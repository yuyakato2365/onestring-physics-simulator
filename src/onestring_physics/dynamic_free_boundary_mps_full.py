"""Full-resident Apple Metal/MPS implementation of the coupled S -> Omega optimizer.

This backend mirrors the CUDA-resident optimizer but avoids PyTorch sparse
matrix kernels. Harmonic boundary response is solved with a matrix-free PCG
whose Laplacian matvec is assembled from the triangle edge list using MPS
index_add operations. UV coordinates, gradients, L-BFGS history, line-search
candidates, safe-step tests and boundary validity tests remain on MPS during
optimization. CPU work is limited to Floater initialization and final audits.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from . import bijective_free_boundary as base
from . import dynamic_free_boundary as previous
from .dynamic_free_boundary_cuda_full import (
    EPS,
    _area_valid_tensor,
    _boundary_direction_tensor,
    _boundary_intersection_count_tensor,
    _evaluate_tensor,
    _full_safe_step_tensor,
    _masked_lbfgs_tensor,
    _triangle_safe_step_tensor,
)
from .reference_bff import triangle_jacobian_diagnostics


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    tris = np.asarray(faces, dtype=np.int64)[:, :3]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _mps_available(torch: Any) -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


class MPSOmegaAccelerator:
    """Tensor objective/validity state compatible with CUDA helper kernels."""

    def __init__(
        self,
        *,
        faces: np.ndarray,
        inverse_surface: np.ndarray,
        surface_areas: np.ndarray,
        boundary_loop: list[int],
        barrier_epsilon: float,
        config: Any,
    ) -> None:
        import torch
        if not _mps_available(torch):
            raise RuntimeError("MPS Omega was requested but torch.backends.mps.is_available() is False")
        self.torch = torch
        self.device = torch.device("mps")
        self.dtype = torch.float32
        self.faces_np = np.asarray(faces, dtype=np.int64)[:, :3]
        self.loop_np = np.asarray(boundary_loop, dtype=np.int64)
        self.faces = torch.tensor(self.faces_np, dtype=torch.long, device=self.device)
        self.inverse_surface = torch.tensor(np.asarray(inverse_surface), dtype=self.dtype, device=self.device)
        self.areas = torch.tensor(np.asarray(surface_areas), dtype=self.dtype, device=self.device)
        self.loop = torch.tensor(self.loop_np, dtype=torch.long, device=self.device)
        self.barrier_epsilon = float(barrier_epsilon)
        self.config = config

        b = len(self.loop_np)
        if b >= 4:
            i, j = np.triu_indices(b, k=1)
            keep = (j != i + 1) & ~((i == 0) & (j == b - 1))
            self.boundary_pair_i = torch.tensor(i[keep], dtype=torch.long, device=self.device)
            self.boundary_pair_j = torch.tensor(j[keep], dtype=torch.long, device=self.device)
        else:
            self.boundary_pair_i = torch.empty(0, dtype=torch.long, device=self.device)
            self.boundary_pair_j = torch.empty(0, dtype=torch.long, device=self.device)

    def _energy_terms(self, uv):
        torch = self.torch
        cfg = self.config
        tri = uv[self.faces]
        d_uv = torch.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], dim=2)
        jac = torch.matmul(d_uv, self.inverse_surface)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        safe_det = det.clamp_min(1.0e-20)

        inv00 = jac[:, 1, 1] / safe_det
        inv01 = -jac[:, 0, 1] / safe_det
        inv10 = -jac[:, 1, 0] / safe_det
        inv11 = jac[:, 0, 0] / safe_det
        frob = torch.sum(jac * jac, dim=(1, 2))
        inv_frob = inv00 * inv00 + inv01 * inv01 + inv10 * inv10 + inv11 * inv11
        conformal = torch.clamp(frob / safe_det - 2.0, min=0.0)
        distortion = torch.sum(
            self.areas * (frob + inv_frob + float(cfg.conformal_weight) * conformal)
        )

        scale = torch.sqrt(safe_det)
        deficit = torch.clamp(float(cfg.minimum_isotropic_scale) - scale, min=0.0)
        shrink = float(cfg.shrink_weight) * torch.sum(self.areas * deficit * deficit)

        boundary_energy = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.loop.numel():
            points = uv[self.loop]
            first = points
            second = torch.roll(points, shifts=-1, dims=0)
            edge = second - first
            squared = torch.sum(edge * edge, dim=1)
            safe_squared = squared.clamp_min(EPS)
            relative = points.unsqueeze(0) - first.unsqueeze(1)
            fraction = torch.clamp(
                torch.sum(relative * edge.unsqueeze(1), dim=2) / safe_squared.unsqueeze(1),
                0.0,
                1.0,
            )
            fraction = torch.where(
                squared.unsqueeze(1) <= EPS,
                torch.full_like(fraction, 0.5),
                fraction,
            )
            closest = first.unsqueeze(1) + fraction.unsqueeze(2) * edge.unsqueeze(1)
            delta = points.unsqueeze(0) - closest
            distance = torch.linalg.vector_norm(delta, dim=2).clamp_min(EPS)
            n = int(self.loop.numel())
            eye = torch.eye(n, dtype=torch.bool, device=self.device)
            incident = eye | torch.roll(eye, shifts=1, dims=1)
            active = (~incident) & (distance < self.barrier_epsilon)
            ratio = torch.where(
                active,
                self.barrier_epsilon / distance - 1.0,
                torch.zeros_like(distance),
            )
            boundary_energy = torch.sum(ratio * ratio)

        total = distortion + float(cfg.boundary_barrier_weight) * boundary_energy + shrink
        return total, distortion, boundary_energy, shrink, det


class MatrixFreeMPSHarmonicResponse:
    """Combinatorial Dirichlet Laplacian solved by matrix-free Jacobi-PCG on MPS."""

    def __init__(
        self,
        faces: np.ndarray,
        vertex_count: int,
        boundary_ids: np.ndarray,
        *,
        tolerance: float = 1.0e-6,
        max_iterations: int = 160,
    ) -> None:
        import torch
        if not _mps_available(torch):
            raise RuntimeError("MPS harmonic response requested but MPS is unavailable")
        self.torch = torch
        self.device = torch.device("mps")
        self.dtype = torch.float32
        self.vertex_count = int(vertex_count)
        self.boundary_ids_np = np.asarray(boundary_ids, dtype=np.int64).reshape(-1)
        boundary_mask = np.zeros(self.vertex_count, dtype=bool)
        boundary_mask[self.boundary_ids_np] = True
        self.interior_ids_np = np.flatnonzero(~boundary_mask).astype(np.int64)
        self.boundary_ids = torch.tensor(self.boundary_ids_np, dtype=torch.long, device=self.device)
        self.interior_ids = torch.tensor(self.interior_ids_np, dtype=torch.long, device=self.device)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.call_count = 0
        self.cg_iterations_total = 0
        self.cg_residual_max = 0.0

        local_i = np.full(self.vertex_count, -1, dtype=np.int64)
        local_i[self.interior_ids_np] = np.arange(len(self.interior_ids_np), dtype=np.int64)
        local_b = np.full(self.vertex_count, -1, dtype=np.int64)
        local_b[self.boundary_ids_np] = np.arange(len(self.boundary_ids_np), dtype=np.int64)

        edges = _unique_edges(faces)
        degree = np.zeros(self.vertex_count, dtype=np.float32)
        if len(edges):
            np.add.at(degree, edges[:, 0], 1.0)
            np.add.at(degree, edges[:, 1], 1.0)

        ii_a, ii_b, ib_i, ib_b = [], [], [], []
        for a_raw, b_raw in edges:
            a, b = int(a_raw), int(b_raw)
            ia, ib = int(local_i[a]), int(local_i[b])
            ba, bb = int(local_b[a]), int(local_b[b])
            if ia >= 0 and ib >= 0:
                ii_a.append(ia)
                ii_b.append(ib)
            elif ia >= 0 and bb >= 0:
                ib_i.append(ia)
                ib_b.append(bb)
            elif ib >= 0 and ba >= 0:
                ib_i.append(ib)
                ib_b.append(ba)

        self.ii_a = torch.tensor(ii_a, dtype=torch.long, device=self.device)
        self.ii_b = torch.tensor(ii_b, dtype=torch.long, device=self.device)
        self.ib_i = torch.tensor(ib_i, dtype=torch.long, device=self.device)
        self.ib_b = torch.tensor(ib_b, dtype=torch.long, device=self.device)
        self.diagonal = torch.tensor(
            degree[self.interior_ids_np], dtype=self.dtype, device=self.device
        ).clamp_min(1.0)

    def _matvec(self, x):
        # L_II x = degree*x - sum_{interior neighbours} x_j
        y = self.diagonal[:, None] * x
        if self.ii_a.numel():
            contrib_a = -x[self.ii_b]
            contrib_b = -x[self.ii_a]
            y.index_add_(0, self.ii_a, contrib_a)
            y.index_add_(0, self.ii_b, contrib_b)
        return y

    def _pcg(self, rhs):
        torch = self.torch
        if rhs.numel() == 0:
            return rhs.clone(), 0, 0.0
        x = torch.zeros_like(rhs)
        r = rhs - self._matvec(x)
        norm_b = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1.0e-20)
        z = r / self.diagonal[:, None]
        p = z.clone()
        rz = torch.sum(r * z, dim=0)
        last = math.inf
        for iteration in range(1, max(1, self.max_iterations) + 1):
            ap = self._matvec(p)
            denom = torch.sum(p * ap, dim=0)
            denom = torch.where(
                torch.abs(denom) < 1.0e-20,
                torch.full_like(denom, 1.0e-20),
                denom,
            )
            alpha = rz / denom
            x = x + p * alpha[None, :]
            r = r - ap * alpha[None, :]
            if iteration == 1 or iteration % 8 == 0 or iteration == self.max_iterations:
                relative = torch.max(torch.linalg.vector_norm(r, dim=0) / norm_b)
                last = float(relative.item())
                if last <= self.tolerance:
                    return x, iteration, last
            z = r / self.diagonal[:, None]
            rz_new = torch.sum(r * z, dim=0)
            safe_rz = torch.where(
                torch.abs(rz) < 1.0e-20,
                torch.full_like(rz, 1.0e-20),
                rz,
            )
            beta = rz_new / safe_rz
            p = z + p * beta[None, :]
            rz = rz_new
        return x, self.max_iterations, float(last)

    def extend_tensor(self, direction):
        torch = self.torch
        out = torch.zeros((self.vertex_count, 2), dtype=self.dtype, device=self.device)
        values = direction[self.boundary_ids] if direction.shape[0] == self.vertex_count else direction
        out[self.boundary_ids] = values
        if not len(self.interior_ids_np):
            return out
        rhs = torch.zeros((len(self.interior_ids_np), 2), dtype=self.dtype, device=self.device)
        if self.ib_i.numel():
            rhs.index_add_(0, self.ib_i, values[self.ib_b])
        solution, iterations, residual = self._pcg(rhs)
        out[self.interior_ids] = solution
        self.call_count += 1
        self.cg_iterations_total += int(iterations)
        self.cg_residual_max = max(self.cg_residual_max, float(residual))
        return out


def full_mps_bijective_free_boundary_parameterization(
    vertices,
    faces,
    config=None,
    progress_callback=None,
):
    """Run the coupled V2 optimization with iterative state resident on Apple MPS."""
    import torch
    if not _mps_available(torch):
        raise RuntimeError("MPS Omega requested but MPS is unavailable")

    cfg = config or previous.BijectiveFreeBoundaryConfig()
    device = torch.device("mps")
    dtype = torch.float32
    started = time.perf_counter()
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    loop, topology = base._extract_single_disk_boundary(tris, len(xyz))

    base._emit_progress(
        progress_callback,
        "S -> Omega FULL MPS",
        0.03,
        f"Apple Metal / MPS | V={len(xyz)}, F={len(tris)}, B={len(loop)}",
    )
    t0 = time.perf_counter()
    uv0 = base._tutte_embedding(xyz, tris, loop, cfg.initial_boundary_shape)
    init_seconds = time.perf_counter() - t0
    initial_uv = uv0.copy()
    t0 = time.perf_counter()
    inverse_surface, areas = base._surface_differentials(xyz, tris)
    diff_seconds = time.perf_counter() - t0

    if (
        not previous._area_valid(uv0, tris, cfg.minimum_signed_double_area)
        or base.boundary_self_intersection_count(uv0, loop)
        or base._check_global_overlap(uv0, tris)
    ):
        raise RuntimeError("full MPS free-boundary initialization is not bijective")

    boundary_ids_np = np.asarray(loop, dtype=int)
    boundary_mask_np = np.zeros(len(xyz), dtype=bool)
    boundary_mask_np[boundary_ids_np] = True
    interior_ids_np = np.flatnonzero(~boundary_mask_np)
    boundary_xyz = xyz[boundary_ids_np]
    barrier_eps = max(
        0.25 * float(np.mean(np.linalg.norm(np.roll(boundary_xyz, -1, axis=0) - boundary_xyz, axis=1))),
        1.0e-8,
    )

    accel = MPSOmegaAccelerator(
        faces=tris,
        inverse_surface=inverse_surface,
        surface_areas=areas,
        boundary_loop=loop,
        barrier_epsilon=barrier_eps,
        config=cfg,
    )
    harmonic = MatrixFreeMPSHarmonicResponse(
        tris, len(xyz), boundary_ids_np, tolerance=1.0e-6, max_iterations=160
    )
    boundary_ids = accel.loop
    interior_ids = torch.tensor(interior_ids_np, dtype=torch.long, device=device)
    uv = torch.tensor(uv0, dtype=dtype, device=device)
    energy, gradient, distortion, boundary_energy, shrink_energy = _evaluate_tensor(accel, uv)
    initial_energy = float(energy.item())
    initial_shrink = float(shrink_energy.item())
    initial_conformal = base._conformal_energy(uv0, tris, inverse_surface, areas)
    initial_scale_min, initial_scale_p05 = previous._isotropic_stats(uv0, tris, inverse_surface)

    history = []
    accepted_boundary = 0
    accepted_interior = 0
    rejected = 0
    counts = {"candidates": 0, "armijo": 0, "validity": 0, "boundary_attempts": 0, "boundary_relax_steps": 0}
    boundary_moves = []
    coupled_interior_moves = []
    log = []
    converged = False
    reason = "maximum_iterations"
    small_streak = 0
    boundary_failures = 0
    max_boundary_failures = 0
    schedule = max(1, int(cfg.interior_steps_per_boundary)) + 1
    max_iterations = max(1, int(cfg.max_iterations))
    gpu_loop_started = time.perf_counter()

    def harmonic_extend(direction):
        return harmonic.extend_tensor(direction)

    def boundary_frame(current_uv):
        p = current_uv[boundary_ids]
        tangent = torch.roll(p, shifts=-1, dims=0) - torch.roll(p, shifts=1, dims=0)
        tangent = tangent / torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp_min(EPS)
        lengths = torch.linalg.vector_norm(torch.roll(p, -1, 0) - p, dim=1)
        valid = lengths[torch.isfinite(lengths) & (lengths > EPS)]
        edge_scale = torch.median(valid) if valid.numel() else torch.ones((), dtype=dtype, device=device)
        return edge_scale

    def relax_interior(start, boundary_target, start_data=None):
        nonlocal rejected
        candidate = start.clone()
        candidate[boundary_ids] = boundary_target
        local_data = _evaluate_tensor(accel, candidate) if start_data is None else start_data
        local_history = []
        accepted = 0
        if interior_ids.numel() == 0:
            return candidate, local_data, accepted
        for _ in range(max(1, min(2, int(cfg.interior_steps_per_boundary)))):
            g = local_data[1]
            direction = _masked_lbfgs_tensor(torch, g, interior_ids, local_history)
            derivative = torch.sum(g * direction)
            if (not bool(torch.isfinite(derivative).item())) or float(derivative.item()) >= -1e-14:
                local_history.clear()
                direction.zero_()
                direction[interior_ids] = -g[interior_ids]
                derivative = torch.sum(g * direction)
            if (not bool(torch.isfinite(derivative).item())) or float(derivative.item()) >= -1e-14:
                break
            rms = torch.linalg.vector_norm(direction[interior_ids]) / math.sqrt(max(1, direction[interior_ids].numel()))
            if float(rms.item()) <= EPS:
                break
            direction = direction / rms
            derivative = derivative / rms
            requested = 0.20 * float(cfg.interior_initial_step_scale) * boundary_frame(candidate)
            safe, _ = _triangle_safe_step_tensor(accel, candidate, direction, cfg.minimum_signed_double_area)
            step = torch.minimum(requested, float(cfg.line_search_safety) * safe) if bool(torch.isfinite(safe).item()) else requested
            accepted_result = None
            for _ls in range(max(1, int(cfg.line_search_max_steps))):
                if float(step.item()) <= torch.finfo(dtype).eps:
                    break
                counts["candidates"] += 1
                trial = candidate + step * direction
                trial[boundary_ids] = boundary_target
                if not bool(_area_valid_tensor(accel, trial, cfg.minimum_signed_double_area).item()):
                    counts["validity"] += 1; rejected += 1; step = step * 0.5; continue
                trial_data = _evaluate_tensor(accel, trial)
                if (not bool(torch.isfinite(trial_data[0]).item())) or bool((trial_data[0] > local_data[0] + 1.0e-4 * step * derivative).item()):
                    counts["armijo"] += 1; rejected += 1; step = step * 0.5; continue
                accepted_result = (trial, trial_data)
                break
            if accepted_result is None:
                break
            trial, trial_data = accepted_result
            s = (trial[interior_ids] - candidate[interior_ids]).reshape(-1).detach()
            y = (trial_data[1][interior_ids] - g[interior_ids]).reshape(-1).detach()
            sy = torch.dot(s, y)
            if float(sy.item()) > 1.0e-12:
                local_history.append((s, y, (1.0 / sy).detach()))
                local_history[:] = local_history[-max(1, int(cfg.lbfgs_history_size)):]
            candidate = trial.detach()
            local_data = tuple(v.detach() for v in trial_data)
            accepted += 1
            counts["boundary_relax_steps"] += 1
        candidate[boundary_ids] = boundary_target
        return candidate, local_data, accepted

    for iteration in range(max_iterations):
        gradient = gradient - torch.mean(gradient, dim=0, keepdim=True)
        phase = "boundary" if ((iteration + 1) % schedule == 0 or interior_ids.numel() == 0) else "interior"

        if phase == "boundary":
            boundary_seed, requested, _edge_scale = _boundary_direction_tensor(accel, uv, gradient, cfg)
            derivative = torch.sum(gradient[boundary_ids] * boundary_seed[boundary_ids])
            predictor_direction = harmonic_extend(boundary_seed)
            safe, safe_reason = _full_safe_step_tensor(accel, uv, predictor_direction, cfg.minimum_signed_double_area)
            phase_gradient = gradient[boundary_ids]
        else:
            direction = _masked_lbfgs_tensor(torch, gradient, interior_ids, history)
            rms = torch.linalg.vector_norm(direction[interior_ids]) / math.sqrt(max(1, direction[interior_ids].numel()))
            if float(rms.item()) > EPS:
                direction = direction / rms
            requested = 0.20 * float(cfg.interior_initial_step_scale) * boundary_frame(uv)
            safe, safe_reason = _triangle_safe_step_tensor(accel, uv, direction, cfg.minimum_signed_double_area)
            derivative = torch.sum(gradient * direction)
            phase_gradient = gradient[interior_ids]

        phase_rms = torch.linalg.vector_norm(phase_gradient) / math.sqrt(max(1, phase_gradient.numel()))
        if float(phase_rms.item()) <= float(cfg.gradient_tolerance):
            full_rms = torch.linalg.vector_norm(gradient) / math.sqrt(max(1, gradient.numel()))
            if float(full_rms.item()) <= float(cfg.gradient_tolerance):
                converged = True; reason = "gradient_tolerance"; break
            continue

        derivative_value = float(derivative.item())
        if (not math.isfinite(derivative_value)) or derivative_value >= -1.0e-14:
            if phase == "interior":
                history.clear(); continue
            boundary_failures += 1
            max_boundary_failures = max(max_boundary_failures, boundary_failures)
            if boundary_failures >= 5:
                reason = "repeated_boundary_direction_not_descent"; break
            continue

        step = requested if not bool(torch.isfinite(safe).item()) else torch.minimum(requested, float(cfg.line_search_safety) * safe)
        result = None
        if phase == "boundary":
            for _attempt in range(max(1, int(cfg.line_search_max_steps))):
                if float(step.item()) <= torch.finfo(dtype).eps:
                    break
                counts["candidates"] += 1; counts["boundary_attempts"] += 1
                boundary_target = uv[boundary_ids] + step * boundary_seed[boundary_ids]
                predictor = uv + harmonic_extend(step * boundary_seed)
                predictor[boundary_ids] = boundary_target
                if (not bool(_area_valid_tensor(accel, predictor, cfg.minimum_signed_double_area).item())) or int(_boundary_intersection_count_tensor(accel, predictor).item()) != 0:
                    counts["validity"] += 1; rejected += 1; step = step * 0.5; continue
                predictor_data = _evaluate_tensor(accel, predictor)
                relaxed, relaxed_data, relaxation_steps = relax_interior(predictor, boundary_target, predictor_data)
                if interior_ids.numel() and relaxation_steps < 1:
                    rejected += 1; step = step * 0.5; continue
                if (not bool(_area_valid_tensor(accel, relaxed, cfg.minimum_signed_double_area).item())) or int(_boundary_intersection_count_tensor(accel, relaxed).item()) != 0:
                    counts["validity"] += 1; rejected += 1; step = step * 0.5; continue
                if (not bool(torch.isfinite(relaxed_data[0]).item())) or bool((relaxed_data[0] >= energy).item()):
                    rejected += 1; step = step * 0.5; continue
                result = (relaxed, relaxed_data, step)
                break
        else:
            for _attempt in range(max(1, int(cfg.line_search_max_steps))):
                if float(step.item()) <= torch.finfo(dtype).eps:
                    break
                counts["candidates"] += 1
                candidate = uv + step * direction
                candidate = candidate - torch.mean(candidate, dim=0, keepdim=True)
                if not bool(_area_valid_tensor(accel, candidate, cfg.minimum_signed_double_area).item()):
                    counts["validity"] += 1; rejected += 1; step = step * 0.5; continue
                candidate_data = _evaluate_tensor(accel, candidate)
                if (not bool(torch.isfinite(candidate_data[0]).item())) or bool((candidate_data[0] > energy + 1.0e-4 * step * derivative).item()):
                    counts["armijo"] += 1; rejected += 1; step = step * 0.5; continue
                result = (candidate, candidate_data, step)
                break

        if result is None:
            if phase == "interior":
                history.clear(); continue
            boundary_failures += 1
            max_boundary_failures = max(max_boundary_failures, boundary_failures)
            if boundary_failures >= 5:
                reason = "repeated_reduced_boundary_line_search_exhausted"; break
            continue

        candidate, new_data, used_step = result
        new_energy, new_gradient, new_distortion, new_boundary, new_shrink = new_data
        relative = abs(float(energy.item()) - float(new_energy.item())) / max(abs(float(energy.item())), 1.0)
        if phase == "interior":
            s = (candidate[interior_ids] - uv[interior_ids]).reshape(-1).detach()
            y = (new_gradient[interior_ids] - gradient[interior_ids]).reshape(-1).detach()
            sy = torch.dot(s, y)
            if float(sy.item()) > 1.0e-12:
                history.append((s, y, (1.0 / sy).detach()))
                history[:] = history[-max(1, int(cfg.lbfgs_history_size)):]
            accepted_interior += 1
        else:
            history.clear(); accepted_boundary += 1; boundary_failures = 0
            delta_b = torch.linalg.vector_norm(candidate[boundary_ids] - uv[boundary_ids], dim=1)
            boundary_moves.append(float(torch.sqrt(torch.mean(delta_b * delta_b)).item()))
            if interior_ids.numel():
                delta_i = torch.linalg.vector_norm(candidate[interior_ids] - uv[interior_ids], dim=1)
                coupled_interior_moves.append(float(torch.sqrt(torch.mean(delta_i * delta_i)).item()))

        uv = candidate.detach()
        energy, gradient, distortion, boundary_energy, shrink_energy = tuple(v.detach() for v in new_data)
        small_streak = small_streak + 1 if relative <= float(cfg.relative_energy_tolerance) else 0
        energy_value = float(energy.item())
        shrink_value = float(shrink_energy.item())
        safe_value = float(safe.item())
        log.append({
            "iteration": int(iteration),
            "phase": phase,
            "energy": energy_value,
            "shrink_energy": shrink_value,
            "accepted_step": float(used_step.item()),
            "safe_step_limit": safe_value,
            "safe_step_reason": safe_reason,
            "gpu_resident_iteration": True,
            "gpu_backend": "mps",
        })

        if phase == "boundary" or iteration % max(1, max_iterations // 200) == 0:
            base._emit_progress(
                progress_callback,
                f"MPS Omega {iteration + 1}/{max_iterations}",
                0.12 + 0.80 * (iteration + 1) / max_iterations,
                f"Metal-resident | phase={phase}; E={energy_value:.6g}; shrink={shrink_value:.4g}; boundary={accepted_boundary}",
            )
        if small_streak >= 2 * schedule:
            converged = True; reason = "relative_energy_tolerance"; break

    gpu_loop_seconds = time.perf_counter() - gpu_loop_started
    uv_np = uv.detach().cpu().numpy().astype(float, copy=False)

    base._emit_progress(progress_callback, "Final CPU validity audit", 0.95, "single MPS->CPU copy; overlap diagnostics")
    boundary_crossings = base.boundary_self_intersection_count(uv_np, loop)
    overlaps = base._check_global_overlap(uv_np, tris) if previous._area_valid(uv_np, tris, cfg.minimum_signed_double_area) else -1
    if (not previous._area_valid(uv_np, tris, cfg.minimum_signed_double_area)) or boundary_crossings or overlaps:
        raise RuntimeError(f"full MPS optimization lost validity: boundary={boundary_crossings}, overlaps={overlaps}")

    diagnostics = triangle_jacobian_diagnostics(xyz, uv_np, tris)
    final_conformal = base._conformal_energy(uv_np, tris, inverse_surface, areas)
    final_scale_min, final_scale_p05 = previous._isotropic_stats(uv_np, tris, inverse_surface)
    displacement = np.linalg.norm(uv_np[boundary_ids_np] - initial_uv[boundary_ids_np], axis=1)
    nonsim, nonsim_rel = base._boundary_nonsimilarity_change(initial_uv[boundary_ids_np], uv_np[boundary_ids_np])
    lambdas = np.asarray(diagnostics["raw_lambda_uv_to_surface_sigma_max"], dtype=float)
    anis = np.asarray(diagnostics["anisotropy"], dtype=float)
    valid_l = lambdas[np.isfinite(lambdas) & (lambdas > 0)]
    valid_a = anis[np.isfinite(anis)]
    floater_info = getattr(base, "_LAST_FLOATER_INITIALIZATION", {}) or {}

    metrics = {
        **topology,
        "parameterization_method": "bijective_free_boundary",
        "parameterization_exactness_label": "coupled_dynamic_full_mps_experiment",
        "flattening_backend": "full_mps_resident_coupled_free_boundary",
        "omega_parameterization_solver": "floater_then_mps_resident_interior_lbfgs_and_reduced_boundary_trials",
        "parameterization_runtime_seconds": float(time.perf_counter() - started),
        "omega_gpu_loop_seconds": float(gpu_loop_seconds),
        "omega_cuda_used": False,
        "omega_mps_available": True,
        "omega_mps_acceleration_used": True,
        "omega_full_cuda_resident": False,
        "omega_full_gpu_resident": True,
        "omega_compute_device": "mps",
        "omega_device_name": "Apple Metal / MPS",
        "omega_torch_dtype": "float32",
        "omega_cpu_gpu_transfer_policy": "initial_UV_to_MPS_once; progress_scalars_only; final_UV_to_CPU_once",
        "omega_iterative_state_residency": "mps",
        "omega_harmonic_backend": "matrix_free_combinatorial_laplacian_pcg",
        "omega_outer_python_control": True,
        "omega_final_global_overlap_audit": "cpu_once",
        "initialization_boundary_shape": cfg.initial_boundary_shape,
        "floater_initialization_mode": str(floater_info.get("mode", "mean_value_arc_length")),
        "floater_fallback_used": bool(floater_info.get("fallback_used", False)),
        "omega_boundary_fixed": False,
        "omega_boundary_shape": "free",
        "optimization_requested_max_iterations": int(max_iterations),
        "optimization_iteration_count": int((log[-1]["iteration"] + 1) if log else 0),
        "optimization_boundary_update_count": int(accepted_boundary),
        "optimization_interior_update_count": int(accepted_interior),
        "optimization_rejected_line_search_step_count": int(rejected),
        "maximum_consecutive_boundary_line_search_failures": int(max_boundary_failures),
        "armijo_rejected_candidate_count": int(counts["armijo"]),
        "local_validity_rejected_candidate_count": int(counts["validity"]),
        "optimization_termination_reason": reason,
        "parameterization_warning": "" if converged else f"Stopped with {reason}; final UV remains bijective.",
        "initial_energy": float(initial_energy),
        "final_energy": float(energy.item()),
        "initial_shrink_energy": float(initial_shrink),
        "final_shrink_energy": float(shrink_energy.item()),
        "initial_conformal_energy": float(initial_conformal),
        "final_conformal_energy": float(final_conformal),
        "initial_isotropic_scale_min": float(initial_scale_min),
        "initial_isotropic_scale_p05": float(initial_scale_p05),
        "final_isotropic_scale_min": float(final_scale_min),
        "final_isotropic_scale_p05": float(final_scale_p05),
        "boundary_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0,
        "boundary_displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
        "boundary_nonsimilarity_change_rms": float(nonsim),
        "boundary_nonsimilarity_change_relative_rms": float(nonsim_rel),
        "boundary_update_accepted_displacement_rms_mean": float(np.mean(boundary_moves)) if boundary_moves else 0.0,
        "boundary_update_accepted_displacement_rms_max": float(np.max(boundary_moves)) if boundary_moves else 0.0,
        "boundary_update_coupled_interior_displacement_rms_mean": float(np.mean(coupled_interior_moves)) if coupled_interior_moves else 0.0,
        "boundary_update_coupled_interior_displacement_rms_max": float(np.max(coupled_interior_moves)) if coupled_interior_moves else 0.0,
        "boundary_harmonic_response_call_count": int(harmonic.call_count),
        "boundary_harmonic_cg_iterations_total": int(harmonic.cg_iterations_total),
        "boundary_harmonic_cg_residual_max": float(harmonic.cg_residual_max),
        "floater_initialization_seconds": float(init_seconds),
        "surface_differentials_seconds": float(diff_seconds),
        "optimization_iteration_log": log,
        "optimization_boundary_attempt_log": [],
        "lambda_min": float(np.min(valid_l)) if len(valid_l) else 0.0,
        "lambda_median": float(np.median(valid_l)) if len(valid_l) else 0.0,
        "lambda_max": float(np.max(valid_l)) if len(valid_l) else 0.0,
        "anisotropy_mean": float(np.mean(valid_a)) if len(valid_a) else 0.0,
        "anisotropy_max": float(np.max(valid_a)) if len(valid_a) else 0.0,
        "per_triangle_lambda": lambdas.tolist(),
        "per_triangle_log_lambda": np.log(np.maximum(lambdas, EPS)).tolist(),
        "per_triangle_anisotropy": anis.tolist(),
        "topology_modified": False,
        "seams_or_cuts_added": False,
    }
    base._emit_progress(
        progress_callback,
        "S -> Omega full MPS complete",
        1.0,
        f"Apple Metal / MPS | iter={metrics['optimization_iteration_count']}/{max_iterations}; E={metrics['final_energy']:.6g}",
    )
    return uv_np, loop, metrics


__all__ = ["full_mps_bijective_free_boundary_parameterization", "MatrixFreeMPSHarmonicResponse"]
