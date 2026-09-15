"""Full-resident CUDA implementation of the coupled S -> Omega optimizer.

Unlike the hybrid CUDA patch, this solver keeps UV coordinates, gradients,
L-BFGS history, search directions, line-search candidates, validity tests and
harmonic boundary response on one CUDA device for the entire optimization
loop. CPU work is intentionally limited to Floater initialization, occasional
progress scalar extraction, and final global diagnostics/overlap audit.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from . import bijective_free_boundary as base
from . import dynamic_free_boundary as previous
from .large_steps_mesh_conditioning import _pcg_solve
from .dynamic_free_boundary_cuda_backend import (
    TorchHarmonicBoundaryResponse,
    TorchOmegaAccelerator,
    _resolve_torch_device,
)
from .reference_bff import triangle_jacobian_diagnostics

EPS = 1.0e-12


def _evaluate_tensor(accel: TorchOmegaAccelerator, uv):
    """Evaluate objective and gradient without leaving the torch device."""
    torch = accel.torch
    x = uv.detach().requires_grad_(True)
    total, distortion, boundary, shrink, det = accel._energy_terms(x)
    invalid = torch.any(~torch.isfinite(det)) | torch.any(det <= EPS)
    if bool(invalid.item()):
        return (
            torch.full((), float("inf"), dtype=accel.dtype, device=accel.device),
            torch.zeros_like(uv),
            torch.full((), float("inf"), dtype=accel.dtype, device=accel.device),
            torch.full((), float("inf"), dtype=accel.dtype, device=accel.device),
            torch.full((), float("inf"), dtype=accel.dtype, device=accel.device),
        )
    grad = torch.autograd.grad(total, x, create_graph=False, retain_graph=False)[0]
    return total.detach(), grad.detach(), distortion.detach(), boundary.detach(), shrink.detach()


def _signed_areas_tensor(accel: TorchOmegaAccelerator, uv):
    tri = uv[accel.faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _area_valid_tensor(accel: TorchOmegaAccelerator, uv, minimum: float):
    torch = accel.torch
    return torch.all(torch.isfinite(uv)) & torch.all(_signed_areas_tensor(accel, uv) > float(minimum))


def _positive_root_tensor(torch, c0, c1, c2, dtype):
    scale = torch.maximum(
        torch.maximum(torch.abs(c0), torch.abs(c1)),
        torch.maximum(torch.abs(c2), torch.ones_like(c0)),
    )
    tol = (1.0e-12 if dtype == torch.float32 else 1.0e-14) * scale
    linear = torch.abs(c2) <= tol
    inf = torch.full_like(c0, float("inf"))
    eps = torch.finfo(dtype).eps
    linear_root = torch.where(torch.abs(c1) > tol, -c0 / c1, inf)
    linear_root = torch.where(
        linear & (linear_root > eps) & torch.isfinite(linear_root), linear_root, inf
    )
    disc = c1 * c1 - 4.0 * c2 * c0
    quadratic = (~linear) & (disc >= -tol)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    denom = torch.where(quadratic, 2.0 * c2, torch.ones_like(c2))
    r1 = (-c1 - sqrt_disc) / denom
    r2 = (-c1 + sqrt_disc) / denom
    r1 = torch.where(quadratic & (r1 > eps) & torch.isfinite(r1), r1, inf)
    r2 = torch.where(quadratic & (r2 > eps) & torch.isfinite(r2), r2, inf)
    return torch.minimum(torch.minimum(linear_root, r1), r2)


def _triangle_safe_step_tensor(accel: TorchOmegaAccelerator, uv, direction, minimum: float):
    torch = accel.torch
    p = uv[accel.faces]
    v = direction[accel.faces]
    a = p[:, 1] - p[:, 0]
    b = p[:, 2] - p[:, 0]
    da = v[:, 1] - v[:, 0]
    db = v[:, 2] - v[:, 0]
    c0 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] - float(minimum)
    c1 = (
        da[:, 0] * b[:, 1] - da[:, 1] * b[:, 0]
        + a[:, 0] * db[:, 1] - a[:, 1] * db[:, 0]
    )
    c2 = da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0]
    roots = _positive_root_tensor(torch, c0, c1, c2, accel.dtype)
    if roots.numel() == 0:
        return torch.full((), float("inf"), dtype=accel.dtype, device=accel.device), "unbounded"
    limit = torch.min(roots)
    return limit, "triangle_degeneracy"


def _boundary_intersection_count_tensor(accel: TorchOmegaAccelerator, uv):
    torch = accel.torch
    if accel.boundary_pair_i.numel() == 0:
        return torch.zeros((), dtype=torch.int64, device=accel.device)
    coords = uv[accel.loop]
    starts = coords
    ends = torch.roll(coords, shifts=-1, dims=0)
    a = starts[accel.boundary_pair_i]
    b = ends[accel.boundary_pair_i]
    c = starts[accel.boundary_pair_j]
    d = ends[accel.boundary_pair_j]

    def orient(start, end, point):
        edge = end - start
        rel = point - start
        return edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0]

    tolerance = 1.0e-7 if accel.dtype == torch.float32 else 1.0e-12

    def on_segment(start, end, point):
        return (
            (torch.minimum(start[:, 0], end[:, 0]) - tolerance <= point[:, 0])
            & (point[:, 0] <= torch.maximum(start[:, 0], end[:, 0]) + tolerance)
            & (torch.minimum(start[:, 1], end[:, 1]) - tolerance <= point[:, 1])
            & (point[:, 1] <= torch.maximum(start[:, 1], end[:, 1]) + tolerance)
        )

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    proper = (o1 * o2 < -tolerance) & (o3 * o4 < -tolerance)
    touching = (
        ((torch.abs(o1) <= tolerance) & on_segment(a, b, c))
        | ((torch.abs(o2) <= tolerance) & on_segment(a, b, d))
        | ((torch.abs(o3) <= tolerance) & on_segment(c, d, a))
        | ((torch.abs(o4) <= tolerance) & on_segment(c, d, b))
    )
    return torch.count_nonzero(proper | touching)


def _full_safe_step_tensor(accel: TorchOmegaAccelerator, uv, direction, minimum: float):
    """GPU first-singularity bound including boundary edge/vertex collisions."""
    torch = accel.torch
    tri_limit, _ = _triangle_safe_step_tensor(accel, uv, direction, minimum)
    limit = tri_limit
    reason = "triangle_degeneracy"
    loop = accel.loop
    count = int(loop.numel())
    if count == 0:
        return limit, reason

    points = uv[loop]
    movement = direction[loop]
    second = torch.roll(points, shifts=-1, dims=0)
    second_d = torch.roll(movement, shifts=-1, dims=0)
    edge = second - points
    d_edge = second_d - movement
    inf = torch.full((), float("inf"), dtype=accel.dtype, device=accel.device)
    boundary_limit = inf
    point_index = torch.arange(count, device=accel.device)

    # Chunk edge rows to keep B^2 temporary memory bounded on large boundaries.
    chunk = 192
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        first = points[start:stop]
        first_d = movement[start:stop]
        e = edge[start:stop]
        de = d_edge[start:stop]
        relative = points.unsqueeze(0) - first.unsqueeze(1)
        d_relative = movement.unsqueeze(0) - first_d.unsqueeze(1)
        c0 = e[:, None, 0] * relative[:, :, 1] - e[:, None, 1] * relative[:, :, 0]
        c1 = (
            de[:, None, 0] * relative[:, :, 1]
            - de[:, None, 1] * relative[:, :, 0]
            + e[:, None, 0] * d_relative[:, :, 1]
            - e[:, None, 1] * d_relative[:, :, 0]
        )
        c2 = de[:, None, 0] * d_relative[:, :, 1] - de[:, None, 1] * d_relative[:, :, 0]
        root = _positive_root_tensor(torch, c0, c1, c2, accel.dtype)
        finite_root = torch.where(torch.isfinite(root), root, torch.zeros_like(root))
        edge_at = e[:, None, :] + finite_root[:, :, None] * de[:, None, :]
        relative_at = relative + finite_root[:, :, None] * d_relative
        denominator = torch.sum(edge_at * edge_at, dim=2)
        fraction = torch.sum(relative_at * edge_at, dim=2) / denominator.clamp_min(EPS)
        rows = torch.arange(start, stop, device=accel.device).unsqueeze(1)
        incident = (point_index.unsqueeze(0) == rows) | (
            point_index.unsqueeze(0) == ((rows + 1) % count)
        )
        valid = (
            torch.isfinite(root)
            & (~incident)
            & (denominator > EPS)
            & (fraction >= -1.0e-9)
            & (fraction <= 1.0 + 1.0e-9)
        )
        local = torch.min(torch.where(valid, root, torch.full_like(root, float("inf"))))
        boundary_limit = torch.minimum(boundary_limit, local)

    if bool((boundary_limit < limit).item()):
        return boundary_limit, "boundary_edge_vertex_collision"
    return limit, reason


def _boundary_frame_tensor(accel: TorchOmegaAccelerator, uv):
    torch = accel.torch
    p = uv[accel.loop]
    tangent = torch.roll(p, shifts=-1, dims=0) - torch.roll(p, shifts=1, dims=0)
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp_min(EPS)
    area = 0.5 * torch.sum(p[:, 0] * torch.roll(p[:, 1], -1) - p[:, 1] * torch.roll(p[:, 0], -1))
    normal = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)
    normal = torch.where((area < 0).reshape(1, 1), -normal, normal)
    lengths = torch.linalg.vector_norm(torch.roll(p, -1, 0) - p, dim=1)
    valid = lengths[torch.isfinite(lengths) & (lengths > EPS)]
    edge_scale = torch.median(valid) if valid.numel() else torch.ones((), dtype=uv.dtype, device=uv.device)
    return tangent, normal, edge_scale


def _boundary_direction_tensor(accel: TorchOmegaAccelerator, uv, gradient, cfg):
    torch = accel.torch
    tangent, normal, edge_scale = _boundary_frame_tensor(accel, uv)
    descent = -gradient[accel.loop]
    n = torch.sum(descent * normal, dim=1)
    t = torch.sum(descent * tangent, dim=1)
    finite = torch.abs(n[torch.isfinite(n)])
    if finite.numel():
        median = torch.median(finite)
        q90 = torch.quantile(finite, 0.90)
        cap = torch.maximum(torch.maximum(median * float(cfg.boundary_gradient_clip_factor), q90), torch.tensor(EPS, dtype=uv.dtype, device=uv.device))
        n = torch.clamp(n, min=-cap, max=cap)
    smooth = float(np.clip(cfg.boundary_normal_smoothing, 0.0, 0.95))
    if smooth > 0.0 and n.numel() >= 3:
        avg = 0.25 * torch.roll(n, 1) + 0.5 * n + 0.25 * torch.roll(n, -1)
        n = (1.0 - smooth) * n + smooth * avg
    vector = n[:, None] * normal + float(cfg.boundary_tangent_weight) * t[:, None] * tangent
    direction = torch.zeros_like(gradient)
    direction[accel.loop] = vector
    derivative = torch.sum(gradient * direction)
    fallback = torch.zeros_like(gradient)
    fallback[accel.loop] = descent
    direction = torch.where((derivative >= -1.0e-14).reshape(1, 1), fallback, direction)
    rms = torch.linalg.vector_norm(direction[accel.loop]) / math.sqrt(max(1, direction[accel.loop].numel()))
    direction = direction / rms.clamp_min(EPS)
    requested = float(cfg.boundary_step_fraction) * max(float(cfg.initial_step_scale), EPS) * edge_scale
    return direction, requested.clamp_min(torch.finfo(uv.dtype).eps), edge_scale


def _masked_lbfgs_tensor(torch, gradient, ids, history):
    out = torch.zeros_like(gradient)
    if ids.numel() == 0:
        return out
    q = gradient[ids].reshape(-1).clone()
    alphas = []
    for s, y, rho in reversed(history):
        a = rho * torch.dot(s, q)
        alphas.append(a)
        q = q - a * y
    if history:
        s, y, _rho = history[-1]
        yy = torch.dot(y, y)
        scale = torch.where(yy > EPS, torch.dot(s, y) / yy, torch.ones_like(yy))
        q = q * torch.clamp(scale, min=1.0e-10)
    for (s, y, rho), a in zip(history, reversed(alphas)):
        q = q + s * (a - rho * torch.dot(y, q))
    out[ids] = -q.reshape((-1, 2))
    return out


def _harmonic_extend_tensor(response: TorchHarmonicBoundaryResponse, boundary_direction):
    torch = response.torch
    out = torch.zeros((response.vertex_count, 2), dtype=response.dtype, device=response.device)
    boundary_ids = torch.tensor(response.boundary_ids, dtype=torch.long, device=response.device)
    if boundary_direction.shape[0] == response.vertex_count:
        values = boundary_direction[boundary_ids]
    else:
        values = boundary_direction
    out[boundary_ids] = values
    if not len(response.interior_ids):
        return out
    rhs = torch.zeros((len(response.interior_ids), 2), dtype=response.dtype, device=response.device)
    if response.ib_i.numel():
        rhs.index_add_(0, response.ib_i, values[response.ib_b])
    solution, iterations, residual = _pcg_solve(
        torch,
        response.matrix,
        response.diagonal,
        rhs,
        tolerance=response.cg_tolerance,
        max_iterations=response.cg_max_iterations,
    )
    interior_ids = torch.tensor(response.interior_ids, dtype=torch.long, device=response.device)
    out[interior_ids] = solution
    response.call_count += 1
    response.cg_iterations_total += int(iterations)
    response.cg_residual_max = max(response.cg_residual_max, float(residual))
    return out


def full_cuda_bijective_free_boundary_parameterization(
    vertices,
    faces,
    config=None,
    progress_callback=None,
):
    """Run the coupled V2-style optimization with all iterative state on CUDA."""
    cfg = config or previous.BijectiveFreeBoundaryConfig()
    torch, device, dtype = _resolve_torch_device("cuda")
    started = time.perf_counter()
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    loop, topology = base._extract_single_disk_boundary(tris, len(xyz))

    base._emit_progress(
        progress_callback,
        "S -> Omega full CUDA initialization",
        0.03,
        f"{torch.cuda.get_device_name(device)} | V={len(xyz)}, F={len(tris)}, B={len(loop)}",
    )
    t0 = time.perf_counter()
    uv0 = base._tutte_embedding(xyz, tris, loop, cfg.initial_boundary_shape)
    init_seconds = time.perf_counter() - t0
    initial_uv = uv0.copy()
    t0 = time.perf_counter()
    inverse_surface, areas = base._surface_differentials(xyz, tris)
    diff_seconds = time.perf_counter() - t0

    # One-time CPU audit before moving the accepted state to CUDA.
    if (
        not previous._area_valid(uv0, tris, cfg.minimum_signed_double_area)
        or base.boundary_self_intersection_count(uv0, loop)
        or base._check_global_overlap(uv0, tris)
    ):
        raise RuntimeError("full CUDA free-boundary initialization is not bijective")

    boundary_ids_np = np.asarray(loop, dtype=int)
    boundary_mask_np = np.zeros(len(xyz), dtype=bool)
    boundary_mask_np[boundary_ids_np] = True
    interior_ids_np = np.flatnonzero(~boundary_mask_np)
    boundary_xyz = xyz[boundary_ids_np]
    barrier_eps = max(
        0.25 * float(np.mean(np.linalg.norm(np.roll(boundary_xyz, -1, axis=0) - boundary_xyz, axis=1))),
        1.0e-8,
    )
    accel = TorchOmegaAccelerator(
        faces=tris,
        inverse_surface=inverse_surface,
        surface_areas=areas,
        boundary_loop=loop,
        barrier_epsilon=barrier_eps,
        config=cfg,
        device="cuda",
    )
    harmonic = TorchHarmonicBoundaryResponse(
        tris,
        len(xyz),
        boundary_ids_np,
        device="cuda",
        cg_tolerance=1.0e-6,
        cg_max_iterations=160,
    )
    boundary_ids = accel.loop
    interior_ids = torch.tensor(interior_ids_np, dtype=torch.long, device=device)
    uv = torch.tensor(uv0, dtype=dtype, device=device)
    data = _evaluate_tensor(accel, uv)
    energy, gradient, distortion, boundary_energy, shrink_energy = data
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
            _, _, edge_scale = _boundary_frame_tensor(accel, candidate)
            requested = 0.20 * float(cfg.interior_initial_step_scale) * edge_scale
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
            boundary_seed, requested, edge_scale = _boundary_direction_tensor(accel, uv, gradient, cfg)
            derivative = torch.sum(gradient[boundary_ids] * boundary_seed[boundary_ids])
            predictor_direction = _harmonic_extend_tensor(harmonic, boundary_seed)
            safe, safe_reason = _full_safe_step_tensor(accel, uv, predictor_direction, cfg.minimum_signed_double_area)
            phase_gradient = gradient[boundary_ids]
            direction = boundary_seed
        else:
            direction = _masked_lbfgs_tensor(torch, gradient, interior_ids, history)
            rms = torch.linalg.vector_norm(direction[interior_ids]) / math.sqrt(max(1, direction[interior_ids].numel()))
            if float(rms.item()) > EPS:
                direction = direction / rms
            _, _, edge_scale = _boundary_frame_tensor(accel, uv)
            requested = 0.20 * float(cfg.interior_initial_step_scale) * edge_scale
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
                predictor = uv + _harmonic_extend_tensor(harmonic, step * boundary_seed)
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
        })

        if phase == "boundary" or iteration % max(1, max_iterations // 200) == 0:
            base._emit_progress(
                progress_callback,
                f"CUDA Omega {iteration + 1}/{max_iterations}",
                0.12 + 0.80 * (iteration + 1) / max_iterations,
                f"GPU-resident | phase={phase}; E={energy_value:.6g}; shrink={shrink_value:.4g}; boundary={accepted_boundary}",
            )
        if small_streak >= 2 * schedule:
            converged = True; reason = "relative_energy_tolerance"; break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_loop_seconds = time.perf_counter() - gpu_loop_started
    uv_np = uv.detach().cpu().numpy().astype(float, copy=False)

    base._emit_progress(progress_callback, "Final CPU validity audit", 0.95, "single GPU->CPU copy; overlap diagnostics")
    boundary_crossings = base.boundary_self_intersection_count(uv_np, loop)
    overlaps = base._check_global_overlap(uv_np, tris) if previous._area_valid(uv_np, tris, cfg.minimum_signed_double_area) else -1
    if (not previous._area_valid(uv_np, tris, cfg.minimum_signed_double_area)) or boundary_crossings or overlaps:
        raise RuntimeError(f"full CUDA optimization lost validity: boundary={boundary_crossings}, overlaps={overlaps}")

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
        "parameterization_exactness_label": "coupled_dynamic_full_cuda_experiment",
        "flattening_backend": "full_cuda_resident_coupled_free_boundary",
        "omega_parameterization_solver": "floater_then_gpu_resident_interior_lbfgs_and_reduced_boundary_trials",
        "parameterization_runtime_seconds": float(time.perf_counter() - started),
        "omega_gpu_loop_seconds": float(gpu_loop_seconds),
        "omega_cuda_used": True,
        "omega_full_cuda_resident": True,
        "omega_compute_device": str(device),
        "omega_device_name": str(torch.cuda.get_device_name(device)),
        "omega_torch_dtype": str(dtype).replace("torch.", ""),
        "omega_cpu_gpu_transfer_policy": "initial_UV_to_GPU_once; progress_scalars_only; final_UV_to_CPU_once",
        "omega_iterative_state_residency": "cuda",
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
        "S -> Omega full CUDA complete",
        1.0,
        f"{metrics['omega_device_name']} | iter={metrics['optimization_iteration_count']}/{max_iterations}; E={metrics['final_energy']:.6g}",
    )
    return uv_np, loop, metrics


__all__ = ["full_cuda_bijective_free_boundary_parameterization"]
