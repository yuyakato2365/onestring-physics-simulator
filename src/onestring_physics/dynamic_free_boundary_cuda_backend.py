"""PyTorch/CUDA kernels for the coupled bijective free-boundary optimizer.

The outer optimization schedule remains in ``dynamic_free_boundary_v2`` so the
research semantics and diagnostics stay comparable. The expensive per-triangle
energy/gradient, shrink penalty, boundary barrier, interior safe-step, boundary
self-intersection test, and harmonic boundary response run on one persistent
PyTorch device. CUDA is selected automatically when available.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .large_steps_mesh_conditioning import _pcg_solve

_EPS = 1.0e-12


def _resolve_torch_device(requested: str = "auto"):
    import torch

    value = str(requested).strip().lower()
    if value not in {"auto", "cuda", "cpu"}:
        raise ValueError("Omega torch device must be auto, cuda, or cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Omega CUDA was requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    elif value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    return torch, device, dtype


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    tris = np.asarray(faces, dtype=np.int64)[:, :3]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


class TorchOmegaAccelerator:
    """Persistent CUDA/CPU tensor backend for the S -> Omega optimizer."""

    def __init__(
        self,
        *,
        faces: np.ndarray,
        inverse_surface: np.ndarray,
        surface_areas: np.ndarray,
        boundary_loop: list[int],
        barrier_epsilon: float,
        config: Any,
        device: str = "auto",
    ) -> None:
        self.torch, self.device, self.dtype = _resolve_torch_device(device)
        torch = self.torch
        self.faces_np = np.asarray(faces, dtype=np.int64)[:, :3]
        self.loop_np = np.asarray(boundary_loop, dtype=np.int64)
        self.faces = torch.tensor(self.faces_np, dtype=torch.long, device=self.device)
        self.inverse_surface = torch.tensor(np.asarray(inverse_surface), dtype=self.dtype, device=self.device)
        self.areas = torch.tensor(np.asarray(surface_areas), dtype=self.dtype, device=self.device)
        self.loop = torch.tensor(self.loop_np, dtype=torch.long, device=self.device)
        self.barrier_epsilon = float(barrier_epsilon)
        self.config = config
        self.energy_call_count = 0
        self.energy_seconds = 0.0
        self.safe_call_count = 0
        self.safe_seconds = 0.0
        self.boundary_check_count = 0
        self.boundary_check_seconds = 0.0

        b = len(self.loop_np)
        if b >= 4:
            i, j = np.triu_indices(b, k=1)
            keep = (j != i + 1) & ~((i == 0) & (j == b - 1))
            self.boundary_pair_i = torch.tensor(i[keep], dtype=torch.long, device=self.device)
            self.boundary_pair_j = torch.tensor(j[keep], dtype=torch.long, device=self.device)
        else:
            self.boundary_pair_i = torch.empty(0, dtype=torch.long, device=self.device)
            self.boundary_pair_j = torch.empty(0, dtype=torch.long, device=self.device)

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return str(self.torch.cuda.get_device_name(self.device))
        return "CPU"

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _tensor(self, values: np.ndarray, *, requires_grad: bool = False):
        return self.torch.tensor(np.asarray(values), dtype=self.dtype, device=self.device, requires_grad=requires_grad)

    def _energy_terms(self, uv):
        torch = self.torch
        cfg = self.config
        tri = uv[self.faces]
        d_uv = torch.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], dim=2)
        jac = torch.matmul(d_uv, self.inverse_surface)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        safe_det = det.clamp_min(1.0e-20)
        inv = torch.empty_like(jac)
        inv[:, 0, 0] = jac[:, 1, 1] / safe_det
        inv[:, 0, 1] = -jac[:, 0, 1] / safe_det
        inv[:, 1, 0] = -jac[:, 1, 0] / safe_det
        inv[:, 1, 1] = jac[:, 0, 0] / safe_det

        frob = torch.sum(jac * jac, dim=(1, 2))
        inv_frob = torch.sum(inv * inv, dim=(1, 2))
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
            safe_squared = squared.clamp_min(_EPS)
            relative = points.unsqueeze(0) - first.unsqueeze(1)
            fraction = torch.clamp(
                torch.sum(relative * edge.unsqueeze(1), dim=2) / safe_squared.unsqueeze(1),
                0.0,
                1.0,
            )
            fraction = torch.where(
                squared.unsqueeze(1) <= _EPS,
                torch.full_like(fraction, 0.5),
                fraction,
            )
            closest = first.unsqueeze(1) + fraction.unsqueeze(2) * edge.unsqueeze(1)
            delta = points.unsqueeze(0) - closest
            distance = torch.linalg.vector_norm(delta, dim=2).clamp_min(_EPS)
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

    def evaluate(self, values: np.ndarray):
        """Return the tuple shape expected by dynamic_free_boundary_v2."""
        started = time.perf_counter()
        self.energy_call_count += 1
        uv = self._tensor(values, requires_grad=True)
        total, distortion, boundary, shrink, det = self._energy_terms(uv)
        invalid = bool(
            self.torch.any(~self.torch.isfinite(det)).item()
            or self.torch.any(det <= _EPS).item()
        )
        if invalid:
            gradient = np.zeros_like(np.asarray(values, dtype=float))
            result = (math.inf, gradient, math.inf, math.inf, math.inf)
        else:
            total.backward()
            gradient = uv.grad.detach().cpu().numpy().astype(float, copy=False)
            result = (
                float(total.detach().item()),
                gradient,
                float(distortion.detach().item()),
                float(boundary.detach().item()),
                float(shrink.detach().item()),
            )
        self._sync()
        self.energy_seconds += time.perf_counter() - started
        return result

    def area_valid(self, values: np.ndarray, minimum: float) -> bool:
        torch = self.torch
        uv = self._tensor(values)
        tri = uv[self.faces]
        a = tri[:, 1] - tri[:, 0]
        b = tri[:, 2] - tri[:, 0]
        signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return bool(torch.all(torch.isfinite(uv)).item() and torch.all(signed > float(minimum)).item())

    def triangle_safe_step(self, values: np.ndarray, direction: np.ndarray, minimum: float) -> tuple[float, str]:
        started = time.perf_counter()
        self.safe_call_count += 1
        torch = self.torch
        uv = self._tensor(values)
        d = self._tensor(direction)
        p = uv[self.faces]
        v = d[self.faces]
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
        scale = torch.maximum(
            torch.maximum(torch.abs(c0), torch.abs(c1)),
            torch.maximum(torch.abs(c2), torch.ones_like(c0)),
        )
        tol = (1.0e-12 if self.dtype == torch.float32 else 1.0e-14) * scale
        linear = torch.abs(c2) <= tol
        inf = torch.full_like(c0, float("inf"))
        eps = torch.finfo(self.dtype).eps
        linear_root = torch.where(torch.abs(c1) > tol, -c0 / c1, inf)
        linear_root = torch.where(
            linear & (linear_root > eps) & torch.isfinite(linear_root),
            linear_root,
            inf,
        )
        disc = c1 * c1 - 4.0 * c2 * c0
        quadratic = (~linear) & (disc >= -tol)
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        denom = torch.where(quadratic, 2.0 * c2, torch.ones_like(c2))
        r1 = (-c1 - sqrt_disc) / denom
        r2 = (-c1 + sqrt_disc) / denom
        r1 = torch.where(quadratic & (r1 > eps) & torch.isfinite(r1), r1, inf)
        r2 = torch.where(quadratic & (r2 > eps) & torch.isfinite(r2), r2, inf)
        roots = torch.minimum(torch.minimum(linear_root, r1), r2)
        limit = float(torch.min(roots).item()) if roots.numel() else math.inf
        self._sync()
        self.safe_seconds += time.perf_counter() - started
        return limit, "triangle_degeneracy" if math.isfinite(limit) else "unbounded"

    def boundary_self_intersection_count(self, values: np.ndarray) -> int:
        started = time.perf_counter()
        self.boundary_check_count += 1
        torch = self.torch
        if self.boundary_pair_i.numel() == 0:
            return 0
        uv = self._tensor(values)
        coords = uv[self.loop]
        starts = coords
        ends = torch.roll(coords, shifts=-1, dims=0)
        a = starts[self.boundary_pair_i]
        b = ends[self.boundary_pair_i]
        c = starts[self.boundary_pair_j]
        d = ends[self.boundary_pair_j]

        def orient(start, end, point):
            edge = end - start
            rel = point - start
            return edge[:, 0] * rel[:, 1] - edge[:, 1] * rel[:, 0]

        tolerance = 1.0e-7 if self.dtype == torch.float32 else 1.0e-12

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
        count = int(torch.count_nonzero(proper | touching).item())
        self._sync()
        self.boundary_check_seconds += time.perf_counter() - started
        return count


class TorchHarmonicBoundaryResponse:
    """CUDA PCG version of the cached Dirichlet displacement predictor."""

    def __init__(
        self,
        faces: np.ndarray,
        vertex_count: int,
        boundary_ids: np.ndarray,
        *,
        device: str = "auto",
        cg_tolerance: float = 1.0e-6,
        cg_max_iterations: int = 160,
    ) -> None:
        self.torch, self.device, self.dtype = _resolve_torch_device(device)
        torch = self.torch
        self.vertex_count = int(vertex_count)
        self.boundary_ids = np.asarray(boundary_ids, dtype=np.int64).reshape(-1)
        boundary_mask = np.zeros(self.vertex_count, dtype=bool)
        boundary_mask[self.boundary_ids] = True
        self.interior_ids = np.flatnonzero(~boundary_mask).astype(np.int64)
        self.cg_tolerance = float(cg_tolerance)
        self.cg_max_iterations = int(cg_max_iterations)
        self.factorization_seconds = 0.0
        self.call_count = 0
        self.solve_seconds = 0.0
        self.cg_iterations_total = 0
        self.cg_residual_max = 0.0

        local_i = np.full(self.vertex_count, -1, dtype=np.int64)
        local_i[self.interior_ids] = np.arange(len(self.interior_ids))
        local_b = np.full(self.vertex_count, -1, dtype=np.int64)
        local_b[self.boundary_ids] = np.arange(len(self.boundary_ids))
        edges = _unique_edges(faces)
        degree = np.zeros(self.vertex_count, dtype=np.float64)
        if len(edges):
            np.add.at(degree, edges[:, 0], 1.0)
            np.add.at(degree, edges[:, 1], 1.0)

        rows, cols, vals = [], [], []
        diag = degree[self.interior_ids]
        for idx, value in enumerate(diag):
            rows.append(idx)
            cols.append(idx)
            vals.append(float(value))
        ib_i, ib_b = [], []
        for a_raw, b_raw in edges:
            a, b = int(a_raw), int(b_raw)
            ia, ib = int(local_i[a]), int(local_i[b])
            if ia >= 0 and ib >= 0:
                rows.extend([ia, ib])
                cols.extend([ib, ia])
                vals.extend([-1.0, -1.0])
            elif ia >= 0 and local_b[b] >= 0:
                ib_i.append(ia)
                ib_b.append(int(local_b[b]))
            elif ib >= 0 and local_b[a] >= 0:
                ib_i.append(ib)
                ib_b.append(int(local_b[a]))

        if len(self.interior_ids):
            indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
            values = torch.tensor(vals, dtype=self.dtype, device=self.device)
            self.matrix = torch.sparse_coo_tensor(
                indices,
                values,
                (len(self.interior_ids), len(self.interior_ids)),
                dtype=self.dtype,
                device=self.device,
            ).coalesce()
            self.diagonal = torch.tensor(diag, dtype=self.dtype, device=self.device)
        else:
            self.matrix = None
            self.diagonal = None
        self.ib_i = torch.tensor(ib_i, dtype=torch.long, device=self.device)
        self.ib_b = torch.tensor(ib_b, dtype=torch.long, device=self.device)

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return str(self.torch.cuda.get_device_name(self.device))
        return "CPU"

    def extend(self, boundary_only_direction: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        self.call_count += 1
        supplied = np.asarray(boundary_only_direction, dtype=float)
        out = np.zeros((self.vertex_count, 2), dtype=float)
        if supplied.shape == out.shape:
            boundary_values_np = supplied[self.boundary_ids]
        elif supplied.shape == (len(self.boundary_ids), 2):
            boundary_values_np = supplied
        else:
            raise ValueError("boundary displacement has incompatible shape")
        out[self.boundary_ids] = boundary_values_np
        if not len(self.interior_ids):
            return out

        torch = self.torch
        boundary_values = torch.tensor(boundary_values_np, dtype=self.dtype, device=self.device)
        rhs = torch.zeros((len(self.interior_ids), 2), dtype=self.dtype, device=self.device)
        if self.ib_i.numel():
            rhs.index_add_(0, self.ib_i, boundary_values[self.ib_b])
        solution, iterations, residual = _pcg_solve(
            torch,
            self.matrix,
            self.diagonal,
            rhs,
            tolerance=self.cg_tolerance,
            max_iterations=self.cg_max_iterations,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        out[self.interior_ids] = solution.detach().cpu().numpy()
        self.cg_iterations_total += int(iterations)
        self.cg_residual_max = max(self.cg_residual_max, float(residual))
        self.solve_seconds += time.perf_counter() - started
        return out


__all__ = [
    "TorchOmegaAccelerator",
    "TorchHarmonicBoundaryResponse",
    "_resolve_torch_device",
]
