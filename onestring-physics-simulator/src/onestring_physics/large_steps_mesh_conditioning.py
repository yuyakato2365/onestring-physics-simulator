"""CUDA-capable Large-Steps mesh conditioning for open triangle surfaces.

The stage applies the differential parameterization from
"Large Steps in Inverse Rendering of Geometry" before S -> Omega. Connectivity
is fixed, 3D boundary vertices are fixed, and interior vertices are optimized in
u=(I+lambda L)v coordinates. The Large-Steps linear system, objective,
orientation checks, local surface projection, and quality snapshots all run on
the selected PyTorch device (CUDA when available/requested).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

_EPS = 1.0e-12
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class LargeStepsMeshConditioningConfig:
    enabled: bool = True
    lambda_: float = 10.0
    max_iterations: int = 120
    learning_rate: float = 0.06
    quality_weight: float = 1.0
    edge_uniformity_weight: float = 0.12
    surface_normal_weight: float = 12.0
    position_weight: float = 0.01
    minimum_orientation_ratio: float = 0.03
    line_search_max_steps: int = 10
    project_to_original_surface: bool = True
    projection_ring: int = 2
    relative_energy_tolerance: float = 1.0e-8
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "auto"  # auto | float32 | float64
    cg_tolerance: float = 1.0e-6
    cg_max_iterations: int = 160
    progress_log_every: int = 5
    maximum_vertex_step_fraction: float = 0.20


def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is None:
        return
    try:
        callback(dict(payload))
    except Exception:
        return


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    tris = np.asarray(faces, dtype=int)[:, :3]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _boundary_vertices(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    tris = np.asarray(faces, dtype=int)[:, :3]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    if not len(boundary_edges):
        return np.asarray([], dtype=int)
    ids = np.unique(boundary_edges.reshape(-1))
    return ids[(ids >= 0) & (ids < int(vertex_count))]


def _triangle_quality_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    tri = xyz[tris]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    lengths = np.stack(
        [np.linalg.norm(e01, axis=1), np.linalg.norm(e12, axis=1), np.linalg.norm(e20, axis=1)],
        axis=1,
    )
    area2 = np.linalg.norm(np.cross(e01, tri[:, 2] - tri[:, 0]), axis=1)
    denom = np.sum(lengths * lengths, axis=1)
    quality = 2.0 * math.sqrt(3.0) * area2 / np.maximum(denom, _EPS)

    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]

    def angle(opposite: np.ndarray, side1: np.ndarray, side2: np.ndarray) -> np.ndarray:
        cosine = (side1 * side1 + side2 * side2 - opposite * opposite) / np.maximum(
            2.0 * side1 * side2, _EPS
        )
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = np.stack([angle(b, a, c), angle(c, a, b), angle(a, b, c)], axis=1)
    minimum_angles = np.min(angles, axis=1)

    edges = _unique_edges(tris)
    edge_lengths = np.linalg.norm(xyz[edges[:, 1]] - xyz[edges[:, 0]], axis=1)
    edge_mean = float(np.mean(edge_lengths)) if len(edge_lengths) else 0.0
    return {
        "triangle_quality_min": float(np.min(quality)) if len(quality) else 0.0,
        "triangle_quality_p05": float(np.percentile(quality, 5.0)) if len(quality) else 0.0,
        "triangle_quality_median": float(np.median(quality)) if len(quality) else 0.0,
        "minimum_angle_degrees": float(np.min(minimum_angles)) if len(minimum_angles) else 0.0,
        "angle_p05_degrees": float(np.percentile(minimum_angles, 5.0)) if len(minimum_angles) else 0.0,
        "edge_length_mean": edge_mean,
        "edge_length_median": float(np.median(edge_lengths)) if len(edge_lengths) else 0.0,
        "edge_length_cv": float(np.std(edge_lengths) / max(edge_mean, _EPS)) if len(edge_lengths) else 0.0,
    }


def _original_face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)[:, :3]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(area2[:, None], _EPS)
    return normals, area2


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    tri = xyz[tris]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(xyz)
    for corner in range(3):
        np.add.at(out, tris[:, corner], cross)
    out /= np.maximum(np.linalg.norm(out, axis=1)[:, None], _EPS)
    return out


def _local_projection_candidates(faces: np.ndarray, vertex_count: int, ring: int) -> list[np.ndarray]:
    tris = np.asarray(faces, dtype=int)[:, :3]
    incident: list[set[int]] = [set() for _ in range(vertex_count)]
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for face_id, face in enumerate(tris):
        ids = [int(v) for v in face]
        for v in ids:
            incident[v].add(face_id)
        for i, v in enumerate(ids):
            neighbors[v].update(ids[:i] + ids[i + 1 :])
    out: list[np.ndarray] = []
    depth = max(1, int(ring))
    for vertex in range(vertex_count):
        seen = {vertex}
        frontier = {vertex}
        for _ in range(depth):
            nxt: set[int] = set()
            for current in frontier:
                nxt.update(neighbors[current])
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
        face_ids: set[int] = set()
        for current in seen:
            face_ids.update(incident[current])
        out.append(np.asarray(sorted(face_ids), dtype=int))
    return out


def _projection_candidate_matrix(
    candidates: list[np.ndarray], movable_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = [np.asarray(candidates[int(v)], dtype=int) for v in np.asarray(movable_ids, dtype=int)]
    width = max((len(ids) for ids in selected), default=1)
    face_ids = np.zeros((len(selected), width), dtype=np.int64)
    valid = np.zeros((len(selected), width), dtype=bool)
    for row, ids in enumerate(selected):
        if len(ids):
            face_ids[row, : len(ids)] = ids
            valid[row, : len(ids)] = True
    return face_ids, valid


def _resolve_device_and_dtype(config: LargeStepsMeshConditioningConfig):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Large Steps mesh conditioning") from exc

    requested = str(config.device).strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Large Steps device must be one of: auto, cuda, cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Large Steps device='cuda' was requested, but torch.cuda.is_available() is False"
            )
        device = torch.device("cuda")
    elif requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    dtype_name = str(config.dtype).strip().lower()
    if dtype_name == "auto":
        dtype = torch.float32 if device.type == "cuda" else torch.float64
    elif dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "float64":
        dtype = torch.float64
    else:
        raise ValueError("Large Steps dtype must be one of: auto, float32, float64")
    return torch, device, dtype


def _build_reduced_system(
    *,
    torch: Any,
    faces: np.ndarray,
    vertices: np.ndarray,
    interior: np.ndarray,
    boundary: np.ndarray,
    lambda_: float,
    device: Any,
    dtype: Any,
):
    vertex_count = len(vertices)
    edges = _unique_edges(faces)
    local = np.full(vertex_count, -1, dtype=np.int64)
    local[np.asarray(interior, dtype=int)] = np.arange(len(interior), dtype=np.int64)
    boundary_mask = np.zeros(vertex_count, dtype=bool)
    boundary_mask[np.asarray(boundary, dtype=int)] = True

    degree = np.zeros(vertex_count, dtype=np.float64)
    if len(edges):
        np.add.at(degree, edges[:, 0], 1.0)
        np.add.at(degree, edges[:, 1], 1.0)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    coupling = np.zeros((len(interior), 3), dtype=np.float64)

    diag = 1.0 + float(lambda_) * degree[np.asarray(interior, dtype=int)]
    for i, value in enumerate(diag):
        rows.append(i)
        cols.append(i)
        values.append(float(value))

    for a_raw, b_raw in edges:
        a, b = int(a_raw), int(b_raw)
        ia, ib = int(local[a]), int(local[b])
        if ia >= 0 and ib >= 0:
            rows.extend([ia, ib])
            cols.extend([ib, ia])
            values.extend([-float(lambda_), -float(lambda_)])
        elif ia >= 0 and boundary_mask[b]:
            coupling[ia] += -float(lambda_) * np.asarray(vertices[b], dtype=float)
        elif ib >= 0 and boundary_mask[a]:
            coupling[ib] += -float(lambda_) * np.asarray(vertices[a], dtype=float)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals = torch.tensor(values, dtype=dtype, device=device)
    matrix = torch.sparse_coo_tensor(
        indices, vals, (len(interior), len(interior)), device=device, dtype=dtype
    ).coalesce()
    diagonal = torch.tensor(diag, dtype=dtype, device=device)
    boundary_coupling = torch.tensor(coupling, dtype=dtype, device=device)
    return matrix, diagonal, boundary_coupling


def _pcg_solve(
    torch: Any,
    matrix: Any,
    diagonal: Any,
    rhs: Any,
    *,
    tolerance: float,
    max_iterations: int,
    initial: Any | None = None,
) -> tuple[Any, int, float]:
    if rhs.numel() == 0:
        return rhs.clone(), 0, 0.0
    x = torch.zeros_like(rhs) if initial is None else initial.clone()
    r = rhs - torch.sparse.mm(matrix, x)
    norm_b = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1.0e-20)
    z = r / diagonal[:, None].clamp_min(1.0e-20)
    p = z.clone()
    rz = torch.sum(r * z, dim=0)
    last_relative = math.inf
    check_every = 8

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        ap = torch.sparse.mm(matrix, p)
        denom = torch.sum(p * ap, dim=0)
        safe_denom = torch.where(
            torch.abs(denom) < 1.0e-30,
            torch.full_like(denom, 1.0e-30),
            denom,
        )
        alpha = rz / safe_denom
        x = x + p * alpha[None, :]
        r = r - ap * alpha[None, :]

        if iteration == 1 or iteration % check_every == 0 or iteration == int(max_iterations):
            relative = torch.max(torch.linalg.vector_norm(r, dim=0) / norm_b)
            last_relative = float(relative.detach().item())
            if last_relative <= float(tolerance):
                return x, iteration, last_relative

        z = r / diagonal[:, None].clamp_min(1.0e-20)
        rz_new = torch.sum(r * z, dim=0)
        safe_rz = torch.where(
            torch.abs(rz) < 1.0e-30,
            torch.full_like(rz, 1.0e-30),
            rz,
        )
        beta = rz_new / safe_rz
        p = z + p * beta[None, :]
        rz = rz_new

    return x, int(max_iterations), float(last_relative)


def _closest_points_to_triangles_torch(torch: Any, points: Any, triangles: Any) -> Any:
    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = torch.sum(ab * ap, dim=-1)
    d2 = torch.sum(ac * ap, dim=-1)

    bp = points - b
    d3 = torch.sum(ab * bp, dim=-1)
    d4 = torch.sum(ac * bp, dim=-1)

    cp = points - c
    d5 = torch.sum(ab * cp, dim=-1)
    d6 = torch.sum(ac * cp, dim=-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    eps = 1.0e-20
    denom_face = (va + vb + vc).clamp_min(eps)
    v_face = vb / denom_face
    w_face = vc / denom_face
    result = a + ab * v_face[..., None] + ac * w_face[..., None]

    denom_bc = (d4 - d3) + (d5 - d6)
    w_bc = (d4 - d3) / torch.where(
        torch.abs(denom_bc) < eps, torch.full_like(denom_bc, eps), denom_bc
    )
    q_bc = b + (c - b) * w_bc[..., None]
    cond_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    result = torch.where(cond_bc[..., None], q_bc, result)

    denom_ac = d2 - d6
    w_ac = d2 / torch.where(
        torch.abs(denom_ac) < eps, torch.full_like(denom_ac, eps), denom_ac
    )
    q_ac = a + ac * w_ac[..., None]
    cond_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    result = torch.where(cond_ac[..., None], q_ac, result)

    cond_c = (d6 >= 0.0) & (d5 <= d6)
    result = torch.where(cond_c[..., None], c, result)

    denom_ab = d1 - d3
    v_ab = d1 / torch.where(
        torch.abs(denom_ab) < eps, torch.full_like(denom_ab, eps), denom_ab
    )
    q_ab = a + ab * v_ab[..., None]
    cond_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    result = torch.where(cond_ab[..., None], q_ab, result)

    cond_b = (d3 >= 0.0) & (d4 <= d3)
    result = torch.where(cond_b[..., None], b, result)

    cond_a = (d1 <= 0.0) & (d2 <= 0.0)
    result = torch.where(cond_a[..., None], a, result)
    return result


def _project_movable_vertices_torch(
    torch: Any,
    vertices: Any,
    original_triangles: Any,
    movable_ids: Any,
    candidate_face_ids: Any,
    candidate_valid: Any,
) -> tuple[Any, Any]:
    if movable_ids.numel() == 0:
        return vertices, torch.zeros((0,), dtype=vertices.dtype, device=vertices.device)
    points = vertices[movable_ids]
    candidate_triangles = original_triangles[candidate_face_ids]
    expanded_points = points[:, None, :].expand(-1, candidate_triangles.shape[1], -1)
    closest = _closest_points_to_triangles_torch(torch, expanded_points, candidate_triangles)
    distance2 = torch.sum((expanded_points - closest) ** 2, dim=-1)
    infinity = torch.full_like(distance2, float("inf"))
    distance2 = torch.where(candidate_valid, distance2, infinity)
    best = torch.argmin(distance2, dim=1)
    row = torch.arange(len(points), device=vertices.device)
    projected = closest[row, best]
    best_distance2 = distance2[row, best]
    out = vertices.clone()
    out[movable_ids] = projected
    return out, best_distance2


def _orientation_ratios_torch(
    torch: Any,
    vertices: Any,
    faces: Any,
    original_face_normals: Any,
    original_area2: Any,
) -> Any:
    tri = vertices[faces]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    signed = torch.sum(cross * original_face_normals, dim=1)
    return signed / original_area2.clamp_min(1.0e-20)


def _energy_torch(
    torch: Any,
    vertices: Any,
    original_vertices: Any,
    faces: Any,
    edges: Any,
    original_vertex_normals: Any,
    target_edge_length: float,
    config: LargeStepsMeshConditioningConfig,
) -> tuple[Any, dict[str, Any]]:
    tri = vertices[faces]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    area2 = torch.linalg.vector_norm(
        torch.cross(e01, tri[:, 2] - tri[:, 0], dim=1), dim=1
    )
    denom = (
        torch.sum(e01 * e01, dim=1)
        + torch.sum(e12 * e12, dim=1)
        + torch.sum(e20 * e20, dim=1)
    ).clamp_min(1.0e-20)
    quality = 2.0 * math.sqrt(3.0) * area2 / denom
    quality_loss = torch.mean((1.0 - quality) ** 2)

    edge_vec = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_len = torch.linalg.vector_norm(edge_vec, dim=1).clamp_min(1.0e-20)
    edge_loss = torch.mean(
        torch.log(edge_len / max(float(target_edge_length), _EPS)) ** 2
    )

    displacement = vertices - original_vertices
    normal_displacement = torch.sum(displacement * original_vertex_normals, dim=1)
    normal_loss = torch.mean(normal_displacement * normal_displacement)
    position_loss = torch.mean(torch.sum(displacement * displacement, dim=1))

    total = (
        float(config.quality_weight) * quality_loss
        + float(config.edge_uniformity_weight) * edge_loss
        + float(config.surface_normal_weight) * normal_loss
        + float(config.position_weight) * position_loss
    )
    return total, {
        "quality_loss": quality_loss,
        "edge_uniformity_loss": edge_loss,
        "surface_normal_loss": normal_loss,
        "position_loss": position_loss,
    }


def _quality_snapshot_torch(torch: Any, vertices: Any, faces: Any, edges: Any) -> dict[str, float]:
    with torch.no_grad():
        tri = vertices[faces]
        e01 = tri[:, 1] - tri[:, 0]
        e12 = tri[:, 2] - tri[:, 1]
        e20 = tri[:, 0] - tri[:, 2]
        lengths = torch.stack(
            [
                torch.linalg.vector_norm(e01, dim=1),
                torch.linalg.vector_norm(e12, dim=1),
                torch.linalg.vector_norm(e20, dim=1),
            ],
            dim=1,
        ).clamp_min(1.0e-20)
        area2 = torch.linalg.vector_norm(
            torch.cross(e01, tri[:, 2] - tri[:, 0], dim=1), dim=1
        )
        quality = 2.0 * math.sqrt(3.0) * area2 / torch.sum(lengths * lengths, dim=1).clamp_min(1.0e-20)

        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]

        def angle(opposite: Any, side1: Any, side2: Any) -> Any:
            cosine = (side1 * side1 + side2 * side2 - opposite * opposite) / (
                2.0 * side1 * side2
            ).clamp_min(1.0e-20)
            return torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))

        minimum_angles = torch.min(
            torch.stack([angle(b, a, c), angle(c, a, b), angle(a, b, c)], dim=1),
            dim=1,
        ).values
        edge_lengths = torch.linalg.vector_norm(
            vertices[edges[:, 1]] - vertices[edges[:, 0]], dim=1
        )
        edge_mean = torch.mean(edge_lengths)
        edge_cv = torch.std(edge_lengths, unbiased=False) / edge_mean.clamp_min(1.0e-20)
        return {
            "minimum_angle_degrees": float(torch.min(minimum_angles).item()),
            "triangle_quality_p05": float(torch.quantile(quality, 0.05).item()),
            "edge_length_cv": float(edge_cv.item()),
        }


def condition_mesh_with_large_steps(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: LargeStepsMeshConditioningConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Redistribute an open triangle mesh using CUDA-capable Large Steps.

    The selected device owns the Large-Steps sparse system, PCG solves, objective,
    local projection, and validity checks. With device="auto", CUDA is preferred
    whenever PyTorch reports an available CUDA device.
    """
    cfg = config or LargeStepsMeshConditioningConfig()
    xyz0 = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    if xyz0.ndim != 2 or xyz0.shape[1] != 3 or len(xyz0) < 3:
        raise ValueError("Large Steps conditioning requires an Nx3 vertex array")
    if tris.ndim != 2 or tris.shape[1] != 3 or not len(tris):
        raise ValueError("Large Steps conditioning requires triangle faces")
    if not cfg.enabled:
        return xyz0.copy(), {"large_steps_conditioning_enabled": False}

    started = time.perf_counter()
    before = _triangle_quality_metrics(xyz0, tris)
    boundary = _boundary_vertices(tris, len(xyz0))
    boundary_mask = np.zeros(len(xyz0), dtype=bool)
    boundary_mask[boundary] = True
    interior = np.flatnonzero(~boundary_mask)
    edges_np = _unique_edges(tris)
    target_edge = max(float(before["edge_length_median"]), _EPS)
    original_vertex_normals_np = _vertex_normals(xyz0, tris)
    original_face_normals_np, original_area2_np = _original_face_normals(xyz0, tris)
    projection_candidates = _local_projection_candidates(tris, len(xyz0), cfg.projection_ring)

    torch, device, dtype = _resolve_device_and_dtype(cfg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    dtype_name = str(dtype).split(".")[-1]

    metrics: dict[str, Any] = {
        "large_steps_conditioning_enabled": True,
        "large_steps_parameterization": "u=(I+lambda*L)v",
        "large_steps_laplacian": "uniform_combinatorial",
        "large_steps_lambda": float(cfg.lambda_),
        "large_steps_boundary_fixed": True,
        "large_steps_connectivity_changed": False,
        "large_steps_project_to_original_surface": bool(cfg.project_to_original_surface),
        "large_steps_requested_iterations": int(cfg.max_iterations),
        "large_steps_boundary_vertex_count": int(len(boundary)),
        "large_steps_interior_vertex_count": int(len(interior)),
        "large_steps_compute_device_requested": str(cfg.device),
        "large_steps_compute_device": str(device),
        "large_steps_cuda_used": bool(device.type == "cuda"),
        "large_steps_device_name": gpu_name,
        "large_steps_dtype": dtype_name,
        "large_steps_cg_tolerance": float(cfg.cg_tolerance),
        "large_steps_cg_max_iterations": int(cfg.cg_max_iterations),
        **{f"large_steps_before_{key}": value for key, value in before.items()},
    }

    _emit_progress(
        progress_callback,
        event="setup",
        iteration=0,
        max_iterations=int(cfg.max_iterations),
        fraction=0.0,
        device=str(device),
        device_name=gpu_name,
        dtype=dtype_name,
        vertex_count=int(len(xyz0)),
        face_count=int(len(tris)),
        interior_vertex_count=int(len(interior)),
        boundary_vertex_count=int(len(boundary)),
        minimum_angle_degrees=float(before["minimum_angle_degrees"]),
        triangle_quality_p05=float(before["triangle_quality_p05"]),
        elapsed_seconds=float(time.perf_counter() - started),
    )

    if not len(interior):
        metrics.update(
            {
                "large_steps_iteration_count": 0,
                "large_steps_termination_reason": "no_interior_vertices",
                "large_steps_runtime_seconds": float(time.perf_counter() - started),
                **{f"large_steps_after_{key}": value for key, value in before.items()},
            }
        )
        _emit_progress(progress_callback, event="done", fraction=1.0, **metrics)
        return xyz0.copy(), metrics

    xyz0_t = torch.tensor(xyz0, dtype=dtype, device=device)
    faces_t = torch.tensor(tris, dtype=torch.long, device=device)
    edges_t = torch.tensor(edges_np, dtype=torch.long, device=device)
    interior_t = torch.tensor(interior, dtype=torch.long, device=device)
    boundary_t = torch.tensor(boundary, dtype=torch.long, device=device)
    vertex_normals_t = torch.tensor(original_vertex_normals_np, dtype=dtype, device=device)
    face_normals_t = torch.tensor(original_face_normals_np, dtype=dtype, device=device)
    original_area2_t = torch.tensor(original_area2_np, dtype=dtype, device=device)
    original_triangles_t = xyz0_t[faces_t]

    candidate_ids_np, candidate_valid_np = _projection_candidate_matrix(
        projection_candidates, interior
    )
    candidate_ids_t = torch.tensor(candidate_ids_np, dtype=torch.long, device=device)
    candidate_valid_t = torch.tensor(candidate_valid_np, dtype=torch.bool, device=device)

    matrix, diagonal, boundary_coupling = _build_reduced_system(
        torch=torch,
        faces=tris,
        vertices=xyz0,
        interior=interior,
        boundary=boundary,
        lambda_=float(cfg.lambda_),
        device=device,
        dtype=dtype,
    )

    def encode(v: Any) -> Any:
        return torch.sparse.mm(matrix, v[interior_t]) + boundary_coupling

    current = xyz0_t.clone()
    u = encode(current)
    adam_m = torch.zeros_like(u)
    adam_v = torch.zeros_like(u)
    beta1, beta2 = 0.9, 0.999
    accepted = 0
    rejected = 0
    reason = "maximum_iterations"
    initial_energy_value: float | None = None
    final_energy_value = math.inf
    last_terms: dict[str, float] = {}
    max_gradient_cg_iterations = 0
    max_direction_cg_iterations = 0
    max_cg_relative_residual = 0.0
    last_step_scale = 0.0

    for iteration in range(max(1, int(cfg.max_iterations))):
        current_var = current.detach().requires_grad_(True)
        energy_t, terms_t = _energy_torch(
            torch,
            current_var,
            xyz0_t,
            faces_t,
            edges_t,
            vertex_normals_t,
            target_edge,
            cfg,
        )
        grad_v = torch.autograd.grad(energy_t, current_var, create_graph=False)[0]
        energy = float(energy_t.detach().item())
        if initial_energy_value is None:
            initial_energy_value = energy
        grad_i = grad_v[interior_t].detach()
        grad_u, cg_grad_iters, cg_grad_residual = _pcg_solve(
            torch,
            matrix,
            diagonal,
            grad_i,
            tolerance=float(cfg.cg_tolerance),
            max_iterations=int(cfg.cg_max_iterations),
        )
        max_gradient_cg_iterations = max(max_gradient_cg_iterations, int(cg_grad_iters))
        max_cg_relative_residual = max(max_cg_relative_residual, float(cg_grad_residual))

        t = iteration + 1
        adam_m = beta1 * adam_m + (1.0 - beta1) * grad_u
        adam_v = beta2 * adam_v + (1.0 - beta2) * (grad_u * grad_u)
        m_hat = adam_m / (1.0 - beta1**t)
        v_hat = adam_v / (1.0 - beta2**t)
        delta_u = -float(cfg.learning_rate) * target_edge * m_hat / (
            torch.sqrt(v_hat) + 1.0e-8
        )

        delta_v, cg_dir_iters, cg_dir_residual = _pcg_solve(
            torch,
            matrix,
            diagonal,
            delta_u,
            tolerance=float(cfg.cg_tolerance),
            max_iterations=int(cfg.cg_max_iterations),
        )
        max_direction_cg_iterations = max(max_direction_cg_iterations, int(cg_dir_iters))
        max_cg_relative_residual = max(max_cg_relative_residual, float(cg_dir_residual))

        rms = torch.linalg.vector_norm(delta_v) / math.sqrt(max(1, delta_v.numel()))
        max_rms = float(cfg.maximum_vertex_step_fraction) * target_edge
        rms_value = float(rms.detach().item())
        if rms_value > max_rms > 0.0:
            delta_v = delta_v * (max_rms / rms_value)

        accepted_candidate = None
        scale = 1.0
        for _line_search_step in range(max(1, int(cfg.line_search_max_steps))):
            with torch.no_grad():
                candidate = current.clone()
                candidate[interior_t] = current[interior_t] + scale * delta_v
                if cfg.project_to_original_surface:
                    candidate, _projection_distance2 = _project_movable_vertices_torch(
                        torch,
                        candidate,
                        original_triangles_t,
                        interior_t,
                        candidate_ids_t,
                        candidate_valid_t,
                    )
                    candidate[boundary_t] = xyz0_t[boundary_t]
                ratios = _orientation_ratios_torch(
                    torch, candidate, faces_t, face_normals_t, original_area2_t
                )
                min_ratio = float(torch.min(ratios).item())
                if not math.isfinite(min_ratio) or min_ratio <= float(cfg.minimum_orientation_ratio):
                    rejected += 1
                    scale *= 0.5
                    continue
                candidate_energy_t, candidate_terms_t = _energy_torch(
                    torch,
                    candidate,
                    xyz0_t,
                    faces_t,
                    edges_t,
                    vertex_normals_t,
                    target_edge,
                    cfg,
                )
                candidate_energy = float(candidate_energy_t.item())
                if not math.isfinite(candidate_energy) or candidate_energy > energy * (1.0 + 1.0e-9):
                    rejected += 1
                    scale *= 0.5
                    continue
                candidate_u = encode(candidate)
                accepted_candidate = (
                    candidate,
                    candidate_u,
                    candidate_energy,
                    {key: float(value.item()) for key, value in candidate_terms_t.items()},
                    min_ratio,
                )
                break

        if accepted_candidate is None:
            reason = "valid_decreasing_step_exhausted"
            _emit_progress(
                progress_callback,
                event="stalled",
                iteration=int(iteration + 1),
                max_iterations=int(cfg.max_iterations),
                fraction=float((iteration + 1) / max(1, int(cfg.max_iterations))),
                energy=float(energy),
                accepted_steps=int(accepted),
                rejected_steps=int(rejected),
                elapsed_seconds=float(time.perf_counter() - started),
                device=str(device),
                device_name=gpu_name,
            )
            break

        current, u, new_energy, last_terms, min_ratio = accepted_candidate
        accepted += 1
        last_step_scale = float(scale)
        relative = abs(energy - new_energy) / max(abs(energy), 1.0)
        final_energy_value = float(new_energy)

        should_log = (
            iteration == 0
            or (iteration + 1) % max(1, int(cfg.progress_log_every)) == 0
            or iteration + 1 == int(cfg.max_iterations)
        )
        if should_log:
            snapshot = _quality_snapshot_torch(torch, current, faces_t, edges_t)
            memory_mb = 0.0
            if device.type == "cuda":
                memory_mb = float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0))
            _emit_progress(
                progress_callback,
                event="iteration",
                iteration=int(iteration + 1),
                max_iterations=int(cfg.max_iterations),
                fraction=float((iteration + 1) / max(1, int(cfg.max_iterations))),
                energy=float(new_energy),
                relative_energy_change=float(relative),
                minimum_angle_degrees=float(snapshot["minimum_angle_degrees"]),
                triangle_quality_p05=float(snapshot["triangle_quality_p05"]),
                edge_length_cv=float(snapshot["edge_length_cv"]),
                minimum_orientation_ratio=float(min_ratio),
                accepted_steps=int(accepted),
                rejected_steps=int(rejected),
                line_search_step_scale=float(scale),
                cg_gradient_iterations=int(cg_grad_iters),
                cg_direction_iterations=int(cg_dir_iters),
                cg_relative_residual=max(float(cg_grad_residual), float(cg_dir_residual)),
                elapsed_seconds=float(time.perf_counter() - started),
                device=str(device),
                device_name=gpu_name,
                cuda_memory_allocated_mb=float(memory_mb),
            )

        if relative <= float(cfg.relative_energy_tolerance):
            reason = "relative_energy_tolerance"
            break

    conditioned_t = current.detach()
    with torch.no_grad():
        if cfg.project_to_original_surface:
            conditioned_t, final_projection_distance2 = _project_movable_vertices_torch(
                torch,
                conditioned_t,
                original_triangles_t,
                interior_t,
                candidate_ids_t,
                candidate_valid_t,
            )
            conditioned_t[boundary_t] = xyz0_t[boundary_t]
        else:
            final_projection_distance2 = torch.zeros(
                (len(interior),), dtype=dtype, device=device
            )
        ratios = _orientation_ratios_torch(
            torch, conditioned_t, faces_t, face_normals_t, original_area2_t
        )
        min_ratio = float(torch.min(ratios).item())
        if min_ratio <= 0.0:
            raise RuntimeError("Large Steps conditioning produced an inverted input triangle")

        deviation = torch.zeros((len(xyz0),), dtype=dtype, device=device)
        if len(interior):
            deviation[interior_t] = torch.sqrt(final_projection_distance2.clamp_min(0.0))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    conditioned = conditioned_t.cpu().numpy().astype(float, copy=False)
    after = _triangle_quality_metrics(conditioned, tris)
    runtime = float(time.perf_counter() - started)
    metrics.update(
        {
            "large_steps_iteration_count": int(accepted),
            "large_steps_rejected_step_count": int(rejected),
            "large_steps_termination_reason": reason,
            "large_steps_initial_energy": float(initial_energy_value or 0.0),
            "large_steps_final_energy": float(
                final_energy_value if math.isfinite(final_energy_value) else (initial_energy_value or 0.0)
            ),
            "large_steps_min_orientation_ratio": float(min_ratio),
            "large_steps_surface_deviation_mean": float(torch.mean(deviation).item()),
            "large_steps_surface_deviation_p95": float(torch.quantile(deviation, 0.95).item()),
            "large_steps_surface_deviation_max": float(torch.max(deviation).item()),
            "large_steps_runtime_seconds": runtime,
            "large_steps_last_step_scale": float(last_step_scale),
            "large_steps_max_gradient_cg_iterations": int(max_gradient_cg_iterations),
            "large_steps_max_direction_cg_iterations": int(max_direction_cg_iterations),
            "large_steps_max_cg_relative_residual": float(max_cg_relative_residual),
            **{f"large_steps_after_{key}": value for key, value in after.items()},
            **{f"large_steps_final_{key}": value for key, value in last_terms.items()},
        }
    )
    if device.type == "cuda":
        metrics["large_steps_cuda_peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )

    _emit_progress(
        progress_callback,
        event="done",
        iteration=int(accepted),
        max_iterations=int(cfg.max_iterations),
        fraction=1.0,
        energy=float(metrics["large_steps_final_energy"]),
        minimum_angle_degrees=float(after["minimum_angle_degrees"]),
        triangle_quality_p05=float(after["triangle_quality_p05"]),
        edge_length_cv=float(after["edge_length_cv"]),
        minimum_orientation_ratio=float(min_ratio),
        accepted_steps=int(accepted),
        rejected_steps=int(rejected),
        elapsed_seconds=runtime,
        termination_reason=reason,
        device=str(device),
        device_name=gpu_name,
        cuda_used=bool(device.type == "cuda"),
    )
    return conditioned, metrics


__all__ = [
    "LargeStepsMeshConditioningConfig",
    "condition_mesh_with_large_steps",
]
