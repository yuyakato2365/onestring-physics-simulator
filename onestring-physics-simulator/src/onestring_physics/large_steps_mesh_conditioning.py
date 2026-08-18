"""Large-Steps mesh conditioning for open triangle surfaces.

This module applies the differential parameterization from
"Large Steps in Inverse Rendering of Geometry" before S -> Omega. It does not
change connectivity. Instead, interior vertices are optimized in the
Large-Steps coordinates u=(I+lambda L)v while boundary vertices stay fixed.
Accepted vertices are projected back to a local patch of the original surface,
so the stage redistributes samples without intentionally changing the target
shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

_EPS = 1.0e-12


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


def _uniform_laplacian(faces: np.ndarray, vertex_count: int):
    try:
        from scipy import sparse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for Large Steps conditioning") from exc
    edges = _unique_edges(faces)
    if not len(edges):
        return sparse.csr_matrix((vertex_count, vertex_count), dtype=float)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = -np.ones(len(row), dtype=float)
    degree = np.bincount(row, minlength=vertex_count).astype(float)
    row = np.concatenate([row, np.arange(vertex_count)])
    col = np.concatenate([col, np.arange(vertex_count)])
    data = np.concatenate([data, degree])
    return sparse.coo_matrix((data, (row, col)), shape=(vertex_count, vertex_count)).tocsr()


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


def _orientation_ratios(
    vertices: np.ndarray,
    faces: np.ndarray,
    original_normals: np.ndarray,
    original_area2: np.ndarray,
) -> np.ndarray:
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)[:, :3]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    signed = np.sum(cross * original_normals, axis=1)
    return signed / np.maximum(original_area2, _EPS)


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, _EPS)
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, _EPS)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), _EPS)
        return b + w * (c - b)
    denom = 1.0 / max(va + vb + vc, _EPS)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


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


def _project_vertices_locally(
    vertices: np.ndarray,
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    candidates: list[np.ndarray],
    movable_ids: np.ndarray,
) -> np.ndarray:
    result = np.asarray(vertices, dtype=float).copy()
    triangles = np.asarray(original_vertices, dtype=float)[np.asarray(original_faces, dtype=int)[:, :3]]
    for vertex_id in np.asarray(movable_ids, dtype=int):
        p = result[vertex_id]
        best = p
        best_distance = math.inf
        for face_id in candidates[vertex_id]:
            tri = triangles[int(face_id)]
            closest = _closest_point_on_triangle(p, tri[0], tri[1], tri[2])
            distance = float(np.dot(p - closest, p - closest))
            if distance < best_distance:
                best_distance = distance
                best = closest
        result[vertex_id] = best
    return result


def _surface_deviation(
    vertices: np.ndarray,
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    candidates: list[np.ndarray],
) -> np.ndarray:
    triangles = np.asarray(original_vertices, dtype=float)[np.asarray(original_faces, dtype=int)[:, :3]]
    out = np.zeros(len(vertices), dtype=float)
    for vertex_id, p in enumerate(np.asarray(vertices, dtype=float)):
        best = math.inf
        for face_id in candidates[vertex_id]:
            tri = triangles[int(face_id)]
            closest = _closest_point_on_triangle(p, tri[0], tri[1], tri[2])
            best = min(best, float(np.linalg.norm(p - closest)))
        out[vertex_id] = 0.0 if not np.isfinite(best) else best
    return out


def _torch_energy_gradient(
    vertices: np.ndarray,
    original_vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    original_vertex_normals: np.ndarray,
    target_edge_length: float,
    config: LargeStepsMeshConditioningConfig,
) -> tuple[float, np.ndarray, dict[str, float]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Large Steps mesh conditioning") from exc

    dtype = torch.float64
    v = torch.tensor(np.asarray(vertices), dtype=dtype, requires_grad=True)
    v0 = torch.tensor(np.asarray(original_vertices), dtype=dtype)
    f = torch.tensor(np.asarray(faces, dtype=np.int64)[:, :3], dtype=torch.long)
    e = torch.tensor(np.asarray(edges, dtype=np.int64), dtype=torch.long)
    normals = torch.tensor(np.asarray(original_vertex_normals), dtype=dtype)

    tri = v[f]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    area2 = torch.linalg.vector_norm(torch.cross(e01, tri[:, 2] - tri[:, 0], dim=1), dim=1)
    denom = (
        torch.sum(e01 * e01, dim=1)
        + torch.sum(e12 * e12, dim=1)
        + torch.sum(e20 * e20, dim=1)
    ).clamp_min(_EPS)
    quality = 2.0 * math.sqrt(3.0) * area2 / denom
    quality_loss = torch.mean((1.0 - quality) ** 2)

    edge_vec = v[e[:, 1]] - v[e[:, 0]]
    edge_len = torch.linalg.vector_norm(edge_vec, dim=1).clamp_min(_EPS)
    edge_loss = torch.mean(torch.log(edge_len / max(float(target_edge_length), _EPS)) ** 2)

    displacement = v - v0
    normal_displacement = torch.sum(displacement * normals, dim=1)
    normal_loss = torch.mean(normal_displacement * normal_displacement)
    position_loss = torch.mean(torch.sum(displacement * displacement, dim=1))

    total = (
        float(config.quality_weight) * quality_loss
        + float(config.edge_uniformity_weight) * edge_loss
        + float(config.surface_normal_weight) * normal_loss
        + float(config.position_weight) * position_loss
    )
    total.backward()
    gradient = v.grad.detach().cpu().numpy()
    terms = {
        "quality_loss": float(quality_loss.detach().cpu()),
        "edge_uniformity_loss": float(edge_loss.detach().cpu()),
        "surface_normal_loss": float(normal_loss.detach().cpu()),
        "position_loss": float(position_loss.detach().cpu()),
    }
    return float(total.detach().cpu()), gradient, terms


def condition_mesh_with_large_steps(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: LargeStepsMeshConditioningConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Redistribute an open triangle mesh using the Large Steps parameterization.

    Connectivity and boundary vertex positions are preserved exactly. The
    optimization variable is the reduced differential coordinate u_I for
    interior vertices. This is the same I + lambda L metric used by Large
    Steps, adapted to a fixed-boundary conditioning objective rather than an
    inverse-rendering loss.
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
    edges = _unique_edges(tris)
    target_edge = max(float(before["edge_length_median"]), _EPS)
    original_vertex_normals = _vertex_normals(xyz0, tris)
    original_face_normals, original_area2 = _original_face_normals(xyz0, tris)
    projection_candidates = _local_projection_candidates(tris, len(xyz0), cfg.projection_ring)

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
        **{f"large_steps_before_{key}": value for key, value in before.items()},
    }
    if not len(interior):
        metrics.update(
            {
                "large_steps_iteration_count": 0,
                "large_steps_termination_reason": "no_interior_vertices",
                "large_steps_runtime_seconds": float(time.perf_counter() - started),
                **{f"large_steps_after_{key}": value for key, value in before.items()},
            }
        )
        return xyz0.copy(), metrics

    try:
        from scipy import sparse
        from scipy.sparse import linalg as spla
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for Large Steps conditioning") from exc

    laplacian = _uniform_laplacian(tris, len(xyz0))
    matrix = sparse.eye(len(xyz0), format="csr") + float(cfg.lambda_) * laplacian
    m_ii = matrix[interior[:, None], interior].tocsc()
    m_ib = matrix[interior[:, None], boundary].tocsr()
    solve = spla.factorized(m_ii)
    boundary_positions = xyz0[boundary]
    boundary_coupling = np.asarray(m_ib @ boundary_positions, dtype=float)

    def encode(v: np.ndarray) -> np.ndarray:
        return np.asarray(m_ii @ v[interior] + boundary_coupling, dtype=float)

    def decode(u: np.ndarray) -> np.ndarray:
        v = xyz0.copy()
        rhs = np.asarray(u - boundary_coupling, dtype=float)
        for axis in range(3):
            v[interior, axis] = np.asarray(solve(rhs[:, axis]), dtype=float)
        v[boundary] = boundary_positions
        return v

    u = encode(xyz0)
    adam_m = np.zeros_like(u)
    adam_v = np.zeros_like(u)
    beta1, beta2 = 0.9, 0.999
    accepted = 0
    rejected = 0
    reason = "maximum_iterations"
    last_energy = math.inf
    initial_energy = None
    last_terms: dict[str, float] = {}

    for iteration in range(max(1, int(cfg.max_iterations))):
        current = decode(u)
        energy, grad_v, terms = _torch_energy_gradient(
            current,
            xyz0,
            tris,
            edges,
            original_vertex_normals,
            target_edge,
            cfg,
        )
        if initial_energy is None:
            initial_energy = float(energy)
        last_terms = terms
        grad_i = grad_v[interior]
        grad_u = np.zeros_like(grad_i)
        for axis in range(3):
            grad_u[:, axis] = np.asarray(solve(grad_i[:, axis]), dtype=float)

        t = iteration + 1
        adam_m = beta1 * adam_m + (1.0 - beta1) * grad_u
        adam_v = beta2 * adam_v + (1.0 - beta2) * (grad_u * grad_u)
        m_hat = adam_m / (1.0 - beta1**t)
        v_hat = adam_v / (1.0 - beta2**t)
        delta = -float(cfg.learning_rate) * target_edge * m_hat / (np.sqrt(v_hat) + 1.0e-8)
        rms = float(np.linalg.norm(delta) / math.sqrt(max(1, delta.size)))
        max_rms = 0.20 * target_edge
        if rms > max_rms:
            delta *= max_rms / rms

        accepted_candidate = None
        scale = 1.0
        for _ in range(max(1, int(cfg.line_search_max_steps))):
            candidate_u = u + scale * delta
            candidate = decode(candidate_u)
            if cfg.project_to_original_surface:
                candidate = _project_vertices_locally(
                    candidate,
                    xyz0,
                    tris,
                    projection_candidates,
                    interior,
                )
                candidate[boundary] = boundary_positions
                candidate_u = encode(candidate)
            ratios = _orientation_ratios(candidate, tris, original_face_normals, original_area2)
            if np.any(~np.isfinite(ratios)) or float(np.min(ratios)) <= float(cfg.minimum_orientation_ratio):
                rejected += 1
                scale *= 0.5
                continue
            candidate_energy, _candidate_grad, candidate_terms = _torch_energy_gradient(
                candidate,
                xyz0,
                tris,
                edges,
                original_vertex_normals,
                target_edge,
                cfg,
            )
            if not np.isfinite(candidate_energy) or candidate_energy > energy * (1.0 + 1.0e-9):
                rejected += 1
                scale *= 0.5
                continue
            accepted_candidate = (candidate_u, float(candidate_energy), candidate_terms)
            break

        if accepted_candidate is None:
            reason = "valid_decreasing_step_exhausted"
            break

        u, new_energy, last_terms = accepted_candidate
        accepted += 1
        relative = abs(energy - new_energy) / max(abs(energy), 1.0)
        last_energy = new_energy
        if relative <= float(cfg.relative_energy_tolerance):
            reason = "relative_energy_tolerance"
            break

    conditioned = decode(u)
    if cfg.project_to_original_surface:
        conditioned = _project_vertices_locally(
            conditioned, xyz0, tris, projection_candidates, interior
        )
        conditioned[boundary] = boundary_positions
    ratios = _orientation_ratios(conditioned, tris, original_face_normals, original_area2)
    if np.any(ratios <= 0.0):
        raise RuntimeError("Large Steps conditioning produced an inverted input triangle")

    after = _triangle_quality_metrics(conditioned, tris)
    deviation = _surface_deviation(conditioned, xyz0, tris, projection_candidates)
    metrics.update(
        {
            "large_steps_iteration_count": int(accepted),
            "large_steps_rejected_step_count": int(rejected),
            "large_steps_termination_reason": reason,
            "large_steps_initial_energy": float(initial_energy if initial_energy is not None else 0.0),
            "large_steps_final_energy": float(last_energy if np.isfinite(last_energy) else (initial_energy or 0.0)),
            "large_steps_min_orientation_ratio": float(np.min(ratios)),
            "large_steps_surface_deviation_mean": float(np.mean(deviation)),
            "large_steps_surface_deviation_p95": float(np.percentile(deviation, 95.0)),
            "large_steps_surface_deviation_max": float(np.max(deviation)),
            "large_steps_runtime_seconds": float(time.perf_counter() - started),
            **{f"large_steps_after_{key}": value for key, value in after.items()},
            **{f"large_steps_final_{key}": value for key, value in last_terms.items()},
        }
    )
    return conditioned, metrics


__all__ = [
    "LargeStepsMeshConditioningConfig",
    "condition_mesh_with_large_steps",
]
