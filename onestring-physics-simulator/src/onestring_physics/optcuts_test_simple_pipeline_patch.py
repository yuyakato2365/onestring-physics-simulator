"""Simplified experimental ``optcuts_test`` path.

User-requested flow:
    official OptCuts -> smooth the cut-boundary/seam -> regenerate Omega while
    preserving the same UV topology -> later clip boundary-crossing M2D panels
    instead of deleting whole cells.

This intentionally removes the earlier experiment that forced the whole Omega
onto a grid-cell outline.  Ordinary ``optcuts`` is untouched.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover
    coo_matrix = None
    spsolve = None


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _boundary_loop(parameterization: Any) -> list[int]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    stored = [int(v) for v in (parameterization.metrics.get("boundary_loop", []) or [])]
    if stored and all(0 <= v < len(uv) for v in stored):
        if len(stored) > 1 and stored[0] == stored[-1]:
            stored = stored[:-1]
        return stored

    faces = np.asarray(parameterization.uv_faces, dtype=int)
    incidence: dict[tuple[int, int], int] = {}
    for face in faces:
        ids = [int(x) for x in face]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            key = _edge_key(a, b)
            incidence[key] = incidence.get(key, 0) + 1
    edges = [edge for edge, count in incidence.items() if count == 1]
    if not edges:
        return []
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = min(adj)
    loop = [start]
    prev = -1
    cur = start
    for _ in range(len(edges) + 2):
        candidates = [n for n in adj.get(cur, []) if n != prev]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return loop


def _signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, dtype=float)[np.asarray(faces, dtype=int)]
    return 0.5 * (
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    )


def _is_valid_like(reference_uv: np.ndarray, candidate_uv: np.ndarray, faces: np.ndarray) -> bool:
    ref = _signed_areas(reference_uv, faces)
    cand = _signed_areas(candidate_uv, faces)
    nz = ref[np.abs(ref) > 1e-14]
    sign = 1.0 if len(nz) == 0 or float(np.median(nz)) >= 0.0 else -1.0
    scale = max(float(np.median(np.abs(nz))) if len(nz) else 1.0, 1e-12)
    return bool(np.all(np.isfinite(cand)) and np.all(sign * cand > max(1e-12, 1e-8 * scale)))


def _smooth_closed_loop(points: np.ndarray, iterations: int = 8, weight: float = 0.18) -> np.ndarray:
    out = np.asarray(points, dtype=float).copy()
    if len(out) < 4:
        return out
    # Conservative Laplacian smoothing.  The centroid and overall scale are
    # restored after every iteration so the seam is regularized rather than
    # progressively shrunk to a point.
    reference_center = np.mean(out, axis=0)
    reference_radius = float(np.sqrt(np.mean(np.sum((out - reference_center) ** 2, axis=1))))
    for _ in range(max(0, int(iterations))):
        prev = np.roll(out, 1, axis=0)
        nxt = np.roll(out, -1, axis=0)
        out = (1.0 - weight) * out + 0.5 * weight * (prev + nxt)
        center = np.mean(out, axis=0)
        radius = float(np.sqrt(np.mean(np.sum((out - center) ** 2, axis=1))))
        out -= center
        if radius > 1e-12 and reference_radius > 1e-12:
            out *= reference_radius / radius
        out += reference_center
    return out


def _harmonic_extend_boundary(
    uv: np.ndarray,
    faces: np.ndarray,
    boundary_ids: np.ndarray,
    boundary_target: np.ndarray,
) -> np.ndarray:
    n = len(uv)
    fixed = np.zeros(n, dtype=bool)
    fixed[boundary_ids] = True
    free = np.flatnonzero(~fixed)
    candidate = np.asarray(uv, dtype=float).copy()
    candidate[boundary_ids] = boundary_target
    if len(free) == 0:
        return candidate

    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in face]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            if a != b:
                edges.add(_edge_key(a, b))

    if coo_matrix is not None and spsolve is not None:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        degree = np.zeros(n, dtype=float)
        for a, b in edges:
            rows.extend([a, b])
            cols.extend([b, a])
            data.extend([-1.0, -1.0])
            degree[a] += 1.0
            degree[b] += 1.0
        rows.extend(range(n)); cols.extend(range(n)); data.extend(degree.tolist())
        L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        fixed_ids = np.flatnonzero(fixed)
        Lii = L[free][:, free]
        Lib = L[free][:, fixed_ids]
        rhs = -(Lib @ boundary_target)
        try:
            candidate[free, 0] = np.asarray(spsolve(Lii, np.asarray(rhs[:, 0]).reshape(-1)), dtype=float)
            candidate[free, 1] = np.asarray(spsolve(Lii, np.asarray(rhs[:, 1]).reshape(-1)), dtype=float)
            return candidate
        except Exception:
            pass

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adjacency[a].append(b); adjacency[b].append(a)
    for _ in range(700):
        old = candidate.copy()
        for vid in free:
            nbrs = adjacency[int(vid)]
            if nbrs:
                candidate[int(vid)] = np.mean(old[np.asarray(nbrs, dtype=int)], axis=0)
        candidate[boundary_ids] = boundary_target
        if float(np.max(np.linalg.norm(candidate - old, axis=1))) < 1e-10:
            break
    return candidate


def _smooth_and_regenerate_omega(parameterization: Any) -> Any:
    uv0 = np.asarray(parameterization.uv_vertices_2d, dtype=float).copy()
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    loop = _boundary_loop(parameterization)
    if len(loop) < 4:
        parameterization.metrics.update({
            "optcuts_test_seam_smoothing_applied": False,
            "optcuts_test_seam_smoothing_reason": "no_valid_boundary_loop",
        })
        return parameterization

    ids = np.asarray(loop, dtype=int)
    target = _smooth_closed_loop(uv0[ids], iterations=8, weight=0.18)

    # Backtrack the smoothing magnitude until the same cut mesh remains bijective.
    alpha = 1.0
    best = uv0.copy()
    accepted = 0.0
    for _ in range(18):
        boundary = uv0[ids] + alpha * (target - uv0[ids])
        candidate = _harmonic_extend_boundary(uv0, faces, ids, boundary)
        if _is_valid_like(uv0, candidate, faces):
            best = candidate
            accepted = alpha
            break
        alpha *= 0.5

    parameterization.uv_vertices_2d = best
    boundary_poly = best[ids]
    parameterization.omega_boundary = np.vstack([boundary_poly, boundary_poly[0]])
    parameterization.method = "optcuts_test"
    parameterization.metrics.update({
        "omega_parameterization_mode": "optcuts_test",
        "requested_omega_parameterization_mode": "optcuts_test",
        "optcuts_test_enabled": True,
        "optcuts_test_model": "official OptCuts -> smooth cut boundary -> harmonic Omega regeneration -> boundary-panel clipping",
        "optcuts_test_seam_topology_preserved": True,
        "optcuts_test_seam_smoothing_applied": bool(accepted > 0.0),
        "optcuts_test_seam_smoothing_alpha": float(accepted),
        "optcuts_test_seam_smoothing_iterations": 8,
        "optcuts_test_seam_smoothing_weight": 0.18,
        "optcuts_test_grid_outline_forcing_removed": True,
        "omega_boundary_shape": "free",
    })
    setattr(parameterization, "_optcuts_test_clip_boundary", True)
    print(
        f"[OPTCUTS-TEST] seam smoothing alpha={accepted:.6f} "
        f"boundary_vertices={len(ids)}; grid-outline forcing disabled"
    )
    return parameterization


def install_optcuts_test_simple_pipeline_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_simple_patch_installed", False):
        return
    base = pipeline._build_surface_parameterization

    def builder(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts_test":
            return base(surface, target, grid, params)
        ordinary = replace(params, omega_parameterization_mode="optcuts")
        parameterization = base(surface, target, grid, ordinary)
        return _smooth_and_regenerate_omega(parameterization)

    pipeline._build_surface_parameterization = builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = builder
    pipeline._onestring_optcuts_test_simple_patch_installed = True


__all__ = ["install_optcuts_test_simple_pipeline_patch"]
