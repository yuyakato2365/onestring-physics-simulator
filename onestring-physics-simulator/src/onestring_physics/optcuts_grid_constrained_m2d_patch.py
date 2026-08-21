"""M2D construction for the grid-constrained OptCuts flow.

The final OptCuts/OneString reparameterization already has straight seam copies
on an h-spaced orthogonal lattice. M2D therefore uses exactly the same lattice.
No second seam is added and no free OptCuts seam is snapped after the fact.

The lattice spacing h is fixed by tile_size, while the lattice phase/origin is
allowed to follow the optimized seam layout. This avoids an unnecessary snap to
world-zero without changing the fabrication unit.

Grid vertex ids are intentionally NOT compacted: downstream OneString code uses
QuadGrid connectivity, so M2D keeps the original fixed-grid numbering and only
filters faces.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .quad_grid import create_quad_grid


def _barycentric_2d(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = np.asarray(tri, dtype=float)
    v0 = b - a
    v1 = c - a
    v2 = np.asarray(point, dtype=float) - a
    denom = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denom) <= 1e-14:
        return None
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denom)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denom)
    return np.asarray([1.0 - u - v, u, v], dtype=float)


def _uv_domain_data(parameterization: Any) -> dict[str, Any]:
    cached = getattr(parameterization, "_onestring_grid_domain_data", None)
    if isinstance(cached, dict):
        return cached
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    triangles = uv[uf]
    tri_lo = np.min(triangles, axis=1)
    tri_hi = np.max(triangles, axis=1)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for face in uf:
        ids = [int(x) for x in face]
        for i in range(3):
            edge_count[tuple(sorted((ids[i], ids[(i + 1) % 3])))] += 1
    boundary = np.asarray(
        [[uv[a], uv[b]] for (a, b), count in edge_count.items() if count == 1],
        dtype=float,
    ) if edge_count else np.zeros((0, 2, 2), dtype=float)
    data = {
        "uv": uv,
        "uv_faces": uf,
        "triangles": triangles,
        "tri_lo": tri_lo,
        "tri_hi": tri_hi,
        "boundary_segments": boundary,
    }
    setattr(parameterization, "_onestring_grid_domain_data", data)
    return data


def _point_triangle_id(point: np.ndarray, data: dict[str, Any], tol: float = 1e-9) -> int:
    p = np.asarray(point, dtype=float)
    mask = np.all(p[None, :] >= data["tri_lo"] - tol, axis=1) & np.all(
        p[None, :] <= data["tri_hi"] + tol, axis=1
    )
    for tri_id in np.flatnonzero(mask):
        bary = _barycentric_2d(p, data["triangles"][int(tri_id)])
        if bary is not None and float(np.min(bary)) >= -tol:
            return int(tri_id)
    return -1


def _segment_hits_open_rect(
    a: np.ndarray,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    eps: float,
) -> bool:
    lower = np.asarray(lo, dtype=float) + float(eps)
    upper = np.asarray(hi, dtype=float) - float(eps)
    if np.any(lower >= upper):
        return False
    p0 = np.asarray(a, dtype=float)
    p1 = np.asarray(b, dtype=float)
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(2):
        if abs(float(d[axis])) <= 1e-15:
            if float(p0[axis]) <= float(lower[axis]) or float(p0[axis]) >= float(upper[axis]):
                return False
            continue
        inv = 1.0 / float(d[axis])
        enter = (float(lower[axis]) - float(p0[axis])) * inv
        leave = (float(upper[axis]) - float(p0[axis])) * inv
        if enter > leave:
            enter, leave = leave, enter
        t0 = max(t0, enter)
        t1 = min(t1, leave)
        if t0 > t1:
            return False
    return t1 >= 0.0 and t0 <= 1.0 and t0 <= t1


def _cell_crossed_by_uv_boundary(points: np.ndarray, data: dict[str, Any], h: float) -> bool:
    pts = np.asarray(points, dtype=float)
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    eps = max(1e-10, 1e-7 * float(h))
    for segment in np.asarray(data["boundary_segments"], dtype=float):
        a, b = segment
        seg_lo = np.minimum(a, b)
        seg_hi = np.maximum(a, b)
        if np.any(seg_hi < lo + eps) or np.any(seg_lo > hi - eps):
            continue
        if _segment_hits_open_rect(a, b, lo, hi, eps):
            return True
    return False


def install_optcuts_grid_constrained_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_grid_constrained_m2d_installed", False):
        return
    base_flatten = pipeline._flatten_to_domain
    base_build = pipeline._build_m2d

    def flatten_on_fixed_lattice(parameterization: Any, grid: Any, params: Any = None):
        domain = base_flatten(parameterization, grid, params)
        if not bool(getattr(parameterization, "metrics", {}).get("optcuts_grid_constrained", False)):
            return domain
        h = max(float(getattr(params, "tile_size", getattr(grid, "tile_size", 0.0))), 1e-8)
        uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
        margin = int(max(0, getattr(params, "omega_overlay_margin", 1))) if params is not None else 1
        metrics = dict(getattr(parameterization, "metrics", {}) or {})
        phase = np.asarray([
            float(metrics.get("grid_phase_u", 0.0)),
            float(metrics.get("grid_phase_v", 0.0)),
        ], dtype=float)
        lo = phase + np.floor((np.min(uv, axis=0) - phase) / h) * h - margin * h
        hi = phase + np.ceil((np.max(uv, axis=0) - phase) / h) * h + margin * h
        nx = max(1, int(round(float((hi[0] - lo[0]) / h))))
        ny = max(1, int(round(float((hi[1] - lo[1]) / h))))
        xs = lo[0] + np.arange(nx + 1, dtype=float) * h
        ys = lo[1] + np.arange(ny + 1, dtype=float) * h
        uu, vv = np.meshgrid(xs, ys, indexing="xy")
        domain.uv_vertices = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        domain.boundary = np.asarray(parameterization.omega_boundary, dtype=float)
        domain.split_lines = []
        domain.overlay_nx = int(nx)
        domain.overlay_ny = int(ny)
        domain.overlay_margin_tiles = int(margin)
        domain.overlay_step_u = float(h)
        domain.overlay_step_v = float(h)
        domain.original_requested_nx = int(getattr(grid, "nx", nx))
        domain.original_requested_ny = int(getattr(grid, "ny", ny))
        setattr(domain, "_optcuts_grid_constrained_parameterization", parameterization)
        setattr(domain, "_optcuts_grid_origin", np.asarray(lo, dtype=float))
        setattr(domain, "_optcuts_grid_phase", phase.copy())
        setattr(domain, "_optcuts_grid_unit", float(h))
        print(
            "[OPTCUTS-GRID-DOMAIN] "
            f"h={h:g} phase=({phase[0]:.6g},{phase[1]:.6g}) "
            f"origin=({lo[0]:.6g},{lo[1]:.6g}) grid={nx}x{ny} legacy_split=disabled"
        )
        return domain

    def build_from_constrained_uv(grid: Any, domain: Any, params: Any = None):
        parameterization = getattr(domain, "_optcuts_grid_constrained_parameterization", None)
        if parameterization is None:
            return base_build(grid, domain, params)
        h = float(getattr(domain, "_optcuts_grid_unit"))
        nx = int(getattr(domain, "overlay_nx"))
        ny = int(getattr(domain, "overlay_ny"))
        overlay_grid = create_quad_grid(nx, ny, h, float(getattr(grid, "gap_size", 0.0)))
        vertices = np.column_stack([
            np.asarray(domain.uv_vertices, dtype=float),
            np.zeros(len(domain.uv_vertices), dtype=float),
        ])
        data = _uv_domain_data(parameterization)
        kept: list[tuple[int, int, int, int]] = []
        kept_triangle_ids: list[int] = []
        outside = 0
        crossing = 0
        for tile in overlay_grid.tiles or []:
            face = tuple(int(x) for x in tile.vertex_ids)
            pts = vertices[np.asarray(face, dtype=int), :2]
            center = np.mean(pts, axis=0)
            tri_id = _point_triangle_id(center, data)
            if tri_id < 0:
                outside += 1
                continue
            if _cell_crossed_by_uv_boundary(pts, data, h):
                crossing += 1
                continue
            kept.append(face)
            kept_triangle_ids.append(int(tri_id))
        if not kept:
            raise RuntimeError("OPTCUTS_GRID_M2D_EMPTY: fixed lattice contains no valid UV cells")
        faces = np.asarray(kept, dtype=int)
        metrics = {
            "optcuts_grid_constrained_m2d": True,
            "optcuts_grid_unit": float(h),
            "optcuts_grid_origin": np.asarray(getattr(domain, "_optcuts_grid_origin"), dtype=float).tolist(),
            "optcuts_grid_phase": np.asarray(getattr(domain, "_optcuts_grid_phase"), dtype=float).tolist(),
            "optcuts_grid_nx": int(nx),
            "optcuts_grid_ny": int(ny),
            "optcuts_grid_total_cell_count": int(nx * ny),
            "optcuts_grid_kept_cell_count": int(len(faces)),
            "optcuts_grid_outside_cell_count": int(outside),
            "optcuts_grid_boundary_crossing_cell_count": int(crossing),
            "optcuts_grid_vertex_ids_preserved": True,
            "optcuts_grid_posthoc_seam_snap": False,
            "optcuts_grid_posthoc_seam_cell_deletion": False,
            "optcuts_grid_seam_aligned_before_m2d": True,
            "number_of_splits": 0,
            "split_locations": [],
            "m2d_grid_overlay": "same fixed h-lattice and optimized phase used by constrained OptCuts seam geometry",
        }
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or getattr(pipeline, "QuadMesh")
        out = cls(vertices, faces, overlay_grid, "M2D", metrics, [])
        setattr(out, "_optcuts_grid_cell_triangle_ids", np.asarray(kept_triangle_ids, dtype=int))
        print(
            "[OPTCUTS-GRID-M2D] "
            f"kept={len(faces)} outside={outside} boundary_crossing={crossing} h={h:g}"
        )
        return out

    pipeline._flatten_to_domain = flatten_on_fixed_lattice
    pipeline._build_m2d = build_from_constrained_uv
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._flatten_to_domain = flatten_on_fixed_lattice
        original._build_m2d = build_from_constrained_uv
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = flatten_on_fixed_lattice
            glb["_build_m2d"] = build_from_constrained_uv
    pipeline._onestring_optcuts_grid_constrained_m2d_installed = True


__all__ = ["install_optcuts_grid_constrained_m2d_patch"]
