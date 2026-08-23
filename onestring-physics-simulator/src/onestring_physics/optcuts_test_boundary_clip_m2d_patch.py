"""Boundary-panel clipping for ``optcuts_test`` M2D.

Unlike the legacy center-crop policy, a grid cell that intersects Omega is kept.
Its panel polygon is clipped to the Omega boundary and represented as four
vertices so the existing downstream QuadMesh pipeline can continue to operate.
Only cells with no intersection with Omega are removed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .quad_grid import create_quad_grid


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            denom = float(y1 - y0)
            if abs(denom) <= 1e-20:
                continue
            x_cross = float(x0 + (x1 - x0) * (y - y0) / denom)
            if x < x_cross:
                inside = not inside
    return inside


def _segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = np.asarray(c, float); d = np.asarray(d, float)
    r = b - a; s = d - c
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) <= 1e-12:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / cross)
    u = float((q[0] * r[1] - q[1] * r[0]) / cross)
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return a + np.clip(t, 0.0, 1.0) * r
    return None


def _inside_axis_aligned_cell(point: np.ndarray, corners: np.ndarray) -> bool:
    p = np.asarray(point, float)
    lo = np.min(corners, axis=0) - 1e-10
    hi = np.max(corners, axis=0) + 1e-10
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _dedupe(points: list[np.ndarray], tol: float = 1e-9) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in points:
        q = np.asarray(p, float)
        if not any(float(np.linalg.norm(q - r)) <= tol for r in out):
            out.append(q)
    return out


def _intersection_polygon(cell: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Collect the local connected intersection polygon between one cell and Omega.

    For the intended fine-grid use the smoothed boundary crosses a cell at most
    once.  The construction is deliberately local: inside cell corners, boundary
    vertices inside the cell, and all cell-edge/boundary intersections are
    collected and cyclically ordered.
    """
    cell = np.asarray(cell, float)
    boundary = np.asarray(boundary, float)
    if len(boundary) > 1 and np.linalg.norm(boundary[0] - boundary[-1]) <= 1e-10:
        boundary = boundary[:-1]
    pts: list[np.ndarray] = []
    for p in cell:
        if _point_in_polygon(p, boundary):
            pts.append(p)
    for p in boundary:
        if _inside_axis_aligned_cell(p, cell):
            pts.append(p)
    cell_edges = [(cell[i], cell[(i + 1) % 4]) for i in range(4)]
    boundary_edges = [(boundary[i], boundary[(i + 1) % len(boundary)]) for i in range(len(boundary))]
    for a, b in cell_edges:
        for c, d in boundary_edges:
            q = _segment_intersection(a, b, c, d)
            if q is not None:
                pts.append(q)
    pts = _dedupe(pts)
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(pts, dtype=float)
    center = np.mean(arr, axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    arr = arr[np.argsort(angles)]
    return arr


def _polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def _quadify(poly: np.ndarray) -> np.ndarray:
    """Represent a clipped boundary panel with four perimeter vertices.

    * 4 vertices: exact.
    * 3 vertices: insert a midpoint on the longest edge; geometry is unchanged.
    * >4 vertices: remove the least geometrically significant perimeter vertices
      until four remain.  This only affects rare cells containing a boundary
      corner; the clipping metrics record how often this approximation is used.
    """
    p = np.asarray(poly, float).copy()
    if len(p) == 3:
        lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        i = int(np.argmax(lengths))
        mid = 0.5 * (p[i] + p[(i + 1) % 3])
        values: list[np.ndarray] = []
        for j in range(3):
            values.append(p[j])
            if j == i:
                values.append(mid)
        return np.asarray(values, float)
    while len(p) > 4:
        scores = []
        for i in range(len(p)):
            a = p[(i - 1) % len(p)]
            b = p[i]
            c = p[(i + 1) % len(p)]
            score = abs(float(np.cross(b - a, c - b)))
            scores.append(score)
        p = np.delete(p, int(np.argmin(scores)), axis=0)
    return p


def install_optcuts_test_boundary_clip_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_boundary_clip_installed", False):
        return
    base = pipeline._build_m2d

    def build_m2d(grid: Any, domain: Any, params: Any = None):
        if not bool(getattr(domain, "_optcuts_test_clip_boundary", False)):
            return base(grid, domain, params)

        overlay_nx = int(getattr(domain, "overlay_nx", grid.nx))
        overlay_ny = int(getattr(domain, "overlay_ny", grid.ny))
        overlay_grid = create_quad_grid(overlay_nx, overlay_ny, grid.tile_size, grid.gap_size)
        vertices_list = [np.asarray([p[0], p[1], 0.0], float) for p in np.asarray(domain.uv_vertices, float)]
        original_vertices = np.asarray(vertices_list, float)
        boundary = np.asarray(domain.boundary, float)
        if len(boundary) > 1 and np.linalg.norm(boundary[0] - boundary[-1]) <= 1e-10:
            boundary_open = boundary[:-1]
        else:
            boundary_open = boundary

        faces: list[tuple[int, int, int, int]] = []
        full_count = 0
        clipped_count = 0
        removed_count = 0
        approximated_gt4 = 0
        triangular_count = 0
        clipped_area = 0.0

        for tile in overlay_grid.tiles or []:
            ids = [int(v) for v in tile.vertex_ids]
            cell = original_vertices[np.asarray(ids, int), :2]
            inside = np.asarray([_point_in_polygon(p, boundary_open) for p in cell], dtype=bool)
            if bool(np.all(inside)):
                faces.append(tuple(ids))
                full_count += 1
                continue

            poly = _intersection_polygon(cell, boundary_open)
            if len(poly) < 3 or _polygon_area(poly) <= 1e-12:
                removed_count += 1
                continue

            if len(poly) == 3:
                triangular_count += 1
            if len(poly) > 4:
                approximated_gt4 += 1
            q = _quadify(poly)
            if len(q) != 4:
                removed_count += 1
                continue
            new_ids: list[int] = []
            for p in q:
                new_ids.append(len(vertices_list))
                vertices_list.append(np.asarray([p[0], p[1], 0.0], float))
            faces.append(tuple(new_ids))
            clipped_count += 1
            clipped_area += _polygon_area(poly)

        if not faces:
            raise RuntimeError("OPTCUTS_TEST_M2D_CLIP_EMPTY: no panel area intersects Omega")

        vertices = np.asarray(vertices_list, dtype=float)
        face_array = np.asarray(faces, dtype=int)
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None)
        if cls is None:
            sample = base(grid, domain, params)
            cls = type(sample)
        metrics = {
            "max_csf_before_split": float(getattr(domain, "csf_before", getattr(domain, "max_csf", 1.0))),
            "max_csf_after_split": float(getattr(domain, "csf_after_split", getattr(domain, "max_csf", 1.0))),
            "number_of_splits": len(getattr(domain, "split_lines", []) or []),
            "split_locations": list(getattr(domain, "split_lines", []) or []),
            "m2d_grid_overlay": "regular grid with Omega-boundary panel clipping",
            "m2d_crop_policy": "clip_boundary_panels",
            "m2d_requested_grid_nx": int(getattr(domain, "original_requested_nx", grid.nx)),
            "m2d_requested_grid_ny": int(getattr(domain, "original_requested_ny", grid.ny)),
            "m2d_overlay_grid_nx": overlay_nx,
            "m2d_overlay_grid_ny": overlay_ny,
            "m2d_overlay_total_quad_count": int(len(overlay_grid.tiles or [])),
            "m2d_full_panel_count": int(full_count),
            "m2d_boundary_clipped_panel_count": int(clipped_count),
            "m2d_removed_fully_outside_panel_count": int(removed_count),
            "m2d_boundary_triangle_quadified_count": int(triangular_count),
            "m2d_boundary_gt4_simplified_count": int(approximated_gt4),
            "m2d_boundary_clipped_area_sum": float(clipped_area),
            "m2d_boundary_panels_deleted_instead_of_clipped": False,
            "optcuts_test_boundary_clip_enabled": True,
        }
        out = cls(vertices, face_array, overlay_grid, "M2D", metrics, list(getattr(domain, "split_lines", []) or []))
        setattr(out, "_optcuts_test_boundary_clipped", True)
        print(
            "[OPTCUTS-TEST-M2D] "
            f"full={full_count} clipped={clipped_count} outside_removed={removed_count} "
            f"tri_quadified={triangular_count} gt4_simplified={approximated_gt4}"
        )
        return out

    pipeline._build_m2d = build_m2d
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_m2d
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d
    pipeline._onestring_optcuts_test_boundary_clip_installed = True


__all__ = ["install_optcuts_test_boundary_clip_m2d_patch"]
