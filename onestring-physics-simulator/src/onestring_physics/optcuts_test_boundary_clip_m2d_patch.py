"""Simple boundary cutting for ``optcuts_test`` M2D.

Requested behavior:
* start from the ordinary regular grid,
* keep full cells that are inside Omega,
* for a boundary-crossing cell, approximate the local Omega boundary by the
  straight chord joining its two cell-boundary intersections,
* keep the inside part of that cell without staircase deletion,
* allow only triangles and quads as final panel polygons.

If the exact straight cut leaves a pentagon (cutting off one corner of a square),
the pentagon is split exactly into one triangle and one quad instead of being
geometrically simplified.  Thus no visible panel has more than four sides.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .quad_grid import create_quad_grid


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = [float(v) for v in np.asarray(point, float)[:2]]
    poly = np.asarray(polygon, float)
    inside = False
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        if (y0 > y) != (y1 > y):
            den = float(y1 - y0)
            if abs(den) > 1e-20:
                x_cross = float(x0 + (x1 - x0) * (y - y0) / den)
                if x < x_cross:
                    inside = not inside
    return inside


def _segment_intersection(a, b, c, d):
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


def _dedupe(points, tol=1e-9):
    out = []
    for p in points:
        p = np.asarray(p, float)
        if not any(float(np.linalg.norm(p - q)) <= tol for q in out):
            out.append(p)
    return out


def _polygon_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def _convex_hull(points):
    """2-D monotone-chain hull; local straight cuts of a square are convex."""
    pts = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points, float)})
    if len(pts) <= 2:
        return np.asarray(pts, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-14:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-14:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], float)


def _boundary_cell_intersections(cell, boundary):
    cell = np.asarray(cell, float)
    boundary = np.asarray(boundary, float)
    cell_edges = [(cell[i], cell[(i + 1) % 4]) for i in range(4)]
    boundary_edges = [(boundary[i], boundary[(i + 1) % len(boundary)]) for i in range(len(boundary))]
    hits = []
    for a, b in cell_edges:
        for c, d in boundary_edges:
            q = _segment_intersection(a, b, c, d)
            if q is not None:
                hits.append(q)
    return _dedupe(hits)


def _farthest_pair(points):
    pts = [np.asarray(p, float) for p in points]
    if len(pts) < 2:
        return None
    best = (pts[0], pts[1])
    best_d = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d > best_d:
                best_d = d
                best = (pts[i], pts[j])
    return best


def _simple_cut_polygon(cell, boundary):
    """Return the inside part of one square after one straight local boundary cut.

    Only the square corners classified inside Omega and two representative
    boundary/cell intersections are used.  Interior vertices of the polyline
    boundary are deliberately ignored; this is what removes the previous jagged
    multi-vertex boundary panels.
    """
    cell = np.asarray(cell, float)
    inside_corners = [p for p in cell if _point_in_polygon(p, boundary)]
    hits = _boundary_cell_intersections(cell, boundary)
    pair = _farthest_pair(hits)
    if pair is None:
        if len(inside_corners) >= 3:
            hull = _convex_hull(inside_corners)
            return hull if len(hull) >= 3 else np.zeros((0, 2), float)
        return np.zeros((0, 2), float)

    pts = list(inside_corners) + [pair[0], pair[1]]
    hull = _convex_hull(_dedupe(pts))
    if len(hull) < 3 or _polygon_area(hull) <= 1e-12:
        return np.zeros((0, 2), float)
    return hull


def _split_to_max_four(poly):
    """Exact decomposition into polygons with at most four vertices."""
    p = np.asarray(poly, float)
    if len(p) <= 4:
        return [p]
    if len(p) == 5:
        # Convex pentagon -> triangle [0,1,2] + quad [0,2,3,4].
        return [p[[0, 1, 2]], p[[0, 2, 3, 4]]]
    # Defensive fallback: fan triangulation.  With the straight-chord model a
    # clipped square should never reach this branch.
    return [p[[0, i, i + 1]] for i in range(1, len(p) - 1)]


def _triangle_as_quad(poly):
    """Legacy downstream representation of a true triangular panel.

    The visible polygon stays triangular.  For the quad-only numerical backend,
    add a midpoint on the longest triangle edge, preserving cyclic order and the
    exact 2-D region.
    """
    p = np.asarray(poly, float)
    if len(p) != 3:
        raise ValueError("triangle_as_quad expects exactly 3 vertices")
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    edge = int(np.argmax(lengths))
    mid = 0.5 * (p[edge] + p[(edge + 1) % 3])
    out = []
    for i in range(3):
        out.append(p[i])
        if i == edge:
            out.append(mid)
    return np.asarray(out, float)


def install_optcuts_test_boundary_clip_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_boundary_clip_installed", False):
        return
    base = pipeline._build_m2d
    base_lift = pipeline._lift_m2d_to_m3d

    def build_m2d(grid: Any, domain: Any, params: Any = None):
        if not bool(getattr(domain, "_optcuts_test_clip_boundary", False)):
            return base(grid, domain, params)

        nx = int(getattr(domain, "overlay_nx", grid.nx))
        ny = int(getattr(domain, "overlay_ny", grid.ny))
        overlay = create_quad_grid(nx, ny, grid.tile_size, grid.gap_size)
        vertices = [np.asarray([p[0], p[1], 0.0], float) for p in np.asarray(domain.uv_vertices, float)]
        original = np.asarray(vertices, float)
        boundary = np.asarray(domain.boundary, float)
        if len(boundary) > 1 and np.linalg.norm(boundary[0] - boundary[-1]) < 1e-10:
            boundary = boundary[:-1]

        vertex_map = {tuple(np.round(v[:2], 10)): i for i, v in enumerate(vertices)}

        def get_id(p):
            key = tuple(np.round(np.asarray(p, float)[:2], 10))
            if key in vertex_map:
                return vertex_map[key]
            idx = len(vertices)
            vertex_map[key] = idx
            vertices.append(np.asarray([p[0], p[1], 0.0], float))
            return idx

        surrogate_faces = []
        polygon_faces = []
        full_cells = 0
        clipped_cells = 0
        removed_cells = 0
        split_pentagons = 0
        triangle_panels = 0
        quad_panels = 0

        for tile in overlay.tiles or []:
            ids = [int(v) for v in tile.vertex_ids]
            cell = original[np.asarray(ids, int), :2]
            corner_inside = [_point_in_polygon(p, boundary) for p in cell]

            if all(corner_inside):
                polygon_faces.append(list(ids))
                surrogate_faces.append(tuple(ids))
                full_cells += 1
                quad_panels += 1
                continue

            cut_poly = _simple_cut_polygon(cell, boundary)
            if len(cut_poly) < 3 or _polygon_area(cut_poly) <= 1e-12:
                removed_cells += 1
                continue

            pieces = _split_to_max_four(cut_poly)
            if len(cut_poly) == 5:
                split_pentagons += 1
            clipped_cells += 1

            for piece in pieces:
                if len(piece) not in (3, 4) or _polygon_area(piece) <= 1e-12:
                    continue
                true_ids = [get_id(p) for p in piece]
                polygon_faces.append(true_ids)
                if len(piece) == 3:
                    triangle_panels += 1
                    q = _triangle_as_quad(piece)
                    surrogate_faces.append(tuple(get_id(p) for p in q))
                else:
                    quad_panels += 1
                    surrogate_faces.append(tuple(true_ids))

        if not surrogate_faces:
            raise RuntimeError("OPTCUTS_TEST_M2D_CLIP_EMPTY")

        verts = np.asarray(vertices, float)
        faces = np.asarray(surrogate_faces, int)
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None)
        if cls is None:
            cls = type(base(grid, domain, params))

        metrics = {
            "m2d_grid_overlay": "regular grid with direct straight boundary cuts",
            "m2d_crop_policy": "simple_chord_cut_max4",
            "m2d_full_cell_count": int(full_cells),
            "m2d_boundary_clipped_cell_count": int(clipped_cells),
            "m2d_removed_fully_outside_cell_count": int(removed_cells),
            "m2d_boundary_pentagon_split_count": int(split_pentagons),
            "m2d_triangle_panel_count": int(triangle_panels),
            "m2d_quad_panel_count": int(quad_panels),
            "m2d_max_visible_panel_degree": 4,
            "m2d_true_polygon_geometry": True,
            "m2d_legacy_quad_surrogate_for_downstream": True,
            "m2d_boundary_model": "two intersections -> one straight cut chord; no boundary polyline vertices inside cell",
            "number_of_splits": len(getattr(domain, "split_lines", []) or []),
            "split_locations": list(getattr(domain, "split_lines", []) or []),
        }

        out = cls(verts, faces, overlay, "M2D", metrics, list(getattr(domain, "split_lines", []) or []))
        setattr(out, "_polygon_faces", [list(map(int, f)) for f in polygon_faces])
        setattr(out, "_optcuts_test_boundary_clipped", True)
        print(
            "[OPTCUTS-TEST-M2D] "
            f"full_cells={full_cells} clipped_cells={clipped_cells} outside_removed={removed_cells} "
            f"triangles={triangle_panels} quads={quad_panels} pentagons_split={split_pentagons} max_degree=4"
        )
        return out

    def lift(target: Any, mesh: Any, parameterization: Any, params: Any):
        lifted, report = base_lift(target, mesh, parameterization, params)
        if hasattr(mesh, "_polygon_faces"):
            setattr(lifted, "_polygon_faces", [list(f) for f in getattr(mesh, "_polygon_faces")])
            setattr(lifted, "_optcuts_test_boundary_clipped", True)
        return lifted, report

    pipeline._build_m2d = build_m2d
    pipeline._lift_m2d_to_m3d = lift
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_m2d
        original._lift_m2d_to_m3d = lift
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d
            glb["_lift_m2d_to_m3d"] = lift
    pipeline._onestring_optcuts_test_boundary_clip_installed = True


__all__ = ["install_optcuts_test_boundary_clip_m2d_patch"]
