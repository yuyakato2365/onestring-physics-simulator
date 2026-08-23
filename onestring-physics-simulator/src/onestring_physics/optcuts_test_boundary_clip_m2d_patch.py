"""Conservative boundary cutting for ``optcuts_test`` M2D.

Every visible M2D panel is required to lie inside Omega.  Boundary-crossing grid
cells are cut with a simple chord, but that chord is accepted only when the
resulting triangle/quad is contained in the bijective Omega domain.  Otherwise
its cut points are retracted toward a known-inside grid corner until containment
is satisfied.  Visible panels are triangles or quads only.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .quad_grid import create_quad_grid


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    p = np.asarray(point, float)[:2]
    x, y = float(p[0]), float(p[1])
    poly = np.asarray(polygon, float)
    # Treat points extremely close to the boundary as inside.
    for i in range(len(poly)):
        a = poly[i]; b = poly[(i + 1) % len(poly)]
        ab = b - a; den = float(np.dot(ab, ab))
        if den > 1e-24:
            t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
            if float(np.linalg.norm(p - (a + t * ab))) <= 1e-9:
                return True
    inside = False
    for i in range(len(poly)):
        x0, y0 = poly[i]; x1, y1 = poly[(i + 1) % len(poly)]
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
        q = np.asarray(p, float)
        if not any(float(np.linalg.norm(q - r)) <= tol for r in out):
            out.append(q)
    return out


def _polygon_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def _convex_hull(points):
    pts = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points, float)})
    if len(pts) <= 2:
        return np.asarray(pts, float)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
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
    cell = np.asarray(cell, float); boundary = np.asarray(boundary, float)
    hits = []
    for i in range(4):
        a, b = cell[i], cell[(i + 1) % 4]
        for j in range(len(boundary)):
            q = _segment_intersection(a, b, boundary[j], boundary[(j + 1) % len(boundary)])
            if q is not None:
                hits.append(q)
    return _dedupe(hits)


def _farthest_pair(points):
    pts = [np.asarray(p, float) for p in points]
    if len(pts) < 2:
        return None
    best = None; best_d = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d > best_d:
                best_d = d; best = (pts[i], pts[j])
    return best


def _panel_inside_omega(poly, boundary, samples_per_edge=9):
    """Conservative containment test for a straight-edged panel."""
    p = np.asarray(poly, float)
    if len(p) < 3 or _polygon_area(p) <= 1e-12:
        return False
    # Vertices and each straight edge must stay inside Omega.
    for v in p:
        if not _point_in_polygon(v, boundary):
            return False
    ts = np.linspace(0.0, 1.0, max(3, int(samples_per_edge)))
    for i in range(len(p)):
        a = p[i]; b = p[(i + 1) % len(p)]
        for t in ts:
            if not _point_in_polygon((1.0 - t) * a + t * b, boundary):
                return False
    # Also test interior samples to catch a non-convex Omega cutting through the face.
    center = np.mean(p, axis=0)
    if not _point_in_polygon(center, boundary):
        return False
    for v in p:
        for t in (0.25, 0.5, 0.75):
            if not _point_in_polygon((1.0 - t) * center + t * v, boundary):
                return False
    return True


def _simple_cut_polygon(cell, boundary):
    """Straight-cut a cell, retracting the cut inward until the panel is in Omega."""
    cell = np.asarray(cell, float)
    inside = [np.asarray(p, float) for p in cell if _point_in_polygon(p, boundary)]
    hits = _boundary_cell_intersections(cell, boundary)
    pair = _farthest_pair(hits)
    if pair is None:
        hull = _convex_hull(inside) if len(inside) >= 3 else np.zeros((0, 2), float)
        return hull if _panel_inside_omega(hull, boundary) else np.zeros((0, 2), float)

    # A known-inside anchor is used only to retract an unsafe chord.  This is
    # deliberately conservative: losing a little area is preferable to leaving
    # the bijective Omega domain.
    if inside:
        anchor = inside[0]
    else:
        mid = 0.5 * (pair[0] + pair[1])
        if not _point_in_polygon(mid, boundary):
            return np.zeros((0, 2), float)
        anchor = mid

    for alpha in (1.0, 0.98, 0.95, 0.90, 0.82, 0.72, 0.60, 0.45, 0.30, 0.15):
        q0 = anchor + alpha * (pair[0] - anchor)
        q1 = anchor + alpha * (pair[1] - anchor)
        hull = _convex_hull(_dedupe(inside + [q0, q1]))
        if len(hull) >= 3 and _panel_inside_omega(hull, boundary):
            return hull
    return np.zeros((0, 2), float)


def _split_to_max_four(poly, boundary):
    p = np.asarray(poly, float)
    if len(p) <= 4:
        return [p] if _panel_inside_omega(p, boundary) else []
    if len(p) == 5:
        candidates = [
            [p[[0,1,2]], p[[0,2,3,4]]],
            [p[[1,2,3]], p[[1,3,4,0]]],
            [p[[2,3,4]], p[[2,4,0,1]]],
        ]
        for pieces in candidates:
            if all(_panel_inside_omega(piece, boundary) for piece in pieces):
                return pieces
        return []
    pieces = [p[[0, i, i + 1]] for i in range(1, len(p) - 1)]
    return pieces if all(_panel_inside_omega(piece, boundary) for piece in pieces) else []


def _triangle_as_quad(poly):
    p = np.asarray(poly, float)
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    edge = int(np.argmax(lengths)); mid = 0.5 * (p[edge] + p[(edge + 1) % 3])
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
        nx = int(getattr(domain, "overlay_nx", grid.nx)); ny = int(getattr(domain, "overlay_ny", grid.ny))
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
            idx = len(vertices); vertex_map[key] = idx
            vertices.append(np.asarray([p[0], p[1], 0.0], float)); return idx

        surrogate_faces = []; polygon_faces = []
        full_cells = clipped_cells = removed_cells = split_pentagons = 0
        triangle_panels = quad_panels = rejected_outside = 0

        for tile in overlay.tiles or []:
            ids = [int(v) for v in tile.vertex_ids]
            cell = original[np.asarray(ids, int), :2]
            # Do NOT trust corner classification alone: the Omega boundary can run
            # through a non-convex cell even when all four corners happen to be inside.
            if _panel_inside_omega(cell, boundary):
                polygon_faces.append(list(ids)); surrogate_faces.append(tuple(ids))
                full_cells += 1; quad_panels += 1; continue

            cut_poly = _simple_cut_polygon(cell, boundary)
            if len(cut_poly) < 3:
                removed_cells += 1; rejected_outside += 1; continue
            pieces = _split_to_max_four(cut_poly, boundary)
            if not pieces:
                removed_cells += 1; rejected_outside += 1; continue
            if len(cut_poly) == 5:
                split_pentagons += 1
            clipped_cells += 1
            for piece in pieces:
                true_ids = [get_id(p) for p in piece]
                polygon_faces.append(true_ids)
                if len(piece) == 3:
                    triangle_panels += 1
                    surrogate_faces.append(tuple(get_id(p) for p in _triangle_as_quad(piece)))
                elif len(piece) == 4:
                    quad_panels += 1; surrogate_faces.append(tuple(true_ids))

        if not surrogate_faces:
            raise RuntimeError("OPTCUTS_TEST_M2D_CLIP_EMPTY")
        verts = np.asarray(vertices, float); faces = np.asarray(surrogate_faces, int)
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None)
        if cls is None:
            cls = type(base(grid, domain, params))
        metrics = {
            "m2d_grid_overlay":"regular grid with conservative straight boundary cuts",
            "m2d_crop_policy":"strict_inside_omega_max4",
            "m2d_full_cell_count":int(full_cells),
            "m2d_boundary_clipped_cell_count":int(clipped_cells),
            "m2d_removed_cell_count":int(removed_cells),
            "m2d_boundary_pentagon_split_count":int(split_pentagons),
            "m2d_triangle_panel_count":int(triangle_panels),
            "m2d_quad_panel_count":int(quad_panels),
            "m2d_rejected_outside_omega_count":int(rejected_outside),
            "m2d_max_visible_panel_degree":4,
            "m2d_panel_subset_of_omega_enforced":True,
            "m2d_true_polygon_geometry":True,
            "m2d_legacy_quad_surrogate_for_downstream":True,
            "number_of_splits":len(getattr(domain,"split_lines",[]) or []),
            "split_locations":list(getattr(domain,"split_lines",[]) or []),
        }
        out = cls(verts, faces, overlay, "M2D", metrics, list(getattr(domain,"split_lines",[]) or []))
        setattr(out, "_polygon_faces", [list(map(int, f)) for f in polygon_faces])
        setattr(out, "_optcuts_test_boundary_clipped", True)
        print(
            "[OPTCUTS-TEST-M2D] "
            f"full={full_cells} clipped={clipped_cells} removed={removed_cells} "
            f"rejected_outside={rejected_outside} triangles={triangle_panels} "
            f"quads={quad_panels} pentagons_split={split_pentagons} subset_of_omega=True"
        )
        return out

    def lift(target: Any, mesh: Any, parameterization: Any, params: Any):
        lifted, report = base_lift(target, mesh, parameterization, params)
        if hasattr(mesh, "_polygon_faces"):
            setattr(lifted, "_polygon_faces", [list(f) for f in getattr(mesh, "_polygon_faces")])
            setattr(lifted, "_optcuts_test_boundary_clipped", True)
        return lifted, report

    pipeline._build_m2d = build_m2d; pipeline._lift_m2d_to_m3d = lift
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_m2d; original._lift_m2d_to_m3d = lift
    for fn in (getattr(pipeline,"build_onestring_design",None), getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None), getattr(original,"build_onestring_design",None) if original is not None else None):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d; glb["_lift_m2d_to_m3d"] = lift
    pipeline._onestring_optcuts_test_boundary_clip_installed = True


__all__ = ["install_optcuts_test_boundary_clip_m2d_patch"]
