"""Use the true physical boundary of the stitched CEPS common refinement.

The CEPS cut graph is stitched before this module runs, so the surface topology
must again be one disk with one boundary loop. The Omega crop boundary is that
loop itself. A convex hull is deliberately forbidden because it can include UV
regions with no CEPS triangles and can make OneString panels bridge the open
bottom boundary.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    return 0.5 * float(
        np.sum(
            pts[:, 0] * np.roll(pts[:, 1], -1)
            - np.roll(pts[:, 0], -1) * pts[:, 1]
        )
    )


def _remove_consecutive_duplicates(points: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if len(values) == 0:
        return values.reshape(0, 2)
    kept = [values[0]]
    for point in values[1:]:
        if float(np.linalg.norm(point - kept[-1])) > tolerance:
            kept.append(point)
    if len(kept) > 1 and float(np.linalg.norm(kept[0] - kept[-1])) <= tolerance:
        kept.pop()
    return np.asarray(kept, dtype=float)


def _outer_boundary(module: Any, uv: np.ndarray, faces: np.ndarray):
    points = np.asarray(uv, dtype=float)
    triangles = np.asarray(faces, dtype=int)[:, :3]
    if points.ndim != 2 or points.shape[1] < 2:
        raise RuntimeError("official CEPS UV output must be an Nx2 array")
    if len(points) == 0 or len(triangles) == 0:
        raise RuntimeError("official CEPS stitched chart is empty")

    loops = module._boundary_loops(triangles)
    if len(loops) != 1:
        raise RuntimeError(
            "stitched official CEPS output must be one topological disk with one "
            f"physical boundary loop; found {len(loops)} loops"
        )

    loop = [int(value) for value in loops[0]]
    polygon = _remove_consecutive_duplicates(points[np.asarray(loop, dtype=int), :2])
    if len(polygon) < 4:
        raise RuntimeError("official CEPS physical boundary collapsed below four vertices")
    area = _polygon_area(polygon)
    if abs(area) <= 1e-14:
        raise RuntimeError("official CEPS physical UV boundary is degenerate")
    if area < 0.0:
        polygon = polygon[::-1]
        loop = loop[::-1]
        area = -area

    span = np.ptp(polygon, axis=0)
    if float(np.min(span)) <= 1e-12:
        raise RuntimeError("official CEPS physical UV boundary collapsed along one axis")

    metrics = dict(getattr(module, "_CEPS_LAST_CHART_METRICS", {}) or {})
    metrics.update(
        {
            "ceps_uv_boundary_loop_count": 1,
            "ceps_surface_boundary_loop_count": 1,
            "ceps_omega_boundary_source": "physical_boundary_loop_of_stitched_common_refinement",
            "ceps_omega_boundary_vertex_count": int(len(polygon)),
            "ceps_omega_boundary_convex_hull_used": False,
            "ceps_internal_uv_seams_excluded_from_omega_boundary": True,
            "ceps_input_open_boundary_preserved": True,
            "ceps_omega_boundary_area": float(area),
            "ceps_omega_boundary_span_u": float(span[0]),
            "ceps_omega_boundary_span_v": float(span[1]),
            "ceps_physical_boundary_loop_vertex_ids": loop,
        }
    )
    return np.vstack([polygon, polygon[0]]), metrics


def install_ceps_outer_boundary(module: Any) -> None:
    """Replace convex-hull recovery with the stitched physical boundary loop."""
    if getattr(module, "_CEPS_OUTER_BOUNDARY_INSTALLED", False):
        return

    def omega_boundary(uv: np.ndarray, faces: np.ndarray):
        return _outer_boundary(module, uv, faces)

    module._omega_boundary = omega_boundary
    module._CEPS_OUTER_BOUNDARY_INSTALLED = True
