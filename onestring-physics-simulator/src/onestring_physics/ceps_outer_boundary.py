"""Recover the external CEPS Omega boundary in the presence of UV seams.

The official common-refinement OBJ may contain duplicated seam vertices. After
pairing 3D and UV vertices, those seams appear as additional topological boundary
loops. The OneString M2D crop stage needs the external polygonal domain, not one
of the internal cut loops. CEPS is invoked here with four positive pi/2 boundary
exterior angles and zero elsewhere, so the intended domain is convex. Its outer
convex hull is therefore the appropriate Omega clipping boundary.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    return 0.5 * float(
        np.sum(
            pts[:, 0] * np.roll(pts[:, 1], -1)
            - np.roll(pts[:, 0], -1) * pts[:, 1]
        )
    )


def _outer_boundary(module: Any, uv: np.ndarray, faces: np.ndarray):
    points = np.asarray(uv, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise RuntimeError("official CEPS UV output must be an Nx2 array")
    points = points[:, :2]
    finite = points[np.all(np.isfinite(points), axis=1)]
    unique = np.unique(np.round(finite, 12), axis=0)
    if len(unique) < 3:
        raise RuntimeError("official CEPS UV output has fewer than three unique points")

    try:
        hull = ConvexHull(unique)
    except Exception as exc:
        raise RuntimeError("official CEPS UV output has no valid two-dimensional outer hull") from exc

    polygon = unique[np.asarray(hull.vertices, dtype=int)]
    area = _polygon_area(polygon)
    if abs(area) <= 1e-14:
        raise RuntimeError("official CEPS outer UV hull is degenerate")
    if area < 0.0:
        polygon = polygon[::-1]
        area = -area

    loops = []
    try:
        loops = module._boundary_loops(np.asarray(faces, dtype=int))
    except Exception:
        loops = []

    span = np.ptp(polygon, axis=0)
    if float(np.min(span)) <= 1e-12:
        raise RuntimeError("official CEPS outer UV hull collapsed along one axis")

    boundary = np.vstack([polygon, polygon[0]])
    return boundary, {
        "ceps_uv_boundary_loop_count": int(len(loops)),
        "ceps_omega_boundary_source": "convex_hull_of_all_paired_ceps_uv_vertices",
        "ceps_omega_boundary_vertex_count": int(len(polygon)),
        "ceps_omega_boundary_convex": True,
        "ceps_internal_uv_seams_excluded_from_omega_boundary": True,
        "ceps_omega_boundary_area": float(area),
        "ceps_omega_boundary_span_u": float(span[0]),
        "ceps_omega_boundary_span_v": float(span[1]),
    }


def install_ceps_outer_boundary(module: Any) -> None:
    """Replace CEPS loop-based boundary extraction with outer-hull recovery."""
    if getattr(module, "_CEPS_OUTER_BOUNDARY_INSTALLED", False):
        return

    def omega_boundary(uv: np.ndarray, faces: np.ndarray):
        return _outer_boundary(module, uv, faces)

    module._omega_boundary = omega_boundary
    module._CEPS_OUTER_BOUNDARY_INSTALLED = True
