"""Apply saved *evaluation* free-boundary BFF leaf charts to real M2D tiles.

Experimental and opt-in. The Split planner and its CSF scores remain unchanged.
Each reBFF chart is mapped to the original UV grid vertices by barycentric
interpolation; a per-piece similarity transform fixes BFF gauge freedom.
Disconnected seam vertex IDs may take different UV coordinates. The ORIGINAL
Omega grid UV is retained separately for M2D -> M3D inverse mapping.

If a chart cannot represent a tile, or tiles flip/collapse, fail loudly.
Never silently fall back to pre-Split UV or pretend that CSF was certified.
"""
from __future__ import annotations

import numpy as np


def _fit_similarity(bff_uv, source_uv):
    """2D similarity with optional reflection: chart UV -> original UV gauge."""
    src = np.asarray(bff_uv, dtype=float)
    dst = np.asarray(source_uv, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 3:
        raise ValueError("POSTCUT_REBFF_M2D_INVALID_CHART: UV correspondence is missing")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("POSTCUT_REBFF_M2D_INVALID_CHART: non-finite UV values")
    source_centroid = src.mean(axis=0)
    target_centroid = dst.mean(axis=0)
    a = src - source_centroid
    b = dst - target_centroid
    denom = float(np.sum(a * a))
    if denom <= 1e-20:
        raise ValueError("POSTCUT_REBFF_M2D_INVALID_CHART: collapsed reBFF UV")
    u, singular, vt = np.linalg.svd(a.T @ b)
    orthogonal = u @ vt
    scale = float(singular.sum() / denom)
    if scale <= 1e-12 or not np.isfinite(scale):
        raise ValueError("POSTCUT_REBFF_M2D_INVALID_CHART: invalid similarity scale")
    return orthogonal * scale, target_centroid - source_centroid @ (orthogonal * scale)


def _interpolate_chart(source_uv, triangle_faces, new_uv, queries, tolerance):
    """Interpolate reBFF UV in old UV triangles, bounded extrapolation at seam.

    The planner's _split_piece can shift its clipping plane by at most 0.6%
    of grid pitch to avoid skinny triangles. A final M2D seam still lies on
    the exact original grid line; allow only a bounded, explicitly counted
    extrapolation across that numerical shift, not a nearest-vertex fallback.
    """
    from .paper_split_reflatten import _locate
    from scipy.spatial import cKDTree

    old = np.asarray(source_uv, float)
    new = np.asarray(new_uv, float)
    faces = np.asarray(triangle_faces, int)
    q = np.asarray(queries, float)
    tri, bary = _locate(old, faces, q)
    extrapolated = 0
    missing = np.flatnonzero(tri < 0)
    if len(missing):
        centers = old[faces].mean(axis=1)
        tree = cKDTree(centers)
        nearest_k = min(max(16, min(64, len(faces))), len(faces))
        _, near = tree.query(q[missing], k=nearest_k)
        near = np.asarray(near).reshape(len(missing), -1)
        for idx, candidates in zip(missing, near):
            point = q[idx]
            best = None
            for fi in np.asarray(candidates).ravel():
                p0, p1, p2 = old[faces[int(fi)]]
                mat = np.column_stack((p1 - p0, p2 - p0))
                det = float(np.linalg.det(mat))
                if abs(det) < 1e-18:
                    continue
                t = np.linalg.solve(mat, point - p0)
                weights = np.asarray([1. - t[0] - t[1], t[0], t[1]])
                inside = np.maximum(weights, 0.)
                inside /= max(float(inside.sum()), 1e-15)
                nearest_point = inside @ np.stack((p0,p1,p2))
                distance = float(np.linalg.norm(nearest_point-point))
                if best is None or distance < best[0]:
                    best = (distance, int(fi), weights)
            if best is None or best[0] > tolerance:
                distance = float("inf") if best is None else best[0]
                raise RuntimeError(
                    "POSTCUT_REBFF_M2D_SEAM_MAPPING_FAILED: M2D vertex "
                    f"{int(idx)} is outside its 3D reBFF leaf chart "
                    f"(distance={distance:.6g}, allowed={tolerance:.6g})"
                )
            tri[idx] = best[1]
            bary[idx] = best[2]
            extrapolated += 1
    if np.any(tri < 0) or not np.isfinite(bary).all():
        raise RuntimeError("POSTCUT_REBFF_M2D_SEAM_MAPPING_FAILED: unmapped vertices")
    result = (bary[:, :, None] * new[faces[tri]]).sum(axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("POSTCUT_REBFF_M2D_INVALID_CHART: interpolated UV is non-finite")
    return result, extrapolated


def _quad_signed_area(vertices, faces):
    xy = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int), :2]
    x, y = xy[:, :, 0], xy[:, :, 1]
    return 0.5 * (x * np.roll(y, -1, axis=1) -
                  y * np.roll(x, -1, axis=1)).sum(axis=1)


def apply_leaf_rebff_uv(canonical, faces, leaf_charts, *, rotation_degrees=0.):
    """Return manufacturing M2D UV, unchanged original vertices and metrics.

    leaf_charts is transient domain-only data, not the serialized Split report.
    It was captured at each terminal 3D patch during Split candidate scoring.
    Original face IDs survive both the simple and final full-line seam passes.
    """
    original = np.asarray(canonical, dtype=float)
    F = np.asarray(faces, int)
    if original.ndim != 2 or original.shape[1] < 2 or F.ndim != 2 or F.shape[1] != 4:
        raise ValueError("POSTCUT_REBFF_M2D_INVALID_MESH: expected 2D quad mesh")
    if not leaf_charts or not len(F):
        raise RuntimeError("POSTCUT_REBFF_M2D_NO_LEAVES: missing terminal 3D reBFF charts")
    angle = np.deg2rad(float(rotation_degrees))
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    query_grid = original[:, :2] @ rotation
    transformed = np.full((len(original), 2), np.nan, dtype=float)
    assigned = np.zeros(len(original), bool)
    covered_faces = np.zeros(len(F), bool)
    leaf_count = 0
    seam_extrapolated = 0
    scale_min, scale_max = float("inf"), 0.0

    for leaf in leaf_charts:
        ids = np.asarray(leaf["face_ids"], dtype=int)
        if not len(ids):
            raise RuntimeError("POSTCUT_REBFF_M2D_NO_LEAVES: empty leaf")
        if np.any(ids < 0) or np.any(ids >= len(F)) or np.any(covered_faces[ids]):
            raise RuntimeError("POSTCUT_REBFF_M2D_TOPOLOGY_MISMATCH: invalid leaf face IDs")
        covered_faces[ids] = True
        vertex_ids = np.unique(F[ids].ravel())
        if np.any(assigned[vertex_ids]):
            # This would mean the final seam has not actually disconnected
            # the pieces. A single M2D vertex cannot own two independent UVs.
            raise RuntimeError(
                "POSTCUT_REBFF_M2D_TOPOLOGY_MISMATCH: "
                "shared vertex IDs across independently reparameterized pieces"
            )
        chart_source = np.asarray(leaf["source_uv"], dtype=float)
        chart_new = np.asarray(leaf["rebff_uv"], dtype=float)
        piece_faces = np.asarray(leaf["piece_faces"], dtype=int)
        matrix, shift = _fit_similarity(chart_new, chart_source)
        oriented = chart_new @ matrix + shift
        tile_xy = query_grid[F[ids]]
        pitch = float(np.median(np.linalg.norm(
            tile_xy[:,1,:] - tile_xy[:,0,:], axis=1)))
        tolerance = max(0.012 * pitch, 1e-9)
        mapped, extrapolated = _interpolate_chart(
            chart_source, piece_faces, oriented, query_grid[vertex_ids],
            tolerance=tolerance,
        )
        transformed[vertex_ids] = mapped
        assigned[vertex_ids] = True
        seam_extrapolated += int(extrapolated)
        leaf_count += 1
        scale = float(np.linalg.norm(matrix[:,0]))
        scale_min = min(scale_min, scale)
        scale_max = max(scale_max, scale)

    if not covered_faces.all() or np.any(~assigned[np.unique(F.ravel())]):
        raise RuntimeError("POSTCUT_REBFF_M2D_TOPOLOGY_MISMATCH: some quads lack a reBFF leaf")

    output = original.copy()
    used = np.flatnonzero(assigned)
    output[used,:2] = transformed[used] @ rotation.T

    # Inversion/collapse cannot be accepted as a substitute for a successful
    # CSF split. Quad convexity is not assumed, but signed area must be
    # strictly oriented the same way as the original and large enough.
    original_area = _quad_signed_area(original, F)
    new_area = _quad_signed_area(output, F)
    eps = np.maximum(np.abs(original_area) * 1e-6, 1e-14)
    if (not np.isfinite(new_area).all() or
            np.any(original_area * new_area <= 0.) or
            np.any(np.abs(new_area) < eps)):
        bad = np.flatnonzero(
            (~np.isfinite(new_area)) | (original_area * new_area <= 0.) |
            (np.abs(new_area) < eps)
        )
        raise RuntimeError(
            "POSTCUT_REBFF_M2D_FLIPPED_OR_COLLAPSED_QUAD: "
            f"{len(bad)} invalid quads, first face={int(bad[0])}"
        )
    return output, dict(
        m2d_postcut_rebff_applied=True,
        m2d_postcut_rebff_leaf_count=int(leaf_count),
        m2d_postcut_rebff_seam_extrapolated_vertex_count=int(seam_extrapolated),
        m2d_postcut_rebff_alignment_scale_min=float(scale_min),
        m2d_postcut_rebff_alignment_scale_max=float(scale_max),
        m2d_postcut_rebff_m3d_inverse_source="original Omega UV, not deformed manufacturing M2D UV",
        m2d_postcut_rebff_geometry_scope="fabrication M2D coordinates only; M3D source correspondence preserved",
        m2d_postcut_rebff_certified=False,
    )
