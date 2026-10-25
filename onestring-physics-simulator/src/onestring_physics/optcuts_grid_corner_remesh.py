"""Optional paired UV/3D boundary-corner refinement for the 10/23 grid Omega.

Subdivide only triangles adjacent to boundary vertices mapped onto grid turns.
Each affected triangle is split into three at its own barycenter, preserving
ALL original triangle edges and the exact piecewise-linear 3D surface.
No T-junction, no change to any original vertex/seam, and no patching of
infeasible endpoint correspondence: preflight runs BEFORE this subdivision.
Added barycenters are unconstrained interior UV variables for subsequent fitting.
"""
from __future__ import annotations
import numpy as np


def subdivide_grid_corner_neighborhood(
    parameterization, boundary_loop, mapped_boundary, grid_outline,
    *, max_faces=600,
):
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    xyz = np.asarray(parameterization.surface_vertices_3d, float)
    faces = np.asarray(parameterization.uv_faces, int)
    source_faces = np.asarray(parameterization.surface_faces, int)
    boundary_loop = np.asarray(boundary_loop, int)
    mapped = np.asarray(mapped_boundary, float)
    outline = np.asarray(grid_outline, float)
    if (len(faces) != len(source_faces) or len(mapped) != len(boundary_loop)
            or uv.shape[1] != 2 or xyz.shape[1] != 3):
        raise ValueError("GRID_CORNER_REMESH_INVALID_PAIRED_SOURCE")
    if max_faces < 1:
        raise ValueError("GRID_CORNER_REMESH_INVALID_BUDGET")
    # These are *existing* source boundary vertices constrained by the
    # selected arclength/DP map, not inferred nearest source positions.
    matched = np.argmin(
        np.linalg.norm(mapped[:, None, :] - outline[None, :, :], axis=-1),
        axis=0)
    if np.max(np.linalg.norm(mapped[matched] - outline, axis=1)) > 1.e-7 * max(
            1., float(np.max(np.ptp(outline, axis=0)))):
        raise ValueError("GRID_CORNER_REMESH_MISSING_ANCHOR")
    anchors = np.unique(boundary_loop[matched])
    candidates = np.flatnonzero(np.isin(faces, anchors).any(axis=1))
    if not len(candidates):
        raise ValueError("GRID_CORNER_REMESH_NO_ADJACENT_FACES")
    # Prioritize triangles closest to source grid-corner anchors. The limit
    # bounds memory and K2D cost on dense scanned Bunny meshes.
    centers = uv[faces[candidates]].mean(axis=1)
    anchor_xy = uv[anchors]
    distance = np.min(np.linalg.norm(
        centers[:, None, :] - anchor_xy[None, :, :], axis=-1), axis=1)
    if len(candidates) > max_faces:
        candidates = candidates[np.argsort(distance, kind="stable")[:max_faces]]
    chosen = set(int(i) for i in candidates)
    new_uv = [p.copy() for p in uv]
    new_xyz = [p.copy() for p in xyz]
    new_faces, new_source_faces = [], []
    for index, (uf, sf) in enumerate(zip(faces, source_faces)):
        if index not in chosen:
            new_faces.append([int(v) for v in uf])
            new_source_faces.append([int(v) for v in sf])
            continue
        u_mid = len(new_uv)
        s_mid = len(new_xyz)
        new_uv.append(np.mean(uv[uf], axis=0))
        new_xyz.append(np.mean(xyz[sf], axis=0))
        for i in range(3):
            j = (i + 1) % 3
            new_faces.append([int(uf[i]), int(uf[j]), u_mid])
            new_source_faces.append([int(sf[i]), int(sf[j]), s_mid])
    parameterization.uv_vertices_2d = np.asarray(new_uv, float)
    parameterization.surface_vertices_3d = np.asarray(new_xyz, float)
    parameterization.uv_faces = np.asarray(new_faces, int)
    parameterization.surface_faces = np.asarray(new_source_faces, int)
    parameterization.triangle_acceleration = None
    diag = {
        "subdivided_triangles": len(chosen),
        "added_interior_vertices": len(chosen),
        "source_triangle_count_before": int(len(faces)),
        "source_triangle_count_after": int(len(new_faces)),
        "corner_anchor_count": len(anchors),
        "original_topology_edges_preserved": True,
        "source_piecewise_linear_surface_preserved": True,
    }
    print("[GRID-CORNER-REMESH] " + str(diag), flush=True)
    return diag
