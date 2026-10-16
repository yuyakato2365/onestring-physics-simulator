"""Experimental dual-grid CSF proxy using four neighboring tile centroids.

The orange area in OneString Supplemental Fig.18 is a four-tile-centroid
linkage cell, NOT the area of an individual rigid tile. This module measures
such cells for the initial cropped M2D Grid and its strictly inverse-mapped
M3D geometry.

The M2D/M3D dual area ratio is NOT a measured expansion from the fully
closed physical mechanism, nor a proof of local conformality or fabricability.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .reference_bff import strict_inverse_map_uv_to_surface


def build_dual_centroid_field(vertices, faces, parameterization):
    """Return complete 4-quad fans and their dual area scales, prior to Split.

    cells[k] indexes four ORIGINAL M2D face IDs in counterclockwise
    centroid order. A fan is valid only when all four quads meet at a
    common M2D vertex, with consecutive faces sharing an edge there.
    Boundary/partial fans are excluded, explicitly and diagnostically.
    """
    uv_grid = np.asarray(vertices, float)[:, :2]
    quads = np.asarray(faces, int)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError("Dual-centroid CSF needs four-vertex M2D quads")
    if np.min(quads, initial=0) < 0 or np.max(quads, initial=-1) >= len(uv_grid):
        raise ValueError("Dual-centroid CSF face indices are invalid")
    source_uv = np.asarray(parameterization.uv_vertices_2d, float)
    source_uv_faces = np.asarray(
        getattr(parameterization, "uv_faces", parameterization.surface_faces), int
    )
    source_xyz = np.asarray(parameterization.surface_vertices_3d, float)
    source_xyz_faces = np.asarray(parameterization.surface_faces, int)
    if source_uv_faces.shape != source_xyz_faces.shape:
        raise ValueError("Dual-centroid CSF requires matching BFF UV/3D triangles")

    used = np.unique(quads.reshape(-1))
    mapped = np.full((len(uv_grid), 3), np.nan)
    for vid in used:
        xyz, _, _, _ = strict_inverse_map_uv_to_surface(
            uv_grid[vid], source_uv, source_uv_faces, source_xyz,
            source_xyz_faces, tolerance=1e-10, vertex_id=int(vid),
        )
        mapped[vid] = xyz
    centers2 = uv_grid[quads].mean(axis=1)
    centers3 = mapped[quads].mean(axis=1)

    corner_to_faces = defaultdict(list)
    for fid, face in enumerate(quads):
        for vid in face:
            corner_to_faces[int(vid)].append(int(fid))

    cells, central_vertices, a2_list, a3_list = [], [], [], []
    nonfour = 0
    invalid_fans = 0
    for vid, incident in sorted(corner_to_faces.items()):
        if len(incident) != 4 or len(set(incident)) != 4:
            nonfour += 1
            continue
        incident = np.asarray(incident, int)
        rel = centers2[incident] - uv_grid[vid, :2]
        order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]), kind="stable")
        ordered = incident[order]
        # Ensure a real grid fan rather than an accidental four-face
        # point contact across disconnected topology.
        if any(
            len(set(map(int, quads[int(ordered[k])])) &
                set(map(int, quads[int(ordered[(k+1)%4])]))) < 2
            for k in range(4)
        ):
            invalid_fans += 1
            continue
        poly2 = centers2[ordered]
        poly3 = centers3[ordered]
        signed_a2 = 0.5 * np.sum(
            poly2[:, 0] * np.roll(poly2[:, 1], -1)
            - poly2[:, 1] * np.roll(poly2[:, 0], -1)
        )
        a2 = float(abs(signed_a2))
        a3 = float(0.5 * (
            np.linalg.norm(np.cross(poly3[1]-poly3[0], poly3[2]-poly3[0]))
            + np.linalg.norm(np.cross(poly3[2]-poly3[0], poly3[3]-poly3[0]))
        ))
        if not (np.isfinite(a2) and np.isfinite(a3) and a2 > 0 and a3 > 0):
            raise ValueError(
                f"Dual-centroid CSF has collapsed centroid polygon at M2D vertex {vid}"
            )
        cells.append(ordered.tolist())
        central_vertices.append(vid)
        a2_list.append(a2)
        a3_list.append(a3)

    face_ids = np.asarray(cells, int).reshape(-1, 4)
    a2 = np.asarray(a2_list, float)
    a3 = np.asarray(a3_list, float)
    j = a3/a2
    result = dict(
        face_ids=face_ids,
        central_vertex_ids=np.asarray(central_vertices, int),
        area2=a2,
        area3=a3,
        jacobian=j,
    )
    result["diagnostics"] = dict(
        method="M2D/M3D four-tile centroid dual area",
        description="four quad centroids around a shared M2D Grid vertex; triangle fan 0-2",
        dual_cell_count=int(len(j)),
        boundary_or_nonfour_grid_vertex_count=int(nonfour),
        invalid_grid_fan_count=int(invalid_fans),
        dual_m2d_areas=a2.tolist(),
        dual_m3d_areas=a3.tolist(),
        dual_area_ratios=j.tolist(),
        dual_member_quad_ids=face_ids.tolist(),
        dual_central_grid_vertex_ids=central_vertices,
        measured_against="M2D centroid area, not physically fully closed configuration",
        source_geometry="uncut M2D and strict BFF-inverse M3D; not K3D/T3D",
        physical_auxetic_bound_certified=False,
    )
    return result


def component_dual_stats(component, current_faces, field):
    """Sample only intact four-quad dual cells INSIDE this Split component.

    A dual cell spanning two disconnected components or a duplicated seam
    corner is no longer a material linkage and is not counted. A region with
    no intact cells has NOT passed a CSF criterion; measurable=False.
    """
    component = np.asarray(component, int)
    quads = np.asarray(current_faces, int)
    members = np.asarray(field["face_ids"], int).reshape(-1, 4)
    values = np.asarray(field["jacobian"], float)
    if len(component) == 0 or len(members) == 0:
        return dict(measurable=False, cell_count=0, span=None, area_span=None,
                    used_dual_ids=[])
    inclusion = np.zeros(len(quads), bool)
    inclusion[component] = True
    complete = np.all(inclusion[members], axis=1)
    # A split may duplicate a corner ID even if some face grouping remains
    # reachable by other edges. Require the original 4-way fan still shares
    # its actual vertex in the current connectivity.
    candidates = np.flatnonzero(complete)
    kept = []
    for i in candidates:
        common = set(map(int, quads[members[i, 0]]))
        for fid in members[i, 1:]:
            common.intersection_update(map(int, quads[fid]))
        if common:
            kept.append(int(i))
    if not kept:
        return dict(measurable=False, cell_count=0, span=None, area_span=None,
                    used_dual_ids=[])
    sample = values[kept]
    area_span = float(np.max(sample)/np.min(sample))
    return dict(measurable=True, cell_count=int(len(kept)),
                span=float(np.sqrt(area_span)), area_span=area_span,
                min_j=float(np.min(sample)), max_j=float(np.max(sample)),
                used_dual_ids=kept)
