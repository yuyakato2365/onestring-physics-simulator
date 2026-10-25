"""Exact preflight of candidate Omega against its frozen production M2D.

Run only for the experimental native Grid-Seam search, *before* selecting
an Omega candidate. This uses the SAME rectilinear production grid tile
topology and the SAME fixed 50% area-mask as the final M2D build. It does
not lift a mesh, alter any UV/surface vertex, remove a tile, or weld seams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .quad_grid import create_quad_grid
from .optcuts_seam_exact_closure import discover_exact_seam_topology


def preview_fixed_grid_seam_topology(parameterization: Any, grid: Any) -> dict:
    """Fail closed if the actual original-mask M2D would fail seam discovery.

    No fake tiling: derives grid dimensions, pitch, origin and mask from the
    *accepted* Omega candidate's production-grid metadata. The face vertex
    IDs and tile order are supplied by create_quad_grid, as in paper_m2d().
    Compaction only renumbers vertex IDs; it does not affect seam provenance.
    """
    metrics = parameterization.metrics
    mask = np.asarray(parameterization._optcuts_area_grid_mask, bool)
    nx = int(metrics["optcuts_area_grid_overlay_nx"])
    ny = int(metrics["optcuts_area_grid_overlay_ny"])
    origin = np.asarray(metrics["optcuts_area_grid_grid_origin"], float)
    step = np.asarray(metrics["optcuts_area_grid_grid_step"], float)
    if (mask.shape != (ny,nx) or origin.shape != (2,) or step.shape != (2,)
            or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(step)) or np.any(step<=0)):
        raise ValueError("GRID_SEAM_PREFLIGHT_FROZEN_LATTICE_INVALID")

    original_grid = create_quad_grid(
        nx, ny, float(grid.tile_size), float(grid.gap_size))
    all_faces = np.asarray(
        [tile.vertex_ids for tile in original_grid.tiles], int)
    assert all_faces.shape == (nx*ny, 4)
    faces = all_faces[mask.reshape(-1)]
    if not len(faces):
        raise ValueError("GRID_SEAM_PREFLIGHT_NO_SELECTED_CELLS")

    x = origin[0] + np.arange(nx+1)*step[0]
    y = origin[1] + np.arange(ny+1)*step[1]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    uv = np.stack((xx, yy),axis=-1).reshape(-1,2)
    # This is the exact full-grid vertex ID convention used downstream,
    # before _compact_quad_mesh() removes unused vertices.
    topology = discover_exact_seam_topology(
        parameterization, uv, faces)
    return {
        "status":str(topology["status"]),
        "seam_grid_edges":int(topology.get("seam_grid_edges",0)),
        "boundary_grid_edges":int(topology.get("boundary_grid_edges",0)),
        "unclassified_boundary_edges":int(
            topology.get("unclassified_boundary_edges",0)),
        "unmatched_edges":int(topology.get("unmatched_edges",0)),
        "seam_stitch_mode":str(topology.get("seam_stitch_mode","unavailable")),
        "grid_face_count":int(len(faces)),
        "exact_mask_unchanged":True,
    }
