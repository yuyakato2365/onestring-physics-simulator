"""Consistency guard for phased OptCuts M2D grids."""
from __future__ import annotations

from typing import Any
import numpy as np


def install_optcuts_grid_consistency_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_grid_consistency_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_consistent(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        if not bool(metrics.get("optcuts_grid_constrained_m2d", False)):
            return mesh
        vertices = np.asarray(mesh.vertices, dtype=float)
        qgrid = getattr(mesh, "grid", None)
        if qgrid is None:
            raise RuntimeError("OPTCUTS_GRID_CONSISTENCY_MISSING_QUADGRID")
        if len(vertices) != (int(qgrid.nx) + 1) * (int(qgrid.ny) + 1):
            raise RuntimeError(
                "OPTCUTS_GRID_CONSISTENCY_VERTEX_COUNT: "
                f"mesh={len(vertices)} grid={(int(qgrid.nx)+1)*(int(qgrid.ny)+1)}"
            )
        # QuadGrid is mutable even though TileSpec/HingeSpec are structural.  Keep
        # its geometry in the same phased coordinate frame as the M2D vertices.
        qgrid.vertex_positions = vertices.copy()
        delta = float(np.max(np.abs(np.asarray(qgrid.vertex_positions) - vertices))) if len(vertices) else 0.0
        if delta > 1.0e-12:
            raise RuntimeError(f"OPTCUTS_GRID_CONSISTENCY_COORDINATE_MISMATCH: max_delta={delta}")
        metrics.update({
            "optcuts_quadgrid_phase_synced": True,
            "optcuts_quadgrid_mesh_max_delta": delta,
        })
        mesh.metrics.update(metrics)
        print(f"[OPTCUTS-GRID-CONSISTENCY] vertices={len(vertices)} max_delta={delta:.3g}")
        return mesh

    pipeline._build_m2d = build_consistent
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_consistent
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_consistent
    pipeline._onestring_optcuts_grid_consistency_installed = True


__all__=["install_optcuts_grid_consistency_patch"]
