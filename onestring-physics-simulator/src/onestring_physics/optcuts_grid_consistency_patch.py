"""Consistency guard for phased native Grid-OptCuts M2D.

QuadGrid stores only the base regular-lattice vertices.  M2D may additionally
contain zero-width duplicate vertices created solely to disconnect the native
OptCuts seam topology.  Synchronize the base prefix with QuadGrid and validate
that every duplicate is geometrically coincident with a base lattice vertex.
"""
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
        base_count = (int(qgrid.nx) + 1) * (int(qgrid.ny) + 1)
        if len(vertices) < base_count:
            raise RuntimeError(
                "OPTCUTS_GRID_CONSISTENCY_VERTEX_COUNT: "
                f"mesh={len(vertices)} base_grid={base_count}"
            )
        base = vertices[:base_count].copy()
        qgrid.vertex_positions = base
        delta = float(np.max(np.abs(np.asarray(qgrid.vertex_positions) - base))) if base_count else 0.0
        if delta > 1.0e-12:
            raise RuntimeError(f"OPTCUTS_GRID_CONSISTENCY_COORDINATE_MISMATCH: max_delta={delta}")

        duplicate_max_distance = 0.0
        if len(vertices) > base_count:
            # Duplicates are expected to be exact copies of a base grid vertex.
            # A brute-force check is acceptable here because only seam vertices
            # are duplicated and this is a one-time guard.
            base_xy = base[:, :2]
            for p in vertices[base_count:, :2]:
                d = float(np.min(np.linalg.norm(base_xy - p[None, :], axis=1)))
                duplicate_max_distance = max(duplicate_max_distance, d)
            h = max(float(metrics.get("optcuts_grid_unit", getattr(qgrid, "tile_size", 1.0))), 1e-12)
            if duplicate_max_distance > max(1e-9, 1e-7 * h):
                raise RuntimeError(
                    "OPTCUTS_GRID_CONSISTENCY_NONCOINCIDENT_DUPLICATE: "
                    f"max_distance={duplicate_max_distance} h={h}"
                )

        metrics.update({
            "optcuts_quadgrid_phase_synced": True,
            "optcuts_quadgrid_mesh_max_delta": delta,
            "optcuts_quadgrid_base_vertex_count": int(base_count),
            "optcuts_quadgrid_duplicate_vertex_count": int(len(vertices) - base_count),
            "optcuts_quadgrid_duplicate_max_distance": float(duplicate_max_distance),
        })
        mesh.metrics.update(metrics)
        print(
            "[OPTCUTS-GRID-CONSISTENCY] "
            f"base_vertices={base_count} duplicates={len(vertices)-base_count} "
            f"base_delta={delta:.3g} duplicate_max_distance={duplicate_max_distance:.3g}"
        )
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


__all__ = ["install_optcuts_grid_consistency_patch"]
