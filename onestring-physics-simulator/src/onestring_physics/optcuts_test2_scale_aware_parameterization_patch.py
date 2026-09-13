"""Scale-aware S->Omega path for the visible ``optcuts_test2`` variant.

This wrapper is installed outermost by the test2 runtime patch.  It replaces only
optcuts_test2's initial official-OptCuts call with scale-aware candidate selection,
then reuses the existing optcuts_test grid-outline reparameterization unchanged.
The ordinary ``optcuts`` and ``optcuts_test`` baselines are therefore preserved.
"""
from __future__ import annotations

from dataclasses import replace
import os
from typing import Any

import numpy as np

from .optcuts_pipeline_patch import _config_from_params
from .optcuts_scale_aware_selection_patch import run_scale_aware_optcuts
from .optcuts_test_boundary_reparameterization_patch import (
    _build_test_targets,
    _quad_union_boundary,
    _reference_domain_without_invalid_optcuts_boundary_diagnostic,
    _staged_boundary_resolve,
)


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == "2"


def _initial_parameterization(pipeline: Any, surface: Any, params: Any):
    vertices = np.asarray(surface.vertices, dtype=float)
    faces = np.asarray(surface.faces, dtype=int)[:, :3]
    result = run_scale_aware_optcuts(vertices, faces, _config_from_params(params))
    loop = [int(v) for v in result.boundary_loops[0]]
    boundary = np.asarray(result.uv_vertices_2d, dtype=float)[loop + [loop[0]]]
    metrics: dict[str, object] = {
        **result.metrics,
        "parameterization_exactness_label": "official_optcuts_scale_aware_outer_selection",
        "parameterization_warning": (
            "The upstream OptCuts binary is unchanged. OneString scale factor participates in "
            "outer candidate selection, followed by the existing optcuts_test reparameterization."
        ),
        "paper_compliance_status": "experimental_nonpaper_parameterization",
        "omega_boundary_mode": "paper_default",
        "omega_parameterization_mode": "optcuts",
        "requested_omega_parameterization_mode": "optcuts",
        "boundary_vertex_count": int(len(loop)),
        "boundary_loop": loop,
        "height_field_shortcut_used": False,
        "harmonic_solve_performed": False,
        "omega_corresponds_to_S": True,
        "omega_correspondence_model": "scale-aware selected official OptCuts cut topology + UV embedding",
        "paper_flow_stage": "S -> Omega by scale-aware official OptCuts candidate selection",
        "bff_implemented": False,
        "optcuts_implemented": True,
        "optcuts_grid_mode": False,
        "optcuts_internal_method": "optcuts_official",
        "omega_boundary_fixed": False,
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape": "free",
        "omega_boundary_constraint_model": "scale-aware selected official OptCuts seam",
        "fallbacks_used": [],
    }
    return pipeline.SurfaceParameterization(
        method="optcuts_official",
        surface_vertices_3d=np.asarray(result.surface_vertices_3d, dtype=float),
        surface_faces=np.asarray(result.surface_faces, dtype=int),
        uv_vertices_2d=np.asarray(result.uv_vertices_2d, dtype=float),
        uv_faces=np.asarray(result.uv_faces, dtype=int),
        omega_boundary=np.asarray(boundary, dtype=float),
        triangle_acceleration=None,
        metrics=metrics,
    )


def install_optcuts_test2_scale_aware_parameterization_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_test2_scale_aware_parameterization_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def builder(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts_test" or not _is_test2():
            return base_builder(surface, target, grid, params)

        ordinary_params = replace(params, omega_parameterization_mode="optcuts")
        parameterization = _initial_parameterization(pipeline, surface, ordinary_params)
        parameterization.method = "optcuts_test"
        parameterization.metrics["omega_parameterization_mode"] = "optcuts_test"
        parameterization.metrics["requested_omega_parameterization_mode"] = "optcuts_test"
        parameterization.metrics["optcuts_test_initial_omega_boundary"] = np.asarray(
            parameterization.omega_boundary, dtype=float
        ).tolist()

        domain = _reference_domain_without_invalid_optcuts_boundary_diagnostic(
            pipeline, parameterization, grid, ordinary_params
        )
        footprint = pipeline._build_reference_m2d(grid, domain, ordinary_params)
        outline = _quad_union_boundary(footprint)
        if len(outline) < 4:
            raise RuntimeError("OPTCUTS_TEST2_SCALE_AWARE_GRID_OUTLINE_EMPTY")

        boundary_targets, counts = _build_test_targets(parameterization, outline)
        uv_final, opt_info = _staged_boundary_resolve(parameterization, boundary_targets)
        parameterization.uv_vertices_2d = np.asarray(uv_final, dtype=float)
        parameterization.omega_boundary = np.asarray(outline, dtype=float)
        parameterization.metrics.update(
            {
                "optcuts_test_enabled": True,
                "optcuts_test2_scale_aware_enabled": True,
                "optcuts_test_model": (
                    "scale-aware selected official OptCuts topology -> initial Omega -> M2D quad footprint -> "
                    "grid-cell outline -> full OptCuts cut-boundary loop mapped monotonically to outline -> "
                    "flip-safe staged Symmetric Dirichlet interior resolve"
                ),
                "optcuts_test_seam_topology_preserved": True,
                "optcuts_test_seam_geometry_fixed_during_resolve": False,
                "optcuts_test_boundary_semantics": (
                    "OptCuts seam copies remain part of the Omega boundary and are re-mapped with that boundary"
                ),
                "optcuts_test_outer_boundary_mapping": "full ordered cut boundary -> grid-outline arclength",
                "optcuts_test_grid_outline_vertex_count": int(len(outline) - 1),
                "optcuts_test_grid_outline": np.asarray(outline, dtype=float).tolist(),
                "optcuts_test_reference_m2d_face_count": int(len(np.asarray(footprint.faces))),
                **counts,
                **{f"optcuts_test_{k}": v for k, v in opt_info.items()},
            }
        )
        setattr(parameterization, "_optcuts_test_grid_outline", np.asarray(outline, dtype=float))
        print(
            "[OPTCUTS-TEST2-SCALE-AWARE] "
            f"selected_initial_scale={parameterization.metrics.get('optcuts_scale_aware_selected_scale_range')} "
            f"hard_feasible={parameterization.metrics.get('optcuts_scale_aware_selected_scale_hard_feasible')} "
            f"footprint_quads={len(np.asarray(footprint.faces))} outline={len(outline)-1} "
            f"stages={opt_info.get('continuation_accepted_stage_count', 0)}"
        )
        return parameterization

    pipeline._build_surface_parameterization = builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = builder

    pipeline._onestring_test2_scale_aware_parameterization_installed = True
    print("[OPTCUTS-TEST2-SCALE-AWARE-ROUTE] installed")


__all__ = ["install_optcuts_test2_scale_aware_parameterization_patch"]
