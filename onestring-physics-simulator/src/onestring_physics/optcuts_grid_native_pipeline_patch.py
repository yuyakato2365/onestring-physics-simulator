"""S -> Omega dispatch for native Grid-Constrained OptCuts.

This wrapper intercepts only ``omega_parameterization_mode == 'optcuts_grid'``.
Official ``optcuts`` remains the untouched authors' backend.  No Python seam
post-processing or constrained continuation solve is performed here.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .optcuts_grid_native_backend import run_native_grid_optcuts
from .optcuts_pipeline_patch import _config_from_params


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric; got {raw!r}") from exc


def install_native_grid_optcuts_pipeline_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_native_grid_optcuts_pipeline_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def native_grid_builder(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts_grid":
            return base_builder(surface, target, grid, params)

        xyz = np.asarray(surface.vertices, dtype=float)
        faces = np.asarray(surface.faces, dtype=int)[:, :3]
        h = max(float(getattr(params, "tile_size", getattr(grid, "tile_size", 0.0))), 1.0e-10)
        angle_deg = float(getattr(
            params,
            "optcuts_grid_angle_degrees",
            _env_float("ONESTRING_OPTCUTS_GRID_ANGLE_DEGREES", 0.0),
        ))
        phase_u = float(getattr(
            params,
            "optcuts_grid_phase_u",
            _env_float("ONESTRING_OPTCUTS_GRID_PHASE_U", 0.0),
        ))
        phase_v = float(getattr(
            params,
            "optcuts_grid_phase_v",
            _env_float("ONESTRING_OPTCUTS_GRID_PHASE_V", 0.0),
        ))
        max_snap_steps = float(getattr(
            params,
            "optcuts_grid_max_snap_steps",
            _env_float("ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS", 2.0),
        ))

        result = run_native_grid_optcuts(
            xyz,
            faces,
            _config_from_params(params),
            grid_h=h,
            angle_degrees=angle_deg,
            phase_u=phase_u,
            phase_v=phase_v,
            max_snap_steps=max_snap_steps,
        )
        if not result.boundary_loops:
            raise RuntimeError("OPTCUTS_GRID_NATIVE_NO_UV_BOUNDARY")
        loop = [int(v) for v in result.boundary_loops[0]]
        boundary = np.asarray(result.uv_vertices_2d, dtype=float)[loop + [loop[0]]]
        metrics: dict[str, object] = {
            **result.metrics,
            "parameterization_exactness_label": "native_grid_constrained_optcuts_cpp",
            "parameterization_warning": (
                "Experimental OneString Grid-OptCuts modifies the OptCuts split candidate search itself."
            ),
            "paper_compliance_status": "experimental_nonpaper_parameterization",
            "omega_parameterization_mode": "optcuts_grid",
            "requested_omega_parameterization_mode": "optcuts_grid",
            "boundary_loop": loop,
            "boundary_vertex_count": int(len(loop)),
            "omega_corresponds_to_S": True,
            "omega_boundary_fixed": False,
            "omega_boundary_forced_rectangle": False,
            "omega_boundary_shape": "free_with_grid_constrained_internal_seams",
            "optcuts_implemented": True,
            "optcuts_grid_constrained": True,
            "optcuts_grid_native": True,
            "optcuts_posthoc_extra_seam": False,
            "optcuts_original_uv_used_as_final": True,
            "optcuts_grid_unit": float(h),
            "grid_phase_u": float(phase_u),
            "grid_phase_v": float(phase_v),
            "optcuts_grid_angle_degrees": float(angle_deg),
            "optcuts_grid_allowed_seam_directions": "two fixed global orthogonal axes",
            "optcuts_grid_seam_geometry": (
                "native OptCuts candidate paths embedded as H/V/H-V/V-H lattice segments before selection"
            ),
            "optcuts_grid_candidate_selection_stage": "inside OptCuts computeLocalLDec/querySplit",
            "optcuts_grid_python_reparameterization_used": False,
            "fallbacks_used": [],
        }
        parameterization = pipeline.SurfaceParameterization(
            method="optcuts_grid_native",
            surface_vertices_3d=np.asarray(result.surface_vertices_3d, dtype=float),
            surface_faces=np.asarray(result.surface_faces, dtype=int),
            uv_vertices_2d=np.asarray(result.uv_vertices_2d, dtype=float),
            uv_faces=np.asarray(result.uv_faces, dtype=int),
            omega_boundary=boundary,
            triangle_acceleration=None,
            metrics=metrics,
        )
        setattr(parameterization, "_onestring_grid_unit", float(h))

        quality_fn = getattr(pipeline, "_omega_quality_metrics", None)
        if callable(quality_fn):
            try:
                parameterization.metrics.update(dict(quality_fn(parameterization)))
                parameterization.metrics["optcuts_generic_quality_audit_status"] = "completed"
            except Exception as exc:
                parameterization.metrics["optcuts_generic_quality_audit_status"] = (
                    f"skipped:{type(exc).__name__}:{exc}"
                )

        print(
            "[OPTCUTS-GRID-NATIVE] "
            f"h={h:g} angle={angle_deg:.3f}deg phase=({phase_u:.6g},{phase_v:.6g}) "
            f"max_snap_steps={max_snap_steps:g} uv_vertices={len(result.uv_vertices_2d)}"
        )
        return parameterization

    pipeline._build_surface_parameterization = native_grid_builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = native_grid_builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = native_grid_builder
    pipeline._onestring_native_grid_optcuts_pipeline_installed = True


__all__ = ["install_native_grid_optcuts_pipeline_patch"]
