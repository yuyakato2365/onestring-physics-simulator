"""Runtime integration of the official OptCuts executable into S -> Omega.

The patch is deliberately opt-in.  Existing BFF, CEPS, free-boundary, and M2D
Split paths are delegated unchanged unless ``omega_parameterization_mode`` is
exactly ``"optcuts"``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig, run_official_optcuts


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false; got {raw!r}")


def _config_from_params(params: Any) -> OptCutsConfig:
    executable = getattr(params, "optcuts_executable", None)
    if not executable:
        executable = os.environ.get("ONESTRING_OPTCUTS_EXECUTABLE") or None
    return OptCutsConfig(
        executable=executable,
        distortion_bound=float(
            getattr(
                params,
                "optcuts_distortion_bound",
                _env_float("ONESTRING_OPTCUTS_DISTORTION_BOUND", 4.1),
            )
        ),
        lambda_init=float(
            getattr(
                params,
                "optcuts_lambda_init",
                _env_float("ONESTRING_OPTCUTS_LAMBDA_INIT", 0.999),
            )
        ),
        method_type=int(
            getattr(
                params,
                "optcuts_method_type",
                _env_int("ONESTRING_OPTCUTS_METHOD_TYPE", 0),
            )
        ),
        use_bijectivity=bool(
            getattr(
                params,
                "optcuts_use_bijectivity",
                _env_bool("ONESTRING_OPTCUTS_USE_BIJECTIVITY", True),
            )
        ),
        initial_cut_option=int(
            getattr(
                params,
                "optcuts_initial_cut_option",
                _env_int("ONESTRING_OPTCUTS_INITIAL_CUT_OPTION", 0),
            )
        ),
        timeout_seconds=float(
            getattr(
                params,
                "optcuts_timeout_seconds",
                _env_float("ONESTRING_OPTCUTS_TIMEOUT_SECONDS", 600.0),
            )
        ),
    )


def _build_optcuts_parameterization(pipeline: Any, surface: Any, target: Any, grid: Any, params: Any):
    del target, grid  # OptCuts works directly on the triangulated target surface.

    surface_vertices = np.asarray(surface.vertices, dtype=float)
    surface_faces = np.asarray(surface.faces, dtype=int)[:, :3]
    result = run_official_optcuts(surface_vertices, surface_faces, _config_from_params(params))
    loop = [int(v) for v in result.boundary_loops[0]]
    boundary = result.uv_vertices_2d[loop + [loop[0]]]

    metrics: dict[str, object] = {
        **result.metrics,
        "parameterization_exactness_label": "official_optcuts_external_backend",
        "parameterization_warning": (
            "Official OptCuts research code is used as an external parameterization backend. "
            "This is not the One String paper's BFF parameterization."
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
        "omega_correspondence_model": (
            "official OptCuts cut topology + UV embedding; M2D inverse mapping uses paired "
            "surface_faces/uv_faces barycentric coordinates"
        ),
        "paper_flow_stage": "S -> Omega by official OptCuts external backend",
        "paper_exactness_warning": (
            "OptCuts is an experimental alternative to OneString's BFF stage, not a paper-faithful replacement."
        ),
        "bff_implemented": False,
        "optcuts_implemented": True,
        "omega_boundary_fixed": False,
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape": "free",
        "omega_boundary_constraint_model": "OptCuts optimized seams and arbitrary embedding",
        "fallbacks_used": [],
    }

    parameterization = pipeline.SurfaceParameterization(
        method="optcuts",
        surface_vertices_3d=np.asarray(result.surface_vertices_3d, dtype=float),
        surface_faces=np.asarray(result.surface_faces, dtype=int),
        uv_vertices_2d=np.asarray(result.uv_vertices_2d, dtype=float),
        uv_faces=np.asarray(result.uv_faces, dtype=int),
        omega_boundary=np.asarray(boundary, dtype=float),
        triangle_acceleration=None,
        metrics=metrics,
    )

    # Reuse the current branch's generic quality audit only if it can handle the
    # separate 3D/UV index spaces produced by seams.  Failure here must not turn
    # into a fallback; the backend's own Jacobian/SVD metrics remain available.
    quality_fn = getattr(pipeline, "_omega_quality_metrics", None)
    if callable(quality_fn):
        try:
            quality = dict(quality_fn(parameterization))
        except Exception as exc:
            parameterization.metrics["optcuts_generic_quality_audit_status"] = (
                f"skipped:{type(exc).__name__}:{exc}"
            )
        else:
            parameterization.metrics.update(quality)
            parameterization.metrics["optcuts_generic_quality_audit_status"] = "completed"

    return parameterization


def install_optcuts_pipeline_patch(pipeline: Any) -> None:
    """Install one non-stacking OptCuts dispatch wrapper on the active pipeline."""
    if getattr(pipeline, "_onestring_optcuts_pipeline_patch_installed", False):
        return

    original_builder = getattr(pipeline, "_build_surface_parameterization")
    pipeline._onestring_pre_optcuts_build_surface_parameterization = original_builder

    def optcuts_dispatch(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts":
            return original_builder(surface, target, grid, params)
        return _build_optcuts_parameterization(pipeline, surface, target, grid, params)

    optcuts_dispatch.__name__ = "_build_surface_parameterization_with_optcuts"
    optcuts_dispatch.__module__ = __name__

    pipeline._build_surface_parameterization = optcuts_dispatch
    original_module = getattr(pipeline, "_original", None)
    if original_module is not None:
        original_module._build_surface_parameterization = optcuts_dispatch

    # build_onestring_design in this repository may execute in the globals of a
    # backed-up compatibility module. Rewire only this one symbol; do not reload
    # or reinstall any Split/K2D wrappers.
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original_module, "build_onestring_design", None) if original_module is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = optcuts_dispatch

    pipeline._onestring_optcuts_pipeline_patch_installed = True
