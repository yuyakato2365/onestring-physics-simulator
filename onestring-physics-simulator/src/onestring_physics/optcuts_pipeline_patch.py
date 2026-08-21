"""Runtime integration of the official OptCuts executable into S -> Omega.

Two modes are intentionally separated:

* ``optcuts``: official OptCuts output is used directly. No OneString grid
  seam constraint, axis snap, or constrained UV solve is applied.
* ``optcuts_grid``: official OptCuts is used as the proposal backend for the
  experimental grid-constrained pipeline.

All non-OptCuts modes delegate unchanged to the pre-existing pipeline.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig, run_official_optcuts
from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust


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
            getattr(params, "optcuts_distortion_bound", _env_float("ONESTRING_OPTCUTS_DISTORTION_BOUND", 4.1))
        ),
        lambda_init=float(
            getattr(params, "optcuts_lambda_init", _env_float("ONESTRING_OPTCUTS_LAMBDA_INIT", 0.999))
        ),
        method_type=int(
            getattr(params, "optcuts_method_type", _env_int("ONESTRING_OPTCUTS_METHOD_TYPE", 0))
        ),
        use_bijectivity=bool(
            getattr(params, "optcuts_use_bijectivity", _env_bool("ONESTRING_OPTCUTS_USE_BIJECTIVITY", True))
        ),
        initial_cut_option=int(
            getattr(params, "optcuts_initial_cut_option", _env_int("ONESTRING_OPTCUTS_INITIAL_CUT_OPTION", 0))
        ),
        timeout_seconds=float(
            getattr(params, "optcuts_timeout_seconds", _env_float("ONESTRING_OPTCUTS_TIMEOUT_SECONDS", 600.0))
        ),
    )


def _align_uv_to_optcuts_seam_axis(parameterization: Any) -> float:
    """Rigidly rotate UV so the dominant OptCuts seam direction becomes +u.

    Used only by ``optcuts_grid``. ``optcuts`` keeps the official UV unchanged.
    """
    payload = extract_connected_seam_payload_robust(parameterization)
    segments = np.asarray(payload.get("segments", np.zeros((0, 2, 2))), dtype=float)
    if len(segments) == 0:
        parameterization.metrics["optcuts_fabrication_axis_status"] = "no_internal_seam"
        parameterization.metrics["optcuts_fabrication_axis_rotation_degrees"] = 0.0
        return 0.0

    moment = np.zeros((2, 2), dtype=float)
    total = 0.0
    for segment in segments:
        direction = np.asarray(segment[1] - segment[0], dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 1e-12 or not np.isfinite(length):
            continue
        unit = direction / length
        moment += length * np.outer(unit, unit)
        total += length
    if total <= 1e-12:
        parameterization.metrics["optcuts_fabrication_axis_status"] = "degenerate_internal_seam"
        parameterization.metrics["optcuts_fabrication_axis_rotation_degrees"] = 0.0
        return 0.0

    values, vectors = np.linalg.eigh(moment)
    axis = np.asarray(vectors[:, int(np.argmax(values))], dtype=float)
    angle = float(math.atan2(axis[1], axis[0]))
    while angle > math.pi / 2.0:
        angle -= math.pi
    while angle <= -math.pi / 2.0:
        angle += math.pi

    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    center = np.mean(uv, axis=0) if len(uv) else np.zeros(2, dtype=float)
    c, s = math.cos(-angle), math.sin(-angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    rotated = (uv - center[None, :]) @ rotation.T + center[None, :]
    parameterization.uv_vertices_2d = rotated

    loop = [int(v) for v in parameterization.metrics.get("boundary_loop", [])]
    if loop:
        parameterization.omega_boundary = rotated[loop + [loop[0]]]

    degrees = float(math.degrees(-angle))
    parameterization.metrics.update({
        "optcuts_fabrication_axis_status": "aligned",
        "optcuts_fabrication_axis_model": "length-weighted principal axis of robust OptCuts seam graph",
        "optcuts_fabrication_axis_rotation_degrees": degrees,
        "optcuts_fabrication_axis_u": [float(math.cos(angle)), float(math.sin(angle))],
        "optcuts_fabrication_axis_v": [float(-math.sin(angle)), float(math.cos(angle))],
        "optcuts_uv_rotation_is_rigid": True,
        "optcuts_distortion_changed_by_axis_alignment": False,
    })
    print(f"[OPTCUTS-AXIS] rigid_uv_rotation_degrees={degrees:.4f} seam_segments={len(segments)}")
    return degrees


def _build_optcuts_parameterization(
    pipeline: Any,
    surface: Any,
    target: Any,
    grid: Any,
    params: Any,
    requested_mode: str = "optcuts",
):
    del target, grid
    if requested_mode not in {"optcuts", "optcuts_grid"}:
        raise ValueError(f"unsupported OptCuts mode: {requested_mode}")

    surface_vertices = np.asarray(surface.vertices, dtype=float)
    surface_faces = np.asarray(surface.faces, dtype=int)[:, :3]
    result = run_official_optcuts(surface_vertices, surface_faces, _config_from_params(params))
    loop = [int(v) for v in result.boundary_loops[0]]
    boundary = result.uv_vertices_2d[loop + [loop[0]]]

    is_grid = requested_mode == "optcuts_grid"
    # Important: existing grid wrappers key off method == "optcuts". Keep the
    # official mode on a distinct internal method so those wrappers cannot touch it.
    internal_method = "optcuts" if is_grid else "optcuts_official"
    metrics: dict[str, object] = {
        **result.metrics,
        "parameterization_exactness_label": "official_optcuts_external_backend",
        "parameterization_warning": (
            "Official OptCuts research code is used as an external parameterization backend. "
            "This is not the One String paper's BFF parameterization."
        ),
        "paper_compliance_status": "experimental_nonpaper_parameterization",
        "omega_boundary_mode": "paper_default",
        "omega_parameterization_mode": requested_mode,
        "requested_omega_parameterization_mode": requested_mode,
        "boundary_vertex_count": int(len(loop)),
        "boundary_loop": loop,
        "height_field_shortcut_used": False,
        "harmonic_solve_performed": False,
        "omega_corresponds_to_S": True,
        "omega_correspondence_model": (
            "official OptCuts cut topology + UV embedding"
            if not is_grid
            else "official OptCuts proposal + experimental OneString grid-constrained processing"
        ),
        "paper_flow_stage": "S -> Omega by official OptCuts external backend",
        "paper_exactness_warning": (
            "OptCuts is an experimental alternative to OneString's BFF stage, not a paper-faithful replacement."
        ),
        "bff_implemented": False,
        "optcuts_implemented": True,
        "optcuts_grid_mode": bool(is_grid),
        "optcuts_internal_method": internal_method,
        "omega_boundary_fixed": False,
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape": "free",
        "omega_boundary_constraint_model": (
            "official OptCuts optimized seams"
            if not is_grid
            else "official OptCuts proposal followed by explicit grid-constrained stage"
        ),
        "fallbacks_used": [],
    }

    parameterization = pipeline.SurfaceParameterization(
        method=internal_method,
        surface_vertices_3d=np.asarray(result.surface_vertices_3d, dtype=float),
        surface_faces=np.asarray(result.surface_faces, dtype=int),
        uv_vertices_2d=np.asarray(result.uv_vertices_2d, dtype=float),
        uv_faces=np.asarray(result.uv_faces, dtype=int),
        omega_boundary=np.asarray(boundary, dtype=float),
        triangle_acceleration=None,
        metrics=metrics,
    )

    if is_grid:
        _align_uv_to_optcuts_seam_axis(parameterization)
    else:
        parameterization.metrics.update({
            "optcuts_fabrication_axis_status": "not_applied_official_mode",
            "optcuts_fabrication_axis_rotation_degrees": 0.0,
            "optcuts_original_uv_used_as_final": True,
        })

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
        if mode not in {"optcuts", "optcuts_grid"}:
            return original_builder(surface, target, grid, params)
        return _build_optcuts_parameterization(
            pipeline,
            surface,
            target,
            grid,
            params,
            requested_mode=mode,
        )

    optcuts_dispatch.__name__ = "_build_surface_parameterization_with_optcuts"
    optcuts_dispatch.__module__ = __name__

    pipeline._build_surface_parameterization = optcuts_dispatch
    original_module = getattr(pipeline, "_original", None)
    if original_module is not None:
        original_module._build_surface_parameterization = optcuts_dispatch

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original_module, "build_onestring_design", None) if original_module is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = optcuts_dispatch

    pipeline._onestring_optcuts_pipeline_patch_installed = True
