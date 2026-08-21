"""Persist the active OptCuts mode across stages that replace mesh metrics.

Several pipeline stages construct a fresh QuadMesh with a new metrics dict.  A
later guard therefore must not infer OptCuts mode only from mesh.metrics.  This
small wrapper records the selected parameterization mode on the active pipeline
module at S->Omega dispatch time.
"""
from __future__ import annotations

from typing import Any


def install_optcuts_run_flag_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_run_flag_patch_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def build_surface_parameterization_with_run_flag(surface: Any, target: Any, grid: Any, params: Any):
        active = str(getattr(params, "omega_parameterization_mode", "")) == "optcuts"
        pipeline._onestring_optcuts_active_run = bool(active)
        original_module = getattr(pipeline, "_original", None)
        if original_module is not None:
            try:
                original_module._onestring_optcuts_active_run = bool(active)
            except Exception:
                pass
        return base_builder(surface, target, grid, params)

    pipeline._build_surface_parameterization = build_surface_parameterization_with_run_flag
    original_module = getattr(pipeline, "_original", None)
    if original_module is not None:
        original_module._build_surface_parameterization = build_surface_parameterization_with_run_flag
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original_module, "build_onestring_design", None) if original_module is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = build_surface_parameterization_with_run_flag

    pipeline._onestring_optcuts_run_flag_patch_installed = True


__all__ = ["install_optcuts_run_flag_patch"]
