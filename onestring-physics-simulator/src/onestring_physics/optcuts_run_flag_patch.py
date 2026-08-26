"""Persist the active OptCuts mode across stages that replace mesh metrics."""
from __future__ import annotations

from typing import Any

from .optcuts_grid_orientation_patch import install_optcuts_grid_orientation_patch
from .optcuts_grid_seam_sidecar_patch import install_optcuts_grid_seam_sidecar_patch
from .optcuts_official_binary_patch import install_official_optcuts_binary_separation


# Install before any OptCuts run. Ordinary ``optcuts`` must never resolve to the
# Grid-OptCuts executable; the latter is reserved for ``optcuts_grid`` only.
install_official_optcuts_binary_separation()

# Install before any Grid-OptCuts run. Orientation stabilizes the backend's
# global frame first; seam transfer then wraps the native pipeline call and
# applies the identical final rigid frame transform to the C++ cohE sidecar.
install_optcuts_grid_orientation_patch()
install_optcuts_grid_seam_sidecar_patch()


def install_optcuts_run_flag_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_run_flag_patch_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def build_surface_parameterization_with_run_flag(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        active = mode in {"optcuts", "optcuts_grid"}
        grid_active = mode == "optcuts_grid"
        pipeline._onestring_optcuts_active_run = bool(active)
        pipeline._onestring_optcuts_grid_active_run = bool(grid_active)
        original_module = getattr(pipeline, "_original", None)
        if original_module is not None:
            try:
                original_module._onestring_optcuts_active_run = bool(active)
                original_module._onestring_optcuts_grid_active_run = bool(grid_active)
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
