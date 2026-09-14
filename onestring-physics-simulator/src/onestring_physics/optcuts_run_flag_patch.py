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


def _install_optcuts_weight_ui_patch() -> None:
    """Loosen K3D weight limits and expose E_Conn as a numeric input.

    The legacy app owns these sidebar widgets.  ``app_optcuts.py`` imports this
    module before executing that app, so a narrow Streamlit wrapper lets the
    OptCuts launcher change only the relevant controls without copying the large
    legacy application file.
    """
    try:
        import streamlit as st
    except Exception:
        return

    if getattr(st, "_onestring_optcuts_weight_ui_patch_installed", False):
        return

    original_number_input = st.number_input
    original_slider = st.slider

    def patched_number_input(label: str, *args: Any, **kwargs: Any):
        if label == "w_planar / EPlanar":
            # Keep the existing default/step but remove the restrictive ceiling.
            kwargs["max_value"] = 1_000_000.0
            return original_number_input(label, *args, **kwargs)
        if label == "w_square / ESquare":
            # The quadrilateral/square regularizer previously stopped at 100.
            # Allow substantially stronger experiments from the UI.
            kwargs["max_value"] = 1_000.0
            return original_number_input(label, *args, **kwargs)
        return original_number_input(label, *args, **kwargs)

    def patched_slider(label: str, *args: Any, **kwargs: Any):
        if label == "hinge connection weight":
            # Legacy call shape is slider(label, min, max, value, step, ...).
            min_value = float(args[0]) if len(args) > 0 else float(kwargs.pop("min_value", 0.1))
            value = float(args[2]) if len(args) > 2 else float(kwargs.pop("value", 8.0))
            step = float(args[3]) if len(args) > 3 else float(kwargs.pop("step", 0.1))
            help_text = kwargs.pop("help", "Weight for E_Conn: pairwise hinge vertex coincidence.")
            key = kwargs.pop("key", None)
            return original_number_input(
                label,
                min_value=min(0.0, min_value),
                max_value=1_000.0,
                value=min(1_000.0, max(0.0, value)),
                step=step,
                help=help_text,
                key=key,
                format="%.3f",
                **kwargs,
            )
        return original_slider(label, *args, **kwargs)

    st.number_input = patched_number_input
    st.slider = patched_slider
    st._onestring_optcuts_weight_ui_patch_installed = True


_install_optcuts_weight_ui_patch()


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
