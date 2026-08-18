"""Streamlit display helpers for the automatic Large Steps conditioning stage."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


def install_large_steps_visualization_patch() -> None:
    try:
        import streamlit as st
        from . import visualization
    except Exception:
        return
    if getattr(visualization, "_LARGE_STEPS_VISUALIZATION_PATCH_INSTALLED", False):
        return

    original = visualization.figure_surface_mesh

    def figure_surface_mesh_with_conditioning(surface: Any, *args: Any, **kwargs: Any):
        show_conditioned = bool(st.session_state.get("_onestring_show_conditioned_surface", False))
        if not show_conditioned:
            return original(surface, *args, **kwargs)

        conditioned = getattr(surface, "_large_steps_conditioned_vertices", None)
        metrics = getattr(surface, "_large_steps_conditioning_metrics", {}) or {}
        if conditioned is None:
            fig = original(surface, *args, **kwargs)
            try:
                fig.update_layout(title="Conditioned S unavailable for this run")
            except Exception:
                pass
            return fig

        conditioned_surface = SimpleNamespace(
            vertices=np.asarray(conditioned, dtype=float),
            faces=np.asarray(surface.faces, dtype=int),
            kind=f"{getattr(surface, 'kind', 'surface')}_large_steps_conditioned",
        )
        fig = original(conditioned_surface, *args, **kwargs)
        before_angle = float(metrics.get("large_steps_before_minimum_angle_degrees", 0.0))
        after_angle = float(metrics.get("large_steps_after_minimum_angle_degrees", 0.0))
        before_q = float(metrics.get("large_steps_before_triangle_quality_p05", 0.0))
        after_q = float(metrics.get("large_steps_after_triangle_quality_p05", 0.0))
        try:
            fig.update_layout(
                title=(
                    "Conditioned S — Large Steps mesh conditioning "
                    f"(min angle {before_angle:.2f}° → {after_angle:.2f}°, "
                    f"quality p05 {before_q:.3f} → {after_q:.3f})"
                )
            )
        except Exception:
            pass
        try:
            st.caption(
                "Large Steps conditioning: fixed connectivity / fixed 3D boundary / "
                f"u=(I+λL)v, λ={float(metrics.get('large_steps_lambda', 0.0)):.3g}. "
                f"Surface deviation max={float(metrics.get('large_steps_surface_deviation_max', 0.0)):.3g}."
            )
        except Exception:
            pass
        return fig

    visualization.figure_surface_mesh = figure_surface_mesh_with_conditioning
    visualization._LARGE_STEPS_VISUALIZATION_PATCH_INSTALLED = True


__all__ = ["install_large_steps_visualization_patch"]
