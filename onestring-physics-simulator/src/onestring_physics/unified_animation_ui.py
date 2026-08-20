"""Single UI route for OneString process/assembly animations.

Synthetic animation choices are rendered immediately underneath the legacy
``View stage`` selector.  The legacy app receives a private sentinel that matches
no static stage, so it never substitutes T3D or another unrelated view.

The same renderer functions are reused for the fresh-run diagnostic sequence:
actual Omega accepted states -> one Split panel view -> actual K2D checkpoints.
No numerical Split/K2D/Omega behavior is changed here.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from . import process_animation_view_patch as process_views
from .view_stage_animation_patch import ANIMATION_VIEWS as ASSEMBLY_ANIMATION_VIEWS


OMEGA_VIEW = process_views.PROCESS_ANIMATION_VIEWS[0]
K2D_VIEW = process_views.PROCESS_ANIMATION_VIEWS[1]
PROCESS_VIEWS = (OMEGA_VIEW, K2D_VIEW)
ALL_SYNTHETIC_VIEWS = tuple(ASSEMBLY_ANIMATION_VIEWS) + PROCESS_VIEWS
LEGACY_STAGE_SENTINEL = "__ONESSTRING_SYNTHETIC_ANIMATION_VIEW__"

_SESSION_SELECTED = "_onestring_unified_animation_view_choice"
_SESSION_OMEGA_FIGURE = "_onestring_exact_omega_animation_figure"
_SESSION_RENDERED = "_onestring_synthetic_view_rendered_this_rerun"


def _figure_title(fig: Any) -> str:
    try:
        title = getattr(getattr(fig, "layout", None), "title", None)
        text = getattr(title, "text", None)
        if text:
            return str(text)
    except Exception:
        pass
    try:
        data = fig.to_plotly_json() if hasattr(fig, "to_plotly_json") else dict(fig)
        title = (data.get("layout", {}) or {}).get("title", "")
        if isinstance(title, dict):
            title = title.get("text", "")
        return str(title or "")
    except Exception:
        return ""


def install_exact_omega_figure_capture() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_exact_omega_figure_capture_installed", False):
        return

    previous = st.plotly_chart

    def plotly_chart_with_omega_capture(fig: Any, *args: Any, **kwargs: Any):
        title = _figure_title(fig)
        if "Accepted Ω state" in title:
            try:
                payload = fig.to_plotly_json() if hasattr(fig, "to_plotly_json") else fig
                st.session_state[_SESSION_OMEGA_FIGURE] = payload
            except Exception:
                pass
        return previous(fig, *args, **kwargs)

    st.plotly_chart = plotly_chart_with_omega_capture
    st._onestring_exact_omega_figure_capture_installed = True


def cache_omega_payload(frames: Any, faces: Any, boundary_loop: Any, summary: Any = None) -> None:
    try:
        import streamlit as st
        st.session_state[process_views._SESSION_OMEGA] = {
            "frames": list(frames),
            "faces": np.asarray(faces, dtype=int).copy(),
            "boundary_loop": np.asarray(boundary_loop, dtype=int).copy(),
            "summary": dict(summary or {}),
        }
    except Exception:
        pass


def _render_exact_omega(optimization_debug_module: Any) -> bool:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except Exception:
        return False

    exact = st.session_state.get(_SESSION_OMEGA_FIGURE)
    if exact:
        st.markdown("### Animation: Omega optimization (accepted states)")
        st.caption(
            "Accepted states recorded during the Omega optimization run."
        )
        st.plotly_chart(go.Figure(exact), config={"responsive": True})
        return True

    payload = st.session_state.get(process_views._SESSION_OMEGA)
    if payload:
        renderer = getattr(
            optimization_debug_module,
            "_onestring_original_omega_renderer_for_unified_view",
            None,
        )
        if renderer is None:
            renderer = optimization_debug_module.render_omega_flip_debug_animation
        st.markdown("### Animation: Omega optimization (accepted states)")
        renderer(
            payload["frames"],
            payload["faces"],
            payload["boundary_loop"],
            payload.get("summary"),
        )
        return True

    status = st.session_state.get("_onestring_omega_process_cache_status")
    st.warning(
        "Omega optimization animation data is not available for this run. "
        f"cache status={status!r}. Re-run the pipeline once after restarting the app."
    )
    return False


def _render_k2d_process() -> bool:
    try:
        import streamlit as st
    except Exception:
        return False
    payload = st.session_state.get(process_views._SESSION_K2D)
    if not payload:
        st.warning(
            "K2D optimization checkpoints are not available for this run. "
            "Re-run the pipeline once after restarting the app."
        )
        return False
    process_views._render_k2d(payload)
    return True


def _resolve_state(state: Any | None = None) -> Any | None:
    if state is not None:
        return state
    try:
        import streamlit as st
    except Exception:
        return None
    for key in ("onestring_state", "_onestring_last_design_state"):
        value = st.session_state.get(key)
        if value is not None:
            return value
    return None


def _render_assembly(selected: str, state: Any | None = None) -> bool:
    try:
        import streamlit as st
        from .animation import assembly_progress_animation
    except Exception:
        return False
    state = _resolve_state(state)
    if state is None:
        st.info("Run the design calculation once to generate this animation view.")
        return True

    if selected == ASSEMBLY_ANIMATION_VIEWS[0]:
        fig = assembly_progress_animation(
            state,
            frame_count=56,
            max_tiles=900,
            show_target=True,
            show_path=False,
            motion_mode="simultaneous_hinge_contraction",
            title="T2D -> T3D assembly: simultaneous hinge contraction",
        )
    elif selected == ASSEMBLY_ANIMATION_VIEWS[1]:
        fig = assembly_progress_animation(
            state,
            frame_count=56,
            max_tiles=900,
            show_target=True,
            show_path=False,
            motion_mode="boundary_string_order",
            title="T2D -> T3D assembly: boundary/string order",
        )
    else:
        fig = assembly_progress_animation(
            state,
            frame_count=56,
            max_tiles=900,
            show_target=True,
            show_path=True,
            motion_mode="simultaneous_hinge_contraction",
            title="T2D -> T3D assembly with string path",
        )
    st.markdown(f"### {selected}")
    st.plotly_chart(fig, config={"responsive": True})
    return True


def _render_selected_value(selected: str, optimization_debug_module: Any, state: Any | None = None) -> bool:
    if selected == OMEGA_VIEW:
        return _render_exact_omega(optimization_debug_module)
    if selected == K2D_VIEW:
        return _render_k2d_process()
    if selected in ASSEMBLY_ANIMATION_VIEWS:
        return _render_assembly(selected, state)
    return False


def install_unified_view_stage_selector() -> None:
    """Add and render all synthetic animations directly at the View-stage site."""
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_unified_animation_selector_installed", False):
        return

    previous = st.selectbox

    def selectbox_with_unified_animations(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label != "View stage":
            return previous(*args, **kwargs)

        if len(args) >= 2:
            options = list(args[1])
            for item in ALL_SYNTHETIC_VIEWS:
                if item not in options:
                    options.append(item)
            args = (args[0], options, *args[2:])
        else:
            options = list(kwargs.get("options", []))
            for item in ALL_SYNTHETIC_VIEWS:
                if item not in options:
                    options.append(item)
            kwargs = {**kwargs, "options": options}

        selected = previous(*args, **kwargs)
        if selected in ALL_SYNTHETIC_VIEWS:
            st.session_state[_SESSION_SELECTED] = selected
            # Render NOW, immediately beneath the selector.  Waiting until the
            # outer launcher regains control places the animation after unrelated
            # legacy sections such as Current approximations.
            try:
                from . import optimization_debug_visualization as opt_debug
                rendered = _render_selected_value(selected, opt_debug, _resolve_state())
                st.session_state[_SESSION_RENDERED] = bool(rendered)
            except Exception as exc:
                st.session_state[_SESSION_RENDERED] = False
                st.exception(exc)
            return LEGACY_STAGE_SENTINEL

        st.session_state[_SESSION_SELECTED] = None
        st.session_state[_SESSION_RENDERED] = False
        return selected

    st.selectbox = selectbox_with_unified_animations
    st._onestring_unified_animation_selector_installed = True


def install_unified_animation_ui() -> None:
    install_exact_omega_figure_capture()
    install_unified_view_stage_selector()


def render_selected_synthetic_view(
    optimization_debug_module: Any,
    *,
    state: Any | None = None,
) -> bool:
    """Fallback renderer for callers that do not go through the patched selector."""
    try:
        import streamlit as st
    except Exception:
        return False
    if bool(st.session_state.get(_SESSION_RENDERED, False)):
        return True
    selected = st.session_state.get(_SESSION_SELECTED)
    if selected not in ALL_SYNTHETIC_VIEWS:
        return False
    rendered = _render_selected_value(selected, optimization_debug_module, state)
    st.session_state[_SESSION_RENDERED] = bool(rendered)
    return bool(rendered)


def render_postrun_process_sequence(
    state: Any,
    optimization_debug_module: Any,
    split_renderer: Callable[[Any, Any, Any | None], None],
) -> None:
    """Show Omega -> Split -> K2D exactly once after a fresh calculation."""
    try:
        import streamlit as st
    except Exception:
        return
    if state is None:
        return

    st.divider()
    st.markdown("## Optimization process animations")
    st.caption(
        "Fresh-run diagnostics: actual accepted Omega states, one Split panel "
        "view, then actual K2D optimization checkpoints."
    )

    _render_exact_omega(optimization_debug_module)

    try:
        split_renderer(
            state.mesh_2d_initial,
            state.mesh_2d_optimized,
            state.mesh_3d_optimized,
        )
    except Exception as exc:
        st.warning(f"Split diagnostic rendering failed: {exc}")

    _render_k2d_process()


__all__ = [
    "ALL_SYNTHETIC_VIEWS",
    "ASSEMBLY_ANIMATION_VIEWS",
    "PROCESS_VIEWS",
    "OMEGA_VIEW",
    "K2D_VIEW",
    "LEGACY_STAGE_SENTINEL",
    "cache_omega_payload",
    "install_unified_animation_ui",
    "install_unified_view_stage_selector",
    "install_exact_omega_figure_capture",
    "render_selected_synthetic_view",
    "render_postrun_process_sequence",
]
