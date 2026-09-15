"""Expose existing OneString assembly animations through the Streamlit View stage selector.

This patch is intentionally UI-only.  It does not change design or deployment
numerics.  Special animation choices are stored in session state while the
legacy app receives a safe static fallback stage; after the legacy app finishes
rendering, ``render_selected_animation_view`` renders the requested animation
from the most recently built ``OneStringDesignState``.
"""
from __future__ import annotations

from typing import Any, Callable


ANIMATION_VIEWS = (
    "Animation: T2D -> T3D (simultaneous hinge contraction)",
    "Animation: T2D -> T3D (boundary/string order)",
    "Animation: T2D -> T3D + string path",
)

_SESSION_SELECTED = "_onestring_view_stage_animation_choice"
_SESSION_STATE = "_onestring_last_design_state"


def install_view_stage_animation_selector() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_view_stage_animation_selector_installed", False):
        return

    previous_selectbox = st.selectbox

    def selectbox_with_animation_views(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label != "View stage":
            return previous_selectbox(*args, **kwargs)

        if len(args) >= 2:
            options = list(args[1])
            for item in ANIMATION_VIEWS:
                if item not in options:
                    options.append(item)
            args = (args[0], options, *args[2:])
        else:
            options = list(kwargs.get("options", []))
            for item in ANIMATION_VIEWS:
                if item not in options:
                    options.append(item)
            kwargs = {**kwargs, "options": options}

        selected = previous_selectbox(*args, **kwargs)
        if selected in ANIMATION_VIEWS:
            st.session_state[_SESSION_SELECTED] = selected
            # The legacy renderer does not know our synthetic stage names.  Keep
            # its normal rendering path alive and replace/add the requested
            # animation after the legacy app body has completed.
            return "T3D"

        st.session_state[_SESSION_SELECTED] = None
        return selected

    st.selectbox = selectbox_with_animation_views
    st._onestring_view_stage_animation_selector_installed = True


def wrap_build_to_cache_state(build_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a thin build wrapper that caches the newest design state in Streamlit."""
    if getattr(build_fn, "_onestring_caches_view_stage_state", False):
        return build_fn

    def build_and_cache(*args: Any, **kwargs: Any) -> Any:
        state = build_fn(*args, **kwargs)
        try:
            import streamlit as st
            st.session_state[_SESSION_STATE] = state
        except Exception:
            pass
        return state

    build_and_cache.__name__ = getattr(build_fn, "__name__", "build_onestring_design")
    build_and_cache.__module__ = getattr(build_fn, "__module__", __name__)
    build_and_cache.__doc__ = getattr(build_fn, "__doc__", None)
    build_and_cache._onestring_caches_view_stage_state = True  # type: ignore[attr-defined]
    build_and_cache._onestring_wrapped_build = build_fn  # type: ignore[attr-defined]
    return build_and_cache


def render_selected_animation_view() -> bool:
    """Render the animation selected through View stage, if one is selected."""
    try:
        import streamlit as st
        from .animation import assembly_progress_animation
    except Exception:
        return False

    selected = st.session_state.get(_SESSION_SELECTED)
    if selected not in ANIMATION_VIEWS:
        return False
    state = st.session_state.get(_SESSION_STATE)
    if state is None:
        st.info("Run the design calculation once to generate this animation view.")
        return True

    if selected == ANIMATION_VIEWS[0]:
        fig = assembly_progress_animation(
            state,
            frame_count=56,
            max_tiles=900,
            show_target=True,
            show_path=False,
            motion_mode="simultaneous_hinge_contraction",
            title="T2D -> T3D assembly: simultaneous hinge contraction",
        )
    elif selected == ANIMATION_VIEWS[1]:
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
    st.plotly_chart(fig, use_container_width=True)
    return True


__all__ = [
    "ANIMATION_VIEWS",
    "install_view_stage_animation_selector",
    "wrap_build_to_cache_state",
    "render_selected_animation_view",
]
