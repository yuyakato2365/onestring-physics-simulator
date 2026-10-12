"""Keep Streamlit view state valid when a rebuilt grid changes panel topology.

This module is intentionally independent of Streamlit so the state transition
can be regression-tested without starting a server or constructing 3D figures.
"""
from __future__ import annotations


def prepare_completed_pipeline_display(session_state, *, old_grid_size, new_grid_size):
    """Reset stale view/component selections ONLY after a successful grid rebuild.

    A requested grid change does not invalidate the last successful result
    until the new pipeline has actually returned. This function must therefore
    be called only after the caller stores the new state, and before it creates
    the View stage selectbox or any keyed panel-number widgets.
    """
    new_size = int(new_grid_size)
    changed = old_grid_size is not None and int(old_grid_size) != new_size
    # Full builds can change the number of T2D trials even with the same grid.
    # A selected trial from a previous run is not necessarily present now.
    session_state.pop("dual_hinge_inspection_trial", None)
    session_state.pop("dual_hinge_show_large_history", None)
    if changed:
        # Don't immediately attempt huge 3D plots from the old stage. Display
        # the lightweight pipeline summary first, preserving the new result.
        session_state["onestring_view_stage"] = "Pipeline View"
        for key in (
            "cross_stage_panel_id",
            "t2d_top_panel_id",
            "t2d_dual_panel_id",
            "t2d_top_pending_panel_id",
            "t2d_dual_pending_panel_id",
            "t2d_top_click_k3d_sel_0",
            "t2d_dual_click_k3d_sel_0",
        ):
            session_state.pop(key, None)
    session_state["onestring_completed_grid_size"] = new_size
    session_state["onestring_grid_view_reset"] = bool(changed)
    return bool(changed)
