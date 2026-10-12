"""Regression checks for result display after grid / topology changes.

No Streamlit server or heavyweight optimization required.
"""
from __future__ import annotations

import ast
from pathlib import Path

from onestring_physics.ui_result_recovery import prepare_completed_pipeline_display


def _previous_state():
    return {
        "onestring_view_stage": "T2D Dual Hinge",
        "cross_stage_panel_id": 299,
        "t2d_dual_panel_id": 299,
        "t2d_top_panel_id": 299,
        "t2d_dual_pending_panel_id": 299,
        "t2d_top_pending_panel_id": 299,
        "dual_hinge_inspection_trial": 10,
        "dual_hinge_show_large_history": True,
        "t2d_dual_show_large_panel_inspector": True,
        "t2d_top_show_large_panel_inspector": True,
        "collision_split_preview_selected": 6,
        "collision_split_preview_stage": "K2D",
        "collision_split_preview_red_overlay": True,
        "collision_split_preview_hinges": True,
        "collision_split_preview_show_large": True,
        "onestring_render_overview": True,
        "unrelated_user_preference": "untouched",
    }


def test_changed_grid_resets_heavy_stage_and_stale_panel_ids():
    state = _previous_state()
    changed = prepare_completed_pipeline_display(
        state, old_grid_size=30, new_grid_size=12,
    )
    assert changed
    assert state["onestring_view_stage"] == "Pipeline View"
    assert state["onestring_completed_grid_size"] == 12
    assert state["onestring_grid_view_reset"] is True
    for key in (
        "cross_stage_panel_id", "t2d_dual_panel_id", "t2d_top_panel_id",
        "t2d_dual_pending_panel_id", "t2d_top_pending_panel_id",
        "dual_hinge_inspection_trial", "dual_hinge_show_large_history",
        "t2d_top_show_large_panel_inspector",
        "t2d_dual_show_large_panel_inspector",
        "collision_split_preview_selected",
        "collision_split_preview_stage",
        "collision_split_preview_red_overlay",
        "collision_split_preview_hinges",
        "collision_split_preview_show_large",
        "onestring_render_overview",
    ):
        assert key not in state
    assert state["unrelated_user_preference"] == "untouched"


def test_same_grid_keeps_requested_stage_but_resets_panel_topology():
    state = _previous_state()
    changed = prepare_completed_pipeline_display(
        state, old_grid_size=20, new_grid_size=20,
    )
    assert not changed
    assert state["onestring_view_stage"] == "T2D Dual Hinge"
    assert state["onestring_completed_grid_size"] == 20
    assert "cross_stage_panel_id" not in state
    assert "dual_hinge_inspection_trial" not in state
    assert "dual_hinge_show_large_history" not in state
    assert state["onestring_render_overview"] is True


def test_first_pipeline_run_preserves_default_selection():
    state = {"onestring_view_stage": "Omega"}
    assert not prepare_completed_pipeline_display(
        state, old_grid_size=None, new_grid_size=20,
    )
    assert state["onestring_view_stage"] == "Omega"
    assert state["onestring_completed_grid_size"] == 20


def test_result_is_published_before_costly_stage_rendering():
    # Avoid accidentally moving the result-status UI back *after* a huge
    # Plotly render in a future refactor.
    path = Path(__file__).resolve().parents[1] / "app_runtime_base.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    assert source.index('st.session_state.onestring_state = build_onestring_design(') < (
        source.index('prepare_completed_pipeline_display(')
    )
    assert source.index('prepare_completed_pipeline_display(') < (
        source.index('view_stage = st.selectbox(')
    )
    assert source.index('_saved_grid_size = st.session_state.get(') < (
        source.index('view_stage = st.selectbox(')
    )
    assert source.index('st.success("計算結果をセッションに保持しました。 "') < (
        source.index('view_stage = st.selectbox(')
    )
    assert 'st.caption(_result_text)' in source
    assert 'Pipeline Viewの描画に失敗しました' in source
    assert '_large_history = _history_tile_count * _history_frame_count > 12000' in source
    assert 'if pose_history and _render_history:' in source
    assert 'if common > 200:' in source
    assert 'render_split_candidate_previews(st, state, split_history)' in source
    assert '棄却・採用した仮Split候補を3Dで描画して比較' in source
    assert source.index('st.session_state["onestring_just_completed_pipeline"] = True') < (
        source.index('    st.rerun()\n\nstate = st.session_state.onestring_state')
    )
    assert source.index('progress_bar.empty()') < (
        source.index('    st.rerun()\n\nstate = st.session_state.onestring_state')
    )
    assert 'st.session_state.pop("onestring_just_completed_pipeline", False)' in source
    assert 'st.subheader("計算済み工程の概要")' in source
    assert source.index('st.subheader("計算済み工程の概要")') < (
        source.index('st.plotly_chart(figure_pipeline_overview(state)')
    )
