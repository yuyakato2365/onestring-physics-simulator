"""Headless smoke test for dated Paper app startup and rerun-state stability."""
from __future__ import annotations

from pathlib import Path


def test_20260923_launcher_renders_sidebar_without_exception_on_repeated_runs():
    from streamlit.testing.v1 import AppTest
    app=Path(__file__).parents[1]/"app_optcuts_20260923_extrusion_aware.py"
    test=AppTest.from_file(str(app),default_timeout=30)
    test.run()
    assert not test.exception, [str(x.message) for x in test.exception]
    # 13 legitimate reruns can be caused by normal setting changes; they
    # must NEVER be called a rerun LOOP or blocked by the old 12-runs guard.
    for _ in range(13):
        test.run()
        assert not test.exception, [str(x.message) for x in test.exception]
    assert len(test.sidebar.checkbox)>0
    assert all("PAPER_UI_RERUN_LOOP_DETECTED" not in str(x.message)
               for x in test.exception)
    assert any("Run OneString pipeline" in str(x.label)
               for x in test.sidebar.button)



def test_grid_aware_is_a_separate_omega_selector_mode_not_checkbox():
    """The 09-18 numerical carrier stays optcuts_test while mode remains distinct."""
    from streamlit.testing.v1 import AppTest
    app=Path(__file__).parents[1]/"app_optcuts_20260923_extrusion_aware.py"
    t=AppTest.from_file(str(app),default_timeout=30).run()
    assert not t.exception, [str(x.message) for x in t.exception]
    selectboxes=[s for s in t.sidebar.selectbox
                 if s.label=="Omega parameterization mode"]
    assert len(selectboxes)==1
    menu=selectboxes[0]
    assert "optcuts_grid_aware" in menu.options
    menu.set_value("optcuts_grid_aware")
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert t.session_state["_onestring_grid_aware_omega_selected"] is True
    assert any("Grid整列の強さ（実験）"==x.label for x in t.sidebar.slider)
    assert not any(x.key=="optcuts_grid_aware_experiment" for x in t.sidebar.checkbox)
    menu=[s for s in t.sidebar.selectbox
          if s.label=="Omega parameterization mode"][0]
    menu.set_value("optcuts_test2")
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert t.session_state["_onestring_grid_aware_omega_selected"] is False
    assert not any(x.label=="Grid整列の強さ（実験）" for x in t.sidebar.slider)
