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
    # Paper is enabled by default and intentionally disables Omega choices.
    paper=[x for x in t.sidebar.checkbox
           if x.label=="Paper-oriented computation mode (2025 Sec.4–5)"]
    assert len(paper)==1
    paper[0].set_value(False)
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    selectboxes=[s for s in t.sidebar.selectbox
                 if s.label=="Omega parameterization mode"]
    assert len(selectboxes)==1
    menu=selectboxes[0]
    assert not menu.disabled
    # AppTest exposes formatted display labels instead of the internal values.
    grid_labels=[x for x in menu.options if "Grid-aware OptCuts Ω" in x]
    assert len(grid_labels)==1
    menu.set_value(grid_labels[0])
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert t.session_state["_onestring_grid_aware_omega_selected"] is True
    assert any("Grid整列の強さ（実験）"==x.label for x in t.sidebar.slider)
    assert not any(x.key=="optcuts_grid_aware_experiment" for x in t.sidebar.checkbox)
    menu=[s for s in t.sidebar.selectbox
          if s.label=="Omega parameterization mode"][0]
    baseline=[x for x in menu.options if "2026-09-18" in x or "optcuts_test2"==x]
    if not baseline:
        baseline=[x for x in menu.options if "2026-09-13" in x]
    assert len(baseline)>=1
    menu.set_value(baseline[0])
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert t.session_state["_onestring_grid_aware_omega_selected"] is False
    assert not any(x.label=="Grid整列の強さ（実験）" for x in t.sidebar.slider)



def test_area_threshold_grid_boundary_is_selectable_and_slider_is_independent():
    """Paper-OFF mode selection and 50% slider are actually interactive."""
    from streamlit.testing.v1 import AppTest
    app=Path(__file__).parents[1]/"app_optcuts_20260923_extrusion_aware.py"
    t=AppTest.from_file(str(app),default_timeout=30).run()
    assert not t.exception, [str(x.message) for x in t.exception]
    paper=[x for x in t.sidebar.checkbox
           if x.label=="Paper-oriented computation mode (2025 Sec.4–5)"]
    assert len(paper)==1
    paper[0].set_value(False)
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    modes=[x for x in t.sidebar.selectbox
           if x.label=="Omega parameterization mode"]
    assert len(modes)==1
    candidate=[x for x in modes[0].options
               if "Area-Threshold Grid Boundary" in x]
    assert len(candidate)==1
    modes[0].set_value(candidate[0])
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert t.session_state["_onestring_area_grid_boundary_omega_selected"] is True
    th=[x for x in t.sidebar.slider
        if x.label=="GridセルのΩ内面積割合・採用閾値"]
    assert len(th)==1
    assert float(th[0].value)==.5
    th[0].set_value(.65)
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert float(t.session_state["optcuts_area_grid_coverage_threshold"])==.65
    modes=[x for x in t.sidebar.selectbox
           if x.label=="Omega parameterization mode"]
    baseline=[x for x in modes[0].options if "2026-09-18" in x]
    assert len(baseline)>0
    modes[0].set_value(baseline[0])
    t.run()
    assert not t.exception, [str(x.message) for x in t.exception]
    assert t.session_state["_onestring_area_grid_boundary_omega_selected"] is False
    assert not any(x.label=="GridセルのΩ内面積割合・採用閾値"
                   for x in t.sidebar.slider)
