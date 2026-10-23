"""Headless smoke test for dated Paper app startup and rerun-state stability."""
from __future__ import annotations

from pathlib import Path


def test_20260923_launcher_renders_sidebar_without_exception_on_repeated_runs():
    from streamlit.testing.v1 import AppTest
    app=Path(__file__).parents[1]/"app_optcuts_20260923_extrusion_aware.py"
    test=AppTest.from_file(str(app),default_timeout=30)
    test.run()
    assert not test.exception, [str(x.message) for x in test.exception]
    # A second run must not mutate a keyed version widget or call
    # st.rerun forever, even with Paper mode selected by default.
    test.run()
    assert not test.exception, [str(x.message) for x in test.exception]
    assert len(test.sidebar.checkbox)>0
