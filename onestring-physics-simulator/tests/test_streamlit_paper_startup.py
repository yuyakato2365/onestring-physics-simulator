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
