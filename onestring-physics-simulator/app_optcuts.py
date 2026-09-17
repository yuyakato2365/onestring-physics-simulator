"""Main OptCuts launcher with the 2026-09-16 paper-grounded UI layer."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onestring_physics.paper_ui_20260916_patch import install_paper_ui_20260916_patch
from onestring_physics.eq5_iteration_history_view import render_eq5_iteration_history

import streamlit as st
st.session_state["eq5_rendered_this_run"] = False
install_paper_ui_20260916_patch()
runpy.run_path(str(ROOT / "app_optcuts_core_20260916.py"), run_name="__main__")

# Diagnostic requested for the experimental Eq.(5) route: after the ordinary
# app has rendered its final K2D result, show the solver trajectory separately.
# This does not alter optimization state or the normal K2D visualization.
try:
    import streamlit as st
    if not st.session_state.get("eq5_rendered_this_run", False):
        render_eq5_iteration_history(st)
except Exception as exc:
    print(f"[PAPER-EQ5-HISTORY-UI] skipped: {exc}")
