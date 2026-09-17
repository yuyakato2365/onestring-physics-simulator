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
from onestring_physics.eq5_iteration_history_view import render_eq5_iteration_history, render_final_k2d_result

import streamlit as st
st.session_state["eq5_rendered_this_run"] = False
install_paper_ui_20260916_patch()
runpy.run_path(str(ROOT / "app_optcuts_core_20260916.py"), run_name="__main__")

# The legacy main K2D renderer expects the old independent-tile FlatTileLayout
# and can be empty for the reconstructed Eq.(5) topology.  Keep numerics and all
# downstream stages untouched; only add a final-result view backed directly by
# the solved Eq.(5) xy stored in this Streamlit session.
try:
    if st.session_state.get("eq5_history"):
        render_final_k2d_result(st)
except Exception as exc:
    print(f"[PAPER-EQ5-FINAL-K2D-UI] skipped: {exc}")

# Diagnostic requested for the experimental Eq.(5) route: show the solver
# trajectory only when it was not already rendered immediately after K2D.
try:
    if not st.session_state.get("eq5_rendered_this_run", False):
        render_eq5_iteration_history(st)
except Exception as exc:
    print(f"[PAPER-EQ5-HISTORY-UI] skipped: {exc}")
