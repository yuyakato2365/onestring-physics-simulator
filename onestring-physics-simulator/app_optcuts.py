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
from onestring_physics.final_k2d_view_stage_patch import install_final_k2d_view_stage_patch

import streamlit as st
st.session_state["eq5_rendered_this_run"] = False
install_paper_ui_20260916_patch()
install_final_k2d_view_stage_patch()
runpy.run_path(str(ROOT / "app_optcuts_core_20260916.py"), run_name="__main__")

# Stage rendering is owned by the legacy View-stage branches.  Do not append a\n# K2D figure unconditionally after them; that made K3D/T3D/T2D views appear to\n# be replaced by K2D output after every Streamlit rerun.\n\n# Diagnostic requested for the experimental Eq.(5) route: show the solver
# trajectory only when it was not already rendered immediately after K2D.
try:
    if not st.session_state.get("eq5_rendered_this_run", False):
        render_eq5_iteration_history(st)
except Exception as exc:
    print(f"[PAPER-EQ5-HISTORY-UI] skipped: {exc}")
