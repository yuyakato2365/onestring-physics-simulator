"""2026-09-20 paper-aligned T3D launcher.

Version: 2026-09-20-paper-t3d
"""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import onestring_physics as package
from onestring_physics import onestring_pipeline as pipeline
from onestring_physics.paper_t3d_20260920_patch import install_paper_t3d_20260920_patch
from onestring_physics.k3d_iteration_history_patch import install_k3d_iteration_history_patch

# Numerical switch: this launcher always uses the paper T3D implementation.
install_paper_t3d_20260920_patch(pipeline)
install_k3d_iteration_history_patch(pipeline)
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design

print(
    "[2026-09-20-PAPER-T3D] enabled: K3D -> T3D uses mesh-normal offset + "
    "Eq.(2) planarity optimization on fixed 8-vertex frustums."
)

# UI flags consumed by the legacy app / split-panel launcher.
os.environ["ONESTRING_PAPER_T3D_20260920"] = "1"
os.environ["ONESTRING_PAPER_T3D_PERSIST_STATIC_VIEW"] = "1"

runpy.run_path(str(ROOT / "app_optcuts_20260916.py"), run_name="__main__")


# Dedicated paper-T3D fallback renderer.  app.py owns the View-stage selector,
# but the nested runpy launchers can make later diagnostic renderers obscure
# that section.  Re-render the selected static stage at the very end so K3D and
# T3D are always visible, and put K3D history directly beside the K3D result.
try:
    import streamlit as st
    from onestring_physics.k3d_iteration_history_view import render_k3d_iteration_history

    _state = st.session_state.get("onestring_state")
    if _state is not None:
        st.markdown("---")
        st.markdown("## Paper-T3D results")
        _tabs = st.tabs(["K3D", "T3D", "K3D objective history"])
        with _tabs[0]:
            try:
                from onestring_physics.visualization import figure_quad_mesh
                st.plotly_chart(
                    figure_quad_mesh(_state.mesh_3d_optimized, title="K3D"),
                    width="stretch",
                    key="paper_t3d_fallback_k3d",
                )
            except Exception as _exc:
                st.warning(f"K3D fallback view skipped: {_exc}")
        with _tabs[1]:
            try:
                from onestring_physics.visualization import figure_tile_assembly
                st.plotly_chart(
                    figure_tile_assembly(_state.tiles_3d),
                    width="stretch",
                    key="paper_t3d_fallback_t3d",
                )
                st.write(_state.tiles_3d.metrics)
            except Exception as _exc:
                st.warning(f"T3D fallback view skipped: {_exc}")
        with _tabs[2]:
            if not render_k3d_iteration_history(st, _state):
                st.warning(
                    "K3D objective history was not recorded for this run. "
                    "Run the pipeline once after pulling this revision; the CPU/SciPy route records it."
                )
except Exception as _exc:
    print(f"[2026-09-20-PAPER-T3D-RESULTS] skipped: {_exc}")
