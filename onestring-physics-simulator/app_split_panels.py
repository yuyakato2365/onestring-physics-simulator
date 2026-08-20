"""Experimental OneString app with semantic Split -> separated Panels.

Run with:
  python -m streamlit run app_split_panels.py --server.port 8502

Split changes the actual M2D/K2D planar geometry. The pre-layout Omega
coordinates are retained for M2D -> M3D inverse mapping.

The execution path is patched in four layers:
1) semantic Split/Panel behavior,
2) direct wiring into legacy build-function globals,
3) a final M2D topology pass that applies complete grid-line cuts,
4) structured Split diagnostics written to logs/split_debug.jsonl.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onestring_physics import onestring_pipeline as pipeline  # noqa: E402
from onestring_physics import optimization_debug_visualization as opt_debug  # noqa: E402
from onestring_physics import final_split_panel_pass as final_split_module  # noqa: E402
from onestring_physics.split_panel_debug_patch import install_split_panel_debug  # noqa: E402
from onestring_physics.split_panel_force_wiring import install_force_split_panel_wiring  # noqa: E402
from onestring_physics.final_split_panel_pass import install_final_split_panel_pass  # noqa: E402
from onestring_physics.split_diagnostics import install_split_diagnostics  # noqa: E402

install_split_panel_debug(pipeline, opt_debug)
install_force_split_panel_wiring(pipeline)
install_final_split_panel_pass(pipeline)
install_split_diagnostics(pipeline, final_split_module, ROOT)

# The final pass wires _build_m2d directly. Keep the downstream semantic wrappers
# visible in every legacy build-function global dictionary as well.
for fn in (
    getattr(pipeline, "build_onestring_design", None),
    getattr(pipeline._original, "build_onestring_design", None),
    getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
):
    glb = getattr(fn, "__globals__", None)
    if isinstance(glb, dict):
        glb["_lift_m2d_to_m3d"] = pipeline._lift_m2d_to_m3d
        glb["_optimize_k2d"] = pipeline._optimize_k2d

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
