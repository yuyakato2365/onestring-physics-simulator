"""Experimental OneString app with semantic Split -> separated Panels.

Run with:
  python -m streamlit run app_split_panels.py --server.port 8502

Unlike the first debug version, Split now changes the actual M2D/K2D planar
geometry. M2D stores the pre-layout Omega coordinates privately so M3D inverse
mapping is still evaluated at the correct c^{-1} coordinates.
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
from onestring_physics.split_panel_debug_patch import install_split_panel_debug  # noqa: E402

install_split_panel_debug(pipeline, opt_debug)

# build_onestring_design() delegates to the backed-up original pipeline module,
# whose functions keep that module's global namespace. Mirror the experimental
# wrappers there as well; otherwise the UI can show a patched function while the
# actual design flow still calls the old M2D/K2D implementation.
pipeline._original._build_m2d = pipeline._build_m2d
pipeline._original._lift_m2d_to_m3d = pipeline._lift_m2d_to_m3d
pipeline._original._optimize_k2d = pipeline._optimize_k2d

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
