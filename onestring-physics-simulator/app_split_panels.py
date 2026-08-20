"""Experimental OneString app with semantic Split -> separated Panels.

Run with:
  python -m streamlit run app_split_panels.py --server.port 8502

Split changes the actual M2D/K2D planar geometry. The pre-layout Omega
coordinates are retained for M2D -> M3D inverse mapping.
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
from onestring_physics.split_panel_force_wiring import install_force_split_panel_wiring  # noqa: E402

# First install the semantic Split implementation itself.
install_split_panel_debug(pipeline, opt_debug)

# Then patch the exact global names resolved by the legacy build function. This
# is intentionally stronger than rebinding module attributes because this repo
# delegates through backed-up pipeline modules whose functions retain their own
# global namespaces.
install_force_split_panel_wiring(pipeline)

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
