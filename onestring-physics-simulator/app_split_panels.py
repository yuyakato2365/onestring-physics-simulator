"""Experimental OneString app with explicit full-grid Split -> Panel debugging.

Run with:
  python -m streamlit run app_split_panels.py --server.port 8502

This installs the Split/Panel debug patch first, then delegates to the normal
app.py wrapper. The default app.py remains untouched while this behavior is
validated.
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
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
