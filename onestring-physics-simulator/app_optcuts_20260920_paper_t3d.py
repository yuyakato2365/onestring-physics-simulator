"""2026-09-20 paper-aligned T3D launcher.

Version: 2026-09-20-paper-t3d
"""
from __future__ import annotations
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

# Numerical switch: this launcher always uses the paper T3D implementation.
install_paper_t3d_20260920_patch(pipeline)
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design

print(
    "[2026-09-20-PAPER-T3D] enabled: K3D -> T3D uses mesh-normal offset + "
    "Eq.(2) planarity optimization on fixed 8-vertex frustums."
)

# The visible Version selector is defined by app_split_panels.py, which is
# reached later through app_optcuts_20260916.py -> app_optcuts.py ->
# app_optcuts_core_20260916.py.  Set an explicit environment flag consumed by
# that app instead of trying to wrap st.selectbox (paper_ui unwraps wrappers).
import os
os.environ["ONESTRING_PAPER_T3D_20260920"] = "1"

runpy.run_path(str(ROOT / "app_optcuts_20260916.py"), run_name="__main__")
