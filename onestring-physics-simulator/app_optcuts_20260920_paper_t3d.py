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
