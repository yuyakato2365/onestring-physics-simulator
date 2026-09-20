"""2026-09-20 paper-aligned T3D launcher.

Version: 2026-09-20-paper-t3d

K3D -> T3D follows the paper route: mesh-normal offset, fixed eight-vertex
quadrilateral frustums, then Eq.(2) planarity optimization of top, bottom and
contact faces. Variable-topology recovery is not used in this version.
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
from onestring_physics.paper_t3d_20260920_patch import (
    install_paper_t3d_20260920_patch,
    install_paper_t3d_20260920_version_ui,
)

# Install the numerical route and the visible version-selector entry before the
# dated launcher builds the rest of the UI/patch stack.
install_paper_t3d_20260920_patch(pipeline)
install_paper_t3d_20260920_version_ui()
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design

print(
    "[2026-09-20-PAPER-T3D] enabled: K3D -> T3D uses mesh-normal offset + "
    "Eq.(2) planarity optimization on fixed 8-vertex frustums."
)

runpy.run_path(str(ROOT / "app_optcuts_20260916.py"), run_name="__main__")
