"""2026-09-20 paper-aligned T3D launcher.

Version: 2026-09-20-paper-t3d

Compared with the k2d-recovery baseline, this version changes only K3D -> T3D:
1. offset shared K3D vertices along mesh vertex normals by the requested thickness;
2. build fixed 8-vertex / 6-quad frustum tiles;
3. optimize top, bottom, and four contact faces with the paper's Eq.(2)
   best-fit-plane projection energy;
4. retain a per-tile rigid top->bottom transform for the later K2D -> T2D step.

Variable-topology half-space clipping, wedge/pyramid recovery, local-thickness
recovery, junction caps, and global T3D clipping are deliberately not used.
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

install_paper_t3d_20260920_patch(pipeline)
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design

print(
    "[2026-09-20-PAPER-T3D] enabled: K3D -> T3D uses mesh-normal offset + "
    "Eq.(2) planarity optimization on fixed 8-vertex frustums."
)

runpy.run_path(str(ROOT / "app_optcuts_20260916.py"), run_name="__main__")
