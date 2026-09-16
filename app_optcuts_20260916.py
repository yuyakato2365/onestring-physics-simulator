"""2026-09-16 OneString / OptCuts launcher.

Purpose
-------
Keep the existing app_optcuts.py launcher untouched, but activate the dated
paper-aligned K2D route before that launcher installs its normal patch stack.

For the visible ``optcuts_test`` mode this enables the existing
``optcuts_paper_k2d_20260914_patch`` implementation, whose policy is:

- K2D is the flat metric/layout stage, not the hinge-closure stage.
- use the ordinary whole-mesh K2D optimization path (EEdge / collision /
  fabrication layout behavior) rather than the OptCuts-specific hard-hinge,
  hard-SAT, or global-SE(2) replacement;
- do not force physical hinge coincidence in K2D;
- defer hinge placement/closure to the later T2D / Section 4.4-style stage.

The numerical implementation lives in the existing dated patch so that this
launcher is a small, reversible switch.  The old app_optcuts.py command remains
available for direct comparison.
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

# Activate the paper-aligned dated route.  The patch itself additionally checks
# that the runtime mode is optcuts_test, so unrelated parameterization modes are
# left unchanged.
os.environ["ONESTRING_PAPER_K2D_20260914"] = "1"

import onestring_physics as package  # noqa: E402
from onestring_physics import onestring_pipeline as pipeline  # noqa: E402
from onestring_physics.optcuts_paper_k2d_20260914_patch import (  # noqa: E402
    install_optcuts_paper_k2d_20260914_patch,
)

install_optcuts_paper_k2d_20260914_patch(pipeline)

# Keep public package aliases synchronized with the patched pipeline before the
# ordinary OptCuts launcher adds its remaining wrappers/UI.
package.onestring_pipeline = pipeline
package.build_onestring_design = pipeline.build_onestring_design

print(
    "[2026-09-16-PAPER-K2D] enabled: ordinary K2D optimization; "
    "OptCuts-specific hard hinge/SAT/global-SE2 K2D replacements disabled; "
    "hinge closure deferred to T2D."
)

runpy.run_path(str(ROOT / "app_optcuts.py"), run_name="__main__")
