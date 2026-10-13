#!/usr/bin/env python3
"""Integration check for the prescribed-corner official BFF rectangle solver."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script without an editable pip install. The full
# simulator itself requires Python 3.11+ (see pyproject.toml).
if sys.version_info < (3, 11):
    raise SystemExit(
        f"OneString requires Python 3.11+, but this is {sys.version.split()[0]}. "
        "Run with your project virtualenv or python3.11."
    )
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from onestring_physics.reference_bff import run_official_bff


def main() -> None:
    # A deliberately non-rectangular eight-sided disk with one center.
    angles=np.linspace(0.0,2.0*np.pi,8,endpoint=False)
    radii=np.array([1.0,1.1,.9,1.05,1.,1.12,.93,1.07])
    rim=np.column_stack((radii*np.cos(angles),radii*np.sin(angles),
                         np.zeros_like(angles)))
    vertices=np.vstack((rim, [[0.,0.,.15]]))
    faces=np.array([[8,k,(k+1)%8] for k in range(8)],int)
    output=run_official_bff(
        vertices,faces,boundary_policy="target_rectangle"
    )
    assert output.metrics["bff_rectangle_verified"]
    assert output.metrics["bff_rectangle_width"]>0
    assert output.metrics["bff_rectangle_height"]>0
    print("RECTANGULAR_BFF_OK")
    print("target_policy:",output.metrics["bff_boundary_policy_effective"])
    print("backend:",output.metrics["parameterization_backend_name"])
    print("rectangle_width:",output.metrics["bff_rectangle_width"])
    print("rectangle_height:",output.metrics["bff_rectangle_height"])
    print("max_boundary_deviation:",output.metrics["bff_rectangle_side_max_deviation"])


if __name__=="__main__":
    main()
