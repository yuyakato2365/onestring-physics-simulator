#!/usr/bin/env python3
"""Make native Grid-OptCuts V4 validate cuts after constrained relaxation.

`allowCutThrough=False` is used by the Grid candidate trial helpers, while the
real accepted OptCuts operation uses the default `True`.  V4 originally checked
triangle inversion inside cutPath/splitEdgeOnBoundary immediately after moving
new seam vertices onto the lattice.  That rejected candidates before the trial
scorer could reparameterize the newly cut topology.

Trials now defer validity to oneStringScoreTrial(), which performs a harmonic
solve with all Grid/boundary vertices fixed.  Real accepted cuts perform the
same constrained relaxation immediately, then enforce inversion and seam-grid
validity before returning to the main OptCuts optimizer.
"""
from __future__ import annotations
from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_RELAX_BEFORE_VALIDATE"


def apply_trial_relax_patch(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False

    old_interior = '''            if(oneStringGridEncodedInterior) {
                oneStringLockGridVertices(*this, {path[0], path[1], path[2], nV});
                if(!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this)) {
                    throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID");
                }
            }
'''
    new_interior = '''            if(oneStringGridEncodedInterior) {
                oneStringLockGridVertices(*this, {path[0], path[1], path[2], nV});
                // ONESTRING_GRID_NATIVE_V4_RELAX_BEFORE_VALIDATE
                // Candidate trials pass allowCutThrough=false and are relaxed/
                // checked by oneStringScoreTrial. A real accepted cut uses the
                // default true and is relaxed here before becoming authoritative.
                if(allowCutThrough) {
                    if(!oneStringRelaxGridTrialUV(*this) ||
                       !checkInversion(true) ||
                       !oneStringAllCohesiveSeamSidesGridAligned(*this)) {
                        throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID");
                    }
                }
            }
'''
    if text.count(old_interior) != 1:
        raise RuntimeError(f"interior Grid validation anchor count={text.count(old_interior)}")
    text = text.replace(old_interior, new_interior, 1)

    old_boundary = '''        if(oneStringGridEncodedBoundary) {
            oneStringGridLockedVert.insert(vI_boundary);
            oneStringGridLockedVert.insert(oneStringGridBoundaryDuplicate);
            oneStringGridLockedVert.insert(vI_interior);
            fixedVert.insert(vI_boundary);
            fixedVert.insert(oneStringGridBoundaryDuplicate);
            fixedVert.insert(vI_interior);
            if(!checkInversion(true)) {
                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVERTED");
            }
        }
'''
    new_boundary = '''        if(oneStringGridEncodedBoundary) {
            oneStringGridLockedVert.insert(vI_boundary);
            oneStringGridLockedVert.insert(oneStringGridBoundaryDuplicate);
            oneStringGridLockedVert.insert(vI_interior);
            fixedVert.insert(vI_boundary);
            fixedVert.insert(oneStringGridBoundaryDuplicate);
            fixedVert.insert(vI_interior);
            if(allowCutThrough) {
                if(!oneStringRelaxGridTrialUV(*this) ||
                   !checkInversion(true) ||
                   !oneStringAllCohesiveSeamSidesGridAligned(*this)) {
                    throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVALID");
                }
            }
        }
'''
    if text.count(old_boundary) != 1:
        raise RuntimeError(f"boundary Grid validation anchor count={text.count(old_boundary)}")
    text = text.replace(old_boundary, new_boundary, 1)

    path.write_text(text, encoding="utf-8")
    print(f"Applied Grid-OptCuts relaxed trial/accepted-cut validation: {path}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_trial_relax_patch(a.root.resolve())
