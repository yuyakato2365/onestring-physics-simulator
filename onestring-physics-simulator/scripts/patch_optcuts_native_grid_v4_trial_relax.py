#!/usr/bin/env python3
"""Defer Grid candidate validation to the scorer, validate accepted cuts once.

Candidate trials use ``allowCutThrough=False``.  Their exact trial topology is
validated by ``oneStringScoreTrial`` after the cut is constructed.  The actual
accepted OptCuts operation uses the default ``allowCutThrough=True`` and is
validated once before returning to the ordinary OptCuts optimizer.

Critically, this patch performs NO harmonic/global solve.  Candidate enumeration
must remain cheap, and accepted cuts are globally relaxed by the normal OptCuts
optimization phase after topology selection.
"""
from __future__ import annotations
from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE"


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
                // ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE
                // Candidate trials are checked by oneStringScoreTrial.  The real
                // accepted cut is checked once here, with no global solve.
                if(allowCutThrough &&
                   (!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this))) {
                    throw std::runtime_error("ONESTRING_GRID_APPLIED_INTERIOR_CUT_INVALID");
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
            if(allowCutThrough &&
               (!checkInversion(true) || !oneStringAllCohesiveSeamSidesGridAligned(*this))) {
                throw std::runtime_error("ONESTRING_GRID_APPLIED_BOUNDARY_CUT_INVALID");
            }
        }
'''
    if text.count(old_boundary) != 1:
        raise RuntimeError(f"boundary Grid validation anchor count={text.count(old_boundary)}")
    text = text.replace(old_boundary, new_boundary, 1)

    path.write_text(text, encoding="utf-8")
    print(f"Applied Grid-OptCuts deferred trial/accepted-cut validation: {path}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_trial_relax_patch(a.root.resolve())
