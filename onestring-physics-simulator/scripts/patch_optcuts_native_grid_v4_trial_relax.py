#!/usr/bin/env python3
"""Validate Grid cuts cheaply while preserving OptCuts global bijectivity.

Candidate trials use ``allowCutThrough=False``. Their exact trial topology is
validated by ``oneStringScoreTrial`` after construction. The actual accepted
operation uses the default ``allowCutThrough=True`` and is validated once before
returning to the ordinary OptCuts optimizer.

Native Grid-OptCuts must ALSO retain the authors' air/scaffold bijectivity
machinery. Local triangle-orientation tests do not prevent two distant UV regions
from overlapping. Official OptCuts normally suppresses the scaffold for its
random one-point initial cut; Grid mode uses a Grid-aware initial boundary, so we
disable that shortcut only for Grid mode.
"""
from __future__ import annotations
from pathlib import Path

TRI_MARKER = "ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE"
MAIN_MARKER = "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD"


def apply_trial_relax_patch(root: Path) -> bool:
    changed = False

    tri_path = root / "src" / "TriMesh.cpp"
    text = tri_path.read_text(encoding="utf-8")
    if TRI_MARKER not in text:
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
                // Candidate trials are checked by oneStringScoreTrial. The real
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
        tri_path.write_text(text, encoding="utf-8")
        print(f"Applied Grid-OptCuts deferred trial/accepted-cut validation: {tri_path}")
        changed = True

    # Preserve the original OptCuts global bijectivity scaffold in Grid mode.
    main_path = root / "src" / "main.cpp"
    main = main_path.read_text(encoding="utf-8")
    if MAIN_MARKER not in main:
        old = '''                        temp.onePointCut(F_component[componentI](0, 0));
                        rand1PInitCut = (n_components == 1);
                        break;
'''
        new = '''                        temp.onePointCut(F_component[componentI](0, 0));
                        // ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD
                        // Grid mode has a lattice-aware initial boundary. Keep
                        // OptCuts' air/scaffold barrier active so distant UV
                        // regions cannot overlap while seam vertices stay fixed.
                        if(oneStringMainGridEnabled()) {
                            rand1PInitCut = false;
                            std::cout << "[ONESTRING-GRID] global_bijectivity_scaffold=enabled" << std::endl;
                        }
                        else {
                            rand1PInitCut = (n_components == 1);
                        }
                        break;
'''
        if main.count(old) != 1:
            raise RuntimeError(f"Grid bijectivity main.cpp anchor count={main.count(old)}")
        main_path.write_text(main.replace(old, new, 1), encoding="utf-8")
        print(f"Enabled Grid-OptCuts global bijectivity scaffold: {main_path}")
        changed = True

    return changed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_trial_relax_patch(a.root.resolve())
