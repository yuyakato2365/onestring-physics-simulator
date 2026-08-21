#!/usr/bin/env python3
"""Add concise diagnostics to native Grid-OptCuts exact trial scoring."""
from __future__ import annotations
from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_TRIAL_DIAGNOSTICS"


def apply_native_grid_diagnostics(root: Path) -> bool:
    path = root / "src" / "TriMesh.cpp"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    score_start = text.find("double oneStringScoreTrial(")
    score_end = text.find("bool oneStringTryInteriorGridCut(", score_start)
    if score_start < 0 or score_end < 0:
        raise RuntimeError("Grid trial score helper not found")
    block = text[score_start:score_end]
    replacements = [
        ("if(!oneStringLockedPreserved(mesh, trial)) return -DBL_MAX;",
         'if(!oneStringLockedPreserved(mesh, trial)) { std::cout << "[ONESTRING-GRID-TRIAL-REJECT] locked_not_preserved" << std::endl; return -DBL_MAX; }'),
        ("if(!oneStringAllLockedOnGrid(trial)) return -DBL_MAX;",
         'if(!oneStringAllLockedOnGrid(trial)) { std::cout << "[ONESTRING-GRID-TRIAL-REJECT] locked_off_grid" << std::endl; return -DBL_MAX; }'),
        ("if(!oneStringAllCohesiveSeamSidesGridAligned(trial)) return -DBL_MAX;",
         'if(!oneStringAllCohesiveSeamSidesGridAligned(trial)) { std::cout << "[ONESTRING-GRID-TRIAL-REJECT] seam_off_grid" << std::endl; return -DBL_MAX; }'),
        ("if(!oneStringRelaxGridTrialUV(trial)) return -DBL_MAX;",
         'if(!oneStringRelaxGridTrialUV(trial)) { std::cout << "[ONESTRING-GRID-TRIAL-REJECT] harmonic_failed" << std::endl; return -DBL_MAX; }'),
        ("if(!trial.checkInversion(true)) return -DBL_MAX;",
         'if(!trial.checkInversion(true)) { std::cout << "[ONESTRING-GRID-TRIAL-REJECT] inversion_after_relax" << std::endl; return -DBL_MAX; }'),
    ]
    for old, new in replacements:
        block = block.replace(old, new)
    block = block.replace(
        "const double sdDec = before - after;",
        'const double sdDec = before - after;\n    std::cout << "[ONESTRING-GRID-TRIAL] before=" << before << " after=" << after << " sdDec=" << sdDec << " seam=" << seamIncrease << std::endl; // ONESTRING_GRID_NATIVE_V4_TRIAL_DIAGNOSTICS'
    )
    text = text[:score_start] + block + text[score_end:]

    # Report topology-operation exceptions rather than silently converting every
    # failure into -DBL_MAX.
    text = text.replace(
        "catch(...) {\n        score = -DBL_MAX;\n        return false;\n    }",
        'catch(const std::exception& e) {\n        std::cout << "[ONESTRING-GRID-TRIAL-REJECT] cut_exception=" << e.what() << std::endl;\n        score = -DBL_MAX;\n        return false;\n    }\n    catch(...) {\n        std::cout << "[ONESTRING-GRID-TRIAL-REJECT] cut_exception=unknown" << std::endl;\n        score = -DBL_MAX;\n        return false;\n    }',
        2,
    )
    path.write_text(text, encoding="utf-8")
    print(f"Applied Grid-OptCuts trial diagnostics: {path}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_native_grid_diagnostics(a.root.resolve())
