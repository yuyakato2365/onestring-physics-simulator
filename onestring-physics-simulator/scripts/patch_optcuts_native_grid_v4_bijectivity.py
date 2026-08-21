#!/usr/bin/env python3
"""Keep OptCuts' global bijectivity scaffold enabled in native Grid mode.

Grid-OptCuts fixes fabrication seam/junction vertices on the lattice, but local
triangle-orientation checks alone do not prevent two distant UV regions from
overlapping.  The authors' air/scaffold machinery is therefore still required.

Official OptCuts suppresses the scaffold for its random one-point initial cut by
setting ``rand1PInitCut``.  Native Grid-OptCuts uses a Grid-aware initial
boundary and must not take that shortcut, so this patch disables the shortcut
only while ONESTRING_OPTCUTS_GRID_NATIVE is active.
"""
from __future__ import annotations
from pathlib import Path

MARKER = "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD"


def apply_grid_bijectivity_patch(root: Path) -> bool:
    path = root / "src" / "main.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False

    old = """                        temp.onePointCut(F_component[componentI](0, 0));
                        rand1PInitCut = (n_components == 1);
                        break;
"""
    new = """                        temp.onePointCut(F_component[componentI](0, 0));
                        // ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD
                        // Official OptCuts skips the air scaffold for its random
                        // one-point initialization.  Native Grid-OptCuts instead
                        // installs a fabrication-grid boundary, and must keep the
                        // global bijectivity scaffold active to prevent distant
                        // UV regions from overlapping while seam vertices remain
                        // lattice-locked.
                        rand1PInitCut = oneStringMainGridEnabled() ? false : (n_components == 1);
                        break;
"""
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Grid bijectivity main.cpp anchor count={count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Enabled native Grid-OptCuts global bijectivity scaffold: {path}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path)
    a = p.parse_args(); apply_grid_bijectivity_patch(a.root.resolve())
