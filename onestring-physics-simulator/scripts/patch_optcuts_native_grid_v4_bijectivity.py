#!/usr/bin/env python3
"""Keep OptCuts' global bijectivity scaffold enabled in native Grid mode.

Grid-OptCuts fixes fabrication seam/junction vertices on the lattice, but local
triangle-orientation checks alone do not prevent distant UV regions from
overlapping.  Native Grid mode must therefore keep OptCuts' air/scaffold
bijectivity machinery active.

Official OptCuts suppresses the scaffold for its random one-point genus-0
initial cut via ``rand1PInitCut``.  Grid-OptCuts uses a Grid-aware initial
boundary and must not take that shortcut.
"""
from __future__ import annotations

from pathlib import Path

SOURCE_MARKER = "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD"
RUNTIME_MARKER = '[ONESTRING-GRID] global_bijectivity_scaffold=enabled'


def apply_grid_bijectivity_patch(root: Path) -> bool:
    path = root / "src" / "main.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if SOURCE_MARKER in text and RUNTIME_MARKER in text:
        print("Native Grid-OptCuts global bijectivity scaffold patch already present.")
        return False

    old = """                        temp.onePointCut(F_component[componentI](0, 0));
                        rand1PInitCut = (n_components == 1);
                        break;
"""
    new = """                        temp.onePointCut(F_component[componentI](0, 0));
                        // ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD
                        // Official OptCuts skips the air scaffold for its random
                        // one-point initialization. Native Grid-OptCuts uses a
                        // fabrication-grid boundary and must keep the global
                        // bijectivity scaffold active.
                        rand1PInitCut = oneStringMainGridEnabled() ? false : (n_components == 1);
                        if(oneStringMainGridEnabled()) {
                            std::cout << "[ONESTRING-GRID] global_bijectivity_scaffold=enabled" << std::endl;
                        }
                        break;
"""

    if SOURCE_MARKER not in text:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"Grid bijectivity main.cpp anchor count={count}")
        text = text.replace(old, new, 1)
    elif RUNTIME_MARKER not in text:
        # Upgrade an earlier scaffold patch that changed rand1PInitCut but did
        # not emit the runtime marker required by the Python bridge.
        needle = """                        rand1PInitCut = oneStringMainGridEnabled() ? false : (n_components == 1);
                        break;
"""
        replacement = """                        rand1PInitCut = oneStringMainGridEnabled() ? false : (n_components == 1);
                        if(oneStringMainGridEnabled()) {
                            std::cout << "[ONESTRING-GRID] global_bijectivity_scaffold=enabled" << std::endl;
                        }
                        break;
"""
        count = text.count(needle)
        if count != 1:
            raise RuntimeError(f"Grid bijectivity runtime-marker upgrade anchor count={count}")
        text = text.replace(needle, replacement, 1)

    path.write_text(text, encoding="utf-8")
    print(f"Enabled native Grid-OptCuts global bijectivity scaffold: {path}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_grid_bijectivity_patch(args.root.resolve())
