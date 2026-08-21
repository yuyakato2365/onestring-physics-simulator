#!/usr/bin/env python3
"""Canonical OneString Grid-OptCuts V4 patch entrypoint.

This is the only patch entrypoint setup_optcuts.py calls. Grid bijectivity is
part of the core main.cpp patch itself; there is no separately-wired bijectivity
fragment anymore.
"""
from __future__ import annotations

from pathlib import Path

# IMPORTANT: keep the imported core patch under a private, unambiguous name.
# Do not alias apply_grid_optcuts_v4 back to apply_native_grid_patch: Python
# resolves globals at call time, which previously made this function call
# itself recursively until RecursionError.
from patch_optcuts_native_grid_v4_final import apply_native_grid_patch as _apply_core_native_grid_patch
from patch_optcuts_native_grid_v4_perf import apply_native_grid_perf_patch
from patch_optcuts_native_grid_v4_diagnostics import apply_native_grid_diagnostics
from patch_optcuts_native_grid_v4_trial_relax import apply_trial_relax_patch

CANONICAL_PATCH_VERSION = "4.3-consolidated-no-recursion"
NATIVE_RUNTIME_MARKER = "[ONESTRING-GRID] native_candidate_search enabled version=4"
GRID_BIJECTIVITY_RUNTIME_MARKER = "[ONESTRING-GRID] global_bijectivity_scaffold=enabled"

_REQUIRED_TRIMESH_MARKERS = (
    "ONESTRING_GRID_NATIVE_V4",
    "ONESTRING_GRID_NATIVE_V4_FAST_SEARCH",
    "ONESTRING_GRID_NATIVE_V4_ORIGINAL_LOCAL_PROPOSAL",
    "ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE",
)
_REQUIRED_MAIN_MARKERS = (
    "native_candidate_search enabled version=4",
    "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD",
    "global_bijectivity_scaffold=",
    "ONESTRING_GRID_BIJECTIVITY_SCAFFOLD_DISABLED",
)


def _require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(
            f"Consolidated Grid-OptCuts V4 source verification failed for {path}: "
            f"missing markers={missing}"
        )


def apply_grid_optcuts_v4(root: Path) -> bool:
    root = root.expanduser().resolve()
    changed = False

    # Core V4 owns main.cpp, including the authoritative Optimizer scaffold bool.
    changed = bool(_apply_core_native_grid_patch(root)) or changed
    changed = bool(apply_native_grid_perf_patch(root)) or changed
    changed = bool(apply_native_grid_diagnostics(root)) or changed
    changed = bool(apply_trial_relax_patch(root)) or changed

    _require_markers(root / "src" / "TriMesh.cpp", _REQUIRED_TRIMESH_MARKERS)
    _require_markers(root / "src" / "main.cpp", _REQUIRED_MAIN_MARKERS)

    print(
        "Verified consolidated Grid-OptCuts V4 source contract "
        f"({CANONICAL_PATCH_VERSION})."
    )
    return changed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_grid_optcuts_v4(args.root)
