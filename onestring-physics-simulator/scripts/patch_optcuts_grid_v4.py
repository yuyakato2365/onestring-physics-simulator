#!/usr/bin/env python3
"""Canonical OneString Grid-OptCuts V4 patch entrypoint.

This is the *only* patch entrypoint setup_optcuts.py should call.

Historically V4 grew as several incremental patch scripts.  Keeping the setup
script aware of each fragment caused wiring omissions (for example, backend
validation could require a patch that setup forgot to apply).  This module owns
the complete V4 patch contract in one place: it applies every internal fragment
in the required order and then verifies the resulting OptCuts sources contain
all mandatory V4 invariants.

The fragment modules remain implementation details for now; callers must not
invoke them individually.
"""
from __future__ import annotations

from pathlib import Path

from patch_optcuts_native_grid_v4_final import apply_native_grid_patch
from patch_optcuts_native_grid_v4_perf import apply_native_grid_perf_patch
from patch_optcuts_native_grid_v4_diagnostics import apply_native_grid_diagnostics
from patch_optcuts_native_grid_v4_trial_relax import apply_trial_relax_patch
from patch_optcuts_native_grid_v4_bijectivity import (
    RUNTIME_MARKER as GRID_BIJECTIVITY_RUNTIME_MARKER,
    apply_grid_bijectivity_patch,
)

CANONICAL_PATCH_VERSION = "4.1-consolidated"
NATIVE_RUNTIME_MARKER = "[ONESTRING-GRID] native_candidate_search enabled version=4"

# Source markers are deliberately checked after *all* fragments run.  A setup
# cannot report success merely because the individual patch functions returned.
_REQUIRED_TRIMESH_MARKERS = (
    "ONESTRING_GRID_NATIVE_V4",
    "ONESTRING_GRID_NATIVE_V4_FAST_SEARCH",
    "ONESTRING_GRID_NATIVE_V4_ORIGINAL_LOCAL_PROPOSAL",
    "ONESTRING_GRID_NATIVE_V4_DEFER_TRIAL_VALIDATE",
)
_REQUIRED_MAIN_MARKERS = (
    "native_candidate_search enabled version=4",
    "ONESTRING_GRID_NATIVE_V4_BIJECTIVITY_SCAFFOLD",
    GRID_BIJECTIVITY_RUNTIME_MARKER,
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
    """Apply the complete Grid-OptCuts V4 contract and verify source state."""
    root = root.expanduser().resolve()
    changed = False

    # Ordering is part of the contract.  Base V4 creates the functions/anchors;
    # performance replaces candidate scoring; diagnostics and accepted-cut
    # validation operate on that final search path; bijectivity edits main.cpp.
    changed = bool(apply_native_grid_patch(root)) or changed
    changed = bool(apply_native_grid_perf_patch(root)) or changed
    changed = bool(apply_native_grid_diagnostics(root)) or changed
    changed = bool(apply_trial_relax_patch(root)) or changed
    changed = bool(apply_grid_bijectivity_patch(root)) or changed

    _require_markers(root / "src" / "TriMesh.cpp", _REQUIRED_TRIMESH_MARKERS)
    _require_markers(root / "src" / "main.cpp", _REQUIRED_MAIN_MARKERS)

    print(
        "Verified consolidated Grid-OptCuts V4 source contract "
        f"({CANONICAL_PATCH_VERSION})."
    )
    return changed


# Backward-compatible name for any external helper that imported the old
# generic function.  New code should call apply_grid_optcuts_v4 explicitly.
apply_native_grid_patch = apply_grid_optcuts_v4


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    apply_grid_optcuts_v4(args.root)
