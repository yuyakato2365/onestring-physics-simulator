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

CANONICAL_PATCH_VERSION = "4.4-hard-grid-locks"
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
_REQUIRED_OPTIMIZER_MARKERS = (
    "ONESTRING_GRID_NATIVE_V4_ZERO_LOCKED_SEARCH_DIR",
    "ONESTRING_GRID_NATIVE_V4_HARD_LOCK_STEP_FORWARD",
)


def _require_markers(path: Path, markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(
            f"Consolidated Grid-OptCuts V4 source verification failed for {path}: "
            f"missing markers={missing}"
        )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def _patch_optimizer_hard_grid_locks(root: Path) -> bool:
    """Make Grid locks a hard optimizer invariant, not only a solver hint.

    OptCuts' sparse pattern treats ``fixedVert`` specially, but ``stepForward``
    still applies the full search direction to every mesh vertex.  Native
    Grid-OptCuts requires a stronger invariant: once a seam/junction vertex has
    been accepted on the fabrication lattice, no subsequent numerical step may
    move it.  We therefore zero those search-direction entries immediately after
    every solve and also skip their coordinate update in ``stepForward``.

    The second guard is intentional redundancy: it protects the lattice even if
    a future solver/scaffold path accidentally produces a non-zero fixed entry.
    """
    path = root / "src" / "Optimizer.cpp"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if all(marker in text for marker in _REQUIRED_OPTIMIZER_MARKERS):
        print("Grid-OptCuts hard optimizer locks already present.")
        return False

    text = _replace_once(
        text,
        """        else {\n            linSysSolver->solve(minusG, searchDir);\n        }\n        if(!mute) { timer_step.stop(); }\n        \n        fractureInitiated = false;\n""",
        """        else {\n            linSysSolver->solve(minusG, searchDir);\n        }\n        if(!mute) { timer_step.stop(); }\n\n        // ONESTRING_GRID_NATIVE_V4_ZERO_LOCKED_SEARCH_DIR\n        // A fabrication-grid seam/junction is a hard positional constraint.\n        // Zeroing here also keeps the scaffold boundary motion consistent,\n        // because Scaffold::stepForward consumes this same searchDir.\n        for(const int vI : result.oneStringGridLockedVert) {\n            if(vI >= 0 && vI < result.V.rows() && vI * 2 + 1 < searchDir.size()) {\n                searchDir[vI * 2] = 0.0;\n                searchDir[vI * 2 + 1] = 0.0;\n            }\n        }\n        \n        fractureInitiated = false;\n""",
        "Optimizer hard-zero Grid lock search direction",
    )

    text = _replace_once(
        text,
        """        for(int vI = 0; vI < data.V.rows(); vI++) {\n            data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];\n            data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];\n        }\n""",
        """        for(int vI = 0; vI < data.V.rows(); vI++) {\n            // ONESTRING_GRID_NATIVE_V4_HARD_LOCK_STEP_FORWARD\n            // Never allow an accepted fabrication-grid seam/junction to drift.\n            if(data.oneStringGridLockedVert.find(vI) != data.oneStringGridLockedVert.end()) {\n                data.V.row(vI) = dataV0.row(vI);\n                continue;\n            }\n            data.V(vI, 0) = dataV0(vI, 0) + stepSize * searchDir[vI * 2];\n            data.V(vI, 1) = dataV0(vI, 1) + stepSize * searchDir[vI * 2 + 1];\n        }\n""",
        "Optimizer hard-skip Grid lock stepForward",
    )

    path.write_text(text, encoding="utf-8")
    print(f"Applied hard Grid seam/junction optimizer locks: {path}")
    return True


def apply_grid_optcuts_v4(root: Path) -> bool:
    root = root.expanduser().resolve()
    changed = False

    # Core V4 owns main.cpp, including the authoritative Optimizer scaffold bool.
    changed = bool(_apply_core_native_grid_patch(root)) or changed
    changed = bool(apply_native_grid_perf_patch(root)) or changed
    changed = bool(apply_native_grid_diagnostics(root)) or changed
    changed = bool(apply_trial_relax_patch(root)) or changed
    changed = bool(_patch_optimizer_hard_grid_locks(root)) or changed

    _require_markers(root / "src" / "TriMesh.cpp", _REQUIRED_TRIMESH_MARKERS)
    _require_markers(root / "src" / "main.cpp", _REQUIRED_MAIN_MARKERS)
    _require_markers(root / "src" / "Optimizer.cpp", _REQUIRED_OPTIMIZER_MARKERS)

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
