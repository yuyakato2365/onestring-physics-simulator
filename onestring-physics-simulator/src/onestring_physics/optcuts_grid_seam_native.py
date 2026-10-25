"""Strict resolver for the isolated C++ Grid-aware OptCuts build.

This mode MUST NOT fall back to Official OptCuts, Grid-OptCuts V4, or the
previous October 23 Python-only cut reranking. A source marker and binary
timestamp protect against stale/misidentified builds.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

from .optcuts_backend import OptCutsUnavailableError

MARKER = "ONESTRING_NATIVE_GRID_SEAM_V1"


def candidate_roots(project: Path) -> list[Path]:
    project = project.resolve()
    base = project / "third_party" / "OptCuts_grid_seam_onestring"
    roots = [base]
    explicit = os.environ.get("ONESTRING_OPTCUTS_GRID_SEAM_ROOT", "").strip()
    if explicit:
        roots.insert(0, Path(explicit).expanduser())
    # Ignored build artifacts are not automatically shared by git worktrees.
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "worktree", "list", "--porcelain"],
            text=True, capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        for row in result.stdout.splitlines():
            if not row.startswith("worktree "):
                continue
            wt = Path(row[len("worktree "):].strip())
            for parent in (wt, wt / "onestring-physics-simulator"):
                roots.append(parent / "third_party" / "OptCuts_grid_seam_onestring")
    seen, unique = set(), []
    for root in roots:
        root = root.resolve()
        if str(root) not in seen:
            unique.append(root)
            seen.add(str(root))
    return unique


def resolve_grid_seam_binary(project: Path) -> Path:
    errors = []
    for root in candidate_roots(project):
        source = root / "src" / "TriMesh.cpp"
        if not source.is_file():
            errors.append(f"{root}: C++ source not built")
            continue
        try:
            text = source.read_text()
        except OSError as exc:
            errors.append(f"{source}: {exc}")
            continue
        if MARKER not in text:
            errors.append(f"{root}: lacks {MARKER} marker")
            continue
        for binary in (root / "build" / "OptCuts_bin",
                       root / "build" / "Release" / "OptCuts_bin"):
            if binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime:
                return binary.resolve()
        errors.append(f"{root}: patched executable missing/stale")
    raise OptCutsUnavailableError(
        "OptCuts Grid-Seam C++ executable unavailable. Build the isolated "
        "experimental version using python3 scripts/setup_optcuts_grid_seam.py. "
        "The ordinary official binary is deliberately not a fallback.\n"
        + "\n".join(errors[:12]))
