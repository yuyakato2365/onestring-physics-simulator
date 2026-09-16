"""Keep the authors' OptCuts binary separate from OneString Grid-OptCuts.

The repository intentionally maintains two executables:

* ``third_party/OptCuts_official``: authors' OptCuts algorithm, with build-only
  compatibility edits if required by modern macOS/CMake. Used by ``optcuts``.
* ``third_party/OptCuts``: OneString's experimental Grid-OptCuts fork. Used only
  by ``optcuts_grid``.

Historically both modes could resolve to ``third_party/OptCuts/build/OptCuts_bin``.
That made the ordinary ``optcuts`` mode execute the heavily modified Grid binary,
which can turn a previously short authors' OptCuts solve into a very long run.
This patch makes that mix-up impossible instead of silently accepting it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _grid_optcuts_root() -> Path:
    return (_project_root() / "third_party" / "OptCuts").resolve()


def _official_optcuts_root() -> Path:
    return (_project_root() / "third_party" / "OptCuts_official").resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def official_executable_candidates(requested: str | None = None) -> list[Path]:
    """Return only candidates that are valid for the ordinary ``optcuts`` mode."""
    candidates: list[Path] = []

    explicit_official = os.environ.get("ONESTRING_OPTCUTS_OFFICIAL_EXECUTABLE", "").strip()
    if explicit_official:
        candidates.append(Path(explicit_official).expanduser())

    # Keep support for an explicitly supplied external authors' binary, but never
    # accept the in-repository Grid-OptCuts build through the legacy field.
    if requested:
        p = Path(requested).expanduser()
        if not _is_under(p, _grid_optcuts_root()):
            candidates.append(p)

    root = _official_optcuts_root()
    candidates.extend(
        [
            root / "build" / "OptCuts_bin",
            root / "build" / "OptCuts_bin.exe",
            root / "build" / "Release" / "OptCuts_bin.exe",
            root / "build" / "Release" / "OptCuts_bin",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def resolve_official_optcuts_executable(requested: str | None = None) -> Path:
    candidates = official_executable_candidates(requested)
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.is_file():
            if _is_under(resolved, _grid_optcuts_root()):
                # This should already have been filtered, but keep the runtime
                # invariant explicit in case the candidate logic changes later.
                continue
            return resolved

    legacy = Path(requested).expanduser() if requested else None
    legacy_note = ""
    if legacy is not None and _is_under(legacy, _grid_optcuts_root()):
        legacy_note = (
            "\nThe configured executable points to third_party/OptCuts, which is the "
            "Grid-OptCuts build and is intentionally rejected for ordinary optcuts."
        )
    shown = "\n  - ".join(str(path) for path in candidates) if candidates else "(none)"
    from .optcuts_backend import OptCutsUnavailableError

    raise OptCutsUnavailableError(
        "A clean Official OptCuts executable was not found. "
        "Build it once with `python3 scripts/setup_optcuts_official.py`.\n"
        f"Tried:\n  - {shown}{legacy_note}"
    )


def install_official_optcuts_binary_separation() -> None:
    """Patch the backend resolver used by run_official_optcuts()."""
    from . import optcuts_backend as backend

    if getattr(backend, "_onestring_official_binary_separation_installed", False):
        return
    backend.resolve_optcuts_executable = resolve_official_optcuts_executable
    backend._onestring_official_binary_separation_installed = True


__all__ = [
    "official_executable_candidates",
    "resolve_official_optcuts_executable",
    "install_official_optcuts_binary_separation",
]
