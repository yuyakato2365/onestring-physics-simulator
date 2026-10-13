#!/usr/bin/env python3
"""Build a rectangle-constrained CLI from the UNMODIFIED official BFF library.

No copy/patch of the original bff-command-line or the user's local worktree.
Usage:
  python3 scripts/setup_bff_rectangle.py
  python3 scripts/setup_bff_rectangle.py --source /path/to/boundary-first-flattening
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]


def locate_source(explicit: str | None = None) -> Path:
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        Path(os.environ["ONESTRING_BFF_SOURCE"]).expanduser()
        if os.environ.get("ONESTRING_BFF_SOURCE") else None,
        PROJECT / "third_party" / "boundary-first-flattening",
        PROJECT.parents[1] / "onestring-physics-simulator"
        / "third_party" / "boundary-first-flattening",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate/"include/bff/project/Bff.h").is_file():
            return candidate.resolve()
    raise SystemExit(
        "Official BFF C++ source not found. Pass --source with the folder "
        "containing include/bff/project/Bff.h; a standalone prebuilt "
        "bff-command-line executable alone is insufficient."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Root of the official BFF git checkout")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--build-dir", type=Path,
                        default=PROJECT/"third_party/bff_rectangle_build")
    args = parser.parse_args()
    src = locate_source(args.source)
    build_dir = args.build_dir.expanduser().resolve()
    print("Official BFF C++ library:", src, flush=True)
    print("Independent rectangle adapter build:", build_dir, flush=True)
    # This CMake adapter calls add_subdirectory() on the official source,
    # without modifying ANY file in the BFF checkout.
    subprocess.run([
        "cmake", "-S", str(PROJECT/"tools/bff_rectangle"), "-B", str(build_dir),
        "-DONESTRING_BFF_SOURCE="+str(src),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ], check=True)
    subprocess.run([
        "cmake","--build",str(build_dir),
        "--target","bff-rect-command-line",
        "--parallel",str(max(1,args.jobs)),
    ], check=True)
    exe=build_dir/"bff-rect-command-line"
    if not exe.is_file():
        exe=build_dir/"Release/bff-rect-command-line.exe"
    if not exe.is_file():
        raise SystemExit("Compiled bff-rect-command-line executable was not produced")
    print("Rectangular BFF executable:", exe, flush=True)
    print('export ONESTRING_BFF_RECTANGLE_EXECUTABLE="'+str(exe)+'"',flush=True)


if __name__ == "__main__":
    main()
