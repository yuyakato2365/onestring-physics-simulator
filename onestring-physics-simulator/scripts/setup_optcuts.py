#!/usr/bin/env python3
"""Clone and build the official OptCuts research code for the OneString bridge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


OFFICIAL_REPOSITORY = "https://github.com/liminchen/OptCuts.git"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def executable_candidates(root: Path) -> list[Path]:
    return [
        root / "build" / "OptCuts_bin",
        root / "build" / "OptCuts_bin.exe",
        root / "build" / "Release" / "OptCuts_bin.exe",
        root / "build" / "Release" / "OptCuts_bin",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "OptCuts",
        help="Destination for the official OptCuts checkout.",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(12, os.cpu_count() or 4)))
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Clone/check only; do not invoke CMake.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if shutil.which("git") is None:
        raise SystemExit("git is required")
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", OFFICIAL_REPOSITORY, str(root)])
    elif not (root / ".git").is_dir():
        raise SystemExit(f"{root} exists but is not a Git checkout")
    else:
        print(f"Using existing OptCuts checkout: {root}")
        print("No automatic git pull is performed; this script never overwrites local upstream edits.")

    if args.no_build:
        return 0
    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required to build OptCuts")

    build = root / "build"
    run(["cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"])
    run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--parallel",
            str(max(1, args.jobs)),
        ]
    )

    for candidate in executable_candidates(root):
        if candidate.is_file():
            print("\nOptCuts executable:")
            print(candidate)
            print("\nSet explicitly if desired:")
            if os.name == "nt":
                print(f'$env:ONESTRING_OPTCUTS_EXECUTABLE = "{candidate}"')
            else:
                print(f'export ONESTRING_OPTCUTS_EXECUTABLE="{candidate}"')
            return 0

    print("Build finished, but OptCuts_bin was not found in the expected locations.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
