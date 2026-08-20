#!/usr/bin/env python3
"""Clone and build the official OptCuts research code for the OneString bridge.

The upstream OptCuts repository predates current CMake policy defaults.  Recent
CMake releases reject the old ``cmake_minimum_required`` version used by the
nested DownloadProject/TBB configure step.  This setup helper applies a tiny,
local-only compatibility patch to the ignored third_party checkout so the
original research code can still be configured without modifying its numerical
implementation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


OFFICIAL_REPOSITORY = "https://github.com/liminchen/OptCuts.git"
CMAKE_POLICY_MINIMUM = "3.5"


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


def patch_nested_cmake_policy(root: Path) -> bool:
    """Pass a modern compatibility floor into OptCuts' nested CMake configure.

    OptCuts vendors an old ``DownloadProject.cmake`` that launches a second
    CMake process for TBB and other dependencies.  Passing
    ``-DCMAKE_POLICY_VERSION_MINIMUM=3.5`` only to the top-level configure does
    not automatically reach that nested process, so inject the same cache value
    there.  The third_party checkout is gitignored; this never edits the upstream
    repository or OneString numerical code.
    """

    path = root / "cmake" / "DownloadProject.cmake"
    if not path.is_file():
        print(f"OptCuts compatibility patch skipped: {path} was not found")
        return False

    text = path.read_text(encoding="utf-8")
    marker = 'CMAKE_POLICY_VERSION_MINIMUM:STRING=3.5'
    if marker in text:
        print("OptCuts nested CMake policy compatibility patch already present.")
        return False

    needle = (
        'execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}"\n'
        '                        -D "CMAKE_MAKE_PROGRAM:FILE=${CMAKE_MAKE_PROGRAM}"\n'
        '                        .'
    )
    replacement = (
        'execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}"\n'
        '                        -D "CMAKE_MAKE_PROGRAM:FILE=${CMAKE_MAKE_PROGRAM}"\n'
        f'                        -D "CMAKE_POLICY_VERSION_MINIMUM:STRING={CMAKE_POLICY_MINIMUM}"\n'
        '                        .'
    )
    if needle not in text:
        raise SystemExit(
            "OptCuts DownloadProject.cmake has an unexpected layout; refusing to patch it automatically. "
            "Inspect cmake/DownloadProject.cmake around its nested execute_process(CMAKE_COMMAND ...) call."
        )

    backup = path.with_suffix(path.suffix + ".onestring-backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"Applied local OptCuts nested CMake compatibility patch: {path}")
    return True


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

    patch_nested_cmake_policy(root)

    if args.no_build:
        return 0
    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required to build OptCuts")

    build = root / "build"
    run(
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_POLICY_VERSION_MINIMUM={CMAKE_POLICY_MINIMUM}",
        ]
    )
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
