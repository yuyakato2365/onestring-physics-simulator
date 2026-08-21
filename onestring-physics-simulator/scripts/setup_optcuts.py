#!/usr/bin/env python3
"""Clone, patch, and build OptCuts for the OneString bridge.

The local checkout receives:
1. compatibility fixes required by modern CMake/GLFW/Eigen; and
2. the explicit OneString native Grid-OptCuts V4 candidate-search patch.

The Grid patch is dormant unless ONESTRING_OPTCUTS_GRID_NATIVE=1, so the same
binary still executes the authors' ordinary OptCuts path for ``optcuts``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

from patch_optcuts_native_grid_v4 import apply_native_grid_patch

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
            "OptCuts DownloadProject.cmake has an unexpected layout; refusing to patch it automatically."
        )
    backup = path.with_suffix(path.suffix + ".onestring-backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"Applied local OptCuts nested CMake compatibility patch: {path}")
    return True


def patch_legacy_glfw_policy(root: Path) -> bool:
    path = root / "ext" / "libigl" / "external" / "glfw" / "CMakeLists.txt"
    if not path.is_file():
        print(f"GLFW compatibility patch skipped: {path} was not found")
        return False
    text = path.read_text(encoding="utf-8")
    old = "cmake_policy(SET CMP0042 OLD)"
    new = "cmake_policy(SET CMP0042 NEW) # OneString modern-CMake compatibility"
    if new in text:
        print("OptCuts GLFW CMP0042 compatibility patch already present.")
        return False
    if old not in text:
        print("OptCuts GLFW CMP0042 OLD policy was not found; no patch needed.")
        return False
    backup = path.with_suffix(path.suffix + ".onestring-backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Applied local OptCuts GLFW CMP0042 compatibility patch: {path}")
    return True


def patch_legacy_eigen_transpositions(root: Path) -> bool:
    """Fix the old Eigen Transpositions assignment for current Apple/Clang builds."""
    path = root / "ext" / "libigl" / "external" / "eigen" / "Eigen" / "src" / "Core" / "Transpositions.h"
    if not path.is_file():
        print(f"Eigen compatibility patch skipped: {path} was not found")
        return False
    text = path.read_text(encoding="utf-8")
    marker = "= trt; // OneString modern-Eigen compatibility"
    if marker in text:
        print("OptCuts Eigen Transpositions compatibility patch already present.")
        return False
    old = "= trt.derived();"
    if old not in text:
        print("OptCuts Eigen trt.derived() assignment was not found; no patch needed.")
        return False
    backup = path.with_suffix(path.suffix + ".onestring-backup")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(old, "= trt; // OneString modern-Eigen compatibility"), encoding="utf-8")
    print(f"Applied local OptCuts Eigen compatibility patch: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "OptCuts",
        help="Destination for the official OptCuts checkout.",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(12, os.cpu_count() or 4)))
    parser.add_argument("--no-build", action="store_true", help="Patch only; do not invoke CMake.")
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
        print("No automatic git pull is performed; local upstream edits are not overwritten.")

    patch_nested_cmake_policy(root)
    patch_legacy_glfw_policy(root)
    patch_legacy_eigen_transpositions(root)
    apply_native_grid_patch(root)

    if args.no_build:
        return 0
    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required to build OptCuts")

    build = root / "build"
    run([
        "cmake", "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_POLICY_VERSION_MINIMUM={CMAKE_POLICY_MINIMUM}",
    ])
    run([
        "cmake", "--build", str(build), "--config", "Release", "--parallel", str(max(1, args.jobs)),
    ])

    for candidate in executable_candidates(root):
        if candidate.is_file():
            print("\nOptCuts executable:")
            print(candidate)
            print("\nNative Grid-OptCuts V4 support is compiled in and dormant for official optcuts mode.")
            if os.name == "nt":
                print(f'$env:ONESTRING_OPTCUTS_EXECUTABLE = "{candidate}"')
            else:
                print(f'export ONESTRING_OPTCUTS_EXECUTABLE="{candidate}"')
            return 0

    print("Build finished, but OptCuts_bin was not found in the expected locations.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
