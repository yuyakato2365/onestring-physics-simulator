#!/usr/bin/env python3
"""Build an untouched authors' OptCuts executable for ordinary ``optcuts`` mode.

Algorithmic source files are not modified. Only build-system / third-party
compatibility edits needed by modern macOS/CMake/Clang are applied. The
Grid-OptCuts fork remains under ``third_party/OptCuts`` and is never used by
this script.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

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


def patch_nested_cmake_policy(root: Path) -> None:
    path = root / "cmake" / "DownloadProject.cmake"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "CMAKE_POLICY_VERSION_MINIMUM:STRING=3.5" in text:
        return
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
    if needle in text:
        path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def patch_legacy_glfw_policy(root: Path) -> None:
    path = root / "ext" / "libigl" / "external" / "glfw" / "CMakeLists.txt"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    old = "cmake_policy(SET CMP0042 OLD)"
    new = "cmake_policy(SET CMP0042 NEW) # OneString modern-CMake compatibility"
    if old in text and new not in text:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_legacy_eigen_transpositions(root: Path) -> None:
    """Apply only the minimal Eigen-vs-modern-Clang compatibility edits.

    The bundled Eigen revision assumes ``Transpose<TranspositionsBase<...>>``
    provides ``derived()``. Modern Clang instantiation exposes that this wrapper
    has no such member. Passing ``trt`` itself is the intended expression object
    and does not change OptCuts numerics.
    """
    path = root / "ext" / "libigl" / "external" / "eigen" / "Eigen" / "src" / "Core" / "Transpositions.h"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")

    replacements = (
        (
            "= trt.derived();",
            "= trt; // OneString modern-Eigen compatibility",
        ),
        (
            "Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt.derived())",
            "Product<OtherDerived, Transpose, AliasFreeProduct>(matrix.derived(), trt)",
        ),
    )

    changed = False
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            changed = True

    if changed:
        path.write_text(text, encoding="utf-8")


def verify_no_grid_patch_markers(root: Path, executable: Path) -> None:
    forbidden = (
        "ONESTRING_GRID_NATIVE_V4",
        "ONESTRING_GRID_NATIVE_V4_INITIAL_SEARCH_FAST",
        "boundary_constrained_reparameterization",
        "[ONESTRING-GRID]",
    )
    data = executable.read_bytes()
    present = [marker for marker in forbidden if marker.encode("utf-8") in data]
    if present:
        raise SystemExit(
            "Official OptCuts build unexpectedly contains OneString Grid-OptCuts markers: "
            + ", ".join(present)
        )
    if root.name == "OptCuts":
        raise SystemExit("Refusing to use third_party/OptCuts for the official build; use OptCuts_official")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "OptCuts_official",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(12, os.cpu_count() or 4)))
    parser.add_argument("--reclone", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    grid_root = (Path(__file__).resolve().parents[1] / "third_party" / "OptCuts").resolve()
    if root == grid_root:
        raise SystemExit("Official OptCuts must not be built in third_party/OptCuts (Grid-OptCuts tree).")

    if shutil.which("git") is None:
        raise SystemExit("git is required")
    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required")

    if args.reclone and root.exists():
        print(f"Removing existing official tree: {root}")
        shutil.rmtree(root)

    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", OFFICIAL_REPOSITORY, str(root)])
    elif not (root / ".git").is_dir():
        raise SystemExit(f"{root} exists but is not a Git checkout")

    # Ensure this remains an authors' tree. Do not apply any Grid-OptCuts patch.
    run(["git", "reset", "--hard", "HEAD"], cwd=root)
    run(["git", "clean", "-fd", "--exclude=build"], cwd=root)

    patch_nested_cmake_policy(root)
    patch_legacy_glfw_policy(root)
    patch_legacy_eigen_transpositions(root)

    build = root / "build"
    run([
        "cmake", "-S", str(root), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=14",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        f"-DCMAKE_POLICY_VERSION_MINIMUM={CMAKE_POLICY_MINIMUM}",
    ])
    run([
        "cmake", "--build", str(build), "--config", "Release", "--parallel", str(max(1, args.jobs)),
    ])

    executable = next((p for p in executable_candidates(root) if p.is_file()), None)
    if executable is None:
        raise SystemExit("Build finished, but OptCuts_bin was not found")
    verify_no_grid_patch_markers(root, executable)

    print("\nOfficial OptCuts executable ready:")
    print(executable)
    print("\nOrdinary `optcuts` will resolve this binary automatically.")
    print("Grid-OptCuts remains isolated under third_party/OptCuts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
