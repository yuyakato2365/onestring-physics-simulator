#!/usr/bin/env python3
"""Clone, patch, build, and runtime-verify canonical Grid-OptCuts V4."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from patch_optcuts_grid_v4 import (
    GRID_BIJECTIVITY_RUNTIME_MARKER,
    NATIVE_RUNTIME_MARKER,
    apply_grid_optcuts_v4,
)
from patch_optcuts_initial_search_perf import apply_initial_search_perf_patch

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


def _verify_binary_contains(executable: Path, markers: tuple[str, ...]) -> None:
    data = executable.read_bytes()
    missing = [m for m in markers if m.encode("utf-8") not in data]
    if missing:
        raise SystemExit(f"OptCuts binary is missing required Grid-OptCuts markers: {missing}")
    print("Verified compiled Grid-OptCuts marker strings.")


def _write_tetra_obj(path: Path) -> None:
    path.write_text(
        "v 1 1 1\n"
        "v -1 -1 1\n"
        "v -1 1 -1\n"
        "v 1 -1 -1\n"
        "f 1 3 2\n"
        "f 1 2 4\n"
        "f 1 4 3\n"
        "f 2 3 4\n",
        encoding="utf-8",
    )


def _write_open_disk_obj(path: Path) -> None:
    # Two-triangle open disk: this deliberately bypasses the closed-surface
    # onePointCut path that previously (incorrectly) owned the scaffold marker.
    path.write_text(
        "v -1 -1 0\n"
        "v 1 -1 0\n"
        "v 1 1 0\n"
        "v -1 1 0\n"
        "f 1 2 3\n"
        "f 1 3 4\n",
        encoding="utf-8",
    )


def _run_runtime_case(executable: Path, root: Path, obj: Path, tag: str) -> None:
    env = os.environ.copy()
    env.update({
        "ONESTRING_OPTCUTS_GRID_NATIVE": "1",
        "ONESTRING_OPTCUTS_GRID_H": "0.25",
        "ONESTRING_OPTCUTS_GRID_ANGLE_RAD": "0",
        "ONESTRING_OPTCUTS_GRID_PHASE_U": "0",
        "ONESTRING_OPTCUTS_GRID_PHASE_V": "0",
        "ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS": "2",
    })
    command = [
        str(executable), "100", str(obj), "0.999", "1", "1", "100", "1", "0", tag,
    ]
    print(f"+ runtime smoke [{tag}]:", " ".join(command))
    combined = ""
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=45.0,
            check=False,
        )
        combined = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0:
            raise SystemExit(
                f"Grid-OptCuts runtime smoke [{tag}] failed:\n" +
                "\n".join(combined.splitlines()[-80:])
            )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        combined = stdout + "\n" + stderr

    missing = [
        marker
        for marker in (NATIVE_RUNTIME_MARKER, GRID_BIJECTIVITY_RUNTIME_MARKER)
        if marker not in combined
    ]
    if missing:
        marker_lines = [line for line in combined.splitlines() if "[ONESTRING-GRID]" in line]
        raise SystemExit(
            f"Grid-OptCuts runtime contract [{tag}] failed: "
            f"missing={missing}; markers_seen={marker_lines[-12:]}"
        )
    print(f"Verified Grid-OptCuts runtime contract [{tag}].")


def _runtime_smoke_test(executable: Path, root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="onestring_grid_optcuts_setup_smoke_") as tmp:
        tmp_path = Path(tmp)
        tetra = tmp_path / "tetra.obj"
        open_disk = tmp_path / "open_disk.obj"
        _write_tetra_obj(tetra)
        _write_open_disk_obj(open_disk)
        _run_runtime_case(executable, root, tetra, "closed_tetra")
        _run_runtime_case(executable, root, open_disk, "open_disk")


def patch_nested_cmake_policy(root: Path) -> None:
    path = root / "cmake" / "DownloadProject.cmake"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if 'CMAKE_POLICY_VERSION_MINIMUM:STRING=3.5' in text:
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
    path = root / "ext" / "libigl" / "external" / "eigen" / "Eigen" / "src" / "Core" / "Transpositions.h"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    old = "= trt.derived();"
    new = "= trt; // OneString modern-Eigen compatibility"
    if old in text and new not in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "OptCuts",
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(12, os.cpu_count() or 4)))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    if shutil.which("git") is None:
        raise SystemExit("git is required")
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", OFFICIAL_REPOSITORY, str(root)])
    elif not (root / ".git").is_dir():
        raise SystemExit(f"{root} exists but is not a Git checkout")

    patch_nested_cmake_policy(root)
    patch_legacy_glfw_policy(root)
    patch_legacy_eigen_transpositions(root)
    apply_grid_optcuts_v4(root)
    apply_initial_search_perf_patch(root)

    if args.no_build:
        return 0
    if shutil.which("cmake") is None:
        raise SystemExit("cmake is required")

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

    _verify_binary_contains(executable, (NATIVE_RUNTIME_MARKER, GRID_BIJECTIVITY_RUNTIME_MARKER))
    _runtime_smoke_test(executable, root)

    print("\nOptCuts executable:")
    print(executable)
    print("\nNative Grid-OptCuts V4 consolidated setup is compiled and runtime-verified on closed and open inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
