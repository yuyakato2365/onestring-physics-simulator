#!/usr/bin/env python3
"""Build the dedicated OneString-aware OptCuts binary.

This script applies the repository-tracked source patch to the local upstream
OptCuts checkout and builds into ``third_party/OptCuts/build_onestring``.
The Streamlit app never rewrites C++ source at runtime.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def check_apply(optcuts: Path, patch: Path, reverse: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "apply"]
    if reverse:
        cmd.append("--reverse")
    cmd.extend(["--check", str(patch)])
    return subprocess.run(
        cmd, cwd=str(optcuts), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optcuts-root",
        type=Path,
        default=None,
        help="upstream OptCuts checkout (default: third_party/OptCuts)",
    )
    args = parser.parse_args()

    root = project_root()
    optcuts = (args.optcuts_root or (root / "third_party" / "OptCuts")).expanduser().resolve()
    patch = root / "vendor" / "optcuts_onestring" / "0001-scale-aware-seam-objective.patch"
    cpp = optcuts / "src" / "TriMesh.cpp"
    if not cpp.is_file():
        raise SystemExit(
            f"OptCuts checkout not found at {optcuts}. Clone https://github.com/liminchen/OptCuts there first."
        )
    if not patch.is_file():
        raise SystemExit(f"tracked OneString patch missing: {patch}")

    # Apply exactly the tracked source change. If a previous experimental
    # runtime-patcher left its backup behind, restore that clean upstream source
    # before applying the tracked patch so old experiments cannot contaminate
    # the dedicated binary.
    forward = check_apply(optcuts, patch)
    reverse = check_apply(optcuts, patch, reverse=True) if forward.returncode != 0 else None
    if forward.returncode != 0 and (reverse is None or reverse.returncode != 0):
        legacy_backup = cpp.with_suffix(".cpp.onestring_original")
        if legacy_backup.is_file():
            print(f"[OPTCUTS-ONESTRING-SOURCE] restoring legacy backup {legacy_backup}")
            shutil.copy2(legacy_backup, cpp)
            forward = check_apply(optcuts, patch)
            reverse = check_apply(optcuts, patch, reverse=True) if forward.returncode != 0 else None

    if forward.returncode == 0:
        run(["git", "apply", str(patch)], cwd=optcuts)
        print("[OPTCUTS-ONESTRING-SOURCE] applied tracked source patch")
    elif reverse is not None and reverse.returncode == 0:
        print("[OPTCUTS-ONESTRING-SOURCE] tracked source patch already applied")
    else:
        sys.stderr.write(forward.stderr)
        raise SystemExit(
            "Local OptCuts source does not match the upstream source expected by the tracked OneString patch. "
            "Restore third_party/OptCuts/src/TriMesh.cpp from upstream and rerun this script."
        )

    build = optcuts / "build_onestring"
    build.mkdir(parents=True, exist_ok=True)
    run([
        "cmake", "-S", str(optcuts), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
    ])
    jobs = str(max(1, min(12, os.cpu_count() or 1)))
    run(["cmake", "--build", str(build), "--config", "Release", "-j", jobs])

    candidates = [
        build / "OptCuts_bin",
        build / "OptCuts_bin.exe",
        build / "Release" / "OptCuts_bin",
        build / "Release" / "OptCuts_bin.exe",
    ]
    binary = next((p for p in candidates if p.is_file()), None)
    if binary is None:
        raise SystemExit(f"OptCuts_bin was not produced under {build}")

    stamp = root / ".onestring_optcuts_binary"
    stamp.write_text(str(binary.resolve()) + "\n", encoding="utf-8")
    print(f"[OPTCUTS-ONESTRING-BUILD] binary={binary.resolve()}")
    print(f"[OPTCUTS-ONESTRING-BUILD] stamp={stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
