#!/usr/bin/env python3
"""Build the dedicated OneString-aware OptCuts binary.

This script applies the repository-tracked source modification to the local
upstream OptCuts checkout and builds into ``third_party/OptCuts/build_onestring``.
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


def _apply_cmd(patch: Path, *, reverse: bool = False, check: bool = False) -> list[str]:
    # The upstream file contains whitespace-only blank lines around the namespace
    # and computeLocalLDec blocks.  The tracked modification is anchored on the
    # actual C++ statements, so ignore whitespace-only context differences while
    # still requiring every semantic anchor to match exactly.
    cmd = [
        "git", "apply", "--recount", "--ignore-space-change", "--ignore-whitespace"
    ]
    if reverse:
        cmd.append("--reverse")
    if check:
        cmd.append("--check")
    cmd.append(str(patch))
    return cmd


def check_apply(optcuts: Path, patch: Path, reverse: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _apply_cmd(patch, reverse=reverse, check=True),
        cwd=str(optcuts), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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
        raise SystemExit(f"tracked OneString source modification missing: {patch}")

    # If the earlier experimental runtime patcher touched the local checkout,
    # restore its clean backup first.  From this point on only the repository-
    # tracked source modification is allowed to define the dedicated binary.
    legacy_backup = cpp.with_suffix(".cpp.onestring_original")
    source_text = cpp.read_text(encoding="utf-8", errors="replace")
    if "ONESTRING_INTERNAL_SCALE_FACTOR_PATCH_V1" in source_text and legacy_backup.is_file():
        print(f"[OPTCUTS-ONESTRING-SOURCE] restoring legacy runtime-patch backup {legacy_backup}")
        shutil.copy2(legacy_backup, cpp)

    forward = check_apply(optcuts, patch)
    reverse = check_apply(optcuts, patch, reverse=True) if forward.returncode != 0 else None

    if forward.returncode == 0:
        run(_apply_cmd(patch), cwd=optcuts)
        print("[OPTCUTS-ONESTRING-SOURCE] applied tracked source modification")
    elif reverse is not None and reverse.returncode == 0:
        print("[OPTCUTS-ONESTRING-SOURCE] tracked source modification already applied")
    else:
        # Print enough diagnostics to distinguish an upstream-revision mismatch
        # from a malformed modification without asking the user to guess.
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(optcuts),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
            ).stdout.strip()
        except Exception:
            head = "unknown"
        sys.stderr.write(forward.stderr)
        raise SystemExit(
            "Tracked OneString OptCuts modification still does not apply. "
            f"Local OptCuts HEAD={head}. This is a build-script/source-modification mismatch, "
            "not a OneString parameter-setting issue."
        )

    # Hard verification: do not compile unless the actual C++ objective contains
    # the scale-factor term.  This prevents a route/setup log from being mistaken
    # for a successfully modified optimizer.
    source_text = cpp.read_text(encoding="utf-8", errors="replace")
    required = (
        "oneStringScalePenalty",
        "oneStringScaleReward",
        "objectiveDec += oneStringScaleReward(*this, candidate)",
        "curEwDec += oneStringScaleReward(*this, candidate)",
        "EwDec += oneStringScaleReward(*this, candidate)",
    )
    missing = [token for token in required if token not in source_text]
    if missing:
        raise SystemExit(
            "OptCuts source verification failed; scale-aware objective is not actually present: "
            + ", ".join(missing)
        )
    print("[OPTCUTS-ONESTRING-OBJECTIVE] verified scale-factor term in TriMesh::computeLocalLDec")

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
