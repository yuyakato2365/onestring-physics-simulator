"""Strict resolver for the isolated C++ Grid-aware OptCuts build.

This mode MUST NOT fall back to Official OptCuts, Grid-OptCuts V4, or the
previous October 23 Python-only cut reranking. A source marker and binary
timestamp protect against stale/misidentified builds.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

from .optcuts_backend import OptCutsUnavailableError

MARKER = "ONESTRING_NATIVE_GRID_SEAM_V3_SAFE_RANK"


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
            errors.append(f"{root}: lacks {MARKER} in TriMesh.cpp")
            continue
        main_source = root / "src" / "main.cpp"
        if not main_source.is_file() or MARKER not in main_source.read_text():
            errors.append(f"{root}: native C++ lambda diagnostic patch missing in main.cpp")
            continue
        for binary in (root / "build" / "OptCuts_bin",
                       root / "build" / "Release" / "OptCuts_bin"):
            if (binary.is_file() and binary.stat().st_mtime >= max(
                    source.stat().st_mtime, main_source.stat().st_mtime)):
                return binary.resolve()
        errors.append(f"{root}: patched executable missing/stale")
    raise OptCutsUnavailableError(
        "OptCuts Grid-Seam C++ executable unavailable. Build the isolated "
        "experimental version using python3 scripts/setup_optcuts_grid_seam.py. "
        "The ordinary official binary is deliberately not a fallback.\n"
        + "\n".join(errors[:12]))


def run_native_with_heartbeat(
    command, *, cwd: str, env: dict[str, str], timeout: float,
    heartbeat_seconds: float = 20.0,
) -> subprocess.CompletedProcess:
    """Run the independent native grid OptCuts fork with periodic progress.

    Maintain the previous run() contract (captured stdout/stderr and
    CompletedProcess). communicate() can safely be retried after a timeout;
    its internal buffers preserve all output. This avoids silent ten-minute
    waits without changing any numerical algorithm or killing valid work.
    """
    if not (timeout > 0.0 and heartbeat_seconds > 0.0):
        raise ValueError("GRID_SEAM_INVALID_TIMEOUT_OR_HEARTBEAT")
    start = time.monotonic()
    proc = subprocess.Popen(command, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    print("[GRID-CUT-NATIVE-RUN] started "
          f"pid={proc.pid} timeout={timeout:g}s executable={command[0]}",
          flush=True)
    while True:
        elapsed = time.monotonic() - start
        remaining = timeout - elapsed
        if remaining <= 0.0:
            proc.kill()
            stdout, stderr = proc.communicate()
            print("[GRID-CUT-NATIVE-RUN] timed_out "
                  f"pid={proc.pid} elapsed={elapsed:.1f}s "
                  "reason=C++_did_not_complete", flush=True)
            raise subprocess.TimeoutExpired(
                command, timeout, output=stdout, stderr=stderr)
        try:
            stdout, stderr = proc.communicate(
                timeout=min(heartbeat_seconds, remaining))
        except subprocess.TimeoutExpired:
            print("[GRID-CUT-NATIVE-RUN] still_running "
                  f"pid={proc.pid} elapsed={time.monotonic()-start:.1f}s "
                  "stage=external_C++_OptCuts", flush=True)
            continue
        duration = time.monotonic() - start
        print("[GRID-CUT-NATIVE-RUN] finished "
              f"pid={proc.pid} elapsed={duration:.1f}s "
              f"exit_code={proc.returncode}", flush=True)
        return subprocess.CompletedProcess(
            command, proc.returncode, stdout, stderr)
