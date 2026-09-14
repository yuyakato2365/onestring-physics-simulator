"""External bridge to the official OptCuts SIGGRAPH Asia 2018 implementation.

This module intentionally does not reimplement OptCuts.  It exports the current
triangle surface to OBJ, invokes the authors' ``OptCuts_bin`` in headless mode,
and imports the resulting 3D/UV face correspondence.

The bridge is isolated from OneString's existing BFF/CEPS/Split numerics.  If
the external executable is unavailable or produces unsupported topology, the
caller gets an explicit error instead of a silent fallback.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

import numpy as np


class OptCutsError(RuntimeError):
    """Base error for the external OptCuts bridge."""


class OptCutsUnavailableError(OptCutsError):
    """Raised when the official executable cannot be found."""


class OptCutsOutputError(OptCutsError):
    """Raised when OptCuts completed without a usable UV OBJ."""


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _log(message: str) -> None:
    print(f"[{_timestamp()}] {message}", flush=True)


def _progress_snapshot() -> str:
    """Best-effort view into the source-modified OptCuts diagnostics.

    The C++ scale-aware build appends one CSV row per diagnostic evaluation.
    Reading the file from the parent process gives useful progress information
    without changing the optimizer or guessing a percentage from wall time.
    """
    active_raw = os.environ.get("ONESTRING_OPTCUTS_ACTIVE_PATH", "").strip()
    diag_raw = os.environ.get("ONESTRING_OPTCUTS_SCALE_DIAG_PATH", "").strip()
    active = bool(active_raw and Path(active_raw).is_file())
    if not diag_raw:
        return f"objective_active={str(active).lower()} diag=unavailable"
    path = Path(diag_raw)
    if not path.is_file():
        return f"objective_active={str(active).lower()} diag_rows=0"
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:
        return (
            f"objective_active={str(active).lower()} "
            f"diag_read_error={type(exc).__name__}"
        )
    if not rows:
        return f"objective_active={str(active).lower()} diag_rows=0"
    global_rows = sum(1 for row in rows if row.get("scope") == "global")
    local_rows = sum(1 for row in rows if row.get("scope") == "local")
    last = rows[-1]
    fields = [
        f"objective_active={str(active).lower()}",
        f"diag_rows={len(rows)}",
        f"global_rows={global_rows}",
        f"local_rows={local_rows}",
        f"last_scope={last.get('scope', '?')}",
        f"last_call={last.get('call', '?')}",
    ]
    scale_range = last.get("scale_range", "")
    violating = last.get("violating", "")
    if scale_range:
        fields.append(f"range={scale_range}")
    if violating:
        fields.append(f"violating={violating}")
    return " ".join(fields)


@dataclass(frozen=True)
class OptCutsConfig:
    executable: str | None = None
    lambda_init: float = 0.999
    method_type: int = 1
    distortion_bound: float = 4.1
    use_bijectivity: bool = True
    initial_cut_option: int = 0
    timeout_seconds: float = 1800.0


@dataclass
class OptCutsResult:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    uv_faces: np.ndarray
    metrics: dict[str, object] = field(default_factory=dict)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _infer_optcuts_root(executable: Path) -> Path:
    cur = executable.resolve().parent
    for candidate in (cur, *cur.parents):
        if candidate.name == "OptCuts":
            return candidate
    return _project_root() / "third_party" / "OptCuts"


def resolve_optcuts_executable(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("ONESTRING_OPTCUTS_BIN", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    stamp = _project_root() / ".onestring_optcuts_binary"
    if stamp.is_file():
        try:
            stamped = stamp.read_text(encoding="utf-8").strip()
            if stamped:
                candidates.append(Path(stamped).expanduser())
        except OSError:
            pass
    root = _project_root()
    candidates.extend([
        root / "third_party" / "OptCuts" / "build_onestring" / "OptCuts_onestring_runner",
        root / "third_party" / "OptCuts" / "build_onestring" / "OptCuts_bin",
        root / "third_party" / "OptCuts" / "build" / "OptCuts_bin",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise OptCutsUnavailableError(
        "OptCuts executable was not found. Build it with scripts/enable_optcuts_scale_sum.py "
        "or set ONESTRING_OPTCUTS_BIN."
    )


def _write_triangle_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in np.asarray(vertices, dtype=float):
            f.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for tri in np.asarray(faces, dtype=int):
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _read_obj_with_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    uv: list[list[float]] = []
    faces: list[list[int]] = []
    uv_faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("vt "):
                p = line.split()
                uv.append([float(p[1]), float(p[2])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) != 3:
                    raise OptCutsOutputError("OptCuts output contains a non-triangle face")
                fv: list[int] = []
                ft: list[int] = []
                for token in tokens:
                    parts = token.split("/")
                    fv.append(int(parts[0]) - 1)
                    if len(parts) < 2 or not parts[1]:
                        raise OptCutsOutputError("OptCuts output OBJ does not contain UV indices")
                    ft.append(int(parts[1]) - 1)
                faces.append(fv)
                uv_faces.append(ft)
    if not vertices or not faces or not uv:
        raise OptCutsOutputError(f"Incomplete OptCuts OBJ: {path}")
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
        np.asarray(uv, dtype=float),
        np.asarray(uv_faces, dtype=int),
    )


def _boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edge_count: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    for tri in np.asarray(faces, dtype=int):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edge_count[key] = edge_count.get(key, 0) + 1
            directed.append((int(a), int(b)))
    boundary = [(a, b) for a, b in directed if edge_count[(min(a, b), max(a, b))] == 1]
    outgoing: dict[int, list[int]] = {}
    for a, b in boundary:
        outgoing.setdefault(a, []).append(b)
    unused = set(boundary)
    loops: list[np.ndarray] = []
    while unused:
        start = next(iter(unused))
        a, b = start
        loop = [a]
        cur_a, cur_b = a, b
        while (cur_a, cur_b) in unused:
            unused.remove((cur_a, cur_b))
            loop.append(cur_b)
            nxt = [v for v in outgoing.get(cur_b, []) if (cur_b, v) in unused]
            if not nxt:
                break
            cur_a, cur_b = cur_b, nxt[0]
            if cur_b == loop[0]:
                break
        if len(loop) > 1 and loop[-1] == loop[0]:
            loop.pop()
        loops.append(np.asarray(loop, dtype=int))
    return loops


def _signed_area(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return 0.0
    q = np.roll(p, -1, axis=0)
    return 0.5 * float(np.sum(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]))


def _normalize_uv_area(
    xyz: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
) -> tuple[np.ndarray, float]:
    tri3 = xyz[faces]
    area3 = 0.5 * float(
        np.sum(np.linalg.norm(np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0]), axis=1))
    )
    tri2 = uv[uv_faces]
    signed2 = 0.5 * (
        (tri2[:, 1, 0] - tri2[:, 0, 0]) * (tri2[:, 2, 1] - tri2[:, 0, 1])
        - (tri2[:, 1, 1] - tri2[:, 0, 1]) * (tri2[:, 2, 0] - tri2[:, 0, 0])
    )
    area2 = float(np.sum(np.abs(signed2)))
    scale = float(np.sqrt(area3 / max(area2, 1.0e-30)))
    centered = np.asarray(uv, dtype=float) - np.mean(uv, axis=0)
    return centered * scale, scale


def _triangle_differential_metrics(
    xyz: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    uv_faces: np.ndarray,
) -> dict[str, object]:
    sig1: list[float] = []
    sig2: list[float] = []
    areas: list[float] = []
    flips: list[bool] = []
    degenerate: list[bool] = []
    sd: list[float] = []

    for f3, f2 in zip(np.asarray(faces, dtype=int), np.asarray(uv_faces, dtype=int)):
        p = np.asarray(xyz[f3], dtype=float)
        q = np.asarray(uv[f2], dtype=float)
        e1 = p[1] - p[0]
        e2 = p[2] - p[0]
        l1 = float(np.linalg.norm(e1))
        normal = np.cross(e1, e2)
        normal_len = float(np.linalg.norm(normal))
        if l1 <= 1e-14 or normal_len <= 1e-14:
            sig1.append(float("nan")); sig2.append(float("nan")); areas.append(0.0)
            flips.append(False); degenerate.append(True); sd.append(float("inf")); continue
        x = e1 / l1
        z = normal / normal_len
        y = np.cross(z, x)
        source = np.asarray(
            [[0.0, l1, float(np.dot(e2, x))], [0.0, 0.0, float(np.dot(e2, y))]], dtype=float
        )[:, 1:]
        target = np.column_stack([q[1] - q[0], q[2] - q[0]])
        det_source = float(np.linalg.det(source))
        signed_area2 = 0.5 * float(np.linalg.det(target))
        is_deg = abs(det_source) <= 1e-14 or abs(signed_area2) <= 1e-14
        if is_deg:
            sig1.append(float("nan")); sig2.append(float("nan")); areas.append(signed_area2)
            flips.append(signed_area2 < 0.0); degenerate.append(True); sd.append(float("inf")); continue
        jac = target @ np.linalg.inv(source)
        s = np.linalg.svd(jac, compute_uv=False)
        a, b = float(max(s)), float(min(s))
        sig1.append(a); sig2.append(b); areas.append(signed_area2)
        flips.append(signed_area2 < 0.0); degenerate.append(False)
        sd.append(a * a + b * b + 1.0 / (a * a) + 1.0 / (b * b))

    finite_sd = np.asarray([v for v in sd if np.isfinite(v)], dtype=float)
    return {
        "per_triangle_sigma1": sig1,
        "per_triangle_sigma2": sig2,
        "per_triangle_symmetric_dirichlet": sd,
        "symmetric_dirichlet_mean": float(np.mean(finite_sd)) if len(finite_sd) else float("inf"),
        "symmetric_dirichlet_max": float(np.max(finite_sd)) if len(finite_sd) else float("inf"),
        "uv_triangle_flip_count": int(np.count_nonzero(flips)),
        "uv_degenerate_triangle_count": int(np.count_nonzero(degenerate)),
        "uv_signed_area_min": float(np.min(areas)) if areas else 0.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_output_obj(root: Path, tag: str, started_at: float) -> Path:
    output_root = root / "output"
    if not output_root.exists():
        raise OptCutsOutputError(f"OptCuts did not create {output_root}")
    candidates = list(output_root.glob("**/finalResult_mesh.obj"))
    tagged = [p for p in candidates if tag in str(p.parent)]
    pool = tagged or [p for p in candidates if p.stat().st_mtime >= started_at - 2.0]
    if not pool:
        raise OptCutsOutputError("OptCuts completed but finalResult_mesh.obj was not found")
    pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pool[0]


def run_official_optcuts(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    config: OptCutsConfig | None = None,
) -> OptCutsResult:
    cfg = config or OptCutsConfig()
    if not (4.0 < float(cfg.distortion_bound) < float("inf")):
        raise ValueError("OptCuts distortion_bound must be > 4")
    if not (0.0 < float(cfg.lambda_init) < 1.0):
        raise ValueError("OptCuts lambda_init must satisfy 0 < lambda_init < 1")
    if int(cfg.method_type) not in {0, 1, 2, 3}:
        raise ValueError("OptCuts method_type must be 0, 1, 2, or 3")
    if int(cfg.initial_cut_option) not in {0, 1}:
        raise ValueError("OptCuts initial_cut_option must be 0 or 1")

    executable = resolve_optcuts_executable(cfg.executable)
    root = _infer_optcuts_root(executable)
    tag = f"onestring_{uuid.uuid4().hex[:12]}"
    started = time.time()

    temp_ctx = tempfile.TemporaryDirectory(prefix="onestring_optcuts_")
    try:
        temp_dir = Path(temp_ctx.name)
        input_obj = temp_dir / "surface.obj"
        stdout_path = temp_dir / "optcuts_stdout.log"
        stderr_path = temp_dir / "optcuts_stderr.log"
        _log(f"[OPTCUTS-PROGRESS] stage=1/4 name=prepare_input tag={tag}")
        _write_triangle_obj(input_obj, surface_vertices, surface_faces)
        command = [
            str(executable), "100", str(input_obj.resolve()),
            f"{float(cfg.lambda_init):.17g}", "1", str(int(cfg.method_type)),
            f"{float(cfg.distortion_bound):.17g}", "1" if cfg.use_bijectivity else "0",
            str(int(cfg.initial_cut_option)), tag,
        ]
        _log(
            "[OPTCUTS-RUN-START] "
            f"tag={tag} vertices={len(surface_vertices)} faces={len(surface_faces)} "
            f"timeout={float(cfg.timeout_seconds):g}s executable={executable}"
        )
        _log("[OPTCUTS-RUN-COMMAND] " + " ".join(command))
        _log(
            f"[OPTCUTS-PROGRESS] stage=2/4 name=optimize_uv_and_cuts tag={tag} "
            "progress=started"
        )

        process: subprocess.Popen[str] | None = None
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_file, \
                    stderr_path.open("w", encoding="utf-8") as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(root),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                )
                heartbeat_interval = 30.0
                next_heartbeat = started + heartbeat_interval
                while True:
                    returncode = process.poll()
                    now = time.time()
                    if returncode is not None:
                        break
                    elapsed_now = now - started
                    if elapsed_now >= float(cfg.timeout_seconds):
                        process.terminate()
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        _log(f"[OPTCUTS-RUN-TIMEOUT] tag={tag} elapsed={elapsed_now:.3f}s")
                        raise OptCutsError(
                            f"Official OptCuts timed out after {cfg.timeout_seconds:g} s"
                        )
                    if now >= next_heartbeat:
                        snapshot = _progress_snapshot()
                        _log(
                            f"[OPTCUTS-RUN-HEARTBEAT] tag={tag} stage=2/4 "
                            f"name=optimize_uv_and_cuts elapsed={elapsed_now:.1f}s "
                            f"pid={process.pid} status=running {snapshot}"
                        )
                        while next_heartbeat <= now:
                            next_heartbeat += heartbeat_interval
                    time.sleep(0.25)
        except OSError as exc:
            elapsed = time.time() - started
            _log(f"[OPTCUTS-RUN-EXEC-ERROR] tag={tag} elapsed={elapsed:.3f}s error={exc}")
            raise OptCutsUnavailableError(f"Failed to execute {executable}: {exc}") from exc

        if process is None:
            raise OptCutsUnavailableError("Failed to start OptCuts process")
        elapsed = time.time() - started
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
        _log(
            f"[OPTCUTS-RUN-END] tag={tag} elapsed={elapsed:.3f}s "
            f"returncode={process.returncode}"
        )
        if process.returncode != 0:
            tail = "\n".join((stdout_text + "\n" + stderr_text).splitlines()[-40:])
            raise OptCutsError(
                f"Official OptCuts exited with code {process.returncode}.\nLast output:\n{tail}"
            )

        _log(f"[OPTCUTS-PROGRESS] stage=3/4 name=load_and_validate_output tag={tag}")
        result_obj = _find_output_obj(root, tag, started)
        _log(f"[OPTCUTS-OUTPUT-FOUND] tag={tag} path={result_obj}")
        xyz, faces, uv, uv_faces = _read_obj_with_uv(result_obj)
        uv, uv_scale = _normalize_uv_area(xyz, faces, uv, uv_faces)
        loops = _boundary_loops(uv_faces)
        if len(loops) != 1:
            raise OptCutsOutputError(
                "The first OneString OptCuts bridge currently requires exactly one UV boundary "
                f"loop after cutting; got {len(loops)}."
            )
        if _signed_area(uv[loops[0]]) < 0.0:
            uv[:, 1] *= -1.0

        _log(f"[OPTCUTS-PROGRESS] stage=4/4 name=compute_quality_metrics tag={tag}")
        differential = _triangle_differential_metrics(xyz, faces, uv, uv_faces)
        metrics: dict[str, object] = {
            "uv_area_normalization_scale": uv_scale,
            "optcuts_elapsed_seconds": elapsed,
            "optcuts_executable": str(executable),
            "optcuts_executable_sha256": _sha256(executable),
            "optcuts_stdout_tail": "\n".join(stdout_text.splitlines()[-80:]),
            "optcuts_stderr_tail": "\n".join(stderr_text.splitlines()[-80:]),
            **differential,
        }
        _log(
            f"[OPTCUTS-RUN-IMPORTED] tag={tag} elapsed={time.time() - started:.3f}s "
            f"output_vertices={len(xyz)} output_faces={len(faces)}"
        )
        _log(f"[OPTCUTS-PROGRESS] stage=4/4 name=complete tag={tag} progress=done")
        return OptCutsResult(xyz, faces, uv, uv_faces, metrics)
    finally:
        temp_ctx.cleanup()
