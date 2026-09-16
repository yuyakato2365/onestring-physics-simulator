"""External bridge to the official OptCuts SIGGRAPH Asia 2018 implementation.

This module intentionally does not reimplement OptCuts.  It exports the current
triangle surface to OBJ, invokes the authors' ``OptCuts_bin`` in headless mode,
and imports the resulting 3D/UV face correspondence.

The bridge is isolated from OneString's existing BFF/CEPS/Split numerics.  If
the external executable is unavailable or produces unsupported topology, the
caller gets an explicit error instead of a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class OptCutsConfig:
    executable: str | None = None
    distortion_bound: float = 4.1
    lambda_init: float = 0.999
    method_type: int = 0
    use_bijectivity: bool = True
    initial_cut_option: int = 0
    timeout_seconds: float = 600.0
    keep_workdir: bool = False


@dataclass
class OptCutsResult:
    surface_vertices_3d: np.ndarray
    surface_faces: np.ndarray
    uv_vertices_2d: np.ndarray
    uv_faces: np.ndarray
    boundary_loops: list[list[int]]
    output_obj: str
    metrics: dict[str, object] = field(default_factory=dict)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_executables() -> list[Path]:
    root = _project_root()
    env = os.environ.get("ONESTRING_OPTCUTS_EXECUTABLE", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    third_party = root / "third_party" / "OptCuts"
    candidates.extend(
        [
            third_party / "build" / "OptCuts_bin",
            third_party / "build" / "OptCuts_bin.exe",
            third_party / "build" / "Release" / "OptCuts_bin.exe",
            third_party / "build" / "Release" / "OptCuts_bin",
        ]
    )
    return candidates


def resolve_optcuts_executable(requested: str | None = None) -> Path:
    candidates = [Path(requested).expanduser()] if requested else _candidate_executables()
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved.is_file():
            return resolved
    shown = "\n  - ".join(str(path) for path in candidates) if candidates else "(none)"
    raise OptCutsUnavailableError(
        "Official OptCuts executable was not found. Set ONESTRING_OPTCUTS_EXECUTABLE "
        "or build the authors' repository under third_party/OptCuts.\n"
        f"Tried:\n  - {shown}"
    )


def _infer_optcuts_root(executable: Path) -> Path:
    for candidate in [executable.parent, *executable.parents]:
        if (candidate / "CMakeLists.txt").is_file() and (candidate / "src").is_dir():
            return candidate
    if executable.parent.name.lower() == "release" and executable.parent.parent.name == "build":
        return executable.parent.parent.parent
    if executable.parent.name == "build":
        return executable.parent.parent
    return executable.parent


def _write_triangle_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("OptCuts input vertices must have shape (n, >=3)")
    if tris.ndim != 2 or tris.shape[1] < 3:
        raise ValueError("OptCuts input faces must have shape (m, >=3)")
    tris = tris[:, :3]
    if len(xyz) == 0 or len(tris) == 0:
        raise ValueError("OptCuts input mesh is empty")
    if int(np.min(tris)) < 0 or int(np.max(tris)) >= len(xyz):
        raise ValueError("OptCuts input face index is out of range")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# OneString -> official OptCuts bridge\n")
        for p in xyz[:, :3]:
            fh.write(f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        for face in tris:
            fh.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")


def _obj_index(raw: str, count: int) -> int:
    value = int(raw)
    if value > 0:
        return value - 1
    if value < 0:
        return count + value
    raise OptCutsOutputError("OBJ index 0 is invalid")


def _read_obj_with_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    uv_faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt" and len(parts) >= 3:
                texcoords.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f" and len(parts) >= 4:
                refs = parts[1:]
                # OptCuts works on triangles, but fan triangulation makes the parser robust.
                for j in range(1, len(refs) - 1):
                    tri_refs = [refs[0], refs[j], refs[j + 1]]
                    face: list[int] = []
                    uv_face: list[int] = []
                    for ref in tri_refs:
                        fields = ref.split("/")
                        face.append(_obj_index(fields[0], len(vertices)))
                        if len(fields) < 2 or fields[1] == "":
                            raise OptCutsOutputError(
                                f"{path.name} does not contain per-corner texture coordinates"
                            )
                        uv_face.append(_obj_index(fields[1], len(texcoords)))
                    faces.append(face)
                    uv_faces.append(uv_face)

    if not vertices or not faces or not texcoords:
        raise OptCutsOutputError(f"OptCuts output OBJ is incomplete: {path}")
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    uv = np.asarray(texcoords, dtype=float)
    uv_tris = np.asarray(uv_faces, dtype=int)
    if len(tris) != len(uv_tris):
        raise OptCutsOutputError("3D and UV face counts differ in OptCuts output")
    if int(np.max(tris)) >= len(xyz) or int(np.max(uv_tris)) >= len(uv):
        raise OptCutsOutputError("OptCuts output OBJ contains out-of-range indices")
    return xyz, tris, uv, uv_tris


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    tris = np.asarray(faces, dtype=int)
    edge_count: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    for tri in tris:
        for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
            edge_count[(min(a, b), max(a, b))] = edge_count.get((min(a, b), max(a, b)), 0) + 1
            directed.append((a, b))
    boundary = [(a, b) for a, b in directed if edge_count[(min(a, b), max(a, b))] == 1]
    if not boundary:
        return []

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    remaining = {tuple(sorted((a, b))) for a, b in boundary}
    loops: list[list[int]] = []
    while remaining:
        first_edge = next(iter(remaining))
        start = first_edge[0]
        current = start
        previous: int | None = None
        loop: list[int] = []
        for _ in range(len(boundary) + 2):
            loop.append(current)
            neighbors = [
                n for n in adjacency.get(current, [])
                if tuple(sorted((current, n))) in remaining and n != previous
            ]
            if not neighbors:
                # Allow the closing edge back to the start.
                neighbors = [
                    n for n in adjacency.get(current, [])
                    if tuple(sorted((current, n))) in remaining
                ]
            if not neighbors:
                break
            nxt = neighbors[0]
            remaining.discard(tuple(sorted((current, nxt))))
            previous, current = current, nxt
            if current == start:
                break
        if current != start or len(loop) < 3:
            raise OptCutsOutputError("OptCuts UV boundary is not a collection of simple loops")
        loops.append(loop)
    loops.sort(key=len, reverse=True)
    return loops


def _signed_area(poly: np.ndarray) -> float:
    pts = np.asarray(poly, dtype=float)
    return 0.5 * float(
        np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    )


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
            sig1.append(float("nan"))
            sig2.append(float("nan"))
            areas.append(0.0)
            flips.append(False)
            degenerate.append(True)
            sd.append(float("inf"))
            continue
        x = e1 / l1
        z = normal / normal_len
        y = np.cross(z, x)
        source = np.asarray(
            [[0.0, l1, float(np.dot(e2, x))], [0.0, 0.0, float(np.dot(e2, y))]],
            dtype=float,
        )[:, 1:]
        target = np.column_stack([q[1] - q[0], q[2] - q[0]])
        det_source = float(np.linalg.det(source))
        signed_area2 = 0.5 * float(np.linalg.det(target))
        is_deg = abs(det_source) <= 1e-14 or abs(signed_area2) <= 1e-14
        if is_deg:
            sig1.append(float("nan"))
            sig2.append(float("nan"))
            areas.append(signed_area2)
            flips.append(signed_area2 < 0.0)
            degenerate.append(True)
            sd.append(float("inf"))
            continue
        jac = target @ np.linalg.inv(source)
        s = np.linalg.svd(jac, compute_uv=False)
        a, b = float(max(s)), float(min(s))
        sig1.append(a)
        sig2.append(b)
        areas.append(signed_area2)
        flips.append(signed_area2 < 0.0)
        degenerate.append(False)
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
        _write_triangle_obj(input_obj, surface_vertices, surface_faces)
        command = [
            str(executable),
            "100",
            str(input_obj.resolve()),
            f"{float(cfg.lambda_init):.17g}",
            "1",
            str(int(cfg.method_type)),
            f"{float(cfg.distortion_bound):.17g}",
            "1" if cfg.use_bijectivity else "0",
            str(int(cfg.initial_cut_option)),
            tag,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=float(cfg.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OptCutsError(
                f"Official OptCuts timed out after {cfg.timeout_seconds:g} s"
            ) from exc
        except OSError as exc:
            raise OptCutsUnavailableError(f"Failed to execute {executable}: {exc}") from exc

        if completed.returncode != 0:
            tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-40:])
            raise OptCutsError(
                f"Official OptCuts exited with code {completed.returncode}.\n"
                f"Last output:\n{tail}"
            )

        result_obj = _find_output_obj(root, tag, started)
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

        differential = _triangle_differential_metrics(xyz, faces, uv, uv_faces)
        metrics: dict[str, object] = {
            "parameterization_backend_name": "official_optcuts_external",
            "parameterization_backend_version": "authors_SIGGRAPH_Asia_2018_code",
            "parameterization_method": "optcuts",
            "omega_parameterization_mode": "optcuts",
            "flattening_backend": "official_optcuts_external",
            "optcuts_executable": str(executable),
            "optcuts_executable_sha256": _sha256(executable),
            "optcuts_working_root": str(root),
            "optcuts_output_obj": str(result_obj),
            "optcuts_command_mode": 100,
            "optcuts_lambda_init": float(cfg.lambda_init),
            "optcuts_method_type": int(cfg.method_type),
            "optcuts_distortion_bound": float(cfg.distortion_bound),
            "optcuts_use_bijectivity": bool(cfg.use_bijectivity),
            "optcuts_initial_cut_option": int(cfg.initial_cut_option),
            "optcuts_runtime_seconds": float(time.time() - started),
            "optcuts_stdout_tail": "\n".join(completed.stdout.splitlines()[-40:]),
            "optcuts_stderr_tail": "\n".join(completed.stderr.splitlines()[-40:]),
            "optcuts_uv_area_normalization_scale": float(uv_scale),
            "optcuts_uv_boundary_loop_count": int(len(loops)),
            "optcuts_uv_boundary_vertex_count": int(len(loops[0])),
            "surface_vertex_count_after_cut_export": int(len(xyz)),
            "surface_triangle_count_after_cut_export": int(len(faces)),
            "uv_vertex_count_after_cut": int(len(uv)),
            **differential,
        }
        return OptCutsResult(
            surface_vertices_3d=xyz,
            surface_faces=faces,
            uv_vertices_2d=uv,
            uv_faces=uv_faces,
            boundary_loops=loops,
            output_obj=str(result_obj),
            metrics=metrics,
        )
    finally:
        if cfg.keep_workdir:
            # TemporaryDirectory cannot be transferred safely; keep_workdir is
            # recorded for API compatibility but outputs live in OptCuts/output.
            pass
        temp_ctx.cleanup()
