"""Strict adapter and diagnostics for the official Boundary First Flattening CLI.

The adapter deliberately contains no local parameterization fallback.  A caller
requesting the paper-reference mode either receives output from the official BFF
command-line program or a :class:`ReferenceBFFUnavailableError`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Literal

import numpy as np


OFFICIAL_BFF_REPOSITORY = "https://github.com/GeometryCollective/boundary-first-flattening"
OFFICIAL_BFF_CLI_COMMIT = "UNSPECIFIED_BY_PREBUILT_BINARY"


class ReferenceBFFError(RuntimeError):
    """Base class for strict reference-mode failures."""


class ReferenceBFFUnavailableError(ReferenceBFFError):
    """Raised when the official BFF backend cannot be executed faithfully."""


class ReferenceMeshValidationError(ReferenceBFFError):
    """Raised when an input mesh does not meet reference-mode preconditions."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


class ReferenceInverseMapError(ReferenceBFFError):
    """Raised instead of projecting a point that has no containing UV triangle."""

    def __init__(self, reason: str, uv_point: np.ndarray, vertex_id: int | None = None):
        self.reason = str(reason)
        self.uv_point = np.asarray(uv_point, dtype=float).copy()
        self.vertex_id = vertex_id
        suffix = f" at M2D vertex {vertex_id}" if vertex_id is not None else ""
        super().__init__(f"Reference inverse map failed ({self.reason}){suffix}: uv={self.uv_point.tolist()}")


@dataclass(frozen=True)
class BFFRunResult:
    uv_vertices: np.ndarray
    metrics: dict[str, Any]


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _boundary_loops_from_edges(edges: list[tuple[int, int]]) -> tuple[list[list[int]], int]:
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    invalid_degree_count = sum(1 for values in adjacency.values() if len(values) != 2)
    unseen = set(adjacency)
    loops: list[list[int]] = []
    while unseen:
        start = min(unseen)
        loop = [start]
        previous = -1
        current = start
        for _ in range(len(adjacency) + 1):
            neighbors = adjacency.get(current, [])
            if len(neighbors) != 2:
                break
            nxt = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if nxt == start:
                break
            if nxt in loop:
                break
            loop.append(nxt)
            previous, current = current, nxt
        for vertex in loop:
            unseen.discard(vertex)
        loops.append(loop)
    return loops, int(invalid_degree_count)


def validate_reference_mesh(vertices: np.ndarray, faces: np.ndarray, area_epsilon: float = 1e-14) -> dict[str, Any]:
    """Validate a consistently oriented manifold triangle disk without repairing it."""

    pts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    diagnostics: dict[str, Any] = {
        "vertex_count": int(len(pts)) if pts.ndim >= 1 else 0,
        "triangle_count": int(len(tris)) if tris.ndim >= 1 else 0,
        "finite_vertex_values": bool(np.all(np.isfinite(pts))) if pts.size else True,
        "non_manifold_edge_count": 0,
        "degenerate_triangle_count": 0,
        "inconsistent_winding_edge_count": 0,
        "boundary_loop_count": 0,
        "boundary_vertex_degree_error_count": 0,
        "connected_component_count": 0,
        "euler_characteristic": 0,
        "disk_topology": False,
        "orientable_consistent_winding": False,
    }
    errors: list[str] = []
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        errors.append("vertices must have shape (n, 3) with n >= 3")
    if tris.ndim != 2 or tris.shape[1] != 3 or len(tris) < 1:
        errors.append("faces must have shape (m, 3) with m >= 1")
    if errors:
        diagnostics["errors"] = errors
        raise ReferenceMeshValidationError("Invalid reference BFF input: " + "; ".join(errors), diagnostics)
    if not diagnostics["finite_vertex_values"]:
        errors.append("NaN / Inf vertex coordinates")
    if int(np.min(tris)) < 0 or int(np.max(tris)) >= len(pts):
        errors.append("face index out of range")
        diagnostics["errors"] = errors
        raise ReferenceMeshValidationError("Invalid reference BFF input: " + "; ".join(errors), diagnostics)

    doubled_area = np.linalg.norm(np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]]), axis=1)
    diagnostics["degenerate_triangle_count"] = int(np.count_nonzero(doubled_area <= 2.0 * float(area_epsilon)))
    if diagnostics["degenerate_triangle_count"]:
        errors.append(f"degenerate triangle count={diagnostics['degenerate_triangle_count']}")

    incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    face_adjacency: list[set[int]] = [set() for _ in range(len(tris))]
    for face_id, (a, b, c) in enumerate(tris):
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            incidence.setdefault(_edge_key(u, v), []).append((int(face_id), 1 if u < v else -1))
    diagnostics["edge_count"] = int(len(incidence))
    diagnostics["non_manifold_edge_count"] = int(sum(len(items) > 2 for items in incidence.values()))
    diagnostics["inconsistent_winding_edge_count"] = int(
        sum(len(items) == 2 and items[0][1] == items[1][1] for items in incidence.values())
    )
    boundary_edges = [edge for edge, items in incidence.items() if len(items) == 1]
    loops, degree_errors = _boundary_loops_from_edges(boundary_edges)
    diagnostics["boundary_loop_count"] = int(len(loops))
    diagnostics["boundary_vertex_degree_error_count"] = int(degree_errors)
    diagnostics["boundary_vertex_count"] = int(sum(len(loop) for loop in loops))
    diagnostics["boundary_loops"] = loops
    for items in incidence.values():
        if len(items) == 2:
            a, b = items[0][0], items[1][0]
            face_adjacency[a].add(b)
            face_adjacency[b].add(a)
    unseen = set(range(len(tris)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for neighbor in face_adjacency[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    diagnostics["connected_component_count"] = int(components)
    used_vertices = np.unique(tris)
    chi = int(len(used_vertices) - len(incidence) + len(tris))
    diagnostics["euler_characteristic"] = chi
    diagnostics["disk_topology"] = bool(components == 1 and len(loops) == 1 and degree_errors == 0 and chi == 1)
    diagnostics["orientable_consistent_winding"] = bool(diagnostics["inconsistent_winding_edge_count"] == 0)

    if diagnostics["non_manifold_edge_count"]:
        errors.append(f"non-manifold edge count={diagnostics['non_manifold_edge_count']}")
    if diagnostics["inconsistent_winding_edge_count"]:
        errors.append(f"inconsistent winding edge count={diagnostics['inconsistent_winding_edge_count']}")
    if components != 1:
        errors.append(f"connected component count={components}")
    if len(loops) != 1 or degree_errors:
        errors.append(f"boundary count={len(loops)}, boundary degree errors={degree_errors}")
    if chi != 1:
        errors.append(f"topology is not a disk (Euler characteristic={chi})")
    if errors:
        diagnostics["errors"] = errors
        raise ReferenceMeshValidationError("Invalid reference BFF input: " + "; ".join(errors), diagnostics)
    diagnostics["errors"] = []
    return diagnostics


def _candidate_executables(explicit: str | os.PathLike[str] | None) -> list[Path]:
    values: list[Path] = []
    if explicit:
        values.append(Path(explicit).expanduser())
    env_value = os.environ.get("ONESTRING_BFF_EXECUTABLE")
    if env_value:
        values.append(Path(env_value).expanduser())
    root = Path(__file__).resolve().parents[2]
    values.extend(
        [
            root / "third_party" / "boundary-first-flattening" / "binaries" / "windows-v1.6" / "bff-command-line.exe",
            root / "third_party" / "boundary-first-flattening" / "build" / "bff-command-line.exe",
            root / "third_party" / "boundary-first-flattening" / "build" / "bff-command-line",
        ]
    )
    for name in ("bff-command-line.exe", "bff-command-line"):
        found = shutil.which(name)
        if found:
            values.append(Path(found))
    unique: list[Path] = []
    seen: set[str] = set()
    for value in values:
        key = os.path.normcase(str(value.resolve(strict=False)))
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def find_official_bff_executable(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidates = _candidate_executables(explicit)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  - ".join(str(path) for path in candidates) or "(no candidates)"
    raise ReferenceBFFUnavailableError(
        "Official Boundary First Flattening CLI is unavailable. No substitute was used.\n"
        "Set ONESTRING_BFF_EXECUTABLE or PipelineParameters.bff_executable to the official bff-command-line binary.\n"
        f"Tried:\n  - {tried}"
    )


def _write_triangle_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# OneString strict reference BFF input\n")
        for x, y, z in np.asarray(vertices, dtype=float):
            stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in np.asarray(faces, dtype=int):
            stream.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def _read_write_only_uv_obj(path: Path, expected_vertex_count: int, expected_faces: np.ndarray) -> np.ndarray:
    values: list[tuple[float, float]] = []
    output_faces: list[list[int]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                if line.startswith("v "):
                    fields = line.split()
                    if len(fields) < 3:
                        raise ValueError(f"malformed vertex record: {line.rstrip()}")
                    values.append((float(fields[1]), float(fields[2])))
                elif line.startswith("f "):
                    fields = line.split()[1:]
                    output_faces.append([int(field.split("/")[0]) - 1 for field in fields])
    except Exception as exc:
        raise ReferenceBFFUnavailableError(f"Could not parse official BFF output {path}: {exc}") from exc
    uv_output_order = np.asarray(values, dtype=float)
    if uv_output_order.shape != (int(expected_vertex_count), 2):
        raise ReferenceBFFUnavailableError(
            "Official BFF output did not preserve one UV vertex per input vertex: "
            f"expected {(int(expected_vertex_count), 2)}, got {uv_output_order.shape}. No fallback was attempted."
        )
    expected = np.asarray(expected_faces, dtype=int)
    output = np.asarray(output_faces, dtype=int)
    if output.shape != expected.shape:
        raise ReferenceBFFUnavailableError(
            f"Official BFF output face list shape changed: expected {expected.shape}, got {output.shape}."
        )
    # The official CLI preserves face order but may renumber vertices.  Recover
    # the input vertex order from corresponding face corners and reject any
    # inconsistent mapping instead of guessing by position.
    input_to_output = np.full(int(expected_vertex_count), -1, dtype=int)
    output_to_input = np.full(int(expected_vertex_count), -1, dtype=int)
    for input_face, output_face in zip(expected, output):
        for input_id, output_id in zip(input_face, output_face):
            input_id = int(input_id)
            output_id = int(output_id)
            if not (0 <= output_id < expected_vertex_count):
                raise ReferenceBFFUnavailableError(f"Official BFF output vertex index {output_id} is out of range.")
            if input_to_output[input_id] not in {-1, output_id} or output_to_input[output_id] not in {-1, input_id}:
                raise ReferenceBFFUnavailableError("Official BFF output vertex renumbering is not one-to-one across face corners.")
            input_to_output[input_id] = output_id
            output_to_input[output_id] = input_id
    if np.any(input_to_output < 0):
        missing = np.flatnonzero(input_to_output < 0).astype(int).tolist()
        raise ReferenceBFFUnavailableError(f"Official BFF output did not reference input vertices {missing[:20]}.")
    uv = uv_output_order[input_to_output]
    if not np.all(np.isfinite(uv)):
        raise ReferenceBFFUnavailableError("Official BFF output contains NaN / Inf UV coordinates.")
    return uv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover_git_commit(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        git_dir = parent / ".git"
        if not git_dir.exists():
            continue
        if git_dir.is_file():
            text = git_dir.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                candidate = Path(text.split(":", 1)[1].strip())
                git_dir = candidate if candidate.is_absolute() else (parent / candidate).resolve()
        head_path = git_dir / "HEAD"
        if not head_path.is_file():
            continue
        head = head_path.read_text(encoding="ascii", errors="replace").strip()
        if not head.startswith("ref:"):
            return head or OFFICIAL_BFF_CLI_COMMIT
        reference = head.split(":", 1)[1].strip()
        loose_ref = git_dir / reference
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="ascii", errors="replace").strip() or OFFICIAL_BFF_CLI_COMMIT
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii", errors="replace").splitlines():
                if line and not line.startswith(("#", "^")):
                    sha, name = line.split(" ", 1)
                    if name.strip() == reference:
                        return sha.strip()
        return OFFICIAL_BFF_CLI_COMMIT
    return OFFICIAL_BFF_CLI_COMMIT


def run_official_bff(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    executable: str | os.PathLike[str] | None = None,
    boundary_policy: Literal["automatic_reference", "boundary_scale_zero", "target_disk"] = "automatic_reference",
    timeout_seconds: float = 120.0,
) -> BFFRunResult:
    """Run the official BFF CLI and return its UV vertices in input order."""

    topology = validate_reference_mesh(vertices, faces)
    exe = find_official_bff_executable(executable)
    if boundary_policy not in {"automatic_reference", "boundary_scale_zero", "target_disk"}:
        raise ReferenceBFFUnavailableError(
            f"Boundary policy {boundary_policy!r} is not exposed faithfully by the official CLI. No substitute was used."
        )
    with tempfile.TemporaryDirectory(prefix="onestring-reference-bff-") as folder:
        root = Path(folder)
        input_path = root / "input.obj"
        output_path = root / "output.obj"
        _write_triangle_obj(input_path, vertices, faces)
        command = [str(exe), str(input_path), str(output_path), "--writeOnlyUVs"]
        if boundary_policy == "target_disk":
            command.append("--flattenToDisk")
        try:
            completed = subprocess.run(
                command,
                cwd=str(exe.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(float(timeout_seconds), 1.0),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReferenceBFFUnavailableError(f"Official BFF execution failed: {type(exc).__name__}: {exc}") from exc
        if completed.returncode != 0 or not output_path.is_file():
            raise ReferenceBFFUnavailableError(
                "Official BFF CLI failed; no fallback was used. "
                f"exit={completed.returncode}, stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-2000:]!r}"
            )
        uv = _read_write_only_uv_obj(output_path, len(vertices), np.asarray(faces, dtype=int))
    version_hint = exe.parent.name if "v" in exe.parent.name.lower() else "UNSPECIFIED_BY_BINARY"
    effective_policy = "boundary_scale_zero" if boundary_policy in {"automatic_reference", "boundary_scale_zero"} else boundary_policy
    return BFFRunResult(
        uv_vertices=uv,
        metrics={
            "parameterization_backend_name": "GeometryCollective/boundary-first-flattening:bff-command-line",
            "parameterization_backend_repository": OFFICIAL_BFF_REPOSITORY,
            "parameterization_backend_version": version_hint,
            "parameterization_backend_commit_sha": _discover_git_commit(exe),
            "parameterization_backend_executable": str(exe),
            "parameterization_backend_sha256": _sha256(exe),
            "bff_boundary_policy_requested": str(boundary_policy),
            "bff_boundary_policy_effective": effective_policy,
            "bff_automatic_cli_call": "BFF::flatten(zero boundary scale factors, givenScaleFactors=true)",
            "onestring_boundary_condition_status": "UNSPECIFIED_IN_PAPER",
            "fallbacks_used": [],
            "input_topology": topology,
        },
    )


def triangle_jacobian_diagnostics(
    surface_vertices: np.ndarray,
    uv_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    degeneracy_epsilon: float = 1e-14,
) -> dict[str, Any]:
    """Compute exact per-triangle differentials for S->UV and UV->S."""

    xyz = np.asarray(surface_vertices, dtype=float)
    uv = np.asarray(uv_vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    n = len(tris)
    sigma_s_to_uv = np.full((n, 2), np.nan, dtype=float)
    sigma_uv_to_s = np.full((n, 2), np.nan, dtype=float)
    area_scale_s_to_uv = np.full(n, np.nan, dtype=float)
    signed_uv_double_area = np.full(n, np.nan, dtype=float)
    surface_area = np.full(n, np.nan, dtype=float)
    uv_degenerate = np.zeros(n, dtype=bool)
    surface_degenerate = np.zeros(n, dtype=bool)

    for tri_id, face in enumerate(tris):
        p0, p1, p2 = xyz[face]
        e1 = p1 - p0
        e2 = p2 - p0
        length = float(np.linalg.norm(e1))
        normal = np.cross(e1, e2)
        doubled_area = float(np.linalg.norm(normal))
        surface_area[tri_id] = 0.5 * doubled_area
        if length <= degeneracy_epsilon or doubled_area <= 2.0 * degeneracy_epsilon:
            surface_degenerate[tri_id] = True
            continue
        x2 = float(np.dot(e2, e1 / length))
        y2 = doubled_area / length
        d_surface = np.asarray([[length, x2], [0.0, y2]], dtype=float)
        q0, q1, q2 = uv[face]
        d_uv = np.column_stack([q1 - q0, q2 - q0])
        det_uv = float(np.linalg.det(d_uv))
        signed_uv_double_area[tri_id] = det_uv
        if abs(det_uv) <= 2.0 * degeneracy_epsilon:
            uv_degenerate[tri_id] = True
            continue
        j_s_to_uv = d_uv @ np.linalg.inv(d_surface)
        j_uv_to_s = d_surface @ np.linalg.inv(d_uv)
        sv_forward = np.linalg.svd(j_s_to_uv, compute_uv=False)
        sv_inverse = np.linalg.svd(j_uv_to_s, compute_uv=False)
        sigma_s_to_uv[tri_id] = np.sort(sv_forward)[::-1]
        sigma_uv_to_s[tri_id] = np.sort(sv_inverse)[::-1]
        area_scale_s_to_uv[tri_id] = abs(float(np.linalg.det(j_s_to_uv)))

    nonzero_orientation = signed_uv_double_area[np.isfinite(signed_uv_double_area) & (np.abs(signed_uv_double_area) > 2.0 * degeneracy_epsilon)]
    reference_sign = 1.0 if len(nonzero_orientation) == 0 or float(np.median(nonzero_orientation)) >= 0.0 else -1.0
    flips = np.isfinite(signed_uv_double_area) & (reference_sign * signed_uv_double_area < -2.0 * degeneracy_epsilon)
    valid = ~(surface_degenerate | uv_degenerate) & np.all(np.isfinite(sigma_uv_to_s), axis=1)
    anisotropy = np.full(n, np.nan, dtype=float)
    anisotropy[valid] = sigma_s_to_uv[valid, 0] / np.maximum(sigma_s_to_uv[valid, 1], 1e-300)
    raw_lambda = np.full(n, np.nan, dtype=float)
    raw_lambda[valid] = sigma_uv_to_s[valid, 0]
    return {
        "mapping_direction": "surface_to_uv J=D_UV*inverse(D_S); auxetic expansion diagnostics use inverse J",
        "sigma_surface_to_uv": sigma_s_to_uv,
        "sigma_uv_to_surface": sigma_uv_to_s,
        "sigma1": sigma_uv_to_s[:, 0],
        "sigma2": sigma_uv_to_s[:, 1],
        "anisotropy": anisotropy,
        "area_scale_surface_to_uv": area_scale_s_to_uv,
        "area_scale_uv_to_surface": np.where(area_scale_s_to_uv > 0.0, 1.0 / area_scale_s_to_uv, np.nan),
        "raw_lambda_uv_to_surface_sigma_max": raw_lambda,
        "log_raw_lambda": np.log(raw_lambda),
        "triangle_flip": flips,
        "uv_degenerate_triangle": uv_degenerate,
        "surface_degenerate_triangle": surface_degenerate,
        "uv_triangle_flip_count": int(np.count_nonzero(flips)),
        "uv_degenerate_triangle_count": int(np.count_nonzero(uv_degenerate)),
        "surface_degenerate_triangle_count": int(np.count_nonzero(surface_degenerate)),
        "valid_triangle_count": int(np.count_nonzero(valid)),
        "surface_triangle_area": surface_area,
    }


def count_internal_triangle_overlaps(uv_vertices: np.ndarray, faces: np.ndarray, tolerance: float = 1e-12) -> int:
    """Count positive-area overlaps between non-adjacent UV triangles."""

    uv = np.asarray(uv_vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    points = uv[tris]
    minimum = np.min(points, axis=1)
    maximum = np.max(points, axis=1)

    def orient(a, b, c) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def proper_intersection(a, b, c, d) -> bool:
        o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
        return bool(o1 * o2 < -tolerance and o3 * o4 < -tolerance)

    def strictly_inside(point, triangle) -> bool:
        signs = np.asarray([orient(triangle[0], triangle[1], point), orient(triangle[1], triangle[2], point), orient(triangle[2], triangle[0], point)])
        return bool(np.all(signs > tolerance) or np.all(signs < -tolerance))

    overlaps = 0
    for first in range(len(tris)):
        ids_first = set(int(value) for value in tris[first])
        for second in range(first + 1, len(tris)):
            if ids_first.intersection(int(value) for value in tris[second]):
                continue
            if np.any(maximum[first] < minimum[second] - tolerance) or np.any(maximum[second] < minimum[first] - tolerance):
                continue
            a = points[first]
            b = points[second]
            intersects = any(
                proper_intersection(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3])
                for i in range(3)
                for j in range(3)
            )
            contained = any(strictly_inside(point, b) for point in a) or any(strictly_inside(point, a) for point in b)
            if intersects or contained:
                overlaps += 1
    return int(overlaps)


def normalize_uv_and_compute_csf(
    surface_vertices: np.ndarray,
    uv_vertices: np.ndarray,
    faces: np.ndarray,
    normalization: Literal["min_to_one_hypothesis_a", "none_unspecified"] = "min_to_one_hypothesis_a",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve BFF's similarity scale with an explicitly labelled hypothesis."""

    uv = np.asarray(uv_vertices, dtype=float).copy()
    initial = triangle_jacobian_diagnostics(surface_vertices, uv, faces)
    raw = np.asarray(initial["raw_lambda_uv_to_surface_sigma_max"], dtype=float)
    valid = raw[np.isfinite(raw) & (raw > 0.0)]
    if not len(valid):
        raise ReferenceBFFUnavailableError("Official BFF output has no valid triangle differential.")
    if normalization == "min_to_one_hypothesis_a":
        uv_scale = float(np.min(valid))
        uv *= uv_scale
        status = "hypothesis_a: scale UV so min(max singular value of J^-1) = 1"
    elif normalization == "none_unspecified":
        uv_scale = 1.0
        status = "UNSPECIFIED_IN_PAPER: official CLI similarity scale retained"
    else:
        raise ValueError(f"unknown reference CSF normalization: {normalization}")
    final = triangle_jacobian_diagnostics(surface_vertices, uv, faces)
    lambda_values = np.asarray(final["raw_lambda_uv_to_surface_sigma_max"], dtype=float)
    final.update(
        {
            "lambda": lambda_values,
            "log_lambda": np.log(lambda_values),
            "lambda_normalization": normalization,
            "lambda_normalization_status": status,
            "lambda_uv_similarity_scale_applied": uv_scale,
            "lambda_min": float(np.nanmin(lambda_values)),
            "lambda_median": float(np.nanmedian(lambda_values)),
            "lambda_max": float(np.nanmax(lambda_values)),
            "lambda_bound": 2.0,
            "lambda_exceeds_bound_triangle_count": int(np.count_nonzero(lambda_values > 2.0)),
        }
    )
    return uv, final


def strict_inverse_map_uv_to_surface(
    uv_point: np.ndarray,
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    *,
    tolerance: float = 1e-10,
    vertex_id: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Map one UV point by a containing triangle only; never use nearest geometry."""

    point = np.asarray(uv_point, dtype=float)
    uv = np.asarray(uv_vertices, dtype=float)
    uv_tris = np.asarray(uv_faces, dtype=int)
    xyz = np.asarray(surface_vertices, dtype=float)
    xyz_tris = np.asarray(surface_faces, dtype=int)
    best: tuple[float, int, np.ndarray] | None = None
    degenerate_count = 0
    for tri_id, face in enumerate(uv_tris):
        a, b, c = uv[face]
        matrix = np.column_stack([b - a, c - a])
        det = float(np.linalg.det(matrix))
        if abs(det) <= 1e-14:
            degenerate_count += 1
            continue
        local = np.linalg.solve(matrix, point - a)
        bary = np.asarray([1.0 - local[0] - local[1], local[0], local[1]], dtype=float)
        violation = float(max(0.0, -np.min(bary), np.max(bary) - 1.0))
        if violation <= float(tolerance):
            score = float(np.max(np.abs(np.minimum(bary, 0.0))))
            if best is None or score < best[0]:
                best = (score, int(tri_id), bary)
    if best is None:
        reason = "uv_degenerate_triangles" if degenerate_count == len(uv_tris) else "outside_omega_or_numerical_tolerance"
        raise ReferenceInverseMapError(reason, point, vertex_id)
    _, tri_id, bary = best
    mapped = bary @ xyz[xyz_tris[tri_id]]
    round_trip = bary @ uv[uv_tris[tri_id]]
    return mapped, tri_id, bary, float(np.linalg.norm(round_trip - point))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_diagnostics_json(path: str | os.PathLike[str], diagnostics: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_ready(diagnostics), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return target
