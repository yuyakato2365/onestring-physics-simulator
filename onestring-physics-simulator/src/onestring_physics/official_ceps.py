"""Bridge from OneString to the official MarkGillespie/CEPS C++ CLI.

The CEPS metric solve, Ptolemy flips, correspondence tracking, and common
refinement are performed by the reference executable. This module writes the
boundary data and imports its ordinary-UV common-refinement OBJ.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .discrete_bff import EPS, _boundary_loop, _corners, _target


@dataclass(frozen=True)
class CepsObjResult:
    surface_vertices: np.ndarray
    surface_faces: np.ndarray
    uv_vertices: np.ndarray
    uv_faces: np.ndarray
    vertex_uv: dict[int, np.ndarray]
    raw_texture_coordinate_count: int


def _write_input_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for p in np.asarray(vertices, float):
            out.write(f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        for f in np.asarray(faces, int)[:, :3]:
            out.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def _write_curvatures(path: Path, vertex_count: int, corners: Iterable[int]) -> None:
    corner_set = set(map(int, corners))
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# 0-indexed CEPS target curvature / boundary exterior angle\n")
        for vertex in range(vertex_count):
            value = math.pi / 2.0 if vertex in corner_set else 0.0
            out.write(f"{vertex} {value:.17g}\n")


def _obj_index(value: str, count: int) -> int:
    index = int(value)
    return index - 1 if index > 0 else count + index


def _weld_uv(values: np.ndarray, decimals: int = 12) -> tuple[np.ndarray, np.ndarray]:
    unique: list[np.ndarray] = []
    lookup: dict[tuple[float, float], int] = {}
    mapping = np.empty(len(values), dtype=int)
    for i, uv in enumerate(np.asarray(values, float)):
        key = tuple(np.round(uv[:2], decimals).tolist())
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(uv[:2].copy())
        mapping[i] = lookup[key]
    return np.asarray(unique, float), mapping


def _parse_ceps_obj(path: Path) -> CepsObjResult:
    vertices: list[list[float]] = []
    textures: list[list[float]] = []
    polygons: list[list[tuple[int, int]]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = raw.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(list(map(float, fields[1:4])))
        elif fields[0] == "vt" and len(fields) >= 3:
            textures.append(list(map(float, fields[1:3])))
        elif fields[0] == "f" and len(fields) >= 4:
            refs: list[tuple[int, int]] = []
            for token in fields[1:]:
                parts = token.split("/")
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    raise RuntimeError(f"CEPS OBJ face lacks texture indices at line {line_number}")
                refs.append((_obj_index(parts[0], len(vertices)), _obj_index(parts[1], len(textures))))
            polygons.append(refs)
    if not vertices or not textures or not polygons:
        raise RuntimeError("official CEPS output OBJ is incomplete")
    surface_faces: list[list[int]] = []
    texture_faces: list[list[int]] = []
    for polygon in polygons:
        for i in range(1, len(polygon) - 1):
            tri = [polygon[0], polygon[i], polygon[i + 1]]
            surface_faces.append([x[0] for x in tri])
            texture_faces.append([x[1] for x in tri])
    surface_vertices = np.asarray(vertices, float)
    raw_uv = np.asarray(textures, float)
    sf, tf = np.asarray(surface_faces, int), np.asarray(texture_faces, int)
    if np.min(sf) < 0 or np.max(sf) >= len(surface_vertices) or np.min(tf) < 0 or np.max(tf) >= len(raw_uv):
        raise RuntimeError("official CEPS output OBJ contains an invalid index")
    uv_vertices, texture_map = _weld_uv(raw_uv)
    uv_faces = texture_map[tf]
    samples: dict[int, list[np.ndarray]] = {}
    for sface, tface in zip(sf, tf):
        for sv, tv in zip(sface, tface):
            samples.setdefault(int(sv), []).append(raw_uv[int(tv)])
    vertex_uv = {v: np.mean(np.asarray(values), axis=0) for v, values in samples.items()}
    return CepsObjResult(surface_vertices, sf, uv_vertices, uv_faces, vertex_uv, len(raw_uv))


def _read_vertex_map(path: Path, expected: int) -> np.ndarray:
    values = np.asarray([int(line) for line in path.read_text().splitlines() if line.strip()], int)
    if len(values) != expected or np.any(values < 0):
        raise RuntimeError(f"official CEPS vertex map is incomplete: got {len(values)}, expected {expected}")
    return values


def _align_uv(
    uv: np.ndarray,
    input_corner_ids: Sequence[int],
    vertex_map: np.ndarray,
    vertex_uv: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    corners = np.asarray([vertex_uv[int(vertex_map[int(i)])] for i in input_corner_ids], float)
    edge = corners[1] - corners[0]
    if np.linalg.norm(edge) <= EPS:
        raise RuntimeError("official CEPS collapsed the first rectangle side")
    angle = -math.atan2(float(edge[1]), float(edge[0]))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[c, -s], [s, c]])
    aligned, aligned_corners = np.asarray(uv, float) @ rotation.T, corners @ rotation.T
    area = 0.5 * np.sum(
        aligned_corners[:, 0] * np.roll(aligned_corners[:, 1], -1)
        - np.roll(aligned_corners[:, 0], -1) * aligned_corners[:, 1]
    )
    if area < 0:
        aligned[:, 1] *= -1
        aligned_corners[:, 1] *= -1
    center = (np.min(aligned_corners, axis=0) + np.max(aligned_corners, axis=0)) / 2.0
    aligned -= center
    aligned_corners -= center
    scale = 2.0 / max(float(np.max(np.ptp(aligned_corners, axis=0))), EPS)
    aligned *= scale
    aligned_corners *= scale
    axis_error = max(
        abs(aligned_corners[1, 1] - aligned_corners[0, 1]),
        abs(aligned_corners[2, 0] - aligned_corners[1, 0]),
        abs(aligned_corners[3, 1] - aligned_corners[2, 1]),
        abs(aligned_corners[0, 0] - aligned_corners[3, 0]),
    )
    return aligned, {
        "ceps_uv_alignment_rotation_radians": angle,
        "ceps_uv_alignment_scale": scale,
        "ceps_aligned_corner_coordinates": aligned_corners.tolist(),
        "ceps_aligned_corner_axis_error": float(axis_error),
    }


def _signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, float)[np.asarray(faces, int)[:, :3]]
    return 0.5 * (
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    )


def _boundary_loops(faces: np.ndarray) -> list[list[int]]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, int)[:, :3]:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(a), int(b))))
            counts[key] = counts.get(key, 0) + 1
    graph: dict[int, list[int]] = {}
    for (a, b), count in counts.items():
        if count == 1:
            graph.setdefault(a, []).append(b)
            graph.setdefault(b, []).append(a)
    loops: list[list[int]] = []
    unused = {edge for edge, count in counts.items() if count == 1}
    while unused:
        a, b = next(iter(unused))
        loop, previous, current = [a], a, b
        unused.discard((min(a, b), max(a, b)))
        for _ in range(len(graph) + 5):
            if current == loop[0]:
                break
            loop.append(current)
            candidates = [x for x in graph.get(current, []) if x != previous]
            if not candidates:
                break
            nxt = candidates[0]
            unused.discard((min(current, nxt), max(current, nxt)))
            previous, current = current, nxt
        if current == loop[0] and len(loop) >= 3:
            loops.append(loop)
    return loops


def _polygon_area(points: np.ndarray) -> float:
    p = np.asarray(points, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _omega_boundary(uv: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    loops = _boundary_loops(faces)
    if loops:
        loop = max(loops, key=lambda x: abs(_polygon_area(uv[np.asarray(x, int)])))
        polygon, source = uv[np.asarray(loop, int)], "largest_uv_boundary_loop"
    else:
        polygon, source = uv[ConvexHull(uv).vertices], "convex_hull_fallback"
    if _polygon_area(polygon) < 0:
        polygon = polygon[::-1]
    return np.vstack([polygon, polygon[0]]), {
        "ceps_uv_boundary_loop_count": len(loops),
        "ceps_omega_boundary_source": source,
        "ceps_omega_boundary_vertex_count": len(polygon),
    }


def _resolve_command(params: Any, project_root: Path) -> tuple[list[str], str]:
    configured = getattr(params, "ceps_command", None)
    if configured:
        command = [str(x) for x in configured] if isinstance(configured, (list, tuple)) else shlex.split(str(configured), posix=os.name != "nt")
        return command, "PipelineParameters.ceps_command"
    env_command = os.environ.get("ONESTRING_CEPS_COMMAND", "").strip()
    if env_command:
        return shlex.split(env_command, posix=os.name != "nt"), "ONESTRING_CEPS_COMMAND"
    explicit = str(getattr(params, "ceps_executable", "") or "").strip()
    environment = os.environ.get("ONESTRING_CEPS_EXECUTABLE", "").strip()
    candidates = [
        (explicit, "PipelineParameters.ceps_executable"),
        (environment, "ONESTRING_CEPS_EXECUTABLE"),
        (str(project_root / "external/CEPS/build/bin/Release/parameterize.exe"), "auto_detected"),
        (str(project_root / "external/CEPS/build/bin/parameterize.exe"), "auto_detected"),
        (r"C:\CEPS\build\bin\Release\parameterize.exe", "auto_detected"),
        (r"C:\CEPS\build\bin\parameterize.exe", "auto_detected"),
    ]
    for candidate, source in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return [str(Path(candidate).expanduser().resolve())], source
    executable = shutil.which("parameterize.exe" if os.name == "nt" else "parameterize")
    if executable:
        return [executable], "PATH"
    raise RuntimeError(
        "Official CEPS executable was not found. Run `powershell -ExecutionPolicy Bypass "
        "-File scripts\\setup_ceps_windows.ps1`, then restart PowerShell or set "
        "`$env:ONESTRING_CEPS_EXECUTABLE` to parameterize.exe."
    )


def official_ceps_rectangle(
    vertices: np.ndarray, faces: np.ndarray, params: Any, *, project_root: Path | None = None
) -> tuple[CepsObjResult, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    pts, tris = np.asarray(vertices, float), np.asarray(faces, int)[:, :3]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4 or len(tris) < 2 or not np.all(np.isfinite(pts)):
        raise RuntimeError("official CEPS requires a finite Vx3 triangle mesh")
    loop, topology = _boundary_loop(tris, len(pts))
    target, target_metrics = _target(pts, loop, params)
    ordered_loop, corner_positions, corner_metrics = _corners(pts, loop, target)
    corner_ids = [ordered_loop[p] for p in corner_positions]
    root = Path(project_root or Path(__file__).resolve().parents[2])
    command_prefix, command_source = _resolve_command(params, root)
    timeout = max(1.0, float(getattr(params, "ceps_timeout_seconds", 600.0)))
    keep_temp = bool(getattr(params, "ceps_keep_temporary_files", False))
    with tempfile.TemporaryDirectory(prefix="onestring-ceps-", dir=getattr(params, "ceps_temporary_directory", None)) as temporary:
        work = Path(temporary)
        input_obj, curvature = work / "input.obj", work / "rectangle_curvatures.txt"
        output_obj, vertex_map_path, log_path = work / "ceps.obj", work / "vertex_map.txt", work / "log.tsv"
        _write_input_obj(input_obj, pts, tris)
        _write_curvatures(curvature, len(pts), corner_ids)
        command = [
            *command_prefix,
            str(input_obj),
            f"--curvatures={curvature}",
            "--noFreeBoundary",
            f"--outputLinearTextureFilename={output_obj}",
            f"--outputVertexMapFilename={vertex_map_path}",
            f"--outputLogFilename={log_path}",
            "--verbose=false",
        ]
        process = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=timeout, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"official CEPS failed ({process.returncode}). stdout={process.stdout[-2000:]!r}; stderr={process.stderr[-2000:]!r}"
            )
        if not output_obj.is_file() or not vertex_map_path.is_file():
            raise RuntimeError("official CEPS did not write the common-refinement OBJ or vertex map")
        parsed = _parse_ceps_obj(output_obj)
        vertex_map = _read_vertex_map(vertex_map_path, len(pts))
        aligned_uv, alignment_metrics = _align_uv(parsed.uv_vertices, corner_ids, vertex_map, parsed.vertex_uv)
        signed = _signed_areas(aligned_uv, parsed.uv_faces)
        if float(np.median(signed)) < 0:
            aligned_uv[:, 1] *= -1
            signed = _signed_areas(aligned_uv, parsed.uv_faces)
        boundary, boundary_metrics = _omega_boundary(aligned_uv, parsed.uv_faces)
        span = np.maximum(np.ptp(boundary[:-1], axis=0), EPS)
        achieved = float(max(span) / min(span))
        requested = float(target_metrics["boundary_target_aspect_ratio"])
        requested = max(requested, 1.0 / max(requested, EPS))
        result = CepsObjResult(parsed.surface_vertices, parsed.surface_faces, aligned_uv, parsed.uv_faces, parsed.vertex_uv, parsed.raw_texture_coordinate_count)
        metrics: dict[str, Any] = {
            **topology, **target_metrics, **corner_metrics, **alignment_metrics, **boundary_metrics,
            "parameterization_method": "ceps",
            "parameterization_exactness_label": "official_ceps_common_refinement_linear_uv_export",
            "flattening_backend": "official_MarkGillespie_CEPS_parameterize_CLI",
            "omega_parameterization_solver": "CEPS_intrinsic_Delaunay_Newton_Ptolemy_common_refinement",
            "omega_boundary_constraint_model": "CEPS_prescribed_boundary_exterior_angles_via_doubled_mesh",
            "omega_boundary_forced_rectangle": True,
            "omega_boundary_fixed": False,
            "omega_boundary_shape": "CEPS polygonal rectangle from four prescribed pi/2 exterior angles",
            "ceps_implemented": True,
            "ceps_backend_used": "official_ceps_cli",
            "ceps_reference_backend": True,
            "ceps_reference_repository": "MarkGillespie/CEPS",
            "ceps_reference_executable": command_prefix[0],
            "ceps_command_source": command_source,
            "ceps_command": command,
            "ceps_common_refinement_used": True,
            "ceps_texture_interpolation": "ordinary_linear_uv_export",
            "ceps_projective_interpolation_used": False,
            "ceps_output_surface_vertex_count": len(parsed.surface_vertices),
            "ceps_output_triangle_count": len(parsed.surface_faces),
            "ceps_output_uv_vertex_count_after_weld": len(aligned_uv),
            "ceps_output_raw_texture_coordinate_count": parsed.raw_texture_coordinate_count,
            "ceps_prescribed_boundary_curvature": True,
            "ceps_boundary_exterior_angle_at_corners": math.pi / 2.0,
            "ceps_boundary_exterior_angle_elsewhere": 0.0,
            "ceps_ptolemy_flips": "performed inside official CEPS backend",
            "ceps_normal_coordinate_correspondence": "represented by official common-refinement output",
            "ceps_process_returncode": process.returncode,
            "ceps_stdout_tail": process.stdout[-4000:],
            "ceps_stderr_tail": process.stderr[-4000:],
            "uv_triangle_flip_count": int(np.sum(signed < -1e-12)),
            "uv_degenerate_triangle_count": int(np.sum(np.abs(signed) <= 1e-12)),
            "uv_min_triangle_area": float(np.min(np.abs(signed))) if len(signed) else 0.0,
            "boundary_loop": list(map(int, ordered_loop)),
            "boundary_achieved_unoriented_aspect_ratio": achieved,
            "boundary_requested_unoriented_aspect_ratio": requested,
            "boundary_aspect_relative_error": abs(achieved - requested) / max(requested, EPS),
            "parameterization_runtime_seconds": time.perf_counter() - started,
            "parameterization_warning": "",
        }
        if metrics["uv_triangle_flip_count"]:
            metrics["parameterization_warning"] = "Official CEPS output contains inverted ordinary-UV triangles; no fallback was substituted."
        if metrics["uv_degenerate_triangle_count"]:
            metrics["parameterization_warning"] = (metrics["parameterization_warning"] + "; degenerate UV triangles detected").strip("; ")
        if keep_temp:
            destination = root / "ceps_debug_output"
            destination.mkdir(parents=True, exist_ok=True)
            for item in work.iterdir():
                shutil.copy2(item, destination / item.name)
            metrics["ceps_temporary_output_directory"] = str(destination)
        return result, boundary, metrics


def install_official_ceps(pipeline_module: Any) -> None:
    """Patch ``omega_parameterization_mode='ceps'`` into the pipeline."""
    if getattr(pipeline_module, "_OFFICIAL_CEPS_PATCH_INSTALLED", False):
        return
    legacy = pipeline_module._build_surface_parameterization

    def build(surface: Any, target: Any, grid: Any, params: Any) -> Any:
        mode = str(getattr(params, "omega_parameterization_mode", "bff"))
        debug = str(getattr(params, "m3d_construction_mode", "mesh_harmonic")) == "analytic_scaled_heightfield_debug"
        if mode not in {"ceps", "ceps_official"} or debug:
            return legacy(surface, target, grid, params)
        if str(getattr(params, "omega_boundary_mode", "paper_default")) != "paper_default":
            raise ValueError("official CEPS mode requires omega_boundary_mode='paper_default'")
        result, boundary, ceps_metrics = official_ceps_rectangle(np.asarray(surface.vertices, float), np.asarray(surface.faces[:, :3], int), params)
        slope = {"mean_slope": 0.0, "max_slope": 0.0} if getattr(target, "kind", "") == "sampled" else pipeline_module._original._heightfield_metric_summary(target, grid)
        metrics = {
            **ceps_metrics,
            "omega_boundary_mode": "paper_default",
            "omega_parameterization_mode": "ceps",
            "requested_omega_parameterization_mode": mode,
            "surface_vertex_count": len(result.surface_vertices),
            "surface_triangle_count": len(result.surface_faces),
            "boundary_vertex_count": len(boundary) - 1,
            "mean_slope": float(slope["mean_slope"]),
            "max_slope": float(slope["max_slope"]),
            "height_field_shortcut_used": False,
            "omega_corresponds_to_S": True,
            "omega_correspondence_model": "official CEPS common refinement with ordinary per-corner UV coordinates",
            "paper_flow_stage": "S -> Omega by official CEPS uniformization and common refinement",
            "paper_exactness_warning": "CEPS is a OneString research extension; the OneString paper specifies BFF for this stage.",
            "omega_warning": str(ceps_metrics.get("parameterization_warning", "")),
        }
        out = pipeline_module._original.SurfaceParameterization(
            method="ceps",
            surface_vertices_3d=result.surface_vertices,
            surface_faces=result.surface_faces,
            uv_vertices_2d=result.uv_vertices,
            uv_faces=result.uv_faces,
            omega_boundary=boundary,
            triangle_acceleration=None,
            metrics=metrics,
        )
        marker = getattr(pipeline_module, "_mark_parameterization_mode", None)
        return marker(out, method="ceps", exactness="official_ceps_common_refinement_linear_uv_export", warning=metrics["parameterization_warning"]) if callable(marker) else out

    pipeline_module._build_surface_parameterization = build
    pipeline_module._original._build_surface_parameterization = build
    pipeline_module._OFFICIAL_CEPS_PATCH_INSTALLED = True
