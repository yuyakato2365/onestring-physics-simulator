"""Fail-fast bridge to Autodesk's official Affine Body Dynamics executable.

This module does not implement an ABD substitute.  It serializes OneString
geometry and constraints, invokes an external Autodesk-derived executable, and
parses its affine transforms.  The stock Autodesk executable can run contact and
joint scenes, but a full OneString run additionally requires an executable that
advertises ``--onestring-manifest`` for the unilateral string constraint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal

import numpy as np

from .t3d_recovery import triangulate_solid


class ABDBackendError(RuntimeError):
    pass


class ABDBackendUnavailableError(ABDBackendError):
    pass


class ABDCapabilityError(ABDBackendError):
    pass


@dataclass
class ShakeTrajectory:
    amplitude: float = 0.0
    frequency_hz: float = 0.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ABDBackendConfig:
    executable: str | None = None
    timestep: float = 0.01
    steps: int = 100
    density: float = 1000.0
    friction: float = 0.25
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    orthogonality_stiffness: float = 1e9
    minimum_separation_distance: float = 1e-6
    barrier_activation_distance: float = 5e-4
    newton_velocity_tolerance: float = 1e-3
    nthreads: int = 0
    timeout_seconds: float = 600.0
    pull_end_ratio: float = 0.75
    shake: ShakeTrajectory = field(default_factory=ShakeTrajectory)
    require_onestring_extension: bool = True


@dataclass
class ABDPreparedJob:
    job_dir: Path
    scene_path: Path
    manifest_path: Path
    output_dir: Path
    body_proxy_local_vertices: np.ndarray
    body_rest_volumes: np.ndarray
    body_rest_edge_lengths: list[np.ndarray]
    guide_count: int


@dataclass
class ABDRunResult:
    frames: list[np.ndarray]
    final_tiles: np.ndarray
    metrics: dict[str, Any]
    collision_counts: list[int]
    frame_logs: list[dict[str, Any]]
    result_json_path: str
    gltf_path: str | None
    npz_path: str


def find_abd_executable(explicit: str | None = None, project_root: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("ONESTRING_ABD_EXECUTABLE") or os.environ.get("ABD_EXECUTABLE")
    if env_value:
        candidates.append(Path(env_value))
    if project_root is not None:
        root = Path(project_root)
        candidates.extend(
            [
                root / "third_party" / "affine-body-dynamics" / "build" / "Release" / "abd_sim.exe",
                root / "third_party" / "affine-body-dynamics" / "build" / "abd_sim.exe",
                root / "third_party" / "affine-body-dynamics" / "build" / "abd_sim",
            ]
        )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return None


def probe_abd_capabilities(executable: str | Path) -> dict[str, Any]:
    path = Path(executable)
    if not path.is_file():
        return {"available": False, "reason": "executable_not_found", "path": str(path)}
    try:
        completed = subprocess.run(
            [str(path), "--help"], capture_output=True, text=True, timeout=30, check=False
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc), "path": str(path)}
    help_text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    return {
        "available": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "path": str(path.resolve()),
        "stock_headless_scene": "--scene-path" in help_text and "--output-path" in help_text,
        "onestring_unilateral_string_extension": "--onestring-manifest" in help_text,
        "help": help_text,
    }


def _triangles_from_proxy() -> list[tuple[int, int, int]]:
    return [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]


def _write_obj(path: Path, vertices: np.ndarray, triangles: list[tuple[int, int, int]]) -> None:
    lines = [f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for p in np.asarray(vertices, dtype=float)]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _mesh_volume(vertices: np.ndarray, triangles: list[tuple[int, int, int]]) -> float:
    verts = np.asarray(vertices, dtype=float)
    return abs(float(sum(np.dot(verts[a], np.cross(verts[b], verts[c])) for a, b, c in triangles) / 6.0))


def _mesh_edge_lengths(vertices: np.ndarray, triangles: list[tuple[int, int, int]]) -> np.ndarray:
    edges = sorted({tuple(sorted((tri[i], tri[(i + 1) % 3]))) for tri in triangles for i in range(3)})
    verts = np.asarray(vertices, dtype=float)
    return np.asarray([np.linalg.norm(verts[a] - verts[b]) for a, b in edges], dtype=float)


def _state_tile_geometry(state: Any) -> tuple[list[np.ndarray], list[list[tuple[int, int, int]]]]:
    solids = getattr(state.tiles_3d, "authoritative_solids", None)
    if solids:
        return (
            [np.asarray(solid.vertices, dtype=float) for solid in solids],
            [triangulate_solid(solid) for solid in solids],
        )
    proxy = np.asarray(state.tiles_3d.vertices, dtype=float)
    return [tile.copy() for tile in proxy], [_triangles_from_proxy() for _ in range(len(proxy))]


def _hinge_manifest(
    state: Any,
    centers: np.ndarray,
    initial_proxy: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    constraints: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    pair_points: dict[tuple[int, int], list[np.ndarray]] = {}
    for hinge in state.hinge_graph.hinges:
        tile_a, tile_b = int(hinge.tile_a), int(hinge.tile_b)
        if tile_a == tile_b or min(tile_a, tile_b) < 0 or max(tile_a, tile_b) >= len(centers):
            continue
        offset = 0 if str(hinge.surface) == "top" else 4
        point_a = initial_proxy[tile_a, offset + int(hinge.local_vertex_a)]
        point_b = initial_proxy[tile_b, offset + int(hinge.local_vertex_b)]
        point = 0.5 * (point_a + point_b)
        pair = tuple(sorted((tile_a, tile_b)))
        pair_points.setdefault(pair, []).append(point)
        constraints.append(
            {
                "type": "pin_joint",
                "bodyA_name": f"tile_{tile_a:04d}",
                "bodyA_point": (point - centers[tile_a]).tolist(),
                "bodyB_name": f"tile_{tile_b:04d}",
            }
        )
    for (tile_a, tile_b), points in pair_points.items():
        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - existing) <= 1e-9 for existing in unique):
                unique.append(point)
        axis = np.zeros(3, dtype=float)
        if len(unique) >= 2:
            raw = unique[-1] - unique[0]
            norm = float(np.linalg.norm(raw))
            if norm > 1e-12:
                axis = raw / norm
        metadata.append(
            {
                "tile_a": tile_a,
                "tile_b": tile_b,
                "anchors": [point.tolist() for point in unique],
                "axis": axis.tolist(),
            }
        )
    return constraints, metadata


def _string_guides(state: Any, centers: np.ndarray, initial_proxy: np.ndarray) -> list[dict[str, Any]]:
    gap_by_id = {int(gap.id): gap for gap in state.gap_graph.gaps}
    guides: list[dict[str, Any]] = []
    for gap_id in state.string_path.gap_ids:
        gap = gap_by_id.get(int(gap_id))
        if gap is None or not gap.surrounding_tiles:
            continue
        tile_id = int(gap.surrounding_tiles[0])
        if tile_id < 0 or tile_id >= len(centers):
            continue
        centroid_2d = np.asarray(gap.centroid_2d, dtype=float)
        tile_top = initial_proxy[tile_id, :4]
        nearest = int(np.argmin(np.linalg.norm(tile_top[:, :2] - centroid_2d[None, :], axis=1)))
        world = tile_top[nearest].copy()
        guides.append(
            {
                "gap_id": int(gap_id),
                "body_id": tile_id,
                "body_name": f"tile_{tile_id:04d}",
                "material_point": (world - centers[tile_id]).tolist(),
                "initial_world_point": world.tolist(),
            }
        )
    return guides


def prepare_abd_job(state: Any, config: ABDBackendConfig, job_dir: str | Path) -> ABDPreparedJob:
    job_dir = Path(job_dir).resolve()
    assets_dir = job_dir / "assets"
    output_dir = job_dir / "result"
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_vertices, tile_triangles = _state_tile_geometry(state)
    target_proxy = np.asarray(state.tiles_3d.vertices, dtype=float)
    initial_proxy = np.asarray(state.tiles_2d_dual_hinge.vertices, dtype=float)
    if initial_proxy.shape != target_proxy.shape:
        raise ABDBackendError("T2D initial proxy and T3D compatibility proxy must have the same shape")
    centers = np.mean(initial_proxy, axis=1)
    proxy_local = initial_proxy - centers[:, None, :]
    bodies: list[dict[str, Any]] = []
    rest_volumes: list[float] = []
    rest_edges: list[np.ndarray] = []
    for tile_id, (vertices_world, triangles) in enumerate(zip(tile_vertices, tile_triangles)):
        target_center = np.mean(target_proxy[tile_id], axis=0)
        source_centered = target_proxy[tile_id] - target_center[None, :]
        initial_centered = initial_proxy[tile_id] - centers[tile_id][None, :]
        u, _singular, vt = np.linalg.svd(source_centered.T @ initial_centered, full_matrices=False)
        rotation = vt.T @ u.T
        if float(np.linalg.det(rotation)) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        local = (np.asarray(vertices_world, dtype=float) - np.mean(vertices_world, axis=0)[None, :]) @ rotation.T
        mesh_path = assets_dir / f"tile_{tile_id:04d}.obj"
        _write_obj(mesh_path, local, triangles)
        bodies.append(
            {
                "mesh": str(mesh_path),
                "position": centers[tile_id].tolist(),
                "rotation": [0.0, 0.0, 0.0],
                "density": float(config.density),
                "oriented": True,
                "type": "dynamic",
                "group_id": int(tile_id),
            }
        )
        rest_volumes.append(_mesh_volume(local, triangles))
        rest_edges.append(_mesh_edge_lengths(local, triangles))
    linear_constraints, hinge_metadata = _hinge_manifest(state, centers, initial_proxy)
    guides = _string_guides(state, centers, initial_proxy)
    initial_length = float(
        sum(
            np.linalg.norm(np.asarray(guides[idx + 1]["initial_world_point"]) - np.asarray(guides[idx]["initial_world_point"]))
            for idx in range(max(0, len(guides) - 1))
        )
    )
    duration = float(config.steps) * float(config.timestep)
    scene = {
        "scene_type": "distance_barrier_rb_problem",
        "solver": "ipc_solver",
        "timestep": float(config.timestep),
        "max_iterations": int(config.steps),
        "distance_barrier_constraint": {
            "trajectory_type": "ACCD",
            "initial_barrier_activation_distance": float(config.barrier_activation_distance),
            "minimum_separation_distance": float(config.minimum_separation_distance),
        },
        "ipc_solver": {"velocity_conv_tol": float(config.newton_velocity_tolerance)},
        "friction_constraints": {"iterations": -1},
        "rigid_body_problem": {
            "coefficient_restitution": -1.0,
            "coefficient_friction": float(config.friction),
            "gravity": list(config.gravity),
            "orthogonality_stiffness": float(config.orthogonality_stiffness),
            "do_intersection_check": True,
            "rigid_bodies": bodies,
            "linear_constraints": linear_constraints,
        },
    }
    manifest = {
        "schema": "onestring-abd-bridge-v1",
        "official_scene_path": str(job_dir / "scene.json"),
        "tile_count": len(bodies),
        "thickness": float(state.tiles_3d.metrics.get("requested_thickness", state.tiles_3d.metrics.get("tile_thickness", 0.0))),
        "density": float(config.density),
        "mass_model": "Autodesk ABD computes mass from closed rest mesh and density",
        "initial_positions": centers.tolist(),
        "initial_orientations_xyz_degrees": [[0.0, 0.0, 0.0] for _ in bodies],
        "hinges": hinge_metadata,
        "string": {
            "model": "unilateral_total_guide_length_constraint",
            "guide_points": guides,
            "initial_length": initial_length,
            "pull_schedule": [
                {"time": 0.0, "command_length": initial_length},
                {"time": duration, "command_length": initial_length * float(config.pull_end_ratio)},
            ],
            "inequality": "L(q) <= L_command(t)",
            "compression_force_when_slack": 0.0,
        },
        "shake_trajectory": asdict(config.shake),
        "friction": float(config.friction),
        "timestep": float(config.timestep),
        "gravity": list(config.gravity),
        "abd_stiffness": float(config.orthogonality_stiffness),
        "required_frame_logs": [
            "newton_iterations", "ccd_seconds", "linear_solve_seconds", "active_contacts",
            "minimum_contact_distance", "orthogonality_error_max", "orthogonality_error_mean",
            "max_edge_length_change_ratio", "volume_change_ratio", "string_length",
            "constraint_violation",
        ],
    }
    scene_path = job_dir / "scene.json"
    manifest_path = job_dir / "onestring_manifest.json"
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return ABDPreparedJob(
        job_dir=job_dir,
        scene_path=scene_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        body_proxy_local_vertices=proxy_local,
        body_rest_volumes=np.asarray(rest_volumes, dtype=float),
        body_rest_edge_lengths=rest_edges,
        guide_count=len(guides),
    )


def _frame_affine_metrics(frame: dict[str, Any], job: ABDPreparedJob) -> tuple[np.ndarray, dict[str, float]]:
    bodies = frame.get("rigid_bodies", [])
    if len(bodies) != len(job.body_proxy_local_vertices):
        raise ABDBackendError("ABD result body count does not match exported OneString tiles")
    world_proxy = np.zeros_like(job.body_proxy_local_vertices)
    orthogonality: list[float] = []
    edge_changes: list[float] = []
    volume_changes: list[float] = []
    for body_id, body in enumerate(bodies):
        position = np.asarray(body["position"], dtype=float)
        transform = np.asarray(body["transform"], dtype=float)
        if transform.shape != (3, 3):
            raise ABDBackendError("ABD result contains a non-3D affine transform")
        world_proxy[body_id] = job.body_proxy_local_vertices[body_id] @ transform.T + position[None, :]
        orthogonality.append(float(np.linalg.norm(transform.T @ transform - np.eye(3), ord="fro")))
        singular = np.linalg.svd(transform, compute_uv=False)
        edge_changes.append(float(np.max(np.abs(singular - 1.0))))
        volume_changes.append(abs(float(np.linalg.det(transform)) - 1.0))
    return world_proxy, {
        "orthogonality_error_max": max(orthogonality, default=0.0),
        "orthogonality_error_mean": float(np.mean(orthogonality)) if orthogonality else 0.0,
        "max_edge_length_change_ratio": max(edge_changes, default=0.0),
        "volume_change_ratio": max(volume_changes, default=0.0),
    }


def run_abd_backend(
    state: Any,
    config: ABDBackendConfig,
    job_dir: str | Path,
    project_root: str | Path | None = None,
) -> ABDRunResult:
    executable = find_abd_executable(config.executable, project_root)
    if executable is None:
        raise ABDBackendUnavailableError(
            "ABD backend selected, but Autodesk abd_sim was not found. Build the official project in Release mode "
            "and set ONESTRING_ABD_EXECUTABLE. No legacy/SAT fallback was used."
        )
    capabilities = probe_abd_capabilities(executable)
    if not capabilities.get("available") or not capabilities.get("stock_headless_scene"):
        raise ABDCapabilityError("The selected executable is not a compatible Autodesk ABD headless runner")
    job = prepare_abd_job(state, config, job_dir)
    command = [
        str(executable), "--ngui", "--scene-path", str(job.scene_path),
        "--output-path", str(job.output_dir), "--output-name", "sim.json",
        "--num-steps", str(int(config.steps)),
    ]
    if int(config.nthreads) > 0:
        command.extend(["--nthreads", str(int(config.nthreads))])
    if config.require_onestring_extension and job.guide_count >= 2:
        if not capabilities.get("onestring_unilateral_string_extension"):
            raise ABDCapabilityError(
                "The official stock ABD executable is available, but it does not implement the required unilateral "
                "OneString guide-length constraint. Build the OneString ABD extension exposing --onestring-manifest; "
                "the simulator will not mislabel stock ABD or legacy SAT as the requested backend."
            )
        command.extend(["--onestring-manifest", str(job.manifest_path)])
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=str(job.job_dir), capture_output=True, text=True,
        timeout=float(config.timeout_seconds), check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise ABDBackendError(
            f"Autodesk ABD failed with exit code {completed.returncode}.\nSTDOUT:\n{completed.stdout[-4000:]}\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )
    result_path = job.output_dir / "sim.json"
    if not result_path.is_file():
        raise ABDBackendError("Autodesk ABD completed without producing sim.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    states = result.get("animation", {}).get("state_sequence", [])
    stats = result.get("stats", {})
    frames: list[np.ndarray] = []
    frame_logs: list[dict[str, Any]] = []
    solver_iterations = stats.get("solver_iterations", [])
    contacts = stats.get("num_contacts", [])
    minimum_distances = stats.get("step_minimum_distances", [])
    onestring_stats = result.get("onestring_stats", {})
    for frame_id, frame in enumerate(states):
        world_proxy, affine_metrics = _frame_affine_metrics(frame, job)
        frames.append(world_proxy)
        frame_logs.append(
            {
                "frame": frame_id,
                "newton_iterations": int(solver_iterations[frame_id - 1]) if frame_id > 0 and frame_id - 1 < len(solver_iterations) else 0,
                "ccd_seconds": onestring_stats.get("ccd_seconds", [None] * len(states))[frame_id] if frame_id < len(onestring_stats.get("ccd_seconds", [])) else None,
                "linear_solve_seconds": onestring_stats.get("linear_solve_seconds", [None] * len(states))[frame_id] if frame_id < len(onestring_stats.get("linear_solve_seconds", [])) else None,
                "active_contacts": int(contacts[frame_id - 1]) if frame_id > 0 and frame_id - 1 < len(contacts) else 0,
                "minimum_contact_distance": float(minimum_distances[frame_id - 1]) if frame_id > 0 and frame_id - 1 < len(minimum_distances) and minimum_distances[frame_id - 1] is not None else None,
                **affine_metrics,
                "string_length": onestring_stats.get("string_length", [None] * len(states))[frame_id] if frame_id < len(onestring_stats.get("string_length", [])) else None,
                "constraint_violation": onestring_stats.get("constraint_violation", [None] * len(states))[frame_id] if frame_id < len(onestring_stats.get("constraint_violation", [])) else None,
            }
        )
    if not frames:
        raise ABDBackendError("Autodesk ABD result contains no animation frames")
    npz_path = job.output_dir / "onestring_abd_frames.npz"
    np.savez_compressed(npz_path, frames=np.asarray(frames), frame_logs=np.asarray(frame_logs, dtype=object))
    gltf_path = job.output_dir / "sim.glb"
    metrics = {
        "physics_backend": "abd",
        "abd_implementation": "Autodesk/affine-body-dynamics external executable",
        "abd_executable": str(executable),
        "abd_capabilities": capabilities,
        "simulation_runtime_seconds": float(elapsed),
        "frame_count": len(frames),
        "final_collision_count": int(frame_logs[-1]["active_contacts"]),
        "max_orthogonality_error": max(float(row["orthogonality_error_max"]) for row in frame_logs),
        "max_edge_length_change_ratio": max(float(row["max_edge_length_change_ratio"]) for row in frame_logs),
        "max_volume_change_ratio": max(float(row["volume_change_ratio"]) for row in frame_logs),
        "legacy_sat_collision_projection_used": False,
        "string_compression_when_slack": 0.0,
        "solver_device_report": onestring_stats.get("device_report", {
            "ccd": "not reported by stock executable",
            "hessian": "not reported by stock executable",
            "newton_solve": "not reported by stock executable",
        }),
    }
    return ABDRunResult(
        frames=frames,
        final_tiles=frames[-1],
        metrics=metrics,
        collision_counts=[int(row["active_contacts"]) for row in frame_logs],
        frame_logs=frame_logs,
        result_json_path=str(result_path),
        gltf_path=str(gltf_path) if gltf_path.is_file() else None,
        npz_path=str(npz_path),
    )
