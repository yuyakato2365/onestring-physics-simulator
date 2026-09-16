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
    use_desk: bool = True
    desk_clearance: float = 5.0e-3
    orthogonality_stiffness: float = 1e9
    minimum_separation_distance: float = 1e-6
    barrier_activation_distance: float = 5e-4
    newton_velocity_tolerance: float = 1e-3
    nthreads: int = 0
    timeout_seconds: float = 600.0
    pull_end_ratio: float = 0.75
    string_stiffness: float = 1.0e6
    string_smoothing_epsilon: float = 1.0e-9
    string_use_exact_hessian: bool = False
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
    initial_layout: str
    collision_skin: float


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
                root / "third_party" / "affine-body-dynamics" / "build-gpu" / "Release" / "abd_sim.exe",
                root / "third_party" / "affine-body-dynamics" / "build-parallel" / "Release" / "abd_sim.exe",
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
            [str(path), "--help"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False
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


def _collision_free_staging_proxy(
    state: Any,
    source_proxy: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, str, float]:
    """Keep the T2D layout and apply only the minimum uniform clearance scale."""
    source = np.asarray(source_proxy, dtype=float)
    tile_count = len(source)
    if tile_count == 0:
        return source.copy(), "empty", 1.0
    source_centers = np.mean(source, axis=1)
    local = source - source_centers[:, None, :]
    layout_center = np.mean(source_centers[:, :2], axis=0)
    relative_centers = source_centers[:, :2] - layout_center[None, :]
    hinge_pairs = {
        tuple(sorted((int(hinge.tile_a), int(hinge.tile_b))))
        for hinge in getattr(getattr(state, "hinge_graph", None), "hinges", [])
        if int(hinge.tile_a) != int(hinge.tile_b)
    }

    def quads_at(scale: float) -> np.ndarray:
        centers = layout_center[None, :] + relative_centers * float(scale)
        return local[:, :4, :2] + centers[:, None, :]

    def pair_has_clearance(a: np.ndarray, b: np.ndarray) -> bool:
        required_gap = 2.0 * max(float(clearance), 0.0)
        for polygon in (a, b):
            for edge_id in range(4):
                edge = polygon[(edge_id + 1) % 4] - polygon[edge_id]
                axis = np.asarray([-edge[1], edge[0]], dtype=float)
                norm = float(np.linalg.norm(axis))
                if norm <= 1e-12:
                    continue
                axis /= norm
                projection_a = a @ axis
                projection_b = b @ axis
                if (
                    float(np.max(projection_a)) + required_gap <= float(np.min(projection_b))
                    or float(np.max(projection_b)) + required_gap <= float(np.min(projection_a))
                ):
                    return True
        return False

    def layout_is_clear(scale: float) -> bool:
        quads = quads_at(scale)
        for tile_a in range(tile_count):
            for tile_b in range(tile_a + 1, tile_count):
                if (tile_a, tile_b) in hinge_pairs:
                    continue
                if not pair_has_clearance(quads[tile_a], quads[tile_b]):
                    return False
        return True

    layout_scale = 1.0
    if not layout_is_clear(layout_scale):
        upper = 1.05
        while upper < 64.0 and not layout_is_clear(upper):
            upper *= 1.35
        if not layout_is_clear(upper):
            raise ABDBackendError(
                "The T2D-relative ABD initial layout contains coincident or persistent "
                "non-hinge panel overlaps that uniform clearance scaling cannot resolve."
            )
        lower = 1.0
        for _ in range(32):
            middle = 0.5 * (lower + upper)
            if layout_is_clear(middle):
                upper = middle
            else:
                lower = middle
        layout_scale = float(upper)

    staged = local.copy()
    staged[:, :, :2] += (
        layout_center[None, :] + relative_centers * layout_scale
    )[:, None, :]
    staged[:, :, 2] += source_centers[:, None, 2]
    layout_name = (
        "t2d_dual_hinge_exact_initial_layout"
        if layout_scale <= 1.0 + 1e-9
        else "t2d_dual_hinge_minimum_uniform_clearance_scale"
    )
    return staged, layout_name, float(layout_scale)


def _hinge_manifest(
    state: Any,
    centers: np.ndarray,
    initial_proxy: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def proxy_vertex_index(hinge: Any, attribute: str) -> int:
        index = int(getattr(hinge, attribute))
        vertex_count = int(initial_proxy.shape[1])
        surface = str(getattr(hinge, "surface", "top")).lower()
        # Canonical Hinge indices are absolute in the eight-vertex proxy:
        # top=0..3 and bottom=4..7. Older external states sometimes stored a
        # bottom-face-local 0..3 index, so retain that unambiguous conversion.
        if surface == "bottom" and 0 <= index < 4 and vertex_count >= 8:
            index += 4
        if index < 0 or index >= vertex_count:
            raise ABDBackendError(
                f"hinge {attribute}={index} is outside the {vertex_count}-vertex "
                f"tile proxy (surface={surface!r})"
            )
        return index

    constraints: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    pair_points: dict[tuple[int, int], list[np.ndarray]] = {}
    for hinge in state.hinge_graph.hinges:
        tile_a, tile_b = int(hinge.tile_a), int(hinge.tile_b)
        if tile_a == tile_b or min(tile_a, tile_b) < 0 or max(tile_a, tile_b) >= len(centers):
            continue
        point_a = initial_proxy[tile_a, proxy_vertex_index(hinge, "local_vertex_a")]
        point_b = initial_proxy[tile_b, proxy_vertex_index(hinge, "local_vertex_b")]
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


def _string_guides(
    state: Any,
    centers: np.ndarray,
    initial_proxy: np.ndarray,
    guide_reference_proxy: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    gap_by_id = {int(gap.id): gap for gap in state.gap_graph.gaps}
    guides: list[dict[str, Any]] = []
    for gap_id in state.string_path.gap_ids:
        gap = gap_by_id.get(int(gap_id))
        if gap is None or not gap.surrounding_tiles:
            continue
        tile_id = int(gap.surrounding_tiles[0])
        if tile_id < 0 or tile_id >= len(centers):
            continue
        centroid = np.asarray(gap.centroid_2d, dtype=float).reshape(-1)
        if centroid.size < 2 or not np.all(np.isfinite(centroid[:2])):
            raise ABDBackendError(f"gap {int(gap_id)} has an invalid centroid_2d")
        # Some pipeline stages retain the lifted z coordinate even in the
        # field named centroid_2d.  Guide ownership is selected in the flat
        # T2D layout, so only x/y participate in this nearest-corner query.
        centroid_xy = centroid[:2]
        tile_top = initial_proxy[tile_id, :4]
        reference_top = tile_top if guide_reference_proxy is None else np.asarray(guide_reference_proxy[tile_id, :4], dtype=float)
        nearest = int(np.argmin(np.linalg.norm(reference_top[:, :2] - centroid_xy[None, :], axis=1)))
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
    target_proxy = np.asarray(state.tiles_3d.vertices, dtype=float)
    # Start from the actual split-spaced T2D Dual-Hinge arrangement. The
    # staging helper preserves its orientations and relative centers, applying
    # only a minimum uniform center scale when non-hinge panels still overlap.
    initial_assembly = state.tiles_2d_dual_hinge
    initial_layout = "t2d_dual_hinge_split_spaced"
    source_initial_proxy = np.asarray(initial_assembly.vertices, dtype=float)
    if source_initial_proxy.shape != target_proxy.shape:
        raise ABDBackendError("T2D initial proxy and T3D compatibility proxy must have the same shape")
    collision_skin = max(
        2.0 * float(config.minimum_separation_distance),
        float(config.barrier_activation_distance),
    )
    initial_proxy, staging_layout, staging_scale = _collision_free_staging_proxy(
        state, source_initial_proxy, collision_skin
    )
    initial_layout = staging_layout
    # ABD simulates the fabricated flat panels.  Their rest/collision geometry
    # must therefore be the exact T2D solids, not recovered T3D solids rotated
    # back by a best-fit transform.  The latter is only approximate when K2D
    # and K3D are not congruent and created hundreds of initial intersections.
    tile_vertices = [tile.copy() for tile in initial_proxy]
    tile_triangles = [_triangles_from_proxy() for _ in range(len(initial_proxy))]
    centers = np.mean(initial_proxy, axis=1)
    proxy_local = initial_proxy - centers[:, None, :]
    bodies: list[dict[str, Any]] = []
    rest_volumes: list[float] = []
    rest_edges: list[np.ndarray] = []
    for tile_id, (vertices_world, triangles) in enumerate(zip(tile_vertices, tile_triangles)):
        local = np.asarray(vertices_world, dtype=float) - centers[tile_id][None, :]
        # Panels that share a design edge are exactly touching in K2D.  IPC's
        # barrier formulation needs a strictly positive starting distance.
        # Apply a small in-plane collision skin while keeping thickness and
        # all hinge/string material points unchanged.
        planar_radius = float(np.max(np.linalg.norm(local[:, :2], axis=1))) if len(local) else 0.0
        if planar_radius > collision_skin:
            local[:, :2] *= (planar_radius - collision_skin) / planar_radius
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
    desk_metadata: dict[str, Any] = {"enabled": False}
    if bool(config.use_desk) and len(initial_proxy):
        desk_top_z = float(np.min(initial_proxy[:, :, 2])) - max(
            float(config.desk_clearance), float(config.minimum_separation_distance)
        )
        desk_points = [
            {
                "body_id": int(tile_id),
                "body_name": f"tile_{tile_id:04d}",
                "material_point": (
                    initial_proxy[tile_id, vertex_id] - centers[tile_id]
                ).tolist(),
            }
            for tile_id in range(len(initial_proxy))
            for vertex_id in range(4, 8)
        ]
        desk_metadata = {
            "enabled": True,
            "model": "smooth_unilateral_support_plane_penalty",
            "top_z": desk_top_z,
            "clearance": float(config.desk_clearance),
            "stiffness": 1.0e6,
            "smoothing_epsilon": max(0.25 * float(config.desk_clearance), 1.0e-6),
            "support_points": desk_points,
            "tangential_friction_model": "none; normal support only",
        }
    linear_constraints, hinge_metadata = _hinge_manifest(state, centers, initial_proxy)
    guides = _string_guides(state, centers, initial_proxy, source_initial_proxy)
    path_points: list[dict[str, Any]] = []
    if guides:
        path_points.append(
            {
                "type": "world_anchor",
                "id": "support",
                "position": list(guides[0]["initial_world_point"]),
            }
        )
        path_points.extend(
            {
                "type": "body_guide",
                "gap_id": int(guide["gap_id"]),
                "body_id": int(guide["body_id"]),
                "body_name": str(guide["body_name"]),
                "material_point": list(guide["material_point"]),
            }
            for guide in guides
        )
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
        "ipc_solver": {
            "velocity_conv_tol": float(config.newton_velocity_tolerance),
            "use_parallel_pcg": True,
            "parallel_pcg_max_iterations": 300,
            "parallel_pcg_tolerance": 1e-5,
        },
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
        "schema": "onestring-abd-bridge-v2",
        "official_scene_path": str(job_dir / "scene.json"),
        "tile_count": len(initial_proxy),
        "simulation_body_count": len(bodies),
        "thickness": float(state.tiles_3d.metrics.get("requested_thickness", state.tiles_3d.metrics.get("tile_thickness", 0.0))),
        "density": float(config.density),
        "mass_model": "Autodesk ABD computes mass from closed rest mesh and density",
        "initial_positions": centers.tolist(),
        "initial_layout": initial_layout,
        "staging_layout_scale": float(staging_scale),
        "rest_collision_geometry": "exact_t2d_thick_panel_proxy",
        "collision_skin": float(collision_skin),
        "desk": desk_metadata,
        "initial_orientations_xyz_degrees": [[0.0, 0.0, 0.0] for _ in bodies],
        "hinges": hinge_metadata,
        "string": {
            "model": "unilateral_total_path_length_constraint",
            "path_points": path_points,
            "guide_points": guides,
            "initial_length": initial_length,
            "stiffness": float(config.string_stiffness),
            "smoothing_epsilon": float(config.string_smoothing_epsilon),
            "use_exact_hessian": bool(config.string_use_exact_hessian),
            "pull_schedule": [
                {"time": 0.0, "command_length": initial_length},
                {"time": duration, "command_length": initial_length * float(config.pull_end_ratio)},
            ],
            "inequality": "L(q) <= L_command(t)",
            "compression_force_when_slack": 0.0,
        },
        "shake_trajectory": {**asdict(config.shake), "target_anchor": "support"},
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
        initial_layout=initial_layout,
        collision_skin=float(collision_skin),
    )


def _frame_affine_metrics(frame: dict[str, Any], job: ABDPreparedJob) -> tuple[np.ndarray, dict[str, float]]:
    bodies = frame.get("rigid_bodies", [])
    tile_count = len(job.body_proxy_local_vertices)
    if len(bodies) < tile_count:
        raise ABDBackendError("ABD result body count does not match exported OneString tiles")
    bodies = bodies[:tile_count]
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
        "--num-steps", str(int(config.steps)), "--log", "warning",
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
    stdout_log = job.job_dir / "abd_stdout.log"
    stderr_log = job.job_dir / "abd_stderr.log"
    try:
        completed = subprocess.run(
            command, cwd=str(job.job_dir), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=float(config.timeout_seconds), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        def _timeout_output(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        timed_stdout = _timeout_output(exc.stdout)
        timed_stderr = _timeout_output(exc.stderr)
        stdout_log.write_text(timed_stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(timed_stderr, encoding="utf-8", errors="replace")
        requested_threads = int(config.nthreads) if int(config.nthreads) > 0 else "ABD automatic"
        raise ABDBackendError(
            f"Autodesk ABD exceeded the {float(config.timeout_seconds):g}-second time limit and was stopped. "
            f"The job requested {int(config.steps)} steps and CPU threads={requested_threads}. "
            "This is a runtime limit, not a geometry failure. Reduce grid size/steps, relax solver settings, "
            "or increase 'ABD timeout (seconds)'. "
            f"Partial logs: {stdout_log}, {stderr_log}"
        ) from None
    elapsed = time.perf_counter() - started
    stdout_log.write_text(completed.stdout or "", encoding="utf-8", errors="replace")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if "initial state contains intersections" in combined_output.lower():
            raise ABDBackendError(
                "Autodesk ABD rejected the initial panel layout because it contains intersections. "
                f"The generated job used initial_layout={job.initial_layout!r} and collision_skin={job.collision_skin:g}. "
                "Reduce the grid/shape complexity or repair the T2D layout before simulation. "
                f"Full logs: {stdout_log}"
            )
        raise ABDBackendError(
            f"Autodesk ABD failed with exit code {completed.returncode}.\nSTDOUT:\n{completed.stdout[-4000:]}\n"
            f"STDERR:\n{completed.stderr[-4000:]}\nFull logs: {stdout_log}, {stderr_log}"
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
                "command_length": onestring_stats.get("command_length", [None] * len(states))[frame_id] if frame_id < len(onestring_stats.get("command_length", [])) else None,
                "constraint_violation": onestring_stats.get("constraint_violation", [None] * len(states))[frame_id] if frame_id < len(onestring_stats.get("constraint_violation", [])) else None,
                "constraint_active": bool(onestring_stats.get("active", [0] * len(states))[frame_id]) if frame_id < len(onestring_stats.get("active", [])) else False,
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
        "abd_initial_layout": job.initial_layout,
        "abd_rest_collision_geometry": "exact_t2d_thick_panel_proxy",
        "abd_collision_skin": float(job.collision_skin),
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
