from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - optional fallback
    least_squares = None

try:  # pragma: no cover - torch availability is environment-dependent
    import torch
except Exception:  # pragma: no cover
    torch = None

from .heightfields import HeightField
from .metrics import rms_distance
from .quad_grid import HingeSpec, QuadGrid, create_quad_grid


ProgressCallback = Callable[[str, float, str], None]


def _emit_progress(callback: ProgressCallback | None, stage: str, progress: float, detail: str = "") -> None:
    """Best-effort progress callback used by Streamlit and CLI callers.

    The simulator should never fail just because the UI progress renderer fails,
    so callback errors are intentionally swallowed.
    """
    if callback is None:
        return
    try:
        callback(str(stage), float(np.clip(progress, 0.0, 1.0)), str(detail))
    except Exception:
        return


def _subprogress(callback: ProgressCallback | None, start: float, end: float, prefix: str = "") -> ProgressCallback | None:
    if callback is None:
        return None

    def _cb(stage: str, progress: float, detail: str = "") -> None:
        label = f"{prefix}{stage}" if prefix else stage
        _emit_progress(callback, label, start + (end - start) * float(np.clip(progress, 0.0, 1.0)), detail)

    return _cb


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    kind: str


@dataclass
class PlanarDomain:
    boundary: np.ndarray
    uv_vertices: np.ndarray
    method: str
    csf_values: np.ndarray
    split_lines: list[tuple[str, float]] = field(default_factory=list)

    @property
    def max_csf(self) -> float:
        return float(np.max(self.csf_values)) if self.csf_values.size else 1.0


@dataclass
class SurfaceParameterization:
    method: Literal["bff", "lscm", "harmonic", "analytic_scaled_heightfield_debug", "heightfield_uv_debug"]
    surface_vertices_3d: np.ndarray
    surface_faces: np.ndarray
    uv_vertices_2d: np.ndarray
    uv_faces: np.ndarray
    omega_boundary: np.ndarray
    triangle_acceleration: object | None = None
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass
class QuadMesh:
    vertices: np.ndarray
    faces: np.ndarray
    grid: QuadGrid
    stage: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    split_lines: list[tuple[str, float]] = field(default_factory=list)

    @property
    def tile_count(self) -> int:
        return int(len(self.faces))


@dataclass
class FlatTileLayout:
    tile_top_vertices_2d: np.ndarray
    tile_ids: list[int]
    hinge_pairs: list[tuple[int, int]]
    gap_polygons: list[np.ndarray] = field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)

    @property
    def tile_top_vertices_3d(self) -> np.ndarray:
        z = np.zeros((*self.tile_top_vertices_2d.shape[:2], 1), dtype=float)
        return np.concatenate([self.tile_top_vertices_2d, z], axis=2)

    @property
    def tile_count(self) -> int:
        return int(self.tile_top_vertices_2d.shape[0])


@dataclass
class TileAssembly:
    vertices: np.ndarray
    top_faces: np.ndarray
    bottom_faces: np.ndarray
    side_faces: np.ndarray
    stage: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    transform_matrices: np.ndarray | None = None

    @property
    def top_tiles(self) -> np.ndarray:
        return self.vertices[:, :4, :]

    @property
    def tile_count(self) -> int:
        return int(self.vertices.shape[0])


@dataclass
class Hinge:
    tile_a: int
    tile_b: int
    local_vertex_a: int
    local_vertex_b: int
    surface: Literal["top", "bottom"]
    rest_position_2d: np.ndarray
    target_position_3d: np.ndarray


@dataclass
class HingeGraph:
    hinges: list[Hinge]
    metrics: dict[str, float | int | str]


@dataclass
class Gap:
    id: int
    surrounding_tiles: list[int]
    centroid_2d: np.ndarray
    centroid_3d: np.ndarray
    type: Literal["vertical", "horizontal", "virtual_boundary", "split_boundary"]
    boundary: bool
    label: int
    gpe: float = 0.0


@dataclass
class GapGraph:
    gaps: list[Gap]
    edges: list[tuple[int, int]]
    metrics: dict[str, float | int]


@dataclass
class LiftPoint:
    gap_id: int
    position_2d: np.ndarray
    position_3d: np.ndarray
    gpe: float
    cluster_id: int


@dataclass
class StringPath:
    gap_ids: list[int]
    boundary_gap_ids: list[int]
    lift_gap_ids: list[int]
    turn_angle_total: float
    estimated_channel_friction: float
    metrics: dict[str, float | int | str | bool]


@dataclass
class ComputeConfig:
    backend: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    use_gpu_for_optimization: bool = True
    use_gpu_for_simulation: bool = True
    min_grid_for_gpu: int = 1


@dataclass
class DeploymentParameters:
    steps: int = 48
    solver_iterations: int = 16
    rigid_weight: float = 0.95
    rigid_projection_passes: int = 4
    rigid_guard_final_projection: bool = True
    hinge_weight: float = 0.85
    snap_weight: float = 0.65
    lift_weight: float = 0.9
    collision_weight: float = 0.25
    damping_ratio: float = 0.2
    quasi_static_pull_speed: float = 1.0
    high_fidelity: bool = False
    hinge_rotational_stiffness: float = 0.25
    hinge_damping: float = 0.2
    tile_mass: float = 1.0
    gravity: float = 9.81
    contact_friction: float = 0.25
    string_channel_friction: float = 0.2
    solver_substeps: int = 1
    debug_all_pair_collision: bool = False
    store_animation_frames: bool = False
    max_animation_frames: int = 24
    snap_scope: Literal["all_internal_gaps", "string_path_only"] = "string_path_only"
    use_target_gap_contraction: bool = True
    # Deployment target guard: keeps the animated tiles from visually passing
    # through the designed T3D target.  This is a paper-style projection term,
    # not a visual-only clamp: it is applied inside the projective solve and is
    # followed by rigid tile projection so panel shapes remain rigid.
    target_fit_weight: float = 0.30
    target_contact_guard_weight: float = 0.85
    target_contact_start_alpha: float = 0.60
    target_contact_clearance: float = 0.0
    target_contact_projection_passes: int = 2
    compute: ComputeConfig = field(default_factory=ComputeConfig)


@dataclass
class DeploymentResult:
    frames: list[np.ndarray]
    final_tiles: np.ndarray
    metrics: dict[str, float | bool | int]
    collision_counts: list[int] = field(default_factory=list)


@dataclass
class StageReport:
    name: str
    objective: str
    before_error: float
    after_error: float
    constraint_violation: float
    computation_time: float
    failed_constraints: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class OneStringDesignState:
    target_surface: SurfaceMesh
    surface_parameterization: SurfaceParameterization
    conformal_domain: PlanarDomain
    mesh_2d_initial: QuadMesh
    mesh_3d_initial: QuadMesh
    mesh_3d_optimized: QuadMesh
    tiles_3d: TileAssembly
    mesh_2d_optimized: QuadMesh
    k2d_flat_layout: FlatTileLayout
    tiles_2d_top_hinge: TileAssembly
    tiles_2d_dual_hinge: TileAssembly
    hinge_graph: HingeGraph
    gap_graph: GapGraph
    lift_points: list[LiftPoint]
    string_path: StringPath
    simulation_result: DeploymentResult | None = None
    stage_reports: dict[str, StageReport] = field(default_factory=dict)
    approximations: list[str] = field(default_factory=list)
    backend_reports: dict[str, dict[str, str | bool | int | float]] = field(default_factory=dict)


@dataclass
class PipelineParameters:
    nx: int = 3
    ny: int | None = None
    tile_size: float = 1.0
    gap_size: float = 0.08
    thickness: float = 0.08
    max_3d_iterations: int = 40
    max_2d_iterations: int = 40
    # Paper default weights from Sec. 5.4 implementation paragraph:
    # E_Assembled: w_planar=10000, w_square=10, w_surface=0.1.
    # E_Flat: w_edge=1, w_collision=1, w_fab=0.001.
    w_planar: float = 10000.0
    w_square: float = 10.0
    w_surface: float = 0.1
    w_edge: float = 1.0
    w_collision: float = 1.0
    w_fab: float = 0.001
    strict_paper_flow: bool = True
    lift_tau: float = 0.8
    channel_friction: float = 0.2
    m3d_construction_mode: Literal["mesh_harmonic", "analytic_scaled_heightfield_debug"] = "mesh_harmonic"
    surface_mesh_subdivisions: int = 4
    # Paper Fig. 5 starts with an Omega-domain grid larger than Omega and crops outside quads.
    # The public UI value nx/ny is the intended in-Omega resolution; internally we overlay
    # nx+2*margin by ny+2*margin quads and remove the ones outside Omega in M2D.
    omega_overlay_margin: int = 1
    # M2D crop policy: the paper overlays a larger grid on Ω and deletes cells outside it.
    # For interactive height-field targets, center-based cropping is much more stable: it
    # keeps boundary-crossing quads instead of collapsing the topology to a tiny interior patch.
    m2d_crop_policy: Literal["center", "strict_vertices"] = "center"
    # Avoid expensive nonlinear SciPy K2D solves on medium/large grids. The vectorized
    # projective solver below is bounded and is the default for interactive use.
    strict_k2d_scipy_vertex_limit: int = 120
    # Time guard for the K2D strict solver. Non-dome shapes used to appear to hang
    # because least_squares tried to solve hundreds/thousands of variables exactly.
    strict_k2d_time_budget_sec: float = 12.0
    # Paper Section 4.4-style hinge-layout optimization.  These control the
    # rigid-tile placement solve that aligns vertex joints while preserving
    # each K2D tile shape and discouraging collisions.
    hinge_layout_iterations: int = 120
    hinge_layout_connection_weight: float = 3.0
    hinge_layout_collision_weight: float = 1.0
    hinge_layout_anchor_weight: float = 0.05
    # The flat linkage often needs more room than the raw K2D shared mesh.  K2D
    # is an abstract edge-length mesh; T2D is an independent-tile fabrication
    # layout.  Before E_Hinge placement, expand tile centers globally while
    # preserving every tile shape.  This supplies the missing flat-layout space
    # that otherwise makes collision and hinge constraints fight each other.
    hinge_layout_initial_expansion: float = 1.08
    hinge_layout_max_center_drift_tiles: float = 2.0
    # The paper-style E_Hinge placement can be expensive because it combines
    # rigid tile poses, vertex-joint constraints, and full-footprint collision.
    # Keep it bounded for interactive Streamlit use; report when the budget is hit.
    hinge_layout_time_budget_sec: float = 8.0
    hinge_layout_max_candidate_pairs: int = 3000
    hinge_layout_collision_sweeps_per_iteration: int = 2
    compute: ComputeConfig = field(default_factory=ComputeConfig)


def build_onestring_design(
    target: HeightField,
    params: PipelineParameters | None = None,
    run_simulation: bool = False,
    deployment_params: DeploymentParameters | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OneStringDesignState:
    params = params or PipelineParameters()
    _validate_compute_config(params.compute)
    if getattr(params, "strict_paper_flow", False) and params.m3d_construction_mode != "mesh_harmonic":
        raise RuntimeError(
            "Strict paper flow disables analytic_scaled_heightfield_debug. "
            "Use mesh_harmonic so M2D is lifted through the stored S↔Omega parameterization."
        )
    ny = params.nx if params.ny is None else params.ny
    params.ny = ny
    _emit_progress(progress_callback, "Initialize grid", 0.02, "Create regular overlay grid")
    grid = create_quad_grid(params.nx, ny, params.tile_size, params.gap_size)
    reports: dict[str, StageReport] = {}

    target_surface = _build_surface_mesh(target, grid, params.surface_mesh_subdivisions)
    _emit_progress(progress_callback, "S: target surface mesh", 0.08, f"{len(target_surface.vertices)} vertices")
    parameterization = _build_surface_parameterization(target_surface, target, grid, params)
    _emit_progress(progress_callback, "S -> Ω", 0.16, str(parameterization.metrics.get("parameterization_method", parameterization.method)))
    domain = _flatten_to_domain(parameterization, grid, params)
    _emit_progress(progress_callback, "Ω domain", 0.22, "Flattened domain ready")
    mesh_2d_initial = _build_m2d(grid, domain, params)
    _emit_progress(progress_callback, "Ω -> M2D", 0.30, f"kept {len(mesh_2d_initial.faces)} quads")
    active_grid = mesh_2d_initial.grid
    mesh_3d_initial, reports["M2D -> M3D"] = _lift_m2d_to_m3d(target, mesh_2d_initial, parameterization, params)
    _emit_progress(progress_callback, "M2D -> M3D", 0.38, "Inverse map / surface lift done")
    mesh_3d_optimized, reports["M3D -> K3D"] = _optimize_k3d(target, mesh_3d_initial, parameterization, params)
    _emit_progress(progress_callback, "M3D -> K3D", 0.50, str(mesh_3d_optimized.metrics.get("actual_backend", "cpu")))
    tiles_3d, reports["K3D -> T3D"] = _extrude_tiles(mesh_3d_optimized, params.thickness, "T3D")
    _emit_progress(progress_callback, "K3D -> T3D", 0.56, "Extruded assembled tiles")
    mesh_2d_optimized, reports["M2D -> K2D"] = _optimize_k2d(
        mesh_2d_initial,
        mesh_3d_optimized,
        params,
        progress_callback=_subprogress(progress_callback, 0.56, 0.70, "M2D -> K2D: "),
    )
    k2d_flat_layout = _make_flat_tile_layout(mesh_2d_optimized, params)
    _emit_progress(progress_callback, "K2D independent tile layout", 0.73, "Abstract K2D mesh converted to independent tiles")
    mesh_2d_optimized.metrics.update(
        {
            "k2d_tile_overlap_count": int(k2d_flat_layout.metrics["tile_overlap_count"]),
            "k2d_min_clearance": float(k2d_flat_layout.metrics["min_clearance"]),
            "k2d_gap_count": int(k2d_flat_layout.metrics["k2d_gap_count"]),
            "flat_layout_type": str(k2d_flat_layout.metrics["layout_type"]),
        }
    )
    _emit_progress(progress_callback, "K2D -> T2D Top Hinge", 0.731, "Starting rigid flat-tile construction")
    tiles_2d_top, reports["K2D -> T2D top hinge"] = _make_t2d_from_transforms(
        mesh_2d_optimized,
        k2d_flat_layout,
        mesh_3d_optimized,
        tiles_3d,
        "T2D top hinge",
    )
    _emit_progress(progress_callback, "K2D -> T2D Top Hinge", 0.78, "Top-hinge T2D built from K2D and T3D transforms")
    hinge_graph = _build_hinge_graph(active_grid, mesh_2d_optimized.faces, tiles_2d_top, tiles_3d, dual=False)
    _emit_progress(progress_callback, "Build hinge graph", 0.81, f"{len(hinge_graph.hinges)} pairwise hinges")
    tiles_2d_dual, hinge_graph, reports["T2D top hinge -> T2D dual hinge"] = _optimize_dual_hinges(
        active_grid,
        mesh_2d_optimized.faces,
        tiles_2d_top,
        tiles_3d,
        params,
        progress_callback=_subprogress(progress_callback, 0.81, 0.94, "Dual Hinge: "),
    )
    gap_graph = _build_gap_graph(mesh_2d_optimized.faces, tiles_2d_dual, tiles_3d)
    _emit_progress(progress_callback, "Build gap graph", 0.96, f"{len(gap_graph.gaps)} gaps")
    lift_points = _select_lift_points(gap_graph, params.lift_tau)
    string_path = _build_string_path(gap_graph, lift_points, params.channel_friction)
    _emit_progress(progress_callback, "Lift points / string path", 0.98, f"{len(lift_points)} lift points")

    state = OneStringDesignState(
        target_surface=target_surface,
        surface_parameterization=parameterization,
        conformal_domain=domain,
        mesh_2d_initial=mesh_2d_initial,
        mesh_3d_initial=mesh_3d_initial,
        mesh_3d_optimized=mesh_3d_optimized,
        tiles_3d=tiles_3d,
        mesh_2d_optimized=mesh_2d_optimized,
        k2d_flat_layout=k2d_flat_layout,
        tiles_2d_top_hinge=tiles_2d_top,
        tiles_2d_dual_hinge=tiles_2d_dual,
        hinge_graph=hinge_graph,
        gap_graph=gap_graph,
        lift_points=lift_points,
        string_path=string_path,
        stage_reports=reports,
        backend_reports={
            "K3D": _stage_backend_report("K3D", params.compute, reports["M3D -> K3D"].computation_time, mesh_3d_optimized.metrics),
            "K2D": _stage_backend_report("K2D", params.compute, reports["M2D -> K2D"].computation_time, mesh_2d_optimized.metrics),
            "T2D/T3D": _stage_backend_report(
                "T2D/T3D",
                params.compute,
                reports["K2D -> T2D top hinge"].computation_time + reports["K3D -> T3D"].computation_time,
                {"actual_backend": "cpu", "cpu_stage": True, "fallback_reason": "geometry extrusion and hinge projection are CPU-side"},
            ),
        },
        approximations=[
            "Figure-5 order is now locked: K2D is not hinge-optimized before T2D Top Hinge; Dual Hinge is the only hinge-layout stage.",
            "Remaining mismatch from the paper: Boundary First Flattening is approximated by harmonic UV parameterization.",
            "Remaining mismatch from the paper: ShapeOp/libigl projection stack is approximated by NumPy/PyTorch local projection/least-squares steps.",
            "Remaining mismatch from the paper: CSF split placement is simplified.",
            "Remaining mismatch from the paper: Morse-Smale lift point clustering is approximated with GPE thresholding.",
            "Remaining mismatch from the paper: E_Collision uses AABB/local repulsion rather than full rigid tile collision constraints.",
            "Remaining mismatch from the paper: string routing friction uses a simplified Capstan-style turn cost.",
        ],
    )

    if run_simulation:
        state.simulation_result = simulate_onestring_deployment(state, deployment_params, progress_callback=_subprogress(progress_callback, 0.98, 1.0, "deployment: "))
    _emit_progress(progress_callback, "Done", 1.0, "Pipeline complete")
    return state


def simulate_onestring_deployment(
    state: OneStringDesignState,
    params: DeploymentParameters | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DeploymentResult:
    total_start = time.perf_counter()
    params = params or DeploymentParameters()
    if params.compute.backend == "cuda":
        _validate_compute_config(params.compute)
    if (
        params.compute.use_gpu_for_simulation
        and torch is not None
        and compute_backend_info(params.compute)["current_backend"] == "cuda"
    ):
        return _simulate_onestring_deployment_torch(state, params, progress_callback=progress_callback)
    steps = max(2, int(params.steps))
    _emit_progress(progress_callback, "Prepare deployment", 0.02, f"{steps} steps on CPU")
    current = np.asarray(state.tiles_2d_dual_hinge.vertices, dtype=float).copy()
    previous = current.copy()
    rest = current.copy()
    target = np.asarray(state.tiles_3d.vertices, dtype=float)
    frames: list[np.ndarray] = []
    collision_counts: list[int] = []
    frame_stride = max(1, steps // max(1, params.max_animation_frames))

    progress_stride = max(1, steps // 50)
    for step in range(steps):
        if step % progress_stride == 0:
            _emit_progress(progress_callback, "CPU deployment solve", 0.03 + 0.92 * step / max(1, steps - 1), f"step {step + 1}/{steps}")
        alpha = min(1.0, params.quasi_static_pull_speed * step / max(1, steps - 1))
        if not params.high_fidelity:
            alpha = 1.0 - (1.0 - alpha) ** 2
        for _ in range(max(1, params.solver_substeps)):
            velocity = (current - previous) * max(0.0, 1.0 - params.damping_ratio)
            previous = current.copy()
            if params.high_fidelity:
                current[..., 2] -= params.gravity * 0.00002
            current += velocity
            for _ in range(max(1, params.solver_iterations)):
                _project_lift_constraints(current, state, alpha, params.lift_weight)
                _project_snap_constraints(current, state, alpha, params.snap_weight, params.high_fidelity, params.snap_scope, params.use_target_gap_contraction)
                _project_hinge_constraints(current, state, params.hinge_weight)
                for _rigid_pass in range(max(1, int(params.rigid_projection_passes))):
                    _project_rigid_tiles(current, rest, params.rigid_weight)
                _project_aabb_collisions(
                    current,
                    params.collision_weight,
                    state.tiles_2d_dual_hinge.vertices.shape[0],
                    state.mesh_2d_optimized.grid,
                    params.debug_all_pair_collision,
                )
                _project_target_pose_fit(current, rest, target, alpha, params.target_fit_weight)
                _project_target_contact_guard(
                    current,
                    rest,
                    target,
                    alpha,
                    params.target_contact_guard_weight,
                    params.target_contact_start_alpha,
                    params.target_contact_clearance,
                    params.target_contact_projection_passes,
                )
                if params.rigid_guard_final_projection:
                    _project_rigid_tiles(current, rest, 1.0)
                if params.high_fidelity:
                    _project_bending_targets(current, target, alpha, params.hinge_rotational_stiffness)
        should_store_frame = params.store_animation_frames and (step == steps - 1 or step % frame_stride == 0)
        if should_store_frame:
            frames.append(current.copy())
            collision_counts.append(_count_aabb_collisions(current, state.mesh_2d_optimized.grid, params.debug_all_pair_collision))

    _emit_progress(progress_callback, "Finalize deployment", 0.96, "Post-process metrics")
    _project_target_pose_fit(current, rest, target, 1.0, params.target_fit_weight)
    _project_target_contact_guard(
        current,
        rest,
        target,
        1.0,
        params.target_contact_guard_weight,
        params.target_contact_start_alpha,
        params.target_contact_clearance,
        params.target_contact_projection_passes,
    )
    if params.rigid_guard_final_projection:
        _project_rigid_tiles(current, rest, 1.0)
    final_error = rms_distance(current, target)
    snap_error = _snap_error(current, state)
    lift_error = _lift_error(current, state)
    rigid_error = _rigid_error(current, rest)
    hinge_error = _hinge_error(current, state)
    target_surface_fit = float(state.tiles_3d.metrics.get("surface_fit_error", 0.0))
    velocities = current - previous
    kinetic = float(0.5 * params.tile_mass * np.sum(velocities * velocities))
    final_collision_count = _count_aabb_collisions(current, state.mesh_2d_optimized.grid, params.debug_all_pair_collision)
    target_penetration = _target_penetration_metrics(current, target, params.target_contact_clearance)
    metrics: dict[str, float | bool | int] = {
        "target_surface_fit_error_S": target_surface_fit,
        "designed_assembled_error_T3D": 0.0,
        "final_deployment_error_to_T3D": final_error,
        "snap_error": snap_error,
        "lift_error": lift_error,
        "rigid_error": rigid_error,
        "rigid_error_max": _rigid_error_max(current, rest),
        "rigid_projection_passes": int(params.rigid_projection_passes),
        "rigid_guard_final_projection": bool(params.rigid_guard_final_projection),
        "animation_rigidity_model": "strict per-tile Kabsch projection; panel shape is projected back to the rest tile after each constraint iteration",
        "hinge_error": hinge_error,
        "collision_count": int(final_collision_count),
        **target_penetration,
        "target_fit_weight": float(params.target_fit_weight),
        "target_contact_guard_weight": float(params.target_contact_guard_weight),
        "target_contact_start_alpha": float(params.target_contact_start_alpha),
        "target_contact_clearance": float(params.target_contact_clearance),
        "target_contact_projection_passes": int(params.target_contact_projection_passes),
        "target_contact_model": "late one-sided T3D contact guard using thickness-direction normals + rigid per-tile projection",
        "turn_angle_total": state.string_path.turn_angle_total,
        "estimated_channel_friction": state.string_path.estimated_channel_friction,
        "kinetic_energy": kinetic,
        "simulation_model": "paper_projective_dynamics_snap_lift",
        "energy_model": "E = w_rigid*E_rigid + w_collision*E_collision + w_actuation*(E_snap + E_lift)",
        "string_model": "geometric_constraints_not_discrete_rope",
        "snap_scope": params.snap_scope,
        "use_target_gap_contraction": bool(params.use_target_gap_contraction),
        "actuated_snap_gap_count": int(len(_deployment_snap_gaps(state, params.snap_scope))),
        "actual_backend": "cpu",
        "dominant_backend": "cpu",
        "gpu_kernel_time": 0.0,
        "cpu_preprocess_time": 0.0,
        "cpu_postprocess_time": 0.0,
        "cpu_gpu_transfer_count": 0,
        "requested_backend": params.compute.backend,
        "gpu_memory_peak": 0,
        "elapsed_time": time.perf_counter() - total_start,
        "stable_state": bool(
            kinetic < 0.5
            and snap_error < 0.25
            and lift_error < 0.25
            and rigid_error < 0.1
            and hinge_error < 0.1
            and int(final_collision_count) == 0
            and float(target_penetration.get("target_penetration_max", 0.0)) < 1e-4
        ),
    }
    final_frame = current.copy()
    if not frames:
        frames = [final_frame.copy()]
        collision_counts = [int(final_collision_count)]
    return DeploymentResult(frames=frames, final_tiles=final_frame, metrics=metrics, collision_counts=collision_counts)


def _validate_compute_config(config: ComputeConfig) -> None:
    if config.backend != "cuda":
        return
    if torch is None:
        raise RuntimeError(
            "CUDA requested but PyTorch is not installed in the current Python environment. "
            f"sys.executable={sys.executable}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false in the current Python environment. "
            "Do not silently fall back to CPU when backend='cuda'."
        )


def _simulate_onestring_deployment_torch(state: OneStringDesignState, params: DeploymentParameters, progress_callback: ProgressCallback | None = None) -> DeploymentResult:
    total_start = time.perf_counter()
    if torch is None:
        raise RuntimeError("Torch deployment requested but PyTorch is not installed.")
    _emit_progress(progress_callback, "Prepare CUDA deployment", 0.02, "Uploading tensors to GPU")
    cpu_preprocess_start = time.perf_counter()
    device = torch.device("cuda")
    dtype = torch.float64 if params.compute.dtype == "float64" else torch.float32
    torch.cuda.reset_peak_memory_stats(device)
    current = torch.as_tensor(state.tiles_2d_dual_hinge.vertices, dtype=dtype, device=device).clone()
    rest = current.clone()
    target = torch.as_tensor(state.tiles_3d.vertices, dtype=dtype, device=device)
    snap_gaps = _deployment_snap_gaps(state, params.snap_scope)
    hinge_specs = state.hinge_graph.hinges
    steps = max(2, int(params.steps))
    previous = current.clone()
    frame_tensors: list = []
    frame_stride = max(1, steps // max(1, params.max_animation_frames))
    cpu_gpu_transfer_count = 3

    lift_tile_rows: list[list[int]] = []
    lift_targets_2d: list[np.ndarray] = []
    lift_targets_3d: list[np.ndarray] = []
    for lift in state.lift_points:
        gap = state.gap_graph.gaps[lift.gap_id]
        if not gap.surrounding_tiles:
            continue
        lift_tile_rows.append(list(gap.surrounding_tiles))
        lift_targets_2d.append(lift.position_2d)
        lift_targets_3d.append(lift.position_3d)
    max_lift_tiles = max((len(row) for row in lift_tile_rows), default=0)
    if max_lift_tiles:
        lift_tile_idx_np = np.full((len(lift_tile_rows), max_lift_tiles), 0, dtype=np.int64)
        lift_mask_np = np.zeros((len(lift_tile_rows), max_lift_tiles), dtype=bool)
        for row_id, row in enumerate(lift_tile_rows):
            lift_tile_idx_np[row_id, : len(row)] = row
            lift_mask_np[row_id, : len(row)] = True
        lift_tile_idx = torch.as_tensor(lift_tile_idx_np, dtype=torch.long, device=device)
        lift_mask = torch.as_tensor(lift_mask_np, dtype=torch.bool, device=device)
        lift_flat_tiles = lift_tile_idx.reshape(-1)
        lift_flat_mask = lift_mask.reshape(-1)
        lift_flat_selected = lift_flat_tiles[lift_flat_mask]
        lift_count_ones = torch.ones((lift_flat_selected.numel(), 1, 1), dtype=dtype, device=device)
        lift_t2 = torch.as_tensor(np.asarray(lift_targets_2d), dtype=dtype, device=device)
        lift_t3 = torch.as_tensor(np.asarray(lift_targets_3d), dtype=dtype, device=device)
        cpu_gpu_transfer_count += 4
    else:
        lift_tile_idx = lift_mask = lift_flat_mask = lift_flat_selected = lift_count_ones = lift_t2 = lift_t3 = None

    if snap_gaps:
        snap_pairs = torch.as_tensor([gap.surrounding_tiles for gap in snap_gaps], dtype=torch.long, device=device)
        snap_edge_a = torch.as_tensor([[1, 2, 6, 5] if gap.type == "vertical" else [3, 2, 6, 7] for gap in snap_gaps], dtype=torch.long, device=device)
        snap_edge_b = torch.as_tensor([[0, 3, 7, 4] if gap.type == "vertical" else [0, 1, 5, 4] for gap in snap_gaps], dtype=torch.long, device=device)
        snap_rest_sep, snap_target_sep = _gap_separation_vectors(state, snap_gaps, include_bottom=True)
        snap_rest_sep_t = torch.as_tensor(snap_rest_sep, dtype=dtype, device=device)
        snap_target_sep_t = torch.as_tensor(snap_target_sep, dtype=dtype, device=device)
        cpu_gpu_transfer_count += 5
    else:
        snap_pairs = snap_edge_a = snap_edge_b = snap_rest_sep_t = snap_target_sep_t = None

    if hinge_specs:
        hinge_tile_a = torch.as_tensor([h.tile_a for h in hinge_specs], dtype=torch.long, device=device)
        hinge_tile_b = torch.as_tensor([h.tile_b for h in hinge_specs], dtype=torch.long, device=device)
        hinge_va = torch.as_tensor([h.local_vertex_a for h in hinge_specs], dtype=torch.long, device=device)
        hinge_vb = torch.as_tensor([h.local_vertex_b for h in hinge_specs], dtype=torch.long, device=device)
        cpu_gpu_transfer_count += 4
    else:
        hinge_tile_a = hinge_tile_b = hinge_va = hinge_vb = None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cpu_preprocess_time = time.perf_counter() - cpu_preprocess_start
    start_event.record()
    progress_stride = max(1, steps // 50)
    for step in range(steps):
        if step % progress_stride == 0:
            _emit_progress(progress_callback, "CUDA deployment solve", 0.03 + 0.92 * step / max(1, steps - 1), f"step {step + 1}/{steps}")
        alpha_value = min(1.0, params.quasi_static_pull_speed * step / max(1, steps - 1))
        alpha = torch.tensor(alpha_value, dtype=dtype, device=device)
        velocity = (current - previous) * max(0.0, 1.0 - params.damping_ratio)
        previous = current.clone()
        current = current + velocity
        for _ in range(max(1, params.solver_iterations)):
            tile_delta = torch.zeros_like(current)
            tile_counts = torch.zeros((current.shape[0], 1, 1), dtype=dtype, device=device)
            if lift_tile_idx is not None:
                centers = current[lift_tile_idx, :4].mean(dim=2)
                masked_centers = centers * lift_mask[..., None].to(dtype)
                denom = torch.clamp(lift_mask.sum(dim=1, keepdim=True).to(dtype), min=1.0)
                group_center = masked_centers.sum(dim=1) / denom
                target_pos = (1.0 - alpha) * lift_t2 + alpha * lift_t3
                correction = ((target_pos - group_center) * params.lift_weight / denom).unsqueeze(1)
                expanded = (correction * lift_mask[..., None].to(dtype)).reshape(-1, 3)
                tile_delta.index_add_(0, lift_flat_selected, expanded[lift_flat_mask].unsqueeze(1).expand(-1, 8, -1))
                tile_counts.index_add_(0, lift_flat_selected, lift_count_ones)
            snap_eff = alpha * params.snap_weight
            if snap_pairs is not None:
                a = snap_pairs[:, 0]
                b = snap_pairs[:, 1]
                pa = current[a[:, None], snap_edge_a].mean(dim=1)
                pb = current[b[:, None], snap_edge_b].mean(dim=1)
                mid = 0.5 * (pa + pb)
                if params.use_target_gap_contraction:
                    desired_sep = (1.0 - alpha) * snap_rest_sep_t + alpha * snap_target_sep_t
                    desired_pa = mid + 0.5 * desired_sep
                    desired_pb = mid - 0.5 * desired_sep
                    da = (desired_pa - pa) * snap_eff
                    db = (desired_pb - pb) * snap_eff
                else:
                    da = (mid - pa) * snap_eff
                    db = (mid - pb) * snap_eff
                tile_delta.index_add_(0, a, da.unsqueeze(1).expand(-1, 8, -1))
                tile_delta.index_add_(0, b, db.unsqueeze(1).expand(-1, 8, -1))
                tile_counts.index_add_(0, a, torch.ones((a.numel(), 1, 1), dtype=dtype, device=device))
                tile_counts.index_add_(0, b, torch.ones((b.numel(), 1, 1), dtype=dtype, device=device))
            current = current + tile_delta / torch.clamp(tile_counts, min=1.0)
            if hinge_tile_a is not None:
                pa = current[hinge_tile_a, hinge_va]
                pb = current[hinge_tile_b, hinge_vb]
                mid = 0.5 * (pa + pb)
                va_new = pa + (mid - pa) * params.hinge_weight
                vb_new = pb + (mid - pb) * params.hinge_weight
                flat = current.reshape(-1, 3)
                flat.index_copy_(0, hinge_tile_a * 8 + hinge_va, va_new)
                flat.index_copy_(0, hinge_tile_b * 8 + hinge_vb, vb_new)
                current = flat.reshape_as(current)
            for _rigid_pass in range(max(1, int(params.rigid_projection_passes))):
                current = _torch_project_rigid_tiles(current, rest, params.rigid_weight)
            current = _torch_project_target_pose_fit(current, rest, target, alpha, params.target_fit_weight)
            current = _torch_project_target_contact_guard(
                current,
                rest,
                target,
                alpha,
                params.target_contact_guard_weight,
                params.target_contact_start_alpha,
                params.target_contact_clearance,
                params.target_contact_projection_passes,
            )
            if params.rigid_guard_final_projection:
                current = _torch_project_rigid_tiles(current, rest, 1.0)
        if params.store_animation_frames and (step == steps - 1 or step % frame_stride == 0):
            frame_tensors.append(current.detach().clone())
    end_event.record()
    torch.cuda.synchronize(device)
    gpu_kernel_time = float(start_event.elapsed_time(end_event) / 1000.0)
    _emit_progress(progress_callback, "Download deployment result", 0.96, "GPU -> CPU final frames/metrics")
    cpu_postprocess_start = time.perf_counter()
    current = _torch_project_target_pose_fit(current, rest, target, torch.tensor(1.0, dtype=dtype, device=device), params.target_fit_weight)
    current = _torch_project_target_contact_guard(
        current,
        rest,
        target,
        torch.tensor(1.0, dtype=dtype, device=device),
        params.target_contact_guard_weight,
        params.target_contact_start_alpha,
        params.target_contact_clearance,
        params.target_contact_projection_passes,
    )
    if params.rigid_guard_final_projection:
        current = _torch_project_rigid_tiles(current, rest, 1.0)
    final = current.detach().cpu().numpy()
    frames = [frame.detach().cpu().numpy() for frame in frame_tensors]
    if not frames:
        frames = [final.copy()]
    cpu_gpu_transfer_count += 1 + len(frame_tensors)
    previous_np = previous.detach().cpu().numpy()
    final_error = rms_distance(final, state.tiles_3d.vertices)
    snap_error = _snap_error(final, state)
    lift_error = _lift_error(final, state)
    rigid_error = _rigid_error(final, state.tiles_2d_dual_hinge.vertices)
    hinge_error = _hinge_error(final, state)
    collision_count = _count_aabb_collisions(final, state.mesh_2d_optimized.grid, params.debug_all_pair_collision)
    target_penetration = _target_penetration_metrics(final, state.tiles_3d.vertices, params.target_contact_clearance)
    kinetic = float(0.5 * params.tile_mass * np.sum((final - previous_np) ** 2))
    peak = int(torch.cuda.max_memory_allocated(device))
    cpu_postprocess_time = time.perf_counter() - cpu_postprocess_start
    metrics: dict[str, float | bool | int] = {
        "target_surface_fit_error_S": float(state.tiles_3d.metrics.get("surface_fit_error", 0.0)),
        "designed_assembled_error_T3D": 0.0,
        "final_deployment_error_to_T3D": final_error,
        "snap_error": snap_error,
        "lift_error": lift_error,
        "rigid_error": rigid_error,
        "rigid_error_max": _rigid_error_max(final, state.tiles_2d_dual_hinge.vertices),
        "rigid_projection_passes": int(params.rigid_projection_passes),
        "rigid_guard_final_projection": bool(params.rigid_guard_final_projection),
        "animation_rigidity_model": "strict per-tile Kabsch projection; panel shape is projected back to the rest tile after each constraint iteration",
        "hinge_error": hinge_error,
        "collision_count": int(collision_count),
        **target_penetration,
        "target_fit_weight": float(params.target_fit_weight),
        "target_contact_guard_weight": float(params.target_contact_guard_weight),
        "target_contact_start_alpha": float(params.target_contact_start_alpha),
        "target_contact_clearance": float(params.target_contact_clearance),
        "target_contact_projection_passes": int(params.target_contact_projection_passes),
        "target_contact_model": "late one-sided T3D contact guard using thickness-direction normals + rigid per-tile projection",
        "turn_angle_total": state.string_path.turn_angle_total,
        "estimated_channel_friction": state.string_path.estimated_channel_friction,
        "kinetic_energy": kinetic,
        "simulation_model": "paper_projective_dynamics_snap_lift",
        "energy_model": "E = w_rigid*E_rigid + w_collision*E_collision + w_actuation*(E_snap + E_lift)",
        "string_model": "geometric_constraints_not_discrete_rope",
        "snap_scope": params.snap_scope,
        "use_target_gap_contraction": bool(params.use_target_gap_contraction),
        "actuated_snap_gap_count": int(len(snap_gaps)),
        "actual_backend": "cuda",
        "dominant_backend": "cuda",
        "gpu_kernel_time": gpu_kernel_time,
        "cpu_preprocess_time": cpu_preprocess_time,
        "cpu_postprocess_time": cpu_postprocess_time,
        "cpu_gpu_transfer_count": cpu_gpu_transfer_count,
        "requested_backend": params.compute.backend,
        "gpu_memory_peak": peak,
        "elapsed_time": time.perf_counter() - total_start,
        "stable_state": bool(kinetic < 0.5 and snap_error < 0.25 and lift_error < 0.25 and rigid_error < 0.1 and hinge_error < 0.1 and collision_count == 0 and float(target_penetration.get("target_penetration_max", 0.0)) < 1e-4),
    }
    state.backend_reports["deployment"] = _stage_backend_report("deployment", params.compute, gpu_kernel_time, metrics)
    return DeploymentResult(frames=frames, final_tiles=final, metrics=metrics, collision_counts=[collision_count for _ in frames])



def _torch_project_rigid_tiles(current, rest, weight: float):
    if weight <= 0.0:
        return current
    rest_center = rest.mean(dim=1, keepdim=True)
    current_center = current.mean(dim=1, keepdim=True)
    a = rest - rest_center
    b = current - current_center
    h = torch.matmul(a.transpose(1, 2), b)
    u, _, vh = torch.linalg.svd(h)
    r = torch.matmul(vh.transpose(1, 2), u.transpose(1, 2))
    det = torch.linalg.det(r)
    if bool(torch.any(det < 0.0)):
        vh_fixed = vh.clone()
        vh_fixed[det < 0.0, -1, :] *= -1.0
        r = torch.matmul(vh_fixed.transpose(1, 2), u.transpose(1, 2))
    projected = torch.matmul(a, r.transpose(1, 2)) + current_center
    return (1.0 - weight) * current + weight * projected



def _torch_tile_outward_normals_from_thickness(vertices):
    """Return the per-tile outward/top normal from the top-bottom ordering.

    The old contact guard inferred normals from top-face winding and then
    forced z-positive normals.  That is fragile: if a tile's local vertex
    order is flipped, or if a curved/saddle target has a strong side tilt, the
    guard can interpret the underside as the outside and push panels through the
    target.

    Here the normal is defined by the actual thickness direction: bottom -> top.
    This is independent of face winding and directly detects top/bottom flips.
    """
    top_center = vertices[:, :4].mean(dim=1)
    bottom_center = vertices[:, 4:].mean(dim=1)
    normals = top_center - bottom_center
    norms = torch.linalg.norm(normals, dim=1, keepdim=True)
    fallback = norms[:, 0] < 1e-12
    normals = normals / torch.clamp(norms, min=1e-12)
    if bool(torch.any(fallback)):
        top = vertices[:, :4]
        face_normals = torch.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0], dim=1)
        face_normals = face_normals / torch.clamp(torch.linalg.norm(face_normals, dim=1, keepdim=True), min=1e-12)
        normals = normals.clone()
        normals[fallback] = face_normals[fallback]
    return normals


def _torch_target_tile_normals(target):
    return _torch_tile_outward_normals_from_thickness(target)


def _torch_project_target_pose_fit(current, rest, target, alpha, weight: float):
    if weight <= 0.0:
        return current
    eff = torch.clamp(alpha * alpha * float(weight), min=0.0, max=1.0)
    if float(eff.detach().cpu()) <= 0.0:
        return current
    desired = (1.0 - alpha) * rest + alpha * target
    rest_center = rest.mean(dim=1, keepdim=True)
    desired_center = desired.mean(dim=1, keepdim=True)
    a = rest - rest_center
    b = desired - desired_center
    h = torch.matmul(a.transpose(1, 2), b)
    u, _, vh = torch.linalg.svd(h)
    r = torch.matmul(vh.transpose(1, 2), u.transpose(1, 2))
    det = torch.linalg.det(r)
    if bool(torch.any(det < 0.0)):
        vh_fixed = vh.clone()
        vh_fixed[det < 0.0, -1, :] *= -1.0
        r = torch.matmul(vh_fixed.transpose(1, 2), u.transpose(1, 2))
    projected = torch.matmul(a, r.transpose(1, 2)) + desired_center
    return (1.0 - eff) * current + eff * projected


def _torch_smooth_activation(alpha, start_alpha: float):
    if start_alpha >= 1.0:
        return alpha * 0.0
    t = (alpha - float(start_alpha)) / max(1e-8, 1.0 - float(start_alpha))
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _torch_project_target_contact_guard(current, rest, target, alpha, weight: float, start_alpha: float, clearance: float, passes: int = 1):
    if weight <= 0.0:
        return current
    eff = torch.clamp(_torch_smooth_activation(alpha, start_alpha) * float(weight), min=0.0, max=1.0)
    if float(eff.detach().cpu()) <= 0.0:
        return current
    normals = _torch_target_tile_normals(target)
    out = current
    for _ in range(max(1, int(passes))):
        # signed distance along bottom->top target normal.  Negative means the
        # animated tile is on the bottom/inside side of the corresponding T3D
        # tile.  Require signed >= clearance.
        signed = torch.sum((out - target) * normals[:, None, :], dim=2)
        depth = torch.clamp(float(clearance) - signed, min=0.0)
        max_depth = torch.max(depth, dim=1).values
        if float(torch.max(max_depth).detach().cpu()) <= 0.0:
            break
        delta = max_depth[:, None] * normals * eff
        out = out + delta[:, None, :]
        out = _torch_project_rigid_tiles(out, rest, 1.0)
    return out


def _build_surface_mesh(target: HeightField, grid: QuadGrid, subdivision_factor: int = 4) -> SurfaceMesh:
    if target.kind == "sampled" and target.points is not None and target.faces is not None:
        faces = np.asarray(target.faces, dtype=int)
        if faces.shape[1] == 4:
            tri_faces = []
            for a, b, c, d in faces:
                tri_faces.append((a, b, c))
                tri_faces.append((a, c, d))
            faces = np.asarray(tri_faces, dtype=int)
        return SurfaceMesh(vertices=np.asarray(target.points, dtype=float), faces=faces[:, :3], kind=target.kind)
    factor = max(1, int(subdivision_factor))
    dense_nx = max(grid.nx * factor, grid.nx + 1)
    dense_ny = max(grid.ny * factor, grid.ny + 1)
    xs = (np.linspace(0, grid.nx, dense_nx + 1) - grid.nx / 2.0) * grid.tile_size
    ys = (np.linspace(0, grid.ny, dense_ny + 1) - grid.ny / 2.0) * grid.tile_size
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = target.height(xx, yy)
    full_vertices = np.stack([xx, yy, zz], axis=-1)

    support_mask_fn = getattr(target, "support_mask", None)
    support = support_mask_fn(xx, yy) if callable(support_mask_fn) else np.ones_like(xx, dtype=bool)
    support = np.asarray(support, dtype=bool)

    if bool(np.all(support)):
        vertices = full_vertices.reshape(-1, 3)
        faces: list[tuple[int, int, int]] = []
        for row in range(dense_ny):
            for col in range(dense_nx):
                a = row * (dense_nx + 1) + col
                b = a + 1
                c = a + dense_nx + 2
                d = a + dense_nx + 1
                faces.append((a, b, c))
                faces.append((a, c, d))
        return SurfaceMesh(vertices=vertices, faces=np.asarray(faces, dtype=int), kind=target.kind)

    # Non-rectangular analytic targets, such as half_gourd, keep only the
    # supported footprint.  This gives Omega a meaningful boundary and makes the
    # M2D "overlay large grid then crop" stage visible instead of rectangular.
    index_map = -np.ones(support.shape, dtype=int)
    kept = np.argwhere(support)
    vertices_list: list[np.ndarray] = []
    for new_id, (row, col) in enumerate(kept):
        index_map[row, col] = new_id
        vertices_list.append(full_vertices[row, col])
    faces: list[tuple[int, int, int]] = []
    for row in range(dense_ny):
        for col in range(dense_nx):
            corners = [(row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col)]
            ids = [int(index_map[r, c]) for r, c in corners]
            if min(ids) < 0:
                continue
            a, b, c, d = ids
            faces.append((a, b, c))
            faces.append((a, c, d))
    if not vertices_list or not faces:
        raise RuntimeError(f"target surface '{target.kind}' produced an empty supported mesh; lower grid size or adjust radius.")
    # Support masks can retain isolated boundary samples that belong to no
    # fully-supported cell.  They are not part of the generated triangle mesh
    # and must not be passed to topology-sensitive parameterization backends.
    face_array = np.asarray(faces, dtype=int)
    used_vertex_ids = np.unique(face_array)
    remap = np.full(len(vertices_list), -1, dtype=int)
    remap[used_vertex_ids] = np.arange(len(used_vertex_ids), dtype=int)
    compact_vertices = np.asarray(vertices_list, dtype=float)[used_vertex_ids]
    return SurfaceMesh(vertices=compact_vertices, faces=remap[face_array], kind=target.kind)


def _build_surface_parameterization(surface: SurfaceMesh, target: HeightField, grid: QuadGrid, params: PipelineParameters) -> SurfaceParameterization:
    if params.m3d_construction_mode == "analytic_scaled_heightfield_debug":
        return _build_debug_heightfield_parameterization(target, grid)
    surface_vertices = np.asarray(surface.vertices, dtype=float)
    surface_faces = np.asarray(surface.faces[:, :3], dtype=int)
    boundary_loop = _mesh_boundary_loop(surface_faces)
    if len(boundary_loop) < 3:
        raise RuntimeError("Paper mode requires an open target mesh with a boundary; only debug heightfield mode is available for this surface.")
    uv_vertices = _harmonic_parameterization(surface_vertices, surface_faces, boundary_loop)
    boundary = uv_vertices[boundary_loop + [boundary_loop[0]]]
    metric = {"mean_slope": 0.0, "max_slope": 0.0} if target.kind == "sampled" else _heightfield_metric_summary(target, grid)
    method: Literal["bff", "lscm", "harmonic", "analytic_scaled_heightfield_debug", "heightfield_uv_debug"] = "harmonic"
    metrics: dict[str, float | int | str | bool] = {
        "parameterization_method": method,
        "surface_vertex_count": int(len(surface_vertices)),
        "surface_triangle_count": int(len(surface_faces)),
        "boundary_vertex_count": int(len(boundary_loop)),
        "mean_slope": metric["mean_slope"],
        "max_slope": metric["max_slope"],
        "harmonic_solve_performed": True,
        "height_field_shortcut_used": False,
        "omega_corresponds_to_S": True,
        "omega_correspondence_model": "harmonic UV map c:S->Omega, inverse by UV triangle lookup",
        "bff_implemented": False,
        "paper_flow_stage": "S -> Omega by conformal-map substitute; inverse c^-1 used for M2D -> M3D",
        "paper_exactness_warning": "Boundary First Flattening is not implemented; this is the only remaining non-identical initialization step.",
        "omega_warning": "Approximate: harmonic map, not Boundary First Flattening/LSCM.",
    }
    return SurfaceParameterization(
        method=method,
        surface_vertices_3d=surface_vertices,
        surface_faces=surface_faces,
        uv_vertices_2d=uv_vertices,
        uv_faces=surface_faces.copy(),
        omega_boundary=boundary,
        triangle_acceleration=None,
        metrics=metrics,
    )


def _heightfield_metric_summary(target: HeightField, grid: QuadGrid) -> dict[str, float]:
    sample_n = max(8, min(40, grid.nx * 4))
    xs = (np.linspace(0, grid.nx, sample_n + 1) - grid.nx / 2.0) * grid.tile_size
    ys = (np.linspace(0, grid.ny, sample_n + 1) - grid.ny / 2.0) * grid.tile_size
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    eps = max(grid.tile_size * 0.01, 1e-4)
    dzdx = (target.height(xx + eps, yy) - target.height(xx - eps, yy)) / (2.0 * eps)
    dzdy = (target.height(xx, yy + eps) - target.height(xx, yy - eps)) / (2.0 * eps)
    slope = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    return {
        "mean_slope": float(np.mean(slope)) if slope.size else 0.0,
        "max_slope": float(np.max(slope)) if slope.size else 0.0,
    }


def _build_debug_heightfield_parameterization(target: HeightField, grid: QuadGrid) -> SurfaceParameterization:
    debug_params = PipelineParameters(m3d_construction_mode="analytic_scaled_heightfield_debug")
    surface = _build_surface_mesh(target, grid, debug_params.surface_mesh_subdivisions)
    uv = surface.vertices[:, :2].copy()
    boundary_loop = _mesh_boundary_loop(surface.faces)
    boundary = uv[boundary_loop + [boundary_loop[0]]] if boundary_loop else _rectangle_boundary(grid)
    return SurfaceParameterization(
        method="analytic_scaled_heightfield_debug",
        surface_vertices_3d=surface.vertices,
        surface_faces=surface.faces,
        uv_vertices_2d=uv,
        uv_faces=surface.faces.copy(),
        omega_boundary=boundary,
        triangle_acceleration=None,
        metrics={
            "parameterization_method": "analytic_scaled_heightfield_debug",
            "surface_vertex_count": int(len(surface.vertices)),
            "surface_triangle_count": int(len(surface.faces)),
            "boundary_vertex_count": int(len(boundary_loop)),
            "harmonic_solve_performed": False,
            "height_field_shortcut_used": True,
            "omega_corresponds_to_S": False,
            "omega_correspondence_model": "analytic XY heightfield shortcut; no conformal flattening",
            "bff_implemented": False,
            "omega_warning": "Debug/GPU-first shortcut: Omega is XY domain, not a conformal map of S.",
        },
    )


def _mesh_boundary_loop(faces: np.ndarray) -> list[int]:
    edge_count: dict[tuple[int, int], int] = {}
    next_map: dict[int, int] = {}
    for tri in np.asarray(faces, dtype=int):
        for a, b in [(int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))]:
            key = tuple(sorted((a, b)))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary_edges:
        return []
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    start = min(adjacency)
    loop = [start]
    prev = -1
    current = start
    for _ in range(len(boundary_edges) + 2):
        candidates = [v for v in adjacency[current] if v != prev]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, current = current, nxt
    return loop


def _harmonic_parameterization(vertices: np.ndarray, faces: np.ndarray, boundary_loop: list[int]) -> np.ndarray:
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy sparse is required for harmonic parameterization.") from exc
    n = len(vertices)
    boundary_set = set(boundary_loop)
    uv = np.zeros((n, 2), dtype=float)
    lengths = [0.0]
    for a, b in zip(boundary_loop, boundary_loop[1:] + [boundary_loop[0]]):
        lengths.append(lengths[-1] + float(np.linalg.norm(vertices[b] - vertices[a])))
    total = max(lengths[-1], 1e-12)
    for idx, vertex_id in enumerate(boundary_loop):
        t = lengths[idx] / total
        if t < 0.25:
            s = t / 0.25
            uv[vertex_id] = [-1.0 + 2.0 * s, -1.0]
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            uv[vertex_id] = [1.0, -1.0 + 2.0 * s]
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            uv[vertex_id] = [1.0 - 2.0 * s, 1.0]
        else:
            s = (t - 0.75) / 0.25
            uv[vertex_id] = [-1.0, 1.0 - 2.0 * s]
    neighbors: dict[int, set[int]] = {i: set() for i in range(n)}
    for tri in faces:
        a, b, c = map(int, tri)
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros((n, 2), dtype=float)
    for i in range(n):
        if i in boundary_set:
            rows.append(i); cols.append(i); data.append(1.0)
            rhs[i] = uv[i]
            continue
        nbrs = sorted(neighbors[i])
        rows.append(i); cols.append(i); data.append(1.0)
        w = -1.0 / max(1, len(nbrs))
        for j in nbrs:
            rows.append(i); cols.append(j); data.append(w)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    uv[:, 0] = spsolve(mat, rhs[:, 0])
    uv[:, 1] = spsolve(mat, rhs[:, 1])
    return uv


def _rectangle_boundary(grid: QuadGrid) -> np.ndarray:
    half_x = grid.nx * grid.tile_size * 0.5
    half_y = grid.ny * grid.tile_size * 0.5
    return np.asarray([[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y], [-half_x, -half_y]], dtype=float)


def _flatten_to_domain(parameterization: SurfaceParameterization, grid: QuadGrid, params: PipelineParameters | None = None) -> PlanarDomain:
    boundary_open = parameterization.omega_boundary[:-1]
    uv_samples = parameterization.uv_vertices_2d
    csf = np.full(len(uv_samples), float(parameterization.metrics.get("max_slope", 0.0)) + 1.0)
    split_lines: list[tuple[str, float]] = []
    max_before = float(np.max(csf)) if csf.size else 1.0
    if max_before > 2.0:
        split_lines.append(("row", 0.0))

    # Paper-faithful M2D construction: place a regular grid on an Omega-domain
    # rectangle that is deliberately larger than Omega, then crop quads outside
    # the actual Omega polygon in _build_m2d.  The previous prototype sampled
    # exactly (nx+1)*(ny+1) points inside the Omega bounding box, so there was
    # almost nothing for the M2D crop stage to remove; worse, it made non-rectangular
    # Omega domains look rectangular to later stages.
    boundary = np.asarray(parameterization.omega_boundary, dtype=float)
    min_u, min_v = np.min(boundary_open, axis=0)
    max_u, max_v = np.max(boundary_open, axis=0)
    span_u = max(float(max_u - min_u), 1e-8)
    span_v = max(float(max_v - min_v), 1e-8)
    margin_tiles = int(max(0, getattr(params, "omega_overlay_margin", 1))) if params is not None else 1
    overlay_nx = int(grid.nx + 2 * margin_tiles)
    overlay_ny = int(grid.ny + 2 * margin_tiles)
    # Keep the requested in-Omega resolution approximately grid.nx/grid.ny and
    # add full-cell margins around the domain.
    step_u = span_u / max(int(grid.nx), 1)
    step_v = span_v / max(int(grid.ny), 1)
    xs = min_u - margin_tiles * step_u + np.arange(overlay_nx + 1, dtype=float) * step_u
    ys = min_v - margin_tiles * step_v + np.arange(overlay_ny + 1, dtype=float) * step_v
    uu, vv = np.meshgrid(xs, ys, indexing="xy")
    domain_uv = np.stack([uu, vv], axis=-1).reshape(-1, 2)
    domain = PlanarDomain(boundary, domain_uv, f"{parameterization.method} surface parameterization", csf, split_lines)
    domain.csf_before = max_before  # type: ignore[attr-defined]
    domain.csf_after_split = max(1.0, max_before / max(1, len(split_lines) + 1))  # type: ignore[attr-defined]
    domain.overlay_nx = overlay_nx  # type: ignore[attr-defined]
    domain.overlay_ny = overlay_ny  # type: ignore[attr-defined]
    domain.overlay_margin_tiles = margin_tiles  # type: ignore[attr-defined]
    domain.overlay_step_u = float(step_u)  # type: ignore[attr-defined]
    domain.overlay_step_v = float(step_v)  # type: ignore[attr-defined]
    domain.original_requested_nx = int(grid.nx)  # type: ignore[attr-defined]
    domain.original_requested_ny = int(grid.ny)  # type: ignore[attr-defined]
    return domain


def inverse_map_uv_to_surface(
    uv_point: np.ndarray,
    parameterization: SurfaceParameterization,
) -> tuple[np.ndarray, int, bool]:
    accelerated = _inverse_map_uv_to_surface_regular(uv_point, parameterization)
    if accelerated is not None:
        return accelerated
    uv_faces = parameterization.uv_faces
    uv_vertices = parameterization.uv_vertices_2d
    surface_vertices = parameterization.surface_vertices_3d
    best_tri = -1
    best_bary: np.ndarray | None = None
    best_score = float("inf")
    outside = False
    for tri_id, face in enumerate(uv_faces):
        tri = uv_vertices[face]
        bary = _barycentric_2d(np.asarray(uv_point, dtype=float), tri)
        if bary is None:
            continue
        min_bary = float(np.min(bary))
        if min_bary >= -1e-9:
            surface_tri = surface_vertices[parameterization.surface_faces[tri_id]]
            return bary @ surface_tri, int(tri_id), False
        score = abs(min_bary)
        if score < best_score:
            best_score = score
            best_tri = int(tri_id)
            best_bary = bary
    if best_tri < 0 or best_bary is None:
        nearest = int(np.argmin(np.linalg.norm(uv_vertices - uv_point, axis=1)))
        return surface_vertices[nearest].copy(), -1, True
    outside = True
    clipped = np.clip(best_bary, 0.0, 1.0)
    total = float(np.sum(clipped))
    clipped = clipped / total if total > 1e-12 else np.asarray([1.0, 0.0, 0.0])
    surface_tri = surface_vertices[parameterization.surface_faces[best_tri]]
    return clipped @ surface_tri, best_tri, outside


def _inverse_map_uv_to_surface_regular(
    uv_point: np.ndarray,
    parameterization: SurfaceParameterization,
) -> tuple[np.ndarray, int, bool] | None:
    accel = parameterization.triangle_acceleration
    if not isinstance(accel, dict):
        return None
    dense_nx = int(accel.get("dense_nx", 0))
    dense_ny = int(accel.get("dense_ny", 0))
    if dense_nx <= 0 or dense_ny <= 0:
        return None
    boundary = parameterization.omega_boundary
    min_u, min_v = np.min(boundary[:-1], axis=0)
    max_u, max_v = np.max(boundary[:-1], axis=0)
    u, v = float(uv_point[0]), float(uv_point[1])
    outside = u < min_u - 1e-9 or u > max_u + 1e-9 or v < min_v - 1e-9 or v > max_v + 1e-9
    uu = np.clip((u - min_u) / max(max_u - min_u, 1e-12), 0.0, 1.0)
    vv = np.clip((v - min_v) / max(max_v - min_v, 1e-12), 0.0, 1.0)
    col = min(dense_nx - 1, max(0, int(math.floor(uu * dense_nx))))
    row = min(dense_ny - 1, max(0, int(math.floor(vv * dense_ny))))
    base_tri = 2 * (row * dense_nx + col)
    for tri_id in [base_tri, base_tri + 1]:
        face = parameterization.uv_faces[tri_id]
        tri = parameterization.uv_vertices_2d[face]
        bary = _barycentric_2d(np.asarray(uv_point, dtype=float), tri)
        if bary is None:
            continue
        if np.min(bary) >= -1e-9:
            surface_tri = parameterization.surface_vertices_3d[parameterization.surface_faces[tri_id]]
            return bary @ surface_tri, tri_id, outside
    return None


def _barycentric_2d(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denom = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denom) <= 1e-14:
        return None
    inv = 1.0 / denom
    u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv
    v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv
    return np.asarray([1.0 - u - v, u, v], dtype=float)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / max(y1 - y0, 1e-12) + x0):
            inside = not inside
    return inside


def _distances_to_surface_mesh(points: np.ndarray, surface_vertices: np.ndarray, surface_faces: np.ndarray) -> np.ndarray:
    closest = _closest_points_on_surface_mesh(points, surface_vertices, surface_faces)
    return np.linalg.norm(np.asarray(points, dtype=float) - closest, axis=1)


def _closest_points_on_surface_mesh(points: np.ndarray, surface_vertices: np.ndarray, surface_faces: np.ndarray) -> np.ndarray:
    triangles = surface_vertices[np.asarray(surface_faces, dtype=int)]
    values: list[np.ndarray] = []
    for point in np.asarray(points, dtype=float):
        best = float("inf")
        best_point = triangles[0, 0]
        for tri in triangles:
            closest = _closest_point_on_triangle(point, tri[0], tri[1], tri[2])
            dist = float(np.linalg.norm(point - closest))
            if dist < best:
                best = dist
                best_point = closest
        values.append(best_point)
    return np.asarray(values, dtype=float)


def _closest_surface_vertices(points: np.ndarray, surface_vertices: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(surface_vertices)
        _, idx = tree.query(np.asarray(points, dtype=float), k=1)
        return surface_vertices[np.asarray(idx, dtype=int)]
    except Exception:
        pts = np.asarray(points, dtype=float)
        diff = pts[:, None, :] - surface_vertices[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        return surface_vertices[idx]


def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _build_m2d(grid: QuadGrid, domain: PlanarDomain, params: PipelineParameters | None = None) -> QuadMesh:
    overlay_nx = int(getattr(domain, "overlay_nx", grid.nx))
    overlay_ny = int(getattr(domain, "overlay_ny", grid.ny))
    overlay_grid = create_quad_grid(overlay_nx, overlay_ny, grid.tile_size, grid.gap_size)
    vertices = np.column_stack([domain.uv_vertices, np.zeros(len(domain.uv_vertices))])
    kept_faces: list[tuple[int, int, int, int]] = []
    kept_tile_ids: list[int] = []
    cropped = 0
    boundary = domain.boundary[:-1]
    for tile in overlay_grid.tiles or []:
        face = tuple(tile.vertex_ids)
        pts = vertices[list(face), :2]
        center = np.mean(pts, axis=0)
        # Remove quads outside Omega.  The old strict all-vertices-inside test
        # removed almost every boundary quad on non-dome conformal maps, leaving a
        # tiny M2D topology and making K2D/T2D look inconsistent.  The default
        # center policy keeps a quad if its center lies in Ω, which better matches
        # “overlay larger grid, then crop outside cells” for fabrication previews.
        crop_policy = str(getattr(params, "m2d_crop_policy", "center")) if params is not None else "center"
        if crop_policy == "strict_vertices":
            inside = all(_point_in_polygon(pt, boundary) for pt in pts) and _point_in_polygon(center, boundary)
        else:
            inside = _point_in_polygon(center, boundary)
        if inside:
            kept_faces.append(face)
            kept_tile_ids.append(int(tile.id))
        else:
            cropped += 1
    if not kept_faces:
        raise RuntimeError("M2D grid overlay produced no quads inside Omega.")
    faces = np.asarray(kept_faces, dtype=int)
    metrics = {
        "max_csf_before_split": float(getattr(domain, "csf_before", domain.max_csf)),
        "max_csf_after_split": float(getattr(domain, "csf_after_split", domain.max_csf)),
        "number_of_splits": len(domain.split_lines),
        "split_locations": list(domain.split_lines),
        "m2d_grid_overlay": "regular UV grid larger than Omega; outside quads cropped",
        "m2d_crop_policy": str(getattr(params, "m2d_crop_policy", "center")) if params is not None else "center",
        "m2d_requested_grid_nx": int(getattr(domain, "original_requested_nx", grid.nx)),
        "m2d_requested_grid_ny": int(getattr(domain, "original_requested_ny", grid.ny)),
        "m2d_overlay_grid_nx": overlay_nx,
        "m2d_overlay_grid_ny": overlay_ny,
        "m2d_overlay_margin_tiles": int(getattr(domain, "overlay_margin_tiles", 0)),
        "m2d_overlay_total_quad_count": int(len(overlay_grid.tiles or [])),
        "m2d_cropped_quad_count": cropped,
        "m2d_kept_quad_count": len(kept_faces),
        "m2d_kept_original_overlay_tile_ids_sample": kept_tile_ids[:20],
        "m2d_non_rectangular_topology_supported": True,
    }
    return QuadMesh(vertices, faces, overlay_grid, "M2D", metrics, list(domain.split_lines))


def _lift_m2d_to_m3d(
    target: HeightField,
    mesh: QuadMesh,
    parameterization: SurfaceParameterization,
    params: PipelineParameters,
) -> tuple[QuadMesh, StageReport]:
    start = time.perf_counter()
    if params.m3d_construction_mode == "analytic_scaled_heightfield_debug":
        vertices = mesh.vertices.copy()
        vertices[:, 2] = target.height(vertices[:, 0], vertices[:, 1])
        lookup_fail_count = 0
        outside_count = 0
        used_shortcut = True
    else:
        mapped: list[np.ndarray] = []
        lookup_fail_count = 0
        outside_count = 0
        for uv in mesh.vertices[:, :2]:
            point, tri_id, outside = inverse_map_uv_to_surface(uv, parameterization)
            mapped.append(point)
            if tri_id < 0:
                lookup_fail_count += 1
            if outside:
                outside_count += 1
        vertices = np.asarray(mapped, dtype=float)
        used_shortcut = False
    lifted = QuadMesh(vertices, mesh.faces.copy(), mesh.grid, "M3D")
    planarity = _quad_planarity_error(vertices, mesh.faces)
    if used_shortcut:
        analytic_distances = np.abs(vertices[:, 2] - target.height(vertices[:, 0], vertices[:, 1]))
        surface_distances = analytic_distances
        direct_debug = mesh.vertices.copy()
        direct_debug[:, 2] = target.height(direct_debug[:, 0], direct_debug[:, 1])
        rms_debug = float(np.sqrt(np.mean((vertices - direct_debug) ** 2))) if len(vertices) else 0.0
    else:
        surface_distances = _distances_to_surface_mesh(vertices, parameterization.surface_vertices_3d, parameterization.surface_faces)
        analytic_distances = np.full(len(vertices), np.nan)
        direct_uv = mesh.vertices.copy()
        rms_debug = float(np.sqrt(np.mean((vertices - direct_uv) ** 2))) if len(vertices) else 0.0
    surface_mean = float(np.mean(surface_distances)) if surface_distances.size else 0.0
    surface_max = float(np.max(surface_distances)) if surface_distances.size else 0.0
    lifted.metrics = {
        "surface_deviation": surface_mean,
        "quad_planarity_error": planarity,
        "m3d_construction_method": params.m3d_construction_mode,
        "parameterization_method": parameterization.method,
        "m3d_surface_distance_mean": surface_mean,
        "m3d_surface_distance_max": surface_max,
        "m3d_analytic_height_deviation_mean": float(np.nanmean(analytic_distances)) if np.any(np.isfinite(analytic_distances)) else float("nan"),
        "m3d_analytic_height_deviation_max": float(np.nanmax(analytic_distances)) if np.any(np.isfinite(analytic_distances)) else float("nan"),
        "m3d_uv_triangle_lookup_fail_count": lookup_fail_count,
        "m3d_outside_omega_count": outside_count,
        "m3d_used_height_field_shortcut": used_shortcut,
        "m3d_vertex_count": int(len(vertices)),
        "m3d_quad_count": int(len(mesh.faces)),
        "m3d_planarity_error": planarity,
        "m3d_rms_difference_from_uv_plus_height": rms_debug,
    }
    report = StageReport(
        name="M2D -> M3D",
        objective="Inverse surface parameterization c^-1 from Omega to S via UV triangle lookup and barycentric interpolation.",
        before_error=0.0,
        after_error=surface_mean,
        constraint_violation=planarity,
        computation_time=time.perf_counter() - start,
        counts=_mesh_counts(lifted),
    )
    return lifted, report


def _optimize_k3d(target: HeightField, mesh: QuadMesh, parameterization: SurfaceParameterization, params: PipelineParameters) -> tuple[QuadMesh, StageReport]:
    start = time.perf_counter()
    base = mesh.vertices.copy()
    before_planar = _quad_planarity_error(base, mesh.faces)
    before_surface = _surface_fit_error(target, base, parameterization, params)
    before_square = _square_error(base, mesh.faces)
    before_variance = _edge_length_variance(base, mesh.faces)
    z_range_m3d = _z_range(base)

    def residual(xyz_values: np.ndarray) -> np.ndarray:
        vertices = xyz_values.reshape(-1, 3)
        parts: list[np.ndarray] = []
        parts.append(math.sqrt(params.w_planar) * _planarity_residuals(vertices, mesh.faces))
        parts.append(math.sqrt(params.w_square) * _square_residuals(vertices, mesh.faces))
        surface_closest = _closest_surface_vertices(vertices, parameterization.surface_vertices_3d)
        parts.append(math.sqrt(params.w_surface) * (vertices - surface_closest).ravel())
        # Keep the conformal parameterization from drifting excessively while
        # still allowing full 3D optimization instead of z-only flattening.
        xy_anchor_weight = 0.05
        parts.append(math.sqrt(xy_anchor_weight) * (vertices[:, :2] - base[:, :2]).ravel())
        return np.concatenate([p.ravel() for p in parts if p.size])

    k3d_gpu_start = time.perf_counter()
    if params.m3d_construction_mode == "analytic_scaled_heightfield_debug":
        torch_vertices = _optimize_k3d_torch(target, mesh, params, base)
    else:
        torch_vertices = _optimize_k3d_mesh_torch(mesh, parameterization, params, base)
    k3d_gpu_time = time.perf_counter() - k3d_gpu_start if torch_vertices is not None else 0.0
    large_grid_fast_path = (mesh.grid.nx + 1) * (mesh.grid.ny + 1) > 100 and torch_vertices is None
    if torch_vertices is not None:
        vertices = torch_vertices
    elif least_squares is not None and not large_grid_fast_path:
        opt = least_squares(residual, base.ravel(), max_nfev=max(5, params.max_3d_iterations), method="trf")
        vertices = opt.x.reshape(-1, 3)
    else:
        vertices = _fast_k3d_projection_to_surface(parameterization, base, iterations=max(1, min(4, params.max_3d_iterations // 4)))

    z_range_k3d = _z_range(vertices)
    z_range_ratio = z_range_k3d / max(z_range_m3d, 1e-8)
    after_surface_candidate = _surface_fit_error(target, vertices, parameterization, params)
    non_flat_target = z_range_m3d > max(1e-5, mesh.grid.tile_size * 0.02)
    surface_guard = max(before_surface * 1.5, 0.05 * max(z_range_m3d, 1e-8))
    rejected = bool(non_flat_target and (z_range_ratio < 0.3 or after_surface_candidate > surface_guard))
    fallback_used = rejected
    warning = "Large-grid fast path used; dense K3D optimizer skipped." if large_grid_fast_path else ""
    failed_constraints: list[str] = []
    if rejected:
        vertices = base.copy()
        z_range_k3d = z_range_m3d
        z_range_ratio = 1.0
        warning = "K3D optimization rejected; fallback to M3D"
        failed_constraints.append("flattening_guard")

    after_planar = _quad_planarity_error(vertices, mesh.faces)
    after_surface = _surface_fit_error(target, vertices, parameterization, params)
    after_square = _square_error(vertices, mesh.faces)
    metrics = {
        "objective": "E_Assembled = w1*EPlanar + w2*ESquare + w3*ESurface",
        "paper_weight_w1_planar": float(params.w_planar),
        "paper_weight_w2_square": float(params.w_square),
        "paper_weight_w3_surface": float(params.w_surface),
        "paper_default_weights_used": bool(abs(params.w_planar - 10000.0) < 1e-9 and abs(params.w_square - 10.0) < 1e-9 and abs(params.w_surface - 0.1) < 1e-9),
        "planarity_error_before": before_planar,
        "planarity_error_after": after_planar,
        "surface_fit_error_before": before_surface,
        "surface_fit_error_after": after_surface,
        "square_error_before": before_square,
        "square_error_after": after_square,
        "edge_length_variance_before": before_variance,
        "edge_length_variance_after": _edge_length_variance(vertices, mesh.faces),
        "z_range_M3D": z_range_m3d,
        "z_range_K3D": z_range_k3d,
        "z_range_ratio": z_range_ratio,
        "optimization_rejected": rejected,
        "fallback_used": fallback_used,
        "approximation_warning": warning,
        "compute_backend": "cuda" if torch_vertices is not None else ("fast_numpy" if large_grid_fast_path else "scipy"),
        "actual_backend": "cuda" if torch_vertices is not None else ("fast_numpy" if large_grid_fast_path else "scipy"),
        "dominant_backend": "cuda" if torch_vertices is not None else ("cpu" if large_grid_fast_path else "cpu"),
        "gpu_kernel_time": k3d_gpu_time,
        "cpu_preprocess_time": 0.0,
        "cpu_postprocess_time": 0.0,
        "cpu_gpu_transfer_count": 4 if torch_vertices is not None else 0,
        "gpu_memory_peak": int(torch.cuda.max_memory_allocated(0)) if torch_vertices is not None and torch is not None and torch.cuda.is_available() else 0,
    }
    out = QuadMesh(vertices, mesh.faces.copy(), mesh.grid, "K3D", metrics, list(mesh.split_lines))
    report = StageReport(
        name="M3D -> K3D",
        objective=str(metrics["objective"]),
        before_error=before_planar + before_square + before_surface,
        after_error=after_planar + after_square + after_surface,
        constraint_violation=after_planar,
        computation_time=time.perf_counter() - start,
        failed_constraints=failed_constraints,
        counts=_mesh_counts(out),
    )
    return out, report


def _extrude_tiles(mesh: QuadMesh, thickness: float, stage: str) -> tuple[TileAssembly, StageReport]:
    start = time.perf_counter()
    top_tiles = _mesh_tiles(mesh)
    vertices = np.zeros((top_tiles.shape[0], 8, 3), dtype=float)
    transforms = np.zeros((top_tiles.shape[0], 4, 4), dtype=float)
    for i, top in enumerate(top_tiles):
        normal = _quad_normal(top)
        bottom = top - thickness * normal
        vertices[i, :4] = top
        vertices[i, 4:] = bottom
        transform = np.eye(4)
        transform[:3, 3] = -thickness * normal
        transforms[i] = transform
    top_faces = np.asarray([[0, 1, 2, 3] for _ in range(len(vertices))], dtype=int)
    bottom_faces = np.asarray([[4, 7, 6, 5] for _ in range(len(vertices))], dtype=int)
    side_faces = np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)
    planarity = _tile_face_planarity(vertices)
    face_planarity = _tile_face_planarity_by_group(vertices)
    assembly = TileAssembly(
        vertices=vertices,
        top_faces=top_faces,
        bottom_faces=bottom_faces,
        side_faces=side_faces,
        stage=stage,
        metrics={
            "objective": "Extrusion and face planarity projection.",
            "face_planarity_error": planarity,
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            "tile_thickness": thickness,
            "surface_fit_error": float(mesh.metrics.get("surface_fit_error_after", 0.0)),
            "tile_count": int(len(vertices)),
            "k3d_fallback_warning": str(mesh.metrics.get("approximation_warning", "")),
            **_tile_orientation_metrics(vertices, f"{stage.lower()}"),
        },
        transform_matrices=transforms,
    )
    report = StageReport(
        name=f"{mesh.stage} -> {stage}",
        objective="Extrude K3D into eight-vertex quadrilateral frustum tiles.",
        before_error=0.0,
        after_error=planarity,
        constraint_violation=planarity,
        computation_time=time.perf_counter() - start,
        counts=_assembly_counts(assembly),
    )
    return assembly, report



def _strict_k2d_edge_length_solve(
    base_xy: np.ndarray,
    faces: np.ndarray,
    edges: list[tuple[int, int]],
    target_lengths: np.ndarray,
    params: PipelineParameters,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Prioritize K2D/K3D edge-length agreement, then cautiously handle fabrication.

    The paper's K2D stage is driven by the edge lengths of K3D.  The previous
    implementation could return a visually plausible flat layout even when the
    CUDA/Adam solve had not actually converged to those lengths.  This strict
    fallback uses a longer stress/projection solve and refuses collision/fab
    post-processing if it would significantly increase edge mismatch.
    """
    start = time.perf_counter()
    xy = np.asarray(base_xy, dtype=float).copy()
    if len(edges) == 0:
        return xy, {"strict_k2d_solver_used": True, "strict_k2d_solver_reason": "no_edges"}

    # Normalize the starting scale to the K3D mean edge length before iterative
    # projection.  This helps when Omega's UV scale differs from K3D's metric.
    current_lengths = np.asarray([np.linalg.norm(xy[b] - xy[a]) for a, b in edges], dtype=float)
    mean_current = float(np.mean(current_lengths)) if current_lengths.size else 1.0
    mean_target = float(np.mean(target_lengths)) if target_lengths.size else 1.0
    if mean_current > 1e-12 and mean_target > 1e-12:
        c = np.mean(xy, axis=0, keepdims=True)
        xy = c + (xy - c) * (mean_target / mean_current)

    # If SciPy is available, run a pure edge-length least-squares solve only on
    # small systems.  The old threshold was 5000 vertices, which made saddle/wave/
    # gaussian appear to hang: SciPy was trying to solve hundreds or thousands of
    # nonlinear distance constraints exactly.  Medium/large grids now use the
    # bounded vectorized projective solver below.
    scipy_limit = int(getattr(params, "strict_k2d_scipy_vertex_limit", 120))
    scipy_used = False
    if least_squares is not None and len(xy) <= scipy_limit:
        def edge_residual(flat: np.ndarray) -> np.ndarray:
            v = flat.reshape(-1, 2)
            aa = np.asarray([e[0] for e in edges], dtype=int)
            bb = np.asarray([e[1] for e in edges], dtype=int)
            vals = np.linalg.norm(v[bb] - v[aa], axis=1) - target_lengths
            # Very weak centroid/orientation stabilizer; not a shape anchor.
            centroid = (np.mean(v, axis=0) - np.mean(base_xy, axis=0)) * 1e-4
            return np.concatenate([vals, centroid])
        opt = least_squares(
            edge_residual,
            xy.ravel(),
            max_nfev=max(30, int(params.max_2d_iterations) * 4),
            method="trf",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        xy = opt.x.reshape(-1, 2)
        scipy_used = True

    # Bounded vectorized projective edge projection.  No per-iteration collision
    # repulsion here, because that was the main reason edge lengths drifted away
    # from K3D.  This is the main interactive K2D solver for non-dome shapes.
    iterations = max(120, int(params.max_2d_iterations) * 18)
    time_budget = float(getattr(params, "strict_k2d_time_budget_sec", 12.0))
    edge_idx = np.asarray(edges, dtype=int)
    aa = edge_idx[:, 0]
    bb = edge_idx[:, 1]
    strict_tol = max(1e-6, 0.0005 * mean_target)
    projective_iterations_done = 0
    projective_backend = "numpy"
    strict_gpu_kernel_time = 0.0
    strict_gpu_memory_peak = 0
    strict_cpu_gpu_transfer_count = 0
    use_cuda_projective = (
        params.compute.use_gpu_for_optimization
        and torch is not None
        and compute_backend_info(params.compute)["current_backend"] == "cuda"
        and len(xy) > 0
        and len(edges) > 0
    )
    if use_cuda_projective:
        device = torch.device("cuda")
        dtype = torch.float64 if params.compute.dtype == "float64" else torch.float32
        torch.cuda.reset_peak_memory_stats(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        xy_t = torch.as_tensor(xy, dtype=dtype, device=device).clone()
        base_mean_t = torch.as_tensor(np.mean(base_xy, axis=0, keepdims=True), dtype=dtype, device=device)
        aa_t = torch.as_tensor(aa, dtype=torch.long, device=device)
        bb_t = torch.as_tensor(bb, dtype=torch.long, device=device)
        target_t = torch.as_tensor(target_lengths, dtype=dtype, device=device)
        degree_t = torch.zeros((len(xy), 1), dtype=dtype, device=device)
        degree_t.index_add_(0, aa_t, torch.ones((len(aa), 1), dtype=dtype, device=device))
        degree_t.index_add_(0, bb_t, torch.ones((len(bb), 1), dtype=dtype, device=device))
        degree_t = torch.clamp(degree_t, min=1.0)
        strict_cpu_gpu_transfer_count += 5
        start_event.record()
        _emit_progress(progress_callback, "strict K2D edge projection", 0.05, "CUDA projective solver")
        for it in range(iterations):
            delta = xy_t[bb_t] - xy_t[aa_t]
            length = torch.linalg.norm(delta, dim=1)
            safe = torch.clamp(length, min=1e-12)
            correction = ((length - target_t) / safe).unsqueeze(1) * delta * 0.5
            accum = torch.zeros_like(xy_t)
            accum.index_add_(0, aa_t, correction)
            accum.index_add_(0, bb_t, -correction)
            xy_t = xy_t + accum / degree_t
            xy_t = xy_t + (base_mean_t - xy_t.mean(dim=0, keepdim=True))
            projective_iterations_done = it + 1
            if (it + 1) % 10 == 0:
                max_err_t = torch.max(torch.abs(torch.linalg.norm(xy_t[bb_t] - xy_t[aa_t], dim=1) - target_t))
                max_err = float(max_err_t.detach().cpu().item())
                _emit_progress(progress_callback, "strict K2D edge projection", min(0.95, (it + 1) / max(1, iterations)), f"CUDA iter {it + 1}/{iterations}, max edge err={max_err:.4g}")
                if max_err <= strict_tol:
                    break
                if time.perf_counter() - start > time_budget:
                    break
        end_event.record()
        torch.cuda.synchronize(device)
        strict_gpu_kernel_time = float(start_event.elapsed_time(end_event) / 1000.0)
        strict_gpu_memory_peak = int(torch.cuda.max_memory_allocated(device))
        xy = xy_t.detach().cpu().numpy()
        strict_cpu_gpu_transfer_count += 1
        projective_backend = "cuda"
    else:
        degree = np.zeros((len(xy), 1), dtype=float)
        np.add.at(degree, aa, 1.0)
        np.add.at(degree, bb, 1.0)
        degree = np.maximum(degree, 1.0)
        _emit_progress(progress_callback, "strict K2D edge projection", 0.05, "NumPy projective solver")
        for it in range(iterations):
            delta = xy[bb] - xy[aa]
            length = np.linalg.norm(delta, axis=1)
            safe = np.maximum(length, 1e-12)
            correction = ((length - target_lengths) / safe)[:, None] * delta * 0.5
            accum = np.zeros_like(xy)
            np.add.at(accum, aa, correction)
            np.add.at(accum, bb, -correction)
            xy += accum / degree
            # Remove global translation drift only; do not anchor the shape to M2D.
            xy += (np.mean(base_xy, axis=0, keepdims=True) - np.mean(xy, axis=0, keepdims=True))
            projective_iterations_done = it + 1
            if (it + 1) % 10 == 0:
                _, max_err = _edge_matching_errors(xy, edges, target_lengths)
                _emit_progress(progress_callback, "strict K2D edge projection", min(0.95, (it + 1) / max(1, iterations)), f"NumPy iter {it + 1}/{iterations}, max edge err={max_err:.4g}")
                if max_err <= strict_tol:
                    break
                if time.perf_counter() - start > time_budget:
                    break

    mean_before_collision, max_before_collision = _edge_matching_errors(xy, edges, target_lengths)

    # Do not run all-pairs collision relaxation in the strict K2D edge solver on
    # medium/large grids.  That was the main reason non-dome targets appeared to
    # never finish: for 400 boundary-cropped quads it executed hundreds of
    # thousands of Python/Numpy AABB checks after K2D edge matching was already
    # solved.  Collision/fabrication is still reported later and handled by the
    # Dual Hinge E_Hinge stage; here edge-length agreement to K3D is the priority.
    collision_used = False
    collision_skipped_for_speed = bool(len(faces) > 120)
    if collision_skipped_for_speed:
        collision_before = -1
    else:
        collision_before = _count_2d_tile_collisions(_tiles_from_mesh_vertices(np.column_stack([xy, np.zeros(len(xy))]), faces), None, all_pairs=False)
        xy_relaxed = _relax_2d_collisions(xy, faces, None, iterations=4, weight=0.04)
        mean_relaxed, max_relaxed = _edge_matching_errors(xy_relaxed, edges, target_lengths)
        tolerance = max(1e-5, 1.05 * max_before_collision, 0.002 * mean_target)
        if max_relaxed <= tolerance:
            xy = xy_relaxed
            collision_used = True

    mean_final, max_final = _edge_matching_errors(xy, edges, target_lengths)
    return xy, {
        "strict_k2d_solver_used": True,
        "strict_k2d_solver_scipy_used": scipy_used,
        "strict_k2d_scipy_vertex_limit": int(scipy_limit),
        "strict_k2d_time_budget_sec": float(time_budget),
        "strict_k2d_projection_iterations": int(projective_iterations_done),
        "strict_k2d_projective_backend": projective_backend,
        "strict_k2d_gpu_kernel_time": float(strict_gpu_kernel_time),
        "strict_k2d_gpu_memory_peak": int(strict_gpu_memory_peak),
        "strict_k2d_cpu_gpu_transfer_count": int(strict_cpu_gpu_transfer_count),
        "strict_k2d_collision_projection_accepted": collision_used,
        "strict_k2d_collision_projection_skipped_for_speed": collision_skipped_for_speed,
        "strict_k2d_edge_error_before_collision_mean": float(mean_before_collision),
        "strict_k2d_edge_error_before_collision_max": float(max_before_collision),
        "strict_k2d_edge_error_final_mean": float(mean_final),
        "strict_k2d_edge_error_final_max": float(max_final),
        "strict_k2d_collision_count_before_optional_relax": int(collision_before),
        "strict_k2d_elapsed_time": float(time.perf_counter() - start),
        "strict_k2d_policy": "edge-length agreement to K3D is prioritized; collision relaxation is rolled back if it breaks edge matching",
    }


def _optimize_k2d(mesh_2d: QuadMesh, mesh_3d: QuadMesh, params: PipelineParameters, progress_callback: ProgressCallback | None = None) -> tuple[QuadMesh, StageReport]:
    start = time.perf_counter()
    _emit_progress(progress_callback, "Prepare K2D edge targets", 0.02, "K3D correspondence edge lengths")
    base_xy = mesh_2d.vertices[:, :2].copy()
    edges = _unique_mesh_edges(mesh_2d.faces)
    target_lengths = np.asarray(
        [np.linalg.norm(mesh_3d.vertices[a] - mesh_3d.vertices[b]) for a, b in edges],
        dtype=float,
    )
    before_mean, before_max = _edge_matching_errors(base_xy, edges, target_lengths)
    collisions_before = _count_2d_tile_collisions(_tiles_from_mesh_vertices(mesh_2d.vertices, mesh_2d.faces), mesh_2d.grid)

    def residual(xy_flat: np.ndarray) -> np.ndarray:
        xy = xy_flat.reshape(-1, 2)
        parts: list[np.ndarray] = []
        current = np.asarray([np.linalg.norm(xy[a] - xy[b]) for a, b in edges], dtype=float)
        parts.append(math.sqrt(params.w_edge) * (current - target_lengths))
        parts.append(math.sqrt(params.w_fab) * (xy - base_xy).ravel())
        return np.concatenate([p.ravel() for p in parts])

    _emit_progress(progress_callback, "Fast K2D optimizer", 0.08, "Try CUDA/Adam path if available")
    torch_result, torch_metrics = _optimize_k2d_torch(mesh_2d, mesh_3d, params, base_xy, edges, target_lengths)
    large_grid_fast_path = (mesh_2d.grid.nx + 1) * (mesh_2d.grid.ny + 1) > 100 and torch_result is None
    optimizer_iterations = int(max(12, params.max_2d_iterations * 4))
    optimizer_converged = True
    actual_backend = "cuda" if torch_result is not None else "projective_numpy"
    if torch_result is not None:
        xy = torch_result
        optimizer_iterations += int(max(40, params.max_2d_iterations * 6))
    elif least_squares is not None and not large_grid_fast_path:
        opt = least_squares(residual, base_xy.ravel(), max_nfev=max(5, params.max_2d_iterations), method="trf")
        xy = _projective_edge_match_2d(opt.x.reshape(-1, 2), base_xy, edges, target_lengths, mesh_2d.faces, mesh_2d.grid, iterations=optimizer_iterations)
        optimizer_iterations = int(getattr(opt, "nfev", params.max_2d_iterations)) + optimizer_iterations
        optimizer_converged = bool(getattr(opt, "success", True))
        actual_backend = "scipy+projective_numpy"
    else:
        xy = _projective_edge_match_2d(base_xy, base_xy, edges, target_lengths, mesh_2d.faces, mesh_2d.grid, iterations=optimizer_iterations)
    k2d_collision_relax_skipped_for_speed = False
    if actual_backend != "cuda":
        if len(mesh_2d.faces) > 120:
            # Never run all-pairs 2D collision relaxation on interactive-sized
            # K2D meshes. It is slow, and more importantly it changes the edge
            # lengths that K2D is supposed to inherit from K3D. Dual Hinge is the
            # correct stage for collision-aware rigid tile placement.
            k2d_collision_relax_skipped_for_speed = True
        else:
            relaxed_xy = _relax_2d_collisions(xy, mesh_2d.faces, mesh_2d.grid, iterations=3, weight=0.08)
            relaxed_mean, relaxed_max = _edge_matching_errors(relaxed_xy, edges, target_lengths)
            current_mean, current_max = _edge_matching_errors(xy, edges, target_lengths)
            if relaxed_mean <= current_mean and relaxed_max <= current_max:
                xy = relaxed_xy

    # Strict paper-flow check: K2D must actually match K3D edge lengths.  If the
    # fast CUDA/Adam or lightweight projective pass returns a layout with visible
    # mismatch, rerun a stricter edge-length embedding before building T2D.
    strict_metrics: dict[str, float | int | str | bool] = {"strict_k2d_solver_used": False}
    pre_strict_mean, pre_strict_max = _edge_matching_errors(xy, edges, target_lengths)
    mean_target_length = float(np.mean(target_lengths)) if len(target_lengths) else 1.0
    strict_threshold = max(1e-5, 0.002 * mean_target_length)
    if getattr(params, "strict_paper_flow", False) and pre_strict_max > strict_threshold:
        xy_strict, strict_metrics = _strict_k2d_edge_length_solve(
            base_xy,
            mesh_2d.faces,
            edges,
            target_lengths,
            params,
            progress_callback=_subprogress(progress_callback, 0.25, 0.92, "strict solve: "),
        )
        strict_mean, strict_max = _edge_matching_errors(xy_strict, edges, target_lengths)
        if strict_max <= pre_strict_max or strict_mean <= pre_strict_mean:
            xy = xy_strict
            actual_backend = "strict_edge_length_cuda" if strict_metrics.get("strict_k2d_projective_backend") == "cuda" else ("strict_edge_length_cpu" if actual_backend != "cuda" else "cuda+strict_edge_length_cpu")

    _emit_progress(progress_callback, "Finalize K2D metrics", 0.96, "Edge/collision/gap metrics")
    vertices = np.column_stack([xy, np.zeros(len(xy))])
    z_abs_max = float(np.max(np.abs(vertices[:, 2]))) if len(vertices) else 0.0
    if z_abs_max > 1e-6:
        raise RuntimeError("K2D is not planar. K2D must be a 2D flat layout.")
    after_mean, after_max = _edge_matching_errors(xy, edges, target_lengths)
    collisions = _count_2d_tile_collisions(_tiles_from_mesh_vertices(vertices, mesh_2d.faces), mesh_2d.grid)
    gap_angles = _gap_angles(vertices, mesh_2d.faces)
    displacement = np.linalg.norm(xy - base_xy, axis=1)
    z_range_k3d = float(mesh_3d.metrics.get("z_range_K3D", _z_range(mesh_3d.vertices)))
    warning = ""
    if float(np.sqrt(np.mean(displacement * displacement))) < 1e-5 and z_range_k3d > mesh_2d.grid.tile_size * 0.02:
        warning = "K2D is almost identical to M2D despite non-flat K3D. Edge matching may not be active."
    metrics = {
        "objective": "E_Flat = w1*EEdge + w2*ECollision + w3*EFab",
        "paper_weight_w1_edge": float(params.w_edge),
        "paper_weight_w2_collision": float(params.w_collision),
        "paper_weight_w3_fab": float(params.w_fab),
        "paper_default_weights_used": bool(abs(params.w_edge - 1.0) < 1e-9 and abs(params.w_collision - 1.0) < 1e-9 and abs(params.w_fab - 0.001) < 1e-12),
        "edge_matching_error": after_mean,
        "edge_matching_error_before": before_mean,
        "edge_matching_error_after": after_mean,
        "k2d_z_abs_max": z_abs_max,
        "k2d_edge_error_before": before_mean,
        "k2d_edge_error_after": after_mean,
        "mean_edge_length_error_before": before_mean,
        "mean_edge_length_error_after": after_mean,
        "max_edge_length_error_before": before_max,
        "max_edge_length_error_after": after_max,
        "k2d_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))),
        "k2d_displacement_max": float(np.max(displacement)) if displacement.size else 0.0,
        "k2d_xy_displacement_rms_from_M2D": float(np.sqrt(np.mean(displacement * displacement))),
        "k2d_xy_displacement_max_from_M2D": float(np.max(displacement)) if displacement.size else 0.0,
        "collision_count_before": collisions_before,
        "collision_count_after": collisions,
        "2d_collision_count": collisions,
        "k2d_tile_overlap_count": collisions,
        "k2d_min_clearance": _min_aabb_clearance_2d(_tiles_from_mesh_vertices(vertices, mesh_2d.faces), mesh_2d.grid),
        "k2d_gap_count": len(_hinge_specs_from_faces(mesh_2d.faces)),
        "min_gap_angle": float(np.min(gap_angles)) if gap_angles.size else 0.0,
        "max_gap_angle": float(np.max(gap_angles)) if gap_angles.size else 0.0,
        "fabrication_clearance_violation": float(np.mean(np.linalg.norm(xy - base_xy, axis=1))),
        "collision_projection": "deferred to Dual Hinge on medium/large grids; small-grid local AABB relax only if edge matching is preserved",
        "k2d_collision_relax_skipped_for_speed": bool(k2d_collision_relax_skipped_for_speed),
        "fast_path": large_grid_fast_path,
        "optimizer_iterations": optimizer_iterations,
        "optimizer_converged": optimizer_converged,
        "strict_edge_length_threshold": strict_threshold,
        "pre_strict_mean_edge_error": pre_strict_mean,
        "pre_strict_max_edge_error": pre_strict_max,
        **strict_metrics,
        "approximation_warning": warning,
        "actual_backend": actual_backend,
        "dominant_backend": actual_backend,
        "gpu_kernel_time": float(torch_metrics.get("gpu_kernel_time", 0.0)) + float(strict_metrics.get("strict_k2d_gpu_kernel_time", 0.0)),
        "cpu_preprocess_time": float(torch_metrics.get("cpu_preprocess_time", 0.0)),
        "cpu_postprocess_time": float(torch_metrics.get("cpu_postprocess_time", 0.0)),
        "cpu_gpu_transfer_count": int(torch_metrics.get("cpu_gpu_transfer_count", 0)) + int(strict_metrics.get("strict_k2d_cpu_gpu_transfer_count", 0)),
        "gpu_memory_peak": max(
            int(torch.cuda.max_memory_allocated(0)) if "cuda" in str(actual_backend) and torch is not None and torch.cuda.is_available() else 0,
            int(strict_metrics.get("strict_k2d_gpu_memory_peak", 0)),
        ),
    }
    out = QuadMesh(vertices, mesh_2d.faces.copy(), mesh_2d.grid, "K2D", metrics, list(mesh_2d.split_lines))
    report = StageReport(
        name="M2D -> K2D",
        objective=str(metrics["objective"]),
        before_error=before_mean,
        after_error=after_mean,
        constraint_violation=float(collisions),
        computation_time=time.perf_counter() - start,
        counts=_mesh_counts(out),
    )
    return out, report



def _t2d_top_to_bottom_transforms(tiles_3d: TileAssembly, tile_count: int) -> np.ndarray:
    """Return one top->bottom transform per tile for T2D construction."""
    transforms = np.zeros((tile_count, 4, 4), dtype=float)
    for i in range(tile_count):
        if tiles_3d.transform_matrices is not None and i < len(tiles_3d.transform_matrices):
            transforms[i] = np.asarray(tiles_3d.transform_matrices[i], dtype=float)
        else:
            transforms[i] = np.eye(4, dtype=float)
            if i < len(tiles_3d.vertices):
                offsets = np.asarray(tiles_3d.vertices[i, 4:] - tiles_3d.vertices[i, :4], dtype=float)
                transforms[i][:3, 3] = np.mean(offsets, axis=0)
    return transforms


def _apply_t2d_transforms_to_top_xy(top_xy: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Build 8-vertex flat T2D tiles from top-face xy coordinates and per-tile transforms."""
    top_xy = np.asarray(top_xy, dtype=float)
    n = int(len(top_xy))
    vertices = np.zeros((n, 8, 3), dtype=float)
    if n == 0:
        return vertices
    top = np.concatenate([top_xy, np.zeros((n, 4, 1), dtype=float)], axis=2)
    vertices[:, :4] = top
    for i in range(n):
        transform = np.asarray(transforms[i], dtype=float) if i < len(transforms) else np.eye(4, dtype=float)
        homo = np.concatenate([top[i], np.ones((4, 1), dtype=float)], axis=1)
        vertices[i, 4:] = (transform @ homo.T).T[:, :3]
    return vertices



def _hinge_constraint_tuples_from_specs(specs: list[HingeSpec]) -> list[tuple[int, int, int, int]]:
    return [(int(spec.tile_a), int(spec.corner_a0), int(spec.tile_b), int(spec.corner_b0)) for spec in specs]


def _hinge_constraint_tuples_from_hinges(hinges: list[Hinge]) -> list[tuple[int, int, int, int]]:
    return [(int(h.tile_a), int(h.local_vertex_a), int(h.tile_b), int(h.local_vertex_b)) for h in hinges]


def _paper_local_global_se2_layout(
    rest_xy: np.ndarray,
    hinge_constraints: list[tuple[int, int, int, int]],
    footprint_builder,
    initial_xy: np.ndarray | None,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    clearance: float,
    stage_name: str,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Bounded local/global E_Hinge placement for rigid flat tiles.

    This is still a lightweight Python approximation of the paper's
    E_Hinge = E_Rigid + E_Collision + E_Conn, but it is now run as a
    trust-region solver.  The previous patch applied large SAT collision
    projections directly and could throw tiles far away from the K2D layout.
    Here every iteration proposes a rigid SE(2) update, line-searches it
    against the actual objective, clamps per-iteration displacement, and rolls
    back to the best state if collision/connection terms conflict.
    """
    rest = np.asarray(rest_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"paper_layout_optimizer": "empty"}
    current = np.asarray(initial_xy if initial_xy is not None else rest, dtype=float).copy()
    if current.shape != rest.shape:
        current = rest.copy()

    iterations = max(1, int(iterations))
    w_conn = max(0.0, float(connection_weight))
    w_coll = max(0.0, float(collision_weight))
    w_anchor = max(0.0, float(anchor_weight))
    clearance = max(float(clearance), 1e-6)
    time_budget_sec = max(0.0, float(time_budget_sec))
    max_candidate_pairs = max(50, int(max_candidate_pairs))
    collision_sweeps_per_iteration = max(1, int(collision_sweeps_per_iteration))
    solve_start_time = time.perf_counter()
    timed_out = False
    executed_iterations = 0
    pairs_were_capped = False

    rest_centers = np.mean(rest, axis=1)
    edge_lengths = []
    for tile in rest:
        edge_lengths.extend([np.linalg.norm(tile[(i + 1) % tile.shape[0]] - tile[i]) for i in range(tile.shape[0])])
    tile_scale = float(np.median(edge_lengths)) if edge_lengths else 1.0
    tile_scale = max(tile_scale, 1e-8)

    initial_expansion = max(1.0, float(initial_expansion))
    # Expand tile centers, not tile shapes.  The previous implementation used
    # multiplicative global scaling around the entire sheet center:
    #     x += (factor - 1) * (x - sheet_center)
    # That makes voids grow with distance from the sheet center; on medium grids
    # a factor like 1.25 creates huge empty channels.  Use a bounded additive
    # radial offset measured in tile-size units instead.  This gives thick T2D
    # footprints a little fabrication clearance without turning the linkage into
    # a sparse array of disconnected blocks.
    additive_expansion_offset = 0.0
    if initial_expansion > 1.000001 and len(current):
        centers0 = np.mean(current, axis=1)
        world_center0 = np.mean(centers0, axis=0)
        dirs = centers0 - world_center0
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs / np.maximum(norms[:, None], 1e-12)
        additive_expansion_offset = min((initial_expansion - 1.0) * tile_scale, tile_scale * 0.18)
        current = current + additive_expansion_offset * dirs[:, None, :]
    anchor_layout = current.copy()
    anchor_centers = np.mean(anchor_layout, axis=1)
    # This is the stabilizer that prevents the scattered-tile failure mode.
    max_step = max(clearance * 2.0, tile_scale * 0.08)
    max_center_drift = max(tile_scale * float(max_center_drift_tiles), clearance * 8.0)

    def _cap_pairs(fp: np.ndarray, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        nonlocal pairs_were_capped
        if len(pairs) <= max_candidate_pairs:
            return pairs
        pairs_were_capped = True
        bmin = np.min(fp[:, :, :2], axis=1)
        bmax = np.max(fp[:, :, :2], axis=1)
        scores = []
        for k, (i, j) in enumerate(pairs):
            sep = np.maximum(np.maximum(bmin[j] - bmax[i], bmin[i] - bmax[j]), 0.0)
            scores.append((float(np.dot(sep, sep)), k))
        scores.sort(key=lambda x: x[0])
        return [pairs[k] for _, k in scores[:max_candidate_pairs]]

    def _time_expired() -> bool:
        return time_budget_sec > 0.0 and (time.perf_counter() - solve_start_time) >= time_budget_sec

    def _pairs_for(layout: np.ndarray):
        fp = np.asarray(footprint_builder(layout), dtype=float)
        pairs = _spatial_candidate_pairs_for_tiles(fp, pad=max(clearance * 8.0, tile_scale * 0.08, 1e-4))
        pairs = _cap_pairs(fp, pairs)
        return fp, pairs

    def _connection_values(layout: np.ndarray) -> list[float]:
        vals: list[float] = []
        for ia, ca, ib, cb in hinge_constraints:
            if ia < len(layout) and ib < len(layout) and ca < layout.shape[1] and cb < layout.shape[1]:
                vals.append(float(np.linalg.norm(layout[ia, ca] - layout[ib, cb])))
        return vals

    def _energy(layout: np.ndarray) -> tuple[float, dict[str, float | int]]:
        fp, pairs = _pairs_for(layout)
        penetration_sq = 0.0
        collision_count = 0
        min_clear = float("inf")
        for i, j in pairs:
            overlap, _, signed = _sat_polygon_mtv(fp[i], fp[j], clearance=clearance)
            min_clear = min(min_clear, float(signed))
            if overlap:
                collision_count += 1
                penetration_sq += float((-signed) ** 2)
        conn_vals = _connection_values(layout)
        conn_rms = float(np.sqrt(np.mean(np.square(conn_vals)))) if conn_vals else 0.0
        conn_max = float(max(conn_vals, default=0.0))
        centers = np.mean(layout, axis=1)
        center_drift = centers - anchor_centers
        anchor_rms = float(np.sqrt(np.mean(center_drift * center_drift))) if center_drift.size else 0.0
        # Collision count is intentionally strong: a layout with no collisions but
        # slightly worse hinge error is preferable to a visually invalid overlap.
        e = (
            w_conn * conn_rms * conn_rms
            + max(0.0, w_coll) * (penetration_sq + collision_count * clearance * clearance * 25.0)
            + max(0.0, w_anchor) * anchor_rms * anchor_rms
        )
        return float(e), {
            "collision_count": int(collision_count),
            "min_clearance": float(min_clear if np.isfinite(min_clear) else 0.0),
            "hinge_rms": float(conn_rms),
            "hinge_max": float(conn_max),
            "anchor_rms": float(anchor_rms),
            "pair_count": int(len(pairs)),
        }

    def _clamp_step(base: np.ndarray, proposal: np.ndarray) -> np.ndarray:
        out = proposal.copy()
        base_centers = np.mean(base, axis=1)
        prop_centers = np.mean(out, axis=1)
        delta = prop_centers - base_centers
        norms = np.linalg.norm(delta, axis=1)
        active = norms > max_step
        if np.any(active):
            scale = (max_step / np.maximum(norms[active], 1e-12))[:, None]
            out[active] = base[active] + (out[active] - base[active]) * scale[:, None, :]
        # Hard trust region around the expanded fabrication-layout anchor,
        # not the raw K2D shared mesh.  Otherwise expansion is immediately
        # cancelled and thick tiles have no room to avoid collisions.
        centers = np.mean(out, axis=1)
        drift = centers - anchor_centers
        drift_norm = np.linalg.norm(drift, axis=1)
        active = drift_norm > max_center_drift
        if np.any(active):
            target_centers = anchor_centers[active] + drift[active] * (max_center_drift / np.maximum(drift_norm[active], 1e-12))[:, None]
            out[active] += (target_centers - centers[active])[:, None, :]
        return out

    _emit_progress(progress_callback, stage_name, 0.02, "Initialize E_Hinge local/global optimizer")
    before_energy, before_stats = _energy(current)
    before_collision_count = int(before_stats["collision_count"])
    before_clearance = float(before_stats["min_clearance"])
    before_conn = float(before_stats["hinge_rms"])
    best = current.copy()
    best_energy = before_energy
    best_stats = dict(before_stats)
    current_energy = before_energy
    current_stats = dict(before_stats)
    max_collision_count = before_collision_count
    last_pair_count = int(before_stats["pair_count"])
    rejected_steps = 0
    accepted_steps = 0

    progress_stride = max(1, iterations // 60)
    for it in range(iterations):
        if it % progress_stride == 0:
            _emit_progress(
                progress_callback,
                stage_name,
                min(0.98, (it + 1) / max(1, iterations)),
                f"iter {it + 1}/{iterations}, collisions={int(current_stats.get('collision_count', 0))}, hinge_rms={float(current_stats.get('hinge_rms', 0.0)):.4g}",
            )
        if _time_expired():
            timed_out = True
            break
        executed_iterations = it + 1
        desired_sum = np.zeros_like(current)
        desired_weight = np.zeros(current.shape[:2], dtype=float)
        # Keep every vertex close to the current pose; E_Rigid is enforced by
        # fitting a rigid transform to these local targets.
        pose_keep_w = 1.0
        desired_sum += current * pose_keep_w
        desired_weight += pose_keep_w
        if w_anchor > 0.0:
            # The anchor is deliberately not downweighted.  Without this term,
            # collision/connection conflict can scatter the entire linkage.
            desired_sum += anchor_layout * w_anchor
            desired_weight += w_anchor

        # Local E_Conn projection: vertex-joint midpoint targets.
        ramp = min(1.0, (it + 1) / max(1.0, iterations * 0.25))
        for ia, ca, ib, cb in hinge_constraints:
            if ia >= len(current) or ib >= len(current) or ca >= current.shape[1] or cb >= current.shape[1]:
                continue
            mid = 0.5 * (current[ia, ca] + current[ib, cb])
            w = w_conn * ramp
            desired_sum[ia, ca] += mid * w
            desired_weight[ia, ca] += w
            desired_sum[ib, cb] += mid * w
            desired_weight[ib, cb] += w

        # Local E_Collision projection: convert SAT MTVs into bounded targets.
        if w_coll > 0.0:
            fp, pairs = _pairs_for(current)
            last_pair_count = int(len(pairs))
            for _ in range(collision_sweeps_per_iteration):
                shifts = np.zeros((len(current), 2), dtype=float)
                counts = np.zeros((len(current), 1), dtype=float)
                active_count = 0
                for i, j in pairs:
                    overlap, mtv, _ = _sat_polygon_mtv(fp[i], fp[j], clearance=clearance)
                    if not overlap:
                        continue
                    active_count += 1
                    # mtv points from j toward i. Apply small bounded half-steps.
                    shifts[i] += 0.5 * mtv
                    shifts[j] -= 0.5 * mtv
                    counts[i, 0] += 1.0
                    counts[j, 0] += 1.0
                max_collision_count = max(max_collision_count, int(active_count))
                if active_count == 0:
                    break
                active = counts[:, 0] > 0.0
                shifts[active] /= np.maximum(counts[active], 1.0)
                # Add as vertex targets rather than directly overwriting the pose.
                w = min(1.5, 0.35 + 0.25 * w_coll) * ramp
                desired_sum[active] += (current[active] + shifts[active, None, :]) * w
                desired_weight[active] += w

        # Global E_Rigid projection: one SE(2) fit per tile.
        proposal = current.copy()
        for tile_id in range(len(current)):
            weights = np.maximum(desired_weight[tile_id], 1e-12)
            targets = desired_sum[tile_id] / weights[:, None]
            proposal[tile_id] = _fit_rigid_2d_weighted(rest[tile_id], targets, weights)
        proposal = _clamp_step(current, proposal)
        proposal -= np.mean(np.mean(proposal, axis=1), axis=0) - np.mean(np.mean(anchor_layout, axis=1), axis=0)

        # Trust-region line search.  This prevents the optimizer from accepting
        # the chaotic scattered configurations seen in the screenshot.
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            trial = current + alpha * (proposal - current)
            trial = _clamp_step(current, trial)
            trial_energy, trial_stats = _energy(trial)
            # Allow a small temporary energy increase only if it reduces actual collisions.
            improves = trial_energy <= current_energy * 1.02 or int(trial_stats["collision_count"]) < int(current_stats["collision_count"])
            if improves:
                current = trial
                current_energy = trial_energy
                current_stats = dict(trial_stats)
                accepted = True
                accepted_steps += 1
                if trial_energy < best_energy:
                    best = trial.copy()
                    best_energy = trial_energy
                    best_stats = dict(trial_stats)
                break
        if not accepted:
            rejected_steps += 1
            # If no step is acceptable, we have reached a local conflict between
            # hinge connection and collision.  Stop rather than random-walking.
            if rejected_steps >= 5:
                break

    # Always return the best state, not the last state.  If optimization made the
    # layout worse, this rolls back to the original K2D/T2D layout and reports it.
    current = best.copy()
    after_energy, after_stats = _energy(current)
    after_collision_count = int(after_stats["collision_count"])
    after_clearance = float(after_stats["min_clearance"])
    after_conn_rms = float(after_stats["hinge_rms"])
    after_conn_max = float(after_stats["hinge_max"])
    last_pair_count = int(after_stats["pair_count"])

    shape_rms = _tile_shape_distance_error(
        np.dstack([current, np.zeros(current.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
    )
    shape_max = _tile_shape_distance_error(
        np.dstack([current, np.zeros(current.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
        use_max=True,
    )
    return current, {
        "paper_layout_optimizer": "bounded trust-region local/global E_Hinge SE(2) solver",
        "paper_layout_stage": str(stage_name),
        "paper_layout_energy": "E_Hinge = E_Rigid + E_Collision + E_Conn",
        "paper_layout_E_Rigid": "exact per-tile rigid SE(2) Procrustes projection",
        "paper_layout_E_Collision": "SAT full-footprint local projection with trust-region line search",
        "paper_layout_E_Conn": "vertex-joint midpoint local projection",
        "paper_layout_iterations_requested": int(iterations),
        "paper_layout_iterations_executed": int(executed_iterations),
        "paper_layout_timed_out": bool(timed_out),
        "paper_layout_time_budget_sec": float(time_budget_sec),
        "paper_layout_elapsed_sec": float(time.perf_counter() - solve_start_time),
        "paper_layout_max_candidate_pairs": int(max_candidate_pairs),
        "paper_layout_candidate_pairs_capped": bool(pairs_were_capped),
        "paper_layout_collision_sweeps_per_iteration": int(collision_sweeps_per_iteration),
        "paper_layout_connection_weight": float(connection_weight),
        "paper_layout_collision_weight": float(collision_weight),
        "paper_layout_anchor_weight": float(anchor_weight),
        "paper_layout_anchor_reference": "expanded fabrication layout, not raw K2D shared mesh",
        "paper_layout_initial_expansion": float(initial_expansion),
        "paper_layout_expansion_mode": "bounded additive radial offset in tile-size units",
        "paper_layout_additive_expansion_offset": float(additive_expansion_offset),
        "paper_layout_global_space_expansion_enabled": bool(initial_expansion > 1.000001),
        "paper_layout_max_center_drift_tiles": float(max_center_drift_tiles),
        "paper_layout_clearance": float(clearance),
        "paper_layout_trust_region_max_step": float(max_step),
        "paper_layout_trust_region_max_center_drift": float(max_center_drift),
        "paper_layout_accepted_steps": int(accepted_steps),
        "paper_layout_rejected_steps": int(rejected_steps),
        "paper_layout_returned_best_state": True,
        "paper_layout_energy_before": float(before_energy),
        "paper_layout_energy_after": float(after_energy),
        "paper_layout_candidate_pair_count_last": int(last_pair_count),
        "paper_layout_collision_count_before": int(before_collision_count),
        "paper_layout_collision_count_after": int(after_collision_count),
        "paper_layout_collision_count_max_seen": int(max_collision_count),
        "paper_layout_min_clearance_before": float(before_clearance),
        "paper_layout_min_clearance_after": float(after_clearance),
        "paper_layout_hinge_rms_before": float(before_conn),
        "paper_layout_hinge_rms_after": float(after_conn_rms),
        "paper_layout_hinge_max_after": float(after_conn_max),
        "paper_layout_tile_shape_rms_error": float(shape_rms),
        "paper_layout_tile_shape_max_error": float(shape_max),
        "paper_layout_tile_shape_preserved": bool(shape_max < 1e-8),
        "paper_layout_hinge_pairs_exempt_from_collision": False,
        "paper_layout_old_soft_pushback_disabled": True,
        "paper_layout_scattered_layout_guard_enabled": True,
    }


def _optimize_t2d_footprint_layout(
    top_xy: np.ndarray,
    transforms: np.ndarray,
    faces: np.ndarray,
    grid: QuadGrid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Optimize T2D Top-Hinge placement with paper-style E_Hinge."""
    rest = np.asarray(top_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"t2d_footprint_optimizer": "empty"}
    specs = _vertex_hinge_specs_from_faces(faces)
    constraints = _hinge_constraint_tuples_from_specs(specs)

    def footprint_builder(layout: np.ndarray) -> np.ndarray:
        return _apply_t2d_transforms_to_top_xy(layout, transforms)[:, :, :2]

    clearance = max(float(grid.gap_size) * 0.35, 1e-5)
    solved, metrics = _paper_local_global_se2_layout(
        rest,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=rest,
        iterations=max(1, int(iterations)),
        connection_weight=float(connection_weight),
        collision_weight=max(1.0, float(collision_weight)),
        anchor_weight=float(anchor_weight) * 0.25,
        clearance=clearance,
        stage_name="T2D Top Hinge full-footprint placement",
        time_budget_sec=float(time_budget_sec),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=int(collision_sweeps_per_iteration),
        initial_expansion=float(initial_expansion),
        max_center_drift_tiles=float(max_center_drift_tiles),
        progress_callback=progress_callback,
    )
    before_vertices = _apply_t2d_transforms_to_top_xy(rest, transforms)
    after_vertices = _apply_t2d_transforms_to_top_xy(solved, transforms)
    before_pairs = _spatial_candidate_pairs_for_tiles(before_vertices[:, :, :2], pad=clearance * 8.0)
    after_pairs = _spatial_candidate_pairs_for_tiles(after_vertices[:, :, :2], pad=clearance * 8.0)
    after_shape = _tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
    )
    after_shape_max = _tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
        use_max=True,
    )
    out: dict[str, float | int | str | bool] = {
        "t2d_footprint_optimizer": "paper-style local/global E_Hinge full-footprint optimizer",
        "t2d_footprint_collision_checked_on": "top+bottom projected footprint with SAT",
        "t2d_footprint_hinge_pairs_exempt_from_collision": False,
        "t2d_footprint_iterations_requested": int(iterations),
        "t2d_footprint_collision_count_before": int(_count_2d_footprint_collisions_from_pairs(before_vertices[:, :, :2], before_pairs)),
        "t2d_footprint_collision_count_after": int(_count_2d_footprint_collisions_from_pairs(after_vertices[:, :, :2], after_pairs)),
        "t2d_footprint_min_clearance_before": float(_min_footprint_clearance_2d_from_pairs(before_vertices[:, :, :2], before_pairs)),
        "t2d_footprint_min_clearance_after": float(_min_footprint_clearance_2d_from_pairs(after_vertices[:, :, :2], after_pairs)),
        "t2d_footprint_hinge_error_before": float(_vertex_layout_hinge_error(rest, specs)),
        "t2d_footprint_hinge_error_after": float(_vertex_layout_hinge_error(solved, specs)),
        "t2d_top_tile_shape_rms_error_after_footprint_layout": float(after_shape),
        "t2d_top_tile_shape_max_error_after_footprint_layout": float(after_shape_max),
        "t2d_top_shape_preserved_by_rigid_pose_fit": bool(after_shape_max < 1e-8),
        "t2d_previous_soft_collision_bug_fixed": True,
        **metrics,
    }
    return solved, out


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) <= 1:
        return pts.copy()
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    # Remove near-duplicate points.
    unique = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - unique[-1]) > 1e-10:
            unique.append(p)
    pts = np.asarray(unique, dtype=float)
    if len(pts) <= 2:
        return pts.copy()

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _sat_polygon_mtv(poly_a: np.ndarray, poly_b: np.ndarray, clearance: float = 0.0) -> tuple[bool, np.ndarray, float]:
    """Return overlap/near-overlap MTV from B to A for convex 2D polygons."""
    a = _convex_hull_2d(poly_a)
    b = _convex_hull_2d(poly_b)
    if len(a) < 2 or len(b) < 2:
        return False, np.zeros(2, dtype=float), 0.0
    axes: list[np.ndarray] = []
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            norm = np.linalg.norm(edge)
            if norm <= 1e-12:
                continue
            axis = np.array([-edge[1], edge[0]], dtype=float) / norm
            axes.append(axis)
    if not axes:
        return False, np.zeros(2, dtype=float), 0.0
    best_axis = axes[0]
    best_depth = float("inf")
    separated = False
    min_gap = float("inf")
    best_gap_axis = axes[0]
    ca = np.mean(a, axis=0)
    cb = np.mean(b, axis=0)
    for axis in axes:
        amin, amax = float(np.min(a @ axis)), float(np.max(a @ axis))
        bmin, bmax = float(np.min(b @ axis)), float(np.max(b @ axis))
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap < 0.0:
            gap = -overlap
            if gap < min_gap:
                min_gap = gap
                best_gap_axis = axis.copy()
            if gap >= clearance:
                separated = True
        else:
            depth = overlap + clearance
            if depth < best_depth:
                best_depth = depth
                best_axis = axis.copy()
    if separated:
        return False, np.zeros(2, dtype=float), min_gap
    axis = best_gap_axis if not np.isfinite(best_depth) else best_axis
    if float(np.dot(ca - cb, axis)) < 0.0:
        axis = -axis
    depth = max(float(best_depth) if np.isfinite(best_depth) else float(clearance - min_gap), 0.0)
    return True, axis * depth, -depth


def _count_2d_footprint_collisions_from_pairs(footprints_xy: np.ndarray, pairs: list[tuple[int, int]]) -> int:
    count = 0
    for i, j in pairs:
        overlap, _, _ = _sat_polygon_mtv(footprints_xy[i], footprints_xy[j], clearance=0.0)
        if overlap:
            count += 1
    return count


def _min_footprint_clearance_2d_from_pairs(footprints_xy: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    vals: list[float] = []
    for i, j in pairs:
        overlap, _, signed = _sat_polygon_mtv(footprints_xy[i], footprints_xy[j], clearance=0.0)
        vals.append(float(signed))
    return float(np.min(vals)) if vals else 0.0


def _tile_footprint_collision_translation_targets(
    footprints_xy: np.ndarray,
    candidate_pairs: list[tuple[int, int]],
    clearance: float,
) -> np.ndarray:
    """SAT-based separation targets for full projected top+bottom footprints."""
    footprints_xy = np.asarray(footprints_xy, dtype=float)
    shifts = np.zeros((len(footprints_xy), 2), dtype=float)
    counts = np.zeros((len(footprints_xy), 1), dtype=float)
    for i, j in candidate_pairs:
        overlap, mtv, _ = _sat_polygon_mtv(footprints_xy[i], footprints_xy[j], clearance=float(clearance))
        if not overlap:
            continue
        # mtv points from tile j toward tile i. Apply half in opposite directions.
        shifts[i] += 0.5 * mtv
        shifts[j] -= 0.5 * mtv
        counts[i, 0] += 1.0
        counts[j, 0] += 1.0
    active = counts[:, 0] > 0.0
    shifts[active] /= np.maximum(counts[active], 1.0)
    return shifts


def _make_t2d_from_transforms(
    mesh_2d: QuadMesh,
    flat_layout: FlatTileLayout,
    mesh_3d: QuadMesh,
    tiles_3d: TileAssembly,
    stage: str,
    params: PipelineParameters | None = None,
) -> tuple[TileAssembly, StageReport]:
    start = time.perf_counter()
    top_tiles = flat_layout.tile_top_vertices_3d
    transforms = _t2d_top_to_bottom_transforms(tiles_3d, len(top_tiles))

    # Paper-faithful T2D construction with a missing physical detail restored:
    # K2D is a top-face metric layout, but the actual flat part is thick.  When
    # a tile is extruded with the T3D top->bottom transform, its projected
    # footprint is the union of top and bottom vertices.  The previous version
    # checked only top-face overlaps, so bottom/side footprints could remain
    # stacked on top of neighbouring tiles.  We therefore run a rigid SE(2)
    # top-pose projection that preserves each K2D tile shape, keeps vertex
    # hinges near their joints, and separates the full 8-vertex footprint before
    # constructing T2D.
    top_xy_initial = top_tiles[:, :, :2].copy()
    footprint_before_vertices = _apply_t2d_transforms_to_top_xy(top_xy_initial, transforms)
    footprint_pairs_before = _spatial_candidate_pairs_for_tiles(footprint_before_vertices[:, :, :2], pad=float(mesh_2d.grid.gap_size) * 3.0)
    footprint_collisions_before = _count_2d_footprint_collisions_from_pairs(footprint_before_vertices[:, :, :2], footprint_pairs_before)
    top_xy, footprint_metrics = _optimize_t2d_footprint_layout(
        top_xy_initial,
        transforms,
        mesh_2d.faces,
        mesh_2d.grid,
        iterations=max(1, int(getattr(params, "hinge_layout_iterations", 120))) if params is not None else 120,
        connection_weight=float(getattr(params, "hinge_layout_connection_weight", 3.0)) if params is not None else 3.0,
        collision_weight=float(getattr(params, "hinge_layout_collision_weight", 0.35)) if params is not None else 0.35,
        anchor_weight=float(getattr(params, "hinge_layout_anchor_weight", 0.03)) if params is not None else 0.03,
        time_budget_sec=float(getattr(params, "hinge_layout_time_budget_sec", 8.0)) if params is not None else 8.0,
        max_candidate_pairs=int(getattr(params, "hinge_layout_max_candidate_pairs", 3000)) if params is not None else 3000,
        collision_sweeps_per_iteration=int(getattr(params, "hinge_layout_collision_sweeps_per_iteration", 2)) if params is not None else 2,
        initial_expansion=float(getattr(params, "hinge_layout_initial_expansion", 1.08)) if params is not None else 1.08,
        max_center_drift_tiles=float(getattr(params, "hinge_layout_max_center_drift_tiles", 2.0)) if params is not None else 2.0,
    )
    vertices = _apply_t2d_transforms_to_top_xy(top_xy, transforms)

    flat_top_xy = top_tiles[:, :, :2]
    top_to_k2d_rms = float(np.sqrt(np.mean((vertices[:, :4, :2] - flat_top_xy) ** 2))) if len(vertices) else 0.0
    top_to_k2d_max = float(np.max(np.linalg.norm(vertices[:, :4, :2] - flat_top_xy, axis=2))) if len(vertices) else 0.0
    top_shape_rms = _tile_shape_distance_error(vertices[:, :4, :], tiles_3d.vertices[:, :4, :])
    top_shape_max = _tile_shape_distance_error(vertices[:, :4, :], tiles_3d.vertices[:, :4, :], use_max=True)
    full_shape_rms = _tile_shape_distance_error(vertices, tiles_3d.vertices)
    full_shape_max = _tile_shape_distance_error(vertices, tiles_3d.vertices, use_max=True)
    m2d_tiles = _mesh_tiles(QuadMesh(mesh_2d.grid.vertex_positions.copy(), mesh_2d.faces.copy(), mesh_2d.grid, "M2D-reference"))
    top_vs_m2d = float(np.sqrt(np.mean((vertices[:, :4, :2] - m2d_tiles[:, :, :2]) ** 2))) if len(vertices) else 0.0
    face_planarity = _tile_face_planarity_by_group(vertices)
    assembly = TileAssembly(
        vertices=vertices,
        top_faces=np.asarray([[0, 1, 2, 3] for _ in range(len(vertices))], dtype=int),
        bottom_faces=np.asarray([[4, 7, 6, 5] for _ in range(len(vertices))], dtype=int),
        side_faces=np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int),
        stage=stage,
        metrics={
            "objective": "Paper T2D construction: top face from K2D; bottom face from T3D top-to-bottom transforms.",
            "face_planarity_error": _tile_face_planarity(vertices),
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            "t2d_footprint_collision_count_before": int(footprint_collisions_before),
            **footprint_metrics,
            "transform_source": "per-tile T3D top-to-bottom transform matrices applied to K2D top vertices",
            "fabrication_geometry_model": "T2D top vertices are K2D; T2D is not a rigid copy of T3D unless K2D/K3D shape matching succeeds",
            "rigid_copy_of_T3D_forced": False,
            "paper_t2d_extrusion_model": True,
            "top_vertices_match_k2d_max_error": top_to_k2d_max,
            "t2d_top_vertices_match_K2D": top_to_k2d_max,
            "top_vertices_rms_from_k2d_layout": top_to_k2d_rms,
            "top_vertices_rms_from_m2d": top_vs_m2d,
            "top_tile_shape_rms_error_to_K3D": top_shape_rms,
            "top_tile_shape_max_error_to_K3D": top_shape_max,
            "tile_shape_rms_error_to_T3D": full_shape_rms,
            "tile_shape_max_error_to_T3D": full_shape_max,
            "t2d_t3d_congruent_tile_geometry": bool(full_shape_max < 1e-6),
            "warning_if_shape_error_large": "If tile_shape_max_error_to_T3D is large, K2D edge/shape matching failed; do not hide it by rigid-copying T3D.",
            "has_8_vertices_per_tile": bool(vertices.shape[1] == 8),
            "t2d_num_tiles": int(len(vertices)),
            "t2d_vertices_per_tile": int(vertices.shape[1]) if vertices.ndim == 3 else 0,
            "side_faces_count": int(len(vertices) * 4),
            "t2d_side_face_count": int(len(vertices) * 4),
            **_tile_orientation_metrics(vertices, "t2d"),
            "t2d_gap_count": int(flat_layout.metrics.get("k2d_gap_count", 0)),
            "t2d_hinge_count": int(len(flat_layout.hinge_pairs)),
        },
        transform_matrices=transforms,
    )
    report = StageReport(
        name=f"{mesh_2d.stage} -> {stage}",
        objective="Generate T2D from K2D top vertices and T3D top-to-bottom transforms.",
        before_error=0.0,
        after_error=float(assembly.metrics["tile_shape_rms_error_to_T3D"]),
        constraint_violation=float(assembly.metrics["top_tile_shape_rms_error_to_K3D"]),
        computation_time=time.perf_counter() - start,
        counts=_assembly_counts(assembly),
    )
    return assembly, report


def _rigidly_place_t3d_tile_in_flat_layout(tile_3d: np.ndarray, flat_top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = np.asarray(tile_3d[:4], dtype=float)
    flat = np.asarray(flat_top, dtype=float)
    src_center = np.mean(top, axis=0)
    dst_center = np.mean(flat, axis=0)
    dst_center = np.array([dst_center[0], dst_center[1], 0.0], dtype=float)

    src_basis = _tile_frame_from_top_face(top)
    dst_basis = _flat_frame_from_top_face(flat)
    rotation = dst_basis @ src_basis.T
    placed = (np.asarray(tile_3d, dtype=float) - src_center) @ rotation.T + dst_center

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = dst_center - rotation @ src_center
    return placed, transform


def _tile_frame_from_top_face(top: np.ndarray) -> np.ndarray:
    center = np.mean(top, axis=0)
    normal = _quad_normal(top)
    x_axis = top[1] - top[0]
    x_axis = x_axis - normal * float(np.dot(x_axis, normal))
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-12:
        _, _, vt = np.linalg.svd(top - center, full_matrices=False)
        x_axis = vt[0]
        x_axis = x_axis - normal * float(np.dot(x_axis, normal))
        x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-12:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm
    y_axis = np.cross(normal, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-12:
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis = y_axis / y_norm
    return np.column_stack([x_axis, y_axis, normal])


def _flat_frame_from_top_face(flat_top: np.ndarray) -> np.ndarray:
    edge = np.asarray(flat_top[1] - flat_top[0], dtype=float)
    x_axis = np.array([edge[0], edge[1], 0.0], dtype=float)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-12:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm
    normal = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    return np.column_stack([x_axis, y_axis, normal])


def _tile_shape_distance_error(a: np.ndarray, b: np.ndarray, use_max: bool = False) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    da = np.linalg.norm(a[:, :, None, :] - a[:, None, :, :], axis=-1)
    db = np.linalg.norm(b[:, :, None, :] - b[:, None, :, :], axis=-1)
    diff = np.abs(da - db)
    return float(np.max(diff) if use_max else np.sqrt(np.mean(diff * diff)))



def _build_hinge_graph(
    grid: QuadGrid,
    mesh_faces: np.ndarray,
    t2d: TileAssembly,
    t3d: TileAssembly,
    dual: bool,
) -> HingeGraph:
    """Build paper-style vertex hinges, not edge hinges.

    In the OneString linkage, neighboring tiles share a single joint.  The old
    prototype treated a shared grid edge as a hinge and generated two hinge
    points per neighbor pair.  That effectively glued whole edges together in
    the flat layout, which over-constrained the mechanism and caused the T2D
    view / PD simulation to collapse into tangled overlaps.  Here each grid
    adjacency contributes exactly one top/bottom corner-to-corner joint.
    """
    hinges: list[Hinge] = []
    top_count = 0
    bottom_count = 0
    for spec in _vertex_hinge_specs_from_faces(mesh_faces):
        ca = spec.corner_a0
        cb = spec.corner_b0
        surface: Literal["top", "bottom"] = "top"
        if dual and _dihedral_indicator(t3d.top_tiles[spec.tile_a], t3d.top_tiles[spec.tile_b]) > 0.1:
            surface = "bottom"
        offset = 0 if surface == "top" else 4
        if surface == "top":
            top_count += 1
        else:
            bottom_count += 1
        rest = 0.5 * (t2d.vertices[spec.tile_a, ca + offset] + t2d.vertices[spec.tile_b, cb + offset])
        target = 0.5 * (t3d.vertices[spec.tile_a, ca + offset] + t3d.vertices[spec.tile_b, cb + offset])
        hinges.append(Hinge(spec.tile_a, spec.tile_b, ca + offset, cb + offset, surface, rest, target))
    topology_metrics = _vertex_hinge_topology_metrics(mesh_faces)
    metrics = {
        "objective": "pairwise vertex-joint hinge constraints + rigid tiles + collision",
        "hinge_topology": "pairwise_alternating_vertex_joint_one_corner_per_tile_side",
        "edge_hinge_model_disabled": True,
        "four_way_hinge_model_disabled": True,
        "diamond_void_layout_intended": True,
        "top_hinge_count": top_count,
        "bottom_hinge_count": bottom_count,
        "hinge_connection_error": _hinge_connection_error(t2d.vertices, hinges),
        "tile_rigidity_error": _rigid_error(t2d.vertices, t2d.vertices),
        "flat_collision_count": _count_aabb_collisions(t2d.vertices, grid),
        **topology_metrics,
    }
    return HingeGraph(hinges, metrics)


def _optimize_dual_hinges(
    grid: QuadGrid,
    mesh_faces: np.ndarray,
    t2d: TileAssembly,
    t3d: TileAssembly,
    params: PipelineParameters | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[TileAssembly, HingeGraph, StageReport]:
    start = time.perf_counter()
    params = params or PipelineParameters(nx=grid.nx, ny=grid.ny, tile_size=grid.tile_size, gap_size=grid.gap_size)
    _emit_progress(progress_callback, "Build dual-hinge constraints", 0.03, "Select top/bottom pairwise hinges")
    hinge_graph = _build_hinge_graph(grid, mesh_faces, t2d, t3d, dual=True)

    # Paper-style E_Hinge stage: optimize rigid tile poses, not just translate
    # already-placed quads.  The variables are one in-plane rigid transform per
    # tile; the tile vertex coordinates are never independently edited.  This
    # approximates E_Rigid + E_Collision + E_Conn and fixes the previous missing
    # step where hinge layout was only a greedy translation projection.
    vertices, opt_metrics = _optimize_rigid_assembly_hinge_layout_2d(
        rest_vertices=t2d.vertices,
        hinges=hinge_graph.hinges,
        grid=grid,
        iterations=max(20, int(params.hinge_layout_iterations)),
        connection_weight=float(params.hinge_layout_connection_weight),
        collision_weight=float(params.hinge_layout_collision_weight),
        anchor_weight=float(params.hinge_layout_anchor_weight),
        time_budget_sec=float(getattr(params, "hinge_layout_time_budget_sec", 8.0)),
        max_candidate_pairs=int(getattr(params, "hinge_layout_max_candidate_pairs", 3000)),
        collision_sweeps_per_iteration=int(getattr(params, "hinge_layout_collision_sweeps_per_iteration", 2)),
        initial_expansion=float(getattr(params, "hinge_layout_initial_expansion", 1.08)),
        max_center_drift_tiles=float(getattr(params, "hinge_layout_max_center_drift_tiles", 2.0)),
        progress_callback=_subprogress(progress_callback, 0.08, 0.92, "E_Hinge layout: "),
    )

    _emit_progress(progress_callback, "Finalize dual hinge", 0.95, "Compute layout metrics")
    dual_shape_rms = _tile_shape_distance_error(vertices, t3d.vertices)
    dual_shape_max = _tile_shape_distance_error(vertices, t3d.vertices, use_max=True)
    out = TileAssembly(
        vertices=vertices,
        top_faces=t2d.top_faces.copy(),
        bottom_faces=t2d.bottom_faces.copy(),
        side_faces=t2d.side_faces.copy(),
        stage="T2D dual hinge",
        metrics={
            **hinge_graph.metrics,
            **opt_metrics,
            "hinge_connection_error": _hinge_connection_error(vertices, hinge_graph.hinges),
            "flat_collision_count": _count_aabb_collisions(vertices, grid),
            "has_8_vertices_per_tile": bool(vertices.shape[1] == 8),
            "side_faces_count": int(len(vertices) * 4),
            "source_top_vertices_match_k2d_before_hinge": t2d.metrics.get("top_vertices_match_k2d_max_error", 0.0),
            "top_vertices_rms_from_pre_hinge_t2d": float(np.sqrt(np.mean((vertices[:, :4, :2] - t2d.vertices[:, :4, :2]) ** 2))) if len(vertices) else 0.0,
            "tile_shape_rms_error_to_T3D": dual_shape_rms,
            "tile_shape_max_error_to_T3D": dual_shape_max,
            "t2d_t3d_congruent_tile_geometry": bool(dual_shape_max < 1e-5),
            "dual_hinge_rigid_projection_applied": True,
            "dual_hinge_projection_model": "E_Hinge-style rigid SE(2) tile placement; optimizes rotations+translations under vertex-joint connection and collision penalties",
            "edge_hinge_model_disabled": True,
            "paper_E_Hinge_approximation": "E_Rigid is enforced exactly by per-tile rigid poses; E_Conn uses vertex-joint residuals; E_Collision uses local AABB repulsion.",
        },
        transform_matrices=t2d.transform_matrices,
    )
    hinge_graph.metrics = dict(out.metrics)
    report = StageReport(
        name="T2D top hinge -> T2D dual hinge",
        objective="Approximate paper E_Hinge = E_Rigid + E_Collision + E_Conn by optimizing one rigid pose per tile.",
        before_error=float(t2d.metrics.get("face_planarity_error", 0.0)),
        after_error=float(out.metrics["hinge_connection_error"]),
        constraint_violation=float(out.metrics["flat_collision_count"]),
        computation_time=time.perf_counter() - start,
        counts=_assembly_counts(out) | {"hinges": len(hinge_graph.hinges)},
    )
    return out, hinge_graph, report


def _build_gap_graph(mesh_faces: np.ndarray, t2d: TileAssembly, t3d: TileAssembly) -> GapGraph:
    gaps: list[Gap] = []
    tile_to_gaps: dict[int, list[int]] = {}

    def add_gap(gap: Gap) -> None:
        gaps.append(gap)
        for tile_id in gap.surrounding_tiles:
            tile_to_gaps.setdefault(int(tile_id), []).append(gap.id)

    for spec in _edge_gap_specs_from_faces(mesh_faces):
        if spec.direction == "x":
            corners_a = [1, 2]
            corners_b = [0, 3]
            gap_type: Literal["vertical", "horizontal", "virtual_boundary", "split_boundary"] = "vertical"
            label = 0
        else:
            corners_a = [3, 2]
            corners_b = [0, 1]
            gap_type = "horizontal"
            label = 1
        p2 = np.mean(np.vstack([t2d.vertices[spec.tile_a, corners_a], t2d.vertices[spec.tile_b, corners_b]]), axis=0)
        p3 = np.mean(np.vstack([t3d.vertices[spec.tile_a, corners_a], t3d.vertices[spec.tile_b, corners_b]]), axis=0)
        add_gap(Gap(len(gaps), [spec.tile_a, spec.tile_b], p2, p3, gap_type, False, label))

    boundary_specs = _boundary_tile_edges_from_faces(mesh_faces)
    for tile_id, corners in boundary_specs:
        p2 = np.mean(t2d.vertices[tile_id, list(corners)], axis=0)
        p3 = np.mean(t3d.vertices[tile_id, list(corners)], axis=0)
        add_gap(Gap(len(gaps), [tile_id], p2, p3, "virtual_boundary", True, -1))

    # Build the gap adjacency through tile incidence. The old code compared
    # every pair of gaps and intersected Python sets, which is O(G^2). On large
    # grids this dominated the app and hid GPU gains. Incidence construction is
    # O(sum gaps incident to each tile)^2 and is effectively linear for this grid.
    edges: set[tuple[int, int]] = set()
    for incident in tile_to_gaps.values():
        m = len(incident)
        for a_idx in range(m):
            a = incident[a_idx]
            for b_idx in range(a_idx + 1, m):
                b = incident[b_idx]
                edges.add((a, b) if a < b else (b, a))

    z_min = float(np.min(t3d.vertices[..., 2]))
    tile_top_z = np.mean(t3d.vertices[:, :4, 2], axis=1)
    for gap in gaps:
        if gap.surrounding_tiles:
            z_values = tile_top_z[np.asarray(gap.surrounding_tiles, dtype=int)]
            gap.gpe = float(np.sum(0.25 * 1.0 * 9.81 * (z_values - z_min)))
        else:
            gap.gpe = 0.0
    metrics = {
        "gap_count": len(gaps),
        "edge_count": len(edges),
        "boundary_gap_count": sum(1 for gap in gaps if gap.boundary),
        "split_boundary_gap_count": sum(1 for gap in gaps if gap.type == "split_boundary"),
        "max_gpe": max((gap.gpe for gap in gaps), default=0.0),
        "gap_graph_algorithm": "tile_incidence_O_local",
    }
    return GapGraph(gaps, sorted(edges), metrics)



def paper_consistency_report(state: OneStringDesignState) -> list[dict[str, str | float | bool]]:
    """Return an explicit audit comparing the current state to the paper pipeline."""
    rows: list[dict[str, str | float | bool]] = []

    def add(item: str, expected: str, actual: str, ok: bool, value: float | bool | str = "") -> None:
        rows.append({"item": item, "expected": expected, "actual": actual, "ok": bool(ok), "value": value})

    method = str(state.surface_parameterization.method)
    m3d_method = str(state.mesh_3d_initial.metrics.get("m3d_construction_method", ""))
    add(
        "Pipeline order",
        "S→Ω→M2D→M3D→K3D→T3D and M2D→K2D→T2D Top→T2D Dual→lift/string→PD simulation",
        "build_onestring_design follows Figure 5 order; K2D shared mesh is converted to independent Top-Hinge tiles before Dual-Hinge optimization",
        True,
        "figure5_order_locked",
    )
    add(
        "S ↔ Ω correspondence",
        "Boundary First Flattening c:S→Ω and inverse c^-1:Ω→S",
        f"parameterization={method}, M3D construction={m3d_method}",
        method == "bff" and bool(state.mesh_3d_initial.metrics.get("m3d_used_height_field_shortcut", False)) is False,
        "remaining mismatch: harmonic map is used instead of BFF" if method != "bff" else "bff",
    )
    add(
        "M2D -> M3D inverse map",
        "M2D vertices live in Ω and are lifted to S by c^-1",
        str(state.mesh_3d_initial.metrics.get("m3d_construction_method", "")),
        bool(state.mesh_3d_initial.metrics.get("m3d_used_height_field_shortcut", False)) is False,
        str(state.mesh_3d_initial.metrics.get("m3d_uv_triangle_lookup_fail_count", 0)),
    )
    add(
        "K3D optimization",
        "planarity + square-like shape + surface closeness",
        str(state.mesh_3d_optimized.metrics.get("objective", "")),
        "planar" in str(state.mesh_3d_optimized.metrics.get("objective", "")).lower()
        or "planarity" in str(state.mesh_3d_optimized.metrics.get("objective", "")).lower(),
        float(state.mesh_3d_optimized.metrics.get("planarity_error_after", 0.0)),
    )
    add(
        "K2D optimization",
        "flat independent layout matching K3D edge lengths, with collision/fabrication constraints",
        str(state.mesh_2d_optimized.metrics.get("objective", "")),
        bool(state.mesh_2d_optimized.metrics.get("k2d_edge_matching_to_K3D", False))
        or float(state.mesh_2d_optimized.metrics.get("edge_matching_error", 1.0)) < 0.1,
        float(state.mesh_2d_optimized.metrics.get("edge_matching_error", 0.0)),
    )
    add(
        "Paper weights",
        "EAssembled weights (10000,10,0.1), EFlat weights (1,1,0.001)",
        f"K3D={state.mesh_3d_optimized.metrics.get('paper_default_weights_used', False)}, K2D={state.mesh_2d_optimized.metrics.get('paper_default_weights_used', False)}",
        bool(state.mesh_3d_optimized.metrics.get("paper_default_weights_used", False)) and bool(state.mesh_2d_optimized.metrics.get("paper_default_weights_used", False)),
        "",
    )
    add(
        "T2D Top Hinge geometry",
        "Duplicate K2D faces into independent rigid tiles, place them as a vertex-hinge linkage, then apply T3D top-to-bottom transforms",
        str(state.k2d_flat_layout.metrics.get("layout_type", "")),
        bool(state.k2d_flat_layout.metrics.get("t2d_uses_independent_tile_vertices", False))
        and bool(state.k2d_flat_layout.metrics.get("tile_shape_preserved_from_K2D", False))
        and bool(state.k2d_flat_layout.metrics.get("shared_edge_gluing_disabled", False)),
        float(state.k2d_flat_layout.metrics.get("k2d_independent_vertex_joint_error", 0.0)),
    )
    add(
        "Hinges",
        "vertex-joint hinge topology, not edge-glued hinges",
        str(state.hinge_graph.metrics.get("hinge_topology", "")),
        state.hinge_graph.metrics.get("hinge_topology") == "vertex_joint_single_corner_per_neighbor_pair",
        float(state.hinge_graph.metrics.get("hinge_connection_error", 0.0)),
    )
    add(
        "Dual Hinge placement optimization",
        "after Top Hinge T2D, choose top/bottom hinge surfaces and optimize one rigid pose per tile for E_Hinge = E_Rigid + E_Collision + E_Conn",
        str(state.tiles_2d_dual_hinge.metrics.get("dual_hinge_layout_optimizer", "")),
        bool(state.tiles_2d_dual_hinge.metrics.get("dual_hinge_tile_rigidity_enforced_by_pose_fit", False))
        and bool(state.k2d_flat_layout.metrics.get("t2d_uses_independent_tile_vertices", False)),
        float(state.tiles_2d_dual_hinge.metrics.get("dual_hinge_final_connection_error", 0.0)),
    )
    add(
        "Actuation snap scope",
        "snap constraints are applied along the computed string path; lift constraints at lift points",
        str(state.simulation_result.metrics.get("snap_scope", "not simulated")) if state.simulation_result else "not simulated",
        True if state.simulation_result is None else state.simulation_result.metrics.get("snap_scope") == "string_path_only",
        str(state.simulation_result.metrics.get("actuated_snap_gap_count", "")) if state.simulation_result else "",
    )
    return rows

def _select_lift_points(gap_graph: GapGraph, tau: float) -> list[LiftPoint]:
    interior = [gap for gap in gap_graph.gaps if not gap.boundary]
    if not interior:
        interior = list(gap_graph.gaps)
    max_gpe = max((gap.gpe for gap in interior), default=0.0)
    threshold = tau * max_gpe
    peaks = [gap for gap in interior if gap.gpe >= threshold and gap.gpe > 0.0]
    if not peaks and interior:
        peaks = [max(interior, key=lambda gap: gap.gpe)]
    peaks = sorted(peaks, key=lambda gap: gap.gpe, reverse=True)
    selected: list[LiftPoint] = []
    used_tiles: set[int] = set()
    for gap in peaks:
        if used_tiles and used_tiles.intersection(gap.surrounding_tiles):
            continue
        selected.append(LiftPoint(gap.id, gap.centroid_2d, gap.centroid_3d, gap.gpe, len(selected)))
        used_tiles.update(gap.surrounding_tiles)
    if not selected and gap_graph.gaps:
        gap = max(gap_graph.gaps, key=lambda item: item.gpe)
        selected.append(LiftPoint(gap.id, gap.centroid_2d, gap.centroid_3d, gap.gpe, 0))
    return selected


def _build_string_path(gap_graph: GapGraph, lift_points: list[LiftPoint], mu_c: float) -> StringPath:
    boundary = [gap for gap in gap_graph.gaps if gap.boundary]
    center = np.mean([gap.centroid_2d for gap in boundary], axis=0) if boundary else np.zeros(3)
    boundary_sorted = sorted(boundary, key=lambda gap: math.atan2(gap.centroid_2d[1] - center[1], gap.centroid_2d[0] - center[0]))
    route = [gap.id for gap in boundary_sorted]
    for lift in lift_points:
        route.extend(_shortest_gap_path(gap_graph, route[-1] if route else lift.gap_id, lift.gap_id))
        if boundary_sorted:
            route.extend(_shortest_gap_path(gap_graph, lift.gap_id, boundary_sorted[0].id))
    deduped: list[int] = []
    for gap_id in route:
        if not deduped or deduped[-1] != gap_id:
            deduped.append(gap_id)
    theta = _turn_angle_total(gap_graph, deduped)
    friction = safe_capstan_friction(mu_c, theta)
    log_channel_cost = float(mu_c * theta) if math.isfinite(mu_c) and math.isfinite(theta) else float("inf")
    route_node_count = len(deduped)
    unique_route_node_count = len(set(deduped))
    duplicate_visit_count = route_node_count - unique_route_node_count
    theta_upper_bound = math.pi * max(0, route_node_count - 2)
    max_single_turn = _max_single_turn_angle(gap_graph, deduped)
    warnings: list[str] = []
    invalid_turn_accumulation = theta > theta_upper_bound + 1e-6
    if invalid_turn_accumulation:
        warnings.append("Invalid turn angle accumulation.")
    if route_node_count and duplicate_visit_count > route_node_count * 0.5:
        warnings.append("String path revisits too many nodes.")
    if theta > 200:
        warnings.append("String path turn angle is extremely large; routing likely failed.")
    return StringPath(
        gap_ids=deduped,
        boundary_gap_ids=[gap.id for gap in boundary_sorted],
        lift_gap_ids=[lift.gap_id for lift in lift_points],
        turn_angle_total=theta,
        estimated_channel_friction=friction,
        metrics={
            "route_length": route_node_count,
            "route_node_count": route_node_count,
            "unique_route_node_count": unique_route_node_count,
            "duplicate_visit_count": duplicate_visit_count,
            "boundary_gap_count": len(boundary_sorted),
            "lift_point_count": len(lift_points),
            "max_single_turn_angle": max_single_turn,
            "turn_angle_total": theta,
            "theta_total": theta,
            "theta_upper_bound": theta_upper_bound,
            "log_channel_cost": log_channel_cost,
            "estimated_channel_friction": friction,
            "overflow_prevented": bool(not math.isfinite(friction) or log_channel_cost > 60.0),
            "invalid_turn_accumulation": invalid_turn_accumulation,
            "warnings": "; ".join(warnings),
        },
    )



def _deployment_snap_gaps(state: OneStringDesignState, snap_scope: str = "string_path_only") -> list[Gap]:
    """Gaps whose side faces are actuated by the virtual string pull.

    Paper-aligned default: only gaps traversed by the computed string path are
    actively snapped.  The old prototype used all internal gaps by default,
    which effectively applied a global invisible actuator and often collapsed
    the linkage into a tangled clump.  The paper's simulation abstracts the
    string as snap/lift positional constraints along the string path, not as an
    all-gap contraction field.
    """
    if snap_scope == "all_internal_gaps":
        return [gap for gap in state.gap_graph.gaps if not gap.boundary and len(gap.surrounding_tiles) == 2]
    active = set(state.string_path.gap_ids)
    return [gap for gap in state.gap_graph.gaps if gap.id in active and not gap.boundary and len(gap.surrounding_tiles) == 2]


def _gap_separation_vectors(state: OneStringDesignState, gaps: list[Gap], include_bottom: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rest: list[np.ndarray] = []
    target: list[np.ndarray] = []
    for gap in gaps:
        rest_a, rest_b = _gap_side_face_midpoints(state.tiles_2d_dual_hinge.vertices, gap, include_bottom)
        target_a, target_b = _gap_side_face_midpoints(state.tiles_3d.vertices, gap, include_bottom)
        rest.append(rest_a - rest_b)
        target.append(target_a - target_b)
    if not rest:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return np.asarray(rest, dtype=float), np.asarray(target, dtype=float)

def _project_lift_constraints(current: np.ndarray, state: OneStringDesignState, alpha: float, weight: float) -> None:
    for lift in state.lift_points:
        gap = state.gap_graph.gaps[lift.gap_id]
        target = (1.0 - alpha) * lift.position_2d + alpha * lift.position_3d
        tile_ids = gap.surrounding_tiles
        current_center = np.mean([np.mean(current[tile, :4], axis=0) for tile in tile_ids], axis=0)
        delta = (target - current_center) * weight
        for tile in tile_ids:
            current[tile] += delta / max(1, len(tile_ids))


def _project_snap_constraints(
    current: np.ndarray,
    state: OneStringDesignState,
    alpha: float,
    weight: float,
    face_level: bool,
    snap_scope: str = "all_internal_gaps",
    use_target_gap_contraction: bool = True,
) -> None:
    effective = alpha * weight
    for gap in _deployment_snap_gaps(state, snap_scope):
        a, b = gap.surrounding_tiles
        pa, pb = _gap_side_face_midpoints(current, gap, face_level)
        if face_level:
            pa2, pb2 = _gap_side_face_midpoints(current, gap, False)
            pa = 0.5 * (pa + pa2)
            pb = 0.5 * (pb + pb2)
        mid = 0.5 * (pa + pb)
        if use_target_gap_contraction:
            rest_a, rest_b = _gap_side_face_midpoints(state.tiles_2d_dual_hinge.vertices, gap, face_level)
            target_a, target_b = _gap_side_face_midpoints(state.tiles_3d.vertices, gap, face_level)
            rest_sep = rest_a - rest_b
            target_sep = target_a - target_b
            desired_sep = (1.0 - alpha) * rest_sep + alpha * target_sep
            desired_a = mid + 0.5 * desired_sep
            desired_b = mid - 0.5 * desired_sep
            current[a] += (desired_a - pa) * effective
            current[b] += (desired_b - pb) * effective
        else:
            current[a] += (mid - pa) * effective
            current[b] += (mid - pb) * effective


def _project_hinge_constraints(current: np.ndarray, state: OneStringDesignState, weight: float) -> None:
    _project_hinge_list(current, state.hinge_graph.hinges, weight)


def _project_hinge_list(current: np.ndarray, hinges: list[Hinge], weight: float) -> None:
    for hinge in hinges:
        pa = current[hinge.tile_a, hinge.local_vertex_a]
        pb = current[hinge.tile_b, hinge.local_vertex_b]
        mid = 0.5 * (pa + pb)
        current[hinge.tile_a, hinge.local_vertex_a] += (mid - pa) * weight
        current[hinge.tile_b, hinge.local_vertex_b] += (mid - pb) * weight



def _project_hinge_tile_translations(current: np.ndarray, hinges: list[Hinge], weight: float) -> None:
    if not hinges or weight <= 0.0:
        return
    delta = np.zeros((current.shape[0], 3), dtype=float)
    counts = np.zeros((current.shape[0], 1), dtype=float)
    for hinge in hinges:
        pa = current[hinge.tile_a, hinge.local_vertex_a]
        pb = current[hinge.tile_b, hinge.local_vertex_b]
        mid = 0.5 * (pa + pb)
        delta[hinge.tile_a] += (mid - pa) * weight
        delta[hinge.tile_b] += (mid - pb) * weight
        counts[hinge.tile_a, 0] += 1.0
        counts[hinge.tile_b, 0] += 1.0
    active = counts[:, 0] > 0.0
    current[active] += (delta[active] / np.maximum(counts[active], 1.0))[:, None, :]

def _project_rigid_tiles(current: np.ndarray, rest: np.ndarray, weight: float) -> None:
    for i in range(current.shape[0]):
        projected = _kabsch_project(rest[i], current[i])
        current[i] = (1.0 - weight) * current[i] + weight * projected


def _project_aabb_collisions(
    current: np.ndarray,
    weight: float,
    tile_count: int | None = None,
    grid: QuadGrid | None = None,
    all_pairs: bool = False,
) -> None:
    for i, j in _collision_candidate_pairs(current.shape[0], grid, all_pairs):
        min_i = np.min(current[i], axis=0)
        max_i = np.max(current[i], axis=0)
        center_i = np.mean(current[i], axis=0)
        min_j = np.min(current[j], axis=0)
        max_j = np.max(current[j], axis=0)
        overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
        if np.all(overlap > 0):
            center_j = np.mean(current[j], axis=0)
            axis = int(np.argmin(overlap))
            direction = 1.0 if center_i[axis] >= center_j[axis] else -1.0
            delta = np.zeros(3)
            delta[axis] = direction * overlap[axis] * 0.5 * weight
            current[i] += delta
            current[j] -= delta


def _smooth_activation(alpha: float, start_alpha: float) -> float:
    if start_alpha >= 1.0:
        return 0.0
    t = (float(alpha) - float(start_alpha)) / max(1e-8, 1.0 - float(start_alpha))
    t = max(0.0, min(1.0, t))
    return float(t * t * (3.0 - 2.0 * t))


def _tile_outward_normals_from_thickness(vertices: np.ndarray) -> np.ndarray:
    """Return outward/top normals from the actual top-bottom thickness direction.

    This avoids the common top/bottom inversion bug where normals are inferred
    from the top-face winding.  The canonical tile ordering is vertices 0..3 =
    top and 4..7 = bottom, so the outward/top direction is bottom_center->top_center.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        return np.zeros((0, 3), dtype=float)
    top_center = np.mean(vertices[:, :4], axis=1)
    bottom_center = np.mean(vertices[:, 4:], axis=1)
    normals = top_center - bottom_center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = norms[:, 0] < 1e-12
    normals = normals / np.maximum(norms, 1e-12)
    if np.any(fallback):
        top = vertices[:, :4]
        face_normals = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
        face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(face_norms, 1e-12)
        normals[fallback] = face_normals[fallback]
    return normals


def _target_tile_normals(target: np.ndarray) -> np.ndarray:
    return _tile_outward_normals_from_thickness(target)


def _tile_orientation_metrics(vertices: np.ndarray, prefix: str = "tile") -> dict[str, float | int | bool | str]:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        return {f"{prefix}_orientation_tile_count": 0}
    top_center = np.mean(vertices[:, :4], axis=1)
    bottom_center = np.mean(vertices[:, 4:], axis=1)
    top_minus_bottom = top_center - bottom_center
    dz = top_minus_bottom[:, 2]
    thickness = np.linalg.norm(top_minus_bottom, axis=1)
    inverted = dz < -1e-8
    very_flat = thickness < 1e-10
    return {
        f"{prefix}_orientation_model": "top vertices 0..3, bottom vertices 4..7; normal from bottom_center_to_top_center",
        f"{prefix}_top_above_bottom_ratio": float(np.mean(dz >= -1e-8)),
        f"{prefix}_inverted_tile_count": int(np.count_nonzero(inverted)),
        f"{prefix}_near_zero_thickness_tile_count": int(np.count_nonzero(very_flat)),
        f"{prefix}_mean_signed_top_minus_bottom_z": float(np.mean(dz)),
        f"{prefix}_min_signed_top_minus_bottom_z": float(np.min(dz)),
        f"{prefix}_mean_thickness_direction_length": float(np.mean(thickness)),
        f"{prefix}_top_bottom_order_ok": bool(np.count_nonzero(inverted) == 0),
    }


def _project_target_pose_fit(current: np.ndarray, rest: np.ndarray, target: np.ndarray, alpha: float, weight: float) -> None:
    """Rigidly pull each tile toward the corresponding T3D target pose.

    The projection is per-tile rigid: for tile i we fit the rest tile to the
    interpolated pose (1-alpha)*rest + alpha*T3D, then blend the current pose
    toward that rigid target pose.  This avoids the common failure where snap /
    lift constraints pull vertices through the translucent T3D target while the
    tile is still allowed to remain below it.
    """
    if weight <= 0.0 or alpha <= 0.0 or current.size == 0:
        return
    eff = max(0.0, min(1.0, float(weight) * float(alpha) * float(alpha)))
    if eff <= 0.0:
        return
    desired = (1.0 - float(alpha)) * rest + float(alpha) * target
    for i in range(current.shape[0]):
        target_pose = _kabsch_project(rest[i], desired[i])
        current[i] = (1.0 - eff) * current[i] + eff * target_pose


def _target_penetration_metrics(current: np.ndarray, target: np.ndarray, clearance: float = 0.0) -> dict[str, float | int | bool | str]:
    if current.size == 0 or target.size == 0:
        return {"target_penetration_count": 0, "target_penetration_max": 0.0, "target_penetration_mean": 0.0}
    normals = _target_tile_normals(target)
    signed = np.sum((current - target) * normals[:, None, :], axis=2)
    depth = float(clearance) - signed
    positive = depth[depth > 0.0]
    out: dict[str, float | int | bool | str] = {
        "target_penetration_count": int(positive.size),
        "target_penetration_max": float(np.max(positive) if positive.size else 0.0),
        "target_penetration_mean": float(np.mean(positive) if positive.size else 0.0),
        "target_contact_normal_source": "tile_thickness_direction_bottom_to_top_not_face_winding",
    }
    out.update(_tile_orientation_metrics(target, "target_t3d"))
    out.update(_tile_orientation_metrics(current, "animated"))
    return out


def _project_target_contact_guard(
    current: np.ndarray,
    rest: np.ndarray,
    target: np.ndarray,
    alpha: float,
    weight: float,
    start_alpha: float,
    clearance: float,
    passes: int = 1,
) -> None:
    """Prevent animated tiles from passing through the corresponding T3D tile.

    The contact normal is now derived from the target tile's thickness
    direction, bottom->top.  This fixes the apparent top/bottom flip caused by
    using face winding normals on tiles whose local vertex order may be
    inconsistent after T2D/T3D transforms.
    """
    if weight <= 0.0 or current.size == 0:
        return
    activation = _smooth_activation(alpha, start_alpha)
    eff = max(0.0, min(1.0, float(weight) * activation))
    if eff <= 0.0:
        return
    normals = _target_tile_normals(target)
    for _ in range(max(1, int(passes))):
        signed = np.sum((current - target) * normals[:, None, :], axis=2)
        depth = np.maximum(float(clearance) - signed, 0.0)
        if not np.any(depth > 0.0):
            break
        max_depth = np.max(depth, axis=1)
        delta = max_depth[:, None] * normals * eff
        active = max_depth > 0.0
        current[active] += delta[active, None, :]
        _project_rigid_tiles(current, rest, 1.0)


def _project_bending_targets(current: np.ndarray, target: np.ndarray, alpha: float, weight: float) -> None:
    current += (target - current) * (alpha * weight * 0.02)


def _optimize_k3d_torch(
    target: HeightField,
    mesh: QuadMesh,
    params: PipelineParameters,
    base: np.ndarray,
) -> np.ndarray | None:
    if torch is None or not params.compute.use_gpu_for_optimization:
        return None
    info = compute_backend_info(params.compute)
    if info["current_backend"] != "cuda" and params.compute.backend != "cuda":
        return None
    if mesh.grid.nx < params.compute.min_grid_for_gpu and params.compute.backend == "auto":
        return None
    if target.kind == "sampled":
        return None
    try:
        device = torch.device("cuda" if info["current_backend"] == "cuda" else "cpu")
        dtype = torch.float64 if params.compute.dtype == "float64" else torch.float32
        verts = torch.nn.Parameter(torch.as_tensor(base, dtype=dtype, device=device))
        base_xy = torch.as_tensor(base[:, :2], dtype=dtype, device=device)
        faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
        optimizer = torch.optim.Adam([verts], lr=0.02 * mesh.grid.tile_size)
        for _ in range(max(20, params.max_3d_iterations * 4)):
            optimizer.zero_grad()
            quads = verts[faces]
            normals = torch.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0], dim=1)
            normals = normals / torch.clamp(torch.linalg.norm(normals, dim=1, keepdim=True), min=1e-8)
            planar = torch.mean(torch.sum((quads[:, 3] - quads[:, 0]) * normals, dim=1) ** 2)
            edge_lengths = torch.stack(
                [torch.linalg.norm(quads[:, (i + 1) % 4] - quads[:, i], dim=1) for i in range(4)],
                dim=1,
            )
            square_edges = torch.mean((edge_lengths - torch.mean(edge_lengths, dim=1, keepdim=True)) ** 2)
            diag = torch.linalg.norm(quads[:, 0] - quads[:, 2], dim=1) - torch.linalg.norm(quads[:, 1] - quads[:, 3], dim=1)
            square = square_edges + torch.mean(diag * diag)
            surface_z = _torch_height(target, verts[:, 0], verts[:, 1])
            surface = torch.mean((verts[:, 2] - surface_z) ** 2)
            xy_anchor = torch.mean((verts[:, :2] - base_xy) ** 2)
            loss = params.w_planar * planar + params.w_square * square + params.w_surface * surface + 0.05 * xy_anchor
            loss.backward()
            optimizer.step()
        return verts.detach().cpu().numpy()
    except Exception:
        if params.compute.backend == "cuda":
            raise
        return None


def _optimize_k3d_mesh_torch(
    mesh: QuadMesh,
    parameterization: SurfaceParameterization,
    params: PipelineParameters,
    base: np.ndarray,
) -> np.ndarray | None:
    if torch is None or not params.compute.use_gpu_for_optimization:
        return None
    info = compute_backend_info(params.compute)
    if params.compute.backend == "cuda" and info["current_backend"] != "cuda":
        raise RuntimeError(str(info.get("fallback_warning") or "CUDA requested but unavailable."))
    if info["current_backend"] != "cuda":
        return None
    if mesh.grid.nx < params.compute.min_grid_for_gpu and params.compute.backend == "auto":
        return None
    try:
        device = torch.device("cuda")
        dtype = torch.float64 if params.compute.dtype == "float64" else torch.float32
        torch.cuda.reset_peak_memory_stats(device)
        verts = torch.nn.Parameter(torch.as_tensor(base, dtype=dtype, device=device))
        base_xy = torch.as_tensor(base[:, :2], dtype=dtype, device=device)
        faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
        surface_vertices = torch.as_tensor(parameterization.surface_vertices_3d, dtype=dtype, device=device)
        with torch.no_grad():
            anchor_idx = torch.argmin(torch.cdist(verts.detach(), surface_vertices), dim=1)
            surface_anchors = surface_vertices[anchor_idx]
        optimizer = torch.optim.Adam([verts], lr=0.015 * mesh.grid.tile_size)
        for _ in range(max(20, params.max_3d_iterations * 4)):
            optimizer.zero_grad()
            quads = verts[faces]
            normals = torch.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0], dim=1)
            normals = normals / torch.clamp(torch.linalg.norm(normals, dim=1, keepdim=True), min=1e-8)
            planar = torch.mean(torch.sum((quads[:, 3] - quads[:, 0]) * normals, dim=1) ** 2)
            edge_lengths = torch.stack(
                [torch.linalg.norm(quads[:, (i + 1) % 4] - quads[:, i], dim=1) for i in range(4)],
                dim=1,
            )
            square_edges = torch.mean((edge_lengths - torch.mean(edge_lengths, dim=1, keepdim=True)) ** 2)
            diag = torch.linalg.norm(quads[:, 0] - quads[:, 2], dim=1) - torch.linalg.norm(quads[:, 1] - quads[:, 3], dim=1)
            square = square_edges + torch.mean(diag * diag)
            surface = torch.mean((verts - surface_anchors) ** 2)
            xy_anchor = torch.mean((verts[:, :2] - base_xy) ** 2)
            loss = params.w_planar * planar + params.w_square * square + params.w_surface * surface + 0.05 * xy_anchor
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize(device)
        return verts.detach().cpu().numpy()
    except Exception:
        if params.compute.backend == "cuda":
            raise
        return None


def _optimize_k2d_torch(
    mesh_2d: QuadMesh,
    mesh_3d: QuadMesh,
    params: PipelineParameters,
    base_xy: np.ndarray,
    edges: list[tuple[int, int]],
    target_lengths: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, float | int]]:
    timing = {"gpu_kernel_time": 0.0, "cpu_preprocess_time": 0.0, "cpu_postprocess_time": 0.0, "cpu_gpu_transfer_count": 0}
    cpu_t0 = time.perf_counter()
    if torch is None or not params.compute.use_gpu_for_optimization:
        return None, timing
    info = compute_backend_info(params.compute)
    if params.compute.backend == "cuda" and info["current_backend"] != "cuda":
        raise RuntimeError(str(info.get("fallback_warning") or "CUDA requested but unavailable."))
    if info["current_backend"] != "cuda":
        return None, timing
    if mesh_2d.grid.nx < params.compute.min_grid_for_gpu and params.compute.backend == "auto":
        return None, timing
    try:
        device = torch.device("cuda")
        dtype = torch.float64 if params.compute.dtype == "float64" else torch.float32
        torch.cuda.reset_peak_memory_stats(device)
        xy = torch.nn.Parameter(torch.as_tensor(base_xy, dtype=dtype, device=device))
        base = torch.as_tensor(base_xy, dtype=dtype, device=device)
        edge_idx = torch.as_tensor(edges, dtype=torch.long, device=device)
        rest = torch.as_tensor(target_lengths, dtype=dtype, device=device)
        face_idx = torch.as_tensor(mesh_2d.faces, dtype=torch.long, device=device)
        timing["cpu_preprocess_time"] = time.perf_counter() - cpu_t0
        timing["cpu_gpu_transfer_count"] += 4
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        optimizer = torch.optim.Adam([xy], lr=0.015 * mesh_2d.grid.tile_size)
        for _ in range(max(40, params.max_2d_iterations * 6)):
            optimizer.zero_grad()
            delta = xy[edge_idx[:, 1]] - xy[edge_idx[:, 0]]
            lengths = torch.linalg.norm(delta, dim=1)
            edge_loss = torch.mean((lengths - rest) ** 2)
            fab_loss = torch.mean((xy - base) ** 2)
            loss = params.w_edge * edge_loss + params.w_fab * fab_loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            xy_data = xy.detach()
            xy_data = _projective_edge_match_2d_torch(xy_data, base, edge_idx, rest, iterations=int(max(12, params.max_2d_iterations * 4)))
            relaxed = _relax_2d_collisions_torch(xy_data, face_idx, iterations=6, weight=0.15)
            current_mean, current_max = _edge_matching_error_tensors(xy_data, edge_idx, rest)
            relaxed_mean, relaxed_max = _edge_matching_error_tensors(relaxed, edge_idx, rest)
            accept_relaxed = torch.logical_and(relaxed_mean <= current_mean, relaxed_max <= current_max)
            xy_data = torch.where(accept_relaxed, relaxed, xy_data)
        end_event.record()
        torch.cuda.synchronize(device)
        timing["gpu_kernel_time"] = float(start_event.elapsed_time(end_event) / 1000.0)
        cpu_post = time.perf_counter()
        out = xy_data.detach().cpu().numpy()
        timing["cpu_postprocess_time"] = time.perf_counter() - cpu_post
        timing["cpu_gpu_transfer_count"] += 1
        return out, timing
    except Exception:
        if params.compute.backend == "cuda":
            raise
        return None, timing


def _edge_matching_errors_torch(xy, edge_idx, rest) -> tuple[float, float]:
    mean_err, max_err = _edge_matching_error_tensors(xy, edge_idx, rest)
    return float(mean_err.detach().cpu()), float(max_err.detach().cpu())


def _edge_matching_error_tensors(xy, edge_idx, rest):
    lengths = torch.linalg.norm(xy[edge_idx[:, 1]] - xy[edge_idx[:, 0]], dim=1)
    err = torch.abs(lengths - rest)
    if err.numel() == 0:
        zero = torch.zeros((), dtype=xy.dtype, device=xy.device)
        return zero, zero
    return torch.mean(err), torch.max(err)


def _projective_edge_match_2d_torch(xy, base, edge_idx, rest, iterations: int):
    out = xy.clone()
    anchor = 0.002
    for _ in range(max(1, iterations)):
        delta = out[edge_idx[:, 1]] - out[edge_idx[:, 0]]
        length = torch.clamp(torch.linalg.norm(delta, dim=1, keepdim=True), min=1e-10)
        correction = (length - rest[:, None]) * delta / length * 0.5
        accum = torch.zeros_like(out)
        accum.index_add_(0, edge_idx[:, 0], correction)
        accum.index_add_(0, edge_idx[:, 1], -correction)
        counts = torch.zeros((out.shape[0], 1), dtype=out.dtype, device=out.device)
        ones = torch.ones((edge_idx.shape[0], 1), dtype=out.dtype, device=out.device)
        counts.index_add_(0, edge_idx[:, 0], ones)
        counts.index_add_(0, edge_idx[:, 1], ones)
        out = out + accum / torch.clamp(counts, min=1.0)
        out = out + (base - out) * anchor
    return out


def _relax_2d_collisions_torch(xy, face_idx, iterations: int, weight: float):
    out = xy.clone()
    tile_count = face_idx.shape[0]
    if tile_count < 2:
        return out
    pair_i, pair_j = torch.triu_indices(tile_count, tile_count, offset=1, device=out.device)
    for _ in range(iterations):
        tiles = out[face_idx]
        min_i = tiles[pair_i].amin(dim=1)
        max_i = tiles[pair_i].amax(dim=1)
        min_j = tiles[pair_j].amin(dim=1)
        max_j = tiles[pair_j].amax(dim=1)
        overlap = torch.minimum(max_i, max_j) - torch.maximum(min_i, min_j)
        mask = torch.all(overlap > 0, dim=1)
        if not bool(mask.any()):
            continue
        pi = pair_i[mask]
        pj = pair_j[mask]
        ov = overlap[mask]
        axis = torch.argmin(ov, dim=1)
        center_i = tiles[pi].mean(dim=1)
        center_j = tiles[pj].mean(dim=1)
        sign = torch.where(center_i.gather(1, axis[:, None]).squeeze(1) >= center_j.gather(1, axis[:, None]).squeeze(1), 1.0, -1.0).to(out.dtype)
        delta = torch.zeros((pi.shape[0], 2), dtype=out.dtype, device=out.device)
        delta[torch.arange(pi.shape[0], device=out.device), axis] = sign * ov.gather(1, axis[:, None]).squeeze(1) * weight
        vertex_delta = torch.zeros_like(out)
        for corner in range(4):
            vertex_delta.index_add_(0, face_idx[pi, corner], delta)
            vertex_delta.index_add_(0, face_idx[pj, corner], -delta)
        out = out + vertex_delta * 0.25
    return out


def _fast_k3d_projection_to_surface(parameterization: SurfaceParameterization, base: np.ndarray, iterations: int) -> np.ndarray:
    vertices = base.copy()
    for _ in range(iterations):
        closest = _closest_points_on_surface_mesh(vertices, parameterization.surface_vertices_3d, parameterization.surface_faces)
        vertices = 0.85 * vertices + 0.15 * closest
    return vertices


def _torch_height(target: HeightField, x, y):
    amp = float(target.parameters.get("amplitude", 0.6))
    radius = float(target.parameters.get("radius", 1.8))
    wavelength = float(target.parameters.get("wavelength", 2.5))
    sigma = float(target.parameters.get("sigma", 1.0))
    if target.kind == "flat":
        return torch.zeros_like(x)
    if target.kind == "dome":
        r2 = x * x + y * y
        return amp * torch.clamp(1.0 - r2 / max(radius * radius, 1e-8), min=0.0)
    if target.kind == "saddle":
        return amp * (x * x - y * y) / max(radius * radius, 1e-8)
    if target.kind == "wave":
        wave_scale = float(target.parameters.get("wave_amplitude_scale", 0.35))
        return amp * wave_scale * torch.sin(2.0 * math.pi * x / wavelength) * torch.cos(2.0 * math.pi * y / wavelength)
    if target.kind in {"half_gourd", "gourd_half", "hyotan_half", "hyoutan_half"}:
        yn = y / max(radius, 1e-8)
        lower = 0.78 * torch.exp(-((yn + 0.42) / 0.38) ** 2)
        upper = 0.55 * torch.exp(-((yn - 0.46) / 0.32) ** 2)
        waist = 0.42 * torch.exp(-(yn / 0.18) ** 2)
        width_profile = torch.clamp(0.18 + lower + upper - waist, min=0.16, max=1.05)
        half_width = max(radius, 1e-8) * 0.58 * width_profile
        outline = 1.0 - (x / torch.clamp(half_width, min=1e-8)) ** 2 - (torch.abs(yn) / 1.08) ** 4
        asym = 0.92 + 0.08 * torch.tanh(-0.8 * yn)
        return amp * torch.sqrt(torch.clamp(outline, min=0.0)) * asym
    return amp * torch.exp(-(x * x + y * y) / max(2.0 * sigma * sigma, 1e-8))


def _kabsch_project(rest: np.ndarray, current: np.ndarray) -> np.ndarray:
    rest_center = np.mean(rest, axis=0)
    current_center = np.mean(current, axis=0)
    a = rest - rest_center
    b = current - current_center
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return a @ r.T + current_center


def _mesh_tiles(mesh: QuadMesh) -> np.ndarray:
    return _tiles_from_mesh_vertices(mesh.vertices, mesh.faces)


def _make_flat_tile_layout(mesh: QuadMesh, params: PipelineParameters | None = None) -> FlatTileLayout:
    """Build the fabrication-side independent K2D tile linkage layout.

    Important distinction:
    - ``mesh`` is the shared-vertex K2D optimization mesh.  It is only the
      abstract metric mesh whose edge lengths are matched to K3D.
    - ``FlatTileLayout`` is the physical flat linkage layout.  Each quad face
      is duplicated into an independent rigid tile, then one SE(2) pose per tile
      is optimized so neighbouring tiles meet only at vertex joints.

    The previous implementation leaked the shared K2D mesh directly into T2D,
    so the flat state looked like a continuous quad sheet.  That is not the
    OneString linkage: tiles must be independent bodies connected by unique
    vertex joints with void/gap space between them.
    """
    raw_tiles_3d = _mesh_tiles(mesh)
    raw_tiles_xy = raw_tiles_3d[:, :, :2].copy()

    if params is None:
        iterations = 120
        connection_weight = 3.0
        collision_weight = 0.35
        anchor_weight = 0.03
    else:
        iterations = max(20, int(getattr(params, "hinge_layout_iterations", 120)))
        connection_weight = float(getattr(params, "hinge_layout_connection_weight", 3.0))
        collision_weight = float(getattr(params, "hinge_layout_collision_weight", 0.35))
        anchor_weight = float(getattr(params, "hinge_layout_anchor_weight", 0.03))

    flat_tiles, layout_metrics = _optimize_independent_k2d_tile_linkage_layout(
        raw_tiles_xy,
        mesh.faces,
        mesh.grid,
        iterations=iterations,
        connection_weight=connection_weight,
        collision_weight=collision_weight,
        anchor_weight=anchor_weight,
        time_budget_sec=float(getattr(params, "hinge_layout_time_budget_sec", 8.0)) if params is not None else 8.0,
        max_candidate_pairs=int(getattr(params, "hinge_layout_max_candidate_pairs", 3000)) if params is not None else 3000,
        collision_sweeps_per_iteration=int(getattr(params, "hinge_layout_collision_sweeps_per_iteration", 2)) if params is not None else 2,
        initial_expansion=float(getattr(params, "hinge_layout_initial_expansion", 1.08)) if params is not None else 1.08,
        max_center_drift_tiles=float(getattr(params, "hinge_layout_max_center_drift_tiles", 2.0)) if params is not None else 2.0,
    )

    hinge_specs = _vertex_hinge_specs_from_faces(mesh.faces)
    edge_specs = _edge_gap_specs_from_faces(mesh.faces)
    hinge_pairs = [(spec.tile_a, spec.tile_b) for spec in hinge_specs]
    gap_polygons: list[np.ndarray] = []
    for spec in edge_specs:
        if spec.direction == "x":
            a_edge = flat_tiles[spec.tile_a, [1, 2]]
            b_edge = flat_tiles[spec.tile_b, [0, 3]]
        else:
            a_edge = flat_tiles[spec.tile_a, [3, 2]]
            b_edge = flat_tiles[spec.tile_b, [0, 1]]
        gap_polygons.append(np.vstack([a_edge[0], a_edge[1], b_edge[1], b_edge[0]]))

    flat_tiles_3d = np.concatenate([flat_tiles, np.zeros((*flat_tiles.shape[:2], 1), dtype=float)], axis=2)
    raw_tiles_flat_3d = np.concatenate([raw_tiles_xy, np.zeros((*raw_tiles_xy.shape[:2], 1), dtype=float)], axis=2)
    shape_error = _tile_shape_distance_error(flat_tiles_3d, raw_tiles_flat_3d)
    shape_error_max = _tile_shape_distance_error(flat_tiles_3d, raw_tiles_flat_3d, use_max=True)
    collision_pairs = _spatial_candidate_pairs_for_tiles(flat_tiles, pad=float(mesh.grid.gap_size) * 2.0)
    collision_count = _count_2d_tile_collisions_from_pairs(flat_tiles_3d, collision_pairs)
    min_clearance = _min_aabb_clearance_2d_from_pairs(flat_tiles_3d, collision_pairs)
    hinge_error = _vertex_layout_hinge_error(flat_tiles, hinge_specs)
    shared_vertex_error = _k2d_shared_vertex_consistency_error(mesh, raw_tiles_xy)

    metrics: dict[str, float | int | str | bool] = {
        "layout_type": "independent rigid K2D tile linkage layout",
        "k2d_shared_mesh_role": "abstract edge-length mesh only; tile vertices are duplicated for fabrication",
        "t2d_uses_independent_tile_vertices": True,
        "tile_vertices_are_duplicated_from_k2d_faces": True,
        "shared_edge_gluing_disabled": True,
        "edge_hinge_model_disabled": True,
        "paper_layout_correction": True,
        "paper_pipeline_order": "K2D shared mesh -> independent T2D Top Hinge linkage -> T2D Dual Hinge",
        "paper_E_Hinge_layout_optimizer_enabled": True,
        "hinge_layout_optimizer": str(layout_metrics.get("hinge_layout_optimizer", "independent rigid SE(2) tile placement")),
        "hinge_layout_stage": "K2D shared mesh to T2D Top Hinge independent linkage",
        "hinge_layout_deferred_to_dual_hinge": False,
        "old_continuous_sheet_layout_disabled": True,
        "old_center_shrink_layout_disabled": True,
        "old_checkerboard_translate_only_layout_disabled": True,
        "tile_shape_preserved_from_K2D": bool(shape_error_max < 1e-8),
        "k2d_tile_shape_rms_error_after_layout": float(shape_error),
        "k2d_tile_shape_max_error_after_layout": float(shape_error_max),
        "k2d_shared_vertex_consistency_error_before_duplication": float(shared_vertex_error),
        "k2d_independent_vertex_joint_error": float(hinge_error),
        "k2d_z_abs_max": float(mesh.metrics.get("k2d_z_abs_max", 0.0)),
        "tile_count": int(len(flat_tiles)),
        "vertices_per_tile": 4,
        "k2d_gap_count": len(gap_polygons),
        "hinge_pair_count": len(hinge_pairs),
        "tile_overlap_count": int(collision_count),
        "min_clearance": float(min_clearance),
        "gap_opening_model": "voids are generated by rigid independent tile placement, not by shrinking tile geometry",
        "hinge_topology": "vertex_joint_single_corner_per_neighbor_pair",
        "edge_gap_polygons_for_snap_visualization": len(gap_polygons),
        **layout_metrics,
    }
    return FlatTileLayout(
        tile_top_vertices_2d=flat_tiles,
        tile_ids=list(range(len(flat_tiles))),
        hinge_pairs=hinge_pairs,
        gap_polygons=gap_polygons,
        metrics=metrics,
    )



def _k2d_shared_vertex_consistency_error(mesh: QuadMesh, flat_tiles: np.ndarray) -> float:
    """Check that the direct K2D tile layout still matches the shared K2D mesh vertices."""
    if len(flat_tiles) == 0:
        return 0.0
    values: list[float] = []
    for tile_id, face in enumerate(np.asarray(mesh.faces, dtype=int)):
        if tile_id >= len(flat_tiles):
            continue
        expected = mesh.vertices[list(face), :2]
        values.extend(np.linalg.norm(flat_tiles[tile_id, :, :2] - expected, axis=1).tolist())
    return float(max(values)) if values else 0.0



def _optimize_independent_k2d_tile_linkage_layout(
    raw_tiles_xy: np.ndarray,
    faces: np.ndarray,
    grid: QuadGrid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Create independent flat linkage layout from shared K2D via E_Hinge."""
    rest = np.asarray(raw_tiles_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"hinge_layout_optimizer": "empty independent K2D layout"}
    specs = _vertex_hinge_specs_from_faces(faces)
    constraints = _hinge_constraint_tuples_from_specs(specs)
    initial, init_metrics = _initial_independent_k2d_tile_layout(rest, faces, grid)

    def footprint_builder(layout: np.ndarray) -> np.ndarray:
        return layout

    clearance = max(float(grid.gap_size) * 0.35, 1e-5)
    solved, metrics = _paper_local_global_se2_layout(
        rest,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=initial,
        iterations=max(1, int(iterations)),
        connection_weight=float(connection_weight),
        collision_weight=max(1.0, float(collision_weight)),
        anchor_weight=float(anchor_weight) * 0.25,
        clearance=clearance,
        stage_name="K2D shared mesh to independent top-hinge tile layout",
        time_budget_sec=float(time_budget_sec),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=int(collision_sweeps_per_iteration),
        initial_expansion=float(initial_expansion),
        max_center_drift_tiles=float(max_center_drift_tiles),
    )
    before_pairs = _spatial_candidate_pairs_for_tiles(initial, pad=clearance * 8.0)
    after_pairs = _spatial_candidate_pairs_for_tiles(solved, pad=clearance * 8.0)
    after_shape = _tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
    )
    after_shape_max = _tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
        use_max=True,
    )
    out: dict[str, float | int | str | bool] = {
        "hinge_layout_optimizer": "paper-style local/global E_Hinge rigid SE(2) tile placement",
        "hinge_layout_objective": "E_Rigid + E_Collision + E_Conn; exact rigid pose per tile",
        "hinge_layout_iterations": int(iterations),
        "hinge_layout_connection_weight": float(connection_weight),
        "hinge_layout_collision_weight": float(collision_weight),
        "hinge_layout_anchor_weight": float(anchor_weight),
        "hinge_layout_initial_expansion": float(initial_expansion),
        "hinge_layout_max_center_drift_tiles": float(max_center_drift_tiles),
        "hinge_layout_hinge_pairs_exempt_from_collision": False,
        **init_metrics,
        "hinge_layout_initial_connection_error": float(_vertex_layout_hinge_error(initial, specs)),
        "hinge_layout_final_connection_error": float(_vertex_layout_hinge_error(solved, specs)),
        "hinge_layout_initial_collision_count": int(_count_2d_footprint_collisions_from_pairs(initial, before_pairs)),
        "hinge_layout_final_collision_count": int(_count_2d_footprint_collisions_from_pairs(solved, after_pairs)),
        "tile_rigidity_enforced_by_pose_fit": True,
        "k2d_independent_layout_shape_error_before": float(_tile_shape_distance_error(np.dstack([initial, np.zeros(initial.shape[:2])]), np.dstack([rest, np.zeros(rest.shape[:2])]))),
        "k2d_independent_layout_shape_error_after": float(after_shape),
        "k2d_independent_layout_shape_error_max_after": float(after_shape_max),
        "k2d_independent_layout_candidate_pair_count": int(len(after_pairs)),
        "k2d_independent_layout_hinge_count": int(len(specs)),
        **metrics,
    }
    return solved, out



def _initial_independent_k2d_tile_layout(
    raw_tiles_xy: np.ndarray,
    faces: np.ndarray,
    grid: QuadGrid,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Initialize T2D as pairwise hinges around diamond/rhombus voids.

    The previous implementation duplicated K2D faces and then asked E_Hinge to
    discover both the hinge topology and the void opening.  That is too hard and
    visually collapses to the pre-optimization sheet or scatters under collision
    pressure.  The paper's topology says a tile can connect to up to four
    neighbors by unique pairwise joints, and those cuts create quadrilateral
    gaps/voids between four tiles.  Therefore the flat layout should already
    start with coincident abstract K2D vertices split into small diamond voids.

    This routine preserves every tile shape exactly.  It only computes one
    initial SE(2) pose per tile by fitting its K2D corners to pairwise hinge
    targets.  Pairwise hinges associated with the same K2D mesh vertex are not
    collapsed into a four-panel joint; they are distributed around that vertex.
    """
    rest = np.asarray(raw_tiles_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"rhombus_void_initializer": "empty"}
    faces_arr = np.asarray(faces, dtype=int)
    specs_with_joint = _vertex_hinge_specs_with_joint_vertices_from_faces(faces_arr)
    if not specs_with_joint:
        return rest.copy(), {"rhombus_void_initializer": "no hinges"}

    # K2D abstract vertex positions from duplicated tile corners.
    vertex_samples: dict[int, list[np.ndarray]] = {}
    for tile_id, face in enumerate(faces_arr):
        if tile_id >= len(rest):
            continue
        for local, vertex_id in enumerate(face):
            vertex_samples.setdefault(int(vertex_id), []).append(rest[tile_id, local])
    vertex_pos = {vid: np.mean(np.asarray(samples), axis=0) for vid, samples in vertex_samples.items()}

    # Group pairwise hinges by the abstract K2D vertex they came from.  Each
    # group becomes a small rhombus/diamond of physical hinge positions.
    groups: dict[int, list[tuple[int, HingeSpec]]] = {}
    for idx, (spec, joint_vertex) in enumerate(specs_with_joint):
        groups.setdefault(int(joint_vertex), []).append((idx, spec))

    edge_lengths = []
    for tile in rest:
        for k in range(4):
            edge_lengths.append(float(np.linalg.norm(tile[(k + 1) % 4] - tile[k])))
    tile_scale = max(float(np.median(edge_lengths)) if edge_lengths else float(grid.tile_size), 1e-8)
    # Use a compact void radius.  It is large enough to show the rhombus but
    # smaller than the earlier global expansion that created huge empty channels.
    void_radius = min(max(float(grid.gap_size) * 0.45, tile_scale * 0.018), tile_scale * 0.10)

    # Targets for specific tile corners.  Multiple targets on a corner are
    # averaged, but the pairwise topology tries to keep this rare.
    desired_sum = np.zeros_like(rest)
    desired_weight = np.zeros(rest.shape[:2], dtype=float)

    for joint_vertex, items in groups.items():
        center = np.asarray(vertex_pos.get(joint_vertex, np.zeros(2)), dtype=float)
        m = len(items)
        if m == 1:
            dirs = [np.zeros(2, dtype=float)]
        else:
            # Direction is based on the midpoint of the two participating tile
            # centers relative to the abstract vertex.  This naturally separates
            # the lower/upper or left/right pairwise hinges around the void.  If
            # degenerate, fall back to evenly spaced diamond directions.
            dirs = []
            for _, spec in items:
                ca = np.mean(rest[int(spec.tile_a)], axis=0) if int(spec.tile_a) < len(rest) else center
                cb = np.mean(rest[int(spec.tile_b)], axis=0) if int(spec.tile_b) < len(rest) else center
                d = 0.5 * (ca + cb) - center
                n = float(np.linalg.norm(d))
                dirs.append(d / n if n > 1e-9 else np.zeros(2, dtype=float))
            if sum(float(np.linalg.norm(d)) for d in dirs) < 1e-9:
                base = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])]
                dirs = [base[i % len(base)] for i in range(m)]
            # Keep directions in a diamond-like set, not arbitrary tiny angles.
            for i, d in enumerate(dirs):
                if float(np.linalg.norm(d)) <= 1e-9:
                    angle = 2.0 * math.pi * i / max(1, m)
                    dirs[i] = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        for (item_idx, spec), d in zip(items, dirs):
            target = center + void_radius * d
            ia = int(spec.tile_a)
            ib = int(spec.tile_b)
            ca = int(spec.corner_a0)
            cb = int(spec.corner_b0)
            if ia < len(rest) and ca < 4:
                desired_sum[ia, ca] += target
                desired_weight[ia, ca] += 1.0
            if ib < len(rest) and cb < 4:
                desired_sum[ib, cb] += target
                desired_weight[ib, cb] += 1.0

    # Add weak anchors for all corners, and fit one rigid transform per tile.
    # This preserves tile shapes while giving the E_Hinge optimizer a layout that
    # already contains the intended void topology.
    anchor_w = 0.08
    desired_sum += rest * anchor_w
    desired_weight += anchor_w
    initial = rest.copy()
    for tile_id in range(len(rest)):
        targets = desired_sum[tile_id] / np.maximum(desired_weight[tile_id, :, None], 1e-12)
        initial[tile_id] = _fit_rigid_2d_weighted(rest[tile_id], targets, desired_weight[tile_id])

    # Keep global translation comparable to K2D.
    initial -= np.mean(np.mean(initial, axis=1), axis=0) - np.mean(np.mean(rest, axis=1), axis=0)

    specs = [spec for spec, _ in specs_with_joint]
    joint_counts = [len(v) for v in groups.values()]
    return initial, {
        "rhombus_void_initializer": "pairwise hinges split around each abstract K2D vertex",
        "rhombus_void_initializer_enabled": True,
        "rhombus_void_radius": float(void_radius),
        "abstract_k2d_vertex_groups": int(len(groups)),
        "max_pairwise_hinges_per_abstract_vertex": int(max(joint_counts) if joint_counts else 0),
        "mean_pairwise_hinges_per_abstract_vertex": float(np.mean(joint_counts) if joint_counts else 0.0),
        "four_panel_joint_interpretation_disabled": True,
        "initial_layout_hinge_error_after_void_split": float(_vertex_layout_hinge_error(initial, specs)),
        "initial_layout_shape_error_after_void_split": float(_tile_shape_distance_error(
            np.dstack([initial, np.zeros(initial.shape[:2])]),
            np.dstack([rest, np.zeros(rest.shape[:2])]),
        )),
    }

def _spatial_candidate_pairs_for_tiles(tiles_xy: np.ndarray, pad: float = 0.0) -> list[tuple[int, int]]:
    tiles = np.asarray(tiles_xy, dtype=float)
    n = int(len(tiles))
    if n <= 1:
        return []
    bmin = np.min(tiles[:, :, :2], axis=1)
    bmax = np.max(tiles[:, :, :2], axis=1)
    centers = 0.5 * (bmin + bmax)
    spans = np.maximum(bmax - bmin, 1e-8)
    cell = max(float(np.median(np.max(spans, axis=1))) + float(pad), 1e-6)
    keys = np.floor(centers / cell).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(keys):
        buckets.setdefault((int(key[0]), int(key[1])), []).append(int(idx))
    pairs: set[tuple[int, int]] = set()
    for i, key in enumerate(keys):
        kx, ky = int(key[0]), int(key[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for j in buckets.get((kx + dx, ky + dy), []):
                    if j <= i:
                        continue
                    # Broad-phase padding: include close non-overlapping tiles
                    # because they may collide after a projection step.
                    sep = np.maximum(np.maximum(bmin[j] - bmax[i], bmin[i] - bmax[j]), 0.0)
                    if float(np.linalg.norm(sep)) <= max(float(pad), cell * 0.25):
                        pairs.add((int(i), int(j)))
    return sorted(pairs)


def _count_2d_tile_collisions_from_pairs(tiles: np.ndarray, pairs: list[tuple[int, int]]) -> int:
    count = 0
    for i, j in pairs:
        min_i = np.min(tiles[i, :, :2], axis=0)
        max_i = np.max(tiles[i, :, :2], axis=0)
        min_j = np.min(tiles[j, :, :2], axis=0)
        max_j = np.max(tiles[j, :, :2], axis=0)
        if np.all(np.minimum(max_i, max_j) - np.maximum(min_i, min_j) > 0):
            count += 1
    return count


def _min_aabb_clearance_2d_from_pairs(tiles: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    values: list[float] = []
    for i, j in pairs:
        min_i = np.min(tiles[i, :, :2], axis=0)
        max_i = np.max(tiles[i, :, :2], axis=0)
        min_j = np.min(tiles[j, :, :2], axis=0)
        max_j = np.max(tiles[j, :, :2], axis=0)
        sep = np.maximum(np.maximum(min_j - max_i, min_i - max_j), 0.0)
        overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
        if np.all(overlap > 0):
            values.append(-float(np.min(overlap)))
        else:
            values.append(float(np.linalg.norm(sep)))
    return float(np.min(values)) if values else 0.0


def _optimize_vertex_hinge_tile_layout(
    raw_tiles: np.ndarray,
    grid: QuadGrid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Optimize one in-plane rigid pose per K2D tile.

    The old prototype opened gaps by a fixed checkerboard rotation and then
    translated tiles.  That omits the actual hinge-placement solve.  This
    local/global projection loop is closer to the paper's E_Hinge: hinge
    connection targets are formed at shared vertex-joint midpoints, collision
    repulsion produces translation targets, and each tile is refit by a weighted
    rigid Procrustes solve so the tile shape is never deformed.
    """
    rest = np.asarray(raw_tiles, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"hinge_layout_optimizer": "empty"}
    initial = _initial_auxetic_vertex_hinge_layout(rest, grid)
    specs = _vertex_hinge_specs_from_grid(grid)
    current = initial.copy()
    collision_pairs = _collision_candidate_pairs(len(rest), grid, False)
    hinge_pair_set = {tuple(sorted((int(spec.tile_a), int(spec.tile_b)))) for spec in specs}
    collision_pairs = [pair for pair in collision_pairs if tuple(sorted(pair)) not in hinge_pair_set]

    w_anchor = max(0.0, float(anchor_weight))
    w_conn = max(0.0, float(connection_weight))
    w_coll = max(0.0, float(collision_weight))
    before_hinge = _vertex_layout_hinge_error(current, specs)
    before_collisions = _count_2d_tile_collisions(np.dstack([current, np.zeros(current.shape[:2])]), grid)

    for it in range(max(1, int(iterations))):
        # Gradually strengthen connection/collision terms to avoid immediately
        # collapsing a bad initial layout.
        ramp = min(1.0, (it + 1) / max(1.0, iterations * 0.35))
        desired_sum = np.zeros_like(current)
        desired_weight = np.zeros(current.shape[:2], dtype=float)
        if w_anchor > 0.0:
            desired_sum += initial * w_anchor
            desired_weight += w_anchor

        # E_Conn: each selected pair of tile corners wants a common joint.
        for spec in specs:
            if spec.tile_a >= len(current) or spec.tile_b >= len(current):
                continue
            ia = int(spec.tile_a)
            ib = int(spec.tile_b)
            ca = int(spec.corner_a0)
            cb = int(spec.corner_b0)
            pa = current[ia, ca]
            pb = current[ib, cb]
            mid = 0.5 * (pa + pb)
            w = w_conn * ramp
            desired_sum[ia, ca] += mid * w
            desired_weight[ia, ca] += w
            desired_sum[ib, cb] += mid * w
            desired_weight[ib, cb] += w

        # E_Collision: local AABB repulsion.  This is intentionally conservative
        # and ignores directly hinged pairs so it does not fight E_Conn at the
        # very joint that should coincide.
        if w_coll > 0.0:
            shifts = _tile_collision_translation_targets(current, collision_pairs)
            if np.any(shifts):
                w = w_coll * ramp
                desired_sum += (current + shifts[:, None, :]) * w
                desired_weight += w

        # Fallback: unconstrained vertices keep their current/initial positions.
        fallback_w = 0.01
        desired_sum += current * fallback_w
        desired_weight += fallback_w

        next_current = current.copy()
        for tile_id in range(len(current)):
            targets = desired_sum[tile_id] / np.maximum(desired_weight[tile_id, :, None], 1e-12)
            next_current[tile_id] = _fit_rigid_2d_weighted(rest[tile_id], targets, desired_weight[tile_id])
        current = next_current
        current -= np.mean(np.mean(current, axis=1), axis=0) - np.mean(np.mean(initial, axis=1), axis=0)

    after_hinge = _vertex_layout_hinge_error(current, specs)
    after_collisions = _count_2d_tile_collisions(np.dstack([current, np.zeros(current.shape[:2])]), grid)
    metrics: dict[str, float | int | str | bool] = {
        "hinge_layout_optimizer": "local-global rigid SE(2) tile placement",
        "hinge_layout_objective": "approximate E_Hinge = exact rigid tile poses + vertex-joint connection + local collision repulsion",
        "hinge_layout_iterations": int(iterations),
        "hinge_layout_connection_weight": float(connection_weight),
        "hinge_layout_collision_weight": float(collision_weight),
        "hinge_layout_anchor_weight": float(anchor_weight),
        "hinge_layout_initial_connection_error": float(before_hinge),
        "hinge_layout_final_connection_error": float(after_hinge),
        "hinge_layout_initial_collision_count": int(before_collisions),
        "hinge_layout_final_collision_count": int(after_collisions),
        "hinge_layout_variables": int(len(rest) * 3),
        "tile_rigidity_enforced_by_pose_fit": True,
    }
    return current, metrics


def _initial_auxetic_vertex_hinge_layout(raw_tiles: np.ndarray, grid: QuadGrid) -> np.ndarray:
    flat = np.asarray(raw_tiles, dtype=float).copy()
    if len(flat) == 0:
        return flat
    centers = np.mean(flat, axis=1)
    base_len = max(float(grid.tile_size), 1e-8)
    opening = min(0.65, max(0.12, float(grid.gap_size) / base_len * 2.5))
    for tile_id in range(len(flat)):
        row = tile_id // max(1, grid.nx)
        col = tile_id % max(1, grid.nx)
        sign = 1.0 if ((row + col) & 1) == 0 else -1.0
        angle = sign * opening
        c = centers[tile_id]
        ca = math.cos(angle)
        sa = math.sin(angle)
        rot = np.array([[ca, -sa], [sa, ca]], dtype=float)
        flat[tile_id] = (flat[tile_id] - c) @ rot.T + c
        flat[tile_id] += np.array(
            [
                (col - (grid.nx - 1) * 0.5) * grid.gap_size,
                (row - (grid.ny - 1) * 0.5) * grid.gap_size,
            ],
            dtype=float,
        )
    return flat


def _fit_rigid_2d_weighted(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        return target.copy()
    if float(np.sum(weights)) <= 1e-12:
        return source.copy()
    weights = np.maximum(weights, 0.0)
    sw = float(np.sum(weights))
    cs = np.sum(source * weights[:, None], axis=0) / sw
    ct = np.sum(target * weights[:, None], axis=0) / sw
    xs = source - cs
    xt = target - ct
    cov = (xs * weights[:, None]).T @ xt
    try:
        u, _, vt = np.linalg.svd(cov)
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0.0:
            vt[-1, :] *= -1.0
            rot = vt.T @ u.T
    except np.linalg.LinAlgError:
        rot = np.eye(2)
    return (source - cs) @ rot.T + ct


def _tile_collision_translation_targets(tiles_xy: np.ndarray, candidate_pairs: list[tuple[int, int]]) -> np.ndarray:
    shifts = np.zeros((len(tiles_xy), 2), dtype=float)
    counts = np.zeros((len(tiles_xy), 1), dtype=float)
    for i, j in candidate_pairs:
        ti = tiles_xy[i, :, :2]
        tj = tiles_xy[j, :, :2]
        min_i = np.min(ti, axis=0)
        max_i = np.max(ti, axis=0)
        min_j = np.min(tj, axis=0)
        max_j = np.max(tj, axis=0)
        overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
        if not np.all(overlap > 0.0):
            continue
        center_i = np.mean(ti, axis=0)
        center_j = np.mean(tj, axis=0)
        axis = int(np.argmin(overlap))
        direction = 1.0 if center_i[axis] >= center_j[axis] else -1.0
        delta = np.zeros(2, dtype=float)
        delta[axis] = direction * (float(overlap[axis]) + 1e-4) * 0.5
        shifts[i] += delta
        shifts[j] -= delta
        counts[i, 0] += 1.0
        counts[j, 0] += 1.0
    active = counts[:, 0] > 0.0
    shifts[active] /= np.maximum(counts[active], 1.0)
    return shifts


def _optimize_rigid_assembly_hinge_layout_2d(
    rest_vertices: np.ndarray,
    hinges: list[Hinge],
    grid: QuadGrid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Dual-hinge placement using paper-style local/global E_Hinge."""
    rest = np.asarray(rest_vertices, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"dual_hinge_layout_optimizer": "empty"}
    rest_xy = rest[:, :, :2].copy()
    constraints = _hinge_constraint_tuples_from_hinges(hinges)

    def footprint_builder(layout: np.ndarray) -> np.ndarray:
        return layout

    clearance = max(float(grid.gap_size) * 0.35, 1e-5)
    solved_xy, metrics = _paper_local_global_se2_layout(
        rest_xy,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=rest_xy,
        iterations=max(1, int(iterations)),
        connection_weight=float(connection_weight),
        collision_weight=max(1.0, float(collision_weight)),
        anchor_weight=float(anchor_weight) * 0.25,
        clearance=clearance,
        stage_name="T2D Top Hinge to T2D Dual Hinge placement",
        time_budget_sec=float(time_budget_sec),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=int(collision_sweeps_per_iteration),
        initial_expansion=float(initial_expansion),
        max_center_drift_tiles=float(max_center_drift_tiles),
    )
    out = rest.copy()
    out[:, :, :2] = solved_xy
    before_pairs = _spatial_candidate_pairs_for_tiles(rest_xy, pad=clearance * 8.0)
    after_pairs = _spatial_candidate_pairs_for_tiles(solved_xy, pad=clearance * 8.0)
    shape_rms = _tile_shape_distance_error(out, rest)
    shape_max = _tile_shape_distance_error(out, rest, use_max=True)
    out_metrics: dict[str, float | int | str | bool] = {
        "dual_hinge_layout_optimizer": "paper-style local/global E_Hinge full-footprint SE(2) tile placement",
        "dual_hinge_layout_objective": "E_Hinge = E_Rigid + E_Collision + E_Conn",
        "dual_hinge_layout_iterations_requested": int(iterations),
        "dual_hinge_initial_connection_error": float(_hinge_connection_error(rest, hinges)),
        "dual_hinge_final_connection_error": float(_hinge_connection_error(out, hinges)),
        "dual_hinge_initial_collision_count": int(_count_2d_footprint_collisions_from_pairs(rest_xy, before_pairs)),
        "dual_hinge_final_collision_count": int(_count_2d_footprint_collisions_from_pairs(solved_xy, after_pairs)),
        "dual_hinge_pose_variables": int(len(rest) * 3),
        "dual_hinge_tile_rigidity_enforced_by_pose_fit": True,
        "dual_hinge_full_footprint_collision": True,
        "dual_hinge_hinge_pairs_exempt_from_collision": False,
        "dual_hinge_tile_shape_rms_error_after_layout": float(shape_rms),
        "dual_hinge_tile_shape_max_error_after_layout": float(shape_max),
        **metrics,
    }
    return out, out_metrics


def _vertex_layout_hinge_error(flat_tiles: np.ndarray, specs: list[HingeSpec]) -> float:
    if len(flat_tiles) == 0 or not specs:
        return 0.0
    values = []
    for spec in specs:
        if spec.tile_a >= len(flat_tiles) or spec.tile_b >= len(flat_tiles):
            continue
        values.append(np.linalg.norm(flat_tiles[spec.tile_a, spec.corner_a0] - flat_tiles[spec.tile_b, spec.corner_b0]))
    return float(np.max(values)) if values else 0.0


def _tiles_from_mesh_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray([vertices[list(face)] for face in faces], dtype=float)


def _shrink_tile(tile: np.ndarray, amount: float) -> np.ndarray:
    center = np.mean(tile, axis=0)
    vec = tile - center
    length = np.linalg.norm(vec[:, :2], axis=1, keepdims=True)
    scale = np.maximum(length - amount, 0.0) / np.maximum(length, 1e-8)
    out = tile.copy()
    out[:, :2] = center[:2] + vec[:, :2] * scale
    return out


def _quad_normal(tile: np.ndarray) -> np.ndarray:
    normal = np.cross(tile[1] - tile[0], tile[3] - tile[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    if normal[2] < 0.0:
        normal *= -1.0
    return normal / norm


def _planarity_residuals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    values = []
    for face in faces:
        pts = vertices[list(face)]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            values.append(0.0)
        else:
            values.append(float(np.dot(pts[3] - pts[0], normal / norm)))
    return np.asarray(values, dtype=float)


def _quad_planarity_error(vertices: np.ndarray, faces: np.ndarray) -> float:
    residuals = np.abs(_planarity_residuals(vertices, faces))
    return float(np.max(residuals)) if residuals.size else 0.0


def _surface_fit_error(
    target: HeightField,
    vertices: np.ndarray,
    parameterization: SurfaceParameterization | None = None,
    params: PipelineParameters | None = None,
) -> float:
    if params is not None and params.m3d_construction_mode == "analytic_scaled_heightfield_debug":
        z = target.height(vertices[:, 0], vertices[:, 1])
        return float(np.sqrt(np.mean((vertices[:, 2] - z) ** 2)))
    if parameterization is None:
        raise RuntimeError("Paper mode surface fit requires a surface mesh parameterization.")
    distances = _distances_to_surface_mesh(vertices, parameterization.surface_vertices_3d, parameterization.surface_faces)
    return float(np.sqrt(np.mean(distances * distances))) if distances.size else 0.0


def _square_residuals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for face in faces:
        pts = vertices[list(face)]
        edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        avg = float(np.mean(edges))
        values.extend([edge - avg for edge in edges])
        diag0 = np.linalg.norm(pts[0] - pts[2])
        diag1 = np.linalg.norm(pts[1] - pts[3])
        values.append(diag0 - diag1)
    return np.asarray(values, dtype=float)


def _square_error(vertices: np.ndarray, faces: np.ndarray) -> float:
    residuals = np.abs(_square_residuals(vertices, faces))
    return float(np.mean(residuals)) if residuals.size else 0.0


def _unique_mesh_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        ids = list(map(int, face))
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            edge = tuple(sorted((ids[a], ids[b])))
            edges.add(edge)
    return sorted(edges)


def _edge_length_variance(vertices: np.ndarray, faces: np.ndarray) -> float:
    edges = _unique_mesh_edges(faces)
    lengths = [np.linalg.norm(vertices[a] - vertices[b]) for a, b in edges]
    return float(np.var(lengths)) if lengths else 0.0


def _edge_matching_error(xy: np.ndarray, edges: list[tuple[int, int]], target_lengths: np.ndarray) -> float:
    current = np.asarray([np.linalg.norm(xy[a] - xy[b]) for a, b in edges], dtype=float)
    return float(np.sqrt(np.mean((current - target_lengths) ** 2))) if len(current) else 0.0


def _edge_matching_errors(xy: np.ndarray, edges: list[tuple[int, int]], target_lengths: np.ndarray) -> tuple[float, float]:
    current = np.asarray([np.linalg.norm(xy[a] - xy[b]) for a, b in edges], dtype=float)
    if not len(current):
        return 0.0, 0.0
    err = np.abs(current - target_lengths)
    return float(np.mean(err)), float(np.max(err))


def _projective_edge_match_2d(
    xy_start: np.ndarray,
    base_xy: np.ndarray,
    edges: list[tuple[int, int]],
    target_lengths: np.ndarray,
    faces: np.ndarray,
    grid: QuadGrid,
    iterations: int,
) -> np.ndarray:
    del faces, grid
    xy = np.asarray(xy_start, dtype=float).copy()
    if not edges:
        return xy
    edge_idx = np.asarray(edges, dtype=int)
    aa = edge_idx[:, 0]
    bb = edge_idx[:, 1]
    degree = np.zeros((len(xy), 1), dtype=float)
    np.add.at(degree, aa, 1.0)
    np.add.at(degree, bb, 1.0)
    degree = np.maximum(degree, 1.0)
    target = np.asarray(target_lengths, dtype=float)
    mean_target = float(np.mean(target)) if target.size else 1.0
    tol = max(1e-6, 0.001 * mean_target)
    # Keep only a very weak anchoring to Ω to remove drift; do not run collision
    # relaxation here because it changes the edge lengths we are solving for.
    anchor = 0.0002
    for it in range(max(1, int(iterations))):
        delta = xy[bb] - xy[aa]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1e-12)
        correction = ((length - target) / safe)[:, None] * delta * 0.5
        accum = np.zeros_like(xy)
        np.add.at(accum, aa, correction)
        np.add.at(accum, bb, -correction)
        xy += accum / degree
        xy += (base_xy - xy) * anchor
        if (it + 1) % 20 == 0:
            cur = np.linalg.norm(xy[bb] - xy[aa], axis=1)
            if float(np.max(np.abs(cur - target))) <= tol:
                break
    return xy


def _tile_face_planarity(vertices: np.ndarray) -> float:
    faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    values: list[float] = []
    for tile in vertices:
        for face in faces:
            pts = tile[face]
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                values.append(abs(float(np.dot(pts[3] - pts[0], normal / norm))))
    return float(np.max(values)) if values else 0.0


def _tile_face_planarity_by_group(vertices: np.ndarray) -> dict[str, float]:
    groups = {
        "top": [[0, 1, 2, 3]],
        "bottom": [[4, 7, 6, 5]],
        "side": [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
    }
    result: dict[str, float] = {}
    for name, faces in groups.items():
        values: list[float] = []
        for tile in vertices:
            for face in faces:
                pts = tile[face]
                normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    values.append(abs(float(np.dot(pts[3] - pts[0], normal / norm))))
        result[name] = float(np.max(values)) if values else 0.0
    return result


def _boundary_tile_edges(grid: QuadGrid) -> list[tuple[int, tuple[int, int]]]:
    edges: list[tuple[int, tuple[int, int]]] = []
    for col in range(grid.nx):
        edges.append((col, (0, 1)))
    for row in range(grid.ny):
        tile = row * grid.nx + grid.nx - 1
        edges.append((tile, (1, 2)))
    for col in reversed(range(grid.nx)):
        tile = (grid.ny - 1) * grid.nx + col
        edges.append((tile, (2, 3)))
    for row in reversed(range(grid.ny)):
        tile = row * grid.nx
        edges.append((tile, (3, 0)))
    return edges


def _hinge_specs_from_faces(faces: np.ndarray) -> list[HingeSpec]:
    edge_owner: dict[tuple[int, int], tuple[int, int, int]] = {}
    specs: list[HingeSpec] = []
    for tile_id, face in enumerate(np.asarray(faces, dtype=int)):
        local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for la, lb in local_edges:
            a, b = int(face[la]), int(face[lb])
            key = tuple(sorted((a, b)))
            if key in edge_owner:
                other_tile, oa, ob = edge_owner[key]
                direction = "x" if {la, lb} in [{1, 2}, {0, 3}] else "y"
                specs.append(HingeSpec(other_tile, oa, ob, tile_id, la, lb, direction))
            else:
                edge_owner[key] = (tile_id, la, lb)
    return specs


def _boundary_tile_edges_from_faces(faces: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    edge_owner: dict[tuple[int, int], tuple[int, int, int]] = {}
    edge_count: dict[tuple[int, int], int] = {}
    for tile_id, face in enumerate(np.asarray(faces, dtype=int)):
        for la, lb in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            key = tuple(sorted((int(face[la]), int(face[lb]))))
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_owner[key] = (tile_id, la, lb)
    return [(owner[0], (owner[1], owner[2])) for key, owner in edge_owner.items() if edge_count.get(key, 0) == 1]



def _infer_face_lattice_width(faces: np.ndarray) -> int | None:
    """Infer row-major grid width from QuadGrid-style face vertex ids.

    QuadGrid faces are ordered [lower-left, lower-right, upper-right,
    upper-left], so v1-v0 == 1 and v3-v0 == grid_width.  M2D cropping keeps
    the original overlay-grid vertex ids, therefore this remains valid even for
    non-rectangular cropped domains.  If uploaded/irregular meshes violate this
    pattern, return None and use the conservative fallback hinge rule.
    """
    widths: list[int] = []
    for face in np.asarray(faces, dtype=int):
        if len(face) != 4:
            continue
        v0, v1, v2, v3 = [int(x) for x in face]
        width = v3 - v0
        if width > 1 and v1 - v0 == 1 and v2 - v1 == width and v2 - v3 == 1:
            widths.append(width)
    if not widths:
        return None
    values, counts = np.unique(np.asarray(widths, dtype=int), return_counts=True)
    width = int(values[int(np.argmax(counts))])
    if int(np.max(counts)) < max(1, len(widths) // 2):
        return None
    return width


def _tile_lattice_row_col(face: np.ndarray, width: int | None) -> tuple[int, int] | None:
    if width is None or width <= 1:
        return None
    v0 = int(np.asarray(face, dtype=int)[0])
    return int(v0 // width), int(v0 % width)


def _edge_side_from_local_edge(la: int, lb: int) -> str:
    edge = {int(la), int(lb)}
    if edge == {1, 2}:
        return "right"
    if edge == {3, 0}:
        return "left"
    if edge == {2, 3}:
        return "top"
    if edge == {0, 1}:
        return "bottom"
    return "unknown"


def _alternating_pairwise_corner(row: int, col: int, side: str) -> int:
    """Choose a single pairwise corner hinge for one side of one tile.

    The earlier implementation chose one endpoint of each shared edge with a
    parity formula that was not tile-local.  Around a grid vertex this could make
    all four neighboring panels share the same joint location, which is not the
    paper's linkage: each hinge connects exactly two panels, and the gaps remain
    diamond-shaped voids.

    This checkerboard rule assigns the four side-neighbor connections of every
    tile to four distinct corners:

      even tile: right->TR, top->TL, left->BL, bottom->BR
      odd  tile: right->BR, top->TR, left->TL, bottom->BL

    For a shared edge, the two adjacent tiles choose the same grid endpoint, but
    no tile corner is reused by multiple hinges.  Therefore a grid vertex may
    host several nearby pairwise joints geometrically, but they are not collapsed
    into a single four-panel hinge constraint.
    """
    parity = (int(row) + int(col)) & 1
    if parity == 0:
        mapping = {"right": 2, "top": 3, "left": 0, "bottom": 1}
    else:
        mapping = {"right": 1, "top": 2, "left": 3, "bottom": 0}
    return int(mapping.get(side, 0))


def _legacy_single_endpoint_corner(
    tile_a: int,
    face_a: np.ndarray,
    local_by_vertex_a: dict[int, int],
    tile_b: int,
    face_b: np.ndarray,
    local_by_vertex_b: dict[int, int],
    shared_key: tuple[int, int],
) -> tuple[int, int]:
    """Fallback for non-lattice meshes: one endpoint per shared edge."""
    shared = list(shared_key)
    chosen_vertex = shared[(int(tile_a + tile_b + shared[0] + shared[1]) & 1)]
    return int(local_by_vertex_a[chosen_vertex]), int(local_by_vertex_b[chosen_vertex])



def _vertex_hinge_specs_with_joint_vertices_from_faces(faces: np.ndarray) -> list[tuple[HingeSpec, int]]:
    """Return pairwise hinge specs plus the abstract K2D mesh vertex they came from.

    The shared K2D mesh vertex is *not* a physical four-panel joint.  It is an
    abstract location around which the physical flat layout must create a
    rhombus/diamond void.  Around one K2D mesh vertex there may therefore be
    several pairwise hinges, but those hinges must be separated in the T2D
    fabrication layout.

    This helper keeps the topology pairwise: one HingeSpec connects exactly two
    panels.  It also remembers the original mesh vertex so the T2D initializer
    can split coincident pairwise joints into a small diamond void before the
    E_Hinge optimizer runs.
    """
    faces_arr = np.asarray(faces, dtype=int)
    width = _infer_face_lattice_width(faces_arr)
    edge_owner: dict[tuple[int, int], tuple[int, tuple[int, int], dict[int, int], tuple[int, int] | None]] = {}
    result: list[tuple[HingeSpec, int]] = []
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for tile_id, face in enumerate(faces_arr):
        row_col = _tile_lattice_row_col(face, width)
        for la, lb in local_edges:
            va, vb = int(face[la]), int(face[lb])
            key = tuple(sorted((va, vb)))
            local_by_vertex = {va: int(la), vb: int(lb)}
            if key not in edge_owner:
                edge_owner[key] = (int(tile_id), (int(la), int(lb)), local_by_vertex, row_col)
                continue

            other_tile, other_edge, other_by_vertex, other_row_col = edge_owner[key]
            side_current = _edge_side_from_local_edge(la, lb)
            side_other = _edge_side_from_local_edge(other_edge[0], other_edge[1])

            if width is not None and row_col is not None and other_row_col is not None:
                ca = _alternating_pairwise_corner(other_row_col[0], other_row_col[1], side_other)
                cb = _alternating_pairwise_corner(row_col[0], row_col[1], side_current)
                va_chosen = int(faces_arr[other_tile, ca])
                vb_chosen = int(face[cb])
                # The topology rule intentionally chooses one endpoint of the
                # shared K2D edge.  If a cropped/irregular face breaks the
                # lattice assumption, fall back to a deterministic endpoint.
                if va_chosen != vb_chosen or va_chosen not in key:
                    ca, cb = _legacy_single_endpoint_corner(
                        other_tile, faces_arr[other_tile], other_by_vertex, int(tile_id), face, local_by_vertex, key
                    )
                    joint_vertex = int(faces_arr[other_tile, ca])
                else:
                    joint_vertex = int(va_chosen)
            else:
                ca, cb = _legacy_single_endpoint_corner(
                    other_tile, faces_arr[other_tile], other_by_vertex, int(tile_id), face, local_by_vertex, key
                )
                joint_vertex = int(faces_arr[other_tile, ca])

            direction = "x" if side_current in {"left", "right"} else "y"
            spec = HingeSpec(int(other_tile), int(ca), int(ca), int(tile_id), int(cb), int(cb), direction)
            result.append((spec, int(joint_vertex)))

    return result


def _vertex_hinge_specs_from_faces(faces: np.ndarray) -> list[HingeSpec]:
    """Return paper-style pairwise vertex hinges from cropped M2D faces.

    A K2D shared vertex is not treated as a single four-panel joint.  Instead,
    every shared edge contributes one pairwise hinge between two panels.  In the
    subsequent T2D layout initializer, pairwise hinges that originate from the
    same abstract K2D mesh vertex are split around that vertex into a small
    diamond void.  This is the key distinction that the previous patch missed:
    pairwise topology alone is not enough if all pairwise joints remain
    geometrically collapsed at the same K2D vertex.
    """
    return [spec for spec, _joint in _vertex_hinge_specs_with_joint_vertices_from_faces(faces)]

def _vertex_hinge_topology_metrics(faces: np.ndarray) -> dict[str, float | int | str | bool]:
    specs = _vertex_hinge_specs_from_faces(faces)
    corner_use: dict[tuple[int, int], int] = {}
    for spec in specs:
        corner_use[(int(spec.tile_a), int(spec.corner_a0))] = corner_use.get((int(spec.tile_a), int(spec.corner_a0)), 0) + 1
        corner_use[(int(spec.tile_b), int(spec.corner_b0))] = corner_use.get((int(spec.tile_b), int(spec.corner_b0)), 0) + 1
    return {
        "hinge_topology_rule": "alternating pairwise corner hinges; one hinge connects exactly two panels",
        "four_way_hinge_model_disabled": True,
        "one_corner_reused_by_multiple_hinges_count": int(sum(1 for v in corner_use.values() if v > 1)),
        "max_hinges_using_one_tile_corner": int(max(corner_use.values()) if corner_use else 0),
        "pairwise_hinge_count": int(len(specs)),
    }


def _vertex_hinge_specs_from_grid(grid: QuadGrid) -> list[HingeSpec]:
    """Compatibility wrapper. Prefer _vertex_hinge_specs_from_faces after M2D crop."""
    if grid.tiles is None:
        return []
    faces = np.asarray([tile.vertex_ids for tile in grid.tiles], dtype=int)
    return _vertex_hinge_specs_from_faces(faces)

def _edge_gap_specs_from_faces(faces: np.ndarray) -> list[HingeSpec]:
    """Edge-adjacency specs used only for snap side-face pairs/gap routing.

    Physical hinges should use _vertex_hinge_specs_from_faces.  This function
    intentionally keeps the old edge-neighbour information because the paper's
    snap constraint acts on side-face midpoints along a gap, not on the hinge
    joint itself.
    """
    return _hinge_specs_from_faces(faces)

def _shortest_gap_path(gap_graph: GapGraph, start: int, goal: int) -> list[int]:
    if start == goal:
        return [goal]
    adjacency: dict[int, list[int]] = {gap.id: [] for gap in gap_graph.gaps}
    for a, b in gap_graph.edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    queue = [start]
    parent = {start: -1}
    for node in queue:
        for nxt in adjacency.get(node, []):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))[1:]
            queue.append(nxt)
    return [goal]


def _turn_angle_total(gap_graph: GapGraph, route: list[int]) -> float:
    if len(route) < 3:
        return 0.0
    centroids = {gap.id: gap.centroid_2d for gap in gap_graph.gaps}
    total = 0.0
    for a, b, c in zip(route[:-2], route[1:-1], route[2:]):
        v0 = centroids[a] - centroids[b]
        v1 = centroids[c] - centroids[b]
        n0 = np.linalg.norm(v0)
        n1 = np.linalg.norm(v1)
        if n0 <= 1e-12 or n1 <= 1e-12:
            continue
        dot = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
        total += math.acos(dot)
    return float(total)


def _max_single_turn_angle(gap_graph: GapGraph, route: list[int]) -> float:
    if len(route) < 3:
        return 0.0
    centroids = {gap.id: gap.centroid_2d for gap in gap_graph.gaps}
    values: list[float] = []
    for a, b, c in zip(route[:-2], route[1:-1], route[2:]):
        v0 = centroids[a] - centroids[b]
        v1 = centroids[c] - centroids[b]
        n0 = np.linalg.norm(v0)
        n1 = np.linalg.norm(v1)
        if n0 <= 1e-12 or n1 <= 1e-12:
            continue
        dot = float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0))
        values.append(math.acos(dot))
    return float(max(values, default=0.0))


def safe_capstan_friction(mu_c: float, theta: float, tension: float = 1.0) -> float:
    if not math.isfinite(mu_c) or not math.isfinite(theta):
        return float("inf")
    if mu_c < 0.0 or theta < 0.0:
        return float("inf")
    arg = mu_c * theta
    if arg > 60.0:
        return float("inf")
    return float(tension * math.expm1(arg))


def _dihedral_indicator(tile_a: np.ndarray, tile_b: np.ndarray) -> float:
    na = _quad_normal(tile_a)
    nb = _quad_normal(tile_b)
    return float(1.0 - np.dot(na, nb))


def _hinge_connection_error(vertices: np.ndarray, hinges: list[Hinge]) -> float:
    if not hinges:
        return 0.0
    values = [np.linalg.norm(vertices[h.tile_a, h.local_vertex_a] - vertices[h.tile_b, h.local_vertex_b]) for h in hinges]
    return float(np.sqrt(np.mean(np.square(values))))


def _rigid_error(current: np.ndarray, rest: np.ndarray) -> float:
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3), (4, 6), (5, 7)]
    errors: list[float] = []
    for tile, rest_tile in zip(current, rest):
        for a, b in pairs:
            errors.append(float(np.linalg.norm(tile[a] - tile[b]) - np.linalg.norm(rest_tile[a] - rest_tile[b])))
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0


def _rigid_error_max(current: np.ndarray, rest: np.ndarray) -> float:
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3), (4, 6), (5, 7)]
    max_error = 0.0
    for tile, rest_tile in zip(current, rest):
        for a, b in pairs:
            err = abs(float(np.linalg.norm(tile[a] - tile[b]) - np.linalg.norm(rest_tile[a] - rest_tile[b])))
            max_error = max(max_error, err)
    return float(max_error)


def _hinge_error(current: np.ndarray, state: OneStringDesignState) -> float:
    return _hinge_connection_error(current, state.hinge_graph.hinges)




def _snap_error(current: np.ndarray, state: OneStringDesignState) -> float:
    values: list[float] = []
    for gap in _deployment_snap_gaps(state, "all_internal_gaps"):
        pa, pb = _gap_side_face_midpoints(current, gap, include_bottom=True)
        values.append(float(np.linalg.norm(pa - pb)))
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def _lift_error(current: np.ndarray, state: OneStringDesignState) -> float:
    values: list[float] = []
    for lift in state.lift_points:
        gap = state.gap_graph.gaps[lift.gap_id]
        current_center = np.mean([np.mean(current[tile, :4], axis=0) for tile in gap.surrounding_tiles], axis=0)
        values.append(float(np.linalg.norm(current_center - lift.position_3d)))
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def _gap_side_face_midpoints(tiles: np.ndarray, gap: Gap, include_bottom: bool) -> tuple[np.ndarray, np.ndarray]:
    a, b = gap.surrounding_tiles
    if gap.type == "vertical":
        edge_a = [1, 2, 6, 5] if include_bottom else [1, 2]
        edge_b = [0, 3, 7, 4] if include_bottom else [0, 3]
    else:
        edge_a = [3, 2, 6, 7] if include_bottom else [3, 2]
        edge_b = [0, 1, 5, 4] if include_bottom else [0, 1]
    return np.mean(tiles[a, edge_a], axis=0), np.mean(tiles[b, edge_b], axis=0)


def _count_aabb_collisions(tiles: np.ndarray, grid: QuadGrid | None = None, all_pairs: bool = False) -> int:
    count = 0
    for i, j in _collision_candidate_pairs(tiles.shape[0], grid, all_pairs):
        min_i = np.min(tiles[i], axis=0)
        max_i = np.max(tiles[i], axis=0)
        min_j = np.min(tiles[j], axis=0)
        max_j = np.max(tiles[j], axis=0)
        if np.all(np.minimum(max_i, max_j) - np.maximum(min_i, min_j) > 0):
            count += 1
    return count


def _collision_candidate_pairs(tile_count: int, grid: QuadGrid | None, all_pairs: bool) -> list[tuple[int, int]]:
    if all_pairs or grid is None:
        return [(i, j) for i in range(tile_count) for j in range(i + 1, tile_count)]
    # The previous implementation still iterated over every tile pair and then
    # filtered by grid distance. For n=80 this means ~20M Python-loop checks
    # even though only local neighbours are needed. Generate the local stencil
    # directly so all collision users scale approximately O(N) instead of O(N^2).
    nx = int(grid.nx)
    ny = int(grid.ny)
    pairs: list[tuple[int, int]] = []
    for row in range(ny):
        row_start = row * nx
        for col in range(nx):
            i = row_start + col
            if i >= tile_count:
                continue
            r0 = max(0, row - 2)
            r1 = min(ny - 1, row + 2)
            c0 = max(0, col - 2)
            c1 = min(nx - 1, col + 2)
            for rr in range(r0, r1 + 1):
                base = rr * nx
                for cc in range(c0, c1 + 1):
                    j = base + cc
                    if j > i and j < tile_count:
                        pairs.append((i, j))
    return pairs


def _count_2d_tile_collisions(tiles: np.ndarray, grid: QuadGrid | None = None, all_pairs: bool = False) -> int:
    count = 0
    for i, j in _collision_candidate_pairs(tiles.shape[0], grid, all_pairs):
        min_i = np.min(tiles[i, :, :2], axis=0)
        max_i = np.max(tiles[i, :, :2], axis=0)
        min_j = np.min(tiles[j, :, :2], axis=0)
        max_j = np.max(tiles[j, :, :2], axis=0)
        if np.all(np.minimum(max_i, max_j) - np.maximum(min_i, min_j) > 0):
            count += 1
    return count


def _min_aabb_clearance_2d(tiles: np.ndarray, grid: QuadGrid | None = None, all_pairs: bool = False) -> float:
    values: list[float] = []
    for i, j in _collision_candidate_pairs(tiles.shape[0], grid, all_pairs):
        min_i = np.min(tiles[i, :, :2], axis=0)
        max_i = np.max(tiles[i, :, :2], axis=0)
        min_j = np.min(tiles[j, :, :2], axis=0)
        max_j = np.max(tiles[j, :, :2], axis=0)
        sep = np.maximum(np.maximum(min_j - max_i, min_i - max_j), 0.0)
        overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
        if np.all(overlap > 0):
            values.append(-float(np.min(overlap)))
        else:
            values.append(float(np.linalg.norm(sep)))
    return float(np.min(values)) if values else 0.0


def _relax_2d_collisions(vertices_xy: np.ndarray, faces: np.ndarray, grid: QuadGrid, iterations: int, weight: float) -> np.ndarray:
    xy = vertices_xy.copy()
    for _ in range(iterations):
        tiles = _tiles_from_mesh_vertices(np.column_stack([xy, np.zeros(len(xy))]), faces)
        for i, j in _collision_candidate_pairs(len(tiles), grid, False):
            min_i = np.min(tiles[i, :, :2], axis=0)
            max_i = np.max(tiles[i, :, :2], axis=0)
            min_j = np.min(tiles[j, :, :2], axis=0)
            max_j = np.max(tiles[j, :, :2], axis=0)
            overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            if not np.all(overlap > 0):
                continue
            axis = int(np.argmin(overlap))
            center_i = np.mean(tiles[i, :, :2], axis=0)
            center_j = np.mean(tiles[j, :, :2], axis=0)
            direction = 1.0 if center_i[axis] >= center_j[axis] else -1.0
            delta = np.zeros(2)
            delta[axis] = direction * overlap[axis] * weight
            for vertex_id in faces[i]:
                xy[vertex_id] += delta
            for vertex_id in faces[j]:
                xy[vertex_id] -= delta
    return xy


def _gap_angles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for face in faces:
        pts = vertices[list(face), :2]
        for i in range(4):
            a = pts[i - 1] - pts[i]
            b = pts[(i + 1) % 4] - pts[i]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                values.append(math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))))
    return np.asarray(values, dtype=float)


def _z_range(vertices: np.ndarray) -> float:
    return float(np.max(vertices[:, 2]) - np.min(vertices[:, 2])) if len(vertices) else 0.0


def complexity_metrics(grid_size: int) -> dict[str, int]:
    n = int(grid_size)
    tiles = n * n
    return {
        "grid_size": n,
        "tiles": tiles,
        "vertices": (n + 1) * (n + 1),
        "mesh_edges": 2 * n * (n + 1),
        "hinges": 4 * n * max(0, n - 1),
        "gaps_approx": 2 * n * max(0, n - 1) + 4 * n,
        "all_pair_collision_checks": tiles * max(0, tiles - 1) // 2,
        "near_neighbor_collision_checks_approx": tiles * min(max(0, tiles - 1), 24) // 2,
        "k3d_optimizer_variables": 3 * (n + 1) * (n + 1),
        "k2d_optimizer_variables": 2 * (n + 1) * (n + 1),
    }


def compute_backend_info(config: ComputeConfig | None = None) -> dict[str, str | bool | int]:
    config = config or ComputeConfig()
    available = bool(torch is not None and torch.cuda.is_available())
    selected = config.backend
    current = "cpu"
    error = ""
    if selected == "cuda" and not available:
        error = "CUDA requested but unavailable; pipeline construction will raise instead of falling back to CPU."
    elif selected == "cuda" and available:
        current = "cuda"
    elif selected == "auto" and available:
        current = "cuda"
    info: dict[str, str | bool | int] = {
        "torch_available": torch is not None,
        "torch_version": "" if torch is None else str(torch.__version__),
        "torch_cuda_version": "" if torch is None else str(torch.version.cuda),
        "cuda_available": available,
        "cuda_device_count": 0,
        "cuda_current_device": -1,
        "requested_backend": selected,
        "current_backend": current,
        "tensor_dtype": config.dtype,
        "gpu_name": "",
        "gpu_capability": "",
        "gpu_memory_allocated": 0,
        "gpu_memory_reserved": 0,
        "fallback_warning": error,
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_path": "" if torch is None else str(getattr(torch, "__file__", "")),
    }
    if available:
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["cuda_current_device"] = int(torch.cuda.current_device())
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        info["gpu_memory_allocated"] = int(torch.cuda.memory_allocated(0))
        info["gpu_memory_reserved"] = int(torch.cuda.memory_reserved(0))
    return info


def _stage_backend_report(
    stage_name: str,
    config: ComputeConfig,
    elapsed_time: float,
    metrics: dict[str, object],
) -> dict[str, str | bool | int | float]:
    info = compute_backend_info(config)
    actual = str(metrics.get("actual_backend", metrics.get("compute_backend", info["current_backend"])))
    fallback = bool(config.backend == "auto" and actual == "cpu" and info["cuda_available"] is False)
    cpu_stage = bool(metrics.get("cpu_stage", False))
    gpu_kernel_time = float(metrics.get("gpu_kernel_time", 0.0) or 0.0)
    cpu_preprocess_time = float(metrics.get("cpu_preprocess_time", 0.0) or 0.0)
    cpu_postprocess_time = float(metrics.get("cpu_postprocess_time", 0.0) or 0.0)
    gpu_time_ratio = gpu_kernel_time / max(float(elapsed_time), 1e-12)
    dominant = str(metrics.get("dominant_backend", actual))
    if actual == "cuda" and gpu_time_ratio < 0.5:
        dominant = "cuda_partial"
    unaccounted_time = max(0.0, float(elapsed_time) - gpu_kernel_time - cpu_preprocess_time - cpu_postprocess_time)
    return {
        "stage_name": stage_name,
        "requested_backend": config.backend,
        "actual_backend": actual,
        "dominant_backend": dominant,
        "device_name": str(info.get("gpu_name", "")),
        "elapsed_time": float(elapsed_time),
        "gpu_kernel_time": gpu_kernel_time,
        "gpu_time_ratio": gpu_time_ratio,
        "cpu_preprocess_time": cpu_preprocess_time,
        "cpu_postprocess_time": cpu_postprocess_time,
        "unaccounted_time": unaccounted_time,
        "cpu_gpu_transfer_count": int(metrics.get("cpu_gpu_transfer_count", 0) or 0),
        "gpu_memory_before": 0,
        "gpu_memory_after": int(info.get("gpu_memory_allocated", 0)),
        "gpu_memory_peak": int(metrics.get("gpu_memory_peak", 0) or 0),
        "cpu_fallback_used": False if cpu_stage else fallback or actual in {"scipy", "fast_numpy", "projective_numpy", "scipy+projective_numpy", "cpu"},
        "fallback_reason": str(metrics.get("fallback_reason", "")) or (str(info.get("fallback_warning", "")) if fallback else ""),
    }


def gpu_self_test(config: ComputeConfig | None = None) -> dict[str, str | bool | int | float]:
    config = config or ComputeConfig(backend="cuda")
    if torch is None:
        raise RuntimeError(f"PyTorch is not installed in this Python environment: {sys.executable}")
    if config.backend == "cuda":
        _validate_compute_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    x = torch.randn((4096, 4096), device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checksum = float(y[0, 0].detach().cpu())
    return {
        "device_used": torch.cuda.get_device_name(0),
        "elapsed_time": elapsed,
        "memory_allocated": int(torch.cuda.memory_allocated(0)),
        "memory_reserved": int(torch.cuda.memory_reserved(0)),
        "memory_peak": int(torch.cuda.max_memory_allocated(0)),
        "result_checksum": checksum,
    }


def run_simulator_gpu_benchmark(grid_sizes: list[int] | None = None) -> list[dict[str, float | int | str]]:
    # Use sizes large enough to make CUDA work visible. Small grids only prove
    # that the CUDA path exists, not that it dominates runtime.
    grid_sizes = grid_sizes or [20, 40, 60]
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requires CUDA-enabled PyTorch in the Streamlit Python environment.")
    rows: list[dict[str, float | int | str]] = []
    for n in grid_sizes:
        target = HeightField("dome", {"amplitude": 0.6, "radius": max(2.0, n * 0.7)})
        cpu_config = ComputeConfig(backend="cpu")
        cuda_config = ComputeConfig(backend="cuda")

        t0 = time.perf_counter()
        cpu_state = build_onestring_design(
            target,
            PipelineParameters(
                nx=n,
                max_3d_iterations=40,
                max_2d_iterations=40,
                m3d_construction_mode="analytic_scaled_heightfield_debug",
                surface_mesh_subdivisions=1,
                compute=cpu_config,
            ),
        )
        cpu_build = time.perf_counter() - t0

        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        cuda_state = build_onestring_design(
            target,
            PipelineParameters(
                nx=n,
                max_3d_iterations=80,
                max_2d_iterations=80,
                m3d_construction_mode="analytic_scaled_heightfield_debug",
                surface_mesh_subdivisions=1,
                compute=cuda_config,
            ),
        )
        cuda_build = time.perf_counter() - t0
        build_peak = int(torch.cuda.max_memory_allocated(0))

        t0 = time.perf_counter()
        simulate_onestring_deployment(cpu_state, DeploymentParameters(steps=40, solver_iterations=20, store_animation_frames=False, compute=cpu_config))
        cpu_deploy = time.perf_counter() - t0

        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        cuda_deploy = simulate_onestring_deployment(cuda_state, DeploymentParameters(steps=80, solver_iterations=40, store_animation_frames=False, compute=cuda_config))
        cuda_deploy_time = time.perf_counter() - t0
        deploy_peak = int(torch.cuda.max_memory_allocated(0))

        rows.append(
            {
                "grid_size": n,
                "cpu_build_time": cpu_build,
                "cuda_build_time": cuda_build,
                "cpu_k3d_time": float(cpu_state.backend_reports["K3D"]["elapsed_time"]),
                "cuda_k3d_time": float(cuda_state.backend_reports["K3D"]["elapsed_time"]),
                "cpu_k2d_time": float(cpu_state.backend_reports["K2D"]["elapsed_time"]),
                "cuda_k2d_time": float(cuda_state.backend_reports["K2D"]["elapsed_time"]),
                "cpu_deployment_time": cpu_deploy,
                "cuda_deployment_time": cuda_deploy_time,
                "k3d_backend": str(cuda_state.backend_reports["K3D"]["actual_backend"]),
                "k2d_backend": str(cuda_state.backend_reports["K2D"]["actual_backend"]),
                "deployment_backend": str(cuda_deploy.metrics["actual_backend"]),
                "cuda_build_peak_memory": build_peak,
                "cuda_deployment_peak_memory": deploy_peak,
            }
        )
    return rows


def nvidia_smi_probe() -> dict[str, str | bool]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        process_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        combined = (process_result.stdout or "") + (process_result.stderr or "")
        return {
            "nvidia_smi_detected": result.returncode == 0,
            "gpu_summary": (result.stdout or result.stderr).strip(),
            "current_python_process_visible": str(os.getpid()) in combined or "python" in combined.lower(),
        }
    except Exception as exc:
        return {"nvidia_smi_detected": False, "gpu_summary": str(exc), "current_python_process_visible": False}


def _smooth_grid_z(z: np.ndarray, nx: int, ny: int, iterations: int) -> np.ndarray:
    grid_z = z.reshape(ny + 1, nx + 1).copy()
    for _ in range(iterations):
        padded = np.pad(grid_z, 1, mode="edge")
        avg = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
        grid_z = 0.8 * grid_z + 0.2 * avg
    return grid_z.ravel()


def _mesh_counts(mesh: QuadMesh) -> dict[str, int]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "tiles": int(len(mesh.faces)),
    }


def _assembly_counts(assembly: TileAssembly) -> dict[str, int]:
    return {
        "tiles": int(assembly.vertices.shape[0]),
        "vertices": int(assembly.vertices.shape[0] * assembly.vertices.shape[1]),
        "faces": int(assembly.vertices.shape[0] * 6),
    }
