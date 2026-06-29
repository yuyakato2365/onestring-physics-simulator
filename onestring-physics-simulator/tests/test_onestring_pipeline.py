from onestring_physics.input_shape import create_builtin_shape
from onestring_physics.onestring_pipeline import (
    DeploymentParameters,
    PipelineParameters,
    build_onestring_design,
    inverse_map_uv_to_surface,
    safe_capstan_friction,
    simulate_onestring_deployment,
)
from onestring_physics.animation import assembly_progress_animation
from onestring_physics.visualization import figure_flat_tile_layout, figure_tile_assembly
import numpy as np


def test_onestring_pipeline_stores_paper_intermediates():
    target = create_builtin_shape("dome", {"amplitude": 0.45, "radius": 2.0})
    state = build_onestring_design(
        target,
        PipelineParameters(nx=3, max_3d_iterations=6, max_2d_iterations=6),
    )

    assert state.target_surface.vertices.shape[1] == 3
    assert state.surface_parameterization.method == "harmonic"
    assert "surface parameterization" in state.conformal_domain.method
    assert state.mesh_2d_initial.stage == "M2D"
    assert state.mesh_3d_initial.stage == "M3D"
    assert state.mesh_3d_optimized.stage == "K3D"
    assert state.mesh_2d_optimized.stage == "K2D"
    assert state.k2d_flat_layout.tile_top_vertices_2d.shape[1:] == (4, 2)
    assert state.tiles_3d.stage == "T3D"
    assert state.tiles_2d_top_hinge.stage == "T2D top hinge"
    assert state.tiles_2d_dual_hinge.stage == "T2D dual hinge"
    assert state.tiles_3d.vertices.shape[1:] == (8, 3)
    assert len(state.hinge_graph.hinges) > 0
    assert len(state.gap_graph.gaps) > 0
    assert len(state.lift_points) > 0
    assert len(state.string_path.boundary_gap_ids) > 0


def test_onestring_actuation_reports_t3d_error_and_constraints():
    target = create_builtin_shape("saddle", {"amplitude": 0.35, "radius": 2.0})
    state = build_onestring_design(
        target,
        PipelineParameters(nx=3, max_3d_iterations=5, max_2d_iterations=5),
    )
    result = simulate_onestring_deployment(
        state,
        DeploymentParameters(steps=5, solver_iterations=2, solver_substeps=1, store_animation_frames=True),
    )

    assert len(result.frames) == 5
    assert result.final_tiles.shape == state.tiles_3d.vertices.shape
    assert "final_deployment_error_to_T3D" in result.metrics
    assert "target_surface_fit_error_S" in result.metrics
    assert "snap_error" in result.metrics
    assert "lift_error" in result.metrics
    assert "collision_count" in result.metrics


def test_k3d_does_not_collapse_for_curved_targets():
    for kind in ["dome", "gaussian", "saddle", "wave"]:
        target = create_builtin_shape(kind, {"amplitude": 0.6, "radius": 2.0, "sigma": 0.9, "wavelength": 2.5})
        state = build_onestring_design(
            target,
            PipelineParameters(nx=3, max_3d_iterations=8, max_2d_iterations=4),
        )
        metrics = state.mesh_3d_optimized.metrics
        assert metrics["z_range_ratio"] >= 0.3
        assert "z_range_M3D" in metrics
        assert "surface_fit_error_after" in metrics


def test_m3d_uses_inverse_parameterization_not_xy_height_lift():
    target = create_builtin_shape("dome", {"amplitude": 0.65, "radius": 2.0})
    state = build_onestring_design(target, PipelineParameters(nx=3, max_3d_iterations=5, max_2d_iterations=5))
    metrics = state.mesh_3d_initial.metrics
    direct = state.mesh_2d_initial.vertices.copy()
    direct[:, 2] = target.height(direct[:, 0], direct[:, 1])

    assert metrics["m3d_construction_method"] == "mesh_harmonic"
    assert metrics["m3d_used_height_field_shortcut"] is False
    assert metrics["m3d_uv_triangle_lookup_fail_count"] == 0
    assert metrics["m3d_outside_omega_count"] >= 0
    assert state.surface_parameterization.metrics["harmonic_solve_performed"] is True
    assert state.surface_parameterization.method == "harmonic"


def test_m3d_vertices_lie_on_target_surface_and_match_m2d_connectivity():
    target = create_builtin_shape("gaussian", {"amplitude": 0.7, "radius": 2.0, "sigma": 0.8})
    state = build_onestring_design(target, PipelineParameters(nx=4, max_3d_iterations=5, max_2d_iterations=5))
    metrics = state.mesh_3d_initial.metrics

    assert state.mesh_3d_initial.vertices.shape[0] == state.mesh_2d_initial.vertices.shape[0]
    assert np.array_equal(state.mesh_3d_initial.faces, state.mesh_2d_initial.faces)
    assert metrics["m3d_surface_distance_mean"] < 1e-10
    assert metrics["m3d_surface_distance_max"] < 1e-9
    assert metrics["m3d_vertex_count"] == state.mesh_2d_initial.vertices.shape[0]
    assert metrics["m3d_quad_count"] == state.mesh_2d_initial.faces.shape[0]


def test_m3d_analytic_scaled_heightfield_debug_mode_is_explicit_shortcut():
    target = create_builtin_shape("wave", {"amplitude": 0.5, "radius": 2.0, "wavelength": 2.5})
    state = build_onestring_design(
        target,
        PipelineParameters(
            nx=3,
            max_3d_iterations=5,
            max_2d_iterations=5,
            m3d_construction_mode="analytic_scaled_heightfield_debug",
            strict_paper_flow=False,
        ),
    )

    assert state.mesh_3d_initial.metrics["m3d_used_height_field_shortcut"] is True
    assert state.surface_parameterization.method == "analytic_scaled_heightfield_debug"


def test_inverse_map_works_for_rotated_non_axis_aligned_surface():
    target = create_builtin_shape("dome", {"amplitude": 0.55, "radius": 2.0})
    state = build_onestring_design(target, PipelineParameters(nx=3, max_3d_iterations=5, max_2d_iterations=5))
    param = state.surface_parameterization
    angle = np.deg2rad(27.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    param.surface_vertices_3d = param.surface_vertices_3d @ rotation.T
    mapped = np.asarray([inverse_map_uv_to_surface(uv, param)[0] for uv in state.mesh_2d_initial.vertices[:, :2]])
    direct = state.mesh_2d_initial.vertices.copy()
    direct[:, 2] = target.height(direct[:, 0], direct[:, 1])

    assert mapped.shape == state.mesh_3d_initial.vertices.shape
    assert np.sqrt(np.mean((mapped - direct) ** 2)) > 1e-3


def test_k2d_not_identical_to_m2d_for_curved_targets():
    target = create_builtin_shape("gaussian", {"amplitude": 0.7, "radius": 2.0, "sigma": 0.85})
    state = build_onestring_design(target, PipelineParameters(nx=4, max_3d_iterations=8, max_2d_iterations=12))
    metrics = state.mesh_2d_optimized.metrics

    assert metrics["k2d_z_abs_max"] < 1e-8
    assert metrics["k2d_displacement_rms"] > 1e-5
    assert metrics["mean_edge_length_error_after"] < metrics["mean_edge_length_error_before"]
    assert metrics["max_edge_length_error_after"] < metrics["max_edge_length_error_before"]


def test_t2d_uses_k2d_top_vertices_and_has_frumstum_geometry():
    target = create_builtin_shape("dome", {"amplitude": 0.6, "radius": 2.0})
    state = build_onestring_design(target, PipelineParameters(nx=3, max_3d_iterations=8, max_2d_iterations=12))
    k2d_tiles = state.k2d_flat_layout.tile_top_vertices_3d

    top_to_k2d = np.max(np.abs(state.tiles_2d_top_hinge.vertices[:, :4, :2] - k2d_tiles[:, :, :2]))
    assert state.tiles_2d_top_hinge.metrics["top_vertices_match_k2d_max_error"] < 0.2
    assert top_to_k2d < 0.2
    assert np.max(np.abs(state.tiles_2d_top_hinge.vertices[:, :4, 2])) < 1e-8
    assert state.tiles_2d_top_hinge.vertices.shape[1] == 8
    assert state.tiles_2d_top_hinge.side_faces.shape == (4, 4)
    assert state.tiles_2d_top_hinge.metrics["side_faces_count"] == state.tiles_2d_top_hinge.tile_count * 4
    assert state.tiles_2d_top_hinge.metrics["t2d_gap_count"] == len(state.k2d_flat_layout.gap_polygons)


def test_k2d_flat_layout_rendering_has_independent_tile_gaps():
    target = create_builtin_shape("gaussian", {"amplitude": 0.7, "radius": 2.0, "sigma": 0.85})
    state = build_onestring_design(target, PipelineParameters(nx=3, max_3d_iterations=8, max_2d_iterations=12))
    fig = figure_flat_tile_layout(state.k2d_flat_layout, hinge_graph=state.hinge_graph)
    trace_names = {getattr(trace, "name", "") for trace in fig.data}

    assert "K2D independent tile faces" in trace_names
    assert "tile gaps / edges" in trace_names
    assert state.k2d_flat_layout.metrics["min_clearance"] >= 0.0


def test_t2d_rendering_includes_tile_mesh_edges_and_hinge_markers():
    target = create_builtin_shape("dome", {"amplitude": 0.5, "radius": 2.0})
    state = build_onestring_design(target, PipelineParameters(nx=3, max_3d_iterations=6, max_2d_iterations=8))
    fig = figure_tile_assembly(state.tiles_2d_dual_hinge, hinge_graph=state.hinge_graph)
    trace_names = {getattr(trace, "name", "") for trace in fig.data}

    assert len(fig.data) <= 4
    assert "tiles" in trace_names
    assert "tile edges" in trace_names
    assert "top hinges" in trace_names or "bottom hinges" in trace_names


def test_assembly_progress_animation_has_frames():
    target = create_builtin_shape("dome", {"amplitude": 0.5, "radius": 2.0})
    state = build_onestring_design(target, PipelineParameters(nx=3, max_3d_iterations=4, max_2d_iterations=4))
    fig = assembly_progress_animation(state, frame_count=12)

    assert len(fig.frames) == 12
    assert len(fig.data) > 0


def test_safe_capstan_friction_does_not_overflow():
    assert np.isinf(safe_capstan_friction(1.0, 1000.0))
    assert safe_capstan_friction(0.2, 2.0) > 0.0
