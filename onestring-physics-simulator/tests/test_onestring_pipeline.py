from onestring_physics.input_shape import create_builtin_shape
from onestring_physics.onestring_pipeline import (
    DeploymentParameters,
    PipelineParameters,
    QuadMesh,
    _canonicalize_faces_by_coincident_vertices,
    _csf_split_lines,
    _detect_parameterization_reflection_symmetry,
    _extrude_tiles,
    _free_layout_parameters,
    _m2d_connected_component_sizes,
    _mirror_csf_split_lines,
    _orient_tile_normals_consistently,
    _parameterization_stretch_csf,
    _surface_peak_uvs,
    _split_m2d_along_existing_grid_line,
    _weld_k3d_duplicate_reference_vertices,
    build_onestring_design,
    export_t2d_stl,
    inverse_map_uv_to_surface,
    safe_capstan_friction,
    simulate_onestring_deployment,
)
from onestring_physics.animation import _simultaneous_hinge_contraction_vertices, assembly_progress_animation
from onestring_physics.visualization import figure_flat_tile_layout, figure_tile_assembly
import numpy as np
from types import SimpleNamespace


def experimental_params(**kwargs):
    values = {
        "omega_boundary_mode": "shape_preserving_experimental",
        "omega_parameterization_mode": "pca_debug",
        "allow_experimental_pipeline": True,
    }
    values.update(kwargs)
    return PipelineParameters(**values)


def test_hinge_layout_defaults_match_streamlit_controls():
    params = PipelineParameters()

    assert params.hinge_layout_connection_weight == 8.0
    assert params.hinge_layout_collision_weight == 4.0
    assert params.hinge_layout_anchor_weight == 0.0
    assert params.hinge_layout_initial_expansion == 1.6
    assert params.hinge_layout_max_center_drift_tiles == 5.0

    free = _free_layout_parameters(
        grid=SimpleNamespace(tile_size=1.0, gap_size=0.08),
        iterations=params.hinge_layout_iterations,
        connection_weight=params.hinge_layout_connection_weight,
        collision_weight=params.hinge_layout_collision_weight,
        anchor_weight=params.hinge_layout_anchor_weight,
        initial_expansion=params.hinge_layout_initial_expansion,
        max_center_drift_tiles=params.hinge_layout_max_center_drift_tiles,
    )
    assert free["anchor_weight"] == 0.0
    assert free["initial_expansion"] == 1.6
    assert free["max_center_drift_tiles"] == 5.0


def test_onestring_pipeline_stores_paper_intermediates():
    target = create_builtin_shape("dome", {"amplitude": 0.45, "radius": 2.0})
    state = build_onestring_design(
        target,
        experimental_params(nx=3, max_3d_iterations=6, max_2d_iterations=6),
    )

    assert state.target_surface.vertices.shape[1] == 3
    assert state.surface_parameterization.method == "pca_debug"
    assert state.surface_parameterization.metrics["parameterization_exactness_label"] == "experimental"
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
    assert state.gap_graph.metrics["lift_point_selection_model"] == "discrete_graph_gpe_peak_coupling"
    assert state.gap_graph.metrics["lift_point_count"] == len(state.lift_points)
    assert state.gap_graph.metrics["lift_point_rows"]
    for row in state.gap_graph.metrics["lift_point_rows"]:
        assert {"gap_id", "gpe", "position_2d", "position_3d", "selection_reason"}.issubset(row)
        assert row["selection_reason"]
    assert state.string_path.metrics["string_path_model"] == "boundary_loop_to_gpe_lift_peaks_weighted_gap_graph_path"
    assert state.string_path.metrics["string_path_boundary_loop_included"] is True
    assert set(state.string_path.boundary_gap_ids).issubset(set(state.string_path.gap_ids))


def test_onestring_actuation_reports_t3d_error_and_constraints():
    target = create_builtin_shape("saddle", {"amplitude": 0.35, "radius": 2.0})
    state = build_onestring_design(
        target,
        experimental_params(nx=3, max_3d_iterations=5, max_2d_iterations=5),
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
            experimental_params(nx=3, max_3d_iterations=8, max_2d_iterations=4),
        )
        metrics = state.mesh_3d_optimized.metrics
        assert metrics["z_range_ratio"] >= 0.3
        assert "z_range_M3D" in metrics
        assert "surface_fit_error_after" in metrics


def test_m3d_uses_inverse_parameterization_not_xy_height_lift():
    target = create_builtin_shape("dome", {"amplitude": 0.65, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=5, max_2d_iterations=5))
    metrics = state.mesh_3d_initial.metrics
    direct = state.mesh_2d_initial.vertices.copy()
    direct[:, 2] = target.height(direct[:, 0], direct[:, 1])

    assert metrics["m3d_construction_method"] == "mesh_harmonic"
    assert metrics["m3d_used_height_field_shortcut"] is False
    assert metrics["m3d_uv_triangle_lookup_fail_count"] == 0
    assert metrics["m3d_outside_omega_count"] >= 0
    assert state.surface_parameterization.metrics["harmonic_solve_performed"] is False
    assert state.surface_parameterization.metrics["omega_boundary_shape_preserved"] is True
    assert state.surface_parameterization.method == "pca_debug"


def test_m3d_vertices_lie_on_target_surface_and_match_m2d_connectivity():
    target = create_builtin_shape("gaussian", {"amplitude": 0.7, "radius": 2.0, "sigma": 0.8})
    state = build_onestring_design(target, experimental_params(nx=4, max_3d_iterations=5, max_2d_iterations=5))
    metrics = state.mesh_3d_initial.metrics

    assert state.mesh_3d_initial.vertices.shape[0] == state.mesh_2d_initial.vertices.shape[0]
    assert np.array_equal(state.mesh_3d_initial.faces, state.mesh_2d_initial.faces)
    assert metrics["m3d_surface_distance_mean"] < 1e-10
    assert metrics["m3d_surface_distance_max"] < 1e-9
    assert metrics["m3d_vertex_count"] == state.mesh_2d_initial.vertices.shape[0]
    assert metrics["m3d_quad_count"] == state.mesh_2d_initial.faces.shape[0]


def test_csf_split_detects_local_stretch_above_two():
    param = SimpleNamespace(
        uv_vertices_2d=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=float,
        ),
        surface_vertices_3d=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [6.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        uv_faces=np.asarray([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=int),
        surface_faces=np.asarray([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=int),
    )

    csf = _parameterization_stretch_csf(param)
    lines = _csf_split_lines(param, csf, threshold=2.0, max_splits=2)

    assert np.max(csf) > 2.0
    assert lines


def test_csf_split_prefers_symmetric_peak_path_over_high_csf_centroid():
    param = SimpleNamespace(
        uv_vertices_2d=np.asarray(
            [
                [-0.6, 0.0],
                [0.6, 0.0],
                [-0.9, 0.45],
                [-0.3, 0.45],
                [0.3, 0.45],
                [0.9, 0.45],
                [-0.9, -0.45],
                [-0.3, -0.45],
                [0.3, -0.45],
                [0.9, -0.45],
            ],
            dtype=float,
        ),
        surface_vertices_3d=np.asarray(
            [
                [-0.6, 0.0, 2.0],
                [0.6, 0.0, 2.0],
                [-0.9, 0.45, 0.0],
                [-0.3, 0.45, 0.0],
                [0.3, 0.45, 0.0],
                [0.9, 0.45, 0.0],
                [-0.9, -0.45, 0.0],
                [-0.3, -0.45, 0.0],
                [0.3, -0.45, 0.0],
                [0.9, -0.45, 0.0],
            ],
            dtype=float,
        ),
        uv_faces=np.asarray([[0, 2, 3], [0, 3, 7], [0, 7, 6], [1, 4, 5], [1, 9, 8], [1, 8, 4]], dtype=int),
        surface_faces=np.asarray([[0, 2, 3], [0, 3, 7], [0, 7, 6], [1, 4, 5], [1, 9, 8], [1, 8, 4]], dtype=int),
    )
    csf = np.ones(10, dtype=float)
    csf[[2, 3, 4, 5]] = 2.5

    peaks = _surface_peak_uvs(param)
    lines = _csf_split_lines(param, csf, threshold=2.0, max_splits=1)

    assert len(peaks) == 2
    assert lines == [("row", 0.0)]


def test_csf_split_duplicates_grid_vertices_without_deleting_quads():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)

    split_vertices, split_faces, duplicate_count = _split_m2d_along_existing_grid_line(vertices, faces, ("col", 1.0))

    assert split_faces.shape == faces.shape
    assert duplicate_count == 2
    assert split_vertices.shape[0] == vertices.shape[0] + 2
    assert _m2d_connected_component_sizes(split_faces) == [1, 1]


def test_detected_symmetry_mirrors_csf_split_lines():
    param = SimpleNamespace(
        surface_vertices_3d=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        uv_vertices_2d=np.asarray(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=float,
        ),
    )

    symmetry = _detect_parameterization_reflection_symmetry(param)
    mirrored = _mirror_csf_split_lines([("col", 0.35)], symmetry["axes"], symmetry["centers"], param.uv_vertices_2d)

    assert 0 in symmetry["axes"]
    assert ("col", 0.35) in mirrored
    assert any(axis == "col" and np.isclose(value, -0.35) for axis, value in mirrored)


def test_m2d_grid_aligns_surface_peak_to_shared_quad_vertex():
    target = create_builtin_shape("dome", {"amplitude": 0.45, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=2, max_2d_iterations=2))
    metrics = state.mesh_2d_initial.metrics
    peak_uv = np.asarray(metrics["m2d_peak_uv_target"], dtype=float)
    uv_vertices = state.mesh_2d_initial.vertices[:, :2]

    assert metrics["m2d_grid_aligned_to_peak_vertex"] is True
    assert np.min(np.linalg.norm(uv_vertices - peak_uv, axis=1)) < 1e-9
    peak_vertex = int(np.argmax(state.mesh_3d_optimized.vertices[:, 2]))
    incident_faces = sum(1 for face in state.mesh_3d_optimized.faces if peak_vertex in set(map(int, face)))
    assert incident_faces >= 4


def test_omega_boundary_preserves_nonrectangular_surface_shape():
    target = create_builtin_shape("half_gourd", {"amplitude": 0.55, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=2, max_2d_iterations=2))
    boundary = state.surface_parameterization.omega_boundary[:-1]
    lo = np.min(boundary, axis=0)
    hi = np.max(boundary, axis=0)
    on_box = (
        np.isclose(boundary[:, 0], lo[0], atol=1e-6)
        | np.isclose(boundary[:, 0], hi[0], atol=1e-6)
        | np.isclose(boundary[:, 1], lo[1], atol=1e-6)
        | np.isclose(boundary[:, 1], hi[1], atol=1e-6)
    )

    assert state.surface_parameterization.metrics["omega_boundary_shape_preserved"] is True
    assert state.surface_parameterization.metrics["omega_boundary_forced_rectangle"] is False
    assert state.surface_parameterization.metrics["parameterization_exactness_label"] == "experimental"
    assert float(np.mean(on_box)) < 0.95


def test_m3d_analytic_scaled_heightfield_debug_mode_is_explicit_shortcut():
    target = create_builtin_shape("wave", {"amplitude": 0.5, "radius": 2.0, "wavelength": 2.5})
    state = build_onestring_design(
        target,
        experimental_params(
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
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=5, max_2d_iterations=5))
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
    state = build_onestring_design(target, experimental_params(nx=4, max_3d_iterations=8, max_2d_iterations=12))
    metrics = state.mesh_2d_optimized.metrics

    assert metrics["k2d_z_abs_max"] < 1e-8
    assert metrics["k2d_displacement_rms"] > 1e-5
    assert metrics["mean_edge_length_error_after"] < metrics["mean_edge_length_error_before"]
    assert metrics["max_edge_length_error_after"] < metrics["max_edge_length_error_before"]


def test_k3d_split_duplicate_vertices_remain_coincident_after_weld():
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    optimized = reference.copy()
    optimized[1] += np.asarray([0.0, 0.0, 0.2])
    optimized[2] += np.asarray([0.0, 0.0, -0.2])

    welded, metrics = _weld_k3d_duplicate_reference_vertices(reference, optimized)

    assert metrics["k3d_split_duplicate_weld_applied"] is True
    assert metrics["k3d_split_duplicate_weld_group_count"] == 1
    assert np.allclose(welded[1], welded[2])


def test_t3d_extrusion_normals_are_oriented_across_shared_edges():
    raw_normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=float)
    faces = np.asarray([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=int)

    oriented, metrics = _orient_tile_normals_consistently(raw_normals, faces)

    assert np.dot(oriented[0], oriented[1]) > 0.0
    assert metrics["t3d_extrusion_normal_flip_count"] == 1
    assert metrics["t3d_extrusion_normal_inconsistent_edge_count"] == 0


def test_t3d_extrusion_normals_are_oriented_across_split_components():
    raw_normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 4, 5, 2],
            [6, 7, 8, 9],
            [7, 10, 11, 8],
        ],
        dtype=int,
    )

    oriented, metrics = _orient_tile_normals_consistently(raw_normals, faces)

    assert metrics["t3d_extrusion_normal_component_count"] == 2
    assert metrics["t3d_extrusion_normal_component_global_flip_count"] == 1
    assert np.all(oriented[:, 2] > 0.0)


def test_split_coincident_edges_are_not_treated_as_open_outer_walls():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int)
    mesh = QuadMesh(vertices, faces, None, "K3D")

    canonical_faces, weld_metrics = _canonicalize_faces_by_coincident_vertices(vertices, faces)
    assembly, _report = _extrude_tiles(mesh, 0.1, "T3D")

    assert weld_metrics["split_virtual_weld_applied"] is True
    assert len({int(v) for v in canonical_faces.reshape(-1)}) == 6
    assert assembly.metrics["split_contact_miter_edge_count"] == 1
    assert assembly.metrics["boundary_side_plane_count"] == 6
    assert sorted(assembly.metrics["split_contact_side_edges"]) == [[0, 1], [1, 3]]


def test_t2d_preserves_t3d_tile_shape_for_animation_and_has_frustum_geometry():
    target = create_builtin_shape("dome", {"amplitude": 0.6, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=8, max_2d_iterations=12))
    assert "t3d_extrusion_normal_flip_count" in state.tiles_3d.metrics
    assert state.tiles_3d.metrics["t3d_extrusion_normal_inconsistent_edge_count"] == 0
    assert state.tiles_3d.metrics["t3d_reversed_extrusion_vertex_count"] == 0

    assert state.tiles_2d_top_hinge.metrics["t2d_t3d_congruent_tile_geometry"] is True
    assert state.tiles_2d_top_hinge.metrics["tile_shape_max_error_to_T3D"] < 1e-6
    assert state.tiles_2d_top_hinge.vertices.shape[1] == 8
    assert state.tiles_2d_top_hinge.side_faces.shape == (4, 4)
    assert state.tiles_2d_top_hinge.metrics["side_faces_count"] == state.tiles_2d_top_hinge.tile_count * 4
    assert state.tiles_2d_top_hinge.metrics["t2d_gap_count"] == len(state.k2d_flat_layout.gap_polygons)


def test_k2d_flat_layout_rendering_has_independent_tile_gaps():
    target = create_builtin_shape("gaussian", {"amplitude": 0.7, "radius": 2.0, "sigma": 0.85})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=8, max_2d_iterations=12))
    fig = figure_flat_tile_layout(state.k2d_flat_layout, hinge_graph=state.hinge_graph)
    trace_names = {getattr(trace, "name", "") for trace in fig.data}

    assert "K2D independent tile faces" in trace_names
    assert "tile gaps / edges" in trace_names
    assert state.k2d_flat_layout.metrics["min_clearance"] >= 0.0


def test_t2d_rendering_includes_tile_mesh_edges_and_hinge_markers():
    target = create_builtin_shape("dome", {"amplitude": 0.5, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=6, max_2d_iterations=8))
    fig = figure_tile_assembly(state.tiles_2d_dual_hinge, hinge_graph=state.hinge_graph)
    trace_names = {getattr(trace, "name", "") for trace in fig.data}

    assert len(fig.data) <= 4
    assert "tiles" in trace_names
    assert "tile edges" in trace_names
    assert "top hinges" in trace_names or "bottom hinges" in trace_names


def test_assembly_progress_animation_has_frames():
    target = create_builtin_shape("dome", {"amplitude": 0.5, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=3, max_3d_iterations=4, max_2d_iterations=4))
    fig = assembly_progress_animation(state, frame_count=12)

    assert len(fig.frames) == 12
    assert len(fig.data) > 0


def test_assembly_progress_animation_keeps_tiles_rigid_between_poses():
    start = np.asarray(
        [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]],
        dtype=float,
    )
    theta = np.deg2rad(70.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    target = start @ rotation.T + np.asarray([[[4.0, -2.0, 1.0]]])

    mid = _simultaneous_hinge_contraction_vertices(start, target, 0.5)

    start_lengths = np.linalg.norm(start[:, :, None, :] - start[:, None, :, :], axis=-1)
    mid_lengths = np.linalg.norm(mid[:, :, None, :] - mid[:, None, :, :], axis=-1)
    assert np.max(np.abs(start_lengths - mid_lengths)) < 1e-10


def test_safe_capstan_friction_does_not_overflow():
    assert np.isinf(safe_capstan_friction(1.0, 1000.0))
    assert safe_capstan_friction(0.2, 2.0) > 0.0


def test_export_t2d_stl_combined_records_metrics():
    target = create_builtin_shape("dome", {"amplitude": 0.4, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=2, max_3d_iterations=3, max_2d_iterations=3))

    data, metrics = export_t2d_stl(state, stage="dual_hinge", panel_size=0.1)

    assert data.startswith(b"solid onestring_t2d")
    assert b"facet normal" in data
    assert metrics["t2d_tile_count"] == state.tiles_2d_dual_hinge.tile_count
    assert metrics["t2d_export_panel_size"] == 0.1
    assert metrics["t2d_vertex_count"] == state.tiles_2d_dual_hinge.tile_count * 8
    assert metrics["t2d_face_count"] == state.tiles_2d_dual_hinge.tile_count * 12
    assert metrics["t2d_nonmanifold_edge_count"] == 0
    assert metrics["t2d_export_model"] == "current_T2D_assembly_vertices_scaled_to_stl"
    assert metrics["t2d_export_thickness"] > 0.0


def test_export_t2d_stl_can_return_per_tile_files():
    target = create_builtin_shape("dome", {"amplitude": 0.3, "radius": 2.0})
    state = build_onestring_design(target, experimental_params(nx=2, max_3d_iterations=2, max_2d_iterations=2))

    files, metrics = export_t2d_stl(state.tiles_2d_top_hinge, separate_tiles=True, panel_size=0.1)

    assert len(files) == state.tiles_2d_top_hinge.tile_count
    assert all(name.endswith(".stl") for name in files)
    assert metrics["t2d_tile_count"] == state.tiles_2d_top_hinge.tile_count


def test_omega_rectangular_debug_and_paper_default_are_explicit():
    target = create_builtin_shape("dome", {"amplitude": 0.35, "radius": 2.0})
    default_state = build_onestring_design(target, PipelineParameters(nx=2, max_3d_iterations=2, max_2d_iterations=2))
    assert default_state.surface_parameterization.method == "bff"
    assert default_state.surface_parameterization.metrics["requested_omega_parameterization_mode"] == "bff"
    assert default_state.surface_parameterization.metrics["flattening_backend"] == "local_bff_rectangular_boundary_cotan_harmonic"
    assert default_state.surface_parameterization.metrics["bff_implemented"] is True
    assert default_state.surface_parameterization.metrics["bff_reference_backend_available"] is False
    assert default_state.surface_parameterization.metrics["omega_boundary_forced_rectangle"] is True
    assert default_state.surface_parameterization.metrics["omega_boundary_shape"] == "rectangular"
    assert default_state.surface_parameterization.metrics["omega_boundary_constraint_model"] == "bff_prescribed_rectangular_boundary_by_3d_arclength"
    assert default_state.surface_parameterization.metrics["parameterization_exactness_label"] == "bff_rectangular_boundary_local"
    assert default_state.surface_parameterization.metrics["bff_boundary_rectangular_correction_applied"] is True
    assert default_state.surface_parameterization.metrics["harmonic_solve_performed"] is True
    assert default_state.surface_parameterization.metrics["uv_triangle_flip_count"] == 0
    boundary = default_state.surface_parameterization.omega_boundary[:-1]
    lo = np.min(boundary, axis=0)
    hi = np.max(boundary, axis=0)
    on_rect = (
        np.isclose(boundary[:, 0], lo[0], atol=1e-8)
        | np.isclose(boundary[:, 0], hi[0], atol=1e-8)
        | np.isclose(boundary[:, 1], lo[1], atol=1e-8)
        | np.isclose(boundary[:, 1], hi[1], atol=1e-8)
    )
    assert float(np.mean(on_rect)) > 0.95
    m2d_metrics = default_state.mesh_2d_initial.metrics
    assert m2d_metrics["m2d_boundary_clipping_used"] is True
    assert m2d_metrics["m2d_boundary_clip_policy_effective"] == "strict_vertices"
    assert m2d_metrics["m2d_general_omega_overlay_rebuilt"] is False
    m2d_centers = np.mean(default_state.mesh_2d_initial.vertices[default_state.mesh_2d_initial.faces][:, :, :2], axis=1)
    assert np.all(m2d_centers[:, 0] >= lo[0] - 1e-8)
    assert np.all(m2d_centers[:, 0] <= hi[0] + 1e-8)
    assert np.all(m2d_centers[:, 1] >= lo[1] - 1e-8)
    assert np.all(m2d_centers[:, 1] <= hi[1] + 1e-8)

    try:
        build_onestring_design(
            target,
            PipelineParameters(
                nx=2,
                max_3d_iterations=2,
                max_2d_iterations=2,
                omega_boundary_mode="shape_preserving_experimental",
                omega_parameterization_mode="pca_debug",
            ),
        )
    except RuntimeError as exc:
        assert "allow_experimental_pipeline=True" in str(exc)
    else:
        raise AssertionError("experimental PCA path must require explicit opt-in")

    rectangular = build_onestring_design(
        target,
        experimental_params(
            nx=2,
            max_3d_iterations=2,
            max_2d_iterations=2,
            omega_boundary_mode="rectangular_debug",
            omega_parameterization_mode="pca_debug",
        ),
    )

    assert rectangular.surface_parameterization.metrics["omega_boundary_mode"] == "rectangular_debug"
    assert rectangular.surface_parameterization.metrics["parameterization_exactness_label"] == "debug"
    assert rectangular.surface_parameterization.metrics["omega_boundary_forced_rectangle"] is True

    try:
        build_onestring_design(
            target,
            PipelineParameters(
                nx=2,
                max_3d_iterations=2,
                max_2d_iterations=2,
                omega_parameterization_mode="paper_like_unimplemented",
            ),
        )
    except NotImplementedError as exc:
        assert "paper_like_unimplemented is not implemented" in str(exc)
    else:
        raise AssertionError("unimplemented paper mode must not silently fallback")


def test_bff_request_uses_rectangular_target_boundary_even_without_reference_backend():
    target = create_builtin_shape("half_gourd", {"amplitude": 0.35, "radius": 2.0})
    state = build_onestring_design(target, PipelineParameters(nx=2, max_3d_iterations=2, max_2d_iterations=2))
    metrics = state.surface_parameterization.metrics
    boundary = state.surface_parameterization.omega_boundary[:-1]
    lo = np.min(boundary, axis=0)
    hi = np.max(boundary, axis=0)
    on_rect = (
        np.isclose(boundary[:, 0], lo[0], atol=1e-8)
        | np.isclose(boundary[:, 0], hi[0], atol=1e-8)
        | np.isclose(boundary[:, 1], lo[1], atol=1e-8)
        | np.isclose(boundary[:, 1], hi[1], atol=1e-8)
    )

    assert state.surface_parameterization.method == "bff"
    assert metrics["requested_omega_parameterization_mode"] == "bff"
    assert metrics["flattening_backend"] == "local_bff_rectangular_boundary_cotan_harmonic"
    assert metrics["bff_implemented"] is True
    assert metrics["bff_reference_backend_available"] is False
    assert metrics["bff_reference_backend_error"]
    assert metrics["parameterization_exactness_label"] == "bff_rectangular_boundary_local"
    assert metrics["omega_boundary_forced_rectangle"] is True
    assert metrics["omega_boundary_shape"] == "rectangular"
    assert metrics["bff_boundary_rectangular_correction_applied"] is True
    assert metrics["uv_triangle_flip_count"] == 0
    assert float(np.mean(on_rect)) > 0.95


def test_rect_harmonic_mode_is_explicitly_not_bff():
    target = create_builtin_shape("dome", {"amplitude": 0.35, "radius": 2.0})
    state = build_onestring_design(
        target,
        PipelineParameters(
            nx=2,
            max_3d_iterations=2,
            max_2d_iterations=2,
            omega_parameterization_mode="rect_harmonic",
        ),
    )
    metrics = state.surface_parameterization.metrics

    assert state.surface_parameterization.method == "rect_harmonic"
    assert metrics["flattening_backend"] == "rect_harmonic"
    assert metrics["parameterization_exactness_label"] == "rectangular_boundary_harmonic"
    assert metrics["bff_implemented"] is False
    assert metrics["omega_boundary_fixed"] is True
    assert metrics["omega_boundary_shape"] == "rectangular"
    assert metrics["omega_boundary_forced_rectangle"] is True


def test_snowman_half_and_full_builtin_shapes_are_available():
    for kind in ["snowman_half", "snowman_full"]:
        target = create_builtin_shape(kind, {"amplitude": 0.5, "radius": 2.0})
        state = build_onestring_design(target, experimental_params(nx=2, max_3d_iterations=2, max_2d_iterations=2))
        assert state.target_surface.vertices.shape[0] > 0
        assert state.surface_parameterization.metrics["parameterization_method"] == "pca_debug"


def test_pipeline_parameters_default_csf_threshold_is_1_9():
    params = PipelineParameters()
    assert params.csf_split_threshold == 1.9



def test_csf_split_defaults_allow_multiple_splits():
    params = PipelineParameters()
    assert params.enable_csf_splits is True
    assert params.csf_split_threshold == 1.9
    assert params.max_csf_splits >= 2
