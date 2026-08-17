import numpy as np
import pytest

from onestring_physics import PipelineParameters, build_onestring_design, create_builtin_shape
from onestring_physics.bijective_free_boundary import (
    BijectiveFreeBoundaryConfig,
    _check_local_validity,
    _energy_and_gradient,
    _energy_and_gradient_bruteforce,
    _extract_single_disk_boundary,
    _lbfgs_direction,
    _safe_step_limit,
    _safe_step_limit_bruteforce,
    _signed_double_areas,
    _surface_differentials,
    bijective_free_boundary_parameterization,
    boundary_self_intersection_count,
)
from onestring_physics.reference_bff import (
    count_internal_triangle_overlaps,
    count_internal_triangle_overlaps_bruteforce,
)
from onestring_physics.visualization import figure_domain


def _grid_mesh(nx: int = 8, ny: int = 7, *, curved: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.8, 0.8, ny)
    vertices = []
    for y in ys:
        for x in xs:
            z = 0.25 * np.sin(1.4 * x) * np.cos(1.2 * y) if curved else 0.0
            vertices.append([x, y, z])
    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def test_flat_grid_is_finite_bijective_and_optimized() -> None:
    vertices, faces = _grid_mesh(curved=False)
    uv, boundary_loop, metrics = bijective_free_boundary_parameterization(vertices, faces)

    assert uv.shape == (len(vertices), 2)
    assert len(boundary_loop) == 2 * (8 - 1) + 2 * (7 - 1)
    assert np.isfinite(uv).all()
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    assert metrics["optimization_succeeded"] is True
    assert metrics["optimization_converged"] is True
    assert metrics["final_energy"] < metrics["initial_energy"]
    assert metrics["onestring_grid_loss_used"] is False
    assert metrics["lambda_directly_optimized"] is False


def test_curved_grid_stays_bijective_with_a_free_boundary() -> None:
    vertices, faces = _grid_mesh(curved=True)
    uv, _boundary_loop, metrics = bijective_free_boundary_parameterization(
        vertices,
        faces,
        BijectiveFreeBoundaryConfig(max_iterations=60),
    )

    assert np.isfinite(uv).all()
    assert metrics["omega_boundary_fixed"] is False
    assert metrics["omega_boundary_shape"] == "free"
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    assert metrics["initial_internal_triangle_overlap_count"] == 0
    assert metrics["initial_uv_triangle_flip_count"] == 0
    assert metrics["optimization_requested_max_iterations"] == 60
    assert metrics["boundary_displacement_rms"] > 0.0
    assert metrics["boundary_displacement_max"] >= metrics["boundary_displacement_rms"]
    assert metrics["initial_boundary_radius_cv"] < 1.0e-12
    assert metrics["final_boundary_radius_cv"] >= 0.0
    assert metrics["initial_boundary_circle_fit_relative_rms"] < 1.0e-12
    assert metrics["final_boundary_circle_fit_relative_rms"] >= 0.0
    assert metrics["boundary_nonsimilarity_change_rms"] > 0.0
    assert np.isfinite(metrics["lambda_min"])
    assert np.isfinite(metrics["log_lambda_max"])
    assert metrics["anisotropy_max"] >= 1.0
    assert metrics["global_overlap_validation_each_step"] is False
    assert metrics["overlap_check_call_count"] == 2
    assert metrics["line_search_candidate_count"] >= metrics["line_search_accepted_candidate_count"]
    assert metrics["energy_gradient_call_count"] >= 1
    assert metrics["safe_step_call_count"] >= metrics["optimization_iteration_count"]
    assert len(metrics["optimization_iteration_log"]) <= 60


def test_spatial_hash_matches_bruteforce_for_random_deformations() -> None:
    vertices, faces = _grid_mesh(nx=7, ny=6, curved=False)
    base_uv = vertices[:, :2]
    for seed in range(20):
        generator = np.random.default_rng(seed)
        uv = base_uv + generator.normal(scale=0.08, size=base_uv.shape)
        stats: dict[str, int | float] = {}
        accelerated = count_internal_triangle_overlaps(uv, faces, stats=stats)
        reference = count_internal_triangle_overlaps_bruteforce(uv, faces)
        assert accelerated == reference
        assert stats["broad_phase_candidate_pair_count"] <= stats["total_possible_pair_count"]


def test_overlap_checker_detects_intentional_nonadjacent_overlap() -> None:
    uv = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.2],
            [1.2, 0.2],
            [0.2, 1.2],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int)

    assert count_internal_triangle_overlaps(uv, faces) == 1
    assert count_internal_triangle_overlaps_bruteforce(uv, faces) == 1


def test_local_validity_rejects_boundary_crossing_flip_and_degeneracy() -> None:
    valid_uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    valid_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    loop = [0, 1, 2, 3]
    valid, intersections = _check_local_validity(valid_uv, valid_faces, loop, 1.0e-12)
    assert valid is True
    assert intersections == 0

    bow_tie = valid_uv[[0, 2, 3, 1]]
    assert boundary_self_intersection_count(bow_tie, loop) == 1
    boundary_valid, boundary_intersections = _check_local_validity(
        bow_tie, valid_faces, loop, 1.0e-12
    )
    assert boundary_valid is False
    assert boundary_intersections == 1

    flipped_faces = np.asarray([[0, 2, 1], [0, 2, 3]], dtype=int)
    assert _check_local_validity(valid_uv, flipped_faces, loop, 1.0e-12)[0] is False

    degenerate_uv = valid_uv.copy()
    degenerate_uv[2] = [0.5, 0.0]
    assert _check_local_validity(degenerate_uv, valid_faces, loop, 1.0e-12)[0] is False


def test_progress_callback_reports_optimizer_stages_without_changing_result() -> None:
    vertices, faces = _grid_mesh(nx=5, ny=4, curved=True)
    progress: list[tuple[str, float, str]] = []
    first = bijective_free_boundary_parameterization(
        vertices,
        faces,
        BijectiveFreeBoundaryConfig(max_iterations=3),
    )
    second = bijective_free_boundary_parameterization(
        vertices,
        faces,
        BijectiveFreeBoundaryConfig(max_iterations=3),
        progress_callback=lambda stage, fraction, detail: progress.append((stage, fraction, detail)),
    )

    np.testing.assert_allclose(first[0], second[0], rtol=0.0, atol=0.0)
    stages = [row[0] for row in progress]
    assert "Extract boundary" in stages
    assert "Floater initialization" in stages
    assert "Initial validity check" in stages
    assert "Final validity check" in stages
    assert "S -> Omega complete" in stages
    assert all(0.0 <= row[1] <= 1.0 for row in progress)


def test_vectorized_energy_and_gradient_matches_scalar_reference() -> None:
    vertices, faces = _grid_mesh(nx=7, ny=6, curved=True)
    loop, _topology = _extract_single_disk_boundary(faces, len(vertices))
    inverse_surface, surface_areas = _surface_differentials(vertices, faces)
    uv = vertices[:, :2].copy()
    arguments = (uv, faces, inverse_surface, surface_areas, loop, 0.8, 0.7)

    vectorized = _energy_and_gradient(*arguments)
    reference = _energy_and_gradient_bruteforce(*arguments)

    np.testing.assert_allclose(vectorized[0], reference[0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(vectorized[1], reference[1], rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(vectorized[2:], reference[2:], rtol=1.0e-12, atol=1.0e-12)


def test_vectorized_safe_step_matches_scalar_reference() -> None:
    vertices, faces = _grid_mesh(nx=7, ny=6, curved=False)
    loop, _topology = _extract_single_disk_boundary(faces, len(vertices))
    uv = vertices[:, :2].copy()
    for seed in range(10):
        direction = np.random.default_rng(seed).normal(scale=0.05, size=uv.shape)
        accelerated_limit, accelerated_reason = _safe_step_limit(uv, direction, faces, loop)
        reference_limit, reference_reason = _safe_step_limit_bruteforce(uv, direction, faces, loop)
        assert accelerated_reason == reference_reason
        np.testing.assert_allclose(accelerated_limit, reference_limit, rtol=1.0e-12, atol=1.0e-12)


def test_safe_step_preserves_configured_positive_area_margin() -> None:
    uv = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=int)
    direction = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, -2.0]], dtype=float)

    limit, reason = _safe_step_limit(
        uv,
        direction,
        faces,
        [0, 1, 2],
        minimum_signed_double_area=0.1,
    )

    assert reason == "triangle_degeneracy"
    assert limit == pytest.approx(0.45)
    candidate = uv + 0.8 * limit * direction
    assert np.all(_signed_double_areas(candidate, faces) > 0.1)


def test_lbfgs_initial_metric_balances_mesh_gradient_scales_and_is_descent() -> None:
    gradient = np.asarray([[1.0, 0.0], [1.0e6, 0.0], [1.0e3, 0.0]], dtype=float)

    direction = _lbfgs_direction(gradient, [])

    assert float(np.sum(gradient * direction)) < 0.0
    np.testing.assert_allclose(np.abs(direction[:, 0]), np.full(3, 1.0e3))


def test_closed_non_disk_mesh_is_rejected() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int)

    with pytest.raises(RuntimeError, match="boundary|disk"):
        bijective_free_boundary_parameterization(vertices, faces)


def test_non_manifold_mesh_is_rejected() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5, 1.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=int)
    with pytest.raises(RuntimeError, match="manifold|disk"):
        bijective_free_boundary_parameterization(vertices, faces)


def test_inconsistent_winding_is_rejected() -> None:
    vertices, faces = _grid_mesh(nx=4, ny=4, curved=False)
    faces = faces.copy()
    faces[len(faces) // 2] = faces[len(faces) // 2][::-1]
    with pytest.raises(RuntimeError, match="embedding|boundary|disk"):
        bijective_free_boundary_parameterization(vertices, faces)


def test_pipeline_mode_is_explicit_and_does_not_replace_bff() -> None:
    target = create_builtin_shape("dome", {"amplitude": 0.35, "radius": 2.0})
    progress: list[tuple[str, float, str]] = []
    state = build_onestring_design(
        target,
        PipelineParameters(
            nx=2,
            max_3d_iterations=2,
            max_2d_iterations=2,
            omega_parameterization_mode="bijective_free_boundary",
            bijective_free_boundary_max_iterations=40,
        ),
        progress_callback=lambda stage, fraction, detail: progress.append((stage, fraction, detail)),
    )
    metrics = state.surface_parameterization.metrics

    assert state.surface_parameterization.method == "bijective_free_boundary"
    assert metrics["parameterization_method"] == "bijective_free_boundary"
    assert metrics["omega_boundary_shape"] == "free"
    assert metrics["omega_boundary_fixed"] is False
    assert metrics.get("bff_backend_used") != "local_discrete_bff"
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    figure = figure_domain(state)
    trace_names = {str(trace.name) for trace in figure.data}
    assert "parameterized UV mesh" in trace_names
    assert "local log(lambda)" in trace_names
    assert "initial Omega boundary" in trace_names
    assert "final optimized Omega boundary" in trace_names
    final_trace = next(trace for trace in figure.data if trace.name == "final optimized Omega boundary")
    loop = metrics["boundary_loop"]
    expected_boundary = state.surface_parameterization.uv_vertices_2d[
        np.asarray(loop + [loop[0]], dtype=int)
    ]
    np.testing.assert_allclose(np.asarray(final_trace.x, dtype=float), expected_boundary[:, 0])
    np.testing.assert_allclose(np.asarray(final_trace.y, dtype=float), expected_boundary[:, 1])
    internal_progress = [row for row in progress if row[0].startswith("S -> Omega: ")]
    assert internal_progress
    assert all(0.08 <= row[1] <= 0.16 for row in internal_progress)
