from __future__ import annotations

import numpy as np

from onestring_physics import bijective_free_boundary_parameterization
from onestring_physics.dynamic_free_boundary import (
    BijectiveFreeBoundaryConfig,
    _boundary_direction,
    _energy_gradient,
)
from onestring_physics.dynamic_free_boundary_v2 import HarmonicBoundaryResponse
from onestring_physics import bijective_free_boundary as legacy_free_boundary
from onestring_physics.fast_t3d_preview import _aabb_sweep_candidates


def _rectangular_disk(nx: int = 8, ny: int = 5):
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.5, 0.5, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = 0.12 * np.exp(-2.0 * (xx * xx + 2.0 * yy * yy))
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return vertices, np.asarray(faces, dtype=int)


def test_default_shrink_penalty_threshold_is_point_nine():
    assert BijectiveFreeBoundaryConfig().minimum_isotropic_scale == 0.9


def test_boundary_direction_is_local_not_global_polynomial_mode():
    count = 20
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    uv = np.column_stack([np.cos(angles), np.sin(angles)])
    gradient = np.zeros_like(uv)
    gradient[0] = np.asarray([-1.0, 0.0])
    settings = BijectiveFreeBoundaryConfig(
        boundary_normal_smoothing=0.0,
        boundary_tangent_weight=0.0,
        boundary_gradient_clip_factor=100.0,
        initial_step_scale=3.0,
    )
    direction, requested, characteristic = _boundary_direction(
        uv, gradient, list(range(count)), settings
    )
    magnitude = np.linalg.norm(direction, axis=1)
    assert magnitude[0] > 0.0
    assert np.count_nonzero(magnitude > 1.0e-12) == 1
    assert requested > 0.30 * characteristic
    assert requested < 0.40 * characteristic


def test_harmonic_boundary_proposal_moves_interior_before_energy_test():
    # 3x3 grid: vertex 4 is the only interior vertex.
    nx = ny = 3
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    faces = np.asarray(faces, dtype=int)
    boundary = np.asarray([0, 1, 2, 5, 8, 7, 6, 3], dtype=int)
    response = HarmonicBoundaryResponse(faces, 9, boundary)

    seed = np.zeros((9, 2), dtype=float)
    seed[2, 0] = 1.0
    seed[5, 0] = 1.0
    seed[8, 0] = 1.0
    coupled = response.extend(seed)

    # Boundary values are preserved exactly, while the center follows the right
    # boundary instead of remaining frozen.
    assert np.allclose(coupled[boundary], seed[boundary])
    assert coupled[4, 0] > 0.0
    assert abs(coupled[4, 1]) < 1.0e-12
    assert response.call_count == 1


def test_shrink_penalty_detects_isotropic_uv_collapse():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    inverse_surface, surface_areas = legacy_free_boundary._surface_differentials(vertices, faces)
    loop = [0, 1, 2, 3]
    settings = BijectiveFreeBoundaryConfig(
        conformal_weight=0.0,
        shrink_weight=1.0,
        minimum_isotropic_scale=0.7,
        boundary_barrier_weight=0.0,
    )
    uv_full = vertices[:, :2].copy()
    uv_small = 0.4 * uv_full
    full = _energy_gradient(uv_full, faces, inverse_surface, surface_areas, loop, 0.01, settings)
    small = _energy_gradient(uv_small, faces, inverse_surface, surface_areas, loop, 0.01, settings)
    assert full[4] == 0.0
    assert small[4] > 0.0
    assert small[0] > full[0]


def test_circle_initialization_gets_real_boundary_updates_on_anisotropic_disk():
    vertices, faces = _rectangular_disk()
    config = BijectiveFreeBoundaryConfig(
        max_iterations=60,
        line_search_max_steps=12,
        initial_boundary_shape="circle",
        initial_step_scale=3.0,
        conformal_weight=2.0,
        shrink_weight=4.0,
        minimum_isotropic_scale=0.65,
        interior_steps_per_boundary=1,
        boundary_normal_smoothing=0.10,
    )
    _uv, _loop, metrics = bijective_free_boundary_parameterization(vertices, faces, config)
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert metrics["boundary_global_low_frequency_basis_used"] is False
    assert metrics["boundary_candidate_interior_fixed"] is False
    assert metrics["boundary_candidate_harmonic_interior_response"] is True
    assert metrics["optimization_boundary_update_count"] > 0
    assert metrics["boundary_nonsimilarity_change_relative_rms"] > 1.0e-5
    attempts = metrics["optimization_boundary_attempt_log"]
    accepted_attempts = [attempt for attempt in attempts if attempt["accepted"]]
    assert accepted_attempts
    required_diagnostics = {
        "boundary_directional_derivative",
        "energy_before_relax",
        "energy_after_harmonic_predictor",
        "energy_after_interior_relax",
        "accepted",
        "reject_reason",
    }
    assert all(required_diagnostics <= set(attempt) for attempt in attempts)
    assert all(
        attempt["boundary_directional_derivative"] < 0.0
        and attempt["energy_after_interior_relax"] < attempt["energy_before_relax"]
        for attempt in accepted_attempts
    )


def test_rectangle_initialization_accepts_reduced_objective_boundary_update():
    vertices, faces = _rectangular_disk()
    config = BijectiveFreeBoundaryConfig(
        max_iterations=12,
        line_search_max_steps=12,
        initial_boundary_shape="rectangle",
        initial_step_scale=3.0,
        conformal_weight=2.0,
        shrink_weight=4.0,
        minimum_isotropic_scale=0.65,
        interior_steps_per_boundary=1,
        boundary_normal_smoothing=0.10,
    )

    _uv, _loop, metrics = bijective_free_boundary_parameterization(
        vertices,
        faces,
        config,
    )

    assert metrics["optimization_boundary_update_count"] > 0
    assert metrics["final_energy"] < metrics["initial_energy"]
    assert metrics["boundary_displacement_rms"] > 0.0
    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["boundary_self_intersection_count"] == 0
    assert metrics["internal_triangle_overlap_count"] == 0
    assert any(
        attempt["accepted"]
        for attempt in metrics["optimization_boundary_attempt_log"]
    )


def test_aabb_sweep_broad_phase_keeps_only_overlapping_boxes():
    boxes = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.5, 1.5, 0.0], [0.5, 1.5, 0.0]],
            [[4.0, 4.0, 0.0], [5.0, 4.0, 0.0], [5.0, 5.0, 0.0], [4.0, 5.0, 0.0]],
        ],
        dtype=float,
    )
    assert _aabb_sweep_candidates(boxes) == [(0, 1)]
