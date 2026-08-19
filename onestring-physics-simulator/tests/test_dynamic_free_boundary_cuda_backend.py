import numpy as np

from onestring_physics import bijective_free_boundary as base
from onestring_physics import dynamic_free_boundary as previous
from onestring_physics.dynamic_free_boundary_cuda_backend import (
    TorchHarmonicBoundaryResponse,
    TorchOmegaAccelerator,
)


def _disk_mesh():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )
    loop = [0, 1, 2, 3]
    return vertices, faces, loop


def test_torch_omega_energy_matches_numpy_reference_on_cpu():
    vertices, faces, loop = _disk_mesh()
    inverse_surface, areas = base._surface_differentials(vertices, faces)
    uv = np.asarray(
        [
            [0.0, 0.0],
            [1.08, 0.02],
            [1.0, 1.0],
            [-0.02, 0.98],
            [0.47, 0.54],
        ],
        dtype=float,
    )
    cfg = previous.BijectiveFreeBoundaryConfig(
        conformal_weight=4.0,
        shrink_weight=8.0,
        minimum_isotropic_scale=0.90,
        boundary_barrier_weight=1.0,
    )
    barrier_epsilon = 0.1

    expected = previous._energy_gradient(
        uv,
        faces,
        inverse_surface,
        areas,
        loop,
        barrier_epsilon,
        cfg,
    )
    backend = TorchOmegaAccelerator(
        faces=faces,
        inverse_surface=inverse_surface,
        surface_areas=areas,
        boundary_loop=loop,
        barrier_epsilon=barrier_epsilon,
        config=cfg,
        device="cpu",
    )
    actual = backend.evaluate(uv)

    assert np.isclose(actual[0], expected[0], rtol=1.0e-8, atol=1.0e-9)
    assert np.allclose(actual[1], expected[1], rtol=1.0e-7, atol=1.0e-8)
    assert np.isclose(actual[2], expected[2], rtol=1.0e-8, atol=1.0e-9)
    assert np.isclose(actual[3], expected[3], rtol=1.0e-8, atol=1.0e-9)
    assert np.isclose(actual[4], expected[4], rtol=1.0e-8, atol=1.0e-9)


def test_torch_triangle_safe_step_matches_numpy_reference_on_cpu():
    _vertices, faces, loop = _disk_mesh()
    uv = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    direction = np.zeros_like(uv)
    direction[4] = [0.9, 0.0]

    cfg = previous.BijectiveFreeBoundaryConfig()
    inverse_surface = np.repeat(np.eye(2)[None, :, :], len(faces), axis=0)
    areas = np.ones(len(faces), dtype=float)
    backend = TorchOmegaAccelerator(
        faces=faces,
        inverse_surface=inverse_surface,
        surface_areas=areas,
        boundary_loop=loop,
        barrier_epsilon=0.1,
        config=cfg,
        device="cpu",
    )

    expected, expected_reason = previous._triangle_safe_step(uv, direction, faces, 1.0e-12)
    actual, actual_reason = backend.triangle_safe_step(uv, direction, 1.0e-12)
    assert actual_reason == expected_reason
    assert np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-12)


def test_torch_harmonic_boundary_response_gives_center_average():
    _vertices, faces, loop = _disk_mesh()
    response = TorchHarmonicBoundaryResponse(
        faces,
        vertex_count=5,
        boundary_ids=np.asarray(loop, dtype=int),
        device="cpu",
        cg_tolerance=1.0e-10,
        cg_max_iterations=50,
    )
    boundary_values = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=float,
    )
    extended = response.extend(boundary_values)

    assert np.allclose(extended[np.asarray(loop)], boundary_values)
    assert np.allclose(extended[4], np.mean(boundary_values, axis=0), atol=1.0e-9)
