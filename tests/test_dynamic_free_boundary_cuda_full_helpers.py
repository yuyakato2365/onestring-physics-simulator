import numpy as np

from onestring_physics import bijective_free_boundary as base
from onestring_physics import dynamic_free_boundary as previous
from onestring_physics.dynamic_free_boundary_cuda_backend import TorchOmegaAccelerator
from onestring_physics.dynamic_free_boundary_cuda_full import (
    _area_valid_tensor,
    _boundary_intersection_count_tensor,
    _full_safe_step_tensor,
    _triangle_safe_step_tensor,
)


def _disk():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
         [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=float
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    loop = [0, 1, 2, 3]
    return vertices, faces, loop


def _backend(vertices, faces, loop):
    inverse_surface, areas = base._surface_differentials(vertices, faces)
    return TorchOmegaAccelerator(
        faces=faces,
        inverse_surface=inverse_surface,
        surface_areas=areas,
        boundary_loop=loop,
        barrier_epsilon=0.1,
        config=previous.BijectiveFreeBoundaryConfig(),
        device="cpu",
    )


def test_full_cuda_tensor_validity_and_triangle_safe_step_match_reference_on_cpu():
    vertices, faces, loop = _disk()
    accel = _backend(vertices, faces, loop)
    torch = accel.torch
    uv_np = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=float
    )
    direction_np = np.zeros_like(uv_np)
    direction_np[4] = [0.9, 0.0]
    uv = torch.tensor(uv_np, dtype=accel.dtype)
    direction = torch.tensor(direction_np, dtype=accel.dtype)

    assert bool(_area_valid_tensor(accel, uv, 1.0e-12).item())
    expected, _ = previous._triangle_safe_step(uv_np, direction_np, faces, 1.0e-12)
    actual, _ = _triangle_safe_step_tensor(accel, uv, direction, 1.0e-12)
    assert np.isclose(float(actual.item()), expected, rtol=1.0e-10, atol=1.0e-12)


def test_full_cuda_boundary_intersection_and_full_safe_step_match_reference_on_cpu():
    vertices, faces, loop = _disk()
    accel = _backend(vertices, faces, loop)
    torch = accel.torch
    uv_np = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=float
    )
    direction_np = np.zeros_like(uv_np)
    direction_np[0] = [0.75, 0.75]
    uv = torch.tensor(uv_np, dtype=accel.dtype)
    direction = torch.tensor(direction_np, dtype=accel.dtype)

    assert int(_boundary_intersection_count_tensor(accel, uv).item()) == 0
    expected, expected_reason = base._safe_step_limit(uv_np, direction_np, faces, loop, 1.0e-12)
    actual, actual_reason = _full_safe_step_tensor(accel, uv, direction, 1.0e-12)
    assert actual_reason == expected_reason
    if np.isfinite(expected):
        assert np.isclose(float(actual.item()), expected, rtol=1.0e-9, atol=1.0e-11)
    else:
        assert not np.isfinite(float(actual.item()))
