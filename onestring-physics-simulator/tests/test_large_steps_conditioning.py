import numpy as np

from onestring_physics.large_steps_mesh_conditioning import (
    LargeStepsMeshConditioningConfig,
    _boundary_vertices,
    _triangle_quality_metrics,
    condition_mesh_with_large_steps,
)


def _skinny_disk():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.08, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )
    return vertices, faces


def test_large_steps_conditioning_improves_skinny_open_disk_and_keeps_boundary():
    vertices, faces = _skinny_disk()
    before = _triangle_quality_metrics(vertices, faces)
    boundary = _boundary_vertices(faces, len(vertices))

    conditioned, metrics = condition_mesh_with_large_steps(
        vertices,
        faces,
        LargeStepsMeshConditioningConfig(
            lambda_=10.0,
            max_iterations=80,
            learning_rate=0.06,
        ),
    )
    after = _triangle_quality_metrics(conditioned, faces)

    assert np.allclose(conditioned[boundary], vertices[boundary])
    assert conditioned.shape == vertices.shape
    assert after["minimum_angle_degrees"] > before["minimum_angle_degrees"]
    assert after["triangle_quality_p05"] > before["triangle_quality_p05"]
    assert metrics["large_steps_connectivity_changed"] is False
    assert metrics["large_steps_boundary_fixed"] is True
    assert metrics["large_steps_surface_deviation_max"] < 1.0e-9


def test_large_steps_conditioning_can_be_disabled():
    vertices, faces = _skinny_disk()
    conditioned, metrics = condition_mesh_with_large_steps(
        vertices,
        faces,
        LargeStepsMeshConditioningConfig(enabled=False),
    )
    assert np.array_equal(conditioned, vertices)
    assert metrics["large_steps_conditioning_enabled"] is False
