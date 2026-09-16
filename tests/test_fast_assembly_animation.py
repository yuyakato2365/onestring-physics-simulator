from __future__ import annotations

import numpy as np

from onestring_physics.fast_assembly_animation_patch import (
    _fast_vertices_at_frame,
    _prepare_rigid_motion,
)


def _quad() -> np.ndarray:
    return np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=float,
    )


def test_fast_rigid_motion_preserves_endpoints() -> None:
    start = np.stack([_quad(), _quad() + np.asarray([2.0, 0.0, 0.0])])
    target = start.copy()
    # Rotate the second tile 90 degrees about x around its centroid, then lift it.
    center = np.mean(target[1], axis=0, keepdims=True)
    local = target[1] - center
    rotation = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    target[1] = local @ rotation.T + center + np.asarray([0.0, 0.0, 1.5])

    rank = np.zeros(2, dtype=float)
    rigid = _prepare_rigid_motion(start, target)
    first = _fast_vertices_at_frame(start, target, rank, 0, 11, "simultaneous_hinge_contraction", rigid)
    last = _fast_vertices_at_frame(start, target, rank, 10, 11, "simultaneous_hinge_contraction", rigid)

    np.testing.assert_allclose(first, start, atol=1e-10)
    np.testing.assert_allclose(last, target, atol=1e-8)


def test_boundary_order_path_is_unchanged_at_endpoints() -> None:
    start = np.stack([_quad(), _quad() + np.asarray([2.0, 0.0, 0.0])])
    target = start + np.asarray([0.0, 0.0, 1.0])
    rank = np.asarray([0.0, 0.72])

    first = _fast_vertices_at_frame(start, target, rank, 0, 11, "boundary_string_order", None)
    last = _fast_vertices_at_frame(start, target, rank, 10, 11, "boundary_string_order", None)

    np.testing.assert_allclose(first, start, atol=1e-10)
    np.testing.assert_allclose(last, target, atol=1e-10)
