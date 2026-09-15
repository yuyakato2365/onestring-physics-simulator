from __future__ import annotations

import numpy as np

from onestring_physics.optcuts_grid_orientation_patch import (
    _orient_primary_loop_to_triangle_majority,
    _triangle_orientation_counts,
)


def _area(poly: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    )


def test_reversed_boundary_walk_does_not_force_global_reflection() -> None:
    uv = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    # Same valid positive UV map, but boundary walker happened to return CW.
    loops = [[0, 3, 2, 1]]
    out = _orient_primary_loop_to_triangle_majority(uv, faces, loops)
    assert _triangle_orientation_counts(uv, faces) == (2, 0, 0)
    assert _area(uv[np.asarray(out[0], dtype=int)]) > 0.0


def test_globally_reversed_map_requests_one_global_reflection() -> None:
    uv = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    loops = [[0, 3, 2, 1]]  # deliberately positive boundary walk
    out = _orient_primary_loop_to_triangle_majority(uv, faces, loops)
    assert _triangle_orientation_counts(uv, faces) == (0, 2, 0)
    assert _area(uv[np.asarray(out[0], dtype=int)]) < 0.0


def test_mixed_triangle_orientation_is_not_hidden() -> None:
    uv = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    # One positive and one negative triangle.  Majority orientation logic must
    # not turn this into a clean map; the backend's later per-triangle audit
    # will still report the remaining local flip.
    faces = np.asarray([[0, 1, 2], [0, 3, 2]], dtype=int)
    pos, neg, deg = _triangle_orientation_counts(uv, faces)
    assert (pos, neg, deg) == (1, 1, 0)
