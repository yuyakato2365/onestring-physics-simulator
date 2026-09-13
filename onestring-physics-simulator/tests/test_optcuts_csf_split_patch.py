from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_csf_split_patch import (
    _normalized_global,
    _part_sigma,
    _split_partition,
    _triangle_raw_lambda,
)


def _single_triangle(surface_scale: float = 2.0, uv_scale: float = 1.0):
    return SimpleNamespace(
        surface_vertices_3d=np.asarray(
            [[0.0, 0.0, 0.0], [surface_scale, 0.0, 0.0], [0.0, surface_scale, 0.0]],
            dtype=float,
        ),
        surface_faces=np.asarray([[0, 1, 2]], dtype=int),
        uv_vertices_2d=np.asarray(
            [[0.0, 0.0], [uv_scale, 0.0], [0.0, uv_scale]], dtype=float
        ),
        uv_faces=np.asarray([[0, 1, 2]], dtype=int),
        metrics={},
    )


def test_triangle_raw_lambda_is_uv_to_surface_stretch():
    parameterization = _single_triangle(surface_scale=2.0, uv_scale=1.0)
    raw = _triangle_raw_lambda(parameterization)
    assert raw.shape == (1,)
    assert np.isclose(raw[0], 2.0, atol=1.0e-12)


def test_global_uv_similarity_scale_is_removed_by_normalization():
    raw_a = _triangle_raw_lambda(_single_triangle(surface_scale=2.0, uv_scale=1.0))
    raw_b = _triangle_raw_lambda(_single_triangle(surface_scale=2.0, uv_scale=10.0))

    normalized_a, _ = _normalized_global(raw_a)
    normalized_b, _ = _normalized_global(raw_b)

    assert np.isclose(normalized_a[0], 1.0)
    assert np.isclose(normalized_b[0], 1.0)


def test_part_sigma_detects_bound_violation():
    raw = np.asarray([1.0, 1.5, 2.0000001], dtype=float)
    sigma = _part_sigma(raw, np.asarray([0, 1, 2], dtype=int))
    assert sigma > 2.0


def test_split_partition_uses_row_and_column_coordinates():
    centroids = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    ids = np.arange(4, dtype=int)

    left, right = _split_partition(centroids, ids, "col", 0.5)
    assert set(left.tolist()) == {0, 2}
    assert set(right.tolist()) == {1, 3}

    bottom, top = _split_partition(centroids, ids, "row", 0.5)
    assert set(bottom.tolist()) == {0, 1}
    assert set(top.tolist()) == {2, 3}
