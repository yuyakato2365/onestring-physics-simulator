from __future__ import annotations

import numpy as np

from onestring_physics.optcuts_test_boundary_reparameterization_patch import (
    _clean_polygon,
    _point_in_polygon_or_boundary,
    _quad_fully_contained,
)


def test_point_on_boundary_counts_as_inside() -> None:
    omega = _clean_polygon(
        np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        )
    )
    assert _point_in_polygon_or_boundary(np.array([0.0, 1.0]), omega, 1e-10)
    assert _point_in_polygon_or_boundary(np.array([1.0, 1.0]), omega, 1e-10)
    assert not _point_in_polygon_or_boundary(np.array([2.1, 1.0]), omega, 1e-10)


def test_strict_crop_keeps_only_fully_contained_quad() -> None:
    omega = _clean_polygon(
        np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
        )
    )
    inside = np.array([[0.2, 0.2], [1.0, 0.2], [1.0, 1.0], [0.2, 1.0]])
    partial = np.array([[1.5, 0.2], [2.2, 0.2], [2.2, 0.9], [1.5, 0.9]])
    boundary_aligned = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    assert _quad_fully_contained(inside, omega)
    assert not _quad_fully_contained(partial, omega)
    assert _quad_fully_contained(boundary_aligned, omega)


def test_concave_notch_rejects_cell_even_when_some_center_rule_could_pass() -> None:
    # Omega is a square with a rectangular notch cut downward from the top.
    omega = _clean_polygon(
        np.array(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 3.0],
                [1.8, 3.0],
                [1.8, 1.4],
                [1.2, 1.4],
                [1.2, 3.0],
                [0.0, 3.0],
            ]
        )
    )
    safe = np.array([[0.1, 0.1], [1.0, 0.1], [1.0, 1.0], [0.1, 1.0]])
    notch_crossing = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])

    assert _quad_fully_contained(safe, omega)
    assert not _quad_fully_contained(notch_crossing, omega)
