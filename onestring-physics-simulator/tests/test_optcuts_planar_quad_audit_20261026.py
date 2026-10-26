"""Regression tests for fail-closed unstructured topology and true 3D quad quality.

These tests do not claim Bunny/OptCuts/T2D end-to-end certification.
"""
import numpy as np
import pytest

from onestring_physics.optcuts_planar_quad_20261026 import (
    PlanarQuadMeshingError, UnstructuredQuadGrid,
)
from onestring_physics.optcuts_planar_quad_refine_20261026 import _quad_penalty


def test_three_quad_owners_on_same_edge_are_rejected_not_turned_into_a_hinge():
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.],
                   [0., -1.], [1., -1.], [2., 0.], [2., 1.]])
    # All three faces use the physical edge (0,1).
    faces = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [0, 1, 6, 7]])
    with pytest.raises(PlanarQuadMeshingError, match="NONMANIFOLD_EDGE"):
        UnstructuredQuadGrid(uv, faces, 1.0, 0.08)


def test_adjacent_quads_with_the_same_oriented_shared_edge_are_rejected():
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.],
                   [2., 0.], [2., 1.]])
    faces = np.array([[0, 1, 2, 3], [1, 2, 5, 4]])
    with pytest.raises(PlanarQuadMeshingError, match="SHARED_EDGE_WINDING"):
        UnstructuredQuadGrid(uv, faces, 1.0, 0.08)


def test_valid_opposite_directed_shared_edge_remains_one_hinge():
    uv = np.array([[0., 0.], [1., 0.], [2., 0.],
                   [0., 1.], [1., 1.], [2., 1.]])
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    grid = UnstructuredQuadGrid(uv, faces, 1.0, 0.08)
    assert grid.num_tiles == 2
    assert len(grid.hinges) == 1
    hinge = grid.hinges[0]
    assert (hinge.tile_a, hinge.tile_b) == (0, 1)
    a = faces[0][[hinge.corner_a0, hinge.corner_a1]]
    b = faces[1][[hinge.corner_b0, hinge.corner_b1]]
    np.testing.assert_array_equal(a, b[::-1])


def test_m3d_diagonal_regularization_uses_3d_geometry():
    uv = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    xyz = np.array([[[0., 0., 0.], [1., 0., 0.],
                     [1., 1., 2.], [0., 1., 0.]]])
    score = float(_quad_penalty(uv, xyz, h_uv=1., h_xyz=1.)[0])
    e3 = np.roll(xyz, -1, axis=1) - xyz
    lens3 = np.linalg.norm(e3, axis=-1)
    variance = float(np.var(np.log(lens3), axis=1)[0])
    size = float(np.mean(np.log(lens3) ** 2, axis=1)[0])
    cos = np.sum(e3 * np.roll(e3, 1, axis=1), axis=-1) / (
        lens3 * np.roll(lens3, 1, axis=1))
    angle = float(np.mean(cos ** 2, axis=1)[0])
    # UV diagonal lengths equal sqrt(2), while XYZ diagonals equal sqrt(6)
    # and sqrt(2). This term was accidentally 0 in the original code.
    xyz_diagonal = float(np.log(np.sqrt(6.) / np.sqrt(2.)) ** 2)
    expected = 1.8 * variance + .7 * angle + .4 * size + .15 * .5 * xyz_diagonal
    assert score == pytest.approx(expected, abs=1e-12)
