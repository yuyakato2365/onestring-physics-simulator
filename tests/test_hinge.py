import numpy as np

from onestring_physics.hinge import PointHingeConstraint


def test_point_hinge_reduces_endpoint_error():
    tiles = np.zeros((2, 4, 3), dtype=float)
    tiles[1, 0] = np.array([1.0, 0.0, 0.0])
    hinge = PointHingeConstraint(0, 0, 1, 0, stiffness=1.0)

    before = np.linalg.norm(tiles[1, 0] - tiles[0, 0])
    hinge.solve(tiles, np.ones(2))
    after = np.linalg.norm(tiles[1, 0] - tiles[0, 0])

    assert after < before
