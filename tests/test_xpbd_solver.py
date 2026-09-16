import numpy as np

from onestring_physics.xpbd_solver import solve_distance_positions


def test_distance_projection_moves_points_toward_rest_length():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([2.0, 0.0, 0.0])

    n0, n1, error = solve_distance_positions(p0, p1, 1.0, 1.0, 1.0, stiffness=1.0)

    assert error > 0.0
    assert np.isclose(np.linalg.norm(n1 - n0), 1.0)
