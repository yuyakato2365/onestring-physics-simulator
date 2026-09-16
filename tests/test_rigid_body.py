import numpy as np

from onestring_physics.rigid_body import RigidTile


def test_rigid_tile_integrates_force():
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    tile = RigidTile.from_corners(0, corners, mass=1.0)

    tile.accumulate_force(np.array([0.0, 0.0, 1.0]))
    tile.integrate(0.1, damping=0.0)

    assert tile.position[2] > 0.0
