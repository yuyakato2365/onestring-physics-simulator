import numpy as np

from onestring_physics.rope import Rope


def test_rope_distance_constraint_reduces_stretch():
    rope = Rope.from_polyline(np.array([[0, 0, 0], [1, 0, 0]], dtype=float), closed=False)
    rope.particles[1].position = np.array([2.0, 0.0, 0.0])

    before = np.linalg.norm(rope.particles[1].position - rope.particles[0].position)
    rope.solve()
    after = np.linalg.norm(rope.particles[1].position - rope.particles[0].position)

    assert after < before
