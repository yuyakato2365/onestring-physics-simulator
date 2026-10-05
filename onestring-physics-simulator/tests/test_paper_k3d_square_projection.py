import numpy as np

from onestring_physics.paper_local_global_solvers import _closest_square_projection


_TEMPLATE = np.array(
    [[-1.0, -1.0, 0.0],
     [ 1.0, -1.0, 0.0],
     [ 1.0,  1.0, 0.0],
     [-1.0,  1.0, 0.0]]
)


def _random_rotation(rng):
    a = rng.normal(size=(3, 3))
    u, _, vt = np.linalg.svd(a)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def test_closest_square_projection_is_identity_for_axis_aligned_square():
    projected = _closest_square_projection(_TEMPLATE)
    np.testing.assert_allclose(projected, _TEMPLATE, atol=1e-12, rtol=1e-12)


def test_closest_square_projection_is_identity_for_random_rigid_similarities():
    rng = np.random.default_rng(20261005)
    for _ in range(1000):
        rotation = _random_rotation(rng)
        scale = 10.0 ** rng.uniform(-2.0, 2.0)
        translation = rng.normal(size=3)
        square = scale * (_TEMPLATE @ rotation.T) + translation
        projected = _closest_square_projection(square)
        np.testing.assert_allclose(projected, square, atol=2e-11 * max(1.0, scale), rtol=2e-11)


def test_closest_square_projection_preserves_reversed_winding_square():
    square = _TEMPLATE[[0, 3, 2, 1]]
    projected = _closest_square_projection(square)
    np.testing.assert_allclose(projected, square, atol=1e-12, rtol=1e-12)
