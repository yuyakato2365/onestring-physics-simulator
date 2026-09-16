from types import SimpleNamespace

import numpy as np

from onestring_physics import abd_backend
from onestring_physics.abd_builtin_compat import install_builtin_shape_abd_compatibility


def _tile(offset_x: float) -> np.ndarray:
    top = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    bottom = top - np.asarray([0.0, 0.0, 0.1])
    return np.vstack([top, bottom]) + np.asarray([offset_x, 0.0, 0.0])


def test_builtin_compat_derives_guides_when_string_path_is_empty():
    install_builtin_shape_abd_compatibility()
    proxy = np.asarray([_tile(0.0), _tile(1.2)])
    centers = np.mean(proxy, axis=1)
    state = SimpleNamespace(
        gap_graph=SimpleNamespace(gaps=[]),
        string_path=SimpleNamespace(gap_ids=[]),
        hinge_graph=SimpleNamespace(
            hinges=[SimpleNamespace(tile_a=0, tile_b=1)]
        ),
    )

    guides = abd_backend._string_guides(state, centers, proxy, proxy)

    assert len(guides) >= 2
    assert all(guide["source"] == "builtin_shape_fallback" for guide in guides)
    first = np.asarray(guides[0]["initial_world_point"], dtype=float)
    second = np.asarray(guides[1]["initial_world_point"], dtype=float)
    assert np.linalg.norm(second - first) > 0.0


def test_builtin_compat_keeps_valid_routed_guides():
    install_builtin_shape_abd_compatibility()
    proxy = np.asarray([_tile(0.0), _tile(1.2)])
    centers = np.mean(proxy, axis=1)
    gaps = [
        SimpleNamespace(id=0, surrounding_tiles=[0], centroid_2d=np.asarray([0.0, 0.0])),
        SimpleNamespace(id=1, surrounding_tiles=[1], centroid_2d=np.asarray([2.2, 1.0])),
    ]
    state = SimpleNamespace(
        gap_graph=SimpleNamespace(gaps=gaps),
        string_path=SimpleNamespace(gap_ids=[0, 1]),
        hinge_graph=SimpleNamespace(hinges=[]),
    )

    guides = abd_backend._string_guides(state, centers, proxy, proxy)

    assert len(guides) == 2
    assert all(guide["source"] == "routed_gap_path" for guide in guides)
