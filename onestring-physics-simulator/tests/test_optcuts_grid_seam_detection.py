from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_grid_constrained_m2d_patch import _internal_seam_segments


def test_separate_uv_ids_with_same_coordinates_are_not_a_seam() -> None:
    # Two triangles share physical edge (0, 1).  OBJ-style vt ids differ across
    # faces even though the UV coordinates are identical.  This is NOT a cut.
    p = SimpleNamespace(
        surface_faces=np.asarray([[0, 1, 2], [1, 0, 3]], dtype=int),
        uv_faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
        uv_vertices_2d=np.asarray(
            [
                [0.0, 0.0],  # physical 0, face 0
                [1.0, 0.0],  # physical 1, face 0
                [0.0, 1.0],
                [1.0, 0.0],  # physical 1, face 1 (different vt id)
                [0.0, 0.0],  # physical 0, face 1 (different vt id)
                [1.0, 1.0],
            ],
            dtype=float,
        ),
    )
    seams = _internal_seam_segments(p)
    assert seams.shape == (0, 2, 2)


def test_geometrically_separated_uv_copies_are_a_seam() -> None:
    p = SimpleNamespace(
        surface_faces=np.asarray([[0, 1, 2], [1, 0, 3]], dtype=int),
        uv_faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
        uv_vertices_2d=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],  # physical 1, second seam side
                [0.0, 1.0],  # physical 0, second seam side
                [1.0, 2.0],
            ],
            dtype=float,
        ),
    )
    seams = _internal_seam_segments(p)
    assert seams.shape == (2, 2, 2)
    got = {tuple(map(tuple, seg)) for seg in seams}
    assert ((0.0, 0.0), (1.0, 0.0)) in got
    assert ((1.0, 1.0), (0.0, 1.0)) in got
