from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_visualization_compat_patch import (
    _optcuts_seam_segments_2d,
    _seam_adjacent_m2d_face_ids,
)


def _state_with_duplicated_zero_width_seam():
    # Two quads share the same geometric seam x=1, but the right quad uses
    # duplicated seam vertices (6, 7), exactly as a topology cut does.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1 seam original
            [1.0, 1.0, 0.0],  # 2 seam original
            [0.0, 1.0, 0.0],  # 3
            [2.0, 0.0, 0.0],  # 4
            [2.0, 1.0, 0.0],  # 5
            [1.0, 0.0, 0.0],  # 6 seam duplicate
            [1.0, 1.0, 0.0],  # 7 seam duplicate
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
            [6, 4, 5, 7],
        ],
        dtype=int,
    )
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        _optcuts_grid_seam_paths=[[1, 2]],
    )
    return SimpleNamespace(mesh_2d_initial=mesh)


def test_final_grid_seam_segments_are_read_from_optcuts_paths():
    state = _state_with_duplicated_zero_width_seam()
    segments = _optcuts_seam_segments_2d(state)
    assert segments.shape == (1, 2, 2)
    assert np.allclose(segments[0], [[1.0, 0.0], [1.0, 1.0]])


def test_seam_adjacency_finds_both_sides_after_vertex_duplication():
    state = _state_with_duplicated_zero_width_seam()
    face_ids = _seam_adjacent_m2d_face_ids(state)
    assert face_ids.tolist() == [0, 1]
