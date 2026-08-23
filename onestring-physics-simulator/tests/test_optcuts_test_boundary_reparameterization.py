from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_test_boundary_reparameterization_patch import (
    _build_test_targets,
    _quad_union_boundary,
)


def test_quad_union_boundary_extracts_outer_rectilinear_loop():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)
    outline = _quad_union_boundary(SimpleNamespace(vertices=vertices, faces=faces))
    assert len(outline) >= 5
    assert np.allclose(outline[0], outline[-1])
    assert set(map(tuple, np.round(outline[:-1], 8))) == {
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (0.0, 1.0),
    }


def test_test_targets_keep_seam_uv_copies_exactly_fixed():
    # Two triangles share physical edge (0, 1) but use distinct UV ids on each
    # side, so that edge is an OptCuts seam.  Vertex 2/3 supply non-seam outer
    # boundary vertices that should project to the grid outline.
    parameterization = SimpleNamespace(
        surface_vertices_3d=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ),
        surface_faces=np.array([[0, 1, 2], [1, 0, 3]], dtype=int),
        uv_vertices_2d=np.array(
            [
                [0.2, 0.2],  # seam copy side A: surface 0
                [0.8, 0.2],  # seam copy side A: surface 1
                [0.2, 0.8],  # outer
                [0.8, 0.25], # seam copy side B: surface 1
                [0.2, 0.25], # seam copy side B: surface 0
                [0.8, 0.8],  # outer
            ]
        ),
        uv_faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )
    outline = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=float
    )
    original = parameterization.uv_vertices_2d.copy()
    targets, info = _build_test_targets(parameterization, outline)
    for uid in (0, 1, 3, 4):
        assert np.allclose(targets[uid], original[uid])
    assert info["seam_fixed_vertex_count"] == 4
    assert info["outer_boundary_vertex_count"] == 2
    assert np.isfinite(targets[2]).all()
    assert np.isfinite(targets[5]).all()
