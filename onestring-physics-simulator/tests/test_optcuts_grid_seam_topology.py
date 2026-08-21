from __future__ import annotations

import numpy as np

from onestring_physics.optcuts_grid_seam_topology_patch import (
    _duplicate_vertices_along_cut_edges,
)


def test_zero_width_cut_duplicates_only_topology() -> None:
    # Two quads sharing vertical edge (1, 4).
    vertices = np.asarray(
        [
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)
    out_v, out_f, duplicated = _duplicate_vertices_along_cut_edges(
        vertices, faces, {(1, 4)}
    )

    assert duplicated == 2
    assert len(out_v) == len(vertices) + 2
    # Geometry is unchanged: duplicated ids occupy exactly the same positions.
    assert np.allclose(np.sort(out_v[:, 0]), np.sort(np.r_[vertices[:, 0], 1.0, 1.0]))
    # The two quads have no shared vertex ids after the shared seam edge is cut.
    assert set(map(int, out_f[0])).isdisjoint(set(map(int, out_f[1])))
    assert int(out_f[0, 1]) != int(out_f[1, 0])
    assert int(out_f[0, 2]) != int(out_f[1, 3])


def test_open_slit_does_not_require_global_component_split() -> None:
    # 2x2 quad grid. Cut only the lower half of the center vertical grid line:
    # this is an open slit whose faces remain globally connected around the tip.
    vertices = np.asarray(
        [[x, y] for y in (0.0, 1.0, 2.0) for x in (0.0, 1.0, 2.0)],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 4, 3], [1, 2, 5, 4],
            [3, 4, 7, 6], [4, 5, 8, 7],
        ],
        dtype=int,
    )
    out_v, out_f, duplicated = _duplicate_vertices_along_cut_edges(
        vertices, faces, {(1, 4)}
    )

    # The boundary endpoint is duplicated even though the mesh as a whole stays
    # connected around the interior slit tip.
    assert duplicated >= 1
    left_lower = out_f[0]
    right_lower = out_f[1]
    assert int(left_lower[1]) != int(right_lower[0])
