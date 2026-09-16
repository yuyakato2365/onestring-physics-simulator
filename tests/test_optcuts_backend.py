from __future__ import annotations

import numpy as np

from onestring_physics.optcuts_backend import (
    _boundary_loops,
    _read_obj_with_uv,
    _triangle_differential_metrics,
)


def test_read_obj_preserves_separate_surface_and_uv_indices(tmp_path):
    path = tmp_path / "cut.obj"
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3",
                "f 1/4 3/5 4/6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    xyz, faces, uv, uv_faces = _read_obj_with_uv(path)

    assert xyz.shape == (4, 3)
    assert faces.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert uv.shape == (6, 2)
    assert uv_faces.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_boundary_loop_detects_single_disk_boundary():
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    loops = _boundary_loops(faces)

    assert len(loops) == 1
    assert set(loops[0]) == {0, 1, 2, 3}


def test_identity_parameterization_has_symmetric_dirichlet_four():
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    uv = xyz[:, :2].copy()

    metrics = _triangle_differential_metrics(xyz, faces, uv, faces)

    assert metrics["uv_triangle_flip_count"] == 0
    assert metrics["uv_degenerate_triangle_count"] == 0
    assert np.isclose(metrics["symmetric_dirichlet_mean"], 4.0)
    assert np.isclose(metrics["symmetric_dirichlet_max"], 4.0)
