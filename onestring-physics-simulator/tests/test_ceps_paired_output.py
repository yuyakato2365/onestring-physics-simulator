from __future__ import annotations

from pathlib import Path

import numpy as np

from onestring_physics.official_ceps import _parse_ceps_obj


def test_ceps_obj_duplicates_surface_vertex_across_uv_seam(tmp_path: Path) -> None:
    path = tmp_path / "seam.obj"
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
                # The same 3D vertex 1 has a different UV on the second face.
                "vt 0.2 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3",
                "f 1/4 3/5 4/6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _parse_ceps_obj(path)

    assert len(result.surface_vertices) == len(result.uv_vertices)
    assert np.array_equal(result.surface_faces, result.uv_faces)
    assert result.surface_vertices.shape == (5, 3)
    assert result.uv_vertices.shape == (5, 2)

    duplicate_origin = np.flatnonzero(
        np.all(np.isclose(result.surface_vertices, np.asarray([0.0, 0.0, 0.0])), axis=1)
    )
    assert len(duplicate_origin) == 2
    assert not np.allclose(
        result.uv_vertices[duplicate_origin[0]],
        result.uv_vertices[duplicate_origin[1]],
    )


def test_ceps_obj_without_seam_keeps_shared_vertices(tmp_path: Path) -> None:
    path = tmp_path / "plain.obj"
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

    result = _parse_ceps_obj(path)

    assert result.surface_vertices.shape == (4, 3)
    assert result.uv_vertices.shape == (4, 2)
    assert np.array_equal(result.surface_faces, result.uv_faces)
