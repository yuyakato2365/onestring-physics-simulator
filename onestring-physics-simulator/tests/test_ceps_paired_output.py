from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onestring_physics import official_ceps
from onestring_physics.official_ceps import _parse_ceps_obj


def test_ceps_obj_stitches_translated_cut_copies_into_one_chart(tmp_path: Path) -> None:
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
                # Same second triangle, translated in the CEPS cut layout.
                "vt 3 0",
                "vt 4 1",
                "vt 3 1",
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
    assert np.allclose(
        result.uv_vertices,
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        atol=1e-10,
    )
    assert np.allclose(result.vertex_uv[0], [0.0, 0.0])
    metrics = official_ceps._CEPS_LAST_CHART_METRICS
    assert metrics["ceps_continuous_chart_reconstructed"] is True
    assert metrics["ceps_internal_cut_seam_edge_count"] == 1
    assert metrics["ceps_convex_hull_boundary_used"] is False
    assert metrics["ceps_artificial_cap_faces_added"] == 0


def test_ceps_obj_without_cut_keeps_original_surface_connectivity(tmp_path: Path) -> None:
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
    assert official_ceps._CEPS_LAST_CHART_METRICS["ceps_internal_cut_seam_edge_count"] == 0


def test_ceps_obj_rejects_non_isometric_cut_instead_of_hiding_it(tmp_path: Path) -> None:
    path = tmp_path / "invalid-seam.obj"
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
                # Shared diagonal length no longer matches the first copy.
                "vt 3 0",
                "vt 5 1",
                "vt 3 1",
                "f 1/1 2/2 3/3",
                "f 1/4 3/5 4/6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="not consistently stitchable"):
        _parse_ceps_obj(path)
