from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_backend import OptCutsOutputError
from onestring_physics.optcuts_grid_seam_sidecar_patch import (
    _read_raw_seam_sidecar,
    _sidecar_path,
    _to_final_fabrication_segments,
)


def test_sidecar_path_tracks_final_result_obj() -> None:
    obj = Path("/tmp/output/run/finalResult_mesh.obj")
    assert _sidecar_path(obj) == Path("/tmp/output/run/finalResult_grid_seams.txt")


def test_read_sidecar_requires_authoritative_file(tmp_path: Path) -> None:
    with pytest.raises(OptCutsOutputError, match="SEAM_SIDECAR_MISSING"):
        _read_raw_seam_sidecar(tmp_path / "missing.txt")


def test_sidecar_roundtrip_and_fabrication_rotation(tmp_path: Path) -> None:
    path = tmp_path / "finalResult_grid_seams.txt"
    path.write_text("0 0 0.1 0\n0.1 0 0.1 0.2\n", encoding="utf-8")
    raw = _read_raw_seam_sidecar(path)
    assert raw.shape == (2, 2, 2)

    rotated = _to_final_fabrication_segments(raw, angle_degrees=90.0, reflected_v=False)
    expected = np.asarray(
        [
            [[0.0, 0.0], [0.0, -0.1]],
            [[0.0, -0.1], [0.2, -0.1]],
        ],
        dtype=float,
    )
    assert np.allclose(rotated, expected, atol=1.0e-12)


def test_sidecar_receives_same_global_v_reflection_as_uv() -> None:
    raw = np.asarray([[[0.0, 0.0], [0.1, 0.0]]], dtype=float)
    normal = _to_final_fabrication_segments(raw, angle_degrees=0.0, reflected_v=False)
    reflected = _to_final_fabrication_segments(raw, angle_degrees=0.0, reflected_v=True)
    assert np.allclose(reflected[:, :, 0], normal[:, :, 0])
    assert np.allclose(reflected[:, :, 1], -normal[:, :, 1])
