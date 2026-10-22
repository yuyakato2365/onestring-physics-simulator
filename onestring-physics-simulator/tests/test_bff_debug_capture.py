"""Forensic BFF captures preserve real source XYZ and raw official UV.

Capture is strictly opt-in and does not repair collapsed/overlapping charts.
"""
import json
from pathlib import Path

import numpy as np

from onestring_physics.reference_bff import (
    _save_rectangular_bff_diagnostics,
)


def _example():
    xyz = np.array([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return xyz, faces


def test_capture_records_raw_collapsed_face_without_modifying_data(tmp_path, monkeypatch):
    xyz, faces = _example()
    uv = xyz[:, :2].copy()
    uv[2] = [2., 0.]
    originals = (xyz.copy(), faces.copy(), uv.copy())
    adapter = tmp_path / "bff-rect-command-line"
    adapter.write_bytes(b"test binary bytes")
    monkeypatch.setenv("ONESTRING_BFF_DEBUG_DIR", str(tmp_path / "captures"))
    prefix = _save_rectangular_bff_diagnostics(
        xyz, faces, uv, executable=adapter,
        corner_phase=0.0625, corner_positions=None, native_precision=17,
    )
    assert prefix is not None
    summary = json.loads(Path(str(prefix) + ".json").read_text())
    arr = np.load(str(prefix) + ".npz")
    assert summary["native_output_precision_digits"] == 17
    assert summary["candidate_corner_phase"] == 0.0625
    assert summary["uv_degenerate_faces"] >= 1
    assert 0 in arr["degenerate_face_ids"]
    assert summary["examined_faces"][0]["source_vertex_ids"] == [0, 1, 2]
    np.testing.assert_array_equal(arr["vertices"], xyz)
    np.testing.assert_array_equal(arr["uv_vertices"], uv)
    for old, current in zip(originals, (xyz, faces, uv)):
        np.testing.assert_array_equal(old, current)


def test_capture_detects_real_foldover(tmp_path, monkeypatch):
    xyz, faces = _example()
    uv = xyz[:, :2].copy()
    uv[2] = [-.2, .2]
    adapter = tmp_path / "bff-rect-command-line"
    adapter.write_bytes(b"sample")
    monkeypatch.setenv("ONESTRING_BFF_DEBUG_DIR", str(tmp_path / "captures"))
    prefix = _save_rectangular_bff_diagnostics(
        xyz, faces, uv, executable=adapter, corner_phase=0.0,
        corner_positions=xyz[[0, 1, 2, 3]], native_precision=17,
    )
    report = json.loads(Path(str(prefix) + ".json").read_text())
    assert report["uv_inverted_faces"] == 1
    assert report["uv_degenerate_faces"] == 0
    assert report["candidate_explicit_corner_xyz"] == xyz.tolist()


def test_capture_off_by_default(tmp_path, monkeypatch):
    xyz, faces = _example()
    adapter = tmp_path / "bff-rect-command-line"
    adapter.write_bytes(b"sample")
    monkeypatch.delenv("ONESTRING_BFF_DEBUG_DIR", raising=False)
    assert _save_rectangular_bff_diagnostics(
        xyz, faces, xyz[:, :2], executable=adapter, corner_phase=0.,
        corner_positions=None, native_precision=17,
    ) is None
    assert not list(tmp_path.glob("*.npz"))
