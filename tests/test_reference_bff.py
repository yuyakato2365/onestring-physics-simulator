import json
import os
from pathlib import Path

import numpy as np
import pytest

from onestring_physics.reference_bff import (
    ReferenceBFFUnavailableError,
    ReferenceMeshValidationError,
    normalize_uv_and_compute_csf,
    run_official_bff,
    strict_inverse_map_uv_to_surface,
    triangle_jacobian_diagnostics,
    validate_reference_mesh,
)


GOLDEN = Path(__file__).parent / "golden" / "official_bff_plane_windows_v1_6.json"


def _fixture():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return data, np.asarray(data["vertices"], dtype=float), np.asarray(data["faces"], dtype=int)


def _similarity_rms(actual, expected):
    a = np.asarray(actual, dtype=float) - np.mean(actual, axis=0)
    b = np.asarray(expected, dtype=float) - np.mean(expected, axis=0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    aligned = a @ rotation
    scale = float(np.sum(aligned * b) / np.sum(aligned * aligned))
    return float(np.sqrt(np.mean(np.sum((scale * aligned - b) ** 2, axis=1))))


def test_reference_mesh_validation_accepts_one_triangle_disk():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    diagnostics = validate_reference_mesh(vertices, np.asarray([[0, 1, 2]], dtype=int))
    assert diagnostics["disk_topology"] is True
    assert diagnostics["boundary_loop_count"] == 1


def test_reference_mesh_validation_rejects_degenerate_triangle_without_repair():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ReferenceMeshValidationError, match="degenerate triangle count=1"):
        validate_reference_mesh(vertices, np.asarray([[0, 1, 2]], dtype=int))


def test_exact_triangle_jacobian_for_planar_identity():
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    diagnostics = triangle_jacobian_diagnostics(vertices, vertices[:, :2], np.asarray([[0, 1, 2]], dtype=int))
    np.testing.assert_allclose(diagnostics["sigma1"], [1.0], atol=1e-12)
    np.testing.assert_allclose(diagnostics["sigma2"], [1.0], atol=1e-12)
    assert diagnostics["uv_triangle_flip_count"] == 0


def test_strict_inverse_map_round_trip_has_no_nearest_fallback():
    data, vertices, faces = _fixture()
    uv = np.asarray(data["official_uv"], dtype=float)
    point = np.mean(uv[faces[0]], axis=0)
    mapped, triangle_id, barycentric, error = strict_inverse_map_uv_to_surface(point, uv, faces, vertices, faces)
    assert triangle_id == 0
    np.testing.assert_allclose(barycentric, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    np.testing.assert_allclose(mapped, np.mean(vertices[faces[0]], axis=0), atol=1e-12)
    assert error < 1e-12


def test_unavailable_official_backend_fails_instead_of_falling_back(tmp_path):
    _data, vertices, faces = _fixture()
    old = os.environ.pop("ONESTRING_BFF_EXECUTABLE", None)
    try:
        with pytest.raises(ReferenceBFFUnavailableError, match="No substitute was used"):
            run_official_bff(vertices, faces, executable=tmp_path / "missing-bff-command-line")
    finally:
        if old is not None:
            os.environ["ONESTRING_BFF_EXECUTABLE"] = old


def test_official_bff_matches_committed_golden_without_skip():
    data, vertices, faces = _fixture()
    result = run_official_bff(vertices, faces, boundary_policy="boundary_scale_zero")
    error = _similarity_rms(result.uv_vertices, np.asarray(data["official_uv"], dtype=float))
    assert error < 1e-10
    assert result.metrics["fallbacks_used"] == []


def test_plane_lambda_normalizes_to_one():
    data, vertices, faces = _fixture()
    uv, diagnostics = normalize_uv_and_compute_csf(vertices, np.asarray(data["official_uv"], dtype=float), faces)
    assert uv.shape == (5, 2)
    np.testing.assert_allclose(diagnostics["lambda"], np.ones(4), atol=1e-12)
    assert diagnostics["lambda_normalization"] == "min_to_one_hypothesis_a"

