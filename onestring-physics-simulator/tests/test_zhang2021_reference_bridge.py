"""Independent tests: no Zhang algorithm is simulated or claimed as reproduced."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from onestring_physics.zhang2021_reference_bridge import (
    Zhang2021BridgeError, audit_reference_quads,
    run_reference_adapter, to_onestring_m2d, export_source_chart,
)


@pytest.fixture
def square_source():
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    xyz = np.column_stack((uv, np.array([0., 0., .15, 0.])))
    return SimpleNamespace(uv_vertices_2d=uv, uv_faces=faces,
                           surface_vertices_3d=xyz, surface_faces=faces,
                           omega_boundary=np.vstack((uv, uv[0])),
                           metrics={}, method="optcuts_test")


def test_valid_output_preserves_paired_inverse_and_open_boundary(square_source):
    uv = square_source.uv_vertices_2d
    result = audit_reference_quads(square_source, uv, [[0, 1, 2, 3]])
    assert np.allclose(result.vertices_3d, square_source.surface_vertices_3d)
    assert result.diagnostics["quad_count"] == 1
    assert result.diagnostics["seam_status"] == "source_open_boundary_only"
    assert result.diagnostics["checked_inverse_samples"] == 9
    assert result.diagnostics["zhang2021_algorithm_verified"] is False
    m2d = to_onestring_m2d(result, tile_size=1., gap_size=.05)
    assert m2d.tile_count == 1
    assert m2d.grid.num_tiles == 1


def test_outside_omega_is_rejected_not_clipped(square_source):
    uv = square_source.uv_vertices_2d.copy()
    uv[2, 0] = 1.2
    with pytest.raises(Zhang2021BridgeError, match="OUTSIDE_SOURCE_OMEGA"):
        audit_reference_quads(square_source, uv, [[0, 1, 2, 3]])


def test_missing_large_region_is_rejected(square_source):
    uv = np.array([[.1, .1], [.9, .1], [.9, .9], [.1, .9]])
    with pytest.raises(Zhang2021BridgeError, match="UNDERCOVER_SOURCE_OMEGA"):
        audit_reference_quads(square_source, uv, [[0, 1, 2, 3]])


def test_inverted_bowtie_rejected(square_source):
    with pytest.raises(Zhang2021BridgeError, match="INVALID_QUAD_TOPOLOGY"):
        audit_reference_quads(
            square_source, square_source.uv_vertices_2d, [[0, 2, 1, 3]])


def test_missing_reference_adapter_is_explicit_error(square_source):
    with pytest.raises(Zhang2021BridgeError, match="NOT_CONFIGURED"):
        run_reference_adapter(square_source, [])


def test_explicit_external_adapter_exchange_npz(square_source):
    # Test our own file-exchange protocol, NOT Zhang's original C++ algorithm.
    script = ("import sys, numpy as np; "
              "d=np.load(sys.argv[1], allow_pickle=False); "
              "np.savez(sys.argv[2], vertices_uv=d['source_uv_vertices'], "
              "faces=np.array([[0,1,2,3]],dtype=int))")
    result = run_reference_adapter(
        square_source,
        [sys.executable, "-c", script, "{input}", "{output}"],
    )
    assert result.diagnostics["status"] == "onestring_geometric_preflight_passed"
    assert result.diagnostics["zhang2021_algorithm_verified"] is False


def test_reference_adapter_does_not_silently_fallback(square_source):
    script = "import sys; sys.exit(17)"
    with pytest.raises(Zhang2021BridgeError, match="REFERENCE_ADAPTER_EXIT_17"):
        run_reference_adapter(square_source,
                              [sys.executable, "-c", script, "{input}", "{output}"])


def test_original_chart_export_has_explicit_source_provenance(square_source, tmp_path):
    dest = tmp_path / "chart.npz"
    export_source_chart(square_source, dest)
    with np.load(dest, allow_pickle=False) as doc:
        assert set(doc.files) == {
            "source_uv_vertices", "source_uv_faces",
            "source_xyz_vertices", "source_xyz_faces", "omega_boundary"}
        assert np.array_equal(doc["source_uv_faces"], square_source.uv_faces)


def test_inconsistent_source_boundary_detected(square_source):
    square_source.omega_boundary = np.array(
        [[0., 0.], [1.1, 0.], [1.1, 1.], [0., 1.], [0., 0.]])
    with pytest.raises(Zhang2021BridgeError,
                       match="SOURCE_BOUNDARY_TRIANGLE_MISMATCH"):
        audit_reference_quads(square_source,
                              square_source.uv_vertices_2d,
                              [[0, 1, 2, 3]])


def test_clockwise_quads_have_valid_orientation(square_source):
    uv = square_source.uv_vertices_2d
    result = audit_reference_quads(square_source, uv, [[3, 2, 1, 0]])
    assert result.diagnostics["quad_count"] == 1


def test_undercovered_boundary_cannot_pass_seam_audit(square_source):
    uv = np.array([[.1, .1], [.9, .1], [.9, .9], [.1, .9]])
    with pytest.raises(Zhang2021BridgeError, match="QUAD_SEAM_PREFLIGHT_FAILED"):
        audit_reference_quads(
            square_source, uv, [[0, 1, 2, 3]],
            max_undercoverage_fraction=.99, require_seam_correspondence=True)


def test_standalone_cli_validates_reference_candidate(square_source, tmp_path):
    import json
    import subprocess
    from pathlib import Path
    source = tmp_path / "source.npz"
    candidate = tmp_path / "candidate.npz"
    report = tmp_path / "report.json"
    export_source_chart(square_source, source)
    np.savez(candidate, vertices_uv=square_source.uv_vertices_2d,
             faces=np.array([[0, 1, 2, 3]], dtype=int))
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, "scripts/audit_zhang2021_candidate.py",
        "--source", str(source), "--candidate", str(candidate),
        "--report", str(report),
    ], cwd=repo_root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    record = json.loads(report.read_text(encoding="utf-8"))
    assert record["quad_count"] == 1
    assert record["zhang2021_algorithm_verified"] is False
