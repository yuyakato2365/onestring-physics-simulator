"""Noninvasive regression coverage for failed Ω -> staircase Grid fitting."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from onestring_physics.optcuts_grid_failure_diagnostics import (
    analyze_failure,
    proper_boundary_crossings,
    reflex_corner_indices,
    save_failure_bundle,
)


def fixture_uv():
    uv = np.array([
        [0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.]
    ])
    face = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
    ], dtype=int)
    loop = np.arange(4, dtype=int)
    outline = np.array([
        [0., 0.], [2., 0.], [2., 1.],
        [1., 1.], [1., 2.], [0., 2.]
    ])
    rejected = uv.copy()
    rejected[4] = [.01, .01]
    harmonic = uv.copy()
    harmonic[4] = [-1., -1.]
    return uv, face, loop, outline, rejected, harmonic


def test_proper_crossings_on_bow_tie_but_not_square():
    square = np.array([[0., 0.], [2., 0.],
                       [2., 2.], [0., 2.]])
    bow_tie = np.array([[0., 0.], [2., 2.],
                        [2., 0.], [0., 2.]])
    assert proper_boundary_crossings(square)["count"] == 0
    result = proper_boundary_crossings(bow_tie)
    assert result["count"] == 1
    assert result["examples"] == [[0, 2]]


def test_reflex_indices_independent_of_polygon_winding():
    _, _, _, outline, _, _ = fixture_uv()
    indices = reflex_corner_indices(outline)
    assert len(indices) == 1
    np.testing.assert_allclose(outline[indices[0]], [1., 1.])
    reversed_polygon = outline[::-1]
    rev_idx = reflex_corner_indices(reversed_polygon)
    assert len(rev_idx) == 1
    np.testing.assert_allclose(reversed_polygon[rev_idx[0]], [1., 1.])


def test_diagnostic_bundle_saved_with_reproducible_uv_and_hotspot_svg(tmp_path):
    uv, faces, loop, outline, rejected, harmonic = fixture_uv()
    uv_copy, rejected_copy = uv.copy(), rejected.copy()
    diagnostics = dict(
        base=uv,
        faces=faces,
        boundary_indices=loop,
        mapped_boundary=uv[loop],
        accepted=uv,
        rejected=rejected,
        harmonic=harmonic,
        grid_outline=outline,
        grid_step=np.ones(2),
        fraction=.995,
        last_valid_fraction=.99,
        history=[
            {"fraction": .99, "accepted": True, "min_area_ratio": .01},
            {"fraction": .995, "accepted": False,
             "min_area_ratio": .001}
        ],
        grid_mask=np.array([[True, False], [True, True]]),
        grid_coverage=np.array([[.75, .2], [1., .55]]),
        source_vertices_3d=np.c_[uv, np.ones(len(uv))],
        source_faces=faces.copy(),
        metadata={"attempt_tag": "test", "phase": 11, "method": "corner_dp"},
        output_dir=tmp_path,
    )
    where = Path(save_failure_bundle(**diagnostics))
    assert where.parent == tmp_path
    assert (where / "diagnosis.json").exists()
    assert (where / "reproduce_uv.npz").exists()
    assert (where / "overview.svg").exists()
    assert (where / "worst_triangle_zoom.svg").exists()
    assert (tmp_path / "index.jsonl").exists()
    report = json.loads((where / "diagnosis.json").read_text())
    assert report["schema"] == "onestring-grid-failure-diagnostic-v1"
    assert report["reflex_corner_count"] == 1
    assert "source_xyz_centroid" in report["ranked_worst_triangles"][0]
    assert report["metadata"]["phase"] == 11
    assert report["boundary_fraction_rejected"] == pytest.approx(.995)
    assert report["rejected_below_existing_0_005_area_guard_count"] >= 0
    assert len(report["ranked_worst_triangles"]) == len(faces)
    assert "diagnostic_hypothesis" in report
    assert "prove" in report["limitations"]
    with np.load(where / "reproduce_uv.npz") as saved:
        np.testing.assert_allclose(saved["base_uv"], uv)
        np.testing.assert_allclose(saved["rejected_uv"], rejected)
        np.testing.assert_array_equal(saved["uv_faces"], faces)
        np.testing.assert_array_equal(saved["source_faces"], faces)
        np.testing.assert_allclose(saved["source_vertices_3d"][:, 2], 1.)
        np.testing.assert_array_equal(saved["grid_mask"], diagnostics["grid_mask"])
        np.testing.assert_array_equal(saved["reflex_corner_indices"], [3])
    overview = (where / "overview.svg").read_text()
    zoom = (where / "worst_triangle_zoom.svg").read_text()
    assert "<svg" in overview and "</svg>" in overview
    assert "<svg" in zoom and "</svg>" in zoom
    assert "worst triangle" in overview
    np.testing.assert_array_equal(uv, uv_copy)
    np.testing.assert_array_equal(rejected, rejected_copy)


def test_intermediate_crossing_is_distinguished_from_corner_compression():
    uv, faces, loop, outline, rejected, harmonic = fixture_uv()
    # Boundary target takes opposite corners in crossed order; though not
    # itself a valid M2D outline, the diagnostic must say *what crossed*.
    crossing_boundary = uv[loop][[0, 2, 1, 3]]
    report, _ = analyze_failure(
        uv, faces, loop, crossing_boundary, uv, rejected, harmonic,
        outline, np.ones(2), 1., .95,
    )
    assert report["proper_boundary_intersections"][
        "intended_full_boundary"]["count"] >= 1


def test_failed_continuation_preserves_its_original_error_and_calls_diagnosis(
    monkeypatch, tmp_path
):
    import onestring_physics.optcuts_grid_boundary_untangle as untangle

    uv, faces, loop, outline, rejected, harmonic = fixture_uv()

    def folded_harmonic(_base, _faces, ids, destination):
        out = harmonic.copy()
        out[ids] = destination
        return out

    def always_invalid(_base, _faces, _loop, _target, _harmonic, _previous):
        return harmonic.copy(), {"solver": "synthetic_stuck", "iterations": 0}

    monkeypatch.setattr(untangle, "untangle_fixed_boundary", always_invalid)
    recorded = []

    def diagnose(**kwargs):
        recorded.append(kwargs["rejected_fraction"])
        return save_failure_bundle(
            base=kwargs["base"], faces=kwargs["faces"],
            boundary_indices=kwargs["loop"], mapped_boundary=kwargs["mapped"],
            accepted=kwargs["last"], rejected=kwargs["rejected"],
            harmonic=kwargs["harmonic"], grid_outline=outline,
            grid_step=np.ones(2), fraction=kwargs["rejected_fraction"],
            last_valid_fraction=kwargs["progress"],
            history=kwargs["history"], output_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="GRID_BOUNDARY_BIJECTIVE_MAPPING_NOT_FOUND") as exc:
        untangle.fit_by_continuation(
            uv, faces, loop, uv[loop], folded_harmonic,
            lambda *_: {"flipped_triangles": 0,
                        "degenerate_triangles": 0,
                        "injectivity_overlap_pairs": 0},
            on_failure=diagnose,
        )
    assert len(recorded) == 1
    assert "diagnostic_bundle=" in str(exc.value)
    assert len(list(tmp_path.glob("*/diagnosis.json"))) == 1


def test_diagnostic_io_failure_never_hides_original_mapping_error(monkeypatch):
    import onestring_physics.optcuts_grid_boundary_untangle as untangle
    uv, faces, loop, outline, rejected, harmonic = fixture_uv()
    monkeypatch.setattr(
        untangle, "untangle_fixed_boundary",
        lambda *_args: (harmonic.copy(), {"iterations": 0}),
    )
    def bad_diagnostic(**_):
        raise PermissionError("intentionally unwritable diagnostics folder")
    with pytest.raises(ValueError, match="BIJECTIVE_MAPPING_NOT_FOUND"):
        untangle.fit_by_continuation(
            uv, faces, loop, uv[loop],
            lambda *_args: harmonic.copy(),
            lambda *_: {}, on_failure=bad_diagnostic,
        )



def test_summary_synthesizes_failure_evidence_and_shareable_zip(tmp_path):
    from scripts.summarize_grid_boundary_failures import summarize
    from zipfile import ZipFile
    uv, faces, loop, outline, rejected, harmonic = fixture_uv()
    for rotation, progress in [(0., .96), (67.5, .99368)]:
        save_failure_bundle(
            base=uv, faces=faces, boundary_indices=loop,
            mapped_boundary=uv[loop], accepted=uv, rejected=rejected,
            harmonic=harmonic, grid_outline=outline,
            grid_step=np.ones(2), fraction=progress+.001,
            last_valid_fraction=progress,
            history=[{"fraction": progress, "accepted": True}],
            metadata={
                "official_seam_start_index": 0,
                "uv_rotation_degrees": rotation,
                "correspondence_attempt": 1,
                "phase": 7,
                "method": "corner_dp",
            },
            output_dir=tmp_path,
        )
    result = summarize(tmp_path, last=10, max_bundles=2)
    assert result["attempts"] == 2
    assert result["best_fraction"] == pytest.approx(.99368)
    assert Path(result["summary"]).exists()
    assert Path(result["csv"]).exists()
    assert Path(result["zip"]).exists()
    summary = Path(result["summary"]).read_text()
    assert "67.5°" in summary
    assert "0.993680" in summary
    assert "Next experiments justified" in summary
    with ZipFile(result["zip"]) as z:
        names = z.namelist()
        assert "analysis_summary.md" in names
        assert "comparison.csv" in names
        assert sum(x.endswith("reproduce_uv.npz") for x in names) == 2
