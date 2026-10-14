"""Regression tests for Sec.4.5-inspired conformal-scale Split candidates.

Synthetic quad grids only; no paper-figure or full BFF result is certified.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import paper_mesh_splitting as splitting


def _fixture(monkeypatch, by_col):
    nx, ny = 4, 2
    verts = np.array([(float(x), float(y), 0.)
                      for y in range(ny + 1) for x in range(nx + 1)])

    def vid(x, y):
        return y * (nx + 1) + x

    faces = np.array([[vid(x, y), vid(x + 1, y),
                       vid(x + 1, y + 1), vid(x, y + 1)]
                      for y in range(ny) for x in range(nx)])
    lo = np.array([by_col[x][0] for y in range(ny) for x in range(nx)])
    hi = np.array([by_col[x][1] for y in range(ny) for x in range(nx)])
    assert np.all(hi >= lo) and np.all(lo > 0)
    peak = vid(2, 1)
    samples = [np.array([peak]) for _ in faces]
    monkeypatch.setattr(splitting, "_face_samples",
                        lambda *_args: (lo, hi, samples))
    curvature = np.zeros(len(verts))
    curvature[peak] = 30.
    monkeypatch.setattr(splitting, "gaussian_curvature",
                        lambda _p: curvature)
    parameterization = SimpleNamespace(
        uv_vertices_2d=verts[:, :2].copy(), metrics={}
    )
    domain = SimpleNamespace(
        parameterization=parameterization,
        csf_split_threshold=2.,
        max_csf_splits=64,
        csf_split_enabled=True,
        reference_grid_rotation_degrees=0.,
    )
    return verts, faces, domain


def _run(monkeypatch, by_col, paper=True):
    vertices, faces, domain = _fixture(monkeypatch, by_col)
    params = SimpleNamespace(paper_fidelity_mode=paper,
                             collision_split_forced_lines=())
    return splitting._split_mesh_single(vertices, faces, domain, params)


def test_paper_split_reduces_linear_conformal_scale_span_and_stops(monkeypatch):
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.02), (1.1, 1.12), (4., 4.02), (4.1, 4.12)])
    assert len(records) == 1
    assert records[0]["worst_sigma_reduction"] > 0
    assert 2 < metrics["max_csf_before_split"] < 2.1
    assert metrics["max_csf_after_split"] < 2
    assert metrics["csf_split_stopping_reason"] == "threshold_met"
    assert metrics["csf_split_candidate_area_bound_satisfied"]
    assert not metrics["csf_split_bound_certified"]
    assert not metrics["csf_split_component_rescale_applied"]
    assert metrics["csf_split_paper_area_span_active"]
    assert not metrics["csf_split_algorithm_same_as_normal"]


def test_paper_does_not_consume_64_cuts_for_irreducible_quad(monkeypatch):
    _, _, records, metrics = _run(monkeypatch, [
        (1., 5.), (1., 1.01), (1., 1.01), (1., 1.01)])
    assert records == []
    assert metrics["max_csf_after_split"] > 2
    assert metrics["csf_split_irreducible_quad_count"] == 2
    assert metrics["csf_split_rejected_components"][0]["reason"] == (
        "within_quad_distortion_floor"
    )
    assert not metrics["csf_split_candidate_area_bound_satisfied"]


def test_paper_does_not_adopt_non_improving_full_cut(monkeypatch):
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.05), (4., 5.), (4., 5.), (1., 1.05)])
    assert records == []
    assert metrics["max_csf_before_split"] == pytest.approx(np.sqrt(5.))
    assert metrics["max_csf_after_split"] == pytest.approx(np.sqrt(5.))
    assert metrics["csf_split_stopping_reason"] == "no_admissible_cut_remaining"
    assert not metrics["csf_split_candidate_area_bound_satisfied"]


def test_off_mode_keeps_its_original_invariant_behavior(monkeypatch):
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.02), (1.1, 1.12), (4., 4.02), (4.1, 4.12)], paper=False)
    assert records == []
    assert metrics["csf_split_algorithm_same_as_normal"]
    assert not metrics["csf_split_paper_area_span_active"]


def test_paper_distinguishes_area_jacobian_from_linear_scale(monkeypatch):
    # Ratio of 3 in *area* corresponds to sqrt(3) in conformal length.
    # It is not a Split trigger at the paper's lambda threshold 2.
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.), (1., 1.01), (2.95, 2.98), (2.99, 3.)])
    assert records == []
    assert metrics["max_csf_before_split"] == pytest.approx(np.sqrt(3.))
    assert metrics["csf_split_stopping_reason"] == "threshold_met"
    assert metrics["csf_split_linear_csf_area_relation"] == (
        "lambda=sqrt(area Jacobian ratio)"
    )
    assert metrics["csf_split_area_ratio_after"] == pytest.approx(3.)


def test_paper_one_split_when_ratio_crosses_linear_threshold(monkeypatch):
    # A tiny margin above area ratio 4 requires a single topological cut.
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.), (1., 1.01), (4.01, 4.03), (4.05, 4.08)])
    assert len(records) == 1
    assert metrics["max_csf_before_split"] == pytest.approx(np.sqrt(4.08))
    assert metrics["max_csf_after_split"] <= 2
    assert metrics["csf_split_stopping_reason"] == "threshold_met"
