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


def _run(monkeypatch, by_col, paper=True, *, allow_neutral=True):
    vertices, faces, domain = _fixture(monkeypatch, by_col)
    params = SimpleNamespace(paper_fidelity_mode=paper,
                             paper_split_allow_neutral_cuts=allow_neutral,
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
    audit = metrics["csf_split_initial_one_cut_exhaustive_audit"]
    assert audit["complete_cut_candidate_count"] > 0
    assert audit["any_single_cut_below_threshold"]
    assert audit["best_single_cut"]["worst_linear_csf"] < 2.0


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
    audit = metrics["csf_split_initial_one_cut_exhaustive_audit"]
    assert not audit["any_single_cut_below_threshold"]
    assert audit["best_single_cut"]["worst_linear_csf"] > 2.0


def test_legacy_strict_one_step_improvement_is_still_available(monkeypatch):
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.05), (4., 5.), (4., 5.), (1., 1.05)], allow_neutral=False)
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
    assert metrics["csf_split_initial_one_cut_exhaustive_audit"] is None


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
    assert metrics["csf_split_candidate_linear_bound_satisfied"]
    assert not metrics["csf_split_candidate_area_bound_satisfied"]


def test_paper_one_split_when_ratio_crosses_linear_threshold(monkeypatch):
    # A tiny margin above area ratio 4 requires a single topological cut.
    _, _, records, metrics = _run(monkeypatch, [
        (1., 1.), (1., 1.01), (4.01, 4.03), (4.05, 4.08)])
    assert len(records) == 1
    assert metrics["max_csf_before_split"] == pytest.approx(np.sqrt(4.08))
    assert metrics["max_csf_after_split"] <= 2
    assert metrics["csf_split_stopping_reason"] == "threshold_met"


def test_experimental_physical_area_limit_can_require_a_cut_where_linear_two_does_not(monkeypatch):
    # Whole component linear ratio sqrt(3.02), lying between sqrt(2) and 2.
    # A generic bisection across x=2 separates both Jacobian bands.
    by_col = [(1.,1.01),(1.1,1.12),(2.90,2.92),(3.0,3.02)]
    v,f,d = _fixture(monkeypatch, by_col)
    legacy=SimpleNamespace(paper_fidelity_mode=True,collision_split_forced_lines=())
    _,_,old_records,old_metrics=splitting._split_mesh_single(v,f,d,legacy)
    assert len(old_records)==0
    assert old_metrics["csf_split_stopping_reason"]=="threshold_met"
    assert old_metrics["csf_split_area_ratio_after"] > 2.

    v,f,d=_fixture(monkeypatch,by_col)
    trial=SimpleNamespace(paper_fidelity_mode=True,
                          paper_split_area_limit_experiment=True,
                          collision_split_forced_lines=())
    _,_,records,metrics=splitting._split_mesh_single(v,f,d,trial)
    assert len(records)==1
    assert metrics["csf_split_effective_linear_threshold"]==pytest.approx(np.sqrt(2.))
    assert metrics["csf_split_requested_area_threshold"]==2.
    assert metrics["csf_split_area_limit_experiment"]
    assert metrics["csf_split_stopping_reason"]=="threshold_met"
    assert metrics["csf_split_candidate_area_bound_satisfied"]
    assert metrics["csf_split_area_ratio_after"] <= 2.
    # CSF fit alone is NOT fabrication certification.
    assert metrics["csf_split_bound_certified"] is False


def test_paper_allows_plateau_step_required_for_three_cut_solution(monkeypatch):
    # Astra's real-code counterexample: four 2-row columns with constant
    # per-quad Jacobians [1,9,1,9]. Any FIRST cut has worst span 3.
    # The reference description asks to keep splitting until each
    # component reaches its scale bound, not to insist each step improves.
    _, _, records, metrics = _run(
        monkeypatch, [(1.,1.),(9.,9.),(1.,1.),(9.,9.)]
    )
    assert len(records)==3, [r.get("axis") for r in records]
    assert metrics["max_csf_before_split"]==pytest.approx(3.)
    assert metrics["max_csf_after_split"]==pytest.approx(1.)
    assert metrics["csf_split_stopping_reason"]=="threshold_met"
    assert metrics["csf_split_neutral_cut_count"]>=1
    assert metrics["csf_split_neutral_cut_policy"]=="allow_hierarchical_plateau"
    for r in records:
        assert len(r["child_face_ids"])==2
        assert all(len(c)>0 for c in r["child_face_ids"])
        assert sum(len(c) for c in r["child_face_ids"])==len(r["face_ids"])


def test_strict_one_step_policy_reproduces_counterexample_failure(monkeypatch):
    _,_,records,metrics=_run(
        monkeypatch, [(1.,1.),(9.,9.),(1.,1.),(9.,9.)],
        allow_neutral=False,
    )
    assert records==[]
    assert metrics["max_csf_after_split"]==pytest.approx(3.)
    assert metrics["csf_split_stopping_reason"]=="no_admissible_cut_remaining"
    assert metrics["csf_split_neutral_cut_count"]==0
