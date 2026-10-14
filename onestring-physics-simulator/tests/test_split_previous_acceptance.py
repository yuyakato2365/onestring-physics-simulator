"""Legacy OFF Split regression plus intentional separation of Paper policy.

The original OFF policy remains invariant; Paper now uses the distinct
per-component area-Jacobian span required for a Sec.4.5 candidate.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import paper_mesh_splitting as split


def _grid(n=4):
    vertices = np.array([[float(i), float(j), 0.]
                         for j in range(n+1) for i in range(n+1)])
    faces = np.array([[j*(n+1)+i,j*(n+1)+i+1,
                       (j+1)*(n+1)+i+1,(j+1)*(n+1)+i]
                      for j in range(n) for i in range(n)], dtype=int)
    return vertices, faces


def _run(monkeypatch, paper_mode, lo, hi, *, threshold=2.0, budget=64,
         forced=()):
    vertices, faces = _grid()
    center = 12  # highest source curvature inside the square
    curvature = np.full(len(vertices), -1.)
    curvature[center] = 10.
    samples = [np.array([center], dtype=int) for _ in faces]
    monkeypatch.setattr(split, "_face_samples", lambda *a: (lo, hi, samples))
    monkeypatch.setattr(split, "gaussian_curvature", lambda *a: curvature)
    domain = SimpleNamespace(
        parameterization=SimpleNamespace(
            metrics={}, uv_vertices_2d=vertices[:, :2],
        ),
        csf_split_enabled=True,
        csf_split_threshold=threshold,
        max_csf_splits=budget,
        reference_grid_rotation_degrees=0.,
    )
    params = SimpleNamespace(
        paper_fidelity_mode=paper_mode,
        collision_split_forced_lines=forced,
    )
    return split._split_mesh_single(vertices, faces, domain, params)


@pytest.mark.parametrize("peak_jacobian", [1.9, 4.2, 12.5])
def test_off_mode_preserves_invariant_floor(monkeypatch, peak_jacobian):
    lo = np.ones(16)
    hi = np.full(16, peak_jacobian)
    off = _run(monkeypatch, False, lo, hi)
    paper = _run(monkeypatch, True, lo, hi)
    assert off[3]["number_of_splits"] == 0
    assert off[3]["max_csf_before_split"] == pytest.approx(
        np.sqrt(peak_jacobian)
    )
    assert off[3]["csf_split_stopping_reason"] == (
        "threshold_met" if peak_jacobian <= 4.0
        else "no_admissible_cut_remaining"
    )
    assert off[3]["csf_split_algorithm_same_as_normal"] is True
    # The area span of the *same* chart is a different physical quantity.
    assert paper[3]["max_csf_before_split"] == pytest.approx(peak_jacobian)
    assert paper[3]["csf_split_algorithm_same_as_normal"] is False
    assert paper[3]["number_of_splits"] == 0


def test_paper_switch_preserves_custom_controls_but_not_csf_formula(monkeypatch):
    lo = np.ones(16)
    hi = np.linspace(1., 9., 16)
    off = _run(monkeypatch, False, lo, hi, threshold=1.7, budget=8)
    paper = _run(monkeypatch, True, lo, hi, threshold=1.7, budget=8)
    assert off[3]["csf_split_threshold"] == 1.7
    assert paper[3]["csf_split_threshold"] == 2.0
    assert off[3]["max_csf_splits"] == paper[3]["max_csf_splits"] == 8
    assert off[3]["max_csf_before_split"] == pytest.approx(3.)
    assert paper[3]["max_csf_before_split"] == pytest.approx(9.)
    assert not paper[3]["csf_split_bound_certified"]


def test_explicit_collision_feedback_is_orthogonal_to_csf_policy(monkeypatch):
    lo = np.ones(16)
    hi = np.ones(16)
    forced = [dict(axis="col", value=2., source_face_id=5)]
    off = _run(monkeypatch, False, lo, hi, forced=forced, budget=0)
    paper = _run(monkeypatch, True, lo, hi, forced=forced, budget=0)
    for x, y in zip(off[:2], paper[:2]):
        np.testing.assert_array_equal(x, y)
    assert off[2] == paper[2]
    assert off[3]["collision_split_count"] == paper[3]["collision_split_count"] == 1
    assert off[3]["csf_split_step_count"] == paper[3]["csf_split_step_count"] == 0
