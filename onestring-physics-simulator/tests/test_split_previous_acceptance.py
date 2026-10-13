"""The Paper checkbox must NOT alter the OFF-mode Split processing.

Run the same synthetic M2D chart through both modes and assert identical
mesh topology, cuts, peak selection, CSF diagnostics and stopping conditions.
The OFF algorithm deliberately has an invariant fixed-chart CSF floor.
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


_METRICS = (
    "number_of_splits", "csf_split_step_count", "max_csf_before_split",
    "max_csf_after_split", "csf_split_component_sigma", "csf_split_status",
    "csf_split_unresolved_component_count", "csf_split_budget_exhausted",
    "csf_split_stopping_reason", "csf_split_stopping_rule",
    "csf_split_rejected_components", "split_boundary_pairs",
    "csf_split_face_sigma", "csf_model",
)


def _assert_identical(a, b):
    for x, y in zip(a[:2], b[:2]):
        np.testing.assert_array_equal(x, y)
    assert a[2] == b[2]
    for name in _METRICS:
        assert a[3][name] == b[3][name], name
    assert a[3]["csf_split_algorithm_same_as_normal"] is True
    assert b[3]["csf_split_algorithm_same_as_normal"] is True


@pytest.mark.parametrize("peak_jacobian", [1.9, 4.2, 12.5])
def test_split_solver_and_stopping_are_identical_in_both_modes(
    monkeypatch, peak_jacobian
):
    lo=np.ones(16)
    hi=np.full(16,peak_jacobian)
    off=_run(monkeypatch, False, lo, hi)
    on=_run(monkeypatch, True, lo, hi)
    _assert_identical(off, on)
    # Under the CURRENT OFF rule, worst-quad CSF is an invariant lower
    # bound and halts the candidate before any cuts are made.
    assert off[3]["number_of_splits"]==0
    assert off[3]["csf_split_stopping_reason"]==(
        "threshold_met" if peak_jacobian<=4.0 else
        "no_admissible_cut_remaining"
    )


def test_split_parity_preserves_custom_threshold_and_budget(monkeypatch):
    lo=np.ones(16)
    hi=np.linspace(1.,9.,16)
    off=_run(monkeypatch,False,lo,hi,threshold=1.7,budget=8)
    on=_run(monkeypatch,True,lo,hi,threshold=1.7,budget=8)
    _assert_identical(off,on)
    assert on[3]["csf_split_threshold"]==1.7
    assert on[3]["max_csf_splits"]==8


def test_split_parity_includes_explicit_collision_feedback_cut(monkeypatch):
    lo=np.ones(16)
    hi=np.ones(16)
    forced=[dict(axis="col",value=2.,source_face_id=5)]
    off=_run(monkeypatch,False,lo,hi,forced=forced,budget=0)
    on=_run(monkeypatch,True,lo,hi,forced=forced,budget=0)
    _assert_identical(off,on)
    assert on[3]["collision_split_count"]==1
    assert on[3]["csf_split_step_count"]==0
