"""Regression tests for geometric screening and exact-seam UV untangling."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import eye

from onestring_physics import optcuts_seamless_joint_transitions as joint
from onestring_physics import optcuts_seamless_lattice_mode as mode
from onestring_physics.optcuts_seamless_uv_untangle import (
    refine_positive_uv, signed_areas,
)


def _square():
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    return SimpleNamespace(
        uv_vertices_2d=uv,
        uv_faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
    )


def test_two_discrete_alternatives_keep_exact_consistency(monkeypatch):
    chart = _square()
    chains = [((100, 101), (1,), (2,))]
    def candidates(uv, chain, h, *, per_chain=24):
        return [
            {"physical": chain[0], "a": chain[1], "b": chain[2],
             "k": 0, "steps": (0, 0), "rms": 0.},
            {"physical": chain[0], "a": chain[1], "b": chain[2],
             "k": 0, "steps": (0, 1), "rms": 1.},
        ]
    monkeypatch.setattr(joint, "candidate_transitions", candidates)
    choices, audit = joint.select_joint_transitions(
        chart.uv_vertices_2d, chains, 1.,
        return_alternatives=True, max_alternatives=6)
    assert len(choices) == 2
    assert [ts[0]["steps"] for ts, _ in choices] == [(0, 0), (0, 1)]
    assert audit["seamless_lattice_joint_alternatives_available"] == 2
    for transitions, _ in choices:
        mode._global_lattice_constraints(4, transitions, 1.)


def test_folded_first_discrete_candidate_is_rejected_for_valid_second(monkeypatch):
    chart = _square()
    chain = ((100, 101), (1,), (2,))
    monkeypatch.setattr(mode, "source_chains", lambda _: [chain])
    def candidates(uv, chain, h, *, per_chain=24):
        return [
            {"physical": chain[0], "a": chain[1], "b": chain[2],
             "k": 0, "steps": (0, 0), "rms": 0.},
            {"physical": chain[0], "a": chain[1], "b": chain[2],
             "k": 0, "steps": (0, 1), "rms": 1.},
        ]
    monkeypatch.setattr(joint, "candidate_transitions", candidates)
    uv, report = mode.solve_seamless_lattice_uv(chart, 1.)
    assert report["seamless_lattice_selected_alternative"] == 1
    assert report["seamless_lattice_geometry_attempts_before_success"] == 1
    assert report["seamless_lattice_selected_discrete_cost"] > 0
    assert report["seamless_lattice_source_relative_area_min"] > 1e-3
    np.testing.assert_allclose(uv[2] - uv[1], [0, 1], atol=1e-8)


def test_orientation_refinement_fixes_fold_in_reduced_coordinates():
    chart = _square()
    target = chart.uv_vertices_2d.copy()
    target[2] = [1., -0.5]
    roots, report = refine_positive_uv(
        eye(8, format="csr"), target.ravel(), target.ravel(),
        component=np.arange(4),
        matrices=np.tile(np.eye(2, dtype=int), (4, 1, 1)),
        offsets=np.zeros((4, 2)),
        fixed={}, columns={i: 2*i for i in range(4)},
        faces=chart.uv_faces,
        reference_uv=chart.uv_vertices_2d, h=1.)
    assert report["refinement_attempted"]
    assert report["refinement_bad_faces"] == 0
    assert np.min(signed_areas(roots.reshape(-1, 2), chart.uv_faces)) > 1e-3


def test_refinement_preserves_exact_affine_seam_relation():
    chart = _square()
    # UV[2] = UV[1] + (0, 1) exactly. Only 3 independent roots exist.
    offsets = np.zeros((4, 2))
    offsets[2] = [0, 1]
    roots = np.array([0., 3., 1., 0., 0., 1.])
    refined, result = refine_positive_uv(
        eye(6, format="csr"), roots, roots,
        component=np.array([0, 1, 1, 2]),
        matrices=np.tile(np.eye(2, dtype=int), (4, 1, 1)),
        offsets=offsets, fixed={},
        columns={0: 0, 1: 2, 2: 4}, faces=chart.uv_faces,
        reference_uv=chart.uv_vertices_2d, h=1.)
    u0, u1, u3 = refined.reshape(-1, 2)
    uv = np.stack((u0, u1, u1 + [0, 1], u3))
    np.testing.assert_allclose(uv[2] - uv[1], [0, 1], atol=1e-12)
    assert result["refinement_bad_faces"] == 0
    assert np.min(signed_areas(uv, chart.uv_faces)) > 1e-3


def test_impossible_collapsed_seam_keeps_failure_no_fallback(monkeypatch):
    chart = _square()
    chain = ((100, 101), (1,), (2,))
    monkeypatch.setattr(mode, "source_chains", lambda _: [chain])
    def candidates(uv, chain, h, *, per_chain=24):
        return [{
            "physical": chain[0], "a": chain[1], "b": chain[2],
            "k": 0, "steps": (0, 0), "rms": 0.,
        }]
    monkeypatch.setattr(joint, "candidate_transitions", candidates)
    with pytest.raises(mode.SeamlessLatticeError, match="UV_FLIP_OR_COLLAPSE") as exc:
        mode.solve_seamless_lattice_uv(chart, 1.)
    assert "no fallback" in str(exc.value)
    assert "repair_attempted=True" in str(exc.value)
