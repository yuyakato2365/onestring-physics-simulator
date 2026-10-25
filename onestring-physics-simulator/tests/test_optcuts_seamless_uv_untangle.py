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


def test_relative_area_stage_repairs_tiny_triangle_without_ignoring_it():
    # The former absolute-area penalty costs O(area) for an equal relative
    # fold: a very small source triangle may remain inverted. Both reference
    # and solved signed-area ratios must be considered instead.
    original = np.array([[0., 0.], [1.e-4, 0.], [0., 1.e-4]])
    faces = np.array([[0, 1, 2]], dtype=int)
    folded = original.copy()
    folded[2, 1] = -1.e-4
    roots, report = refine_positive_uv(
        eye(6, format="csr"), folded.ravel(), folded.ravel(),
        component=np.arange(3),
        matrices=np.tile(np.eye(2, dtype=int), (3, 1, 1)),
        offsets=np.zeros((3, 2)), fixed={},
        columns={i: 2*i for i in range(3)},
        faces=faces, reference_uv=original, h=1.)
    final = roots.reshape((-1, 2))
    assert report["refinement_bad_faces"] == 0
    assert (signed_areas(final, faces) / signed_areas(original, faces)).min() > 1.e-3
    assert any(entry["stage"] == "relative_area"
               for entry in report["refinement_history"])


def test_refined_gradient_matches_finite_difference_for_reduced_roots(monkeypatch):
    # Check the chain-rule and area derivative used by L-BFGS-B with
    # nonidentity rotation and constrained duplicate UV vertices.
    from scipy.optimize import minimize as real_minimize
    inspected = []
    def checking_minimize(fn, x, **kwargs):
        f0, grad = fn(x)
        eps = 1.e-7
        for i in range(len(x)):
            perturb = np.zeros(len(x))
            perturb[i] = eps
            fd = (fn(x + perturb)[0] - fn(x - perturb)[0]) / (2. * eps)
            np.testing.assert_allclose(grad[i], fd, rtol=1.e-4, atol=2.e-5)
        inspected.append(True)
        return real_minimize(fn, x, **kwargs)

    from onestring_physics import optcuts_seamless_uv_untangle as untangle
    monkeypatch.setattr(untangle, "minimize", checking_minimize)
    original = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    root0 = np.array([0.2, 0.1])
    root1 = np.array([1., -0.3])
    root3 = np.array([-0.3, 1.1])
    roots = np.r_[root0, root1, root3]
    matrices = np.tile(np.eye(2, dtype=int), (4, 1, 1))
    matrices[2] = np.array([[0, -1], [1, 0]])
    offsets = np.zeros((4, 2))
    offsets[2] = [1., 0.]
    # UV2 is R*UV1 + (1, 0); reduced variables are roots 0,1,3.
    out, info = refine_positive_uv(
        eye(6, format="csr"), roots, roots,
        component=np.array([0, 1, 1, 2]),
        matrices=matrices, offsets=offsets, fixed={},
        columns={0: 0, 1: 2, 2: 4},
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        reference_uv=original, h=1., maxiter=10,
        penalties=(16.,))
    assert inspected
    assert np.all(np.isfinite(out))
