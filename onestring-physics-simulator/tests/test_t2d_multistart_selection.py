"""Regressions for T2D-only SE(2) multi-start selection and audit history.

Runs only the dual hinge solver with a small synthetic pair of rigid panels.
It does not rebuild Omega/M2D/K3D/K2D/T3D or execute the Streamlit UI.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import paper_hinge_solver, paper_t2d, solid_collision_audit


def _tile(x_offset):
    square = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float
    ) + [float(x_offset), 0.0]
    top = np.c_[square, np.zeros(4)]
    bottom = np.c_[square, -0.2 * np.ones(4)]
    return np.concatenate([top, bottom], axis=0)


def _data():
    vertices = np.stack((_tile(0.0), _tile(0.55)))
    identity = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    t2d = SimpleNamespace(
        vertices=vertices.copy(),
        transform_matrices=identity.copy(),
        top_to_bottom_transforms=identity.copy(),
        metrics={"section_4_3_linkage": True, "paper_classification": {}},
        stage="T2D top hinge",
    )
    t3d = SimpleNamespace(vertices=vertices.copy())
    params = SimpleNamespace(
        hinge_layout_connection_weight=8.0,
        hinge_layout_collision_weight=4.0,
        hinge_layout_anchor_weight=0.0,
        hinge_layout_initial_expansion=1.0,
        hinge_layout_max_candidate_pairs=3000,
        hinge_layout_time_budget_sec=0.0,
        hinge_layout_iterations=35,
        hinge_layout_restart_count=6,
        hinge_layout_perturb_translation=0.35,
        hinge_layout_perturb_rotation_deg=20.0,
        hinge_layout_random_seed=123,
        hinge_layout_search_mode="multi_start",
    )
    pipeline = SimpleNamespace(
        _tile_shape_distance_error=lambda *a, **k: 0.0,
        StageReport=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    return vertices, t2d, t3d, params, pipeline


@pytest.fixture(autouse=True)
def _stub_non_search_diagnostics(monkeypatch):
    def build_hinges(*args, **kwargs):
        hinge = SimpleNamespace(
            tile_a=0,
            tile_b=1,
            local_vertex_a=1,
            local_vertex_b=0,
            rest_position_2d=np.zeros(3),
        )
        return SimpleNamespace(hinges=[hinge], metrics={})

    monkeypatch.setattr(paper_t2d, "build_hinge_graph", build_hinges)
    monkeypatch.setattr(paper_t2d, "validate_flat_linkage", lambda *a, **k: {})
    monkeypatch.setattr(
        solid_collision_audit, "audit_solid_collisions",
        lambda *a, **k: {"solid_collision_pair_count": 0},
    )


def _solve(mode="multi_start", collision_mode="soft", legacy_drift=None):
    original, top, t3d, params, pipeline = _data()
    params.hinge_layout_search_mode = mode
    params.hinge_layout_collision_mode = collision_mode
    if legacy_drift is not None:
        # Legacy user-settings must no longer constrain either active solver.
        params.hinge_layout_max_center_drift_tiles = legacy_drift
    result, graph, report = paper_hinge_solver.optimize_hinge_poses(
        None, None, top, t3d, params, pipeline
    )
    np.testing.assert_array_equal(top.vertices, original)
    return original, top, result, graph, report


def test_single_mode_has_only_baseline_trial():
    _, _, out, _, _ = _solve("single")
    metrics = out.metrics
    assert metrics["dual_hinge_trials_requested"] == 1
    assert metrics["dual_hinge_trials_completed"] == 1
    assert metrics["dual_hinge_selected_trial"] == 0
    assert len(metrics["dual_hinge_trial_pose_histories"]) == 1
    assert metrics["dual_hinge_search_mode"] == "single"


def test_multistart_selects_best_and_reports_70_30_sources():
    original, top, out, _, _ = _solve()
    m = out.metrics
    trials = m["dual_hinge_trial_summary"]
    assert len(trials) == m["dual_hinge_trials_requested"] == 7
    assert m["dual_hinge_trials_completed"] == 7
    assert [x["trial_id"] for x in trials] == list(range(7))
    assert trials[0]["initial_source"] == "baseline Top Hinge"
    assert sum(x["initial_source"] == "perturbed best" for x in trials) == 4
    assert sum(x["initial_source"] == "perturbed Top Hinge" for x in trials) == 2
    assert sum(bool(x["became_best"]) for x in trials) >= 1
    winner = trials[m["dual_hinge_selected_trial"]]
    assert winner["initial_source"] == m["dual_hinge_selected_source"]
    assert np.isclose(winner["final_EHinge"], m["EHinge"], rtol=1e-10)
    assert m["EHinge"] <= min(x["final_EHinge"] for x in trials) + 1e-9
    assert m["EHinge"] <= trials[0]["final_EHinge"] + 1e-9
    assert len(m["dual_hinge_trial_pose_histories"]) == 7

    winning_trajectory = m["dual_hinge_trial_pose_histories"][m["dual_hinge_selected_trial"]]
    poses = winning_trajectory["pose_history"]
    assert poses[0]["iteration"] == 0
    assert poses[-1]["iteration"] == winner["iterations"]
    np.testing.assert_allclose(poses[-1]["poses"], m["dual_hinge_pose_history"][-1]["poses"])
    scale = 1.0
    pose = np.asarray(poses[-1]["poses"])
    centers = original[:, :4, :2].mean(axis=1)
    local = original[:, :, :2] - centers[:, None, :]
    expected = np.empty_like(local)
    for i, (angle, tx, ty) in enumerate(pose):
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        expected[i] = local[i] @ rot.T + centers[i] + scale * np.array([tx, ty])
    np.testing.assert_allclose(out.vertices[:, :, :2], expected, atol=1e-8)


def test_multistart_is_seed_reproducible():
    _, _, first, _, _ = _solve()
    _, _, second, _, _ = _solve()
    first_metrics, second_metrics = first.metrics, second.metrics
    assert first_metrics["dual_hinge_selected_trial"] == second_metrics["dual_hinge_selected_trial"]
    np.testing.assert_allclose(first.vertices, second.vertices, rtol=1e-9, atol=1e-9)
    for a, b in zip(first_metrics["dual_hinge_trial_summary"], second_metrics["dual_hinge_trial_summary"]):
        assert a["initial_source"] == b["initial_source"]
        np.testing.assert_allclose(a["final_EHinge"], b["final_EHinge"], rtol=1e-9, atol=1e-9)


def test_hard_solver_repairs_overlapping_rigid_tiles():
    original, top, out, _, _ = _solve(mode="single", collision_mode="hard")
    metrics = out.metrics
    assert metrics["dual_hinge_collision_constraint_mode"] == "hard"
    assert metrics["hard_collision_feasible"] is True
    assert metrics["hard_collision_status"] == "verified_feasible"
    assert metrics["flat_collision_count"] == 0
    assert metrics["collision_pair_count"] == 0
    assert metrics["hard_collision_repair_source_trial"] == 0
    assert metrics["hard_collision_repair_attempts"]
    assert metrics["hard_collision_history"]
    final_pose = np.asarray(metrics["dual_hinge_pose_history"][-1]["poses"])
    centers = original[:, :4, :2].mean(axis=1)
    local = original[:, :, :2]-centers[:, None, :]
    expected = np.empty_like(local)
    for i, (angle, tx, ty) in enumerate(final_pose):
        c, sn = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -sn], [sn, c]])
        expected[i] = local[i]@rot.T + centers[i] + np.array([tx,ty])
    np.testing.assert_allclose(out.vertices[:, :, :2], expected, atol=1e-8)
    assert metrics["hard_collision_repair_elapsed_sec"] >= 0
    assert metrics["hard_collision_refinement_elapsed_sec"] >= 0


def test_legacy_drift_setting_does_not_constrain_soft_or_hard():
    for collision_mode in ("soft", "hard"):
        _, _, output, _, _ = _solve(
            mode="single", collision_mode=collision_mode, legacy_drift=0.0,
        )
        assert output.metrics["dual_hinge_translation_bounds_enabled"] is False
        assert output.metrics["dual_hinge_translation_bounds"] == "unbounded in Soft and Hard"
        if collision_mode == "hard":
            assert output.metrics["hard_collision_feasible"] is True


def test_soft_mode_is_backward_compatible():
    _, _, out, _, _ = _solve("single", collision_mode="soft")
    metrics = out.metrics
    assert metrics["dual_hinge_collision_constraint_mode"] == "soft"
    assert metrics["hard_collision_feasible"] is None
    assert not metrics["hard_collision_history"]


@pytest.mark.parametrize("collision_mode", ["soft", "hard"])
def test_lbfgsb_does_not_receive_center_translation_bounds(monkeypatch, collision_mode):
    # Verify the actual scipy entry point (not only a metrics flag).
    actual_minimize=paper_hinge_solver.minimize
    seen=[]

    def checked_minimize(*args, **kwargs):
        seen.append(kwargs.get("bounds"))
        assert kwargs.get("bounds") is None
        return actual_minimize(*args, **kwargs)

    monkeypatch.setattr(paper_hinge_solver, "minimize", checked_minimize)
    _, _, output, _, _ = _solve(
        mode="single", collision_mode=collision_mode, legacy_drift=0.0,
    )
    assert seen == [None]
    assert output.metrics["dual_hinge_translation_bounds_enabled"] is False
