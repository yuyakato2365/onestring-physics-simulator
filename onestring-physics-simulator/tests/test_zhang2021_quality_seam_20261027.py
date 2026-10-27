"""Zhang2021 mode revision (2026-10-27 修正指導書): no count bias, panel uniformity gates,
hard OptCuts seam closure, phased selection.  Pure Python; no reference binaries needed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from test_zhang2021_quad_20261027 import cylinder_chart, grid_quads


# ------------------------------------------------------------ Fix A: count
def test_there_is_no_panel_count_term():
    from onestring_physics.zhang2021_candidate_selection import (
        DEFAULT_SELECTION_WEIGHTS, candidate_score, parse_weights)
    assert "count" not in DEFAULT_SELECTION_WEIGHTS
    with pytest.raises(ValueError, match="no panel-count term"):
        parse_weights("count=1")
    w = parse_weights("")
    good = {"e_size": 0.01, "e_angle": 0.01, "e_aspect": 0.0, "area_cv": 0.1, "worst_panel_smooth_max": 0.1}
    # identical panel quality -> identical score whatever the panel count / N_tar was
    a, _ = candidate_score(good, good, 0.05, 0.01, w)
    b, _ = candidate_score(dict(good), dict(good), 0.05, 0.01, w)
    assert a == b
    bad = {"e_size": 0.4, "e_angle": 0.3, "e_aspect": 0.5, "area_cv": 0.9, "worst_panel_smooth_max": 1.5}
    assert candidate_score(bad, bad, 0.05, 0.01, w)[0] > a


def test_density_input_is_not_a_goal():
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    assert not hasattr(mode, "DEFAULT_TARGET_QUADS")
    p = SimpleNamespace(zhang2021_target_quads=35, zhang2021_candidate_density_factors="1,2,4")
    assert mode.candidate_densities(p, 10.0) == [35, 70, 140]
    p = SimpleNamespace(zhang2021_target_quads=35, zhang2021_target_edge_length=0.5,
                        zhang2021_candidate_density_factors="1")
    assert mode.candidate_densities(p, 10.0) == [40]          # area / l^2


def test_low_quality_candidate_is_rejected_whatever_its_count():
    from onestring_physics.zhang2021_candidate_selection import feasibility
    m = {"zhang2021_m2d_quads_valid": True, "zhang2021_onestring_chart_injective": True,
         "zhang2021_m2d_topology_valid": True, "zhang2021_onestring_final_chart_area_relative_error": 0.0,
         "zhang2021_seam_preflight_mode": "hard", "zhang2021_m2d_quality_passed": True,
         "zhang2021_m3d_quality_passed": False, "zhang2021_m3d_quality_failures": ["edge ratio 6 > 3"],
         "zhang2021_odd_valence_interior_vertices": 0}
    checks = {c["constraint"]: c["ok"] for c in feasibility(m, route_needs_even_valence=False)}
    assert checks["C6_m3d_gate"] is False and not all(checks.values())


# ----------------------------------------------------- Fix B: uniformity
def test_thin_rectangles_pass_scaled_jacobian_but_fail_the_gate():
    from onestring_physics.zhang2021_panel_quality import panel_quality
    from onestring_physics.zhang2021_quad_metrics import scaled_jacobians
    uv, quads = grid_quads(2, 2, 0.0, 6.0, 0.0, 1.0)          # 3 x 0.5 panels
    assert np.min(scaled_jacobians(uv, quads)) == pytest.approx(1.0)
    rep = panel_quality(uv, quads, stage="M3D", planar_2d_orientation=1.0)
    assert not rep.passed and any("edge ratio" in f for f in rep.failures)
    square = panel_quality(*grid_quads(3, 3), stage="M3D", planar_2d_orientation=1.0)
    assert square.passed and square.metrics["e_size"] == pytest.approx(0.0, abs=1e-12)


def test_mixed_panel_sizes_and_worst_panel_are_detected():
    from onestring_physics.zhang2021_panel_quality import panel_quality
    xs = np.r_[0.0, 0.2, 1.2, 2.2, 3.2]                       # one narrow column
    uv = np.array([[x, y] for y in (0.0, 1.0) for x in xs])
    quads = np.array([[i, i + 1, i + 6, i + 5] for i in range(4)])
    rep = panel_quality(uv, quads, stage="M3D", planar_2d_orientation=1.0)
    assert not rep.passed
    assert rep.metrics["worst_edge_ratio_panel"] == 0 and rep.metrics["smallest_panel"] == 0
    assert rep.metrics["size_ratio_max_min"] > 2.0
    # thresholds are user settings, parsed scale-free
    from onestring_physics.zhang2021_panel_quality import PanelQualityThresholds
    loose = PanelQualityThresholds.from_text("max_edge_ratio=10, max_size_ratio=10, max_size_spread_p95_p5=10,"
                                             "max_e_size=10, max_area_cv=10")
    assert panel_quality(uv * 1000.0, quads, stage="M3D", thresholds=loose, planar_2d_orientation=1.0).passed
    with pytest.raises(ValueError):
        PanelQualityThresholds.from_text("max_panels=35")


def test_nonconvex_lifted_panel_is_flagged():
    from onestring_physics.zhang2021_panel_quality import panel_quality
    X = np.array([[0, 0, 0], [2, 0, 0], [1, 0.3, 0], [0, 2, 0.0]])   # dart (reflex corner 2)
    rep = panel_quality(X, np.array([[0, 1, 2, 3]]), stage="M3D")
    assert rep.metrics["flipped_or_nonconvex_panels"] == 1 and not rep.passed


# --------------------------------------------------------- Fix C: seams
def _cylinder_plan(monkeypatch, *, optimize=True):
    from onestring_physics import optcuts_zhang2021_quad_20261027 as mode
    xyz, f3, uv, f2 = cylinder_chart()
    quv, quads = grid_quads(6, 3, 0.0, 2 * np.pi, 0.0, 1.0)

    def fake_runner(*a, **k):
        return SimpleNamespace(quad_vertices=quv, quads=quads, metrics={
            "zhang2021_flipped_quads": 0, "zhang2021_quad_boundary_loops": 1, "zhang2021_n_patches": 1,
            "zhang2021_j_min": 1.0, "zhang2021_d_h_percent_bbox": 0.0})
    monkeypatch.setattr(mode, "run_reference_pipeline", fake_runner)
    return mode.plan_zhang2021_m2d(xyz, f3, uv, f2, SimpleNamespace(zhang2021_panel_optimization=optimize))


def test_open_cylinder_seam_is_hard_and_open_boundary_stays_open(monkeypatch):
    plan = _cylinder_plan(monkeypatch)
    m = plan.metrics
    assert m["zhang2021_seam_preflight_mode"] == "hard"
    # top and bottom circles are genuine open boundary of S: never constrained
    assert m["zhang2021_seam_preflight_open_boundary_edges"] == 12
    assert m["zhang2021_m2d_quads_valid"] is True


def test_nested_partition_is_exactly_eliminated_and_interleaving_is_refused():
    from onestring_physics.zhang2021_seam_reduction import (
        SeamReductionError, build_seam_reduction, nested_partition_audit)
    # coarse side a0-a1 (0..1), fine side b0-bt-b1 with bt at s=0.25; a0=b0, a1=b1
    a0, a1, b0, bt, b1 = 0, 1, 2, 3, 4
    rows = [[(a0, 1.0), (b0, -1.0)], [(a1, 1.0), (b1, -1.0)], [(a0, 0.75), (a1, 0.25), (bt, -1.0)]]
    red = build_seam_reduction(5, [], rows)
    X = np.random.default_rng(0).normal(size=(red.m, 3))
    x = red.expand(X)
    assert red.residual(x) < 1e-12 and len(red.t_vertices) == 1
    assert np.allclose(x[bt], 0.75 * x[a0] + 0.25 * x[a1])
    # interleaved: a T-vertex hosted by an edge ending at another T-vertex
    rows_bad = [[(0, 1.0), (2, -1.0)], [(0, 0.5), (1, 0.5), (3, -1.0)], [(2, 0.5), (3, 0.5), (1, -1.0)]]
    assert nested_partition_audit(4, [], rows_bad)["eliminable"] is False
    with pytest.raises(SeamReductionError, match="INTERLEAVED"):
        build_seam_reduction(4, [], rows_bad)


def test_unequal_cut_counts_are_aligned_as_nested_partition():
    from onestring_physics.zhang2021_seam_alignment import _match_subset
    small, large = [0.3, 0.7], [0.1, 0.32, 0.5, 0.69, 0.9]
    assert _match_subset(small, large) == [1, 3]
    assert _match_subset([], large) == []


def test_tip_valence_repair_by_diagonal_collapse_and_edge_rotation():
    from onestring_physics.zhang2021_seam_alignment import (
        _valid_after, diagonal_collapse, edge_rotation_towards)
    uv, quads = grid_quads(3, 3)
    boundary = {0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4}
    # corner 0 has valence 1; collapsing its quad merges interior vertex 5 into it
    out = diagonal_collapse(quads, 0, 0)
    assert np.sum(out == 0) == 3 and 5 not in out and _valid_after(out, boundary)
    # rotate the edge shared by quads 0 and 1 towards corner 0 -> one more panel at 0
    rot = edge_rotation_towards(quads, 0, 1, 0)
    assert rot is not None and np.sum(rot == 0) == 2


def test_soft_seam_is_never_accepted_by_the_k3d_gate():
    from onestring_physics.optcuts_zhang2021_quad_20261027 import Zhang2021QuadModeError
    from onestring_physics.zhang2021_verification import k3d_gate
    uv, quads = grid_quads(2, 1)
    X = np.c_[uv, np.zeros(len(uv))]
    out = SimpleNamespace(vertices=X, faces=quads, metrics={
        "seam_stitch_mode": "soft", "seam_exact_affine_rows": [[[0, 1.0], [2, -1.0]]]})
    with pytest.raises(Zhang2021QuadModeError, match="not hard-closed"):
        k3d_gate(out, SimpleNamespace(zhang2021_k3d_gate="max_edge_ratio=10"), route="optcuts_test2")
    # the same geometry with the seam closed exactly is accepted
    X2 = X.copy()
    X2[2] = X2[0]
    out = SimpleNamespace(vertices=X, faces=quads, metrics={"seam_stitch_mode": "hard", "seam_exact_affine_rows": []})
    k3d_gate(out, SimpleNamespace(), route="optcuts_test2")


def test_paper_k3d_keeps_the_seam_closed_at_every_step(monkeypatch):
    from onestring_physics.paper_local_global_solvers import optimize_paper_local_global_k3d
    from onestring_physics.zhang2021_seam_reduction import build_seam_reduction
    monkeypatch.setenv("ONESTRING_PAPER_K3D_ITERATIONS", "5")
    monkeypatch.setenv("ONESTRING_K3D_PLANARITY_POLISH", "0")
    xyz, f3, uv, f2 = cylinder_chart(n_theta=12, n_z=2)
    quv, quads = grid_quads(4, 2, 0.0, 2 * np.pi, 0.0, 1.0)
    th = quv[:, 0]
    X = np.c_[np.cos(th), np.sin(th), quv[:, 1]] + 0.01 * np.random.default_rng(1).normal(size=(len(quv), 3))
    left = [i for i in range(len(quv)) if quv[i, 0] == 0.0]
    right = [i for i in range(len(quv)) if np.isclose(quv[i, 0], 2 * np.pi)]
    groups = [sorted(pair) for pair in zip(left, right)]
    red = build_seam_reduction(len(X), groups, [])
    param = SimpleNamespace(surface_vertices_3d=xyz, surface_faces=f3)
    pipeline = SimpleNamespace(StageReport=lambda **kw: SimpleNamespace(**kw))

    class _Mesh(SimpleNamespace):
        def __init__(self, v, f, g, s, m, sl):
            super().__init__(vertices=v, faces=f, grid=g, stage=s, metrics=m, split_lines=sl)

    mesh = _Mesh(X, quads, None, "M3D", {}, [])
    out, _ = optimize_paper_local_global_k3d(None, mesh, param, SimpleNamespace(), pipeline=pipeline,
                                             seam_reduction=red)
    gap = max(float(np.linalg.norm(out.vertices[a] - out.vertices[b])) for a, b in groups)
    assert gap == 0.0
    assert out.metrics["k3d_seam_reduction_applied"] is True


# ------------------------------------------------------- verification
def test_final_verification_requires_collision_free_t2d_and_closed_seams():
    from onestring_physics.zhang2021_verification import final_verification
    uv, quads = grid_quads(2, 2)
    X = np.c_[uv, np.zeros(len(uv))]
    k3d = SimpleNamespace(vertices=X, faces=quads, metrics={"zhang2021_k3d_gate_passed": True,
                                                           "zhang2021_k3d_seam_rows": 0})
    ok_t2d = SimpleNamespace(metrics={"collision_pair_count": 0, "solid_collision_pair_count": 0,
                                      "hard_collision_status": "verified_feasible", "hinge_max": 0.0})
    state = SimpleNamespace(mesh_3d_optimized=k3d, tiles_3d=SimpleNamespace(metrics={"paper_t3d_panel_collision_count": 0}),
                            mesh_2d_optimized=SimpleNamespace(faces=quads), tiles_2d_dual_hinge=ok_t2d)
    assert final_verification(state, SimpleNamespace())["passed"] is True
    state.tiles_2d_dual_hinge = SimpleNamespace(metrics={"collision_pair_count": 3, "hinge_max": 0.0})
    v = final_verification(state, SimpleNamespace())
    assert v["passed"] is False and any(c["check"] == "2D collisions" and not c["ok"] for c in v["checks"])
