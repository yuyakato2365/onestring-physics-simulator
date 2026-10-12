"""Two-cut lookahead must never commit a numerically invalid intermediate cut."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import onestring_pipeline as pipeline
from onestring_physics import paper_collision_split as collision
from onestring_physics.paper_collision_split_lookahead import (
    promising_rejected_first_cut, validate_real_split_trial,
)


def _grid_states():
    """One 2x2 grid, then one real column seam and one real partial row seam."""
    v0 = np.array([[i, j, 0.] for j in range(3) for i in range(3)], float)
    f0 = np.array([[0, 1, 4, 3], [1, 2, 5, 4],
                   [3, 4, 7, 6], [4, 5, 8, 7]], int)
    v1 = np.concatenate((v0, v0[[1, 4, 7]]))
    f1 = f0.copy()
    for face_id in (1, 3):
        f1[face_id] = [({1: 9, 4: 10, 7: 11}.get(int(i), int(i)))
                       for i in f1[face_id]]
    # Split the LEFT component horizontally, leaving the right connected.
    v2 = np.concatenate((v1, v1[[3, 4]]))
    f2 = f1.copy()
    f2[2] = [({3: 12, 4: 13}.get(int(i), int(i))) for i in f2[2]]
    seam1 = dict(complete_boundary_to_boundary=True,
                 split_boundary_pairs=[[0, 1, 1, 3], [2, 1, 3, 3]])
    seam2 = dict(complete_boundary_to_boundary=True,
                 split_boundary_pairs=[[0, 2, 2, 0]])

    def make_state(v, f, seams, collisions, energy, max_depth):
        quads = v[f]
        tiles = np.concatenate((quads, quads + [0., 0., -.1]), axis=1)
        return SimpleNamespace(
            conformal_domain=SimpleNamespace(paper_split_policy=True),
            mesh_2d_initial=SimpleNamespace(
                vertices=v.copy(), faces=f.copy(),
                metrics=dict(
                    collision_split_count=len(seams),
                    collision_split_final_pass_applied_count=len(seams),
                    collision_split_step_analysis=list(seams),
                ),
            ),
            mesh_2d_optimized=SimpleNamespace(vertices=v.copy(), faces=f.copy()),
            tiles_2d_top_hinge=SimpleNamespace(vertices=tiles.copy()),
            tiles_2d_dual_hinge=SimpleNamespace(
                vertices=tiles.copy(), metrics={},
            ),
            hinge_graph=SimpleNamespace(hinges=[]),
            fixture_quality=dict(
                status="ok", collision_pairs=collisions,
                collision_energy=energy, penetration_max=max_depth,
                hinge_rms=.05, tile_scale=1.,
            ),
        )
    return (
        make_state(v0, f0, [], 9, 9., .20),
        make_state(v1, f1, [seam1], 7, 5., .21),
        make_state(v2, f2, [seam1, seam2], 3, 2., .15),
    )


def test_promising_first_cut_allows_local_max_worsening_but_not_bad_energy():
    before = dict(status="ok", collision_pairs=9, collision_energy=9.)
    after = dict(status="ok", collision_pairs=7, collision_energy=5.,
                 penetration_max=.21)
    assert promising_rejected_first_cut(before, after)
    assert not promising_rejected_first_cut(before, dict(after, collision_energy=10.))
    # The original accept_if_improved allows up to max(2, ceil(0.02*p0))
    # additional pairs. Temporary lookahead parents should use that allowance.
    assert promising_rejected_first_cut(before, dict(after, collision_pairs=10))
    assert not promising_rejected_first_cut(before, dict(after, collision_pairs=12))
    # A zero-contact result with disconnected hinges has no collision hotspot
    # from which to propose the second cut. Do not starve viable parents.
    assert not promising_rejected_first_cut(before, dict(after, collision_pairs=0,
                                                       collision_energy=0.))
    assert not promising_rejected_first_cut(before, dict(after, collision_energy=float("nan")))


def test_second_seam_must_really_add_a_new_disconnected_component():
    original, first, second = _grid_states()
    validate_real_split_trial(original, first, 1)
    validate_real_split_trial(first, second, 2)
    with pytest.raises(RuntimeError, match="did not add"):
        validate_real_split_trial(second, second, 2)


def _run_stub(monkeypatch, *, enabled=True, second_max_depth=.15):
    original, first, second = _grid_states()
    second.fixture_quality["penetration_max"] = second_max_depth
    built=[]

    def fake_build(_target, params, **kwargs):
        forced=tuple(getattr(params, "collision_split_forced_lines", ()))
        built.append(len(forced))
        return (original, first, second)[len(forced)]

    def fake_rank(state, **kwargs):
        n=int(state.mesh_2d_initial.metrics["collision_split_count"])
        if n >= 2:
            return [], dict(status="no_more_candidates")
        return [dict(axis="col" if n == 0 else "row",
                     value=1., source_face_id=0,
                     boundary_distance_tiles=1.0, score=1.0)], dict(status="ok")

    monkeypatch.setattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", fake_build)
    monkeypatch.setattr(collision, "rank_collision_split_candidates", fake_rank)
    monkeypatch.setattr(collision, "collision_quality",
                        lambda state: dict(state.fixture_quality))
    monkeypatch.setattr(collision, "collision_component_audit",
                        lambda state: dict(status="ok"))

    params=SimpleNamespace(
        omega_parameterization_mode="",
        collision_split_enabled=True,
        collision_split_max_additional=2,
        collision_split_candidate_attempts=1,
        collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_gap_multiplier=1.,
        collision_split_forced_lines=(),
        collision_split_lookahead_enabled=enabled,
        collision_split_lookahead_children=1,
    )
    result=pipeline.build_onestring_design(object(), params)
    return original, first, second, result, built


def test_two_cuts_are_accepted_atomically_only_when_pair_passes(monkeypatch):
    original, first, second, result, built=_run_stub(monkeypatch)
    assert result is second
    assert built == [0, 1, 2]
    metrics=result.tiles_2d_dual_hinge.metrics
    assert metrics["collision_split_newly_accepted"] == 2
    assert len(metrics["collision_split_accepted_lines"]) == 2
    steps=metrics["collision_split_feedback_steps"]
    assert len(steps)==2 and all(item["accepted"] for item in steps)
    assert steps[0]["single_step_decision_reason"] == "max_penetration_worsened"
    assert steps[1]["acceptance_mode"] == "lookahead_second_cut"
    assert steps[1]["lookahead_baseline"]["collision_pairs"] == 9
    previews=result.collision_split_candidate_previews
    assert len(previews)==2 and all(p["accepted"] for p in previews)
    assert previews[1]["lookahead_baseline_metrics"]["collision_pairs"] == 9
    np.testing.assert_array_equal(result.mesh_2d_initial.faces,
                                  second.mesh_2d_initial.faces)
    assert not hasattr(original, "collision_split_candidate_previews")


def test_invalid_second_cut_rejects_entire_pair_and_preserves_original(monkeypatch):
    original, first, second, result, built=_run_stub(
        monkeypatch, second_max_depth=.22,
    )
    assert built == [0, 1, 2]
    assert result is original
    m=result.tiles_2d_dual_hinge.metrics
    assert m["collision_split_newly_accepted"] == 0
    assert m["collision_split_accepted_lines"] == []
    assert all(not row["accepted"] for row in m["collision_split_feedback_steps"])
    assert all(not p["accepted"] for p in result.collision_split_candidate_previews)
    assert m["collision_split_feedback_steps"][1]["decision_reason"] == (
        "max_penetration_worsened"
    )


def test_lookahead_disabled_reproduces_original_single_cut_behavior(monkeypatch):
    original, first, second, result, built=_run_stub(monkeypatch, enabled=False)
    assert result is original
    assert built == [0, 1]
    assert len(result.tiles_2d_dual_hinge.metrics["collision_split_feedback_steps"])==1


def test_collision_free_but_hinge_broken_candidate_does_not_starve_lookahead(monkeypatch):
    """Regression: skip a 0-contact parent and explore the next energy-improving
    but maximum-penetration-worsened parent (the observed real-world scenario).
    """
    import copy
    original, first, second = _grid_states()
    first_bad = copy.deepcopy(first)
    first_bad.fixture_quality.update(
        collision_pairs=0, collision_energy=0., penetration_max=0.,
        hinge_rms=.30,  # forbidden: baseline=.05, ceiling=.15
    )
    first_good = first
    second.fixture_quality["hinge_rms"] = .05
    built=[]

    def fake_build(_target, params, **_kwargs):
        forced=tuple(getattr(params, "collision_split_forced_lines", ()))
        built.append(tuple(str(c["name"]) for c in forced))
        if not forced:
            return original
        if len(forced)==1:
            return first_bad if forced[0]["name"]=="collision_zero" else first_good
        return second

    def candidate(name):
        return dict(name=name, axis="col", value=1., source_face_id=0,
                    boundary_distance_tiles=1., score=1.)

    def fake_rank(state, **_kwargs):
        if state is original:
            return ([candidate("collision_zero"), candidate("max_worse")],
                    dict(status="ok"))
        if state is first_good:
            return [candidate("second")], dict(status="ok")
        return [], dict(status="no_collisions")

    monkeypatch.setattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", fake_build)
    monkeypatch.setattr(collision, "rank_collision_split_candidates", fake_rank)
    monkeypatch.setattr(collision, "collision_quality",
                        lambda state: dict(state.fixture_quality))
    monkeypatch.setattr(collision, "collision_component_audit",
                        lambda _: dict(status="ok"))
    params=SimpleNamespace(
        omega_parameterization_mode="",
        collision_split_enabled=True, collision_split_max_additional=2,
        collision_split_candidate_attempts=2, collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_gap_multiplier=1., collision_split_forced_lines=(),
        collision_split_lookahead_enabled=True,
        collision_split_lookahead_children=1,
    )
    result=pipeline.build_onestring_design(object(),params)
    assert result is second
    assert built == [(), ("collision_zero",), ("max_worse",),
                     ("max_worse","second")]
    steps=result.tiles_2d_dual_hinge.metrics["collision_split_feedback_steps"]
    assert len(steps)==3
    assert steps[0]["decision_reason"]=="hinge_connection_regression"
    assert not steps[0]["accepted"]
    assert steps[1]["single_step_decision_reason"]=="max_penetration_worsened"
    assert steps[1]["accepted"] and steps[2]["accepted"]
    assert len(result.tiles_2d_dual_hinge.metrics["collision_split_accepted_lines"])==2


def test_observed_243_to_245_pair_growth_is_eligible_for_lookahead():
    """Observed regression: ~18% lower penetration energy, +2 pairs."""
    before=dict(status="ok", collision_pairs=243, collision_energy=.0064)
    after=dict(status="ok", collision_pairs=245, collision_energy=.0053,
               penetration_max=.0214, hinge_rms=.0094)
    assert promising_rejected_first_cut(before, after)
    # Bound temporary pair growth to the original 2% allowance (ceil(4.86)=5).
    assert promising_rejected_first_cut(
        before,dict(after, collision_pairs=248)
    )
    assert not promising_rejected_first_cut(
        before,dict(after, collision_pairs=249)
    )


def test_lookahead_picks_minor_pair_growth_parent_if_energy_is_lower(monkeypatch):
    """Two rejected first cuts: select lower-energy intermediate with +1 pair
    even though it misses the single-cut >=20% extra-pair improvement rule.
    """
    import copy
    original, parent, second=_grid_states()
    worse_max=copy.deepcopy(parent)
    worse_max.fixture_quality.update(collision_pairs=7,collision_energy=8.5,
                                      penetration_max=.21,hinge_rms=.05)
    extra_pair=parent
    extra_pair.fixture_quality.update(collision_pairs=10,collision_energy=7.7,
                                      penetration_max=.19,hinge_rms=.05)
    second.fixture_quality.update(hinge_rms=.05)
    built=[]

    def make(name):
        return dict(name=name,axis="col",value=1.,source_face_id=0,
                    boundary_distance_tiles=1.,score=1.)

    def fake_build(_target,params,**_kwargs):
        forced=tuple(getattr(params,"collision_split_forced_lines",()))
        built.append(tuple(c["name"] for c in forced))
        if not forced:
            return original
        if len(forced)==1:
            return worse_max if forced[0]["name"]=="worse_max" else extra_pair
        return second

    def fake_rank(state,**_kwargs):
        if state is original:
            return [make("worse_max"), make("extra_pair")],dict(status="ok")
        return [make("next_cut")],dict(status="ok")

    monkeypatch.setattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",fake_build)
    monkeypatch.setattr(collision,"rank_collision_split_candidates",fake_rank)
    monkeypatch.setattr(collision,"collision_quality",
                        lambda st:dict(st.fixture_quality))
    monkeypatch.setattr(collision,"collision_component_audit",
                        lambda st:dict(status="ok"))
    params=SimpleNamespace(
        omega_parameterization_mode="",
        collision_split_enabled=True,collision_split_max_additional=2,
        collision_split_candidate_attempts=2,
        collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_gap_multiplier=1.,
        collision_split_forced_lines=(),
        collision_split_lookahead_enabled=True,
        collision_split_lookahead_children=1,
    )
    result=pipeline.build_onestring_design(object(),params)
    assert result is second
    assert built==[(),("worse_max",),("extra_pair",),
                   ("extra_pair","next_cut")]
    steps=result.tiles_2d_dual_hinge.metrics["collision_split_feedback_steps"]
    assert steps[0]["decision_reason"]=="max_penetration_worsened"
    assert steps[1]["single_step_decision_reason"]=="insufficient_reduction_for_extra_pairs"
    assert steps[1]["acceptance_mode"]=="lookahead_first_cut"
    assert steps[2]["accepted"] is True
    assert len(result.tiles_2d_dual_hinge.metrics["collision_split_accepted_lines"])==2
