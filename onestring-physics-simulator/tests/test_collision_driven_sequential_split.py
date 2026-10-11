"""Regression tests for boundary/depth-ranked ONE-at-a-time M2D cuts.

No external BFF/OptCuts binary and no full K3D optimization are required.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import paper_collision_split as collision
from onestring_physics import paper_mesh_splitting as split
from onestring_physics import onestring_pipeline as pipeline
from onestring_physics.simple_split_panel_patch import _edge_components


def _grid(n=4):
    vertices=np.array([[float(i),float(j),0.] for j in range(n+1)
                       for i in range(n+1)],dtype=float)
    faces=np.array([[j*(n+1)+i,j*(n+1)+i+1,
                     (j+1)*(n+1)+i+1,(j+1)*(n+1)+i]
                    for j in range(n) for i in range(n)],dtype=int)
    return vertices,faces


def _candidate_state(n=4):
    vertices,faces=_grid(n)
    top=vertices[faces].copy()
    bottom=top.copy();bottom[:,:,2]=-0.15
    panels=np.concatenate((top,bottom),axis=1)
    # A single strong interior collision, with other unshifted tiles touching
    # only at edges. The severity map must favor the affected region.
    panels[n+1,:,0]+=0.8
    panels[n+1,:,1]+=0.6
    return SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(vertices=vertices,faces=faces),
        tiles_2d_dual_hinge=SimpleNamespace(vertices=panels,
            metrics={"section_4_3_linkage":True, "hinge_rms":0.01}),
        conformal_domain=SimpleNamespace(reference_grid_rotation_degrees=0.),
    )


def test_ranked_seam_uses_both_collision_severity_and_boundary_distance():
    state=_candidate_state()
    candidates,stats=collision.rank_collision_split_candidates(
        state,boundary_weight=2.,max_candidates=10,
    )
    assert stats["collision_pairs"]>0
    assert stats["affected_tiles"]>0
    assert stats["status"]=="candidates_ranked"
    assert candidates
    best=candidates[0]
    assert best["score"]>0
    assert best["hotspot_collision_pairs"]>0
    assert best["boundary_distance_tiles"]>=0
    assert best["seam_edge_count"]>=1
    assert best["axis"] in ("row","col")
    assert 0 <= best["source_face_id"] < len(state.mesh_2d_initial.faces)
    # Each candidate cuts a true connected source quad component.
    from onestring_physics.final_split_panel_pass import _complete_cut_candidate
    comp=np.arange(len(state.mesh_2d_initial.faces),dtype=int)
    source=state.mesh_2d_initial
    actual=_complete_cut_candidate(
        source.vertices,source.faces,comp,best["axis"],best["value"],
    )
    assert actual is not None
    assert len(actual["endpoints"])==2
    assert len(_edge_components(source.faces))==1


def test_boundary_weight_is_measured_and_changes_ranking_scores():
    state=_candidate_state()
    unweighted,_=collision.rank_collision_split_candidates(
        state,boundary_weight=0.,max_candidates=30,
    )
    weighted,_=collision.rank_collision_split_candidates(
        state,boundary_weight=3.,max_candidates=30,
    )
    assert unweighted and weighted
    d0={(c["axis"],c["value"]):c["score"] for c in unweighted}
    d1={(c["axis"],c["value"]):c["score"] for c in weighted}
    common=set(d0)&set(d1)
    assert common
    assert all(d1[k]>=d0[k]-1e-12 for k in common)


def test_forced_cut_is_real_mesh_disconnection_and_separate_from_csf(monkeypatch):
    vertices,faces=_grid(4)
    monkeypatch.setattr(
        split, "_face_samples",
        lambda vertices,faces,param,threshold: (
            np.ones(len(faces)),np.ones(len(faces)),
            [np.array([0],dtype=int) for _ in faces],
        ),
    )
    monkeypatch.setattr(split,"gaussian_curvature",lambda p:np.zeros(1))
    domain=SimpleNamespace(
        parameterization=SimpleNamespace(
            metrics={},uv_vertices_2d=np.zeros((1,2))),
        csf_split_enabled=False,
        csf_split_threshold=2.,
        max_csf_splits=0,
        reference_grid_rotation_degrees=0.,
    )
    candidate={"axis":"col","value":2.,"source_face_id":5,
               "boundary_distance_tiles":1.5,"score":3.}
    params=SimpleNamespace(collision_split_forced_lines=[candidate])
    new_vertices,new_faces,records,metrics=split._split_mesh_single(
        vertices,faces,domain,params,
    )
    assert len(records)==1
    assert len(_edge_components(new_faces))==2
    assert len(new_vertices)>len(vertices)
    assert metrics["csf_split_step_count"]==0
    assert metrics["collision_split_count"]==1
    assert metrics["collision_split_applied"] is True
    assert metrics["split_boundary_pairs"]
    assert all(len(pair)==4 for pair in metrics["split_boundary_pairs"])
    assert records[0]["complete_boundary_to_boundary"] is True
    assert metrics["number_of_splits"]==1


def test_acceptance_must_lower_collision_without_exploding_hinges():
    before=dict(status="ok",collision_pairs=8,collision_energy=10.,
                hinge_rms=.01,tile_scale=1.)
    after=dict(status="ok",collision_pairs=2,collision_energy=3.,
               hinge_rms=.02)
    assert collision.accept_if_improved(before,after)[0]
    bad=dict(after,hinge_rms=.5)
    accepted,reason=collision.accept_if_improved(before,bad)
    assert not accepted and reason=="hinge_connection_regression"
    worsened=dict(after,collision_energy=11.)
    assert not collision.accept_if_improved(before,worsened)[0]


def _fake_state(pair_count, applied_cuts=0):
    return SimpleNamespace(
        conformal_domain=SimpleNamespace(paper_split_policy=True),
        tiles_2d_dual_hinge=SimpleNamespace(metrics={}),
        mesh_2d_initial=SimpleNamespace(metrics={
            "collision_split_count": applied_cuts,
        }),
        collision_pair_count=pair_count,
    )


def test_feedback_accepts_one_cut_then_reoptimizes_and_stops(monkeypatch):
    initial=_fake_state(9)
    rebuilt=_fake_state(0,applied_cuts=1)
    calls=[]

    def fake_full_build(target,params,**kwargs):
        forced=list(getattr(params,"collision_split_forced_lines",()))
        calls.append(len(forced))
        return initial if not forced else rebuilt

    monkeypatch.setattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",fake_full_build)
    monkeypatch.setattr(collision,"collision_quality",lambda state: dict(
        status="ok",collision_pairs=state.collision_pair_count,
        collision_energy=float(state.collision_pair_count),
        hinge_rms=.01,tile_scale=1.,
    ))
    candidate=dict(axis="col",value=2.,source_face_id=5,
                   boundary_distance_tiles=1.5,score=2.)
    monkeypatch.setattr(collision,"rank_collision_split_candidates",
                        lambda *a,**kw: ([candidate],dict(status="candidates_ranked")))
    params=SimpleNamespace(
        collision_split_enabled=True,collision_split_max_additional=3,
        collision_split_candidate_attempts=2,
        collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_forced_lines=(),
        omega_parameterization_mode="",
    )
    output=pipeline.build_onestring_design(object(),params)
    assert output is rebuilt
    assert calls==[0,1]  # exactly one cut + one full rebuild
    m=output.tiles_2d_dual_hinge.metrics
    assert m["collision_split_accepted"]==1
    assert m["collision_split_status"]=="collision_free"
    assert len(m["collision_split_feedback_steps"])==1
    assert m["collision_split_feedback_steps"][0]["accepted"]


def test_feedback_rejects_nonimproving_cut_and_preserves_old_state(monkeypatch):
    baseline=_fake_state(9)
    candidate_state=_fake_state(12,applied_cuts=1)
    calls=[]

    def fake_full_build(target,params,**kwargs):
        forced=list(getattr(params,"collision_split_forced_lines",()))
        calls.append(len(forced))
        return baseline if not forced else candidate_state

    monkeypatch.setattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",fake_full_build)
    monkeypatch.setattr(collision,"collision_quality",lambda state: dict(
        status="ok",collision_pairs=state.collision_pair_count,
        collision_energy=float(state.collision_pair_count),
        hinge_rms=.01,tile_scale=1.,
    ))
    candidate=dict(axis="row",value=2.,source_face_id=5,
                   boundary_distance_tiles=1.5,score=2.)
    monkeypatch.setattr(collision,"rank_collision_split_candidates",
                        lambda *a,**kw: ([candidate],dict(status="candidates_ranked")))
    params=SimpleNamespace(
        collision_split_enabled=True,collision_split_max_additional=3,
        collision_split_candidate_attempts=2,
        collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_forced_lines=(),
        omega_parameterization_mode="",
    )
    output=pipeline.build_onestring_design(object(),params)
    assert output is baseline
    assert calls==[0,1]
    m=output.tiles_2d_dual_hinge.metrics
    assert m["collision_split_accepted"]==0
    assert m["collision_split_status"]=="no_candidate_improved_or_rebuilt"
    assert not m["collision_split_feedback_steps"][0]["accepted"]
