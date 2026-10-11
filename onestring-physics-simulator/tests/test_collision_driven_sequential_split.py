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
                penetration_max=.2,hinge_rms=.01,tile_scale=1.)
    after=dict(status="ok",collision_pairs=2,collision_energy=3.,
               penetration_max=.1,hinge_rms=.02)
    assert collision.accept_if_improved(before,after)[0]
    bad=dict(after,hinge_rms=.5)
    accepted,reason=collision.accept_if_improved(before,bad)
    assert not accepted and reason=="hinge_connection_regression"
    worsened=dict(after,collision_energy=11.)
    assert not collision.accept_if_improved(before,worsened)[0]


def _fake_state(pair_count, applied_cuts=0):
    # An intentionally tiny but REAL M2D source topology: a 4x4 grid with
    # either zero, one or two complete disconnected seams.
    from collections import defaultdict
    from onestring_physics.final_split_panel_pass import _complete_cut_once
    vertices,faces=_grid(4)
    seam_records=[]
    for k in range(applied_cuts):
        axis="col" if k==0 else "row"
        anchor=5 if k==0 else 10
        comp=next(comp for comp in _edge_components(faces) if anchor in comp)
        owners=defaultdict(list)
        for fi in comp:
            for corner in range(4):
                key=tuple(sorted((int(faces[fi,corner]),
                                  int(faces[fi,(corner+1)%4]))))
                owners[key].append((int(fi),corner))
        vertices2,subset,record=_complete_cut_once(vertices,faces[comp],axis,2.)
        assert record is not None
        updated=faces.copy();updated[comp]=subset
        pairs=[]
        for owners_for_edge in owners.values():
            if len(owners_for_edge)!=2:
                continue
            (a,ea),(b,eb)=owners_for_edge
            if set(updated[a,[ea,(ea+1)%4]])!=set(updated[b,[eb,(eb+1)%4]]):
                pairs.append([a,ea,b,eb])
        assert pairs
        seam_records.append(dict(
            complete_boundary_to_boundary=True,
            split_boundary_pairs=pairs,
            collision_source_face_id=anchor,
        ))
        vertices,faces=vertices2,updated
    return SimpleNamespace(
        conformal_domain=SimpleNamespace(paper_split_policy=True),
        tiles_2d_dual_hinge=SimpleNamespace(metrics={}),
        mesh_2d_initial=SimpleNamespace(faces=faces,vertices=vertices,
            metrics={
                "collision_split_count": applied_cuts,
                "collision_split_final_pass_applied_count": applied_cuts,
                "collision_split_step_analysis":seam_records,
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
        penetration_max=.1 if state.collision_pair_count else 0.,
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
        penetration_max=.1 if state.collision_pair_count else 0.,
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



def test_two_cuts_are_proposed_sequentially_with_fresh_collision_map(monkeypatch):
    states=[_fake_state(10,0),_fake_state(4,1),_fake_state(0,2)]
    forced_history=[]
    scored_states=[]

    def fake_full_build(target,params,**kwargs):
        forced=list(getattr(params,"collision_split_forced_lines",()))
        forced_history.append([dict(x) for x in forced])
        return states[len(forced)]

    def fake_rank(state,**kwargs):
        # Verify the candidate is re-ranked *after* the first cut.
        scored_states.append(state)
        # Compare identities, never SimpleNamespace values: these contain
        # NumPy arrays of different shapes after each topological cut, and
        # list.index() invokes equality (which raises on ndarray comparison).
        i=next(i for i, candidate_state in enumerate(states)
               if candidate_state is state)
        proposal=dict(
            axis="col" if i==0 else "row",
            value=2.0,source_face_id=5 if i==0 else 10,
            boundary_distance_tiles=1.0+i,score=3.0,
        )
        return [proposal],dict(status="candidates_ranked")

    monkeypatch.setattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",fake_full_build)
    monkeypatch.setattr(collision,"rank_collision_split_candidates",fake_rank)
    monkeypatch.setattr(collision,"collision_quality",lambda state: dict(
        status="ok",collision_pairs=state.collision_pair_count,
        collision_energy=float(state.collision_pair_count),
        penetration_max=.1 if state.collision_pair_count else 0.,
        hinge_rms=.01,tile_scale=1.,
    ))
    params=SimpleNamespace(
        collision_split_enabled=True,collision_split_max_additional=4,
        collision_split_candidate_attempts=1,
        collision_split_boundary_weight=2.,
        collision_split_min_relative_improvement=.03,
        collision_split_forced_lines=(),
        omega_parameterization_mode="",
    )
    result=pipeline.build_onestring_design(object(),params)
    assert result is states[2]
    assert len(scored_states)==2 and scored_states==states[:2]
    assert [len(v) for v in forced_history]==[0,1,2]
    assert forced_history[2][0]["axis"]=="col"
    assert forced_history[2][1]["axis"]=="row"
    metrics=result.tiles_2d_dual_hinge.metrics
    assert metrics["collision_split_status"]=="collision_free"
    assert metrics["collision_split_accepted"]==2
    assert all(x["accepted"] for x in metrics["collision_split_feedback_steps"])
    assert len(metrics["collision_split_feedback_steps"])==2


def test_rebuild_with_missing_seam_is_rejected(monkeypatch):
    baseline=_fake_state(9,0)
    # Energy improves, but the new M2D did not actually get cut!
    uncaught_fake=_fake_state(0,0)
    monkeypatch.setattr(
        pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",
        lambda target,params,**kwargs: baseline
        if not getattr(params,"collision_split_forced_lines",()) else uncaught_fake,
    )
    monkeypatch.setattr(collision,"collision_quality",lambda state: dict(
        status="ok",collision_pairs=state.collision_pair_count,
        collision_energy=float(state.collision_pair_count),
        penetration_max=.1 if state.collision_pair_count else 0.,
        hinge_rms=.01,tile_scale=1.,
    ))
    candidate=dict(axis="col",value=2.,source_face_id=5,
                   boundary_distance_tiles=1.5,score=2.)
    monkeypatch.setattr(collision,"rank_collision_split_candidates",
                        lambda *a,**kw: ([candidate],dict(status="candidates_ranked")))
    params=SimpleNamespace(
        collision_split_enabled=True,collision_split_max_additional=1,
        collision_split_candidate_attempts=1,
        collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_forced_lines=(),
        omega_parameterization_mode="",
    )
    result=pipeline.build_onestring_design(object(),params)
    assert result is baseline
    report=result.tiles_2d_dual_hinge.metrics
    assert report["collision_split_accepted"]==0
    assert "topology not propagated" in report["collision_split_feedback_steps"][0]["decision_reason"]



def test_final_pass_preserves_col_collision_cut_without_csf():
    from onestring_physics.final_split_panel_pass import (
        _split_axis_value, _complete_cut_once, install_final_split_panel_pass,
    )
    assert _split_axis_value(("col",2.)) == ("column",2.)
    vertices, faces = _grid(4)
    new_vertices,new_faces,record=_complete_cut_once(vertices,faces,"col",2.)
    assert record is not None
    class Mesh:
        def __init__(self,vertices,faces,grid,stage,metrics,split_lines):
            self.vertices=vertices
            self.faces=faces
            self.grid=grid
            self.stage=stage
            self.metrics=metrics
            self.split_lines=split_lines

    mesh=Mesh(
        new_vertices,new_faces,SimpleNamespace(tile_size=1.,gap_size=.1),
        "M2D",dict(
            csf_split_applied=False,
            csf_split_step_analysis=[],
            collision_split_applied=True,
            collision_split_count=1,
            collision_split_step_analysis=[
                {"collision_source_face_id":5,"axis":"col","snapped_value":2.}
            ],
            split_locations=[["col",2.]],
        ),[("col",2.)],
    )
    mesh._split_panel_source_vertices=new_vertices.copy()
    fake=SimpleNamespace(
        _build_m2d=lambda grid,domain,params=None:mesh,
        _original=SimpleNamespace(QuadMesh=Mesh),
    )
    install_final_split_panel_pass(fake)
    result=fake._build_m2d(mesh.grid,SimpleNamespace(),None)
    assert result.metrics["final_split_panel_pass_applied"] is True
    assert result.metrics["collision_split_final_pass_applied_count"]==1
    assert len(_edge_components(result.faces))==2
    assert len(result.metrics["final_split_panel_per_line"])==1
    assert result.metrics["final_split_panel_per_line"][0]["collision_driven"]


def test_final_pass_keeps_source_face_anchor_on_sequential_collisions():
    from onestring_physics.final_split_panel_pass import (
        _complete_cut_once, install_final_split_panel_pass,
    )
    vertices, faces = _grid(4)
    vertices, faces, row_record = _complete_cut_once(vertices,faces,"row",2.)
    # The second cut belongs only to the upper-right or upper-left component;
    # it cannot silently be applied to the wrong component.
    upper_anchor=10
    upper_component=next(c for c in _edge_components(faces) if upper_anchor in c)
    vertices_after,subset,col_record=_complete_cut_once(
        vertices,faces[upper_component],"col",2.,
    )
    assert col_record is not None
    faces_after=faces.copy();faces_after[upper_component]=subset
    class Mesh:
        def __init__(self,vertices,faces,grid,stage,metrics,split_lines):
            self.vertices=vertices
            self.faces=faces
            self.grid=grid
            self.stage=stage
            self.metrics=metrics
            self.split_lines=split_lines
    m=Mesh(vertices_after,faces_after,
           SimpleNamespace(tile_size=1.,gap_size=.1),
           "M2D",{
               "csf_split_applied":True,
               "csf_split_step_analysis":[{"axis":"row","snapped_value":2.}],
               "collision_split_applied":True,
               "collision_split_count":1,
               "collision_split_step_analysis":[
                   {"collision_source_face_id":upper_anchor}],
               "split_locations":[["row",2.],["col",2.]],
           },[("row",2.),("col",2.)])
    m._split_panel_source_vertices=vertices_after.copy()
    fake=SimpleNamespace(
        _build_m2d=lambda grid,domain,params=None:m,
        _original=SimpleNamespace(QuadMesh=Mesh),
    )
    install_final_split_panel_pass(fake)
    result=fake._build_m2d(m.grid,SimpleNamespace(),None)
    assert result.metrics["collision_split_final_pass_applied_count"]==1
    assert len(_edge_components(result.faces))==3
    final_collisions=[r for r in result.metrics["final_split_panel_per_line"]
                      if r.get("collision_driven",False)]
    assert len(final_collisions)==1
    assert final_collisions[0]["collision_source_face_id"]==upper_anchor


def test_split_accepts_deeper_contact_reduction_with_one_extra_pair():
    # Regression from actual UI: 168 -> 169 contacts, but 0.0036 -> 0.0021
    # summed squared depth (~42% lower). The old condition rejected this
    # beneficial Split solely due to a one-pair count increase.
    before=dict(status="ok",collision_pairs=168,collision_energy=.0036,
                penetration_max=.0162,hinge_rms=.0084,tile_scale=1.)
    after=dict(status="ok",collision_pairs=169,collision_energy=.0021,
               penetration_max=.0130,hinge_rms=.0085,tile_scale=1.)
    accepted,reason=collision.accept_if_improved(before,after,
                                                 minimum_relative_improvement=.03)
    assert accepted
    assert reason=="penetration_severity_reduced_with_minor_pair_increase"


def test_split_rejects_many_extra_pairs_even_when_summed_severity_falls():
    before=dict(status="ok",collision_pairs=168,collision_energy=.0036,
                penetration_max=.0162,hinge_rms=.0084,tile_scale=1.)
    after=dict(status="ok",collision_pairs=173,collision_energy=.0026,
               penetration_max=.012,hinge_rms=.0085,tile_scale=1.)
    accepted,reason=collision.accept_if_improved(before,after)
    assert not accepted
    assert reason=="too_many_new_collision_pairs"


def test_split_rejects_worse_maximum_penetration_despite_lower_total_energy():
    before=dict(status="ok",collision_pairs=168,collision_energy=.0036,
                penetration_max=.0162,hinge_rms=.0084,tile_scale=1.)
    after=dict(status="ok",collision_pairs=169,collision_energy=.0021,
               penetration_max=.019,hinge_rms=.0085,tile_scale=1.)
    accepted,reason=collision.accept_if_improved(before,after)
    assert not accepted
    assert reason=="max_penetration_worsened"


def test_split_rejects_minor_total_energy_improvement_if_pair_count_rises():
    before=dict(status="ok",collision_pairs=168,collision_energy=.0036,
                penetration_max=.0162,hinge_rms=.0084,tile_scale=1.)
    after=dict(status="ok",collision_pairs=169,collision_energy=.0033,
               penetration_max=.015,hinge_rms=.0085,tile_scale=1.)
    assert collision.accept_if_improved(before,after)[1]=="insufficient_reduction_for_extra_pairs"


def test_split_requires_depth_metrics_for_acceptance():
    before=dict(status="ok",collision_pairs=168,collision_energy=.0036,
                hinge_rms=.0084,tile_scale=1.)
    after=dict(status="ok",collision_pairs=169,collision_energy=.0021,
               hinge_rms=.0085,tile_scale=1.)
    assert collision.accept_if_improved(before,after)==(
        False,"missing_collision_or_hinge_diagnostics"
    )
