"""Draw rejected Split trials without accidentally accepting them or exporting STLs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.paper_collision_split_preview import (
    capture_split_trial_geometry, collision_tile_severity,
    hinge_gap_segments, rejection_comparison, acceptance_checklist,
)
from onestring_physics.paper_collision_split_preview_view import (
    _preview_figure, _severity_colors,
)
from onestring_physics import paper_collision_split as collision
from onestring_physics import onestring_pipeline as pipeline


def _pair(overlap=True):
    square=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    sep=.55 if overlap else 2.
    top=np.stack((square,square+[sep,0.]))
    dual=np.concatenate((
        np.concatenate((top,np.zeros((2,4,1))),axis=-1),
        np.concatenate((top,np.full((2,4,1),-.12)),axis=-1),
    ),axis=1)
    faces=np.arange(8,dtype=int).reshape(2,4)
    xyz=np.concatenate((
        np.column_stack((top[0],np.zeros(4))),
        np.column_stack((top[1],np.zeros(4))),
    ),axis=0)
    hinges=[SimpleNamespace(
        tile_a=0,local_vertex_a=1,tile_b=1,local_vertex_b=0,
    )]
    return SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(vertices=xyz.copy(),faces=faces.copy(),
                                         metrics={}),
        mesh_2d_optimized=SimpleNamespace(vertices=xyz.copy(),faces=faces.copy()),
        tiles_2d_top_hinge=SimpleNamespace(vertices=dual.copy()),
        tiles_2d_dual_hinge=SimpleNamespace(vertices=dual.copy(),metrics={}),
        hinge_graph=SimpleNamespace(hinges=hinges),
    )


def test_snapshot_keeps_all_four_geometry_stages_but_no_full_state_or_stl():
    state=_pair()
    snap=capture_split_trial_geometry(state)
    assert snap is not None
    assert set(snap["stages"])=={"M2D","K2D","T2D Top","T2D Dual"}
    assert snap["tile_count"]==2
    assert snap["hinges"].shape==(1,4)
    assert snap["valid_for_fabrication_export"] is False
    assert "optimizer_history" not in snap and "string_path" not in snap
    actual=state.tiles_2d_dual_hinge.vertices[0,0,0]
    state.tiles_2d_dual_hinge.vertices[0,0,0]=999.
    assert snap["stages"]["T2D Dual"]["vertices"][0,0,0]==actual
    assert snap["stages"]["T2D Dual"]["vertices"].dtype==np.float64


def test_missing_partial_geometry_returns_none_without_mutation():
    state=SimpleNamespace(
        tiles_2d_dual_hinge=SimpleNamespace(metrics={}),
    )
    assert capture_split_trial_geometry(state) is None


def test_collision_heatmap_uses_actual_sat_and_shared_scale():
    before=capture_split_trial_geometry(_pair(overlap=True))
    after=capture_split_trial_geometry(_pair(overlap=False))
    s0=collision_tile_severity(before["stages"]["T2D Dual"]["vertices"])
    s1=collision_tile_severity(after["stages"]["T2D Dual"]["vertices"])
    assert s0["pair_count"]==1
    assert s0["max_depth"]==pytest.approx(.45)
    assert s0["energy"]==pytest.approx(.45**2)
    assert np.count_nonzero(s0["strength"])==2
    assert s1["pair_count"]==0
    assert np.max(s1["strength"])==0
    colors0=_severity_colors(s0["strength"],float(max(s0["strength"])))
    colors1=_severity_colors(s1["strength"],float(max(s0["strength"])))
    assert len(colors0)==len(colors1)==2
    assert all(color!="#92c8c2" for color in colors0)
    assert all(color=="#92c8c2" for color in colors1)


def test_preview_figures_use_same_final_trial_geometry_not_reoptimized():
    after=capture_split_trial_geometry(_pair())
    source=after["stages"]["T2D Dual"]["vertices"].copy()
    fig,diagnostics=_preview_figure(
        after,"T2D Dual",title="rejected",severity_max=.5,
        include_hinges=True,red_collision_overlay=False,
    )
    assert len(fig.data)>=2
    assert diagnostics["pair_count"]==1
    np.testing.assert_array_equal(after["stages"]["T2D Dual"]["vertices"],source)
    assert any("hinge" in (str(trace.name)).lower() for trace in fig.data)


def test_hinge_gap_segments_are_true_endpoints():
    trial=capture_split_trial_geometry(_pair())
    segment=hinge_gap_segments(
        trial["stages"]["T2D Dual"]["vertices"],trial["hinges"],
    )
    assert len(segment["gap"])==1
    assert segment["gap"][0]==pytest.approx(.45)
    np.testing.assert_allclose(segment["start"][0],[1.,0.,0.])
    np.testing.assert_allclose(segment["end"][0],[.55,0.,0.])


def test_rejection_details_reproduce_reason_thresholds():
    before=dict(collision_pairs=272,collision_energy=.0057,
                penetration_max=.016,hinge_rms=.0084,tile_scale=1.)
    after=dict(collision_pairs=241,collision_energy=.0038,
               penetration_max=.017,hinge_rms=.009)
    facts=rejection_comparison(before,after)
    assert facts["collision_delta"]==-31
    assert facts["penetration_energy_improvement_percent"]==pytest.approx(100/3)
    assert facts["max_penetration_delta"]==pytest.approx(.001)
    assert facts["extra_pairs_allowance"]==6
    assert facts["hinge_rms_rejection_threshold"]==pytest.approx(.1)


def test_rejected_full_rebuild_keeps_viewable_trial_but_not_authoritative_state(monkeypatch):
    # Two independent synthetic completed full-pipeline states with a genuine
    # disconnected seam in the rejected one. Avoid importing another test file
    # or building actual heavy K3D/OptCuts stages in the preview regression.
    from collections import defaultdict
    from onestring_physics.final_split_panel_pass import _complete_cut_once

    side=4
    verts=np.array([[float(i),float(j),0.]
                    for j in range(side+1) for i in range(side+1)],float)
    faces=np.array([
        [j*(side+1)+i,j*(side+1)+i+1,
         (j+1)*(side+1)+i+1,(j+1)*(side+1)+i]
        for j in range(side) for i in range(side)
    ],dtype=np.int32)
    v_cut, f_cut, cut_record=_complete_cut_once(verts,faces,"col",2.)
    assert cut_record is not None
    edge_owners=defaultdict(list)
    for fi,face in enumerate(faces):
        for k in range(4):
            edge=tuple(sorted((int(face[k]),int(face[(k+1)%4]))))
            edge_owners[edge].append((fi,k))
    broken=[]
    for adjacent in edge_owners.values():
        if len(adjacent)!=2:
            continue
        (a,ea),(b,eb)=adjacent
        if set(f_cut[a,[ea,(ea+1)%4]])!=set(f_cut[b,[eb,(eb+1)%4]]):
            broken.append([a,ea,b,eb])
    assert broken

    def fake_state(vertex_array,face_array,collision_count,cut_count):
        quads=np.asarray(vertex_array,dtype=float)[face_array]
        panels=np.concatenate((
            quads,
            quads+np.array([0.,0.,-.1]),
        ),axis=1)
        records=([dict(
            collision_source_face_id=5,
            complete_boundary_to_boundary=True,
            split_boundary_pairs=broken,
        )] if cut_count else [])
        return SimpleNamespace(
            conformal_domain=SimpleNamespace(paper_split_policy=True),
            mesh_2d_initial=SimpleNamespace(
                vertices=vertex_array.copy(), faces=face_array.copy(),
                metrics=dict(
                    collision_split_count=cut_count,
                    collision_split_final_pass_applied_count=cut_count,
                    collision_split_step_analysis=records,
                ),
            ),
            mesh_2d_optimized=SimpleNamespace(
                vertices=vertex_array.copy(),faces=face_array.copy(),
            ),
            tiles_2d_top_hinge=SimpleNamespace(vertices=panels.copy()),
            tiles_2d_dual_hinge=SimpleNamespace(vertices=panels.copy(),metrics={}),
            hinge_graph=SimpleNamespace(hinges=[]),
            collision_pair_count=collision_count,
        )
    original=fake_state(verts,faces,9,0)
    rejected=fake_state(v_cut,f_cut,12,1)

    def fake_build(_target,params,**kwargs):
        forced=getattr(params,"collision_split_forced_lines",())
        return rejected if forced else original
    monkeypatch.setattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",fake_build)
    monkeypatch.setattr(collision,"rank_collision_split_candidates",
                        lambda *_args,**_kwargs: (
                            [dict(axis="col",value=2.,source_face_id=5,
                                  boundary_distance_tiles=1.5,score=3.)],
                            dict(status="candidates_ranked")))
    monkeypatch.setattr(collision,"collision_quality",lambda state: dict(
        status="ok",collision_pairs=state.collision_pair_count,
        collision_energy=float(state.collision_pair_count),
        penetration_max=.1,hinge_rms=.01,tile_scale=1.,
    ))
    monkeypatch.setattr(collision,"collision_component_audit",
                        lambda *_:dict(status="ok"))
    params=SimpleNamespace(
        collision_split_enabled=True,collision_split_max_additional=2,
        collision_split_candidate_attempts=1,collision_split_boundary_weight=1.,
        collision_split_min_relative_improvement=.03,
        collision_split_forced_lines=(),omega_parameterization_mode="",
    )
    result=pipeline.build_onestring_design(object(),params)
    assert result is original
    assert result.tiles_2d_dual_hinge.metrics["collision_split_accepted"]==0
    rows=result.tiles_2d_dual_hinge.metrics["collision_split_feedback_steps"]
    assert len(rows)==1 and rows[0]["accepted"] is False
    assert rows[0]["preview_index"]==0
    candidates=result.collision_split_candidate_previews
    assert len(candidates)==1
    assert candidates[0]["after"]["stages"]["T2D Dual"]["vertices"].shape[0]==16
    assert not candidates[0]["accepted"]
    assert "collision_split_candidate_previews" not in result.tiles_2d_dual_hinge.metrics
    # The authoritative topology is unchanged after a rejected trial.
    np.testing.assert_array_equal(result.mesh_2d_initial.faces,faces)
    assert candidates[0]["after"]["stages"]["M2D"]["faces"].max()>faces.max()



def test_rejection_threshold_checklist_exposes_exact_reason():
    before=dict(collision_pairs=272,collision_energy=.0057,
                penetration_max=.016,hinge_rms=.0084,tile_scale=1.)
    after=dict(collision_pairs=241,collision_energy=.0038,
               penetration_max=.017,hinge_rms=.009)
    checks=acceptance_checklist(before,after,min_relative_improvement=.03)
    assert len(checks)==3
    by_name={row["criterion"]:row for row in checks}
    assert by_name["最大SAT貫通深さの非悪化"]["ok"] is False
    assert by_name["貫通エネルギー改善率"]["ok"] is True
    assert by_name["残存ヒンジRMS上限"]["ok"] is True


def test_collision_pair_increase_criteria_only_when_needed():
    before=dict(collision_pairs=168,collision_energy=.0036,
                penetration_max=.018,hinge_rms=.0084,tile_scale=1.)
    after=dict(collision_pairs=169,collision_energy=.0021,
               penetration_max=.012,hinge_rms=.009)
    checks=acceptance_checklist(before,after)
    assert len(checks)==5
    assert all(row["ok"] for row in checks)
    after_more=dict(after,collision_pairs=180)
    more=acceptance_checklist(before,after_more)
    assert any(row["criterion"]=="衝突ペア増加数の上限" and not row["ok"]
               for row in more)



def test_top_hinge_preview_uses_top_corners_even_for_bottom_selected_dual_joints():
    state=_pair()
    state.hinge_graph.hinges[0].local_vertex_a=5
    state.hinge_graph.hinges[0].local_vertex_b=4
    snap=capture_split_trial_geometry(state)
    np.testing.assert_array_equal(snap["hinges"],[[0,5,1,4]])
    np.testing.assert_array_equal(snap["top_hinges"],[[0,1,1,0]])
    fig,_=_preview_figure(
        snap,"T2D Top",title="top",severity_max=.5,
        include_hinges=True,red_collision_overlay=False,
    )
    assert any("hinge" in str(trace.name).lower() for trace in fig.data)
