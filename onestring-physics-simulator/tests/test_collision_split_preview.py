"""Draw rejected Split trials without accidentally accepting them or exporting STLs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.paper_collision_split_preview import (
    capture_split_trial_geometry, collision_tile_severity,
    hinge_gap_segments, rejection_comparison,
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
    from tests.test_collision_driven_sequential_split import _fake_state
    original=_fake_state(9,0)
    rejected=_fake_state(12,1)
    for state in (original,rejected):
        extra=_pair()
        state.mesh_2d_initial.vertices=np.asarray(
            extra.mesh_2d_initial.vertices).copy()
        # Preserve actual synthetic detached source topology and its vertex
        # count: the preview must use the SAME faces as the completed rebuild.
        if np.max(state.mesh_2d_initial.faces)>=len(state.mesh_2d_initial.vertices):
            # This test models a bigger regular grid instead of only two quads.
            faces=state.mesh_2d_initial.faces
            vertices=np.zeros((int(faces.max())+1,3),float)
            for row in range(len(vertices)):
                vertices[row,:2]=(row%6,row//6)
            state.mesh_2d_initial.vertices=vertices
        state.mesh_2d_optimized=SimpleNamespace(
            vertices=state.mesh_2d_initial.vertices.copy(),
            faces=state.mesh_2d_initial.faces.copy(),
        )
        n=len(state.mesh_2d_initial.faces)
        tiles=np.zeros((n,8,3),float)
        for i,face in enumerate(state.mesh_2d_initial.faces):
            tiles[i,:4]=state.mesh_2d_initial.vertices[face]
            tiles[i,4:]=tiles[i,:4]
            tiles[i,4:,2]=-.1
        state.tiles_2d_top_hinge=SimpleNamespace(vertices=tiles.copy())
        state.tiles_2d_dual_hinge.vertices=tiles.copy()

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
