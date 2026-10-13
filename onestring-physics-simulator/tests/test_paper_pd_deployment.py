"""Paper Section 5.4 / Supplement E actuation contract regression.

Tests paper Snap/Lift geometric *objectives*, not equivalence to unpublished
PD global matrix solver, exact channel geometry or IPC nonpenetration.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from onestring_physics.onestring_pipeline import DeploymentParameters
from onestring_physics.paper_pd_deployment import (
    PaperPDInputError,_validate_route,_opposite_side_pairs,
    _rigid_fit, simulate_paper_pd,
)


def _tile(x,y,z=0.):
    top=np.array([[x,y,z],[x+1,y,z],[x+1,y+1,z],[x,y+1,z]],float)
    return np.vstack((top,top+[0,0,-.10]))


def _state(extra_four=False):
    rest=np.stack([_tile(0.,0.),_tile(1.6,0.)] if not extra_four else
                  [_tile(0.,0.),_tile(2.,0.),_tile(0.,2.),_tile(2.,2.)])
    target=rest.copy()
    target[:,:,2]+=.5
    tiles=list(range(len(rest)))
    def gap(gid,owner,boundary,xy):
        return SimpleNamespace(id=gid,surrounding_tiles=owner,boundary=boundary,
                               centroid_2d=np.array([*xy,0.]),
                               centroid_3d=np.array([*xy,.5]))
    gaps=[
        gap(0,[0],True,(-.1,.5)),
        gap(1,tiles,False,(1.3,.5) if not extra_four else (1.5,1.5)),
        gap(2,[len(rest)-1],True,(3.,.5)),
        gap(3,[0],True,(.5,-.1)),
    ]
    path=SimpleNamespace(gap_ids=[0,1,2,3,0],
                         metrics={
                             "string_path_valid":True,
                             "string_path_status":"optimal",
                             "string_path_model":"Sec. 5.3 minimum-turn closed-cycle MILP",
                         })
    return SimpleNamespace(
        tiles_2d_dual_hinge=SimpleNamespace(vertices=rest),
        tiles_3d=SimpleNamespace(vertices=target),
        gap_graph=SimpleNamespace(gaps=gaps,metrics={
            "paper_section_5_3_graph":True,"split_boundary_missing_pairs":[],
        }),
        string_path=path,
        hinge_graph=SimpleNamespace(hinges=[]),
        lift_points=[SimpleNamespace(
            gap_id=1,position_3d=np.array([1.3,.5,.75]),
        )],
    )


def _params(**kw):
    p=dict(physics_backend="paper_pd",steps=14,solver_iterations=7,
           max_animation_frames=7,paper_pd_timestep=.012,
           gravity=0.,collision_weight=.4,snap_weight=.8,
           lift_weight=.9,hinge_weight=.95,damping_ratio=.35)
    p.update(kw)
    return DeploymentParameters(**p)


def test_valid_closed_milp_route_selects_physical_side_face_snap_and_lift():
    state=_state()
    snaps,lifts,info=_validate_route(state)
    assert len(snaps)==1
    assert snaps[0].a==0 and snaps[0].b==1
    assert len(snaps[0].face_a)==4 and len(snaps[0].face_b)==4
    assert len(lifts)==1
    assert info["route_closed"] is True


def test_quad_gap_creates_opposite_two_face_pairs_instead_of_skipping():
    state=_state(extra_four=True)
    snaps,lifts,_=_validate_route(state)
    assert len(snaps)==2
    assert len({x.a for x in snaps}|{x.b for x in snaps})==4
    assert len(lifts)==1


def test_invalid_routing_fails_without_creating_fictitious_actuation():
    state=_state()
    state.string_path.metrics["string_path_valid"]=False
    with pytest.raises(PaperPDInputError,match="INVALID_ROUTE"):
        simulate_paper_pd(state,_params())
    state=_state()
    state.string_path.gap_ids=[0,1,2,3]
    with pytest.raises(PaperPDInputError,match="OPEN_OR_REPEATED"):
        simulate_paper_pd(state,_params())
    state=_state()
    state.string_path.metrics["string_path_model"]="boundary_loop_greedy_fallback"
    with pytest.raises(PaperPDInputError,match="NOT_MILP"):
        simulate_paper_pd(state,_params())


def test_canonical_paper_pd_starts_with_actual_t2d_and_uses_only_snap_lift():
    state=_state()
    result=simulate_paper_pd(state,_params())
    assert len(result.frames)>=3
    np.testing.assert_array_equal(result.frames[0],state.tiles_2d_dual_hinge.vertices)
    assert np.isfinite(np.asarray(result.frames)).all()
    assert result.metrics["physics_backend"]=="paper_pd"
    assert result.metrics["t3d_target_tile_pose_actuator"] is False
    assert result.metrics["target_fit_weight"]==0
    assert result.metrics["target_contact_guard_weight"]==0
    assert result.metrics["actuated_snap_gap_count"]==1
    assert result.metrics["actuated_lift_gap_count"]==1
    assert len(result.collision_counts)==len(result.frames)
    assert len(result.snap_error_history)==len(result.frames)
    assert len(result.lift_error_history)==len(result.frames)
    # Rigid projection preserves original T2D panel dimensions.
    old=np.linalg.norm(state.tiles_2d_dual_hinge.vertices[:,1]
                       -state.tiles_2d_dual_hinge.vertices[:,0],axis=1)
    for frame in result.frames:
        np.testing.assert_allclose(
            np.linalg.norm(frame[:,1]-frame[:,0],axis=1),old,atol=1e-8
        )


def test_unselected_t3d_panel_poses_do_not_actuate_paper_pd():
    a=_state()
    b=_state()
    b.tiles_3d.vertices+=np.array([0.,0.,20.])
    pa=simulate_paper_pd(a,_params(steps=8,solver_iterations=4))
    pb=simulate_paper_pd(b,_params(steps=8,solver_iterations=4))
    np.testing.assert_allclose(pa.final_tiles,pb.final_tiles,atol=1e-9)
    assert pb.metrics["final_deployment_error_to_T3D"] > pa.metrics["final_deployment_error_to_T3D"]


def test_rigid_batch_kabsch_guards_against_reflections():
    state=_state()
    rest=state.tiles_2d_dual_hinge.vertices
    target=rest.copy()
    target[:,:,0]*=-1
    p,_,rot=_rigid_fit(rest,target)
    np.testing.assert_allclose(np.linalg.det(rot),np.ones(len(rest)),atol=1e-9)
    assert np.isfinite(p).all()
