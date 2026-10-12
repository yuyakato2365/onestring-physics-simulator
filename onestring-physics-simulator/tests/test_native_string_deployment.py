"""Regression checks for approximate T2D rigid-panel/string physics.

These test the CONSTRAINT CONTRACT, not equivalence to the OneString paper's
rigid-body simulator or experimental fabrication.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.native_string_deployment import (
    NativeDeploymentError,
    _prepare,
    _string_constraint,
    _string_geometry,
    _contacts,
    simulate_native_string_deployment,
)
from onestring_physics.onestring_pipeline import DeploymentParameters


def _tile(x, y, z=0.):
    top=np.array([[x,y,z],[x+1,y,z],[x+1,y+1,z],[x,y+1,z]],float)
    return np.vstack((top, top+[0.,0.,-.1]))


def _two_panel_state():
    a=_tile(0.,0.)
    b=_tile(1.,1.)
    avertices=np.stack((a,b))
    hinge=SimpleNamespace(tile_a=0,tile_b=1,
                          local_vertex_a=2,local_vertex_b=0,
                          surface="top")
    gaps=[
        SimpleNamespace(id=0,surrounding_tiles=[0],
                        centroid_2d=np.array([0.,0.])),
        SimpleNamespace(id=1,surrounding_tiles=[1],
                        centroid_2d=np.array([2.,2.])),
    ]
    return SimpleNamespace(
        tiles_2d_dual_hinge=SimpleNamespace(vertices=avertices),
        tiles_3d=SimpleNamespace(vertices=avertices.copy()),
        hinge_graph=SimpleNamespace(hinges=[hinge]),
        gap_graph=SimpleNamespace(gaps=gaps),
        string_path=SimpleNamespace(gap_ids=[0,1],metrics={}),
    )


def test_native_frame_zero_is_exact_t2d_and_panels_stay_rigid():
    state=_two_panel_state()
    params=DeploymentParameters(
        physics_backend="native",steps=12,solver_iterations=12,
        solver_substeps=2,hinge_weight=0.95,collision_weight=0.7,
        native_pull_end_ratio=0.75,native_anchor_height_tiles=2.0,
        damping_ratio=0.45,gravity=9.81,max_animation_frames=12,
        store_animation_frames=True,
    )
    result=simulate_native_string_deployment(state,params)
    assert len(result.frames)>=3
    np.testing.assert_array_equal(result.frames[0],state.tiles_2d_dual_hinge.vertices)
    assert np.isfinite(np.asarray(result.frames)).all()
    initial_lengths=np.linalg.norm(
        np.diff(state.tiles_2d_dual_hinge.vertices,axis=1),axis=-1,
    )
    for frame in result.frames:
        lengths=np.linalg.norm(np.diff(frame,axis=1),axis=-1)
        np.testing.assert_allclose(lengths,initial_lengths,atol=1e-8)
    assert result.metrics["t3d_target_used_as_actuator"] is False
    assert result.metrics["initial_layout"]=="exact_t2d_dual_hinge"
    assert result.metrics["actual_string_length"] <= (
        result.metrics["initial_string_length"] + 1e-6
    )
    assert result.metrics["collision_model"]=="3d_obb_sat_discrete"
    # This elementary pin-hinged test must physically rise with routed pull:
    # an animation that remains exactly flat would not meet the main request.
    assert result.metrics["rise_from_t2d"] > 1e-4
    assert len(result.string_path_frames)==len(result.frames)
    assert len(result.string_command_history)==len(result.frames)
    np.testing.assert_allclose(
        result.string_path_frames[0][1:],
        [state.tiles_2d_dual_hinge.vertices[0,0],
         state.tiles_2d_dual_hinge.vertices[1,2]],
    )


def test_string_actuation_is_tension_only_and_shortens_length():
    state=_two_panel_state()
    m=_prepare(state,anchor_height_tiles=3.)
    initial,_,_=_string_geometry(m)
    # Rope already slack at its rest length: no compression force.
    assert _string_constraint(m, initial*1.1,1.)==0.
    np.testing.assert_array_equal(m.rest,state.tiles_2d_dual_hinge.vertices)
    old=m.centers.copy()
    assert _string_constraint(m,initial*.9,1.)>0.
    after,_,_=_string_geometry(m)
    assert after < initial
    assert not np.array_equal(m.centers,old)


def test_no_string_or_fake_guides_fails_instead_of_using_T3D_goal():
    state=_two_panel_state()
    state.string_path.gap_ids=[0]
    with pytest.raises(NativeDeploymentError,match="UNDERCONSTRAINED"):
        _prepare(state,anchor_height_tiles=2.)
    state.string_path.gap_ids=[0,2]
    with pytest.raises(NativeDeploymentError,match="MISSING"):
        _prepare(state,anchor_height_tiles=2.)


def test_discrete_obb_contacts_disambiguate_stacked_vs_separated_panels():
    state=_two_panel_state()
    m=_prepare(state,anchor_height_tiles=2.)
    m.centers[1]=m.centers[0].copy()
    assert _contacts(m,apply=False,strength=0.)>=1
    m.centers[1] += np.array([0.,0.,5.])
    assert _contacts(m,apply=False,strength=0.)==0


def test_native_uses_actual_geometry_not_t3d_goal():
    a=_two_panel_state()
    b=_two_panel_state()
    b.tiles_3d.vertices += np.array([0.,0.,20.])
    config=DeploymentParameters(
        physics_backend="native",steps=6,solver_iterations=5,
        solver_substeps=1,gravity=0.,damping_ratio=.5,
        native_pull_end_ratio=.9,max_animation_frames=5,
    )
    ra=simulate_native_string_deployment(a,config)
    rb=simulate_native_string_deployment(b,config)
    np.testing.assert_allclose(ra.final_tiles,rb.final_tiles,atol=1e-12)
    assert rb.metrics["final_deployment_error_to_T3D"] > (
        ra.metrics["final_deployment_error_to_T3D"]
    )
