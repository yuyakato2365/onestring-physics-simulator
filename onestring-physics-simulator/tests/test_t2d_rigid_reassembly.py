"""Solid T2D->3D rigid-placement diagnostics (not physical actuation)."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from onestring_physics.t2d_rigid_reassembly import (
    proper_rigid_fit, rigid_reassemble_t2d, obb_overlap_3d_counts,
)


def _tile(x=0.,y=0.):
    top=np.array([[x,y,0.],[x+1.,y,0.],[x+1.,y+1.,0.],[x,y+1.,0.]])
    return np.concatenate((top,top+np.array([0.,0.,-.1])),axis=0)


def _state(rest,target,hinges=()):
    return SimpleNamespace(
        tiles_2d_dual_hinge=SimpleNamespace(vertices=rest.copy()),
        tiles_3d=SimpleNamespace(vertices=target.copy()),
        hinge_graph=SimpleNamespace(hinges=list(hinges)),
    )


def test_kabsch_repositions_actual_t2d_without_distorting_shape():
    rest=_tile()
    rot=Rotation.from_euler("xyz",[.37,-.51,.21]).as_matrix()
    target=rest@rot.T+np.array([4.,3.,2.])
    placed,err=proper_rigid_fit(rest,target)
    np.testing.assert_allclose(placed,target,atol=1e-11)
    assert err<1e-11
    assert np.linalg.det(rot)==pytest.approx(1.)
    np.testing.assert_allclose(
        np.linalg.norm(placed[:,None]-placed[None,:],axis=-1),
        np.linalg.norm(rest[:,None]-rest[None,:],axis=-1),
    )


def test_reassembly_does_not_copy_distorted_target_into_t2d_panels():
    rest=np.array([_tile()])
    target=rest.copy()
    target[0,2,0]+=0.3
    placed,report=rigid_reassemble_t2d(_state(rest,target))
    assert report["kinematic_only"]
    assert report["rigid_panels_preserved"]
    assert report["max_internal_distance_change"] < 1e-10
    assert report["total_target_vertex_rms"]>0.02
    assert report["max_tile_target_fit_rms"]>0.02
    assert not np.allclose(placed,target)
    np.testing.assert_allclose(
        np.linalg.norm(placed[0,0]-placed[0,1]),
        np.linalg.norm(rest[0,0]-rest[0,1]),atol=1e-10
    )


def test_flat_collision_free_can_be_overlapping_after_target_reassembly():
    rest=np.stack((_tile(0.,0.),_tile(2.,0.)))
    # Both correspond to the same T3D target area, so a kinematically
    # placed T2D set can overlap despite zero initial T2D collisions.
    target=np.stack((_tile(0.,0.),_tile(0.,0.)))
    placed,report=rigid_reassemble_t2d(_state(rest,target))
    assert report["initial_t2d_3d_obb_pairs"]==0
    assert report["reassembled_3d_obb_pairs"]==1
    assert report["reassembled_3d_obb_max_depth"]>0
    assert obb_overlap_3d_counts(placed)["pairs"]==1


def test_reassembled_hinge_gap_is_reported_without_adjusting_panels():
    rest=np.stack((_tile(0.,0.),_tile(1.,0.)))
    target=rest.copy()
    target[1]+=np.array([0.,0.,1.])
    hinge=SimpleNamespace(tile_a=0,tile_b=1,local_vertex_a=1,
                          local_vertex_b=0,surface="top")
    placed,report=rigid_reassemble_t2d(_state(rest,target,[hinge]))
    assert report["hinge_rms"]==pytest.approx(1.)
    assert report["hinge_max"]==pytest.approx(1.)
    assert report["hinge_count"]==1


def test_zero_or_invalid_tiles_do_not_silently_generate_physics():
    with pytest.raises(ValueError):
        proper_rigid_fit(np.zeros((4,3)),np.zeros((4,3)))
    with pytest.raises(ValueError):
        obb_overlap_3d_counts(np.zeros((3,4,3)))
