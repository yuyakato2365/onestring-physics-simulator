"""Focused geometry and feasibility tests for hard XY footprint mode."""
from __future__ import annotations

import time

import numpy as np

from onestring_physics.hard_collision_solver import (
    footprint_contacts, repair_nonpenetration, refine_feasible_pose,
)


def _fixture(separation=0.55):
    square=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    tiles=np.stack([np.tile(square,(2,1)),
                    np.tile(square+[separation,0.],(2,1))])
    centers=tiles[:,:4].mean(axis=1)
    local=tiles-centers[:,None,:]
    faces=np.arange(16).reshape(2,8)
    reference=np.zeros((2,3))
    def positions(pose):
        angle=pose[:,0]
        c,s=np.cos(angle)[:,None],np.sin(angle)[:,None]
        rotated=np.stack((c*local[:,:,0]-s*local[:,:,1],
                          s*local[:,:,0]+c*local[:,:,1]),axis=-1)
        return rotated+centers[:,None,:]+pose[:,None,1:],rotated
    return tiles,faces,reference,positions


def test_contact_depth_and_zero_contact():
    original,faces,reference,positions=_fixture()
    ii,jj,n,sign,depth=footprint_contacts(original,faces)
    assert len(depth)==1 and np.isclose(depth[0],0.45)
    separate,_,_,positions_separate=_fixture(1.0)
    assert len(footprint_contacts(separate,faces)[-1])==0


def test_complete_containment_uses_translation_depth_not_interval_intersection():
    outer=np.array([[0,0],[4,0],[4,4],[0,4]]*2,float)
    inner=np.array([[1,1],[2,1],[2,2],[1,2]]*2,float)
    xy=np.stack([outer,inner])
    faces=np.arange(16).reshape(2,8)
    depth=footprint_contacts(xy,faces)[-1]
    assert len(depth)==1 and np.isclose(depth[0],2.0)


def test_repair_is_rigid_and_collision_free():
    tiles,faces,reference,positions=_fixture()
    candidate,feasible,frames,diagnostics=repair_nonpenetration(
        reference,positions,faces,reference,5.0,1.0,
        time.perf_counter()+10.0,np.random.default_rng(4),
    )
    assert feasible
    assert len(footprint_contacts(positions(candidate)[0],faces)[-1])==0
    assert len(frames)>=1
    assert diagnostics[-1]['collisions']==0
    for original,moved in zip(tiles,positions(candidate)[0]):
        np.testing.assert_allclose(
            np.linalg.norm(original[:,None,:]-original[None,:,:],axis=-1),
            np.linalg.norm(moved[:,None,:]-moved[None,:,:],axis=-1),
            atol=1e-12,
        )


def test_infeasible_bounds_are_not_certified():
    tiles,faces,reference,positions=_fixture()
    candidate,feasible,frames,diagnostics=repair_nonpenetration(
        reference,positions,faces,reference,0.,1.,
        time.perf_counter()+10.0,np.random.default_rng(4),
        max_iterations=50,
    )
    assert not feasible
    assert len(footprint_contacts(positions(candidate)[0],faces)[-1])>0


def test_refinement_never_reintroduces_penetration():
    tiles,faces,reference,positions=_fixture()
    candidate,feasible,_,_=repair_nonpenetration(
        reference,positions,faces,reference,5.,1.,
        time.perf_counter()+10.,np.random.default_rng(9),
    )
    assert feasible

    def objective(pose):
        # Drives the panels towards their original overlapping centers.
        # This unconstrained objective would prefer a penetrating pose.
        delta=pose-reference
        return float(np.sum(delta**2)), 2.*delta

    after,history=refine_feasible_pose(
        candidate,positions,faces,reference,5.,1.,
        time.perf_counter()+10.,np.random.default_rng(9),
        objective,iterations=12,
    )
    assert len(footprint_contacts(positions(after)[0],faces)[-1])==0
    assert objective(after)[0] <= objective(candidate)[0]+1e-10
