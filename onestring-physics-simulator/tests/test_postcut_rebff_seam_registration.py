"""Real physical S seam, not synthetic UV packing, drives T2D placement."""
from types import SimpleNamespace

import numpy as np
import pytest

from postcut_rebff_seam_registration import (
    _boundary_point_groups, _rigid_2d, align_t2d_at_source_split_seams
)


def make_case(*, separate_by=(3.,4.), alter_source=False):
    # Two disk-shaped source triangles on opposite sides of x=1.
    patch_a={
        "xyz":np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float),
        "faces":np.array([[0,1,2],[0,2,3]],int),
    }
    patch_b={
        "xyz":np.array([[1,0,0],[2,0,0],[2,1,0],[1,1,0]],float),
        "faces":np.array([[0,1,2],[0,2,3]],int),
    }
    if alter_source:
        patch_b["xyz"][:,0]+=2.
    pieces=[patch_a,patch_b]
    tile3d=np.array([
        patch_a["xyz"],
        patch_b["xyz"],
    ])
    t2d=tile3d.copy()
    t2d[1,:,:2]+=np.asarray(separate_by,float)
    # 8 vertices per real thick tile; bottom copies top shifted in Z.
    tile8=np.concatenate([t2d,t2d-np.array([0,0,0.1])],axis=1)
    a=SimpleNamespace(
        vertices=tile8.copy(),transform_matrices=np.tile(np.eye(4),(2,1,1)),
        metrics={},
    )
    b=SimpleNamespace(
        vertices=tile8.copy(),transform_matrices=np.tile(np.eye(4),(2,1,1)),
        metrics={},
    )
    hinges=SimpleNamespace(hinges=[],metrics={})
    state=SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(faces=np.array([[0,1,2,3],[4,5,6,7]])),
        mesh_3d_initial=SimpleNamespace(
            vertices=tile3d.reshape(-1,3),
            faces=np.array([[0,1,2,3],[4,5,6,7]],int),
        ),
        tiles_2d_dual_hinge=a, tiles_2d_top_hinge=b,hinge_graph=hinges,
    )
    return state,pieces


def test_physical_seam_points_detected():
    _,pieces=make_case()
    shared=_boundary_point_groups(pieces)
    assert (0,1) in shared
    assert len(shared[(0,1)])==2
    assert {tuple(p) for p in shared[(0,1)]}=={(1.,0.,0.),(1.,1.,0.)}


def test_seam_aligned_without_changing_panel_dimensions():
    state,pieces=make_case()
    old=state.tiles_2d_dual_hinge.vertices.copy()
    before=np.linalg.norm(old[:,None,:,:]-old[:,:,None,:],axis=-1)
    result=align_t2d_at_source_split_seams(state,pieces,[0,1])
    after=np.linalg.norm(
        state.tiles_2d_dual_hinge.vertices[:,None,:,:]-
        state.tiles_2d_dual_hinge.vertices[:,:,None,:],axis=-1)
    assert np.allclose(before,after,atol=1e-10)
    assert result["split_seam_pair_count"]==1
    assert result["split_seam_anchor_pair_count"]==2
    assert result["split_seam_fit_rms_max"]<1e-10
    a=state.tiles_2d_dual_hinge.vertices
    # Shared source physical seam = x1 in tile a / x0 in tile b.
    assert np.allclose(a[0,[1,2],:2],a[1,[0,3],:2],atol=1e-10)
    assert np.allclose(a,state.tiles_2d_top_hinge.vertices)
    assert result["split_seam_matching_certified"] is False


def test_non_matching_source_pieces_do_not_fake_success():
    state,pieces=make_case(alter_source=True)
    with pytest.raises(RuntimeError,match="Cannot align re-BFF T2D"):
        align_t2d_at_source_split_seams(state,pieces,[0,1])


def test_alignment_preserves_3d_heights_and_transform_matrices():
    state,pieces=make_case(separate_by=(10.,-10.))
    original=state.tiles_2d_dual_hinge.vertices.copy()
    align_t2d_at_source_split_seams(state,pieces,[0,1])
    assert np.allclose(
        state.tiles_2d_dual_hinge.vertices[:,:,2], original[:,:,2])
    assert not np.allclose(
        state.tiles_2d_dual_hinge.transform_matrices[1],np.eye(4))


def test_rigid_fit_cannot_scalably_deform_a_nonmatching_seam():
    x=np.array([[0.,0.],[0.,1.],[0.,2.]])
    y=np.array([[0.,0.],[0.,2.],[0.,4.]])
    r,t,residual=_rigid_2d(x,y)
    assert residual>0.5
    assert abs(np.linalg.det(r)-1.)<1e-10
    assert np.allclose(r.T@r,np.eye(2),atol=1e-10)


def test_grid_cut_exposes_same_physical_seam_as_rectangle_corners():
    from postcut_rebff_iterative import _cut_child_rectangular_corners
    from scripts.test_lilium_postcut_bff_hypothesis import clip_piece
    x=np.array([[i/2.,j/2.,0.] for j in range(3) for i in range(3)])
    f=np.array([
        [3*j+i,3*j+i+1,3*(j+1)+i+1]
        for j in range(2) for i in range(2)
    ]+[
        [3*j+i,3*(j+1)+i+1,3*(j+1)+i]
        for j in range(2) for i in range(2)
    ],int)
    chart={'xyz':x, 'new_uv':x[:,:2], 'faces':f}
    seam_ends=[]
    for positive in (False,True):
        xyz,uv,faces,_=clip_piece(x,x[:,:2],f,1,0.5,positive)
        corners=_cut_child_rectangular_corners(
            chart,xyz,uv,faces,'row',0.5,positive
        )
        assert corners.shape==(4,3)
        seam=corners[np.abs(corners[:,1]-.5)<1e-8]
        assert seam.shape==(2,3)
        seam_ends.append(np.asarray(sorted(seam.tolist())))
    assert np.allclose(seam_ends[0],seam_ends[1])


def test_required_rectangular_children_receive_explicit_corners():
    from postcut_rebff_iterative import _split_and_rebff
    from scripts.test_lilium_postcut_bff_hypothesis import clip_piece
    x=np.array([[i/2.,j/2.,0.] for j in range(3) for i in range(3)])
    f=np.array([
        [3*j+i,3*j+i+1,3*(j+1)+i+1]
        for j in range(2) for i in range(2)
    ]+[
        [3*j+i,3*(j+1)+i+1,3*(j+1)+i]
        for j in range(2) for i in range(2)
    ],int)
    parent={'xyz':x,'new_uv':x[:,:2],'faces':f,'side':'initial'}
    seen=[]
    def fake_bff(xyz,faces,*,rectangle_corner_positions):
        corners=np.asarray(rectangle_corner_positions)
        seen.append(corners)
        assert corners.shape==(4,3)
        return xyz[:,:2],{'bff_rectangle_verified':True}
    fake_bff.seam_corners_required=True
    children=_split_and_rebff(parent,'row',.5,fake_bff)
    assert len(children)==2
    assert len(seen)==2
    assert all(c['seam_rectangle_corners_constrained'] for c in children)


def test_independent_rectangular_bff_uvs_share_a_single_grid_after_similarity():
    from postcut_rebff_seam_registration import align_rectangular_bff_charts_on_source_seam
    _,patches=make_case()
    for p in patches:
        p['seam_rectangle_corners_constrained']=True
    patches[0]['new_uv']=patches[0]['xyz'][:,:2].copy()
    # Different global scale/translation from the independently solved BFF.
    patches[1]['new_uv']=.5*patches[1]['xyz'][:,:2]+[7.,-3.]
    before=[p['new_uv'].copy() for p in patches]
    audit=align_rectangular_bff_charts_on_source_seam(patches)
    assert audit['postcut_seam_uv_pair_count']==1
    # Common physical Split endpoints are now identical UV points.
    assert np.allclose(
        patches[0]['new_uv'][[1,2]],patches[1]['new_uv'][[0,3]],atol=1e-10
    )
    assert np.allclose(before[0][1]-before[0][2],
                       patches[0]['new_uv'][1]-patches[0]['new_uv'][2])
    assert audit['postcut_seam_uv_preserves_csf_ratio']
    # Child rectangle stays on the opposite side of the physical seam.
    assert np.mean(patches[0]['new_uv'][:,0])<0.
    assert np.mean(patches[1]['new_uv'][:,0])>0.


def test_rebff_charts_without_seam_corner_constraints_are_rejected():
    from postcut_rebff_seam_registration import align_rectangular_bff_charts_on_source_seam
    _,patches=make_case()
    patches[0]['new_uv']=patches[0]['xyz'][:,:2].copy()
    patches[1]['new_uv']=patches[1]['xyz'][:,:2].copy()
    with pytest.raises(RuntimeError,match='did not constrain BOTH'):
        align_rectangular_bff_charts_on_source_seam(patches)


def test_registered_common_omega_grid_is_used_for_t2d_seam():
    state,patches=make_case(separate_by=(8.,-6.))
    state.mesh_2d_initial.vertices=state.mesh_3d_initial.vertices.copy()
    state.mesh_2d_initial.metrics={'postcut_seam_shared_grid_vertex_count':2}
    audit=align_t2d_at_source_split_seams(state,patches,[0,1])
    assert audit['split_seam_alignment']=='shared_source_seam_grid_rigid_SE2'
    assert audit['split_seam_anchor_pair_count']==2
    assert audit['split_seam_fit_rms_max']<1e-10
    tiles=state.tiles_2d_dual_hinge.vertices
    assert np.allclose(tiles[0,[1,2],:2],tiles[1,[0,3],:2],atol=1e-10)
