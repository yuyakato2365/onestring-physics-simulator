"""Exact full rectangular Omega coverage and identical cut-seam grid tests."""
import numpy as np
import pytest

from postcut_rebff_exact_rectangle_grid import (
    rectangle_bounds, grid_at_pitch, choose_exact_grid, matching_seam_edges,
)


def patch(x0,x1,y0,y1, *, offset=(0.,0.)):
    # Tris of a proper open disk. Underlying source S is a rectangle.
    xyz=np.array([
        [x0,y0,0.],[x1,y0,0.],[x1,y1,0.],[x0,y1,0.]
    ],float)
    uv=xyz[:,:2].copy()+np.asarray(offset,float)
    return {
        "xyz":xyz, "new_uv":uv,
        "faces":np.array([[0,1,2],[0,2,3]],int),
        "bff_metadata":{"bff_rectangle_verified":True},
        "seam_rectangle_corners_constrained":True,
    }


def area_of_quads(mesh):
    x=np.asarray(mesh["vertices"],float)
    f=np.asarray(mesh["faces"],int)
    return float(np.sum(np.abs(np.cross(
        x[f[:,1],:2]-x[f[:,0],:2],
        x[f[:,3],:2]-x[f[:,0],:2]
    ))))


def test_rectangle_is_tiled_to_exact_edges_even_if_not_multiple_of_pitch():
    p=patch(0.,1.037,0.,.923)
    (mesh,)=grid_at_pitch([p],.2)
    assert np.allclose(mesh["rectangle_bounds"][1],[1.037,.923])
    assert np.isclose(area_of_quads(mesh),1.037*.923,atol=1e-11)
    assert np.isclose(mesh["area_fraction_covered"],1.)
    assert np.max(mesh["vertices"][:,0])==1.037
    assert np.max(mesh["vertices"][:,1])==.923
    assert mesh["min_cell_width"]>0 and mesh["min_cell_height"]>0
    assert len(mesh["faces"])>0


def test_two_rectangles_same_grid_seam_segments_and_no_loss():
    left=patch(0.,1.037,0.,.923)
    right=patch(1.037,2.831,0.,.923)
    meshes=grid_at_pitch([left,right],.22)
    matched=matching_seam_edges(meshes)
    assert matched[(0,1)]>=2
    for m in meshes:
        assert m["area_fraction_covered"]==pytest.approx(1.,abs=1e-11)
    assert area_of_quads(meshes[0])==pytest.approx(1.037*.923)
    assert area_of_quads(meshes[1])==pytest.approx((2.831-1.037)*.923)
    # Every common seam edge must have exactly the same endpoints on BOTH
    # patches (we're not merely measuring nearby panel corners).
    def edge_intervals(m):
        v=m["vertices"][:,:2]
        f=m["faces"]
        pairs={}
        for quad in f:
            for i,j in zip(quad,np.roll(quad,-1)):
                key=tuple(sorted((int(i),int(j))))
                pairs[key]=pairs.get(key,0)+1
        return sorted(
            (round(min(v[a,1],v[b,1]),9),
             round(max(v[a,1],v[b,1]),9))
            for (a,b),c in pairs.items()
            if c==1 and abs(v[a,0]-1.037)<1e-9
            and abs(v[b,0]-1.037)<1e-9
        )
    assert edge_intervals(meshes[0])==edge_intervals(meshes[1])


def test_candidate_density_respects_exact_rectangle_coverage():
    meshes_src=[patch(0.,1.13,0.,1.17),patch(1.13,2.05,0.,1.17)]
    pitch,meshes=choose_exact_grid(meshes_src, 80,grid=9)
    assert pitch>0.
    assert sum(len(m["faces"]) for m in meshes)>0
    assert matching_seam_edges(meshes).get((0,1),0)>=1
    assert all(abs(m["area_fraction_covered"]-1.)<1e-10 for m in meshes)


def test_grid_rejects_overlapping_rectangles_and_nonrectangular_boundary():
    with pytest.raises(RuntimeError,match="overlap"):
        grid_at_pitch([patch(0,1,0,1),patch(.5,1.5,0,1)],.2)
    twisted=patch(0.,1.,0.,1.)
    twisted["new_uv"][2]=[.7,1.]
    with pytest.raises(RuntimeError,match="not an axis-aligned rectangle|missing a real corner"):
        rectangle_bounds(twisted)


def test_no_grid_cell_when_pitch_exceeds_rectangle_would_be_wrong():
    p=patch(0.,.03,0.,.04)
    meshes=grid_at_pitch([p],pitch=.5)
    assert len(meshes[0]["faces"])==1
    assert meshes[0]["area_fraction_covered"]==pytest.approx(1.)



def test_four_chart_source_corner_contacts_do_not_block_full_m2d_assembly():
    # Reproduces the failure class from a recursively split rectangular Omega:
    # adjacent patches must share the SAME lattice edges, but diagonally
    # opposed charts share only one source vertex and MUST NOT demand an edge.
    from postcut_rebff_full_pipeline import _assemble_patches
    patches=[]
    for i,(x0,y0) in enumerate([(0.,0.),(1.,0.),(0.,1.),(1.,1.)]):
        p=patch(x0,x0+1.,y0,y0+1.)
        # Each source chart was independently BFF'ed, with a different
        # conformal uniform scale and UV-space translation.
        p["new_uv"]=p["new_uv"]*(.5+i*.125)+[5*i,-3*i]
        patches.append(p)
    result=_assemble_patches(patches,None,8,64)
    audit=result["audit"]
    assert audit["postcut_rebff_final_quad_count"]>0
    assert all(
        abs(v-1.)<1e-10
        for v in audit["postcut_omega_rectangles_area_fraction_covered"]
    )
    matched=audit["postcut_seam_exact_uv_edge_pairs"]
    for pair in ("0-1","0-2","1-3","2-3"):
        assert matched.get(pair,0)>0
    assert "0-3" not in matched and "1-2" not in matched



def test_recursive_joint_atlas_seam_edges_and_physical_points_match():
    """A second-stage T-junction keeps every S seam physically registered.

    Raw child BFFs deliberately have independent nonlinear seam sampling.
    A mere per-island similarity cannot align those seam interior vertices.
    """
    from postcut_rebff_iterative import _split_and_rebff
    from postcut_rebff_joint_atlas import verify_joint_source_seams
    from postcut_rebff_full_pipeline import _assemble_patches
    from postcut_rebff_seam_registration import _boundary_point_groups

    xyz=np.array([[i/6.,j/6.,0.] for j in range(7) for i in range(7)])
    faces=np.array([
        [7*j+i,7*j+i+1,7*(j+1)+i+1]
        for j in range(6) for i in range(6)
    ]+[
        [7*j+i,7*(j+1)+i+1,7*(j+1)+i]
        for j in range(6) for i in range(6)
    ],int)
    original={"xyz":xyz,"new_uv":xyz[:,:2].copy(),
              "faces":faces,"side":"initial"}
    call=[0]
    def nonlinear_raw_bff(x,f,*,rectangle_corner_positions):
        call[0]+=1
        uv=np.asarray(x[:,:2],float).copy()
        uv[:,0]=uv[:,0]*(1.2+0.07*call[0])+5.*call[0]
        uv[:,1]=uv[:,1]*(1.4+0.03*call[0])+.025*np.sin(
            2*np.pi*uv[:,1]
        )-3.*call[0]
        return uv,{"bff_rectangle_verified":True}
    nonlinear_raw_bff.seam_corners_required=True
    first=_split_and_rebff(original,"row",.5,nonlinear_raw_bff)
    assert len(first)==2
    pieces=[]
    for parent in first:
        pieces.extend(_split_and_rebff(
            parent,"col",.5,nonlinear_raw_bff
        ))
    assert len(pieces)==4
    assert all(p["postcut_joint_atlas"] for p in pieces)
    assert all(p["independent_bff_post"]["linear_span"]>0 for p in pieces)
    source_pairs=_boundary_point_groups(pieces)
    assert len(source_pairs)>=4
    seam=verify_joint_source_seams(pieces)
    assert seam["postcut_seam_uv_full_physical_correspondence_verified"]
    assert seam["postcut_seam_interior_correspondence_error_max"]<1e-8
    packed=_assemble_patches(pieces,None,8,64)
    matched=packed["audit"]["postcut_seam_exact_uv_edge_pairs"]
    assert all(matched.get(f"{a}-{b}",0)>=1 for a,b in source_pairs)
    assert all(
        abs(fraction-1.)<1e-10 for fraction in
        packed["audit"]["postcut_omega_rectangles_area_fraction_covered"]
    )



def test_eight_recursive_rectangles_retain_full_source_seams():
    from postcut_rebff_iterative import _split_and_rebff
    from postcut_rebff_joint_atlas import verify_joint_source_seams
    from postcut_rebff_full_pipeline import _assemble_patches
    from postcut_rebff_seam_registration import _boundary_point_groups

    xyz=np.array([[i/8.,j/8.,0.] for j in range(9) for i in range(9)])
    triangles=np.array([
        [9*j+i,9*j+i+1,9*(j+1)+i+1]
        for j in range(8) for i in range(8)
    ]+[
        [9*j+i,9*(j+1)+i+1,9*(j+1)+i]
        for j in range(8) for i in range(8)
    ],int)
    root={"xyz":xyz,"new_uv":xyz[:,:2].copy(),
          "faces":triangles,"side":"initial"}
    call=[0]
    def rebff(x,f,*,rectangle_corner_positions):
        call[0]+=1
        uv=np.asarray(x[:,:2],float).copy()
        uv[:,0]=uv[:,0]*(1.3+.01*call[0])+3.*call[0]
        uv[:,1]=uv[:,1]*(1.7+.02*call[0])+0.03*np.sin(
            np.pi*uv[:,1]
        )-5.*call[0]
        return uv,{"bff_rectangle_verified":True}
    rebff.seam_corners_required=True
    first=_split_and_rebff(root,"row",.5,rebff)
    four=[]
    for p in first:
        four.extend(_split_and_rebff(p,"col",.5,rebff))
    eight=[]
    for i,p in enumerate(four):
        lo=np.min(p["new_uv"],axis=0)
        hi=np.max(p["new_uv"],axis=0)
        axis="row" if i%2==0 else "col"
        dim={"row":1,"col":0}[axis]
        eight.extend(_split_and_rebff(
            p,axis,float((lo[dim]+hi[dim])/2.),rebff
        ))
    assert len(eight)==8
    seams=_boundary_point_groups(eight)
    assert len(seams)>=8
    metrics=verify_joint_source_seams(eight)
    assert metrics["postcut_seam_interior_correspondence_error_max"]<1e-8
    packed=_assemble_patches(eight,None,12,144)
    matched=packed["audit"]["postcut_seam_exact_uv_edge_pairs"]
    assert all(matched.get(f"{a}-{b}",0)>0 for a,b in seams)
    assert all(
        abs(coverage-1.)<1e-10 for coverage in
        packed["audit"]["postcut_omega_rectangles_area_fraction_covered"]
    )
