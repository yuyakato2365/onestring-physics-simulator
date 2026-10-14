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
