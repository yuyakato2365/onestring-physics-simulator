"""Exact free-BFF seam correspondence: no false physical/T2D certification."""
import numpy as np
import pytest

from postcut_free_bff_matched_seam import match_free_bff_split_children
from scripts.test_lilium_postcut_bff_hypothesis import field


def _patch(xmin,xmax, distorted):
    nx,ny=2,4
    xy=np.array([(xmin+(xmax-xmin)*i/nx, float(j)/ny)
                 for j in range(ny+1) for i in range(nx+1)])
    def vid(i,j):
        return j*(nx+1)+i
    f=np.array([t for j in range(ny) for i in range(nx)
                for t in ([vid(i,j),vid(i+1,j),vid(i+1,j+1)],
                          [vid(i,j),vid(i+1,j+1),vid(i,j+1)])],int)
    xyz=np.column_stack((xy,np.zeros(len(xy))))
    uv=xy.copy()
    if distorted:
        uv[:,0]=1.5*xy[:,0]+.12*np.sin(np.pi*xy[:,1])+.35
        uv[:,1]=1.5*xy[:,1]+.10*np.sin(np.pi*xy[:,1])-.25
    else:
        uv[:,0]=xy[:,0]+.035*np.sin(np.pi*xy[:,1])
        uv[:,1]=xy[:,1]
    return {
        "xyz":xyz,"faces":f,"original_uv":xy,"new_uv":uv,
        "independent_bff_uv":uv.copy(),
        "independent_bff_post":field(xyz,uv,f),
        "post":field(xyz,uv,f),
        "side":("right" if distorted else "left"),
    }


def test_harmonic_seam_matching_is_exact_and_keeps_all_face_orientations():
    a=_patch(0,1,False)
    b=_patch(1,2,True)
    raw=[a["independent_bff_uv"].copy(),b["independent_bff_uv"].copy()]
    children=match_free_bff_split_children(
        [a,b], "col",1., evaluate=field
    )
    assert children[0]["postcut_free_seam_aligned"]
    assert children[1]["postcut_free_seam_aligned"]
    assert children[0]["postcut_free_seam_audit"]["seam_sample_count"]==5
    assert children[0]["postcut_free_seam_audit"]["seam_uv_error_max"]<1e-12
    for patch,original in zip(children,raw):
        assert np.array_equal(patch["independent_bff_uv"],original)
        assert np.isfinite(patch["post"]["linear_span"])
        assert patch["postcut_free_seam_audit"]["physical_T2D_seam_alignment_verified"] is False
        assert patch["postcut_free_seam_audit"]["preserves_original_global_grid"] is False
        uv=patch["new_uv"];faces=patch["faces"];xy=uv[faces]
        signed=np.cross(xy[:,1]-xy[:,0],xy[:,2]-xy[:,0])
        assert np.min(signed)>0
    seam_a=a["new_uv"][np.isclose(a["original_uv"][:,0],1)]
    seam_b=b["new_uv"][np.isclose(b["original_uv"][:,0],1)]
    assert np.allclose(seam_a,seam_b,atol=1e-12)


def test_split_requires_identical_physical_seam_samples():
    a=_patch(0,1,False)
    b=_patch(1,2,True)
    b["xyz"][3,0]+=.001
    # Select only the on-seam XYZ vertex; index 3 is 0.0 in x, so change 3D
    # of vertex 5 that actually lies on x=1.
    b["xyz"][5,0]+=.001
    with pytest.raises(ValueError,match="seam source vertices differ"):
        match_free_bff_split_children([a,b],"col",1.,evaluate=field)
