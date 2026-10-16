"""Synthetic regressions for the multi-condition shared-seam experiment."""
import numpy as np
from postcut_free_seam_sweep import sweep_free_bff_shared_seam


def _piece(x0, x1, wavy):
    points=np.array([(x0+(x1-x0)*i/3, float(j)/5)
                     for j in range(6) for i in range(4)])
    faces=[]
    for j in range(5):
        for i in range(3):
            a=j*4+i
            faces.extend([[a,a+1,a+5],[a,a+5,a+4]])
    triangles=np.array(faces,int)
    xyz=np.column_stack((points,np.zeros(len(points))))
    uv=points.copy()
    if wavy:
        uv[:,0]=points[:,0]*1.2+.06*np.sin(np.pi*points[:,1])+.4
        uv[:,1]=points[:,1]*1.2+.04*np.sin(np.pi*points[:,1])-.3
    return {
        "xyz":xyz,"faces":triangles,
        "original_uv":points,
        "new_uv":uv,"independent_bff_uv":uv.copy(),
        "independent_bff_post":{"linear_span":1.0},
    }


def _field(xyz,uv,faces):
    x=np.asarray(xyz)[faces]
    u=np.asarray(uv)[faces]
    a3=np.linalg.norm(np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]),axis=1)
    d1=u[:,1]-u[:,0];d2=u[:,2]-u[:,0]
    a2=np.abs(d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0])
    valid=(a2>1e-16)&(a3>1e-16)
    j=a3[valid]/a2[valid]
    return {
        "linear_span":float(np.sqrt(j.max()/j.min())),
        "area_span":float(j.max()/j.min()),
        "linear_span_pct1_99":float(np.sqrt(np.percentile(j,99)/np.percentile(j,1))),
        "jac_min":float(j.min()),"jac_max":float(j.max()),
    }


def test_sweep_explores_distinct_registration_and_boundary_conditions():
    patches=[_piece(0,1,False),_piece(1,2,True)]
    before=[p["new_uv"].copy() for p in patches]
    sweep=sweep_free_bff_shared_seam(
        patches,"col",1.0,evaluate=_field
    )
    assert sweep["candidates_count"]>=60
    assert sweep["source_seam_points"]==6
    assert sweep["source_xyz_max_error"]==0.
    assert sweep["valid_exact_seam_count"]>=3
    assert sweep["best_exact_seam"]["uv_seam_max_error"]<1e-9
    assert any("sliding_outer_boundary" in r["name"] for r in sweep["all_candidates"])
    assert any("fixed_outer_boundary" in r["name"] for r in sweep["all_candidates"])
    assert any("affine" in r["name"] for r in sweep["all_candidates"])
    assert any("straight_parent_grid" in r["name"] for r in sweep["all_candidates"])
    assert all(np.array_equal(p["new_uv"],uv)
               for p,uv in zip(patches,before))
    assert not sweep["best_exact_seam"]["physical_grid_compatibility_proven"]
