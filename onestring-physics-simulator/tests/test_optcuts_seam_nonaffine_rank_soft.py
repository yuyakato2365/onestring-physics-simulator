"""Failure-reproduction: many physical cut edges, mismatched UV Grid intervals.

The previous affine hard projection has a 1-dimensional seam nullspace
for a curved source boundary mapped to unequal Grid partitions, forcing
the entire seam to a point before any SmoothMax iteration. Soft seam
energy must NOT project to that degenerate feasible subspace.
"""
import numpy as np
import pytest
from onestring_physics.optcuts_seam_affine_constraints import (
    source_cut_affine_rows,AffineSeamProjector,
)
from onestring_physics.optcuts_seam_soft_constraints import (
    seam_matrix_rank_diagnostics,soft_seam_energy_gradient,
    residual_stats,remap_rows,
)
from onestring_physics.optcuts_test_k3d_augmented_lagrangian_patch import (
    _augmented_lagrangian_planarize,_quad_triangle_area_diagnostics,
)


def nonaffine_chart(n_source_edges=8):
    faces=[]
    vertices=[]
    seam_vertex_ids=[]
    for side, knots in (
        (0,[0.,.5,1.]),(1,[0.,.23,.67,1.])
    ):
        x=0. if side==0 else 2.
        ys=np.asarray(knots,float)
        inner=[]
        outer=[]
        for y in ys:
            inner.append(len(vertices))
            vertices.append([x,y])
            outer.append(len(vertices))
            vertices.append([x+(-1. if side==0 else 1.),y])
        seam_vertex_ids.extend(inner)
        for i in range(len(ys)-1):
            faces.append([outer[i],inner[i],inner[i+1],outer[i+1]])
    uv=np.asarray(vertices,float)
    source=[]
    for j in range(n_source_edges):
        ta=j/n_source_edges
        tb=(j+1)/n_source_edges
        for side in (0,1):
            # Both sides represent the SAME physical source edge, but
            # the UV arc-length assignment is *not* the same affine map.
            y0=ta if side==0 else ta+.12*np.sin(np.pi*ta)
            y1=tb if side==0 else tb+.12*np.sin(np.pi*tb)
            x=0. if side==0 else 2.
            source.append((
                np.asarray([[x,y0],[x,y1]],float),
                np.asarray([[0.,ta,0.],[0.,tb,0.]],float),
                (j,j+1),side
            ))
    rows,metadata=source_cut_affine_rows(
        uv,np.asarray(faces,int),source
    )
    xyz=np.zeros((len(uv),3),float)
    for index,(u,y) in enumerate(uv):
        xyz[index]=[
            -1. if u==-1. else 1. if u==3. else 0.,
            y,
            .12*np.sin(np.pi*y),
        ]
    return uv,np.asarray(faces,int),xyz,seam_vertex_ids,rows,metadata


def test_many_source_edges_reveal_old_hard_system_rank_collapse():
    uv,f,xyz,seam,rows,meta=nonaffine_chart(8)
    assert meta["source_cut_physical_edge_count"]==8
    audit=seam_matrix_rank_diagnostics(rows,len(uv))
    assert audit["rank_evaluated"]
    assert audit["seam_vertices"]>=7
    assert audit["seam_nullity"]<=2, audit
    assert audit["seam_rank"]>=audit["seam_vertices"]-2
    # Preserve this negative witness: zero linear residual is NOT proof
    # of nondegenerate, useful seam closure.
    projector=AffineSeamProjector(len(xyz),rows)
    old=projector.project(xyz,strict=True)
    assert projector.residual(old)<1e-8
    before=np.linalg.norm(np.diff(xyz[seam[:3]],axis=0),axis=1)
    after=np.linalg.norm(np.diff(old[seam[:3]],axis=0),axis=1)
    assert after.min()<.6*before.min(),(before,after)


def test_soft_nonaffine_seam_analytic_gradient_matches_finite_difference():
    uv,f,xyz,seam,rows,meta=nonaffine_chart(6)
    energy,gradient=soft_seam_energy_gradient(xyz,rows,23.,scale=.5)
    assert energy>0
    for i,d in ((seam[1],1),(seam[3],1),(seam[-2],2)):
        plus=xyz.copy();minus=xyz.copy()
        plus[i,d]+=1e-6;minus[i,d]-=1e-6
        ep=soft_seam_energy_gradient(plus,rows,23.,scale=.5)[0]
        em=soft_seam_energy_gradient(minus,rows,23.,scale=.5)[0]
        assert gradient[i,d]==pytest.approx(
            (ep-em)/2e-6,rel=1e-5,abs=1e-6
        )


def test_soft_nonaffine_smoothmax_keeps_seam_noncollapsed():
    uv,faces,xyz,seam,rows,metadata=nonaffine_chart(8)
    initial=residual_stats(xyz,rows)["rms"]
    result,metrics=_augmented_lagrangian_planarize(
        xyz,faces,square_weight=3.,anchor_weight=2.,
        outer_iterations=2,inner_iterations=45,
        restoration_sweeps=2,final_polish_sweeps=5,
        soft_seam_rows=rows,soft_seam_weight=30.,
    )
    assert metrics["k3d_smoothmax_enabled"]
    assert metrics["k3d_seam_stitch_mode"]=="soft"
    assert not metrics["k3d_al_source_seam_hard_equality"]
    assert np.isfinite(result).all()
    ratio=_quad_triangle_area_diagnostics(xyz,result,faces)
    assert ratio["collapsed_or_inverted_triangle_count"]==0,ratio
    after=residual_stats(result,rows)["rms"]
    assert after<initial,(initial,after)
    assert np.linalg.norm(result[seam[1]]-result[seam[0]])>.15


def test_soft_rows_survive_extrusion_aware_geometry_reindex():
    from onestring_physics.extrusion_aware import geometry_view
    from types import SimpleNamespace
    uv,faces,xyz,seam,rows,metadata=nonaffine_chart(5)
    mesh=SimpleNamespace(
        vertices=xyz,faces=faces,
        metrics={"seam_exact_preflight_certified":True,
                 "seam_stitch_mode":"soft",
                 "seam_exact_affine_rows":rows,
                 "seam_exact_vertex_groups":[]},
    )
    mapping=np.arange(len(xyz),dtype=int)
    reduced=geometry_view(mesh,mapping)
    assert reduced.metrics["seam_affine_rows_reindexed_for_reduction"]
    assert residual_stats(
        xyz,reduced.metrics["seam_exact_affine_rows"]
    )["rms"]==pytest.approx(residual_stats(xyz,rows)["rms"])
