"""Reproduce the Bunny failure: the original cut ENDS MID Grid edge.

The (425,426) report is 64% cut / 36% truly open and the (772,815)
report is 95% cut / 5% truly open, with 100% source boundary coverage.
The correct treatment is a physical cut sub-interval (not classifying
the whole Grid edge as either seam or open boundary).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_seam_affine_constraints import (
    AffineSeamProjector,source_cut_affine_rows,
)
from onestring_physics.optcuts_seam_exact_closure import (
    discover_exact_seam_topology,_covered_fraction_on_source_boundary,
)
from onestring_physics.optcuts_test_k3d_augmented_lagrangian_patch import (
    _augmented_lagrangian_planarize,
)
from onestring_physics.paper_extrusion import extrude_paper_face_planarity


def mixed_chart():
    # LEFT quad covers 0<=y<=1. RIGHT quad covers .4<=y<=1.
    # Both are legitimate 4-corner fabricatable panels. The LEFT
    # perimeter has a 40% genuine source-open / 60% physical seam split
    # in the middle of one unsplit Grid quad edge.
    uv=np.array([
        [0,0],[1,0],[1,1],[0,1],
        [2,.4],[3,.4],[3,1],[2,1],
    ],float)
    faces=np.array([
        [0,1,2,3],[4,5,6,7]
    ],int)
    # Source physical mesh is topological left pentagon plus right quad.
    # Physical x=1, y>=.4 is 2-sided; below .4 it is a *real* opening.
    xyz=np.array([
        [0,0,0],[1,0,0],[1,.4,0],[1,1,0],
        [0,1,0],[2,.4,0],[2,1,0],
    ],float)
    sf=np.array([
        [0,1,2],[0,2,4],[2,3,4],
        [2,5,6],[2,6,3],
    ],int)
    source_uv=np.array([
        [0,0],[1,0],[1,.4],[1,1],[0,1],
        [2,.4],[2,1],[3,.4],[3,1],
    ],float)
    uf=np.array([
        [0,1,2],[0,2,4],[2,3,4],
        [5,7,8],[5,8,6],
    ],int)
    chart=SimpleNamespace(
        surface_vertices_3d=xyz,
        surface_faces=sf,
        uv_vertices_2d=source_uv,
        uv_faces=uf,
    )
    return chart,uv,faces


def test_mixed_cut_open_grid_edge_not_misclassified():
    chart,uv,faces=mixed_chart()
    result=discover_exact_seam_topology(chart,uv,faces)
    assert result["status"]=="conforming"
    assert result["unclassified_boundary_edges"]==0
    assert result["mixed_seam_open_grid_edges"]==1
    assert not result["affine_exact_stitchable"]
    assert result["seam_stitch_mode"]=="soft"
    assert result["affine_metrics"]["source_cut_physical_edge_count"]==1
    assert result["affine_metrics"]["source_cut_constraint_row_count"]>=2
    assert len(result["affine_rows"])>=2
    mixed=result["mixed_grid_edge_sample"][0]
    assert mixed[:2]==[1,2]
    assert mixed[2]==pytest.approx(.6)
    assert mixed[3]==pytest.approx(.4)
    # The physical opening remains; only the cut subset is stitched.
    assert result["source_open_grid_edges"]>0


def test_affine_projector_closes_partial_grid_cut_without_adding_grid_vertices():
    chart,uv,faces=mixed_chart()
    topo=discover_exact_seam_topology(chart,uv,faces)
    # Move right seam away from left; do NOT move the source UV Grid.
    xyz=np.c_[uv,np.zeros(len(uv))]
    projector=AffineSeamProjector(len(xyz),topo["affine_rows"])
    assert projector.residual(xyz)>.5
    solved=projector.project(xyz,strict=True)
    assert solved.shape==xyz.shape
    assert projector.residual(solved)<1e-10
    # t=0 is the cut/open junction inside LEFT edge 1->2;
    # the RIGHT start vertex 4 must coincide with that *edge interior*.
    left_junction=.6*solved[1]+.4*solved[2]
    np.testing.assert_allclose(left_junction,solved[4],atol=1e-10)
    np.testing.assert_allclose(solved[2],solved[7],atol=1e-10)


def test_smoothmax_with_mixed_seam_affine_constraint():
    chart,uv,faces=mixed_chart()
    topo=discover_exact_seam_topology(chart,uv,faces)
    reference=np.c_[uv,np.zeros(len(uv))]
    reference[2,0]+=.07
    solved,metrics=_augmented_lagrangian_planarize(
        reference,faces,
        square_weight=10.,
        anchor_weight=.1,
        outer_iterations=3,inner_iterations=70,
        restoration_sweeps=3,final_polish_sweeps=8,
        affine_rows=topo["affine_rows"],
        equality_groups=topo["groups"],
    )
    affine=AffineSeamProjector(len(solved),topo["affine_rows"])
    assert metrics["k3d_smoothmax_enabled"]
    assert metrics["k3d_al_source_seam_affine_constraint_count"]>=2
    assert affine.residual(solved)<=1e-10
    np.testing.assert_allclose(
        .6*solved[1]+.4*solved[2],solved[4],atol=1e-10
    )
    assert metrics["k3d_al_source_seam_affine_max_residual"]<=1e-10


def test_paper_t3d_preserves_both_partial_cut_seam_and_original_open_end():
    chart,uv,faces=mixed_chart()
    topo=discover_exact_seam_topology(chart,uv,faces)
    xyz=np.c_[uv,np.zeros(len(uv))]
    p=AffineSeamProjector(len(xyz),topo["affine_rows"])
    xyz=p.project(xyz,strict=True)
    obj=SimpleNamespace(
        vertices=xyz,faces=faces,stage="K3D",
        metrics={
            "seam_exact_vertex_groups":topo["groups"],
            "seam_exact_affine_rows":topo["affine_rows"],
            "paper_t3d_planarity_iterations":4,
        },
    )
    class Assembly:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    class Report:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    pipeline=SimpleNamespace(
        TileAssembly=Assembly,StageReport=Report,
        _tile_face_planarity_by_group=lambda *_:
            {"top":0.,"bottom":0.,"side":0.},
        _assembly_counts=lambda *_:{},
    )
    assembly,_=extrude_paper_face_planarity(obj,.1,"T3D",pipeline)
    assert p.residual(obj.vertices)<1e-9
    assert assembly.metrics["t3d_seam_affine_row_count"]>=2
    assert assembly.metrics["t3d_seam_affine_max_residual"]<1e-9
    top=assembly.vertices[:,:4]
    bot=assembly.vertices[:,4:]
    np.testing.assert_allclose(.6*top[0,1]+.4*top[0,2],top[1,0],atol=1e-9)
    np.testing.assert_allclose(.6*bot[0,1]+.4*bot[0,2],bot[1,0],atol=1e-9)
    assert not np.isnan(assembly.vertices).any()



@pytest.mark.parametrize("open_fraction",[
    .35554409,  # Bunny 425-426: 64.445591% physical cut
    .05102818,  # Bunny 772-815: 94.897182% physical cut
])
def test_actual_bunny_mixed_edge_cut_fractions(open_fraction):
    chart,uv,faces=mixed_chart()
    # Relocate the original *physical seam endpoint* without modifying
    # the 4-corner left M2D Grid edge. Both the UV source and the
    # right quad start at the new original seam endpoint.
    chart.surface_vertices_3d[[2,5],1]=open_fraction
    chart.uv_vertices_2d[[2,5,7],1]=open_fraction
    uv[[4,5],1]=open_fraction
    topo=discover_exact_seam_topology(chart,uv,faces)
    assert topo["status"]=="conforming"
    assert topo["mixed_seam_open_grid_edges"]==1
    mixed=topo["mixed_grid_edge_sample"][0]
    assert mixed[:2]==[1,2]
    assert mixed[2]==pytest.approx(1.-open_fraction,abs=1e-7)
    assert mixed[3]==pytest.approx(open_fraction,abs=1e-7)
    projector=AffineSeamProjector(8,topo["affine_rows"])
    pts=projector.project(np.c_[uv,np.zeros(len(uv))],strict=True)
    assert projector.residual(pts)<=1e-10
    np.testing.assert_allclose(
        (1.-open_fraction)*pts[1]+open_fraction*pts[2],
        pts[4],atol=1e-10
    )
