"""Native UV source-seam pairing, hard K3D and Paper T3D closure regression."""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_seam_exact_closure import (
    SeamTopologyMismatch,
    discover_exact_seam_topology,
    enforce_shared_positions,
    fit_k3d_with_exact_pairs,
    max_seam_gap,
)
from onestring_physics.paper_extrusion import (
    _solve,extrude_paper_face_planarity
)


def original_seam():
    # Physical source is a continuous 2x1 planar rectangle.
    # Its middle x=1 source edge has two disconnected UV copies, x=1 and
    # x=2 respectively. Thus the source seam is genuine, not proximity
    # of unrelated K3D panels.
    xyz=np.array([
        [0,0,0],[1,0,0],[2,0,0],
        [0,1,0],[1,1,0],[2,1,0],
    ],float)
    sf=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4]],int)
    uv=np.array([
        [0.,0.],[1.,0.],[1.,1.],[0.,1.],
        [2.,0.],[3.,0.],[3.,1.],[2.,1.]
    ])
    uf=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]],int)
    return SimpleNamespace(
        surface_vertices_3d=xyz,surface_faces=sf,
        uv_vertices_2d=uv,uv_faces=uf
    )


def conforming():
    chart=original_seam()
    faces=np.array([[0,1,2,3],[4,5,6,7]],int)
    return chart,chart.uv_vertices_2d.copy(),faces


def test_finds_exact_3d_seam_correspondence_from_original_surface_edges():
    chart,uv,faces=conforming()
    result=discover_exact_seam_topology(chart,uv,faces)
    assert result["status"]=="conforming"
    assert result["unmatched_edges"]==0
    assert result["seam_grid_edges"]==2
    assert len(result["paired_edges"])==1
    assert {frozenset(g) for g in result["groups"]}=={
        frozenset([1,4]),frozenset([2,7])
    }


def test_rejects_different_uv_boundary_samplings_not_nearest_weld():
    chart,uv,faces=conforming()
    # Right hand seam subdivided at y=0.5, whereas the left uses one edge.
    uv=np.array([
        [0,0],[1,0],[1,1],[0,1],
        [2,0],[3,0],[3,.5],[2,.5],[3,1],[2,1]
    ],float)
    faces=np.array([[0,1,2,3],[4,5,6,7],[7,6,8,9]],int)
    result=discover_exact_seam_topology(chart,uv,faces)
    assert result["status"]=="nonconforming"
    assert result["seam_grid_edges"]==3
    assert result["unmatched_edges"]==3
    assert len(result["paired_edges"])==0


def test_exact_k3d_group_closure_survives_consensus_planarization():
    chart,uv,faces=conforming()
    topo=discover_exact_seam_topology(chart,uv,faces)
    vertices=np.array([
        [0,0,0],[1.02,0,.03],[1.02,1,.04],[0,1,0],
        [.98,0,-.02],[2,0,0],[2,1,.02],[.98,1,-.04]
    ],float)
    assert max_seam_gap(vertices,topo["groups"])>.01
    solved,metrics=fit_k3d_with_exact_pairs(
        vertices,faces,topo["groups"],iterations=8
    )
    assert max_seam_gap(solved,topo["groups"])<=1e-10
    assert metrics["seam_k3d_gap_after"]<=1e-10
    np.testing.assert_allclose(solved[1],solved[4],atol=1e-10)
    np.testing.assert_allclose(solved[2],solved[7],atol=1e-10)


def test_paper_eq2_solver_hard_top_and_bottom_equality():
    source=np.array([
        [0.,0,0],[.4,0,0],[.9,0,0],
        [0.,0,1],[.42,0,1],[.88,0,1]
    ])
    rows=[([0,1,2],source[:3]+.1,1.),
          ([3,4,5],source[3:]+.2,1.)]
    out=_solve(len(source),rows,source,1.,
               equality_groups=[[0,1],[3,4]])
    np.testing.assert_array_equal(out[0],out[1])
    np.testing.assert_array_equal(out[3],out[4])


def test_paper_extrusion_preserves_closed_seam_on_top_and_bottom():
    chart,uv,faces=conforming()
    topo=discover_exact_seam_topology(chart,uv,faces)
    xyz=np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [1,0,0],[2,0,0],[2,1,0],[1,1,0]
    ],float)
    mesh=SimpleNamespace(vertices=xyz.copy(),faces=faces,stage="K3D",
        metrics={"seam_exact_vertex_groups":topo["groups"],
                 "k3d_smoothmax_enabled":True,
                 "k3d_smoothmax_weight":10.,
                 "paper_t3d_planarity_iterations":5})
    # Actual upstream helpers are omitted, but the shared top/bottom solve
    # and output geometry must still follow real K3D+seam equality groups.
    class Assembly:
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs)
    class Report:
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs)
    pipeline=SimpleNamespace(
        TileAssembly=Assembly, StageReport=Report,
        _tile_face_planarity_by_group=lambda _tiles:
            {"top":0.,"bottom":0.,"side":0.},
        _assembly_counts=lambda _assembly:{},
    )
    result,_=extrude_paper_face_planarity(
        mesh,.1,"T3D",pipeline
    )
    assert result.metrics["t3d_seam_exact_equality_group_count"]==2
    assert result.metrics["t3d_final_k3d_smoothmax_audited"] is True
    assert np.isfinite(
        result.metrics["t3d_final_k3d_smoothmax_energy_after"]
    )
    assert result.metrics["t3d_final_k3d_smoothmax_energy_before"]>=0.
    assert result.metrics["t3d_seam_top_gap_max"]==0.
    assert result.metrics["t3d_seam_bottom_gap_max"]==0.
    np.testing.assert_array_equal(
        result.vertices[0,1],result.vertices[1,0])
    np.testing.assert_array_equal(
        result.vertices[0,2],result.vertices[1,3])
    np.testing.assert_array_equal(
        result.vertices[0,5],result.vertices[1,4])
    np.testing.assert_array_equal(
        result.vertices[0,6],result.vertices[1,7])
    np.testing.assert_array_equal(mesh.vertices[1],mesh.vertices[4])
    np.testing.assert_array_equal(mesh.vertices[2],mesh.vertices[7])



def test_launcher_seam_hook_routes_m2d_uv_to_k3d_and_retains_exact_groups():
    from onestring_physics.optcuts_seam_exact_closure import (
        install_optcuts_exact_seam_closure,
    )
    chart,uv,faces=conforming()
    chart.method="optcuts_test"
    chart.metrics={"optcuts_area_grid_boundary_enabled":True}
    original_xyz=np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [1,0,0],[2,0,0],[2,1,0],[1,1,0],
    ],float)
    def source_lift(target,mesh,parameterization,params):
        return SimpleNamespace(
            stage="M3D",vertices=original_xyz.copy(),
            faces=faces.copy(),grid=mesh.grid,metrics={}),None
    def source_k3d(target,mesh,parameterization,params):
        geometry=mesh.vertices.copy()
        geometry[1]+=[.015,0,.01]
        geometry[4]-=[.008,0,.015]
        out=SimpleNamespace(
            stage="K3D",vertices=geometry,faces=mesh.faces.copy(),
            metrics={},grid=mesh.grid)
        return out,None
    owner=SimpleNamespace(
        _lift_m2d_to_m3d=source_lift,
        _optimize_k3d=source_k3d,
    )
    install_optcuts_exact_seam_closure(owner)
    m2d=SimpleNamespace(
        stage="M2D",vertices=np.c_[uv,np.zeros(len(uv))],
        faces=faces.copy(),grid=SimpleNamespace(tile_size=1.),
    )
    m3d,_=owner._lift_m2d_to_m3d(None,m2d,chart,None)
    k3d,_=owner._optimize_k3d(None,m3d,chart,None)
    assert k3d.metrics["seam_exact_closure_enabled"]
    assert k3d.metrics["seam_exact_paired_grid_edges"]==1
    assert k3d.metrics["seam_k3d_gap_after"]==pytest.approx(0.)
    np.testing.assert_allclose(k3d.vertices[1],k3d.vertices[4],atol=1e-10)


def test_launcher_rejects_nonconforming_real_seam_not_claiming_exact_fit():
    from onestring_physics.optcuts_seam_exact_closure import (
        install_optcuts_exact_seam_closure,
    )
    chart,_,_=conforming()
    chart.method="optcuts_test"
    chart.metrics={"optcuts_area_grid_boundary_enabled":True}
    uv=np.array([
        [0,0],[1,0],[1,1],[0,1],
        [2,0],[3,0],[3,.5],[2,.5],[3,1],[2,1]
    ],float)
    faces=np.array([[0,1,2,3],[4,5,6,7],[7,6,8,9]],int)
    def source_lift(_target,mesh,_param,_params):
        return SimpleNamespace(
            stage="M3D",vertices=np.c_[mesh.vertices[:,:2],np.zeros(len(uv))],
            faces=mesh.faces.copy(),metrics={}),None
    def source_opt(_target,mesh,_param,_params):
        return SimpleNamespace(
            stage="K3D",vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),metrics={}),None
    owner=SimpleNamespace(
        _lift_m2d_to_m3d=source_lift,_optimize_k3d=source_opt)
    install_optcuts_exact_seam_closure(owner)
    mesh=SimpleNamespace(vertices=np.c_[uv,np.zeros(len(uv))],faces=faces)
    lifted,_=owner._lift_m2d_to_m3d(None,mesh,chart,None)
    with pytest.raises(SeamTopologyMismatch,match="SEAM_K3D_NONCONFORMING_GRID_EDGE_SAMPLING"):
        owner._optimize_k3d(None,lifted,chart,None)



def test_topological_uv_seam_detected_even_when_two_copies_touch_in_uv():
    """Seam identity depends on paired source incidence, not UV distance."""
    from onestring_physics.optcuts_seam_exact_closure import (
        _cut_uv_segments
    )
    chart,_,_=conforming()
    # Distinct UV vertex IDs encode the cut, even when its two border
    # realizations numerically meet. The chart itself would overlap if
    # accepted without reparameterization; this tests ONLY classification.
    chart.uv_vertices_2d[4,0]=1.
    chart.uv_vertices_2d[7,0]=1.
    segments,open_lines=_cut_uv_segments(chart,include_open=True)
    assert len(segments)==2
    assert len(open_lines)==6
    paired=[s for s in segments if s[2]==(1,4)]
    assert len(paired)==2


def test_unclassified_grid_edge_reports_real_source_boundary_coverage():
    chart,uv,faces=conforming()
    # Reproduce the category of fault from Bunny: one 2D Grid edge is no
    # longer precisely on the chart's UV source boundary.
    bad_uv=uv.copy()
    bad_uv[1,0]-=.15
    out=discover_exact_seam_topology(chart,bad_uv,faces)
    assert out["status"]=="nonconforming"
    assert out["unclassified_boundary_edges"]>0
    sample=out["unclassified_boundary_diagnostics"]
    assert any(v["all_source_boundary_coverage"]<1.
               for v in sample)
    assert all("uv_a" in v and "uv_b" in v and
               "nearest_cut_distances" in v and
               "likely_reason" in v for v in sample)
    assert any(v["likely_reason"]==
               "M2D_grid_edge_not_on_actual_omega_boundary"
               for v in sample)


def test_grid_edge_requires_full_source_cut_coverage_not_endpoints_only():
    from onestring_physics.optcuts_seam_exact_closure import (
        _covered_fraction_on_source_boundary
    )
    a=np.array([0.,0.])
    b=np.array([1.,0.])
    # The cut occupies only the endpoints' immediate neighborhoods;
    # a straight quad edge cannot legally bridge the intervening gap.
    lines=[np.array([[0.,0.],[.25,0.]]),
           np.array([[.75,0.],[1.,0.]])]
    assert _covered_fraction_on_source_boundary(a,b,lines,1e-10)==pytest.approx(.5)
    assert _covered_fraction_on_source_boundary(
        a,b,lines+[np.array([[.25,0.],[.75,0.]])],1e-10
    )==pytest.approx(1.)
    assert _covered_fraction_on_source_boundary(
        a,b,[np.array([[0.,.1],[1.,.1]])],1e-10
    )==pytest.approx(0.)
