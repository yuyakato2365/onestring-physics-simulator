"""Area-threshold Omega grid boundary -> interior remap regression tests."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from onestring_physics.optcuts_area_grid_boundary import (
    GridBoundaryReparameterizationError, _triangle_rectangle_area,
    _intersection_ratios, _grid_boundary, _boundary_target_map,
    construct_grid_target, apply_area_grid_boundary,
    fix_domain_lattice, keep_cells_on_fixed_lattice,
)


def test_triangle_rectangle_intersection_is_exact():
    tri=np.array([[0.,0.],[1.,0.],[0.,1.]])
    assert _triangle_rectangle_area(tri,np.array([0.,0.]),np.array([1.,1.]))==pytest.approx(.5)
    assert _triangle_rectangle_area(tri,np.array([0.,0.]),np.array([.5,.5]))==pytest.approx(.25)
    assert _triangle_rectangle_area(tri,np.array([1.,1.]),np.array([2.,2.]))==pytest.approx(0.)
    area=_intersection_ratios(tri,np.array([[0,1,2]]),np.zeros(2),
                              np.array([.5,.5]),2,2)
    np.testing.assert_allclose(area,np.array([[1.,.5],[.5,0.]]))


def test_50_percent_rule_includes_half_crossed_cell_but_not_outside():
    uv=np.array([[0.,0.],[2.,0.],[0.,2.]])
    tri=np.array([[0,1,2]])
    tgt=construct_grid_target(uv,tri,[0,1,2],nx=2,ny=2,threshold=.5)
    assert tgt["mask"].shape==(4,4)
    assert tgt["mask"][0,0]
    assert tgt["mask"][1,1]
    assert not tgt["mask"][3,3]
    assert tgt["coverage"][1,2]==pytest.approx(.5)
    assert np.any(tgt["mask"] & (tgt["coverage"]<.999))


def test_selected_cell_boundary_must_have_disk_topology():
    with pytest.raises(GridBoundaryReparameterizationError,match="MULTI_COMPONENT_OR_HOLE"):
        _grid_boundary(np.array([[1,0,1]],bool))
    with pytest.raises(GridBoundaryReparameterizationError,match="MULTI_COMPONENT_OR_HOLE"):
        _grid_boundary(np.array([[1,1,1],[1,0,1],[1,1,1]],bool))


def test_ordered_boundary_map_matches_all_grid_corners():
    # Four UV boundary corners and a matching four-corner grid.
    uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    target=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    mapped=_boundary_target_map(uv,[0,1,2,3],target)
    np.testing.assert_allclose(mapped,target)
    with pytest.raises(GridBoundaryReparameterizationError,match="NOT_ENOUGH_VERTICES"):
        _boundary_target_map(uv,[0,1,2],target)


def _grid_chart(n=8):
    xs=np.linspace(0.,1.,n+1)
    xx,yy=np.meshgrid(xs,xs,indexing="ij")
    uv=np.stack([xx.ravel(),yy.ravel()],axis=1)
    faces=[]
    for i in range(n):
        for j in range(n):
            a=i*(n+1)+j;b=(i+1)*(n+1)+j
            faces.extend([[a,b,b+1],[a,b+1,a+1]])
    faces=np.asarray(faces,int)
    boundary=[]
    for i in range(n):
        boundary.append(i*(n+1))
    for j in range(n):
        boundary.append(n*(n+1)+j)
    for i in range(n,0,-1):
        boundary.append(i*(n+1)+n)
    for j in range(n,0,-1):
        boundary.append(j)
    # This list traces bottom -> right -> top -> left CCW.
    return uv,faces,boundary


def test_square_boundary_reparameterization_preserves_chart_and_grid():
    uv,faces,boundary=_grid_chart()
    p=SimpleNamespace(
        uv_vertices_2d=uv.copy(),
        uv_faces=faces,
        surface_vertices_3d=np.c_[uv,np.zeros(len(uv))],
        surface_faces=faces.copy(),
        omega_boundary=uv[np.r_[boundary,boundary[0]]].copy(),
        triangle_acceleration="old-uv-index",
        metrics={},
    )
    params=SimpleNamespace(optcuts_area_grid_coverage_threshold=.5,omega_overlay_margin=1)
    grid=SimpleNamespace(nx=4,ny=4)
    out=apply_area_grid_boundary(p,grid,params)
    assert out.metrics["optcuts_area_grid_boundary_status"]=="accepted_exact_grid_outline"
    assert out.metrics["optcuts_area_grid_cell_count"]==64
    assert out.triangle_acceleration is None
    assert out._optcuts_area_grid_mask.shape==(8,8)
    assert out.metrics["optcuts_area_grid_original_mean_sd"]==pytest.approx(
        out.metrics["optcuts_area_grid_final_mean_sd"],abs=.05)
    domain=SimpleNamespace()
    domain=fix_domain_lattice(domain,out)
    domain.parameterization=out
    f=np.array([[0,1,10,9]],int)
    fake=SimpleNamespace(
        vertices=np.c_[domain.uv_vertices,np.zeros(len(domain.uv_vertices))],
        faces=np.asarray([[
          j*(domain.overlay_nx+1)+i,
          j*(domain.overlay_nx+1)+i+1,
          (j+1)*(domain.overlay_nx+1)+i+1,
          (j+1)*(domain.overlay_nx+1)+i,
        ] for j in range(domain.overlay_ny)
          for i in range(domain.overlay_nx)],int),
    )
    keep=keep_cells_on_fixed_lattice(fake,domain)
    assert keep.sum()==64


def test_mode_is_isolated_and_threshold_wired_to_ui_and_cache_key():
    root=Path(__file__).parents[1]
    selector=(root/"app_optcuts_core_20260916.py").read_text()
    ui=(root/"app_runtime_base.py").read_text()
    pipe=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    carrier=(root/"src/onestring_physics/optcuts_test_boundary_reparameterization_patch.py").read_text()
    assert '"optcuts_area_grid_boundary"' in selector
    assert 'value in {"optcuts_test2","optcuts_grid_aware",' in selector
    assert "GridセルのΩ内面積割合・採用閾値" in ui
    assert "value=0.50" in ui
    assert "optcuts_area_grid_coverage_threshold" in ui
    assert "optcuts_area_grid_coverage_threshold: float = 0.50" in pipe
    assert "area_boundary_domain" in carrier
    assert "keep_cells_on_fixed_lattice" in carrier
    assert "apply_area_grid_boundary" in carrier
