"""Independent tests of reBFF manufacturing M2D geometry (opt-in only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onestring_physics.paper_split_reflatten import plan_reflatten_cuts
from onestring_physics.final_split_panel_pass import _complete_cut_once
from onestring_physics.postcut_rebff_m2d import (
    apply_leaf_rebff_uv, _quad_signed_area,
)


def _fixture(n=12, grid=6):
    x = np.linspace(-1., 1., n+1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    source_uv = np.stack((xx.ravel(), yy.ravel()), axis=1)
    surface_xyz = np.column_stack((source_uv, np.zeros(len(source_uv))))
    tri = []
    for i in range(n):
        for j in range(n):
            a = i*(n+1)+j
            b = a+n+1
            tri.extend(((a,b,b+1),(a,b+1,a+1)))
    g = np.linspace(-0.8,0.8,grid+1)
    gx,gy=np.meshgrid(g,g,indexing="ij")
    grid_uv=np.stack((gx.ravel(),gy.ravel()),axis=1)
    quad=np.asarray([
        [i*(grid+1)+j,(i+1)*(grid+1)+j,
         (i+1)*(grid+1)+j+1,i*(grid+1)+j+1]
        for i in range(grid) for j in range(grid)
    ],dtype=int)
    return surface_xyz,np.asarray(tri,int),source_uv,grid_uv,quad


def _smooth_nonlinear_flatten(V,F):
    # A legitimate per-piece UV change: horizontal shift varies with height.
    # It preserves positive orientation and has no need for native BFF in CI.
    return np.column_stack([V[:,0]+0.09*V[:,1]**2, V[:,1]])


def test_postcut_rebff_projects_actual_m2d_without_changing_cut_decision():
    xyz,tri,uv,grid_uv,quads=_fixture()
    cuts_old,report_old=plan_reflatten_cuts(
        grid_uv,quads,xyz,tri,uv,
        flatten=_smooth_nonlinear_flatten,max_cuts=1,force_one_cut=True,
    )
    chart_leaves=[]
    cuts_new,report_new=plan_reflatten_cuts(
        grid_uv,quads,xyz,tri,uv,
        flatten=_smooth_nonlinear_flatten,max_cuts=1,force_one_cut=True,
        leaf_chart_sink=chart_leaves,
    )
    assert cuts_old==cuts_new
    assert report_old["initial_sigma"]==pytest.approx(report_new["initial_sigma"])
    assert report_old["final_max_sigma"]==pytest.approx(report_new["final_max_sigma"])
    assert report_old["steps"]==report_new["steps"]
    assert len(cuts_new)==1
    assert len(chart_leaves)==2

    new_source,new_faces,cut_record=_complete_cut_once(
        grid_uv,quads,cuts_new[0]["axis"],cuts_new[0]["value"])
    assert cut_record is not None
    original_uv=new_source.copy()
    mapped,metrics=apply_leaf_rebff_uv(
        new_source,new_faces,chart_leaves,
    )
    assert metrics["m2d_postcut_rebff_applied"]
    assert metrics["m2d_postcut_rebff_leaf_count"]==2
    assert np.max(np.linalg.norm(mapped[:,:2]-original_uv[:,:2],axis=1))>1e-4
    assert np.all(_quad_signed_area(mapped,new_faces)*
                  _quad_signed_area(original_uv,new_faces)>0)
    # The input M2D source UV survives for the inverse M3D mapping.
    np.testing.assert_allclose(new_source,original_uv)


def test_postcut_rebff_refuses_missing_leaf_coverage():
    xyz,tri,uv,grid_uv,quads=_fixture()
    with pytest.raises(RuntimeError,match="POSTCUT_REBFF_M2D_NO_LEAVES"):
        apply_leaf_rebff_uv(grid_uv,quads,[])


def test_postcut_rebff_refuses_collapsed_uv():
    xyz,tri,uv,grid_uv,quads=_fixture(grid=1)
    leaf=dict(face_ids=[0],source_uv=grid_uv,
              rebff_uv=np.zeros_like(grid_uv),
              piece_faces=np.array([[0,1,2],[0,2,3]],int))
    with pytest.raises(ValueError,match="collapsed reBFF UV"):
        apply_leaf_rebff_uv(grid_uv,quads,[leaf])


def test_m2d_rebff_checkbox_pipeline_and_both_cut_owners_wired():
    root=Path(__file__).parents[1]
    app=(root/"app_runtime_base.py").read_text()
    params=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    planner=(root/"src/onestring_physics/paper_split_reflatten.py").read_text()
    simple=(root/"src/onestring_physics/simple_split_panel_patch.py").read_text()
    final=(root/"src/onestring_physics/final_split_panel_pass.py").read_text()
    assert 'key="split_apply_rebff_to_m2d"' in app
    assert 'paper_split_apply_rebff_m2d=paper_split_apply_rebff_m2d' in app
    assert 'paper_split_apply_rebff_m2d: bool = False' in params
    assert 'leaf_chart_sink=leaf_charts' in planner
    assert "domain._split_rebff_leaf_charts=" in planner
    assert 'apply_leaf_rebff_uv(' in simple
    assert 'apply_leaf_rebff_uv(' in final
    assert 'out._split_panel_source_vertices = canonical' in simple
    assert 'setattr(out, "_split_panel_source_vertices", canonical.copy())' in final
