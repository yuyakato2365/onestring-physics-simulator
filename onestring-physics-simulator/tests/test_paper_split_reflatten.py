"""Behavioral and wiring checks for the opt-in reflatten-area Split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onestring_physics.paper_split_reflatten import (
    _clip, _face_components, lobe_tip_corner_positions,
    plan_reflatten_cuts, tile_area_ratios,
)


def _grid_mesh(n=24):
    x=np.linspace(-1,1,n+1)
    X,Y=np.meshgrid(x,x,indexing="ij")
    uv=np.stack([X.ravel(),Y.ravel()],1)
    V=np.column_stack([uv,np.zeros(len(uv))])
    faces=[]
    for i in range(n):
        for j in range(n):
            a=i*(n+1)+j;b=(i+1)*(n+1)+j
            c=b+1;d=a+1
            faces.extend([[a,b,c],[a,c,d]])
    return V,np.asarray(faces),uv


def test_surface_clip_preserves_total_area_and_topology():
    v,f,uv=_grid_mesh()
    vn,un,left,right=_clip(v,f,uv,0,.123)
    area=lambda ff:float(np.abs(np.cross(
        vn[ff[:,1]]-vn[ff[:,0]],vn[ff[:,2]]-vn[ff[:,0]])[:,2]).sum()/2.)
    assert area(left)+area(right)==pytest.approx(4.)
    assert area(left)==pytest.approx(2*1.123)
    assert len(_face_components(left))==len(_face_components(right))==1


def test_flat_tile_span_is_one():
    v,f,uv=_grid_mesh()
    quads=np.array([[[-.5,-.5],[0.,-.5],[0.,0.],[-.5,0.]]])
    np.testing.assert_allclose(tile_area_ratios((v,f,uv),quads,uv),[1.],rtol=1e-6)


def test_flat_m2d_needs_no_split():
    v,f,uv=_grid_mesh()
    g=np.linspace(-.9,.9,7)
    X,Y=np.meshgrid(g,g,indexing="ij")
    xy=np.stack([X.ravel(),Y.ravel()],1)
    faces=np.array([[i*7+j,(i+1)*7+j,(i+1)*7+j+1,i*7+j+1]
                   for i in range(6) for j in range(6)])
    cuts,r=plan_reflatten_cuts(xy,faces,v,f,uv,
         flatten=lambda V,F:V[:,:2].copy())
    assert not cuts and r["satisfied"]
    assert r["initial_sigma"]==pytest.approx(1.,rel=1e-6)


def test_lobe_tip_corners_find_four_lobes():
    t=np.linspace(0,2*np.pi,160,endpoint=False)
    r=1.+.35*np.cos(4*t)
    V=np.vstack([np.column_stack([r*np.cos(t),r*np.sin(t),np.zeros_like(t)]),
                 [[0.,0.,0.]]])
    F=np.array([[i,(i+1)%160,160] for i in range(160)])
    tips=lobe_tip_corner_positions(V,F,window=4)
    deg=np.sort(np.degrees(np.arctan2(tips[:,1],tips[:,0]))%360)
    np.testing.assert_allclose(deg,[0.,90.,180.,270.],atol=5.)


def test_registered_reflatten_criterion_does_not_replace_fixed_uv():
    from onestring_physics import paper_mesh_splitting as split
    from onestring_physics import paper_split_reflatten as module
    import inspect
    src=inspect.getsource(split.split_mesh)
    assert "paper_split_criterion" in src
    assert "paper_reference_bff" in src
    assert "split_mesh_reflatten" in src
    assert "return _split_mesh_single" in src


def test_ui_and_runtime_flags_reach_parameter_constructor():
    root=Path(__file__).parents[1]
    app=(root/"app_runtime_base.py").read_text()
    pipe=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    for name in ("paper_split_criterion","paper_split_curvature_rings",
                 "reference_grid_fit_rectangle"):
        assert name+"=" in app
        assert name in pipe
    assert '"lobe_tips"' in app
    assert 'if selection_mode == "lobe_tips":' in pipe


def test_existing_k3d_al_and_collision_split_wiring_unchanged():
    root=Path(__file__).parents[1]
    launcher=(root/"app_optcuts_core_20260916.py").read_text()
    app=(root/"app_runtime_base.py").read_text()
    pipe=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    assert "install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline)" in launcher
    assert "collision_split_enabled=collision_split_enabled" in app
    assert "if bool(getattr(active_params,'collision_split_enabled',False)):" in pipe


def test_free_bff_failures_are_never_silently_certified():
    v,f,uv=_grid_mesh(n=12)
    g=np.linspace(-.8,.8,5)
    X,Y=np.meshgrid(g,g,indexing="ij")
    xy=np.stack([X.ravel(),Y.ravel()],1)
    faces=np.array([[i*5+j,(i+1)*5+j,(i+1)*5+j+1,i*5+j+1]
                   for i in range(4) for j in range(4)])
    def broken(*args):
        raise RuntimeError("official BFF failed")
    with pytest.raises(RuntimeError,match="official BFF failed"):
        plan_reflatten_cuts(xy,faces,v,f,uv,flatten=broken)



def test_reflatten_seams_do_not_drop_later_collision_driven_seams(monkeypatch):
    """When full-pipeline collision feedback rebuilds M2D, both cut classes survive."""
    from types import SimpleNamespace
    from onestring_physics import paper_mesh_splitting as split
    from onestring_physics import paper_split_reflatten as module
    from onestring_physics import reference_bff as bff

    observed = {}
    original=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    faces=np.array([[0,1,2,3]])
    tri=np.array([[0,1,2],[0,2,3]])

    monkeypatch.setattr(module, "plan_reflatten_cuts",
        lambda *a,**kw: (
          [dict(axis="col",value=.5,source_face_id=0)],
          dict(initial_sigma=2.1,final_max_sigma=1.8,satisfied=True,
               steps=[dict(parent_face_ids=[0],child_face_ids=[[0],[0]],
                           parent_sigma=2.1,child_sigma=[1.8,1.7],
                           curvature_vertex=1,curvature=.9,candidates=[])],
               leaves=[dict(satisfied=True,sigma=1.8)],
          ),
        ))
    monkeypatch.setattr(split,"chart_geometry",lambda p:(np.column_stack([original,np.zeros(4)]),tri,None))
    monkeypatch.setattr(bff,"run_official_bff",lambda *a,**kw:None)

    def fake_split(v,f,d,p):
        observed["forced"]=list(p.collision_split_forced_lines)
        observed["budget"]=d.max_csf_splits
        return v.copy(),f.copy(),[
          dict(axis="col",snapped_value=.5,duplicated_vertices=2),
          dict(axis="row",snapped_value=.75,duplicated_vertices=3,
               collision_driven=True)
        ],dict(split_boundary_pairs=[])
    monkeypatch.setattr(split,"_split_mesh_single",fake_split)
    p=SimpleNamespace(
        bff_executable=None,bff_timeout_seconds=30,
        paper_split_curvature_rings=3,
        collision_split_forced_lines=(dict(axis="row",value=.75,source_face_id=0),)
    )
    chart=SimpleNamespace(uv_vertices_2d=original,metrics={})
    domain=SimpleNamespace(parameterization=chart,
        reference_grid_rotation_degrees=0,max_csf_splits=12)
    v,f,records,metrics=module.split_mesh_reflatten(original,faces,domain,p)
    assert observed["budget"]==0
    assert [c["axis"] for c in observed["forced"]]==["col","row"]
    assert domain.max_csf_splits==12
    assert p.collision_split_forced_lines==(
        dict(axis="row",value=.75,source_face_id=0),)
    assert metrics["csf_split_step_count"]==1
    assert metrics["collision_split_count"]==1
    assert len(metrics["csf_split_step_analysis"])==1
    assert len(metrics["collision_split_step_analysis"])==1
    assert metrics["csf_split_duplicated_vertex_count"]==2
    assert metrics["collision_split_duplicated_vertex_count"]==3


def test_figure5_css_stays_with_every_stage_render():
    from onestring_physics import paper_ui_20260916_patch as ui
    html=ui._fig(b"\x89PNG placeholder","M2D",.5)
    assert html.startswith("<style>")
    assert "class='os-hi'" in html
    assert ".os-hi{position:absolute" in html
