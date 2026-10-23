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
    assert "paper_reference_bff" not in src  # No initial-Omega method restriction
    assert "not in {" not in src  # Legacy initial Omega is accepted
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



def test_3d_split_replay_preserves_original_m3d_vertices():
    from types import SimpleNamespace
    from onestring_physics.reflatten_split_view import figure_reflatten_split_3d

    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,.1],[0.,1.,0.]])
    originals=vertices.copy()
    faces=np.array([[0,1,2,3]])
    m2d=SimpleNamespace(
        vertices=vertices[:,:2].copy(),faces=faces,
        metrics=dict(csf_split_criterion="reflatten_tile_area",
                     csf_split_initial_components=[dict(face_ids=[0],sigma=1.2)],
                     csf_split_step_analysis=[]),
    )
    state=SimpleNamespace(mesh_2d_initial=m2d,
        mesh_3d_initial=SimpleNamespace(vertices=vertices,faces=faces))
    fig,rows=figure_reflatten_split_3d(state,0)
    assert fig is not None and rows[0]["M2D panel count"]==1
    np.testing.assert_array_equal(vertices,originals)


@pytest.mark.parametrize("initial_method", [
    "paper_reference_bff", "rectangular_harmonic_legacy",
    "lscm_free_boundary", "bff",
])
def test_reflatten_dispatches_identically_for_paper_and_normal_omega(
        monkeypatch, initial_method):
    """Paper OFF must use the exact same BFF/area-span Split engine."""
    from types import SimpleNamespace
    from onestring_physics import paper_mesh_splitting as split
    from onestring_physics import paper_split_reflatten as module

    marker=object()
    seen=[]
    def fake_reflatten(vertices,faces,domain,params):
        seen.append((domain.parameterization.method,
                     params.paper_fidelity_mode))
        return marker
    monkeypatch.setattr(module,"split_mesh_reflatten",fake_reflatten)
    chart=SimpleNamespace(method=initial_method)
    domain=SimpleNamespace(parameterization=chart)
    for paper in (True,False):
        p=SimpleNamespace(paper_split_criterion="reflatten_tile_area",
                          paper_fidelity_mode=paper)
        assert split.split_mesh(None,None,domain,p) is marker
    assert seen==[(initial_method,True),(initial_method,False)]


def test_normal_ui_exposes_reflatten_option_and_keeps_legacy_default():
    root=Path(__file__).parents[1]
    app=(root/"app_runtime_base.py").read_text()
    # In Paper OFF + legacy Omega, keep 'fixed_uv' as the default (index=0),
    # but do not disable the reflatten alternative.
    selection=app.split('paper_split_criterion = st.selectbox(',1)[1].split(
        'paper_split_curvature_rings =',1)[0]
    assert 'index=1 if _reflatten_bff_path else 0' in selection
    assert 'disabled=not _reflatten_bff_path' not in selection
    assert '["fixed_uv", "reflatten_tile_area"]' in selection
    assert 'if paper_split_criterion == "reflatten_tile_area":' in app


def test_normal_reflatten_reports_actual_initial_omega_method():
    code=(Path(__file__).parents[1]/
          "src/onestring_physics/paper_split_reflatten.py").read_text()
    assert 'csf_split_initial_omega_method=source_method' in code
    assert 'csf_split_initial_omega_is_official_bff=' in code
    assert 'csf_split_normal_flow_enabled=' in code
    assert '"paper_fidelity_mode"' in code



def test_validation_forced_split_is_opt_in_and_does_not_fake_csf():
    """A flat chart already satisfies sigma=1: debug must still exercise a cut."""
    v, tri, uv = _grid_mesh(n=12)
    g = np.linspace(-0.8, 0.8, 7)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    quads = np.array([
        [i*7+j, (i+1)*7+j, (i+1)*7+j+1, i*7+j+1]
        for i in range(6) for j in range(6)
    ])
    bff_mock = lambda V, F: V[:, :2].copy()

    natural, natural_report = plan_reflatten_cuts(
        xy, quads, v, tri, uv, flatten=bff_mock)
    assert natural == []
    assert natural_report["initial_sigma"] == pytest.approx(1.)
    assert not natural_report["validation_cut_generated"]

    forced, report = plan_reflatten_cuts(
        xy, quads, v, tri, uv, flatten=bff_mock,
        force_one_cut=True, max_cuts=1)
    assert len(forced) == len(report["steps"]) == 1
    assert report["validation_force_one_cut_requested"]
    assert report["validation_cut_generated"]
    assert forced[0]["validation_forced"] is True
    assert report["steps"][0]["validation_forced"] is True
    assert report["initial_sigma"] == pytest.approx(1.)
    assert report["final_max_sigma"] == pytest.approx(1., abs=1e-6)
    assert report["satisfied"]


def test_debug_force_one_cut_is_exposed_to_regular_pipeline():
    root = Path(__file__).parents[1]
    app = (root/"app_runtime_base.py").read_text()
    core = (root/"src/onestring_physics/paper_split_reflatten.py").read_text()
    pipeline = (root/"src/onestring_physics/onestring_pipeline.py").read_text()
    assert 'paper_split_validation_force_one_cut: bool = False' in pipeline
    assert 'paper_split_validation_force_one_cut=paper_split_validation_force_one_cut' in app
    assert 'csf_split_validation_forced_count' in core
    assert 'force_one_cut=bool(getattr(params,"paper_split_validation_force_one_cut",False))' in core



def _example_area_split_grid(n=12, grid=8):
    v, tri, uv = _grid_mesh(n=n)
    g = np.linspace(-0.84, 0.84, grid+1)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    quads = np.asarray([
        [i*(grid+1)+j, (i+1)*(grid+1)+j,
         (i+1)*(grid+1)+j+1, i*(grid+1)+j+1]
        for i in range(grid) for j in range(grid)
    ])
    return v, tri, uv, xy, quads


def test_fixed_chart_off_uses_original_omega_without_calling_bff():
    """OFF scores the inherited UV; not a BFF call or a silent replacement UV."""
    v, tri, uv, xy, quads = _example_area_split_grid()
    def forbidden_bff(*_args):
        raise AssertionError("reBFF was called despite OFF mode")
    cuts, report = plan_reflatten_cuts(
        xy, quads, v, tri, uv,
        flatten=forbidden_bff, reflatten=False)
    assert cuts == []
    assert report["initial_sigma"] == pytest.approx(1.)
    assert report["final_max_sigma"] == pytest.approx(1.)
    assert report["reflatten_per_piece"] is False
    assert report["evaluation_uv"] == "original_omega_fixed_chart"


def test_fixed_chart_cut_uses_same_candidate_engine_and_real_area_ratio():
    """One patch has much larger true 3D area; UV and tiles remain unchanged."""
    v, tri, uv, xy, quads = _example_area_split_grid(n=24, grid=8)
    v[:, 2] = 4.0 * np.maximum(v[:, 0], 0.0)
    cuts, report = plan_reflatten_cuts(
        xy, quads, v, tri, uv, reflatten=False, max_cuts=1)
    assert report["initial_sigma"] > 2.
    assert len(cuts) == len(report["steps"]) == 1
    assert cuts[0]["axis"] in ("row", "col")
    assert report["steps"][0]["parent_sigma"] == pytest.approx(report["initial_sigma"])
    assert report["steps"][0]["validation_forced"] is False
    assert report["reflatten_per_piece"] is False


def test_fixed_chart_forced_cut_is_tagged_and_keeps_sigma_real():
    """Existing validation switch exercises the same topology with reBFF OFF."""
    v, tri, uv, xy, quads = _example_area_split_grid()
    cuts, report = plan_reflatten_cuts(
        xy, quads, v, tri, uv, reflatten=False, force_one_cut=True,
        max_cuts=1)
    assert len(cuts) == 1
    assert cuts[0]["validation_forced"] is True
    assert report["validation_cut_generated"] is True
    assert report["initial_sigma"] == pytest.approx(1.)
    assert report["final_max_sigma"] == pytest.approx(1., abs=1e-6)


def test_fixed_chart_dispatch_and_rebff_dispatch_share_area_cut_owner(monkeypatch):
    """Only criterion switches: no change to legacy fixed_uv."""
    from types import SimpleNamespace
    from onestring_physics import paper_mesh_splitting as splitting
    from onestring_physics import paper_split_reflatten as engine

    seen = []
    def fixed(*args):
        seen.append("fixed_chart")
        return 17
    def rebff(*args):
        seen.append("rebff")
        return 23
    monkeypatch.setattr(engine, "split_mesh_fixed_chart_area", fixed)
    monkeypatch.setattr(engine, "split_mesh_reflatten", rebff)
    domain = SimpleNamespace(parameterization=SimpleNamespace(method="lscm_free_boundary"))
    for criterion, expected in (("fixed_chart_tile_area",17),("reflatten_tile_area",23)):
        params = SimpleNamespace(paper_split_criterion=criterion)
        assert splitting.split_mesh(None,None,domain,params) == expected
    assert seen == ["fixed_chart","rebff"]


def test_fixed_chart_ui_toggle_and_result_metadata_are_wired():
    root = Path(__file__).parents[1]
    app = (root/"app_runtime_base.py").read_text()
    engine = (root/"src/onestring_physics/paper_split_reflatten.py").read_text()
    viz = (root/"src/onestring_physics/visualization.py").read_text()
    assert 'key="split_area_enable_rebff"' in app
    assert 'paper_split_criterion = "fixed_chart_tile_area"' in app
    assert 'csf_split_bff_evaluation_enabled=bool(reflatten)' in engine
    assert 'fixed_chart_area_cut=not bool(reflatten)' in engine
    assert "record.get('fixed_chart_area_cut', False)" in viz
