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



def test_new_mode_runs_parameterization_domain_and_m2d_as_one_fixed_lattice():
    """Integration without the unavailable native OptCuts CLI.

    Exercise the real optcuts_test wrapper: source Omega -> area threshold
    -> deformed chart -> frozen domain lattice -> exact selected M2D tiles.
    """
    from dataclasses import dataclass
    from onestring_physics.optcuts_test_boundary_reparameterization_patch import (
        install_optcuts_test_boundary_reparameterization_patch,
    )

    uv,tri,loop=_grid_chart()
    grid=SimpleNamespace(nx=4,ny=4)

    @dataclass
    class Params:
        omega_parameterization_mode: str = "optcuts_test"
        optcuts_try_alternative_initial_cut: bool = True
        optcuts_area_grid_boundary_enabled: bool = True
        optcuts_area_grid_coverage_threshold: float = .5
        omega_overlay_margin: int = 1

    def source_chart(*_):
        return SimpleNamespace(
            _onestring_optcuts_candidates=[SimpleNamespace(
                surface_vertices_3d=np.c_[uv,np.zeros(len(uv))],
                surface_faces=tri.copy(), uv_vertices_2d=uv.copy(),
                uv_faces=tri.copy(), metrics={},
            )],
            _onestring_optcuts_candidate_qualities=[{
                "distortion_area_mean": 2., "seam_length": 0.,
                "csf_over_2_area_fraction": 0.,
                "flipped_triangles": 0, "degenerate_triangles": 0,
                "injectivity_overlap_pairs": 0,
            }],
            method="optcuts_official",
            uv_vertices_2d=uv.copy(),
            uv_faces=tri.copy(),
            surface_vertices_3d=np.c_[uv,np.zeros(len(uv))],
            surface_faces=tri.copy(),
            omega_boundary=uv[np.r_[loop,loop[0]]].copy(),
            triangle_acceleration=None,
            metrics={},
        )

    def flatten_domain(chart,*_):
        # The actual free-boundary overlay in the dated app is 2*nx by
        # 2*ny, with an extra cell of margin on ALL four sides.
        # The same lattice is used for selection and final M2D.
        n=2*grid.nx+2;m=2*grid.ny+2
        origin=np.asarray(chart.omega_boundary,float).min(axis=0)[:2]
        extent=np.ptp(np.asarray(chart.omega_boundary,float),axis=0)
        step=np.maximum(extent/np.array([2*grid.nx,2*grid.ny]),1e-8)
        x=origin[0]+(np.arange(n+1)-1)*step[0]
        y=origin[1]+(np.arange(m+1)-1)*step[1]
        xx,yy=np.meshgrid(x,y,indexing="xy")
        return SimpleNamespace(
            boundary=chart.omega_boundary.copy(),
            parameterization=chart,uv_vertices=np.stack([xx,yy],axis=-1).reshape(-1,2),
            overlay_nx=n,overlay_ny=m,overlay_margin_tiles=1,
            overlay_step_u=step[0],overlay_step_v=step[1],
            split_lines=[],
        )

    def m2d_source(_grid,domain,*_):
        verts=np.c_[domain.uv_vertices,np.zeros(len(domain.uv_vertices))]
        n,m=domain.overlay_nx,domain.overlay_ny
        faces=np.asarray([
            [j*(n+1)+i,j*(n+1)+i+1,
             (j+1)*(n+1)+i+1,(j+1)*(n+1)+i]
            for j in range(m) for i in range(n)],int)
        return SimpleNamespace(vertices=verts,faces=faces,
                               grid=grid,stage="M2D",metrics={},split_lines=[])

    owner=SimpleNamespace(
        _build_surface_parameterization=source_chart,
        _flatten_to_domain=flatten_domain,
        _build_m2d=m2d_source,
    )
    install_optcuts_test_boundary_reparameterization_patch(owner)
    params=Params()
    chart=owner._build_surface_parameterization(None,None,grid,params)
    # Fitting must NOT occur before real M2D lattice is available.
    assert len(chart._onestring_optcuts_candidates)==1
    assert not chart.metrics.get("optcuts_area_grid_boundary_enabled",False)
    domain=owner._flatten_to_domain(chart,grid,params)
    assert chart._onestring_optcuts_candidates==[]
    assert chart.metrics["optcuts_area_grid_multistart_downstream_swap_enabled"] is False
    assert chart.metrics["optcuts_area_grid_lattice_source"]=="real_production_m2d_overlay"
    assert domain.overlay_nx==10 and domain.overlay_ny==10
    m2d=owner._build_m2d(grid,domain,params)
    assert len(m2d.faces)==64
    assert m2d.metrics["paper_m2d_candidate_face_count"]==100
    assert m2d.metrics["optcuts_area_grid_boundary_applied_to_m2d"]
    assert m2d.metrics["optcuts_area_grid_coverage_threshold"]==.5
    assert chart.metrics["optcuts_area_grid_boundary_status"]=="accepted_exact_grid_outline"
    np.testing.assert_allclose(domain.boundary,chart.omega_boundary)
    assert int(np.sum(domain._optcuts_area_grid_mask))==len(m2d.faces)

    # The previously successful optcuts_test2 carrier stays on its old path
    # when the new mode is not selected.
    original=owner._build_surface_parameterization(None,None,grid,
        Params(optcuts_area_grid_boundary_enabled=False))
    assert not original.metrics.get("optcuts_area_grid_boundary_enabled",False)
    assert original.metrics["optcuts_test_grid_outline_reparameterization"] is False
    assert len(original._onestring_optcuts_candidates)==1



def test_actual_m2d_lattice_phase_is_preserved_not_invented():
    from onestring_physics.optcuts_area_grid_boundary import (
        _actual_overlay_lattice,
    )
    uv,tri,loop=_grid_chart()
    origin=np.array([-.0625,-.1875])
    h=np.array([.125,.125])
    nx=10;ny=10
    x=origin[0]+np.arange(nx+1)*h[0]
    y=origin[1]+np.arange(ny+1)*h[1]
    xx,yy=np.meshgrid(x,y,indexing="xy")
    domain=SimpleNamespace(
        overlay_nx=nx,overlay_ny=ny,overlay_margin_tiles=1,
        uv_vertices=np.stack([xx,yy],axis=-1).reshape(-1,2),
    )
    target=construct_grid_target(
        uv,tri,loop,nx=4,ny=4,threshold=.5,domain=domain,
    )
    assert target["grid_source"]=="real_production_m2d_overlay"
    np.testing.assert_allclose(target["origin"],origin)
    np.testing.assert_allclose(target["step"],h)
    assert target["mask"].shape==(ny,nx)
    assert target["mask"][0,0] is np.False_ or not bool(target["mask"][0,0])
    assert np.any(target["mask"])
    invalid=SimpleNamespace(
        overlay_nx=nx,overlay_ny=ny,overlay_margin_tiles=1,
        uv_vertices=domain.uv_vertices.copy(),
    )
    invalid.uv_vertices[5,0]+=.03
    with pytest.raises(GridBoundaryReparameterizationError,match="LATTICE_MISMATCH"):
        _actual_overlay_lattice(invalid)


def test_subdivision_adds_paired_3d_uv_boundary_vertices_with_same_geometry():
    from onestring_physics.optcuts_area_grid_boundary import (
        _ensure_boundary_vertices,
    )
    from onestring_physics.optcuts_backend import _boundary_loops
    uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    faces=np.array([[0,1,2],[0,2,3]],int)
    p=SimpleNamespace(
        uv_vertices_2d=uv.copy(),
        uv_faces=faces.copy(),
        surface_vertices_3d=np.c_[uv,np.zeros(4)],
        surface_faces=faces.copy(),
    )
    loop=_boundary_loops(faces)[0]
    new_loop,inserted=_ensure_boundary_vertices(p,12,loop)
    assert inserted==8
    assert len(new_loop)==12
    assert len(p.uv_vertices_2d)==12
    assert len(p.surface_vertices_3d)==12
    assert len(p.uv_faces)==len(p.surface_faces)==10
    for triangle_uv,triangle_xyz in zip(p.uv_faces,p.surface_faces):
        np.testing.assert_allclose(
            p.uv_vertices_2d[triangle_uv],
            p.surface_vertices_3d[triangle_xyz,:2],
        )
    assert len(_boundary_loops(p.uv_faces))==1


def test_area_mode_m2d_does_not_call_prior_strict_crop():
    """Unlike the old code, already-removed >=50% quads must survive."""
    from onestring_physics.optcuts_test_boundary_reparameterization_patch import (
        install_optcuts_test_boundary_reparameterization_patch,
    )
    uv,tri,loop=_grid_chart()
    p=SimpleNamespace(
        method="optcuts_official",uv_vertices_2d=uv.copy(),uv_faces=tri.copy(),
        surface_vertices_3d=np.c_[uv,np.zeros(len(uv))],
        surface_faces=tri.copy(),
        omega_boundary=uv[np.r_[loop,loop[0]]].copy(),metrics={},
        triangle_acceleration=None,
    )
    grid=SimpleNamespace(nx=4,ny=4)
    calls={"strict":0}
    def fake_legacy_crop(*_args,**_kwargs):
        calls["strict"]+=1
        raise AssertionError("M2D pre-crop would permanently remove valid boundary quads")
    from dataclasses import dataclass
    @dataclass
    class Params:
        omega_parameterization_mode:str="optcuts_test"
        optcuts_try_alternative_initial_cut:bool=True
        optcuts_area_grid_boundary_enabled:bool=True
        optcuts_area_grid_coverage_threshold:float=.5
        omega_overlay_margin:int=1
    def actual_domain(param, *_):
        x=np.linspace(-.125,1.125,11)
        xx,yy=np.meshgrid(x,x,indexing="xy")
        return SimpleNamespace(
            boundary=param.omega_boundary.copy(),parameterization=param,
            uv_vertices=np.stack([xx,yy],axis=-1).reshape(-1,2),
            overlay_nx=10,overlay_ny=10,overlay_margin_tiles=1,
            split_lines=[],
        )
    class FakeQuadMesh:
        def __init__(self,vertices,faces,grid,stage,metrics,split_lines):
            self.vertices=vertices;self.faces=faces;self.grid=grid
            self.stage=stage;self.metrics=metrics;self.split_lines=split_lines
    class FakeGrid:
        def __init__(self,n,m):
            self.tiles=[SimpleNamespace(vertex_ids=(
                j*(n+1)+i,j*(n+1)+i+1,
                (j+1)*(n+1)+i+1,(j+1)*(n+1)+i))
                for j in range(m) for i in range(n)]
    origin=SimpleNamespace(
        QuadMesh=FakeQuadMesh,
        create_quad_grid=lambda n,m,*_args: FakeGrid(n,m),
    )
    owner=SimpleNamespace(
        _original=origin,
        _build_surface_parameterization=lambda *_args:p,
        _flatten_to_domain=actual_domain,
        _build_m2d=fake_legacy_crop,
    )
    install_optcuts_test_boundary_reparameterization_patch(owner)
    params=Params()
    chart=owner._build_surface_parameterization(None,None,grid,params)
    domain=owner._flatten_to_domain(chart,grid,params)
    # QuadMesh uses original 50% mask directly, even if previous crop would fail.
    grid.tile_size=1.;grid.gap_size=0.
    m2d=owner._build_m2d(grid,domain,params)
    assert calls["strict"]==0
    assert len(m2d.faces)==64



def test_no_flip_untangling_repairs_a_folded_harmonic_interior():
    """The user's 25%-boundary-motion failure must trigger an actual
    interior optimization, not immediate rejection or threshold reduction.
    """
    from onestring_physics.optcuts_grid_boundary_untangle import (
        untangle_fixed_boundary, oriented_area_ratio,
        fit_by_continuation,
    )
    uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
    faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]],int)
    loop=np.array([0,1,2,3])
    # Target boundary stays a simple square. A wrong harmonic interior
    # guess lies outside, folding three triangles.
    harmonic=uv.copy()
    harmonic[4]=[-1.,-1.]
    assert np.min(oriented_area_ratio(uv,harmonic,faces))<0.
    out,diag=untangle_fixed_boundary(
        uv,faces,loop,uv[loop],harmonic,uv)
    np.testing.assert_allclose(out[loop],uv[loop])
    assert float(np.min(oriented_area_ratio(uv,out,faces)))>=.005
    assert diag["solver"]=="orientation_barrier_lbfgsb"

    def fake_harmonic(base,faces,loop,targets):
        folded=base.copy()
        folded[loop]=targets
        folded[4]=[-1.,-1.]
        return folded

    def check(candidate):
        ratio=oriented_area_ratio(uv,candidate,faces)
        return dict(
            flipped_triangles=int(np.count_nonzero(ratio<=0)),
            degenerate_triangles=int(np.count_nonzero(ratio<.005)),
            injectivity_overlap_pairs=0,
        )
    fitted,history=fit_by_continuation(
        uv,faces,loop,uv[loop],fake_harmonic,check)
    assert len(history)>=4
    assert any(row.get("solver")=="orientation_barrier_lbfgsb"
               for row in history)
    assert all(row["accepted"] for row in history)
    assert np.min(oriented_area_ratio(uv,fitted,faces))>=.005


def test_boundary_phase_selection_uses_global_displacement():
    from onestring_physics.optcuts_area_grid_boundary import (
        _boundary_target_candidates,
    )
    uv=np.array([[0.,0.],[2.,0.],[2.,1.],[0.,1.]])
    reversed_start=np.roll(uv,2,axis=0)
    phases=_boundary_target_candidates(uv,np.arange(4),reversed_start)
    assert len(phases)==4
    np.testing.assert_allclose(phases[0][2],uv)
    assert phases[0][0]<1e-12



def test_corner_dp_search_preserves_boundary_order_and_grid_corners():
    """A nonuniform sampled contour needs more than arc-length landmarks."""
    from onestring_physics.optcuts_area_grid_boundary import (
        _monotone_corner_indices_dp, _boundary_target_candidates,
    )
    old = np.array([
        [0., 0.], [.3, 0.], [1., 0.], [2., 0.], [2., 1.],
        [2., 2.], [1., 2.], [0., 2.], [0., 1.],
    ])
    outline = np.array([
        [0., 0.], [2., 0.], [2., .3],
        [.6, .3], [.6, 2.], [0., 2.],
    ])
    old_lengths = np.linalg.norm(np.roll(old, -1, axis=0) - old, axis=1)
    new_lengths = np.linalg.norm(np.roll(outline, -1, axis=0) - outline, axis=1)
    old_t = np.r_[0., np.cumsum(old_lengths[:-1])] / old_lengths.sum()
    times = np.r_[0., np.cumsum(new_lengths[:-1])] / new_lengths.sum()
    indices = _monotone_corner_indices_dp(old, old_t, outline, times)
    assert indices[0] == 0
    assert len(set(indices)) == len(outline)
    assert all(b > a for a, b in zip(indices, indices[1:]))
    candidates = _boundary_target_candidates(
        old, np.arange(len(old)), outline, max_phases=6,
    )
    assert any(method == "corner_dp" for _, _, _, method in candidates)
    assert all(len(mapped) == len(old) for _, _, mapped, _ in candidates)
    # Each candidate exactly interpolates the SAME corners, never softens
    # the 50% cell footprint or reduces the number of mandatory Grid turns.
    for _, phase, mapped, method in candidates:
        shifted = np.roll(outline, -phase, axis=0)
        assert all(np.min(np.linalg.norm(mapped - corner, axis=1)) < 1e-10
                   for corner in shifted)


def test_corner_dp_square_regression_retains_zero_displacement():
    from onestring_physics.optcuts_area_grid_boundary import (
        _boundary_target_candidates,
    )
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    result = _boundary_target_candidates(
        square, np.arange(4), square, max_phases=4,
    )
    assert len(result) == 4
    assert result[0][0] == pytest.approx(0.)
    np.testing.assert_allclose(result[0][2], square)



def test_grid_aware_seam_complexity_prefers_fewer_staircase_corners():
    from onestring_physics.optcuts_grid_aware_seam_selection import (
        grid_outline_complexity, grid_seam_selection_score,
    )
    # A four-corner square versus a staircase with six reflex corners.
    square = dict(
        grid_outline=np.array([[0, 0], [5, 0], [5, 5], [0, 5]], float),
        step=np.ones(2), mask=np.ones((5, 5), bool),
    )
    stairs = dict(
        grid_outline=np.array([
            [0, 0], [6, 0], [6, 1], [5, 1], [5, 2], [6, 2],
            [6, 6], [0, 6], [0, 4], [1, 4], [1, 3], [0, 3],
        ], float),
        step=np.ones(2), mask=np.ones((6, 6), bool),
    )
    a = grid_outline_complexity(square)
    b = grid_outline_complexity(stairs)
    audit = {"distortion_area_mean": 2., "seam_length": 1.,
             "csf_over_2_area_fraction": .01}
    assert a["corners"] == 4 and a["reflex_corners"] == 0
    assert b["corners"] > a["corners"]
    assert b["reflex_corners"] > 0
    assert (grid_seam_selection_score(a, audit, reference_sd=2.)
            < grid_seam_selection_score(b, audit, reference_sd=2.))


def test_grid_aware_seam_ranking_uses_each_original_production_grid():
    from onestring_physics.optcuts_grid_aware_seam_selection import (
        rank_grid_aware_seam_candidates,
    )
    uv, faces, loop = _grid_chart()
    center = np.array([.5, .5])
    theta = np.deg2rad(35.)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    rotated = (uv - center) @ rotation.T + center
    def candidate(points):
        return SimpleNamespace(
            surface_vertices_3d=np.c_[uv, np.zeros(len(uv))],
            surface_faces=faces.copy(),
            uv_vertices_2d=points.copy(),
            uv_faces=faces.copy(),
            metrics={},
        )
    square = candidate(uv)
    diagonal = candidate(rotated)
    official_quality = {
        "distortion_area_mean": 2., "seam_length": 1.,
        "csf_over_2_area_fraction": .0,
        "flipped_triangles": 0, "degenerate_triangles": 0,
        "injectivity_overlap_pairs": 0,
    }
    param = SimpleNamespace(
        method="optcuts_test", metrics={},
        surface_vertices_3d=diagonal.surface_vertices_3d.copy(),
        surface_faces=faces.copy(),
        uv_vertices_2d=rotated.copy(),
        uv_faces=faces.copy(),
        omega_boundary=rotated[np.r_[loop, loop[0]]],
        triangle_acceleration=None,
        _onestring_optcuts_candidates=[diagonal, square],
        _onestring_optcuts_candidate_qualities=[
            dict(official_quality), dict(official_quality)],
    )
    grid = SimpleNamespace(nx=4, ny=4)
    params = SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,
        optcuts_area_grid_orientation_search_enabled=False)
    def actual_grid(chart, _grid, _params):
        boundary = np.asarray(chart.omega_boundary)
        step = np.ptp(boundary, axis=0) / (2 * grid.nx)
        start = np.min(boundary, axis=0) - step
        x = start[0] + np.arange(11) * step[0]
        y = start[1] + np.arange(11) * step[1]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return SimpleNamespace(
            uv_vertices=np.stack([xx, yy], axis=-1).reshape(-1, 2),
            overlay_nx=10, overlay_ny=10, overlay_margin_tiles=1,
            boundary=boundary.copy(), parameterization=chart,
        )
    ranked, report = rank_grid_aware_seam_candidates(
        param, grid, params, actual_grid)
    assert len(ranked) == 2
    assert all(item["status"] == "scored" for item in report)
    assert report[0]["corners"] > report[1]["corners"]
    assert report[0]["reflex_corners"] > report[1]["reflex_corners"]
    assert ranked[0][2]["candidate_index"] == 1
    # Scoring never mutates the original chart or its OptCuts starts.
    np.testing.assert_allclose(param.uv_vertices_2d, rotated)
    assert len(param._onestring_optcuts_candidates) == 2
    for chart, domain, row in ranked:
        assert row["real_grid_origin"] == pytest.approx(
            domain.uv_vertices[0].tolist())
        assert row["retained_cells"] > 0



def test_grid_seam_selection_tries_next_official_cut_after_fit_failure(monkeypatch):
    """Only a geometrically accepted fit is committed to the pipeline state."""
    from dataclasses import dataclass
    from onestring_physics.optcuts_area_grid_boundary import (
        GridBoundaryReparameterizationError,
    )
    from onestring_physics.optcuts_test_boundary_reparameterization_patch import (
        install_optcuts_test_boundary_reparameterization_patch,
    )
    import onestring_physics.optcuts_area_grid_boundary as boundary_module

    uv, faces, loop = _grid_chart()
    result = SimpleNamespace(
        surface_vertices_3d=np.c_[uv, np.zeros(len(uv))],
        surface_faces=faces.copy(), uv_vertices_2d=uv.copy(),
        uv_faces=faces.copy(), metrics={},
    )
    metrics = {
        "distortion_area_mean": 2., "seam_length": 1.,
        "csf_over_2_area_fraction": 0.,
        "flipped_triangles": 0, "degenerate_triangles": 0,
        "injectivity_overlap_pairs": 0,
    }
    original = SimpleNamespace(
        method="optcuts_test", metrics={},
        surface_vertices_3d=result.surface_vertices_3d.copy(),
        surface_faces=faces.copy(), uv_vertices_2d=uv.copy(),
        uv_faces=faces.copy(),
        omega_boundary=uv[np.r_[loop, loop[0]]].copy(),
        triangle_acceleration=None,
        _onestring_optcuts_candidates=[result, result],
        _onestring_optcuts_candidate_qualities=[dict(metrics), dict(metrics)],
    )
    grid = SimpleNamespace(nx=4, ny=4)
    @dataclass
    class Params:
        omega_parameterization_mode: str = "optcuts_test"
        optcuts_try_alternative_initial_cut: bool = True
        optcuts_area_grid_boundary_enabled: bool = True
        optcuts_area_grid_coverage_threshold: float = .5
        optcuts_area_grid_orientation_search_enabled: bool = False

    def domain_for(chart, *_args):
        x = np.linspace(-.125, 1.125, 11)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        return SimpleNamespace(
            uv_vertices=np.stack([xx, yy], axis=-1).reshape(-1, 2),
            overlay_nx=10, overlay_ny=10, overlay_margin_tiles=1,
            boundary=chart.omega_boundary.copy(), parameterization=chart,
            split_lines=[],
        )

    invoked = []
    def fake_fit(chart, _grid, _params, *, domain):
        invoked.append(int(chart.metrics["optcuts_area_grid_seam_start_index"]))
        if len(invoked) == 1:
            raise GridBoundaryReparameterizationError("synthetic first fit failure")
        # Only the control-flow/lattice-reuse branch is mocked. Full UV fitting
        # remains covered separately by the geometric regression tests.
        chart._optcuts_area_grid_mask = np.ones((10, 10), bool)
        chart._optcuts_area_grid_coverage = np.ones((10, 10), float)
        chart._optcuts_area_grid_source_boundary = chart.omega_boundary.copy()
        chart._optcuts_area_grid_target_outline = chart.omega_boundary.copy()
        chart.metrics.update(
            optcuts_area_grid_boundary_enabled=True,
            optcuts_area_grid_grid_origin=[-.125, -.125],
            optcuts_area_grid_grid_step=[.125, .125],
            optcuts_area_grid_overlay_nx=10,
            optcuts_area_grid_overlay_ny=10,
            optcuts_area_grid_margin=1,
        )
        return chart

    monkeypatch.setattr(boundary_module, "apply_area_grid_boundary", fake_fit)
    owner = SimpleNamespace(
        _build_surface_parameterization=lambda *_: original,
        _flatten_to_domain=domain_for,
        _build_m2d=lambda *_: None,
    )
    install_optcuts_test_boundary_reparameterization_patch(owner)
    domain = owner._flatten_to_domain(original, grid, Params())
    assert invoked == [0, 1]
    assert domain.parameterization is original
    assert original.metrics["optcuts_area_grid_seam_selected_start"] == 1
    assert (original.metrics["optcuts_area_grid_seam_ranking"][0]["status"]
            == "boundary_fit_failed")
    assert (original.metrics["optcuts_area_grid_seam_ranking"][1]["status"]
            == "boundary_fit_accepted")
    assert len(original._onestring_optcuts_candidates) == 0


def test_grid_seam_score_rejects_excessive_original_uv_distortion():
    from onestring_physics.optcuts_grid_aware_seam_selection import (
        grid_seam_selection_score,
    )
    shape = {"corners": 4, "reflex_corners": 0,
             "perimeter_grid_units": 20., "retained_cells": 25}
    base = dict(distortion_area_mean=2., seam_length=1.,
                csf_over_2_area_fraction=0.)
    bad = dict(distortion_area_mean=8., seam_length=1.,
               csf_over_2_area_fraction=.4)
    assert (grid_seam_selection_score(shape, bad, reference_sd=2.)
            > grid_seam_selection_score(shape, base, reference_sd=2.))



def test_one_official_seam_has_real_uv_orientations_not_fake_seams():
    """The fixed-grid production geometry changes while the seam IDs do not."""
    from onestring_physics.optcuts_grid_aware_seam_selection import (
        rank_grid_aware_seam_candidates,
    )
    uv, tri, loop = _grid_chart(n=6)
    chart = SimpleNamespace(
        method="optcuts_test", metrics={"boundary_loop": list(loop)},
        surface_vertices_3d=np.c_[uv, np.zeros(len(uv))],
        surface_faces=tri.copy(), uv_vertices_2d=uv.copy(),
        uv_faces=tri.copy(), omega_boundary=uv[np.r_[loop,loop[0]]],
        triangle_acceleration=None,
    )
    grid = SimpleNamespace(nx=3, ny=3)
    params = SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,
        optcuts_area_grid_orientation_search_enabled=True,
    )
    def actual_domain(candidate, *_args):
        pts = np.asarray(candidate.omega_boundary, dtype=float)
        lo = pts.min(axis=0)
        step = np.maximum(np.ptp(pts, axis=0)/6., 1e-8)
        xx, yy = np.meshgrid(
            lo[0] + (np.arange(9) - 1)*step[0],
            lo[1] + (np.arange(9) - 1)*step[1],
            indexing="xy",
        )
        return SimpleNamespace(
            uv_vertices=np.stack([xx, yy], axis=-1).reshape(-1, 2),
            overlay_nx=8, overlay_ny=8, overlay_margin_tiles=1,
            parameterization=candidate,
        )
    ranked, report = rank_grid_aware_seam_candidates(
        chart, grid, params, actual_domain,
    )
    assert len(report) == 4
    assert len(ranked) >= 2
    assert {x["uv_rotation_degrees"] for x in report} == {
        0., 22.5, 45., 67.5}
    assert {x["source_start_index"] for x in report} == {0}
    assert all(np.array_equal(c.uv_faces, tri) for c, _, _ in ranked)
    np.testing.assert_allclose(chart.uv_vertices_2d, uv)
    assert all(np.isfinite(x["score"]) for _, _, x in ranked)


def test_grid_seam_orientation_toggle_only_one_chart_when_disabled():
    from onestring_physics.optcuts_grid_aware_seam_selection import (
        rank_grid_aware_seam_candidates,
    )
    uv, tri, loop = _grid_chart(n=6)
    chart = SimpleNamespace(
        method="optcuts_test", metrics={},
        surface_vertices_3d=np.c_[uv, np.zeros(len(uv))],
        surface_faces=tri, uv_vertices_2d=uv,
        uv_faces=tri, omega_boundary=uv[np.r_[loop, loop[0]]],
        triangle_acceleration=None,
    )
    grid = SimpleNamespace(nx=3, ny=3)
    params = SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,
        optcuts_area_grid_orientation_search_enabled=False,
    )
    def build(c, *_args):
        x = np.linspace(-1/6, 7/6, 9)
        xx, yy = np.meshgrid(x, x, indexing="xy")
        return SimpleNamespace(
            uv_vertices=np.stack([xx, yy], axis=-1).reshape(-1, 2),
            overlay_nx=8, overlay_ny=8, overlay_margin_tiles=1,
        )
    ranks, rows = rank_grid_aware_seam_candidates(chart, grid, params, build)
    assert len(rows) == len(ranks) == 1
    assert rows[0]["uv_rotation_degrees"] == 0.



def test_boundary_ear_preflight_proves_zero_area_without_optimization():
    """Three fixed boundary vertices cannot be untangled by interior UV."""
    from onestring_physics.optcuts_grid_ear_correspondence import (
        fixed_boundary_ear_report, map_boundary_with_corner_indices,
    )
    uv = np.array([
        [0.,0.], [1.,-.4], [2.,0.],
        [2.,1.], [1.1,2.], [0.,1.], [1.,.7],
    ])
    triangles = np.array([
        [0,1,2], [0,2,6], [2,3,6],
        [3,4,6], [4,5,6], [5,0,6],
    ],int)
    loop=np.arange(6)
    corners=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    edge_lengths=np.linalg.norm(np.roll(uv[loop],-1,axis=0)-
                                uv[loop],axis=1)
    target=map_boundary_with_corner_indices(
        uv[loop],edge_lengths,corners,[0,3,4,5])
    before=fixed_boundary_ear_report(uv,triangles,loop,target)
    assert 0 in before["bad_ids"]
    assert before["minimum_fixed_ratio"] <= 1e-10


def test_boundary_ear_repair_keeps_exact_grid_corners_and_uv_topology():
    from onestring_physics.optcuts_grid_ear_correspondence import (
        fixed_boundary_ear_report, repair_boundary_ears,
    )
    uv = np.array([
        [0.,0.], [1.,-.4], [2.,0.],
        [2.,1.], [1.1,2.], [0.,1.], [1.,.7],
    ])
    triangles = np.array([
        [0,1,2], [0,2,6], [2,3,6],
        [3,4,6], [4,5,6], [5,0,6],
    ],int)
    loop=np.arange(6)
    corners=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    edge_lengths=np.linalg.norm(np.roll(uv[loop],-1,axis=0)-
                                uv[loop],axis=1)
    source=uv.copy()
    faces_before=triangles.copy()
    target, revised, edits, check=repair_boundary_ears(
        uv,triangles,loop,corners,[0,3,4,5],edge_lengths)
    assert len(check["bad_ids"]) == 0
    assert check["minimum_fixed_ratio"] >= .005
    assert len(edits) == 1
    assert revised[1] == 1
    assert all(b>a for a,b in zip(revised,revised[1:]))
    for corner in corners:
        assert min(np.linalg.norm(target-corner,axis=1)) < 1e-10
    np.testing.assert_array_equal(uv,source)
    np.testing.assert_array_equal(triangles,faces_before)


def test_correspondence_search_can_use_ear_aware_corner_assignment():
    from onestring_physics.optcuts_grid_ear_correspondence import (
        fixed_boundary_ear_report,
    )
    from onestring_physics.optcuts_area_grid_boundary import (
        _boundary_target_candidates,
    )
    uv = np.array([
        [0.,0.], [1.,-.4], [2.,0.],
        [2.,1.], [1.1,2.], [0.,1.], [1.,.7],
    ])
    faces=np.array([
        [0,1,2],[0,2,6],[2,3,6],
        [3,4,6],[4,5,6],[5,0,6],
    ],int)
    outline=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    ranked=_boundary_target_candidates(
        uv,np.arange(6),outline,max_phases=4,faces=faces)
    assert ranked
    assert all(
        not len(fixed_boundary_ear_report(
            uv,faces,np.arange(6),mapped)["bad_ids"])
        for _,_,mapped,_ in ranked
    )



def test_collapsed_ear_raises_recoverable_exception_then_reassigns_grid_corner():
    """Do not just skip a UV ear that becomes a straight Grid segment."""
    from onestring_physics.optcuts_grid_ear_correspondence import (
        FixedBoundaryEarCollapseError,
        map_boundary_with_corner_indices,
        require_noncollapsed_boundary_ears,
        recover_boundary_ear_correspondences,
    )
    uv=np.array([
        [0.,0.],[1.,-.4],[2.,0.],[2.,1.],[1.1,2.],[0.,1.],[1.,.7]
    ])
    f=np.array([
        [0,1,2],[0,2,6],[2,3,6],[3,4,6],[4,5,6],[5,0,6]
    ],int)
    loop=np.arange(6)
    corners=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    edge=np.linalg.norm(
        np.roll(uv[loop],-1,axis=0)-uv[loop],axis=1)
    orig=[0,3,4,5]
    target=map_boundary_with_corner_indices(uv[loop],edge,corners,orig)
    with pytest.raises(FixedBoundaryEarCollapseError) as captured:
        require_noncollapsed_boundary_ears(uv,f,loop,target)
    assert 0 in captured.value.face_ids
    before_uv=uv.copy()
    before_faces=f.copy()
    options=recover_boundary_ear_correspondences(
        uv,f,loop,corners,orig,edge)
    assert options
    for option in options:
        fixed=require_noncollapsed_boundary_ears(
            uv,f,loop,option["mapped"])
        assert len(fixed["bad_ids"])==0
        assert min(fixed["ratios"])>=.005
        assert all(
            b>a for a,b in zip(option["indices"],option["indices"][1:]))
        for corner in corners:
            assert min(np.linalg.norm(
                option["mapped"]-corner,axis=1)) < 1e-10
        assert option["edits"]
    np.testing.assert_array_equal(uv,before_uv)
    np.testing.assert_array_equal(f,before_faces)


def test_exception_recovery_only_changes_correspondence_not_surface():
    """Multiple trials must not introduce invented Grid/cut geometry."""
    from onestring_physics.optcuts_area_grid_boundary import (
        _boundary_target_candidates,
    )
    from onestring_physics.optcuts_grid_ear_correspondence import (
        require_noncollapsed_boundary_ears,
    )
    uv=np.array([
        [0.,0.],[1.,-.4],[2.,0.],[2.,1.],[1.1,2.],[0.,1.],[1.,.7]
    ])
    f=np.array([
        [0,1,2],[0,2,6],[2,3,6],[3,4,6],[4,5,6],[5,0,6]
    ],int)
    outline=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    original_uv=uv.copy()
    candidates=_boundary_target_candidates(
        uv,np.arange(6),outline,max_phases=4,faces=f)
    assert any("ear_exception_recovered" in name
               for _,_,_,name in candidates)
    for _,_,mapped,_ in candidates:
        require_noncollapsed_boundary_ears(
            uv,f,np.arange(6),mapped)
        for target_corner in outline:
            assert np.any(np.linalg.norm(
                mapped-target_corner,axis=1) < 1e-9)
    np.testing.assert_array_equal(uv,original_uv)



def _distortion_retry_fixture(monkeypatch, *, second_mean_sd=4.5):
    """Synthetic fixed-grid trial: first fit bijective but high distortion."""
    # These older tests isolate the correspondence fallback from the
    # separate SD-refinement feature (their fake quality lacks SD geometry).
    monkeypatch.setenv("ONESTRING_GRID_SD_REFINEMENT","0")
    from onestring_physics.optcuts_area_grid_boundary import (
        GridBoundaryReparameterizationError,
    )
    import onestring_physics.optcuts_area_grid_boundary as boundary_module
    import onestring_physics.optcuts_grid_boundary_untangle as untangler
    import onestring_physics.optcuts_quality_audit as audit_module

    uv=np.array([
        [0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]
    ],float)
    faces=np.array([
        [0,1,4],[1,2,4],[2,3,4],[3,0,4]
    ],int)
    p=SimpleNamespace(
        uv_vertices_2d=uv.copy(), uv_faces=faces.copy(),
        surface_vertices_3d=np.c_[uv,np.zeros(len(uv))],
        surface_faces=faces.copy(),
        omega_boundary=uv[np.array([0,1,2,3,0])].copy(),
        metrics={}, triangle_acceleration=None,
    )
    target=dict(
        mask=np.ones((2,2),bool),
        coverage=np.ones((2,2),float),
        grid_outline=uv[:4].copy(),
        step=np.array([.5,.5]), origin=np.array([0.,0.]),
        nx=2, ny=2, margin=0,
        grid_source="real_production_m2d_overlay",
    )
    monkeypatch.setattr(
        boundary_module,"construct_grid_target",
        lambda *_args,**_kwargs:target,
    )
    # Two distinct corner matching / fitting trials with the same
    # prescribed exact grid. Only one meets the ORIGINAL quality cutoff.
    candidates=[
        (1.,"bad_phase",uv[:4].copy(),"corner_dp_ear_exception_recovered_1"),
        (2.,"good_phase",uv[:4].copy(),"corner_dp_ear_exception_recovered_2"),
    ]
    candidates=[(x,idx,points,method)
                for idx,(x,_phase,points,method) in enumerate(candidates)]
    monkeypatch.setattr(
        boundary_module,"_boundary_target_candidates",
        lambda *_args,**_kwargs:candidates,
    )
    num={"calls":0}
    def fake_fit(base,faces,loop,mapped,harmonic,assess,**_kwargs):
        num["calls"]+=1
        result=base.copy()
        result[4,0] = .56 if num["calls"]==1 else .44
        return result,[{"fraction":1., "accepted":True}]
    monkeypatch.setattr(untangler,"fit_by_continuation",fake_fit)

    def fake_quality(res):
        cx=float(res.uv_vertices_2d[4,0])
        if cx>.53:return {
            "distortion_area_mean":8.663,
            "flipped_triangles":0,"injectivity_overlap_pairs":0,
        }
        if cx<.47:return {
            "distortion_area_mean":second_mean_sd,
            "flipped_triangles":0,"injectivity_overlap_pairs":0,
        }
        return {
            "distortion_area_mean":4.1,
            "flipped_triangles":0,"injectivity_overlap_pairs":0,
        }
    monkeypatch.setattr(audit_module,"quality",fake_quality)
    return p, SimpleNamespace(nx=4,ny=4), SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,omega_overlay_margin=0
    ), num


def test_grid_fit_tries_second_correspondence_when_first_is_too_distorted(
    monkeypatch
):
    """Exactly reached boundary does not mean the mapping is acceptable."""
    from onestring_physics.optcuts_area_grid_boundary import (
        apply_area_grid_boundary,
    )
    p,grid,params,num=_distortion_retry_fixture(monkeypatch)
    output=apply_area_grid_boundary(p,grid,params,domain=object())
    assert output is p
    assert num["calls"]==2
    assert p.metrics["optcuts_area_grid_boundary_selected_attempt"]==2
    assert p.metrics["optcuts_area_grid_boundary_quality_rejections"]==1
    assert p.metrics["optcuts_area_grid_boundary_full_boundary_attempts"]==2
    assert p.metrics["optcuts_area_grid_final_mean_sd"]==pytest.approx(4.5)
    assert p.metrics["optcuts_area_grid_boundary_mean_sd_limit"]==pytest.approx(5.33)
    np.testing.assert_allclose(p.omega_boundary[:4],np.array([
        [0.,0.],[1.,0.],[1.,1.],[0.,1.]
    ]))
    assert p._optcuts_area_grid_mask.sum()==4


def test_grid_fit_does_not_claim_success_when_all_full_boundary_maps_exceed_sd(
    monkeypatch
):
    """No lowering distortion guard and no half-committed failed UV."""
    from onestring_physics.optcuts_area_grid_boundary import (
        GridBoundaryReparameterizationError,apply_area_grid_boundary,
    )
    p,grid,params,num=_distortion_retry_fixture(
        monkeypatch,second_mean_sd=10.7)
    old_uv=p.uv_vertices_2d.copy()
    old_omega=p.omega_boundary.copy()
    with pytest.raises(
        GridBoundaryReparameterizationError,
        match="ALL_ACCEPTABLE_CORRESPONDENCES_FAILED",
    ) as err:
        apply_area_grid_boundary(p,grid,params,domain=object())
    assert num["calls"]==2
    assert "quality_rejected=2" in str(err.value)
    assert "boundary_completed=2" in str(err.value)
    assert "mean_sd_limit=5.33" in str(err.value)
    np.testing.assert_allclose(p.uv_vertices_2d,old_uv)
    np.testing.assert_allclose(p.omega_boundary,old_omega)
    assert not p.metrics.get("optcuts_area_grid_boundary_enabled",False)
