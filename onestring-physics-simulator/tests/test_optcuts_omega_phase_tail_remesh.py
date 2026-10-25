"""Ablation tests: ① phase ③ tail SD ⑤ topology-preserving local remesh."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from onestring_physics.optcuts_grid_aware_seam_selection import (
    _shift_production_lattice,
)
from onestring_physics.optcuts_grid_corner_remesh import (
    subdivide_grid_corner_neighborhood,
)
from onestring_physics.optcuts_grid_sd_refinement import (
    SymmetricDirichletProblem,
)


def _area3d(xyz, faces):
    tri = xyz[faces]
    return .5*np.linalg.norm(np.cross(
        tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)


def test_phase_search_shifts_only_candidate_grid_before_mask():
    xx, yy = np.meshgrid(np.arange(5.), np.arange(5.), indexing="xy")
    base = np.stack((xx, yy), axis=-1).reshape((-1, 2))
    domain = SimpleNamespace(
        overlay_nx=4, overlay_ny=4, uv_vertices=base.copy())
    shifted = _shift_production_lattice(domain, .5, .5)
    np.testing.assert_allclose(shifted.uv_vertices, base + .5)
    np.testing.assert_allclose(domain.uv_vertices, base)
    assert shifted.overlay_nx == domain.overlay_nx
    assert shifted.overlay_ny == domain.overlay_ny


def test_corner_remesh_preserves_surface_exactly_and_adds_uv_freedom():
    xy = np.array([[0., 0.], [2., 0.], [2., 2.],
                   [0., 2.], [1., 1.]])
    xyz = np.column_stack((xy, [0., 0., .2, 0., .1]))
    faces = np.array([[0, 1, 4], [1, 2, 4],
                      [2, 3, 4], [3, 0, 4]], int)
    obj = SimpleNamespace(
        uv_vertices_2d=xy.copy(), surface_vertices_3d=xyz.copy(),
        uv_faces=faces.copy(), surface_faces=faces.copy(),
        triangle_acceleration="stale",
    )
    before_area = _area3d(xyz, faces).sum()
    report = subdivide_grid_corner_neighborhood(
        obj, np.array([0, 1, 2, 3]), xy[:4], xy[:4])
    assert report["subdivided_triangles"] == 4
    assert report["added_interior_vertices"] == 4
    assert len(obj.uv_faces) == 12
    assert len(obj.surface_faces) == 12
    np.testing.assert_allclose(_area3d(
        obj.surface_vertices_3d, obj.surface_faces).sum(), before_area)
    np.testing.assert_allclose(obj.surface_vertices_3d[:len(xyz)], xyz)
    np.testing.assert_allclose(obj.uv_vertices_2d[:len(xy)], xy)
    assert obj.triangle_acceleration is None
    for i, face in enumerate(faces):
        np.testing.assert_allclose(obj.uv_vertices_2d[len(xy)+i],
                                   xy[face].mean(axis=0))
        np.testing.assert_allclose(obj.surface_vertices_3d[len(xyz)+i],
                                   xyz[face].mean(axis=0))
    # Every original edge is present; centroid splits create no T-junctions.
    def edge_set(ff):
        return {tuple(sorted((int(t[i]),int(t[(i+1)%3]))))
                for t in ff for i in range(3)}
    assert edge_set(faces).issubset(edge_set(obj.uv_faces))


def test_tail_sd_analytic_gradient_and_energy():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    source_faces = np.array([[0, 1, 2]])
    uv = np.array([[0., 0.], [1.2, 0.15], [-.15, .8]])
    model = SymmetricDirichletProblem(xyz, source_faces,
                                     source_faces, uv)
    tau = 4.2
    energy, grad = model.energy_gradient(
        uv, tail_threshold=tau, tail_strength=.5)
    assert np.isfinite(energy) and np.isfinite(grad).all()
    np.testing.assert_allclose(energy, model.tail_score(
        uv,tau,strength=.5), atol=1e-9)
    eps = 1e-6
    for i in range(len(uv)):
        for j in range(2):
            pos=uv.copy()
            neg=uv.copy()
            pos[i,j]+=eps
            neg[i,j]-=eps
            plus=model.energy_gradient(pos,tail_threshold=tau)[0]
            minus=model.energy_gradient(neg,tail_threshold=tau)[0]
            np.testing.assert_allclose(grad[i,j],(plus-minus)/(2*eps),
                                       rtol=1e-4,atol=5e-5)


def test_four_modes_are_distinct_and_independently_cached():
    root=Path(__file__).resolve().parents[1]
    launcher=(root/"app_optcuts_core_20260916.py").read_text()
    ui=(root/"app_runtime_base.py").read_text()
    core=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    for suffix in ("phase_search","tail_sd","adaptive_remesh","combined"):
        assert '"optcuts_area_grid_'+suffix+'"' in launcher
    for flag in (
        "optcuts_area_grid_phase_search_enabled",
        "optcuts_area_grid_tail_sd_enabled",
        "optcuts_area_grid_adaptive_remesh_enabled",
    ):
        assert flag+"=optcuts_area_grid_" in ui
        assert flag+": bool = False" in core
    assert '"phase_search": value in {' in launcher
    assert '"tail_sd": value in {' in launcher
    assert '"adaptive_remesh": value in {' in launcher


def _square_chart(n=4):
    x,y=np.meshgrid(np.linspace(0.,1.,n+1),
                    np.linspace(0.,1.,n+1),indexing="ij")
    uv=np.stack((x.ravel(),y.ravel()),axis=1)
    f=[]
    for i in range(n):
        for j in range(n):
            a=i*(n+1)+j
            b=(i+1)*(n+1)+j
            f.extend([[a,b,b+1],[a,b+1,a+1]])
    faces=np.asarray(f,int)
    loop=([i*(n+1) for i in range(n)]
          +[n*(n+1)+j for j in range(n)]
          +[i*(n+1)+n for i in range(n,0,-1)]
          +[j for j in range(n,0,-1)])
    return SimpleNamespace(
        uv_vertices_2d=uv.copy(),uv_faces=faces.copy(),
        surface_vertices_3d=np.c_[uv,np.zeros(len(uv))],
        surface_faces=faces.copy(),
        omega_boundary=uv[np.r_[loop,loop[0]]].copy(),
        triangle_acceleration=None,metrics={})


def test_corner_remesh_mode_integrates_into_actual_area_grid_fitting():
    from onestring_physics.optcuts_area_grid_boundary import apply_area_grid_boundary
    chart=_square_chart()
    source_count=len(chart.uv_faces)
    params=SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,omega_overlay_margin=1,
        optcuts_area_grid_adaptive_remesh_enabled=True,
        optcuts_area_grid_tail_sd_enabled=False)
    chart=apply_area_grid_boundary(chart,SimpleNamespace(nx=2,ny=2),params)
    metrics=chart.metrics
    assert metrics["optcuts_area_grid_adaptive_remesh_enabled"] is True
    assert metrics["optcuts_area_grid_boundary_status"]=="accepted_exact_grid_outline"
    assert metrics["optcuts_area_grid_remesh_diagnostic"]["subdivided_triangles"]>0
    assert len(chart.uv_faces)>source_count
    assert len(chart.uv_faces)==len(chart.surface_faces)
    assert metrics["optcuts_area_grid_min_signed_area_ratio"]>=.005


def test_tail_mode_calls_refinement_even_if_average_sd_passes(monkeypatch):
    from onestring_physics import optcuts_grid_sd_refinement as sd
    from onestring_physics.optcuts_area_grid_boundary import apply_area_grid_boundary
    chart=_square_chart()
    calls=[]
    real_refine=sd.refine_exact_grid_uv
    def monitored(*a,**k):
        calls.append(k.get("tail_aware"))
        return real_refine(*a,**k)
    monkeypatch.setattr(sd,"refine_exact_grid_uv",monitored)
    params=SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,omega_overlay_margin=1,
        optcuts_area_grid_tail_sd_enabled=True)
    out=apply_area_grid_boundary(chart,SimpleNamespace(nx=2,ny=2),params)
    assert calls==[True]
    assert out.metrics["optcuts_area_grid_sd_refinement_attempts"]==1
    assert out.metrics["optcuts_area_grid_boundary_status"]=="accepted_exact_grid_outline"


def test_phase_mode_evaluates_multiple_shifted_original_lattices():
    from onestring_physics.optcuts_grid_aware_seam_selection import (
        rank_grid_aware_seam_candidates,
    )
    chart=_square_chart()
    chart._onestring_optcuts_candidates=[]
    chart._onestring_optcuts_candidate_qualities=[]
    def chart_domain(chart,grid,_params):
        uv=chart.uv_vertices_2d
        start=np.min(uv,axis=0)-.25
        x=start[0]+np.arange(9)*.1875
        y=start[1]+np.arange(9)*.1875
        xx,yy=np.meshgrid(x,y,indexing="xy")
        return SimpleNamespace(uv_vertices=np.stack(
            (xx,yy),axis=-1).reshape((-1,2)),overlay_nx=8,
            overlay_ny=8,overlay_margin_tiles=1)
    params=SimpleNamespace(
        optcuts_area_grid_coverage_threshold=.5,
        optcuts_area_grid_orientation_search_enabled=False,
        optcuts_area_grid_phase_search_enabled=True)
    ranked,report=rank_grid_aware_seam_candidates(
        chart,SimpleNamespace(nx=2,ny=2),params,chart_domain)
    assert len(report)==4
    assert len(ranked)>=1
    assert {(r["grid_phase_u"],r["grid_phase_v"]) for r in report}=={
        (0.,0.),(.5,0.),(0.,.5),(.5,.5)}
    for _chart,domain,row in ranked:
        np.testing.assert_allclose(domain.uv_vertices[0],
                                   row["real_grid_origin"])
