"""Numerical and geometric regression for low-distortion fixed-Grid UV."""

from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_grid_sd_refinement import (
    SymmetricDirichletProblem,
    _boundary_sliders,
    refine_exact_grid_uv,
)
from onestring_physics.optcuts_quality_audit import quality


def chart_with_center():
    xy=np.array([
        [0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]
    ],float)
    faces=np.array([
        [0,1,4],[1,2,4],[2,3,4],[3,0,4]
    ],int)
    return xy,np.c_[xy,np.zeros(len(xy))],faces


def audit_factory(xyz,faces):
    def audit(uv):
        return quality(SimpleNamespace(
            surface_vertices_3d=xyz,
            surface_faces=faces,
            uv_vertices_2d=np.asarray(uv,float),
            uv_faces=faces,
        ))
    return audit


def test_analytic_sd_matches_existing_svd_audit():
    base,xyz,faces=chart_with_center()
    after=base.copy()
    after[4]=[.72,.31]
    p=SymmetricDirichletProblem(xyz,faces,faces,base)
    measured=audit_factory(xyz,faces)(after)
    assert p.exact_mean_sd(after)==pytest.approx(
        measured["distortion_area_mean"],rel=1e-11,abs=1e-11
    )
    score,gradient=p.energy_gradient(after)
    assert score==pytest.approx(
        measured["distortion_area_mean"],rel=1e-11,abs=1e-11
    )
    assert gradient.shape==after.shape
    assert np.isfinite(gradient).all()


def test_analytic_gradient_matches_central_difference():
    base,xyz,faces=chart_with_center()
    after=base.copy()
    after[4]=[.68,.35]
    problem=SymmetricDirichletProblem(xyz,faces,faces,base)
    _,jac=problem.energy_gradient(after)
    for vid,axis in [(4,0),(4,1),(0,0),(1,1)]:
        eps=1e-6
        right=after.copy();left=after.copy()
        right[vid,axis]+=eps;left[vid,axis]-=eps
        numerical=(
            problem.energy_gradient(right)[0]
            -problem.energy_gradient(left)[0]
        )/(2*eps)
        assert jac[vid,axis]==pytest.approx(
            numerical,rel=2e-5,abs=2e-5
        )


def test_exact_grid_boundary_unchanged_and_interior_sd_reduced():
    base,xyz,faces=chart_with_center()
    initial=base.copy()
    initial[4]=[.81,.25]
    initial_copy=initial.copy()
    before=audit_factory(xyz,faces)(initial)
    assert before["distortion_area_mean"]>4.1
    improved,log=refine_exact_grid_uv(
        xyz,faces,faces,base,initial,np.arange(4),base[:4],
        assess=audit_factory(xyz,faces),
        mean_sd_limit=4.1, interior_maxiter=125,
        sliding_maxiter=40,progress=False,
    )
    after=audit_factory(xyz,faces)(improved)
    assert after["distortion_area_mean"]<before["distortion_area_mean"]
    assert after["distortion_area_mean"]<=4.1
    assert log["status"]=="within_limit"
    assert log["improvement"]>0
    assert log["min_area_ratio"]>=.005
    assert after["flipped_triangles"]==0
    assert after["injectivity_overlap_pairs"]==0
    np.testing.assert_array_equal(improved[:4],initial[:4])
    np.testing.assert_array_equal(initial,initial_copy)


def chart_with_sliding_vertices():
    base=np.array([
        [0.,0.],[.25,0.],[1.,0.],[1.,1.],
        [.65,1.],[0.,1.],[.5,.5]
    ],float)
    faces=np.asarray([
        [0,1,6],[1,2,6],[2,3,6],
        [3,4,6],[4,5,6],[5,0,6]
    ],int)
    outline=np.array([
        [0.,0.],[1.,0.],[1.,1.],[0.,1.]
    ])
    xyz=np.c_[base,np.zeros(len(base))]
    return base,xyz,faces,outline


def test_sliding_boundaries_remain_on_same_grid_edges_in_order():
    base,xyz,faces,corners=chart_with_sliding_vertices()
    moved=base.copy()
    moved[1]=[.08,0.]
    moved[4]=[.89,1.]
    moved[6]=[.67,.31]
    source=moved.copy()
    a=_boundary_sliders(moved,np.arange(6),corners)
    ids,dirs,t,lo,hi=a
    assert set(ids)=={1,4}
    assert np.all(lo<t) and np.all(t<hi)
    before=audit_factory(xyz,faces)(moved)
    refined,log=refine_exact_grid_uv(
        xyz,faces,faces,base,moved,np.arange(6),corners,
        assess=audit_factory(xyz,faces),
        mean_sd_limit=4.01,interior_maxiter=40,
        sliding_maxiter=115,progress=False,
    )
    after=audit_factory(xyz,faces)(refined)
    assert after["distortion_area_mean"] < before["distortion_area_mean"]
    assert np.all(np.diff(refined[[0,1,2],0])>0)
    assert np.all(np.diff(refined[[3,4,5],0])<0)
    np.testing.assert_allclose(refined[[0,2,3,5]],base[[0,2,3,5]])
    assert refined[1,1] == pytest.approx(0,abs=1e-12)
    assert refined[4,1] == pytest.approx(1,abs=1e-12)
    assert after["flipped_triangles"]==0
    assert after["injectivity_overlap_pairs"]==0
    assert log["min_area_ratio"]>=.005
    assert any(x["phase"]=="grid_edge_slide" for x in log["phases"])
    np.testing.assert_array_equal(source,moved)


def test_energy_guard_preserves_triangulation_and_rejects_invalid_start():
    base,xyz,faces=chart_with_center()
    baseline=base.copy()
    invalid=base.copy()
    invalid[4]=[2.,2.]
    with pytest.raises(ValueError,match="GRID_SD_START_INVALID_MIN_AREA_RATIO"):
        refine_exact_grid_uv(
            xyz,faces,faces,base,invalid,np.arange(4),base[:4],
            assess=audit_factory(xyz,faces),mean_sd_limit=4.1,
            progress=False,
        )
    np.testing.assert_array_equal(base,baseline)


def test_no_change_if_initial_grid_chart_already_meets_same_sd_limit():
    base,xyz,faces=chart_with_center()
    before=base.copy()
    after,diagnostic=refine_exact_grid_uv(
        xyz,faces,faces,base,base.copy(),np.arange(4),base[:4],
        assess=audit_factory(xyz,faces),mean_sd_limit=4.2,
        progress=False,
    )
    assert diagnostic["status"]=="already_within_limit"
    np.testing.assert_array_equal(after,before)
