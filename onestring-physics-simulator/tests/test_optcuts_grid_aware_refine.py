"""Grid-aware (UV-only) OptCuts ablation: regression and safety tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from onestring_physics.optcuts_backend import OptCutsConfig, OptCutsResult, run_official_optcuts
from onestring_physics.optcuts_grid_aware_refine import (
    _seam_sides, _seam_quality, _cell_yield_proxy,
    _seam_line_targets, _solve_harmonic_displacement,
    _oriented_and_nondegenerate, refine_optcuts_grid_aware,
)
from onestring_physics.optcuts_quality_audit import quality


def _two_patches():
    """Two valid UV triangles sharing a canonical 3D diagonal, cut in UV.

    A diagonal rather than grid-aligned seam makes seam alignment measurable.
    """
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    sf=np.array([[0,1,2],[0,2,3]],int)
    uv=np.array([[0.,0.],[1.,0.],[1.,1.],
                 [0.,.55],[1.,1.55],[0.,1.55]])
    uf=np.array([[0,1,2],[3,4,5]],int)
    return OptCutsResult(xyz,sf,uv,uf,[[0,1,2]],"synthetic.obj",{})


def test_seam_detected_by_3d_even_when_uv_chart_is_disconnected():
    obj=_two_patches()
    sides=_seam_sides(obj)
    assert len(sides)==1
    assert _seam_quality(obj.uv_vertices_2d,sides,8,8)>0.
    assert np.isfinite(_cell_yield_proxy(obj.uv_vertices_2d,[0,1,2],8,8))


def test_fixed_seam_targets_and_harmonic_step_reduce_diagonal_alignment():
    obj=_two_patches()
    uv=obj.uv_vertices_2d
    sides=_seam_sides(obj)
    target,weights=_seam_line_targets(uv,sides,8,8)
    assert np.count_nonzero(weights)>=4
    new=_solve_harmonic_displacement(uv,obj.uv_faces,target,weights,1.)
    assert np.linalg.norm(new-uv)>0.
    # The optimizer is permitted to reject the candidate if high distortion.
    # It must not change the source's 3D/UV indexing.
    final=refine_optcuts_grid_aware(obj,nx=8,ny=8,strength=1.0)
    np.testing.assert_array_equal(final.surface_faces,obj.surface_faces)
    np.testing.assert_array_equal(final.uv_faces,obj.uv_faces)
    assert final.metrics["optcuts_grid_aware_enabled"] is True
    assert final.metrics["optcuts_grid_aware_status"] in (
        "accepted","rejected_retained_original")
    if final.metrics["optcuts_grid_aware_status"]=="accepted":
        assert final.metrics["optcuts_grid_aware_alignment_after"] < final.metrics["optcuts_grid_aware_alignment_before"]
        assert final.metrics["optcuts_grid_aware_mean_symmetric_dirichlet_after"] <= (
            final.metrics["optcuts_grid_aware_mean_symmetric_dirichlet_before"]*1.005+1e-10)
        assert _oriented_and_nondegenerate(uv,final.uv_vertices_2d,obj.uv_faces)
    else:
        np.testing.assert_array_equal(final.uv_vertices_2d,uv)


def test_beneficial_sheared_seam_is_actually_refined():
    """Regression against a no-op-only experiment.

    The two 3D triangles share a horizontal source seam. Each separated
    UV copy has a sheared, diagonal seam, so flattening it improves SD.
    """
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,-1.,0.]])
    sf=np.array([[0,1,2],[1,0,3]],int)
    uv=np.array([[0.,.30],[1.,.60],[1.,1.60],
                 [1.,-.05],[0.,-.35],[0.,-1.35]])
    uf=np.array([[0,1,2],[3,4,5]],int)
    obj=OptCutsResult(xyz,sf,uv,uf,[[0,1,2]],"sheared.obj",{})
    baseline=_seam_quality(uv,_seam_sides(obj),8,8)
    assert baseline>0.
    out=refine_optcuts_grid_aware(obj,nx=8,ny=8,strength=3.)
    assert out.metrics["optcuts_grid_aware_status"]=="accepted", out.metrics
    assert out.metrics["optcuts_grid_aware_alignment_after"]<baseline
    assert quality(out)["injectivity_overlap_pairs"]==0
    np.testing.assert_array_equal(out.uv_faces,uf)
    np.testing.assert_array_equal(out.surface_faces,sf)


def test_no_seam_rejects_without_touching_original_geometry():
    obj=_two_patches()
    obj.uv_vertices_2d[3]=obj.uv_vertices_2d[0]
    obj.uv_vertices_2d[4]=obj.uv_vertices_2d[2]
    assert _seam_sides(obj)==[]
    result=refine_optcuts_grid_aware(obj,nx=8,ny=8)
    assert result.metrics["optcuts_grid_aware_status"]=="no_internal_seam_unchanged"
    np.testing.assert_array_equal(result.uv_vertices_2d,obj.uv_vertices_2d)


def test_grid_aware_requires_legal_strength_and_grid_range():
    obj=_two_patches()
    with pytest.raises(ValueError,match="grid-aware"):
        refine_optcuts_grid_aware(obj,nx=0,ny=8)
    with pytest.raises(ValueError,match="strength"):
        refine_optcuts_grid_aware(obj,nx=8,ny=8,strength=-1)


def test_off_path_does_not_call_grid_refiner(monkeypatch):
    from onestring_physics import optcuts_backend as backend
    obj=_two_patches()
    def fake_official(*args,**kwargs):
        return replace(obj,metrics={})
    monkeypatch.setattr(backend,"_run_official_optcuts_once",fake_official)
    def forbidden(*args,**kwargs):
        raise AssertionError("grid-aware optimizer invoked with OFF")
    monkeypatch.setattr(backend,"_refine_uv_for_grid_aware",forbidden)
    cfg=OptCutsConfig(csf_tail_refine=False,grid_aware_enabled=False,
                      try_alternative_initial_cut=False)
    # Baseline still visits the wrapper; the wrapper must return immediately.
    monkeypatch.setattr(backend,"_refine_uv_for_grid_aware",
        lambda result,cfg: result if not cfg.grid_aware_enabled else forbidden())
    got=run_official_optcuts(obj.surface_vertices_3d,obj.surface_faces,cfg)
    np.testing.assert_array_equal(got.uv_vertices_2d,obj.uv_vertices_2d)
    assert "optcuts_grid_aware_status" not in got.metrics


def test_param_config_and_ui_are_only_optcuts_nonpaper():
    from types import SimpleNamespace
    from onestring_physics.optcuts_pipeline_patch import _config_from_params
    grid=SimpleNamespace(nx=14,ny=15)
    p=SimpleNamespace(omega_parameterization_mode="optcuts",
                      paper_fidelity_mode=False,optcuts_grid_aware_enabled=True,
                      optcuts_grid_aware_strength=4.)
    c=_config_from_params(p,grid)
    assert c.grid_aware_enabled is True
    assert (c.grid_aware_nx,c.grid_aware_ny,c.grid_aware_strength)==(14,15,4.)
    p.paper_fidelity_mode=True
    assert not _config_from_params(p,grid).grid_aware_enabled
    p.paper_fidelity_mode=False
    p.omega_parameterization_mode="optcuts_grid"
    assert not _config_from_params(p,grid).grid_aware_enabled
    p.omega_parameterization_mode="paper_reference_bff"
    assert not _config_from_params(p,grid).grid_aware_enabled

    root=Path(__file__).parents[1]
    app=(root/"app_runtime_base.py").read_text()
    pipe=(root/"src/onestring_physics/onestring_pipeline.py").read_text()
    assert 'key="optcuts_grid_aware_experiment"' in app
    assert 'optcuts_grid_aware_enabled=optcuts_grid_aware_enabled' in app
    assert "optcuts_grid_aware_enabled: bool = False" in pipe
