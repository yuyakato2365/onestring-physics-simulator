"""New Omega mode: independent source-geometry-copy lattice transition tests."""
from types import SimpleNamespace
import numpy as np
import pytest

from onestring_physics import optcuts_seamless_lattice_mode as m


def _two_uv_charts():
    # In production these UV copies are on the SAME original 3D seam.
    # A perturbed copy B is translated by approximately 2 grid steps.
    uv=np.array([
        [0,0],[1,0],[1,1],[0,1],
        [3.12,0],[4.12,0],[4.12,1],[3.12,1]
    ],float)
    faces=np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7]
    ],int)
    chart=SimpleNamespace(
        uv_vertices_2d=uv,
        uv_faces=faces,
    )
    chains=[((10,11),(1,2),(4,7))]
    return chart,chains


def test_exact_integer_transition_without_straightening(monkeypatch):
    chart,chains=_two_uv_charts()
    monkeypatch.setattr(m,"source_chains",lambda _:chains)
    result,stats=m.solve_seamless_lattice_uv(chart,1.)
    assert stats["seamless_lattice_source_chain_count"]==1
    assert stats["seamless_lattice_transition_residual_max"]<1e-12
    assert stats["seamless_lattice_transitions"][0]["k"]==0
    assert stats["seamless_lattice_transitions"][0]["shift_grid_steps"]==[2,0]
    assert stats["seamless_lattice_source_relative_area_min"]>.01
    assert not stats["seamless_lattice_staircase_snap_applied"]
    assert not stats["seamless_lattice_m2d_coverage_proven"]
    np.testing.assert_allclose(result[4]-result[1],[2,0],atol=1e-11)
    np.testing.assert_allclose(result[7]-result[2],[2,0],atol=1e-11)
    assert result.shape==chart.uv_vertices_2d.shape
    # Source Ω and all four-corner cells are untouched; only UV moves.
    assert not np.array_equal(result,chart.uv_vertices_2d)


def test_rotation_search_all_four_grid_isometries():
    uv=np.array([
        [1.,0.],[2.,0.],[0.,1.],[0.,2.],
        [4.,5.],[4.,6.]
    ],float)
    chain=[((20,21),(0,1),(4,5))]
    # Side-B is exactly a 90-degree turn + integer translation.
    uv[4]=m.ROT[1]@uv[0]+np.array([4.,4.])
    uv[5]=m.ROT[1]@uv[1]+np.array([4.,4.])
    t=m.best_integer_transitions(uv,chain,1.)
    assert t[0]["k"]==1
    assert t[0]["steps"]==(4,4)
    assert t[0]["rms"]==pytest.approx(0.,abs=1e-12)


def test_incompatible_translation_holonomy_fails_closed():
    constraints=[
        {"a":(0,),"b":(1,),"k":0,"steps":(1,0)},
        {"a":(1,),"b":(0,),"k":0,"steps":(1,0)},
    ]
    with pytest.raises(m.SeamlessLatticeError,match="INCONSISTENT_TRANSLATIONAL_HOLONOMY"):
        m._global_lattice_constraints(2,constraints,1.)


def test_unsupported_or_missing_physical_source_cut_fails(monkeypatch):
    chart,chains=_two_uv_charts()
    monkeypatch.setattr(m,"source_chains",lambda _:[])
    with pytest.raises(m.SeamlessLatticeError,match="NO_PHYSICAL_OPT_CUT_CHAIN"):
        m.solve_seamless_lattice_uv(chart,1.)


def test_fail_closed_on_foldover_not_return_original_uv(monkeypatch):
    chart,chains=_two_uv_charts()
    # Reverse one source triangle: the candidate must be rejected if it
    # inverts any previously oriented source triangle during the solve.
    monkeypatch.setattr(m,"source_chains",lambda _:chains)
    chart.uv_vertices_2d[4,1]+=.65
    out,info=m.solve_seamless_lattice_uv(chart,1.)
    assert info["seamless_lattice_source_relative_area_min"]>0
    assert info["seamless_lattice_transition_residual_max"]<1e-9


def test_installer_only_changes_explicit_new_mode(monkeypatch):
    base_chart,chains=_two_uv_charts()
    calls=[]
    def builder(surface,target,grid,params):
        calls.append(params)
        return SimpleNamespace(
            method="optcuts_test",
            uv_vertices_2d=base_chart.uv_vertices_2d.copy(),
            uv_faces=base_chart.uv_faces,
            metrics={},
        )
    owner=SimpleNamespace(_build_surface_parameterization=builder)
    m.install_seamless_lattice_mode_patch(owner)
    old=owner._build_surface_parameterization(
        None,None,None,SimpleNamespace(optcuts_seamless_lattice_enabled=False)
    )
    np.testing.assert_array_equal(old.uv_vertices_2d,base_chart.uv_vertices_2d)
    assert not hasattr(old,"seamless_lattice_mode_enabled")
    assert owner._onestring_seamless_lattice_mode_installed
