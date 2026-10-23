"""Default K3D AL and final exact-Seam stage must really minimize SmoothMax.

The mode `optcuts_test` is the numerical carrier for the visible
"OptCuts Ω + Area-Threshold Grid Boundary" UI. Do NOT require the older
optional shape-preserving Version selector to activate E_Square.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from onestring_physics import optcuts_test_k3d_augmented_lagrangian_patch as al
from onestring_physics.optcuts_seam_exact_closure import (
    fit_k3d_with_exact_pairs,max_seam_gap,
)


def skewed_pair():
    v=np.array([
        [0.,0.,0.],[1.2,0.,0.],[1.15,1.1,0.],[0.,1.,0.],
        [1.2,0.,0.],[2.5,0.,0.],[2.55,1.2,0.],[1.15,1.1,0.],
    ])
    f=np.array([[0,1,2,3],[4,5,6,7]],int)
    groups=[[1,4],[2,7]]
    return v,f,groups


def test_default_al_uses_actual_smoothmax_square_without_shape_mode(monkeypatch):
    monkeypatch.setenv("ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY","0")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","12")
    v,f,_=skewed_pair()
    baseline=al._square_energy_and_gradient(
        v,f,al._face_scales(v,f),10.
    )[0]
    solved,diag=al._augmented_lagrangian_planarize(
        v,f,anchor_weight=.2,square_weight=10.,
        outer_iterations=3,inner_iterations=80,
        restoration_sweeps=2,final_polish_sweeps=15,
    )
    actual=al._square_energy_and_gradient(
        solved,f,al._face_scales(v,f),10.
    )[0]
    assert not diag["k3d_shape_preserving_mode"]
    assert diag["k3d_global_square_regularization_enabled"] is True
    assert diag["k3d_smoothmax_enabled"] is True
    assert diag["k3d_smoothmax_alpha"]==4.
    assert diag["k3d_smoothmax_beta"]==12.
    assert diag["k3d_smoothmax_weight"]==10.
    assert actual<baseline


def test_square_weight_zero_disables_default_smoothmax(monkeypatch):
    monkeypatch.setenv("ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY","0")
    v,f,_=skewed_pair()
    result,diag=al._augmented_lagrangian_planarize(
        v,f,square_weight=0.,anchor_weight=.5,
        outer_iterations=1,inner_iterations=10,
        restoration_sweeps=1,final_polish_sweeps=0,
    )
    assert diag["k3d_global_square_regularization_enabled"] is False
    assert diag["k3d_smoothmax_enabled"] is False
    assert diag["k3d_smoothmax_weight"]==0.
    np.testing.assert_allclose(result,v,rtol=0,atol=1e-6)


def test_seam_closure_polishes_actual_smoothmax_with_exact_constraints(monkeypatch):
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","12")
    v,f,groups=skewed_pair()
    assert max_seam_gap(v,groups)==0.
    solved,diag=fit_k3d_with_exact_pairs(
        v,f,groups,iterations=3,square_weight=10.
    )
    assert max_seam_gap(solved,groups)<=1e-12
    assert diag["k3d_seam_smoothmax_attempted"] is True
    assert diag["k3d_seam_smoothmax_weight"]==10.
    assert diag["k3d_seam_smoothmax_energy_final"]<=(
        diag["k3d_seam_smoothmax_energy_before"]+1e-12
    )
    assert diag["seam_k3d_gap_after"]<=1e-12
    if diag["k3d_seam_smoothmax_accepted"]:
        assert diag["k3d_seam_smoothmax_energy_final"]<(
            diag["k3d_seam_smoothmax_energy_before"]
        )


def test_seam_smoothmax_is_a_true_energy_optimization_not_only_a_diagnostic(
    monkeypatch
):
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    v,f,groups=skewed_pair()
    result,diag=fit_k3d_with_exact_pairs(
        v,f,groups,iterations=2,square_weight=10.
    )
    assert diag["k3d_seam_smoothmax_accepted"] is True
    assert diag["k3d_seam_smoothmax_energy_final"]<(
        diag["k3d_seam_smoothmax_energy_before"]
    )
    assert diag["k3d_seam_smoothmax_planarity_trial"]<=(
        diag["k3d_seam_smoothmax_planarity_limit"]
    )
    np.testing.assert_allclose(result[1],result[4],atol=1e-12)
    np.testing.assert_allclose(result[2],result[7],atol=1e-12)


def test_smoothmax_beta_zero_continuation_stays_enabled(monkeypatch):
    monkeypatch.setenv("ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY","0")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","0")
    v,f,_=skewed_pair()
    _,diag=al._augmented_lagrangian_planarize(
        v,f,square_weight=3.,outer_iterations=1,
        inner_iterations=10,restoration_sweeps=1,
        final_polish_sweeps=0,
    )
    assert diag["k3d_smoothmax_enabled"]
    assert diag["k3d_smoothmax_beta"]==0.



def test_smoothmax_al_hard_seam_equality_even_from_open_initial_gap(monkeypatch):
    """Prove soft square penalty cannot reopen truly shared seam unknowns."""
    monkeypatch.setenv("ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY","0")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","12")
    v,faces,groups=skewed_pair()
    v[4]+=[-.12,.02,.035]
    v[7]+=[.15,-.01,-.035]
    assert max_seam_gap(v,groups)>.1
    solved,diag=al._augmented_lagrangian_planarize(
        v,faces,anchor_weight=.2,square_weight=10.,
        outer_iterations=3,inner_iterations=90,
        restoration_sweeps=5,final_polish_sweeps=25,
        equality_groups=groups,
    )
    # Hard constraint, exactly identical floats, not mere small penalty.
    np.testing.assert_array_equal(solved[1],solved[4])
    np.testing.assert_array_equal(solved[2],solved[7])
    assert diag["k3d_al_source_seam_hard_equality"] is True
    assert diag["k3d_al_source_seam_group_count"]==2
    assert diag["k3d_al_source_seam_reduced_unknowns"]==6
    assert diag["k3d_al_source_seam_max_gap"]==0.
    assert diag["k3d_smoothmax_enabled"] is True
    assert np.isfinite(solved).all()


def test_renumbered_geometry_view_maps_seam_groups_for_al_not_stale_original():
    from onestring_physics.extrusion_aware import geometry_view
    from types import SimpleNamespace
    v,faces,groups=skewed_pair()
    mesh=SimpleNamespace(
        vertices=v,faces=faces,
        metrics={"seam_exact_preflight_certified":True,
                 "seam_exact_vertex_groups":groups},
    )
    # Merge a nonseam pair; seam endpoints must be re-indexed.
    mapping=np.array([0,1,2,3,4,5,6,7],int)
    mapping[6]=5
    mapping[7]=6
    reduced=geometry_view(mesh,mapping)
    assert len(reduced.vertices)==7
    assert reduced.metrics["seam_exact_vertex_groups"]==[
        [1,4],[2,6]
    ]
    assert reduced.metrics["seam_exact_groups_reindexed_for_reduction"]
    assert mesh.metrics["seam_exact_vertex_groups"]==groups
