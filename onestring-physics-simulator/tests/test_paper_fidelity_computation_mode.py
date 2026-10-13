"""Regression: paper mode changes REAL pipeline parameters and Split logic.

Tests do not claim the paper's original implementation or exact BFF output.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.onestring_pipeline import PipelineParameters
from onestring_physics.paper_fidelity_mode import (
    MODE_ID, configure_paper_fidelity, check_paper_fidelity,
)
from onestring_physics.paper_mesh_splitting import _component_area_span


def test_reference_preset_is_independent_and_sets_actual_compute_options():
    original=PipelineParameters(
        omega_parameterization_mode="bff",csf_split_threshold=3.1,
        max_csf_splits=2,collision_split_enabled=True,
        collision_split_lookahead_enabled=True,
        hinge_layout_collision_mode="soft",hinge_layout_iterations=40,
        hinge_layout_time_budget_sec=5.,
        t3d_variable_topology_enabled=True,
        allow_legacy_normal_prism_emergency_fallback=True,
    )
    selected,overrides=configure_paper_fidelity(original)
    assert original.omega_parameterization_mode=="bff"
    assert original.collision_split_enabled
    assert selected is not original
    assert selected.paper_fidelity_mode
    assert selected.omega_parameterization_mode=="paper_reference_bff"
    assert selected.bff_boundary_policy=="target_rectangle"
    assert selected.csf_split_threshold==2.0
    assert selected.max_csf_splits>=32
    assert selected.hinge_layout_collision_mode=="hard"
    assert selected.hinge_layout_iterations>=700
    assert selected.hinge_layout_time_budget_sec>=30.
    assert selected.localize_csf_splits is False
    assert selected.collision_split_enabled is False
    assert selected.collision_split_forced_lines==()
    assert not selected.t3d_variable_topology_enabled
    assert not selected.allow_legacy_normal_prism_emergency_fallback
    assert "omega_parameterization_mode" in overrides
    assert "paper_fidelity_mode" in overrides


def test_split_candidate_area_expansion_is_componentwise_not_global_sqrt():
    # Mixed area-Jacobian range: min=1, max=5 exceeds sigma=2.
    # A complete cut separating both regimes permits independent scale
    # normalization with area expansion 1 in each component.
    lo=np.array([1.,1.1,4.6,4.9])
    hi=np.array([1.2,1.4,5.,5.3])
    combined=_component_area_span(lo,hi,[0,1,2,3])
    after=max(_component_area_span(lo,hi,[0,1]),
              _component_area_span(lo,hi,[2,3]))
    assert combined>4.
    assert after<2.
    # This potential reduction does NOT rescale actual M2D UV coordinates.


def test_split_candidates_do_not_accept_nonphysical_jacobians():
    with pytest.raises(ValueError,match="Jacobian"):
        _component_area_span([0.,1.],[2.,3.],[0,1])
    with pytest.raises(ValueError,match="Jacobian"):
        _component_area_span([1.],[float("nan")],[0])


def _state(*,pairs=0,solid=0,hard=True,valid=True,split=False):
    return SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(metrics={
            "csf_split_bound_certified":not split,
            "csf_split_candidate_area_bound_satisfied":True,
        }),
        tiles_2d_dual_hinge=SimpleNamespace(metrics={
            "dual_hinge_final_collision_count":pairs,
            "solid_collision_pair_count":solid,
            "hard_collision_feasible":hard,
        }),
        string_path=SimpleNamespace(metrics={
            "string_path_model":"Sec. 5.3 minimum-turn closed-cycle MILP",
            "string_path_valid":valid,
        }),
    )


def test_reference_report_never_claims_full_fidelity_for_approximations():
    params,_=configure_paper_fidelity(PipelineParameters())
    report=check_paper_fidelity(_state(split=True),params)
    assert report.mode==MODE_ID
    assert not report.verified
    assert report.stages["Split"]=="uncertified"
    assert any("UV scaling" in x or "UV scaling" in x for x in report.reasons)
    assert report.stages["T2D contact"]=="no detected flat SAT intersections"
    assert report.stages["T2D solid contact"]=="zero detected solid overlaps"


def test_reference_reports_overlaps_even_if_flat_sat_says_zero():
    params,_=configure_paper_fidelity(PipelineParameters())
    report=check_paper_fidelity(_state(pairs=0,solid=7,valid=False,hard=False),params)
    assert report.stages["T2D solid contact"]=="failed"
    assert report.stages["T2D feasible"]=="not demonstrated"
    assert report.stages["String route"]=="not demonstrated"
    assert not report.verified


def test_reference_guard_rejects_legacy_parameters_in_core_pipeline():
    from onestring_physics.onestring_pipeline import build_onestring_design
    params=PipelineParameters(paper_fidelity_mode=True,
                              omega_parameterization_mode="bff")
    with pytest.raises(ValueError,match="paper_reference_bff"):
        build_onestring_design(object(),params)
