"""Preflight must not misrepresent fixed-UV cuts as paper Fig. 6."""
from types import SimpleNamespace

from onestring_physics.paper_reproduction_audit import audit_paper_reproduction


def _state(*,before=4.2,after=4.2,split_count=2,collisions=190,
           hinge=.008,collision_mode="soft",collision_split_count=0):
    return SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(metrics=dict(
            max_csf_before_split=before,
            max_csf_after_split=after,
            csf_split_threshold=2.0,
            csf_split_step_count=split_count,
            csf_split_status="no_admissible_curvature_guided_grid_cut",
            csf_split_exactness_label=(
                "fixed-chart inverse-length CSF; no reparameterization, "
                "not Fig. 6 reproduction"
            ),
            collision_split_count=collision_split_count,
        )),
        tiles_2d_dual_hinge=SimpleNamespace(metrics=dict(
            dual_hinge_final_collision_count=collisions,
            hinge_connection_error=hinge,
            dual_hinge_collision_constraint_mode=collision_mode,
        )),
    )


def test_legacy_bff_and_unchanged_split_csf_report_real_mismatch():
    state=_state()
    params=SimpleNamespace(omega_parameterization_mode="bff")
    result=audit_paper_reproduction(state,params)
    codes={x["code"] for x in result["warnings"]}
    assert result["omega_mode"]=="bff"
    assert result["csf_split_count"]==2
    assert result["csf_before"]==4.2
    assert result["csf_after"]==4.2
    assert result["verdict"]=="NOT_PAPER_REPRODUCTION"
    assert {
        "NOT_OFFICIAL_BFF",
        "FIXED_CHART_CSF_SPLIT",
        "CSF_BOUND_UNSATISFIED",
        "SPLIT_DID_NOT_REDUCE_REPORTED_CSF",
        "T2D_NONPENETRATION_UNSATISFIED",
        "SOFT_COLLISION_NO_FEASIBILITY_GUARANTEE",
    }.issubset(codes)


def test_paper_reference_bff_does_not_silence_unreproduced_csf_cut():
    result=audit_paper_reproduction(
        _state(before=3.5,after=3.5,collisions=0,collision_mode="hard"),
        SimpleNamespace(omega_parameterization_mode="paper_reference_bff"),
    )
    codes={x["code"] for x in result["warnings"]}
    assert "NOT_OFFICIAL_BFF" not in codes
    assert "FIXED_CHART_CSF_SPLIT" in codes
    assert "SPLIT_DID_NOT_REDUCE_REPORTED_CSF" in codes
    assert "SOFT_COLLISION_NO_FEASIBILITY_GUARANTEE" not in codes


def test_collision_split_is_separate_heuristic_not_the_paper_section45():
    result=audit_paper_reproduction(
        _state(split_count=0,collision_split_count=2),
        SimpleNamespace(omega_parameterization_mode="paper_reference_bff"),
    )
    assert result["collision_split_count"]==2
    assert any(x["code"]=="COLLISION_SPLIT_IS_PROJECT_EXTENSION"
               for x in result["warnings"])


def test_missing_optional_metrics_do_not_crash_diagnostic():
    state=SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(metrics={}),
        tiles_2d_dual_hinge=SimpleNamespace(metrics={}),
    )
    result=audit_paper_reproduction(
        state,SimpleNamespace(omega_parameterization_mode="paper_reference_bff")
    )
    assert result["csf_before"] is None
    assert result["t2d_collision_count"] is None
    assert result["csf_split_count"]==0
    assert result["verdict"]=="NOT_PAPER_REPRODUCTION"
