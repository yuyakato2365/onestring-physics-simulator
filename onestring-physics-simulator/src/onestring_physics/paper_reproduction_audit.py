"""Nonmutating validation of OneString paper-baseline prerequisites.

This is deliberately a diagnostic, NOT a substitute for the Sec.4.5
conformal reparameterization or Sec.4.4 exact collision optimization.
Never infer that a plotted flat layout is fabricable from a soft penalty.
"""
from __future__ import annotations

import math


def audit_paper_reproduction(state, params) -> dict:
    mesh=getattr(state,"mesh_2d_initial",None)
    split=dict(getattr(mesh,"metrics",{}) or {})
    dual=getattr(getattr(state,"tiles_2d_dual_hinge",None),"metrics",{}) or {}
    domain=getattr(state,"conformal_domain",None)
    omega_mode=str(getattr(params,"omega_parameterization_mode","unknown"))
    threshold=float(split.get("csf_split_threshold",
                              getattr(params,"csf_split_threshold",2.)))
    n_csf=int(split.get("csf_split_step_count",
                        len(split.get("csf_split_step_analysis",[]) or [])))
    n_collision=int(split.get("collision_split_count",0))
    sigma_before=split.get("max_csf_before_split")
    sigma_after=split.get("max_csf_after_split")
    exactness=str(split.get("csf_split_exactness_label",""))
    split_status=str(split.get("csf_split_status","unknown"))
    collision_count=dual.get("dual_hinge_final_collision_count",
                             dual.get("flat_collision_count"))
    hinge=dual.get("hinge_connection_error",None)
    collision_mode=str(dual.get("dual_hinge_collision_constraint_mode",
                               getattr(params,"hinge_layout_collision_mode","unknown")))
    warnings=[]
    if omega_mode in ("bff","rectangular_harmonic_legacy"):
        warnings.append({
            "code":"NOT_OFFICIAL_BFF",
            "stage":"S → Ω",
            "detail":(
                "The selected 'bff' is a historical alias of rectangular "
                "cotangent harmonic flattening, NOT the official BFF CLI. "
                "Use 'paper_reference_bff' and verify its boundary policy."
            ),
        })
    elif omega_mode!="paper_reference_bff":
        warnings.append({
            "code":"NON_REFERENCE_OMEGA",
            "stage":"S → Ω",
            "detail":"The current Omega parameterization mode is not the official paper-reference BFF route.",
        })
    if not exactness or ("no reparameterization" in exactness
                          or "not Fig. 6 reproduction" in exactness):
        warnings.append({
            "code":"FIXED_CHART_CSF_SPLIT",
            "stage":"M2D Split (Sec. 4.5)",
            "detail":(
                "Current cuts disconnect M2D, but local CSF is read from the "
                "original fixed UV chart. No post-cut conformal map/refit is "
                "performed; do NOT interpret this as Fig. 6 CSF reduction."
            ),
        })
    if (isinstance(sigma_after,(float,int))
            and math.isfinite(float(sigma_after))
            and sigma_after > threshold+1e-8):
        warnings.append({
            "code":"CSF_BOUND_UNSATISFIED",
            "stage":"M2D Split (Sec. 4.5)",
            "detail":f"Residual CSF={sigma_after:.5g} exceeds limit {threshold:.5g}; "
                     f"CSF Split status: {split_status}.",
        })
    if (n_csf and isinstance(sigma_before,(float,int))
            and isinstance(sigma_after,(float,int))
            and math.isfinite(float(sigma_before))
            and math.isfinite(float(sigma_after))
            and abs(float(sigma_before)-float(sigma_after))
                <= 1e-9*max(1.,abs(float(sigma_before)))):
        warnings.append({
            "code":"SPLIT_DID_NOT_REDUCE_REPORTED_CSF",
            "stage":"M2D Split (Sec. 4.5)",
            "detail":(
                f"{n_csf} complete CSF split(s) were added, but the reported "
                f"maximum CSF stayed {float(sigma_before):.5g} → "
                f"{float(sigma_after):.5g}. This is expected for a fixed map "
                "and is not evidence of the paper's claimed scale reduction."
            ),
        })
    if isinstance(collision_count,(float,int)) and collision_count>0:
        warnings.append({
            "code":"T2D_NONPENETRATION_UNSATISFIED",
            "stage":"T2D Dual Hinge (Eq. 6)",
            "detail":f"{int(collision_count)} final 2D footprint SAT collision pair(s) remain.",
        })
    if collision_mode=="soft":
        warnings.append({
            "code":"SOFT_COLLISION_NO_FEASIBILITY_GUARANTEE",
            "stage":"T2D Dual Hinge (Eq. 6)",
            "detail":"Soft L-BFGS-B penalties are not a certificate of nonpenetrating thick panels.",
        })
    if n_collision:
        warnings.append({
            "code":"COLLISION_SPLIT_IS_PROJECT_EXTENSION",
            "stage":"M2D Split",
            "detail":(
                f"{n_collision} collision-driven split(s) were introduced "
                "by a project-specific T2D-feedback procedure. Sec. 4.5 "
                "uses high Gaussian curvature / CSF, not collision-SAT ranking."
            ),
        })
    return {
        "verdict":("NOT_PAPER_REPRODUCTION" if warnings else "NO_OBVIOUS_CONFIG_MISMATCH_NOT_VERIFIED"),
        "omega_mode":omega_mode,
        "csf_limit":threshold,
        "csf_before":sigma_before,
        "csf_after":sigma_after,
        "csf_split_count":n_csf,
        "collision_split_count":n_collision,
        "csf_split_status":split_status,
        "csf_model":split.get("csf_model"),
        "csf_split_exactness_label":exactness or "missing",
        "collision_mode":collision_mode,
        "t2d_collision_count":collision_count,
        "t2d_hinge_rms":hinge,
        "warnings":warnings,
        "source_mesh_present":bool(mesh is not None),
    }
