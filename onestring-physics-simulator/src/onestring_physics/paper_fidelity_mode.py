"""OneString 2025 paper-oriented, validated *computational preset*.

The reference is Zaman et al., TOG 44(6) 2025, Secs. 4.2–5.4,
Supplement Fig.18 and Appendix E.

This is intentionally separate from the original experimental UI presets.
It enables the stages currently closest to the publication, but DOES NOT
claim identical meshes, boundary conditions, optimizer weights, IPC, or
a literal 3D channel. A failed paper condition is reported, never silently
reclassified as a successful paper reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real


MODE_ID = "paper_reference_2025_strict"


@dataclass(frozen=True)
class FidelityOutcome:
    mode: str
    stages: dict[str, str]
    reasons: tuple[str, ...]
    verified: bool


def configure_paper_fidelity(params):
    """Copy the user's parameters and select paper-oriented stages explicitly.

    This MUST be called before the costly pipeline, never retroactively.
    The hard T2D nonpenetration refinement is an implementation-specific
    feasibility safeguard, not the paper authors' exact Eq.(6) optimizer.
    """
    p=replace(params)
    overrides={
        "paper_fidelity_mode": True,
        "omega_parameterization_mode": "paper_reference_bff",
        "omega_boundary_mode": "paper_default",
        "bff_boundary_policy": "target_rectangle",
        "m2d_crop_policy": "strict_vertices",
        # Split uses exactly the normal OFF algorithm: same CSF estimator,
        # component selection, curvature ranking, improvement checks,
        # stopping conditions, threshold and budget. Only non-Split Paper
        # stages (e.g. rectangular BFF) have a different computation preset.
        "collision_split_enabled": False,
        "collision_split_lookahead_enabled": False,
        "collision_split_forced_lines": (),
        "hinge_layout_collision_mode": "hard",
        "hinge_layout_search_mode": "multi_start",
        "hinge_layout_restart_count": max(6,int(getattr(params,"hinge_layout_restart_count",6))),
        "hinge_layout_iterations": max(700,int(getattr(params,"hinge_layout_iterations",700))),
        "hinge_layout_time_budget_sec": max(30.,float(getattr(params,"hinge_layout_time_budget_sec",30.))),
        "hinge_layout_max_candidate_pairs": max(
            6000,int(getattr(params,"hinge_layout_max_candidate_pairs",6000))
        ),
        "hinge_layout_initial_expansion": 1.0,
        "hinge_layout_anchor_weight": 0.0,
        "reference_stop_on_required_split": False,
        "t3d_variable_topology_enabled": False,
        "allow_legacy_normal_prism_emergency_fallback": False,
        "t3d_intersection_trim_enabled": False,
        "use_curvature_adaptive_grid": False,
        "use_extrusion_aware_split": False,
        "use_extrusion_aware_k3d": False,
    }
    for name,value in overrides.items():
        setattr(p,name,value)
    return p,overrides


def check_paper_fidelity(state, params) -> FidelityOutcome:
    """Return a conservative status; approximate feasibility never becomes verified."""
    if not getattr(params,"paper_fidelity_mode",False):
        raise ValueError("Paper fidelity report requires the actual reference-mode pipeline")
    m=getattr(getattr(state,"mesh_2d_initial",None),"metrics",{}) or {}
    t=getattr(getattr(state,"tiles_2d_dual_hinge",None),"metrics",{}) or {}
    stages={}
    reasons=[]
    # BFF is a REQUIREMENT, not inferred from a misleading 'bff' alias.
    omega=str(getattr(params,"omega_parameterization_mode",""))
    rectangle=str(getattr(params,"bff_boundary_policy",""))=="target_rectangle"
    stages["Omega"]=(
        "official BFF with prescribed rectangular boundary requested"
        if omega=="paper_reference_bff" and rectangle else "invalid"
    )
    if not rectangle:
        reasons.append("Paper rectangle mode requires four BFF boundary exterior angles")
    if omega!="paper_reference_bff":
        reasons.append("Official BFF parameterization not selected")
    if "paper_reference_bff" not in omega:
        reasons.append("Check that external BFF completed without fallback")
    if not m.get("csf_split_bound_certified",False):
        stages["Split"]="uncertified"
        reasons.append(
            "Complete curvature-guided cuts do not themselves certify "
            "the physical 1-to-2 expansion bound; per-piece UV scaling "
            "or conformal remapping still requires geometric verification."
        )
    else:
        stages["Split"]="certified by recorded chart metrics"
    collisions=t.get("dual_hinge_final_collision_count",
                     t.get("flat_collision_count",None))
    if collisions is None:
        stages["T2D contact"]="unknown"
        reasons.append("T2D SAT contact result was not recorded")
    elif not isinstance(collisions,Real) or not isfinite(float(collisions)):
        stages["T2D contact"]="invalid"
        reasons.append("T2D collision result is nonfinite")
    elif int(collisions)>0:
        stages["T2D contact"]="failed"
        reasons.append(f"{int(collisions)} colliding pairs remain in the flat layout")
    else:
        stages["T2D contact"]="no detected flat SAT intersections"
    solids=t.get("solid_collision_pair_count",None)
    if solids is None:
        stages["T2D solid contact"]="unmeasured"
        reasons.append("Thick solid 3D contact audit was not provided")
    elif isinstance(solids,Real) and isfinite(float(solids)) and float(solids)==0.:
        stages["T2D solid contact"]="zero detected solid overlaps"
    else:
        stages["T2D solid contact"]="failed"
        reasons.append(f"Thick T2D solid collision audit found {solids} overlaps")
    hard=t.get("hard_collision_feasible",False)
    if not hard:
        stages["T2D feasible"]="not demonstrated"
        reasons.append("Hard nonpenetration refinement did not certify the final flat layout")
    else:
        stages["T2D feasible"]="flat-footprint feasible"
    path=getattr(state,"string_path",None)
    route=dict(getattr(path,"metrics",{}) or {})
    if (route.get("string_path_model")=="Sec. 5.3 minimum-turn closed-cycle MILP"
        and route.get("string_path_valid") is True):
        stages["String route"]="Sec.5.3 discrete MILP valid"
    else:
        stages["String route"]="not demonstrated"
        reasons.append("Paper Sec.5.3 closed-cycle MILP route is not valid")
    stages["Paper PD"]="available separately (requires animation run)"
    reasons.append(
        "Exact numerical settings, physical channel geometry and "
        "continuous collision guarantees have not been validated against the publication."
    )
    return FidelityOutcome(
        mode=MODE_ID,
        stages=stages,
        reasons=tuple(reasons),
        verified=False,
    )
