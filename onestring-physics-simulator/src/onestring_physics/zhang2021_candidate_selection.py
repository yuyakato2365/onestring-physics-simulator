"""Phased candidate selection for the 2026-10-27 Zhang2021 mode (OneString, not the paper).

Phase 1  feasibility (hard constraints, scale free):
         C1 UV bijective (chart injective, every M2D quad simple + CCW)
         C2 valid quad topology (all-quad disk, no doublets)
         C3 S coverage and exact UV<->3D correspondence (chart tiles the quad region)
         C4 OptCuts seams hard-closable in 3D (matched / nested partitions,
            seam nodes with a valid 3D valence) - soft is never feasible
         C6 panel thinness / size spread: independent M2D (validity) and M3D
            (uniformity) gates
         C7 route linkage (the 09-18 route's Sec. 4.3 corner linkage needs even
            interior valence)
         (C5 K3D planarity / non-degeneracy and the T2D part of C7 are checked
         on the real downstream result in Phase 3.)
Phase 2  quality score of the feasible candidates (no panel-count term):
         E(c) = w_size E_size + w_shape E_shape + w_area E_area + w_dist E_distortion
              + w_fit E_surface + w_worst E_worst,  M3D terms + w_m2d * M2D terms
Phase 3  K3D / T3D / K2D / T2D verification (zhang2021_verification)
Phase 4  final-quality comparison of the verified candidates.

Every candidate keeps its rejection reasons; the table is part of the result.
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_SELECTION_WEIGHTS = {
    "size": 1.0,        # E_size   = mean log^2(edge / geometric-mean edge)        (3D)
    "shape": 1.0,       # E_shape  = mean cos^2(angle) + mean log^2(aspect)        (3D)
    "area": 0.5,        # E_area   = (panel area CV)^2                             (3D)
    "distortion": 1.0,  # E_dist   = final chart SD mean / OptCuts SD mean - 1
    "fit": 2.0,         # E_surface= boundary Hausdorff(Q, Omega_0) / bbox diagonal
    "worst": 0.5,       # E_worst  = smooth max of the per-panel deviation         (3D)
    "m2d": 0.1,         # weight of the same terms measured on M2D (UV)
}


def parse_weights(text: Any) -> dict[str, float]:
    out = dict(DEFAULT_SELECTION_WEIGHTS)
    if text is None or (isinstance(text, str) and not text.strip()):
        return out
    if isinstance(text, dict):
        items = text.items()
    else:
        items = []
        for part in str(text).replace(";", ",").split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"ZHANG2021_SELECTION_WEIGHTS_INVALID: {part!r} (expected name=value)")
            k, v = part.split("=", 1)
            items.append((k.strip(), v.strip()))
    for k, v in items:
        if k == "count":
            raise ValueError("ZHANG2021_SELECTION_WEIGHTS_INVALID: there is no panel-count term")
        if k not in out:
            raise ValueError(f"ZHANG2021_SELECTION_WEIGHTS_INVALID: unknown term {k!r}; "
                             f"known: {sorted(out)}")
        val = float(v)
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"ZHANG2021_SELECTION_WEIGHTS_INVALID: {k}={v}")
        out[k] = val
    return out


def quality_terms(report_metrics: dict) -> dict[str, float]:
    m = report_metrics
    return {
        "size": float(m.get("e_size", np.inf)),
        "shape": float(m.get("e_angle", np.inf)) + float(m.get("e_aspect", np.inf)),
        "area": float(m.get("area_cv", np.inf)) ** 2,
        "worst": float(m.get("worst_panel_smooth_max", np.inf)),
    }


def candidate_score(m3d: dict, m2d: dict, distortion: float, fit: float,
                    weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    """E(c) of the module doc; returns (score, weighted terms).  No count term."""
    t3 = quality_terms(m3d)
    t2 = quality_terms(m2d)
    terms = {k: weights[k] * t3[k] for k in ("size", "shape", "area", "worst")}
    terms["distortion"] = weights["distortion"] * max(0.0, float(distortion))
    terms["fit"] = weights["fit"] * max(0.0, float(fit))
    terms["m2d"] = weights["m2d"] * sum(weights[k] * t2[k] for k in ("size", "shape", "area", "worst"))
    score = float(sum(terms.values()))
    return score, terms


def feasibility(plan_metrics: dict, *, route_needs_even_valence: bool) -> list[dict]:
    """Phase 1 hard constraints of one candidate -> [{'constraint', 'ok', 'detail'}]."""
    m = plan_metrics
    out = []

    def add(name, ok, detail):
        out.append({"constraint": name, "ok": bool(ok), "detail": str(detail)})

    add("C1_uv_bijective", m.get("zhang2021_m2d_quads_valid", False)
        and m.get("zhang2021_onestring_chart_injective", False),
        f"chart injective={m.get('zhang2021_onestring_chart_injective')}, "
        f"invalid M2D quads={m.get('zhang2021_m2d_invalid_quads')}")
    add("C2_quad_topology", m.get("zhang2021_m2d_topology_valid", False),
        m.get("zhang2021_m2d_topology_detail", ""))
    add("C3_surface_coverage", float(m.get("zhang2021_onestring_final_chart_area_relative_error", 1.0)) <= 1e-8,
        f"chart/quad-region area error {m.get('zhang2021_onestring_final_chart_area_relative_error')}")
    seam_mode = str(m.get("zhang2021_seam_preflight_mode", "unavailable"))
    seam_ok = seam_mode in ("hard", "no_cut")
    add("C4_seam_hard_closable", seam_ok,
        f"preflight={m.get('zhang2021_seam_preflight_status')}/{seam_mode}; "
        f"nested branches={m.get('zhang2021_onestring_seam_nested_branches')}, "
        f"T-vertices={m.get('zhang2021_onestring_seam_t_vertices')}; "
        f"{m.get('zhang2021_seam_preflight_reason', '')}")
    add("C6_m2d_gate", m.get("zhang2021_m2d_quality_passed", False), "; ".join(m.get("zhang2021_m2d_quality_failures", [])))
    add("C6_m3d_gate", m.get("zhang2021_m3d_quality_passed", False), "; ".join(m.get("zhang2021_m3d_quality_failures", [])))
    odd = int(m.get("zhang2021_odd_valence_interior_vertices", 0))
    add("C7_route_linkage", (odd == 0) or not route_needs_even_valence,
        f"odd-valence interior vertices={odd} (09-18 route needs 0)")
    return out


def summarize_row(row: dict) -> str:
    st = row.get("status", "?")
    head = (f"tol={row.get('precondition_tol')} eps={row.get('zhang_eps')} N_tar={row.get('n_tar')}: "
            f"{st}")
    if row.get("quads") is not None:
        head += f", {row['quads']} quads"
    if row.get("score") is not None:
        head += f", E={row['score']:.4g}"
    if row.get("rejections"):
        head += " | " + "; ".join(row["rejections"])[:400]
    elif row.get("reason"):
        head += " | " + str(row["reason"])[:300]
    return head


__all__ = ["DEFAULT_SELECTION_WEIGHTS", "parse_weights", "candidate_score", "feasibility", "quality_terms",
           "summarize_row"]
