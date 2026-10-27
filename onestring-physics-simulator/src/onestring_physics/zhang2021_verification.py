"""Phase 3 / 4 of the 2026-10-27 Zhang2021 mode: downstream verification (OneString).

* K3D gate (called right after K3D, both downstream routes): planar panels
  must pass the K3D size/shape gate and every OptCuts seam must be closed
  exactly (hard mode, constraint residual ~ 0).  A soft or open seam is a
  failure, never a warning.
* Final verification after T3D / K2D / T2D: seam still closed in T3D, no T3D
  panel collision, T2D collision free, hinge connection error below a
  scale-free limit.
* Candidate loop: the pipeline is re-run with the next quality-ranked
  feasible candidate (OptCuts and the candidate evaluation are cached) until
  one passes; with ``zhang2021_compare_verified_candidates`` all attempts are
  run and the best final quality is adopted (Phase 4).  If none passes the
  run fails explicitly with the full table and a diagnostics folder.
"""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np


def _mean_edge(vertices: np.ndarray, faces: np.ndarray) -> float:
    X = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    if not len(f):
        return 0.0
    e = np.concatenate([f[:, [k, (k + 1) % 4]] for k in range(4)])
    e = np.unique(np.sort(e, axis=1), axis=0)
    return float(np.mean(np.linalg.norm(X[e[:, 1]] - X[e[:, 0]], axis=1)))


def seam_closure_report(vertices: np.ndarray, faces: np.ndarray, metrics: dict) -> dict:
    """Measured OptCuts seam closure of a 3D result (absolute and relative)."""
    from .optcuts_seam_soft_constraints import residual_stats

    rows = list(metrics.get("seam_exact_affine_rows") or [])
    groups = list(metrics.get("seam_exact_vertex_groups") or [])
    mode = str(metrics.get("seam_stitch_mode", "none"))
    l_ref = max(_mean_edge(vertices, faces), 1e-300)
    out = {"seam_mode": mode, "seam_rows": len(rows), "seam_vertex_groups": len(groups),
           "seam_gap_max": 0.0, "seam_gap_rms": 0.0, "reference_edge_length": l_ref}
    X = np.asarray(vertices, float)
    if rows:
        st = residual_stats(X, rows)
        out["seam_gap_max"], out["seam_gap_rms"] = float(st["max"]), float(st["rms"])
    if groups:
        gap = max((float(np.max(np.linalg.norm(X[np.asarray(g, int)] - X[int(g[0])], axis=1))) for g in groups),
                  default=0.0)
        out["seam_gap_max"] = max(out["seam_gap_max"], gap)
    out["seam_gap_max_relative"] = out["seam_gap_max"] / l_ref
    out["seam_gap_rms_relative"] = out["seam_gap_rms"] / l_ref
    return out


def k3d_gate(out: Any, params: Any, *, route: str, raise_on_failure: bool = True) -> dict:
    from .optcuts_zhang2021_quad_20261027 import Zhang2021QuadModeError, stage_thresholds
    from .zhang2021_panel_quality import panel_quality

    X = np.asarray(out.vertices, float)
    F = np.asarray(out.faces, int)
    rep = panel_quality(X, F, stage="K3D", thresholds=stage_thresholds(params, "k3d"))
    seam = seam_closure_report(X, F, out.metrics)
    tol = float(getattr(params, "zhang2021_seam_closure_tolerance", 1e-6))
    has_cut = bool(seam["seam_rows"] or seam["seam_vertex_groups"])
    preflight_cut = str(out.metrics.get("zhang2021_seam_preflight_mode", "")) not in ("", "no_cut")
    failures = [f"K3D quality: {f}" for f in rep.failures]
    if has_cut or preflight_cut:
        if seam["seam_mode"] != "hard":
            failures.append(f"OptCuts seam not hard-closed (mode={seam['seam_mode']}); soft closure is never "
                            "accepted")
        if seam["seam_gap_max_relative"] > tol:
            failures.append(f"OptCuts seam gap {seam['seam_gap_max']:.3g} = {seam['seam_gap_max_relative']:.3g} x "
                            f"mean edge > {tol:g}")
    metrics = {**rep.prefixed("zhang2021_k3d"),
               **{f"zhang2021_k3d_{k}": v for k, v in seam.items()},
               "zhang2021_k3d_gate_route": route,
               "zhang2021_k3d_gate_passed": not failures,
               "zhang2021_k3d_gate_failures": failures}
    out.metrics.update(metrics)
    print(f"[ZHANG2021-K3D-GATE] route={route} passed={not failures} seam={seam['seam_mode']} "
          f"gap={seam['seam_gap_max']:.3g} (rel {seam['seam_gap_max_relative']:.3g}) "
          f"min/max angle={rep.metrics['angle_min_deg']:.1f}/{rep.metrics['angle_max_deg']:.1f} "
          f"edge ratio={rep.metrics['edge_ratio_max']:.2f} " + ("; ".join(failures)[:600] if failures else ""),
          flush=True)
    if failures and raise_on_failure:
        raise Zhang2021QuadModeError("ZHANG2021_K3D_GATE_FAILED: " + "; ".join(failures))
    return metrics


def final_verification(state: Any, params: Any) -> dict:
    """K3D / T3D / K2D / T2D checks on a finished run -> {'passed', 'checks', 'score', ...}."""
    checks: list[dict] = []

    def add(stage, name, ok, value, limit):
        checks.append({"stage": stage, "check": name, "ok": bool(ok), "value": value, "limit": limit})

    k3d = getattr(state, "mesh_3d_optimized", None)
    t3d = getattr(state, "tiles_3d", None)
    k2d = getattr(state, "mesh_2d_optimized", None)
    t2d = getattr(state, "tiles_2d_dual_hinge", None)
    km = dict(getattr(k3d, "metrics", {}) or {})
    add("K3D", "K3D gate (quality + hard seam closure)", km.get("zhang2021_k3d_gate_passed", False),
        "; ".join(km.get("zhang2021_k3d_gate_failures", [])) or "passed", "all limits")
    l_ref = _mean_edge(k3d.vertices, k3d.faces) if k3d is not None else 1.0
    tol = float(getattr(params, "zhang2021_seam_closure_tolerance", 1e-6))
    tm = dict(getattr(t3d, "metrics", {}) or {})
    if t3d is None:
        add("T3D", "T3D present", False, "missing", "present")
    else:
        if km.get("zhang2021_k3d_seam_rows") or km.get("zhang2021_k3d_seam_vertex_groups"):
            mode = str(tm.get("t3d_seam_stitch_mode", "none"))
            add("T3D", "seam mode", mode == "hard", mode, "hard")
            gap = max(float(tm.get("t3d_seam_top_gap_max", 0.0) or 0.0),
                      float(tm.get("t3d_seam_bottom_gap_max", 0.0) or 0.0),
                      float(tm.get("t3d_seam_affine_max_residual", 0.0) or 0.0))
            add("T3D", "seam gap / mean edge", gap / max(l_ref, 1e-300) <= tol, gap / max(l_ref, 1e-300), tol)
        coll = tm.get("paper_t3d_panel_collision_count")
        limit = int(getattr(params, "zhang2021_t3d_max_panel_collisions", 0))
        if coll is not None:
            add("T3D", "panel collisions", int(coll) <= limit, int(coll), limit)
    if k2d is None:
        add("K2D", "K2D present", False, "missing", "present")
    else:
        add("K2D", "K2D panel count", len(np.asarray(k2d.faces)) == len(np.asarray(k3d.faces)) if k3d is not None else False,
            int(len(np.asarray(k2d.faces))), "= K3D panels")
    hm = dict(getattr(t2d, "metrics", {}) or {})
    if t2d is None:
        add("T2D", "T2D present", False, "missing", "present")
    else:
        coll = hm.get("collision_pair_count", hm.get("dual_hinge_final_collision_count"))
        add("T2D", "2D collisions", coll is not None and int(coll) == 0, coll, 0)
        if "solid_collision_pair_count" in hm:
            add("T2D", "3D solid collisions", int(hm["solid_collision_pair_count"]) == 0,
                int(hm["solid_collision_pair_count"]), 0)
        if "hard_collision_status" in hm:
            add("T2D", "hard non-penetration", hm["hard_collision_status"] == "verified_feasible",
                hm["hard_collision_status"], "verified_feasible")
        hinge = hm.get("hinge_max", hm.get("hinge_connection_max_error", hm.get("hinge_connection_error")))
        ratio = float(getattr(params, "zhang2021_t2d_max_hinge_error_ratio", 0.05))
        if hinge is not None:
            add("T2D", "hinge connection error / mean edge", float(hinge) / max(l_ref, 1e-300) <= ratio,
                float(hinge) / max(l_ref, 1e-300), ratio)
    passed = all(c["ok"] for c in checks)
    # Phase 4 score: K3D panel quality (no count term) + relative hinge error
    from .zhang2021_candidate_selection import parse_weights, quality_terms
    w = parse_weights(getattr(params, "zhang2021_selection_weights", ""))
    t = quality_terms({k[len("zhang2021_k3d_"):]: v for k, v in km.items() if k.startswith("zhang2021_k3d_")})
    hinge_rel = next((c["value"] for c in checks if c["check"].startswith("hinge")), 0.0)
    score = float(sum(w[k] * t[k] for k in ("size", "shape", "area", "worst")) + float(hinge_rel or 0.0))
    return {"passed": bool(passed), "checks": checks, "final_score": score}


def _stage_of(exc: BaseException) -> str:
    text = str(exc)
    for key, stage in (("K3D", "K3D"), ("T3D", "T3D"), ("K2D", "K2D"), ("T2D", "T2D"), ("SEAM", "seam"),
                       ("M2D", "M2D")):
        if key in text[:120]:
            return stage
    return "pipeline"


def install_zhang2021_verification_loop(pipeline: Any) -> None:
    """Wrap ``build_onestring_design``; inactive unless the Zhang2021 mode is selected."""
    from .optcuts_zhang2021_quad_20261027 import (
        Zhang2021CandidatesExhausted, Zhang2021QuadModeError, mode_enabled)

    if getattr(pipeline, "_zhang2021_verification_loop_installed", False):
        return
    inner = pipeline.build_onestring_design

    def zhang2021_build_onestring_design(target, params=None, run_simulation=False, deployment_params=None,
                                         progress_callback=None):
        if params is None or not mode_enabled(params):
            return inner(target, params, run_simulation=run_simulation, deployment_params=deployment_params,
                         progress_callback=progress_callback)
        from .optcuts_zhang2021_quad_20261027 import structured_grid_enabled
        if structured_grid_enabled(params):
            return _structured_single_run(inner, target, params, run_simulation, deployment_params,
                                          progress_callback)
        attempts = max(1, int(getattr(params, "zhang2021_verification_attempts", 3)))
        compare = bool(getattr(params, "zhang2021_compare_verified_candidates", False))
        cache: dict = {}
        log: list[dict] = []
        passed: list[tuple[float, int, Any]] = []
        last_state = None
        for rank in range(attempts):
            p = copy.copy(params)
            p.zhang2021_candidate_rank = rank
            p._zhang2021_run_cache = cache
            p._zhang2021_progress = progress_callback
            started = time.perf_counter()
            try:
                state = inner(target, p, run_simulation=run_simulation, deployment_params=deployment_params,
                              progress_callback=progress_callback)
            except Zhang2021CandidatesExhausted:
                break
            except Zhang2021QuadModeError as exc:
                if "ZHANG2021_NO_FEASIBLE_CANDIDATE" in str(exc):
                    raise
                log.append({"rank": rank, "status": "failed", "stage": _stage_of(exc), "error": str(exc)[:1500],
                            "seconds": time.perf_counter() - started})
                print(f"[ZHANG2021-VERIFY] candidate rank {rank} failed downstream: {str(exc)[:400]}", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001 - any downstream failure rejects THIS candidate
                log.append({"rank": rank, "status": "failed", "stage": _stage_of(exc),
                            "error": f"{type(exc).__name__}: {str(exc)[:1500]}",
                            "seconds": time.perf_counter() - started})
                print(f"[ZHANG2021-VERIFY] candidate rank {rank} failed downstream: "
                      f"{type(exc).__name__}: {str(exc)[:400]}", flush=True)
                continue
            last_state = state
            verdict = final_verification(state, p)
            row = {"rank": rank, "status": "passed" if verdict["passed"] else "rejected",
                   "final_score": verdict["final_score"], "checks": verdict["checks"],
                   "seconds": time.perf_counter() - started,
                   "quads": int(len(np.asarray(state.mesh_2d_initial.faces)))}
            log.append(row)
            bad = [f"{c['stage']}:{c['check']}={c['value']} (limit {c['limit']})" for c in verdict["checks"] if not c["ok"]]
            print(f"[ZHANG2021-VERIFY] candidate rank {rank}: {'PASSED' if verdict['passed'] else 'REJECTED'} "
                  + ("; ".join(bad)[:600] if bad else f"final score {verdict['final_score']:.4g}"), flush=True)
            if verdict["passed"]:
                passed.append((verdict["final_score"], rank, state))
                if not compare:
                    break
        if passed:
            passed.sort(key=lambda t: (t[0], t[1]))
            score, rank, state = passed[0]
            summary = {"zhang2021_verification_log": log, "zhang2021_verification_adopted_rank": int(rank),
                       "zhang2021_verification_adopted_final_score": float(score),
                       "zhang2021_verification_attempts_run": int(len(log)),
                       "zhang2021_all_constraints_satisfied": True}
            for name in ("surface_parameterization", "mesh_2d_initial", "mesh_3d_optimized", "tiles_2d_dual_hinge"):
                obj = getattr(state, name, None)
                if obj is not None and isinstance(getattr(obj, "metrics", None), dict):
                    obj.metrics.update(summary)
            return state
        from .zhang2021_diagnostics import write_verification_diagnostics
        where = write_verification_diagnostics(log, last_state, params)
        lines = []
        for r in log:
            if r["status"] == "failed":
                lines.append(f"rank {r['rank']}: failed at {r['stage']}: {r['error'][:400]}")
            else:
                bad = [f"{c['stage']} {c['check']}={c['value']} (limit {c['limit']})" for c in r["checks"] if not c["ok"]]
                lines.append(f"rank {r['rank']}: rejected: " + "; ".join(bad))
        raise Zhang2021QuadModeError(
            f"ZHANG2021_NO_CANDIDATE_PASSED_VERIFICATION after {len(log)} downstream attempt(s) "
            f"(zhang2021_verification_attempts={attempts}); diagnostics: {where}\n" + "\n".join(lines))

    zhang2021_build_onestring_design.__wrapped__ = inner
    pipeline.build_onestring_design = zhang2021_build_onestring_design
    pipeline._zhang2021_verification_loop_installed = True


def _structured_single_run(inner, target, params, run_simulation, deployment_params, progress_callback):
    """2026-10-28 structured grid: one run, every check reported, nothing rejected or retried."""
    p = copy.copy(params)
    p.zhang2021_candidate_rank = 0
    p._zhang2021_run_cache = {}
    p._zhang2021_progress = progress_callback
    started = time.perf_counter()
    state = inner(target, p, run_simulation=run_simulation, deployment_params=deployment_params,
                  progress_callback=progress_callback)
    verdict = final_verification(state, p)
    pm = dict(getattr(getattr(state, "surface_parameterization", None), "metrics", {}) or {})
    m2d = dict(getattr(getattr(state, "mesh_2d_initial", None), "metrics", {}) or {})
    km = dict(getattr(getattr(state, "mesh_3d_optimized", None), "metrics", {}) or {})
    unresolved = list(pm.get("structured_grid_unresolved", []) or []) + list(m2d.get("structured_grid_m2d_unresolved", []) or [])
    for f in km.get("zhang2021_k3d_gate_failures", []) or []:
        if "seam" in f.lower():
            unresolved.append(f)
    for c in verdict["checks"]:
        if not c["ok"] and "seam" in c["check"].lower():
            unresolved.append(f"{c['stage']} {c['check']}={c['value']} (limit {c['limit']})")
    quality_notes = [f"{c['stage']} {c['check']}={c['value']} (limit {c['limit']})"
                     for c in verdict["checks"] if not c["ok"] and "seam" not in c["check"].lower()]
    summary = {
        "structured_grid_run_completed": True,
        "structured_grid_geometrically_verified": not unresolved,
        "structured_grid_unresolved_all": unresolved,
        "structured_grid_quality_notes": quality_notes,
        "structured_grid_final_checks": verdict["checks"],
        "structured_grid_total_seconds": float(time.perf_counter() - started),
    }
    from .zhang2021_diagnostics import write_structured_grid_diagnostics
    summary["structured_grid_diagnostics_dir"] = write_structured_grid_diagnostics(state, summary, p)
    for name in ("surface_parameterization", "mesh_2d_initial", "mesh_3d_optimized", "tiles_2d_dual_hinge"):
        obj = getattr(state, name, None)
        if obj is not None and isinstance(getattr(obj, "metrics", None), dict):
            obj.metrics.update(summary)
    print("[SG-RESULT] run completed; geometrically verified: "
          f"{summary['structured_grid_geometrically_verified']}"
          + (f"; UNRESOLVED: {'; '.join(unresolved)[:800]}" if unresolved else "")
          + (f"; quality (display only): {'; '.join(quality_notes)[:600]}" if quality_notes else "")
          + f"; diagnostics: {summary['structured_grid_diagnostics_dir']}", flush=True)
    return state


__all__ = ["k3d_gate", "final_verification", "seam_closure_report", "install_zhang2021_verification_loop"]
