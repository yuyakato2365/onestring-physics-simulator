"""Joint hard-feasibility projection for optcuts_test K2D.

Final K2D must simultaneously satisfy:
1. each tile remains rigid (SE(2) only),
2. all declared vertex hinges coincide within a strict tolerance,
3. no positive-area tile overlap remains.

The ordinary SE(2) K2D solver and collision-only hard projection are retained as
initialization.  This patch then alternates rigid hinge projections with
Gauss-Seidel SAT collision projections.  The collision phase deliberately does
NOT use global centre expansion because that would destroy hinge coincidence.
If the two hard constraints cannot be satisfied simultaneously, the layout is
reported as infeasible instead of silently weakening either constraint.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _rigid_fit_2d(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orientation-preserving R,t minimizing ||source R + t - target||."""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    if len(src) == 0:
        return np.eye(2), np.zeros(2)
    cs = np.mean(src, axis=0)
    cd = np.mean(dst, axis=0)
    a = src - cs
    b = dst - cd
    if len(src) == 1:
        return np.eye(2), cd - cs
    try:
        u, _s, vt = np.linalg.svd(a.T @ b)
        r = u @ vt
        if np.linalg.det(r) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vt
    except Exception:
        r = np.eye(2)
    t = cd - cs @ r
    return r, t


def _hinge_errors(tiles: np.ndarray, constraints: list[tuple[int, int, int, int]]) -> np.ndarray:
    x = np.asarray(tiles, dtype=float)
    vals: list[float] = []
    for ia, ca, ib, cb in constraints:
        if 0 <= ia < len(x) and 0 <= ib < len(x) and 0 <= ca < x.shape[1] and 0 <= cb < x.shape[1]:
            vals.append(float(np.linalg.norm(x[ia, ca] - x[ib, cb])))
    return np.asarray(vals, dtype=float)


def _hard_hinge_sweep(tiles: np.ndarray, constraints: list[tuple[int, int, int, int]]) -> np.ndarray:
    """One simultaneous rigid per-tile projection toward all incident hinge targets."""
    x = np.asarray(tiles, dtype=float)
    if not constraints or len(x) == 0:
        return x.copy()
    src_by_tile: list[list[np.ndarray]] = [[] for _ in range(len(x))]
    dst_by_tile: list[list[np.ndarray]] = [[] for _ in range(len(x))]
    for ia, ca, ib, cb in constraints:
        if not (0 <= ia < len(x) and 0 <= ib < len(x) and 0 <= ca < x.shape[1] and 0 <= cb < x.shape[1]):
            continue
        pa = x[ia, ca].copy()
        pb = x[ib, cb].copy()
        mid = 0.5 * (pa + pb)
        src_by_tile[ia].append(pa)
        dst_by_tile[ia].append(mid)
        src_by_tile[ib].append(pb)
        dst_by_tile[ib].append(mid)

    out = x.copy()
    for tile_id in range(len(x)):
        if not src_by_tile[tile_id]:
            continue
        src = np.asarray(src_by_tile[tile_id], dtype=float)
        dst = np.asarray(dst_by_tile[tile_id], dtype=float)
        r, t = _rigid_fit_2d(src, dst)
        out[tile_id] = x[tile_id] @ r + t
    return out


def _collision_gs_sweep(mod: Any, pipeline: Any, tiles: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """Sequential SAT projection, preserving each tile as a rigid body."""
    x = np.asarray(tiles, dtype=float).copy()
    bad = mod._penetrating_pairs(pipeline, x, tol)
    if not bad:
        return x, 0
    _pair_fn, sat_fn = mod._collision_backend(pipeline)
    scale = max(float(mod._tile_scale(x)), 1e-12)
    moved = 0
    for i, j, _mtv0, _depth0 in sorted(bad, key=lambda item: item[3], reverse=True):
        overlap, mtv, signed = sat_fn(x[i], x[j], clearance=0.0)
        mtv = np.asarray(mtv, dtype=float)
        depth = float(np.linalg.norm(mtv)) if overlap else 0.0
        if not overlap or depth <= tol or float(signed) >= -tol:
            continue
        direction = mtv / max(depth, 1e-15)
        sep = mtv + direction * max(4.0 * tol, 5e-10 * scale)
        x[i] += 0.5 * sep[None, :]
        x[j] -= 0.5 * sep[None, :]
        moved += 1
    return x, moved


def install_optcuts_test_k2d_hard_hinge_patch(pipeline: Any) -> None:
    from . import optcuts_test_k2d_relative_layout_patch as mod

    if getattr(mod, "_onestring_k2d_hard_hinge_installed", False):
        return

    base_build = mod._build_rigid_k2d_layout

    def build_with_joint_hard_constraints(pipeline_obj: Any, mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        solved, metrics = base_build(pipeline_obj, mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        solved = np.asarray(solved, dtype=float)
        faces = np.asarray(mesh_2d.faces, dtype=int)
        constraints = mod._hinge_constraints(pipeline_obj, faces)
        scale = max(float(mod._tile_scale(solved)), 1e-12)
        hinge_tol = max(scale * float(getattr(params, "k2d_hard_hinge_relative_tolerance", 5e-4)), 1e-9)
        collision_tol = max(scale * 1e-10, 1e-13)
        max_outer = max(50, int(getattr(params, "k2d_hard_joint_max_iterations", 600)))
        collision_sweeps = max(1, int(getattr(params, "k2d_hard_joint_collision_sweeps", 2)))

        before_err = _hinge_errors(solved, constraints)
        before_max = float(np.max(before_err)) if before_err.size else 0.0
        best = solved.copy()
        best_score = (len(mod._penetrating_pairs(pipeline_obj, best, collision_tol)), before_max)
        converged = False
        iterations_used = 0

        x = solved.copy()
        for outer in range(max_outer):
            # Project toward exact hinge coincidence using only rigid SE(2) tile moves.
            x = _hard_hinge_sweep(x, constraints)
            # Restore hard non-overlap without centre-expansion fallback.
            for _ in range(collision_sweeps):
                x, moved = _collision_gs_sweep(mod, pipeline_obj, x, collision_tol)
                if moved == 0:
                    break

            errors = _hinge_errors(x, constraints)
            max_hinge = float(np.max(errors)) if errors.size else 0.0
            overlaps = mod._penetrating_pairs(pipeline_obj, x, collision_tol)
            score = (len(overlaps), max_hinge)
            if score < best_score:
                best = x.copy()
                best_score = score
            iterations_used = outer + 1
            if len(overlaps) == 0 and max_hinge <= hinge_tol:
                converged = True
                best = x.copy()
                best_score = score
                break

        final = best
        final_err = _hinge_errors(final, constraints)
        final_max = float(np.max(final_err)) if final_err.size else 0.0
        final_rms = float(np.sqrt(np.mean(final_err * final_err))) if final_err.size else 0.0
        final_overlaps = mod._penetrating_pairs(pipeline_obj, final, collision_tol)

        metrics = dict(metrics or {})
        metrics.update({
            "onestring_k2d_hard_hinge_applied": True,
            "onestring_k2d_hard_hinge_model": "alternating rigid SE(2) hinge projection + Gauss-Seidel SAT non-overlap projection",
            "onestring_k2d_hard_hinge_constraint_count": int(len(constraints)),
            "onestring_k2d_hard_hinge_tolerance": float(hinge_tol),
            "onestring_k2d_hard_hinge_max_before": float(before_max),
            "onestring_k2d_hard_hinge_max_after": float(final_max),
            "onestring_k2d_hard_hinge_rms_after": float(final_rms),
            "onestring_k2d_joint_hard_collision_tolerance": float(collision_tol),
            "onestring_k2d_joint_hard_final_overlap_count": int(len(final_overlaps)),
            "onestring_k2d_joint_hard_iterations": int(iterations_used),
            "onestring_k2d_joint_hard_feasible": bool(len(final_overlaps) == 0 and final_max <= hinge_tol),
            "onestring_k2d_hard_nonoverlap_final_penetration_count": int(len(final_overlaps)),
            "onestring_k2d_hard_nonoverlap_satisfied": bool(len(final_overlaps) == 0),
        })

        print(
            "[OPTCUTS-TEST-K2D-JOINT-HARD] "
            f"hinges={len(constraints)} hinge_max={before_max:.6g}->{final_max:.6g} "
            f"hinge_tol={hinge_tol:.6g} overlaps={len(final_overlaps)} "
            f"iters={iterations_used} feasible={metrics['onestring_k2d_joint_hard_feasible']}"
        )

        if final_overlaps or final_max > hinge_tol:
            raise RuntimeError(
                "OPTCUTS_TEST_K2D_JOINT_HARD_INFEASIBLE: could not satisfy rigid-tile hinge coincidence "
                "and non-overlap simultaneously; "
                f"overlaps={len(final_overlaps)}, max_hinge_error={final_max:.9g}, "
                f"hinge_tolerance={hinge_tol:.9g}, iterations={iterations_used}."
            )
        return final, metrics

    mod._build_rigid_k2d_layout = build_with_joint_hard_constraints
    mod._onestring_k2d_hard_hinge_installed = True


__all__ = ["install_optcuts_test_k2d_hard_hinge_patch"]
