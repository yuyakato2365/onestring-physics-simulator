"""Hard collision-feasibility projection for optcuts_test K2D.

This patch replaces the old averaged/Jacobi MTV relaxation used after the
OneString-style SE(2) K2D solve.  The authoritative K2D result must satisfy
positive-area non-overlap.  We first use sequential Gauss-Seidel SAT projections;
if that local projection stalls, we keep every tile rigid and minimally expand
tile centres about the global centroid until a collision-free configuration is
found.  The expansion amount is then binary-searched down to the smallest
feasible uniform factor.

Point/edge contact is allowed.  Only positive-area penetration is forbidden.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _scale_tile_centres(tiles: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(tiles, dtype=float)
    if len(x) == 0:
        return x.copy()
    centres = np.mean(x, axis=1)
    global_centre = np.mean(centres, axis=0)
    local = x - centres[:, None, :]
    new_centres = global_centre[None, :] + float(alpha) * (centres - global_centre[None, :])
    return local + new_centres[:, None, :]


def _inject_tiny_centre_separation(tiles: np.ndarray, scale: float) -> np.ndarray:
    """Break exact/near-exact coincident tile centres without changing tile shape."""
    x = np.asarray(tiles, dtype=float).copy()
    if len(x) <= 1:
        return x
    centres = np.mean(x, axis=1)
    eps = max(float(scale) * 1e-5, 1e-9)
    # Deterministic golden-angle offsets.  This is used only if uniform centre
    # expansion cannot create a feasible configuration because centres coincide.
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(len(x)):
        for j in range(i):
            if np.linalg.norm(centres[i] - centres[j]) <= eps:
                angle = golden * float(i + 1)
                offset = eps * np.array([np.cos(angle), np.sin(angle)], dtype=float)
                x[i] += offset[None, :]
                centres[i] += offset
                break
    return x


def install_optcuts_test_k2d_hard_feasibility_patch() -> None:
    from . import optcuts_test_k2d_relative_layout_patch as mod

    if getattr(mod, "_onestring_k2d_true_hard_feasibility_installed", False):
        return

    def hard_nonoverlap_project(
        pipeline: Any,
        tiles: np.ndarray,
        *,
        max_sweeps: int,
        penetration_tolerance: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        x = np.asarray(tiles, dtype=float).copy()
        before = mod._penetrating_pairs(pipeline, x, penetration_tolerance)
        initial_count = len(before)
        max_depth_before = max((item[3] for item in before), default=0.0)
        scale = max(float(mod._tile_scale(x)), 1e-8)
        pair_fn, sat_fn = mod._collision_backend(pipeline)
        if pair_fn is None or sat_fn is None:
            raise RuntimeError("OPTCUTS_TEST_K2D_HARD_COLLISION_BACKEND_UNAVAILABLE")

        sweeps_used = 0
        # True sequential feasibility projection: every correction is committed
        # immediately before the next pair is evaluated.  This avoids the old
        # averaged MTV cancellation that left most of the layout overlapping.
        for sweep in range(max(1, int(max_sweeps))):
            bad = mod._penetrating_pairs(pipeline, x, penetration_tolerance)
            if not bad:
                break
            # Resolve deepest penetrations first.
            bad = sorted(bad, key=lambda item: item[3], reverse=True)
            moved = False
            for i, j, _old_mtv, _old_depth in bad:
                overlap, mtv, signed = sat_fn(x[i], x[j], clearance=0.0)
                mtv = np.asarray(mtv, dtype=float)
                depth = float(np.linalg.norm(mtv)) if overlap else 0.0
                tol = max(float(penetration_tolerance), scale * 1e-12, 1e-12)
                if not overlap or depth <= tol or float(signed) >= -tol:
                    continue
                # Slight over-projection prevents the next floating-point SAT
                # evaluation from classifying the same pair as penetrating.
                direction = mtv / max(depth, 1e-15)
                separation = mtv + direction * max(tol * 2.0, scale * 1e-10)
                x[i] += 0.5 * separation[None, :]
                x[j] -= 0.5 * separation[None, :]
                moved = True
            sweeps_used = sweep + 1
            if not moved:
                break

        after_gs = mod._penetrating_pairs(pipeline, x, penetration_tolerance)
        gs_count = len(after_gs)
        expansion_used = False
        expansion_factor = 1.0
        expansion_search_steps = 0
        coincident_jitter_used = False

        if after_gs:
            # A hard feasible fallback.  Uniformly scale tile centres while each
            # tile itself remains a rigid body.  Since tile extents are bounded,
            # sufficiently large alpha must separate distinct centres.
            base = x.copy()
            low = 1.0
            high = 1.02
            feasible = None
            for _ in range(80):
                candidate = _scale_tile_centres(base, high)
                if not mod._penetrating_pairs(pipeline, candidate, penetration_tolerance):
                    feasible = candidate
                    break
                low = high
                high *= 1.15
                expansion_search_steps += 1
                if high > 64.0:
                    break

            if feasible is None:
                # Exact coincident centres are invariant under centre scaling.
                # Give those centres an infinitesimal deterministic separation,
                # then repeat the same hard-feasibility search.
                base = _inject_tiny_centre_separation(base, scale)
                coincident_jitter_used = True
                low = 1.0
                high = 1.02
                for _ in range(100):
                    candidate = _scale_tile_centres(base, high)
                    if not mod._penetrating_pairs(pipeline, candidate, penetration_tolerance):
                        feasible = candidate
                        break
                    low = high
                    high *= 1.15
                    expansion_search_steps += 1
                    if high > 128.0:
                        break

            if feasible is None:
                remaining = mod._penetrating_pairs(pipeline, base, penetration_tolerance)
                raise RuntimeError(
                    "OPTCUTS_TEST_K2D_HARD_FEASIBILITY_FAILED: could not construct a collision-free "
                    f"rigid K2D layout; remaining_pairs={len(remaining)}."
                )

            # Binary search the smallest feasible uniform centre expansion.
            for _ in range(36):
                mid = 0.5 * (low + high)
                candidate = _scale_tile_centres(base, mid)
                if mod._penetrating_pairs(pipeline, candidate, penetration_tolerance):
                    low = mid
                else:
                    high = mid
                    feasible = candidate
            x = np.asarray(feasible, dtype=float)
            expansion_used = True
            expansion_factor = float(high)

        final_bad = mod._penetrating_pairs(pipeline, x, penetration_tolerance)
        if final_bad:
            raise RuntimeError(
                "OPTCUTS_TEST_K2D_HARD_FEASIBILITY_ASSERTION_FAILED: authoritative K2D still has "
                f"{len(final_bad)} positive-area overlaps."
            )

        max_depth_after = max((item[3] for item in final_bad), default=0.0)
        metrics = {
            "onestring_k2d_hard_nonoverlap_applied": True,
            "onestring_k2d_hard_nonoverlap_model": "hard SAT feasibility: Gauss-Seidel MTV projection + minimal rigid centre expansion fallback",
            "onestring_k2d_hard_nonoverlap_touching_allowed": True,
            "onestring_k2d_hard_nonoverlap_initial_penetration_count": int(initial_count),
            "onestring_k2d_hard_nonoverlap_after_gauss_seidel_count": int(gs_count),
            "onestring_k2d_hard_nonoverlap_final_penetration_count": 0,
            "onestring_k2d_hard_nonoverlap_max_depth_before": float(max_depth_before),
            "onestring_k2d_hard_nonoverlap_max_depth_after": float(max_depth_after),
            "onestring_k2d_hard_nonoverlap_sweeps": int(sweeps_used),
            "onestring_k2d_hard_nonoverlap_satisfied": True,
            "onestring_k2d_hard_nonoverlap_is_authoritative_constraint": True,
            "onestring_k2d_hard_nonoverlap_expansion_fallback_used": bool(expansion_used),
            "onestring_k2d_hard_nonoverlap_expansion_factor": float(expansion_factor),
            "onestring_k2d_hard_nonoverlap_expansion_search_steps": int(expansion_search_steps),
            "onestring_k2d_hard_nonoverlap_coincident_centre_jitter_used": bool(coincident_jitter_used),
        }
        print(
            "[OPTCUTS-TEST-K2D-HARD-FEASIBLE] "
            f"penetrations={initial_count}->{gs_count}->0 "
            f"sweeps={sweeps_used} expansion={expansion_used} alpha={expansion_factor:.6g}"
        )
        return x, metrics

    mod._hard_nonoverlap_project = hard_nonoverlap_project
    mod._onestring_k2d_true_hard_feasibility_installed = True


__all__ = ["install_optcuts_test_k2d_hard_feasibility_patch"]
