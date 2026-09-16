"""Hard collision-feasibility projection for optcuts_test K2D.

The authoritative K2D result must satisfy positive-area non-overlap.  We first
use sequential Gauss-Seidel SAT projections; if that local projection stalls, we
keep every tile rigid and minimally expand tile centres about the global centroid
until a collision-free configuration is found.

Point/edge contact is allowed. Only positive-area penetration is forbidden.
The feasibility solve and the final layout assertion deliberately use the same
strict tolerance based on the actual tile scale, so a configuration accepted by
the hard solver cannot be rejected later by a slightly different threshold.
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
    x = np.asarray(tiles, dtype=float).copy()
    if len(x) <= 1:
        return x
    centres = np.mean(x, axis=1)
    eps = max(float(scale) * 1e-5, 1e-9)
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

    # Keep one authoritative tolerance policy for both the hard feasibility solve
    # and the final make_layout() assertion.  The old code used tile_size in the
    # solver and median actual edge length in the assertion; clipped boundary
    # tiles made those differ enough to create a one-pair false failure.
    def strict_tol_for_tiles(tiles: np.ndarray, requested: float | None = None) -> float:
        scale = max(float(mod._tile_scale(np.asarray(tiles, dtype=float))), 1e-12)
        tol = max(scale * 1e-10, 1e-13)
        if requested is not None and np.isfinite(float(requested)):
            tol = min(tol, max(float(requested), 1e-13))
        return float(tol)

    def hard_nonoverlap_project(
        pipeline: Any,
        tiles: np.ndarray,
        *,
        max_sweeps: int,
        penetration_tolerance: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        x = np.asarray(tiles, dtype=float).copy()
        effective_tol = strict_tol_for_tiles(x, penetration_tolerance)
        before = mod._penetrating_pairs(pipeline, x, effective_tol)
        initial_count = len(before)
        max_depth_before = max((item[3] for item in before), default=0.0)
        scale = max(float(mod._tile_scale(x)), 1e-8)
        pair_fn, sat_fn = mod._collision_backend(pipeline)
        if pair_fn is None or sat_fn is None:
            raise RuntimeError("OPTCUTS_TEST_K2D_HARD_COLLISION_BACKEND_UNAVAILABLE")

        sweeps_used = 0
        for sweep in range(max(1, int(max_sweeps))):
            bad = mod._penetrating_pairs(pipeline, x, effective_tol)
            if not bad:
                break
            bad = sorted(bad, key=lambda item: item[3], reverse=True)
            moved = False
            for i, j, _old_mtv, _old_depth in bad:
                overlap, mtv, signed = sat_fn(x[i], x[j], clearance=0.0)
                mtv = np.asarray(mtv, dtype=float)
                depth = float(np.linalg.norm(mtv)) if overlap else 0.0
                if not overlap or depth <= effective_tol or float(signed) >= -effective_tol:
                    continue
                direction = mtv / max(depth, 1e-15)
                separation = mtv + direction * max(effective_tol * 4.0, scale * 5e-10)
                x[i] += 0.5 * separation[None, :]
                x[j] -= 0.5 * separation[None, :]
                moved = True
            sweeps_used = sweep + 1
            if not moved:
                break

        after_gs = mod._penetrating_pairs(pipeline, x, effective_tol)
        gs_count = len(after_gs)
        expansion_used = False
        expansion_factor = 1.0
        expansion_search_steps = 0
        coincident_jitter_used = False

        if after_gs:
            base = x.copy()
            low = 1.0
            high = 1.02
            feasible = None
            for _ in range(80):
                candidate = _scale_tile_centres(base, high)
                if not mod._penetrating_pairs(pipeline, candidate, effective_tol):
                    feasible = candidate
                    break
                low = high
                high *= 1.15
                expansion_search_steps += 1
                if high > 64.0:
                    break

            if feasible is None:
                base = _inject_tiny_centre_separation(base, scale)
                coincident_jitter_used = True
                low = 1.0
                high = 1.02
                for _ in range(100):
                    candidate = _scale_tile_centres(base, high)
                    if not mod._penetrating_pairs(pipeline, candidate, effective_tol):
                        feasible = candidate
                        break
                    low = high
                    high *= 1.15
                    expansion_search_steps += 1
                    if high > 128.0:
                        break

            if feasible is None:
                remaining = mod._penetrating_pairs(pipeline, base, effective_tol)
                raise RuntimeError(
                    "OPTCUTS_TEST_K2D_HARD_FEASIBILITY_FAILED: could not construct a collision-free "
                    f"rigid K2D layout; remaining_pairs={len(remaining)}."
                )

            for _ in range(40):
                mid = 0.5 * (low + high)
                candidate = _scale_tile_centres(base, mid)
                if mod._penetrating_pairs(pipeline, candidate, effective_tol):
                    low = mid
                else:
                    high = mid
                    feasible = candidate
            # Add a tiny safety margin beyond the binary-search boundary.  This is
            # still many orders of magnitude below a visible layout change, but it
            # avoids a pair landing exactly on the SAT numerical threshold.
            safe_alpha = high * (1.0 + 5e-9)
            x = _scale_tile_centres(base, safe_alpha)
            expansion_used = True
            expansion_factor = float(safe_alpha)

        final_bad = mod._penetrating_pairs(pipeline, x, effective_tol)
        if final_bad:
            raise RuntimeError(
                "OPTCUTS_TEST_K2D_HARD_FEASIBILITY_ASSERTION_FAILED: authoritative K2D still has "
                f"{len(final_bad)} positive-area overlaps."
            )

        metrics = {
            "onestring_k2d_hard_nonoverlap_applied": True,
            "onestring_k2d_hard_nonoverlap_model": "hard SAT feasibility: Gauss-Seidel MTV projection + minimal rigid centre expansion fallback",
            "onestring_k2d_hard_nonoverlap_touching_allowed": True,
            "onestring_k2d_hard_nonoverlap_initial_penetration_count": int(initial_count),
            "onestring_k2d_hard_nonoverlap_after_gauss_seidel_count": int(gs_count),
            "onestring_k2d_hard_nonoverlap_final_penetration_count": 0,
            "onestring_k2d_hard_nonoverlap_max_depth_before": float(max_depth_before),
            "onestring_k2d_hard_nonoverlap_max_depth_after": 0.0,
            "onestring_k2d_hard_nonoverlap_sweeps": int(sweeps_used),
            "onestring_k2d_hard_nonoverlap_satisfied": True,
            "onestring_k2d_hard_nonoverlap_is_authoritative_constraint": True,
            "onestring_k2d_hard_nonoverlap_effective_tolerance": float(effective_tol),
            "onestring_k2d_hard_nonoverlap_expansion_fallback_used": bool(expansion_used),
            "onestring_k2d_hard_nonoverlap_expansion_factor": float(expansion_factor),
            "onestring_k2d_hard_nonoverlap_expansion_search_steps": int(expansion_search_steps),
            "onestring_k2d_hard_nonoverlap_coincident_centre_jitter_used": bool(coincident_jitter_used),
        }
        print(
            "[OPTCUTS-TEST-K2D-HARD-FEASIBLE] "
            f"penetrations={initial_count}->{gs_count}->0 "
            f"tol={effective_tol:.3g} sweeps={sweeps_used} "
            f"expansion={expansion_used} alpha={expansion_factor:.9g}"
        )
        return x, metrics

    def layout_collision_metrics(pipeline: Any, tiles: np.ndarray) -> tuple[int, float]:
        t = np.asarray(tiles, dtype=float)
        if len(t) < 2:
            return 0, 0.0
        tol = strict_tol_for_tiles(t)
        bad = mod._penetrating_pairs(pipeline, t, tol)
        min_clear = -max((item[3] for item in bad), default=0.0)
        return int(len(bad)), float(min_clear)

    mod._hard_nonoverlap_project = hard_nonoverlap_project
    mod._layout_collision_metrics = layout_collision_metrics
    mod._onestring_k2d_true_hard_feasibility_installed = True


__all__ = ["install_optcuts_test_k2d_hard_feasibility_patch"]
