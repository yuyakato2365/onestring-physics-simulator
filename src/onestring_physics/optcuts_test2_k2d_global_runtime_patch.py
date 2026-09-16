"""Run the global all-hinge K2D feasibility solve directly in optcuts_test2.

The original optcuts_test path is intentionally left on the previous kinematic
spanning-tree solver for comparison.  optcuts_test2 bypasses that solver and
starts from the collision-free rigid K2D checkpoint, then optimizes every tile
SE(2) pose simultaneously against every physical point-hinge coincidence.

This is Phase 1 of the new formulation: hinge feasibility is solved first with
collision disabled.  Residual overlaps are reported non-fatally so we can tell
whether hinge closure itself is feasible before adding non-overlap inequalities.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from .optcuts_k2d_global_history import global_hinge_solve


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "1").strip() == "2"


def install_optcuts_test2_k2d_global_runtime_patch(
    pipeline: Any,
    global_base_builder: Callable[..., tuple[np.ndarray, dict[str, Any]]],
) -> None:
    """Make optcuts_test2 use the global all-hinge solver in the normal pipeline.

    ``global_base_builder`` must be captured before the legacy hard-hinge wrapper
    is installed.  This is what actually bypasses the spanning-tree/loop solver
    rather than running it and overwriting its result afterwards.
    """
    from . import optcuts_test_k2d_relative_layout_patch as mod

    if getattr(mod, "_onestring_test2_global_runtime_installed", False):
        return

    legacy_builder = mod._build_rigid_k2d_layout

    def build_with_global_all_hinges(
        pipeline_obj: Any,
        mesh_2d: Any,
        mesh_3d: Any,
        params: Any,
        progress_callback=None,
    ):
        if not _is_test2():
            return legacy_builder(
                pipeline_obj,
                mesh_2d,
                mesh_3d,
                params,
                progress_callback=progress_callback,
            )

        # IMPORTANT: this builder is the collision-free rigid K2D builder that
        # was captured *before* the legacy spanning-tree hard-hinge wrapper.
        initial, metrics = global_base_builder(
            pipeline_obj,
            mesh_2d,
            mesh_3d,
            params,
            progress_callback=progress_callback,
        )
        initial = np.asarray(initial, dtype=float)
        faces = np.asarray(mesh_2d.faces, dtype=int)
        constraints = mod._hinge_constraints(pipeline_obj, faces)

        if len(initial) == 0 or not constraints:
            out = dict(metrics or {})
            out.update({
                "onestring_k2d_global_all_hinge_applied": True,
                "onestring_k2d_global_all_hinge_phase": "phase1_hinge_only",
                "onestring_k2d_global_all_hinge_feasible": True,
            })
            return initial, out

        max_nfev = max(20, int(getattr(params, "k2d_global_hinge_max_nfev", 300)))
        anchor_weight = max(0.0, float(getattr(params, "k2d_global_hinge_anchor_weight", 1e-5)))

        print(
            "[OPTCUTS-TEST2-K2D-GLOBAL-START] "
            f"tiles={len(initial)} hinges={len(constraints)} max_nfev={max_nfev} "
            "model=all-tile-SE2/all-hinge-coincidence collision=OFF"
        )

        final, global_metrics = global_hinge_solve(
            initial,
            constraints,
            max_nfev=max_nfev,
            anchor_weight=anchor_weight,
        )

        scale = max(float(mod._tile_scale(final)), 1e-12)
        collision_tol = max(scale * 1e-10, 1e-13)
        overlaps = mod._penetrating_pairs(pipeline_obj, final, collision_tol)
        overlap_pairs = [(int(i), int(j)) for i, j, _mtv, _depth in overlaps]
        overlap_tile_ids = sorted({v for pair in overlap_pairs for v in pair})

        hinge_tol = float(global_metrics.get("hinge_tolerance", max(scale * 5e-4, 1e-9)))
        hinge_errors = np.asarray(
            [
                np.linalg.norm(final[ia, ca] - final[ib, cb])
                for ia, ca, ib, cb in constraints
            ],
            dtype=float,
        )
        violating_edges = [int(i) for i, err in enumerate(hinge_errors) if float(err) > hinge_tol]
        hinge_violation_tiles: set[int] = set()
        for edge_id in violating_edges:
            ia, _ca, ib, _cb = constraints[edge_id]
            hinge_violation_tiles.update((int(ia), int(ib)))

        hinge_feasible = bool(global_metrics.get("all_hinges_satisfied", False))
        metrics_out = dict(metrics or {})
        metrics_out.update(global_metrics)
        metrics_out.update({
            "onestring_k2d_global_all_hinge_applied": True,
            "onestring_k2d_global_all_hinge_phase": "phase1_hinge_only",
            "onestring_k2d_global_all_hinge_model": (
                "all tile SE(2) poses optimized simultaneously; every physical hinge is a zero-gap point-coincidence residual; "
                "no spanning tree / no privileged tree hinges; collision intentionally disabled in Phase 1"
            ),
            "onestring_k2d_hinge_constraint_semantics": "zero-gap hinge coincidence; all hinges equal in one global system",
            "onestring_k2d_global_all_hinge_feasible": hinge_feasible,
            "onestring_k2d_joint_hard_feasible": hinge_feasible and len(overlaps) == 0,
            "onestring_k2d_kinematic_feasible": hinge_feasible and len(overlaps) == 0,
            "onestring_k2d_kinematic_constraint_status": (
                "all hinge coincidences satisfied; Phase 1 collision diagnostic may still overlap"
                if hinge_feasible
                else "GLOBAL HINGE FEASIBILITY NOT SATISFIED; best-effort layout returned"
            ),
            # Compatibility keys for the existing diagnostic visualization.  In
            # global mode these are ALL violating hinges, not only loop hinges.
            "onestring_k2d_kinematic_violating_loop_edge_ids": violating_edges,
            "onestring_k2d_kinematic_hinge_violation_tile_ids": sorted(hinge_violation_tiles),
            "onestring_k2d_hard_hinge_max_after": float(global_metrics.get("hinge_max_error", 0.0)),
            "onestring_k2d_hard_hinge_tolerance": hinge_tol,
            "onestring_k2d_residual_overlap_pairs": [list(pair) for pair in overlap_pairs],
            "onestring_k2d_residual_overlap_pair_count": int(len(overlap_pairs)),
            "onestring_k2d_overlap_tile_ids": overlap_tile_ids,
            "onestring_k2d_joint_hard_final_overlap_count": int(len(overlap_pairs)),
            "onestring_k2d_hard_nonoverlap_final_penetration_count": int(len(overlap_pairs)),
            "onestring_k2d_hard_nonoverlap_satisfied": bool(len(overlap_pairs) == 0),
            "onestring_k2d_phase1_collision_enabled": False,
            "onestring_k2d_tile_rigidity_preserved": True,
        })

        print(
            "[OPTCUTS-TEST2-K2D-GLOBAL] "
            f"hinge_max={float(global_metrics.get('hinge_max_error', 0.0)):.6g} "
            f"hinge_rms={float(global_metrics.get('hinge_rms_error', 0.0)):.6g} "
            f"tol={hinge_tol:.6g} violations={len(violating_edges)} "
            f"overlaps_diagnostic={len(overlap_pairs)} hinge_feasible={hinge_feasible}"
        )
        if not hinge_feasible:
            print(
                "[OPTCUTS-TEST2-K2D-GLOBAL-NONFATAL] all-hinge coincidence was not fully satisfied; "
                "continuing so the failed geometry can be inspected"
            )
        elif overlap_pairs:
            print(
                "[OPTCUTS-TEST2-K2D-GLOBAL-PHASE1] hinge feasibility passed; collisions are diagnostic only in Phase 1 "
                f"(overlaps={len(overlap_pairs)})"
            )

        return np.asarray(final, dtype=float), metrics_out

    mod._build_rigid_k2d_layout = build_with_global_all_hinges
    mod._onestring_test2_global_runtime_installed = True


__all__ = ["install_optcuts_test2_k2d_global_runtime_patch"]
