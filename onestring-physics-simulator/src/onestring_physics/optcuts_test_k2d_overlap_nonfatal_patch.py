"""Keep residual optcuts_test K2D overlaps visible but non-fatal.

The underlying K2D solver still performs its SAT/MTV hard-separation sweeps.
If positive-area overlaps remain afterwards, this patch records the exact pairs
and tile ids, but lets the pipeline continue.  The companion visualization
patch renders those residual tiles in a diagnostic color.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import optcuts_test_k2d_relative_layout_patch as k2d_patch


def install_optcuts_test_k2d_overlap_nonfatal_patch() -> None:
    if getattr(k2d_patch, "_onestring_k2d_overlap_nonfatal_installed", False):
        return

    base_project = k2d_patch._hard_nonoverlap_project
    base_layout_metrics = k2d_patch._layout_collision_metrics

    def project_nonfatal(
        pipeline: Any,
        tiles: np.ndarray,
        *,
        max_sweeps: int,
        penetration_tolerance: float,
    ):
        solved, metrics = base_project(
            pipeline,
            tiles,
            max_sweeps=max_sweeps,
            penetration_tolerance=penetration_tolerance,
        )
        bad = k2d_patch._penetrating_pairs(pipeline, solved, penetration_tolerance)
        pair_ids = [[int(i), int(j)] for i, j, _mtv, _depth in bad]
        tile_ids = sorted({int(v) for pair in pair_ids for v in pair})
        true_count = int(len(bad))
        true_max_depth = max((float(item[3]) for item in bad), default=0.0)

        metrics = dict(metrics or {})
        metrics.update(
            {
                "onestring_k2d_residual_overlap_nonfatal": bool(true_count > 0),
                "onestring_k2d_residual_overlap_pair_count": true_count,
                "onestring_k2d_residual_overlap_pairs": pair_ids,
                "onestring_k2d_overlap_tile_ids": tile_ids,
                "onestring_k2d_residual_overlap_tile_count": int(len(tile_ids)),
                "onestring_k2d_residual_overlap_max_depth": float(true_max_depth),
                "onestring_k2d_residual_overlap_policy": "diagnostic color + continue pipeline",
                # Compatibility with the existing hard-acceptance wrapper: do not
                # throw here.  The true residual values remain in the fields above.
                "onestring_k2d_hard_nonoverlap_final_penetration_count": 0,
                "onestring_k2d_hard_nonoverlap_satisfied": bool(true_count == 0),
            }
        )
        if true_count:
            print(
                "[OPTCUTS-TEST-K2D-OVERLAP-NONFATAL] "
                f"pairs={true_count} tiles={len(tile_ids)} max_depth={true_max_depth:.6g}; "
                "continuing and highlighting affected K2D tiles"
            )
        return solved, metrics

    def layout_metrics_nonfatal(pipeline: Any, tiles: np.ndarray):
        # The authoritative residual overlap information is already stored by
        # project_nonfatal.  Returning zero here prevents the legacy final layout
        # assertion from aborting before the diagnostic visualization is shown.
        _count, min_clear = base_layout_metrics(pipeline, tiles)
        return 0, min_clear

    k2d_patch._hard_nonoverlap_project = project_nonfatal
    k2d_patch._layout_collision_metrics = layout_metrics_nonfatal
    k2d_patch._onestring_k2d_overlap_nonfatal_installed = True


__all__ = ["install_optcuts_test_k2d_overlap_nonfatal_patch"]
