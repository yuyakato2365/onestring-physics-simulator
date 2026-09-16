"""Compatibility patch for collision-free ABD staging of flat T2D panels.

The original staging helper ignored hinge-connected panel pairs while checking
clearance.  That is acceptable for a purely topological layout check, but ABD's
IPC initialization rejects any touching or intersecting collision meshes before
pin joints are solved.  Builtin shapes commonly contain many such exact edge
contacts, so the exported scene could fail even though the 2D non-hinge check
reported an exact layout.

This patch keeps panel orientations and relative ordering, but applies the
smallest uniform center expansion that gives every panel pair a strictly
positive in-plane clearance.  Pin-joint anchors are generated after staging,
so the solver can subsequently close those small initial gaps without changing
the intended connectivity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import abd_backend as _abd


_ORIGINAL_STAGING = _abd._collision_free_staging_proxy
_INSTALLED = False


def _pair_has_clearance(a: np.ndarray, b: np.ndarray, required_gap: float) -> bool:
    """Return True when two convex quads are separated by at least required_gap."""

    for polygon in (a, b):
        for edge_id in range(len(polygon)):
            edge = polygon[(edge_id + 1) % len(polygon)] - polygon[edge_id]
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= 1.0e-12:
                continue
            axis /= norm
            projection_a = a @ axis
            projection_b = b @ axis
            if (
                float(np.max(projection_a)) + required_gap <= float(np.min(projection_b))
                or float(np.max(projection_b)) + required_gap <= float(np.min(projection_a))
            ):
                return True
    return False


def _all_pair_collision_free_staging_proxy(
    state: Any,
    source_proxy: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, str, float]:
    """Apply the minimum safe uniform expansion for all panel pairs.

    Unlike the original implementation, hinge-connected pairs are not skipped:
    IPC requires a strictly positive initial distance even for bodies joined by
    a pin constraint.
    """

    source = np.asarray(source_proxy, dtype=float)
    tile_count = int(len(source))
    if tile_count == 0:
        return source.copy(), "empty", 1.0
    if source.ndim != 3 or source.shape[1] < 4 or source.shape[2] != 3:
        raise _abd.ABDBackendError(
            f"ABD T2D staging expected (tile, vertex, 3) vertices with at least four vertices; got {source.shape}"
        )
    if not np.all(np.isfinite(source)):
        raise _abd.ABDBackendError("ABD T2D staging contains non-finite panel vertices")

    source_centers = np.mean(source, axis=1)
    local = source - source_centers[:, None, :]
    layout_center = np.mean(source_centers[:, :2], axis=0)
    relative_centers = source_centers[:, :2] - layout_center[None, :]

    # prepare_abd_job later shrinks every collision mesh in-plane by collision_skin.
    # Requiring this smaller center gap before that shrink is conservative without
    # scattering panels excessively.
    required_gap = max(float(clearance), 1.0e-8)

    def quads_at(scale: float) -> np.ndarray:
        centers = layout_center[None, :] + relative_centers * float(scale)
        return local[:, :4, :2] + centers[:, None, :]

    def layout_is_clear(scale: float) -> bool:
        quads = quads_at(scale)
        for tile_a in range(tile_count):
            for tile_b in range(tile_a + 1, tile_count):
                if not _pair_has_clearance(quads[tile_a], quads[tile_b], required_gap):
                    return False
        return True

    layout_scale = 1.0
    if not layout_is_clear(layout_scale):
        upper = 1.0005
        while upper < 64.0 and not layout_is_clear(upper):
            upper = 1.0 + (upper - 1.0) * 1.8
        if not layout_is_clear(upper):
            raise _abd.ABDBackendError(
                "The ABD initial T2D layout contains coincident panel centers or persistent overlaps. "
                "All panel pairs, including hinge-connected pairs, were checked and uniform center "
                "expansion up to 64x could not create positive IPC clearance."
            )
        lower = 1.0
        for _ in range(40):
            middle = 0.5 * (lower + upper)
            if layout_is_clear(middle):
                upper = middle
            else:
                lower = middle
        layout_scale = float(upper)

    staged = local.copy()
    staged[:, :, :2] += (
        layout_center[None, :] + relative_centers * layout_scale
    )[:, None, :]
    staged[:, :, 2] += source_centers[:, None, 2]

    layout_name = (
        "t2d_dual_hinge_exact_positive_clearance"
        if layout_scale <= 1.0 + 1.0e-10
        else "t2d_dual_hinge_all_pair_minimum_clearance_scale"
    )
    return staged, layout_name, layout_scale


def install_abd_layout_compatibility() -> None:
    """Install the all-pair positive-clearance staging patch once."""

    global _INSTALLED
    if _INSTALLED:
        return
    _abd._collision_free_staging_proxy = _all_pair_collision_free_staging_proxy
    _INSTALLED = True


__all__ = ["install_abd_layout_compatibility"]
