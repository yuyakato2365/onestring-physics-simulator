"""Topology and conservative two-cut lookahead helpers for Collision-driven Split.

Lookahead may temporarily explore a first cut that fails single-step numerical
acceptance. No cut is committed until the final two-cut layout passes the
original accept_if_improved() test against the LAST ACCEPTED baseline.
"""
from __future__ import annotations

import math

import numpy as np


def promising_rejected_first_cut(before: dict, after: dict) -> bool:
    """Bound lookahead to first cuts with fewer collisions and lower total energy.

    An increased *maximum* SAT penetration is allowed in this temporary state,
    since the second cut is intended to address such a remaining hotspot.
    Require a remaining collision: collision-ranked second cuts cannot be
    generated from a collision-free layout, even if its hinges are disconnected.
    This does NOT weaken the final committed-state acceptance test.
    """
    if before.get("status") != "ok" or after.get("status") != "ok":
        return False
    try:
        p0 = int(before["collision_pairs"])
        p1 = int(after["collision_pairs"])
        e0 = float(before["collision_energy"])
        e1 = float(after["collision_energy"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return False
    return (0 < p1 <= p0
            and math.isfinite(e0) and math.isfinite(e1)
            and 0 <= e1 < e0)


def validate_real_split_trial(before_state, trial_state, expected_cut_count: int) -> None:
    """Raise unless a full rebuild really preserves all M2D seam topology.

    Verify the cumulative seam count, every cut's disconnected edge pairs,
    and a strict increase of M2D connected components from the preceding
    trial. Applies equally to a one-cut and the second cut of a lookahead.
    """
    from .simple_split_panel_patch import _edge_components

    mesh = trial_state.mesh_2d_initial
    metrics = mesh.metrics
    applied = int(metrics.get("collision_split_count", -1))
    final_applied = metrics.get("collision_split_final_pass_applied_count")
    if applied != expected_cut_count:
        raise RuntimeError(
            "Collision Split topology not propagated through M2D: "
            f"expected={expected_cut_count}, initial_cuts={applied}"
        )
    if final_applied is not None and int(final_applied) != expected_cut_count:
        raise RuntimeError(
            "Collision Split lost during optional final M2D topology pass"
        )
    seams = metrics.get("collision_split_step_analysis", [])
    if len(seams) != expected_cut_count:
        raise RuntimeError("Collision Split source seam records missing")
    faces = np.asarray(mesh.faces, dtype=int)
    for seam in seams:
        if (not seam.get("complete_boundary_to_boundary", False)
                or not seam.get("split_boundary_pairs")):
            raise RuntimeError("Collision Split was not a complete real source seam")
        for a, ea, b, eb in seam["split_boundary_pairs"]:
            edge_a = set(map(int, faces[a, [ea, (ea + 1) % 4]]))
            edge_b = set(map(int, faces[b, [eb, (eb + 1) % 4]]))
            if edge_a == edge_b:
                raise RuntimeError("Collision Split edge was rewelded")
    old_components = len(_edge_components(
        np.asarray(before_state.mesh_2d_initial.faces, dtype=int)
    ))
    new_components = len(_edge_components(faces))
    if new_components <= old_components:
        raise RuntimeError(
            "Collision Split did not add a real disconnected M2D component"
        )
