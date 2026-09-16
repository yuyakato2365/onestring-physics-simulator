"""Injective paired-line grid fusion for OneString/OptCuts.

A physical OptCuts seam has two UV boundary copies.  Collapsing both copies onto
one geometric line removes the very planar freedom introduced by the cut and is
not generally compatible with an injective UV embedding.  This module keeps the
user's fabrication constraints while preserving that necessary degree of freedom:

* each physical seam chain is represented by a PAIR of straight parallel lines;
* all lines use one global orthogonal frame (u/v only);
* the existing tile_size is the fixed lattice unit h;
* each copy line lies on that h-lattice;
* paired lines use the smallest lattice separation close to the official OptCuts
  proposal (never an arbitrary post-hoc staircase);
* corresponding points on the two copies use the same normalized 3D seam
  arclength, so paired boundaries remain cleanly registered;
* no extra central seam is inserted in M2D.  The two straight UV boundaries are
  the actual two sides of the physical cut.

This is deliberately different from the rejected v2 zero-width model, which can
force thousands of triangle flips for legitimate OptCuts cut topologies.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import optcuts_grid_constrained_parameterization_patch as constrained_param
from . import optcuts_pipeline_patch as optcuts_pipeline


def _paired_line_hard_targets(
    parameterization: Any,
    h: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h = max(float(h), 1.0e-12)
    uv0 = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    records = constrained_param._surface_seam_records(parameterization)
    physical_chains = constrained_param._physical_chains(records)
    lookup = constrained_param._record_lookup(records)

    entries: list[dict[str, Any]] = []
    copy_chains: list[list[int]] = []
    for physical in physical_chains:
        if len(physical) >= 3 and int(physical[0]) == int(physical[-1]):
            raise RuntimeError(
                "OPTCUTS_GRID_CONSTRAINT_CLOSED_SEAM: closed physical seam needs more than one straight segment"
            )
        side0, side1 = constrained_param._uv_copy_chains_for_physical_chain(physical, lookup)
        if len(side0) < 2 or len(side1) < 2:
            continue
        if len(side0) != len(physical) or len(side1) != len(physical):
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_COPY_CHAIN_LENGTH_MISMATCH")
        entries.append({
            "physical": [int(v) for v in physical],
            "copies": [[int(v) for v in side0], [int(v) for v in side1]],
        })
        copy_chains.extend([side0, side1])

    if not entries:
        return uv0.copy(), np.full_like(uv0, np.nan), {
            "seam_chain_count": 0,
            "seam_copy_chain_count": 0,
            "axis_rotation_degrees": 0.0,
            "constrained_vertex_count": 0,
            "paired_straight_seam_copies": True,
            "zero_width_physical_seam": False,
        }

    angle = constrained_param._dominant_axis_angle(uv0, copy_chains)
    rotated, _ = constrained_param._rotate_uv(uv0, -angle)

    # One H/V choice per PHYSICAL chain.  Both copies necessarily share it.
    for entry in entries:
        deltas = []
        for copy in entry["copies"]:
            pts = rotated[np.asarray(copy, dtype=int)]
            deltas.append(pts[-1] - pts[0])
        d = np.mean(np.asarray(deltas, dtype=float), axis=0)
        entry["axis"] = 0 if abs(float(d[0])) >= abs(float(d[1])) else 1

    # Choose a common lattice phase for each coordinate.  Unlike v2, the lattice
    # origin is not assumed to be world zero; this avoids needless <= h/2 shifts.
    seam_samples = np.asarray(sorted({u for e in entries for c in e["copies"] for u in c}), dtype=int)
    phases = np.zeros(2, dtype=float)
    for coord in (0, 1):
        values = rotated[seam_samples, coord]
        residual = np.mod(values, h)
        # Circular mean on a period-h lattice.
        angles = 2.0 * np.pi * residual / h
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        if mean_angle < 0.0:
            mean_angle += 2.0 * np.pi
        phases[coord] = h * mean_angle / (2.0 * np.pi)

    def snap(value: float, coord: int) -> float:
        phase = float(phases[coord])
        return phase + round((float(value) - phase) / h) * h

    # For every physical seam, find two PARALLEL lattice lines.  Their separation
    # is the nearest positive integer multiple of h to the original OptCuts copy
    # separation, so the embedding is disturbed as little as possible.
    for entry in entries:
        axis = int(entry["axis"])
        const_coord = 1 - axis
        means = [
            float(np.mean(rotated[np.asarray(copy, dtype=int), const_coord]))
            for copy in entry["copies"]
        ]
        center = 0.5 * (means[0] + means[1])
        original_sep = abs(means[1] - means[0])
        sep_steps = max(1, int(round(original_sep / h)))

        # Place the first line on the global lattice, then the second exactly an
        # integer number of units away.  Preserve the side ordering from OptCuts.
        c0 = snap(means[0], const_coord)
        sign = 1.0 if means[1] >= means[0] else -1.0
        c1 = c0 + sign * sep_steps * h
        # If the alternative placement is closer in total squared displacement,
        # anchor side1 and derive side0 instead.
        alt1 = snap(means[1], const_coord)
        alt0 = alt1 - sign * sep_steps * h
        cost_a = (c0 - means[0]) ** 2 + (c1 - means[1]) ** 2
        cost_b = (alt0 - means[0]) ** 2 + (alt1 - means[1]) ** 2
        if cost_b < cost_a:
            c0, c1 = alt0, alt1
        entry["constants"] = [float(c0), float(c1)]
        entry["separation_steps"] = int(sep_steps)

    uv_to_surface = constrained_param._uv_vertex_to_surface_vertex(parameterization)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    targets = np.full_like(rotated, np.nan)
    boundary_segments: list[list[list[float]]] = []
    separations: list[int] = []

    # Endpoint varying coordinates are kept close to the OptCuts proposal and
    # snapped to the SAME lattice phase.  Each UV copy keeps its own endpoint;
    # forcing copy endpoints to coincide is exactly what made v2 non-injective.
    for entry in entries:
        physical = entry["physical"]
        axis = int(entry["axis"])
        side_targets: list[tuple[np.ndarray, np.ndarray]] = []
        for side_id, copy in enumerate(entry["copies"]):
            const = float(entry["constants"][side_id])
            p0 = rotated[int(copy[0])].copy()
            p1 = rotated[int(copy[-1])].copy()
            p0[axis] = snap(float(p0[axis]), axis)
            p1[axis] = snap(float(p1[axis]), axis)
            p0[1 - axis] = const
            p1[1 - axis] = const
            if abs(float(p1[axis] - p0[axis])) < h:
                proposal_delta = float(rotated[int(copy[-1]), axis] - rotated[int(copy[0]), axis])
                p1[axis] = p0[axis] + (h if proposal_delta >= 0.0 else -h)
            side_targets.append((p0, p1))

        xyz = surface_xyz[np.asarray(physical, dtype=int)]
        lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = float(cumulative[-1])
        tvals = cumulative / total if total > 1.0e-12 else np.linspace(0.0, 1.0, len(physical))

        for side_id, copy in enumerate(entry["copies"]):
            p0, p1 = side_targets[side_id]
            for u, t in zip(copy, tvals):
                target = (1.0 - float(t)) * p0 + float(t) * p1
                if np.all(np.isfinite(targets[int(u)])):
                    # A UV vertex can be a junction of multiple seam chains.  Exact
                    # consistency is required; averaging would violate straightness.
                    if np.linalg.norm(targets[int(u)] - target) > max(1.0e-8, 1.0e-5 * h):
                        raise RuntimeError(
                            f"OPTCUTS_GRID_CONSTRAINT_JUNCTION_CONFLICT: uv={int(u)} "
                            f"old={targets[int(u)].tolist()} new={target.tolist()}"
                        )
                targets[int(u)] = target
            boundary_segments.append([p0.tolist(), p1.tolist()])
        separations.append(int(entry["separation_steps"]))

    fixed = np.all(np.isfinite(targets), axis=1)
    info = {
        "seam_chain_count": int(len(entries)),
        "seam_copy_chain_count": int(2 * len(entries)),
        "axis_rotation_degrees": float(np.degrees(-angle)),
        "constrained_vertex_count": int(np.count_nonzero(fixed)),
        "paired_straight_seam_copies": True,
        "zero_width_physical_seam": False,
        "seam_copy_lines_are_coincident": False,
        "seam_copy_separation_steps": separations,
        "seam_copy_separation_min_steps": int(min(separations)) if separations else 0,
        "seam_copy_separation_max_steps": int(max(separations)) if separations else 0,
        "physical_seam_segments": boundary_segments,
        "grid_phase_u": float(phases[0]),
        "grid_phase_v": float(phases[1]),
        "grid_origin_is_optimized_phase": True,
    }
    return rotated, targets, info


def install_injective_paired_grid_fusion() -> None:
    # The constrained solver owns orientation; disable the old rigid pre-rotation.
    def no_legacy_axis_rotation(parameterization: Any) -> float:
        parameterization.metrics["optcuts_legacy_axis_alignment_disabled"] = True
        return 0.0

    optcuts_pipeline._align_uv_to_optcuts_seam_axis = no_legacy_axis_rotation
    constrained_param._build_hard_seam_targets = _paired_line_hard_targets


__all__ = ["install_injective_paired_grid_fusion", "_paired_line_hard_targets"]
