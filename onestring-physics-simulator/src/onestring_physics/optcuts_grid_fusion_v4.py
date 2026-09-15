"""Global constraint compiler for the grid-constrained OneString/OptCuts seam.

v3 still chose each chain endpoint locally and only afterwards checked shared UV
junctions.  That is backwards: a junction is one lattice point and must be solved
once for the whole seam network.  This module compiles the complete seam network
before assigning any hard UV target.

Policy:
* one global orthogonal frame;
* fixed lattice spacing h=tile_size (phase may translate globally);
* every UV seam-copy chain is one horizontal or vertical straight segment;
* same-axis copy chains that share a UV junction share one lattice line;
* every UV junction is assigned exactly one lattice point and reused by every
  incident chain;
* no chain-local endpoint mutation is allowed;
* zero-length snapped chains are repaired only through a globally free endpoint;
  otherwise an explicit structural error is raised;
* a final preflight verifies lattice alignment, straightness, shared junctions,
  nonzero chain length, and unintended H/V segment crossings before optimization.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import optcuts_grid_constrained_parameterization_patch as constrained_param
from . import optcuts_pipeline_patch as optcuts_pipeline


def _junction_consistent_hard_targets(
    parameterization: Any,
    h: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h = max(float(h), 1.0e-12)
    uv0 = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    records = constrained_param._surface_seam_records(parameterization)
    physical_chains = constrained_param._physical_chains(records)
    lookup = constrained_param._record_lookup(records)

    physical_entries: list[dict[str, Any]] = []
    all_copy_chains: list[list[int]] = []
    for physical in physical_chains:
        if len(physical) >= 3 and int(physical[0]) == int(physical[-1]):
            raise RuntimeError(
                "OPTCUTS_GRID_CONSTRAINT_CLOSED_SEAM: closed physical seam cannot be represented by one straight segment"
            )
        side0, side1 = constrained_param._uv_copy_chains_for_physical_chain(physical, lookup)
        if len(side0) < 2 or len(side1) < 2:
            continue
        if len(side0) != len(physical) or len(side1) != len(physical):
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_COPY_CHAIN_LENGTH_MISMATCH")
        physical_entries.append({
            "physical": [int(x) for x in physical],
            "copies": [[int(x) for x in side0], [int(x) for x in side1]],
        })
        all_copy_chains.extend([side0, side1])

    if not physical_entries:
        return uv0.copy(), np.full_like(uv0, np.nan), {
            "seam_chain_count": 0,
            "seam_copy_chain_count": 0,
            "constrained_vertex_count": 0,
            "grid_phase_u": 0.0,
            "grid_phase_v": 0.0,
            "junction_count": 0,
            "constraint_preflight_passed": True,
        }

    angle = constrained_param._dominant_axis_angle(uv0, all_copy_chains)
    rotated, _ = constrained_param._rotate_uv(uv0, -angle)

    # One H/V direction per physical chain; both UV copies share that direction.
    for entry in physical_entries:
        deltas = []
        for copy in entry["copies"]:
            pts = rotated[np.asarray(copy, dtype=int)]
            deltas.append(pts[-1] - pts[0])
        d = np.mean(np.asarray(deltas, dtype=float), axis=0)
        entry["axis"] = 0 if abs(float(d[0])) >= abs(float(d[1])) else 1

    # Translate, but do not resize, the fixed h-lattice to minimize seam motion.
    seam_ids = np.asarray(sorted({u for e in physical_entries for c in e["copies"] for u in c}), dtype=int)
    phases = np.zeros(2, dtype=float)
    for coord in (0, 1):
        residual = np.mod(rotated[seam_ids, coord], h)
        angles = 2.0 * np.pi * residual / h
        z = np.mean(np.exp(1j * angles))
        phase_angle = float(np.angle(z))
        if phase_angle < 0.0:
            phase_angle += 2.0 * np.pi
        phases[coord] = h * phase_angle / (2.0 * np.pi)

    def snap(value: float, coord: int) -> float:
        return float(phases[coord] + round((float(value) - float(phases[coord])) / h) * h)

    # Flatten physical chains into UV-copy chains.  These are the actual UV
    # boundaries, and junction consistency is defined by shared UV endpoint ids.
    side_entries: list[dict[str, Any]] = []
    for pi, entry in enumerate(physical_entries):
        axis = int(entry["axis"])
        for side_id, copy in enumerate(entry["copies"]):
            side_entries.append({
                "physical_id": int(pi),
                "side_id": int(side_id),
                "axis": axis,
                "ids": [int(x) for x in copy],
            })

    # Union same-axis chains that meet at exactly the same UV junction.  Their
    # constant coordinate must be identical; deciding it per chain is the v3 bug.
    parent = list(range(len(side_entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    endpoint_incident: dict[int, list[int]] = defaultdict(list)
    for si, entry in enumerate(side_entries):
        endpoint_incident[int(entry["ids"][0])].append(si)
        endpoint_incident[int(entry["ids"][-1])].append(si)
    for touching in endpoint_incident.values():
        by_axis: dict[int, list[int]] = defaultdict(list)
        for si in touching:
            by_axis[int(side_entries[si]["axis"])].append(si)
        for same_axis in by_axis.values():
            for j in range(1, len(same_axis)):
                union(int(same_axis[0]), int(same_axis[j]))

    line_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for si, entry in enumerate(side_entries):
        line_groups[(int(entry["axis"]), find(si))].append(si)

    # Choose one lattice line per global line group, using all its chain samples.
    group_constant: dict[tuple[int, int], float] = {}
    for key, members in line_groups.items():
        axis, _root = key
        const_coord = 1 - int(axis)
        ids = sorted({u for si in members for u in side_entries[si]["ids"]})
        mean_value = float(np.mean(rotated[np.asarray(ids, dtype=int), const_coord]))
        group_constant[key] = snap(mean_value, const_coord)
        for si in members:
            side_entries[si]["line_group"] = key
            side_entries[si]["constant"] = float(group_constant[key])

    # Solve every UV junction once.  A vertical incident line fixes x; a
    # horizontal incident line fixes y.  The other coordinate is the nearest
    # lattice point to the original junction.  Same-axis incidents were unioned,
    # so multiple candidates must now agree by construction.
    junction_xy: dict[int, np.ndarray] = {}
    junction_fixed_coord: dict[int, np.ndarray] = {}
    for uv_id, touching in endpoint_incident.items():
        x_values = sorted({
            float(side_entries[si]["constant"])
            for si in touching if int(side_entries[si]["axis"]) == 1
        })
        y_values = sorted({
            float(side_entries[si]["constant"])
            for si in touching if int(side_entries[si]["axis"]) == 0
        })
        tol = max(1.0e-9, 1.0e-6 * h)
        if len(x_values) > 1 and max(x_values) - min(x_values) > tol:
            raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINT_VERTICAL_JUNCTION_CONFLICT: uv={uv_id} values={x_values}")
        if len(y_values) > 1 and max(y_values) - min(y_values) > tol:
            raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINT_HORIZONTAL_JUNCTION_CONFLICT: uv={uv_id} values={y_values}")
        p = rotated[int(uv_id)].copy()
        fixed = np.zeros(2, dtype=bool)
        if x_values:
            p[0] = float(x_values[0]); fixed[0] = True
        else:
            p[0] = snap(float(p[0]), 0)
        if y_values:
            p[1] = float(y_values[0]); fixed[1] = True
        else:
            p[1] = snap(float(p[1]), 1)
        junction_xy[int(uv_id)] = p
        junction_fixed_coord[int(uv_id)] = fixed

    # Repair only a truly free endpoint when two snapped endpoints collapse.
    # Never mutate a coordinate fixed by an orthogonal incident line.
    repaired_short = 0
    for entry in side_entries:
        axis = int(entry["axis"])
        a = int(entry["ids"][0]); b = int(entry["ids"][-1])
        pa = junction_xy[a]; pb = junction_xy[b]
        if abs(float(pb[axis] - pa[axis])) >= 0.5 * h:
            continue
        delta = float(rotated[b, axis] - rotated[a, axis])
        step = h if delta >= 0.0 else -h
        if not bool(junction_fixed_coord[b][axis]):
            junction_xy[b] = pb.copy(); junction_xy[b][axis] = pa[axis] + step
            repaired_short += 1
        elif not bool(junction_fixed_coord[a][axis]):
            junction_xy[a] = pa.copy(); junction_xy[a][axis] = pb[axis] - step
            repaired_short += 1
        else:
            raise RuntimeError(
                f"OPTCUTS_GRID_CONSTRAINT_ZERO_LENGTH_CHAIN: endpoints=({a},{b}) axis={axis} "
                "both endpoint coordinates are fixed by the global junction graph"
            )

    uv_to_surface = constrained_param._uv_vertex_to_surface_vertex(parameterization)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    targets = np.full_like(rotated, np.nan)
    segments: list[dict[str, Any]] = []

    for entry in side_entries:
        ids = entry["ids"]
        axis = int(entry["axis"])
        const = float(entry["constant"])
        p0 = junction_xy[int(ids[0])].copy()
        p1 = junction_xy[int(ids[-1])].copy()
        p0[1 - axis] = const
        p1[1 - axis] = const
        surface_ids = [int(uv_to_surface[u]) for u in ids]
        if any(s < 0 for s in surface_ids):
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_SEAM_VERTEX_WITHOUT_SURFACE_VERTEX")
        xyz = surface_xyz[np.asarray(surface_ids, dtype=int)]
        lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = float(cumulative[-1])
        tvals = cumulative / total if total > 1.0e-12 else np.linspace(0.0, 1.0, len(ids))
        for u, t in zip(ids, tvals):
            target = (1.0 - float(t)) * p0 + float(t) * p1
            if np.all(np.isfinite(targets[int(u)])):
                if np.linalg.norm(targets[int(u)] - target) > max(1.0e-8, 1.0e-6 * h):
                    raise RuntimeError(
                        f"OPTCUTS_GRID_CONSTRAINT_INTERNAL_ASSIGNMENT_CONFLICT: uv={int(u)} "
                        f"old={targets[int(u)].tolist()} new={target.tolist()}"
                    )
            targets[int(u)] = target
        segments.append({
            "uv0": int(ids[0]), "uv1": int(ids[-1]), "axis": axis,
            "p0": p0.tolist(), "p1": p1.tolist(),
            "physical_id": int(entry["physical_id"]), "side_id": int(entry["side_id"]),
        })

    # Global preflight: all constraints must already be mutually consistent before
    # the expensive continuation optimizer sees them.
    tol = max(1.0e-8, 1.0e-5 * h)
    fixed = np.all(np.isfinite(targets), axis=1)
    for entry in side_entries:
        ids = np.asarray(entry["ids"], dtype=int)
        pts = targets[ids]
        if not np.all(np.isfinite(pts)):
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_UNASSIGNED_SEAM_VERTEX")
        axis = int(entry["axis"]); const_coord = 1 - axis
        if float(np.max(np.abs(pts[:, const_coord] - pts[0, const_coord]))) > tol:
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_NONSTRAIGHT_CHAIN")
        for endpoint in (pts[0], pts[-1]):
            lattice_residual = np.abs((endpoint - phases) / h - np.round((endpoint - phases) / h))
            if float(np.max(lattice_residual)) > 1.0e-5:
                raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_ENDPOINT_OFF_LATTICE")
        if abs(float(pts[-1, axis] - pts[0, axis])) < 0.5 * h:
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_ZERO_LENGTH_CHAIN")

    # Detect unintended perpendicular crossings.  A crossing is allowed only when
    # the two segments explicitly share the same UV endpoint (a real junction).
    crossing_count = 0
    for i in range(len(segments)):
        a = segments[i]
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if int(a["axis"]) == int(b["axis"]):
                continue
            shared = {int(a["uv0"]), int(a["uv1"])} & {int(b["uv0"]), int(b["uv1"])}
            hseg = a if int(a["axis"]) == 0 else b
            vseg = b if int(a["axis"]) == 0 else a
            hp0 = np.asarray(hseg["p0"], dtype=float); hp1 = np.asarray(hseg["p1"], dtype=float)
            vp0 = np.asarray(vseg["p0"], dtype=float); vp1 = np.asarray(vseg["p1"], dtype=float)
            x = float(vp0[0]); y = float(hp0[1])
            inside_h = min(hp0[0], hp1[0]) + tol < x < max(hp0[0], hp1[0]) - tol
            inside_v = min(vp0[1], vp1[1]) + tol < y < max(vp0[1], vp1[1]) - tol
            if inside_h and inside_v and not shared:
                crossing_count += 1
    if crossing_count:
        raise RuntimeError(
            f"OPTCUTS_GRID_CONSTRAINT_UNINTENDED_SEGMENT_CROSSING: count={crossing_count}; "
            "the rectilinear seam-network layout itself is not planar"
        )

    # Separation diagnostics are measured after the global solve, not imposed as
    # conflicting per-chain hard constraints.
    separations: list[int] = []
    by_physical: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in side_entries:
        by_physical[int(entry["physical_id"])].append(entry)
    for copies in by_physical.values():
        if len(copies) == 2 and int(copies[0]["axis"]) == int(copies[1]["axis"]):
            axis = int(copies[0]["axis"]); cc = 1 - axis
            c0 = float(copies[0]["constant"]); c1 = float(copies[1]["constant"])
            separations.append(int(round(abs(c1 - c0) / h)))

    print(
        "[OPTCUTS-GRID-CONSTRAINT-PREFLIGHT] "
        f"copy_chains={len(side_entries)} junctions={len(junction_xy)} "
        f"line_groups={len(line_groups)} short_repairs={repaired_short} crossings=0"
    )

    return rotated, targets, {
        "seam_chain_count": int(len(physical_entries)),
        "seam_copy_chain_count": int(len(side_entries)),
        "axis_rotation_degrees": float(np.degrees(-angle)),
        "constrained_vertex_count": int(np.count_nonzero(fixed)),
        "paired_straight_seam_copies": True,
        "zero_width_physical_seam": False,
        "seam_copy_lines_are_globally_solved": True,
        "junction_count": int(len(junction_xy)),
        "line_group_count": int(len(line_groups)),
        "short_chain_repair_count": int(repaired_short),
        "constraint_preflight_passed": True,
        "unintended_segment_crossing_count": 0,
        "seam_copy_separation_steps": separations,
        "seam_copy_separation_min_steps": int(min(separations)) if separations else 0,
        "seam_copy_separation_max_steps": int(max(separations)) if separations else 0,
        "physical_seam_segments": [[s["p0"], s["p1"]] for s in segments],
        "grid_phase_u": float(phases[0]),
        "grid_phase_v": float(phases[1]),
        "grid_origin_is_optimized_phase": True,
    }


def install_global_grid_constraint_fusion() -> None:
    def no_legacy_axis_rotation(parameterization: Any) -> float:
        parameterization.metrics["optcuts_legacy_axis_alignment_disabled"] = True
        return 0.0

    optcuts_pipeline._align_uv_to_optcuts_seam_axis = no_legacy_axis_rotation
    constrained_param._build_hard_seam_targets = _junction_consistent_hard_targets


__all__ = ["install_global_grid_constraint_fusion", "_junction_consistent_hard_targets"]
