"""Global orthogonal-segment constraint compiler for OneString/OptCuts.

This is the strict implementation of the intended seam model:
* tile_size h is fixed in advance;
* one global orthogonal frame is estimated from the OptCuts proposal;
* the physical OptCuts cut network is decomposed into straight H/V segment runs;
  a bend is represented by a junction between two straight segments, never by a
  staircase approximation;
* closed cut loops are supported as multiple straight segment runs when their
  proposal contains direction changes;
* same-axis segments sharing a UV junction share one lattice line;
* every UV junction is solved once globally and reused by all incident segments;
* no chain-local endpoint snapping or mutation is allowed;
* all hard constraints are preflighted before UV optimization.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import optcuts_grid_constrained_parameterization_patch as constrained_param
from . import optcuts_pipeline_patch as optcuts_pipeline


def _segment_optcuts_chains(parameterization: Any, h: float):
    uv0 = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    records = constrained_param._surface_seam_records(parameterization)
    physical_chains = constrained_param._physical_chains(records)
    lookup = constrained_param._record_lookup(records)
    raw: list[dict[str, Any]] = []
    copies_all: list[list[int]] = []
    for physical in physical_chains:
        side0, side1 = constrained_param._uv_copy_chains_for_physical_chain(physical, lookup)
        if len(side0) < 2 or len(side1) < 2:
            continue
        if len(side0) != len(physical) or len(side1) != len(physical):
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_COPY_CHAIN_LENGTH_MISMATCH")
        raw.append({"physical": [int(x) for x in physical], "copies": [[int(x) for x in side0], [int(x) for x in side1]]})
        copies_all.extend([side0, side1])
    if not raw:
        return uv0.copy(), [], 0.0

    angle = constrained_param._dominant_axis_angle(uv0, copies_all)
    rotated, _ = constrained_param._rotate_uv(uv0, -angle)
    segmented: list[dict[str, Any]] = []

    for raw_id, entry in enumerate(raw):
        physical = list(entry["physical"])
        side0 = list(entry["copies"][0]); side1 = list(entry["copies"][1])
        closed = len(physical) >= 3 and int(physical[0]) == int(physical[-1])
        edge_count = len(physical) - 1
        if edge_count <= 0:
            continue
        axes: list[int] = []
        for i in range(edge_count):
            d0 = rotated[int(side0[i + 1])] - rotated[int(side0[i])]
            d1 = rotated[int(side1[i + 1])] - rotated[int(side1[i])]
            d = 0.5 * (d0 + d1)
            axes.append(0 if abs(float(d[0])) >= abs(float(d[1])) else 1)

        if closed:
            transitions = [i for i in range(edge_count) if axes[i] != axes[(i - 1) % edge_count]]
            if not transitions:
                raise RuntimeError(
                    "OPTCUTS_GRID_CONSTRAINT_CLOSED_COLLINEAR_LOOP: a closed seam loop cannot be represented by one straight segment"
                )
            start = int(transitions[0])
            unique_p = physical[:-1]; unique0 = side0[:-1]; unique1 = side1[:-1]
            order = list(range(start, edge_count)) + list(range(0, start))
            physical = [unique_p[i] for i in order] + [unique_p[start]]
            side0 = [unique0[i] for i in order] + [unique0[start]]
            side1 = [unique1[i] for i in order] + [unique1[start]]
            axes = [axes[i] for i in order]

        run_start = 0
        while run_start < len(axes):
            axis = int(axes[run_start])
            run_end = run_start
            while run_end + 1 < len(axes) and int(axes[run_end + 1]) == axis:
                run_end += 1
            sl = slice(run_start, run_end + 2)
            pseg = physical[sl]; c0 = side0[sl]; c1 = side1[sl]
            if len(pseg) >= 2:
                segmented.append({
                    "raw_id": int(raw_id),
                    "axis": axis,
                    "physical": [int(x) for x in pseg],
                    "copies": [[int(x) for x in c0], [int(x) for x in c1]],
                })
            run_start = run_end + 1
    return rotated, segmented, float(angle)


def _global_targets(parameterization: Any, h: float):
    h = max(float(h), 1.0e-12)
    rotated, segments_physical, angle = _segment_optcuts_chains(parameterization, h)
    if not segments_physical:
        return rotated.copy(), np.full_like(rotated, np.nan), {
            "seam_chain_count": 0, "seam_segment_count": 0, "seam_copy_chain_count": 0,
            "constrained_vertex_count": 0, "grid_phase_u": 0.0, "grid_phase_v": 0.0,
            "constraint_preflight_passed": True,
        }

    seam_ids = np.asarray(sorted({u for e in segments_physical for c in e["copies"] for u in c}), dtype=int)
    phases = np.zeros(2, dtype=float)
    for coord in (0, 1):
        residual = np.mod(rotated[seam_ids, coord], h)
        theta = 2.0 * np.pi * residual / h
        z = np.mean(np.exp(1j * theta))
        a = float(np.angle(z)); a = a + 2.0 * np.pi if a < 0.0 else a
        phases[coord] = h * a / (2.0 * np.pi)

    def snap(value: float, coord: int) -> float:
        return float(phases[coord] + round((float(value) - float(phases[coord])) / h) * h)

    side_entries: list[dict[str, Any]] = []
    for pi, entry in enumerate(segments_physical):
        for side_id, copy in enumerate(entry["copies"]):
            side_entries.append({
                "segment_id": int(pi), "raw_id": int(entry["raw_id"]), "side_id": int(side_id),
                "axis": int(entry["axis"]), "ids": [int(x) for x in copy],
            })

    parent = list(range(len(side_entries)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b: parent[b] = a

    endpoint_incident: dict[int, list[int]] = defaultdict(list)
    for si, e in enumerate(side_entries):
        endpoint_incident[int(e["ids"][0])].append(si)
        endpoint_incident[int(e["ids"][-1])].append(si)
    for touching in endpoint_incident.values():
        by_axis: dict[int, list[int]] = defaultdict(list)
        for si in touching: by_axis[int(side_entries[si]["axis"])].append(si)
        for same in by_axis.values():
            for j in range(1, len(same)): union(int(same[0]), int(same[j]))

    line_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for si, e in enumerate(side_entries): line_groups[(int(e["axis"]), find(si))].append(si)
    for key, members in line_groups.items():
        axis = int(key[0]); cc = 1 - axis
        ids = sorted({u for si in members for u in side_entries[si]["ids"]})
        const = snap(float(np.mean(rotated[np.asarray(ids, dtype=int), cc])), cc)
        for si in members:
            side_entries[si]["constant"] = float(const); side_entries[si]["line_group"] = key

    junction_xy: dict[int, np.ndarray] = {}
    junction_fixed: dict[int, np.ndarray] = {}
    tol = max(1.0e-9, 1.0e-6 * h)
    for uv_id, touching in endpoint_incident.items():
        xs = sorted({float(side_entries[si]["constant"]) for si in touching if int(side_entries[si]["axis"]) == 1})
        ys = sorted({float(side_entries[si]["constant"]) for si in touching if int(side_entries[si]["axis"]) == 0})
        if len(xs) > 1 and max(xs) - min(xs) > tol:
            raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINT_VERTICAL_JUNCTION_CONFLICT: uv={uv_id} values={xs}")
        if len(ys) > 1 and max(ys) - min(ys) > tol:
            raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINT_HORIZONTAL_JUNCTION_CONFLICT: uv={uv_id} values={ys}")
        p = rotated[int(uv_id)].copy(); fixed = np.zeros(2, dtype=bool)
        if xs: p[0] = xs[0]; fixed[0] = True
        else: p[0] = snap(float(p[0]), 0)
        if ys: p[1] = ys[0]; fixed[1] = True
        else: p[1] = snap(float(p[1]), 1)
        junction_xy[int(uv_id)] = p; junction_fixed[int(uv_id)] = fixed

    short_repairs = 0
    for e in side_entries:
        axis = int(e["axis"]); a = int(e["ids"][0]); b = int(e["ids"][-1])
        pa = junction_xy[a]; pb = junction_xy[b]
        if abs(float(pb[axis] - pa[axis])) >= 0.5 * h: continue
        sign = 1.0 if float(rotated[b, axis] - rotated[a, axis]) >= 0.0 else -1.0
        if not bool(junction_fixed[b][axis]):
            junction_xy[b] = pb.copy(); junction_xy[b][axis] = pa[axis] + sign * h; short_repairs += 1
        elif not bool(junction_fixed[a][axis]):
            junction_xy[a] = pa.copy(); junction_xy[a][axis] = pb[axis] - sign * h; short_repairs += 1
        else:
            raise RuntimeError(
                f"OPTCUTS_GRID_CONSTRAINT_ZERO_LENGTH_SEGMENT: endpoints=({a},{b}) axis={axis}; "
                "both coordinates are fixed by orthogonal junction constraints"
            )

    uv_to_surface = constrained_param._uv_vertex_to_surface_vertex(parameterization)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    targets = np.full_like(rotated, np.nan)
    drawn: list[dict[str, Any]] = []
    for e in side_entries:
        ids = e["ids"]; axis = int(e["axis"]); const = float(e["constant"])
        p0 = junction_xy[int(ids[0])].copy(); p1 = junction_xy[int(ids[-1])].copy()
        p0[1-axis] = const; p1[1-axis] = const
        sids = [int(uv_to_surface[u]) for u in ids]
        if any(s < 0 for s in sids): raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_SEAM_VERTEX_WITHOUT_SURFACE_VERTEX")
        xyz = surface_xyz[np.asarray(sids, dtype=int)]
        lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)]); total = float(cumulative[-1])
        tvals = cumulative / total if total > 1.0e-12 else np.linspace(0.0, 1.0, len(ids))
        for u, t in zip(ids, tvals):
            target = (1.0-float(t))*p0 + float(t)*p1
            if np.all(np.isfinite(targets[int(u)])) and np.linalg.norm(targets[int(u)]-target) > max(1.0e-8,1.0e-6*h):
                raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINT_INTERNAL_ASSIGNMENT_CONFLICT: uv={int(u)}")
            targets[int(u)] = target
        drawn.append({"uv0":int(ids[0]),"uv1":int(ids[-1]),"axis":axis,"p0":p0.tolist(),"p1":p1.tolist()})

    # Full preflight before the nonlinear solve.
    check_tol = max(1.0e-8, 1.0e-5*h)
    for e in side_entries:
        ids=np.asarray(e["ids"],dtype=int); pts=targets[ids]; axis=int(e["axis"]); cc=1-axis
        if not np.all(np.isfinite(pts)): raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_UNASSIGNED_SEAM_VERTEX")
        if float(np.max(np.abs(pts[:,cc]-pts[0,cc]))) > check_tol: raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_NONSTRAIGHT_SEGMENT")
        for p in (pts[0],pts[-1]):
            r=np.abs((p-phases)/h-np.round((p-phases)/h))
            if float(np.max(r))>1.0e-5: raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_ENDPOINT_OFF_LATTICE")
        if abs(float(pts[-1,axis]-pts[0,axis]))<0.5*h: raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_PREFLIGHT_ZERO_LENGTH_SEGMENT")

    crossings=0
    for i,a in enumerate(drawn):
        for b in drawn[i+1:]:
            if int(a["axis"])==int(b["axis"]): continue
            shared={int(a["uv0"]),int(a["uv1"])} & {int(b["uv0"]),int(b["uv1"])}
            hs=a if int(a["axis"])==0 else b; vs=b if int(a["axis"])==0 else a
            hp0=np.asarray(hs["p0"]); hp1=np.asarray(hs["p1"]); vp0=np.asarray(vs["p0"]); vp1=np.asarray(vs["p1"])
            x=float(vp0[0]); y=float(hp0[1])
            ih=min(hp0[0],hp1[0])+check_tol<x<max(hp0[0],hp1[0])-check_tol
            iv=min(vp0[1],vp1[1])+check_tol<y<max(vp0[1],vp1[1])-check_tol
            if ih and iv and not shared: crossings+=1
    if crossings:
        raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINT_UNINTENDED_SEGMENT_CROSSING: count={crossings}")

    print(
        "[OPTCUTS-GRID-CONSTRAINT-PREFLIGHT] "
        f"physical_runs={len(segments_physical)} copy_segments={len(side_entries)} "
        f"junctions={len(junction_xy)} line_groups={len(line_groups)} short_repairs={short_repairs} crossings=0"
    )
    return rotated,targets,{
        "seam_chain_count":int(len(set(int(e["raw_id"]) for e in segments_physical))),
        "seam_segment_count":int(len(segments_physical)),
        "seam_copy_chain_count":int(len(side_entries)),
        "axis_rotation_degrees":float(np.degrees(-angle)),
        "constrained_vertex_count":int(np.count_nonzero(np.all(np.isfinite(targets),axis=1))),
        "junction_count":int(len(junction_xy)),"line_group_count":int(len(line_groups)),
        "short_chain_repair_count":int(short_repairs),"constraint_preflight_passed":True,
        "unintended_segment_crossing_count":0,
        "physical_seam_segments":[[d["p0"],d["p1"]] for d in drawn],
        "grid_phase_u":float(phases[0]),"grid_phase_v":float(phases[1]),
        "grid_origin_is_optimized_phase":True,
        "orthogonal_segment_network":True,
    }


def install_orthogonal_segment_grid_fusion() -> None:
    def no_legacy_axis_rotation(parameterization: Any) -> float:
        parameterization.metrics["optcuts_legacy_axis_alignment_disabled"] = True
        return 0.0
    optcuts_pipeline._align_uv_to_optcuts_seam_axis = no_legacy_axis_rotation
    constrained_param._build_hard_seam_targets = _global_targets


__all__=["install_orthogonal_segment_grid_fusion","_global_targets","_segment_optcuts_chains"]
