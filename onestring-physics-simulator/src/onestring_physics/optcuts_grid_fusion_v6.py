"""Global integer-lattice embedding for OptCuts seam networks.

This replaces the v5 "snap then repair" logic. The whole seam endpoint graph is
embedded on the fixed h-lattice at once:

* each seam run is horizontal or vertical in one global frame;
* horizontal runs enforce equal y, vertical runs enforce equal x;
* every endpoint/junction is represented by shared integer lattice variables;
* run lengths are soft-fitted to the OptCuts proposal but constrained to be at
  least one lattice unit;
* same physical cut copies are softly registered along the varying coordinate;
* a global least-squares seed plus integer coordinate descent removes zero-length
  runs before the UV continuation solver is called.

The solver never mutates one chain endpoint independently after junctions have
been assigned, removing the JUNCTION_CONFLICT / ZERO_LENGTH repair loop.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import optcuts_grid_constrained_parameterization_patch as constrained_param
from . import optcuts_pipeline_patch as optcuts_pipeline


def _edge_axes_smoothed(rotated: np.ndarray, side0: list[int], side1: list[int], closed: bool) -> list[int]:
    n = len(side0) - 1
    if n <= 0:
        return []
    costs = np.zeros((n, 2), dtype=float)
    for i in range(n):
        d0 = rotated[int(side0[i + 1])] - rotated[int(side0[i])]
        d1 = rotated[int(side1[i + 1])] - rotated[int(side1[i])]
        d = 0.5 * (d0 + d1)
        norm = max(float(np.linalg.norm(d)), 1e-12)
        costs[i, 0] = abs(float(d[1])) / norm
        costs[i, 1] = abs(float(d[0])) / norm

    turn = 0.28
    if not closed:
        dp = np.full((n, 2), np.inf, dtype=float)
        prev = np.full((n, 2), -1, dtype=int)
        dp[0] = costs[0]
        for i in range(1, n):
            for s in (0, 1):
                vals = [dp[i - 1, p] + (turn if p != s else 0.0) for p in (0, 1)]
                p = int(np.argmin(vals))
                dp[i, s] = costs[i, s] + vals[p]
                prev[i, s] = p
        state = int(np.argmin(dp[-1]))
        axes = [0] * n
        axes[-1] = state
        for i in range(n - 1, 0, -1):
            axes[i - 1] = int(prev[i, axes[i]])
        return axes

    best_cost = np.inf
    best_axes: list[int] | None = None
    for start in (0, 1):
        dp = np.full((n, 2), np.inf, dtype=float)
        prev = np.full((n, 2), -1, dtype=int)
        dp[0, start] = costs[0, start]
        for i in range(1, n):
            for s in (0, 1):
                vals = [dp[i - 1, p] + (turn if p != s else 0.0) for p in (0, 1)]
                p = int(np.argmin(vals))
                dp[i, s] = costs[i, s] + vals[p]
                prev[i, s] = p
        for end in (0, 1):
            value = dp[-1, end] + (turn if end != start else 0.0)
            if value < best_cost:
                axes = [0] * n
                axes[-1] = end
                for i in range(n - 1, 0, -1):
                    axes[i - 1] = int(prev[i, axes[i]])
                best_cost = float(value)
                best_axes = axes
    axes = best_axes or [int(np.argmin(c)) for c in costs]
    if len(set(axes)) == 1 and n >= 2:
        current = axes[0]
        other = 1 - current
        gain = costs[:, other] - costs[:, current]
        i0 = int(np.argmin(gain))
        axes[i0] = other
        if n >= 4:
            candidates = [(abs(i - i0), i) for i in range(n) if i != i0]
            _, i1 = max(candidates)
            axes[int(i1)] = other
    return axes


def _segment_optcuts_chains(parameterization: Any):
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
            raise RuntimeError("OPTCUTS_GRID_EMBEDDING_SOURCE_TOPOLOGY: copy-chain length mismatch")
        raw.append({
            "physical": [int(x) for x in physical],
            "copies": [[int(x) for x in side0], [int(x) for x in side1]],
        })
        copies_all.extend([side0, side1])
    if not raw:
        return uv0.copy(), [], 0.0

    angle = constrained_param._dominant_axis_angle(uv0, copies_all)
    rotated, _ = constrained_param._rotate_uv(uv0, -angle)
    segmented: list[dict[str, Any]] = []

    for raw_id, entry in enumerate(raw):
        physical = list(entry["physical"])
        side0 = list(entry["copies"][0])
        side1 = list(entry["copies"][1])
        closed = len(physical) >= 3 and int(physical[0]) == int(physical[-1])
        edge_count = len(physical) - 1
        if edge_count <= 0:
            continue
        axes = _edge_axes_smoothed(rotated, side0, side1, closed)

        if closed:
            transitions = [i for i in range(edge_count) if axes[i] != axes[(i - 1) % edge_count]]
            if not transitions:
                raise RuntimeError("OPTCUTS_GRID_EMBEDDING_SOURCE_TOPOLOGY: closed seam has no orthogonal turn")
            start = int(transitions[0])
            unique_p, unique0, unique1 = physical[:-1], side0[:-1], side1[:-1]
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
            pseg, c0, c1 = physical[sl], side0[sl], side1[sl]
            if len(pseg) >= 2:
                segmented.append({
                    "raw_id": int(raw_id),
                    "axis": axis,
                    "physical": [int(x) for x in pseg],
                    "copies": [[int(x) for x in c0], [int(x) for x in c1]],
                })
            run_start = run_end + 1
    return rotated, segmented, float(angle)


class _DSU:
    def __init__(self, items: list[int]):
        self.parent = {int(x): int(x) for x in items}

    def find(self, x: int) -> int:
        x = int(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def _integer_axis_solve(
    endpoint_ids: list[int],
    dsu: _DSU,
    coordinate_values: np.ndarray,
    phase: float,
    h: float,
    varying_edges: list[tuple[int, int, float, float]],
    soft_pairs: list[tuple[int, int, float]],
) -> tuple[dict[int, int], dict[int, int], dict[str, Any]]:
    roots = sorted({dsu.find(v) for v in endpoint_ids})
    rindex = {r: i for i, r in enumerate(roots)}
    endpoint_group = {v: rindex[dsu.find(v)] for v in endpoint_ids}
    groups: dict[int, list[int]] = defaultdict(list)
    for v, g in endpoint_group.items():
        groups[g].append(v)

    rv = np.asarray(coordinate_values, dtype=float).reshape(-1)
    anchors = np.zeros(len(roots), dtype=float)
    for g, ids in groups.items():
        anchors[g] = float(np.mean([(rv[v] - phase) / h for v in ids]))

    edges: list[tuple[int, int, float, float, bool]] = []
    for a, b, desired, weight in varying_edges:
        ga, gb = endpoint_group[int(a)], endpoint_group[int(b)]
        if ga == gb:
            return {}, endpoint_group, {
                "ok": False,
                "reason": "varying_edge_self_loop",
                "endpoints": (int(a), int(b)),
            }
        d = float(desired)
        if abs(d) < 0.5:
            d = 1.0 if d >= 0.0 else -1.0
        edges.append((ga, gb, d, float(weight), True))
    for a, b, weight in soft_pairs:
        ga, gb = endpoint_group[int(a)], endpoint_group[int(b)]
        if ga != gb:
            edges.append((ga, gb, 0.0, float(weight), False))

    n = len(roots)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    anchor_w = 0.25
    for g in range(n):
        row = np.zeros(n, dtype=float)
        row[g] = np.sqrt(anchor_w)
        rows.append(row)
        rhs.append(np.sqrt(anchor_w) * anchors[g])
    for ga, gb, d, weight, _hard in edges:
        row = np.zeros(n, dtype=float)
        w = np.sqrt(max(weight, 1e-8))
        row[gb], row[ga] = w, -w
        rows.append(row)
        rhs.append(w * d)
    A = np.asarray(rows, dtype=float)
    b = np.asarray(rhs, dtype=float)
    continuous, *_ = np.linalg.lstsq(A, b, rcond=None)
    x = np.rint(continuous).astype(int)

    incident: dict[int, list[int]] = defaultdict(list)
    for ei, (ga, gb, *_rest) in enumerate(edges):
        incident[ga].append(ei)
        incident[gb].append(ei)

    hard_penalty = 1.0e5

    def edge_cost(ei: int, values: np.ndarray) -> float:
        ga, gb, d, weight, hard = edges[ei]
        delta = int(values[gb] - values[ga])
        c = float(weight) * (float(delta) - float(d)) ** 2
        if hard and delta == 0:
            c += hard_penalty
        return c

    def local_cost(g: int, value: int, values: np.ndarray) -> float:
        old = int(values[g])
        values[g] = int(value)
        c = anchor_w * (float(values[g]) - anchors[g]) ** 2
        for ei in incident.get(g, []):
            c += edge_cost(ei, values)
        values[g] = old
        return float(c)

    for _ in range(50):
        changed = False
        for g in range(n):
            base = int(x[g])
            center = int(round(float(continuous[g])))
            candidates = sorted(set([base, center] + [base + d for d in (-5, -3, -2, -1, 1, 2, 3, 5)]))
            best = min(candidates, key=lambda v: local_cost(g, int(v), x))
            if int(best) != base:
                x[g] = int(best)
                changed = True
        if not changed:
            break

    zero_edges = []
    for ga, gb, _d, _weight, hard in edges:
        if hard and int(x[gb] - x[ga]) == 0:
            zero_edges.append((ga, gb))
    if zero_edges:
        return {}, endpoint_group, {
            "ok": False,
            "reason": "integer_embedding_zero_length",
            "zero_edges": zero_edges[:16],
        }

    result = {root: int(x[rindex[root]]) for root in roots}
    return result, endpoint_group, {
        "ok": True,
        "group_count": int(n),
        "edge_count": int(len(varying_edges)),
        "soft_pair_count": int(len(soft_pairs)),
        "max_anchor_shift_steps": float(np.max(np.abs(x - anchors))) if len(x) else 0.0,
    }


def _global_targets(parameterization: Any, h: float):
    h = max(float(h), 1.0e-12)
    rotated, segments_physical, angle = _segment_optcuts_chains(parameterization)
    if not segments_physical:
        return rotated.copy(), np.full_like(rotated, np.nan), {
            "seam_chain_count": 0,
            "seam_segment_count": 0,
            "seam_copy_chain_count": 0,
            "constrained_vertex_count": 0,
            "grid_phase_u": 0.0,
            "grid_phase_v": 0.0,
            "constraint_preflight_passed": True,
        }

    seam_ids = np.asarray(sorted({u for e in segments_physical for c in e["copies"] for u in c}), dtype=int)
    phases = np.zeros(2, dtype=float)
    for coord in (0, 1):
        residual = np.mod(rotated[seam_ids, coord], h)
        theta = 2.0 * np.pi * residual / h
        z = np.mean(np.exp(1j * theta))
        a = float(np.angle(z))
        if a < 0.0:
            a += 2.0 * np.pi
        phases[coord] = h * a / (2.0 * np.pi)

    side_entries: list[dict[str, Any]] = []
    by_segment: dict[int, list[int]] = defaultdict(list)
    for pi, entry in enumerate(segments_physical):
        for side_id, copy in enumerate(entry["copies"]):
            idx = len(side_entries)
            side_entries.append({
                "segment_id": int(pi),
                "raw_id": int(entry["raw_id"]),
                "side_id": int(side_id),
                "axis": int(entry["axis"]),
                "ids": [int(x) for x in copy],
            })
            by_segment[int(pi)].append(idx)

    endpoints = sorted({int(e["ids"][0]) for e in side_entries} | {int(e["ids"][-1]) for e in side_entries})
    dsu_x, dsu_y = _DSU(endpoints), _DSU(endpoints)
    for e in side_entries:
        a, b, axis = int(e["ids"][0]), int(e["ids"][-1]), int(e["axis"])
        if axis == 0:
            dsu_y.union(a, b)
        else:
            dsu_x.union(a, b)

    varying_x: list[tuple[int, int, float, float]] = []
    varying_y: list[tuple[int, int, float, float]] = []
    for e in side_entries:
        a, b, axis = int(e["ids"][0]), int(e["ids"][-1]), int(e["axis"])
        delta = float((rotated[b, axis] - rotated[a, axis]) / h)
        desired = float(int(round(delta)))
        if desired == 0.0:
            desired = 1.0 if delta >= 0.0 else -1.0
        (varying_x if axis == 0 else varying_y).append((a, b, desired, 2.5))

    soft_x: list[tuple[int, int, float]] = []
    soft_y: list[tuple[int, int, float]] = []
    for _pi, sis in by_segment.items():
        if len(sis) != 2:
            continue
        e0, e1 = side_entries[sis[0]], side_entries[sis[1]]
        axis = int(e0["axis"])
        pairs = [
            (int(e0["ids"][0]), int(e1["ids"][0])),
            (int(e0["ids"][-1]), int(e1["ids"][-1])),
        ]
        (soft_x if axis == 0 else soft_y).extend((a, b, 0.6) for a, b in pairs)

    x_roots, _x_group, xinfo = _integer_axis_solve(
        endpoints, dsu_x, rotated[:, 0], float(phases[0]), h, varying_x, soft_x
    )
    y_roots, _y_group, yinfo = _integer_axis_solve(
        endpoints, dsu_y, rotated[:, 1], float(phases[1]), h, varying_y, soft_y
    )
    if not bool(xinfo.get("ok")) or not bool(yinfo.get("ok")):
        raise RuntimeError(
            "OPTCUTS_GRID_EMBEDDING_INFEASIBLE: global integer lattice solve failed; "
            f"x={xinfo} y={yinfo}. The current OptCuts cut topology/HV assignment is "
            "not embeddable at this tile_size without changing the cut proposal."
        )

    endpoint_xy: dict[int, np.ndarray] = {}
    for u in endpoints:
        rx, ry = dsu_x.find(u), dsu_y.find(u)
        endpoint_xy[int(u)] = np.asarray([
            float(phases[0]) + float(x_roots[rx]) * h,
            float(phases[1]) + float(y_roots[ry]) * h,
        ], dtype=float)

    uv_to_surface = constrained_param._uv_vertex_to_surface_vertex(parameterization)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    targets = np.full_like(rotated, np.nan)
    drawn: list[dict[str, Any]] = []

    for e in side_entries:
        ids = [int(x) for x in e["ids"]]
        axis = int(e["axis"])
        p0, p1 = endpoint_xy[ids[0]].copy(), endpoint_xy[ids[-1]].copy()
        if abs(float(p1[1 - axis] - p0[1 - axis])) > 1e-8 * max(h, 1.0):
            raise RuntimeError("OPTCUTS_GRID_EMBEDDING_INTERNAL: constant-coordinate equality lost")
        if abs(float(p1[axis] - p0[axis])) < 0.5 * h:
            raise RuntimeError("OPTCUTS_GRID_EMBEDDING_INTERNAL: zero-length segment survived integer solve")

        sids = [int(uv_to_surface[u]) for u in ids]
        if any(s < 0 for s in sids):
            raise RuntimeError("OPTCUTS_GRID_EMBEDDING_SOURCE_TOPOLOGY: seam vertex without surface vertex")
        xyz = surface_xyz[np.asarray(sids, dtype=int)]
        lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = float(cumulative[-1])
        tvals = cumulative / total if total > 1e-12 else np.linspace(0.0, 1.0, len(ids))
        for u, t in zip(ids, tvals):
            target = (1.0 - float(t)) * p0 + float(t) * p1
            if np.all(np.isfinite(targets[u])) and np.linalg.norm(targets[u] - target) > max(1e-8, 1e-6 * h):
                raise RuntimeError(
                    "OPTCUTS_GRID_EMBEDDING_INTERNAL: shared seam vertex received inconsistent global target"
                )
            targets[u] = target
        drawn.append({
            "uv0": ids[0],
            "uv1": ids[-1],
            "axis": axis,
            "p0": p0.tolist(),
            "p1": p1.tolist(),
        })

    defects: list[str] = []
    tol = max(1e-8, 1e-5 * h)
    for e in side_entries:
        ids = np.asarray(e["ids"], dtype=int)
        pts = targets[ids]
        axis = int(e["axis"])
        if not np.all(np.isfinite(pts)):
            defects.append(f"unassigned:{ids[0]}-{ids[-1]}")
            continue
        if float(np.max(np.abs(pts[:, 1 - axis] - pts[0, 1 - axis]))) > tol:
            defects.append(f"nonstraight:{ids[0]}-{ids[-1]}")
        if abs(float(pts[-1, axis] - pts[0, axis])) < 0.5 * h:
            defects.append(f"zero_length:{ids[0]}-{ids[-1]}")
        for p in (pts[0], pts[-1]):
            residual = np.abs((p - phases) / h - np.round((p - phases) / h))
            if float(np.max(residual)) > 1e-5:
                defects.append(f"off_lattice:{ids[0]}-{ids[-1]}")
                break

    crossings = 0
    for i, a in enumerate(drawn):
        for b in drawn[i + 1:]:
            if int(a["axis"]) == int(b["axis"]):
                continue
            shared = {int(a["uv0"]), int(a["uv1"])} & {int(b["uv0"]), int(b["uv1"])}
            hs = a if int(a["axis"]) == 0 else b
            vs = b if int(a["axis"]) == 0 else a
            hp0, hp1 = np.asarray(hs["p0"]), np.asarray(hs["p1"])
            vp0, vp1 = np.asarray(vs["p0"]), np.asarray(vs["p1"])
            x, y = float(vp0[0]), float(hp0[1])
            inside_h = min(hp0[0], hp1[0]) + tol < x < max(hp0[0], hp1[0]) - tol
            inside_v = min(vp0[1], vp1[1]) + tol < y < max(vp0[1], vp1[1]) - tol
            if inside_h and inside_v and not shared:
                crossings += 1
    if crossings:
        defects.append(f"unintended_crossings:{crossings}")

    if defects:
        raise RuntimeError(
            "OPTCUTS_GRID_EMBEDDING_INFEASIBLE: global preflight failed after integer solve; "
            f"defects={defects[:24]}"
        )

    print(
        "[OPTCUTS-GRID-EMBEDDING] "
        f"physical_runs={len(segments_physical)} copy_segments={len(side_entries)} "
        f"junctions={len(endpoints)} x_groups={xinfo['group_count']} y_groups={yinfo['group_count']} "
        f"crossings=0 max_shift_steps=({xinfo['max_anchor_shift_steps']:.3g},"
        f"{yinfo['max_anchor_shift_steps']:.3g})"
    )
    return rotated, targets, {
        "seam_chain_count": int(len(set(int(e["raw_id"]) for e in segments_physical))),
        "seam_segment_count": int(len(segments_physical)),
        "seam_copy_chain_count": int(len(side_entries)),
        "axis_rotation_degrees": float(np.degrees(-angle)),
        "constrained_vertex_count": int(np.count_nonzero(np.all(np.isfinite(targets), axis=1))),
        "junction_count": int(len(endpoints)),
        "constraint_preflight_passed": True,
        "unintended_segment_crossing_count": 0,
        "physical_seam_segments": [[d["p0"], d["p1"]] for d in drawn],
        "grid_phase_u": float(phases[0]),
        "grid_phase_v": float(phases[1]),
        "grid_origin_is_optimized_phase": True,
        "orthogonal_segment_network": True,
        "integer_lattice_embedding": True,
        "x_embedding": xinfo,
        "y_embedding": yinfo,
    }


def install_integer_lattice_grid_fusion() -> None:
    def no_legacy_axis_rotation(parameterization: Any) -> float:
        parameterization.metrics["optcuts_legacy_axis_alignment_disabled"] = True
        return 0.0
    optcuts_pipeline._align_uv_to_optcuts_seam_axis = no_legacy_axis_rotation
    constrained_param._build_hard_seam_targets = _global_targets


__all__ = ["install_integer_lattice_grid_fusion", "_global_targets"]
