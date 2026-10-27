"""OneString extension (NOT part of Zhang et al. 2021): seam-count matching.

Zhang et al. mesh the planar chart Omega without knowing that two pieces of
its boundary are the two UV copies of one OptCuts cut.  Panel edges on the
two copies can be paired exactly in 3D ("hard" seam) only if both copies
carry the same number of quad boundary vertices:

* a cut branch between two seam nodes (junctions / cut-open transitions)
  needs n_A == n_B interior quad vertices on its two copies;
* a slit (a branch ending at a cut tip) is one boundary stretch
  node -> tip -> node; it is balanced when the number of quad vertices
  strictly inside the stretch is odd, the middle one being assigned to the
  tip (parity condition).

This module refines the paper's quad mesh Q by **poly-chord insertion**
(sheet insertion): the chord of quads starting at a boundary edge is split
lengthwise, which adds one boundary vertex at each of its two ends and
keeps the mesh all-quad, conforming and convex (each quad is cut between
points of two opposite edges).  A Dijkstra search over the imbalance state
(differences for branches, parities for slits) selects the set of chord
types of minimum total added quads; the chords are then inserted one by
one.  Q's vertices are never moved and no quad is removed.  Remaining
imbalances (when no combination of insertions can fix them) are reported
and the corresponding seams stay "soft".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import heapq

import numpy as np

from .zhang2021_onestring_boundary import (
    InitialCorrespondence,
    SourceBoundary,
    _cyclic_strictly_increasing,
    _fraction_3d,
    branch_pairs,
    indices_inside,
    quad_boundary_loop,
    run_interval,
)


# --------------------------------------------------------------- chords
@dataclass
class Chord:
    quads: list[int]                 # quads in order
    entry: list[int]                 # local index i: entering rung is (q[i], q[i+1])
    start_edge: tuple[int, int]      # boundary edge in loop order
    end_edge: tuple[int, int]        # boundary edge in loop order


def _edge_quads(quads: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    out: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for qi, q in enumerate(quads):
        for i in range(4):
            a, b = int(q[i]), int(q[(i + 1) % 4])
            out[(min(a, b), max(a, b))].append((qi, i))
    return out


def trace_chord(quads: np.ndarray, edge_quads, start_quad: int, start_local: int) -> Chord | None:
    """Walk the poly-chord through opposite edges; None if it is not simple."""
    q_list, entry = [], []
    qi, i = start_quad, start_local
    seen = set()
    start = (int(quads[qi][i]), int(quads[qi][(i + 1) % 4]))
    while True:
        if qi in seen:
            return None
        seen.add(qi)
        q_list.append(qi)
        entry.append(i)
        q = quads[qi]
        a, b = int(q[(i + 2) % 4]), int(q[(i + 3) % 4])      # exit edge in quad orientation
        owners = [o for o in edge_quads[(min(a, b), max(a, b))] if o[0] != qi]
        if not owners:
            return Chord(quads=q_list, entry=entry, start_edge=start, end_edge=(a, b))
        q2, j = owners[0]
        q2v = quads[q2]
        if not (int(q2v[j]) == b and int(q2v[(j + 1) % 4]) == a):
            return None
        qi, i = q2, j


def insert_chord(uv: np.ndarray, quads: np.ndarray, chord: Chord, t_start: float,
                 t_end: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Split every quad of the chord between its entering and exiting rungs.

    Rung k is (L_k, R_k); its new vertex is L_k + t_k (R_k - L_k), t_k linear
    from ``t_start`` (first rung) to ``t_end`` (last rung).  For the first
    rung L_0 -> R_0 is the start edge in loop order; for the last rung
    R_n -> L_n is the end edge in loop order.
    """
    uv = np.asarray(uv, float)
    quads = np.asarray(quads, int)
    n = len(chord.quads)
    rungs = []
    for qi, i in zip(chord.quads, chord.entry):
        q = quads[qi]
        rungs.append((int(q[i]), int(q[(i + 1) % 4])))
    q = quads[chord.quads[-1]]
    i_last = chord.entry[-1]
    rungs.append((int(q[(i_last + 3) % 4]), int(q[(i_last + 2) % 4])))
    new_pts, new_ids = [], []
    for k, (L, R) in enumerate(rungs):
        t = t_start + (t_end - t_start) * (k / n)
        new_pts.append(uv[L] + t * (uv[R] - uv[L]))
        new_ids.append(len(uv) + k)
    uv2 = np.concatenate([uv, np.asarray(new_pts)])
    out = [list(map(int, q)) for q in quads]
    for k, (qi, i) in enumerate(zip(chord.quads, chord.entry)):
        q = quads[qi]
        a0, a1, a2, a3 = (int(q[(i + r) % 4]) for r in range(4))
        m_in, m_out = new_ids[k], new_ids[k + 1]
        out[qi] = [a0, m_in, m_out, a3]
        out.append([m_in, a1, a2, m_out])
    return uv2, np.asarray(out, int), new_ids[0], new_ids[-1]


# ---------------------------------------------------------- constraints
@dataclass
class _Constraint:
    kind: str                         # "pair" (n_A == n_B) or "slit" (parity)
    run_a: tuple[int, int]            # pair: side +1;  slit: node -> tip
    run_b: tuple[int, int]            # pair: side -1;  slit: tip -> node
    tip: int = -1                     # slit: loop position of the tip


def seam_constraints(src: SourceBoundary) -> list[_Constraint]:
    out = []
    for ra, rb in branch_pairs(src):
        (a, b), (a2, b2) = ra, rb
        if b == a2 and src.node_kind.get(b) == "tip":
            out.append(_Constraint("slit", ra, rb, b))
        elif b2 == a and src.node_kind.get(a) == "tip":
            out.append(_Constraint("slit", rb, ra, a))
        else:
            out.append(_Constraint("pair", ra, rb))
    return out


def _state(src: SourceBoundary, cons: list[_Constraint], s: np.ndarray) -> tuple[int, ...]:
    P = src.perimeter
    out = []
    for c in cons:
        na = len(indices_inside(s, *run_interval(src, c.run_a), P))
        nb = len(indices_inside(s, *run_interval(src, c.run_b), P))
        out.append(na - nb if c.kind == "pair" else (na + nb) % 2)
    return tuple(out)


def _label(src: SourceBoundary, cons: list[_Constraint], s_mid: float) -> tuple[int, int]:
    """(constraint index, effect of adding one vertex there) or (-1, 0)."""
    P = src.perimeter
    for ci, c in enumerate(cons):
        for run, sign in ((c.run_a, +1), (c.run_b, -1)):
            lo, hi = run_interval(src, run)
            v = s_mid if s_mid >= lo else s_mid + P
            if lo < v < hi:
                return ci, (sign if c.kind == "pair" else 1)
    return -1, 0


def _apply_effect(state, cons, effects) -> tuple[int, ...]:
    st = list(state)
    for ci, e in effects:
        if ci < 0:
            continue
        if cons[ci].kind == "pair":
            st[ci] += e
        else:
            st[ci] = (st[ci] + 1) % 2
    return tuple(st)


def _plan(state0, cons, moves: dict[tuple, int], max_cost: int):
    """Dijkstra over imbalance states; moves: effect-key -> cheapest chord length."""
    goal = tuple(0 for _ in cons)
    if state0 == goal:
        return []
    bound = max([abs(x) for x in state0] + [1]) + 2
    best = {state0: 0}
    heap = [(0, state0, ())]
    while heap:
        cost, st, path = heapq.heappop(heap)
        if st == goal:
            return list(path)
        if cost > best.get(st, 1 << 60) or cost > max_cost:
            continue
        for key, c in moves.items():
            nst = _apply_effect(st, cons, key)
            if any(abs(x) > bound for x in nst):
                continue
            nc = cost + c
            if nc <= max_cost and nc < best.get(nst, 1 << 60):
                best[nst] = nc
                heapq.heappush(heap, (nc, nst, path + (key,)))
    return None


@dataclass
class SeamMatchingResult:
    quad_uv: np.ndarray
    quads: np.ndarray
    init: InitialCorrespondence
    inserted_chords: int
    added_quads: int
    state_before: dict[str, int]
    state_after: dict[str, int]
    constraints: list[dict]
    log: list[dict] = field(default_factory=list)
    accepted: bool = True
    rejection_reason: str = ""


def _xyz_at(src: SourceBoundary, s: float) -> np.ndarray:
    j, t, _ = src.point_at(s)
    return src.xyz[j] + t * (src.xyz[(j + 1) % src.n] - src.xyz[j])


def _unmatched_targets(src: SourceBoundary, s: np.ndarray, c: _Constraint, side: int) -> list[float]:
    """3D-arc fractions (in the deficient side's direction) of the surplus side's
    vertices left unmatched by an order-preserving minimum-cost matching."""
    P = src.perimeter
    mine, other = (c.run_a, c.run_b) if side == +1 else (c.run_b, c.run_a)
    m0, m1 = run_interval(src, mine)
    o0, o1 = run_interval(src, other)
    fb = [_fraction_3d(src, m0, m1, v) for v, _ in indices_inside(s, m0, m1, P)]
    fa = sorted(1.0 - _fraction_3d(src, o0, o1, v) for v, _ in indices_inside(s, o0, o1, P))
    m, n = len(fa), len(fb)
    if m <= n:
        return []
    INF = float("inf")
    # D[i][j]: best cost using fa[:i] with fb[:j] all matched
    D = np.full((m + 1, n + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        D[i, 0] = 0.0
        for j in range(1, min(i, n) + 1):
            D[i, j] = min(D[i - 1, j], D[i - 1, j - 1] + abs(fa[i - 1] - fb[j - 1]))
    i, j, unmatched = m, n, []
    while i > 0:
        if j > 0 and np.isclose(D[i, j], D[i - 1, j - 1] + abs(fa[i - 1] - fb[j - 1])):
            i, j = i - 1, j - 1
        else:
            unmatched.append(fa[i - 1])
            i -= 1
    return sorted(unmatched)


def _edge_fraction_range(src: SourceBoundary, c: _Constraint, side: int, sa: float, sb: float):
    P = src.perimeter
    mine = c.run_a if side == +1 else c.run_b
    m0, m1 = run_interval(src, mine)
    fa = _fraction_3d(src, m0, m1, sa if sa >= m0 else sa + P)
    fb = _fraction_3d(src, m0, m1, sb if sb >= m0 else sb + P)
    return fa, fb


def _boundary_chords(src, cons, uv, qs, s_of):
    """Every simple chord starting at a boundary edge, with its effect key."""
    P = src.perimeter
    loop = quad_boundary_loop(qs)
    nxt = {loop[k]: loop[(k + 1) % len(loop)] for k in range(len(loop))}
    eq = _edge_quads(qs)
    out = []
    for k in range(len(loop)):
        a, b = loop[k], loop[(k + 1) % len(loop)]
        owner = eq[(min(a, b), max(a, b))]
        if len(owner) != 1:
            continue
        qi, i = owner[0]
        if not (int(qs[qi][i]) == a and int(qs[qi][(i + 1) % 4]) == b):
            continue
        ch = trace_chord(qs, eq, qi, i)
        if ch is None:
            continue
        ea, eb = ch.end_edge
        if nxt.get(ea) != eb:
            continue
        ends = []
        for (x, y) in ((a, b), (ea, eb)):
            sx, sy = s_of[x], s_of[y]
            ends.append(_label(src, cons, sx + 0.5 * ((sy - sx) % P)))
        if (a, b) == (ea, eb):
            continue
        key = tuple(sorted(ends))
        out.append((key, ch, (a, b), (ea, eb)))
    return out


# A cut tip is an interior point of S: in 3D the panels around it must fill
# 360 degrees.  With the two sides of the slit paired exactly, a tip carried by
# one quad folds that quad onto itself and a tip carried by two quads makes two
# panels share two edges (degenerate); at least three are required.
MIN_TIP_VALENCE = 3


def _reassign_tips(src: SourceBoundary, cons: list[_Constraint], s: np.ndarray,
                   loop: list[int] | None = None, valence: np.ndarray | None = None) -> list[dict]:
    """Assign the middle vertex of every balanced slit stretch to its tip.

    A middle vertex incident to a single quad is refused: both boundary edges
    at the tip would belong to one panel, which folds onto itself in 3D once
    the two sides of the slit are paired.
    """
    P = src.perimeter
    report = []
    for c in cons:
        if c.kind != "slit":
            continue
        lo, _ = run_interval(src, c.run_a)
        _, hi = run_interval(src, c.run_b)
        if hi <= lo:
            hi += P
        tip_s = float(src.cum[c.tip])
        tip_u = tip_s if tip_s >= lo else tip_s + P
        inside = indices_inside(s, lo, hi, P)          # includes the current tip vertex
        T = len(inside)
        if T % 2 == 0 or T == 0:
            report.append({"tip": int(c.tip), "vertices_in_stretch": T, "status": "unbalanced"})
            continue
        mid = T // 2
        vals = [v for v, _ in inside]
        idx = [k for _, k in inside]
        if loop is not None and valence is not None:
            val_mid = int(valence[loop[idx[mid]]])
            if val_mid < MIN_TIP_VALENCE:
                report.append({"tip": int(c.tip), "vertices_in_stretch": T,
                               "status": "invalid_tip_valence", "middle_vertex_valence": val_mid})
                continue
        # keep strict order: only the values that would cross the tip are
        # re-spaced (evenly between their last valid neighbour and the tip)
        left = list(vals[:mid])
        right = list(vals[mid + 1:])
        bad = next((q for q in range(len(left)) if left[q] >= tip_u), len(left))
        if bad < len(left):
            base = left[bad - 1] if bad > 0 else lo
            nb = len(left) - bad
            left[bad:] = [base + (tip_u - base) * (q + 1) / (nb + 1) for q in range(nb)]
        good = next((q for q in range(len(right)) if right[q] > tip_u), len(right))
        if good > 0:
            top = right[good] if good < len(right) else hi
            right[:good] = [tip_u + (top - tip_u) * (q + 1) / (good + 1) for q in range(good)]
        new = left + [tip_u] + right
        moved = int(np.argmin(np.abs(np.asarray(vals) - tip_u)))
        for k, v in zip(idx, new):
            s[k] = v % P
        rep = {"tip": int(c.tip), "vertices_in_stretch": T, "status": "balanced",
               "tip_vertex_shift_in_stretch": int(mid - moved)}
        if loop is not None and valence is not None:
            rep["tip_vertex_valence"] = int(valence[loop[idx[mid]]])
        report.append(rep)
    return report


def match_seam_counts(src: SourceBoundary, quad_uv: np.ndarray, quads: np.ndarray,
                      init: InitialCorrespondence, *, max_added_fraction: float = 0.6,
                      max_insertions: int = 32) -> SeamMatchingResult:
    """Minimum-added-quads poly-chord insertion making every cut matchable."""
    P = src.perimeter
    cons = seam_constraints(src)
    uv = np.asarray(quad_uv, float).copy()
    qs = np.asarray(quads, int).copy()
    s_of = {int(v): float(x) for v, x in zip(init.q_loop, init.s)}
    budget = int(max_added_fraction * len(qs))
    log: list[dict] = []

    def loop_and_s():
        loop = quad_boundary_loop(qs)
        return loop, np.asarray([s_of[v] for v in loop])

    loop, s = loop_and_s()
    state0 = _state(src, cons, s)
    added = inserted = 0
    for _ in range(int(max_insertions)):
        state = _state(src, cons, s)
        if all(x == 0 for x in state):
            break
        chords = _boundary_chords(src, cons, uv, qs, s_of)
        moves: dict[tuple, int] = {}
        for key, ch, _, _ in chords:
            if _apply_effect(state, cons, key) == state:
                continue
            moves[key] = min(moves.get(key, 1 << 30), len(ch.quads))
        plan = _plan(state, cons, moves, budget - added)
        if not plan:
            break
        key = plan[0]
        # Among the chords realising this effect, prefer one whose start edge
        # (on a deficient pair side) contains an unmatched vertex of the
        # opposite copy, so the new vertex lands where its partner is; never
        # re-split the same small edge repeatedly.
        best_c = None
        for k2, ch, (a, b), (ea, eb) in chords:
            if k2 != key:
                continue
            sa, sb = s_of[a], s_of[b]
            ci, eff = _label(src, cons, sa + 0.5 * ((sb - sa) % P))
            t0, score = 0.5, 1.0
            if ci >= 0 and cons[ci].kind == "pair" and _apply_effect(state, cons, ((ci, eff),))[ci] == 0 or (
                    ci >= 0 and cons[ci].kind == "pair"
                    and abs(_apply_effect(state, cons, ((ci, eff),))[ci]) < abs(state[ci])):
                lo_f, hi_f = _edge_fraction_range(src, cons[ci], eff, sa, sb)
                targets = [f for f in _unmatched_targets(src, s, cons[ci], eff) if lo_f < f < hi_f]
                if targets and hi_f > lo_f:
                    f = max(targets, key=lambda x: min(x - lo_f, hi_f - x))
                    t0 = float(np.clip((f - lo_f) / (hi_f - lo_f), 0.25, 0.75))
                    score = 0.0
                else:
                    score = 0.5
            elen = float(np.linalg.norm(uv[a] - uv[b]))
            rank = (score, len(ch.quads), -elen)
            if best_c is None or rank < best_c[0]:
                best_c = (rank, ch, (a, b), (ea, eb), t0)
        _, ch, (a, b), (ea, eb), t0 = best_c
        sa, sb = s_of[a], s_of[b]
        uv, qs, v_start, v_end = insert_chord(uv, qs, ch, t0, 0.5)
        s_of[v_start] = (sa + t0 * ((sb - sa) % P)) % P
        se_a, se_b = s_of[ea], s_of[eb]
        s_of[v_end] = (se_a + 0.5 * ((se_b - se_a) % P)) % P
        added += len(ch.quads)
        inserted += 1
        log.append({"start_edge": [int(a), int(b)], "end_edge": [int(ea), int(eb)],
                    "chord_quads": len(ch.quads), "effect": [list(e) for e in key]})
        loop, s = loop_and_s()
        if not _cyclic_strictly_increasing(s, P):
            raise RuntimeError("ZHANG2021_ONESTRING_SEAM_MATCHING_BROKE_ORDER")
    loop, s = loop_and_s()
    valence = np.bincount(qs.ravel(), minlength=len(uv))
    tips = _reassign_tips(src, cons, s, loop, valence)
    if not _cyclic_strictly_increasing(s, P):
        raise RuntimeError("ZHANG2021_ONESTRING_TIP_REASSIGNMENT_BROKE_ORDER")
    state1 = _state(src, cons, s)
    names = [f"{c.kind}{i}" for i, c in enumerate(cons)]
    complete = all(x == 0 for x in state1) and all(t["status"] == "balanced" for t in tips)
    constraints = ([{"name": n, "kind": c.kind, "run_a": list(c.run_a), "run_b": list(c.run_b),
                     "tip": int(c.tip)} for n, c in zip(names, cons)] + [{"tip_reassignment": tips}])
    if not complete:
        # Never hand on a partially matched mesh: the existing seam closure is
        # all-or-nothing ("hard" needs every branch), so extra quads would buy
        # nothing.  Keep the paper's Q and report why matching was rejected.
        loop0 = list(init.q_loop)
        return SeamMatchingResult(
            quad_uv=np.asarray(quad_uv, float), quads=np.asarray(quads, int), init=init,
            inserted_chords=0, added_quads=0,
            state_before=dict(zip(names, map(int, state0))), state_after=dict(zip(names, map(int, state0))),
            constraints=constraints, log=log, accepted=False,
            rejection_reason=("unbalanced: " + str(dict(zip(names, map(int, state1))))
                              if not all(x == 0 for x in state1)
                              else f"cut tip carried by fewer than {MIN_TIP_VALENCE} quads "
                                   "(hard closure would fold or degenerate the tip panels)"))
    init2 = InitialCorrespondence(q_loop=loop, s=s, projection_distance=np.zeros(len(loop)),
                                  snapped_nodes=init.snapped_nodes, unsnapped_nodes=init.unsnapped_nodes)
    return SeamMatchingResult(
        quad_uv=uv, quads=qs, init=init2, inserted_chords=inserted, added_quads=added,
        state_before=dict(zip(names, map(int, state0))), state_after=dict(zip(names, map(int, state1))),
        constraints=constraints, log=log, accepted=True)
