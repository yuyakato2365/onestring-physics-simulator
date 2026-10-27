"""OneString extension (NOT Zhang et al. 2021): hard-closable seam sampling.

Zhang's planar quad mesh Q does not know that two stretches of the chart
boundary are the two UV copies of one OptCuts cut.  For the panels to close
**exactly** in 3D (hard seam) every cut branch must be sampled consistently:

* matched partition: both copies carry their quad vertices at the same 3D
  points (vertex <-> vertex);
* nested partition: the copy with fewer vertices is sampled at a subset of
  the other copy's 3D points; the extra vertices are T-vertices that lie on
  the coarse copy's panel edge at a fixed 3D arc-length parameter
  (vertex <-> edge parameter).  Both are exact linear constraints that the
  K3D solvers enforce as equalities (``zhang2021_seam_reduction``).

The subset is chosen by an order-preserving minimum-displacement matching of
3D arc-length fractions (dynamic programming); matched vertices are moved to
their mean fraction, nothing else is moved.  Interleaved partitions (which
would force a whole seam branch straight) are never produced.

Seam nodes (cut tips, junctions, cut/open transitions) are quad corners.  A
cut tip / junction is an interior point of S, so the panels around it must
close a full turn in 3D: its 3D valence (sum of the quad valences of all its
UV copies) must be >= 3 (one panel folds onto itself, two panels give 180
degree corners).  A deficient node is repaired by a local diagonal collapse
of one of its quads (the opposite interior corner is merged into the node);
this changes no boundary vertex and keeps the parity of every interior
valence.  If no valid collapse exists the candidate is rejected, never
passed with a warning.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .zhang2021_onestring_boundary import (
    SourceBoundary,
    _cyclic_strictly_increasing,
    _fraction_3d,
    _param_from_fraction_3d,
    indices_inside,
    quad_boundary_loop,
    run_interval,
)


class SeamAlignmentError(RuntimeError):
    pass


INTERIOR_NODE_KINDS = ("tip", "junction")


# ------------------------------------------------------------------ valence
def node_copies(src: SourceBoundary, q_loop: list[int], s: np.ndarray) -> dict[int, list[dict]]:
    """3D node id -> its UV copies (loop position of the Q vertex placed on it)."""
    P = src.perimeter
    out: dict[int, list[dict]] = defaultdict(list)
    for j in src.nodes:
        ns = float(src.cum[j]) % P
        d = np.abs((np.asarray(s) - ns + P / 2) % P - P / 2)
        k = int(np.argmin(d))
        exact = bool(d[k] <= 1e-9 * max(P, 1.0))
        out[int(src.xyz_id[j])].append({"loop_position": int(j), "kind": src.node_kind.get(j, "?"),
                                        "q_index": int(q_loop[k]) if exact else -1, "k": k if exact else -1})
    return out


def node_valence_report(src: SourceBoundary, q_loop: list[int], s: np.ndarray, quads: np.ndarray,
                        *, max_interior_valence: int = 8) -> list[dict]:
    val = np.bincount(np.asarray(quads, int).ravel())
    rows = []
    for xid, copies in sorted(node_copies(src, q_loop, s).items()):
        kinds = sorted({c["kind"] for c in copies})
        interior = not ("transition" in kinds)
        vals = [int(val[c["q_index"]]) if c["q_index"] >= 0 and c["q_index"] < len(val) else 0 for c in copies]
        total = int(sum(vals))
        lo = 3 if interior else 2
        ok = all(c["q_index"] >= 0 for c in copies) and lo <= total <= (max_interior_valence if interior else max_interior_valence - 2)
        rows.append({"xyz_id": int(xid), "kinds": kinds, "copies": len(copies), "uv_valences": vals,
                     "valence_3d": total, "required_min": lo, "interior_point": interior,
                     "ok": bool(ok)})
    return rows


def _edge_faces(quads: np.ndarray) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for qi, q in enumerate(quads):
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            out[(min(a, b), max(a, b))].append(qi)
    return out


def _valid_after(quads: np.ndarray, boundary: set[int]) -> bool:
    ef = _edge_faces(quads)
    if any(len(v) > 2 for v in ef.values()):
        return False
    for q in quads:
        if len(set(int(x) for x in q)) != 4:
            return False
    val = np.bincount(np.asarray(quads, int).ravel())
    used = np.flatnonzero(val)
    for v in used:
        if int(v) not in boundary and val[v] < 3:
            return False
    try:
        loop = quad_boundary_loop(quads)
    except Exception:  # noqa: BLE001 - topological check
        return False
    return set(loop) == boundary


def diagonal_collapse(quads: np.ndarray, qi: int, keep: int) -> np.ndarray:
    """Remove quad qi and merge its corner opposite to ``keep`` into ``keep``."""
    q = [int(x) for x in quads[qi]]
    k = q.index(int(keep))
    x = q[(k + 2) % 4]
    out = [list(map(int, f)) for i, f in enumerate(quads) if i != qi]
    for f in out:
        for j in range(4):
            if f[j] == x:
                f[j] = int(keep)
    return np.asarray(out, int)


def edge_rotation_towards(quads: np.ndarray, qi: int, rj: int, t: int) -> np.ndarray | None:
    """Rotate the edge shared by quads qi, rj so that the new diagonal of their hexagon ends at t.

    The union of two quads sharing an edge is a hexagon h0..h5; the shared
    edge is one of its three long diagonals.  Replacing it by (t, t+3) adds
    one quad to t (valence +1) and removes one from each old endpoint.
    """
    Q = [int(x) for x in quads[qi]]
    R = [int(x) for x in quads[rj]]
    shared = set(Q) & set(R)
    if len(shared) != 2:
        return None
    i = next((k for k in range(4) if Q[k] in shared and Q[(k + 1) % 4] in shared), None)
    if i is None:
        return None
    u, v = Q[i], Q[(i + 1) % 4]
    j = next((k for k in range(4) if R[k] == v and R[(k + 1) % 4] == u), None)
    if j is None:
        return None
    H = [v, Q[(i + 2) % 4], Q[(i + 3) % 4], u, R[(j + 2) % 4], R[(j + 3) % 4]]
    if len(set(H)) != 6 or t not in H or t in (u, v):
        return None
    k = H.index(t)
    new_a = [H[k], H[(k + 1) % 6], H[(k + 2) % 6], H[(k + 3) % 6]]
    new_b = [H[(k + 3) % 6], H[(k + 4) % 6], H[(k + 5) % 6], H[k]]
    out = [list(map(int, f)) for f in quads]
    out[qi] = new_a
    out[rj] = new_b
    return np.asarray(out, int)


def _repair_candidates(quads: np.ndarray, t: int, boundary: set[int]):
    """(kind, trial quads, description) of local ops that add panels at vertex t."""
    ef = _edge_faces(quads)
    for qi in np.flatnonzero(np.any(quads == t, axis=1)):
        q = [int(x) for x in quads[qi]]
        x = q[(q.index(t) + 2) % 4]
        if x not in boundary:
            yield "diagonal_collapse", diagonal_collapse(quads, int(qi), t), {"quad": int(qi), "merged_vertex": x}
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            if t in (a, b):
                continue
            owners = [f for f in ef[(min(a, b), max(a, b))] if f != qi]
            if len(owners) != 1:
                continue
            trial = edge_rotation_towards(quads, int(qi), int(owners[0]), t)
            if trial is not None:
                yield "edge_rotation", trial, {"quads": [int(qi), int(owners[0])], "old_edge": [a, b]}


def repair_node_valences(src: SourceBoundary, q_loop: list[int], s: np.ndarray, quads: np.ndarray, *,
                         max_interior_valence: int = 8, max_rounds: int = 16) -> tuple[np.ndarray, list[dict]]:
    """Local quad-topology repairs until every interior seam node has 3D valence >= 3.

    Operations (boundary vertices and their order never change):
    * diagonal collapse of a node quad (opposite interior corner merged into the node),
    * edge rotation of a quad edge opposite the node (adds one panel at the node).
    Every trial must keep a valid disk quad mesh (manifold edges, no doublets:
    interior valence >= 3).  Nodes that cannot be repaired are reported.
    """
    quads = np.asarray(quads, int).copy()
    log: list[dict] = []
    boundary = set(int(v) for v in q_loop)
    failed: set[int] = set()
    for _ in range(int(max_rounds)):
        report = node_valence_report(src, q_loop, s, quads, max_interior_valence=max_interior_valence)
        bad = [r for r in report if r["interior_point"] and r["valence_3d"] < 3
               and all(v > 0 for v in r["uv_valences"]) and r["xyz_id"] not in failed]
        if not bad:
            break
        r = bad[0]
        copies = node_copies(src, q_loop, s)[r["xyz_id"]]
        val = np.bincount(quads.ravel(), minlength=int(quads.max()) + 1)
        best = None
        for c in copies:
            t = c["q_index"]
            for kind, trial, info in _repair_candidates(quads, t, boundary):
                if not _valid_after(trial, boundary):
                    continue
                new_total = r["valence_3d"] - int(val[t]) + int(np.sum(trial == t))
                if new_total < 3 or new_total > max_interior_valence:
                    continue
                # prefer a result near the regular valence 4, then the op that
                # keeps the panel count (edge rotation)
                score = (abs(new_total - 4), 0 if kind == "edge_rotation" else 1)
                if best is None or score < best[0]:
                    best = (score, trial, kind, info, int(t), new_total)
        if best is None:
            failed.add(r["xyz_id"])
            log.append({"xyz_id": r["xyz_id"], "kinds": r["kinds"], "valence_3d": r["valence_3d"],
                        "status": "no_valid_local_repair"})
            continue
        _, quads, kind, info, t, new_total = best
        log.append({"xyz_id": r["xyz_id"], "kinds": r["kinds"], "valence_3d_before": r["valence_3d"],
                    "valence_3d_after": int(new_total), "operation": kind, "node_vertex": t, **info,
                    "status": "repaired"})
    return quads, log


def compact_vertices(quad_uv: np.ndarray, quads: np.ndarray, q_loop: list[int]):
    """Drop unreferenced vertices; returns (uv, quads, q_loop, old->new map)."""
    used = np.unique(np.asarray(quads, int))
    remap = -np.ones(len(quad_uv), int)
    remap[used] = np.arange(len(used))
    return (np.asarray(quad_uv, float)[used], remap[np.asarray(quads, int)],
            [int(remap[v]) for v in q_loop], remap)


# ---------------------------------------------------------------- alignment
@dataclass
class SeamAlignment:
    s: np.ndarray
    pairs: list[tuple[int, int]] = field(default_factory=list)        # loop positions (k_a, k_b)
    t_vertices: list[int] = field(default_factory=list)              # loop positions
    branches: list[dict] = field(default_factory=list)


def _match_subset(f_small: list[float], f_large: list[float]) -> list[int]:
    """Order-preserving injective matching of all f_small into f_large (min sum |diff|)."""
    m, n = len(f_small), len(f_large)
    if m == 0:
        return []
    if m > n:
        raise SeamAlignmentError("ZHANG2021_ONESTRING_SEAM_MATCH_SIZES")
    INF = float("inf")
    D = np.full((m + 1, n + 1), INF)
    take = np.zeros((m + 1, n + 1), bool)          # True: f_small[i-1] <-> f_large[j-1]
    D[0, :] = 0.0
    for i in range(1, m + 1):
        for j in range(i, n + 1):
            skip = D[i, j - 1]
            match = D[i - 1, j - 1] + abs(f_small[i - 1] - f_large[j - 1])
            if match <= skip:
                D[i, j], take[i, j] = match, True
            else:
                D[i, j] = skip
    out = [0] * m
    i, j = m, n
    while i > 0:
        if take[i, j]:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return out


def align_branches(src: SourceBoundary, constraints, s: np.ndarray, *, min_gap_fraction: float = 1e-3) -> SeamAlignment:
    """Matched / nested 3D sampling of every cut branch (see module doc)."""
    P = src.perimeter
    s = np.asarray(s, float).copy()
    res = SeamAlignment(s=s)
    for ci, c in enumerate(constraints):
        lo_a, hi_a = run_interval(src, c.run_a)
        lo_b, hi_b = run_interval(src, c.run_b)
        ia = indices_inside(s, lo_a, hi_a, P)
        ib = indices_inside(s, lo_b, hi_b, P)
        fa = [_fraction_3d(src, lo_a, hi_a, v) for v, _ in ia]
        # side B runs opposite: its fraction in side A's direction is 1 - f
        fb_pairs = sorted((1.0 - _fraction_3d(src, lo_b, hi_b, v), k) for v, k in ib)
        fb = [f for f, _ in fb_pairs]
        kb = [k for _, k in fb_pairs]
        ka = [k for _, k in ia]
        entry = {"constraint": ci, "kind": c.kind, "vertices_a": len(fa), "vertices_b": len(fb)}
        if len(fa) <= len(fb):
            small_f, small_k, large_f, large_k, small_side = fa, ka, fb, kb, "a"
        else:
            small_f, small_k, large_f, large_k, small_side = fb, kb, fa, ka, "b"
        match = _match_subset(small_f, large_f)
        new_f_small = list(small_f)
        new_f_large = list(large_f)
        shifts = []
        for i, j in enumerate(match):
            f = 0.5 * (small_f[i] + large_f[j])
            shifts.append(abs(small_f[i] - large_f[j]))
            new_f_small[i] = f
            new_f_large[j] = f
            pa, pb = (small_k[i], large_k[j]) if small_side == "a" else (large_k[j], small_k[i])
            res.pairs.append((int(pa), int(pb)))
        matched = set(match)
        t_idx = [j for j in range(len(large_f)) if j not in matched]
        # keep T-vertices strictly between their (moved) neighbours
        order = np.asarray(new_f_large)
        for j in t_idx:
            lo = order[j - 1] if j > 0 else 0.0
            hi = order[j + 1] if j + 1 < len(order) else 1.0
            if not (lo + min_gap_fraction < order[j] < hi - min_gap_fraction):
                order[j] = 0.5 * (lo + hi)
        new_f_large = list(order)
        if np.any(np.diff(np.r_[0.0, new_f_large, 1.0]) <= 0) or np.any(np.diff(np.r_[0.0, new_f_small, 1.0]) <= 0):
            raise SeamAlignmentError(f"ZHANG2021_ONESTRING_SEAM_ALIGNMENT_ORDER: constraint {ci}")
        res.t_vertices.extend(int(large_k[j]) for j in t_idx)
        for f, k in zip(new_f_small if small_side == "a" else new_f_large, ka):
            s[k] = _param_from_fraction_3d(src, lo_a, hi_a, f) % P
        for f, k in zip(new_f_large if small_side == "a" else new_f_small, kb):
            s[k] = _param_from_fraction_3d(src, lo_b, hi_b, 1.0 - f) % P
        entry.update({"matched": len(match), "t_vertices": len(t_idx),
                      "partition": "matched" if not t_idx else "nested",
                      "max_alignment_shift_fraction": float(max(shifts) if shifts else 0.0)})
        res.branches.append(entry)
    if not _cyclic_strictly_increasing(s, P):
        raise SeamAlignmentError("ZHANG2021_ONESTRING_SEAM_ALIGNMENT_BROKE_ORDER")
    res.s = s
    return res


__all__ = ["SeamAlignmentError", "SeamAlignment", "align_branches", "node_valence_report",
           "repair_node_valences", "diagonal_collapse", "compact_vertices", "node_copies"]
