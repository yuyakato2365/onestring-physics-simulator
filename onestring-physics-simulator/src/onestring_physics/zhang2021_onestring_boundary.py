"""OneString extension (NOT part of Zhang et al. 2021): boundary correspondence.

The reference pipeline returns a planar quad mesh Q whose boundary
approximates the OptCuts chart boundary dOmega within the paper's two-sided
Hausdorff bound; the final optimisation moves Q's boundary vertices freely
(they are generally *not* on dOmega).  OneString needs every panel corner to
correspond to an exact point of the cut/open boundary of the 3D surface S,
and both UV copies of an OptCuts cut to be subdivided compatibly.

This module decides, for every boundary vertex of Q, a point of dOmega
(an arc-length parameter on the source UV boundary loop):

1. closest-point projection (must be strictly monotone; fail otherwise),
2. optional snapping of the nearest Q vertex onto every *seam node*
   (cut tip, cut junction, cut/open transition) so that panel corners sit on
   the nodes where the two sides of a cut separate,
3. optional *pair alignment*: when the two UV copies of a cut branch carry
   the same number of Q vertices, both copies are assigned the same 3D
   positions along the branch (exact "hard" seam), otherwise left as is
   (soft seam, handled downstream by affine constraints),

and realises the chosen points on the source triangulation by splitting the
boundary edge (and, for a cut edge, its twin copy, so the 3D mesh stays
conforming).  Q itself is never modified.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


class BoundaryCorrespondenceError(RuntimeError):
    pass


def uv_to_xyz_vertex_map(f3: np.ndarray, f2: np.ndarray, n_uv: int) -> np.ndarray:
    f3 = np.asarray(f3, int)
    f2 = np.asarray(f2, int)
    out = np.full(int(n_uv), -1, int)
    for a, b in zip(f3.reshape(-1), f2.reshape(-1)):
        if out[b] == -1:
            out[b] = a
        elif out[b] != a:
            raise BoundaryCorrespondenceError(
                f"UV vertex {int(b)} maps to two different 3D vertices ({int(out[b])}, {int(a)})")
    if np.any(out < 0):
        raise BoundaryCorrespondenceError("UV vertex without a paired 3D vertex")
    return out


def oriented_boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Boundary loops following the faces' orientation (face on the left for CCW)."""
    cnt: dict[tuple[int, int], int] = defaultdict(int)
    directed: dict[tuple[int, int], tuple[int, int]] = {}
    for t in np.asarray(faces, int):
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            k = (min(a, b), max(a, b))
            cnt[k] += 1
            directed[k] = (a, b)
    nxt: dict[int, int] = {}
    for k, c in cnt.items():
        if c == 1:
            a, b = directed[k]
            if a in nxt:
                raise BoundaryCorrespondenceError("non-manifold boundary vertex")
            nxt[a] = b
    loops, seen = [], set()
    for s in sorted(nxt):
        if s in seen:
            continue
        loop = [s]
        seen.add(s)
        x = nxt[s]
        while x != s:
            if x in seen:
                raise BoundaryCorrespondenceError("boundary is not a set of simple loops")
            loop.append(x)
            seen.add(x)
            x = nxt[x]
        loops.append(loop)
    return loops


@dataclass
class SourceBoundary:
    """One oriented UV boundary loop of the source chart with 3D provenance."""

    loop: list[int]                      # UV vertex ids, in face orientation
    uv: np.ndarray                       # (n, 2) positions of loop vertices
    xyz_id: np.ndarray                   # 3D vertex id of each loop vertex
    cum: np.ndarray                      # (n+1,) cumulative UV arc length
    xyz: np.ndarray                      # (n, 3) 3D positions of loop vertices
    cum3: np.ndarray                     # (n+1,) cumulative 3D arc length
    edge_kind: list[str]                 # "cut" / "open" for edge j = (j, j+1)
    twin: np.ndarray                     # twin edge index (cut) or -1
    nodes: list[int] = field(default_factory=list)   # loop vertex positions
    node_kind: dict[int, str] = field(default_factory=dict)   # "tip" / "junction" / "transition"

    @property
    def n(self) -> int:
        return len(self.loop)

    @property
    def perimeter(self) -> float:
        return float(self.cum[-1])

    def point_at(self, s: float) -> tuple[int, float, np.ndarray]:
        """(edge index j, fraction t, UV point) for arc parameter s."""
        s = float(s) % self.perimeter
        j = int(np.searchsorted(self.cum, s, side="right") - 1)
        j = min(max(j, 0), self.n - 1)
        length = self.cum[j + 1] - self.cum[j]
        t = 0.0 if length <= 0 else (s - self.cum[j]) / length
        a = self.uv[j]
        b = self.uv[(j + 1) % self.n]
        return j, float(t), a + t * (b - a)

    def project(self, q: np.ndarray) -> tuple[float, float]:
        a = self.uv
        b = np.roll(self.uv, -1, axis=0)
        d = b - a
        dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
        t = np.clip(np.einsum("ij,ij->i", q[None, :] - a, d) / dd, 0.0, 1.0)
        proj = a + t[:, None] * d
        dist = np.linalg.norm(proj - q[None, :], axis=1)
        j = int(np.argmin(dist))
        return float(self.cum[j] + t[j] * (self.cum[j + 1] - self.cum[j])), float(dist[j])


def build_source_boundary(uv: np.ndarray, f2: np.ndarray, u2x: np.ndarray,
                          xyz: np.ndarray) -> SourceBoundary:
    loops = oriented_boundary_loops(f2)
    if len(loops) != 1:
        raise BoundaryCorrespondenceError(
            f"ZHANG2021_ONESTRING_SOURCE_NOT_DISK: {len(loops)} UV boundary loops "
            "(the OneString adapter supports a single OptCuts disk chart)")
    loop = loops[0]
    P = np.asarray(uv, float)[loop]
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    xid = np.asarray(u2x, int)[loop]
    X = np.asarray(xyz, float)[xid]
    seg3 = np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1)
    cum3 = np.r_[0.0, np.cumsum(seg3)]
    n = len(loop)
    key_pos: dict[tuple[int, int], list[int]] = defaultdict(list)
    for j in range(n):
        a, b = int(xid[j]), int(xid[(j + 1) % n])
        key_pos[(min(a, b), max(a, b))].append(j)
    twin = np.full(n, -1, int)
    kind = []
    for j in range(n):
        a, b = int(xid[j]), int(xid[(j + 1) % n])
        pos = key_pos[(min(a, b), max(a, b))]
        if len(pos) == 2:
            twin[j] = pos[0] if pos[1] == j else pos[1]
            kind.append("cut")
        elif len(pos) == 1:
            kind.append("open")
        else:
            raise BoundaryCorrespondenceError("3D edge appears more than twice on the UV boundary")
    # Seam nodes: vertex j where the incoming and outgoing edges are not two
    # consecutive edges of one branch side (twin continuity breaks), where the
    # edge kind changes, or a slit tip (incoming edge is twin of outgoing).
    nodes = []
    node_kind: dict[int, str] = {}
    for j in range(n):
        ein, eout = (j - 1) % n, j
        if kind[ein] != kind[eout]:
            nodes.append(j)
            node_kind[j] = "transition"      # cut <-> open boundary
            continue
        if kind[eout] == "cut":
            if twin[eout] == ein:          # slit tip a->b->a
                nodes.append(j)
                node_kind[j] = "tip"
                continue
            # continuity: the other side runs in reverse
            if twin[ein] != (twin[eout] + 1) % n:
                nodes.append(j)
                node_kind[j] = "junction"
    return SourceBoundary(loop=loop, uv=P, xyz_id=xid, cum=cum, xyz=X, cum3=cum3,
                          edge_kind=kind, twin=twin, nodes=nodes, node_kind=node_kind)


def quad_boundary_loop(quads: np.ndarray) -> list[int]:
    cnt: dict[tuple[int, int], int] = defaultdict(int)
    directed = {}
    for q in np.asarray(quads, int):
        for i in range(4):
            a, b = int(q[i]), int(q[(i + 1) % 4])
            k = (min(a, b), max(a, b))
            cnt[k] += 1
            directed[k] = (a, b)
    nxt = {}
    for k, c in cnt.items():
        if c == 1:
            a, b = directed[k]
            nxt[a] = b
    if not nxt:
        raise BoundaryCorrespondenceError("quad mesh has no boundary")
    s = min(nxt)
    loop = [s]
    x = nxt[s]
    while x != s:
        loop.append(x)
        x = nxt[x]
    if len(loop) != len(nxt):
        raise BoundaryCorrespondenceError(
            "ZHANG2021_ONESTRING_QUAD_NOT_DISK: quad mesh boundary has several loops")
    return loop


@dataclass
class Correspondence:
    q_loop: list[int]
    s: np.ndarray                 # arc parameter on source loop per Q boundary vertex
    projection_distance: np.ndarray
    snapped_nodes: int
    aligned_pairs: int
    misaligned_branches: int
    branch_report: list[dict]
    regularized_points: int = 0
    unsnapped_nodes: list[dict] = field(default_factory=list)


def _cyclic_strictly_increasing(s: np.ndarray, perimeter: float) -> bool:
    k = int(np.argmin(s))
    ss = np.r_[s[k:], s[:k]]
    return bool(np.all(np.diff(ss) > 0) and ss[-1] - ss[0] < perimeter)


@dataclass
class InitialCorrespondence:
    """Projection of Q's boundary onto dOmega plus seam-node snapping."""

    q_loop: list[int]
    s: np.ndarray
    projection_distance: np.ndarray
    snapped_nodes: int
    unsnapped_nodes: list[dict]


def _projection_candidates(src: SourceBoundary, q: np.ndarray, slack: float):
    """Local minima (along dOmega) of the distance from q to the source loop."""
    a = src.uv
    b = np.roll(src.uv, -1, axis=0)
    d = b - a
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
    t = np.clip(np.einsum("ij,ij->i", q[None, :] - a, d) / dd, 0.0, 1.0)
    dist = np.linalg.norm(a + t[:, None] * d - q[None, :], axis=1)
    par = src.cum[:-1] + t * (src.cum[1:] - src.cum[:-1])
    dmin = float(dist.min())
    keep = (dist <= np.roll(dist, 1)) & (dist <= np.roll(dist, -1)) & (dist <= dmin + slack)
    idx = np.flatnonzero(keep)
    if not len(idx):
        idx = np.asarray([int(np.argmin(dist))])
    # collapse plateaus (an endpoint shared by two edges appears twice)
    out = {}
    for j in idx:
        key = round(float(par[j]) % src.perimeter, 12)
        if key not in out or dist[j] < out[key]:
            out[key] = float(dist[j])
    return sorted(out.items())


def monotone_projection(src: SourceBoundary, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order-preserving correspondence of a closed polyline with dOmega.

    Plain closest-point projection is not order preserving where two pieces
    of dOmega are closer than the quad mesh's boundary error (the two sides
    of a narrow cut).  Among the local projection candidates of every point
    the cyclically increasing assignment of minimum total distance is chosen
    (dynamic programming); if none exists the correspondence fails.
    """
    P = src.perimeter
    pts = np.asarray(points, float)
    m = len(pts)
    span = float(np.max(np.ptp(src.uv, axis=0)))
    first = [_projection_candidates(src, q, 0.0) for q in pts]
    base = max(c[0][1] for c in first)
    cands = [_projection_candidates(src, q, max(2.0 * base, 0.02 * span)) for q in pts]
    best_total, best_path = np.inf, None
    # anchor: the point with the fewest candidates
    k0 = int(np.argmin([len(c) for c in cands]))
    order = [(k0 + r) % m for r in range(m)]
    for s0, d0 in cands[k0]:
        prev = [(d0, s0, -1)]               # (cost, unwrapped param, back pointer)
        back = [prev]
        ok = True
        for k in order[1:]:
            cur = []
            for sc, dc in cands[k]:
                u = sc if sc > s0 else sc + P
                if u >= s0 + P:
                    continue
                best = None
                for bi, (cost, up, _) in enumerate(prev):
                    if up < u and (best is None or cost < best[0]):
                        best = (cost, bi)
                if best is not None:
                    cur.append((best[0] + dc, u, best[1]))
            if not cur:
                ok = False
                break
            back.append(cur)
            prev = cur
        if not ok:
            continue
        bi = int(np.argmin([c[0] for c in prev]))
        if prev[bi][0] < best_total:
            path = []
            for level in range(len(back) - 1, -1, -1):
                cost, u, pb = back[level][bi]
                path.append(u)
                bi = pb
            best_total, best_path = prev[int(np.argmin([c[0] for c in prev]))][0], path[::-1]
    if best_path is None:
        raise BoundaryCorrespondenceError(
            "ZHANG2021_ONESTRING_NO_MONOTONE_BOUNDARY_CORRESPONDENCE: the quad boundary cannot be "
            "mapped onto dOmega in order (orientation mismatch or self-overlapping approximation)")
    s = np.empty(m)
    for r, k in enumerate(order):
        s[k] = best_path[r] % P
    dist = np.asarray([np.linalg.norm(pts[k] - src.point_at(s[k])[2]) for k in range(m)])
    if not _cyclic_strictly_increasing(s, P):
        raise BoundaryCorrespondenceError("ZHANG2021_ONESTRING_MONOTONE_PROJECTION_INTERNAL")
    return s, dist


def initial_correspondence(src: SourceBoundary, quad_uv: np.ndarray, quads: np.ndarray,
                           *, snap_nodes: bool = True) -> InitialCorrespondence:
    q_loop = quad_boundary_loop(quads)
    Q = np.asarray(quad_uv, float)[q_loop]
    s, dist = monotone_projection(src, Q)
    snapped = 0
    unsnapped: list[dict] = []
    if snap_nodes and src.nodes:
        snapped, unsnapped = _snap_nodes(src, s)
        tips = [u for u in unsnapped if u["kind"] == "tip"]
        if tips:
            raise BoundaryCorrespondenceError(
                "ZHANG2021_ONESTRING_CUT_TIP_NOT_A_QUAD_CORNER: no quad boundary vertex can be "
                f"assigned to cut tip(s) {tips}; a panel edge would fold around the tip in 3D")
    return InitialCorrespondence(q_loop=q_loop, s=s, projection_distance=dist,
                                 snapped_nodes=snapped, unsnapped_nodes=unsnapped)


def finalize_correspondence(src: SourceBoundary, init: InitialCorrespondence, *,
                            align_pairs: bool = True, min_split_fraction: float = 0.1) -> Correspondence:
    P = src.perimeter
    s = np.asarray(init.s, float).copy()
    aligned, misaligned, report, pairs = 0, 0, [], []
    if align_pairs:
        aligned, misaligned, report, pairs = _align_cut_branches(src, s)
    if not _cyclic_strictly_increasing(s, P):
        raise BoundaryCorrespondenceError("ZHANG2021_ONESTRING_SEAM_ALIGNMENT_BROKE_ORDER")
    regularized = 0
    if min_split_fraction > 0:
        regularized = _regularize_split_points(src, s, float(min_split_fraction), pairs)
        if not _cyclic_strictly_increasing(s, P):
            raise BoundaryCorrespondenceError("ZHANG2021_ONESTRING_REGULARIZATION_BROKE_ORDER")
    return Correspondence(q_loop=list(init.q_loop), s=s, projection_distance=init.projection_distance,
                          snapped_nodes=init.snapped_nodes, aligned_pairs=aligned,
                          misaligned_branches=misaligned, branch_report=report,
                          regularized_points=regularized, unsnapped_nodes=list(init.unsnapped_nodes))


def compute_correspondence(src: SourceBoundary, quad_uv: np.ndarray, quads: np.ndarray,
                           *, snap_nodes: bool = True, align_pairs: bool = True,
                           min_split_fraction: float = 0.1) -> Correspondence:
    init = initial_correspondence(src, quad_uv, quads, snap_nodes=snap_nodes)
    return finalize_correspondence(src, init, align_pairs=align_pairs,
                                   min_split_fraction=min_split_fraction)


def branch_pairs(src: SourceBoundary) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Pairs of runs ((a, b), (a2, b2)): the two UV copies of one cut branch."""
    n = src.n
    runs = _branches(src)
    out, done = [], set()
    for a, b in runs:
        if (a, b) in done:
            continue
        ta = src.twin[a]
        if ta < 0:
            continue
        twin_end = (ta + 1) % n
        twin_run = next(((a2, b2) for a2, b2 in runs if b2 == twin_end), None)
        if twin_run is None:
            continue
        done.add((a, b))
        done.add(twin_run)
        out.append(((a, b), twin_run))
    return out


def run_interval(src: SourceBoundary, run: tuple[int, int]) -> tuple[float, float]:
    """Unwrapped arc-parameter interval [s0, s1] of a run (s1 > s0)."""
    P = src.perimeter
    a, b = run
    s0, s1 = float(src.cum[a]), (float(src.cum[b]) if b != 0 else P)
    if s1 <= s0:
        s1 += P
    return s0, s1


def indices_inside(s: np.ndarray, lo: float, hi: float, P: float) -> list[tuple[float, int]]:
    """(unwrapped parameter, index) of the entries strictly inside (lo, hi)."""
    tol = 1e-12 * max(P, 1.0)
    idx = []
    for k, val in enumerate(s):
        v = val if val >= lo - tol else val + P
        if lo + tol < v < hi - tol:
            idx.append((v, k))
    return sorted(idx)


def _snap_nodes(src: SourceBoundary, s: np.ndarray) -> tuple[int, list[dict]]:
    """Assign a quad boundary vertex to every seam node, preserving order.

    For a node lying between consecutive quad vertices k-1 and k, moving
    either of them onto the node keeps the cyclic order; the nearer unused
    one is taken.  Cut tips are processed first (they are mandatory: a quad
    edge spanning a tip would fold onto itself in 3D), then junctions, then
    cut/open transitions.  Only the correspondence changes, never Q.
    """
    P = src.perimeter
    m = len(s)
    order = {"tip": 0, "junction": 1, "transition": 2}
    nodes = sorted(src.nodes, key=lambda j: (order.get(src.node_kind.get(j, ""), 3), j))
    used: set[int] = set()
    snapped = 0
    unsnapped: list[dict] = []
    for j in nodes:
        ns = float(src.cum[j]) % P
        delta = (s - ns + P / 2) % P - P / 2
        exact = np.flatnonzero(np.abs(delta) <= 1e-12 * max(P, 1.0))
        if len(exact):
            s[int(exact[0])] = ns          # exactly on the node (no round-off residue)
            used.add(int(exact[0]))
            snapped += 1
            continue
        after = np.where(delta > 0, delta, np.inf)
        k_next = int(np.argmin(after))
        k_prev = (k_next - 1) % m
        cands = sorted(((abs(float(delta[k])), k) for k in (k_prev, k_next)))
        done = False
        for _, k in cands:
            if k in used:
                continue
            lo, hi = s[(k - 1) % m], s[(k + 1) % m]
            if 0 < (ns - lo) % P < (hi - lo) % P:
                s[k] = ns
                used.add(k)
                snapped += 1
                done = True
                break
        if not done:
            unsnapped.append({"loop_position": int(j), "kind": src.node_kind.get(j, "?"),
                              "xyz_id": int(src.xyz_id[j])})
    return snapped, unsnapped


def _regularize_split_points(src: SourceBoundary, s: np.ndarray, gamma: float,
                             pairs: list[tuple[int, int]] | None = None) -> int:
    """Avoid sliver triangles when the chosen points are realised as vertices.

    (1) a point within ``gamma`` (fraction of the source edge) of an existing
        source vertex is moved onto that vertex, unless a neighbouring Q
        vertex already uses it;
    (2) on a cut edge, a point of one UV side and a point of the twin side
        whose 3D fractions differ by less than ``gamma`` are both moved to
        their mean 3D fraction (they then split the 3D edge once).
    Only the correspondence changes; Q is never modified.
    """
    n, P, m = src.n, src.perimeter, len(s)
    partner = {}
    for ka, kb in pairs or []:
        partner[int(ka)] = int(kb)
        partner[int(kb)] = int(ka)

    def snap_candidate(k):
        j, t, _ = src.point_at(s[k])
        if 0.0 < t < gamma:
            cand = src.cum[j] % P
        elif 1.0 - gamma < t < 1.0:
            cand = src.cum[j + 1] % P
        else:
            return None
        prev_s, next_s = s[(k - 1) % m], s[(k + 1) % m]
        if (cand - prev_s) % P > 0 and (next_s - cand) % P > 0 and (cand - prev_s) % P < (next_s - prev_s) % P:
            return cand
        return None

    changed = 0
    done = set()
    for k in range(m):
        if k in done:
            continue
        cand = snap_candidate(k)
        if k in partner:
            kk = partner[k]
            done.update((k, kk))
            cand2 = snap_candidate(kk)
            if cand is not None and cand2 is not None:      # keep the hard pair aligned
                s[k], s[kk] = cand, cand2
                changed += 2
            continue
        if cand is not None:
            s[k] = cand
            changed += 1
    # (2) cross-side merge on cut edges (3D fraction measured from the min 3D id)
    by_edge: dict[tuple[int, int], list[tuple[float, int, int]]] = defaultdict(list)
    for k in range(m):
        if k in partner:
            continue
        j, t, _ = src.point_at(s[k])
        if not (0.0 < t < 1.0) or src.edge_kind[j] != "cut":
            continue
        a, b = int(src.xyz_id[j]), int(src.xyz_id[(j + 1) % n])
        f = t if a < b else 1.0 - t
        by_edge[(min(a, b), max(a, b))].append((f, k, j))
    for key, items in by_edge.items():
        sides = {}
        for f, k, j in items:
            sides.setdefault(j, []).append((f, k))
        if len(sides) != 2:
            continue
        (ja, la), (jb, lb) = sides.items()
        used_b = set()
        for fa, ka in sorted(la):
            best = None
            for fb, kb in sorted(lb):
                if kb in used_b or abs(fa - fb) >= gamma:
                    continue
                if best is None or abs(fa - fb) < abs(fa - best[0]):
                    best = (fb, kb)
            if best is None:
                continue
            fb, kb = best
            used_b.add(kb)
            fm = 0.5 * (fa + fb)
            for kk, jj in ((ka, ja), (kb, jb)):
                a, b = int(src.xyz_id[jj]), int(src.xyz_id[(jj + 1) % n])
                t_new = fm if a < b else 1.0 - fm
                s[kk] = (src.cum[jj] + t_new * (src.cum[jj + 1] - src.cum[jj])) % P
            changed += 2
    return changed


def _branches(src: SourceBoundary) -> list[tuple[int, int]]:
    """Maximal runs of cut edges between consecutive nodes: (start vertex, end vertex)."""
    n = src.n
    nodes = sorted(set(src.nodes))
    if not nodes:
        return []
    runs = []
    for i, a in enumerate(nodes):
        b = nodes[(i + 1) % len(nodes)]
        if src.edge_kind[a] == "cut":
            runs.append((a, b))
    return runs


def _align_cut_branches(src: SourceBoundary, s: np.ndarray) -> tuple[int, int, list[dict], list[tuple[int, int]]]:
    """Give both UV copies of each cut branch the same Q sample positions in 3D."""
    n = src.n
    P = src.perimeter
    runs = _branches(src)
    by_start = {a: b for a, b in runs}
    done = set()
    aligned_pairs = 0
    misaligned = 0
    report = []
    pairs: list[tuple[int, int]] = []
    for a, b in runs:
        if (a, b) in done:
            continue
        # twin side: edge a's twin ends at the twin run's END vertex
        ta = src.twin[a]
        if ta < 0:
            continue
        twin_end = (ta + 1) % n          # vertex position where twin run ends
        # find twin run (a2, b2) with b2 == twin_end
        twin_run = next(((a2, b2) for a2, b2 in runs if b2 == twin_end), None)
        if twin_run is None:
            continue
        done.add((a, b))
        done.add(twin_run)
        a2, b2 = twin_run
        sa0, sa1 = src.cum[a], src.cum[b] if b != 0 else P
        if sa1 <= sa0:
            sa1 += P
        sb0, sb1 = src.cum[a2], src.cum[b2] if b2 != 0 else P
        if sb1 <= sb0:
            sb1 += P

        ia = indices_inside(s, sa0, sa1, P)
        ib = indices_inside(s, sb0, sb1, P)
        entry = {"side_a": [int(a), int(b)], "side_b": [int(a2), int(b2)],
                 "quad_vertices_a": len(ia), "quad_vertices_b": len(ib)}
        if len(ia) != len(ib):
            misaligned += 1
            entry["status"] = "soft (different Q subdivision counts)"
            report.append(entry)
            continue
        # Map to a common 3D-length fraction along the branch.
        frac_a = [_fraction_3d(src, sa0, sa1, v) for v, _ in ia]
        frac_b = [1.0 - _fraction_3d(src, sb0, sb1, v) for v, _ in ib][::-1]
        for (va, ka), fa, fb, (vb, kb) in zip(ia, frac_a, frac_b, ib[::-1]):
            f = 0.5 * (fa + fb)
            s[ka] = _param_from_fraction_3d(src, sa0, sa1, f) % P
            s[kb] = _param_from_fraction_3d(src, sb0, sb1, 1.0 - f) % P
            pairs.append((int(ka), int(kb)))
            aligned_pairs += 1
        entry["status"] = "hard (aligned)"
        report.append(entry)
    return aligned_pairs, misaligned, report, pairs


def _arc3(src: SourceBoundary, s: float) -> float:
    """3D arc-length coordinate of loop parameter s (UV->3D is linear per edge)."""
    P = src.perimeter
    s_mod = float(s) % P
    j, t, _ = src.point_at(s_mod)
    base = src.cum3[j] + t * (src.cum3[j + 1] - src.cum3[j])
    # keep the unwrapped turn count so intervals crossing s=0 stay monotone
    turns = np.floor(float(s) / P)
    return float(base + turns * src.cum3[-1])


def _param_from_arc3(src: SourceBoundary, a3: float) -> float:
    total3 = src.cum3[-1]
    turns = np.floor(a3 / total3)
    r = a3 - turns * total3
    j = int(np.searchsorted(src.cum3, r, side="right") - 1)
    j = min(max(j, 0), src.n - 1)
    length3 = src.cum3[j + 1] - src.cum3[j]
    t = 0.0 if length3 <= 0 else (r - src.cum3[j]) / length3
    return float(src.cum[j] + t * (src.cum[j + 1] - src.cum[j]) + turns * src.perimeter)


def _fraction_3d(src: SourceBoundary, s0: float, s1: float, s: float) -> float:
    """Fraction of 3D cut length of parameter s inside branch side [s0, s1]."""
    a0, a1, a = _arc3(src, s0), _arc3(src, s1), _arc3(src, s)
    return float((a - a0) / max(a1 - a0, 1e-300))


def _param_from_fraction_3d(src: SourceBoundary, s0: float, s1: float, f: float) -> float:
    a0, a1 = _arc3(src, s0), _arc3(src, s1)
    return _param_from_arc3(src, a0 + f * (a1 - a0))
