"""2026-10-28 Omega mode: OptCuts -> piecewise smooth Omega -> structured quad grid (OneString).

    S -> official OptCuts chart Omega_0 (3D cut + paired UV, unchanged)
      -> Omega_1: every boundary run between features (seam nodes, sharp
         corners of the open boundary) replaced by a straight line or a
         smoothing spline (max deviation <= tol), corners kept, bijective
         SLIM inside, UV<->3D triangle correspondence kept exactly
      -> logical domain: an orthogonal polygon of the integer lattice
         (polyomino, like the old 50%-area grid footprint).  Every feature
         gets a turn of +90 / 0 / -90 deg (cut tips -90: a reflex corner
         carries 3 panels, so the tip is surrounded by 3 panels after the
         seam is closed); integer side lengths from a small ILP:
         closed polygon, equal counts on both copies of every cut branch,
         close to 3D length / target panel edge
      -> regular grid of the logical domain (interior valence 4, no
         T-junctions); boundary grid points placed by 3D arc length on the
         smooth boundary (both copies of a cut at the same 3D points),
         interior by a harmonic / Winslow-type solve
      -> Brackbill-Saltzman-type optimisation (orthogonality + cell size +
         smoothness, measured in the 3D metric of the chart) of the interior
         vertices, no flips
      -> M3D lift, then the unchanged 2026-09-18 K3D/T3D/K2D/T2D route
         (OptCuts seams as shared K3D variables).

Quality is reported, never a reason to stop.  Grid inversion, invalid
connectivity and seam mismatch are reported as unresolved.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import time

import numpy as np

MODE = "optcuts_structured_grid_20261028"
MODE_LABEL = "2026-10-28 | OptCuts → piecewise smooth Ω (line+spline) → structured grid (Brackbill–Saltzman) + 09-18"
FLAG = "optcuts_structured_grid_20261028_enabled"

DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])


class StructuredGridError(RuntimeError):
    pass


def mode_enabled(params: Any) -> bool:
    return bool(getattr(params, FLAG, False))


# ------------------------------------------------------------ logical domain
def _turning_angles(src, feats: list[int], scale: float) -> dict[int, float]:
    P = src.perimeter
    cum = np.asarray(src.cum[:-1], float)
    out = {}
    for j in feats:
        # signed turning of a CCW boundary: + convex corner, - reflex corner (cut tips are reflex)
        din = src.uv[j] - src.point_at((cum[j] - scale) % P)[2]
        dout = src.point_at((cum[j] + scale) % P)[2] - src.uv[j]
        out[j] = float(np.degrees(np.arctan2(din[0] * dout[1] - din[1] * dout[0], din @ dout)))
    return out


def logical_domain(src, feats: list[int], h3d: float, scale: float, *, min_ratio: float = 0.7,
                   max_rounds: int = 12, cone_pairs: list | None = None, w_cone: float = 0.5,
                   w_head: float = 0.5, shape_weight: float = 0.05, turn_weight: float = 0.3) -> dict:
    """Turns (+1 / 0 / -1 quarter turns), directions and integer side lengths of the orthogonal polygon.

    One mixed-integer program chooses everything together:
      * direction one-hot delta[k, d] of every side k (E/N/W/S), consecutive sides turn by -1/0/+1
        quarter turns, total turning +4 (closed CCW polygon), cut tips forced to -1 (reflex corner,
        3 panels around the tip once the cut is closed);
      * integer side lengths n_k >= 1, closure, equal counts on both copies of every cut branch;
      * cost: |n_k - L_k| / L_k (L_k = 3D run length / target panel edge)
              + w_turn |turning angle / 90 deg - turn_k|
              + w_pos |corner_k - chart position of feature k| (shape term, used when needed).
    Self-intersecting answers are excluded by lazy non-overlap constraints (stage 2).

    Seam cones (``cone_pairs``: the two UV copies of one 3D point of a cut): the turns t_A, t_B of
    the two copies may add up to -1 / 0 / +1, i.e. the point gets 3D valence 5 / 4 / 3 once the
    seam is closed (an exceptional valence ON the seam; the M2D interior stays valence 4).  Each
    cone costs ``w_cone``; the logical heading is kept close to the chart's heading along the whole
    boundary (``w_head``), so the curvature of caps such as the ears is absorbed by seam cones
    instead of by bending the interior grid.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from shapely.geometry import Polygon
    from .zhang2021_onestring_seam_matching import seam_constraints

    feats = sorted(feats)
    m = len(feats)
    if m < 4 and not any(src.node_kind.get(j) for j in feats):
        raise StructuredGridError(f"SG_TOO_FEW_BOUNDARY_FEATURES: {m} (need >= 4 corners)")
    cone_pairs = list(cone_pairs or [])
    if cone_pairs:
        ang = _chord_turning(src, feats)
    else:
        ang = _turning_angles(src, feats, scale)
    tips = {k for k, j in enumerate(feats) if src.node_kind.get(j) == "tip"}
    L = []
    run_uv = []
    for k in range(m):
        a, b = feats[k], feats[(k + 1) % m]
        lo = float(src.cum3[a])
        hi = float(src.cum3[b]) if b > a else float(src.cum3[b]) + float(src.cum3[-1])
        L.append(max(hi - lo, 1e-12) / h3d)
        lo = float(src.cum[a])
        hi = float(src.cum[b]) if b > a else float(src.cum[b]) + float(src.perimeter)
        run_uv.append(hi - lo)
    L = np.asarray(L)
    # Shape target: feature positions of the chart rotated so that the chords align best with the axes.
    uvf = np.asarray([src.uv[j] for j in feats], float)
    chord = np.roll(uvf, -1, axis=0) - uvf
    clen = np.linalg.norm(chord, axis=1)
    phi = np.arctan2(chord[:, 1], chord[:, 0])
    theta = float(np.angle(np.sum(clen * np.exp(4j * phi))) / 4.0) if np.sum(clen) > 0 else 0.0
    R = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    unit = float(np.sum(L) / max(float(np.sum(run_uv)), 1e-300))
    T = ((uvf - uvf[0]) @ R.T) * unit
    pos = {j: k for k, j in enumerate(feats)}
    pairs = _subside_pairs(src, feats, seam_constraints(src))
    cones = [(pos[a], pos[b]) for a, b in cone_pairs]
    P_c = len(cones)
    # variable layout
    iD = lambda k, d: 4 * k + d                     # noqa: E731  delta (binary)
    iX = lambda k, d: 4 * m + 4 * k + d             # noqa: E731  x = n_k * delta (integer)
    iT = lambda k, s: 8 * m + 3 * k + (s + 1)       # noqa: E731  turn one-hot (binary), s in -1, 0, 1
    iE = lambda k: 11 * m + k                       # noqa: E731  |n_k - L_k|
    iF = lambda k, ax: 12 * m + 2 * k + ax          # noqa: E731  |P_k - T_k|
    iZ = lambda q: 14 * m + q                       # noqa: E731  |t_A + t_B| of seam cone candidate q
    iH = lambda k: 14 * m + P_c + k                 # noqa: E731  heading error after side k
    iO = 15 * m + P_c                               # free heading offset
    nv = 15 * m + P_c + 1
    M = int(10 * max(L.max(), 4) + 10)
    w_turn = float(turn_weight) if cone_pairs else 1.0
    # A side may not be squeezed: n_k >= max(1, floor(min_ratio * L_k)) (protrusions keep their cells).
    n_lo = np.maximum(1, np.floor(min_ratio * L)).astype(int)
    A, lb, ub = [], [], []

    def row(entries, lo, hi):
        r = np.zeros(nv)
        for i, v in entries:
            r[i] += v
        A.append(r); lb.append(lo); ub.append(hi)

    for k in range(m):
        row([(iD(k, d), 1) for d in range(4)], 1, 1)
        for d in range(4):
            row([(iX(k, d), 1), (iD(k, d), -M)], -np.inf, 0)
        row([(iX(k, d), 1) for d in range(4)], float(n_lo[k]), np.inf)
        km = (k - 1) % m
        row([(iT(k, s), 1) for s in (-1, 0, 1)], 1, 1)
        for d in range(4):
            row([(iD(km, d), 1), (iD(k, (d + 2) % 4), 1)], -np.inf, 1)
            for s_ in (-1, 0, 1):
                row([(iT(k, s_), 1), (iD(km, d), -1), (iD(k, (d + s_) % 4), -1)], -1, np.inf)
        if k in tips:
            row([(iT(k, -1), 1)], 1, 1)
        nk = [(iX(k, d), 1) for d in range(4)]
        row(nk + [(iE(k), -1)], -np.inf, L[k])
        row([(i, -v) for i, v in nk] + [(iE(k), -1)], -np.inf, -L[k])
        for ax in (0, 1):
            pk = [(iX(i, d), float(DIRS[d][ax])) for i in range(k) for d in range(4)]
            row(pk + [(iF(k, ax), -1)], -np.inf, T[k, ax])
            row([(i, -v) for i, v in pk] + [(iF(k, ax), -1)], -np.inf, -T[k, ax])
    row([(iT(k, s_), s_) for k in range(m) for s_ in (-1, 1)], 4, 4)
    for ax in (0, 1):
        row([(iX(k, d), float(DIRS[d][ax])) for k in range(m) for d in range(4)], 0, 0)
    for ka, kb in pairs:
        row([(iX(ka, d), 1) for d in range(4)] + [(iX(kb, d), -1) for d in range(4)], 0, 0)
    for q, (ka, kb) in enumerate(cones):
        tsum = [(iT(ka, s_), s_) for s_ in (-1, 1)] + [(iT(kb, s_), s_) for s_ in (-1, 1)]
        row(tsum, -1, 1)                                      # 3D valence 3..5 on the seam
        row(tsum + [(iZ(q), -1)], -np.inf, 0)
        row([(i, -v) for i, v in tsum] + [(iZ(q), -1)], -np.inf, 0)
    if cone_pairs:
        C = np.cumsum([ang[j] / 90.0 for j in feats])
        for k in range(m):
            tk = [(iT(i, s_), s_) for i in range(k + 1) for s_ in (-1, 1)]
            # h_k >= |C_k - sum_{i<=k} t_i - o|
            row([(i, -v) for i, v in tk] + [(iO, -1), (iH(k), -1)], -np.inf, -C[k])
            row(tk + [(iO, 1), (iH(k), -1)], -np.inf, C[k])
    integrality = np.r_[np.ones(11 * m), np.zeros(4 * m + P_c + 1)]
    lo_b = np.zeros(nv)
    lo_b[iO] = -np.inf
    hi_b = np.r_[np.ones(4 * m), np.full(4 * m, M), np.ones(3 * m), np.full(4 * m + P_c + 1, np.inf)]
    attempts = []
    nogood: list[list[int]] = []
    chosen = None
    w_pos = float(shape_weight)
    for _round in range(max_rounds):
        # stage 1: turns / directions (lengths soft, with the hard lower bounds)
        c = np.zeros(nv)
        for k in range(m):
            c[iE(k)] = 1.0 / max(L[k], 1.0)
            for s_ in (-1, 0, 1):
                c[iT(k, s_)] = w_turn * abs(ang[feats[k]] / 90.0 - s_)
            c[iF(k, 0)] = c[iF(k, 1)] = w_pos / max(float(np.mean(L)), 1.0)
            if cone_pairs:
                c[iH(k)] = w_head
        for q in range(P_c):
            c[iZ(q)] = w_cone
        A2, lb2, ub2 = list(A), list(lb), list(ub)
        for ng in nogood:
            r = np.zeros(nv); r[ng] = 1
            A2.append(r); lb2.append(-np.inf); ub2.append(m - 1)
        res = milp(c, constraints=LinearConstraint(np.asarray(A2), lb2, ub2), integrality=integrality,
                   bounds=Bounds(lo_b, hi_b), options={"time_limit": 90.0 if cone_pairs else 8.0, "mip_rel_gap": 0.02})
        if res.x is None:
            attempts.append({"round": _round, "status": f"no turn assignment: {res.message}"})
            break
        x = res.x
        dirs = [int(np.argmax([x[iD(k, d)] for d in range(4)])) for k in range(m)]
        turn = [int(round(sum(s_ * x[iT(k, s_)] for s_ in (-1, 1)))) for k in range(m)]
        # stage 2: integer side lengths for these directions, simple polygon enforced by lazy
        # non-overlap constraints between non-adjacent sides (never by shortening sides)
        n, info = _lengths_for_directions(dirs, L, n_lo, pairs, T, w_pos)
        attempts.append({"round": _round, "turns": turn, **info,
                         "lengths": None if n is None else n.tolist()})
        if n is not None:
            pts = [np.zeros(2, int)]
            for k in range(m):
                pts.append(pts[-1] + n[k] * DIRS[dirs[k]])
            poly = np.asarray(pts[:-1])
            pg = Polygon(poly)
            if np.all(pts[-1] == 0) and pg.is_valid and pg.area > 0:
                chosen = (w_pos, dirs, n, turn, poly)
                break
        nogood.append([iT(k, turn[k]) for k in range(m)])     # rotation-invariant cut on the turns
    if chosen is None:
        raise StructuredGridError(
            "SG_LOGICAL_POLYGON_NOT_FOUND: no simple orthogonal polygon with every side >= "
            f"{min_ratio:g} x its 3D length was found (attempts {attempts})")
    w_pos, dirs, n, turn, poly = chosen
    cone_report = []
    for (a, b), (ka, kb) in zip(cone_pairs, cones):
        ssum = turn[ka] + turn[kb]
        if ssum:
            cone_report.append({"xyz_id": int(src.xyz_id[a]), "uv_copies": [int(a), int(b)],
                                "turns": [int(turn[ka]), int(turn[kb])], "valence_3d": int(4 - ssum)})
    return {"seam_cones": cone_report, "cone_candidates": len(cone_pairs),
            "features": feats, "turn": turn, "dirs": dirs, "lengths": n.tolist(),
            "target_lengths": L.tolist(), "polygon": poly, "cut_pairs": pairs,
            "convex": int(sum(1 for t in turn if t == 1)), "reflex": int(sum(1 for t in turn if t == -1)),
            "shape_weight": w_pos, "attempts": attempts, "turning_deg": [float(ang[j]) for j in feats]}


def _chord_turning(src, feats: list[int]) -> dict[int, float]:
    """Turning (deg, + convex / - reflex) of the polygon through the feature points of a CCW loop."""
    uvf = np.asarray([src.uv[j] for j in feats], float)
    din = uvf - np.roll(uvf, 1, axis=0)
    dout = np.roll(uvf, -1, axis=0) - uvf
    a = np.degrees(np.arctan2(din[:, 0] * dout[:, 1] - din[:, 1] * dout[:, 0], np.sum(din * dout, axis=1)))
    return {j: float(a[k]) for k, j in enumerate(feats)}


def _cyclic_positions(n: int, a: int, b: int) -> list[int]:
    """Loop positions a, a+1, ..., b (cyclic, inclusive)."""
    cnt = (b - a) % n
    return [(a + i) % n for i in range(cnt + 1)]


def _subside_pairs(src, feats: list[int], cons) -> list[tuple[int, int]]:
    """Pairs (side k_A, side k_B) of the two UV copies of every cut branch, split at the features."""
    n = src.n
    fset = set(feats)
    pos = {j: k for k, j in enumerate(feats)}
    out = []
    for con in cons:
        FA = [j for j in _cyclic_positions(n, *con.run_a) if j in fset]
        FB = [j for j in _cyclic_positions(n, *con.run_b) if j in fset]
        if len(FA) != len(FB) or len(FA) < 2:
            raise StructuredGridError("SG_CUT_BRANCH_FEATURES_UNPAIRED: the two copies of a cut branch carry "
                                      f"different feature counts ({len(FA)} vs {len(FB)})")
        r = len(FA) - 1
        for i in range(r):
            a0, a1 = FA[i], FA[i + 1]
            b0, b1 = FB[r - 1 - i], FB[r - i]
            if src.xyz_id[a0] != src.xyz_id[b1] or src.xyz_id[a1] != src.xyz_id[b0]:
                raise StructuredGridError("SG_CUT_BRANCH_FEATURES_NOT_AT_THE_SAME_3D_POINTS")
            out.append((pos[a0], pos[b0]))
    return out


def seam_cone_candidates(src, base_feats: list[int], h3d: float, spacing: float = 2.0,
                         open_spacing: float = 2.0) -> tuple[list[int], list[tuple[int, int]]]:
    """Extra feature points: paired copies of 3D cut vertices every ~spacing panels along every cut
    branch (seam cone candidates) and points every ~open_spacing panels on open runs."""
    from .zhang2021_onestring_seam_matching import seam_constraints

    n = src.n
    tot3 = float(src.cum3[-1])
    d3 = lambda a, b: (float(src.cum3[b]) - float(src.cum3[a])) % tot3     # noqa: E731
    feats = set(int(j) for j in base_feats)
    pairs = []
    for con in seam_constraints(src):
        a0, a1 = con.run_a
        posB = {int(src.xyz_id[j]): j for j in _cyclic_positions(n, *con.run_b)[1:-1]}
        length = d3(a0, a1)
        last = a0
        for j in _cyclic_positions(n, a0, a1)[1:-1]:
            if d3(last, j) >= spacing * h3d and d3(j, a1) >= 0.6 * spacing * h3d:
                q = posB.get(int(src.xyz_id[j]))
                if q is not None:
                    pairs.append((int(j), int(q)))
                    feats.update((int(j), int(q)))
                    last = j
        del length
    base = sorted(feats)
    for k in range(len(base)):
        a, b = base[k], base[(k + 1) % len(base)]
        run = _cyclic_positions(n, a, b)
        if not all(src.edge_kind[j] == "open" for j in run[:-1]):
            continue
        last = a
        for j in run[1:-1]:
            if d3(last, j) >= open_spacing * h3d and d3(j, b) >= 0.6 * open_spacing * h3d:
                feats.add(int(j))
                last = j
    return sorted(feats), pairs


def _locate_bary(Vsrc: np.ndarray, F: np.ndarray, pts: np.ndarray, Vdst: np.ndarray):
    """Map points through the piecewise-linear map Vsrc -> Vdst on triangles F (None if any is outside)."""
    from matplotlib.tri import Triangulation
    finder = Triangulation(Vsrc[:, 0], Vsrc[:, 1], F).get_trifinder()
    t = finder(pts[:, 0], pts[:, 1])
    if np.any(t < 0):
        return None
    a, b, c = F[t, 0], F[t, 1], F[t, 2]
    e1, e2, d = Vsrc[b] - Vsrc[a], Vsrc[c] - Vsrc[a], pts - Vsrc[a]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    l1 = (d[:, 0] * e2[:, 1] - d[:, 1] * e2[:, 0]) / det
    l2 = (e1[:, 0] * d[:, 1] - e1[:, 1] * d[:, 0]) / det
    return Vdst[a] + l1[:, None] * (Vdst[b] - Vdst[a]) + l2[:, None] * (Vdst[c] - Vdst[a])


def _untangle_triangles(P: np.ndarray, F: np.ndarray, free: np.ndarray, iters: int = 400) -> np.ndarray:
    """Make every triangle positively oriented: min sum max(0, eps - det)^2 over the free vertices."""
    from scipy.optimize import minimize
    P = np.asarray(P, float).copy()
    e1 = P[F[:, 1]] - P[F[:, 0]]
    e2 = P[F[:, 2]] - P[F[:, 0]]
    det0 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    eps = 1e-3 * float(np.median(np.abs(det0)))

    def fg(z):
        Q = P.copy()
        Q[free] = z.reshape(-1, 2)
        a, b, c = Q[F[:, 0]], Q[F[:, 1]], Q[F[:, 2]]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        v = np.maximum(eps - det, 0.0)
        E = float(np.sum(v * v))
        g = -2.0 * v
        G = np.zeros_like(Q)
        # d det / d a, b, c
        np.add.at(G, F[:, 0], g[:, None] * np.c_[b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]])
        np.add.at(G, F[:, 1], g[:, None] * np.c_[c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]])
        np.add.at(G, F[:, 2], g[:, None] * np.c_[a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]])
        return E, G[free].ravel()

    res = minimize(fg, P[free].ravel(), jac=True, method="L-BFGS-B", options={"maxiter": iters})
    P[free] = res.x.reshape(-1, 2)
    return P


def _slim_polygon_map(Dc, Dg, quads, logical, bset_c, bpoly, cuv, cf, cxyz, cf3, bnd, harmonic_flips):
    """Bijective chart -> logical polygon map with the least 3D distortion (symmetric Dirichlet, SLIM).

    Start: the composite Tutte map (bijective), then SLIM w.r.t. the 3D triangles with the polygon
    outline fixed.  The lattice is pulled back through the optimised map, so the unavoidable
    distortion (chart vs. polygon) is spread over the chart instead of concentrated in fans."""
    from .zhang2021_onestring_reparam import _SDProblem, _slim_iterations, progressive_boundary_slim

    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    # the lattice disk image is the polygon inscribed in the circle through its boundary vertices:
    # pull the chart's disk image radially onto it (a bijection of the disk onto that polygon)
    bang = np.sort(np.mod(np.arctan2(Dg[bnd, 1], Dg[bnd, 0]), 2 * np.pi))
    th = np.mod(np.arctan2(Dc[:, 1], Dc[:, 0]), 2 * np.pi)
    i1 = np.searchsorted(bang, th) % len(bang)
    i0 = (i1 - 1) % len(bang)
    a0, a1 = bang[i0], bang[i1]
    span = np.mod(a1 - a0, 2 * np.pi)
    mid = a0 + 0.5 * span
    rho = np.cos(0.5 * span) / np.maximum(np.cos(th - mid), 1e-12)
    Dc = Dc * (rho * (1.0 - 1e-9))[:, None]
    P0 = _locate_bary(Dg, tris, Dc, logical)
    if P0 is None:
        print("[SG-PHI] composite map: a chart vertex fell outside the lattice disk image", flush=True)
        return None
    e1 = P0[cf[:, 1]] - P0[cf[:, 0]]
    e2 = P0[cf[:, 2]] - P0[cf[:, 0]]
    nflip = int(np.sum(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0))
    if nflip:
        P0 = _untangle_triangles(P0, cf, np.setdiff1d(np.arange(len(P0)), bset_c))
        e1 = P0[cf[:, 1]] - P0[cf[:, 0]]
        e2 = P0[cf[:, 2]] - P0[cf[:, 0]]
        nflip = int(np.sum(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0))
    if nflip:
        print(f"[SG-PHI] composite map has {nflip} flipped/degenerate chart triangles; SLIM skipped", flush=True)
        return None
    X = cxyz[cf3]
    area3d = 0.5 * float(np.sum(np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)))
    Ppoly = logical[tris]
    parea = 0.5 * float(np.sum(np.abs((Ppoly[:, 1, 0] - Ppoly[:, 0, 0]) * (Ppoly[:, 2, 1] - Ppoly[:, 0, 1])
                                      - (Ppoly[:, 1, 1] - Ppoly[:, 0, 1]) * (Ppoly[:, 2, 0] - Ppoly[:, 0, 0]))))
    scale = float(np.sqrt(area3d / max(parea, 1e-300)))
    prob = _SDProblem(cxyz, cf3, cf, len(cuv))
    free = np.setdiff1d(np.arange(len(cuv)), bset_c)
    try:
        E0 = float(prob.energy(P0 * scale)[0])
        # composite boundary points already lie on the polygon outline; slide them to the exact
        # 3D arc-length positions without flips, then optimise the interior
        cur, _E, _st = progressive_boundary_slim(prob, P0 * scale, cf, bset_c, bpoly * scale, max_steps=400,
                                                 step_slim_iters=2, final_iters=0, error_prefix="SG_PHI")
        cur, E1, iters = _slim_iterations(prob, cur, free, iters=200, tol=1e-6)
    except Exception as exc:  # noqa: BLE001 - fall back to the composite map
        print(f"[SG-PHI] SLIM failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    gfree = np.setdiff1d(np.arange(len(logical)), bnd)
    out = _locate_bary(cur, cf, logical[gfree] * scale, cuv)
    if out is None:
        return None
    return gfree, out, {"phi_model": "bijective SLIM map chart -> logical polygon (3D symmetric Dirichlet), "
                                     "started from the composite Tutte map",
                        "phi_sd_before": E0, "phi_sd_after": float(E1), "phi_slim_iterations": int(iters),
                        "harmonic_direct_flipped_triangles": harmonic_flips}


def _lengths_for_directions(dirs, L, n_lo, pairs, T, w_pos, *, max_iter: int = 40):
    """Integer side lengths for fixed directions: closure, equal cut copies, n >= n_lo, simple polygon.

    Non-adjacent sides i, j must be >= 1 lattice unit apart (L-inf) on the integer lattice; the
    disjunction (left / right / below / above) is added lazily for pairs that touch.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    m = len(dirs)
    D = np.asarray([DIRS[d] for d in dirs], float)
    # vars: n (m, int) | e (m) | f (2m) | 4 binaries per separated pair
    sep_pairs: list[tuple[int, int]] = []
    hi_n = np.maximum(n_lo, np.ceil(1.6 * L) + 3)

    def pos_rows(k):          # P_k (start of side k) = sum_{i<k} n_i D_i  -> coefficient rows (2, m)
        r = np.zeros((2, m))
        r[:, :k] = D[:k].T
        return r

    def box(k):               # (xlo, xhi, ylo, yhi) of side k as coefficient rows over n
        p0 = pos_rows(k)
        p1 = p0.copy(); p1[:, k] += D[k]
        out = []
        for ax in (0, 1):
            if D[k][ax] >= 0:
                out += [p0[ax], p1[ax]]
            else:
                out += [p1[ax], p0[ax]]
        return out            # xlo, xhi, ylo, yhi

    Mbig = float(4 * np.sum(hi_n) + 10)
    n_sol = None
    for it in range(max_iter):
        nb = 4 * len(sep_pairs)
        nv = 4 * m + nb
        A, lb, ub = [], [], []

        def add(coefs_n, extra, lo, hi):
            r = np.zeros(nv)
            r[:m] = coefs_n
            for i, v in extra:
                r[i] += v
            A.append(r); lb.append(lo); ub.append(hi)

        for ax in (0, 1):
            add(D[:, ax], [], 0.0, 0.0)
        for ka, kb in pairs:
            cn = np.zeros(m); cn[ka] = 1; cn[kb] = -1
            add(cn, [], 0.0, 0.0)
        for k in range(m):
            cn = np.zeros(m); cn[k] = 1
            add(cn, [(m + k, -1)], -np.inf, L[k])
            add(-cn, [(m + k, -1)], -np.inf, -L[k])
            pr = pos_rows(k)
            for ax in (0, 1):
                fi = 2 * m + 2 * k + ax
                add(pr[ax], [(fi, -1)], -np.inf, T[k, ax])
                add(-pr[ax], [(fi, -1)], -np.inf, -T[k, ax])
        for q, (i, j) in enumerate(sep_pairs):
            bi = 4 * m + 4 * q
            xi_lo, xi_hi, yi_lo, yi_hi = box(i)
            xj_lo, xj_hi, yj_lo, yj_hi = box(j)
            # b0: xi_hi + 1 <= xj_lo ; b1: xj_hi + 1 <= xi_lo ; b2/b3 same in y
            for t, (a_hi, b_lo) in enumerate(((xi_hi, xj_lo), (xj_hi, xi_lo), (yi_hi, yj_lo), (yj_hi, yi_lo))):
                add(a_hi - b_lo, [(bi + t, Mbig)], -np.inf, Mbig - 1.0)
            r = np.zeros(nv); r[bi:bi + 4] = 1
            A.append(r); lb.append(1.0); ub.append(np.inf)
        c = np.zeros(nv)
        c[m:2 * m] = 1.0 / np.maximum(L, 1.0)
        c[2 * m:4 * m] = max(w_pos, 1e-3) / max(float(np.mean(L)), 1.0)
        integ = np.r_[np.ones(m), np.zeros(3 * m), np.ones(nb)]
        lo = np.r_[n_lo, np.zeros(3 * m + nb)]
        hi = np.r_[hi_n, np.full(3 * m, np.inf), np.ones(nb)]
        res = milp(c, constraints=LinearConstraint(np.asarray(A), lb, ub), integrality=integ,
                   bounds=Bounds(lo, hi), options={"time_limit": 30.0})
        if res.x is None:
            return None, {"status": f"infeasible for these turns ({res.message})", "separated_pairs": len(sep_pairs),
                          "lazy_iterations": it + 1}
        n_sol = np.rint(res.x[:m]).astype(int)
        touching = _touching_pairs(dirs, n_sol)
        new = [pq for pq in touching if pq not in sep_pairs]
        if not new:
            return n_sol, {"status": "simple", "separated_pairs": len(sep_pairs), "lazy_iterations": it + 1}
        sep_pairs.extend(new)
    return None, {"status": "lazy separation did not converge", "separated_pairs": len(sep_pairs)}


def _touching_pairs(dirs, n) -> list[tuple[int, int]]:
    """Non-adjacent sides closer than 1 lattice unit (L-inf), i.e. touching or crossing."""
    m = len(dirs)
    P = [np.zeros(2, int)]
    for k in range(m):
        P.append(P[-1] + n[k] * DIRS[dirs[k]])
    boxes = []
    for k in range(m):
        a, b = P[k], P[k + 1]
        boxes.append((min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])))
    out = []
    for i in range(m):
        for j in range(i + 2, m):
            if i == 0 and j == m - 1:
                continue
            bi, bj = boxes[i], boxes[j]
            gx = max(bj[0] - bi[1], bi[0] - bj[1])
            gy = max(bj[2] - bi[3], bi[2] - bj[3])
            if max(gx, gy) < 1:
                out.append((i, j))
    return out


def lattice_grid(dom: dict) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int], list[tuple[int, int, int]]]:
    """Cells of the orthogonal polygon -> (logical vertices, quads CCW, index map, boundary walk)."""
    from matplotlib.path import Path as _Path
    poly = np.asarray(dom["polygon"])
    path = _Path(poly)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    xs, ys = np.meshgrid(np.arange(xmin, xmax) + 0.5, np.arange(ymin, ymax) + 0.5)
    centers = np.c_[xs.ravel(), ys.ravel()]
    inside = path.contains_points(centers)
    vid: dict[tuple[int, int], int] = {}

    def v(x, y):
        key = (int(x), int(y))
        if key not in vid:
            vid[key] = len(vid)
        return vid[key]

    quads = []
    for (cx, cy) in centers[inside]:
        x0, y0 = int(np.floor(cx)), int(np.floor(cy))
        quads.append([v(x0, y0), v(x0 + 1, y0), v(x0 + 1, y0 + 1), v(x0, y0 + 1)])
    logical = np.zeros((len(vid), 2))
    for (x, y), i in vid.items():
        logical[i] = (x, y)
    walk = []                       # (vertex, segment k, offset j)
    p = poly[0].copy()
    for k, (dk, nk) in enumerate(zip(dom["dirs"], dom["lengths"])):
        for j in range(nk):
            key = (int(p[0]), int(p[1]))
            if key not in vid:
                raise StructuredGridError("SG_LOGICAL_BOUNDARY_NOT_ON_GRID")
            walk.append((vid[key], k, j))
            p = p + DIRS[dk]
    return logical, np.asarray(quads, int), vid, walk


def grid_line_triples(vid: dict[tuple[int, int], int]) -> np.ndarray:
    out = []
    for (x, y), i in vid.items():
        for dx, dy in ((1, 0), (0, 1)):
            a, b = vid.get((x - dx, y - dy)), vid.get((x + dx, y + dy))
            if a is not None and b is not None:
                out.append((a, i, b))
    return np.asarray(out, int).reshape(-1, 3)


def place_grid(src, dom: dict, logical: np.ndarray, quads: np.ndarray, walk,
               chart_uv: np.ndarray | None = None, chart_f2: np.ndarray | None = None,
               chart_xyz: np.ndarray | None = None, chart_f3: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Grid vertices in Omega_1: boundary by 3D arc length; interior pulled back from the logical domain.

    phi: Omega_1 -> logical polygon (boundary by the same 3D arc-length correspondence, interior
    harmonic on the chart triangles).  When phi is injective every lattice point is located in
    phi(Omega_1) and mapped back by barycentric interpolation, so every grid vertex lies inside
    Omega_1 and the grid follows protrusions.  Otherwise: harmonic grid graph (reported).
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from .zhang2021_onestring_boundary import _fraction_3d, _param_from_fraction_3d, run_interval

    feats = dom["features"]
    m = len(feats)
    P = src.perimeter
    poly = np.asarray(dom["polygon"], float)
    uv = np.full((len(logical), 2), np.nan)
    for vtx, k, j in walk:
        a, b = feats[k], feats[(k + 1) % m]
        lo, hi = run_interval(src, (a, b))
        s_ = _param_from_fraction_3d(src, lo, hi, j / dom["lengths"][k]) % P
        uv[vtx] = src.point_at(s_)[2]
    bnd = np.array(sorted({w[0] for w in walk}))
    info = {"placement": "grid-graph harmonic"}
    if chart_uv is not None and chart_f2 is not None:
        try:
            got = _pullback_interior(src, dom, logical, quads, bnd, walk, np.asarray(chart_uv, float),
                                     np.asarray(chart_f2, int), chart_xyz, chart_f3)
        except Exception as exc:  # noqa: BLE001 - reported, falls back below
            got, info = None, {"placement": "grid-graph harmonic", "pullback_failed": f"{type(exc).__name__}: {exc}"}
        if got is not None:
            free, vals, pinfo = got
            uv[free] = vals
            info = {"placement": "pull-back of the lattice through phi: Omega_1 -> logical polygon", **pinfo}
            return uv, info
        info.setdefault("pullback_failed", "phi not injective")
    n = len(logical)
    rows, cols = [], []
    for q in quads:
        for k in range(4):
            rows += [q[k], q[(k + 1) % 4]]
            cols += [q[(k + 1) % 4], q[k]]
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    Lap = (sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A).tocsr()
    free = np.setdiff1d(np.arange(n), bnd)
    if len(free):
        Lff = Lap[free][:, free].tocsc()
        Lfb = Lap[free][:, bnd]
        for c in range(2):
            uv[free, c] = spsolve(Lff, -Lfb @ uv[bnd, c])
    return uv, info


def _tutte_disk(n: int, edges_w: tuple, bset: np.ndarray, bpos: np.ndarray) -> np.ndarray:
    """Convex-combination (Tutte/Floater) embedding with the boundary on the unit circle."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    I, J, W = edges_w
    A = sparse.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    Lap = (sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A).tocsr()
    X = np.zeros((n, 2))
    X[bset] = bpos
    free = np.setdiff1d(np.arange(n), bset)
    if len(free):
        Lff = Lap[free][:, free].tocsc()
        Lfb = Lap[free][:, bset]
        for c in range(2):
            X[free, c] = spsolve(Lff, -Lfb @ bpos[:, c])
    return X


def _pullback_interior(src, dom, logical, quads, bnd, walk, cuv, cf, cxyz=None, cf3=None):
    """Interior grid points as phi^-1(lattice) with phi = (Tutte disk map of the lattice polygon)^-1 o
    (Tutte disk map of Omega_1).  Both disk maps are bijective (positive weights, convex circle
    boundary, same boundary correspondence: 3D arc length <-> polygon arc length), so the composite
    is a bijection Omega_1 -> logical polygon and every grid vertex lies inside Omega_1."""
    from matplotlib.tri import Triangulation
    from .zhang2021_onestring_boundary import _fraction_3d, run_interval

    feats = dom["features"]
    m = len(feats)
    P = src.perimeter
    lengths = np.asarray(dom["lengths"], float)
    cum_n = np.r_[0.0, np.cumsum(lengths)]
    total = float(cum_n[-1])
    loop = np.asarray(src.loop, int)
    cum = np.asarray(src.cum[:-1], float)
    t_chart = np.full(len(cuv), np.nan)
    for k in range(m):
        a, b = feats[k], feats[(k + 1) % m]
        lo, hi = run_interval(src, (a, b))
        cnt = (b - a) % len(loop) or len(loop)
        for jpos in (np.arange(a, a + cnt) % len(loop)):
            sv = cum[jpos]
            if sv < lo - 1e-12:
                sv += P
            t_chart[loop[jpos]] = cum_n[k] + _fraction_3d(src, lo, hi, sv) * lengths[k]
    bset_c = np.flatnonzero(~np.isnan(t_chart))
    circ = lambda t: np.c_[np.cos(2 * np.pi * t / total), np.sin(2 * np.pi * t / total)]   # noqa: E731
    # chart weights: clamped cotangent (positive -> Floater embedding)
    I, J, W = [], [], []
    for tr in cf:
        for k in range(3):
            i, j, o = tr[k], tr[(k + 1) % 3], tr[(k + 2) % 3]
            u, v = cuv[i] - cuv[o], cuv[j] - cuv[o]
            cr = abs(u[0] * v[1] - u[1] * v[0])
            w = max(0.5 * float(u @ v) / max(cr, 1e-15), 1e-6)
            I += [i, j]; J += [j, i]; W += [w, w]
    # 1st choice: direct harmonic map Omega_1 -> logical polygon (low distortion when the polygon
    # follows the chart's shape, e.g. with seam cones); used only if it has no flipped triangle.
    poly = np.asarray(dom["polygon"], float)
    dirs_v = np.asarray([DIRS[d] for d in dom["dirs"]], float)
    kk = np.clip(np.searchsorted(cum_n, t_chart[bset_c], side="right") - 1, 0, m - 1)
    bpoly = poly[kk] + (t_chart[bset_c] - cum_n[kk])[:, None] * dirs_v[kk]
    Dp = _tutte_disk(len(cuv), (I, J, W), bset_c, bpoly)
    e1 = Dp[cf[:, 1]] - Dp[cf[:, 0]]
    e2 = Dp[cf[:, 2]] - Dp[cf[:, 0]]
    o0 = cuv[cf[:, 1]] - cuv[cf[:, 0]]
    o1 = cuv[cf[:, 2]] - cuv[cf[:, 0]]
    flips = int(np.sum((e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) * np.sign(o0[:, 0] * o1[:, 1] - o0[:, 1] * o1[:, 0]) <= 0))
    if flips == 0:
        tri = Triangulation(Dp[:, 0], Dp[:, 1], cf)
        finder = tri.get_trifinder()
        gfree = np.setdiff1d(np.arange(len(logical)), bnd)
        pts = np.asarray(logical, float)[gfree]
        t = finder(pts[:, 0], pts[:, 1])
        if np.all(t >= 0):
            out = np.zeros((len(gfree), 2))
            for r, (ti, p) in enumerate(zip(t, pts)):
                a, b, c = cf[ti]
                T = np.c_[Dp[b] - Dp[a], Dp[c] - Dp[a]]
                l1, l2 = np.linalg.solve(T, p - Dp[a])
                out[r] = cuv[a] + l1 * (cuv[b] - cuv[a]) + l2 * (cuv[c] - cuv[a])
            return gfree, out, {"phi_model": "harmonic map Omega_1 -> logical polygon (no flip)"}
    Dc = _tutte_disk(len(cuv), (I, J, W), bset_c, circ(t_chart[bset_c]))
    # lattice: triangulated quads, uniform weights
    nl = len(logical)
    I, J = [], []
    for q in quads:
        for (i, j) in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0]), (q[0], q[2])):
            I += [i, j]; J += [j, i]
    t_grid = np.zeros(nl)
    for vtx, k, j in walk:
        t_grid[vtx] = cum_n[k] + j
    Dg = _tutte_disk(nl, (I, J, np.ones(len(I))), bnd, circ(t_grid[bnd]))
    if cxyz is not None and cf3 is not None:
        got = _slim_polygon_map(Dc, Dg, quads, np.asarray(logical, float), bset_c, bpoly, cuv, cf,
                                np.asarray(cxyz, float), np.asarray(cf3, int), bnd, flips)
        if got is not None:
            return got
    tri = Triangulation(Dc[:, 0], Dc[:, 1], cf)
    finder = tri.get_trifinder()
    gfree = np.setdiff1d(np.arange(nl), bnd)
    pts = Dg[gfree]
    t = finder(pts[:, 0], pts[:, 1])
    if np.any(t < 0):
        return None
    out = np.zeros((len(gfree), 2))
    for r, (ti, p) in enumerate(zip(t, pts)):
        a, b, c = cf[ti]
        T = np.c_[Dc[b] - Dc[a], Dc[c] - Dc[a]]
        l1, l2 = np.linalg.solve(T, p - Dc[a])
        out[r] = cuv[a] + l1 * (cuv[b] - cuv[a]) + l2 * (cuv[c] - cuv[a])
    return gfree, out, {"phi_model": "composite of two Tutte disk maps (bijective)",
                        "harmonic_direct_flipped_triangles": flips}


# ------------------------------------------------------------------ planner
def plan_structured_grid(xyz, f3, uv, f2, params: Any):
    """Omega_0 -> Omega_1 -> structured grid M2D plan (Zhang2021M2DPlan container, report-only gates)."""
    from .optcuts_zhang2021_quad_20261027 import Zhang2021QuadModeError, _params_with, plan_zhang2021_m2d
    from .zhang2021_onestring_boundary import build_source_boundary, oriented_boundary_loops, uv_to_xyz_vertex_map
    from .zhang2021_onestring_precondition import boundary_features, precondition_omega
    from .zhang2021_onestring_reparam import signed_uv_areas
    from .zhang2021_panel_optimizer import PanelOptimizerWeights

    started = time.perf_counter()
    xyz = np.asarray(xyz, float); f3 = np.asarray(f3, int); uv = np.asarray(uv, float); f2 = np.asarray(f2, int)
    tols = [float(t) for t in str(getattr(params, "structured_grid_smoothing_tolerances", "0.01,0.005,0")).split(",")
            if t.strip()]
    keep = float(getattr(params, "structured_grid_corner_scale", 0.04))
    smoothing_log = []
    pre = None
    for tol in tols:
        cand = precondition_omega(xyz, f3, uv, f2, tol, node_keep_ratio=keep, boundary_model="spline",
                                  sd_mean_ratio_max=float(getattr(params, "zhang2021_precondition_sd_mean_ratio_max", 1.08)),
                                  sd_p95_ratio_max=float(getattr(params, "zhang2021_precondition_sd_p95_ratio_max", 1.30)))
        smoothing_log.append({"tol": tol, "accepted": cand.accepted, "reason": cand.reason})
        print(f"[SG-OMEGA] boundary smoothing tol={tol}: {'accepted' if cand.accepted else cand.reason}", flush=True)
        if cand.accepted:
            pre = cand
            break
    if pre is None:
        raise StructuredGridError("SG_OMEGA_SMOOTHING_REJECTED: " + str(smoothing_log))
    sa = signed_uv_areas(pre.uv, pre.f2)
    mirror = np.array([-1.0, 1.0]) if np.all(sa < 0) else np.array([1.0, 1.0])
    uvm = pre.uv * mirror
    u2x = uv_to_xyz_vertex_map(pre.f3, pre.f2, len(pre.uv))
    src = build_source_boundary(uvm, pre.f2, u2x, pre.xyz)
    diag = float(np.linalg.norm(np.ptp(uvm, axis=0)))
    area3d = float(0.5 * np.sum(np.linalg.norm(np.cross(xyz[f3[:, 1]] - xyz[f3[:, 0]], xyz[f3[:, 2]] - xyz[f3[:, 0]]), axis=1)))
    h3d = float(getattr(params, "structured_grid_panel_edge", 0.0) or 0.0)
    if h3d <= 0:
        h3d = float(np.sqrt(area3d / max(int(getattr(params, "structured_grid_auto_panels", 150)), 4)))
    feats = boundary_features(src, keep * diag)
    ratio0 = float(getattr(params, "structured_grid_min_side_ratio", 0.7))
    dom, errors = None, []
    cone_pairs = []
    if bool(getattr(params, "structured_grid_seam_cones", True)):
        feats, cone_pairs = seam_cone_candidates(
            src, feats, h3d, spacing=float(getattr(params, "structured_grid_cone_spacing", 1.0)),
            open_spacing=float(getattr(params, "structured_grid_cone_spacing", 1.0)))
    for ratio in (ratio0, 0.5 * ratio0):
        try:
            dom = logical_domain(src, feats, h3d, keep * diag, min_ratio=ratio, cone_pairs=cone_pairs,
                                 w_cone=float(getattr(params, "structured_grid_cone_weight", 0.5)),
                                 shape_weight=float(getattr(params, "structured_grid_shape_weight", 0.05)),
                                 turn_weight=float(getattr(params, "structured_grid_turn_weight", 1.0)))
            dom["min_side_ratio"] = ratio
            break
        except StructuredGridError as exc:
            errors.append(str(exc)[:400])
            print(f"[SG-GRID] min side ratio {ratio:g}: {str(exc)[:300]}", flush=True)
    if dom is None:
        raise StructuredGridError("; ".join(errors))
    logical, quads, vid, walk = lattice_grid(dom)
    grid_uv, placement = place_grid(src, dom, logical, quads, walk, chart_uv=uvm, chart_f2=pre.f2,
                                    chart_xyz=pre.xyz, chart_f3=pre.f3)
    grid_uv = grid_uv * mirror
    print(f"[SG-GRID] seam cones: {len(dom.get('seam_cones', []))} of {dom.get('cone_candidates', 0)} candidates "
          f"(3D valence 3: {sum(1 for c in dom.get('seam_cones', []) if c['valence_3d'] == 3)}, "
          f"5: {sum(1 for c in dom.get('seam_cones', []) if c['valence_3d'] == 5)})", flush=True)
    print(f"[SG-GRID] features={len(feats)} convex={dom['convex']} reflex={dom['reflex']} sides={dom['lengths']} "
          f"-> {len(quads)} quads, panel edge target {h3d:.4g}", flush=True)
    p = _params_with(params, zhang2021_repair_node_valence=False, zhang2021_seam_matching=False,
                     zhang2021_hard_seams=True)
    fake = SimpleNamespace(quad_vertices=grid_uv, quads=quads, metrics={
        "zhang2021_flipped_quads": 0, "zhang2021_quad_boundary_loops": 1, "zhang2021_n_patches": 1,
        "zhang2021_j_min": float("nan"), "zhang2021_d_h_percent_bbox": 0.0})
    try:
        plan = plan_zhang2021_m2d(pre.xyz, pre.f3, pre.uv, pre.f2, p, zhang_result=fake,
                                  panel_line_triples=grid_line_triples(vid), strict_node_valence=False,
                                  panel_weights=_panel_weights(params),
                                  report_only=True, keep_interior_3d=True)
    except Zhang2021QuadModeError as exc:
        raise StructuredGridError(str(exc)) from exc
    m = plan.metrics
    unresolved = []
    if not m.get("zhang2021_m2d_quads_valid", False):
        unresolved.append(f"M2D grid inversion/self-intersection: {m.get('zhang2021_m2d_invalid_quads')} quads")
    if m.get("panel_optimizer_skipped"):
        unresolved.append("M2D: " + str(m.get("panel_optimizer_skipped")))
    if not m.get("zhang2021_m2d_topology_valid", False):
        unresolved.append("M2D connectivity: " + str(m.get("zhang2021_m2d_topology_detail")))
    if str(m.get("zhang2021_seam_preflight_mode")) not in ("hard", "no_cut"):
        unresolved.append(f"seam not exactly closable: {m.get('zhang2021_seam_preflight_mode')} "
                          f"{m.get('zhang2021_seam_preflight_reason', '')}")
    bad_nodes = [r for r in m.get("zhang2021_onestring_node_valence_3d", []) if not r.get("ok")]
    if bad_nodes:
        unresolved.append(f"seam nodes with invalid 3D valence: {bad_nodes[:4]}")
    plan.metrics.update({
        "requested_omega_parameterization_mode": MODE,
        "structured_grid_20261028_enabled": True,
        "structured_grid_mode_label": MODE_LABEL,
        "structured_grid_smoothing": smoothing_log,
        "structured_grid_smoothing_tol": float(pre.tol_ratio),
        "structured_grid_features": int(len(feats)),
        "structured_grid_convex_corners": dom["convex"],
        "structured_grid_reflex_corners": dom["reflex"],
        "structured_grid_side_lengths": dom["lengths"],
        "structured_grid_target_side_lengths": [round(x, 3) for x in dom["target_lengths"]],
        "structured_grid_panel_edge_target": h3d,
        "structured_grid_quads": int(len(quads)),
        "structured_grid_initial_placement": placement,
        "structured_grid_min_side_ratio": dom.get("min_side_ratio"),
        "structured_grid_seam_cones": dom.get("seam_cones", []),
        "structured_grid_seam_cone_candidates": int(dom.get("cone_candidates", 0)),
        "structured_grid_seam_cone_valence3": int(sum(1 for c in dom.get("seam_cones", []) if c["valence_3d"] == 3)),
        "structured_grid_seam_cone_valence5": int(sum(1 for c in dom.get("seam_cones", []) if c["valence_3d"] == 5)),
        "structured_grid_logical_attempts": dom.get("attempts", []),
        "structured_grid_interior_valence_all_4": bool(_interior_valence_ok(plan.quads)),
        "structured_grid_unresolved": unresolved,
        "structured_grid_geometrically_verified": not unresolved,
        "structured_grid_seconds": float(time.perf_counter() - started),
    })
    plan.source = SimpleNamespace(xyz=xyz.copy(), f3=f3.copy(), uv=uv.copy(), f2=f2.copy())
    print(f"[SG-PLAN] {len(plan.quads)} quads; M3D quality gate (report only) "
          f"{'passed' if m.get('zhang2021_m3d_quality_passed') else 'not met: ' + '; '.join(m.get('zhang2021_m3d_quality_failures', []))[:300]}; "
          f"unresolved={unresolved or 'none'}", flush=True)
    return plan


PRIORITY_WEIGHTS = {
    # panel optimiser (3D metric) weights per priority; "shape" = square-like panels first
    "shape": dict(angle=2.0, size=0.1, aspect=1.5, angle_sum=1.0),
    "balanced": dict(angle=1.0, size=0.5, aspect=1.0, angle_sum=1.0),
    "size": dict(angle=0.5, size=4.0, aspect=0.5, angle_sum=1.0),
}


def _panel_weights(params: Any):
    from .zhang2021_panel_optimizer import PanelOptimizerWeights
    pr = str(getattr(params, "structured_grid_priority", "shape") or "shape")
    w = dict(PRIORITY_WEIGHTS.get(pr, PRIORITY_WEIGHTS["shape"]))
    return PanelOptimizerWeights(smooth=float(getattr(params, "structured_grid_smoothness_weight", 1.0)), **w)


def _interior_valence_ok(quads: np.ndarray) -> bool:
    quads = np.asarray(quads, int)
    edges: dict = {}
    for q in quads:
        for k in range(4):
            e = tuple(sorted((int(q[k]), int(q[(k + 1) % 4]))))
            edges[e] = edges.get(e, 0) + 1
    bnd = {v for e, c in edges.items() if c == 1 for v in e}
    val = np.bincount(quads.ravel())
    return all(val[v] == 4 for v in np.unique(quads) if v not in bnd)


__all__ = ["MODE", "MODE_LABEL", "FLAG", "mode_enabled", "plan_structured_grid", "logical_domain", "lattice_grid",
           "StructuredGridError"]
