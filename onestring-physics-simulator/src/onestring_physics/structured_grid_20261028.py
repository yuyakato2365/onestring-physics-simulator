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

import os
from types import SimpleNamespace
from typing import Any
import time

import numpy as np

MODE = "optcuts_structured_grid_20261028"
MODE_LABEL = "2026-10-28 | OptCuts → piecewise smooth Ω (line+spline) → structured grid (Brackbill–Saltzman) + 09-18"
FLAG = "optcuts_structured_grid_20261028_enabled"

DIRS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

# 2026-11-03 "seam zipper" mode (params.structured_grid_seam_zipper): every OptCuts seam closes with 4 panels
# around each of its points, the tip included.  In the logical polygon the two copies of a cut branch then lie on
# top of each other (a zero-width slit along grid edges): the tip is a U-turn (360 deg, 4 panels instead of the
# reflex 270 deg corner = 3 panels), corners on the two copies come in complementary pairs (t_A + t_B = 0, no seam
# cones), junctions get 4 panels, and the lattice vertices along the slit are duplicated (one copy per side).
ZIPPER = False


def _slit_runs(m: int, pairs) -> list[list[int]]:
    """Maximal cyclic runs of consecutive sides that are copies of a cut branch (one run per slit)."""
    S = sorted({int(k) for pr in pairs for k in pr})
    if not S:
        return []
    Sset = set(S)
    if len(S) == m:
        return [list(range(m))]
    starts = [k for k in S if (k - 1) % m not in Sset]
    runs = []
    for k0 in starts:
        run = [k0]
        while (run[-1] + 1) % m in Sset:
            run.append((run[-1] + 1) % m)
        runs.append(run)
    return runs


def _zipper_exempt(m: int, pairs) -> set:
    """Side pairs that legitimately touch in a zipper polygon: sides of one slit and the two outer sides that meet
    the slit at its mouth."""
    out = set()
    for run in _slit_runs(m, pairs):
        grp = list(run) + [(run[0] - 1) % m, (run[-1] + 1) % m]
        for a in grp:
            for b in grp:
                if a != b:
                    out.add((min(a, b), max(a, b)))
    return out


def _zipper_polygon_ok(poly, dirs, n, pairs) -> tuple[bool, str]:
    """Weakly simple orthogonal polygon whose slits are zero-width excursions inside a simple outer polygon."""
    from shapely.geometry import LineString, Polygon
    m = len(dirs)
    P = [np.asarray(poly[0], int)]
    for k in range(m):
        P.append(P[-1] + int(n[k]) * DIRS[dirs[k]])
    if np.any(P[-1] != P[0]):
        return False, "not closed"
    P = P[:-1]
    for ka, kb in pairs:
        a0, a1 = P[ka], P[(ka + 1) % m]
        b0, b1 = P[kb], P[(kb + 1) % m]
        if not (np.all(a0 == b1) and np.all(a1 == b0)):
            return False, f"cut copies {ka}/{kb} do not coincide"
    drop = set()
    slits = []
    for run in _slit_runs(m, pairs):
        if np.any(P[run[0]] != P[(run[-1] + 1) % m]):
            return False, "slit does not return to its mouth"
        drop |= set(run[1:])               # keep the mouth point (start of the first copy side)
        slits.append([P[k] for k in run] + [P[(run[-1] + 1) % m]])
    outer = [P[k] for k in range(m) if k not in drop]
    clean = [outer[0]]
    for q in outer[1:]:
        if np.any(q != clean[-1]):
            clean.append(q)
    if len(clean) >= 2 and np.all(clean[0] == clean[-1]):
        clean = clean[:-1]
    if len(clean) < 3:
        return False, "outer polygon degenerate"
    pg = Polygon(np.asarray(clean, float))
    if not (pg.is_valid and pg.area > 0):
        return False, "outer polygon not simple"
    for sl in slits:
        ls = LineString(np.asarray(sl, float))
        if not pg.buffer(1e-9).covers(ls):
            return False, "slit leaves the outer polygon"
    return True, "ok"


def polygon_is_simple(poly, dirs=None, n=None, pairs=None) -> bool:
    """Polygon validity used by the planner (zipper polygons: weakly simple with zero-width slits)."""
    from shapely.geometry import Polygon
    if ZIPPER and dirs is not None and pairs:
        return _zipper_polygon_ok(poly, dirs, n, pairs)[0]
    pg = Polygon(np.asarray(poly, float))
    return bool(pg.is_valid and pg.area > 0)


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
                   w_head: float = 0.5, shape_weight: float = 0.05, turn_weight: float = 0.3,
                   cone_curvature=None, w_curv: float = 1.0, forced_cones: dict | None = None,
                   side_coupling=None, w_field: float = 1.0, target_positions=None,
                   heading_src=None, heading_length_weighted: bool = False,
                   milp_time_limit: float | None = None, turn3d=None) -> dict:
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
    if turn3d is not None:
        ang = _chord_turning_3d(src, feats, turn3d[0], turn3d[1])
    tips = {k for k, j in enumerate(feats) if src.node_kind.get(j) == "tip"}
    if ZIPPER:
        # a zero-width slit: the boundary turns back on itself at the tip (-180 deg)
        ang = dict(ang)
        for k in tips:
            ang[feats[k]] = -180.0
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
    if target_positions is not None:
        # corner targets given in lattice units (e.g. the 10-23 crop frame)
        tp = np.asarray(target_positions, float)
        T = tp - tp[0]
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
    iR = lambda q: 15 * m + P_c + 1 + q             # noqa: E731  |t_A + t_B - curvature/(pi/2)| of candidate q
    kq = None if cone_curvature is None else np.asarray(cone_curvature, float)
    # smooth rotation field: |(C_k - C_l) - (H_k - H_l)| for side pairs tied through the interior
    fpairs = []
    if side_coupling is not None:
        Wc = np.asarray(side_coupling, float)
        fpairs = [(k, l, float(Wc[k, l])) for k in range(m) for l in range(k) if Wc[k, l] > 1e-6]
    iW = lambda p: 15 * m + 2 * P_c + 1 + p         # noqa: E731
    nv = 15 * m + 2 * P_c + 1 + len(fpairs)
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
            if ZIPPER and k in tips:
                row([(iD(k, (d + 2) % 4), 1), (iD(km, d), -1)], 0, 0)     # U-turn: the copies coincide
                continue
            row([(iD(km, d), 1), (iD(k, (d + 2) % 4), 1)], -np.inf, 1)
            for s_ in (-1, 0, 1):
                row([(iT(k, s_), 1), (iD(km, d), -1), (iD(k, (d + s_) % 4), -1)], -1, np.inf)
        if k in tips:
            if ZIPPER:
                row([(iT(k, 0), 1)], 1, 1)          # placeholder; the tip turns by -2 quarter turns (4 panels)
            else:
                row([(iT(k, -1), 1)], 1, 1)
        nk = [(iX(k, d), 1) for d in range(4)]
        row(nk + [(iE(k), -1)], -np.inf, L[k])
        row([(i, -v) for i, v in nk] + [(iE(k), -1)], -np.inf, -L[k])
        for ax in (0, 1):
            pk = [(iX(i, d), float(DIRS[d][ax])) for i in range(k) for d in range(4)]
            row(pk + [(iF(k, ax), -1)], -np.inf, T[k, ax])
            row([(i, -v) for i, v in pk] + [(iF(k, ax), -1)], -np.inf, -T[k, ax])
    ztip = 2 * len(tips) if ZIPPER else 0
    tip_before = np.cumsum([1 if (ZIPPER and k in tips) else 0 for k in range(m)])
    row([(iT(k, s_), s_) for k in range(m) for s_ in (-1, 1)], 4 + ztip, 4 + ztip)
    # every 3D vertex shared by several polygon corners (seam junctions) gets 3..5 panels around it:
    # sum over its copies of (2 - t_k) in [3, 5]
    groups3d: dict = {}
    for k, j in enumerate(feats):
        if src.node_kind.get(j) == "junction":
            groups3d.setdefault(int(src.xyz_id[j]), []).append(k)
    for ks in groups3d.values():
        if len(ks) >= 2:
            c_ = len(ks)
            if ZIPPER:
                row([(iT(k, s_), s_) for k in ks for s_ in (-1, 1)], 2 * c_ - 4, 2 * c_ - 4)   # 4 panels
            else:
                row([(iT(k, s_), s_) for k in ks for s_ in (-1, 1)], 2 * c_ - 5, 2 * c_ - 3)
    if ZIPPER:
        # where a cut meets the open boundary the 3D boundary vertex keeps 2..3 panels (no fold, no spike)
        groups_tr: dict = {}
        for k, j in enumerate(feats):
            if src.node_kind.get(j) == "transition":
                groups_tr.setdefault(int(src.xyz_id[j]), []).append(k)
        for ks in groups_tr.values():
            if len(ks) == 2:
                row([(iT(k, s_), s_) for k in ks for s_ in (-1, 1)], 1, 2)
    for ax in (0, 1):
        row([(iX(k, d), float(DIRS[d][ax])) for k in range(m) for d in range(4)], 0, 0)
    for ka, kb in pairs:
        row([(iX(ka, d), 1) for d in range(4)] + [(iX(kb, d), -1) for d in range(4)], 0, 0)
    for q, (ka, kb) in enumerate(cones):
        tsum = [(iT(ka, s_), s_) for s_ in (-1, 1)] + [(iT(kb, s_), s_) for s_ in (-1, 1)]
        if ZIPPER:
            row(tsum, 0, 0)                                   # complementary corners only: 4 panels on the seam
        elif forced_cones and q in forced_cones:
            sv = float(forced_cones[q])                        # cone required by the cross field
            row(tsum, sv, sv)
        else:
            row(tsum, -1, 1)                                  # 3D valence 3..5 on the seam
        row(tsum + [(iZ(q), -1)], -np.inf, 0)
        row([(i, -v) for i, v in tsum] + [(iZ(q), -1)], -np.inf, 0)
        if kq is not None:
            # the 3D surface's own curvature next to this seam point asks for a cone of k_q quarter turns
            row(tsum + [(iR(q), -1)], -np.inf, float(kq[q]))
            row([(i, -v) for i, v in tsum] + [(iR(q), -1)], -np.inf, -float(kq[q]))
    if cone_pairs:
        # heading reference: the boundary as it really is (Omega_1) when given, so a long side that
        # leaves its axis is seen even if the PolySquare copy straightened it
        hang = _chord_turning(heading_src, feats) if heading_src is not None else ang
        if ZIPPER:
            hang = dict(hang)
            for k in tips:
                hang[feats[k]] = -180.0
        C = np.cumsum([hang[j] / 90.0 for j in feats])
        for k in range(m):
            tk = [(iT(i, s_), s_) for i in range(k + 1) for s_ in (-1, 1)]
            ck = -2.0 * float(tip_before[k])                 # zipper tips: -2 quarter turns each
            # h_k >= |C_k - sum_{i<=k} t_i - o|
            row([(i, -v) for i, v in tk] + [(iO, -1), (iH(k), -1)], -np.inf, -C[k] + ck)
            row(tk + [(iO, 1), (iH(k), -1)], -np.inf, C[k] - ck)
    if fpairs:
        Cc = np.cumsum([_chord_turning(src, feats)[j] / 90.0 for j in feats])
        for p_, (k, l, _w) in enumerate(fpairs):
            tk = [(iT(i, s_), s_) for i in range(l + 1, k + 1) for s_ in (-1, 1)]
            dC = float(Cc[k] - Cc[l])
            cz = -2.0 * float(tip_before[k] - tip_before[l])
            row([(i, -v) for i, v in tk] + [(iW(p_), -1)], -np.inf, -dC + cz)
            row(tk + [(iW(p_), -1)], -np.inf, dC - cz)
    integrality = np.r_[np.ones(11 * m), np.zeros(4 * m + 2 * P_c + 1 + len(fpairs))]
    lo_b = np.zeros(nv)
    lo_b[iO] = -np.inf
    hi_b = np.r_[np.ones(4 * m), np.full(4 * m, M), np.ones(3 * m), np.full(4 * m + 2 * P_c + 1 + len(fpairs), np.inf)]
    attempts = []
    nogood: list[list[int]] = []
    chosen = None
    w_pos = float(shape_weight)
    from .run_trace import trace as _trace
    _trace("grid.logical_domain", "start", min_ratio=min_ratio, max_rounds=max_rounds, zipper=bool(ZIPPER), features=m,
           tips=sorted(tips), cone_pairs=cone_pairs, turn_reference=("surface" if turn3d is not None else "omega"),
           side_target_panels=L, side_min_panels=n_lo, turning_deg=[float(ang[j]) for j in feats],
           feature_kind=[src.node_kind.get(j, "corner") for j in feats], feature_uv=uvf,
           seam_pairs=pairs, milp_time_limit=milp_time_limit)
    for _round in range(max_rounds):
        # stage 1: turns / directions (lengths soft, with the hard lower bounds)
        c = np.zeros(nv)
        for k in range(m):
            c[iE(k)] = 1.0 / max(L[k], 1.0)
            for s_ in (-1, 0, 1):
                c[iT(k, s_)] = w_turn * abs(ang[feats[k]] / 90.0 - s_)
            c[iF(k, 0)] = c[iF(k, 1)] = w_pos / max(float(np.mean(L)), 1.0)
            if cone_pairs:
                # a heading error costs per panel along the side (long sides off their axis are what
                # shears the grid), not once per side
                c[iH(k)] = w_head * (max(L[k], 0.5) if heading_length_weighted else 1.0)
        for p_, (_k, _l, wkl) in enumerate(fpairs):
            c[iW(p_)] = w_field * wkl
        for q in range(P_c):
            c[iZ(q)] = w_cone
            if kq is not None:
                c[iR(q)] = w_curv
        A2, lb2, ub2 = list(A), list(lb), list(ub)
        for ng in nogood:
            r = np.zeros(nv); r[ng] = 1
            A2.append(r); lb2.append(-np.inf); ub2.append(m - 1)
        res = milp(c, constraints=LinearConstraint(np.asarray(A2), lb2, ub2), integrality=integrality,
                   bounds=Bounds(lo_b, hi_b), options={"time_limit": (float(milp_time_limit) if milp_time_limit else (90.0 if cone_pairs else 8.0)),
                                                       "mip_rel_gap": 0.02})
        if res.x is None:
            attempts.append({"round": _round, "status": f"no turn assignment: {res.message}"})
            _trace("grid.logical_domain", "attempt", min_ratio=min_ratio, **attempts[-1])
            break
        x = res.x
        dirs = [int(np.argmax([x[iD(k, d)] for d in range(4)])) for k in range(m)]
        turn = [int(round(sum(s_ * x[iT(k, s_)] for s_ in (-1, 1)))) for k in range(m)]
        if ZIPPER:
            turn = [(-2 if k in tips else t_) for k, t_ in enumerate(turn)]
        # stage 2: integer side lengths for these directions, simple polygon enforced by lazy
        # non-overlap constraints between non-adjacent sides (never by shortening sides)
        n, info = _lengths_for_directions(dirs, L, n_lo, pairs, T, w_pos)
        attempts.append({"round": _round, "turns": turn, **info, "milp_status": str(getattr(res, "message", ""))[:120],
                         "lengths": None if n is None else n.tolist()})
        if n is not None:
            pts = [np.zeros(2, int)]
            for k in range(m):
                pts.append(pts[-1] + n[k] * DIRS[dirs[k]])
            poly = np.asarray(pts[:-1])
            if ZIPPER:
                ok_z, why_z = _zipper_polygon_ok(poly, dirs, n, pairs)
                attempts[-1]["zipper_check"] = why_z
                if ok_z:
                    chosen = (w_pos, dirs, n, turn, poly)
                    break
            else:
                pg = Polygon(poly)
                if np.all(pts[-1] == 0) and pg.is_valid and pg.area > 0:
                    chosen = (w_pos, dirs, n, turn, poly)
                    break
        _trace("grid.logical_domain", "attempt", min_ratio=min_ratio, dirs=dirs, **attempts[-1])
        nogood.append([iT(k, (0 if turn[k] == -2 else turn[k])) for k in range(m)])     # rotation-invariant cut on the turns
    _trace("grid.logical_domain", "result", min_ratio=min_ratio, found=chosen is not None, rounds=len(attempts),
           chosen_turns=(chosen[3] if chosen else None), chosen_lengths=(chosen[2] if chosen else None))
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
    # Super-runs: consecutive sides joined at points where the logical boundary goes straight on
    # (both copies, for seam candidates); grid points are spread by 3D arc length over the whole
    # run, so integer rounding of short sides does not change boundary edge lengths.
    sup = _super_runs(src, feats, turn, n, cones)
    return {"super_runs": sup, "cones_idx": cones, "n_lo": n_lo.tolist(), "T": T.tolist(),
            "seam_cones": cone_report, "cone_candidates": len(cone_pairs),
            "features": feats, "turn": turn, "dirs": dirs, "lengths": n.tolist(),
            "target_lengths": L.tolist(), "polygon": poly, "cut_pairs": pairs,
            "convex": int(sum(1 for t in turn if t == 1)), "reflex": int(sum(1 for t in turn if t == -1)),
            "zipper_tips": int(sum(1 for t in turn if t == -2)),
            "shape_weight": w_pos, "attempts": attempts, "turning_deg": [float(ang[j]) for j in feats]}


def _chord_turning_3d(src, feats: list[int], nrm_loop: np.ndarray, orient: float) -> dict[int, float]:
    """Turning (deg) of the polygon through the feature points measured on the 3D surface: the chords to the
    neighbouring features projected onto the tangent plane at the feature (geodesic turning), signed like the
    chart's CCW loop (``orient``).  2026-11-03: where a conformal chart bends a straight 3D rim (strong scale
    gradient), the chart's own turning puts spurious convex/reflex corner pairs there (degenerate panels)."""
    X = np.asarray([src.xyz[j] for j in feats], float)
    din = X - np.roll(X, 1, axis=0)
    dout = np.roll(X, -1, axis=0) - X
    out = {}
    for k, j in enumerate(feats):
        n = np.asarray(nrm_loop[j], float)
        a = din[k] - (din[k] @ n) * n
        b = dout[k] - (dout[k] @ n) * n
        out[j] = float(orient * np.degrees(np.arctan2(n @ np.cross(a, b), a @ b)))
    return out


def surface_turn_reference(src, feats, xyz, f3):
    """(vertex normals per loop position, orientation sign) for _chord_turning_3d."""
    xyz = np.asarray(xyz, float); f3 = np.asarray(f3, int)
    fn = np.cross(xyz[f3[:, 1]] - xyz[f3[:, 0]], xyz[f3[:, 2]] - xyz[f3[:, 0]])
    vn = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(vn, f3[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
    nrm_loop = vn[np.asarray(src.xyz_id, int)]
    a2 = _chord_turning(src, sorted(feats))
    a3 = _chord_turning_3d(src, sorted(feats), nrm_loop, 1.0)
    orient = 1.0 if sum(a2[j] * a3[j] for j in a2) >= 0 else -1.0
    return nrm_loop, orient


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


def seam_curvature(src, cone_pairs, xyz, f3, h3d: float, radius: float = 1.5) -> np.ndarray:
    """Integrated Gaussian curvature (angle defect) of the 3D surface next to every seam-cone
    candidate, in quarter turns: each interior 3D vertex within radius*h3d is credited to its
    nearest seam site (cone candidates and cut tips, which already carry a valence-3 cone)."""
    from scipy.spatial import cKDTree
    X = np.asarray(xyz, float)
    F = np.asarray(f3, int)
    n = len(X)
    angsum = np.zeros(n)
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, v = X[b] - X[a], X[c] - X[a]
        cs = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        np.add.at(angsum, a, np.arccos(np.clip(cs, -1, 1)))
    from collections import Counter
    ec = Counter(tuple(sorted((int(F[t, k]), int(F[t, (k + 1) % 3])))) for t in range(len(F)) for k in range(3))
    bnd = {v for e, c in ec.items() if c == 1 for v in e}
    defect = 2 * np.pi - angsum
    defect[list(bnd)] = 0.0
    tips = [j for j, kd in src.node_kind.items() if kd == "tip"]
    sites = [int(src.xyz_id[a]) for a, _b in cone_pairs] + [int(src.xyz_id[j]) for j in tips]
    if not sites:
        return np.zeros(0)
    tree = cKDTree(X[sites])
    dist, owner = tree.query(X, distance_upper_bound=radius * h3d)
    k = np.zeros(len(sites))
    ok = np.isfinite(dist)
    np.add.at(k, owner[ok], defect[ok])
    return k[:len(cone_pairs)] / (0.5 * np.pi)


def seam_cone_candidates(src, base_feats: list[int], h3d: float, spacing: float = 2.0,
                         open_spacing: float = 2.0, forced_xyz: dict | None = None):
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
            if forced_xyz and int(src.xyz_id[j]) in forced_xyz and posB.get(int(src.xyz_id[j])) is not None:
                q = posB[int(src.xyz_id[j])]
                pairs.append((int(j), int(q)))
                feats.update((int(j), int(q)))
                last = j
                continue
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


def _find_triangles(V: np.ndarray, F: np.ndarray, pts: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Index of a triangle of (V, F) containing each point (-1 if none).  Unlike matplotlib's TriFinder this
    accepts coincident vertices (the two copies of a zipper slit map to the same logical points)."""
    V = np.asarray(V, float); F = np.asarray(F, int); pts = np.asarray(pts, float)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = B - A, C - A
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    ok = np.abs(det) > 1e-300
    lo, hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
    out = np.full(len(pts), -1, int)
    for s0 in range(0, len(pts), chunk):
        P = pts[s0:s0 + chunk]
        inb = ((P[:, None, 0] >= lo[None, :, 0] - 1e-9) & (P[:, None, 0] <= hi[None, :, 0] + 1e-9) &
               (P[:, None, 1] >= lo[None, :, 1] - 1e-9) & (P[:, None, 1] <= hi[None, :, 1] + 1e-9) & ok[None, :])
        for r, row in enumerate(inb):
            cand = np.flatnonzero(row)
            if not len(cand):
                continue
            d = P[r] - A[cand]
            l1 = (d[:, 0] * e2[cand, 1] - d[:, 1] * e2[cand, 0]) / det[cand]
            l2 = (e1[cand, 0] * d[:, 1] - e1[cand, 1] * d[:, 0]) / det[cand]
            mn = np.minimum(np.minimum(l1, l2), 1 - l1 - l2)
            best = int(np.argmax(mn))
            if mn[best] >= -1e-9:
                out[s0 + r] = int(cand[best])
    return out


def _locate_bary(Vsrc: np.ndarray, F: np.ndarray, pts: np.ndarray, Vdst: np.ndarray):
    """Map points through the piecewise-linear map Vsrc -> Vdst on triangles F (None if any is outside)."""
    if ZIPPER:
        t = _find_triangles(Vsrc, F, pts)
    else:
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


def _untangle_elliptic(P: np.ndarray, F: np.ndarray, free: np.ndarray, ref: np.ndarray,
                       outer: int = 30, iters: int = 400) -> np.ndarray:
    """Foldover-free map by regularised elliptic smoothing (Garanzha, Kaporin, Kudryavtseva,
    Protais, Ray, Sokolov 2021, "Foldover-free maps in 50 lines of code"): minimise
    sum_t A_t [ |J|^2 / chi(det J, eps) + 0.5 (det J^2 + 1) / chi(det J, eps) ],
    chi(D, eps) = (D + sqrt(eps^2 + D^2)) / 2, with eps shrinking until no triangle is inverted.
    J maps the reference triangle (``ref`` positions, e.g. the chart) to P; boundary vertices fixed."""
    from scipy.optimize import minimize
    P = np.asarray(P, float).copy()
    F = np.asarray(F, int)
    ref = np.asarray(ref, float)
    if ref.ndim == 3:                                     # per-triangle reference shapes (t, 3, 2)
        R0 = np.stack([ref[:, 1] - ref[:, 0], ref[:, 2] - ref[:, 0]], axis=2)
    else:
        R0 = np.stack([ref[F[:, 1]] - ref[F[:, 0]], ref[F[:, 2]] - ref[F[:, 0]]], axis=2)     # (t, 2, 2) columns
    detR = R0[:, 0, 0] * R0[:, 1, 1] - R0[:, 0, 1] * R0[:, 1, 0]
    Rinv = np.stack([np.stack([R0[:, 1, 1], -R0[:, 0, 1]], 1), np.stack([-R0[:, 1, 0], R0[:, 0, 0]], 1)], 1) / detR[:, None, None]
    area = 0.5 * np.abs(detR)
    area = area / area.sum()

    def jac(Q):
        M = np.stack([Q[F[:, 1]] - Q[F[:, 0]], Q[F[:, 2]] - Q[F[:, 0]]], axis=2)
        return M @ Rinv

    def fg(z, eps):
        Q = P.copy()
        Q[free] = z.reshape(-1, 2)
        J = jac(Q)
        D = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        Fr = np.sum(J * J, axis=(1, 2))
        sq = np.sqrt(eps * eps + D * D)
        chi = 0.5 * (D + sq)
        dchi = 0.5 * (1.0 + D / sq)
        E = float(np.sum(area * (Fr / chi + 0.5 * (D * D + 1.0) / chi)))
        cof = np.stack([np.stack([J[:, 1, 1], -J[:, 1, 0]], 1), np.stack([-J[:, 0, 1], J[:, 0, 0]], 1)], 1)
        dJ = (2.0 / chi)[:, None, None] * J + ((-Fr * dchi / chi ** 2) + 0.5 * (2.0 * D / chi - (D * D + 1.0) * dchi / chi ** 2))[:, None, None] * cof
        dJ *= area[:, None, None]
        dM = dJ @ np.transpose(Rinv, (0, 2, 1))           # dE/dM, columns = edge 1, edge 2
        G = np.zeros_like(Q)
        np.add.at(G, F[:, 1], dM[:, :, 0])
        np.add.at(G, F[:, 2], dM[:, :, 1])
        np.add.at(G, F[:, 0], -dM[:, :, 0] - dM[:, :, 1])
        return E, G[free].ravel()

    for _ in range(outer):
        J = jac(P)
        D = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        mind = float(D.min())
        if mind > 0:
            break
        eps = float(np.sqrt(1e-8 + 0.04 * mind * mind))
        res = minimize(fg, P[free].ravel(), args=(eps,), jac=True, method="L-BFGS-B", options={"maxiter": iters})
        P[free] = res.x.reshape(-1, 2)
    return P


def _area_refine(prob, uv, free, w_area: float, iters: int = 400):
    """Flip-free L-BFGS on  sum_t A_t [ w_area log^2(det J_t) + (|J_t|_F^2 / det J_t - 2) ].

    J_t maps the 3D triangle to the (area-normalised) logical polygon: log^2 det J pulls every
    triangle towards its 3D area (equal-area map, Moser), the MIPS term keeps it from shearing."""
    from .zhang2021_panel_optimizer import _lbfgs
    uv = np.asarray(uv, float).copy()
    n = len(uv)
    wt = prob.weight
    Dx, Dy = prob.Dx, prob.Dy
    DxT, DyT = Dx.T.tocsr(), Dy.T.tocsr()

    def fun(z):
        P = uv.copy()
        P[free] = z.reshape(-1, 2)
        J = prob.jacobians(P)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if np.any(det <= 1e-14):
            return np.inf, None
        fro = np.sum(J * J, axis=(1, 2))
        ld = np.log(det)
        E = float(np.dot(wt, w_area * ld * ld + fro / det - 2.0))
        cof = np.empty_like(J)
        cof[:, 0, 0] = J[:, 1, 1]; cof[:, 0, 1] = -J[:, 1, 0]
        cof[:, 1, 0] = -J[:, 0, 1]; cof[:, 1, 1] = J[:, 0, 0]
        dE = (w_area * 2.0 * ld / det)[:, None, None] * cof + (2.0 / det)[:, None, None] * J \
            - (fro / (det * det))[:, None, None] * cof
        dE *= wt[:, None, None]
        gu = DxT @ dE[:, 0, 0] + DyT @ dE[:, 0, 1]
        gv = DxT @ dE[:, 1, 0] + DyT @ dE[:, 1, 1]
        G = np.c_[gu, gv]
        return E, G[free].ravel()

    E0, _ = fun(uv[free].ravel())
    z, E1, it = _lbfgs(fun, uv[free].ravel(), max_iter=iters)
    uv[free] = z.reshape(-1, 2)
    J = prob.jacobians(uv)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    ws = wt / wt.sum()
    return uv, {"phi_area_weight": w_area, "phi_area_energy_before": float(E0), "phi_area_energy_after": float(E1),
                "phi_area_iterations": int(it),
                "phi_area_scale_cv": float(np.sqrt(np.dot(ws, (det - np.dot(ws, det)) ** 2)) / max(np.dot(ws, det), 1e-300))}


def _slim_polygon_map(Dc, Dg, quads, logical, bset_c, bpoly, cuv, cf, cxyz, cf3, bnd, harmonic_flips,
                      area_weight=0.0, seam_out=None):
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
        free_c = np.setdiff1d(np.arange(len(P0)), bset_c)
        P0 = _untangle_triangles(P0, cf, free_c)
        e1 = P0[cf[:, 1]] - P0[cf[:, 0]]
        e2 = P0[cf[:, 2]] - P0[cf[:, 0]]
        nflip = int(np.sum(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0))
        if nflip:
            P0 = _untangle_elliptic(P0, cf, free_c, cuv)
            e1 = P0[cf[:, 1]] - P0[cf[:, 0]]
            e2 = P0[cf[:, 2]] - P0[cf[:, 0]]
            nflip = int(np.sum(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0))
    if nflip:
        bad = np.flatnonzero(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0)
        import os as _os
        if _os.environ.get("SG_DUMP_FLIP"):
            import pickle as _pk
            _pk.dump({"P0": P0, "cf": cf, "bset": bset_c, "bpoly": bpoly, "cuv": cuv, "logical": logical,
                      "quads": quads}, open(_os.environ["SG_DUMP_FLIP"], "wb"))
        bs = set(int(v) for v in bset_c)
        print(f"[SG-PHI] composite map has {nflip} flipped/degenerate chart triangles; SLIM skipped "
              f"(e.g. {[(cf[t].tolist(), [int(v) in bs for v in cf[t]]) for t in bad[:3]]})", flush=True)
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
        if seam_out is not None:
            seam_out.update({"phi": cur.copy(), "scale": scale, "prob": prob})
        area_info = {}
        if area_weight > 0:
            # equal-area pull-back: every lattice cell then carries the same 3D surface area
            cur, area_info = _area_refine(prob, cur, free, float(area_weight),
                                          iters=2000)
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
                        **area_info,
                        "harmonic_direct_flipped_triangles": harmonic_flips}


def _super_runs(src, feats, turn, n, cones):
    m = len(feats)
    keep = np.array([turn[k] != 0 or bool(src.node_kind.get(feats[k])) for k in range(m)])
    for ka, kb in cones:
        if turn[ka] != 0 or turn[kb] != 0:
            keep[ka] = keep[kb] = True
    sup = [None] * m
    starts = [k for k in range(m) if keep[k]] or [0]
    for si, k0 in enumerate(starts):
        k1 = starts[(si + 1) % len(starts)]
        span = [(k0 + i) % m for i in range(((k1 - k0) % m) or m)]
        N = int(sum(n[k] for k in span))
        off = 0
        for k in span:
            sup[k] = (int(k0), int(span[-1]), int(off), N)
            off += int(n[k])
    return sup


def _seamless_slim(prob, uv0, groups, gauge: int, iters: int = 60, tol: float = 1e-6):
    """Free-boundary SLIM (3D symmetric Dirichlet) under seamless constraints.

    groups: list of (R, [(a0, b0), (a1, b1), ...]): chart vertices a_i (one copy of a cut piece)
    and b_i (the other copy) with x_b = R x_a + t (same t inside a group).  Every iterate keeps
    the constraints exactly (KKT solve) and is flip-free (bounded step)."""
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from .zhang2021_onestring_reparam import _max_feasible_step
    uv = np.asarray(uv0, float).copy()
    nV = len(uv)
    rows, cols, vals = [], [], []
    r = 0
    for R, prs in groups:
        a0, b0 = prs[0]
        for a, b in prs[1:]:
            # (x_b - R x_a) - (x_b0 - R x_a0) = 0, two rows
            for i in range(2):
                for vtx, sgn in ((b, 1.0), (b0, -1.0)):
                    rows.append(r + i); cols.append(i * nV + vtx); vals.append(sgn)
                for j in range(2):
                    if R[i][j]:
                        for vtx, sgn in ((a, -1.0), (a0, 1.0)):
                            rows.append(r + i); cols.append(j * nV + vtx); vals.append(sgn * R[i][j])
            r += 2
    for i in range(2):
        rows.append(r); cols.append(i * nV + gauge); vals.append(1.0); r += 1
    C = sparse.csr_matrix((vals, (rows, cols)), shape=(r, 2 * nV))
    x0 = np.r_[uv[:, 0], uv[:, 1]]
    d = C @ x0
    E, J, _ = prob.energy(uv)
    if not np.isfinite(E):
        return uv, {"seamless_status": "start not injective"}
    E_start = E
    sw = np.sqrt(prob.area)
    Dx, Dy = prob.Dx, prob.Dy
    done = 0
    for _ in range(int(iters)):
        U, S, Vt = np.linalg.svd(J)
        Rt = U @ Vt
        sv = np.clip(S, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            w2 = np.where(np.abs(sv - 1) > 1e-8, (sv - sv ** -3) / (sv - 1), 4.0)
        w = np.sqrt(np.maximum(w2, 1e-12))
        W = np.einsum("tij,tj,tkj->tik", U, w, U)
        WR = W @ Rt
        blocks, rhs = [], []
        for rr in (0, 1):
            for c_, D in ((0, Dx), (1, Dy)):
                Au = sparse.diags(sw * W[:, rr, 0]) @ D
                Av = sparse.diags(sw * W[:, rr, 1]) @ D
                blocks.append(sparse.hstack([Au, Av]))
                rhs.append(sw * WR[:, rr, c_])
        A = sparse.vstack(blocks).tocsc()
        b = np.concatenate(rhs)
        H = (A.T @ A).tocsc()
        H = H + sparse.identity(H.shape[0], format="csc") * (1e-10 * float(H.diagonal().mean()))
        K = sparse.bmat([[H, C.T], [C, None]], format="csc")
        sol = splu(K).solve(np.r_[A.T @ b, d])
        xn = sol[:2 * nV]
        x = np.r_[uv[:, 0], uv[:, 1]]
        dx = xn - x
        duv = np.c_[dx[:nV], dx[nV:]]
        dJ = prob.jacobians(duv)
        amax = _max_feasible_step(J, dJ)
        alpha = min(1.0, 0.8 * amax)
        accepted = False
        while alpha > 1e-12:
            trial = uv + alpha * duv
            Et, Jt, _ = prob.energy(trial)
            if np.isfinite(Et) and Et <= E:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        rel = (E - Et) / max(E, 1e-30)
        uv, E, J = trial, Et, Jt
        done += 1
        if rel < tol:
            break
    viol = float(np.max(np.abs(C @ np.r_[uv[:, 0], uv[:, 1]] - d))) if C.shape[0] else 0.0
    return uv, {"seamless_sd_before": float(E_start), "seamless_sd_after": float(E), "seamless_iterations": done,
                "seamless_constraint_violation": viol}


def _rot_between(da, db) -> list:
    """Integer rotation matrix R (multiple of 90 deg) with R da = db (unit lattice directions)."""
    for R in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]]):
        if np.allclose(np.asarray(R) @ np.asarray(da), np.asarray(db)):
            return R
    raise StructuredGridError("SG_SEAMLESS_ROTATION_NOT_FOUND")


def refine_domain_seamless(src, dom: dict, seam_out: dict, w_pos: float = 2.0) -> tuple[dict | None, dict]:
    """Re-size the logical polygon to the seamless, free-boundary low-distortion flattening.

    The bijective chart -> polygon map phi (seam_out) satisfies, for every cut, copy B = R copy A + t
    with the rotation R fixed by the chosen turns.  Relaxing the polygon outline under exactly
    these constraints (free-boundary seamless SLIM) gives the shape the surface itself wants for
    the chosen cones; its corner positions and side lengths become the new integer side lengths."""
    feats = dom["features"]
    m = len(feats)
    dirs = dom["dirs"]
    loop = np.asarray(src.loop, int)
    n = len(loop)
    groups = []
    for ka, kb in dom["cut_pairs"]:
        PA = _cyclic_positions(n, feats[ka], feats[(ka + 1) % m])
        PB = _cyclic_positions(n, feats[kb], feats[(kb + 1) % m])
        posB = {int(src.xyz_id[j]): j for j in PB}
        prs = [(int(loop[a]), int(loop[posB[int(src.xyz_id[a])]])) for a in PA if int(src.xyz_id[a]) in posB]
        if len(prs) < 2:
            continue
        R = _rot_between(DIRS[dirs[ka]], -DIRS[dirs[kb]])
        groups.append((R, prs))
    phi, scale, prob = seam_out["phi"], seam_out["scale"], seam_out["prob"]
    star, info = _seamless_slim(prob, phi, groups, gauge=int(loop[feats[0]]))
    Tst = star[loop[np.asarray(feats)]] / scale
    Tst = Tst - Tst[0]
    Lst = np.array([max(float((Tst[(k + 1) % m] - Tst[k]) @ DIRS[dirs[k]]), 0.3) for k in range(m)])
    n_lo = np.maximum(1, np.floor(0.7 * Lst)).astype(int)
    nn, linfo = _lengths_for_directions(dirs, Lst, n_lo, dom["cut_pairs"], Tst, w_pos)
    info.update({"seamless_relength": linfo.get("status")})
    if nn is None:
        return None, info
    pts = [np.zeros(2, int)]
    for k in range(m):
        pts.append(pts[-1] + nn[k] * DIRS[dirs[k]])
    from shapely.geometry import Polygon
    poly = np.asarray(pts[:-1])
    if ZIPPER:
        if not _zipper_polygon_ok(poly, dirs, nn, dom["cut_pairs"])[0]:
            info["seamless_relength"] = "not a zipper polygon"
            return None, info
    elif np.any(pts[-1] != 0) or not Polygon(poly).is_valid:
        info["seamless_relength"] = "not a simple closed polygon"
        return None, info
    new = dict(dom)
    new.update({"lengths": nn.tolist(), "polygon": poly,
                "super_runs": _super_runs(src, feats, dom["turn"], nn, dom.get("cones_idx", [])),
                "seamless_lengths_before": list(dom["lengths"])})
    return new, info


def _lengths_for_directions(dirs, L, n_lo, pairs, T, w_pos, *, max_iter: int = 40, under_weight: float = 1.0):
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
            add(-float(under_weight) * cn, [(m + k, -1)], -np.inf, -float(under_weight) * L[k])
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
        if ZIPPER:
            ex = _zipper_exempt(m, pairs)
            touching = [pq for pq in touching if (min(pq), max(pq)) not in ex]
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
    if ZIPPER and dom.get("cut_pairs"):
        return _lattice_grid_zipper(dom)
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


def _ccw_angle(a, b) -> float:
    """Counter-clockwise angle in [0, 2 pi) from direction a to direction b."""
    t = float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    return t if t >= 0 else t + 2 * np.pi


def _lattice_grid_zipper(dom: dict):
    """lattice_grid for a zipper polygon: cells of the outer polygon (slits removed); every lattice point on a slit
    except the tip gets a second vertex for the quads on the far side, so the M2D is cut along the seam while the
    two copies keep the same logical position (they close with 4 panels in 3D)."""
    from matplotlib.path import Path as _Path
    poly = np.asarray(dom["polygon"], int)
    dirs, lengths = list(dom["dirs"]), [int(x) for x in dom["lengths"]]
    m = len(dirs)
    P = [poly[0].copy()]
    for k in range(m):
        P.append(P[-1] + lengths[k] * DIRS[dirs[k]])
    P = P[:-1]
    runs = _slit_runs(m, dom["cut_pairs"])
    drop = set()
    for run in runs:
        drop |= set(run[1:])
    outer = np.asarray([P[k] for k in range(m) if k not in drop], float)
    path = _Path(outer)
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
    quads = np.asarray(quads, int)
    logical = [None] * len(vid)
    for (x, y), i in vid.items():
        logical[i] = (x, y)
    qc = np.asarray([[np.mean([logical[i][0] for i in q]), np.mean([logical[i][1] for i in q])] for q in quads])
    dupB: dict[tuple[int, int], int] = {}      # slit point -> vertex used by the quads on the B side
    side_copy: dict[int, str] = {}            # side index -> "A" (towards the tip) / "B" (back to the mouth)
    after_run: set = set()
    for run in runs:
        tipk = next((k for k in run[1:] if dirs[k] == (dirs[(k - 1) % m] + 2) % 4), None)
        if tipk is None:
            raise StructuredGridError("SG_ZIPPER_SLIT_WITHOUT_TIP")
        ia = run.index(tipk)
        for k in run[:ia]:
            side_copy[k] = "A"
        for k in run[ia:]:
            side_copy[k] = "B"
        after_run.add((run[-1] + 1) % m)
        # slit path from the mouth to the tip along copy A
        pts = [P[run[0]].copy()]
        for k in run[:ia]:
            for _j in range(lengths[k]):
                pts.append(pts[-1] + DIRS[dirs[k]])
        d_o_in = DIRS[dirs[(run[0] - 1) % m]].astype(float)
        for i, p in enumerate(pts[:-1]):                       # the tip (last point) stays shared
            d_out = (pts[i + 1] - p).astype(float)
            back = -d_o_in if i == 0 else -(p - pts[i - 1]).astype(float)
            span_A = _ccw_angle(d_out, back) or 2 * np.pi
            key = (int(p[0]), int(p[1]))
            if key not in vid:
                raise StructuredGridError("SG_ZIPPER_SLIT_POINT_NOT_ON_GRID")
            iv = vid[key]
            new_id = len(logical)
            used = False
            for qi in np.flatnonzero((quads == iv).any(axis=1)):
                w = qc[qi] - p
                if _ccw_angle(d_out, w) >= span_A:              # right of the slit path: B side
                    quads[qi][quads[qi] == iv] = new_id
                    used = True
            if used:
                logical.append(key)
                dupB[key] = new_id
    logical = np.asarray(logical, float)
    walk = []
    p = P[0].copy()
    for k, (dk, nk) in enumerate(zip(dirs, lengths)):
        p = P[k].copy()
        for j in range(nk):
            key = (int(p[0]), int(p[1]))
            if key not in vid:
                raise StructuredGridError("SG_LOGICAL_BOUNDARY_NOT_ON_GRID")
            use_b = key in dupB and (side_copy.get(k) == "B" or (k in after_run and j == 0))
            walk.append((dupB[key] if use_b else vid[key], k, j))
            p = p + DIRS[dk]
    lattice_grid.last_zipper = {"slits": len(runs), "duplicated": len(dupB)}
    return logical, quads, vid, walk


def grid_line_triples_topo(logical: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """(a, i, b) along grid lines from the quad edges (slit copies are never neighbours)."""
    nb: dict = {}
    for q in np.asarray(quads, int):
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            nb.setdefault(a, set()).add(b)
            nb.setdefault(b, set()).add(a)
    L = np.asarray(logical, float)
    out = []
    for i, ns in nb.items():
        for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            a = [j for j in ns if np.allclose(L[j], L[i] - e)]
            b = [j for j in ns if np.allclose(L[j], L[i] + e)]
            if len(a) == 1 and len(b) == 1:
                out.append((a[0], i, b[0]))
    return np.asarray(out, int).reshape(-1, 3)


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
               chart_xyz: np.ndarray | None = None, chart_f3: np.ndarray | None = None,
               area_weight: float = 0.0, seam_out: dict | None = None) -> tuple[np.ndarray, dict]:
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
    sup = dom.get("super_runs") or [(k, k, 0, dom["lengths"][k]) for k in range(m)]
    for vtx, k, j in walk:
        k0, k1, off, N = sup[k]
        a, b = feats[k0], feats[(k1 + 1) % m]
        lo, hi = run_interval(src, (a, b))
        s_ = _param_from_fraction_3d(src, lo, hi, (off + j) / N) % P
        uv[vtx] = src.point_at(s_)[2]
    bnd = np.array(sorted({w[0] for w in walk}))
    info = {"placement": "grid-graph harmonic"}
    if chart_uv is not None and chart_f2 is not None:
        try:
            got = _pullback_interior(src, dom, logical, quads, bnd, walk, np.asarray(chart_uv, float),
                                     np.asarray(chart_f2, int), chart_xyz, chart_f3, area_weight, seam_out)
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


def _pullback_interior(src, dom, logical, quads, bnd, walk, cuv, cf, cxyz=None, cf3=None, area_weight=0.0,
                       seam_out=None):
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
    sup = dom.get("super_runs") or [(k, k, 0, int(lengths[k])) for k in range(m)]
    for k in range(m):
        k0, k1, off, N = sup[k]
        if k != k0:
            continue
        a, b = feats[k0], feats[(k1 + 1) % m]
        lo, hi = run_interval(src, (a, b))
        cnt = (b - a) % len(loop) or len(loop)
        for jpos in (np.arange(a, a + cnt) % len(loop)):
            sv = cum[jpos]
            if sv < lo - 1e-12:
                sv += P
            t_chart[loop[jpos]] = (cum_n[k0] + _fraction_3d(src, lo, hi, sv) * N) % total
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
        gfree = np.setdiff1d(np.arange(len(logical)), bnd)
        pts = np.asarray(logical, float)[gfree]
        if ZIPPER:
            t = _find_triangles(Dp, cf, pts)
        else:
            tri = Triangulation(Dp[:, 0], Dp[:, 1], cf)
            finder = tri.get_trifinder()
            t = finder(pts[:, 0], pts[:, 1])
        if np.all(t >= 0):
            out = np.zeros((len(gfree), 2))
            for r, (ti, p) in enumerate(zip(t, pts)):
                a, b, c = cf[ti]
                T = np.c_[Dp[b] - Dp[a], Dp[c] - Dp[a]]
                l1, l2 = np.linalg.solve(T, p - Dp[a])
                out[r] = cuv[a] + l1 * (cuv[b] - cuv[a]) + l2 * (cuv[c] - cuv[a])
            if seam_out is not None and cxyz is not None and cf3 is not None:
                from .zhang2021_onestring_reparam import _SDProblem
                X3 = np.asarray(cxyz, float)[np.asarray(cf3, int)]
                a3 = 0.5 * float(np.sum(np.linalg.norm(np.cross(X3[:, 1] - X3[:, 0], X3[:, 2] - X3[:, 0]), axis=1)))
                Pq = np.asarray(logical, float)[np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])]
                pa = 0.5 * float(np.sum(np.abs((Pq[:, 1, 0] - Pq[:, 0, 0]) * (Pq[:, 2, 1] - Pq[:, 0, 1])
                                              - (Pq[:, 1, 1] - Pq[:, 0, 1]) * (Pq[:, 2, 0] - Pq[:, 0, 0]))))
                sc = float(np.sqrt(a3 / max(pa, 1e-300)))
                seam_out.update({"phi": Dp * sc, "scale": sc,
                                 "prob": _SDProblem(np.asarray(cxyz, float), np.asarray(cf3, int), cf, len(cuv))})
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
                                np.asarray(cxyz, float), np.asarray(cf3, int), bnd, flips, area_weight, seam_out)
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
    """Omega_0 -> Omega_1 -> structured grid M2D plan (Zhang2021M2DPlan container, report-only gates).

    2026-10-30: ``params.structured_grid_size_band`` ('0.6,1.6') switches the grid placement and the
    panel step to "shape first" (panel size follows the map within the band, only shape is fought)."""
    from . import aligned_grid_20261028 as _ag
    from .optcuts_zhang2021_quad_20261027 import _size_band
    from .feature_lines_20261029 import apply_feature_preset
    global ZIPPER
    band = _size_band(apply_feature_preset(params))
    previous = _ag.SHAPE_FIRST
    _ag.SHAPE_FIRST = band
    if band is not None:
        print(f"[SG-SHAPE-FIRST] grid placement + panel step: shape first, panel size band {band}", flush=True)
    zip_prev = ZIPPER
    ZIPPER = bool(getattr(apply_feature_preset(params), "structured_grid_seam_zipper", False))   # preset may switch it on
    if ZIPPER:
        print("[SG-ZIPPER] seams close with 4 panels everywhere (tips = U-turns, complementary seam corners, "
              "no seam cones, no cone-first branches)", flush=True)
    from .run_trace import trace as _trace, enabled as _trace_on
    if _trace_on():
        try:   # 2026-11-04: the planner's input next to the trace, so a failure can be replayed offline
            import pickle as _pk
            _f2 = np.asarray(f2, int)
            _E = np.sort(np.concatenate([_f2[:, [0, 1]], _f2[:, [1, 2]], _f2[:, [2, 0]]]), 1)
            _u, _c = np.unique(_E, axis=0, return_counts=True)
            _trace("grid.plan", "start", uv_vertices=len(uv), faces=len(_f2), boundary_edges=int((_c == 1).sum()),
                   zipper=bool(ZIPPER), turn_reference=str(getattr(params, "structured_grid_turn_reference", "omega")),
                   panels=getattr(params, "structured_grid_auto_panels", None),
                   preset=getattr(params, "structured_grid_feature_preset", None))
            _pv = {k: v for k, v in (vars(params) if hasattr(params, "__dict__") else {}).items()
                   if isinstance(v, (int, float, str, bool, type(None)))}
            with open(os.path.join(os.environ["ONESTRING_TRACE_DIR"], "grid_planner_input.pkl"), "wb") as _fh:
                _pk.dump({"xyz": np.asarray(xyz), "f3": np.asarray(f3), "uv": np.asarray(uv), "f2": _f2, "params": _pv}, _fh)
        except Exception:  # noqa: BLE001
            pass
    if bool(getattr(params, "structured_grid_size_from_grid", False)) and int(getattr(params, "nx", 0) or 0) > 0:
        # 2026-11-04: the sidebar "grid size" sets the panel size: a grid of nx cells spans the longer side of
        # Omega's bounding box (measured in the 3D metric), as in the paper route's overlay grid
        import copy as _copy
        _x = np.asarray(xyz, float); _f3 = np.asarray(f3, int); _u = np.asarray(uv, float); _f2 = np.asarray(f2, int)
        _a3 = float(0.5 * np.sum(np.linalg.norm(np.cross(_x[_f3[:, 1]] - _x[_f3[:, 0]], _x[_f3[:, 2]] - _x[_f3[:, 0]]), axis=1)))
        _q = _u[_f2]; _au = float(0.5 * np.sum(np.abs((_q[:, 1, 0] - _q[:, 0, 0]) * (_q[:, 2, 1] - _q[:, 0, 1])
                                                    - (_q[:, 1, 1] - _q[:, 0, 1]) * (_q[:, 2, 0] - _q[:, 0, 0]))))
        _h = float(np.sqrt(_a3 / max(_au, 1e-30)) * float(np.max(np.ptp(_u, axis=0))) / int(params.nx))
        params = _copy.copy(params)
        params.structured_grid_panel_edge = _h
        print(f"[SG-SIZE] grid size {int(getattr(params, 'nx'))} -> panel edge {_h:.4g} in 3D (about {_a3 / _h ** 2:.0f} panels)", flush=True)
    try:
        out = _plan_structured_grid(xyz, f3, uv, f2, params)
        _trace("grid.plan", "done")
        return out
    except Exception as _exc:
        import traceback as _tb
        _trace("grid.plan", "failed", error_type=type(_exc).__name__, error=str(_exc), traceback=_tb.format_exc())
        raise
    finally:
        _ag.SHAPE_FIRST = previous
        ZIPPER = zip_prev


def print(*args, **kwargs):  # noqa: A001 - 2026-11-04: planner messages are also written to the trace log
    import builtins
    builtins.print(*args, **kwargs)
    try:
        msg = " ".join(str(a_) for a_ in args)
        if msg.startswith("[SG"):
            from .run_trace import trace as _trace
            _trace("grid.message", msg.split("]", 1)[0].lstrip("["), text=msg)
    except Exception:  # noqa: BLE001
        pass


def _plan_structured_grid(xyz, f3, uv, f2, params: Any):
    from .optcuts_zhang2021_quad_20261027 import Zhang2021QuadModeError, _params_with, plan_zhang2021_m2d
    from .zhang2021_onestring_boundary import build_source_boundary, oriented_boundary_loops, uv_to_xyz_vertex_map
    from .zhang2021_onestring_precondition import boundary_features, precondition_omega
    from .zhang2021_onestring_reparam import signed_uv_areas
    from .zhang2021_panel_optimizer import PanelOptimizerWeights

    started = time.perf_counter()
    from .feature_lines_20261029 import apply_feature_preset
    params = apply_feature_preset(params)          # 2026-10-29 modes: crease-line settings
    xyz = np.asarray(xyz, float); f3 = np.asarray(f3, int); uv = np.asarray(uv, float); f2 = np.asarray(f2, int)
    try:   # input snapshot (OptCuts chart + parameters) so a failing run can be reproduced offline
        import pickle
        from pathlib import Path
        snap = Path(__file__).resolve().parents[2] / "logs" / "structured_grid_last_input.pkl"
        snap.parent.mkdir(parents=True, exist_ok=True)
        pv = {k: v for k, v in (vars(params) if hasattr(params, "__dict__") else {}).items()
              if isinstance(v, (int, float, str, bool, type(None)))}
        with open(snap, "wb") as fh:
            pickle.dump({"xyz": xyz, "f3": f3, "uv": uv, "f2": f2, "params": pv}, fh)
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    cone_info: dict = {"enabled": False}
    forced_xyz: dict = {}
    feature_lines_used = None
    area3d0 = float(0.5 * np.sum(np.linalg.norm(np.cross(xyz[f3[:, 1]] - xyz[f3[:, 0]], xyz[f3[:, 2]] - xyz[f3[:, 0]]), axis=1)))
    h_req = float(getattr(params, "structured_grid_panel_edge", 0.0) or 0.0) or \
        float(np.sqrt(area3d0 / max(int(getattr(params, "structured_grid_auto_panels", 150)), 4)))
    prune_report: list = []
    prune_ratio = float(getattr(params, "structured_grid_prune_branch_panels", 1.5) or 0.0)
    if str(getattr(params, "structured_grid_placement", "aligned")) in ("quadmesh", "pedp"):
        prune_ratio = max(prune_ratio, 3.0)        # free singularities: short slits only cost quality
    if prune_ratio > 0:
        from .aligned_grid_20261028 import prune_short_branches
        try:
            uv, f2, prune_report = prune_short_branches(xyz, f3, uv, f2, h_req, max_ratio=prune_ratio)
        except Exception as exc:  # noqa: BLE001 - keep the OptCuts cuts
            prune_report = [{"error": f"{type(exc).__name__}: {exc}"}]
        print(f"[SG-PRUNE] sub-panel cut branches sewn back: {prune_report}", flush=True)
    direction_model = str(getattr(params, "structured_grid_direction_model", "field") or "field")
    if bool(getattr(params, "structured_grid_cone_cuts", True)) and not ZIPPER and \
            str(getattr(params, "structured_grid_placement", "aligned")) not in ("quadmesh", "pedp"):
        cone_info, uv, f2, forced_xyz = _cone_first_seams(xyz, f3, uv, f2, params)
    tols = [float(t) for t in str(getattr(params, "structured_grid_smoothing_tolerances", "0.01,0.005,0")).split(",")
            if t.strip()]
    keep = float(getattr(params, "structured_grid_corner_scale", 0.04))
    smoothing_log = []
    pre = None
    if str(getattr(params, "structured_grid_boundary_smoothing", "taubin")) == "taubin":
        cand = precondition_omega(xyz, f3, uv, f2, 0.01, node_keep_ratio=keep, boundary_model="taubin",
                                  smooth_scale=h_req,
                                  sd_mean_ratio_max=float(getattr(params, "structured_grid_taubin_sd_mean_ratio_max", 1.12)),
                                  sd_p95_ratio_max=float(getattr(params, "zhang2021_precondition_sd_p95_ratio_max", 1.30)))
        smoothing_log.append({"model": "taubin", "scale": h_req, "accepted": cand.accepted, "reason": cand.reason})
        print(f"[SG-OMEGA] Taubin boundary low-pass (panel scale {h_req:.4g}): "
              f"{'accepted' if cand.accepted else cand.reason}", flush=True)
        if cand.accepted:
            pre = cand
            tols = []
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
    placement_model = str(getattr(params, "structured_grid_placement", "aligned") or "aligned")
    if placement_model == "pedp":
        # Xiao, Kang, Fu, Chen 2018: PolySquare-enhanced domain partition, applied to Omega_1
        from .pedp_grid_20261028 import pedp_m2d
        from .zhang2021_onestring_reparam import _SDProblem
        grid_uv, quads, lattice_ix, placement = pedp_m2d(
            _SDProblem(pre.xyz, pre.f3, pre.f2, len(uvm)), uvm, pre.f2, src.loop, h3d,
            sigma_ratio=float(getattr(params, "structured_grid_pedp_sigma", 1.0)),
            alpha=float(getattr(params, "structured_grid_pedp_alpha", 20.0)),
            pixel_ratio=float(getattr(params, "structured_grid_pedp_pixel_ratio", 0.5)),
            target=int(round(area3d / h3d ** 2)))
        dom = {"convex": 0, "reflex": 0, "lengths": [], "target_lengths": [], "seam_cones": [],
               "cone_candidates": 0, "attempts": []}
        feats, direction_model, ps_info = [], "polysquare (PeDP)", {}
        vid = {k: v[0] for k, v in lattice_ix.items() if len(v) == 1}
        if getattr(pedp_m2d, "last", None) is not None:
            pedp_m2d.last.update({"xyz": pre.xyz, "f3": pre.f3})
        print(f"[SG-PEDP] {placement}", flush=True)
    elif placement_model == "quadmesh":
        # comparison: unstructured quad mesh of Omega_1 (interior singularities allowed)
        from .quadmesh_20261028 import quadmesh_omega
        grid_uv, quads, placement = quadmesh_omega(
            src, uvm, pre.f2, pre.xyz, pre.f3, h3d,
            lam_o=float(getattr(params, "structured_grid_bs_orthogonality", 0.3)),
            lam_v=float(getattr(params, "structured_grid_bs_volume", 0.3)),
            target_count=area3d / h3d ** 2)
        dom = {"convex": 0, "reflex": 0, "lengths": [], "target_lengths": [], "seam_cones": [],
               "cone_candidates": 0, "attempts": []}
        feats, vid, direction_model, ps_info = [], {}, "none", {}
        print(f"[SG-QUADMESH] {placement}", flush=True)
    else:
        feats = boundary_features(src, keep * diag)
        ratio0 = float(getattr(params, "structured_grid_min_side_ratio", 0.7))
        dom, errors = None, []
        cone_pairs = []
        if bool(getattr(params, "structured_grid_seam_cones", True)):
            feats, cone_pairs = seam_cone_candidates(
                src, feats, h3d, spacing=float(getattr(params, "structured_grid_cone_spacing", 1.0)),
                open_spacing=float(getattr(params, "structured_grid_cone_spacing", 1.0)), forced_xyz=forced_xyz)
        forced_idx = {q: int(forced_xyz[int(src.xyz_id[a])]) for q, (a, _b) in enumerate(cone_pairs)
                      if int(src.xyz_id[a]) in forced_xyz}
        cone_info["forced_seam_cones_used"] = len(forced_idx)
        cone_k = None
        if cone_pairs and float(getattr(params, "structured_grid_curvature_weight", 1.0)) > 0:
            cone_k = seam_curvature(src, cone_pairs, pre.xyz, pre.f3, h3d)
        coupling, w_head = None, 0.5
        src_dir, uv_dir = src, uvm
        ps_info: dict = {}
        placement_model = str(getattr(params, "structured_grid_placement", "aligned") or "aligned")
        crop = None
        crop_targets = None
        shape_w = float(getattr(params, "structured_grid_shape_weight", 0.05))
        if placement_model == "crop_bs":
            # 10-23 rule: whole cells kept by their area fraction inside Omega_1; the logical polygon
            # is the crop straightened (corner positions pulled to the chart in the crop frame)
            from .crop_bs_grid_20261028 import area_threshold_crop, crop_frame_positions
            crop = area_threshold_crop(uvm, pre.f2, h3d,
                                       threshold=float(getattr(params, "structured_grid_crop_threshold", 0.5)))
            crop_targets = crop_frame_positions(crop, src.uv[np.asarray(sorted(feats))], h3d)
            direction_model = "crop"
            forced_idx = {}
            shape_w = float(getattr(params, "structured_grid_crop_shape_weight", 2.0))
            print(f"[SG-CROP] area-threshold crop: angle {crop['angle']} deg, {crop['cells']} whole cells kept, "
                  f"{crop['staircase_corners']} staircase corners", flush=True)
        if direction_model == "field" and str(getattr(params, "structured_grid_direction_source", "polysquare")) == "polysquare":
            # read the boundary headings from a PolySquare-deformed copy of Omega_1 (curved ends of
            # lobes become straight sides with clear corners); only the MILP sees it
            from .aligned_grid_20261028 import polysquare_deform
            from .zhang2021_onestring_reparam import _SDProblem
            try:
                uv_dir, ps_info = polysquare_deform(_SDProblem(pre.xyz, pre.f3, pre.f2, len(uvm)), uvm, src.loop,
                                                    iters=60, w_end=float(getattr(params, "structured_grid_polysquare_weight", 20000.0)))
                src_dir = build_source_boundary(uv_dir, pre.f2, u2x, pre.xyz)
                if list(src_dir.loop) != list(src.loop):
                    raise StructuredGridError("SG_POLYSQUARE_LOOP_CHANGED")
                print(f"[SG-POLYSQUARE] {ps_info}", flush=True)
            except Exception as exc:  # noqa: BLE001 - headings from Omega_1 itself
                src_dir, uv_dir = src, uvm
                ps_info = {"polysquare_failed": f"{type(exc).__name__}: {exc}"}
                print(f"[SG-POLYSQUARE] skipped: {ps_info}", flush=True)
        if direction_model == "field":
            from .aligned_grid_20261028 import side_coupling
            coupling = side_coupling(src_dir, sorted(feats), uv_dir, pre.f2)
            forced_idx = {}                     # directions come from the chart's own rotation field
            w_head = float(getattr(params, "structured_grid_field_heading_weight", 0.1))
        turn3d = None
        if str(getattr(params, "structured_grid_turn_reference", "omega")) == "surface":
            turn3d = surface_turn_reference(src, feats, pre.xyz, pre.f3)
            print(f"[SG-TURN] corner costs from the 3D surface's geodesic turning (orientation {turn3d[1]:+.0f})", flush=True)

        def solve_dom(forced, w_head_override=None, time_limit=None, max_rounds=12, ratios=None):
            for ratio in (ratios or (ratio0, 0.5 * ratio0)):
                try:
                    d_ = logical_domain(src_dir, feats, h3d, keep * diag, min_ratio=ratio, cone_pairs=cone_pairs,
                                        cone_curvature=cone_k, w_curv=float(getattr(params, "structured_grid_curvature_weight", 1.0)),
                                        side_coupling=coupling, w_field=float(getattr(params, "structured_grid_field_weight", 5.0)),
                                        heading_src=(src if str(getattr(params, "structured_grid_heading_source", "omega")) == "omega" else None),
                                        heading_length_weighted=bool(getattr(params, "structured_grid_heading_length_weighted", True)),
                                        w_head=(w_head if w_head_override is None else w_head_override),
                                        forced_cones=forced, target_positions=crop_targets,
                                        w_cone=float(getattr(params, "structured_grid_cone_weight", 0.5)),
                                        shape_weight=shape_w,
                                        turn_weight=float(getattr(params, "structured_grid_turn_weight", 1.0)),
                                        milp_time_limit=time_limit, max_rounds=max_rounds, turn3d=turn3d)
                    d_["min_side_ratio"] = ratio
                    return d_
                except StructuredGridError as exc:
                    errors.append(str(exc)[:400])
                    from .run_trace import trace as _trace
                    _trace("grid.logical_domain", "ratio_failed", min_ratio=ratio, error=str(exc))
                    print(f"[SG-GRID] min side ratio {ratio:g}: {str(exc)[:300]}", flush=True)
            return None

        dom = solve_dom(forced_idx)
        if dom is None:
            raise StructuredGridError("; ".join(errors))
        logical, quads, vid, walk = lattice_grid(dom)
        priority = str(getattr(params, "structured_grid_priority", "shape") or "shape")
        area_w = float(getattr(params, "structured_grid_area_weight", -1.0))
        if area_w < 0:
            area_w = {"size": 50.0, "balanced": 5.0}.get(priority, 0.0)
        seamless_log = []
        aligned_info: dict = {}
        grid_uv = None
        feature_opts = None
        if placement_model == "aligned" and str(getattr(params, "structured_grid_feature_mode", "off") or "off") != "off":
            # 2026-10-29: grid lines along panel-scale creases / lattice aligned with the principal
            # bend directions (see feature_lines_20261029)
            from .feature_lines_20261029 import build_features
            try:
                feature_opts = build_features(xyz, f3, h3d, params)
                print(f"[SG-FEATURES] mode {feature_opts['mode']}, design stage {feature_opts['design']}, "
                      f"creases {feature_opts.get('crease_stats')}, align weight {feature_opts.get('align_weight')}", flush=True)
            except Exception as exc:  # noqa: BLE001 - plain aligned placement
                feature_opts = None
                print(f"[SG-FEATURES] skipped: {type(exc).__name__}: {exc}", flush=True)
        if placement_model == "aligned":
            from .aligned_grid_20261028 import merge_straight_features, place_grid_aligned, refine_domain_aligned
            chart = (uvm, pre.f2, pre.xyz, pre.f3)
            from .aligned_grid_20261028 import cone_feedback, current_cones
            try:
                feedback_log = []
                best = None
                forced_fb = dict(forced_idx)
                dom_k = dom
                for rr in range(int(getattr(params, "structured_grid_cone_feedback_rounds", 0)) + 1):
                    if dom_k is None:
                        break
                    dom_r = merge_straight_features(src, dom_k)
                    lg_r, q_r, vid_r, walk_r = lattice_grid(dom_r)
                    new, info_r = refine_domain_aligned(
                        src, dom_r, lg_r, q_r, walk_r, chart, h3d,
                        min_3d_ratio=float(getattr(params, "structured_grid_min_3d_length_ratio", 0.0)),
                        under_weight=float(getattr(params, "structured_grid_shortfall_weight", 1.0)),
                        area_weight=float(getattr(params, "structured_grid_aligned_area_weight", 1.0)),
                        features=feature_opts)
                    star = info_r.pop("_star", None)
                    sd_c = float(info_r.get("aligned_sd_after", np.inf))
                    print(f"[SG-ALIGNED] round {rr}: {len(dom_k['features'])} -> {len(dom_r['features'])} corners, "
                          f"seam cones {len(dom_k.get('seam_cones', []))}, continuous SD {sd_c:.4f}, "
                          f"rounding {info_r.get('aligned_rounding')} [{time.perf_counter() - started:.0f} s]", flush=True)
                    feedback_log.append({"round": rr, "continuous_sd": sd_c, "seam_cones": len(dom_k.get("seam_cones", [])),
                                         "rounding": info_r.get("aligned_rounding")})
                    if new is not None and (best is None or sd_c < best[0]):
                        best = (sd_c, new, info_r)
                    if star is None or not cone_pairs:
                        break
                    forced_fb, added = cone_feedback(src, cone_pairs, star, chart, h3d,
                                                     {**current_cones(src, cone_pairs, dom_k), **forced_fb},
                                                     threshold=float(getattr(params, "structured_grid_cone_feedback_threshold", 0.35)),
                                                     radius=float(getattr(params, "structured_grid_cone_feedback_radius", 2.0)))
                    feedback_log[-1]["added_cones"] = added
                    print(f"[SG-ALIGNED] cone feedback: {added}", flush=True)
                    if not added:
                        break
                    dom_k = solve_dom(forced_fb)
                # Robustness: the design MILP can land on a poor turn assignment (time limit / solver
                # version / near ties).  When the continuous aligned map is clearly worse than the chart,
                # re-solve the design with other heading weights and keep the best continuous map.
                sd_ref = float(ps_info.get("polysquare_sd", np.nan)) if ps_info else np.nan
                if not np.isfinite(sd_ref):
                    sd_ref = 4.0
                accept = float(getattr(params, "structured_grid_design_sd_ratio", 1.08)) * sd_ref
                budget = float(getattr(params, "structured_grid_design_time_budget", 240.0))
                t_alt = time.perf_counter()
                if best is None or best[0] > accept:
                    for fac in tuple(float(x) for x in str(getattr(params, "structured_grid_design_alternatives", "0.5,2,0.25")).split(",") if x.strip()):
                        print(f"[SG-ALIGNED] solving design alternative heading x{fac:g} "
                              f"({time.perf_counter() - t_alt:.0f} s of {budget:g} s budget used)", flush=True)
                        if time.perf_counter() - t_alt > budget:
                            print(f"[SG-ALIGNED] design alternatives stopped: time budget {budget:g} s used", flush=True)
                            break
                        # bounded: short MILP time limit, few no-good rounds, first side ratio only
                        dom_a = solve_dom(forced_fb, w_head_override=w_head * fac, time_limit=30.0, max_rounds=3,
                                          ratios=(ratio0,))
                        if dom_a is None:
                            continue
                        dom_ar = merge_straight_features(src, dom_a)
                        lg_a, q_a, vid_a, walk_a = lattice_grid(dom_ar)
                        new_a, info_a = refine_domain_aligned(
                            src, dom_ar, lg_a, q_a, walk_a, chart, h3d,
                            min_3d_ratio=float(getattr(params, "structured_grid_min_3d_length_ratio", 0.0)),
                            under_weight=float(getattr(params, "structured_grid_shortfall_weight", 1.0)),
                            area_weight=float(getattr(params, "structured_grid_aligned_area_weight", 1.0)),
                            features=feature_opts)
                        info_a.pop("_star", None)
                        sd_a = float(info_a.get("aligned_sd_after", np.inf))
                        print(f"[SG-ALIGNED] design alternative heading x{fac:g}: continuous SD {sd_a:.4f}, "
                              f"rounding {info_a.get('aligned_rounding')}", flush=True)
                        feedback_log.append({"alternative_heading_factor": fac, "continuous_sd": sd_a,
                                             "rounding": info_a.get("aligned_rounding")})
                        if new_a is not None and (best is None or sd_a < best[0] - 1e-9):
                            best = (sd_a, new_a, info_a)
                            dom_k = dom_a
                        if best is not None and best[0] <= accept:
                            break
                if best is None:
                    raise StructuredGridError("SG_ALIGNED_NO_ROUNDED_POLYGON")
                _sd, new, aligned_info = best
                print(f"[SG-ALIGNED] chosen design: continuous SD {_sd:.4f} (accept <= {accept:.4f})", flush=True)
                aligned_info["cone_feedback"] = feedback_log
                dom_try = new
                lg_t, q_t, vid_t, walk_t = lattice_grid(dom_try)
                g_uv, pl = place_grid_aligned(src, dom_try, lg_t, q_t, walk_t, chart, h3d,
                                              area_weight=float(getattr(params, "structured_grid_aligned_area_weight", 1.0)),
                                              features=feature_opts)
                if feature_opts is not None:
                    fr = pl.get("features") or {}
                    print(f"[SG-FEATURES] placement: SD {fr.get('sd_without_features')} -> {fr.get('aligned_sd_after')}, "
                          f"crease offset {fr.get('feature_offset_before_mean_cells')} -> {fr.get('feature_offset_after_mean_cells')} cells, "
                          f"align residual {fr.get('align_residual_before')} -> {fr.get('align_residual_after')}", flush=True)
                    place_grid_aligned.last_feature_lines = {"lines": feature_opts.get("lines", []),
                                                             "kinds": feature_opts.get("line_kinds", []),
                                                             "dropped_lines": feature_opts.get("dropped_lines", []),
                                                             "dropped_kinds": feature_opts.get("dropped_kinds", [])}
                    feature_lines_used = feature_opts.get("lines") or None
                dom, logical, quads, vid, walk = dom_try, lg_t, q_t, vid_t, walk_t
                grid_uv, placement = g_uv, {**pl, "aligned_refine": aligned_info}
            except Exception as exc:  # noqa: BLE001 - reported, previous placement below
                aligned_info = {"aligned_failed": f"{type(exc).__name__}: {exc}"}
                print(f"[SG-ALIGNED] failed, falling back to the pull-back placement: {aligned_info}", flush=True)
                grid_uv = None
        for _r in range(int(getattr(params, "structured_grid_seamless_rounds", 1))
                        if grid_uv is None and placement_model != "crop_bs" else 0):
            so: dict = {}
            place_grid(src, dom, logical, quads, walk, chart_uv=uvm, chart_f2=pre.f2, chart_xyz=pre.xyz,
                       chart_f3=pre.f3, area_weight=0.0, seam_out=so)
            if "phi" not in so:
                break
            try:
                new, sinfo = refine_domain_seamless(src, dom, so)
            except Exception as exc:  # noqa: BLE001 - keep the current polygon
                new, sinfo = None, {"seamless_error": f"{type(exc).__name__}: {exc}"}
            seamless_log.append(sinfo)
            print(f"[SG-SEAMLESS] round {_r}: {sinfo}", flush=True)
            if new is None:
                break
            dom = new
            logical, quads, vid, walk = lattice_grid(dom)
        if grid_uv is None:
            grid_uv, placement = place_grid(src, dom, logical, quads, walk, chart_uv=uvm, chart_f2=pre.f2,
                                            chart_xyz=pre.xyz, chart_f3=pre.f3, area_weight=area_w)
            placement["seamless_refinement"] = seamless_log
            placement.update(aligned_info)
        if placement_model == "crop_bs":
            from .crop_bs_grid_20261028 import brackbill_saltzman
            bnd_v = sorted({w_[0] for w_ in walk})
            grid_uv, bs_info = brackbill_saltzman(
                grid_uv, quads, bnd_v, uvm, pre.f2, pre.xyz, pre.f3, h3d,
                lam_o=float(getattr(params, "structured_grid_bs_orthogonality", 0.3)),
                lam_v=float(getattr(params, "structured_grid_bs_volume", 0.3)))
            placement.update(bs_info)
            placement["crop"] = {"angle": crop["angle"], "cells": crop["cells"],
                                 "staircase_corners": crop["staircase_corners"]}
            print(f"[SG-BS] Brackbill-Saltzman: {bs_info}", flush=True)
            place_grid.last_crop = {"crop": crop, "dom": dom, "logical": logical, "quads": quads,
                                    "grid_uv": grid_uv, "chart": (uvm, pre.f2, pre.xyz, pre.f3), "h": h3d}   # diagnostics
    grid_uv = grid_uv * mirror
    print(f"[SG-GRID] seam cones: {len(dom.get('seam_cones', []))} of {dom.get('cone_candidates', 0)} candidates "
          f"(3D valence 3: {sum(1 for c in dom.get('seam_cones', []) if c['valence_3d'] == 3)}, "
          f"5: {sum(1 for c in dom.get('seam_cones', []) if c['valence_3d'] == 5)})", flush=True)
    print(f"[SG-GRID] features={len(feats)} convex={dom['convex']} reflex={dom['reflex']} sides={dom['lengths']} "
          f"-> {len(quads)} quads, panel edge target {h3d:.4g}", flush=True)
    principal_field = None
    if float(getattr(params, "structured_grid_panel_principal_weight", 0.0)) > 0:
        # principal bend directions at panel scale for the panel-quality step (feature_lines_20261029)
        from .feature_lines_20261029 import panel_bend_field, panel_principal
        try:
            _fld = panel_bend_field(xyz, f3, h3d)
            _b1, _b2, _d1, _d2 = panel_principal(_fld)
            principal_field = {"V": xyz, "d1": _d1, "d2": _d2, "aniso": np.degrees(_b1 - _b2),
                               "lo": float(getattr(params, "structured_grid_align_aniso_lo", 10.0)),
                               "full": float(getattr(params, "structured_grid_align_aniso_full", 30.0))}
        except Exception as exc:  # noqa: BLE001 - plain panel-quality step
            print(f"[SG-FEATURES] principal field skipped: {type(exc).__name__}: {exc}", flush=True)
    extra_fixed = None
    crease_slide = None
    hold = str(getattr(params, "structured_grid_feature_crease_hold", "slide") or "slide")
    if isinstance(placement, dict) and hold in ("fix", "slide"):
        fr_ = placement.get("features") or {}
        cv = fr_.get("crease_grid_vertices")
        if cv and hold == "fix":
            extra_fixed = np.asarray(cv, int)
            print(f"[SG-FEATURES] {len(cv)} grid vertices on creases kept in place by the panel-quality step", flush=True)
        elif cv and feature_lines_used:
            crease_slide = {"vertices": np.asarray(cv, int), "line_of": np.asarray(fr_["crease_grid_vertex_lines"], int),
                            "lines": feature_lines_used,
                            "weight": float(getattr(params, "structured_grid_feature_slide_weight", 100.0))}
            print(f"[SG-FEATURES] {len(cv)} grid vertices on creases may only slide along them in the panel-quality step", flush=True)
    p = _params_with(params, zhang2021_repair_node_valence=False, zhang2021_seam_matching=False,
                     zhang2021_hard_seams=True)
    fake = SimpleNamespace(quad_vertices=grid_uv, quads=quads, metrics={
        "zhang2021_flipped_quads": 0, "zhang2021_quad_boundary_loops": 1, "zhang2021_n_patches": 1,
        "zhang2021_j_min": float("nan"), "zhang2021_d_h_percent_bbox": 0.0})
    try:
        plan = plan_zhang2021_m2d(pre.xyz, pre.f3, pre.uv, pre.f2, p, zhang_result=fake,
                                  panel_line_triples=(grid_line_triples_topo(logical, quads) if ZIPPER
                                                      else grid_line_triples(vid)), strict_node_valence=False,
                                  panel_weights=_panel_weights(params),
                                  report_only=True, keep_interior_3d=True,
                                  extra_fixed=extra_fixed, principal_field=principal_field,
                                  crease_slide=crease_slide,
                                  boundary_slide=({"max_shift": float(getattr(params, "structured_grid_boundary_slide_max", 0.35))}
                                                  if bool(getattr(params, "structured_grid_panel_boundary_slide", False)) else None))
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
        "structured_grid_cone_first_seams": cone_info,
        "structured_grid_pruned_branches": prune_report,
        "structured_grid_direction_model": direction_model,
        "structured_grid_polysquare": ps_info,
        "structured_grid_placement_model": placement_model,
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
    "size": dict(angle=0.5, size=0.0, aspect=0.5, angle_sum=1.0, area=4.0),
}


def _panel_weights(params: Any):
    from .zhang2021_panel_optimizer import PanelOptimizerWeights
    pr = str(getattr(params, "structured_grid_priority", "shape") or "shape")
    w = dict(PRIORITY_WEIGHTS.get(pr, PRIORITY_WEIGHTS["shape"]))
    # 2026-10-30: keep panels from getting too small / make sizes more even
    if float(getattr(params, "structured_grid_panel_min_size_weight", 0.0) or 0.0) > 0:
        w["min_size"] = float(params.structured_grid_panel_min_size_weight)
        w["min_ratio"] = float(getattr(params, "structured_grid_panel_min_size_ratio", 0.6))
    if float(getattr(params, "structured_grid_panel_neighbor_size_weight", 0.0) or 0.0) > 0:
        w["neighbor_size"] = float(params.structured_grid_panel_neighbor_size_weight)
    if getattr(params, "structured_grid_panel_size_weight", None) not in (None, ""):
        w["size"] = float(params.structured_grid_panel_size_weight)
    return PanelOptimizerWeights(smooth=float(getattr(params, "structured_grid_smoothness_weight", 1.0)),
                                 planar=float(getattr(params, "structured_grid_panel_planar_weight", 0.0)),
                                 principal=float(getattr(params, "structured_grid_panel_principal_weight", 0.0)), **w)


def _cone_first_seams(xyz, f3, uv, f2, params):
    """Cross-field cones -> new short cut branches + forced seam cones (see cross_field_cones_20261028)."""
    from .cross_field_cones_20261028 import boundary_simple, chart_cut_edges, open_slits, plan_cone_cuts, split_chart
    info: dict = {"enabled": True}
    try:
        area3d = float(0.5 * np.sum(np.linalg.norm(np.cross(xyz[f3[:, 1]] - xyz[f3[:, 0]], xyz[f3[:, 2]] - xyz[f3[:, 0]]), axis=1)))
        h = float(getattr(params, "structured_grid_panel_edge", 0.0) or 0.0)
        if h <= 0:
            h = float(np.sqrt(area3d / max(int(getattr(params, "structured_grid_auto_panels", 150)), 4)))
        cut, op = chart_cut_edges(f3, f2)
        plan = plan_cone_cuts(xyz, f3, cut, op, h,
                              max_branch_ratio=float(getattr(params, "structured_grid_max_branch_panels", 6.0)),
                              snap_negative=str(getattr(params, "structured_grid_cone_snap", "positive")) == "all",
                              snap_positive=str(getattr(params, "structured_grid_cone_snap", "positive")) in ("all", "positive"))
        info.update({"singularities": plan["singularities_raw"], "pairs_cancelled": plan["pairs_cancelled"],
                     "report": plan["report"]})
        branches = plan["branches"]
        from .zhang2021_onestring_reparam import signed_uv_areas

        def attempt(brs):
            uv_s, f2_s = split_chart(uv, f2, f3, brs)
            uv_o, oinfo = open_slits(xyz, f3, uv_s, f2_s, touched=getattr(split_chart, "touched", None))
            sa_o = signed_uv_areas(uv_o, f2_s)
            ok = (np.all(sa_o > 0) or np.all(sa_o < 0)) and bool(boundary_simple(uv_o, f2_s))
            return ok, uv_o, f2_s, oinfo

        if branches:
            ok, uv_o, f2_s, oinfo = attempt(branches)
            if not ok:
                # an opened slit overlaps another part of the chart (Omega no longer simple):
                # drop the offending branch (one at a time) instead of all of them
                for k in range(len(branches)):
                    sub = branches[:k] + branches[k + 1:]
                    if not sub:
                        break
                    ok, uv_o, f2_s, oinfo = attempt(sub)
                    if ok:
                        info["branches_dropped"] = f"branch {k} (opened chart not simple / flipped)"
                        branches = sub
                        break
            info["slit_opening"] = oinfo
            info["opened_boundary_simple"] = bool(ok)
            if ok:
                uv, f2 = uv_o, f2_s
                info["new_branches"] = len(branches)
                info["new_seam_length"] = float(sum(np.sum(np.linalg.norm(np.diff(xyz[b], axis=0), axis=1))
                                                    for b in branches))
            else:
                info["new_branches"] = 0
                info["branches_dropped"] = "opened chart has flipped triangles or is not simple"
        print(f"[SG-CONES] cross-field singularities {plan['singularities_raw']} ({plan['pairs_cancelled']} close "
              f"+/- pairs cancelled); new seam branches {info.get('new_branches', 0)} "
              f"(length {info.get('new_seam_length', 0.0):.3f}); seam cones snapped {len(plan['forced'])}", flush=True)
        return info, uv, f2, dict(plan["forced"])
    except Exception as exc:  # noqa: BLE001 - keep the OptCuts seams
        info["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[SG-CONES] skipped: {info['error']}", flush=True)
        return info, uv, f2, {}


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
