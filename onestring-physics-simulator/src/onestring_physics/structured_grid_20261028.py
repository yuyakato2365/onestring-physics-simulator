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
        a = src.point_at((cum[j] - scale) % P)[2] - src.uv[j]
        b = src.point_at((cum[j] + scale) % P)[2] - src.uv[j]
        interior = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)) % 360.0
        out[j] = 180.0 - interior
    return out


def logical_domain(src, feats: list[int], h3d: float, scale: float) -> dict:
    """Turns (+1 / 0 / -1 quarter turns), directions and integer side lengths of the orthogonal polygon.

    One mixed-integer program chooses everything together:
      * direction one-hot delta[k, d] of every side k (E/N/W/S), consecutive sides turn by -1/0/+1
        quarter turns, total turning +4 (closed CCW polygon), cut tips forced to -1 (reflex corner,
        3 panels around the tip once the cut is closed);
      * integer side lengths n_k >= 1, closure, equal counts on both copies of every cut branch;
      * cost: |n_k - L_k| / L_k (L_k = 3D run length / target panel edge)
              + w_turn |turning angle / 90 deg - turn_k|
              + w_pos |corner_k - chart position of feature k| (shape term, used when needed).
    Self-intersecting answers are excluded by no-good cuts and a stronger shape term.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from shapely.geometry import Polygon
    from .zhang2021_onestring_seam_matching import seam_constraints

    feats = sorted(feats)
    m = len(feats)
    if m < 4 and not any(src.node_kind.get(j) for j in feats):
        raise StructuredGridError(f"SG_TOO_FEW_BOUNDARY_FEATURES: {m} (need >= 4 corners)")
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
    pairs = []
    for con in seam_constraints(src):
        ka, kb = pos.get(con.run_a[0]), pos.get(con.run_b[0])
        if ka is None or kb is None or feats[(ka + 1) % m] != con.run_a[1] or feats[(kb + 1) % m] != con.run_b[1]:
            raise StructuredGridError("SG_CUT_BRANCH_NOT_A_SINGLE_SIDE: a cut branch contains extra features")
        pairs.append((ka, kb))
    # variable layout
    iD = lambda k, d: 4 * k + d                     # noqa: E731  delta (binary)
    iX = lambda k, d: 4 * m + 4 * k + d             # noqa: E731  x = n_k * delta (integer)
    iT = lambda k, s: 8 * m + 3 * k + (s + 1)       # noqa: E731  turn one-hot (binary), s in -1, 0, 1
    iE = lambda k: 11 * m + k                       # noqa: E731  |n_k - L_k|
    iF = lambda k, ax: 12 * m + 2 * k + ax          # noqa: E731  |P_k - T_k|
    nv = 14 * m
    M = int(10 * max(L.max(), 4) + 10)
    w_turn = 1.0
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
        row([(iX(k, d), 1) for d in range(4)], 1, np.inf)
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
    integrality = np.r_[np.ones(11 * m), np.zeros(3 * m)]
    lo_b = np.zeros(nv)
    hi_b = np.r_[np.ones(4 * m), np.full(4 * m, M), np.ones(3 * m), np.full(3 * m, np.inf)]
    attempts = []
    nogood: list[list[int]] = []
    chosen = None
    for w_pos in (0.0, 0.0, 0.0, 0.05, 0.05, 0.2, 0.2, 1.0, 1.0, 5.0):
        c = np.zeros(nv)
        for k in range(m):
            c[iE(k)] = 1.0 / max(L[k], 1.0)
            for s_ in (-1, 0, 1):
                c[iT(k, s_)] = w_turn * abs(ang[feats[k]] / 90.0 - s_)
            c[iF(k, 0)] = c[iF(k, 1)] = w_pos / max(float(np.mean(L)), 1.0)
        A2, lb2, ub2 = list(A), list(lb), list(ub)
        for ng in nogood:
            r = np.zeros(nv); r[ng] = 1
            A2.append(r); lb2.append(-np.inf); ub2.append(m - 1)
        res = milp(c, constraints=LinearConstraint(np.asarray(A2), lb2, ub2), integrality=integrality,
                   bounds=Bounds(lo_b, hi_b), options={"time_limit": 8.0})
        if res.x is None:
            attempts.append({"w_pos": w_pos, "status": f"no solution: {res.message}"})
            continue
        x = res.x
        dirs = [int(np.argmax([x[iD(k, d)] for d in range(4)])) for k in range(m)]
        n = np.rint([sum(x[iX(k, d)] for d in range(4)) for k in range(m)]).astype(int)
        turn = [int(round(sum(s_ * x[iT(k, s_)] for s_ in (-1, 1)))) for k in range(m)]
        pts = [np.zeros(2, int)]
        for k in range(m):
            pts.append(pts[-1] + n[k] * DIRS[dirs[k]])
        poly = np.asarray(pts[:-1])
        pg = Polygon(poly)
        ok = bool(np.all(pts[-1] == 0) and pg.is_valid and pg.area > 0)
        attempts.append({"w_pos": w_pos, "status": "simple" if ok else "self-intersecting",
                         "lengths": n.tolist(), "turns": turn})
        if ok:
            chosen = (w_pos, dirs, n, turn, poly)
            break
        nogood.append([iT(k, turn[k]) for k in range(m)])     # rotation-invariant cut
    if chosen is None:
        raise StructuredGridError("SG_LOGICAL_POLYGON_SELF_INTERSECTS: no simple orthogonal polygon was found for "
                                  f"the boundary features (attempts {attempts})")
    w_pos, dirs, n, turn, poly = chosen
    return {"features": feats, "turn": turn, "dirs": dirs, "lengths": n.tolist(),
            "target_lengths": L.tolist(), "polygon": poly, "cut_pairs": pairs,
            "convex": int(sum(1 for t in turn if t == 1)), "reflex": int(sum(1 for t in turn if t == -1)),
            "shape_weight": w_pos, "attempts": attempts, "turning_deg": [float(ang[j]) for j in feats]}


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


def place_grid(src, dom: dict, logical: np.ndarray, quads: np.ndarray, walk) -> np.ndarray:
    """Boundary grid points by 3D arc length on the smooth chart boundary; interior harmonic."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from .zhang2021_onestring_boundary import _param_from_fraction_3d, run_interval

    feats = dom["features"]
    m = len(feats)
    P = src.perimeter
    uv = np.full((len(logical), 2), np.nan)
    for vtx, k, j in walk:
        a, b = feats[k], feats[(k + 1) % m]
        lo, hi = run_interval(src, (a, b))
        s = _param_from_fraction_3d(src, lo, hi, j / dom["lengths"][k]) % P
        uv[vtx] = src.point_at(s)[2]
    bnd = np.array(sorted({w[0] for w in walk}))
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
    return uv


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
    dom = logical_domain(src, feats, h3d, keep * diag)
    logical, quads, vid, walk = lattice_grid(dom)
    grid_uv = place_grid(src, dom, logical, quads, walk) * mirror
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
                                  panel_weights=PanelOptimizerWeights(
                                      smooth=float(getattr(params, "structured_grid_smoothness_weight", 1.0))),
                                  report_only=True)
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
