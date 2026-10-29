"""Crease lines at panel scale, and the constraints that put grid lines on them (2026-10-29).

A K3D panel collapses when the surface bends strongly *inside* one panel: the two halves of the
panel want different planes and only a degenerate quad fits both.  If instead a grid line runs
along the bend, the bend becomes a hinge between two flat panels.  Nothing here is specific to a
shape; every threshold is a bend angle per panel (curvature x panel edge h).

Crease extraction (Ohtake, Belyaev, Seidel 2004 "Ridge-valley lines on meshes via implicit surface
fitting", simplified to a per-vertex quadric fit):

1. principal curvatures and directions from a weighted quadric fit over a disc of radius ``r`` (half
   a panel by default: a panel cannot see finer bends);
2. crest value  s = -k_min (valleys) or k_max (ridges), cross direction = its principal direction;
3. extremality  e = ds/dc  (derivative of s along its own direction); crest points are the zero
   crossings of e on mesh edges where s is a maximum (marching triangles on e = 0);
4. hysteresis on the bend per panel  s*h: points above ``bend_lo`` form lines, a line is kept if it
   reaches ``bend_hi`` somewhere and is at least ``min_length`` panels long; the crease must also be
   anisotropic (cross bend clearly larger than the bend along it);
5. small gaps are bridged when the directions agree, the lines are smoothed and resampled.

Grid constraints (``crease_rows``): every crease sample, located in a chart triangle, becomes a row
``sum_k beta_k x_ax(v_k) = c``: the map coordinate across the crease equals a lattice line ``c``.
The axis follows the heading of the crease in the unconstrained map; a crease that turns by more
than 45 degrees becomes a staircase of axis-parallel pieces.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ geometry helpers
def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
    # outward (most normals point away from the centroid); only the valley / ridge labels depend on it
    if np.mean(np.einsum("ij,ij->i", V - V.mean(axis=0), N)) < 0:
        N = -N
    return N


def principal_curvature_quadric(V: np.ndarray, F: np.ndarray, radius: float):
    """k_max >= k_min (convex positive w.r.t. the outward normal) and their unit 3D directions,
    from a Gaussian-weighted quadric fit z = a x^2 + b xy + c y^2 + d x + e y over the neighbours
    within ``radius`` whose normals agree (other sheets of the surface are ignored)."""
    from scipy.spatial import cKDTree
    N = vertex_normals(V, F)
    tree = cKDTree(V)
    nV = len(V)
    kmax = np.zeros(nV); kmin = np.zeros(nV)
    dmax = np.zeros((nV, 3)); dmin = np.zeros((nV, 3))
    nbrs = tree.query_ball_point(V, radius)
    sig2 = (0.6 * radius) ** 2
    for i in range(nV):
        idx = np.asarray(nbrs[i], int)
        n = N[i]
        for _pass in range(2):
            ok = idx[(N[idx] @ n) > 0.3]
            if len(ok) < 7:
                ok = np.asarray(tree.query(V[i], k=min(12, nV))[1], int)
            t1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            P = V[ok] - V[i]
            x, y, z = P @ t1, P @ t2, P @ n
            w = np.exp(-(x * x + y * y) / sig2)
            A = np.c_[x * x, x * y, y * y, x, y] * w[:, None]
            coef = np.linalg.lstsq(A, z * w, rcond=None)[0]
            a, b, c, d, e = coef
            if _pass == 0:
                n2 = n - d * t1 - e * t2          # tilt the frame to the fitted tangent plane
                n = n2 / np.linalg.norm(n2)
        g = np.sqrt(1.0 + d * d + e * e)
        # second fundamental form / first fundamental form of the graph at the origin
        II = np.array([[2 * a, b], [b, 2 * c]]) / g
        I = np.array([[1 + d * d, d * e], [d * e, 1 + e * e]])
        S = np.linalg.solve(I, II)
        ev, evec = np.linalg.eig(S)
        ev = np.real(ev); evec = np.real(evec)
        order = np.argsort(ev)
        # z grows along +n: a convex bump (z < 0 around) has negative ev -> flip the sign
        k_lo, k_hi = -ev[order[1]], -ev[order[0]]
        v_lo, v_hi = evec[:, order[1]], evec[:, order[0]]
        kmin[i], kmax[i] = k_lo, k_hi
        for vec, out in ((v_lo, dmin), (v_hi, dmax)):
            d3 = vec[0] * t1 + vec[1] * t2
            d3 -= (d3 @ N[i]) * N[i]
            out[i] = d3 / max(np.linalg.norm(d3), 1e-300)
    return kmax, kmin, dmax, dmin, N


def _sample(tree, V, N, vals, P, nrm, k=6):
    """inverse-distance interpolation of a vertex field at points P (neighbours on the same sheet)."""
    d, j = tree.query(P, k=k)
    same = np.einsum("ijk,ik->ij", N[j], nrm) > 0.3
    w = np.where(same, 1.0 / np.maximum(d, 1e-9) ** 2, 0.0)
    w[np.all(~same, axis=1), 0] = 1.0
    return np.sum(w * vals[j], axis=1) / np.sum(w, axis=1)


# ------------------------------------------------------------------ bend per panel
def panel_bend_field(V, F, h: float, n_dirs: int = 12, normal_radius_ratio: float = 0.25):
    """Signed bend of the surface across one panel, for every vertex.

    For a tangent direction d the bend is the signed angle between the (locally averaged) normals at
    p - h/2 d and p + h/2 d: what a panel of edge h centred at p would have to fold.  Positive =
    convex.  Returns (bend (nV, n_dirs) radians, directions (nV, n_dirs, 3), smoothed normals)."""
    from scipy.spatial import cKDTree
    V = np.asarray(V, float); F = np.asarray(F, int)
    N0 = vertex_normals(V, F)
    tree = cKDTree(V)
    # normals averaged over a small disc on the same sheet (sub-panel noise does not count)
    nb = tree.query_ball_point(V, normal_radius_ratio * h)
    Ns = np.zeros_like(N0)
    for i, idx in enumerate(nb):
        idx = np.asarray(idx, int)
        idx = idx[(N0[idx] @ N0[i]) > 0.3]
        m = N0[idx].sum(axis=0)
        Ns[i] = m / max(np.linalg.norm(m), 1e-300)
    t1 = np.cross(Ns, np.where(np.abs(Ns[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(Ns, t1)
    th = np.pi * np.arange(n_dirs) / n_dirs
    D = np.cos(th)[None, :, None] * t1[:, None, :] + np.sin(th)[None, :, None] * t2[:, None, :]
    bend = np.zeros((len(V), n_dirs))

    def normal_at(P, ref):
        d, j = tree.query(P, k=4)
        same = np.einsum("ijk,ik->ij", Ns[j], ref) > 0.0
        w = np.where(same, 1.0 / np.maximum(d, 1e-9), 0.0)
        w[np.all(~same, axis=1), 0] = 1.0
        n = np.einsum("ij,ijk->ik", w, Ns[j])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    for k in range(n_dirs):
        d = D[:, k]
        npl = normal_at(V + 0.5 * h * d, Ns)
        nmi = normal_at(V - 0.5 * h * d, Ns)
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", npl, nmi), -1.0, 1.0))
        sgn = np.sign(np.einsum("ij,ij->i", npl - nmi, d))
        bend[:, k] = sgn * ang
    return bend, D, Ns


def panel_principal(field):
    """Panel-scale principal bends from the bend field: bend(theta) ~ c0 + c2 cos 2t + s2 sin 2t.
    Returns b1 >= b2 (signed bend per panel, radians) and unit 3D directions d1 (of b1), d2."""
    bend, D, Ns = field
    nd = bend.shape[1]
    th = np.pi * np.arange(nd) / nd
    c0 = bend.mean(axis=1)
    c2 = 2.0 * (bend * np.cos(2 * th)).mean(axis=1)
    s2 = 2.0 * (bend * np.sin(2 * th)).mean(axis=1)
    amp = np.sqrt(c2 * c2 + s2 * s2)
    t1 = 0.5 * np.arctan2(s2, c2)
    t1v, t2v = D[:, 0], D[:, nd // 2]                  # theta = 0 and 90 degrees in each vertex frame
    d1 = np.cos(t1)[:, None] * t1v + np.sin(t1)[:, None] * t2v
    d2 = np.cross(Ns, d1)
    return c0 + amp, c0 - amp, d1, d2


def face_alignment(xyz, f3, V, field, aniso_lo_deg: float = 10.0, aniso_full_deg: float = 30.0):
    """Per chart triangle: the panel-scale principal direction in the triangle's local frame
    (x along edge 0->1, y = n x x, as in _SDProblem) and a weight in [0, 1] that grows with the
    anisotropy (difference of the principal bends per panel)."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, float); f3 = np.asarray(f3, int)
    b1, b2, d1, _d2 = panel_principal(field)
    j = cKDTree(np.asarray(V, float)).query(xyz)[1]
    an = np.degrees(b1 - b2)[j]
    dv = d1[j]
    D3 = dv[f3]                                         # (nT, 3, 3)
    sgn = np.where(np.einsum("ij,ikj->ik", D3[:, 0], D3) >= 0, 1.0, -1.0)
    dm = np.einsum("ik,ikj->ij", sgn, D3)
    X = xyz[f3]
    e1 = X[:, 1] - X[:, 0]
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    n = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ey = np.cross(n, e1)
    loc = np.c_[np.einsum("ij,ij->i", dm, e1), np.einsum("ij,ij->i", dm, ey)]
    loc /= np.maximum(np.linalg.norm(loc, axis=1, keepdims=True), 1e-300)
    a = an[f3].mean(axis=1)
    w = np.clip((a - aniso_lo_deg) / max(aniso_full_deg - aniso_lo_deg, 1e-9), 0.0, 1.0)
    return loc, w, a


# ------------------------------------------------------------------ crest points and lines
def crease_lines(V, F, h: float, kinds: str = "valley", bend_hi_deg: float = 35.0, bend_lo_deg: float = 20.0,
                 anisotropy: float = 1.5, min_length_ratio: float = 2.0, gap_ratio: float = 0.75,
                 spacing_ratio: float = 0.25, field=None) -> dict:
    """Panel-scale crease polylines of the surface (V, F).

    kinds: "valley" (concave folds such as the base of a neck or an ear), "ridge" (convex folds) or
    "both".  Thresholds are bend angles per panel (degrees), lengths are in panels.
    Returns {"lines": [{"kind", "xyz" (n,3), "bend" (n,) deg, "length_panels"}], "stats": ...}."""
    from scipy.spatial import cKDTree
    V = np.asarray(V, float); F = np.asarray(F, int)
    if field is None:
        field = panel_bend_field(V, F, h)
    bend, D, Ns = field
    n_dirs = bend.shape[1]
    tree = cKDTree(V)
    elen = float(np.mean(np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)))
    delta = max(0.6 * elen, 0.15 * h)
    lo, hi = np.radians(bend_lo_deg), np.radians(bend_hi_deg)
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=0)
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(E)}
    FE = np.array([[eid[tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))] for k in range(3)] for f in F])
    out_lines, stats = [], {}
    todo = {"valley": ["valley"], "ridge": ["ridge"], "both": ["valley", "ridge"]}[kinds]
    for kind in todo:
        sb = -bend if kind == "valley" else bend
        kc = np.argmax(sb, axis=1)
        s = sb[np.arange(len(V)), kc]                          # bend across the crease (>0)
        along = np.abs(bend[np.arange(len(V)), (kc + n_dirs // 2) % n_dirs])
        cdir = D[np.arange(len(V)), kc]
        ok = (s >= lo) & (s >= anisotropy * along)
        sp = _sample(tree, V, Ns, s, V + delta * cdir, Ns)
        sm = _sample(tree, V, Ns, s, V - delta * cdir, Ns)
        e = (sp - sm) / (2 * delta)                            # extremality along the cross direction
        c2 = sp + sm - 2 * s                                   # < 0 at a maximum
        a, b = E[:, 0], E[:, 1]
        sg = np.where(np.einsum("ij,ij->i", cdir[a], cdir[b]) >= 0, 1.0, -1.0)
        ea, eb = e[a], sg * e[b]
        cross = (ea * eb <= 0) & (ea != eb) & (ok[a] | ok[b]) & ((c2[a] < 0) | (c2[b] < 0))
        t = np.where(cross, ea / np.where(ea - eb == 0, 1.0, ea - eb), 0.0)
        sv = (1 - t) * s[a] + t * s[b]
        cross &= (t >= 0) & (t <= 1) & (sv >= lo)
        P_e = (1 - t)[:, None] * V[a] + t[:, None] * V[b]
        on = cross[FE]                                         # (nF, 3)
        seg = []
        for f in np.where(on.sum(axis=1) >= 2)[0]:
            ks = [int(FE[f, k]) for k in range(3) if on[f, k]]
            if len(ks) == 3:
                ks = sorted(ks, key=lambda q: -sv[q])[:2]
            seg.append((ks[0], ks[1]))
        chains = _chains(seg)
        lines = [(P_e[np.asarray(ch)], sv[np.asarray(ch)]) for ch in chains]
        lines = _bridge(lines, gap_ratio * h)
        kept = []
        for P, svv in lines:
            L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
            if L < min_length_ratio * h or svv.max() < hi:
                continue
            P2, sv2 = _smooth_resample(P, svv, spacing_ratio * h)
            kept.append({"kind": kind, "xyz": P2, "bend": np.degrees(sv2), "length_panels": L / h})
        stats[kind] = {"crest_points": int(cross.sum()), "segments": len(seg), "chains": len(chains),
                       "kept": len(kept), "kept_length_panels": float(sum(k_["length_panels"] for k_ in kept))}
        out_lines += kept
    return {"lines": out_lines, "stats": stats, "bend_hi_deg": bend_hi_deg, "bend_lo_deg": bend_lo_deg}


def _chains(seg):
    """split a segment graph into chains between nodes of degree != 2."""
    adj: dict = {}
    for a, b in seg:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used = set()
    chains = []

    def walk(start, nxt):
        ch = [start, nxt]
        used.add(frozenset((start, nxt)))
        while len(adj[ch[-1]]) == 2:
            cand = [q for q in adj[ch[-1]] if frozenset((ch[-1], q)) not in used]
            if not cand:
                break
            used.add(frozenset((ch[-1], cand[0])))
            ch.append(cand[0])
        return ch

    for v in adj:
        if len(adj[v]) != 2:
            for w in adj[v]:
                if frozenset((v, w)) not in used:
                    chains.append(walk(v, w))
    for v in adj:                      # closed loops
        for w in adj[v]:
            if frozenset((v, w)) not in used:
                chains.append(walk(v, w))
    return chains


def _bridge(lines, gap):
    """join chain ends closer than ``gap`` whose end directions agree (< 40 degrees)."""
    lines = [(np.asarray(P), np.asarray(s)) for P, s in lines if len(P) >= 2]
    changed = True
    while changed:
        changed = False
        best = None
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue
                Pi, Pj = lines[i][0], lines[j][0]
                for ei in (0, 1):
                    for ej in (0, 1):
                        a = Pi[-1] if ei else Pi[0]
                        ta = (Pi[-1] - Pi[-2]) if ei else (Pi[0] - Pi[1])
                        b = Pj[-1] if ej else Pj[0]
                        tb = (Pj[-2] - Pj[-1]) if ej else (Pj[1] - Pj[0])
                        d = float(np.linalg.norm(b - a))
                        if d > gap:
                            continue
                        ta = ta / max(np.linalg.norm(ta), 1e-12); tb = tb / max(np.linalg.norm(tb), 1e-12)
                        link = (b - a) / max(d, 1e-12)
                        if ta @ tb < np.cos(np.radians(40)) or (d > 1e-9 and ta @ link < np.cos(np.radians(50))):
                            continue
                        if best is None or d < best[0]:
                            best = (d, i, j, ei, ej)
        if best is not None:
            _d, i, j, ei, ej = best
            Pi, si = lines[i]
            Pj, sj = lines[j]
            if not ei:
                Pi, si = Pi[::-1], si[::-1]
            if ej:
                Pj, sj = Pj[::-1], sj[::-1]
            new = (np.r_[Pi, Pj], np.r_[si, sj])
            lines = [l_ for k, l_ in enumerate(lines) if k not in (i, j)] + [new]
            changed = True
    return lines


def _smooth_resample(P, s, step, iters=6):
    P = P.copy()
    for _ in range(iters):
        P[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = np.r_[0.0, np.cumsum(seg)]
    n = max(2, int(np.ceil(t[-1] / step)) + 1)
    tt = np.linspace(0.0, t[-1], n)
    Q = np.c_[[np.interp(tt, t, P[:, k]) for k in range(3)]].T
    return Q, np.interp(tt, t, s)


# ------------------------------------------------------------------ chart location and grid rows
def _closest_on_triangles(P, A, B, C):
    """closest points of P (n,3) on triangles (n,3) each (Ericson), with barycentrics."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    bary = np.zeros((len(P), 3))
    den = np.where(np.abs(va + vb + vc) > 1e-300, va + vb + vc, 1.0)
    v, w = vb / den, vc / den
    bary[:] = np.c_[1 - v - w, v, w]
    # regions outside the face
    m = (d1 <= 0) & (d2 <= 0); bary[m] = [1, 0, 0]
    m = (d3 >= 0) & (d4 <= d3); bary[m] = [0, 1, 0]
    m = (d6 >= 0) & (d5 <= d6); bary[m] = [0, 0, 1]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = np.where(m, d1 / np.where(d1 - d3 == 0, 1, d1 - d3), 0); bary[m] = np.c_[1 - t, t, 0 * t][m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = np.where(m, d2 / np.where(d2 - d6 == 0, 1, d2 - d6), 0); bary[m] = np.c_[1 - t, 0 * t, t][m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = np.where(m, (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6)), 0)
    bary[m] = np.c_[0 * t, 1 - t, t][m]
    Q = bary[:, :1] * A + bary[:, 1:2] * B + bary[:, 2:] * C
    return Q, bary


def locate_in_chart(points: np.ndarray, xyz: np.ndarray, f3: np.ndarray, k: int = 24):
    """closest surface point: (face id, barycentric, distance) on the chart's 3D triangles
    (candidate triangles by centroid distance; numpy / scipy only)."""
    from scipy.spatial import cKDTree
    X = np.asarray(xyz, float); F = np.asarray(f3, int); P = np.asarray(points, float)
    T = X[F]
    k = min(k, len(F))
    _, cand = cKDTree(T.mean(axis=1)).query(P, k=k)
    cand = np.asarray(cand).reshape(len(P), k)
    best_d = np.full(len(P), np.inf); best_t = np.zeros(len(P), int); best_b = np.zeros((len(P), 3))
    for j in range(k):
        t = cand[:, j]
        Q, b = _closest_on_triangles(P, T[t, 0], T[t, 1], T[t, 2])
        d = np.linalg.norm(Q - P, axis=1)
        better = d < best_d
        best_d[better], best_t[better], best_b[better] = d[better], t[better], b[better]
    return best_t, np.clip(best_b, 0.0, 1.0), best_d


def crease_rows(lines: list, chart, phi0: np.ndarray, h: float, boundary_loop=None, trim_ratio: float = 0.25,
                min_piece_ratio: float = 0.5, snap: str = "nearest", switch_ratio: float = 2.0,
                segmentation: str = "dp", resolve: bool = True):
    """Constraint rows for the map x (2 nV unknowns): sum_k beta_k x_ax(v_k) = c.

    phi0: the unconstrained aligned map (chart vertex -> lattice plane, lattice spacing h); it decides
    the axis of every piece (a crease running along u is put on a line v = c and vice versa) and the
    nearest lattice line c.  Samples closer than ``trim_ratio`` panels to the chart boundary are left
    out (boundary vertices already slide on their own sides).  Returns a dict with sparse-ready rows
    and a per-piece report."""
    uvm, f2, xyz, f3 = chart
    f2 = np.asarray(f2, int)
    nV = len(phi0)
    pieces = []
    bpoly = None
    if boundary_loop is not None:
        bl = np.asarray(boundary_loop, int)
        bpoly = (phi0[bl], phi0[np.roll(bl, -1)])
    for li, ln in enumerate(lines):
        if not isinstance(ln, dict):
            ln = {"xyz": ln, "kind": "line"}
        P = np.asarray(ln["xyz"], float)
        tri, bary, dist = locate_in_chart(P, xyz, f3)
        vv = f2[tri]                                   # chart vertex ids (n, 3)
        X = np.einsum("ij,ijk->ik", bary, phi0[vv])    # map position of every sample
        d3 = np.linalg.norm(np.diff(P, axis=0), axis=1)
        dm = np.linalg.norm(np.diff(X, axis=0), axis=1)
        jump = dm > 3.0 * d3 + 0.5 * h                 # the crease crosses a seam: split there
        cut = np.r_[0, np.where(jump)[0] + 1, len(P)]
        for a, b in zip(cut[:-1], cut[1:]):
            if b - a < 2:
                continue
            idx = np.arange(a, b)
            if bpoly is not None:
                bd = _dist_to_segments(X[idx], bpoly[0], bpoly[1])
                idx = idx[bd >= trim_ratio * h]
            if len(idx) < 2:
                continue
            # split where consecutive kept samples are not neighbours (a trimmed stretch in between)
            brk = np.r_[0, np.where(np.diff(idx) > 1)[0] + 1, len(idx)]
            for a2, b2 in zip(brk[:-1], brk[1:]):
                run = idx[a2:b2]
                if len(run) >= 2:
                    runs_fn = _axis_runs_majority if segmentation == "majority" else _axis_runs
                    pieces += runs_fn(li, ln, run, X, vv, bary, h, min_piece_ratio, snap, switch_ratio)
    if resolve:
        pieces = _resolve_conflicts(pieces, h)
    rows = []
    for pc in pieces:
        for j in range(len(pc["samples"])):
            rows.append((pc["ax"], pc["verts"][j], pc["bary"][j], pc["c"], pc["w"][j]))
    return {"rows": rows, "pieces": pieces}


def _resolve_conflicts(pieces, h):
    """Two pieces of different lines on the same lattice line with overlapping extent cannot both
    be satisfied: the shorter one moves to the neighbouring lattice line on its own side, or is
    dropped when that one is taken too."""
    def ext(pc):
        o = pc["_other"]
        return o.min(), o.max()
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i]["length_panels"])
    kept = []
    for i in order:
        pc = pieces[i]
        cands = [pc["c"], pc["c"] + np.sign(-pc["offset_cells"] or 1.0) * h]
        for c in cands:
            lo, hi = ext(pc)
            clash = any(q["ax"] == pc["ax"] and abs(q["c"] - c) < 0.5 * h and q["line"] != pc["line"]
                        and not (ext(q)[1] < lo - 0.5 * h or ext(q)[0] > hi + 0.5 * h) for q in kept)
            if not clash:
                pc["offset_cells"] = float(pc["offset_cells"] + (c - pc["c"]) / h)
                pc["c"] = float(c)
                kept.append(pc)
                break
    return kept


def _axis_labels(seg, slen, h, switch_ratio):
    """Axis label per segment by dynamic programming: cost of a label = length x sin^2 of the angle
    to that lattice axis, plus ``switch_ratio`` panels of length for every change of axis.  Every
    segment gets a label (no leftovers); a crease only becomes a staircase where it really turns."""
    n = len(seg)
    cu = seg[:, 1] ** 2 / np.maximum(slen, 1e-12)     # along u -> deviation is the v component
    cv = seg[:, 0] ** 2 / np.maximum(slen, 1e-12)
    unary = np.c_[cu, cv]                              # label 0: runs along u, label 1: runs along v
    lam = switch_ratio * h
    cost = unary[0].copy()
    back = np.zeros((n, 2), int)
    for k in range(1, n):
        new = np.empty(2)
        for l_ in (0, 1):
            stay, swap = cost[l_], cost[1 - l_] + lam
            back[k, l_] = l_ if stay <= swap else 1 - l_
            new[l_] = min(stay, swap) + unary[k, l_]
        cost = new
    lab = np.zeros(n, int)
    lab[-1] = int(np.argmin(cost))
    for k in range(n - 1, 0, -1):
        lab[k - 1] = back[k, lab[k]]
    return lab


def _axis_runs(li, ln, run, X, vv, bary, h, min_piece_ratio, snap, switch_ratio=2.0):
    """Split a crease piece into axis-parallel runs by its heading in the map (see _axis_labels)."""
    Y = X[run]
    seg = np.diff(Y, axis=0)
    slen = np.linalg.norm(seg, axis=1)
    lab0 = _axis_labels(seg, slen, h, switch_ratio)
    horiz = lab0 == 0                                  # runs along u -> constrain v (ax 1)
    out = []
    starts = np.r_[0, np.where(np.diff(horiz.astype(int)) != 0)[0] + 1, len(horiz)]
    for a, b in zip(starts[:-1], starts[1:]):
        L = float(np.sum(slen[a:b]))
        if L < min_piece_ratio * h:
            continue
        samp = run[a:b + 1]
        ax = 1 if horiz[a] else 0
        vals = X[samp, ax]
        w = np.r_[slen[a:b], 0.0] * 0.5 + np.r_[0.0, slen[a:b]] * 0.5
        mean = float(np.sum(w * vals) / max(np.sum(w), 1e-12))
        c = h * np.round(mean / h) if snap == "nearest" else mean
        heading = np.degrees(np.arctan2(*(Y[-1] - Y[0])[::-1]))
        out.append({"line": li, "kind": ln["kind"], "ax": ax, "c": float(c), "offset_cells": float((c - mean) / h),
                    "spread_cells": float(np.sqrt(np.sum(w * (vals - mean) ** 2) / max(np.sum(w), 1e-12)) / h),
                    "length_panels": L / h, "samples": samp, "verts": vv[samp], "bary": bary[samp], "w": w / h,
                    "heading_deg": float(heading), "_other": X[samp, 1 - ax]})
    return out


def _axis_runs_majority(li, ln, run, X, vv, bary, h, min_piece_ratio, snap, switch_ratio=2.0):
    """First version (2026-10-29 comparison "valley + fix + orientation"): axis per segment by a
    1-panel majority filter of its heading; runs shorter than ``min_piece_ratio`` are left out."""
    Y = X[run]
    seg = np.diff(Y, axis=0)
    slen = np.linalg.norm(seg, axis=1)
    horiz = np.abs(seg[:, 0]) >= np.abs(seg[:, 1])
    t = np.r_[0.0, np.cumsum(slen)]
    mid = 0.5 * (t[:-1] + t[1:])
    lab = horiz.copy()
    for k in range(len(seg)):
        win = np.abs(mid - mid[k]) <= 0.5 * h
        lab[k] = np.sum(slen[win] * horiz[win]) >= 0.5 * np.sum(slen[win])
    out = []
    starts = np.r_[0, np.where(np.diff(lab.astype(int)) != 0)[0] + 1, len(lab)]
    for a, b in zip(starts[:-1], starts[1:]):
        L = float(np.sum(slen[a:b]))
        if L < min_piece_ratio * h:
            continue
        samp = run[a:b + 1]
        ax = 1 if lab[a] else 0
        vals = X[samp, ax]
        w = np.r_[slen[a:b], 0.0] * 0.5 + np.r_[0.0, slen[a:b]] * 0.5
        mean = float(np.sum(w * vals) / max(np.sum(w), 1e-12))
        c = h * np.round(mean / h) if snap == "nearest" else mean
        heading = np.degrees(np.arctan2(*(Y[-1] - Y[0])[::-1]))
        out.append({"line": li, "kind": ln["kind"], "ax": ax, "c": float(c), "offset_cells": float((c - mean) / h),
                    "spread_cells": float(np.sqrt(np.sum(w * (vals - mean) ** 2) / max(np.sum(w), 1e-12)) / h),
                    "length_panels": L / h, "samples": samp, "verts": vv[samp], "bary": bary[samp], "w": w / h,
                    "heading_deg": float(heading), "_other": X[samp, 1 - ax]})
    return out


def _dist_to_segments(P, A, B):
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-30)
    out = np.full(len(P), np.inf)
    for i in range(len(P)):
        t = np.clip(((P[i] - A) * AB).sum(axis=1) / L2, 0.0, 1.0)
        d = np.linalg.norm(A + t[:, None] * AB - P[i], axis=1)
        out[i] = d.min()
    return out


def rows_to_matrix(rows, nV: int):
    """sparse B (n_rows x 2 nV), targets c, weights w."""
    from scipy import sparse
    I, J, Vv = [], [], []
    c = np.zeros(len(rows)); w = np.zeros(len(rows))
    for r, (ax, verts, bary, cc, ww) in enumerate(rows):
        for k in range(3):
            I.append(r); J.append(ax * nV + int(verts[k])); Vv.append(float(bary[k]))
        c[r] = cc; w[r] = ww
    return sparse.csr_matrix((Vv, (I, J)), shape=(len(rows), 2 * nV)), c, w


def straightness_matrix(pieces, nV: int):
    """Rows (B_{j+1} - B_j) x = 0 for consecutive samples of every axis-parallel piece: the crease is
    a straight lattice-parallel line whose position is still free (used before the corners are
    rounded, when the lattice lines are not known yet)."""
    from scipy import sparse
    I, J, Vv, w = [], [], [], []
    r = 0
    for pc in pieces:
        ax = pc["ax"]
        for j in range(len(pc["samples"]) - 1):
            for verts, bary, sg in ((pc["verts"][j + 1], pc["bary"][j + 1], 1.0), (pc["verts"][j], pc["bary"][j], -1.0)):
                for k in range(3):
                    I.append(r); J.append(ax * nV + int(verts[k])); Vv.append(sg * float(bary[k]))
            w.append(0.5 * (pc["w"][j] + pc["w"][j + 1]))
            r += 1
    return sparse.csr_matrix((Vv, (I, J)), shape=(r, 2 * nV)), np.zeros(r), np.asarray(w, float)


# Omega parameterization modes (UI) -> feature settings.  Every other setting is the 2026-10-28
# structured-grid mode unchanged.
FEATURE_PRESETS = {
    # crease lines of the concave kind only; grid line pulled onto them, those grid vertices fixed
    # in the panel-quality step; lattice also turned toward the principal bend directions
    "valley_fix_align": {
        "structured_grid_feature_mode": "crease+align", "structured_grid_feature_kinds": "valley",
        "structured_grid_feature_crease_hold": "fix", "structured_grid_feature_schedule": "1,3,10,30",
        "structured_grid_feature_min_mean_bend": 0.0, "structured_grid_feature_min_length": 2.0,
        "structured_grid_align_weight": 10.0, "structured_grid_feature_design": False,
        "structured_grid_feature_trim": 0.6, "structured_grid_feature_min_piece": 1.5,
        "structured_grid_feature_segmentation": "majority", "structured_grid_feature_resolve": False,
    },
    # valley and ridge lines that bend strongly as a whole (mean >= 50 deg per panel, >= 3 panels):
    # drawn into Omega like extra boundaries, the grid follows them exactly and matches the boundary;
    # in the panel-quality step their grid vertices may only slide along the lines
    "strong_lines": {
        "structured_grid_feature_mode": "crease", "structured_grid_feature_kinds": "both",
        "structured_grid_feature_crease_hold": "slide", "structured_grid_feature_slide_weight": 300.0,
        "structured_grid_feature_schedule": "1,3,10,30,100,300,1000",
        "structured_grid_feature_min_mean_bend": 50.0, "structured_grid_feature_min_length": 3.0,
        "structured_grid_feature_design": False,
        "structured_grid_feature_trim": 0.25, "structured_grid_feature_min_piece": 0.5,
        "structured_grid_feature_segmentation": "dp", "structured_grid_feature_resolve": True,
    },
}


def apply_feature_preset(params):
    """params with the settings of ``structured_grid_feature_preset`` (off = unchanged)."""
    import copy
    preset = str(getattr(params, "structured_grid_feature_preset", "off") or "off")
    if preset not in FEATURE_PRESETS:
        return params
    out = copy.copy(params)
    for k, v in FEATURE_PRESETS[preset].items():
        setattr(out, k, v)
    return out


def build_features(V, F, h: float, params) -> dict | None:
    """Feature options for the aligned placement from the pipeline parameters (None = off).

    structured_grid_feature_mode: off | crease | align | crease+align
    structured_grid_feature_kinds: valley | ridge | both
    structured_grid_feature_bend_hi / _bend_lo (deg per panel), _min_length (panels),
    structured_grid_feature_schedule ("1,3,10,30"), structured_grid_align_weight,
    structured_grid_align_aniso_lo / _full (deg per panel), structured_grid_feature_design (bool)."""
    mode = str(getattr(params, "structured_grid_feature_mode", "off") or "off")
    if mode == "off":
        return None
    field = panel_bend_field(V, F, h)
    feat = {"mode": mode, "iters": int(getattr(params, "structured_grid_feature_iters", 40)),
            "design": bool(getattr(params, "structured_grid_feature_design", False)),
            "trim": float(getattr(params, "structured_grid_feature_trim", 0.25)),
            "min_piece": float(getattr(params, "structured_grid_feature_min_piece", 0.5)),
            "segmentation": str(getattr(params, "structured_grid_feature_segmentation", "dp")),
            "resolve": bool(getattr(params, "structured_grid_feature_resolve", True)),
            "surface_vertices": np.asarray(V, float), "field": field}
    if "crease" in mode:
        cl = crease_lines(V, F, h, kinds=str(getattr(params, "structured_grid_feature_kinds", "valley")),
                          bend_hi_deg=float(getattr(params, "structured_grid_feature_bend_hi", 35.0)),
                          bend_lo_deg=float(getattr(params, "structured_grid_feature_bend_lo", 20.0)),
                          min_length_ratio=float(getattr(params, "structured_grid_feature_min_length", 2.0)),
                          field=field)
        # keep only lines that bend strongly as a whole: mean bend per panel >= min_mean_bend (deg)
        mm = float(getattr(params, "structured_grid_feature_min_mean_bend", 0.0) or 0.0)
        strong = [ln for ln in cl["lines"] if float(np.mean(ln["bend"])) >= mm]
        weak = [ln for ln in cl["lines"] if float(np.mean(ln["bend"])) < mm]
        feat["lines"] = [ln["xyz"] for ln in strong]
        feat["line_kinds"] = [ln["kind"] for ln in strong]
        feat["dropped_lines"] = [ln["xyz"] for ln in weak]
        feat["dropped_kinds"] = [ln["kind"] for ln in weak]
        feat["line_mean_bend"] = [float(np.mean(ln["bend"])) for ln in strong]
        feat["crease_stats"] = cl["stats"]
        feat["schedule"] = tuple(float(x) for x in str(getattr(params, "structured_grid_feature_schedule", "1,3,10,30")).split(",") if x.strip())
    if "align" in mode:
        feat["align_weight"] = float(getattr(params, "structured_grid_align_weight", 10.0))
        feat["align_aniso_lo"] = float(getattr(params, "structured_grid_align_aniso_lo", 10.0))
        feat["align_aniso_full"] = float(getattr(params, "structured_grid_align_aniso_full", 30.0))
    return feat
