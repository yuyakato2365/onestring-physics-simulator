"""M2D by PolySquare-enhanced domain partition (Xiao, Kang, Fu, Chen, CAGD 62 (2018) 29-43),
steps 3.1-3.2.1 of the paper applied to the smoothed chart Omega_1.

1. Boundary-aligned deformation (Sec. 3.1.1): pre-rotation, then
       E_deform = E_iso + lambda E_align,
   E_iso = mean_t exp(1/2 (delta_conf + delta_area))   (isometric AMIPS w.r.t. the 3D triangles),
   E_align = mean_i (b_i . P(G_sigma(n_i)))^2 / l^2     (Gaussian-smoothed boundary normals snapped
   to the closest axis), lambda x10 per round until the vertices stop moving.
2. Pixelation + homotopic morphological simplification (Sec. 3.1.2): square pixels of the panel
   size h, kept when >= 50 % inside; boundary glitches removed while
   E_mor = E_simp (corners) + alpha E_dev (distance of the polysquare boundary to the deformed one)
   decreases and the topology is unchanged.
3. Back projection and quad optimisation (Sec. 3.2.1): polysquare boundary vertices -> closest
   point of the deformed boundary, foldover elimination and conformal AMIPS on the four virtual
   triangles of every quad, re-projection to Omega_1 by barycentric coordinates and the same
   optimisation with the boundary on Omega_1.
Interior vertices of the result all have valence 4 (pixel lattice).  Seams of the cut are not
treated specially (both copies are discretised independently, as in the paper's planar setting).
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ 1. boundary-aligned deformation
def _amips_iso(prob):
    wt = prob.weight
    Dx, Dy = prob.Dx, prob.Dy
    DxT, DyT = Dx.T.tocsr(), Dy.T.tocsr()

    def f(P):
        J = prob.jacobians(P)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        if np.any(det <= 1e-14):
            return np.inf, None
        fro = np.sum(J * J, axis=(1, 2))
        dc = fro / (2.0 * det)
        da = 0.5 * (det + 1.0 / det)
        e = np.exp(np.minimum(0.5 * (dc + da), 60.0))
        E = float(np.dot(wt, e))
        cof = np.empty_like(J)
        cof[:, 0, 0] = J[:, 1, 1]; cof[:, 0, 1] = -J[:, 1, 0]
        cof[:, 1, 0] = -J[:, 0, 1]; cof[:, 1, 1] = J[:, 0, 0]
        ddc = J / det[:, None, None] - (fro / (2.0 * det * det))[:, None, None] * cof
        dda = (0.5 * (1.0 - 1.0 / (det * det)))[:, None, None] * cof
        dE = (wt * e * 0.5)[:, None, None] * (ddc + dda)
        G = np.c_[DxT @ dE[:, 0, 0] + DyT @ dE[:, 0, 1], DxT @ dE[:, 1, 0] + DyT @ dE[:, 1, 1]]
        return E, G
    return f


def _smoothed_axes(P, loop, sigma):
    """P(G_sigma(n_i)) for every boundary edge i (closest axis of the smoothed normal)."""
    a, b = P[loop], P[np.roll(loop, -1)]
    e = b - a
    L = np.linalg.norm(e, axis=1)
    n = np.c_[e[:, 1], -e[:, 0]] / np.maximum(L, 1e-300)[:, None]
    s = np.r_[0.0, np.cumsum(L)[:-1]] + 0.5 * L
    per = float(L.sum())
    d = np.abs(s[:, None] - s[None, :])
    d = np.minimum(d, per - d)
    W = L[None, :] * np.exp(-d * d / (2.0 * sigma * sigma))
    g = W @ n
    ax = np.zeros_like(g)
    big = np.abs(g[:, 0]) >= np.abs(g[:, 1])
    ax[big, 0] = np.sign(g[big, 0]); ax[~big, 1] = np.sign(g[~big, 1])
    return ax, float(np.mean(L))


def boundary_aligned_deformation(prob, uv0, loop, sigma, rounds: int = 6, lam0: float = 1.0, iters: int = 300):
    from .zhang2021_panel_optimizer import _lbfgs
    loop = np.asarray(loop, int)
    P = np.asarray(uv0, float).copy()
    # pre-rotation (Eq. 8): the global rotation that brings the boundary normals closest to the axes
    e = P[np.roll(loop, -1)] - P[loop]
    L = np.linalg.norm(e, axis=1)
    th = float(np.angle(np.sum(L * np.exp(4j * np.arctan2(e[:, 1], e[:, 0])))) / 4.0)
    R = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    P = (P - P.mean(0)) @ R.T
    iso = _amips_iso(prob)
    nV = len(P)
    lam = lam0
    diag = float(np.linalg.norm(np.ptp(P, axis=0)))
    log = []
    for r in range(rounds):
        ax, lbar = _smoothed_axes(P, loop, sigma)
        i0, i1 = loop, np.roll(loop, -1)
        nb = len(loop)

        def fun(z):
            Q = z.reshape(nV, 2)
            Ei, Gi = iso(Q)
            if not np.isfinite(Ei):
                return np.inf, None
            b = Q[i1] - Q[i0]
            dot = np.sum(b * ax, axis=1)
            Ea = float(np.sum(dot * dot)) / (nb * lbar * lbar)
            gb = (2.0 * dot / (nb * lbar * lbar))[:, None] * ax
            G = Gi.copy()
            np.add.at(G, i1, lam * gb)
            np.add.at(G, i0, -lam * gb)
            return Ei + lam * Ea, G.ravel()

        z, E, it = _lbfgs(fun, P.ravel(), max_iter=iters)
        Pn = z.reshape(nV, 2)
        move = float(np.max(np.linalg.norm(Pn - P, axis=1)))
        P = Pn
        log.append({"round": r, "lambda": lam, "energy": float(E), "iterations": int(it), "max_move": move})
        if move < 1e-4 * diag:
            break
        lam *= 10.0
    return P, {"pre_rotation_deg": float(np.degrees(th)), "rounds": log}


# ------------------------------------------------------------------ 2. pixelation + morphology
def _boundary_dist(Bpoly, pts):
    """Distance of points to a closed polyline."""
    a = Bpoly
    b = np.roll(Bpoly, -1, axis=0)
    ab = b - a
    L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
    out = np.empty(len(pts))
    for k in range(0, len(pts), 512):
        p = pts[k:k + 512][:, None, :]
        t = np.clip(np.sum((p - a) * ab, axis=2) / L2, 0, 1)
        q = a + t[..., None] * ab
        out[k:k + 512] = np.sqrt(np.min(np.sum((p - q) ** 2, axis=2), axis=1))
    return out


def _corners(mask):
    M = np.pad(mask, 1).astype(int)
    q = M[:-1, :-1] + M[1:, :-1] + M[:-1, 1:] + M[1:, 1:]
    diag = (M[:-1, :-1] == M[1:, 1:]) & (M[1:, :-1] == M[:-1, 1:]) & (M[:-1, :-1] != M[1:, :-1])
    return (q == 1) | (q == 3) | diag


def _topology(mask):
    from scipy import ndimage
    fg = ndimage.label(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])[1]
    bg = ndimage.label(~np.pad(mask, 1), structure=np.ones((3, 3)))[1]
    return fg, bg


def pixelate(P, f2, loop, h, threshold: float = 0.5, alpha: float = 2.0, windows=(3, 2, 1)):
    from .optcuts_area_grid_boundary import _intersection_ratios
    Bp = P[np.asarray(loop, int)]
    best = None
    for ox in (0.0, 0.25, 0.5, 0.75):              # grid phase that follows the polysquare sides best
        for oy in (0.0, 0.25, 0.5, 0.75):
            org = P.min(axis=0) - h * np.array([1.0 - ox, 1.0 - oy])
            nx_, ny_ = (np.ceil((P.max(axis=0) - org) / h).astype(int) + 1)
            cv = _intersection_ratios(P, f2, org, np.array([h, h]), int(nx_), int(ny_))
            m_ = cv >= threshold
            gy_, gx_ = np.mgrid[0:ny_ + 1, 0:nx_ + 1]
            D_ = _boundary_dist(Bp, org + h * np.c_[gx_.ravel(), gy_.ravel()]).reshape(ny_ + 1, nx_ + 1) / h
            Mp_ = np.pad(m_, 1).astype(int)
            q_ = Mp_[:-1, :-1] + Mp_[1:, :-1] + Mp_[:-1, 1:] + Mp_[1:, 1:]
            dev = float(D_[(q_ > 0) & (q_ < 4)].mean())
            if best is None or dev < best[0]:
                best = (dev, org, int(nx_), int(ny_), cv, D_)
    _dev, origin, nx, ny, cov, D = best
    mask = cov >= threshold

    def energy(m):
        c = _corners(m)
        Mp = np.pad(m, 1).astype(int)
        q = Mp[:-1, :-1] + Mp[1:, :-1] + Mp[:-1, 1:] + Mp[1:, 1:]
        bnd = (q > 0) & (q < 4)
        return float(c.sum()) + alpha * float(D[bnd].sum())

    topo0 = _topology(mask)
    E = energy(mask)
    start = energy(mask)
    changed = True
    passes = 0
    while changed and passes < 20:
        changed = False
        passes += 1
        for w in windows:
            Mp = np.pad(mask, 1).astype(int)
            q = Mp[:-1, :-1] + Mp[1:, :-1] + Mp[:-1, 1:] + Mp[1:, 1:]
            near = np.argwhere((q > 0) & (q < 4))
            for (j, i) in near:
                for val in (False, True):           # opening (remove) / closing (fill) of a w x w block
                    j0, i0 = max(0, j - w // 2), max(0, i - w // 2)
                    j1, i1 = min(ny, j0 + w), min(nx, i0 + w)
                    blk = mask[j0:j1, i0:i1]
                    if blk.size == 0 or np.all(blk == val):
                        continue
                    trial = mask.copy()
                    trial[j0:j1, i0:i1] = val
                    Et = energy(trial)
                    if Et < E - 1e-9 and _topology(trial) == topo0:
                        mask, E, changed = trial, Et, True
    return mask, {"origin": origin, "nx": int(nx), "ny": int(ny), "cells_initial": int((cov >= threshold).sum()),
                  "cells": int(mask.sum()), "corners_initial": int(_corners(cov >= threshold).sum()),
                  "corners": int(_corners(mask).sum()), "E_mor_before": start, "E_mor_after": E, "passes": passes}


# ------------------------------------------------------------------ 3. quad mesh, back projection, optimisation
def _pixel_quads(mask, origin, h, P, f2, split: bool = True):
    """Quads of the kept pixels; lattice vertices shared only by pixels on the same side of the
    deformed boundary (a cut slit separates them)."""
    from matplotlib.path import Path
    ny, nx = mask.shape
    kept = {(j, i) for j, i in np.argwhere(mask)}
    # pixels adjacent across a lattice edge are connected unless the segment between their
    # centres leaves the domain (i.e. crosses a slit)
    from matplotlib.tri import Triangulation
    finder = Triangulation(P[:, 0], P[:, 1], f2).get_trifinder()

    def inside(pt):
        return finder(pt[0], pt[1]) >= 0

    def connected(a, b):
        if not split:
            return True
        ca = origin + h * (np.array([a[1], a[0]]) + 0.5)
        cb = origin + h * (np.array([b[1], b[0]]) + 0.5)
        return all(inside(ca + t * (cb - ca)) for t in (0.25, 0.5, 0.75))

    vid = {}
    verts = []

    def vertex(j, i, pix):
        # pixels around lattice vertex (j, i): (j-1,i-1) (j-1,i) (j,i) (j,i-1) in cyclic order
        ring = [(j - 1, i - 1), (j - 1, i), (j, i), (j, i - 1)]
        k = ring.index(pix)
        group = {pix}
        for step in (1, -1):
            cur = pix
            for s in range(1, 4):
                nxt = ring[(k + step * s) % 4]
                if nxt not in kept or not connected(cur, nxt):
                    break
                group.add(nxt)
                cur = nxt
        key = (j, i, min(group))
        if key not in vid:
            vid[key] = len(verts)
            verts.append(origin + h * np.array([i, j], float))
        return vid[key]

    quads = []
    for (j, i) in sorted(kept):
        quads.append([vertex(j, i, (j, i)), vertex(j, i + 1, (j, i)), vertex(j + 1, i + 1, (j, i)), vertex(j + 1, i, (j, i))])
    lattice = {}
    for (j, i, g), v in vid.items():
        lattice.setdefault((i, j), []).append(v)
    return np.asarray(verts, float), np.asarray(quads, int), lattice


def _monotone_boundary(Y, Q, Bpoly):
    """Put the quad boundary vertices on the closed polyline Bpoly in the same cyclic order
    (isotonic regression of their arc parameters), as the seam/boundary bookkeeping needs."""
    from .zhang2021_onestring_boundary import oriented_boundary_loops
    tri = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    bl = list(oriented_boundary_loops(tri)[0])
    a = Bpoly
    b = np.roll(Bpoly, -1, axis=0)
    ab = b - a
    L = np.linalg.norm(ab, axis=1)
    cum = np.r_[0.0, np.cumsum(L)]
    per = float(cum[-1])
    L2 = np.maximum(L * L, 1e-300)

    def param(p):
        t = np.clip(np.sum((p - a) * ab, axis=1) / L2, 0, 1)
        k = int(np.argmin(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))
        return cum[k] + t[k] * L[k]

    s = np.array([param(Y[v]) for v in bl])
    ds = np.mod(np.diff(np.r_[s, s[0]]), per)
    ds = np.where(ds > 0.5 * per, ds - per, ds)
    k0 = int(np.argmax(ds))                    # start after the largest forward gap
    order = [(k0 + 1 + i) % len(bl) for i in range(len(bl))]
    t = np.r_[0.0, np.cumsum(ds[order][:-1])]
    # pool-adjacent-violators: nondecreasing fit
    vals, wts = [], []
    for x in t:
        vals.append(x); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            w = wts[-2] + wts[-1]
            vals[-2:] = [v]; wts[-2:] = [w]
    fit = np.concatenate([[v] * int(w) for v, w in zip(vals, wts)])
    span = max(float(fit[-1] - fit[0]), 1e-12)
    eps = 1e-6 * per
    fit = fit + eps * np.arange(len(fit))
    if fit[-1] - fit[0] >= per - eps:
        fit = fit[0] + (fit - fit[0]) * (per - 2 * eps * len(fit)) / (fit[-1] - fit[0])
    del span
    Z = Y.copy()
    s0 = s[order[0]]
    for i, k in enumerate(order):
        u = (s0 + fit[i]) % per
        j = int(np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(L) - 1))
        Z[bl[k]] = a[j] + ((u - cum[j]) / max(L[j], 1e-300)) * ab[j]
    return Z


def _corner_tris(Q):
    return np.concatenate([Q[:, [0, 1, 3]], Q[:, [1, 2, 0]], Q[:, [2, 3, 1]], Q[:, [3, 0, 2]]])


def _quad_boundary(Q):
    from collections import Counter
    e = Counter(tuple(sorted((int(q[k]), int(q[(k + 1) % 4])))) for q in Q for k in range(4))
    return sorted({v for k, c in e.items() if c == 1 for v in k})


def _project_to_boundary(X, Q, bverts, Bpoly):
    """Closest point of the closed polyline Bpoly for each boundary vertex, on the side of the
    vertex's own quads (slit walls are close to each other)."""
    a = Bpoly
    b = np.roll(Bpoly, -1, axis=0)
    ab = b - a
    L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
    nout = np.c_[ab[:, 1], -ab[:, 0]]
    cent = {}
    for qi, q in enumerate(Q):
        c = X[q].mean(axis=0)
        for v in q:
            cent.setdefault(int(v), []).append(c)
    Y = X.copy()
    for v in bverts:
        p = X[v]
        t = np.clip(np.sum((p - a) * ab, axis=1) / L2, 0, 1)
        qpt = a + t[:, None] * ab
        d = np.linalg.norm(qpt - p, axis=1)
        c = np.mean(cent[v], axis=0)
        ok = np.sum((c - qpt) * nout, axis=1) < 0          # own quads on the inner side
        if np.any(ok):
            d = np.where(ok, d, np.inf)
        Y[v] = qpt[int(np.argmin(d))]
    return Y


def _conf_amips(X, Q, free, iters=400):
    """Conformal AMIPS on the four virtual right-isosceles triangles of every quad (Eq. 12)."""
    from .zhang2021_panel_optimizer import _lbfgs
    C = _corner_tris(Q)

    def fun(z):
        P = X.copy()
        P[free] = z.reshape(-1, 2)
        a = P[C[:, 1]] - P[C[:, 0]]
        b = P[C[:, 2]] - P[C[:, 0]]
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if np.any(det <= 1e-14):
            return np.inf, None
        fro = np.sum(a * a, 1) + np.sum(b * b, 1)
        dc = fro / (2.0 * det)
        e = np.exp(np.minimum(dc, 60.0))
        E = float(np.mean(e))
        ddet_a = np.c_[b[:, 1], -b[:, 0]]
        ddet_b = np.c_[-a[:, 1], a[:, 0]]
        k = e / len(C)
        ga = k[:, None] * (a / det[:, None] - (fro / (2 * det * det))[:, None] * ddet_a)
        gb = k[:, None] * (b / det[:, None] - (fro / (2 * det * det))[:, None] * ddet_b)
        G = np.zeros_like(P)
        np.add.at(G, C[:, 1], ga)
        np.add.at(G, C[:, 2], gb)
        np.add.at(G, C[:, 0], -ga - gb)
        return E, G[free].ravel()

    E0, _ = fun(X[free].ravel())
    if not np.isfinite(E0):
        return X, False
    z, _E, _it = _lbfgs(fun, X[free].ravel(), max_iter=iters)
    Y = X.copy()
    Y[free] = z.reshape(-1, 2)
    return Y, True


def _optimise_quads(X, Q, Bpoly, rounds: int = 4):
    """Foldover elimination + conformal AMIPS with the boundary vertices on Bpoly (Sec. 3.2.1)."""
    from .structured_grid_20261028 import _untangle_elliptic
    bverts = _quad_boundary(Q)
    free = np.setdiff1d(np.arange(len(X)), bverts)
    C = _corner_tris(Q)
    ref = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (len(C), 1, 1))
    ok = False
    for _ in range(rounds):
        X = _project_to_boundary(X, Q, bverts, Bpoly)
        X = _untangle_elliptic(X, C, free, ref, outer=30, iters=500)
        X, ok = _conf_amips(X, Q, free)
    return X, ok


def _chords(Q):
    from collections import defaultdict
    e2q = defaultdict(list)
    for qi, q in enumerate(Q):
        for k in range(4):
            e2q[tuple(sorted((int(q[k]), int(q[(k + 1) % 4]))))].append((qi, k))
    seen = set()
    out = []
    for e in e2q:
        if e in seen:
            continue
        edges, quads, stack = [], set(), [e]
        seen.add(e)
        while stack:
            cur = stack.pop()
            edges.append(cur)
            for qi, k in e2q[cur]:
                quads.add(qi)
                q = Q[qi]
                opp = tuple(sorted((int(q[(k + 2) % 4]), int(q[(k + 3) % 4]))))
                if opp not in seen:
                    seen.add(opp)
                    stack.append(opp)
        out.append((edges, sorted(quads)))
    return out


def chord_collapse(X, Q, target: int, bset: set, max_steps: int = 2000):
    """Boundary-preserving poly-chord collapse (Daniels et al. 2008, Sec. 3.2.2 of the paper):
    repeatedly remove the narrowest chord (strip of quads) until about ``target`` quads remain.
    A pixel mesh has no interior singularity, so every collapse keeps interior valence 4."""
    from .zhang2021_onestring_boundary import oriented_boundary_loops
    X = np.asarray(X, float).copy()
    Q = np.asarray(Q, int).copy()
    bset = set(int(v) for v in bset)
    steps = 0
    while len(Q) > target and steps < max_steps:
        steps += 1
        cands = []
        for edges, quads in _chords(Q):
            if len(quads) == 0 or len(Q) - len(quads) < target * 0.97:
                continue
            w = float(np.mean([np.linalg.norm(X[a] - X[b]) for a, b in edges]))
            es = set(edges)
            other = [np.linalg.norm(X[Q[qi][k]] - X[Q[qi][(k + 1) % 4]]) for qi in quads for k in range(4)
                     if tuple(sorted((int(Q[qi][k]), int(Q[qi][(k + 1) % 4])))) not in es]
            l = float(np.mean(other)) if other else w
            cands.append((w / max(l, 1e-300), edges, quads))     # thinnest strips first: keeps quads square
        if not cands:
            break
        cands.sort(key=lambda c: c[0])
        done = False
        for w, edges, quads in cands[:20]:
            parent = list(range(len(X)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            for a, b in edges:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            rm = set(quads)
            keep = [q for i, q in enumerate(Q) if i not in rm]
            Qn = np.asarray([[find(int(v)) for v in q] for q in keep], int)
            if len(Qn) == 0 or any(len(set(q)) < 4 for q in Qn):
                continue
            tri = np.concatenate([Qn[:, [0, 1, 2]], Qn[:, [0, 2, 3]]])
            try:
                if len(oriented_boundary_loops(tri)) != 1:
                    continue
            except Exception:  # noqa: BLE001
                continue
            groups = {}
            for v in range(len(X)):
                groups.setdefault(find(v), []).append(v)
            Xn = X.copy()
            for r, mem in groups.items():
                if len(mem) > 1:
                    bm = [m for m in mem if m in bset]
                    Xn[r] = X[bm].mean(axis=0) if bm else X[mem].mean(axis=0)
                    if bm:
                        bset.add(r)
            used = np.unique(Qn)
            remap = -np.ones(len(X), int)
            remap[used] = np.arange(len(used))
            X, Q = Xn[used], remap[Qn]
            bset = {int(remap[v]) for v in bset if remap[v] >= 0}
            done = True
            break
        if not done:
            break
    return X, Q, steps


def pedp_m2d(prob, uvm, f2, loop, h: float, *, sigma_ratio: float = 1.0, alpha: float = 2.0,
             threshold: float = 0.5, pixel_ratio: float = 0.5, target: int | None = None):
    """Omega_1 (uvm, CCW) -> (M2D vertices in Omega_1, quads, lattice index, info)."""
    loop = np.asarray(loop, int)
    Pd, dinfo = boundary_aligned_deformation(prob, uvm, loop, sigma=sigma_ratio * h)
    eta = float(pixel_ratio) * h                     # fine pixels; chords are collapsed afterwards
    mask, pinfo = pixelate(Pd, f2, loop, eta, threshold=threshold, alpha=alpha)
    # a pixel whose centre lies outside the deformed domain would bridge a narrow inlet
    from matplotlib.tri import Triangulation
    from scipy import ndimage
    fd = Triangulation(Pd[:, 0], Pd[:, 1], f2).get_trifinder()
    jj, ii = np.nonzero(mask)
    cen = pinfo["origin"] + eta * (np.c_[ii, jj] + 0.5)
    out = fd(cen[:, 0], cen[:, 1]) < 0
    trial = mask.copy()
    trial[jj[out], ii[out]] = False
    lab, ncomp = ndimage.label(trial, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if ncomp >= 1:
        sizes = ndimage.sum(trial, lab, range(1, ncomp + 1))
        trial = lab == (1 + int(np.argmax(sizes))) if ncomp > 1 else trial
    pinfo["removed_outside_pixels"] = int(mask.sum() - trial.sum())
    mask = trial
    X, Q, lattice = _pixel_quads(mask, pinfo["origin"], eta, Pd, f2)
    from .zhang2021_onestring_boundary import oriented_boundary_loops
    tri = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    if len(oriented_boundary_loops(tri)) != 1:
        # a slit that does not reach the outer boundary would become an inner cut: keep the
        # lattice whole there (the slit is narrower than a pixel)
        X, Q, lattice = _pixel_quads(mask, pinfo["origin"], eta, Pd, f2, split=False)
    # back projection onto the deformed domain, optimisation there
    X, ok1 = _optimise_quads(X, Q, Pd[loop])
    # re-projection to Omega_1 by barycentric coordinates in the deformed mesh, optimisation again
    from .structured_grid_20261028 import _locate_bary
    from matplotlib.tri import Triangulation
    finder = Triangulation(Pd[:, 0], Pd[:, 1], f2).get_trifinder()
    t = finder(X[:, 0], X[:, 1])
    Y = np.empty_like(X)
    inside = t >= 0
    if np.any(inside):
        Y[inside] = _locate_bary(Pd, f2, X[inside], uvm)
    if np.any(~inside):                                  # on the boundary: same edge fraction on Omega_1
        a, b = Pd[loop], Pd[np.roll(loop, -1)]
        ab = b - a
        L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
        for v in np.flatnonzero(~inside):
            tt = np.clip(np.sum((X[v] - a) * ab, axis=1) / L2, 0, 1)
            k = int(np.argmin(np.linalg.norm(a + tt[:, None] * ab - X[v], axis=1)))
            Y[v] = uvm[loop[k]] + tt[k] * (uvm[loop[(k + 1) % len(loop)]] - uvm[loop[k]])
    Y, ok2 = _optimise_quads(Y, Q, uvm[loop])
    n_fine = len(Q)
    steps = 0
    if target and len(Q) > target:
        Y, Q, steps = chord_collapse(Y, Q, int(target), set(_quad_boundary(Q)))
        lattice = {}
        Y, ok2 = _optimise_quads(Y, Q, uvm[loop], rounds=3)
    Y = _monotone_boundary(Y, Q, uvm[loop])
    from .structured_grid_20261028 import _untangle_elliptic
    bv = _quad_boundary(Q)
    free = np.setdiff1d(np.arange(len(Y)), bv)
    C = _corner_tris(Q)
    Y = _untangle_elliptic(Y, C, free, np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (len(C), 1, 1)),
                           outer=30, iters=500)
    Y, ok2 = _conf_amips(Y, Q, free)
    from collections import Counter
    bnd = set(_quad_boundary(Q))
    val = Counter(int(v) for q in Q for v in q)
    irr = sum(1 for v, c in val.items() if v not in bnd and c != 4)
    info = {"placement": "PolySquare-enhanced domain partition (Xiao et al. 2018): boundary-aligned "
                         "deformation, pixelation + morphology, back projection, foldover elimination, "
                         "conformal AMIPS", "deformation": dinfo,
            "pixelation": {k: v for k, v in pinfo.items() if k != "origin"},
            "quads": int(len(Q)), "fine_quads": int(n_fine), "chord_collapses": int(steps),
            "interior_irregular_vertices": irr,
            "optimised_on_deformed": ok1, "optimised_on_omega1": ok2}
    pedp_m2d.last = {"deformed": Pd, "mask": mask, "origin": pinfo["origin"], "X_deformed": X, "M2D": Y,
                     "quads": Q, "uvm": uvm, "f2": f2, "loop": loop, "h": h}      # diagnostics only
    return Y, Q, lattice, info


__all__ = ["boundary_aligned_deformation", "pixelate", "pedp_m2d"]
