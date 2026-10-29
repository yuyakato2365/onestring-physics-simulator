"""OneString extension (NOT Zhang et al. 2021): 3D panel-quality optimisation of M2D.

After the seam-aware boundary correspondence the chart boundary equals the
boundary polygon of the quad mesh, and every boundary vertex is an exact
point of the OptCuts cut / open boundary of S, paired across every cut.
Those vertices stay fixed.  The interior vertices of the quad mesh are moved
inside the final chart so that the *lifted* panels (M3D, exact barycentric
lift through the chart) become uniform in size and close to squares:

    b_q = w_angle * mean_k cos^2(theta_k)                (corner angles, 3D)
        + w_size  * mean_k log^2(l_k / L)                (edge length vs. reference L)
        + w_aspect* mean_k (log l_k - mean_j log l_j)^2  (in-panel aspect)
        + w_sum   * (sum_k theta_k - 2 pi)^2             (non-convex / folded corners)
    E   = mean_q b_q + alpha * LSE_beta(b)  +  barrier(UV corner areas)

L is the geometric-mean edge length of the lifted mesh at the start (so the
optimisation equalises sizes without any panel-count target).  The barrier
keeps every UV quad simple and positively oriented (no flips, the lift stays a
bijection onto S); a tangled start (e.g. after a cut-tip valence repair) is first
untangled.  Topology is never changed here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class PanelOptimizationError(RuntimeError):
    pass


@dataclass
class PanelOptimizerWeights:
    angle: float = 1.0
    size: float = 0.5
    aspect: float = 1.0
    angle_sum: float = 1.0
    worst_alpha: float = 1.0
    worst_beta: float = 20.0
    barrier: float = 1e-2
    barrier_ratio: float = 0.05
    smooth: float = 0.0     # Brackbill-Saltzman smoothness on grid-line triples (structured grids)
    area: float = 0.0       # log^2(panel area / L^2): equal panel AREAS (structured grids, size priority)
    planar: float = 0.0     # (det[P1-P0, P2-P0, P3-P0] / L^3)^2: lifted panels close to planar (2026-10-29)
    principal: float = 0.0  # sin^2(2 theta) of both panel directions vs the principal bend directions (2026-10-29)


@dataclass
class ChartLift:
    """Exact piecewise-linear UV -> 3D map of a chart (paired triangles)."""

    uv: np.ndarray
    f2: np.ndarray
    xyz: np.ndarray
    f3: np.ndarray
    _tree: object = field(default=None, repr=False)

    def __post_init__(self):
        from scipy.spatial import cKDTree
        self.uv = np.asarray(self.uv, float)
        self.f2 = np.asarray(self.f2, int)
        self.xyz = np.asarray(self.xyz, float)
        self.f3 = np.asarray(self.f3, int)
        P = self.uv[self.f2]
        self.a = P[:, 0]
        self.e1 = P[:, 1] - P[:, 0]
        self.e2 = P[:, 2] - P[:, 0]
        self.det = self.e1[:, 0] * self.e2[:, 1] - self.e1[:, 1] * self.e2[:, 0]
        if np.any(np.abs(self.det) <= 0):
            raise PanelOptimizationError("chart has a degenerate UV triangle")
        X = self.xyz[self.f3]
        D = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]], axis=2)          # (t, 3, 2)
        M = np.stack([self.e1, self.e2], axis=2)                               # (t, 2, 2)
        self.J = D @ np.linalg.inv(M)                                          # (t, 3, 2)
        self.X0 = X[:, 0]
        self.centers = P.mean(axis=1)
        self._tree = cKDTree(self.centers)

    def bary(self, tri: np.ndarray, p: np.ndarray) -> np.ndarray:
        d = p - self.a[tri]
        s = (d[:, 0] * self.e2[tri, 1] - d[:, 1] * self.e2[tri, 0]) / self.det[tri]
        t = (self.e1[tri, 0] * d[:, 1] - self.e1[tri, 1] * d[:, 0]) / self.det[tri]
        return np.c_[1.0 - s - t, s, t]

    def locate(self, p: np.ndarray, tri_hint: np.ndarray | None = None, tol: float = 1e-10) -> np.ndarray:
        p = np.asarray(p, float).reshape(-1, 2)
        out = np.full(len(p), -1, int) if tri_hint is None else np.asarray(tri_hint, int).copy()
        need = np.ones(len(p), bool)
        if tri_hint is not None:
            w = self.bary(out, p)
            need = np.min(w, axis=1) < -tol
        n = len(self.f2)
        for i in np.flatnonzero(need):
            found = False
            for k in (8, 32, 128, 512, n):
                k = min(k, n)
                _, cand = self._tree.query(p[i], k=k)
                cand = np.atleast_1d(cand)
                w = self.bary(cand, np.repeat(p[i][None], len(cand), axis=0))
                mins = np.min(w, axis=1)
                j = int(np.argmax(mins))
                if mins[j] >= -tol:
                    out[i] = int(cand[j])
                    found = True
                    break
                if k >= n:
                    break
            if not found:
                out[i] = -1
        return out

    def lift(self, p: np.ndarray, tri: np.ndarray) -> np.ndarray:
        d = p - self.a[tri]
        return self.X0[tri] + np.einsum("tij,tj->ti", self.J[tri], d)


def _quad_terms(X: np.ndarray, quads: np.ndarray, L: float, w: PanelOptimizerWeights):
    """Per-panel badness b_q and dE/dX for E = mean b + alpha LSE(b)."""
    Q = X[quads]                                       # (m, 4, 3)
    nxt = np.roll(Q, -1, axis=1) - Q                    # u_k = X_{k+1} - X_k
    prv = np.roll(Q, 1, axis=1) - Q                     # v_k = X_{k-1} - X_k
    lu = np.maximum(np.linalg.norm(nxt, axis=2), 1e-300)
    lv = np.maximum(np.linalg.norm(prv, axis=2), 1e-300)
    dot = np.einsum("mkd,mkd->mk", nxt, prv)
    c = np.clip(dot / (lu * lv), -1.0, 1.0)
    theta = np.arccos(c)
    y = np.log(lu)                                      # log edge length, edge k = (k, k+1)
    ybar = y.mean(axis=1, keepdims=True)
    lr = np.log(lu / L)
    sum_dev = theta.sum(axis=1) - 2.0 * np.pi
    b = (w.angle * np.mean(c * c, axis=1) + w.size * np.mean(lr * lr, axis=1)
         + w.aspect * np.mean((y - ybar) ** 2, axis=1) + w.angle_sum * sum_dev ** 2)
    if w.area > 0:
        d1 = Q[:, 2] - Q[:, 0]
        d2 = Q[:, 3] - Q[:, 1]
        cr = np.cross(d1, d2)
        crn = np.maximum(np.linalg.norm(cr, axis=1), 1e-300)
        A = 0.5 * crn
        la = np.log(A / (L * L))
        b = b + w.area * la * la
    m = len(quads)
    if w.worst_alpha > 0 and w.worst_beta > 0:
        z = w.worst_beta * b
        zmax = float(np.max(z))
        ex = np.exp(z - zmax)
        soft = ex / np.sum(ex)
        lse = (zmax + np.log(np.mean(ex))) / w.worst_beta
    else:
        soft = np.zeros(m)
        lse = 0.0
    E = float(np.mean(b) + w.worst_alpha * lse)
    qw = 1.0 / m + w.worst_alpha * soft                 # dE/db_q
    # d b / d c_k, d b / d y_k (edge k), d b / d theta_k
    dbdc = qw[:, None] * w.angle * 2.0 * c / 4.0
    sin = np.sqrt(np.maximum(1.0 - c * c, 1e-12))
    dbdtheta = qw[:, None] * w.angle_sum * 2.0 * sum_dev[:, None] * np.ones((1, 4))
    dbdc = dbdc - dbdtheta / sin                        # theta = arccos(c)
    dbdy = qw[:, None] * (w.size * 2.0 * lr / 4.0 + w.aspect * 2.0 * (y - ybar) / 4.0)
    # c_k = u.v/(|u||v|)
    dcdu = prv / (lu * lv)[..., None] - (c / (lu * lu))[..., None] * nxt
    dcdv = nxt / (lu * lv)[..., None] - (c / (lv * lv))[..., None] * prv
    gu = dbdc[..., None] * dcdu + (dbdy / (lu * lu))[..., None] * nxt   # y_k = log|u_k|
    gv = dbdc[..., None] * dcdv
    G = np.zeros_like(X)
    if w.area > 0:
        chat = cr / crn[:, None]
        dbdA = (qw * w.area * 2.0 * la / A)[:, None]
        g1 = dbdA * 0.5 * np.cross(d2, chat)          # dA/d d1
        g2 = dbdA * 0.5 * np.cross(chat, d1)          # dA/d d2
        np.add.at(G, quads[:, 2], g1)
        np.add.at(G, quads[:, 0], -g1)
        np.add.at(G, quads[:, 3], g2)
        np.add.at(G, quads[:, 1], -g2)
    for k in range(4):
        kn, kp = (k + 1) % 4, (k - 1) % 4
        np.add.at(G, quads[:, kn], gu[:, k])
        np.add.at(G, quads[:, k], -gu[:, k])
        np.add.at(G, quads[:, kp], gv[:, k])
        np.add.at(G, quads[:, k], -gv[:, k])
    return E, G, b


def uv_corner_areas(P: np.ndarray, quads: np.ndarray) -> np.ndarray:
    Q = P[quads]
    u = np.roll(Q, -1, axis=1) - Q
    v = np.roll(Q, 1, axis=1) - Q
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]          # > 0 for a convex CCW corner


def quad_validity(P: np.ndarray, quads: np.ndarray):
    """Binding corner of every quad for 'simple and positively oriented'.

    A quad is a valid (simple, CCW) polygon iff one of its diagonals splits it
    into two positive triangles, i.e. (A1 > 0 and A3 > 0) or (A0 > 0 and A2 > 0)
    with A_k the corner areas.  Returns (g, corner): g_q = max over the two
    diagonals of the smaller corner area, and the corner index realising it.
    UV quads may be non-convex (the chart is distorted near cut tips); the
    lifted 3D panels are gated for convexity separately.
    """
    A = uv_corner_areas(P, quads)
    m02 = np.minimum(A[:, 0], A[:, 2])
    m13 = np.minimum(A[:, 1], A[:, 3])
    use13 = m13 >= m02
    c13 = np.where(A[:, 1] <= A[:, 3], 1, 3)
    c02 = np.where(A[:, 0] <= A[:, 2], 0, 2)
    corner = np.where(use13, c13, c02)
    g = np.where(use13, m13, m02)
    return g, corner, A


def _corner_area_gradient(P: np.ndarray, quads: np.ndarray, coeff: np.ndarray, corner: np.ndarray) -> np.ndarray:
    """Gradient of sum_q coeff_q * A_{q, corner_q} w.r.t. P."""
    m = len(quads)
    idx = np.arange(m)
    k = corner
    v_k = quads[idx, k]
    v_n = quads[idx, (k + 1) % 4]
    v_p = quads[idx, (k - 1) % 4]
    u = P[v_n] - P[v_k]
    v = P[v_p] - P[v_k]
    gu = coeff[:, None] * np.stack([v[:, 1], -v[:, 0]], axis=-1)
    gv = coeff[:, None] * np.stack([-u[:, 1], u[:, 0]], axis=-1)
    G = np.zeros_like(P)
    np.add.at(G, v_n, gu)
    np.add.at(G, v_p, gv)
    np.add.at(G, v_k, -gu - gv)
    return G


def _barrier(P: np.ndarray, quads: np.ndarray, ref: np.ndarray, w: PanelOptimizerWeights):
    """sum (log(g_q/(r0 ref_q)))^2 for g_q < r0 ref_q; +inf if a quad is not simple/CCW."""
    g, corner, _ = quad_validity(P, quads)
    if np.any(g <= 0):
        return np.inf, None
    r = g / (w.barrier_ratio * ref)
    active = r < 1.0
    lg = np.where(active, np.log(np.minimum(r, 1.0)), 0.0)
    E = float(w.barrier * np.sum(lg * lg) / len(quads))
    coeff = np.where(active, w.barrier * 2.0 * lg / np.maximum(g, 1e-300) / len(quads), 0.0)
    return E, _corner_area_gradient(P, quads, coeff, corner)


def _lbfgs(fun, x0, *, max_iter=300, m=8, gtol=1e-9, ftol=1e-12):
    x = np.asarray(x0, float).copy()
    f, g = fun(x)
    if not np.isfinite(f):
        raise PanelOptimizationError("optimiser started from an infeasible point")
    S, Y = [], []
    it = 0
    for it in range(1, int(max_iter) + 1):
        q = g.copy()
        alphas = []
        for s, y in reversed(list(zip(S, Y))):
            rho = 1.0 / max(float(y @ s), 1e-300)
            a = rho * float(s @ q)
            alphas.append((a, rho, s, y))
            q -= a * y
        if S:
            gamma = float(S[-1] @ Y[-1]) / max(float(Y[-1] @ Y[-1]), 1e-300)
            q *= gamma
        else:
            q *= 1.0 / max(float(np.linalg.norm(g)), 1e-300)
        for a, rho, s, y in reversed(alphas):
            b = rho * float(y @ q)
            q += s * (a - b)
        d = -q
        slope = float(g @ d)
        if slope >= 0:
            S.clear(); Y.clear()
            d = -g / max(float(np.linalg.norm(g)), 1e-300)
            slope = float(g @ d)
        step = 1.0
        accepted = False
        for _ in range(40):
            xn = x + step * d
            fn, gn = fun(xn)
            if np.isfinite(fn) and fn <= f + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if S:
                S.clear(); Y.clear()
                continue
            break
        s_vec, y_vec = xn - x, gn - g
        if float(s_vec @ y_vec) > 1e-12:
            S.append(s_vec); Y.append(y_vec)
            if len(S) > m:
                S.pop(0); Y.pop(0)
        rel = (f - fn) / max(abs(f), 1e-300)
        x, f, g = xn, fn, gn
        if rel < ftol or float(np.max(np.abs(g))) < gtol:
            break
    return x, f, it


def untangle(P: np.ndarray, quads: np.ndarray, free: np.ndarray, *, margin: float = 0.02,
             max_iter: int = 600) -> tuple[np.ndarray, dict]:
    """Make every quad simple and positively oriented by moving the free vertices only."""
    quads = np.asarray(quads, int)
    P = np.asarray(P, float).copy()
    g, _, A = quad_validity(P, quads)
    ref_area = float(np.median(np.abs(A))) if len(A) else 1.0
    if np.all(g > margin * ref_area):
        return P, {"untangle_needed": False, "untangle_iterations": 0}
    free = np.asarray(free, int)
    tau = margin * ref_area

    def fun(z):
        Q = P.copy()
        Q[free] = z.reshape(-1, 2)
        gq, corner, _ = quad_validity(Q, quads)
        d = np.minimum(gq - 2.0 * tau, 0.0)
        E = float(np.sum(d * d)) / ref_area ** 2
        G = _corner_area_gradient(Q, quads, 2.0 * d / ref_area ** 2, corner)
        return E, G[free].ravel()

    z, E, it = _lbfgs(fun, P[free].ravel(), max_iter=max_iter, gtol=1e-14, ftol=1e-16)
    P[free] = z.reshape(-1, 2)
    g, _, _ = quad_validity(P, quads)
    return P, {"untangle_needed": True, "untangle_iterations": int(it),
               "untangle_min_quad_validity_ratio": float(np.min(g) / ref_area),
               "untangle_success": bool(np.all(g > 0)), "untangle_invalid_quads": int(np.sum(g <= 0))}


def optimize_panel_quality(lift: ChartLift, quad_uv: np.ndarray, quads: np.ndarray, fixed: np.ndarray, *,
                           weights: PanelOptimizerWeights | None = None, max_iter: int = 300,
                           reference_length: float | None = None,
                           line_triples: np.ndarray | None = None,
                           allow_invalid: bool = False,
                           principal_field: dict | None = None,
                           crease_slide: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Move non-fixed vertices (UV) to optimise the lifted panels; returns (uv, xyz, report).

    ``line_triples`` (a, i, b): consecutive vertices along a grid line; with
    ``weights.smooth > 0`` adds sum |X_a - 2 X_i + X_b|^2 / L^2 (3D smoothness).
    """
    w = weights or PanelOptimizerWeights()
    quads = np.asarray(quads, int)
    P = np.asarray(quad_uv, float).copy()
    fixed = np.asarray(fixed, bool)
    free = np.flatnonzero(~fixed)
    report: dict = {"panel_optimizer_free_vertices": int(len(free)),
                    "panel_optimizer_fixed_vertices": int(np.sum(fixed))}
    if not len(free):
        tri = lift.locate(P)
        if np.any(tri < 0):
            raise PanelOptimizationError("ZHANG2021_ONESTRING_QUAD_VERTEX_OUTSIDE_CHART")
        return P, lift.lift(P, tri), report
    P, ureport = untangle(P, quads, free)
    report.update(ureport)
    if ureport.get("untangle_needed") and not ureport.get("untangle_success") and allow_invalid:
        P = _pull_inside(lift, P, free)
        tri = lift.locate(P)
        report["panel_optimizer_skipped"] = "untangle failed; grid left as is (inverted quads reported)"
        return P, lift.lift(P, tri), report
    if ureport.get("untangle_needed") and not ureport.get("untangle_success"):
        raise PanelOptimizationError(
            "ZHANG2021_ONESTRING_PANEL_UNTANGLE_FAILED: a valid (convex, unflipped) UV embedding of the "
            f"modified quad mesh was not found ({ureport.get('untangle_invalid_quads')} quad(s) remain "
            f"invalid, min validity ratio {ureport.get('untangle_min_quad_validity_ratio'):.3g})")
    if allow_invalid:
        P = _pull_inside(lift, P, free)
    tri = lift.locate(P)
    if np.any(tri < 0):
        raise PanelOptimizationError(
            f"ZHANG2021_ONESTRING_QUAD_VERTEX_OUTSIDE_CHART: {int(np.sum(tri < 0))} vertices")
    X = lift.lift(P, tri)
    ref_area = np.maximum(np.abs(np.sum(uv_corner_areas(P, quads), axis=1)) / 4.0, 1e-300)
    if reference_length is None:
        edges = np.unique(np.sort(np.concatenate([quads[:, [k, (k + 1) % 4]] for k in range(4)]), axis=1), axis=0)
        el = np.linalg.norm(X[edges[:, 1]] - X[edges[:, 0]], axis=1)
        reference_length = float(np.exp(np.mean(np.log(np.maximum(el, 1e-300)))))
    L = float(reference_length)
    state = {"tri": tri}
    tri_lines = None
    if line_triples is not None and len(line_triples) and w.smooth > 0:
        tri_lines = np.asarray(line_triples, int).reshape(-1, 3)
    report["panel_optimizer_smooth_triples"] = 0 if tri_lines is None else int(len(tri_lines))
    cs = None
    if crease_slide is not None and len(crease_slide.get("vertices", [])):
        # vertices that lie on a crease may slide along it but not leave it (2026-10-29)
        cv = np.asarray(crease_slide["vertices"], int)
        cl = np.asarray(crease_slide["line_of"], int)
        segs = [(np.asarray(Lp, float)[:-1], np.asarray(Lp, float)[1:]) for Lp in crease_slide["lines"]]
        cs = (cv, cl, segs, float(crease_slide.get("weight", 100.0)))
        report["panel_optimizer_crease_slide_vertices"] = int(len(cv))

    def crease_proj(Xz):
        out = np.empty((len(cs[0]), 3))
        for k, (v, li) in enumerate(zip(cs[0], cs[1])):
            A, B = cs[2][li]
            AB = B - A
            t = np.clip(np.einsum("ij,ij->i", Xz[v] - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-30), 0.0, 1.0)
            Pp = A + t[:, None] * AB
            out[k] = Pp[int(np.argmin(np.linalg.norm(Pp - Xz[v], axis=1)))]
        return out
    pf = None
    if principal_field is not None and w.principal > 0:
        # per panel: principal bend directions of the surface near its centre and an anisotropy
        # weight in [0, 1] (frozen during the optimisation; the field is smooth at panel scale)
        from scipy.spatial import cKDTree
        cen = X[quads].mean(axis=1)
        j = cKDTree(np.asarray(principal_field["V"], float)).query(cen)[1]
        an = np.asarray(principal_field["aniso"], float)[j]
        lo, full = float(principal_field.get("lo", 10.0)), float(principal_field.get("full", 30.0))
        pf = (np.asarray(principal_field["d1"], float)[j], np.asarray(principal_field["d2"], float)[j],
              np.clip((an - lo) / max(full - lo, 1e-9), 0.0, 1.0))
        report["panel_optimizer_principal_weighted_panels"] = float(np.sum(pf[2]))

    def fun(z):
        Pz = P.copy()
        Pz[free] = z.reshape(-1, 2)
        Eb, Gb = _barrier(Pz, quads, ref_area, w)
        if not np.isfinite(Eb):
            return np.inf, None
        t = lift.locate(Pz, state["tri"])
        if np.any(t < 0):
            return np.inf, None
        state["tri"] = t
        Xz = lift.lift(Pz, t)
        Eq, GX, _ = _quad_terms(Xz, quads, L, w)
        if tri_lines is not None:
            a, i, b = tri_lines[:, 0], tri_lines[:, 1], tri_lines[:, 2]
            r = Xz[a] - 2.0 * Xz[i] + Xz[b]
            c = w.smooth / (L * L * len(tri_lines))
            Eq = Eq + c * float(np.sum(r * r))
            GX = GX.copy()
            np.add.at(GX, a, 2.0 * c * r)
            np.add.at(GX, b, 2.0 * c * r)
            np.add.at(GX, i, -4.0 * c * r)
        if w.planar > 0:
            # planarity of every lifted panel: triple product of three edge vectors from corner 0
            # (6 x tetrahedron volume), relative to L^3; a planar quad has zero volume
            Q0, Q1, Q2, Q3 = (Xz[quads[:, k]] for k in range(4))
            a1, a2, a3 = Q1 - Q0, Q2 - Q0, Q3 - Q0
            c23, c31, c12 = np.cross(a2, a3), np.cross(a3, a1), np.cross(a1, a2)
            tau = np.einsum("ij,ij->i", a1, c23) / L ** 3
            cp = w.planar / len(quads)
            Eq = Eq + cp * float(np.sum(tau * tau))
            g = (2.0 * cp * tau / L ** 3)[:, None]
            GX = GX.copy()
            np.add.at(GX, quads[:, 1], g * c23)
            np.add.at(GX, quads[:, 2], g * c31)
            np.add.at(GX, quads[:, 3], g * c12)
            np.add.at(GX, quads[:, 0], -g * (c23 + c31 + c12))
        if cs is not None:
            r_ = Xz[cs[0]] - crease_proj(Xz)
            cw = cs[3] / (len(cs[0]) * L * L)
            Eq = Eq + cw * float(np.sum(r_ * r_))
            GX = GX.copy()
            np.add.at(GX, cs[0], 2.0 * cw * r_)
        if pf is not None:
            # both panel directions (mid-edge vectors) along the principal directions:
            # f = sin^2(2 theta) = 4 p^2 q^2 / (p^2 + q^2)^2, p = e.d1, q = e.d2
            d1q, d2q, wq = pf
            Q0, Q1, Q2, Q3 = (Xz[quads[:, k]] for k in range(4))
            cpa = w.principal / len(quads)
            GX = GX.copy()
            for e, sg in (((Q1 + Q2 - Q0 - Q3) * 0.5, (-0.5, 0.5, 0.5, -0.5)),
                          ((Q2 + Q3 - Q0 - Q1) * 0.5, (-0.5, -0.5, 0.5, 0.5))):
                pp = np.einsum("ij,ij->i", e, d1q); qq = np.einsum("ij,ij->i", e, d2q)
                ss = np.maximum(pp * pp + qq * qq, 1e-30)
                f = 4.0 * pp * pp * qq * qq / (ss * ss)
                Eq = Eq + cpa * float(np.dot(wq, f))
                dfp = 8.0 * pp * qq * qq * (qq * qq - pp * pp) / ss ** 3
                dfq = 8.0 * pp * pp * qq * (pp * pp - qq * qq) / ss ** 3
                ge = (cpa * wq)[:, None] * (dfp[:, None] * d1q + dfq[:, None] * d2q)
                for k in range(4):
                    np.add.at(GX, quads[:, k], sg[k] * ge)
        GP = np.einsum("nij,ni->nj", lift.J[t], GX) + Gb
        return Eq + Eb, GP[free].ravel()

    E0, _ = fun(P[free].ravel())
    z, E1, it = _lbfgs(fun, P[free].ravel(), max_iter=max_iter)
    P[free] = z.reshape(-1, 2)
    tri = lift.locate(P, state["tri"])
    X = lift.lift(P, tri)
    report.update({"panel_optimizer_energy_before": float(E0), "panel_optimizer_energy_after": float(E1),
                   "panel_optimizer_iterations": int(it), "panel_optimizer_reference_length": L,
                   "panel_optimizer_weights": dict(vars(w))})
    if np.any(quad_validity(P, quads)[0] <= 0) and not allow_invalid:
        raise PanelOptimizationError("ZHANG2021_ONESTRING_PANEL_OPTIMIZER_FLIPPED_A_QUAD")
    return P, X, report


def _pull_inside(lift: ChartLift, P: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Move vertices that left the chart to the nearest chart-triangle centroid (report-only modes)."""
    P = P.copy()
    tri = lift.locate(P)
    out = np.flatnonzero(tri < 0)
    if len(out):
        cen = lift.uv[lift.f2].mean(axis=1)
        for i in out:
            P[i] = cen[int(np.argmin(np.linalg.norm(cen - P[i], axis=1)))]
    return P


__all__ = ["ChartLift", "PanelOptimizerWeights", "PanelOptimizationError", "optimize_panel_quality",
           "untangle", "uv_corner_areas", "quad_validity"]
