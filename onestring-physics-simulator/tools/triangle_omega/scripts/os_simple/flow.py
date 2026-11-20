"""Projected-Newton solver for UV maps with element energies written in JAX.

Energy terms (all optional, weights given by the caller):
  distortion   per triangle  A_t/A * [ SD + w_conf (MIPS - 2) ]          (same form as the OptCuts fork, ANGLE_W)
  align        per triangle  w_align * alpha_t * A_t/A * sin^2(2 phi_t), phi_t = angle of J_t d1_t
                (4-fold: the max-curvature direction is pulled onto the x or y axis)
  chord        per boundary chord  w_chord * l3/L * ((|dx|_e + |dy|_e)/|d| - 1)   (L1: straight axis-aligned runs)
  seam         per seam edge pair  w_seam * (|eA| - |eB|)^2 / l3 / Ls          (both banks get the same length)
Steps never flip a triangle (largest injective step) and never let the outline cross itself (shapely test).
Free variables can be restricted (fixed vertices) for the boundary-fitting continuation.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from shapely.geometry import LinearRing


def _tri_E(u, Xinv, aw, d1, al, wc, wa):
    E = jnp.stack([u[2:4] - u[0:2], u[4:6] - u[0:2]], axis=1)
    J = E @ Xinv
    fro2 = jnp.sum(J * J); det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    sd = fro2 * (1.0 + 1.0 / det ** 2)
    mips = fro2 / det
    g = J @ d1
    s2 = (2.0 * g[0] * g[1]) ** 2 / (jnp.sum(g * g) ** 2 + 1e-30)
    return aw * (sd + wc * (mips - 2.0)) + wa * al * aw * s2


def _chord_E(u, w, eps):
    d = u[2:4] - u[0:2]
    n2 = jnp.sum(d * d)
    return w * ((jnp.sqrt(d[0] ** 2 + eps ** 2 * n2) + jnp.sqrt(d[1] ** 2 + eps ** 2 * n2)) / jnp.sqrt(n2) - 1.0)


def _seam_E(u, w):
    la = jnp.sqrt(jnp.sum((u[2:4] - u[0:2]) ** 2)); lb = jnp.sqrt(jnp.sum((u[6:8] - u[4:6]) ** 2))
    return w * (la - lb) ** 2


def _bar_E(u, d0, k):
    p, a, b = u[0:2], u[2:4], u[4:6]
    e = b - a
    t = jnp.clip(jnp.dot(p - a, e) / jnp.dot(e, e), 0.0, 1.0)
    q = a + t * e
    d = jnp.sqrt(jnp.sum((p - q) ** 2) + 1e-30)
    return jnp.where(d < d0, -k * (d - d0) ** 2 * jnp.log(d / d0), 0.0)


def _psd(H):
    w, Q = jnp.linalg.eigh(H)
    return (Q * jnp.maximum(w, 0.0)[..., None, :]) @ jnp.swapaxes(Q, -1, -2)


_tri_val = jax.jit(jax.vmap(_tri_E, in_axes=(0, 0, 0, 0, 0, None, None)))
_tri_grad = jax.jit(jax.vmap(jax.grad(_tri_E), in_axes=(0, 0, 0, 0, 0, None, None)))
_tri_hess = jax.jit(lambda *a: _psd(jax.vmap(jax.hessian(_tri_E), in_axes=(0, 0, 0, 0, 0, None, None))(*a)))
_ch_val = jax.jit(jax.vmap(_chord_E, in_axes=(0, 0, None)))
_ch_grad = jax.jit(jax.vmap(jax.grad(_chord_E), in_axes=(0, 0, None)))
_ch_hess = jax.jit(lambda *a: _psd(jax.vmap(jax.hessian(_chord_E), in_axes=(0, 0, None))(*a)))
_se_val = jax.jit(jax.vmap(_seam_E, in_axes=(0, 0)))
_se_grad = jax.jit(jax.vmap(jax.grad(_seam_E), in_axes=(0, 0)))
_se_hess = jax.jit(lambda *a: _psd(jax.vmap(jax.hessian(_seam_E), in_axes=(0, 0))(*a)))
_ba_val = jax.jit(jax.vmap(_bar_E, in_axes=(0, None, None)))
_ba_grad = jax.jit(jax.vmap(jax.grad(_bar_E), in_axes=(0, None, None)))
_ba_hess = jax.jit(lambda *a: _psd(jax.vmap(jax.hessian(_bar_E), in_axes=(0, None, None))(*a)))


def local_frames(X, F):
    """Per triangle: 2x2 inverse of the 3D edge matrix in a local orthonormal frame, and the frame (e1, e2)."""
    P0, P1, P2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    e1 = P1 - P0; e1 /= np.linalg.norm(e1, axis=1)[:, None]
    n = np.cross(P1 - P0, P2 - P0); n /= np.linalg.norm(n, axis=1)[:, None]
    e2 = np.cross(n, e1)
    a = P1 - P0; b = P2 - P0
    Xl = np.stack([np.stack([np.sum(a * e1, 1), np.sum(b * e1, 1)], 1), np.stack([np.sum(a * e2, 1), np.sum(b * e2, 1)], 1)], 1)
    return np.linalg.inv(Xl), e1, e2


class UVProblem:
    def __init__(self, X, F, loop, *, w_conf=16.0, align_dir=None, align_alpha=None, w_align=0.0,
                 chords=None, chord_w=None, w_chord=0.0, chord_eps=0.05, seam_pairs=None, seam_w=None, w_seam=0.0, bar_d0=0.0, bar_k=1.0, run_Q=None, w_run=0.0):
        self.anc_idx = np.zeros(0, int); self.anc_tgt = np.zeros((0, 2)); self.anc_w = np.zeros(0); self.w_anchor = 0.0
        self.X = np.asarray(X, float); self.F = np.asarray(F, int); self.loop = np.asarray(loop, int)
        self.n = len(X)
        self.Xinv, e1, e2 = local_frames(self.X, self.F)
        A3 = 0.5 * np.linalg.norm(np.cross(self.X[self.F[:, 1]] - self.X[self.F[:, 0]], self.X[self.F[:, 2]] - self.X[self.F[:, 0]]), axis=1)
        self.aw = A3 / A3.sum()
        if align_dir is None:
            self.d1 = np.tile([1.0, 0.0], (len(F), 1)); self.al = np.zeros(len(F))
        else:
            d = np.stack([np.sum(align_dir * e1, 1), np.sum(align_dir * e2, 1)], 1)
            self.d1 = d / np.maximum(np.linalg.norm(d, axis=1), 1e-12)[:, None]; self.al = np.asarray(align_alpha, float)
        self.w_conf, self.w_align = w_conf, w_align
        self.ch = np.zeros((0, 2), int) if chords is None else np.asarray(chords, int)
        self.chw = np.zeros(0) if chord_w is None else np.asarray(chord_w, float)
        self.w_chord, self.eps = w_chord, chord_eps
        self.sp_ = np.zeros((0, 4), int) if seam_pairs is None else np.asarray(seam_pairs, int)
        self.spw = np.zeros(0) if seam_w is None else np.asarray(seam_w, float)
        self.w_seam = w_seam
        self.bar_d0, self.bar_k = bar_d0, bar_k
        self.run_Q, self.w_run = run_Q, w_run        # constant quadratic form u^T Q u (straight runs)
        Lp = self.loop
        seg = np.linalg.norm(self.X[np.roll(Lp, -1)] - self.X[Lp], axis=1)
        self.cum = np.r_[0.0, np.cumsum(seg)][:-1]; self.Ltot = float(seg.sum())
        self.fixed = np.zeros(self.n, bool)
        self._build_index()

    def _build_index(self):
        def dofs(idx):
            idx = np.asarray(idx, int)
            if idx.size == 0:
                return np.zeros((0, 2 * (idx.shape[1] if idx.ndim == 2 else 1)), int)
            return np.stack([2 * idx, 2 * idx + 1], -1).reshape(len(idx), -1)
        self.td = dofs(self.F); self.cd = dofs(self.ch); self.sd = dofs(self.sp_)

    def bar_pairs(self, U):
        """(vertex, edge-start, edge-end) for boundary vertices closer than d0 to a non-incident boundary edge."""
        if self.bar_d0 <= 0:
            return np.zeros((0, 3), int)
        from scipy.spatial import cKDTree
        L = self.loop; m = len(L)
        A = U[L]; B = U[np.roll(L, -1)]
        mid = 0.5 * (A + B); half = 0.5 * np.linalg.norm(B - A, axis=1)
        tree = cKDTree(mid)
        cand = tree.query_ball_point(A, self.bar_d0 + half.max())
        out = []
        for k, js in enumerate(cand):
            for j in js:
                if j == k or j == (k - 1) % m:
                    continue
                dl = min(abs(self.cum[k] - self.cum[j]), abs(self.cum[k] - self.cum[(j + 1) % m]))
                dl = min(dl, self.Ltot - dl)
                if dl < 4.0 * self.bar_d0:          # neighbours along the outline (also both banks next to a tip)
                    continue
                a, b, p = A[j], B[j], A[k]
                e = b - a; t = np.clip(np.dot(p - a, e) / max(np.dot(e, e), 1e-30), 0, 1)
                if np.linalg.norm(p - (a + t * e)) < self.bar_d0:
                    out.append((L[k], L[j], L[(j + 1) % m]))
        return np.array(out, int).reshape(-1, 3)

    def energy(self, U, parts=False):
        u = U.reshape(-1)
        det = _dets(U, self.F)
        if np.any(det <= 0):
            return np.inf
        et = np.asarray(_tri_val(u[self.td], self.Xinv, self.aw, self.d1, self.al, self.w_conf, self.w_align))
        ec = np.asarray(_ch_val(u[self.cd], self.chw * self.w_chord, self.eps)) if len(self.ch) and self.w_chord else np.zeros(0)
        es = np.asarray(_se_val(u[self.sd], self.spw * self.w_seam)) if len(self.sp_) and self.w_seam else np.zeros(0)
        bp = self.bar_pairs(U)
        eb = np.asarray(_ba_val(u[np.stack([2 * bp, 2 * bp + 1], -1).reshape(len(bp), -1)], self.bar_d0, self.bar_k)) if len(bp) else np.zeros(0)
        if not np.all(np.isfinite(eb)):
            return np.inf
        er = float(self.w_run * u @ (self.run_Q @ u)) if (self.run_Q is not None and self.w_run) else 0.0
        if self.w_anchor and len(self.anc_idx):
            er += float(self.w_anchor * np.sum(self.anc_w[:, None] * (U[self.anc_idx] - self.anc_tgt) ** 2))
        if parts:
            return dict(tri=float(et.sum()), chord=float(ec.sum()), seam=float(es.sum()), barrier=float(eb.sum()), run=er)
        return float(et.sum() + ec.sum() + es.sum() + eb.sum() + er)

    def distortion_parts(self, U):
        """Area-weighted SD and MIPS, and the alignment residual, measured separately."""
        u = U.reshape(-1)
        sd = np.asarray(_tri_val(u[self.td], self.Xinv, self.aw, self.d1, self.al, 0.0, 0.0)).sum()
        mi = (np.asarray(_tri_val(u[self.td], self.Xinv, self.aw, self.d1, self.al, 1.0, 0.0)).sum() - sd) + 2.0
        al = np.asarray(_tri_val(u[self.td], self.Xinv, self.aw, self.d1, self.al, 0.0, 1.0)).sum() - sd
        return dict(sd=float(sd), mips=float(mi), align=float(al / max(np.sum(self.aw * self.al), 1e-12)))

    def grad_hess(self, U):
        u = U.reshape(-1); N = 2 * self.n
        g = np.zeros(N); rows, cols, vals = [], [], []
        def add(d, ge, He):
            np.add.at(g, d.reshape(-1), np.asarray(ge).reshape(-1))
            k = d.shape[1]
            rows.append(np.repeat(d, k, axis=1).reshape(-1)); cols.append(np.tile(d, (1, k)).reshape(-1)); vals.append(np.asarray(He).reshape(-1))
        a = (u[self.td], self.Xinv, self.aw, self.d1, self.al, self.w_conf, self.w_align)
        add(self.td, _tri_grad(*a), _tri_hess(*a))
        if len(self.ch) and self.w_chord:
            a = (u[self.cd], self.chw * self.w_chord, self.eps)
            add(self.cd, _ch_grad(*a), _ch_hess(*a))
        if len(self.sp_) and self.w_seam:
            a = (u[self.sd], self.spw * self.w_seam)
            add(self.sd, _se_grad(*a), _se_hess(*a))
        bp = self.bar_pairs(U)
        if len(bp):
            bd = np.stack([2 * bp, 2 * bp + 1], -1).reshape(len(bp), -1)
            a = (u[bd], self.bar_d0, self.bar_k)
            add(bd, _ba_grad(*a), _ba_hess(*a))
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
        if self.run_Q is not None and self.w_run:
            g = g + 2.0 * self.w_run * (self.run_Q @ u)
            H = H + 2.0 * self.w_run * self.run_Q
        if self.w_anchor and len(self.anc_idx):
            dof = np.stack([2 * self.anc_idx, 2 * self.anc_idx + 1], -1).ravel()
            ww = np.repeat(self.anc_w, 2) * 2.0 * self.w_anchor
            g = g.copy(); np.add.at(g, dof, ww * (U[self.anc_idx] - self.anc_tgt).ravel())
            H = H + sp.coo_matrix((ww, (dof, dof)), shape=H.shape).tocsr()
        return g, H

    def max_step(self, U, P):
        a = P[self.F[:, 1]] - P[self.F[:, 0]]; b = P[self.F[:, 2]] - P[self.F[:, 0]]
        A = U[self.F[:, 1]] - U[self.F[:, 0]]; B = U[self.F[:, 2]] - U[self.F[:, 0]]
        qa = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        qb = A[:, 0] * b[:, 1] + a[:, 0] * B[:, 1] - A[:, 1] * b[:, 0] - a[:, 1] * B[:, 0]
        qc = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        t = np.full(len(qa), np.inf)
        lin = np.abs(qa) < 1e-30
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -qc[lin] / qb[lin]; t[lin] = np.where(r > 0, r, np.inf)
            disc = qb ** 2 - 4 * qa * qc
            ok = (~lin) & (disc >= 0)
            sq = np.sqrt(np.where(ok, disc, 0))
            r1 = (-qb - sq) / (2 * qa); r2 = (-qb + sq) / (2 * qa)
            r1 = np.where(r1 > 0, r1, np.inf); r2 = np.where(r2 > 0, r2, np.inf)
            t = np.where(ok, np.minimum(r1, r2), t)
        return float(min(1.0, 0.9 * t.min()))

    def outline_ok(self, U):
        return LinearRing(U[self.loop]).is_simple

    def solve(self, U, tol=1e-9, max_guard=2000, record=None, check_outline=True):
        U = U.copy()
        free = np.repeat(~self.fixed, 2)
        E = self.energy(U)
        hist = [E]
        for it in range(max_guard):
            g, H = self.grad_hess(U)
            Hf = H[free][:, free]; gf = g[free]
            reg = 1e-9 * (Hf.diagonal().mean() + 1e-30)
            try:
                p = -spla.spsolve((Hf + reg * sp.eye(Hf.shape[0])).tocsc(), gf)
            except Exception:
                p = -gf
            dec = float(-gf @ p)
            if dec <= tol * max(abs(E), 1e-30):
                break
            P = np.zeros(2 * self.n); P[free] = p; P = P.reshape(-1, 2)
            t = self.max_step(U, P)
            accepted = False
            while t > 1e-12:
                Un = U + t * P
                En = self.energy(Un)
                if En <= E - 1e-4 * t * dec and (not check_outline or self.outline_ok(Un)):
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                break
            rel = (E - En) / max(abs(E), 1e-30)
            U, E = Un, En
            hist.append(E)
            if record is not None:
                record(U, E)
            if rel < 1e-12:
                break
        return U, hist


def _dets(U, F):
    a = U[F[:, 1]] - U[F[:, 0]]; b = U[F[:, 2]] - U[F[:, 0]]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
