"""Prototype: 10-23 area-threshold crop -> straightened logical polygon -> Brackbill-Saltzman grid.

1. ``area_threshold_crop``: a square grid of panel size h is laid over Omega_1 (rotation chosen
   among 0 / 22.5 / 45 / 67.5 deg for the fewest staircase corners); every cell whose area inside
   Omega_1 is >= threshold is kept whole (the 2026-10-23 rule).  It fixes how many cells each
   part of the chart gets.
2. The staircase is straightened by the logical-domain MILP: the corner positions of the
   orthogonal polygon are pulled to the chart positions in the same grid frame (strong shape
   term), so each region keeps the crop's cell count while a diagonal boundary run becomes one
   straight side (every boundary vertex then has exactly two panels).
3. ``brackbill_saltzman``: the interior grid points are placed by minimising the discrete
   Brackbill-Saltzman functional  I_S + lambda_O I_O + lambda_V I_V  (smoothness |x_xi|^2+|x_eta|^2
   over J, orthogonality (x_xi . x_eta)^2, weighted volume w J^2 with w = (3D area / chart area)^2
   so cells tend to equal 3D area); boundary points stay on the smooth Omega_1 boundary.
"""
from __future__ import annotations

import numpy as np


def _rot(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def area_threshold_crop(uv, f2, h: float, threshold: float = 0.5, angles=(0.0, 22.5, 45.0, 67.5)) -> dict:
    """Whole-cell crop of a square grid (cell h) over the chart, best of a few rotations."""
    from .optcuts_area_grid_boundary import _intersection_ratios
    uv = np.asarray(uv, float)
    best = None
    for ang in angles:
        R = _rot(-ang)
        P = uv @ R.T
        origin = P.min(axis=0) - 1e-9
        nx, ny = (np.ceil(np.ptp(P, axis=0) / h).astype(int) + 1)
        cov = _intersection_ratios(P, f2, origin, np.array([h, h]), int(nx), int(ny))
        mask = cov >= threshold
        # staircase corners of the kept set = lattice vertices where the 2x2 pattern is not straight
        M = np.pad(mask, 1).astype(int)
        q = M[:-1, :-1] + M[1:, :-1] + M[:-1, 1:] + M[1:, 1:]
        diag = (M[:-1, :-1] == M[1:, 1:]) & (M[1:, :-1] == M[:-1, 1:]) & (M[:-1, :-1] != M[1:, :-1])
        corners = int(np.sum((q == 1) | (q == 3) | diag))
        cand = {"angle": float(ang), "R": R, "origin": origin, "mask": mask, "coverage": cov,
                "cells": int(mask.sum()), "staircase_corners": corners}
        if best is None or corners < best["staircase_corners"]:
            best = cand
    return best


def crop_frame_positions(crop: dict, pts: np.ndarray, h: float) -> np.ndarray:
    """Chart points in the crop's lattice units."""
    return (np.asarray(pts, float) @ crop["R"].T - crop["origin"]) / h


def brackbill_saltzman(grid_uv, quads, fixed, chart_uv, chart_f2, chart_xyz, chart_f3, h: float,
                       lam_o: float = 0.3, lam_v: float = 0.3, iters: int = 3000):
    """Interior grid points minimising the discrete Brackbill-Saltzman functional (corner form).

    Each quad contributes, at each of its four corners with edge vectors a (xi) and b (eta),
    (|a|^2+|b|^2)/J + lam_o (a.b)^2 + lam_v w J^2,  J = a x b  (coordinates in units of h).
    J <= 0 anywhere is rejected (the smoothness term is a barrier), so the grid never folds."""
    from matplotlib.tri import Triangulation
    from .zhang2021_panel_optimizer import _lbfgs
    X = np.asarray(grid_uv, float) / h
    Q = np.asarray(quads, int)
    n = len(X)
    fixed = np.asarray(sorted(set(int(v) for v in fixed)), int)
    free = np.setdiff1d(np.arange(n), fixed)
    # area weight from the chart: w = (3D area / chart area)^2 at every grid point
    cuv = np.asarray(chart_uv, float)
    cf = np.asarray(chart_f2, int)
    X3 = np.asarray(chart_xyz, float)[np.asarray(chart_f3, int)]
    a3 = 0.5 * np.linalg.norm(np.cross(X3[:, 1] - X3[:, 0], X3[:, 2] - X3[:, 0]), axis=1)
    e1, e2 = cuv[cf[:, 1]] - cuv[cf[:, 0]], cuv[cf[:, 2]] - cuv[cf[:, 0]]
    a2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    s_t = a3 / np.maximum(a2, 1e-300)
    s_t /= float(np.sum(a3) / np.sum(a2))
    t = Triangulation(cuv[:, 0], cuv[:, 1], cf).get_trifinder()(X[:, 0] * h, X[:, 1] * h)
    s_v = np.where(t >= 0, s_t[np.clip(t, 0, None)], 1.0)
    # corner (p0, pa, pb) index triples, CCW quads
    C = np.concatenate([Q[:, [0, 1, 3]], Q[:, [1, 2, 0]], Q[:, [2, 3, 1]], Q[:, [3, 0, 2]]])
    w = (s_v[C[:, 0]] ** 2)

    def fun(z):
        P = X.copy()
        P[free] = z.reshape(-1, 2)
        a = P[C[:, 1]] - P[C[:, 0]]
        b = P[C[:, 2]] - P[C[:, 0]]
        J = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if np.any(J <= 1e-12):
            return np.inf, None
        s2 = np.sum(a * a, 1) + np.sum(b * b, 1)
        ab = np.sum(a * b, 1)
        E = float(np.sum(s2 / J + lam_o * ab * ab + lam_v * w * J * J))
        dJa = np.c_[b[:, 1], -b[:, 0]]
        dJb = np.c_[-a[:, 1], a[:, 0]]
        coefJ = -s2 / J ** 2 + 2.0 * lam_v * w * J
        ga = (2.0 / J)[:, None] * a + coefJ[:, None] * dJa + (2.0 * lam_o * ab)[:, None] * b
        gb = (2.0 / J)[:, None] * b + coefJ[:, None] * dJb + (2.0 * lam_o * ab)[:, None] * a
        G = np.zeros_like(P)
        np.add.at(G, C[:, 1], ga)
        np.add.at(G, C[:, 2], gb)
        np.add.at(G, C[:, 0], -ga - gb)
        return E, G[free].ravel()

    E0, _ = fun(X[free].ravel())
    untangled = False
    if not np.isfinite(E0):
        # non-convex / folded start: make every quad corner positive first (elliptic untangling of
        # the corner triangles towards square corners), then minimise the functional
        from .structured_grid_20261028 import _untangle_elliptic
        ref_tri = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (len(C), 1, 1))
        X = _untangle_elliptic(X, C, free, ref_tri, outer=40, iters=600)
        untangled = True
        E0, _ = fun(X[free].ravel())
        if not np.isfinite(E0):
            return np.asarray(grid_uv, float), {"bs_status": "start grid folded (untangling failed)"}
    z, E1, it = _lbfgs(fun, X[free].ravel(), max_iter=iters)
    X[free] = z.reshape(-1, 2)
    return X * h, {"bs_untangled_start": untangled, "bs_energy_before": float(E0), "bs_energy_after": float(E1), "bs_iterations": int(it),
                   "bs_lambda_orth": lam_o, "bs_lambda_vol": lam_v}


__all__ = ["area_threshold_crop", "crop_frame_positions", "brackbill_saltzman"]
