"""Shape-first UV refinement with a fixed boundary (2026-10-30, used by the 10-23 area-grid mode).

The 10-23 mode keeps the square grid fixed in Omega and re-solves the 3D -> Omega map.  Its interior
is a harmonic interpolation (optionally followed by a symmetric-Dirichlet refinement), which also
fights area changes.  Here the energy is the symmetric Dirichlet energy of J / s, where s is the
similarity scale sqrt(det J) clamped to the panel-size band: inside the band only the stretch ratio
r = sigma1 / sigma2 is penalised (E = 2 (r + 1/r)), so a square grid cell becomes a square panel
whose size may follow the map; outside the band the size is pulled back to the band edge.
"""
from __future__ import annotations

import numpy as np

from .optcuts_grid_sd_refinement import _metric_3d


def _terms(uv, faces, g11, g12, g22, det_src, orient, s_lo, s_hi):
    p = uv[faces]
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    rho = orient * det / det_src                      # sigma1 sigma2
    F = g11 * np.sum(a * a, 1) + 2 * g12 * np.sum(a * b, 1) + g22 * np.sum(b * b, 1)   # |J|_F^2
    return a, b, det, rho, F


def shape_first_refine(xyz, xyz_faces, uv_faces, start_uv, fixed_vertices, size_band, *, max_iter=400):
    """Return (uv, report).  ``size_band`` = (lo, hi) for the panel size relative to the global scale."""
    from .zhang2021_panel_optimizer import PanelOptimizationError, _lbfgs
    faces = np.asarray(uv_faces, int)
    uv0 = np.asarray(start_uv, float).copy()
    g11, g12, g22, det_src, w = _metric_3d(xyz, xyz_faces)
    p = uv0[faces]; a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]
    det0 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    orient = np.sign(np.sum(det0)) * np.ones(len(faces))
    if np.any(orient * det0 <= 0):
        raise ValueError("SHAPE_FIRST_START_FLIPPED")
    s0 = float(np.sqrt(np.dot(w, orient * det0 / det_src)))    # global UV / 3D scale (area weighted)
    lo, hi = float(size_band[0]), float(size_band[1])
    s_lo, s_hi = s0 / hi, s0 / lo                               # sigma = UV length / 3D length
    free = np.setdiff1d(np.arange(len(uv0)), np.asarray(fixed_vertices, int))

    def energy(uv, need_grad=True):
        a, b, det, rho, F = _terms(uv, faces, g11, g12, g22, det_src, orient, s_lo, s_hi)
        if np.any(rho <= 0):
            return np.inf, None
        s2 = np.clip(rho, s_lo ** 2, s_hi ** 2)                 # s^2, with s = clamp(sqrt(rho))
        gfac = 1.0 / s2 + s2 / rho ** 2
        E = float(np.dot(w, F * gfac))
        if not need_grad:
            return E, None
        inside = (rho > s_lo ** 2) & (rho < s_hi ** 2)
        dg = np.where(inside, -2.0 / rho ** 2, -2.0 * s2 / rho ** 3)   # d gfac / d rho (C1)
        dFa = 2 * (g11[:, None] * a + g12[:, None] * b)
        dFb = 2 * (g12[:, None] * a + g22[:, None] * b)
        drho = (orient / det_src)[:, None]
        dDa = np.stack([b[:, 1], -b[:, 0]], 1); dDb = np.stack([-a[:, 1], a[:, 0]], 1)
        ga = w[:, None] * (gfac[:, None] * dFa + (F * dg)[:, None] * drho * dDa)
        gb = w[:, None] * (gfac[:, None] * dFb + (F * dg)[:, None] * drho * dDb)
        G = np.zeros_like(uv)
        np.add.at(G, faces[:, 0], -(ga + gb)); np.add.at(G, faces[:, 1], ga); np.add.at(G, faces[:, 2], gb)
        return E, G

    def fun(z):
        uv = uv0.copy(); uv[free] = z.reshape(-1, 2)
        E, G = energy(uv)
        if not np.isfinite(E):
            return np.inf, None
        return E, G[free].ravel()

    E0, _ = energy(uv0, need_grad=False)
    try:
        z, E1, it = _lbfgs(fun, uv0[free].ravel(), max_iter=max_iter)
    except PanelOptimizationError as exc:
        raise ValueError(f"SHAPE_FIRST_OPTIMISER: {exc}") from exc
    uv = uv0.copy(); uv[free] = z.reshape(-1, 2)

    def ratio_stats(u):
        a, b, det, rho, F = _terms(u, faces, g11, g12, g22, det_src, orient, s_lo, s_hi)
        r2 = F / rho                                            # r + 1/r
        r = 0.5 * (r2 + np.sqrt(np.maximum(r2 * r2 - 4, 0)))
        size = s0 / np.sqrt(rho)
        o = np.argsort(r); c = np.cumsum(w[o])
        return float(r[o][np.searchsorted(c, 0.5)]), float(r[o][np.searchsorted(c, 0.9)]), \
            float(np.percentile(size, 5)), float(np.percentile(size, 95))
    before, after = ratio_stats(uv0), ratio_stats(uv)
    return uv, {"shape_first_energy_before": float(E0), "shape_first_energy_after": float(E1),
                "shape_first_iterations": int(it), "shape_first_size_band": [lo, hi],
                "shape_first_stretch_p50_p90_before": before[:2], "shape_first_stretch_p50_p90_after": after[:2],
                "shape_first_size_p05_p95_after": after[2:]}
