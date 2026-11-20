"""Final-frame metrics of an OptCuts dump: distortion, seams, outline loops, roundness P^2/(4 pi A), narrowest neck."""
import sys, glob, json, numpy as np
import os
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the scripts/ folder
sys.path.insert(0, S + '/os_simple')
from common import read_dump, boundary_loop
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
def sd(X, F, U):
    P = X[F]; e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]; n = np.cross(e1, e2); a3 = 0.5 * np.linalg.norm(n, axis=1)
    t1 = e1 / np.linalg.norm(e1, axis=1)[:, None]; t2 = np.cross(n / (2 * a3)[:, None], t1)
    Q = np.stack([np.stack([(e1 * t1).sum(1), (e1 * t2).sum(1)], 1), np.stack([(e2 * t1).sum(1), (e2 * t2).sum(1)], 1)], 2)
    W = U[F]; R = np.stack([W[:, 1] - W[:, 0], W[:, 2] - W[:, 0]], 2); s = np.linalg.svd(R @ np.linalg.inv(Q), compute_uv=False)
    return float(((s[:, 0]**2 + s[:, 0]**-2 + s[:, 1]**2 + s[:, 1]**-2) * a3).sum() / a3.sum()), a3.sum()
def metrics(U, X, F):
    loops = boundary_loop(F); SD, area3 = sd(X, F, U); R = np.sqrt(area3 / np.pi)
    keyc = {}
    for L in loops:
        for a, b in zip(L, L[1:] + L[:1]):
            k = tuple(sorted((tuple(np.round(X[a], 9)), tuple(np.round(X[b], 9))))); keyc.setdefault(k, []).append((a, b))
    seam = sum(np.linalg.norm(X[v[0][0]] - X[v[0][1]]) for v in keyc.values() if len(v) == 2)
    P = sum(np.linalg.norm(U[b] - U[a]) for L in loops for a, b in zip(L, L[1:] + L[:1]))
    W = U[F]; A = abs(0.5 * ((W[:, 1, 0] - W[:, 0, 0]) * (W[:, 2, 1] - W[:, 0, 1]) - (W[:, 1, 1] - W[:, 0, 1]) * (W[:, 2, 0] - W[:, 0, 0]))).sum()
    n = len(U); E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    comps = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)), directed=False)[0] - (n - len(np.unique(F)))
    # narrowest neck: outline points facing each other across the chart, far apart along the outline (same rule as the term)
    L = loops[0]; m = len(L); u = U[L]; x3 = X[L]
    seg3 = np.linalg.norm(np.roll(x3, -1, 0) - x3, axis=1); arc = np.r_[0, np.cumsum(seg3)][:-1]; Lt = seg3.sum()
    t = np.roll(u, -1, 0) - np.roll(u, 1, 0); nin = np.c_[-t[:, 1], t[:, 0]]; nin /= np.linalg.norm(nin, axis=1)[:, None]
    sgn = 1.0 if A > 0 else 1.0
    best = np.inf
    D = np.linalg.norm(u[:, None] - u[None], axis=2)
    s = np.abs(arc[:, None] - arc[None]); s = np.minimum(s, Lt - s)
    same3 = np.linalg.norm(x3[:, None] - x3[None], axis=2) < 1e-12
    with np.errstate(invalid='ignore', divide='ignore'):
        e = (u[None] - u[:, None]) / D[..., None]
        ci = (e * nin[:, None]).sum(2); cj = -(e * nin[None]).sum(2)
    D0 = float(__import__("os").environ.get("D0", "0.25")); ok = (s > 2 * D0 * R) & (np.minimum(ci, cj) > 0.2) & ~same3 & (D > 0)
    neck = float(D[ok].min() / R) if ok.any() else None
    return dict(SD=round(SD, 4), seam_rel=round(seam / R, 3), loops=[len(l) for l in loops], components=int(comps),
                roundness=round(P * P / (4 * np.pi * A) - 1, 3), narrowest_neck_over_R=None if neck is None else round(neck, 3))
if __name__ == '__main__':
    for d in sys.argv[1:]:
        fs = sorted(glob.glob(d + '/f*.bin')); it, lam, U, X, F = read_dump(fs[-1])
        print(d, 'iterations', it, metrics(U, X, F))
