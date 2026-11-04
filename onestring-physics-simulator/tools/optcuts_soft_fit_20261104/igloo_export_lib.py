"""2026-11-03: shared helpers for the Igloo page exporters (taken from export_igloo6_full.py)."""
import sys, os, glob, json, base64, re, pickle
import numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hinge_angles import angles

VARIANTS = [("plain", "igloo_plain", 0.0), ("soft", "igloo_soft_w16", 16.0), ("hard", "igloo_hard_w4096", 4096.0)]
RUNS = [(f"{g}_{v}", v, g) for g in ("fit", "clip") for v in ("plain", "soft", "hard")]
b16 = lambda a: base64.b64encode(np.asarray(a, "<u2").tobytes()).decode()
R = lambda a, d=4: np.round(np.asarray(a, float), d).tolist()

def read_frames(dump):
    fr = []
    for fn in sorted(glob.glob(os.path.join(dump, "f*.bin"))):
        b = open(fn, "rb").read()
        it, = np.frombuffer(b, np.int32, 1, 0); lam, = np.frombuffer(b, np.float64, 1, 4); nV, nF = np.frombuffer(b, np.int32, 2, 12)
        A = np.frombuffer(b, np.float64, nV * 5, 20).reshape(nV, 5); Fm = np.frombuffer(b, np.int32, nF * 3, 20 + nV * 40).reshape(nF, 3)
        fr.append(dict(it=int(it), lam=float(lam), uv=A[:, :2].copy(), x=A[:, 2:].copy(), F=Fm.copy()))
    return fr

def seams_of(X, F):
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True); bE = u[c == 1]
    keys = {}; pid = np.array([keys.setdefault(tuple(p), len(keys)) for p in np.round(X, 9)])
    ge = np.sort(pid[bE], 1); gk = ge[:, 0] * len(keys) + ge[:, 1]; gu, gc = np.unique(gk, return_counts=True); dbl = set(gu[gc == 2].tolist())
    s = bE[np.array([k in dbl for k in gk], bool)] if len(bE) else bE
    return s, (float(np.linalg.norm(X[s[:, 0]] - X[s[:, 1]], axis=1).sum() / 2) if len(s) else 0.0)

def tri_stats(X, F, uv):
    P = X[F]; e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]; a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    ux = e1 / np.linalg.norm(e1, axis=1, keepdims=True); n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True); uy = np.cross(n, ux)
    L = np.stack([np.stack([np.einsum("ij,ij->i", e1, ux), np.einsum("ij,ij->i", e1, uy)], 1),
                  np.stack([np.einsum("ij,ij->i", e2, ux), np.einsum("ij,ij->i", e2, uy)], 1)], 2)
    Q = uv[F]; U = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], 2); s = np.linalg.svd(U @ np.linalg.inv(L), compute_uv=False)
    s1, s2 = s[:, 0], s[:, 1]; area = s1 * s2; K = s1 / s2; A = a3.sum()
    return dict(SD=float(((s1**2 + s1**-2 + s2**2 + s2**-2) * a3).sum() / A), K_mean=float((K * a3).sum() / A),
                K_p99=float(np.percentile(K, 99)), area_span=float(np.percentile(area, 99) / np.percentile(area, 1))), K, area

def process(dump):
    fr = read_frames(dump); topos = []; frames = []
    allu = np.concatenate([f["uv"] for f in fr]); lo = allu.min(0); hi = allu.max(0); c = (lo + hi) / 2; h = float((hi - lo).max() / 2)
    for f in fr:
        if not topos or topos[-1]["F"].shape != f["F"].shape or not (topos[-1]["F"] == f["F"]).all() or len(topos[-1]["x"]) != len(f["x"]):
            s, slen = seams_of(f["x"], f["F"]); topos.append(dict(F=f["F"], x=f["x"], seam=s.tolist(), seamLen=slen))
        t = len(topos) - 1; st, K, ar = tri_stats(f["x"], f["F"], f["uv"])
        frames.append(dict(topo=t, it=f["it"], lam=round(f["lam"], 6),
                           uv=base64.b64encode(np.round((f["uv"] - c) / h * 32000).astype("<i2").tobytes()).decode(),
                           seamLen=round(topos[t]["seamLen"], 5), **{k: round(v, 5) for k, v in st.items()}))
    keys = {}; X0 = []
    def vid(p):
        if p not in keys: keys[p] = len(X0); X0.append(p)
        return keys[p]
    enc = []; pv = pf = None
    for i, t in enumerate(topos):
        vm = np.array([vid(tuple(p)) for p in np.round(t["x"], 6)]); f = t["F"]
        if i == 0:
            enc.append(dict(v=b16(vm), f=b16(f.ravel()), nF=len(f), seam=t["seam"], seamLen=t["seamLen"]))
        else:
            n = min(len(vm), len(pv))
            dv = [[int(j), int(vm[j])] for j in np.flatnonzero(vm[:n] != pv[:n])] + [[int(j), int(vm[j])] for j in range(n, len(vm))]
            if f.shape == pf.shape: df = [[int(j), *map(int, f[j])] for j in np.flatnonzero((f != pf).any(1))]; full = None
            else: df = None; full = b16(f.ravel())
            enc.append(dict(nV=len(vm), dv=dv, df=df, f=full, nF=len(f), seam=t["seam"], seamLen=t["seamLen"]))
        pv, pf = vm, f
    return dict(uvScale=h, X0=R(np.array(X0), 4), topos=enc, frames=frames)

def qa(V, F):
    P = np.asarray(V, float)[np.asarray(F, int)]
    if P.shape[2] == 2: P = np.concatenate([P, np.zeros(P.shape[:2] + (1,))], 2)
    return 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1)

