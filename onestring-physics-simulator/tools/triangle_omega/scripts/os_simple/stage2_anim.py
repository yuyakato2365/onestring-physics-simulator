"""Stage 2 animation: every dumped OptCuts iteration (UV chart + seams on the 3D surface)."""
import sys, os, glob, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
sys.path.insert(0, os.path.dirname(__file__))
from common import read_dump, boundary_loop
from anim import Recorder, setup3d, draw_mesh3d, pick_frames, SEAM

dump_dir, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(dump_dir, "f*.bin")))
rec = Recorder(out, "anim_2_optcuts")
first = read_dump(files[0])
bounds = (first[3].min(0), first[3].max(0))
for i in pick_frames(len(files), 90):
    it, lam, U, X, F = read_dump(files[i])
    loops = boundary_loop(F)
    # seam = UV boundary edges whose 3D edge is interior (two UV copies)
    seg2, seg3 = [], []
    keyc = {}
    for L in loops:
        for a, b in zip(L, L[1:] + L[:1]):
            k = tuple(sorted((tuple(np.round(X[a], 9)), tuple(np.round(X[b], 9)))))
            keyc[k] = keyc.get(k, 0) + 1
    for L in loops:
        for a, b in zip(L, L[1:] + L[:1]):
            k = tuple(sorted((tuple(np.round(X[a], 9)), tuple(np.round(X[b], 9)))))
            if keyc[k] == 2:
                seg2.append(U[[a, b]]); seg3.append(X[[a, b]])
            else:
                seg2.append(U[[a, b]])
    fig = plt.figure(figsize=(12, 4.6), dpi=90)
    ax = fig.add_subplot(1, 3, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.add_collection(PolyCollection(U[F], facecolor="#e3e9ef", edgecolor="#c0c9d2", lw=0.1))
    ax.add_collection(LineCollection(seg2, colors="#1d2733", lw=0.8))
    ax.autoscale()
    seam_len = sum(np.linalg.norm(s[1] - s[0]) for s in seg3) / 2
    for j, (el, az) in enumerate(((12, -62), (12, 118))):
        a3 = fig.add_subplot(1, 3, 2 + j, projection="3d"); setup3d(a3, bounds, el, az)
        draw_mesh3d(a3, X, F, lines=[(s, SEAM, 1.8) for s in seg3])
    fig.suptitle(f"OptCuts (sigma1 = sigma2 penalty, bound on plain SD 4.1)  iteration {it}   lambda {lam:.4f}   "
                 f"seam length {seam_len*100:.1f} cm", fontsize=10)
    fig.subplots_adjust(0, 0, 1, 0.92, 0, 0)
    rec.add(fig)
rec.finish(fps=12)
print("frames", rec.n)
