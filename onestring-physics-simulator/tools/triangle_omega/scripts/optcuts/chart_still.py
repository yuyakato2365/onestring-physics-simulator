"""Still image of an OptCuts result: Omega coloured by the local shrink (CSF) and the seams on the 3D surface.
usage: chart_still.py DIR_WITH_result.pkl   -> DIR/view/final_chart.png"""
import sys, pickle, numpy as np
import os
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the scripts/ folder
sys.path.insert(0, S + '/os_simple'); sys.path.insert(0, S + '/neck')
from common import boundary_loop
from analyze import metrics
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from anim import setup3d, draw_mesh3d
D = sys.argv[1]; os.makedirs(os.path.join(D, 'view'), exist_ok=True)
r = pickle.load(open(os.path.join(D, 'result.pkl'), 'rb')); U, F = np.asarray(r['uv']), np.asarray(r['f2'])
X = np.zeros((len(U), 3)); X[F.ravel()] = np.asarray(r['xyz'])[np.asarray(r['f3']).ravel()]
mt = metrics(U, X, F)
P = X[F]; a3 = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
W = U[F]; a2 = 0.5 * np.abs((W[:, 1, 0] - W[:, 0, 0]) * (W[:, 2, 1] - W[:, 0, 1]) - (W[:, 1, 1] - W[:, 0, 1]) * (W[:, 2, 0] - W[:, 0, 0]))
csf = np.sqrt(a3 / a2) / np.sqrt(a3.sum() / a2.sum())
print(mt, 'CSF max', csf.max().round(3), 'p95', np.percentile(csf, 95).round(3), 'min', csf.min().round(3))
loops = boundary_loop(F)
keyc = {}
for L in loops:
    for a, b in zip(L, L[1:] + L[:1]):
        k = tuple(sorted((tuple(np.round(X[a], 9)), tuple(np.round(X[b], 9))))); keyc.setdefault(k, []).append((a, b))
seg = [v[0] for v in keyc.values() if len(v) == 2]
fig = plt.figure(figsize=(14, 5.6), dpi=100)
ax = fig.add_axes([0.0, 0.15, 0.33, 0.72]); ax.set_aspect('equal'); ax.axis('off')
pc = PolyCollection(U[F], array=csf, cmap='viridis', edgecolor='none'); pc.set_clim(csf.min(), csf.max()); ax.add_collection(pc)
for L in loops: ax.add_collection(LineCollection([U[[a, b]] for a, b in zip(L, L[1:] + L[:1])], colors='#1d2733', lw=0.8))
ax.autoscale(); cax = fig.add_axes([0.05, 0.08, 0.22, 0.025]); cb = fig.colorbar(pc, cax=cax, orientation='horizontal'); cb.set_label('CSF (local shrink relative to the chart average)', fontsize=8)
for j, az in enumerate((-62, 118)):
    a3x = fig.add_subplot(1, 3, 2 + j, projection='3d'); setup3d(a3x, (X.min(0), X.max(0)), 12, az)
    draw_mesh3d(a3x, X, F, lines=[(X[[a, b]], '#c8324f', 1.6) for a, b in seg])
fig.suptitle(f"bunny_open_bottom_repaired, OptCuts as in the 2026-10-23 area-threshold mode (official objective, b_d 4.1, lambda_init 0.999, bijective)\n"
             f"SD {mt['SD']:.3f}, seams {mt['seam_rel']:.2f} R, pieces {mt['components']}, outline loops {len(mt['loops'])}, roundness {mt['roundness']:.2f}, "
             f"CSF max {csf.max():.2f} (CSF-tail post-step: not adopted, nothing above 1.8)", fontsize=10)
fig.savefig(os.path.join(D, 'view', 'final_chart.png'))
