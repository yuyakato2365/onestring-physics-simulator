"""Interactive 3D page (three.js) of the final M3D: panels coloured by angle error / size, the 3D surface (ghost), the
OptCuts seams and the panel vertices on seams.
usage: build_viewer.py RUN_DIR (with grid/result.pkl and remesh3d.pkl) CHART_PKL (OptCuts result of the same surface) OUT.html
The page loads three.js r128 + OrbitControls 0.128.0 from cdn.jsdelivr.net.  Coordinates are scaled by 0.6469 and the
"size" colour is the mean panel edge x 6.749, as in the reference page."""
import os, sys, re, json, pickle, numpy as np
run, chart, out_html = sys.argv[1:4]; HERE = os.path.dirname(os.path.abspath(__file__)); SC = 0.6469
def bedges(T):
    E = np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1); u, c = np.unique(E, axis=0, return_counts=True); return u[c == 1]
c = pickle.load(open(chart, 'rb')); U = np.asarray(c['uv']); F2 = np.asarray(c['f2']); X3 = np.asarray(c['xyz'], float); F3 = np.asarray(c['f3'])
X = np.zeros((len(U), 3)); X[F2.ravel()] = X3[F3.ravel()]
key = {}
for a, b in bedges(F2): key.setdefault(tuple(sorted((tuple(np.round(X[a], 9)), tuple(np.round(X[b], 9))))), []).append((a, b))
seams = [np.round(X[q] * SC, 4).tolist() for v in key.values() if len(v) == 2 for q in v[0]]
rim = [np.round(X3[q] * SC, 4).tolist() for e in bedges(F3) for q in e]
M = pickle.load(open(os.path.join(run, 'remesh3d.pkl'), 'rb')); Xp, T, kv = M['X'], M['T'], M['kept_vertices']
R = pickle.load(open(os.path.join(run, 'grid', 'result.pkl'), 'rb')); w = np.asarray(R['wid'])
sv0 = set(int(w[i]) for i in set(bedges(np.asarray(R['Tg'])).ravel())) - set(bedges(np.asarray(R['TW'])).ravel())
rm = {int(o): i for i, o in enumerate(kv)}; seamv = sorted(rm[i] for i in sv0 if i in rm)
P = Xp[T]
def ang(u, v): return np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))
Aa = np.stack([ang(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), ang(P[:, 2] - P[:, 1], P[:, 0] - P[:, 1]), ang(P[:, 0] - P[:, 2], P[:, 1] - P[:, 2])], 1)
el = np.linalg.norm(P - P[:, [1, 2, 0]], axis=2).mean(1)
b = set(bedges(T).ravel()); E = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1), axis=0)
deg = np.zeros(len(Xp), int); np.add.at(deg, E.ravel(), 1); vals = {}
for i in range(len(Xp)):
    if i not in b: vals[int(deg[i])] = vals.get(int(deg[i]), 0) + 1
st = dict(panels=len(T), interior6=vals.get(6, 0), sd_final=None, dev_mean=round(float(np.abs(Aa - 60).mean()), 1),
          dev_p95=round(float(np.percentile(np.abs(Aa - 60), 95)), 1), seam_gap_max=json.load(open(os.path.join(run, 'grid', 'summary.json')))['seam_partner_gap_max'],
          other_valence=', '.join(f'{k}:{v}' for k, v in sorted(vals.items()) if k != 6), size_ratio=round(float(np.percentile(el, 95) / np.percentile(el, 5)), 2))
from pathlib import Path
settings = json.loads((Path(run) / 'settings.json').read_text())
size_label = 'Uniform target edge length' if settings['mode'] == 'uniform' else 'Curvature and thickness sizing'
run_label = size_label + ' · Rolling Guidance ' + ('ON' if settings['rolling_guidance'] else 'OFF')
data = dict(surface=dict(v=np.round(X3 * SC, 4).tolist(), f=F3.tolist()), seams=seams, rim=rim,
            runs=dict(runK=dict(label=run_label, v=np.round(Xp * SC, 4).tolist(), f=T.tolist(), dev=np.round(np.abs(Aa - 60).max(1), 1).tolist(),
                                seamv=seamv, size=np.round(el / np.median(el), 3).tolist(), stats=st)))
h = open(os.path.join(HERE, 'viewer_head.html')).read(); t = open(os.path.join(HERE, 'viewer_tail.html')).read()
h = re.sub(r'\s*<button id="b-run(?!K")[^"]*"[^>]*>.*?</button>', '', h)                      # keep only the final run's button
t = re.sub(r"document\.getElementById\('b-run(?!K')[^']*'\)\.onclick=\(\)=>setRun\('[^']*'\);\n", '', t)
t = re.sub(r"\['runK'(,'[^']*')*\]\.forEach", "['runK'].forEach", t)
h = h.replace('RUN_SETTINGS', run_label)
vendor = Path(HERE).parents[1] / 'vendor'
h = re.sub(r'<link[^>]+fonts.googleapis.com[^>]*>', '', h)
h = re.sub(r'<script src="[^"]+/(three.min.js|OrbitControls.js)"></script>',
           lambda m: '<script>' + (vendor/m[1]).read_text() + '</script>', h)
open(out_html, 'w').write(h + json.dumps(data) + t); print('wrote', out_html, st)
