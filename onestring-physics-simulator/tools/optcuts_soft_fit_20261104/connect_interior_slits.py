"""2026-11-04: make an OptCuts result usable by the one-chart pipeline when OptCuts left a seam inside the surface that
does not reach the boundary (an interior slit = an extra UV boundary loop).

For every such slit:
  * take its two tips (the slit vertices that were not duplicated) and find the shortest edge path on the original
    surface from a tip to a vertex of the original surface boundary, through vertices that lie on no seam and no
    boundary (so the slit simply grows to the boundary: no branch, no junction with another seam);
  * cut the surface along that path (the vertices on it are duplicated, one copy per side);
then re-solve the UV map with the same soft objective as the OptCuts run (symmetric Dirichlet + w * (s1/s2 + s2/s1 - 2),
area weighted), starting from the OptCuts UV (the new cut starts closed), by L-BFGS with a step bound that keeps every
triangle's orientation, until the energy stops decreasing (relative decrease < 1e-10 over 50 iterations).

usage: python3 connect_interior_slits.py DUMP_DIR MESH.obj W_ANGLE [OUT_DIR]
  DUMP_DIR holds the OptCuts frames f*.bin (the last one is used); OUT_DIR (default DUMP_DIR) receives
  optcuts_chart.pkl, cut.obj, uv.npy and connect.json (and the frames of the re-solve as r*.bin-like records in connect.json)
"""
import sys, os, glob, json, pickle, heapq
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from optcuts_multistart import load_mesh

dump, mesh = sys.argv[1], sys.argv[2]; W = float(sys.argv[3]); out = sys.argv[4] if len(sys.argv) > 4 else dump
os.makedirs(out, exist_ok=True)
os.environ.setdefault("ONESTRING_TRACE_DIR", os.path.join(out, "trace"))
from onestring_physics.run_trace import trace
fn = sorted(glob.glob(os.path.join(dump, "f*.bin")))[-1]
b = open(fn, "rb").read(); nV, nF = np.frombuffer(b, np.int32, 2, 12)
A = np.frombuffer(b, np.float64, nV * 5, 20).reshape(nV, 5).copy(); F2 = np.frombuffer(b, np.int32, nF * 3, 20 + nV * 40).reshape(nF, 3).copy()
UV, X = A[:, :2].copy(), A[:, 2:].copy()
V0, F0 = load_mesh(mesh)
d, vid = cKDTree(V0).query(X); assert d.max() < 1e-6, d.max()
assert (vid[F2] == F0).all(), "face order differs from the input mesh"

def boundary_loops(F, n):
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1); u, c = np.unique(E, axis=0, return_counts=True); bE = u[c == 1]
    g = coo_matrix((np.ones(len(bE)), (bE[:, 0], bE[:, 1])), shape=(n, n)); _, lab = connected_components(g, directed=False)
    loops = {}
    for e in bE: loops.setdefault(lab[e[0]], set()).update(e.tolist())
    return sorted(loops.values(), key=len, reverse=True), bE

loops, bE2 = boundary_loops(F2, nV)
trace("connect", "input", frame=os.path.basename(fn), uv_vertices=int(nV), boundary_loops=[len(l) for l in loops])
print(f"[CONNECT] {os.path.basename(fn)}: {nV} uv vertices, {len(loops)} boundary loops (sizes {[len(l) for l in loops]})", flush=True)
copies = np.bincount(vid, minlength=len(V0))                     # >1: the vertex lies on a seam
E0 = np.sort(np.concatenate([F0[:, [0, 1]], F0[:, [1, 2]], F0[:, [2, 0]]]), 1); u0, c0 = np.unique(E0, axis=0, return_counts=True)
bnd0 = set(u0[c0 == 1].ravel().tolist())                        # original surface boundary
nbr = [[] for _ in range(len(V0))]
for a, c in u0: L = float(np.linalg.norm(V0[a] - V0[c])); nbr[a].append((c, L)); nbr[c].append((a, L))
seam_touch = set(np.flatnonzero(copies > 1).tolist())
targets = {v for v in bnd0 if v not in seam_touch}               # boundary vertices that are not seam mouths
cut_edges = set(); report = []
for li, loop in enumerate(loops[1:], 1):
    ov = [int(vid[v]) for v in loop]; cnt = {}
    for o in ov: cnt[o] = cnt.get(o, 0) + 1
    tips = [o for o, k in cnt.items() if k == 1]                 # the slit ends (not duplicated)
    assert len(tips) == 2, (li, len(tips))
    dist = {t: 0.0 for t in tips}; prev = {}; pq = [(0.0, t) for t in tips]; heapq.heapify(pq); end = None
    while pq:
        dv, v = heapq.heappop(pq)
        if dv > dist.get(v, np.inf): continue
        if v in targets: end = v; break
        for w, L in nbr[v]:
            if w in seam_touch and w not in tips: continue      # never touch another seam (or this slit's sides)
            if w in bnd0 and w not in targets: continue
            nd = dv + L
            if nd < dist.get(w, np.inf): dist[w] = nd; prev[w] = v; heapq.heappush(pq, (nd, w))
    assert end is not None, "no path to the boundary"
    path = [end]
    while path[-1] in prev: path.append(prev[path[-1]])
    path = path[::-1]
    for a, c in zip(path[:-1], path[1:]): cut_edges.add((min(a, c), max(a, c)))
    report.append(dict(loop=li, slit_vertices=len(loop), tip=int(path[0]), end=int(path[-1]), path_edges=len(path) - 1, path_length=float(dist[end]),
                       tip_xyz=V0[path[0]].round(4).tolist(), end_xyz=V0[path[-1]].round(4).tolist()))
    trace("connect", "slit", **report[-1], path=path)
    print(f"[CONNECT] slit {li} ({len(loop)} loop vertices): tip {path[0]} -> boundary vertex {path[-1]}, {len(path) - 1} edges, length {dist[end]:.4f}", flush=True)

# ---- cut the UV mesh along the new edges: for every uv vertex, faces around it joined across edges that are not cut ----
inc = [[] for _ in range(nV)]
for t, f in enumerate(F2):
    for k in range(3): inc[f[k]].append(t)
F2n = F2.copy(); UVn = [UV[i] for i in range(nV)]; Vmap = list(vid)
for v in range(nV):
    ts = inc[v]
    if len(ts) < 2: continue
    o = vid[v]; touched = any((min(o, w), max(o, w)) in cut_edges for w in (vid[x] for t in ts for x in F2[t]) if w != o)
    if not touched: continue
    par = {t: t for t in ts}
    def fd(t):
        while par[t] != t: par[t] = par[par[t]]; t = par[t]
        return t
    for i, t in enumerate(ts):
        for s in ts[i + 1:]:
            sh = set(F2[t].tolist()) & set(F2[s].tolist()); sh.discard(v)
            for w in sh:                                          # t and s share the edge (v, w)
                if (min(o, vid[w]), max(o, vid[w])) not in cut_edges: par[fd(t)] = fd(s)
    groups = {}
    for t in ts: groups.setdefault(fd(t), []).append(t)
    for gi, g in enumerate(sorted(groups.values(), key=len, reverse=True)):
        if gi == 0: continue
        nvid = len(UVn); UVn.append(UV[v].copy()); Vmap.append(o)
        for t in g: F2n[t][F2n[t] == v] = nvid
UVn = np.array(UVn); Vmap = np.array(Vmap)
loops_n, _ = boundary_loops(F2n, len(UVn))
print(f"[CONNECT] after the cut: {len(UVn)} uv vertices, {len(loops_n)} boundary loop(s)", flush=True)
assert len(loops_n) == 1

# ---- re-solve the UV map: area-weighted  f + f/d^2 + W (f/d - 2)  with f = |J|_F^2, d = det J ----
P = V0[F0]; e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]; a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1); At = a3.sum()
ux = e1 / np.linalg.norm(e1, axis=1, keepdims=True); nn = np.cross(e1, e2); nn /= np.linalg.norm(nn, axis=1, keepdims=True); uy = np.cross(nn, ux)
Lr = np.stack([np.stack([(e1 * ux).sum(1), (e1 * uy).sum(1)], 1), np.stack([(e2 * ux).sum(1), (e2 * uy).sum(1)], 1)], 2)   # columns = edges
Li = np.linalg.inv(Lr)
def jac(U):
    Q = U[F2n]; B = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], 2); return B @ Li
sgn = np.sign(np.median(np.linalg.det(jac(UVn))))
if sgn < 0: UVn[:, 0] *= -1                                      # work with positive orientation, flip back at the end
def energy(U, grad=True, parts=False):
    J = jac(U); f = (J ** 2).sum((1, 2)); dt = np.linalg.det(J)
    if (dt <= 0).any(): return np.inf, None
    e = f + f / dt ** 2 + W * (f / dt - 2.0); E = float((a3 * e).sum() / At)
    if parts: return E, float((a3 * (f + f / dt ** 2)).sum() / At)
    if not grad: return E, None
    cof = np.stack([np.stack([J[:, 1, 1], -J[:, 1, 0]], 1), np.stack([-J[:, 0, 1], J[:, 0, 0]], 1)], 1)
    dE = 2 * J * (1 + 1 / dt ** 2 + W / dt)[:, None, None] + (f * (-2 / dt ** 3 - W / dt ** 2))[:, None, None] * cof
    dB = (a3 / At)[:, None, None] * (dE @ np.transpose(Li, (0, 2, 1)))
    g = np.zeros_like(U); np.add.at(g, F2n[:, 1], dB[:, :, 0]); np.add.at(g, F2n[:, 2], dB[:, :, 1]); np.add.at(g, F2n[:, 0], -dB[:, :, 0] - dB[:, :, 1])
    return E, g
def max_step(U, D):
    Q = U[F2n]; R = D[F2n]; B0 = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], 2); B1 = np.stack([R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]], 2)
    a = np.linalg.det(B1); c = np.linalg.det(B0); bq = B0[:, 0, 0] * B1[:, 1, 1] + B1[:, 0, 0] * B0[:, 1, 1] - B0[:, 0, 1] * B1[:, 1, 0] - B1[:, 0, 1] * B0[:, 1, 0]
    t = np.full(len(a), np.inf)
    lin = np.abs(a) < 1e-300; m = lin & (bq < 0); t[m] = -c[m] / bq[m]
    q = ~lin; disc = bq[q] ** 2 - 4 * a[q] * c[q]; ok = disc >= 0; r = np.full(q.sum(), np.inf)
    s = np.sqrt(np.where(ok, disc, 0)); r1 = (-bq[q] - s) / (2 * a[q]); r2 = (-bq[q] + s) / (2 * a[q])
    for rr in (r1, r2): r = np.where(ok & (rr > 0) & (rr < r), rr, r)
    t[q] = r; return float(t.min())
E0_, SD0 = energy(UVn, parts=True); x = UVn.copy(); E, g = energy(x); S, Y = [], []; hist = [E]; frames = [x.copy()]
for it in range(1, 200001):
    q = g.copy(); al = []
    for s_, y_ in reversed(list(zip(S, Y))):
        r_ = 1.0 / (y_ * s_).sum(); a_ = r_ * (s_ * q).sum(); q -= a_ * y_; al.append((r_, a_))
    if S: q *= (S[-1] * Y[-1]).sum() / (Y[-1] * Y[-1]).sum()
    for (s_, y_), (r_, a_) in zip(zip(S, Y), reversed(al)): q += s_ * (a_ - r_ * (y_ * q).sum())
    D = -q
    if (D * g).sum() >= 0: D = -g; S, Y = [], []
    t = min(1.0, 0.8 * max_step(x, D)); En = np.inf
    while t > 1e-20:
        xn = x + t * D; En, gn = energy(xn)
        if En <= E + 1e-4 * t * (D * g).sum(): break
        t *= 0.5
    if not np.isfinite(En) or En > E: break
    S.append(xn - x); Y.append(gn - g); S, Y = S[-10:], Y[-10:]; x, E, g = xn, En, gn; hist.append(E)
    if it % 25 == 0: frames.append(x.copy())
    if len(hist) > 50 and hist[-51] - hist[-1] < 1e-10 * abs(hist[-1]): break
frames.append(x.copy())
E1, SD1 = energy(x, parts=True)
print(f"[CONNECT] UV re-solve: {it} L-BFGS iterations, objective {E0_:.5f} -> {E1:.5f}, SD {SD0:.5f} -> {SD1:.5f}", flush=True)
if sgn < 0: x[:, 0] *= -1; frames = [np.c_[-fr[:, 0], fr[:, 1]] for fr in frames]
# simple outer boundary (no global overlap)?
from shapely.geometry import Polygon
E_ = np.sort(np.concatenate([F2n[:, [0, 1]], F2n[:, [1, 2]], F2n[:, [2, 0]]]), 1)
dE_ = {}
for f in F2n:
    for a, c in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])): dE_[(a, c)] = True
nxt = {a: c for (a, c) in dE_ if (c, a) not in dE_}
start = next(iter(nxt)); ring = [start]
while nxt[ring[-1]] != start: ring.append(nxt[ring[-1]])
simple = Polygon(x[ring]).is_valid
trace("connect", "resolve_lbfgs", iterations=int(it), objective_before=E0_, objective_after=E1, SD_before=SD0, SD_after=SD1)
print(f"[CONNECT] outer boundary {len(ring)} vertices, simple polygon: {simple}", flush=True)
trace("connect", "outer_boundary", vertices=len(ring), simple=bool(simple))
chart = dict(xyz=V0, f3=F0, uv=x, f2=F2n)
pickle.dump(chart, open(os.path.join(out, "optcuts_chart.pkl"), "wb"))
Vc = np.zeros((len(x), 3)); Vc[F2n.ravel()] = V0[F0.ravel()]
trimesh.Trimesh(Vc, F2n, process=False).export(os.path.join(out, "cut.obj")); np.save(os.path.join(out, "uv.npy"), x)
json.dump(dict(source_frame=os.path.basename(fn), w_angle=W, slits=report, uv_vertices_before=int(nV), uv_vertices_after=int(len(x)),
               objective_before=E0_, objective_after=E1, SD_before=SD0, SD_after=SD1, iterations=int(it), outer_boundary_simple=bool(simple),
               energy_history=[float(h) for h in hist[::max(1, len(hist) // 400)]],
               resolve_frames=[np.round(fr, 6).tolist() for fr in frames[:: max(1, len(frames) // 40)]] if False else None),
          open(os.path.join(out, "connect.json"), "w"))
np.save(os.path.join(out, "connect_frames.npy"), np.array(frames, np.float32))
pickle.dump(dict(F2=F2n, vmap=Vmap, path_edges=sorted(cut_edges)), open(os.path.join(out, "connect_topology.pkl"), "wb"))
print("[CONNECT] saved", flush=True)
