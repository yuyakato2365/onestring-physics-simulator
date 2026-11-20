"""Curvature-limited local INFLATION of the original 3D mesh (vertex positions only; connectivity, cuts and Omega unchanged).

criterion  sphere-equivalent radius  rho_r(v) = 1 / sqrt( max(Kbar_r(v), 0) ),  Kbar_r = (sum of angle deficits of the
           vertices within surface distance r of v) / (their area).  Only convex points more curved than a sphere of
           radius r violate it (rho_r < r); cylinder-like parts such as the edge of an ear have K ~ 0 and are left alone.
offset     variable outward offset  x' = x + delta(x) n(x):  delta = delta0 * smoothstep(1 - d / L), d = surface distance from
           the violating vertices of the ORIGINAL mesh, L = 4 delta0 (the offset changes slowly enough not to raise a
           bump of its own), n = vertex normals smoothed over about one ring; open-rim vertices do not move.  An offset
           surface has every convex radius at least the offset, so pointed tips become round caps and the ear is not
           shortened.  The moved vertices are then relaxed tangentially (uniform Laplacian displacement with its normal
           component removed, until it changes by less than 1e-6) so the triangles stay well shaped.
search     delta0 grows by 20 % from 0.1 r until the criterion holds everywhere, then bisection (relative 1e-3) for the
           smallest delta0 that satisfies it.
usage: inflate.py MESH.obj|CHART_PKL OUT_DIR   (env FAIR_R, default 0.08)
outputs: OUT_DIR/inflated.obj (input of OptCuts), chart_inflated.pkl, inflate_fields.pkl, inflate_summary.json, anim_inflation.mp4
"""
import sys, os, json, pickle, numpy as np, igl, scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the scripts/ folder
sys.path.insert(0, S + '/os_simple'); from anim import Recorder, pick_frames, setup3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
chart, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); r_ = float(os.environ.get('FAIR_R', '0.08'))
if chart.lower().endswith('.obj'):
    vs, fs = [], []
    for ln in open(chart):
        t = ln.split()
        if not t: continue
        if t[0] == 'v': vs.append([float(x) for x in t[1:4]])
        elif t[0] == 'f': fs.append([int(x.split('/')[0]) - 1 for x in t[1:4]])
    c = dict(xyz=np.array(vs), f3=np.array(fs))
else:
    c = pickle.load(open(chart, 'rb'))
V0 = np.asarray(c['xyz'], float); F = np.asarray(c['f3'], int); n = len(V0)
bf = igl.boundary_facets(F); bf = np.asarray(bf[0] if isinstance(bf, tuple) else bf); rim = np.zeros(n, bool); rim[np.unique(bf.ravel())] = True
E = igl.edges(F)
def graph(V):
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return sp.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
def sphere_radius(V):
    G = graph(V)
    k = igl.gaussian_curvature(V, F); k[rim] = 0.0
    m = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI).diagonal()
    # Keep only a bounded set of geodesic discs in memory for full-size scans.
    Kbar = np.empty(n)
    for start in range(0, n, 128):
        ids = np.arange(start, min(start + 128, n))
        inD = np.isfinite(dijkstra(G, indices=ids, limit=r_))
        Kbar[ids] = (inD @ k) / (inD @ m)
    return 1 / np.sqrt(np.maximum(Kbar, 1e-12)), G
A_ = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
deg = np.asarray(A_.sum(1)).ravel()
rho0, G0 = sphere_radius(V0); bad0 = np.where(rho0 < r_)[0]
d0 = dijkstra(G0, indices=bad0, min_only=True) if len(bad0) else np.full(n, np.inf)
N0 = igl.per_vertex_normals(V0, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
N0 = N0 + (A_ @ N0) / deg[:, None]; N0 /= np.linalg.norm(N0, axis=1)[:, None]      # smoothed over one ring
frames = [(V0.copy(), rho0.copy())]
def offset(delta0):
    L_ = 4 * delta0; t = np.clip(1 - d0 / L_, 0, 1); dl = delta0 * t * t * (3 - 2 * t); dl[rim] = 0
    V = V0 + dl[:, None] * N0; mov = dl > 0
    prev = None
    while True:                                                    # tangential relaxation of the moved vertices
        N = igl.per_vertex_normals(V, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
        dlt = (A_ @ V) / deg[:, None] - V; dlt -= (dlt * N).sum(1)[:, None] * N; dlt[~mov] = 0
        V = V + 0.5 * dlt; ch = np.abs(dlt).max()
        if ch < 1e-6 or (prev is not None and ch > prev * 0.999): break
        prev = ch
    rho, _ = sphere_radius(V); frames.append((V.copy(), rho.copy()))
    print(f'[INFLATE] delta0 {delta0:.4f}: worst r/rho {(r_ / rho).max():.3f}, violating vertices {(rho < r_).sum()}', flush=True)
    return V, rho
if len(bad0) == 0: V, rho, dsel = V0.copy(), rho0, 0.0
else:
    lo, hi = 0.0, 0.1 * r_
    while True:
        V, rho = offset(hi)
        if (rho >= r_).all(): break
        lo, hi = hi, hi * 1.2
        if hi > np.linalg.norm(np.ptp(V0, axis=0)):
            raise ValueError("指定した丸め半径をモデルの寸法内で満たせません。")
    Vb, rb = V, rho
    while (hi - lo) > 1e-3 * hi:
        mid = 0.5 * (lo + hi); V, rho = offset(mid)
        if (rho >= r_).all(): hi, Vb, rb = mid, V, rho
        else: lo = mid
    V, rho, dsel = Vb, rb, hi
crease_info=None
threshold=float(os.environ.get('TRI_CREASE_ANGLE','180'))
if threshold<180:
    from round_creases import round_creases
    V,crease_info=round_creases(V,F,threshold,r_,lambda x: np.flatnonzero(sphere_radius(x)[0]<r_) if r_ else [],
                                inflate_residual=os.environ.get("TRI_CREASE_INFLATE")=="1")
    rho,_=sphere_radius(V)
    print('[CREASE]',json.dumps(crease_info),flush=True)
it = len(frames) - 1
move = np.linalg.norm(V - V0, axis=1)
def angles(V_):
    P = V_[F]; a = []
    for i in range(3):
        u = P[:, (i + 1) % 3] - P[:, i]; v = P[:, (i + 2) % 3] - P[:, i]
        a.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1))))
    return np.array(a)
res = dict(r=r_, delta0=float(dsel), evaluations=it, worst_ratio_start=float((r_ / rho0).max()), worst_ratio_end=float((r_ / rho).max()),
           min_sphere_radius_start=float(rho0.min()), min_sphere_radius_end=float(rho.min()),
           moved_vertices=int((move > 1e-6).sum()), max_move=float(move.max()),
           min_angle_start=float(angles(V0).min()), min_angle_end=float(angles(V).min()),
           area_change=float(igl.doublearea(V, F).sum() / igl.doublearea(V0, F).sum() - 1))
if crease_info is not None:res["crease_rounding"]=crease_info
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(out, 'inflate_summary.json'), 'w'), indent=1)
c2 = dict(c); c2['xyz'] = V; pickle.dump(c2, open(os.path.join(out, 'chart_inflated.pkl'), 'wb'))
with open(os.path.join(out, 'inflated.obj'), 'w') as fo:
    for v in V: fo.write('v %.9f %.9f %.9f\n' % tuple(v))
    for t in F + 1: fo.write('f %d %d %d\n' % tuple(t))
pickle.dump(dict(V0=V0, V=V, F=F, rho0=rho0, rho=rho, r=r_), open(os.path.join(out, 'inflate_fields.pkl'), 'wb'))
# ---- animation: both ears, coloured by sphere-equivalent radius / r ----
ears = [(np.array([0.11, 0.86, -0.31]), -70), (np.array([-0.40, 0.82, -0.63]), 110)]
keep = (np.linalg.norm(V0[F].mean(1) - ears[0][0], axis=1) < 0.3) | (np.linalg.norm(V0[F].mean(1) - ears[1][0], axis=1) < 0.3)
cm = plt.get_cmap('magma')
rec = Recorder(out, 'anim_inflation')
for k in pick_frames(len(frames), 80):
    Vk, rk = frames[k]; fig = plt.figure(figsize=(11, 5.6), dpi=90)
    for j, (cc, az) in enumerate(ears):
        ax = fig.add_subplot(1, 2, j + 1, projection='3d'); sel = keep & (np.linalg.norm(V0[F].mean(1) - cc, axis=1) < 0.3)
        val = np.clip((rk[F[sel]].min(1)) / r_, 0, 2) / 2
        ax.add_collection3d(Poly3DCollection(Vk[F[sel]], facecolors=cm(0.15 + 0.8 * val), edgecolors='#30343a', linewidths=0.15))
        rr = 0.2; ax.set_xlim(cc[0] - rr, cc[0] + rr); ax.set_ylim(cc[1] - rr, cc[1] + rr); ax.set_zlim(cc[2] - rr, cc[2] + rr)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=10, azim=az, vertical_axis='y'); ax.axis('off')
        ax.set_title(['right ear', 'left ear'][j], fontsize=9)
    fig.suptitle(f'curvature-limited local inflation (r = {r_}), step {k}: sphere-equivalent radius / r (dark = sharper than r), worst r/rho {(r_ / rk).max():.2f}', fontsize=9)
    rec.add(fig)
rec.finish(fps=10)
