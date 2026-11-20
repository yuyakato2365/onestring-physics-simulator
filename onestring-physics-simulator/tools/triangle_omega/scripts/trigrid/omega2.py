"""Omega2: Omega re-mapped (angle preserving) so that a UNIFORM lattice on Omega2 gives panels whose 3D size follows the
surface curvature.

1. target 3D panel size s*(x).  Principal curvatures k1, k2 are measured on the fine 3D mesh (quadric fit, radius 3
   rings; open-rim vertices take their interior neighbours' values).  Two criteria, each evaluated with the curvature
   AVERAGED OVER THE PANEL'S OWN SIZE (heat smoothing at a set of scales, the scale chosen self-consistently):
     bending criterion   s <= c sqrt(R / (kmax_s + 1/R))        chord deviation of a panel ~ s^2 kmax; a cylinder such as
                                                                 the side of an ear is limited only by this
     Gaussian criterion  s <= c / sqrt(|K_s| + 1/R^2)           K_s = SIGNED Gaussian curvature averaged over the panel;
                                                                 the angle deficit a vertex has to absorb is ~ K_s s^2, so
                                                                 caps (ear tips, K > 0) and saddles (ear roots, K < 0) get
                                                                 small panels while noise of either sign cancels out
     thickness criterion s <= THICK_FACTOR * t(x)               t = local thickness of the shape (shape diameter: median
                                                                 length of rays shot inward within 30 degrees of -normal
                                                                 to the opposite side), so a panel is never wider than a
                                                                 thin part such as the ear and cannot cut across its edge
                                                                 (THICK_FACTOR default 1; this criterion is absolute and
                                                                 is not rescaled by c)
   R = sqrt(area / pi) (both criteria agree on a sphere of radius R); c is found by bisection so that the field asks for
   NP (default 500) panels.  The field is capped at max / SIZE_RATIO (default 8) and made smooth by a growth limit
     s*(x) = min_y [ s(y) + GRADE * d(x, y) ]   (d = edge-path distance on the 3D mesh, GRADE default 0.3),
   which keeps the small sizes at the ear tips and lets the size grow by at most about 30 % per panel away from them.
2. target spacing on Omega   sigma(u) = s*(x(u)) / lambda(u),  lambda = local Omega -> 3D scale sqrt(A3 / A2).
3. phi = harmonic function closest to log sigma (least squares weighted by the number of panels each region receives,
   area / sigma^2; unknowns = phi on Omega's outline, interior = harmonic extension with the cotangent Laplacian of the
   planar Omega mesh).  A harmonic phi keeps the lattice equilateral with every interior vertex 6-valent; its extremes
   lie on the outline (the ear tips are on it).
4. Omega2: Omega's triangles scaled by exp(-phi) and assembled with 2D ARAP (local-global) until the relative energy
   change is below 1e-9; a uniform lattice on Omega2 has spacing proportional to exp(phi) on Omega.
usage: omega2.py OMEGA_SMOOTH_UV_PKL CHART_PKL OUT_DIR
"""
import sys, os, json, pickle, numpy as np, igl
from omega2_metric import relative_stretch
import scipy.sparse as sp, scipy.sparse.linalg as spla
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the scripts/ folder
sys.path.insert(0, S + '/os_simple')
from common import CutMesh
from anim import Recorder, pick_frames
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

u2pkl, chart, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
RATIO = float(os.environ.get('SIZE_RATIO', '8')); GRADE = float(os.environ.get('GRADE', '0.3')); NP = int(os.environ.get('NP', '500')); THICK_FACTOR = float(os.environ.get('THICK_FACTOR', '1.0'))
r = pickle.load(open(chart, 'rb'))
cm = CutMesh(r['xyz'], r['uv'], r['f3'], r['f2'])
U = pickle.load(open(u2pkl, 'rb')); F = cm.F; X = cm.X; V3, F3 = cm.V3, cm.FV
n = len(U); L = cm.loop; onb = np.zeros(n, bool); onb[L] = True
def cross2(a_, b_): return a_[..., 0] * b_[..., 1] - a_[..., 1] * b_[..., 0]
A2t = 0.5 * cross2(U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]])
A3t = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
# ---- 1. curvature and target 3D size on the 3D surface ----
NP_area = igl.doublearea(V3, F3).sum() / 2
uniform = os.environ.get('TRI_SIZE_MODE', 'adaptive') == 'uniform'
if uniform:
    # One equilateral triangle has area sqrt(3) * edge**2 / 4.
    edge = np.sqrt(4 * NP_area / (np.sqrt(3) * NP))
    s_star = np.full(len(V3), edge)
    s0 = s_star.copy()
    kmax = Kabs = s_b = s_g = thick = kap_s = np.full(len(V3), np.nan)
    thick_active = np.zeros(len(V3), dtype=bool)
    c_ = None
else:
    pd1, pd2, k1, k2, _ = igl.principal_curvature(V3, F3, radius=3)
    kmax_raw = np.maximum(np.abs(k1), np.abs(k2)); K_raw = k1 * k2
    E3 = igl.edges(F3); n3 = len(V3)
    Adj = sp.coo_matrix((np.ones(2 * len(E3)), (np.r_[E3[:, 0], E3[:, 1]], np.r_[E3[:, 1], E3[:, 0]])), shape=(n3, n3)).tocsr()
    bf = igl.boundary_facets(F3); bf = np.asarray(bf[0] if isinstance(bf, tuple) else bf)
    rim = np.zeros(n3, bool); rim[np.unique(bf.ravel())] = True
    def fill_rim(v):                                  # quadric fits on the open rim are unreliable: take interior neighbours' values
        v = v.copy(); known = ~rim
        while not known.all():
            cnt = Adj @ known.astype(float); acc = Adj @ np.where(known, v, 0.0)
            new = (~known) & (cnt > 0); v[new] = acc[new] / cnt[new]; known = known | new
        return v
    kmax_raw, K_raw = fill_rim(kmax_raw), fill_rim(K_raw)
    Lc = -igl.cotmatrix(V3, F3); Mm = igl.massmatrix(V3, F3, igl.MASSMATRIX_TYPE_VORONOI); mv = Mm.diagonal()
    h3 = igl.edge_lengths(V3, F3).mean()
    A3tot = igl.doublearea(V3, F3).sum() / 2; R3 = np.sqrt(A3tot / np.pi)
    scales = h3 * 1.2 ** np.arange(0, 60); scales = scales[scales < R3]       # candidate panel sizes
    km_s, K_s = [], []
    for r_ in scales:                                  # curvature averaged over a disc with the area of one vertex star
        rho = r_ * np.sqrt(np.sqrt(3) / (2 * np.pi)); slv = spla.factorized((Mm + rho ** 2 * Lc).tocsc())
        km_s.append(slv(Mm @ kmax_raw)); K_s.append(slv(Mm @ K_raw))
    km_s, K_s = np.array(km_s), np.array(K_s)          # (scales, vertices)
    lim_b = np.sqrt(R3 / (np.maximum(km_s, 0) + 1 / R3)); lim_g = 1 / np.sqrt(np.abs(K_s) + 1 / R3 ** 2)
    # ---- local thickness (shape diameter function) ----
    import trimesh
    tm = trimesh.Trimesh(V3, F3, process=False); NV3 = igl.per_vertex_normals(V3, F3); FN3 = igl.per_face_normals(V3, F3, np.array([0., 0., 1.]))
    dirs = [np.zeros(2)] + [0.5 * np.array([np.cos(a_), np.sin(a_)]) for a_ in np.linspace(0, 2 * np.pi, 8, endpoint=False)]  # tan 30deg ~ 0.58
    t1 = np.cross(NV3, np.where(np.abs(NV3[:, :1]) < 0.9, [[1., 0, 0]], [[0., 1, 0]])); t1 /= np.linalg.norm(t1, axis=1)[:, None]; t2 = np.cross(NV3, t1)
    hits = np.full((n3, len(dirs)), np.inf)
    for k, (a_, b_) in enumerate(dirs):
        D_ = -NV3 + a_ * t1 + b_ * t2; D_ /= np.linalg.norm(D_, axis=1)[:, None]
        loc, ir, it_ = tm.ray.intersects_location(V3 - 1e-5 * NV3, D_, multiple_hits=False)
        good = (FN3[it_] * D_[ir]).sum(1) > 0                                     # leaves the solid through the opposite side
        hits[ir[good], k] = np.linalg.norm(loc[good] - V3[ir[good]], axis=1)
    thick = np.array([np.median(h[np.isfinite(h)]) if np.isfinite(h).sum() >= 3 else np.inf for h in hits])
    fin_ = np.isfinite(thick); thick[fin_] = np.maximum(thick[fin_], h3)
    s_thick = THICK_FACTOR * thick
    w3 = np.linalg.norm(V3[E3[:, 0]] - V3[E3[:, 1]], axis=1)
    from scipy.sparse.csgraph import dijkstra
    def size_field(c):
        lim = c * np.minimum(lim_b, lim_g)                                     # allowed size when averaged at scale r
        ok = lim >= scales[:, None]                                            # scale r is self-consistent
        j = np.where(ok.any(0), ok.shape[0] - 1 - np.argmax(ok[::-1], 0), 0)  # largest consistent scale
        s_ = np.where(ok.any(0), np.minimum(lim[j, np.arange(n3)], scales[np.minimum(j + 1, len(scales) - 1)]), scales[0])
        s_ = np.minimum(s_, s_thick)                                           # never wider than the local thickness
        s_ = np.clip(s_, s_.max() / RATIO, None)
        Gd = sp.coo_matrix((np.r_[w3, w3, s_ / GRADE], (np.r_[E3[:, 0], E3[:, 1], np.full(n3, n3)], np.r_[E3[:, 1], E3[:, 0], np.arange(n3)])),
                           shape=(n3 + 1, n3 + 1)).tocsr()
        ss = GRADE * dijkstra(Gd, indices=n3)[:n3]
        gauss = (lim_g[j, np.arange(n3)] < lim_b[j, np.arange(n3)])
        return s_, ss, gauss, j
    def n_panels(ss): return float(np.sum(mv / (np.sqrt(3) / 4 * ss ** 2)))
    lo_c, hi_c = 1e-3, 10.0
    while hi_c / lo_c > 1 + 1e-6:                                                # bisection on c for NP panels
        mid = np.sqrt(lo_c * hi_c); (lo_c, hi_c) = (mid, hi_c) if n_panels(size_field(mid)[1]) > NP else (lo_c, mid)
    c_ = np.sqrt(lo_c * hi_c); s0, s_star, gauss, jsc = size_field(c_)
    kmax = km_s[jsc, np.arange(n3)]; Kabs = np.abs(K_s[jsc, np.arange(n3)]); kap_s = kmax
    s_b = c_ * lim_b[jsc, np.arange(n3)]; s_g = c_ * lim_g[jsc, np.arange(n3)]
    thick_active = s_thick <= c_ * np.minimum(lim_b, lim_g)[jsc, np.arange(n3)]
    print('[OMEGA2] size field: c %.4f, panels %.0f, s0 ratio %.2f, s* ratio %.2f, Gaussian criterion active on %.0f %%, thickness criterion on %.0f %% of the area' % (
        c_, n_panels(s_star), s0.max() / s0.min(), s_star.max() / s_star.min(), 100 * (mv * gauss).sum() / mv.sum(), 100 * (mv * thick_active).sum() / mv.sum()), flush=True)
# ---- 2. target spacing on Omega ----
lam_t = np.sqrt(A3t / np.abs(A2t))
Av = np.zeros(n); np.add.at(Av, F.ravel(), np.repeat(np.abs(A2t) / 3, 3))
lam_v = np.zeros(n); np.add.at(lam_v, F.ravel(), np.repeat(lam_t * np.abs(A2t) / 3, 3)); lam_v /= Av
vid = cm.v_of
psi = np.log(s_star[vid]) - np.log(lam_v)
# ---- 3. harmonic least squares ----
Lp = -igl.cotmatrix(np.c_[U, np.zeros(n)], F).tocsr()
I_ = np.where(~onb)[0]; B_ = np.where(onb)[0]
solveI = spla.factorized(Lp[I_][:, I_].tocsc())
LIB = Lp[I_][:, B_]
Hm = np.zeros((n, len(B_))); Hm[B_, np.arange(len(B_))] = 1.0
cols = LIB.toarray()
for j in range(len(B_)): Hm[I_, j] = -solveI(cols[:, j])
wp = Av * np.exp(-2 * psi); W = np.sqrt(wp / wp.sum())          # weight = panels per region
# small smoothness term on the outline values (second differences along the loop) keeps them from oscillating
pos = {v: i for i, v in enumerate(B_)}; m = len(L); D2 = sp.lil_matrix((m, len(B_)))
for i in range(m): D2[i, pos[L[i - 1]]] += 1; D2[i, pos[L[i]]] -= 2; D2[i, pos[L[(i + 1) % m]]] += 1
Amat = np.vstack([W[:, None] * Hm, 1e-3 * D2.toarray()]); rhs = np.r_[W * psi, np.zeros(m)]
b = np.linalg.lstsq(Amat, rhs, rcond=None)[0]; phi = Hm @ b
resid = phi - psi
fit = dict(psi_range=float(np.ptp(psi)), phi_range=float(np.ptp(phi)), residual_rms=float(np.sqrt(np.sum(Av * resid ** 2) / Av.sum())),
           residual_p95_abs=float(np.percentile(np.abs(resid), 95)), size_ratio_target=float(np.exp(np.ptp(np.log(s_star)))),
           harmonic_fit_ratio=float(np.exp(np.ptp(phi))))
print('[OMEGA2] fit', fit, flush=True)
# ---- 4. ARAP assembly of the scaled triangles ----
strength = float(os.environ.get('TRI_STRETCH', '1'))
phi = strength * phi
phit = phi[F].mean(1); sc = np.exp(-phit)
E1 = (U[F[:, 1]] - U[F[:, 0]]) * sc[:, None]; E2 = (U[F[:, 2]] - U[F[:, 0]]) * sc[:, None]
Rest = np.stack([E1, E2], 2)                                            # 2x2 rest edge matrix per triangle
Rinv = np.linalg.inv(Rest)
# global matrix: sum_t A_t |J_t - R_t|^2 with J_t = D_t Rinv_t, D_t = [u1-u0, u2-u0]
At = 0.5 * np.abs(cross2(E1, E2))
G = sp.lil_matrix((2 * len(F), n))
for t in range(len(F)):
    for k in range(2):        # rows of Rinv applied to the two edges -> gradient operator rows
        c = Rinv[t][:, k]
        G[2 * t + k, F[t, 1]] += c[0]; G[2 * t + k, F[t, 2]] += c[1]; G[2 * t + k, F[t, 0]] -= c[0] + c[1]
G = G.tocsr(); Wt = sp.diags(np.repeat(At, 2))
K = (G.T @ Wt @ G).tolil(); K[0, :] = 0; K[0, 0] = 1.0; K = K.tocsc(); solveK = spla.factorized(K)
Uc = U * np.exp(-phi.mean())
from shapely.geometry import LinearRing
def injective_chart(uv):
    tri = uv[F]
    return (cross2(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]) > 0).all() and LinearRing(uv[L]).is_simple

guard_chart = os.environ.get('TRI_NORMALIZED') == '1'
if guard_chart and not injective_chart(Uc):
    raise ValueError('Ω の外周が自己交差しているか、反転面があります。')
limited_step = False
frames = [Uc.copy()]; Eprev = None; it = 0; E = 0.0
while strength > 0:
    D = np.stack([Uc[F[:, 1]] - Uc[F[:, 0]], Uc[F[:, 2]] - Uc[F[:, 0]]], 2); J = D @ Rinv
    Uu, Ss, Vt = np.linalg.svd(J); Rt = Uu @ Vt
    neg = np.linalg.det(Rt) < 0
    if neg.any(): Uu[neg, :, 1] *= -1; Rt[neg] = Uu[neg] @ Vt[neg]
    E = float(np.sum(At * ((J - Rt) ** 2).sum((1, 2))))
    rhsx = G.T @ Wt @ Rt[:, 0, :].reshape(-1); rhsy = G.T @ Wt @ Rt[:, 1, :].reshape(-1)
    rhsx[0] = Uc[0, 0]; rhsy[0] = Uc[0, 1]
    proposed = np.c_[solveK(rhsx), solveK(rhsy)]
    if guard_chart and not injective_chart(proposed):
        fraction = 0.5
        while fraction > 1e-8 and not injective_chart(Uc + fraction*(proposed-Uc)):
            fraction *= .5
        if fraction > 1e-8:
            Uc += fraction*(proposed-Uc)
        limited_step = True
        it += 1
        frames.append(Uc.copy())
        print('[OMEGA2] limited ARAP at the first boundary contact or face inversion', flush=True)
        break
    Uc = proposed; it += 1; frames.append(Uc.copy())
    if Eprev is not None and abs(Eprev - E) <= 1e-9 * max(E, 1e-30): break
    Eprev = E
flips = int((cross2(Uc[F[:, 1]] - Uc[F[:, 0]], Uc[F[:, 2]] - Uc[F[:, 0]]) <= 0).sum())
print(f'[OMEGA2] ARAP iterations {it}, energy {E:.3e}, flipped triangles {flips}', flush=True)
if flips and os.environ.get('TRI_NORMALIZED')=='1':
    raise ValueError(f'Ω2 に向きが反転した面が {flips} 枚あります。伸縮を弱めてください。')
local_scale = relative_stretch(U, Uc, F)
fit['applied_harmonic_ratio'] = float(np.exp(np.ptp(phi)))
fit['size_ratio_on_omega2'] = float(local_scale.max() / local_scale.min())
fit['relative_scale_p05_p50_p95'] = np.percentile(local_scale, [5, 50, 95]).tolist()
print('[OMEGA2] actual area-equivalent local length scale (global zoom removed):',
      fit['relative_scale_p05_p50_p95'], flush=True)
extra_chart = {'original_uv': r['uv']} if os.environ.get('TRI_NORMALIZED') == '1' else {}
pickle.dump(dict(xyz=r['xyz'], f3=r['f3'], uv=Uc, f2=r['f2'], **extra_chart), open(os.path.join(out, 'omega2_chart.pkl'), 'wb'))
pickle.dump(dict(U_omega=U, U_omega2=Uc, phi=phi, psi=psi, s_star=s_star, kap_s=kap_s, s0=s0, s_b=s_b, s_g=s_g, kmax=kmax, Kabs=Kabs, thick=thick, thick_active=thick_active), open(os.path.join(out, 'omega2_fields.pkl'), 'wb'))
json.dump(dict(fit=fit, stretch=strength, arap_iterations=it, injectivity_limited=limited_step, flips=flips, size_ratio_cap=RATIO, grade=GRADE, c=c_, size_mode=os.environ.get("TRI_SIZE_MODE", "adaptive"), panels_requested=NP, panels_estimated=NP if uniform else n_panels(s_star)), open(os.path.join(out, 'omega2_summary.json'), 'w'), indent=1)
# ---- figures ----
fig, axs = plt.subplots(1, 3, figsize=(17, 5.8), dpi=100)
for ax, val, title in ((axs[0], psi, 'log target spacing on Omega (from 3D curvature)'), (axs[1], phi, 'harmonic fit phi (used)')):
    pc = PolyCollection(U[F], array=val[F].mean(1), cmap='viridis', edgecolor='none'); pc.set_clim(np.percentile(psi, 1), np.percentile(psi, 99))
    ax.add_collection(pc); ax.autoscale(); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=10); fig.colorbar(pc, ax=ax, fraction=0.04)
axs[2].add_collection(PolyCollection(Uc[F], facecolor='#e3e9ef', edgecolor='#9aa7b4', lw=0.08)); axs[2].autoscale(); axs[2].set_aspect('equal'); axs[2].axis('off')
axs[2].set_title('Omega2 (Omega scaled by exp(-phi)); a uniform lattice is laid here', fontsize=10)
fig.suptitle(f"target 3D size ratio {fit['size_ratio_target']:.2f} (cap {RATIO}); realised on Omega2 {fit['size_ratio_on_omega2']:.2f}; "
             f"fit residual rms {fit['residual_rms']:.3f} (log scale)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(out, 'omega2_fields.png'))
rec = Recorder(out, 'anim_omega2_arap'); lo = np.minimum(U.min(0), Uc.min(0)); hi = np.maximum(U.max(0), Uc.max(0))
for k in pick_frames(len(frames), 80):
    f_ = plt.figure(figsize=(7, 6), dpi=90); ax = f_.add_axes([0, 0, 1, 0.92]); ax.set_aspect('equal'); ax.axis('off')
    ax.add_collection(PolyCollection(frames[k][F], facecolor='#e3e9ef', edgecolor='#9aa7b4', lw=0.08)); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    f_.suptitle(f'Omega -> Omega2 (ARAP with scaled rest triangles), iteration {k}', fontsize=9); rec.add(f_)
rec.finish(fps=10)
