"""Triangle panels on an OptCuts Omega with CLOSED seams.

1. OptCuts Omega (input chart).
2. Omega's outline is pulled onto a Gaussian-smoothed copy of itself (interior: symmetric Dirichlet vs the 3D surface).
3. Equilateral lattice (every interior vertex 6-valent; connectivity never changed afterwards except whole lattice
   triangles added/removed at the outline, which keeps every interior vertex 6-valent).
   The initial mask uses the selected half-area or centroid rule. In the Mac Kagome workflow,
   ordered chords are fitted jointly on paired banks; feasible lattice candidates are ranked
   by the deviation of their planned boundary polylines from each original Omega2 section.
   Seam rule: every seam node (seam tip, junction, seam/rim meeting point) gets its own panel vertex, pinned on it;
   between two nodes, the two banks of a seam must carry the same number of panel vertices: lattice triangles are
   added/removed along the bank with the wrong count until all pairs agree.
   Relaxation (gradient preconditioned by (I + lam L)): equilateral symmetric Dirichlet + pull of every outline vertex
   to its target; targets: nodes -> the node; i-th vertex of a bank and its partner on the other bank -> the points of
   their banks at ONE common position along the 3D seam (mean of their current positions); rim -> closest rim point.
   The pull weight is doubled after each converged stage until all outline vertices are within 1e-3 a.
4. Panels overlapping Omega by < 50 % are dropped (then 3 is repeated).  Omega's outline edges are split at the 3D seam
   positions of the panel corners (so both banks get a vertex at exactly the same 3D point) and Omega is deformed so
   that its outline runs through the panel corners and along the panel edges.
5. M3D: panel corners take the 3D point of their Omega outline vertex, other panel vertices are mapped barycentrically;
   panel vertices on the two banks of a seam are then the same 3D points and are welded.
usage: pipeline2.py CHART_PKL OUT_DIR [N_PANELS]
"""
import sys, os, json, pickle, subprocess, numpy as np
from collections import Counter
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the scripts/ folder
sys.path.insert(0, S + '/os_simple')
from common import CutMesh
from boundary_design import distribute_lift_distortion, allocate_boundary_edges, arc_fractions, resample, place_corners, angle_report, shape_limited_grid
def boundary_loop(F):
    """outline loops of a triangle set; a vertex where the outline passes twice (pinch) appears twice in its loop"""
    he = set()
    for f in F:
        for k in range(3): he.add((int(f[k]), int(f[(k + 1) % 3])))
    out_ = {}
    for (x, y) in he:
        if (y, x) not in he: out_.setdefault(x, []).append(y)
    used = set(); loops = []
    for s0 in list(out_):
        for y0 in out_[s0]:
            if (s0, y0) in used: continue
            Lp = [s0]; used.add((s0, y0)); x = y0; guard = 0
            while x != s0 and guard < 4 * len(he):
                nxts = [y for y in out_.get(x, []) if (x, y) not in used]
                if not nxts: break
                Lp.append(x); used.add((x, nxts[0])); x = nxts[0]; guard += 1
            loops.append(Lp)
    loops.sort(key=len, reverse=True)
    return loops
from flow import UVProblem
from anim import Recorder, pick_frames
import shapely
from shapely.geometry import Polygon, LinearRing
from matplotlib.path import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.tri as mtri

chart, out = sys.argv[1], sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 500
os.makedirs(out, exist_ok=True)
r = pickle.load(open(chart, 'rb'))
cm = CutMesh(r['xyz'], r['uv'], r['f3'], r['f2'])
U = cm.U.copy(); F = cm.F.copy(); X = cm.X.copy(); L = cm.loop.copy(); m0 = len(L)
nodes, runs, pairs = cm.branches()
def cross2(u, v): return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
a = np.sqrt(Polygon(U[L]).area / (NP * np.sqrt(3) / 4))
log = {}
focus = os.environ.get('TRI_GRID_FOCUS')
if focus:
    vertex, radius = focus.split(',')
    center = U[int(vertex)]
    displacement = U-center
    focused = U + 1.15*displacement*np.exp(-np.sum(displacement**2,axis=1)/(2*(float(radius)*a)**2))[:,None]
    triangle = focused[F]
    if (cross2(triangle[:,1]-triangle[:,0], triangle[:,2]-triangle[:,0]) <= 0).any() or not LinearRing(focused[L]).is_simple:
        sys.exit(42)
    U = focused
    a = np.sqrt(Polygon(U[L]).area / (NP*np.sqrt(3)/4))
    log['sampling_focus'] = dict(chart_vertex=int(vertex), radius_in_pitches=float(radius))
feedback = os.environ.get('TRI_CURVATURE_FEEDBACK')
if feedback:
    from curvature_sampling import focused_chart, fan_angles
    with open(feedback, 'rb') as stream:
        previous_mesh = pickle.load(stream)
    depth_floor = float(os.environ['TRI_DEPTH_FLOOR_MM'])
    lateral = float(os.environ['TRI_LATERAL_MM'])
    U, sampling = focused_chart(previous_mesh,U,L,F,X,a,depth_floor,lateral)
    log['curvature_sampling'] = sampling
    a = np.sqrt(Polygon(U[L]).area / (NP*np.sqrt(3)/4))
lost_tips = Counter()
# ---- outline sections between nodes: seam runs (paired) and rim sections ----
node_pos = sorted(nodes)                                    # loop positions (vertex L[p])
if not node_pos and os.environ.get('TRI_NORMALIZED') == '1' and not np.any(cm.is_seam_edge):
    # A disk with no native cuts still needs boundary anchors for ordered relaxation.
    ring = U[L]
    tangent = np.roll(ring, -1, axis=0) - ring
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-15)
    turns = np.arccos(np.clip((np.roll(tangent, 1, axis=0) * tangent).sum(1), -1., 1.))
    node_pos = np.flatnonzero(turns > np.pi / 5).tolist()
    if len(node_pos) < 3:
        length = np.linalg.norm(np.roll(ring,-1,axis=0)-ring,axis=1)
        arc = np.r_[0., np.cumsum(length)[:-1]]
        node_pos = sorted(set(node_pos + [int(np.abs(arc-t).argmin()) for t in length.sum()*np.arange(3)/3]))
sections = []                                               # (start_pos, n_edges, kind, run id)
run_of_edge = {e: ri for ri, rr in enumerate(runs) for e in rr}
for i, p in enumerate(node_pos):
    q = node_pos[(i + 1) % len(node_pos)]; n = (q - p) % m0 or m0
    sections.append(dict(start=p, n=n, kind='seam' if p in run_of_edge else 'rim', run=run_of_edge.get(p)))
partner = {}
for ri, rj in pairs: partner[ri] = rj; partner[rj] = ri
sec_of_run = {s_['run']: k for k, s_ in enumerate(sections) if s_['kind'] == 'seam'}
seg3 = np.linalg.norm(X[np.roll(L, -1)] - X[L], axis=1)      # 3D length of every outline edge (loop order)
for s_ in sections:
    idx = (s_['start'] + np.arange(s_['n'])) % m0; s_['len3'] = float(seg3[idx].sum())
print('[SEAM] sections', [(s_['kind'], s_['n'], round(s_['len3'] / a, 2)) for s_ in sections], flush=True)
log['sections'] = [dict(kind=s_['kind'], edges=s_['n'], length_over_a=s_['len3'] / a, run=s_['run']) for s_ in sections]

def sd_mean(Uc, Fc, Xc):
    P = Xc[Fc]; e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]; n = np.cross(e1, e2); a3 = 0.5 * np.linalg.norm(n, axis=1)
    t1 = e1 / np.linalg.norm(e1, axis=1)[:, None]; t2 = np.cross(n / (2 * a3)[:, None], t1)
    Q = np.stack([np.stack([(e1 * t1).sum(1), (e1 * t2).sum(1)], 1), np.stack([(e2 * t1).sum(1), (e2 * t2).sum(1)], 1)], 2)
    W = Uc[Fc]; R = np.stack([W[:, 1] - W[:, 0], W[:, 2] - W[:, 0]], 2); s = np.linalg.svd(R @ np.linalg.inv(Q), compute_uv=False)
    return float(((s[:, 0]**2 + s[:, 0]**-2 + s[:, 1]**2 + s[:, 1]**-2) * a3).sum() / a3.sum())

def untangle_after_boundary_snap(pulled, snapped, faces, boundary):
    """With the target outline fixed, restore positive area near collapsed faces."""
    from scipy.optimize import minimize
    def areas(tri):
        return cross2(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    reference = areas(pulled[faces])
    bad = areas(snapped[faces]) < .2 * reference
    if not bad.any():
        return snapped
    vertices = np.unique(faces[bad])
    for _ in range(3):
        vertices = np.unique(faces[np.isin(faces, vertices).any(axis=1)])
    free = np.setdiff1d(vertices, boundary)
    if not len(free):
        return None
    region = np.isin(faces, free).any(axis=1)
    local_faces = faces[region]
    base = reference[region]
    points = snapped.copy()
    def objective(z):
        points[free] = z.reshape(-1, 2)
        tri = points[local_faces]
        deficit = np.minimum(areas(tri)-.2*base, 0.)
        gradient = np.zeros_like(points)
        for j in range(3):
            edge = tri[:, (j+1)%3] - tri[:, (j+2)%3]
            derivative = edge[:, [1,0]] * [1., -1.]
            np.add.at(gradient, local_faces[:, j], (2*deficit/base)[:, None]*derivative)
        return (deficit**2/base).sum(), gradient[free].ravel()
    result = minimize(objective, points[free].ravel(), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=1000, ftol=1e-22, gtol=1e-12))
    points[free] = result.x.reshape(-1, 2)
    return points if np.all(areas(points[faces]) >= .1*reference) else None


def fit_outline(Uc, Fc, Xc, Lc, target, tag, frames, repair_snap=False):
    prob = UVProblem(Xc, Fc, Lc, w_conf=0.0)
    s3 = np.linalg.norm(Xc[np.roll(Lc, -1)] - Xc[Lc], axis=1)
    prob.anc_idx = Lc.copy(); prob.anc_tgt = target.copy(); prob.anc_w = (0.5 * (s3 + np.roll(s3, 1)) / s3.sum()) / a ** 2
    w = 1.0; steps = []
    while True:
        prob.w_anchor = w
        Uc, hist = prob.solve(Uc, record=None if os.environ.get('TRI_SKIP_VIDEOS')=='1' else lambda U_, E: frames.append((tag, w, U_.copy(), Fc)), check_outline=False)
        err = float(np.abs(Uc[Lc] - target).max() / a)
        steps.append(dict(w=w, iterations=len(hist) - 1, max_outline_error_over_a=err, SD=sd_mean(Uc, Fc, Xc)))
        print(f'[{tag}]', steps[-1], flush=True)
        if err < 1e-4: break
        if len(steps) > 4 and err > 0.99 * steps[-2]['max_outline_error_over_a']:
            print(f'[{tag}] stalled', flush=True); break
        w *= 2.0
    Us = Uc.copy(); Us[Lc] = target
    if repair_snap:
        Us = untangle_after_boundary_snap(Uc, Us, Fc, Lc)
    if Us is None or cross2(Us[Fc[:, 1]] - Us[Fc[:, 0]], Us[Fc[:, 2]] - Us[Fc[:, 0]]).min() <= 0:
        print(f'[{tag}] snapping flips a triangle; outline left at the converged pull', flush=True)
        steps.append(dict(snapped=False)); return Uc, steps
    prob.w_anchor = 0.0; prob.fixed[:] = False; prob.fixed[Lc] = True
    Uc, hist = prob.solve(Us, record=None if os.environ.get('TRI_SKIP_VIDEOS')=='1' else lambda U_, E: frames.append((tag + ' (outline fixed)', 0, U_.copy(), Fc)), check_outline=False)
    steps.append(dict(snapped=True, iterations=len(hist) - 1, SD=sd_mean(Uc, Fc, Xc)))
    print(f'[{tag}]', steps[-1], flush=True)
    return Uc, steps

frames = []
log['SD_optcuts'] = sd_mean(U, F, X)
# ================= step 2: smooth outline =================
ring = U[L]; segu = np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1); cu = np.r_[0, np.cumsum(segu)][:-1]; Lu = segu.sum()
h = a / 4; sres = np.arange(0, Lu, h)
k_ = np.clip(np.searchsorted(cu, sres, side='right') - 1, 0, m0 - 1); R0 = ring[k_] + ((sres - cu[k_]) / segu[k_])[:, None] * (np.roll(ring, -1, 0)[k_] - ring[k_])
sig = a
while not os.environ.get('SKIP_STEP2'):
    xs = np.arange(-int(4 * sig / h), int(4 * sig / h) + 1); g = np.exp(-0.5 * (xs * h / sig) ** 2); g /= g.sum()
    C = sum(gi * np.roll(R0, -xi, 0) for xi, gi in zip(xs, g))
    if LinearRing(C).is_simple: break
    sig *= 0.5
C = C if 'C' in dir() else R0
kk = cu / h; k0 = np.floor(kk).astype(int) % len(C); k1 = (k0 + 1) % len(C); t_ = (kk - np.floor(kk))[:, None]
if os.environ.get('SKIP_STEP2'):        # the input chart already has a smooth outline (e.g. the size-scaled Omega2)
    U2 = U.copy(); log['step2'] = 'skipped (input outline already smooth)'
else:
    U2, log['step2'] = fit_outline(U.copy(), F, X, L, (1 - t_) * C[k0] + t_ * C[k1], 'step 2: smooth outline', frames)
log['SD_after_step2'] = sd_mean(U2, F, X)
if os.environ.get('STOP_AFTER_STEP2'):                       # only produce the smoothed Omega (for building Omega2 from it)
    pickle.dump(U2, open(os.path.join(out, 'U2.pkl'), 'wb')); json.dump(log, open(os.path.join(out, 'summary_step2.json'), 'w'), indent=1, default=str)
    lo_, hi_ = U.min(0) - a, U.max(0) + a; rec = Recorder(out, 'anim_omega_smooth')
    for k in pick_frames(len(frames), 120):
        tag, w, Uc, Fc = frames[k]
        fig = plt.figure(figsize=(8, 7), dpi=90); ax = fig.add_axes([0, 0, 1, 0.9]); ax.set_aspect('equal'); ax.axis('off')
        ax.add_collection(PolyCollection(Uc[Fc], facecolor='#e3e9ef', edgecolor='#c0c9d2', lw=0.1)); ax.set_xlim(lo_[0], hi_[0]); ax.set_ylim(lo_[1], hi_[1])
        fig.suptitle(f'{tag}   pull weight {w:g}', fontsize=9); rec.add(fig)
    rec.finish(fps=12); print('[STEP2] saved', log['SD_after_step2'], flush=True); sys.exit(0)
O = U2[L]                                                    # Omega' outline polygon (loop order, ccw)

# ---- Omega' outline: position = (edge index along the loop) + fraction; 3D arclength inside a section ----
oa, ob = O, np.roll(O, -1, 0); oab = ob - oa; oab2 = (oab * oab).sum(1)
def closest_pos(q, edges=None):
    E_ = np.arange(m0) if edges is None else np.asarray(edges)
    t = np.clip(((q - oa[E_]) * oab[E_]).sum(1) / oab2[E_], 0, 1); d = np.linalg.norm(oa[E_] + t[:, None] * oab[E_] - q, axis=1)
    j = int(np.argmin(d)); return int(E_[j]), float(t[j]), float(d[j])
def uv_at(e, f): return O[e] + f * oab[e]
def sec_edges(k): s_ = sections[k]; return (s_['start'] + np.arange(s_['n'])) % m0
def sec_param(k, e, f):
    """3D arclength from the section start of the point (edge e, fraction f) of section k"""
    E_ = sec_edges(k); i = int(np.where(E_ == e)[0][0]); return float(seg3[E_[:i]].sum() + f * seg3[e])
def sec_point(k, t):
    E_ = sec_edges(k); c = np.r_[0, np.cumsum(seg3[E_])]; t = min(max(t, 0.0), c[-1])
    i = min(int(np.searchsorted(c, t, side='right') - 1), len(E_) - 1); f = (t - c[i]) / seg3[E_[i]]
    return int(E_[i]), float(f)

# ================= step 3: lattice =================
initial_rule = os.environ.get('TRI_INITIAL_RULE', 'area_fraction')
if initial_rule not in ('area_fraction', 'centroid'):
    raise ValueError('Unknown initial panel selection rule: '+initial_rule)
paired_protrusions = os.environ.get('TRI_PAIRED_PROTRUSIONS') == '1'
outline_path = Path(O)
domain = Polygon(O).buffer(0)
shapely.prepare(domain)

def initial_area_mask(points, faces):
    triangles = shapely.polygons(points[faces])
    keep = shapely.covers(domain, triangles)
    boundary = ~keep & shapely.intersects(domain, triangles)
    cut = triangles[boundary]
    keep[boundary] = shapely.area(shapely.intersection(domain, cut)) >= .5 * shapely.area(cut)
    return keep


def outside_adjacency(points, faces):
    """Faces extending outside Ω2, and pairs joined by a shared edge outside Ω2."""
    outside = ~shapely.covers(domain, shapely.polygons(points[faces]))
    ids = np.flatnonzero(outside)
    edges = np.sort(faces[ids][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = np.repeat(ids, 3)[order]
    shared = np.flatnonzero((edges[1:] == edges[:-1]).all(axis=1))
    # A point contact or an edge entirely on the outline cannot support a protrusion.
    shared = shared[~shapely.covers(domain, shapely.linestrings(points[edges[shared]]))]
    return outside, owners[shared], owners[shared + 1]


def unsupported_panels(keep, adjacency):
    outside, left, right = adjacency
    isolated = keep & outside
    paired = keep[left] & keep[right]
    isolated[left[paired]] = False
    isolated[right[paired]] = False
    return isolated


def clean_selection(faces, keep, adjacency):
    if adjacency is None:
        return clean(faces, keep)
    rejected = np.zeros_like(keep)
    while True:
        isolated = unsupported_panels(keep, adjacency)
        rejected |= isolated
        keep = clean(faces, keep & ~rejected)
        if np.any(keep & rejected):
            # Topology repair would restore a rejected face; try another lattice.
            return np.zeros_like(keep)
        if not unsupported_panels(keep, adjacency).any():
            return keep


def build_lattice(theta, off):
    e1 = a * np.array([np.cos(theta), np.sin(theta)]); e2 = a * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
    lo, hi = O.min(0) - 3 * a, O.max(0) + 3 * a
    M = np.linalg.inv(np.c_[e1, e2]); cs = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]]) @ M.T
    i0, j0 = np.floor(cs.min(0)).astype(int) - 1; i1, j1 = np.ceil(cs.max(0)).astype(int) + 1
    I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
    P = (I[..., None] * e1 + J[..., None] * e2 + off[0] * e1 + off[1] * e2).reshape(-1, 2)
    ni, nj = I.shape; idx = np.arange(ni * nj).reshape(ni, nj)
    up = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3)
    dn = np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)
    T = np.r_[up, dn]
    return P, T
def valid_set(T, keep):
    """keep mask -> (ok, reason): connected, one outline loop, no triangle on <= 1 edge, no pinched vertex"""
    K = T[keep]
    if len(K) == 0: return False
    E = np.sort(np.r_[K[:, [0, 1]], K[:, [1, 2]], K[:, [2, 0]]], 1)
    u, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    if ((cnt[inv] == 2).reshape(3, -1).sum(0) <= 1).any(): return False
    loops = boundary_loop(K)
    if len(loops) != 1: return False
    # pinch: an outline vertex visited twice
    Lp = loops[0]
    return len(Lp) == len(set(Lp))
def clean(T, keep):
    keep = keep.copy()
    for _guard in range(10 * len(T)):            # each pass removes or adds triangles; a cycle means no valid set
        if _guard == 10 * len(T) - 1: return np.zeros_like(keep)
        K = np.where(keep)[0]; KT = T[K]
        if not len(K):
            return keep
        E = np.sort(np.r_[KT[:, [0, 1]], KT[:, [1, 2]], KT[:, [2, 0]]], 1)
        u, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
        bad = (cnt[inv] == 2).reshape(3, -1).sum(0) <= 1
        if bad.any(): keep[K[bad]] = False; continue
        # largest component
        tid = np.tile(np.arange(len(KT)), 3); o = np.lexsort((E[:, 1], E[:, 0])); Es, ts = E[o], tid[o]
        same = (Es[1:] == Es[:-1]).all(1)
        nc, lab = connected_components(sp.coo_matrix((np.ones(same.sum()), (ts[:-1][same], ts[1:][same])), shape=(len(KT),) * 2), directed=False)
        if nc > 1: keep[K[lab != np.bincount(lab).argmax()]] = False; continue
        loops = boundary_loop(KT)
        Lp = loops[0]
        if len(loops) == 1 and len(Lp) == len(set(Lp)): return keep
        # holes / pinches: fill holes by adding nothing; remove the triangles at pinched vertices
        cntv = {}
        for v in Lp: cntv[v] = cntv.get(v, 0) + 1
        pin = [v for v, c in cntv.items() if c > 1]
        if pin:
            for v in pin:
                ts_ = K[(KT == v).any(1)]; keep[ts_[0]] = False
            continue
        if len(loops) > 1:   # interior hole: add the lattice triangles inside it
            for Lh in loops[1:]:
                poly_h = Path(P_[Lh]); cand = np.where(~keep)[0]; cen = P_[T[cand]].mean(1)
                keep[cand[poly_h.contains_points(cen)]] = True
            continue
        return keep

all_paired = os.environ.get('TRI_NORMALIZED')=='1' and len(pairs)>0 and np.all(cm.is_seam_edge)
def outline_state(P, T, keep):
    """panel outline loop (ccw) and, for each outline vertex: section, edge, fraction; node pins; per-section lists"""
    K = T[keep]; Lp = np.array(boundary_loop(K)[0])
    if not LinearRing(P[Lp]).is_ccw: Lp = Lp[::-1]
    # node pins: for every Omega' node, the nearest panel outline vertex (distinct, cyclic order kept)
    if metric_boundary:
        N = len(Lp)
        distances = np.sum((P[Lp,None]-O[node_pos])**2,axis=2)/a**2
        nearest = distances.argmin(0)
        measured = (np.roll(nearest,-1)-nearest)%N
        targets = [metric_targets]
        if measured.sum() == N:
            targets.extend([(.5*metric_targets+.5*measured),(.15*metric_targets+.85*measured)])
        best_pins = None
        physical_slots = None
        seam_ids = list(sec_of_run.values())
        for target in targets:
            key = (N,tuple(target))
            if key not in count_cache:
                count_cache[key] = allocate_boundary_edges(target, count_groups, N)
            slots = count_cache[key]
            if slots is None:
                continue
            if physical_slots is None:
                physical_slots = slots
            # All allocations spend the same perimeter. Avoid borrowing from an
            # already short seam just to put a junction nearer its planar image.
            if seam_ids:
                lower = np.minimum(physical_slots,np.floor(metric_targets)).astype(int)
                upper = np.maximum(physical_slots,np.ceil(metric_targets)).astype(int)
                if np.any(slots[seam_ids]<lower[seam_ids]) or np.any(slots[seam_ids]>upper[seam_ids]):
                    continue
            offsets = np.r_[0,np.cumsum(slots)[:-1]]
            positions = (np.arange(N)[:,None]+offsets)%N
            errors = distances[positions,np.arange(len(slots))]
            scores = (3*errors.max(1)+2*errors.mean(1)
                      + np.mean((slots-metric_targets)**2))
            phase = np.argmin(scores)
            if best_pins is None or scores[phase] < best_pins[0]:
                best_pins = scores[phase], Lp[positions[phase]]
        if best_pins is None:
            raise ValueError('No integer allocation for this perimeter')
        pins = dict(zip(node_pos,map(int,best_pins[1])))
    elif all_paired:
        # Paired runs must use the same number of boundary edges. Quantize
        # their average arclength together, then find the cyclic phase with
        # the smallest squared displacement of the cut-graph junctions.
        N=len(Lp); groups=np.array([(sec_of_run[i],sec_of_run[j]) for i,j in pairs])
        if N<len(node_pos):raise ValueError('Too few boundary edges for seam branches')
        lengths=np.array([np.linalg.norm(oab[sec_edges(k)],axis=1).sum() for k in range(len(sections))])
        ideal=lengths[groups].mean(1)*N/lengths.sum()
        allocation=np.ones(len(groups),int)
        for _ in range(N//2-len(groups)):allocation[np.argmax(ideal-allocation)]+=1
        slots=np.empty(len(sections),int);slots[groups.ravel()]=np.repeat(allocation,2)
        if N%2:slots[np.argmax(lengths/slots)]+=1
        offsets=np.r_[0,np.cumsum(slots)[:-1]]
        positions=(np.arange(N)[:,None]+offsets)%N
        best=np.argmin(((P[Lp[positions]]-O[node_pos])**2).sum(axis=(1,2)))
        pins=dict(zip(node_pos,map(int,Lp[positions[best]])))
    else:
        pins = {}
        for p in node_pos:
            d = np.linalg.norm(P[Lp] - O[p], axis=1)
            for j in np.argsort(d):
                if int(Lp[j]) not in pins.values(): pins[p] = int(Lp[j]); break
    pinpos = {v: p for p, v in pins.items()}
    # walk the panel outline from the pin of the first node; sections between consecutive pins
    start = int(np.where(Lp == pins[node_pos[0]])[0][0]); Lr = np.roll(Lp, -start)
    order_ok = [pinpos[v] for v in Lr if v in pinpos] == node_pos
    per_sec = {k: [] for k in range(len(sections))}; k = -1
    for v in Lr:
        if v in pinpos: k = node_pos.index(pinpos[v]); continue
        per_sec[k].append(int(v))
    return Lr, pins, per_sec, order_ok
def mismatch(per_sec):
    return sum(abs(len(per_sec[sec_of_run[ri]]) - len(per_sec[sec_of_run[rj]])) for ri, rj in pairs)
def fit_cost(P, T, keep):
    K = T[keep]; Lp = np.array(boundary_loop(K)[0]); d = np.array([closest_pos(q)[2] for q in P[Lp]])
    return float(np.mean((d / a) ** 2))

ordered_boundary = os.environ.get('TRI_ORDERED_BOUNDARY') == '1'
metric_boundary = ordered_boundary and os.environ.get('TRI_SEAM_SPACING') == '1'
if metric_boundary:
    tri3 = X[F]
    areas = .5*np.linalg.norm(np.cross(tri3[:,1]-tri3[:,0],tri3[:,2]-tri3[:,0]),axis=1)
    relative = np.ones(len(X))
    strength = float(os.environ.get('TRI_STRETCH','0'))
    if strength and os.environ.get('TRI_SIZE_MODE') == 'adaptive':
        with open(os.path.join(os.path.dirname(chart),'omega2_fields.pkl'),'rb') as field_file:
            relative = pickle.load(field_file)['s_star'][cm.v_of]**strength
    scale3 = np.sqrt(4*np.sum(areas/relative[F].mean(1)**2)/(np.sqrt(3)*NP))
    sizes = relative[L]*scale3
    metric_edges = seg3 / (.5*(sizes+np.roll(sizes,-1)))
    metric_targets = np.array([metric_edges[sec_edges(k)].sum() for k in range(len(sections))])
    count_groups = [(sec_of_run[i],sec_of_run[j]) for i,j in pairs]
    count_groups += [(k,) for k,s in enumerate(sections) if s['kind']=='rim']
    count_cache = {}
section_curves = []
for k in range(len(sections)):
    edges = sec_edges(k)
    curve = O[np.r_[edges, (edges[-1]+1)%len(O)]]
    section_curves.append(curve)


def section_vertices(k, pins, per_sec):
    start = sections[k]['start']
    end = (start+sections[k]['n']) % len(O)
    return [pins[start], *per_sec[k], pins[end]]


def boundary_score(points, pins, per_sec):
    # Compare corresponding arcs, including the Ω2 vertices between panel corners.
    # Nearest points on the opposite bank cannot hide a missed turn.
    distances = []
    for k, curve in enumerate(section_curves):
        outer = points[section_vertices(k, pins, per_sec)]
        arc = arc_fractions(curve)
        grid_arc = arc_fractions(outer)
        samples = np.union1d(arc, grid_arc)
        delta = resample(curve, arc, samples) - resample(outer, grid_arc, samples)
        distances.extend((delta*delta).sum(axis=1)/a**2)
    cost = np.mean(distances) + max(distances)
    if metric_boundary:
        errors = np.array([len(per_sec[k])+1 for k in range(len(sections))])-metric_targets
        cost += np.mean(errors**2)+np.max(errors**2)
    return cost


def designed_targets(points, pins, per_sec, pitch):
    targets = {}
    groups = [[(sec_of_run[i],False), (sec_of_run[j],True)] for i,j in pairs]
    groups.extend([[(k,False)] for k,s in enumerate(sections) if s['kind']=='rim'])
    for group in groups:
        ids, banks = [], []
        for k,reverse in group:
            curve = section_curves[k]
            lengths = seg3[sec_edges(k)]
            vertices = section_vertices(k, pins, per_sec)
            if reverse:
                curve, lengths, vertices = curve[::-1], lengths[::-1], vertices[::-1]
            ids.append(vertices)
            banks.append((curve, np.r_[0.,np.cumsum(lengths)]/lengths.sum(), points[vertices]))
        spacing = None
        if metric_boundary:
            k = group[0][0]
            spacing = banks[0][1], np.r_[0.,np.cumsum(metric_edges[sec_edges(k)])]
        for vertices, corners in zip(ids, place_corners(banks,pitch,spacing)):
            targets.update(zip(vertices, corners))
    return np.array(list(targets)), np.array(list(targets.values()))


cands_l = []
prior_path = os.path.join(os.path.dirname(chart), 'curvature_reference_grid.json')
prior_lattice = None
if feedback and os.path.isfile(prior_path):
    with open(prior_path) as stream:
        prior_lattice = json.load(stream)['lattice']
for th in np.radians(np.arange(0, 60, 5)):
    for ox in np.linspace(0, 1, 4, endpoint=False):
        for oy in np.linspace(0, 1, 4, endpoint=False):
            if prior_lattice and (abs((np.degrees(th)-prior_lattice['rotation_deg']+30)%60-30)>5.01
                                  or (ox,oy) != tuple(prior_lattice['offset'])):
                continue
            P_, T_ = build_lattice(th, (ox, oy))
            if initial_rule == 'centroid':
                initial = outline_path.contains_points(P_[T_].mean(axis=1))
            else:
                initial = initial_area_mask(P_, T_)
            adjacency = outside_adjacency(P_, T_) if paired_protrusions else None
            keep = clean_selection(T_, initial, adjacency)
            if not valid_set(T_, keep): continue
            try:
                Lr, pins, per_sec, ok = outline_state(P_, T_, keep)
            except Exception:
                continue
            if not ok: continue
            sc = (boundary_score(P_, pins, per_sec) if ordered_boundary else fit_cost(P_, T_, keep)) + 0.5 * mismatch(per_sec)
            cands_l.append((sc, th, (ox, oy), P_, T_, keep, int(initial.sum()), int(unsupported_panels(initial, adjacency).sum()) if paired_protrusions else 0))
cands_l.sort(key=lambda c_: c_[0])
def try_ops(P_, T_, adjacency, keep, per_sec, discarded=()):
    """candidate single-triangle changes touching the outline; returns the best (mismatch, cost, keep)"""
    Lp = set(boundary_loop(T_[keep])[0])
    cand = [t for t in range(len(T_)) if t not in discarded and len(set(T_[t]) & Lp) >= 2]
    cur = mismatch(per_sec); res = None
    for t in cand:
        k2 = keep.copy(); k2[t] = not k2[t]
        if adjacency is not None and unsupported_panels(k2, adjacency).any(): continue
        if not valid_set(T_, k2): continue
        try:
            Lr2, pins2, ps2, ok2 = outline_state(P_, T_, k2)
        except Exception:
            continue
        if not ok2: continue
        mm = mismatch(ps2)
        if mm >= cur: continue
        c = boundary_score(P_, pins2, ps2) if ordered_boundary else fit_cost(P_, T_, k2)
        if res is None or (mm, c) < (res[0], res[1]): res = (mm, c, k2)
    return res
# ---- relaxation with paired seam targets ----
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

def relax(P_, T_, keep, frames_r):
    used = np.unique(T_[keep]); remap = -np.ones(len(P_), int); remap[used] = np.arange(len(used))
    Xg = P_[used].copy(); Tg = remap[T_[keep]]
    sgn = np.sign(cross2(Xg[Tg[:, 1]] - Xg[Tg[:, 0]], Xg[Tg[:, 2]] - Xg[Tg[:, 0]])); Tg = np.where(sgn[:, None] > 0, Tg, Tg[:, [0, 2, 1]])
    Lr, pins, per_sec, ok = outline_state(P_, T_, keep)
    pin_g = {p: remap[v] for p, v in pins.items()}; sec_g = {k: [remap[v] for v in vs] for k, vs in per_sec.items()}
    B = np.array([remap[v] for v in Lr])
    ap = np.sqrt(Polygon(O).area / (len(Tg) * np.sqrt(3) / 4))
    if ordered_boundary:
        ids, targets = designed_targets(Xg, pin_g, sec_g, ap)
        source = Xg.copy()
        source[ids] = targets
        Xg, _ = shape_limited_grid(Xg, Tg, B, source[B])
        frames_r.append((Xg.copy(), Tg, 0., 0.))
        return Xg, Tg, B, pin_g, sec_g, [], ap, source
    Rinv = np.linalg.inv(np.array([[ap, ap / 2], [0, ap * np.sqrt(3) / 2]]))
    Tj, Rj = jnp.array(Tg), jnp.array(Rinv)
    def e_shape(Xj):
        Pp = Xj[Tj]; D = jnp.stack([Pp[:, 1] - Pp[:, 0], Pp[:, 2] - Pp[:, 0]], 2); J = D @ Rj
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]; fro = (J ** 2).sum((1, 2))
        return jnp.mean(fro + fro / det ** 2) - 4.0
    eg = jax.jit(jax.value_and_grad(e_shape))
    def targets(Xc):
        tg = {}
        for p, g_ in pin_g.items(): tg[g_] = O[p]
        params = {}
        for k, vs in sec_g.items():
            s_ = sections[k]
            for g_ in vs:
                e, f, _ = closest_pos(Xc[g_], sec_edges(k)); params[g_] = sec_param(k, e, f)
        for ri, rj in pairs:
            ki, kj = sec_of_run[ri], sec_of_run[rj]; Li = sections[ki]['len3']
            A_, B_ = sec_g[ki], sec_g[kj][::-1]
            n = min(len(A_), len(B_))
            tA = np.array([params[g_] for g_ in A_[:n]]); tB = Li - np.array([params[g_] for g_ in B_[:n]])
            tm = 0.5 * (tA + tB)
            tm = np.clip(np.maximum.accumulate(tm), 0, Li)               # keep the order along the seam
            for i in range(n):
                e, f = sec_point(ki, tm[i]); tg[A_[i]] = uv_at(e, f)
                e, f = sec_point(kj, Li - tm[i]); tg[B_[i]] = uv_at(e, f)
            for g_ in A_[n:] + B_[n:]: e, f, _ = closest_pos(Xc[g_], sec_edges(sec_of_run[ri] if g_ in A_ else sec_of_run[rj])); tg[g_] = uv_at(e, f)
        for k, s_ in enumerate(sections):
            if s_['kind'] == 'rim':
                for g_ in sec_g[k]: e, f, _ = closest_pos(Xc[g_], sec_edges(k)); tg[g_] = uv_at(e, f)
        idx = np.array(list(tg.keys())); return idx, np.array([tg[i] for i in idx])
    max_iterations = 1500 if ordered_boundary else 10000
    nV = len(Xg); Eu = np.unique(np.sort(np.r_[Tg[:, [0, 1]], Tg[:, [1, 2]], Tg[:, [2, 0]]], 1), axis=0)
    Ad = sp.coo_matrix((np.ones(2 * len(Eu)), (np.r_[Eu[:, 0], Eu[:, 1]], np.r_[Eu[:, 1], Eu[:, 0]])), shape=(nV, nV)).tocsr()
    pre = spla.factorized((sp.eye(nV) + 10.0 * (sp.diags(np.asarray(Ad.sum(1)).ravel()) - Ad)).tocsc())
    def detmin(Xn): Pp = Xn[Tg]; return cross2(Pp[:, 1] - Pp[:, 0], Pp[:, 2] - Pp[:, 0]).min()
    w = 1.0; step = 1e-3; stages = []
    while True:
        best_e, best_it, it = np.inf, 0, 0
        while True:
            idx, tgt = targets(Xg)
            def E_(Xn):
                es, gg = eg(jnp.array(Xn)); dd = Xn[idx] - tgt
                G = np.asarray(gg).copy(); G[idx] += w * 2 * dd / (len(idx) * ap ** 2)
                return float(es) + w * float((dd ** 2).sum() / (len(idx) * ap ** 2)), G
            E0, G = E_(Xg); D = np.column_stack([pre(G[:, 0]), pre(G[:, 1])]); t = 4 * step
            while True:
                Xn = Xg - t * D
                if detmin(Xn) > 0 and E_(Xn)[0] < E0: break
                t *= 0.5
                if t < 1e-14: Xn = Xg; break
            step = max(t, 1e-8); Xg = Xn; it += 1; En = E_(Xg)[0]
            if os.environ.get("TRI_SKIP_VIDEOS") != "1":
                frames_r.append((Xg.copy(), Tg, w, En))
            if En < best_e * (1 - 1e-7): best_e, best_it = En, it
            # Advance the penalty on very slow stages. The seam-gap acceptance test still applies.
            if it - best_it >= 50 or it >= max_iterations: break
        idx, tgt = targets(Xg); gap = np.linalg.norm(Xg[idx] - tgt, axis=1).max() / ap
        stages.append(dict(w=w, iterations=it, energy=best_e, max_target_gap_over_a=float(gap))); print('[RELAX]', stages[-1], flush=True)
        if gap < 1e-3: break
        if w >= 2**24:
            raise ArithmeticError('Seam penalty stalled')
        w *= 2.0
    idx, tgt = targets(Xg); Xg[idx] = tgt                     # exact on the outline
    if ordered_boundary and (detmin(Xg) <= 0 or not LinearRing(Xg[B]).is_simple):
        raise ArithmeticError("Panel boundary folds after snapping")
    return Xg, Tg, B, pin_g, sec_g, stages, ap, Xg

def valid_seam_topology(faces, pin_vertices, section_vertices):
    """Predict welded connectivity, including the paired fins removed by remesh3d."""
    from scipy.cluster.hierarchy import DisjointSet
    import igl

    vertices = np.unique(faces)
    groups = DisjointSet(vertices)
    source_pins = {}
    for position, vertex in pin_vertices.items():
        source_id = cm.v_of[L[position]]
        groups.merge(vertex, source_pins.setdefault(source_id, vertex))
    for left, right in pairs:
        for a, b in zip(section_vertices[sec_of_run[left]], reversed(section_vertices[sec_of_run[right]])):
            groups.merge(a, b)
    representatives = np.arange(vertices.max() + 1)
    representatives[vertices] = [groups[v] for v in vertices]
    welded = representatives[faces]
    while len(welded):
        canonical = np.sort(welded, axis=1)
        if (np.diff(canonical, axis=1) == 0).any():
            return False
        _, ids, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
        if counts.max() == 1:
            break
        degree = np.bincount(welded.ravel())
        if ordered_boundary:
            for position, corner in pin_vertices.items():
                if degree[representatives[corner]] == 2:
                    lost_tips[int(L[position])] += 1
            return False
        fins = (counts[ids] == 2) & (degree[welded] == 2).any(axis=1)
        if not fins.any():
            return False
        welded = welded[~fins]
    if not len(welded) or not igl.is_edge_manifold(welded)[0]:
        return False
    return igl.is_vertex_manifold(welded)[np.unique(welded)].all() and igl.facet_components(welded)[0] == 1


def ranked_candidates(candidates):
    # Compare feasible outlines in small batches before spending time on the UV solve.
    batch = []
    for index, (_, th, off, points, faces, keep, initial_count, isolated) in enumerate(candidates):
        adjacency = outside_adjacency(points, faces) if paired_protrusions else None
        ring, pins, per_sec, _ = outline_state(points, faces, keep)
        mm0 = mismatch(per_sec)
        while mismatch(per_sec) > 0:
            change = try_ops(points, faces, adjacency, keep, per_sec)
            if change is None:
                break
            keep = change[2]
            ring, pins, per_sec, _ = outline_state(points, faces, keep)
        if mismatch(per_sec) or ((ordered_boundary or all_paired) and not valid_seam_topology(faces[keep], pins, per_sec)):
            continue
        score = 0.
        if ordered_boundary:
            ids, corners = designed_targets(points, pins, per_sec, a)
            source = points.copy()
            source[ids] = corners
            layout, chart_points = shape_limited_grid(points, faces[keep], ring, source[ring])
            boundary_faces = faces[keep & np.isin(faces,ring).any(1)]
            vertices, inverse = np.unique(boundary_faces, return_inverse=True)
            import igl
            chart3 = np.column_stack((U2,np.zeros(len(U2))))
            query = np.column_stack((chart_points[vertices],np.zeros(len(vertices))))
            _, fi, nearest = igl.point_mesh_squared_distance(query, chart3, F)
            fine = chart3[F[fi]]
            u,v = fine[:,1,:2]-fine[:,0,:2], fine[:,2,:2]-fine[:,0,:2]
            q = nearest[:,:2]-fine[:,0,:2]
            det = cross2(u,v)
            b1,b2 = cross2(q,v)/det, cross2(u,q)/det
            bary = np.column_stack((1.-b1-b2,b1,b2))
            estimate = (X[F[fi]]*bary[:,:,None]).sum(axis=1)
            observed = angle_report(estimate[inverse.reshape(-1,3)])
            quality_cost = (observed['max_deviation'] + observed['p95_deviation'])/20.
            mapping = dict(zip(ring, layout[ring]))
            deviations = []
            for k, curve in enumerate(section_curves):
                polyline = [mapping[v] for v in section_vertices(k, pins, per_sec)]
                deviations.append(shapely.hausdorff_distance(shapely.LineString(curve), shapely.LineString(polyline), densify=.25)/a)
            score = quality_cost + .25*(max(deviations) + np.mean(deviations))
        batch.append((score, index, th, off, points, faces, keep, initial_count, isolated, adjacency, mm0))
        if len(batch) == (6 if ordered_boundary else 1):
            yield from sorted(batch, key=lambda row: row[0])
            batch.clear()
    yield from sorted(batch, key=lambda row: row[0])


def lift_grid(Xg, Tg, B, pin_g, sec_g, boundary_source, keep, T_, adjacency, ap, log, frames):
    if paired_protrusions:
        log['lattice']['unsupported_in_final_selection'] = int(unsupported_panels(keep, adjacency).sum())
    log['panels'] = int(len(Tg))
    if ordered_boundary:
        panel_domain = Polygon(Xg[B])
        log['boundary_design'] = dict(method='panel_shape', reference_pitch=float(a),
            max_gap_over_pitch=float(shapely.hausdorff_distance(domain.boundary, panel_domain.boundary, densify=.25)/a),
            area_difference_fraction=float(domain.symmetric_difference(panel_domain).area/domain.area))

    # ================= step 4b: split Omega' outline at the panel corners, fit Omega to the panel outline =================
    # position of every panel outline vertex on Omega's outline: (loop edge, fraction); seam partners share the 3D point
    corner_pos = {}
    for p, g_ in pin_g.items(): corner_pos[g_] = (p, 0.0)
    for k, vs in sec_g.items():
        for g_ in vs: e, f, _ = closest_pos(boundary_source[g_], sec_edges(k)); corner_pos[g_] = (e, f)
    # seam partners: make the fraction identical in 3D (they were pulled to one common seam position)
    for ri, rj in pairs:
        ki, kj = sec_of_run[ri], sec_of_run[rj]; Li = sections[ki]['len3']; A_, B_ = sec_g[ki], sec_g[kj][::-1]
        for ga, gb in zip(A_, B_):
            ta = sec_param(ki, *corner_pos[ga]); tb = Li - sec_param(kj, *corner_pos[gb]); tm = 0.5 * (ta + tb)
            corner_pos[ga] = sec_point(ki, tm); corner_pos[gb] = sec_point(kj, Li - tm)
    if metric_boundary:
        log['boundary_design']['seam_spacing'] = []
    # split Omega's outline edges
    Uw, Fw, Xw = U2.copy(), F.copy(), X.copy(); Lw = list(L)
    splits = {}
    for g_, (e, f) in corner_pos.items():
        if f < 1e-9: continue
        if f > 1 - 1e-9: corner_pos[g_] = ((e + 1) % m0, 0.0); continue
        splits.setdefault(e, []).append((f, g_))
    edge_face = {}
    for fi, ff in enumerate(Fw):
        for i in range(3): edge_face[(ff[i], ff[(i + 1) % 3])] = fi
    corner_vertex = {}
    for g_, (e, f) in corner_pos.items():
        if f < 1e-9 and (e not in splits or all(g2 != g_ for _, g2 in splits.get(e, []))): corner_vertex[g_] = int(L[e])
    source_uv = r.get("original_uv")
    uv_split = list(source_uv) if source_uv is not None else []
    new_loop = []
    Fw = [list(ff) for ff in Fw]; Uw = list(Uw); Xw = list(Xw)
    for e in range(m0):
        va, vb = int(L[e]), int(L[(e + 1) % m0]); new_loop.append(va)
        if e not in splits: continue
        fi = edge_face[(va, vb)]; ff = Fw[fi]; vc = [x for x in ff if x not in (va, vb)][0]
        chain = [va]
        for f, g_ in sorted(splits[e]):
            nv = len(Uw); Uw.append((1 - f) * U2[va] + f * U2[vb]); Xw.append((1 - f) * X[va] + f * X[vb])
            if source_uv is not None:
                uv_split.append((1-f)*source_uv[va]+f*source_uv[vb])
            chain.append(nv); new_loop.append(nv); corner_vertex[g_] = nv
        chain.append(vb)
        for i in range(len(chain)-1):
            child = [chain[i], chain[i+1], vc]
            if i == 0:
                index = fi
                Fw[index] = child
            else:
                index = len(Fw)
                Fw.append(child)
            for u, v in zip(child, child[1:]+child[:1]):
                edge_face[(u, v)] = index
    Uw, Xw, Fw = np.array(Uw), np.array(Xw), np.array(Fw); Lw = np.array(new_loop)
    if paired_protrusions:
        # A chord between two fixed boundary vertices can force its adjacent UV
        # faces onto one straight panel edge. Give each such chord a free midpoint.
        edges, counts = np.unique(np.sort(Fw[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1),
                                  axis=0, return_counts=True)
        chords = edges[(counts == 2) & np.isin(edges, Lw).all(axis=1)]
        log['step4_split_interior_chords'] = len(chords)
        for u, v in chords:
            mid = len(Uw)
            Uw = np.vstack((Uw, (Uw[u]+Uw[v])/2))
            Xw = np.vstack((Xw, (Xw[u]+Xw[v])/2))
            if source_uv is not None:
                uv_split.append((uv_split[u]+uv_split[v])/2)
            adjacent = np.flatnonzero((Fw == u).any(axis=1) & (Fw == v).any(axis=1))
            for fi in adjacent:
                face = Fw[fi]
                i = next(i for i in range(3) if {face[i], face[(i+1)%3]} == {u,v})
                first, second, other = np.roll(face, -i)
                Fw[fi] = [first, mid, other]
                Fw = np.vstack((Fw, [mid, second, other]))
    # targets: corners -> panel vertex; other outline vertices -> on the panel edge between the corners around them,
    # at their share of 3D arclength
    Bc = list(B)                                    # panel outline (ccw), corner vertex for each
    cv = [corner_vertex[g_] for g_ in Bc]
    pos_in_loop = {int(v): i for i, v in enumerate(Lw)}
    ci = [pos_in_loop[v] for v in cv]
    s3w = np.linalg.norm(Xw[np.roll(Lw, -1)] - Xw[Lw], axis=1)
    tgt = np.zeros((len(Lw), 2)); nl = len(Lw)
    order = np.argsort(ci); ci_s = np.array(ci)[order]; Bc_s = [Bc[i] for i in order]
    for j in range(len(ci_s)):
        i0, i1 = ci_s[j], ci_s[(j + 1) % len(ci_s)]
        idx = np.arange(i0, i1 if i1 > i0 else i1 + nl) % nl
        l = np.r_[0, np.cumsum(s3w[idx])]; tot = l[-1]
        p0, p1 = Xg[Bc_s[j]], Xg[Bc_s[(j + 1) % len(ci_s)]]
        tgt[idx] = p0 + (l[:-1] / tot)[:, None] * (p1 - p0)
    # Fit the panel outline in the metric of the chart that received the grid.
    # Using the 3D surface as the rest metric here undoes Omega2's interior scale
    # field. Keep the real Xw separately for the barycentric lift below.
    if source_uv is not None:
        fit_reference = np.column_stack((Uw, np.zeros(len(Uw))))
        log['step4_reference'] = 'omega2'
    else:
        fit_reference = Xw
    U4, log['step4'] = fit_outline(Uw.copy(), Fw, fit_reference, Lw, tgt, 'step 4: fit Omega to the panel outline', frames, repair_snap=paired_protrusions)
    log['SD_final'] = sd_mean(U4, Fw, Xw)

    # ================= step 5: M3D, welded across the seams =================
    V3 = np.zeros((len(Xg), 3))
    for g_ in Bc: V3[g_] = Xw[corner_vertex[g_]]
    if source_uv is not None:
        uv_split = np.asarray(uv_split)
        original_grid = np.empty_like(Xg)
        original_grid[Bc] = uv_split[[corner_vertex[v] for v in Bc]]
    if metric_boundary:
        Xg, stats = distribute_lift_distortion(Xg,Tg,Bc,V3[Bc],U4,Fw,Xw,ap,domain)
        log['boundary_design']['global_relaxation'] = stats
    inner = np.setdiff1d(np.arange(len(Xg)), Bc)
    try:
        finder = mtri.Triangulation(U4[:, 0], U4[:, 1], Fw).get_trifinder(); fid = finder(Xg[inner, 0], Xg[inner, 1])
    except RuntimeError:
        fid = -np.ones(len(inner), int)
    cen = U4[Fw].mean(1)
    for v, f in zip(inner, fid):
        p = Xg[v]
        cand = [f] if f >= 0 else np.argsort(np.linalg.norm(cen - p, axis=1))[:30]
        best_ = None
        for ff in cand:
            A_, B_, C_ = U4[Fw[ff]]; l_ = np.linalg.solve(np.c_[B_ - A_, C_ - A_], p - A_); b = np.r_[1 - l_.sum(), l_]
            bc = np.clip(b, 0, None); bc /= bc.sum(); dd = np.linalg.norm(bc @ U4[Fw[ff]] - p)
            if best_ is None or dd < best_[0]: best_ = (dd, ff, bc)
        V3[v] = best_[2] @ Xw[Fw[best_[1]]]
        if source_uv is not None:
            original_grid[v] = best_[2] @ uv_split[Fw[best_[1]]]
    # weld identical 3D points (seam partners and node copies)
    key = np.round(V3 / ap, 7); uniq, wid = np.unique(key, axis=0, return_inverse=True)
    VW = np.zeros((len(uniq), 3)); np.add.at(VW, wid, V3); VW /= np.bincount(wid)[:, None]; TW = wid[Tg]
    TW = TW[(TW[:, 0] != TW[:, 1]) & (TW[:, 1] != TW[:, 2]) & (TW[:, 0] != TW[:, 2])]
    partner_gap = []
    for ri, rj in pairs:
        A_, B_ = sec_g[sec_of_run[ri]], sec_g[sec_of_run[rj]][::-1]
        partner_gap += [float(np.linalg.norm(V3[x] - V3[y])) for x, y in zip(A_, B_)]
    for p_ in node_pos:
        same = [g_ for q, g_ in pin_g.items() if cm.v_of[L[q]] == cm.v_of[L[p_]]]
        partner_gap += [float(np.linalg.norm(V3[same[0]] - V3[s_])) for s_ in same[1:]]
    Ew = np.unique(np.sort(np.r_[TW[:, [0, 1]], TW[:, [1, 2]], TW[:, [2, 0]]], 1), axis=0)
    degW = np.bincount(Ew.ravel(), minlength=len(VW))
    bW = set(boundary_loop(TW)[0]) if len(boundary_loop(TW)) else set()
    loopsW = boundary_loop(TW)
    if os.environ.get('TRI_NORMALIZED') == '1' and np.all(cm.is_seam_edge) and loopsW:
        raise ValueError('閉じた入力に対してM3Dに未接続の境界が残っています。')
    bWall = set(v for Lx in loopsW for v in Lx)
    innerW = np.array([v for v in range(len(VW)) if v not in bWall and degW[v] > 0])
    seamW = np.unique(wid[[g_ for vs in sec_g.values() for g_ in vs] + list(pin_g.values())])
    seamW_inner = [v for v in seamW if v not in bWall]
    if metric_boundary:
        for ri,rj in pairs:
            k = sec_of_run[ri]
            ids = section_vertices(k,pin_g,sec_g)
            chord = np.linalg.norm(np.diff(V3[ids],axis=0),axis=1)
            log['boundary_design']['seam_spacing'].append(dict(runs=[int(ri),int(rj)],
                target=float(metric_targets[k]), edges=len(chord),
                shortest_chord=float(chord.min()), longest_chord=float(chord.max())))
    if ordered_boundary:
        mask = np.isin(Tg,B).any(1)
        log['boundary_design'].update(m2d=angle_report(Xg[Tg[mask]]), inverse_m3d=angle_report(VW[wid[Tg[mask]]]))
    P3 = VW[TW]
    def ang(u, v): return np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))
    A3 = np.stack([ang(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0]), ang(P3[:, 2] - P3[:, 1], P3[:, 0] - P3[:, 1]), ang(P3[:, 0] - P3[:, 2], P3[:, 1] - P3[:, 2])], 1)
    el3 = np.linalg.norm(np.stack([P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 1], P3[:, 0] - P3[:, 2]], 1), axis=2)
    log.update(seam_partner_gap_max=float(max(partner_gap)) if partner_gap else None, outline_loops_after_weld=len(loopsW),
               interior_valence_3D={int(k): int(v) for k, v in zip(*np.unique(degW[innerW], return_counts=True))},
               seam_vertex_valence_3D={int(k): int(v) for k, v in zip(*np.unique(degW[seamW_inner], return_counts=True))},
               M3D_angle_dev_mean=float(np.abs(A3 - 60).mean()), M3D_angle_dev_p95=float(np.percentile(np.abs(A3 - 60), 95)),
               M3D_angle_range=[float(A3.min()), float(A3.max())],
               M3D_edge_over_median_p5_p95=[float(np.percentile(el3 / np.median(el3), 5)), float(np.percentile(el3 / np.median(el3), 95))])
    grid_extra = {"original_grid": original_grid, "lattice_up": np.flatnonzero(keep) < len(T_)//2} if source_uv is not None else {}
    if os.environ.get('TRI_BOUNDARY_LIFT')=='1':
        grid_extra['_source_uv_split'] = uv_split if source_uv is not None else None
    return dict(**grid_extra, VW=VW, TW=TW, Xg=Xg, Tg=Tg, U4=U4, Fw=Fw, Xw=Xw, Lw=Lw, sec_g=sec_g, pin_g=pin_g, sections=sections, pairs=pairs,
                     sec_of_run=sec_of_run, O=O, wid=wid, X=X, F=F, L=L)


candidates = cands_l if ordered_boundary else cands_l[:12]
best = None
trials = []
upstream_log = log.copy()
upstream_frames = frames
for score, c_i, th, off, P_, T_, keep, initial_count, isolated_initial, adjacency, mm0 in ranked_candidates(candidates):
    log = upstream_log.copy()
    Lr, pins, per_sec, ok = outline_state(P_, T_, keep)
    print(f'[GRID] candidate {c_i}: rotation {np.degrees(th):.0f} deg, mismatch {mm0} -> 0, boundary score {score:.4f}', flush=True)
    log['lattice'] = dict(rotation_deg=float(np.degrees(th)), offset=list(off), panels=int(keep.sum()), candidate=c_i,
                          initial_rule=initial_rule, initial_panels=initial_count)
    if paired_protrusions:
        log['lattice'].update(min_protruding_group=2, isolated_at_initial_selection=isolated_initial)
    if initial_rule == 'area_fraction':
        log['lattice']['initial_min_fraction'] = .5

    log['bank_mismatch_after_equalising'] = int(mismatch(per_sec))
    log['bank_counts'] = {f'{ri}-{rj}': [len(per_sec[sec_of_run[ri]]), len(per_sec[sec_of_run[rj]])] for ri, rj in pairs}

    relax_frames = []
    seen_masks = set()
    discarded = set()
    accepted = False
    while True:
        key = keep.tobytes()
        if key in seen_masks:
            print('[TRIM] repeated lattice; trying next candidate', flush=True)
            break
        seen_masks.add(key)
        try:
            Xg, Tg, B, pin_g, sec_g, log['relax'], ap, boundary_source = relax(P_, T_, keep, relax_frames)
        except ArithmeticError as error:
            print('[GRID]', error, flush=True)
            break
        # step 4a: overlap test
        def _frac(t):                                              # a panel squeezed flat on the outline counts as outside
            a_ = Polygon(Xg[t]).area
            return Polygon(Xg[t]).intersection(domain).area / a_ if a_ > 1e-9 * ap ** 2 else 0.0
        frac = np.array([_frac(t) for t in Tg])
        if (frac >= 0.5).all():
            if not valid_seam_topology(Tg, pin_g, sec_g):
                print('[GRID] nonmanifold or disconnected welded panels; trying next lattice', flush=True)
                break
            accepted = True
            break
        removed = np.flatnonzero(keep)[frac < 0.5]
        print(f'[TRIM] {len(removed)} panels below overlap threshold', flush=True)
        if ordered_boundary:
            discarded.update(removed)
        keep[removed] = False
        keep = clean_selection(T_, keep, adjacency)
        if any(keep[t] for t in discarded) or not valid_set(T_, keep):
            break
        try:
            Lr, pins, per_sec, ok = outline_state(P_, T_, keep)
        except ValueError:
            if not metric_boundary: raise
            break
        while mismatch(per_sec) > 0:
            r_ = try_ops(P_, T_, adjacency, keep, per_sec, discarded)
            if r_ is None: break
            keep = r_[2]; Lr, pins, per_sec, ok = outline_state(P_, T_, keep)
        if not ok or mismatch(per_sec): break
        print('[TRIM] dropped panels, redoing the relaxation', flush=True)

    if not accepted:
        continue
    trial_frames = []
    result = lift_grid(Xg, Tg, B, pin_g, sec_g, boundary_source, keep, T_, adjacency, ap, log, trial_frames)
    merit = 0.
    if ordered_boundary:
        shape = log['boundary_design']['inverse_m3d']
        merit = shape['max_deviation'] + shape['p95_deviation'] + 3.*log['boundary_design']['max_gap_over_pitch']
        from lift_quality import surface_quality
        quality = surface_quality(result['VW'], result['TW'], X)
        print(f'[GRID] candidate {c_i} welded surface: {quality}', flush=True)
        if not quality['valid'] or quality['max_distance_over_pitch'] > 1.:
            if 'farthest_reference_vertex' in quality:
                lost_tips[quality['farthest_reference_vertex']] += 1
            continue
        merit += 50.*quality['max_distance_over_pitch'] + 10.*quality['p95_distance_over_pitch']
        trial = dict(candidate=c_i, surface_quality=quality, **shape)
        if metric_boundary:
            report = log['boundary_design']
            spacing = report['seam_spacing']
            count_error = max((abs(p['edges']-p['target']) for p in spacing),default=0.)
            merit += 5*count_error + 20*shape['outside_20deg']/shape['panels']
            trial.update(seam_spacing=spacing,global_relaxation=report['global_relaxation'])
        if feedback:
            angles = fan_angles(result['VW'],result['TW'])
            opening_limit = np.degrees(2*np.arctan2(lateral,depth_floor))
            merit += angles.max() + np.maximum(angles-opening_limit,0.).mean()
            trial.update(max_fan_angle_deg=float(angles.max()), exceeding_vertices=int((angles>opening_limit).sum()))
        trials.append(dict(**trial,score=merit))
        print(f'[GRID] lifted candidate {c_i}: boundary angles {shape["min_angle"]:.1f}–{shape["max_angle"]:.1f}', flush=True)
    if best is None or merit < best[0]:
        best = merit, result, log, relax_frames, trial_frames
    if not ordered_boundary or len(trials) >= (6 if metric_boundary else 4) or shape['max_deviation'] <= 20.:
        break
if best is None:
    if feedback:
        raise ValueError("局所的に格子を増やした候補が形状・継ぎ目の条件を満たしません。元の実行は保存されています。")
    if focus:
        sys.exit(42)
    for vertex, votes in lost_tips.most_common(3):
        for radius in (3., 2.5, 3.5):
            print(f'[GRID] retry: locally increasing sampling around chart vertex {vertex} '
                  f'({votes} collapsed candidates, radius {radius} pitches)', flush=True)
            retry = subprocess.run([sys.executable, *sys.argv],
                                   env=dict(os.environ, TRI_GRID_FOCUS=f'{vertex},{radius}'))
            if retry.returncode != 42:
                sys.exit(retry.returncode)
    raise ValueError('目標枚数で、継ぎ目の溶接・外形・押し出し方向を同時に保てる格子が見つかりません。')
_, result, log, relax_frames, trial_frames = best
if focus or feedback:
    result['sampling_uv'] = U
frames = upstream_frames + trial_frames
Xg, Tg, sec_g, pin_g = (result[key] for key in ('Xg','Tg','sec_g','pin_g'))
if ordered_boundary:
    log['boundary_design']['lifted_candidates'] = trials
if os.environ.get('TRI_BOUNDARY_LIFT')=='1':
    from boundary_lift import improve_boundary_lift
    correction = improve_boundary_lift(result,result.pop('_source_uv_split'))
    log['boundary_lift'] = correction
    if 'after' in correction and 'boundary_design' in log:
        log['boundary_design']['inverse_m3d'] = correction['after']
    triangles = result['VW'][result['TW']]
    angles = []
    for j in range(3):
        left = triangles[:,j-1]-triangles[:,j]
        right = triangles[:,(j+1)%3]-triangles[:,j]
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(left,right),axis=1),(left*right).sum(1))))
    deviation = np.abs(np.asarray(angles)-60.)
    lengths = np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2)
    log.update(M3D_angle_dev_mean=float(deviation.mean()),M3D_angle_dev_p95=float(np.percentile(deviation,95)),
               M3D_angle_range=[float(np.min(angles)),float(np.max(angles))],
               M3D_edge_over_median_p5_p95=np.percentile(lengths/np.median(lengths),[5,95]).tolist())
    print('[M3D] boundary lift:',correction,flush=True)
json.dump(log, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
pickle.dump(result, open(os.path.join(out, 'result.pkl'), 'wb'))
print(json.dumps({k: v for k, v in log.items() if k not in ('step2', 'step4', 'relax', 'sections')}, indent=1), flush=True)

# ================= animations and M2D still =================
cols = plt.cm.tab10(np.arange(10))
def draw_m2d(ax, Xc, Tc, show_banks=True):
    ax.add_collection(PolyCollection(Xc[Tc], facecolor='#f2e3c4', edgecolor='#5a4a33', lw=0.35))
    ax.plot(*np.vstack([O, O[:1]]).T, color='#9aa7b4', lw=0.7, ls='--')
    if show_banks:
        for c_, (ri, rj) in enumerate(pairs):
            for k in (sec_of_run[ri], sec_of_run[rj]):
                vs = sec_g[k]
                if vs: ax.scatter(Xc[vs, 0], Xc[vs, 1], s=9, color=cols[c_ % 10], zorder=5)
        pv = list(pin_g.values()); ax.scatter(Xc[pv, 0], Xc[pv, 1], s=22, marker='s', facecolor='none', edgecolor='k', zorder=6)
    ax.set_aspect('equal'); ax.axis('off')
lo, hi = O.min(0) - a, O.max(0) + a
rec = Recorder(out, 'anim_grid_relax')
for k in pick_frames(len(relax_frames), 140):
    Xc, Tc, w, E = relax_frames[k]
    fig = plt.figure(figsize=(8, 7), dpi=90); ax = fig.add_axes([0, 0, 1, 0.9]); draw_m2d(ax, Xc, Tc, show_banks=False)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    fig.suptitle(f'panel relaxation with paired seam banks: iteration {k + 1}, pull weight {w:g}, energy {E:.4f}\ngrey dashed = smoothed Omega outline', fontsize=9)
    rec.add(fig)
rec.finish(fps=15)
rec = Recorder(out, 'anim_omega_fit')
for k in pick_frames(len(frames), 120):
    tag, w, Uc, Fc = frames[k]
    fig = plt.figure(figsize=(8, 7), dpi=90); ax = fig.add_axes([0, 0, 1, 0.9]); ax.set_aspect('equal'); ax.axis('off')
    ax.add_collection(PolyCollection(Uc[Fc], facecolor='#e3e9ef', edgecolor='#c0c9d2', lw=0.1))
    if tag.startswith('step 4'): ax.add_collection(PolyCollection(Xg[Tg], facecolor='none', edgecolor='#d08a2e', lw=0.4))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    fig.suptitle(f'{tag}   pull weight {w:g}' + ('\norange = panels' if tag.startswith('step 4') else ''), fontsize=9)
    rec.add(fig)
rec.finish(fps=12)
fig = plt.figure(figsize=(9, 8), dpi=110); ax = fig.add_axes([0, 0, 1, 0.9]); draw_m2d(ax, Xg, Tg)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
fig.suptitle(f"M2D: {len(Tg)} panels. Coloured dots = panel vertices on seam banks (same colour = the two banks of one seam, "
             f"same count on both: {log['bank_counts']}); squares = seam nodes", fontsize=8.5)
fig.savefig(os.path.join(out, 'M2D.png'))
