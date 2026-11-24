"""LargeSteps-style relaxation of the welded M3D on the 3D surface (connectivity fixed, so the valences do not change).

energy  = mean_e ((|e| - l_e) / l_e)^2  +  W_SHAPE * mean_t (1 / q_t - 1)
          l_e : target edge length = curvature-based size s*(x) at the edge midpoint, scaled so the median matches the
                current median edge length (the panel count is kept);  q_t = 4 sqrt3 A / (sum of squared edges), 1 for an
                equilateral triangle; 1/q - 1 grows with the square of the deviation from equilateral, so it still pulls
          firmly on nearly equilateral panels (the earlier (1 - q)^2 was almost flat there)
fins    = before relaxing, every "fin" is removed: a vertex with only two panels that share the same three vertices
          (front and back of one sheet; it appears where a seam tip gets one lattice triangle from each bank).  The vertex and
          both panels are deleted, so the edge across the fin's base is shared by the panels on either side and the
          surface stays closed.  The same two panels are dropped from M2D (removed_panels in remesh3d.pkl, indices into
          grid/result.pkl TW / Tg).
no new crossing panels: the tilt of a panel = angle between its normal and the normal of the fine-surface triangle
          closest to its centre.  After every trial step, the vertices of any panel that would stand more than 60
          degrees off the surface AND more than it did at the start of the relaxation (+0.5 degree) are put back where
          they were, repeated until no such panel is left; the same is done for any panel that would turn by more than
          60 degrees in one step.  Only then is the energy test made, so one stubborn panel holds back its own vertices
          instead of shrinking the step for the whole mesh.  So the relaxation never
          pulls a panel across a thin part such as the ear, while the rest of the mesh relaxes freely.
step    = gradient preconditioned by (I + lam L) (uniform graph Laplacian, lam = 10), then every vertex is projected back
          onto the surface LOCALLY: each vertex remembers the triangle of the fine surface it lies on, and is returned to
          the closest point among the surface triangles within a surface distance of 2 mean fine-mesh edges of that
          triangle whose normals face the same way as the surface under the vertex at the START (dot > 0), so it neither
          jumps through a thin part (the ear) nor creeps round its edge step by step to the opposite face (open-rim vertices go onto the rim); a step is accepted only if the energy decreases and no panel
          turns by more than 60 degrees.  Stops when the best energy has not improved by a relative 1e-6 over 50 steps.
usage: remesh3d.py RUN_DIR (with grid/result.pkl and omega2_fields.pkl) CHART_PKL
"""
import sys, os, json, pickle, igl, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the scripts/ folder
sys.path.insert(0, S + '/even'); sys.path.insert(0, S + '/os_simple')
from dmesh import Ref, _closest_on_tris
from anim import Recorder, pick_frames, setup3d, draw_mesh3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

run, chart = sys.argv[1], sys.argv[2]; W_SHAPE = float(os.environ.get('W_SHAPE', '100'))
R = pickle.load(open(os.path.join(run, 'grid', 'result.pkl'), 'rb')); Fd = pickle.load(open(os.path.join(run, 'omega2_fields.pkl'), 'rb'))
r = pickle.load(open(chart, 'rb')); V3, F3 = np.asarray(r['xyz'], float), np.asarray(r['f3'], int)
ref = Ref(V3, F3); s_star = Fd['s_star']
X, T = R['VW'].copy(), R['TW'].copy()
keep_t = np.ones(len(T), bool)
while True:                                                       # remove fins until none is left
    srt = np.sort(T, 1); fin = None
    for v in range(len(X)):
        tv = np.where(keep_t & (T == v).any(1))[0]
        if len(tv) == 2 and set(T[tv[0]]) == set(T[tv[1]]): fin = (v, tv); break
    if fin is None: break
    keep_t[fin[1]] = False; print(f'[FIN] removed vertex {fin[0]} and panels {fin[1].tolist()}', flush=True)
removed_panels = np.where(~keep_t)[0]
used = np.unique(T[keep_t]); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
X, T = X[used], remap[T[keep_t]]
import igl
if (not igl.is_edge_manifold(T)[0] or not igl.is_vertex_manifold(T).all()
        or igl.facet_components(T)[0] != 1):
    raise ValueError('継ぎ目でパネルが分岐、または分離しているため、M3Dを確定できません。')
nV = len(X)
E = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1), axis=0)
cnt = {}
for e in map(tuple, np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1)): cnt[e] = cnt.get(e, 0) + 1
isb = np.zeros(nV, bool)
for (a_, b_), c_ in cnt.items():
    if c_ == 1: isb[[a_, b_]] = True
A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(nV, nV)).tocsr()
pre = spla.factorized((sp.eye(nV) + 10.0 * (sp.diags(np.asarray(A.sum(1)).ravel()) - A)).tocsc())
Ej, Tj = jnp.array(E), jnp.array(T)
small_weight = float(os.environ.get("TRI_SMALL_WEIGHT","0"))
# Material targets belong to the inverse-mapped points. Keep them fixed as
# vertices move, including at thin features where a nearest midpoint can land
# on the opposite sheet. Blend the sizing field exactly as in Omega2.
X_start = X.copy()
fine_size = ref.interp(X_start, s_star)
strength = float(os.environ.get('TRI_STRETCH', '1'))
vertex_size = np.exp(strength * np.log(fine_size))
initial_edges = np.linalg.norm(X[E[:,0]] - X[E[:,1]], axis=1)
edge_size = vertex_size[E].mean(1)
size_scale = np.median(initial_edges / edge_size)
edge_size *= size_scale
panel_size = jnp.array(size_scale * vertex_size[T].mean(1))
# The realized chart and its rounded outline can differ locally from the
# target field. Retain the assigned panel footprint during surface relaxation.
initial_panel_size = np.linalg.norm(X[T] - np.roll(X[T], 1, axis=1), axis=2).mean(1)
panel_size = jnp.minimum(panel_size, initial_panel_size)
local_size = np.full(nV, np.inf)
np.minimum.at(local_size, E.ravel(), np.repeat(initial_edges, 2))
report_path = os.path.join(run, 'grid', 'summary.json')
grid_report = json.load(open(report_path)) if os.path.isfile(report_path) else {}
shape_first = grid_report.get('boundary_design', {}).get('method') == 'panel_shape'
if shape_first:
    boundary_ids = list(R['pin_g'].values()) + [v for section in R['sec_g'].values() for v in section]
    mapped = R['wid'][R['Tg']]
    valid = (np.diff(np.sort(mapped,axis=1),axis=1)!=0).all(1)
    boundary_faces = np.isin(R['Tg'][valid],boundary_ids).any(1)[keep_t]
    shape_first = boundary_faces.any()
if shape_first:
    boundary_triangles = jnp.asarray(T[boundary_faces])
move_limit = .4*local_size
if shape_first:
    perimeter_vertices = np.unique(T[boundary_faces])
    # A short edge is precisely what needs to open up. Use the local target
    # spacing as the trust radius on this band, still projecting every trial.
    spacing = np.bincount(E.ravel(), np.repeat(edge_size,2), minlength=nV)/np.bincount(E.ravel(),minlength=nV)
    move_limit[perimeter_vertices] = np.maximum(move_limit[perimeter_vertices], .7*spacing[perimeter_vertices])
def energy(Xj, ell):
    d = jnp.linalg.norm(Xj[Ej[:, 0]] - Xj[Ej[:, 1]], axis=1)
    P = Xj[Tj]; a = P[:, 1] - P[:, 0]; b = P[:, 2] - P[:, 0]; c = P[:, 2] - P[:, 1]
    q = 4 * jnp.sqrt(3.0) * 0.5 * jnp.linalg.norm(jnp.cross(a, b), axis=1) / ((a * a).sum(1) + (b * b).sum(1) + (c * c).sum(1))
    size_penalty = 0.
    if small_weight:
        lengths = jnp.stack([jnp.linalg.norm(v, axis=1) for v in (a,b,c)], axis=1)
        # Give the slimmest panels additional influence: a mean over hundreds
        # of well-shaped triangles otherwise hides a single nearly flat one.
        size_penalty = W_SHAPE*jnp.mean(jnp.maximum(0., 1/q-1.5)**2)
        size_penalty += small_weight*jnp.mean(jnp.maximum(0., .8-lengths.mean(1)/panel_size)**2)
    if shape_first:
        panels = Xj[boundary_triangles]
        left = jnp.roll(panels, 1, axis=1)-panels
        right = jnp.roll(panels, -1, axis=1)-panels
        cosine = (left*right).sum(2)/(jnp.linalg.norm(left,axis=2)*jnp.linalg.norm(right,axis=2))
        violation = jnp.maximum(jnp.maximum(jnp.cos(jnp.deg2rad(80.))-cosine,
                                             cosine-jnp.cos(jnp.deg2rad(40.))), 0.)
        size_penalty += W_SHAPE * (jnp.mean(violation**2) + jnp.sqrt(jnp.mean(violation**4)+1e-24))
    return size_penalty + jnp.mean(((d - ell) / ell) ** 2) + W_SHAPE * jnp.mean(1 / q - 1)
eg = jax.jit(jax.value_and_grad(energy)); ef = jax.jit(energy)
def targets(Xc):
    if small_weight:
        return edge_size
    s = ref.interp(0.5 * (Xc[E[:, 0]] + Xc[E[:, 1]]), s_star)
    d = np.linalg.norm(Xc[E[:, 0]] - Xc[E[:, 1]], axis=1); return s * np.median(d) / np.median(s)
# ---- local projection: surface triangles near each fine-surface triangle ----
from scipy.sparse.csgraph import dijkstra
FN3 = np.cross(V3[F3[:, 1]] - V3[F3[:, 0]], V3[F3[:, 2]] - V3[F3[:, 0]]); FN3 /= np.linalg.norm(FN3, axis=1)[:, None]
TT = igl.triangle_triangle_adjacency(F3)[0]; Cf = V3[F3].mean(1)
ia = np.repeat(np.arange(len(F3)), 3); ib = TT.ravel(); okk = ib >= 0; ia, ib = ia[okk], ib[okk]
Gf = sp.coo_matrix((np.linalg.norm(Cf[ia] - Cf[ib], axis=1), (ia, ib)), shape=(len(F3),) * 2).tocsr()
R_LOC = 2.0 * igl.edge_lengths(V3, F3).mean()
nbr = []
for c0 in range(0, len(F3), 400):
    D_ = dijkstra(Gf, indices=np.arange(c0, min(c0 + 400, len(F3))), limit=R_LOC)
    for k, row in enumerate(D_):
        f0 = c0 + k; cand = np.where(np.isfinite(row))[0]
        nbr.append(cand[(FN3[cand] @ FN3[f0]) > 0])
KMAX = max(len(c) for c in nbr); NB = np.full((len(F3), KMAX), -1)
for f0, c in enumerate(nbr): NB[f0, :len(c)] = c
cur_f = ref.closest(X)[1]                                   # surface triangle under each vertex (start: on the surface)
n_start = FN3[cur_f]                                        # surface normal under each vertex at the start
if small_weight:
    from scipy.spatial import cKDTree
    fine_triangles=V3[F3]
    surface_tree=cKDTree(Cf)
    radii=np.linalg.norm(fine_triangles-Cf[:,None],axis=2).max(1)
    fine_u=fine_triangles[:,1]-fine_triangles[:,0]
    fine_v=fine_triangles[:,2]-fine_triangles[:,0]

def crossing_vertices(start,end):
    # Continuous segment/triangle intersection. Ignore the two endpoint
    # contacts because the old and projected positions lie on the surface.
    direction=end-start
    length=np.linalg.norm(direction,axis=1)
    moving=np.flatnonzero(length>1e-9)
    if not len(moving):return moving
    midpoint=(start+end)/2
    groups=surface_tree.query_ball_point(midpoint[moving],.5*length[moving]+radii.max())
    i=np.repeat(moving,[len(g) for g in groups])
    f=np.asarray([j for g in groups for j in g],int)
    nearby=np.linalg.norm(Cf[f]-midpoint[i],axis=1)<=radii[f]+.5*length[i]
    i=i[nearby];f=f[nearby]
    h=np.cross(direction[i],fine_v[f]);det=np.einsum('ij,ij->i',fine_u[f],h)
    valid=np.abs(det)>1e-12*length[i]*np.linalg.norm(fine_u[f],axis=1)*np.linalg.norm(fine_v[f],axis=1)
    i=i[valid];f=f[valid];h=h[valid];det=det[valid]
    offset=start[i]-fine_triangles[f,0];q=np.cross(offset,fine_u[f])
    u=np.einsum('ij,ij->i',offset,h)/det
    v=np.einsum('ij,ij->i',direction[i],q)/det
    t=np.einsum('ij,ij->i',fine_v[f],q)/det
    hit=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1.+1e-9)&(t*length[i]>1e-8)&((1.-t)*length[i]>1e-8)
    return np.unique(i[hit])

def project(Xc, return_faces=False, previous=None, current_faces=None):
    previous = X if previous is None else previous
    current_faces = cur_f if current_faces is None else current_faces
    Y = Xc.copy(); i = np.where(~isb)[0]
    C = NB[current_faces[i]]; valid = C >= 0; Cs = np.where(valid, C, 0)
    pts = np.repeat(Xc[i], KMAX, 0); fa = Cs.ravel()
    Q = _closest_on_tris(pts, V3[F3[fa, 0]], V3[F3[fa, 1]], V3[F3[fa, 2]]).reshape(len(i), KMAX, 3)
    valid &= np.einsum('ikd,id->ik', FN3[Cs], n_start[i]) > 0     # never onto a face turned away from the start side
    d2 = ((Q - Xc[i][:, None]) ** 2).sum(2); d2[~valid] = np.inf; k = np.argmin(d2, 1)
    Y[i] = Q[np.arange(len(i)), k]; nf = current_faces.copy(); nf[i] = Cs[np.arange(len(i)), k]
    missing = i[~valid.any(1)]
    Y[missing] = previous[missing]; nf[missing] = current_faces[missing]
    b = np.where(isb)[0]
    if len(b): Y[b] = ref.project_boundary(Xc[b])
    if small_weight:
        crossed=crossing_vertices(previous,Y)
        Y[crossed]=previous[crossed]; nf[crossed]=current_faces[crossed]
    return (Y, nf) if return_faces else Y
def fn(Xc):
    n = np.cross(Xc[T[:, 1]] - Xc[T[:, 0]], Xc[T[:, 2]] - Xc[T[:, 0]]); return n / np.linalg.norm(n, axis=1)[:, None]
def ang_stats(Xc):
    P = Xc[T]
    def ang(u, v): return np.degrees(np.arccos(np.clip((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))
    Aa = np.stack([ang(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), ang(P[:, 2] - P[:, 1], P[:, 0] - P[:, 1]), ang(P[:, 0] - P[:, 2], P[:, 1] - P[:, 2])], 1)
    e3 = np.linalg.norm(P - P[:, [1, 2, 0]], axis=2).mean(1)
    return dict(angle_dev_mean=float(np.abs(Aa - 60).mean()), angle_dev_p95=float(np.percentile(np.abs(Aa - 60), 95)),
                angle_min=float(Aa.min()), angle_max=float(Aa.max()), size_ratio_p95_p5=float(np.percentile(e3, 95) / np.percentile(e3, 5))), Aa
def panel_quality(points):
    tri = points[T]
    lengths2 = np.sum((tri - np.roll(tri, 1, axis=1))**2, axis=2)
    area2 = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
    return 2*np.sqrt(3.)*area2/np.maximum(lengths2.sum(1), 1e-30)

quality_floor = np.minimum(panel_quality(X), .65)
before, _ = ang_stats(X)
FN3g = igl.per_face_normals(V3, F3, np.array([0., 0., 1.]))
def tilt(Xc):
    _, fi_, _ = igl.point_mesh_squared_distance(Xc[T].mean(1), V3, F3)
    n = np.cross(Xc[T[:, 1]] - Xc[T[:, 0]], Xc[T[:, 2]] - Xc[T[:, 0]]); n /= np.linalg.norm(n, axis=1)[:, None]
    return np.degrees(np.arccos(np.clip((n * FN3g[fi_]).sum(1), -1, 1)))
tilt0 = tilt(X); cap = np.maximum(60.0, tilt0 + 0.5); reverted_total = 0
def keep_flat(Xn, nf, Xo, fo):
    global reverted_total
    Xn, nf = Xn.copy(), nf.copy(); no = fn(Xo)
    while True:
        bad = (tilt(Xn) > cap) | ((fn(Xn) * no).sum(1) <= 0.5)   # would cut across a thin part, or turn by more than 60 deg
        if small_weight:
            too_far = np.linalg.norm(Xn-X_start, axis=1) > move_limit
            bad |= too_far[T].any(1) | (panel_quality(Xn) < quality_floor-1e-10)
        if not bad.any(): return Xn, nf
        v = np.unique(T[bad]); Xn[v] = Xo[v]; nf[v] = fo[v]; reverted_total += len(v)
quiet_steps=0
quiet_windows=0
ell = targets(X); frames = [X.copy()]; hist = []; best, best_it, it, step = np.inf, 0, 0, 1e-3
initial_energy = float(ef(jnp.array(X), jnp.array(ell)))
while True:
    it += 1
    e0, g = eg(jnp.array(X), jnp.array(ell)); e0 = float(e0); g = np.asarray(g)
    D = np.column_stack([pre(g[:, k]) for k in range(3)]); t = 4 * step; n0 = fn(X)
    while True:
        Xn, nf = project(X - t * D, True); Xn, nf = keep_flat(Xn, nf, X, cur_f); en = float(ef(jnp.array(Xn), jnp.array(ell)))
        if en < e0: break
        t *= 0.5
        if t < 1e-12: Xn, en, nf = X, e0, cur_f; break
    step = max(t, 1e-8); X = Xn; cur_f = nf; ell = targets(X)
    e = float(ef(jnp.array(X), jnp.array(ell))); hist.append(e); frames.append(X.copy())
    if small_weight:
        quiet_steps = quiet_steps+1 if (e0-en)<=1e-7*max(1.,abs(e0)) else 0
        if it%25==0:print(f'[M3D] iteration {it}, energy {e:.8g}, relative decrease {(e0-en)/max(1.,abs(e0)):.3g}',flush=True)
        if it%10==0 and len(hist)>10:
            relative=(hist[-11]-e)/max(1.,abs(hist[-11]))
            quiet_windows=quiet_windows+1 if relative<1e-4 else 0
        if en>=e0 or quiet_steps>=8 or quiet_windows>=2:break
    if e < best * (1 - 1e-6): best, best_it = e, it
    if it - best_it >= 50: break
if shape_first:
    # Independent vertex moves allow one sharp corner to improve when a
    # simultaneous step for its entire fan is rejected by the surface guards.
    def face_cost(points):
        tri = points[T]
        lengths = np.linalg.norm(np.roll(tri,1,axis=1)-tri, axis=2)
        q = np.maximum(panel_quality(points),1e-8)
        scale_error = np.maximum(.8-lengths.mean(1)/np.asarray(panel_size),0.)
        return (1/q-1)**2 * np.where(boundary_faces,4.,1.) + scale_error**2
    colors = np.full(nV,-1,int)
    for v in perimeter_vertices:
        occupied = set(colors[A.indices[A.indptr[v]:A.indptr[v+1]]])
        colors[v] = next(k for k in range(len(occupied)+1) if k not in occupied)
    for sweep in range(20):
        moved = 0
        for color in range(colors.max()+1):
            selected = np.flatnonzero(colors==color)
            _, gradient = eg(jnp.asarray(X), jnp.asarray(ell))
            direction = np.array(gradient)[selected]
            tangent_normal = FN3[cur_f[selected]]
            direction -= (direction*tangent_normal).sum(1)[:,None]*tangent_normal
            direction /= np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-30)
            cost0 = face_cost(X)
            best = np.zeros(len(selected))
            positions, new_faces = X[selected].copy(), cur_f[selected].copy()
            for scale in (.12,.04,.012):
                proposal = X.copy()
                proposal[selected] -= scale*spacing[selected,None]*direction
                trial, nf = project(proposal,True)
                other = colors!=color
                trial[other], nf[other] = X[other], cur_f[other]
                trial,nf = keep_flat(trial,nf,X,cur_f)
                change = np.bincount(T.ravel(), np.repeat(face_cost(trial)-cost0,3),minlength=nV)[selected]
                improve = change < best-1e-9
                best[improve] = change[improve]
                positions[improve] = trial[selected[improve]]
                new_faces[improve] = nf[selected[improve]]
            X[selected], cur_f[selected] = positions, new_faces
            moved += np.count_nonzero(best)
        if not moved:
            break
    frames.append(X.copy())
    hist.append(float(ef(jnp.asarray(X),jnp.asarray(ell))))
boundary_polish = None
if shape_first and os.environ.get('TRI_BOUNDARY_LIFT')=='1':
    from boundary_lift import polish_welded
    X,cur_f,boundary_polish = polish_welded(
        X,cur_f,T,boundary_faces,colors,spacing,np.asarray(panel_size),FN3,
        lambda trial,old,faces: project(trial,True,old,faces),keep_flat)
    frames.append(X.copy())
    hist.append(float(ef(jnp.asarray(X),jnp.asarray(ell))))
surface_check = None
relaxation_checkpoint = None
if shape_first:
    from lift_quality import surface_quality
    # Check previously projected iterates, so reverting a step cannot put a
    # vertex between two surface sheets. Apply the rim adjustment afterwards.
    for index in range(len(frames)-1, -1, -1):
        checked = surface_quality(frames[index], T, V3)
        if checked['valid'] and checked['max_distance_over_pitch'] <= 1.:
            X, surface_check = frames[index].copy(), checked
            if index != len(frames)-1:
                relaxation_checkpoint = index
                frames.append(X.copy())
                hist.append(float(ef(jnp.asarray(X), jnp.asarray(ell))))
            break
    else:
        raise ValueError('元の曲面を保持して押し出せるM3Dの緩和ステップがありません。')
rim_summary = None
rim_limit = float(os.environ.get('TRI_RIM_MAX_FRACTION', '0'))
if rim_limit > 0:
    from flexible_rim import adjust_rim
    adjusted, rim_summary = adjust_rim(X, T, rim_limit, (V3,F3))
    if shape_first:
        checked = surface_quality(adjusted, T, V3)
        if checked['valid'] and checked['max_distance_over_pitch'] <= 1.:
            X, surface_check = adjusted, checked
        else:
            rim_summary = dict(status='rejected', reason='surface_quality', trial=rim_summary)
    else:
        X = adjusted
    frames.append(X.copy())
    hist.append(float(ef(jnp.asarray(X), jnp.asarray(ell))))
surface_fit = None
if shape_first:
    from surface_fit import fit_surface
    X, surface_fit = fit_surface(X, T, V3, F3)
    surface_check = surface_quality(X, T, V3)
    frames.append(X.copy())
    hist.append(float(ef(jnp.asarray(X), jnp.asarray(ell))))
fan_summary = None
settings_path = os.path.join(run, 'settings.json')
if os.path.isfile(settings_path):
    with open(settings_path) as stream:
        cfg = json.load(stream)
    if (cfg.get('extrusion_mode') == 'outer_layer' and cfg.get('panel_depth_mode') == 'mm'
            and cfg.get('depth_tolerance_mm') is not None):
        min_depth = cfg['panel_depth_value'] - cfg['depth_tolerance_mm']
        if min_depth > 0:
            from fan_clearance import relax_fans
            X, fan_summary = relax_fans(X, T, V3, F3, min_depth, cfg.get('outer_lateral_mm', 1.))
            if fan_summary and fan_summary['steps']:
                frames.append(X.copy())
                hist.append(float(ef(jnp.asarray(X), jnp.asarray(ell))))
                if shape_first:
                    surface_check = surface_quality(X, T, V3)
from reference_folds import repair_reference_folds
candidate, fold_report = repair_reference_folds(X, T, V3, F3)
if fold_report['moved_vertices']:
    accepted = surface_quality(candidate, T, V3) if shape_first else None
    if accepted is None or (accepted['valid'] and accepted['max_distance_over_pitch'] <= 1.):
        X = candidate
        frames.append(X.copy())
        hist.append(float(ef(jnp.asarray(X), jnp.asarray(ell))))
        if accepted is not None:
            surface_check = accepted
    else:
        fold_report.update(status='surface_quality_rejected', moved_vertices=0,
                           remaining_edges=fold_report['detected_edges'])
print('[M3D reference folds]', fold_report, flush=True)
if relaxation_checkpoint is not None:
    surface_check['accepted_relaxation_checkpoint'] = relaxation_checkpoint
after, Aa = ang_stats(X)
res = dict(reference_folds=fold_report, panels=len(T), fins_removed=int(len(removed_panels) // 2), crossing_panels_start=int((tilt0 > 60).sum()), crossing_panels_end=int((tilt(X) > 60).sum()), iterations=it, energy_start=initial_energy, energy_end=hist[-1], before=before, after=after, w_shape=W_SHAPE, small_panel_weight=small_weight)
if small_weight:
    res.update(relative_energy_tolerance_per_10_steps=1e-4, local_target_range=[float(edge_size.min()),float(edge_size.max())],
               max_displacement_over_local_edge=float((np.linalg.norm(X-X_start,axis=1)/local_size).max()))
if shape_first:
    from boundary_design import angle_report
    res['reference_surface_fit'] = surface_fit
    res['surface_quality'] = surface_check
    res['boundary_k3d'] = angle_report(X[T[boundary_faces]])
    if boundary_polish is not None:
        res['boundary_polish'] = boundary_polish
if rim_summary is not None:
    res['flexible_rim'] = rim_summary
if fan_summary is not None:
    res['extrusion_fan_relaxation'] = dict(fan_summary, revision=2)
    feedback_path = os.path.join(run,'curvature_feedback.pkl')
    if os.path.isfile(feedback_path):
        from curvature_sampling import fan_angles
        with open(feedback_path,'rb') as stream:
            previous = pickle.load(stream)
        prior = fan_angles(previous['X'],previous['T'])
        current = fan_angles(X,T)
        res['curvature_sampling_comparison'] = dict(before_max=float(prior.max()), after_max=float(current.max()))
json.dump(res, open(os.path.join(run, 'remesh3d_summary.json'), 'w'), indent=1)
pickle.dump(dict(X=X, T=T, removed_panels=removed_panels, kept_vertices=used), open(os.path.join(run, 'remesh3d.pkl'), 'wb'))
print(json.dumps(res, indent=1))
if fold_report.get('remaining_edges', 0):
    raise ValueError(f"参照面にない強い折れ込みが {fold_report['remaining_edges']} 辺で残りました。M3Dの途中結果を保存しました。")

rec = Recorder(run, 'anim_remesh3d'); b3 = (V3.min(0), V3.max(0))
for k in pick_frames(len(frames), 100):
    fig = plt.figure(figsize=(11, 5.4), dpi=90)
    for j, az in enumerate((-62, 118)):
        ax = fig.add_subplot(1, 2, j + 1, projection='3d'); setup3d(ax, b3, 12, az); draw_mesh3d(ax, frames[k], T, edge='#33404d', lw=0.3)
    fig.suptitle(f'LargeSteps-style relaxation on the surface, iteration {k}' + (f', energy {hist[k - 1]:.4f}' if k else ''), fontsize=10)
    rec.add(fig)
rec.finish(fps=12)
