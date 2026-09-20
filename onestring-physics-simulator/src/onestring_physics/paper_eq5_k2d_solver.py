"""Section 4.3 on a corner-connected linkage (not an edge-glued quad mesh).

EEdge and EFab are the squared projection distances in Eq. (3)/Appendix A.
ECollision is an explicitly documented SAT penetration-depth surrogate, including
non-neighbour pairs. Sparse-preconditioned L-BFGS minimizes their *sum*.
This is not the paper's Shape-Up local/global implementation. See
 docs/K2D_EQ5_RECONSTRUCTION.md for the exact discretization and limitations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.optimize import OptimizeResult

_LAST_EQ5_HISTORY = None


def get_last_eq5_history():
    return _LAST_EQ5_HISTORY


@dataclass
class LinkageTopology:
    """Explicit index spaces: source metric mesh, tiles, and physical joints."""
    source_faces: np.ndarray
    faces: np.ndarray
    source_vertex_ids: np.ndarray
    hinges: np.ndarray  # tile_a, local_corner_a, tile_b, local_corner_b
    gaps: np.ndarray  # joint, first ray endpoint, second ray endpoint; CCW void


def build_linkage_topology(mesh):
    """Encode one alternating corner joint per uncut side adjacency.

    Pairwise joints are merged by construction. Distinct joints at coincident
    source positions remain distinct, and split vertex IDs are never re-welded.
    H(M2D) is the resulting set of linkage angle stencils, not a new set of tiles.
    """
    sf = np.asarray(mesh.faces, dtype=int)
    if sf.ndim != 2 or sf.shape[1] != 4 or len(sf) == 0:
        raise ValueError('Eq.5 requires a nonempty quad mesh')
    if np.min(sf) < 0 or np.max(sf) >= len(mesh.vertices):
        raise ValueError('M2D face indices are out of bounds')
    owners = {}
    for f, face in enumerate(sf):
        if len(set(face)) != 4:
            raise ValueError('Repeated quad corner')
        for k in range(4):
            key = tuple(sorted((int(face[k]), int(face[(k+1)%4]))))
            owners.setdefault(key, []).append((f, k))
    adjacency = [[] for _ in sf]
    pairs = []
    for edge, entries in owners.items():
        if len(entries) > 2:
            raise ValueError('Nonmanifold source edge; cannot encode quad linkage')
        if len(entries) == 2:
            (a, ea), (b, eb) = entries
            if sf[a, ea] != sf[b, (eb+1)%4]:
                raise ValueError('Source faces must be consistently oriented')
            adjacency[a].append(b); adjacency[b].append(a)
            pairs.append((a, ea, b, eb))
    colors = np.full(len(sf), -1, dtype=int)
    width = int(mesh.grid.nx) + 1
    for seed in range(len(sf)):
        if colors[seed] >= 0:
            continue
        v = int(sf[seed, 0])
        colors[seed] = (v // width + v % width) % 2
        stack = [seed]
        while stack:
            a = stack.pop()
            for b in adjacency[a]:
                wanted = 1-colors[a]
                if colors[b] < 0:
                    colors[b] = wanted; stack.append(b)
                elif colors[b] != wanted:
                    raise ValueError('Non-bipartite quad adjacency requires singularity handling')
    parent = np.arange(4*len(sf))
    hinges, ray_pairs = [], []
    used = set()
    for a, ea, b, eb in pairs:
        ca = (ea+1)%4 if colors[a] == 0 else ea
        cb = eb if colors[a] == 0 else (eb+1)%4
        if sf[a,ca] != sf[b,cb]:
            raise ValueError('Incompatible quad edge ordering')
        qa, qb = 4*a+ca, 4*b+cb
        if qa in used or qb in used:
            raise ValueError('A corner cannot belong to more than one pairwise joint')
        used.update((qa, qb)); parent[qb] = qa
        hinges.append((a, ca, b, cb))
        ra = 4*a+(ea if ca != ea else (ea+1)%4)
        rb = 4*b+(eb if cb != eb else (eb+1)%4)
        if colors[a] != 0:
            ra, rb = rb, ra
        ray_pairs.append((qa, ra, rb))
    # Every corner has at most one partner: no transitive four-way welds.
    roots, inverse = np.unique(parent, return_inverse=True)
    faces = inverse.reshape(-1, 4)
    gaps = inverse[np.asarray(ray_pairs, int)].reshape(-1, 3) if ray_pairs else np.empty((0,3), int)
    topology = LinkageTopology(sf.copy(), faces, sf.reshape(-1)[roots],
                               np.asarray(hinges, int).reshape(-1,4), gaps)
    return topology, np.asarray(mesh.vertices, float)[topology.source_vertex_ids, :2].copy()


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c*v[...,0]-s*v[...,1], s*v[...,0]+c*v[...,1]), axis=-1)


def project_angle_vectors(a, b, theta_min):
    """Exact Appendix A projection (lengths are allowed to change).

    On an active angle boundary beta, write x=r*u, y=t*R(beta)*u.
    The best u is the principal eigenvector of aa^T + cc^T, c=R(-beta)b;
    r=a.u, t=c.u. We test both angle orientations and nonnegative-radius
    boundary candidates, so unequal ray lengths are treated exactly. At zero
    angle the tied solution opens into the combinatorially ordered void.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = np.sum(a*b, axis=1)
    angles = np.arctan2(np.abs(_cross(a,b)), dot)
    target = np.clip(angles, theta_min, np.pi/2)
    active = np.abs(target-angles) > 1e-14
    pa, pb = a.copy(), b.copy()
    if not np.any(active):
        return pa, pb, angles
    aa, bb, beta = a[active], b[active], target[active]
    best = np.full(len(aa), np.inf)
    xa, xb = aa.copy(), bb.copy()
    for sign in (1., -1.):
        c = _rotate(bb, -sign*beta)
        matrix = aa[:,:,None]*aa[:,None,:] + c[:,:,None]*c[:,None,:]
        _, vectors = np.linalg.eigh(matrix)
        u = vectors[:,:,-1]
        u *= np.where(np.sum(u*(aa+c), axis=1) < 0, -1., 1.)[:,None]
        candidates = [u, -u,
                      aa/np.maximum(np.linalg.norm(aa,axis=1)[:,None],1e-30),
                      c/np.maximum(np.linalg.norm(c,axis=1)[:,None],1e-30)]
        for direction in candidates:
            r = np.maximum(np.sum(aa*direction,axis=1),0.)
            t = np.maximum(np.sum(c*direction,axis=1),0.)
            x = r[:,None]*direction
            y = _rotate(t[:,None]*direction, sign*beta)
            cost = np.sum((x-aa)**2+(y-bb)**2,axis=1)
            improve = cost < best-1e-15*np.maximum(1.,best.clip(max=1e10))
            xa[improve], xb[improve], best[improve] = x[improve], y[improve], cost[improve]
    pa[active], pb[active] = xa, xb
    return pa, pb, angles


def collision_energy_gradient(xy, faces, tolerance=1e-10):
    """Squared convex SAT translation depth; includes derivative of each axis.

    This surrogate is not claimed to equal Konakovic et al.'s local half-plane
    projection. All overlapping AABBs are tested, with no candidate cap. Contact
    at a shared corner/edge has zero energy. Convexity is checked by the caller.
    """
    polygons = xy[faces]
    lo, hi = polygons.min(axis=1), polygons.max(axis=1)
    # Sweep broad phase: memory scales with nearby pairs, not all F^2 pairs.
    order = np.argsort(lo[:,0], kind='stable')
    pair_a, pair_b = [], []
    for pos, i in enumerate(order):
        end = np.searchsorted(lo[order,0], hi[i,0], side='left')
        candidates = order[pos+1:end]
        keep = (hi[candidates,1] > lo[i,1]) & (lo[candidates,1] < hi[i,1])
        candidates = candidates[keep]
        pair_a.extend([i]*len(candidates)); pair_b.extend(candidates)
    ii, jj = np.asarray(pair_a,int), np.asarray(pair_b,int)
    grad = np.zeros_like(xy)
    if len(ii) == 0:
        return 0., grad, 0
    a, b = polygons[ii], polygons[jj]
    edges = np.concatenate((np.roll(a,-1,axis=1)-a,np.roll(b,-1,axis=1)-b),axis=1)
    lengths = np.linalg.norm(edges,axis=2)
    normals = np.stack((-edges[...,1],edges[...,0]),axis=-1)/np.maximum(lengths[...,None],1e-30)
    ap = np.einsum('pvd,pad->pav',a,normals)
    bp = np.einsum('pvd,pad->pav',b,normals)
    d1, d2 = ap.max(axis=2)-bp.min(axis=2), bp.max(axis=2)-ap.min(axis=2)
    depths = np.minimum(d1,d2)
    depths[lengths <= 1e-30] = np.inf  # repeated hull padding is not an axis
    axis = np.argmin(depths,axis=1)
    depth = depths[np.arange(len(ii)),axis]
    live = depth > 0.
    rows = np.flatnonzero(live)
    if len(rows) == 0:
        return 0., grad, 0
    ax, dp = axis[rows], depth[rows]
    n = normals[rows,ax]
    forward = d1[rows,ax] <= d2[rows,ax]
    ah = np.where(forward,ap[rows,ax].argmax(axis=1),ap[rows,ax].argmin(axis=1))
    bh = np.where(forward,bp[rows,ax].argmin(axis=1),bp[rows,ax].argmax(axis=1))
    sign = np.where(forward,1.,-1.)
    aid, bid = faces[ii[rows],ah], faces[jj[rows],bh]
    delta = sign[:,None]*(xy[aid]-xy[bid])
    weight = 2*dp
    np.add.at(grad,aid,(weight*sign)[:,None]*n)
    np.add.at(grad,bid,-(weight*sign)[:,None]*n)
    # n=J e/|e|. Chain rule for the selected support axis.
    dn = (delta-n*np.sum(delta*n,axis=1)[:,None])/lengths[rows,ax,None]
    de = np.stack((dn[:,1],-dn[:,0]),axis=1)*weight[:,None]
    m = faces.shape[1]
    owner = np.where(ax<m,ii[rows],jj[rows]); local = ax%m
    np.add.at(grad,faces[owner,local],-de)
    np.add.at(grad,faces[owner,(local+1)%m],de)
    return float(dp@dp), grad, int(np.count_nonzero(dp > tolerance))


class FlatObjective:
    def __init__(self, topology, target_vertices, initial_xy, weights, theta_min):
        self.topology = topology
        self.faces = topology.faces
        self.edges = np.sort(np.stack((self.faces,np.roll(self.faces,-1,axis=1)),axis=-1).reshape(-1,2),axis=1)
        self.edges = np.unique(self.edges,axis=0)
        source = topology.source_vertex_ids
        self.targets = np.linalg.norm(target_vertices[source[self.edges[:,1]]]-target_vertices[source[self.edges[:,0]]],axis=1)
        if np.any(self.targets <= 0) or not np.all(np.isfinite(self.targets)):
            raise ValueError('K3D contains invalid target edge lengths')
        self.scale = float(np.median(self.targets))
        self.weights = np.asarray(weights,float)
        self.theta_min = theta_min
        p = initial_xy[self.faces]
        self.orientation = np.sign(_cross(p[:,1]-p[:,0],p[:,2]-p[:,1]))
        if not self.valid(initial_xy):
            raise ValueError('Initial M2D contains inverted/degenerate/nonconvex quads')

    def valid(self, xy):
        if not np.all(np.isfinite(xy)):
            return False
        p = xy[self.faces]
        e = np.roll(p,-1,axis=1)-p
        return bool(np.all(_cross(e,np.roll(e,-1,axis=1))*self.orientation[:,None] > self.scale**2*1e-10))

    def evaluate(self, xy):
        g = np.zeros_like(xy)
        a,b = self.edges.T
        d = xy[b]-xy[a]; lengths = np.linalg.norm(d,axis=1)
        error = lengths-self.targets
        eg = 2*error[:,None]*d/np.maximum(lengths[:,None],1e-30)
        np.add.at(g,a,-self.weights[0]*eg); np.add.at(g,b,self.weights[0]*eg)
        ee = float(error@error)
        ec,cg,ncoll = collision_energy_gradient(xy,self.faces,self.scale*1e-8)
        g += self.weights[1]*cg
        gaps = self.topology.gaps
        if len(gaps):
            center,first,second = gaps.T
            va,vb = xy[first]-xy[center],xy[second]-xy[center]
            pa,pb,angles = project_angle_vectors(va,vb,self.theta_min)
            ra,rb = va-pa,vb-pb
            ef = float(np.sum(ra*ra+rb*rb))
            np.add.at(g,first,2*self.weights[2]*ra)
            np.add.at(g,second,2*self.weights[2]*rb)
            np.add.at(g,center,-2*self.weights[2]*(ra+rb))
            violations = int(np.count_nonzero((angles<self.theta_min-1e-6)|(angles>np.pi/2+1e-6)))
        else:
            ef,violations,angles = 0.,0,np.zeros(0)
        stats = dict(EFlat=float(self.weights@np.array([ee,ec,ef])),EEdge=ee,ECollision=ec,EFab=ef,
                     collisions=ncoll,fab_violations=violations,
                     gap_min_deg=float(np.degrees(angles.min())) if len(angles) else 0.,
                     gap_max_deg=float(np.degrees(angles.max())) if len(angles) else 0.,
                     edge_rms=float(np.sqrt(np.mean(error**2))),edge_mean=float(np.mean(np.abs(error))))
        return stats['EFlat'],g,stats


def _minimize_flat(objective, xy0, iterations, callback):
    """L-BFGS with an edge/stencil Laplacian inverse as its initial metric.

    The sparse solve is a preconditioner, not an additional energy/anchor. An
    Armijo line search evaluates the full nonlinear objective on every trial.
    Its tiny diagonal regularizes translation/null modes only in the metric.
    """
    scale=objective.scale
    origin=xy0.mean(axis=0)
    x=(xy0-origin)/scale
    gaps=objective.topology.gaps
    pairs=np.vstack((objective.edges,gaps[:,[0,1]],gaps[:,[0,2]]))
    weights=np.r_[np.full(len(objective.edges),objective.weights[0]),
                  np.full(2*len(gaps),objective.weights[2])]
    rows=np.repeat(np.arange(len(pairs)),2)
    matrix=sparse.coo_matrix((np.tile([-1.,1.],len(pairs))*np.repeat(np.sqrt(weights),2),
                             (rows,pairs.ravel())),shape=(len(pairs),len(x))).tocsc()
    metric=matrix.T @ matrix
    damping=max(float(np.mean(metric.diagonal())),1.)*1e-5/4.
    solve=factorized(2*(metric+sparse.eye(len(x))*damping).tocsc())
    def evaluate(z):
        xy=z*scale+origin
        if not objective.valid(xy):
            return float('inf'),np.zeros_like(z)
        e,g,_=objective.evaluate(xy)
        return e/(scale*scale),g/scale
    energy,gradient=evaluate(x)
    memory=[]
    message='iteration limit reached'
    success=False
    completed=0
    for _ in range(iterations):
        if np.linalg.norm(gradient,np.inf)<1e-9:
            message='gradient tolerance reached';success=True;break
        q=gradient.copy();alphas=[]
        for step,change,rho in reversed(memory):
            alpha=rho*np.sum(step*q);q-=alpha*change;alphas.append(alpha)
        direction=solve(q)
        if memory:
            step,change,_=memory[-1]
            gamma=np.sum(step*change)/max(float(np.sum(change*solve(change))),1e-30)
            direction*=np.clip(gamma,.001,1000.)
        for (step,change,rho),alpha in zip(memory,reversed(alphas)):
            beta=rho*np.sum(change*direction);direction+=step*(alpha-beta)
        direction=-direction
        slope=float(np.sum(gradient*direction))
        if slope>=0:
            memory.clear();direction=-solve(gradient);slope=float(np.sum(gradient*direction))
        alpha=1.
        for _trial in range(40):
            candidate=x+alpha*direction
            value,new_gradient=evaluate(candidate)
            if value<=energy+1e-4*alpha*slope:
                break
            alpha*=.5
        else:
            message='line search stalled';break
        step=candidate-x;change=new_gradient-gradient;curvature=float(np.sum(step*change))
        if curvature>1e-12:
            memory.append((step,change,1./curvature));memory=memory[-20:]
        x,energy,gradient=candidate,value,new_gradient
        completed+=1
        callback(x.ravel())
    return OptimizeResult(x=x.ravel(),nit=completed,success=success,message=message)


def _setting(name, default):
    value = float(os.environ.get(name,default))
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def optimize_paper_eq5(mesh_2d,mesh_3d,params,*,progress_callback=None,pipeline=None):
    global _LAST_EQ5_HISTORY
    start = time.perf_counter()
    if not np.array_equal(mesh_2d.faces,mesh_3d.faces) or len(mesh_2d.vertices)!=len(mesh_3d.vertices):
        raise ValueError('M2D and K3D must have identical source topology')
    topology,xy0 = build_linkage_topology(mesh_2d)
    weights = [_setting('ONESTRING_EQ5_W_EDGE',getattr(params,'w_edge',1.)),
               _setting('ONESTRING_EQ5_W_COLLISION',getattr(params,'w_collision',1.)),
               _setting('ONESTRING_EQ5_W_FAB',getattr(params,'w_fab',.001))]
    theta = _setting('ONESTRING_EQ5_THETA_MIN_DEG',5.)
    if theta > 90 or not any(weights):
        raise ValueError('Require theta_min <= 90 and at least one positive weight')
    iterations = max(1,int(_setting('ONESTRING_EQ5_ITERATIONS',max(80,getattr(params,'max_2d_iterations',40)*6))))
    objective = FlatObjective(topology,np.asarray(mesh_3d.vertices),xy0,weights,math.radians(theta))
    scale = objective.scale
    origin = xy0.mean(axis=0)
    records,snapshots = [],[]
    history = dict(faces=topology.faces.copy(),snapshots=snapshots,records=records,initial_xy=xy0.copy())
    _LAST_EQ5_HISTORY = history
    previous = xy0.copy()
    def record(xy,step,initial=False,final=False):
        _,_,stats = objective.evaluate(xy)
        row = dict(iteration=len(records),step=float(step),**stats)
        records.append(row)
        if initial or final or row['iteration']%10==0:
            snapshots.append(dict(row,xy=xy.copy(),initial=initial,final=final))
        print('[PAPER-EQ5-ITER] '+' '.join(f'{k}={v:.9g}' if isinstance(v,float) else f'{k}={v}' for k,v in row.items()),flush=True)
        return row
    record(xy0,0.,initial=True)
    print(f'[PAPER-EQ5-TOPOLOGY] source_vertices={len(mesh_2d.vertices)} linkage_vertices={len(xy0)} tiles={len(topology.faces)} pairwise_hinges={len(topology.hinges)} gap_angles={len(topology.gaps)}',flush=True)
    def callback(z):
        nonlocal previous
        xy = z.reshape(-1,2)*scale+origin
        step = float(np.linalg.norm(xy-previous));previous=xy.copy()
        row=record(xy,step)
        if row['EFlat']>records[-2]['EFlat']+1e-12*scale*scale:
            raise RuntimeError('Eq.5 accepted an increasing objective')
        if progress_callback:
            progress_callback('Paper Eq.5 K2D',min(1.,row['iteration']/iterations),f"iter {row['iteration']}/{iterations}; EFlat={row['EFlat']:.5g}; collisions={row['collisions']}; fab={row['fab_violations']}")
    result = _minimize_flat(objective,xy0,iterations,callback)
    xy = result.x.reshape(-1,2)*scale+origin
    # No unscored hinge/rigid/centroid projection after the solve.
    _,_,final = objective.evaluate(xy)
    if not objective.valid(xy) or final['EFlat']>records[0]['EFlat']+1e-12*scale*scale:
        raise RuntimeError('Eq.5 failed finite/geometry/objective validation')
    if snapshots[-1]['iteration']==records[-1]['iteration']:
        snapshots[-1]['final']=True
    else:
        snapshots.append(dict(records[-1],xy=xy.copy(),final=True))
    history['final_xy']=xy.copy()
    history['solver_message']=str(result.message)
    feasible = final['collisions']==0 and final['fab_violations']==0
    metrics = dict(getattr(mesh_2d,'metrics',{}))
    metrics.update(final)
    metrics.update(objective='EFlat = w_edge*EEdge + w_collision*SAT_depth_squared + w_fab*EFab',
                   paper_eq5_unified=True,paper_eq5_auxetic_topology=False,
                   paper_eq5_pairwise_linkage=True,paper_eq5_fab_topology_verified=True,
                   paper_fab_gap_constraint_count=len(topology.gaps),
                   paper_fab_violation_count=final['fab_violations'],
                   paper_fab_gap_min_deg=final['gap_min_deg'],paper_fab_gap_max_deg=final['gap_max_deg'],
                   collision_count_after=final['collisions'],edge_matching_error=final['edge_mean'],
                   optimizer_iterations=int(result.nit),optimizer_success=bool(result.success),
                   optimizer_message=str(result.message),fabrication_feasible=feasible,theta_min_deg=theta,
                   convergence_status=('feasible_stationary' if feasible and result.success else 'feasible_iteration_limit' if feasible else 'residual_constraints'),
                   actual_backend='sparse_preconditioned_lbfgs_analytic_gradient',
                   paper_collision_discretization='all-pair convex SAT squared penetration depth; differs from cited local half-plane projection',
                   paper_fab_projection='exact vector-pair nearest angle-cone projection; variable lengths',
                   paper_weight_w1_edge=weights[0],paper_weight_w2_collision=weights[1],paper_weight_w3_fab=weights[2],
                   eq5_history_snapshot_count=len(snapshots))
    out=type(mesh_2d)(np.column_stack((xy,np.zeros(len(xy)))),topology.faces.copy(),mesh_2d.grid,'K2D',metrics,list(getattr(mesh_2d,'split_lines',[])))
    out.linkage_topology=topology
    # Store diagnostics with the result, not only in process-global UI state.
    out.eq5_history=history
    report_type=getattr(pipeline,'StageReport',None)
    if report_type is None:
        raise RuntimeError('Eq.5 requires the pipeline StageReport type')
    report=report_type(name='M2D -> K2D',objective=metrics['objective'],
                       before_error=records[0]['EFlat'],after_error=final['EFlat'],
                       constraint_violation=float(final['collisions']+final['fab_violations']),
                       computation_time=time.perf_counter()-start,
                       counts={'vertices':len(xy),'quads':len(topology.faces),'hinges':len(topology.hinges)})
    print(f"[PAPER-EQ5-DONE] status={metrics['convergence_status']} solver={result.message} seconds={report.computation_time:.3f}",flush=True)
    return out,report


__all__=['optimize_paper_eq5','get_last_eq5_history','LinkageTopology','build_linkage_topology']
