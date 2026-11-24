"""Rigid rotating triangles and inward, dihedral-bisected panel solids."""
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, least_squares
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

from job import panel_depth_spec
from shell_geometry import shell_topology
from contact import collision_energy_gradient, solid_contacts, hinge_clearance_bounds


def read(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def unit(v):
    return v/np.linalg.norm(v,axis=-1,keepdims=True)


def boundary_panel_ids(out, count):
    ids = np.arange(count)
    if (out/'grid/result.pkl').is_file() and (out/'remesh3d.pkl').is_file():
        grid = read(out/'grid/result.pkl')
        removed = read(out/'remesh3d.pkl')['removed_panels']
        ids = np.delete(np.arange(len(grid['Tg'])), removed)
        boundary = list(grid['pin_g'].values()) + [v for vs in grid['sec_g'].values() for v in vs]
        return ids, np.isin(grid['Tg'][ids], boundary).any(axis=1)
    return ids, np.zeros(count,bool)


def kagome_joints(faces, up):
    owners={}
    for i,f in enumerate(faces):
        for k in range(3):owners.setdefault(tuple(sorted((f[k],f[(k+1)%3]))),[]).append((i,k))
    hinges=[]
    for entries in owners.values():
        if len(entries)==1:continue
        (a,k),(b,j)=entries
        ka=(k+int(up[a]))%3; kb=(j+int(up[b]))%3
        if up[a]==up[b] or faces[a,ka]!=faces[b,kb]:raise ValueError('Invalid alternating hinge topology')
        hinges.append((a,ka,b,kb))
    return np.asarray(hinges,int).reshape(-1,4)


def tiles(out):
    mesh=read(out/'remesh3d.pkl')
    faces=mesh['T']
    top=mesh['X'][faces]
    normal=unit(np.cross(top[:,1]-top[:,0],top[:,2]-top[:,0]))
    edges=np.roll(top,-1,axis=1)-top
    path = out/'settings.json'
    cfg = json.loads(path.read_text()) if path.is_file() else {}
    mode, value = panel_depth_spec(cfg)
    depth = value*np.linalg.norm(edges,axis=2).mean(axis=1) if mode=='edge_ratio' else np.full(len(top),value)
    if cfg.get('extrusion_mode')=='outer_layer':
        from fixed_skin import build_shell
        # K2D only needs the base triangulation to compute the planar pose.
        seed=np.concatenate([top,top-normal[:,None]*depth[:,None,None]],axis=1)
        build_shell(out,mesh,seed,flat_layout(out,seed),cfg,depth)
        return
    # Fit the incident inward offset planes once per mesh vertex. Both owners
    # of an edge then use exactly the same four side corners and diagonal.
    from scipy.optimize import linprog
    shift = np.zeros_like(mesh['X'])
    panel_ids, boundary = boundary_panel_ids(out,len(faces))
    on_boundary = np.zeros(len(shift),bool)
    on_boundary[faces[boundary]] = True
    for v in range(len(shift)):
        ids = np.where((faces==v).any(axis=1))[0]
        N, h = normal[ids], depth[ids]
        fit = np.linalg.lstsq(np.vstack([N, .12*np.eye(3)]), np.r_[-h, np.zeros(3)], rcond=None)[0]
        if np.any(N @ fit >= 0):
            # A sharply folded fan can require a different inward direction.
            lp = linprog([0.,0.,0.,-1.], A_ub=np.c_[N, np.ones(len(N))], b_ub=np.zeros(len(N)),
                         bounds=[(-1,1)]*3+[(0,1)],method='highs')
            if not lp.success or lp.x[3] < 1e-8:
                raise ValueError(f'頂点 {v} の周囲のパネルに共通する内向きの方向がありません。')
            fit = lp.x[:3]*np.median(h)/np.median(-N@lp.x[:3])
        shift[v] = fit * min(1., 2*h.min()/np.linalg.norm(fit))
        if on_boundary[v] and np.min(-N@shift[v]/h)<.85:
            # A common displacement must serve every incident side wall. Fit the
            # weakest corner as well as the mean rather than leaving one face thin.
            length_scale = h.min()
            normals = -N*(length_scale/h)[:,None]
            start = shift[v]/length_scale
            fraction = normals@start
            def objective(z):
                return -z[3] + .01*(z[:3]@z[:3]) + .02*np.mean((normals@z[:3]-1.)**2)
            trial = minimize(objective, np.r_[start,fraction.min()], method='SLSQP',
                bounds=[(-2.,2.)]*3+[(0.,1.)],
                constraints=({'type':'ineq','fun':lambda z: normals@z[:3]-z[3]},
                             {'type':'ineq','fun':lambda z: max(1.1,fraction.max())-normals@z[:3]},
                             {'type':'ineq','fun':lambda z: 4.-z[:3]@z[:3]}))
            if trial.success and (normals@trial.x[:3]).min()>fraction.min():
                shift[v] = length_scale*trial.x[:3]
    surface_tris, tets = shell_topology(faces)
    owners = np.arange(len(faces))[:,None,None]
    def volumes(points):
        tetra = points[owners, tets]
        return np.linalg.det(tetra[:,:,1:]-tetra[:,:,:1])
    ideal = np.concatenate([top, top-depth[:,None,None]*normal[:,None,:]], axis=1)
    reference_volume = volumes(ideal)
    from igl.copyleft.cgal import remesh_self_intersections
    n = len(shift)
    pairs = faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    _, inverse, counts = np.unique(np.sort(pairs,axis=1),axis=0,return_inverse=True,return_counts=True)
    rim = np.where(counts[inverse].reshape(-1,3)==1)
    node_ids = np.column_stack([faces,faces+n])
    rim_tris = surface_tris[rim[0],2+2*rim[1]]
    rim_tris2 = surface_tris[rim[0],3+2*rim[1]]
    walls = np.vstack([node_ids[rim[0][:,None],rim_tris],node_ids[rim[0][:,None],rim_tris2]])
    shell_faces = np.vstack([faces,faces[:,::-1]+n,walls])
    requested_shift = shift.copy()
    def invalid_vertices(offset):
        bottom = (mesh['X']+offset)[faces]
        solid = np.concatenate([top,bottom],axis=1)
        bad = np.any(volumes(solid)/reference_volume <= 1e-9, axis=1)
        if bad.any():
            return np.unique(faces[bad])
        shell = np.vstack([mesh['X'], mesh['X']+offset])
        crossings = remesh_self_intersections(shell,shell_faces,detect_only=True)[2]
        if not len(crossings): return np.empty(0, int)
        if ((crossings < len(faces)).all(axis=1)).any():
            raise ValueError('K3Dの外面が既に自己交差しているため、T3Dを押し出せません。')
        affected = shell_faces[crossings].ravel()
        return np.unique(affected[affected>=n]-n)
    for _ in range(60):
        bad = invalid_vertices(shift)
        if not len(bad): break
        shift[bad] *= .5
    else:
        raise ValueError('共有側面を保った内向き押し出しで、反転する要素があります。')
    # Halving finds a valid shell but leaves healthy neighbours under-extruded.
    # Restore available depth locally; accept only entire intersection-free shells.
    if np.any(shift != requested_shift):
        for _ in range(6):
            increment = .5*(requested_shift-shift)
            while np.any(increment):
                bad = invalid_vertices(shift+increment)
                if not len(bad):
                    shift += increment
                    break
                if not np.any(increment[bad]): break
                increment[bad] = 0.
    # A shift shared by every incident face keeps mating side walls identical.
    # Where a closed hinge necessarily intersects below the pivot, search the
    # fan for an inward direction that retains the depth already obtained.
    grid = read(out/'grid/result.pkl') if (out/'grid/result.pkl').exists() else {}
    joints = kagome_joints(grid['Tg'][panel_ids], grid['lattice_up'][panel_ids]) if 'lattice_up' in grid else []
    repaired_vertices = 0
    if len(joints):
        frame_x = edges[:,0]/np.linalg.norm(edges[:,0],axis=1)[:,None]
        basis = np.stack((frame_x,np.cross(normal,frame_x),normal),axis=1)
        local_top = (top-top[:,:1]) @ basis.transpose(0,2,1)
        def rest_solids(offset, selected):
            delta = offset[faces[selected]] @ basis[selected].transpose(0,2,1)
            return np.concatenate((local_top[selected],local_top[selected]+delta),axis=1)
        bounds = hinge_clearance_bounds(rest_solids(shift,np.arange(len(faces))),tets,joints)
        pivots = faces[joints[:,0],joints[:,1]]
        concerned = np.unique(pivots[(bounds > 1e-8) & on_boundary[pivots]])
        for v in concerned:
            fans = np.flatnonzero((faces==v).any(1))
            local_joints = joints[pivots==v].copy()
            local_joints[:,[0,2]] = np.searchsorted(fans,local_joints[:,[0,2]])
            def clearance():
                return hinge_clearance_bounds(rest_solids(shift,fans),tets[fans],local_joints)
            original = shift[v].copy()
            start_bounds = clearance()
            merit = (start_bounds.max(),start_bounds.sum())
            best_shift = original
            nn, h = normal[fans], depth[fans]
            achieved = -nn@original
            floor = -np.einsum('fvi,fi->fv',shift[faces[fans]],nn).max(axis=1)
            # Preserve the minimum depth of each participating panel, with
            # freedom to redistribute excess depth at this particular corner.
            axis = unit(original)
            tx = unit(np.cross(axis, np.eye(3)[np.argmin(abs(axis))]))
            ty = np.cross(axis,tx)
            for radius in (.05,.15,.35,.65,1.):
                for theta in np.arange(12)*np.pi/6:
                    edge_shift = original + radius*h.min()*(np.cos(theta)*tx+np.sin(theta)*ty)
                    actual = -nn@edge_shift
                    if np.any(actual<=0):
                        continue
                    edge_shift *= max(1.,np.max(floor/actual))
                    if (np.linalg.norm(edge_shift)>2*h.min()+1e-10 or
                        np.any(-nn@edge_shift>np.maximum(1.15*h,achieved)+1e-10)):
                        continue
                    shift[v]=edge_shift
                    cost=clearance()
                    key=(cost.max(),cost.sum())
                    if key < merit and not len(invalid_vertices(shift)):
                        best_shift,merit=edge_shift.copy(),key
                    shift[v]=original
                if merit[0]<1e-10:
                    break
            shift[v]=best_shift
            repaired_vertices += int(not np.array_equal(original,best_shift))
    from planar_sides import fit_sides
    baseline_bottom = (mesh['X']+shift)[faces]
    shift, side_fit = fit_sides(mesh['X'],faces,shift,normal,depth,invalid_vertices)
    bottom = (mesh['X']+shift)[faces]
    solid = np.concatenate([top,bottom],axis=1)
    inward = -np.einsum('fvi,fi->fv',bottom-top,normal)
    if cfg.get('extrusion_mode')=='fixed_skin':
        from fixed_skin import build_shell
        layout=flat_layout(out,solid)
        build_shell(out,mesh,solid,layout,cfg,depth)
        return
    np.savez_compressed(out/'tiles.npz',K3D=top,T3D=solid,depth=inward.mean(1),requested_depth=depth,
                        surface_tris=surface_tris,tets=tets,baseline_bottom=baseline_bottom)
    fraction = (inward/depth[:,None]).min(1)
    boundary_report = {}
    if boundary.any():
        worst = np.flatnonzero(boundary)[np.argmin(fraction[boundary])]
        boundary_report = dict(boundary_minimum_depth_fraction=float(fraction[worst]),
                               boundary_thinnest_panel_id=int(panel_ids[worst]))
    (out/'tiles_summary.json').write_text(json.dumps(dict(panels=len(top),miter_apex_panels=[],
        **boundary_report,side_planarity=side_fit,
        hinge_offset_vertices_adjusted=repaired_vertices,
        corner_depth_range=[float(inward.min()), float(inward.max())],
        minimum_depth_fraction=float((inward/depth[:,None]).min()),shared_edges=int((counts==2).sum()),
        shell_self_intersections=0,side_construction='shared_vertices_and_diagonal'),indent=2))
    print('panels',len(top),'inward corner depth',inward.min(),inward.max())


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.stack([c,-s,s,c],axis=-1).reshape(-1,2,2)


def regular_cells(grid):
    # Recover the lattice in the original face-corner order, including old runs.
    faces, up = grid['Tg'], grid['lattice_up']
    offsets = np.where(up[:,None,None], [[[0,0],[1,0],[0,1]]], [[[1,0],[1,1],[0,1]]])
    neighbours = [[] for _ in grid['Xg']]
    for face, uv in zip(faces, offsets):
        for i in range(3):
            j = (i+1)%3
            neighbours[face[i]].append((face[j],uv[j]-uv[i]))
            neighbours[face[j]].append((face[i],uv[i]-uv[j]))
    coords = np.zeros((len(neighbours),2),int)
    seen = {int(faces[0,0])}
    queue = list(seen)
    for i in queue:
        for j, delta in neighbours[i]:
            if j not in seen:
                coords[j] = coords[i]+delta
                seen.add(j)
                queue.append(j)
    return (coords @ [[1.,0.],[.5,np.sqrt(3.)/2]])[faces]


def flat_layout(out, solid):
    grid=read(out/'grid/result.pkl'); mesh=read(out/'remesh3d.pkl')
    ids=np.delete(np.arange(len(grid['Tg'])),mesh['removed_panels'])
    faces=grid['Tg'][ids]
    if not np.array_equal(mesh['kept_vertices'][mesh['T']],grid['wid'][faces]):
        raise ValueError('M2D and K3D panel IDs disagree')
    up=grid['lattice_up'][ids]
    uv=grid['original_grid'][faces]
    top=solid[:,:3]; centers=top.mean(1)
    e=unit(top[:,1]-top[:,0]); n=unit(np.cross(e,top[:,2]-top[:,0]))
    basis=np.stack((e,np.cross(n,e),n),axis=1)
    local=(solid-centers[:,None])@basis.transpose(0,2,1)
    # Characteristic length is the square root of area. Find the maximum overlap
    # of [1/r, 2/r] in log scale. Tie-break toward a 30 degree opening.
    area_uv=np.abs(np.linalg.det(uv[:,1:]-uv[:,:1]))
    area_ref=np.linalg.norm(np.cross(top[:,1]-top[:,0],top[:,2]-top[:,0]),axis=1)
    ratio=np.sqrt(area_uv/area_ref)
    if np.any(ratio<=0): raise ValueError('Zero-area mapped Omega panel')
    starts=np.sort(-np.log(ratio)); ends=starts+np.log(2.)
    preferred=np.log(np.sqrt(3.)/np.median(ratio))
    candidates=np.unique(np.r_[starts,ends,preferred])
    counts=np.searchsorted(starts,candidates+1e-13,'right')-np.searchsorted(ends,candidates-1e-13,'left')
    choices=candidates[counts==counts.max()]
    scale=np.exp(choices[np.argmin(abs(choices-preferred))])
    lam=scale*ratio
    theta=np.pi/3-np.arccos(np.clip(lam,1.,2.)/2)
    U,_,Vh=np.linalg.svd(local[:,:3,:2].transpose(0,2,1)@(uv-uv.mean(1)[:,None]))
    align=U@Vh
    U[:,:,-1]*=np.linalg.det(align)[:,None]
    local[:,:,:2]=local[:,:,:2]@(U@Vh)@rotation(np.where(up,-theta,theta)).transpose(0,2,1)
    local[:,:,:2]+=scale*uv.mean(1)[:,None]
    hinges=kagome_joints(faces,up)
    return dict(K2D=local[:,:3],rest=local,hinges=hinges,panel_ids=ids,opening=lam,scale=scale,
                        regular_cells=regular_cells(grid)[ids], lattice_up=up)


def flat(out):
    with np.load(out/'tiles.npz') as shell:
        if 'layout_rest' in shell:
            layout={key[7:]:shell[key] for key in shell.files if key.startswith('layout_')}
        else:
            layout=flat_layout(out,shell['T3D'])
    np.savez_compressed(out/'flat.npz',**layout)
    ids,hinges,lam,scale=(layout[k] for k in ('panel_ids','hinges','opening','scale'))
    (out/'hinges.json').write_text(json.dumps(dict(panel_ids=ids.tolist(),count=len(hinges),hinges=hinges.tolist())))
    print('global scale',scale,'lambda out of range',np.sum((lam<1-1e-10)|(lam>2+1e-10)),'of',len(lam))


def closure(out):
    cfg=json.loads((out/'settings.json').read_text()) if (out/'settings.json').exists() else {}
    if cfg.get('extrusion_mode') in ('fixed_skin','outer_layer'):
        from triangle_linkage import make_linkage
        make_linkage(out,cfg)
        return
    data=np.load(out/'flat.npz')
    rest=data['rest'];joints=data['hinges']
    if not len(joints):
        np.savez_compressed(out/'closure.npz',T2D=rest)
        (out/'closure.json').write_text(json.dumps(dict(hinges=0,max_gap=0,max_penetration=0,overlap_pairs=0,tolerance=0,converged=len(rest)==1)))
        return
    a,ka,b,kb=joints.T
    centers=rest[:,:3,:2].mean(1)
    edge=np.linalg.norm(rest[:,0]-rest[:,1],axis=1).mean()
    local=(rest[:,:,:2]-centers[:,None])/edge
    with np.load(out/'tiles.npz') as shell:
        tets = shell.get('tets')
    cfg_path=out/'settings.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()).get('avoid_t2d_overlap',False):
        from separated_layout import separated_layout, validate_separated
        if 'regular_cells' in data:
            cells,up=data['regular_cells'],data['lattice_up']
        else:
            grid=read(out/'grid/result.pkl')
            cells=regular_cells(grid)[data['panel_ids']]
            up=grid['lattice_up'][data['panel_ids']]
        def span_score(s):
            g=np.linalg.norm(s[a,ka,:2]-s[b,kb,:2],axis=1)
            return g.max()+.3*g.mean()
        best=min((separated_layout(rest,joints,cells,up,np.deg2rad(angle),steps=45)
                  for angle in (15.,45.)),key=span_score)
        solid=separated_layout(rest,joints,cells,up,0.,steps=35,compact_from=best)
        report=validate_separated(rest,solid,tets,joints)
        np.savez_compressed(out/'closure.npz',T2D=solid)
        (out/'closure.json').write_text(json.dumps(report,indent=2))
        print(report)
        return
    base=centers/edge
    n=len(rest)
    hulls=[ConvexHull(p[:,:2]).vertices.tolist() for p in rest]
    width=max(map(len,hulls))
    indices=np.array([[6*i+k for k in h+[h[-1]]*(width-len(h))] for i,h in enumerate(hulls)])
    def pose(q):
        q=q.reshape(n,3)
        rel=local@rotation(q[:,0]).transpose(0,2,1)
        return rel+(base+q[:,1:])[:,None],rel
    def delta(q):
        p,_=pose(q)
        return (p[a,ka]-p[b,kb]).ravel()
    rows=np.repeat(np.arange(2*len(a)),4)
    cols=np.stack([np.c_[3*a,3*a+1,3*b,3*b+1],np.c_[3*a,3*a+2,3*b,3*b+2]],axis=1).ravel()
    ones=np.ones(len(a))
    def jacobian(q):
        _,rel=pose(q)
        A=rel[a,ka]; B=rel[b,kb]
        values=np.stack([np.c_[-A[:,1],ones,B[:,1],-ones],np.c_[A[:,0],ones,-B[:,0],-ones]],axis=1)
        return coo_matrix((values.ravel(),(rows,cols)),shape=(2*len(a),3*n)).tocsr()
    def close(q, iterations=25):
        for _ in range(iterations):
            d=delta(q)
            if np.max(np.abs(d),initial=0.)<1e-9: break
            J=jacobian(q)
            correction=J.T@spsolve((J@J.T+1e-9*eye(J.shape[0])).tocsc(),d)
            q=q-correction/max(1.,np.abs(correction).max()/.25)
        return q
    q=least_squares(delta,np.zeros(3*n),jac=jacobian,max_nfev=220,gtol=1e-11,xtol=1e-12,ftol=1e-12).x
    def solid_eval(q, return_pairs=False, power=2):
        p,_=pose(q)
        solid=np.empty_like(rest)
        solid[:,:,:2]=p; solid[:,:,2]=rest[:,:,2]/edge
        return solid_contacts(solid,base+q.reshape(n,3)[:,1:],tets,return_pairs=return_pairs,power=power)
    def objective(q, weight):
        d=delta(q)
        cost,grad,_,_=solid_eval(q)
        return d@d+weight*cost,2*(jacobian(q).T@d)+weight*grad.ravel()
    # First find a low-contact basin. Then minimize penetration in the
    # nullspace of the hinge constraints, retracting every accepted trial.
    # Continue from an exactly closed rotating-triangle mechanism. The
    # intermediate shapes are solver initializations; the final SAT test uses
    # all six original vertices and the original tetrahedra for each tile.
    seed = 'K2D'
    if 'regular_cells' in data:
        cells = data['regular_cells']
        cell_centers = cells.mean(1)
        cell_centers -= cell_centers.mean(0)
        ideal_shape = cells-cells.mean(1)[:,None,:]
        material = local[:,:3].copy()
        def seed_score(q):
            gap = np.linalg.norm(delta(q).reshape(-1,2),axis=1).max()
            cost,_,_,depth = solid_eval(q)
            return max(gap,depth), gap*gap+cost
        best_score = seed_score(q)
        for opening in (1.55, 1.85):
            angle = np.pi/3-np.arccos(opening/2)
            desired = ideal_shape @ rotation(np.where(data['lattice_up'],-angle,angle)).transpose(0,2,1)
            # Keep the guide in the orientation of the requested K2D layout.
            U,_,Vh = np.linalg.svd(cell_centers.T@(base-base.mean(0)))
            U[:,-1] *= np.linalg.det(U@Vh)
            align = U@Vh
            desired = desired@align
            dot = (material*desired).sum((1,2))
            cross = (material[:,:,0]*desired[:,:,1]-material[:,:,1]*desired[:,:,0]).sum(1)
            angles = np.arctan2(cross,dot)
            guide = desired @ rotation(angles)
            trial = np.column_stack((angles, opening*cell_centers@align+base.mean(0)-base)).ravel()
            for fraction in np.linspace(.1,1.,10):
                local[:,:3] = guide*(1.-fraction)+material*fraction
                trial = close(trial)
            local[:,:3] = material
            score = seed_score(trial)
            if score < best_score:
                q, best_score, seed = trial, score, f'Kagome {opening}'
    q=minimize(objective,q,args=(.25,),jac=True,method='L-BFGS-B',options=dict(maxiter=700,ftol=1e-14)).x
    q=close(q)
    step=1.
    for iteration in range(250):
        cost,grad=objective(q,1.)
        if cost<1e-12: break
        J=jacobian(q)
        correction=spsolve((J@J.T+1e-10*eye(J.shape[0])).tocsc(),J@grad)
        direction=J.T@correction-grad
        step=min(step*1.6,1./max(1e-8,np.max(np.abs(direction))))
        while step>1e-8:
            trial=close(q+step*direction)
            if np.max(np.abs(delta(trial)),initial=0.)<1e-7 and objective(trial,1.)[0]<cost-1e-13:
                q=trial
                break
            step*=.5
        else: break
    q=close(q)
    separated = None
    if 'regular_cells' in data:
        from separated_layout import separated_layout
        separated = separated_layout(rest, joints, data['regular_cells'], data['lattice_up'], np.deg2rad(15.))
        # Project the collision-free embedding onto the point-hinge constraints
        # from its own basin, and retain it only if the physical contacts improve.
        target = separated[:,:3,:2]/edge
        c = target.mean(1)
        vec = target-c[:,None]
        mat = local[:,:3]
        angle = np.arctan2((mat[:,:,0]*vec[:,:,1]-mat[:,:,1]*vec[:,:,0]).sum(1),(mat*vec).sum((1,2)))
        trial = np.column_stack((angle,c-base)).ravel()
        def closed_score(x):
            gap = np.linalg.norm(delta(x).reshape(-1,2),axis=1).max()
            _,_,count,depth = solid_eval(x)
            return (gap>1e-6,count,depth)
        score = closed_score(q)
        from scipy.sparse import vstack
        identity = eye(len(q),format='csr')
        for stiffness in (.1,.02,.003):
            anchor = trial.copy()
            trial = least_squares(lambda x: np.r_[delta(x),stiffness*(x-anchor)],trial,
                jac=lambda x: vstack([jacobian(x), stiffness*identity],format='csr'),
                max_nfev=100,ftol=1e-9).x
            candidate = close(trial, iterations=250)
            candidate_score=closed_score(candidate)
            if candidate_score < score:
                q, score, seed = candidate, candidate_score, 'separation-constrained seed'
    # On the closed mechanism, prioritize deep intersections. Keep candidates
    # only when the peak depth and the number of colliding solids improve or tie.
    for _ in range(80):
        cost,gradient,count,peak = solid_eval(q,power=6)
        if not count:break
        J=jacobian(q)
        grad=gradient.ravel()
        direction=J.T@spsolve((J@J.T+1e-10*eye(J.shape[0])).tocsc(),J@grad)-grad
        direction/=max(np.max(np.abs(direction)),1e-14)
        for step in (.15,.05,.015,.005,.001):
            trial=close(q+step*direction)
            if np.max(np.abs(delta(trial)),initial=0.)>1e-7:continue
            energy,_,contacts,depth=solid_eval(trial,power=6)
            if energy<cost*(1.-1e-6) and contacts<=count and depth<=peak:
                q=trial
                break
        else:break
    p,_=pose(q)
    _,_,_,diag=collision_energy_gradient(p.reshape(-1,2),indices,return_diagnostics=True)
    gap=float(np.linalg.norm(p[a,ka]-p[b,kb],axis=1).max(initial=0))
    solid=rest.copy();solid[:,:,:2]=p*edge
    report=dict(layout_seed=seed,hinges=len(joints),max_gap=gap*edge,max_penetration=diag['penetration_max']*edge,
                overlap_pairs=diag['collision_pair_count'],tolerance=edge*1e-3)
    _,_,count,depth,pairs,penetration=solid_eval(q, return_pairs=True)
    ids, boundary = boundary_panel_ids(out,len(rest))
    touches = boundary[pairs].any(1)
    boundary_hinges = boundary[a] | boundary[b]
    report.update(boundary_overlap_pairs=int(touches.sum()),
                  boundary_contact_panel_ids=ids[pairs[touches]].tolist(),
                  boundary_max_penetration=float(penetration[touches].max(initial=0.)*edge),
                  boundary_max_gap=float(np.linalg.norm(p[a,ka]-p[b,kb],axis=1)[boundary_hinges].max(initial=0.)*edge))
    report.update(footprint_penetration=report['max_penetration'],max_penetration=depth*edge,
                  overlap_pairs=count,converged=bool(max(gap,depth)<=1e-3))
    # At a common depth, a disk inside each panel and centred on its hinge
    # endpoint is invariant under every planar rotation about that hinge.
    if tets is None:
        thickness=-rest[:,3:,2].mean(1)
        sample_depth=.5*np.minimum(thickness[a],thickness[b])
        def inscribed_radius(ids,corners):
            face=rest[ids,:3,:2]
            mid=face+(rest[ids,3:,:2]-face)*(sample_depth/thickness[ids])[:,None,None]
            side=np.roll(mid,-1,axis=1)-mid
            diff=rest[ids,corners,:2][:,None]-mid
            return ((side[:,:,0]*diff[:,:,1]-side[:,:,1]*diff[:,:,0])/np.linalg.norm(side,axis=2)).min(1)
        ra=inscribed_radius(a,ka);rb=inscribed_radius(b,kb)
        always=(ra>edge*1e-9)&(rb>edge*1e-9)
        report.update(unavoidable_hinge_pairs=joints[always].tolist(),
                      separation_lower_bound=float(np.max((ra+rb)[always],initial=0.)))
    else:
        bounds = hinge_clearance_bounds(rest,tets,joints)
        blocked = bounds>report['tolerance']
        report.update(unavoidable_hinge_pairs=joints[blocked].tolist(),
                      boundary_unavoidable_hinges=int((blocked & boundary_hinges).sum()),
                      separation_lower_bound=float(bounds.max(initial=0.)))
    if separated is not None:
        _,_,contacts,penetration=solid_contacts(separated,separated[:,:3,:2].mean(1),tets)
        spans=np.linalg.norm(separated[a,ka,:2]-separated[b,kb,:2],axis=1)
        report['separated_layout']=dict(overlap_pairs=contacts,max_penetration=penetration,
            max_gap=float(spans.max()),median_gap=float(np.median(spans)))
        np.savez_compressed(out/'closure.npz',T2D=solid,separated=separated)
    else:
        np.savez_compressed(out/'closure.npz',T2D=solid)
    (out/'closure.json').write_text(json.dumps(report,indent=2))
    print(report)


if __name__=='__main__':
    stage=sys.argv[1]; out=Path(sys.argv[2])
    if stage=='print':
        from manufacturing import manufacture
        manufacture(out)
    else:
        from drift_preview import ThicknessConflict
        try:
            {'tiles':tiles,'flat':flat,'closure':closure}[stage](out)
        except ThicknessConflict as exc:
            print(exc,flush=True)
            sys.exit(3)
