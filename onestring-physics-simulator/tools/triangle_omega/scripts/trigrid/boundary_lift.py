"""Adjust where a fixed planar panel grid samples its fitted surface chart."""
import numpy as np
from matplotlib.tri import Triangulation
from boundary_design import angle_report, independent_stars


def cross2(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def max_cosine(edges):
    return np.maximum.reduce([-np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-24)
                              for a,b in zip(edges,edges[1:]+edges[:1])])


def shape_cost(p,q,r):
    # Base is q--r. Use all three choices of base so the cost is symmetric.
    total = 0.
    for a,b,c in ((p,q,r),(q,r,p),(r,p,q)):
        base,side = c-b,a-b
        length2 = np.maximum(np.sum(base*base,axis=-1),1e-24)
        foot = np.sum(base*side,axis=-1)/length2
        height = np.linalg.norm(np.cross(base,side),axis=-1)/length2
        total += ((foot-.5)**2 + (height-np.sqrt(3)/2)**2)/np.maximum(height,.001)**2
    return total/3.


def max_angle_error(p,q,r):
    a,b,c = q-p,r-q,p-r
    cross = np.linalg.norm(np.cross(a,-c),axis=-1)
    angles = [np.arctan2(cross,-np.sum(l*r,axis=-1)) for l,r in ((c,a),(a,b),(b,c))]
    return np.max(np.abs(np.asarray(angles)-np.pi/3),axis=0)


def improve_boundary_lift(grid, source_uv):
    faces = grid['Tg']
    layout = grid['Xg']
    sample = layout.copy()
    welded = grid['wid']
    xyz = grid['VW'][welded].copy()
    boundary = np.unique(list(grid['pin_g'].values())+[v for bank in grid['sec_g'].values() for v in bank]).astype(int)
    boundary_faces = np.isin(faces,boundary).any(1)
    if not boundary_faces.any():
        return dict(moved=0, reason='no boundary panels')
    before = angle_report(xyz[faces[boundary_faces]])
    stats = dict(moved=0, before=before, after=before)
    # Keep both copies of every seam vertex paired, while the interior shares the fit.
    free = np.setdiff1d(np.arange(len(layout)),boundary)
    if not len(free):
        return stats
    uv, fine = grid['U4'],grid['Fw']
    chart, surface = uv[fine], grid['Xw'][fine]
    try:
        finder = Triangulation(*uv.T,triangles=fine).get_trifinder()
        inverse = np.linalg.inv((chart[:,1:]-chart[:,:1]).transpose(0,2,1))
    except (RuntimeError,np.linalg.LinAlgError):
        return dict(**stats,reason='chart unavailable')
    initial_face = finder(*layout[free].T)
    if np.any(initial_face<0):
        return dict(**stats,reason='interior vertex outside chart')
    def interpolate(points, face_id, values):
        bary = np.einsum('nij,nj->ni',inverse[face_id],points-chart[face_id,0])
        corners = values[fine[face_id]]
        return corners[:,0] + np.einsum('nij,ni->nj',corners[:,1:]-corners[:,:1],bary)
    surf_normal = np.cross(surface[:,1]-surface[:,0],surface[:,2]-surface[:,0])
    reference_normal = np.zeros_like(xyz)
    reference_normal[free] = surf_normal[initial_face]
    tri = xyz[faces]
    panel_normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    orig = xyz.copy()
    lengths = np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2)
    pitch = np.median(np.linalg.norm(layout[faces]-np.roll(layout[faces],1,axis=1),axis=2))
    pitch3 = np.median(lengths)
    reference_size = np.maximum(lengths.mean(1),1e-12)
    interior = angle_report(tri[~boundary_faces]) if (~boundary_faces).any() else None
    angle_limit = np.full(len(tri),np.inf)
    min_cos = 1.
    if interior is not None:
        angle_limit[~boundary_faces] = np.deg2rad(max(20.,interior['max_deviation']))
        min_cos = np.cos(np.deg2rad(interior['min_angle']))
    weight = np.where(boundary_faces,1.5,1.)
    stars = independent_stars(len(layout),faces,free)
    angles = np.arange(8)*np.pi/4
    directions = np.column_stack((np.cos(angles),np.sin(angles)))
    for step in (.18,.12,.08,.045,.02,.008):
        for sweep in range(8):
            changes = 0
            for ids,fi,others in stars:
                q,r = sample[others[:,:,0]],sample[others[:,:,1]]
                Q,R = xyz[others[:,:,0]],xyz[others[:,:,1]]
                def cost(probe):
                    P = probe[:,None,:]
                    shape = shape_cost(P,Q,R)
                    size = (np.linalg.norm(Q-P,axis=2)+np.linalg.norm(R-P,axis=2)+np.linalg.norm(Q-R,axis=2))/3
                    err = max_angle_error(P,Q,R)
                    over = np.maximum(0.,err-np.deg2rad(18.))
                    return (weight[fi]*(shape+6*over**2) + 7.*over**2 + .15*(size/reference_size[fi]-1.)**2).sum(1)
                best_x,best_v = sample[ids].copy(),xyz[ids].copy()
                best = cost(best_v)
                for direction in directions:
                    proposal = sample[ids]+step*pitch*direction
                    face_id = finder(*proposal.T)
                    safe = np.maximum(face_id,0)
                    mapped = interpolate(proposal,safe,grid['Xw'])
                    # Preserve orientation in both charts and stay on the starting sheet.
                    normal = np.cross(Q-mapped[:,None],R-mapped[:,None])
                    dot = np.sum(normal*panel_normal[fi],axis=2)
                    ok = ((face_id>=0) & (np.sum(surf_normal[safe]*reference_normal[ids],axis=1)>0)
                          & (np.linalg.norm(mapped-orig[ids],axis=1)<.75*pitch3)
                          & np.all(cross2(q-proposal[:,None],r-proposal[:,None])>1e-10*pitch**2,axis=1)
                          & np.all(max_angle_error(mapped[:,None],Q,R)<=angle_limit[fi]+1e-9,axis=1)
                          & np.all(dot>.35*np.linalg.norm(normal,axis=2)*np.linalg.norm(panel_normal[fi],axis=2),axis=1))
                    edges = [Q-mapped[:,None],R-Q,mapped[:,None]-R]
                    cosine = max_cosine(edges)
                    ok &= np.all(boundary_faces[fi] | (cosine<=min_cos+1e-8),axis=1)
                    value = cost(mapped)
                    take = ok & (value<best-1e-10)
                    best[take],best_x[take],best_v[take] = value[take],proposal[take],mapped[take]
                changes += np.any(best_x!=sample[ids],axis=1).sum()
                sample[ids],xyz[ids] = best_x,best_v
            if not changes:
                break
    moved = free[np.any(sample[free]!=layout[free],axis=1)]
    after = angle_report(xyz[faces[boundary_faces]])
    if any(after[k]>before[k]+1e-7 for k in ('max_deviation','p95_deviation','outside_20deg')):
        return dict(**stats,reason='boundary angle check',trial=after)
    if interior is not None:
        interior_after = angle_report(xyz[faces[~boundary_faces]])
        if interior_after['outside_20deg']>interior['outside_20deg']:
            return dict(**stats,reason='excess interior distortion')
    if len(np.unique(welded[moved]))!=len(moved) or np.any(np.bincount(welded)[welded[moved]]!=1):
        return dict(**stats,reason='shared interior vertex')
    from lift_quality import surface_quality
    adjusted = grid['VW'].copy()
    adjusted[welded[moved]] = xyz[moved]
    geometry = surface_quality(adjusted, grid['TW'], grid['X'])
    if not geometry['valid'] or geometry['max_distance_over_pitch'] > 1.:
        return dict(**stats, reason='welded surface quality', geometry=geometry)
    grid['VW'] = adjusted
    grid['inverse_sample_uv'] = sample
    if source_uv is not None and len(moved):
        grid['original_grid'][moved] = interpolate(sample[moved],finder(*sample[moved].T),source_uv)
    return dict(moved=len(moved),before=before,after=after)


def polish_welded(x, references, faces, boundary, colors, spacing, target_size, normals, project, guard):
    """Improve boundary stars on the welded surface, retaining the local projection guards."""
    initial = x.copy()
    def details(points):
        tri = points[faces]
        p,q,r = tri[:,0],tri[:,1],tri[:,2]
        err = max_angle_error(p,q,r)
        size = (np.linalg.norm(p-q,axis=1)+np.linalg.norm(q-r,axis=1)+np.linalg.norm(r-p,axis=1))/3
        cost = np.where(boundary,4.,1.)*(shape_cost(p,q,r)+10*np.maximum(err-np.deg2rad(17.),0.)**2)
        return cost+2*(size/target_size-1.)**2,err

    def metrics(err,mask):
        return (np.count_nonzero(err[mask]>np.deg2rad(20.)+1e-8),float(err[mask].max(initial=0.)))
    _,start = details(x)
    baseline = metrics(start,boundary)
    inside = metrics(start,~boundary)
    best,best_ref=x.copy(),references.copy()
    best_rank=baseline[0]/boundary.sum()+baseline[1]+np.percentile(start[boundary],95)
    for step in (.12,.08,.045,.022,.009):
        for color in range(colors.max()+1):
            selected = np.flatnonzero(colors==color)
            n = normals[references[selected]]
            tangent = np.cross(n,np.eye(3)[np.argmin(np.abs(n),axis=1)])
            tangent /= np.linalg.norm(tangent,axis=1)[:,None]
            side = np.cross(n,tangent)
            base,errors = details(x)
            best_change = np.zeros(len(selected))
            positions,where = x[selected].copy(),references[selected].copy()
            for theta in np.arange(8)*np.pi/4:
                trial = x.copy()
                trial[selected] += step*spacing[selected,None]*(np.cos(theta)*tangent+np.sin(theta)*side)
                trial,ref = project(trial,x,references)
                trial[colors!=color],ref[colors!=color]=x[colors!=color],references[colors!=color]
                trial,ref=guard(trial,ref,x,references)
                value,angle=details(trial)
                change=np.bincount(faces.ravel(),np.repeat(value-base,3),minlength=len(x))[selected]
                # A move cannot make an interior incident panel newly violate
                # the angle band, or worsen any existing severe interior angle.
                bad=(~boundary)&(angle>np.maximum(errors,np.deg2rad(20.))+1e-8)
                allowed=~np.isin(selected,faces[bad])
                take=allowed & (change<best_change-1e-9)
                best_change[take],positions[take],where[take]=change[take],trial[selected[take]],ref[selected[take]]
            x[selected],references[selected]=positions,where
        _,err=details(x)
        count,worst=metrics(err,boundary)
        rank=count/max(1,boundary.sum())+worst+np.percentile(err[boundary],95)
        if count<=baseline[0] and worst<=baseline[1]+1e-8 and rank<best_rank and metrics(err,~boundary)[0]<=inside[0]:
            best,best_ref,best_rank=x.copy(),references.copy(),rank
    return best,best_ref,dict(moved=int(np.any(best!=initial,axis=1).sum()),before=angle_report(initial[faces[boundary]]),
                              after=angle_report(best[faces[boundary]]))
