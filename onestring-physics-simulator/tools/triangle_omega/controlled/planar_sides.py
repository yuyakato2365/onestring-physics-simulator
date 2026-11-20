"""Align the two extrusion directions in each side wall without disconnecting panels."""
import numpy as np
from scipy.optimize import minimize


def fit_sides(vertices, faces, offset, normal, depth, invalid_vertices):
    edge = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    i, j = edge.T
    axis = vertices[j] - vertices[i]
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    scale = np.linalg.norm(offset, axis=1)
    initial = offset / scale[:, None]
    height = -np.einsum('fvc,fc->fv', offset[faces], normal)
    minimum = np.maximum(.9*height, np.minimum(height, .8*depth[:, None]))
    maximum = np.maximum(1.2*depth[:, None], height)
    def project(v):
        return v - (v*axis).sum(1)[:, None]*axis
    def residual(x):
        a, b = project(x[i]), project(x[j])
        la = np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-10)
        lb = np.maximum(np.linalg.norm(b, axis=1)[:, None], 1e-10)
        return a/la, b/lb, la, lb
    def metrics(shift):
        # shell_topology uses the diagonal from outer i to inner j (i < j).
        v0, v1 = vertices[i], vertices[j]
        u0, u1 = v0+shift[i], v1+shift[j]
        n0 = np.cross(v1-v0,u1-v0)
        n1 = np.cross(u1-v0,u0-v0)
        n0 /= np.maximum(np.linalg.norm(n0,axis=1)[:,None],1e-20)
        n1 /= np.maximum(np.linalg.norm(n1,axis=1)[:,None],1e-20)
        angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(n0,n1),axis=1),(n0*n1).sum(1)))
        warp=np.abs(np.sum((u0-v0)*n0,axis=1))
        return dict(median_deg=float(np.median(angle)),p95_deg=float(np.percentile(angle,95)),
                    max_deg=float(angle.max()),p95_plane_distance=float(np.percentile(warp,95)))
    def energy(flat):
        x=flat.reshape(-1,3)
        a,b,la,lb=residual(x)
        diff=a-b
        sq=(diff*diff).sum(1)
        g=2*(1+2*sq[:,None])*diff
        anchor=x-initial
        grad=.01*anchor
        np.add.at(grad,i,project(g-(g*a).sum(1)[:,None]*a)/la)
        np.add.at(grad,j,-project(g-(g*b).sum(1)[:,None]*b)/lb)
        actual=-np.einsum('fvc,fc->fv',x[faces]*scale[faces,None],normal)
        h=(actual-height)/depth[:,None]
        error=(np.minimum(actual-minimum,0)+np.maximum(actual-maximum,0))/depth[:,None]
        np.add.at(grad,faces,-(.12*h+800*error)[:,:,None]*(scale[faces]/depth[:,None])[:,:,None]*normal[:,None,:])
        return float((sq+sq**2).sum()+.005*(anchor**2).sum()+.06*(h**2).sum()+400*(error**2).sum()),grad.ravel()
    solved=minimize(energy,initial.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=300,ftol=1e-11))
    move=solved.x.reshape(-1,3)*scale[:,None]-offset
    # Project each proposed move onto its incident face-depth halfspaces.
    # A radial line search alone would freeze a corner already on one bound,
    # even when it can slide along that bound and make its side walls planar.
    point,corner=np.where(np.ones_like(height,dtype=bool))
    incidence=faces[point,corner]
    for vertex in range(len(vertices)):
        ids=incidence==vertex
        matrix=-normal[point[ids]]
        lower=(minimum-height)[point[ids],corner[ids]]/scale[vertex]
        upper=(maximum-height)[point[ids],corner[ids]]/scale[vertex]
        goal=move[vertex]/scale[vertex]
        projected=matrix@goal
        if np.all(projected>=lower) and np.all(projected<=upper):continue
        constraints=np.vstack((matrix,-matrix))
        limit=np.r_[upper,-lower]
        projection=minimize(lambda t: (.5*np.sum((t-goal)**2),t-goal),np.zeros(3),jac=True,
            constraints=[dict(type='ineq',fun=lambda t:limit-constraints@t,jac=lambda t:-constraints)],
            method='SLSQP',options=dict(ftol=1e-12,maxiter=50))
        if projection.success and np.max(constraints@projection.x-limit)<1e-9:
            move[vertex]=projection.x*scale[vertex]
        else:
            move[vertex]=0
    # Retain the extrusion routine's two-depth miter limit, including any
    # longer shank already accepted by the preceding hinge repair.
    radius=np.full(len(vertices),np.inf)
    np.minimum.at(radius,faces,np.broadcast_to(2*depth[:,None],faces.shape))
    radius=np.maximum(radius,scale)
    over=np.linalg.norm(offset+move,axis=1)>radius
    length2=np.sum(move[over]**2,axis=1)
    along=np.sum(offset[over]*move[over],axis=1)
    root=np.sqrt(np.maximum(0,along**2+length2*(radius[over]**2-scale[over]**2)))
    move[over]*=np.clip((root-along)/np.maximum(length2,1e-30),0,1)[:,None]
    before=metrics(offset)
    best=before; selected=offset.copy()
    for step in (1.,.5,.25,.125):
        change=step*move.copy()
        while True:
            bad=invalid_vertices(offset+change)
            if not len(bad):break
            if not np.any(change[bad]):break
            change[bad]=0
        if len(bad):continue
        after=metrics(offset+change)
        if after['p95_deg']<best['p95_deg'] and after['max_deg']<=before['max_deg']+1e-7:
            best,selected=after,offset+change
    return selected,dict(side_count=len(edge),before=before,after=best)
