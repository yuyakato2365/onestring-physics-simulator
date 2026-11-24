"""Redistribute a locally collapsed offset surface over its two-ring neighbourhood."""
import numpy as np
from scipy.optimize import minimize


def repair_offset(faces, base, initial, symmetric=False):
    # Coordinates in mm, indexed by the original shared mesh vertices.
    F=faces
    ref=base[F]
    cross=np.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0])
    area=np.linalg.norm(cross,axis=1)
    if (area<=1e-12).any(): raise ValueError("押し出しの基準面が退化しています。")
    normal=cross/area[:,None]
    edges=np.unique(np.sort(F[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    u,v=edges.T
    def ratio_grad(x,sign):
        p=(base+sign*(x-base))[F];a=p[:,1]-p[:,0];b=p[:,2]-p[:,0]
        signed=np.einsum('ij,ij->i',np.cross(a,b),normal)/area
        ga=np.cross(b,normal);gb=np.cross(normal,a)
        return signed,sign*np.stack([-ga-gb,ga,gb],1)/area[:,None,None]
    signs=(1,-1) if symmetric else (1,)
    worst=np.min([ratio_grad(initial,s)[0] for s in signs],axis=0)
    active=worst<.65
    for _ in range(2):
        active=np.isin(F,F[active]).any(1)
    ids=np.unique(F[active])
    if not len(ids):return initial.copy()
    length=np.linalg.norm(base[u]-base[v],axis=1)
    local=np.full(len(base),np.inf)
    np.minimum.at(local,u,length);np.minimum.at(local,v,length)
    bound=.16*local[ids]/np.sqrt(3)
    def energy(values):
        move=np.zeros_like(base);move[ids]=values.reshape(-1,3)
        diff=move[u]-move[v]
        loss=(move**2).sum()+.6*(diff**2).sum();grad=2*move
        np.add.at(grad,u,1.2*diff);np.add.at(grad,v,-1.2*diff)
        for sign in signs:
            ratio,g=ratio_grad(initial+move,sign)
            deficit=np.minimum(ratio-.8,0);weight=25*area
            loss+=(weight*deficit**2).sum()
            np.add.at(grad,F,(2*weight*deficit)[:,None,None]*g)
        return loss,grad[ids].ravel()
    limits=np.repeat(bound,3)
    res=minimize(energy,np.zeros_like(limits),jac=True,method='L-BFGS-B',
                 bounds=list(zip(-limits,limits)),options={'maxiter':180,'ftol':1e-11})
    result=initial.copy();result[ids]+=res.x.reshape(-1,3)
    # Retain the existing corner thickness at thin spots and stop thicker
    # corners at 2 mm. A shared vertex must satisfy every incident panel.
    initial_height=np.einsum('nki,ni->nk',(initial-base)[F],normal)
    change=np.einsum('nki,ni->nk',(result-initial)[F],normal)
    minimum=np.minimum(initial_height,1. if symmetric else 2.)
    fraction=np.ones_like(change)
    thinning=change < -1e-12
    fraction[thinning]=np.minimum(1., (initial_height-minimum)[thinning]/-change[thinning])
    step=np.ones(len(base))
    np.minimum.at(step,F,fraction)
    return initial+(result-initial)*step[:,None]
