"""Preview the shared-vertex shell when strict face thickness is infeasible."""
import numpy as np
from scipy.optimize import minimize


class ThicknessConflict(ValueError):
    pass


def save_preview(out, X, F, normals, depth, tolerance, threshold, mm):
    owners=[[] for _ in X]
    for i,face in enumerate(F):
        for v in face: owners[v].append(i)
    delta=np.zeros_like(X)
    for v,ids in enumerate(owners):
        if not ids: continue
        n=normals[ids]
        # Prefer normal depths close to the requested value. Regularisation
        # selects a finite, local displacement even for nearly opposing planes.
        target=depth[ids]
        H=n.T@n+.02*np.eye(3); rhs=n.T@target
        d=np.linalg.solve(H,rhs)
        if (n@d<0).any():
            res=minimize(lambda q:(.5*q@H@q-rhs@q,H@q-rhs),np.zeros(3),jac=True,
                         constraints=[dict(type='ineq',fun=lambda q:n@q,jac=lambda q:n)],method='SLSQP')
            d=res.x if res.success else np.zeros(3)
        delta[v]=d
    shift=delta[F];height=np.einsum('nki,ni->nk',shift,normals)
    lateral=np.linalg.norm(shift-height[:,:,None]*normals[:,None],axis=2).max(1)
    np.savez_compressed(out/'drift_preview.npz',solid=np.concatenate([X[F],(X+delta)[F]],1)/mm,
                        lateral=lateral, depth_min=height.min(1),depth_max=height.max(1),
                        off_depth=((height<depth[:,None]-tolerance-1e-5)|(height>depth[:,None]+1e-5)).any(1),
                        threshold=float(threshold))
