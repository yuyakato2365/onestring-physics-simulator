"""Bijective experimental UV continuation for a prescribed grid-cell-union boundary.

This is *not* an official BFF or proof of existence of a bijective map.
It never changes the selected M2D cells, the boundary destination, or
OptCuts' seam topology. Infeasible goals raise clear diagnostic errors.
"""
import numpy as np


def oriented_area_ratio(base,vertices,faces):
    a=np.asarray(base,float)[faces]
    b=np.asarray(vertices,float)[faces]
    def cross(t):
        u=t[:,1]-t[:,0];v=t[:,2]-t[:,0]
        return u[:,0]*v[:,1]-u[:,1]*v[:,0]
    ref=cross(a)
    if np.any(np.abs(ref)<1e-20):
        raise ValueError("SOURCE_DEGENERATE_TRIANGLES")
    return np.sign(ref)*cross(b)/np.abs(ref)


def untangle_fixed_boundary(base,faces,loop,target,harmonic,previous=None):
    """Minimize a harmonic-position term + signed-area inversion barrier.

    Gradients of signed double areas are exact. Hard boundary coordinates
    are excluded from optimization. A final independent overlap audit
    is mandatory in the caller: positive local areas alone aren't enough.
    """
    from scipy.optimize import minimize
    base=np.asarray(base,float);faces=np.asarray(faces,int)
    bound=np.asarray(loop,int)
    boundary=np.zeros(len(base),bool);boundary[bound]=True
    free=np.flatnonzero(~boundary)
    out=base.copy();out[bound]=target
    if not len(free):return out,{"solver":"no_free_interior","iterations":0}
    r=base[faces]
    a=r[:,1]-r[:,0];b=r[:,2]-r[:,0]
    d=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if np.any(np.abs(d)<1e-20):
        raise ValueError("SOURCE_DEGENERATE_TRIANGLES")
    sign=np.sign(d);area=np.abs(d)
    scale=max(float(np.ptp(base,axis=0).max()),1e-8)
    n=len(free);m=len(faces)
    harmonic_i=harmonic[free];original_i=base[free]
    x=(.5*previous[free]+.5*harmonic_i
       if previous is not None else harmonic_i.copy())
    diag={"solver":"orientation_barrier_lbfgsb","iterations":0}
    for strength in (100.,3000.,100000.):
        def objective(v):
            uv=out.copy();uv[free]=v.reshape(-1,2)
            tri=uv[faces];p,q,r=tri[:,0],tri[:,1],tri[:,2]
            dx=q-p;dy=r-p
            det=dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
            deficit=np.minimum(sign*det/area-.015,0.)
            e=(uv[free]-harmonic_i)/scale
            anchor=(uv[free]-original_i)/scale
            score=(np.sum(e**2)+.02*np.sum(anchor**2))/n
            score+=strength*np.mean(deficit**2)
            grad=np.zeros_like(uv)
            grad[free]+=(2*e+.04*anchor)/(n*scale)
            coeff=2*strength/m*deficit*sign/area
            derivatives=(
                np.column_stack([q[:,1]-r[:,1],r[:,0]-q[:,0]]),
                np.column_stack([r[:,1]-p[:,1],p[:,0]-r[:,0]]),
                np.column_stack([p[:,1]-q[:,1],q[:,0]-p[:,0]]),
            )
            for i,derivative in enumerate(derivatives):
                np.add.at(grad,faces[:,i],coeff[:,None]*derivative)
            return float(score),grad[free].ravel()
        opt=minimize(objective,x.ravel(),method="L-BFGS-B",jac=True,
                     options={"maxiter":240,"gtol":1e-8,"ftol":1e-12})
        x=np.asarray(opt.x,float).reshape(-1,2)
        out[free]=x
        diag["iterations"]+=int(opt.nit)
        diag["min_area_ratio"]=float(np.min(oriented_area_ratio(base,out,faces)))
        if diag["min_area_ratio"]>=.005:break
    return out,diag


def fit_by_continuation(base,faces,loop,mapped,harmonic_dirichlet,assess):
    """Refit at adaptive boundary positions; never suppress an invalid result."""
    progress=0.
    step=.25
    last=np.asarray(base,float).copy()
    history=[]
    for _ in range(70):
        if progress>=1.-1e-9:
            return last,history
        if step<1./2048.:break
        fraction=min(1.,progress+step)
        target=base[loop]+fraction*(mapped-base[loop])
        harmonic=harmonic_dirichlet(base,faces,loop,target)
        ratio=float(np.min(oriented_area_ratio(base,harmonic,faces)))
        row={"fraction":float(fraction),"harmonic_min_area_ratio":ratio}
        if ratio<.005:
            candidate,details=untangle_fixed_boundary(
                base,faces,loop,target,harmonic,last)
            row.update(details)
        else:
            candidate=harmonic
            row["solver"]="harmonic"
        ratio=float(np.min(oriented_area_ratio(base,candidate,faces)))
        row["min_area_ratio"]=ratio
        valid=np.isfinite(candidate).all() and ratio>=.005
        if valid:
            metrics=assess(candidate)
            valid=not (metrics["flipped_triangles"]
                       or metrics["degenerate_triangles"]
                       or metrics["injectivity_overlap_pairs"])
        row["accepted"]=bool(valid)
        history.append(row)
        if valid:
            last=candidate
            progress=fraction
            step=min(.3,step*1.4)
        else:
            step*=.5
    raise ValueError(
        "GRID_BOUNDARY_BIJECTIVE_MAPPING_NOT_FOUND: "
        f"last_valid_fraction={progress:.5g}; "
        "no safe inverse mapping for this exact grid-cell outline"
    )
