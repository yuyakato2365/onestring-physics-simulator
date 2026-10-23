"""Bijective experimental UV continuation for a prescribed grid-cell-union boundary.

This is *not* an official BFF or proof of existence of a bijective map.
It never changes the selected M2D cells, the boundary destination, or
OptCuts' seam topology. Infeasible goals raise clear diagnostic errors.
"""
import numpy as np
import time


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
    last_heartbeat = [time.monotonic()]
    optimization_started = last_heartbeat[0]
    for strength in (100.,3000.,100000.):
        iterations_at_strength = [0]

        def heartbeat(_x):
            # L-BFGS-B can spend a long time inside a single continuation
            # step; this emits health signals without altering its objective.
            iterations_at_strength[0] += 1
            now = time.monotonic()
            if now - last_heartbeat[0] >= 10.:
                print(
                    "[GRID-UV-OPT] optimizer active "
                    f"barrier={strength:g} "
                    f"lbfgsb_iterations={iterations_at_strength[0]} "
                    f"elapsed={now-optimization_started:.1f}s",
                    flush=True,
                )
                last_heartbeat[0] = now
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
                     callback=heartbeat,
                     options={"maxiter":240,"gtol":1e-8,"ftol":1e-12})
        x=np.asarray(opt.x,float).reshape(-1,2)
        out[free]=x
        diag["iterations"]+=int(opt.nit)
        diag["min_area_ratio"]=float(np.min(oriented_area_ratio(base,out,faces)))
        if diag["min_area_ratio"]>=.005:break
    return out,diag


def fit_by_continuation(base,faces,loop,mapped,harmonic_dirichlet,assess,*,on_failure=None):
    """Refit at adaptive boundary positions; never suppress an invalid result."""
    progress=0.
    step=.25
    last=np.asarray(base,float).copy()
    history=[]
    # Capture the last rejected step for failure-only visual diagnosis.
    rejected_uv=None
    rejected_harmonic_uv=None
    rejected_fraction=None
    started = time.monotonic()
    last_progress_log = started
    print(
        f"[GRID-UV-CONT] starting; vertices={len(base)} "
        f"triangles={len(faces)}; full_boundary_fraction=1",
        flush=True,
    )
    for _ in range(70):
        if progress>=1.-1e-9:
            print(
                f"[GRID-UV-CONT] complete; "
                f"steps={len(history)} "
                f"elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
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
            row["flipped_triangles"]=int(metrics["flipped_triangles"])
            row["degenerate_triangles"]=int(metrics["degenerate_triangles"])
            row["uv_overlap_pairs"]=int(metrics["injectivity_overlap_pairs"])
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
            rejected_uv=np.asarray(candidate,float).copy()
            rejected_harmonic_uv=np.asarray(harmonic,float).copy()
            rejected_fraction=float(fraction)
            step*=.5
        now=time.monotonic()
        if (now-last_progress_log>=10. or progress>=1.-1e-9):
            print(
                "[GRID-UV-CONT] "
                f"step={len(history)}/70 accepted={valid} "
                f"target_fraction={fraction:.5f} "
                f"last_valid_fraction={progress:.5f} "
                f"min_signed_area_ratio={ratio:.5g} "
                f"next_step={step:.5g} "
                f"elapsed={now-started:.1f}s",
                flush=True,
            )
            last_progress_log=now
    # This is failure of the attempted numerical continuation, not a proof
    # that no topology-preserving map to the prescribed outline exists.
    rejected=next((item for item in reversed(history)
                   if not item["accepted"]),None)
    diagnostic=""
    if rejected is not None:
        diagnostic=(
            f"; last_rejected_fraction={rejected['fraction']:.5g}"
            f"; harmonic_min_area_ratio={rejected['harmonic_min_area_ratio']:.5g}"
            f"; optimized_min_area_ratio={rejected['min_area_ratio']:.5g}"
            f"; flipped={rejected.get('flipped_triangles', 'not_audited')}"
            f"; degenerate={rejected.get('degenerate_triangles', 'not_audited')}"
            f"; overlap={rejected.get('uv_overlap_pairs', 'not_audited')}"
        )
    print(
        "[GRID-UV-CONT] failed "
        f"last_valid_fraction={progress:.5f} "
        f"steps={len(history)} "
        f"elapsed={time.monotonic()-started:.1f}s",
        flush=True,
    )
    saved_diagnosis=""
    if on_failure is not None and rejected_uv is not None:
        try:
            path=on_failure(
                base=np.asarray(base,float),
                faces=np.asarray(faces,int),
                loop=np.asarray(loop,int),
                mapped=np.asarray(mapped,float),
                last=np.asarray(last,float),
                rejected=rejected_uv,
                harmonic=rejected_harmonic_uv,
                rejected_fraction=rejected_fraction,
                progress=float(progress),
                history=history,
            )
            if path:
                saved_diagnosis=f"; diagnostic_bundle={path}"
        except Exception as diag_exc:
            # Do not let diagnostics I/O hide the original solver error.
            print(
                "[GRID-DIAG] save_failed="
                f"{type(diag_exc).__name__}: {diag_exc}",
                flush=True,
            )
    raise ValueError(
        "GRID_BOUNDARY_BIJECTIVE_MAPPING_NOT_FOUND: "
        f"last_valid_fraction={progress:.5g}"
        +diagnostic
        +saved_diagnosis
        +"; current matching and numerical continuation did not yield "
         "a nonoverlapping UV map (not an infeasibility proof)"
    )
