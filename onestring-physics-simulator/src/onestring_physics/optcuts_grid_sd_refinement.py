"""Low-distortion refinement AFTER an exact, bijective Grid-boundary UV fit.

Unlike the existing harmonic + orientation-barrier solver, this module
minimizes the actual 3D-to-UV area-weighted symmetric Dirichlet energy
reported by optcuts_quality_audit. Its source is the paired 3D triangles,
not the previous UV chart.

Stages:
  1. Interior UV vertices free; all original Grid-boundary vertices fixed.
  2. If requested, interior free and non-corner boundary vertices can slide
     *only* on their ORIGINAL Grid outline edges, within disjoint ordered
     intervals. Every original Grid corner remains fixed.

The original source mesh, UV triangle topology, exact selected Grid mask and
outline, signed-area >=0.005 barrier and full overlap audit are immutable.
If optimization/audit fails or yields no improvement, keep the input UV.

This is a local numerical search, NOT a guarantee that a low-distortion map
exists with the selected seam and footprint.
"""
from __future__ import annotations

import time
import numpy as np


def _metric_3d(xyz, source_faces):
    tri=np.asarray(xyz,float)[np.asarray(source_faces,int)]
    a=tri[:,1]-tri[:,0]
    b=tri[:,2]-tri[:,0]
    aa=np.einsum("ij,ij->i",a,a)
    ab=np.einsum("ij,ij->i",a,b)
    bb=np.einsum("ij,ij->i",b,b)
    detg=aa*bb-ab*ab
    if not np.isfinite(detg).all() or np.any(detg<=1e-24):
        raise ValueError("GRID_SD_SOURCE_DEGENERATE_3D_TRIANGLE")
    det_source=np.sqrt(detg)
    area=det_source*.5
    total=area.sum()
    if not np.isfinite(total) or total<=0:
        raise ValueError("GRID_SD_ZERO_SOURCE_AREA")
    return bb/detg, -ab/detg, aa/detg, det_source, area/total


class SymmetricDirichletProblem:
    """Vectorized exact SD and analytic UV gradient on paired triangles.

    For each 3D triangle let G = [e1,e2]^T [e1,e2], and let U be the
    2x2 matrix of mapped UV edges. The squared Frobenius norm of the
    differential is F=tr(U G^-1 U^T). In 2D:
       SD = F * (1 + det(G)/det(U)^2).
    This is ||J||_F^2+||J^-1||_F^2, identical to the SVD-based audit.
    """

    def __init__(self, xyz, xyz_faces, uv_faces, original_uv,
                 *, min_area_ratio=.005):
        self.faces=np.asarray(uv_faces,int)
        if self.faces.shape!=np.asarray(xyz_faces,int).shape:
            raise ValueError("GRID_SD_PAIRED_FACE_COUNT_MISMATCH")
        self.g11,self.g12,self.g22,self.det_source,self.weight=_metric_3d(
            xyz,xyz_faces
        )
        self.base=np.asarray(original_uv,float)
        a=self.base[self.faces[:,1]]-self.base[self.faces[:,0]]
        b=self.base[self.faces[:,2]]-self.base[self.faces[:,0]]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if np.any(np.abs(det)<1e-20) or not np.isfinite(det).all():
            raise ValueError("GRID_SD_SOURCE_UV_DEGENERATE")
        self.original_det=np.abs(det)
        self.orientation=np.sign(det)
        self.min_area_ratio=float(min_area_ratio)

    def area_ratios(self, uv):
        p=np.asarray(uv,float)[self.faces]
        a=p[:,1]-p[:,0]
        b=p[:,2]-p[:,0]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        return self.orientation*det/self.original_det

    def per_triangle_sd(self, uv):
        """Exact 3D-to-UV symmetric-Dirichlet energy for each triangle."""
        pts=np.asarray(uv,float)[self.faces]
        a=pts[:,1]-pts[:,0]
        b=pts[:,2]-pts[:,0]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if np.any(np.abs(det)<1e-20):
            return np.full(len(self.faces),np.inf)
        f2=(self.g11*np.sum(a*a,axis=1)
            +2*self.g12*np.sum(a*b,axis=1)
            +self.g22*np.sum(b*b,axis=1))
        return f2*(1+(self.det_source/det)**2)

    def tail_threshold(self, uv, quantile=.95):
        sd=self.per_triangle_sd(uv)
        order=np.argsort(sd)
        total=np.cumsum(self.weight[order])
        return float(sd[order[min(len(order)-1,
            int(np.searchsorted(total, quantile)))]] )

    def tail_score(self, uv, threshold, strength=.50):
        sd=self.per_triangle_sd(uv)
        temperature=max(.25,.05*float(threshold))
        return float(np.dot(self.weight,sd)
            + strength*np.dot(self.weight,temperature*np.logaddexp(
                0.,(sd-threshold)/temperature)))

    def energy_gradient(self, uv, *, tail_threshold=None, tail_strength=.50):
        """Return area-weighted SD (+ optional soft tail) and UV gradient."""
        points=np.asarray(uv,float)
        p=points[self.faces]
        a=p[:,1]-p[:,0]
        b=p[:,2]-p[:,0]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        sign=self.orientation
        safe_signed=sign*det
        cutoff=self.original_det*self.min_area_ratio*.25
        # Clamp ONLY the energy denominator in infeasible trial evaluations.
        # Such points can never be accepted by the hard area audit.
        active=safe_signed>cutoff
        denom=sign*np.maximum(safe_signed,cutoff)
        f2=(self.g11*np.sum(a*a,axis=1)
            +2*self.g12*np.sum(a*b,axis=1)
            +self.g22*np.sum(b*b,axis=1))
        invdet2=(self.det_source/denom)**2
        factor=1+invdet2
        sd=f2*factor
        if not np.isfinite(sd).all():
            raise FloatingPointError("GRID_SD_NONFINITE_ENERGY")
        grad_a=2*factor[:,None]*(self.g11[:,None]*a+
                                 self.g12[:,None]*b)
        grad_b=2*factor[:,None]*(self.g12[:,None]*a+
                                 self.g22[:,None]*b)
        # Derivative of det(U) = cross(a,b); when clamped, the
        # denominator is constant but the area-barrier remains active.
        dD=(-2*f2*invdet2/denom)*active
        grad_a+=dD[:,None]*np.stack([b[:,1],-b[:,0]],axis=1)
        grad_b+=dD[:,None]*np.stack([-a[:,1],a[:,0]],axis=1)
        # Smooth positive-excess loss above a *fixed source-candidate*
        # 95th-percentile SD. Unlike mean SD it specifically targets the
        # high-distortion tail; fully differentiable with analytic gradient.
        additional_loss=0.
        scales=np.ones_like(sd)
        if tail_threshold is not None:
            temperature=max(.25,.05*float(tail_threshold))
            z=np.clip((sd-float(tail_threshold))/temperature,-60.,60.)
            additional_loss=float(tail_strength*np.dot(
                self.weight,temperature*np.logaddexp(0.,z)))
            scales += tail_strength/(1.+np.exp(-z))
        grad=np.zeros_like(points)
        wa=(self.weight*scales)[:,None]
        np.add.at(grad,self.faces[:,0],-wa*(grad_a+grad_b))
        np.add.at(grad,self.faces[:,1],wa*grad_a)
        np.add.at(grad,self.faces[:,2],wa*grad_b)
        energy=float(np.dot(sd,self.weight))+additional_loss
        # Strict global positivity is checked outside the minimizer.
        # This smooth continuation barrier helps L-BFGS-B return feasible
        # trials when a step crosses the >0.005 area threshold.
        ratio=safe_signed/self.original_det
        short=np.minimum(ratio-self.min_area_ratio*1.1,0)
        if np.any(short<0):
            strength=8.
            penalty=strength*np.mean((short/self.min_area_ratio)**2)
            dpen=(2*strength/len(short)/self.min_area_ratio**2
                  *short*sign/self.original_det)
            da=np.stack([b[:,1],-b[:,0]],axis=1)
            db=np.stack([-a[:,1],a[:,0]],axis=1)
            q=dpen[:,None]
            np.add.at(grad,self.faces[:,0],-q*(da+db))
            np.add.at(grad,self.faces[:,1],q*da)
            np.add.at(grad,self.faces[:,2],q*db)
            energy+=float(penalty)
        return energy,grad

    def exact_mean_sd(self, uv):
        """Same area-weighted SD as optcuts_quality_audit on feasible UV."""
        pts=np.asarray(uv,float)[self.faces]
        a=pts[:,1]-pts[:,0];b=pts[:,2]-pts[:,0]
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if np.any(np.abs(det)<1e-20):
            return float("inf")
        f2=(self.g11*np.sum(a*a,axis=1)
            +2*self.g12*np.sum(a*b,axis=1)
            +self.g22*np.sum(b*b,axis=1))
        return float(np.dot(
            f2*(1+(self.det_source/det)**2),self.weight
        ))


def _boundary_sliders(uv, loop, outline, *, tolerance=1e-7):
    """Independent disjoint sliding intervals on original staircase edges.

    Fixed corner vertices stay exact. Bounds are halfway to adjacent
    neighbors on the same edge and shrink slightly to forbid collapse.
    This is conservative; a future solver could use coupled monotone
    inequalities to allow larger shifts.
    """
    xyz=np.asarray(uv,float)
    boundary=np.asarray(loop,int)
    corners=np.asarray(outline,float)
    if len(corners)>1 and np.linalg.norm(corners[0]-corners[-1])<=tolerance:
        corners=corners[:-1]
    n=len(corners)
    if n<3:
        raise ValueError("GRID_SD_GRID_OUTLINE_INVALID")
    length=max(float(np.max(np.ptp(corners,axis=0))),1.)
    tol=tolerance*length
    starts=corners
    vectors=np.roll(corners,-1,axis=0)-corners
    lengths2=np.sum(vectors*vectors,axis=1)
    if np.any(lengths2<=tol*tol):
        raise ValueError("GRID_SD_GRID_OUTLINE_ZERO_EDGE")
    # Assumes a simple, valid, orthogonal Grid polyomino contour.
    # For every non-corner source boundary vertex, find its ORIGINAL
    # geometric Grid edge; never let it switch to a different edge.
    assignments={}
    for vertex in boundary:
        point=xyz[vertex]
        if np.any(np.linalg.norm(corners-point,axis=1)<=tol):
            continue
        diff=point[None,:]-starts
        t=np.einsum("ij,ij->i",diff,vectors)/lengths2
        closest=starts+t[:,None]*vectors
        distance=np.linalg.norm(closest-point[None,:],axis=1)
        valid=np.flatnonzero(
            (distance<=tol)&(t>1e-9)&(t<1-1e-9)
        )
        if len(valid)!=1:
            raise ValueError(
                "GRID_SD_BOUNDARY_VERTEX_NOT_ON_UNIQUE_GRID_EDGE: "
                f"vertex_id={int(vertex)}; count={len(valid)}"
            )
        j=int(valid[0])
        assignments[int(vertex)]=(j,float(t[j]))
    var_ids=[]
    directions=[]
    positions=[]
    lows=[]
    highs=[]
    for segment in range(n):
        group=sorted(
            [(index,t) for index,(seg,t) in assignments.items()
             if seg==segment],key=lambda pair:pair[1]
        )
        all_t=[0.]+[t for _,t in group]+[1.]
        for k,(idx,t) in enumerate(group,1):
            before=all_t[k-1]
            after=all_t[k+1]
            if not (before<t<after):
                raise ValueError("GRID_SD_BOUNDARY_ORDER_INVALID")
            eps=min(1e-7,(t-before)*1e-3,(after-t)*1e-3)
            lo=(before+t)*.5+eps
            hi=(after+t)*.5-eps
            if lo>=t or hi<=t:
                continue
            var_ids.append(idx)
            directions.append(vectors[segment])
            positions.append(t)
            lows.append(lo)
            highs.append(hi)
    return (
        np.asarray(var_ids,int),
        np.asarray(directions,float).reshape(-1,2),
        np.asarray(positions,float),
        np.asarray(lows,float),
        np.asarray(highs,float),
    )


def refine_exact_grid_uv(
    source_xyz, source_faces, uv_faces, original_uv,
    full_grid_uv, boundary_loop, grid_outline, *,
    assess, mean_sd_limit, min_area_ratio=.005,
    interior_maxiter=85, sliding_maxiter=100, progress=True,
    tail_aware=False,
):
    """Find a lower-area-mean-SD candidate without changing the Grid mask.

    The reference (original_uv) is the unmodified paired OptCuts chart
    *before* boundary fitting; full_grid_uv is the successful bijective
    UV AFTER continuation. Only commit a candidate with an independently
    audited 0.005 area ratio, no inversions, no overlap, exact corners,
    and strictly lower mean SD than the starting mapping.

    Returns (accepted_or_input_uv, diagnostics). On numerical failure,
    return the unchanged successful full-grid UV.
    """
    from scipy.optimize import minimize

    base=np.asarray(original_uv,float)
    start=np.asarray(full_grid_uv,float)
    faces=np.asarray(uv_faces,int)
    loop=np.asarray(boundary_loop,int)
    corners=np.asarray(grid_outline,float)
    if len(corners)>1 and np.allclose(corners[0],corners[-1]):
        corners=corners[:-1]
    if base.shape!=start.shape or base.shape[1]!=2:
        raise ValueError("GRID_SD_UV_PAIR_SHAPE_MISMATCH")
    if not np.isfinite(start).all():
        raise ValueError("GRID_SD_INITIAL_UV_NONFINITE")
    # Source 3D geometry and face pairing remain unmodified.
    problem=SymmetricDirichletProblem(
        source_xyz,source_faces,faces,base,
        min_area_ratio=min_area_ratio,
    )
    fixed_boundary=np.zeros(len(start),bool)
    fixed_boundary[loop]=True
    interior=np.flatnonzero(~fixed_boundary)
    if np.min(problem.area_ratios(start))<min_area_ratio:
        raise ValueError("GRID_SD_START_INVALID_MIN_AREA_RATIO")
    source_quality=assess(start)
    sd_before=float(source_quality["distortion_area_mean"])
    if any(int(source_quality[k]) for k in
           ("flipped_triangles","degenerate_triangles",
            "injectivity_overlap_pairs")):
        raise ValueError("GRID_SD_START_INVALID_GLOBAL_INJECTIVITY")
    exact=problem.exact_mean_sd(start)
    # Prevent any accidental mismatch between the gradient's energy and
    # the quality metric that determines acceptance.
    if not np.isfinite(exact) or not np.isclose(
        exact,sd_before,rtol=1e-6,atol=1e-7
    ):
        raise ValueError(
            f"GRID_SD_ENERGY_AUDIT_MISMATCH: analytic={exact:.9g} "
            f"quality={sd_before:.9g}"
        )
    tail_threshold=(problem.tail_threshold(start) if tail_aware else None)
    best=start.copy()
    best_quality=source_quality
    best_sd=sd_before
    best_score=(problem.tail_score(best,tail_threshold)
                if tail_aware else sd_before)
    tail_p95_before=float(problem.tail_threshold(start))
    phases=[]
    started=time.monotonic()
    cell_size=max(float(np.max(np.linalg.norm(
        np.roll(corners,-1,axis=0)-corners,axis=1))),1e-12)
    tol=cell_size*1e-7+1e-10

    def audit(candidate, label):
        nonlocal best,best_sd,best_quality,best_score
        if not np.isfinite(candidate).all():
            return "nonfinite"
        min_ratio=float(np.min(problem.area_ratios(candidate)))
        if min_ratio<min_area_ratio:
            return f"min_area_ratio={min_ratio:.6g}"
        if not np.array_equal(candidate.shape,start.shape):
            return "shape"
        if np.min(np.linalg.norm(
            candidate[loop,None,:]-corners[None,:,:],axis=2
        ),axis=0).max()>tol:
            return "grid_corner_missing"
        result=assess(candidate)
        if any(int(result[k]) for k in
               ("flipped_triangles","degenerate_triangles",
                "injectivity_overlap_pairs")):
            return "global_injectivity_or_overlap"
        sd=float(result["distortion_area_mean"])
        if not np.isfinite(sd):
            return "nonfinite_sd"
        score=(problem.tail_score(candidate,tail_threshold)
               if tail_aware else sd)
        # Tail experiments may trade up to 2% mean SD for substantial tail
        # improvement, but cannot exceed the original strict mean SD gate.
        admissible=(sd<=mean_sd_limit and sd<=sd_before*1.02) if tail_aware else True
        if admissible and score < best_score-1e-7:
            best_score=score
            best=candidate.copy()
            best_sd=sd
            best_quality=result
            if progress:
                print(
                    "[GRID-SD] improved "
                    f"phase={label} mean_sd={sd:.6g} "
                    f"limit={mean_sd_limit:.6g} "
                    f"min_area_ratio={min_ratio:.6g}",
                    flush=True,
                )
            return "improved"
        return "no_improvement"

    if sd_before<=mean_sd_limit and not tail_aware:
        return best,{
            "status":"already_within_limit",
            "mean_sd_before":sd_before,
            "mean_sd_after":best_sd,
            "improvement":0.,
            "phases":phases,
        }

    def solve(label, use_sliders, maxiter):
        nonlocal best
        initial=best.copy()
        slider_ids=np.empty(0,int)
        slider_dirs=np.empty((0,2))
        slider_t=np.empty(0)
        lo=np.empty(0)
        hi=np.empty(0)
        if use_sliders:
            slider_ids,slider_dirs,slider_t,lo,hi=_boundary_sliders(
                initial,loop,corners
            )
        n=len(interior)*2
        x0=np.r_[initial[interior].ravel(),slider_t]
        bounds=[(None,None)]*n + list(zip(lo,hi))
        best_feasible=[initial.copy()]
        best_feasible_energy=[problem.exact_mean_sd(initial)]
        last_log=[time.monotonic()]
        iterations=[0]
        def unpack(x):
            uv=initial.copy()
            uv[interior]=x[:n].reshape(-1,2)
            if len(slider_ids):
                delta=x[n:]-slider_t
                uv[slider_ids]=initial[slider_ids]+delta[:,None]*slider_dirs
            return uv
        def objective(x):
            uv=unpack(x)
            loss,gradient=problem.energy_gradient(
                uv,tail_threshold=tail_threshold)
            # Retain only orientation-safe, locally feasible iterates.
            # Full global overlap audit remains mandatory at completion.
            if np.min(problem.area_ratios(uv))>=min_area_ratio:
                value=problem.exact_mean_sd(uv)
                if value<best_feasible_energy[0]:
                    best_feasible_energy[0]=value
                    best_feasible[0]=uv.copy()
            g=np.r_[gradient[interior].ravel(),
                    np.einsum("ij,ij->i",
                              gradient[slider_ids],slider_dirs)]
            return loss,g
        def heartbeat(_x):
            iterations[0]+=1
            now=time.monotonic()
            if progress and now-last_log[0]>=10.:
                print(
                    f"[GRID-SD] active phase={label} "
                    f"iterations={iterations[0]}/{maxiter} "
                    f"elapsed={now-started:.1f}s",
                    flush=True,
                )
                last_log[0]=now
        try:
            optimization=minimize(
                objective,x0,method="L-BFGS-B",jac=True,
                bounds=bounds,callback=heartbeat,
                options={"maxiter":int(maxiter),"ftol":1e-11,
                         "gtol":1e-7,"maxls":35},
            )
            # Audit both the returned UV and a feasible objective
            # evaluation; never accept the penalized, inverted result.
            tried=[unpack(optimization.x),best_feasible[0]]
            results=[audit(p,label) for p in tried]
            status=",".join(results)
            phases.append({
                "phase":label,
                "slider_vertices":int(len(slider_ids)),
                "iterations":int(optimization.nit),
                "solver_success":bool(optimization.success),
                "status":status,
                "best_sd":float(best_sd),
            })
        except (FloatingPointError,ValueError,OverflowError) as exc:
            phases.append({
                "phase":label,
                "slider_vertices":int(len(slider_ids)),
                "status":f"solver_error:{type(exc).__name__}:{exc}",
                "best_sd":float(best_sd),
            })
        if progress:
            print(
                f"[GRID-SD] phase_done={label} "
                f"sd={best_sd:.6g} limit={mean_sd_limit:.6g} "
                f"status={phases[-1]['status']}",
                flush=True,
            )

    if len(interior):
        solve("interior",False,interior_maxiter)
    if best_sd>mean_sd_limit or tail_aware:
        try:
            solve("grid_edge_slide",True,sliding_maxiter)
        except ValueError as exc:
            phases.append({
                "phase":"grid_edge_slide",
                "status":f"slider_setup_unavailable:{exc}",
                "best_sd":float(best_sd),
            })
            if progress:
                print("[GRID-SD] slider_setup_unavailable="+str(exc),
                      flush=True)
    return best,{
        "status":"within_limit" if best_sd<=mean_sd_limit
                 else "improved_but_over_limit"
                 if best_sd<sd_before-1e-7 else "no_improvement",
        "mean_sd_before":sd_before,
        "mean_sd_after":best_sd,
        "improvement":sd_before-best_sd,
        "phases":phases,
        "min_area_ratio":float(np.min(problem.area_ratios(best))),
        "tail_aware":bool(tail_aware),
        "tail_p95_before":tail_p95_before,
        "tail_p95_after":float(problem.tail_threshold(best)),
        "tail_score_before":float(problem.tail_score(start,tail_threshold)
            if tail_aware else sd_before),
        "tail_score_after":float(problem.tail_score(best,tail_threshold)
            if tail_aware else best_sd),
        "elapsed_seconds":time.monotonic()-started,
    }
