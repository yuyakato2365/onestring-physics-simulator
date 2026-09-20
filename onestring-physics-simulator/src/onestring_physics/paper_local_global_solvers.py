"""Paper-aligned local/global solvers for OneString Sec. 4.2 and 4.3.

This module intentionally keeps the OptCuts/M2D front-end unchanged and replaces
only M3D->K3D and M2D->K2D.

Important fidelity note:
- K3D uses explicit local projections for planarity, square/edge length, and
  closest-point-to-target-surface, followed by a sparse global least-squares step.
- K2D uses explicit edge/fabrication/non-penetration projections followed by a
  sparse global least-squares step.
- The paper cites Konakovic et al. for non-penetration. Their implementation
  details are not fully specified in OneString; here the closest separating
  translation of convex quads is computed with SAT and used AS A PROJECTION.
  This is materially different from the older SAT-depth penalty: collision is
  not differentiated as an energy; it produces projected collision-free targets.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize, least_squares

from .paper_eq5_k2d_solver import build_linkage_topology, project_angle_vectors


def _env_float(name, default):
    try:
        x=float(os.environ.get(name, default))
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _env_int(name, default):
    try: return max(1, int(float(os.environ.get(name, default))))
    except Exception: return int(default)


def _edges(faces):
    f=np.asarray(faces,int)
    e=np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,3]],f[:,[3,0]]))
    return np.unique(np.sort(e,axis=1),axis=0)


def _best_fit_plane_projection(q):
    c=q.mean(axis=0)
    _,_,vt=np.linalg.svd(q-c,full_matrices=False)
    n=vt[-1]
    return q-np.outer((q-c)@n,n)


def _closest_square_projection(q):
    """Similarity-fit a square in the quad best-fit plane, preserving winding."""
    planar=_best_fit_plane_projection(q)
    c=planar.mean(axis=0)
    _,_,vt=np.linalg.svd(planar-c,full_matrices=False)
    basis=vt[:2].T
    p=(planar-c)@basis
    template=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    # Orthogonal Procrustes + uniform scale.
    h=template.T@p
    u,_,v=np.linalg.svd(h)
    r=u@v
    if np.linalg.det(r)<0:
        u[:,-1]*=-1; r=u@v
    tr=template@r
    scale=float(np.sum(tr*p)/max(np.sum(tr*tr),1e-30))
    return c+(scale*tr)@basis.T


def _closest_point_triangle(p,a,b,c):
    # Ericson region tests.
    ab=b-a; ac=c-a; ap=p-a
    d1=np.dot(ab,ap); d2=np.dot(ac,ap)
    if d1<=0 and d2<=0: return a
    bp=p-b; d3=np.dot(ab,bp); d4=np.dot(ac,bp)
    if d3>=0 and d4<=d3: return b
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:
        v=d1/(d1-d3); return a+v*ab
    cp=p-c; d5=np.dot(ab,cp); d6=np.dot(ac,cp)
    if d6>=0 and d5<=d6: return c
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:
        w=d2/(d2-d6); return a+w*ac
    va=d3*d6-d5*d4
    if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:
        w=(d4-d3)/((d4-d3)+(d5-d6)); return b+w*(c-b)
    denom=1.0/max(va+vb+vc,1e-30); v=vb*denom; w=vc*denom
    return a+ab*v+ac*w


def _surface_project(points,target,parameterization=None):
    """Closest-point projection onto the actual input surface mesh.

    The pipeline's target is normally a HeightField (it has no vertices/faces).
    SurfaceParameterization, however, carries the exact source surface mesh used
    by the pipeline. Prefer it, and only accept target.vertices/faces for callers
    that explicitly provide a mesh target.
    """
    if parameterization is not None:
        tv=np.asarray(parameterization.surface_vertices_3d,float)
        tf=np.asarray(parameterization.surface_faces,int)
    elif hasattr(target,"vertices") and hasattr(target,"faces"):
        tv=np.asarray(target.vertices,float)
        tf=np.asarray(target.faces,int)
    else:
        raise TypeError("Surface projection requires SurfaceParameterization.surface_vertices_3d/surface_faces")
    if tf.ndim != 2 or tf.shape[1] < 3 or len(tf)==0:
        raise ValueError("Target surface mesh must contain triangles")
    tris=tv[tf[:,:3]]
    centers=tris.mean(axis=1)
    try:
        from scipy.spatial import cKDTree
        tree=cKDTree(centers)
        k=min(32,len(tris))
        _,idx=tree.query(points,k=k)
        idx=np.asarray(idx,int)
        if idx.ndim==1: idx=idx[:,None]
    except Exception:
        idx=np.tile(np.arange(len(tris)),(len(points),1))
    out=np.empty_like(points)
    for i,p in enumerate(points):
        best=None; bd=np.inf
        for tid in np.atleast_1d(idx[i]):
            a,b,c=tris[int(tid)]
            q=_closest_point_triangle(p,a,b,c)
            d=float(np.dot(q-p,q-p))
            if d<bd: bd=d; best=q
        out[i]=best
    return out


def _solve_constraints(n,dim,constraints,anchor=None,anchor_weight=1e-8):
    """Solve sum w ||A x-b||^2. Constraint=(ids, coeffs, target, weight)."""
    rows=[]; cols=[]; data=[]; rhs=[]; row=0
    for ids,coeff,target,weight in constraints:
        sw=math.sqrt(max(float(weight),0.0))
        if sw==0: continue
        target=np.asarray(target,float).reshape(dim)
        for d in range(dim):
            for vid,coef in zip(ids,coeff):
                rows.append(row); cols.append(int(vid)*dim+d); data.append(sw*float(coef))
            rhs.append(sw*target[d]); row+=1
    if anchor is not None and anchor_weight>0:
        sw=math.sqrt(anchor_weight)
        for i,p in enumerate(np.asarray(anchor,float)):
            for d in range(dim):
                rows.append(row);cols.append(i*dim+d);data.append(sw);rhs.append(sw*p[d]);row+=1
    A=sparse.coo_matrix((data,(rows,cols)),shape=(row,n*dim)).tocsr()
    x=lsqr(A,np.asarray(rhs),atol=1e-10,btol=1e-10,iter_lim=max(500,4*n))[0]
    return x.reshape(n,dim)



def minimum_displacement_planarity_polish(vertices,faces,*,tolerance=1e-8,max_iterations=300):
    """Sparse minimum-displacement quad planarization.

    Uses an augmented-residual continuation solve instead of dense SLSQP.
    Each quad coplanarity residual depends on only four vertices (12 scalar
    variables), so the finite-difference sparsity pattern keeps the trust-region
    subproblems sparse even for thousands of K3D vertices.
    """
    x0=np.asarray(vertices,float).copy()
    faces=np.asarray(faces,int)
    if len(faces)==0:
        return x0,{"success":True,"iterations":0,"planarity_max":0.0,
                   "planarity_rms":0.0,"displacement_rms":0.0,
                   "displacement_max":0.0,"message":"empty mesh","tolerance":float(tolerance)}

    n=len(x0)
    center=x0.mean(axis=0)
    scale=max(float(np.linalg.norm(np.ptp(x0,axis=0))),1e-12)
    y0=(x0-center)/scale
    flat0=y0.ravel()

    # Normalize the scalar triple product by a characteristic local edge scale^3
    # so rho has a stable meaning across mesh resolutions.
    q0=y0[faces]
    edge0=np.linalg.norm(np.roll(q0,-1,axis=1)-q0,axis=2)
    local_scale=np.maximum(np.mean(edge0,axis=1),1e-8)
    volume_scale=local_scale**3

    # Residual = [minimum displacement, progressively strengthened coplanarity].
    # Deliberately planarity-only: square/surface terms belong to the preceding K3D solve.
    w_ref=_env_float("ONESTRING_K3D_POLISH_REFERENCE_WEIGHT",1.0)

    def coplanarity(flat):
        y=flat.reshape((-1,3))
        q=y[faces]
        a=q[:,1]-q[:,0]; b=q[:,2]-q[:,0]; c=q[:,3]-q[:,0]
        return np.einsum("ij,ij->i",a,np.cross(b,c))/volume_scale

    # Sparse dependency pattern: reference rows are diagonal; each
    # coplanarity row touches only the four vertices of its quad.
    rows=list(range(3*n)); cols=list(range(3*n))
    for fi,f in enumerate(faces):
        row=3*n+fi
        for vid in f:
            for d in range(3):
                rows.append(row); cols.append(3*int(vid)+d)
    jac_pattern=sparse.coo_matrix(
        (np.ones(len(rows),dtype=float),(rows,cols)),
        shape=(3*n+len(faces),3*n),
    ).tocsr()

    current=flat0.copy()
    total_nfev=0
    stages=0
    message=""
    # Continuation: enforce planarity progressively instead of making the first
    # trust-region system ill-conditioned with an enormous penalty.
    rhos=(1e1,1e2,1e3,1e4,1e5,1e6)
    per_stage=max(20,int(max_iterations)//len(rhos))
    print(
        f"[K3D-PLANARITY-POLISH] start sparse vertices={n} quads={len(faces)} "
        f"tol={tolerance:g} stages={len(rhos)}",
        flush=True,
    )
    for rho in rhos:
        sr=math.sqrt(max(w_ref,1e-16)); sp=math.sqrt(rho)
        def residual(flat):
            return np.concatenate((sr*(flat-flat0),sp*coplanarity(flat)))

        result=least_squares(
            residual,current,jac_sparsity=jac_pattern,method="trf",
            tr_solver="lsmr",x_scale="jac",
            ftol=1e-9,xtol=1e-9,gtol=1e-9,
            max_nfev=per_stage,verbose=0,
        )
        current=result.x
        total_nfev+=int(getattr(result,"nfev",0)); stages+=1
        candidate=current.reshape((-1,3))*scale+center
        dev=[]
        for f in faces:
            q=candidate[f]
            dev.extend(np.linalg.norm(q-_best_fit_plane_projection(q),axis=1))
        max_plan=float(np.max(dev)) if dev else 0.0
        print(
            f"[K3D-PLANARITY-POLISH] rho={rho:g} nfev={getattr(result,'nfev',0)} "
            f"planarity_max={max_plan:.6g}",
            flush=True,
        )
        message=str(getattr(result,"message",""))
        if max_plan<=max(float(tolerance),1e-10):
            break

    polished=current.reshape((-1,3))*scale+center
    deviations=[]
    for f in faces:
        q=polished[f]
        deviations.extend(np.linalg.norm(q-_best_fit_plane_projection(q),axis=1))
    dev=np.asarray(deviations,float)
    disp=np.linalg.norm(polished-x0,axis=1)
    max_plan=float(np.max(dev)) if dev.size else 0.0
    # Penalized continuation may stop just above an extremely strict tolerance;
    # downstream geometry is still accepted only when it reaches a practical
    # geometric threshold in mesh units.
    practical_tol=max(float(tolerance),1e-7*scale)
    success=bool(np.all(np.isfinite(polished)) and max_plan<=practical_tol)
    print(
        f"[K3D-PLANARITY-POLISH] done success={success} stages={stages} "
        f"nfev={total_nfev} planarity_max={max_plan:.6g} "
        f"disp_rms={float(np.sqrt(np.mean(disp*disp))) if disp.size else 0.0:.6g}",
        flush=True,
    )
    return polished,{
        "success":success,
        "solver_success":success,
        "iterations":int(total_nfev),
        "stages":int(stages),
        "planarity_max":max_plan,
        "planarity_rms":float(np.sqrt(np.mean(dev*dev))) if dev.size else 0.0,
        "displacement_rms":float(np.sqrt(np.mean(disp*disp))) if disp.size else 0.0,
        "displacement_max":float(np.max(disp)) if disp.size else 0.0,
        "message":message,
        "tolerance":float(tolerance),
        "practical_tolerance":float(practical_tol),
        "reference_weight":float(w_ref),
    }


def optimize_paper_local_global_k3d(target,mesh,parameterization,params,*,pipeline=None):
    """Paper-aligned Eq.(1) local/global K3D."""
    start=time.perf_counter()
    x=np.asarray(mesh.vertices,float).copy()
    faces=np.asarray(mesh.faces,int)
    edges=_edges(faces)
    # Use the weights selected in the Streamlit UI / PipelineParams.  The
    # previous paper-aligned route ignored these values and silently fell back
    # to environment defaults (1, 1, 0.01), so changing the UI had no effect on
    # the authoritative K3D solve.
    w_planar=float(getattr(params,"w_planar",_env_float("ONESTRING_PAPER_K3D_W_PLANAR",1.0)))
    w_square=float(getattr(params,"w_square",_env_float("ONESTRING_PAPER_K3D_W_SQUARE",1.0)))
    w_surface=float(getattr(params,"w_surface",_env_float("ONESTRING_PAPER_K3D_W_SURFACE",0.01)))
    iterations=_env_int("ONESTRING_PAPER_K3D_ITERATIONS",40)

    # Paper edge target: mean of the mean edge lengths of incident quads.
    face_mean=np.zeros(len(faces))
    for fi,f in enumerate(faces):
        q=x[f]; face_mean[fi]=np.mean(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))
    owners={}
    for fi,f in enumerate(faces):
        for k in range(4):
            key=tuple(sorted((int(f[k]),int(f[(k+1)%4])))); owners.setdefault(key,[]).append(fi)
    edge_target={}
    for a,b in edges:
        fs=owners.get((int(a),int(b)),[])
        edge_target[(int(a),int(b))]=float(np.mean(face_mean[fs])) if fs else float(np.linalg.norm(x[b]-x[a]))

    # ---- read-only instrumentation -------------------------------------------
    # These helpers only measure the state this solver produces.  They never feed
    # back into the constraints, so the K3D result is bit-for-bit what it was
    # before the history was recorded.  No second/shadow solve is performed.
    def _planarity_stats(v):
        """Sum of squared best-fit-plane residuals, its per-face mean, and the
        per-vertex distances to each quad's best-fit plane (geometric units)."""
        squared_sum=0.0; per_face=[]; deviations=[]
        for f in faces:
            q=v[f]
            diff=q-_best_fit_plane_projection(q)
            face_sq=float(np.sum(diff*diff))
            squared_sum+=face_sq; per_face.append(face_sq)
            deviations.append(np.linalg.norm(diff,axis=1))
        dev=np.concatenate(deviations) if deviations else np.zeros(0)
        return squared_sum,(float(np.mean(per_face)) if per_face else 0.0),dev

    def _square_energy(v):
        """E_Square = E_Shape (closest-square projection) + E_Length (Eq. 3)."""
        total=0.0
        for f in faces:
            q=v[f]; diff=q-_closest_square_projection(q)
            total+=float(np.sum(diff*diff))
        for a,b in edges:
            d=v[int(b)]-v[int(a)]; ln=float(np.linalg.norm(d))
            if ln<1e-12: continue
            diff=d-edge_target[(int(a),int(b))]*d/ln
            total+=float(np.dot(diff,diff))
        return total

    def _checkpoint(v,iteration,step,surface_points=None):
        planar_sum,planar_mean,dev=_planarity_stats(v)
        square_sum=_square_energy(v)
        proj=_surface_project(v,target,parameterization) if surface_points is None else surface_points
        offsets=v-proj
        surface_sum=float(np.sum(offsets*offsets))
        return dict(
            iteration=int(iteration),
            step=float(step),
            EAssembled=w_planar*planar_sum+w_square*square_sum+w_surface*surface_sum,
            w1EPlanar=w_planar*planar_sum,
            w2ESquare=w_square*square_sum,
            w3ESurface=w_surface*surface_sum,
            EPlanarSum=planar_sum,
            ESquareSum=square_sum,
            ESurfaceSum=surface_sum,
            planarity_max=float(np.max(dev)) if dev.size else 0.0,
            planarity_rms=float(np.sqrt(np.mean(dev*dev))) if dev.size else 0.0,
            planarity_mean=float(np.mean(dev)) if dev.size else 0.0,
            # Legacy per-face means kept for k3d_planarity_residual / k3d_surface_residual.
            EPlanar=planar_mean,
            ESurface=float(np.mean(np.sum(offsets*offsets,axis=1))),
        )
    # --------------------------------------------------------------------------

    records=[_checkpoint(x,0,0.0)]
    for it in range(iterations):
        constraints=[]
        # Local P_P and P_Q.
        for f in faces:
            q=x[f]
            pp=_best_fit_plane_projection(q)
            pq=_closest_square_projection(q)
            for local,vid in enumerate(f):
                constraints.append(([int(vid)],[1.],pp[local],w_planar))
                constraints.append(([int(vid)],[1.],pq[local],w_square))
        # E_Length projection of each edge vector to target K3D tile scale.
        for a,b in edges:
            d=x[b]-x[a]; ln=float(np.linalg.norm(d))
            if ln<1e-12: continue
            t=edge_target[(int(a),int(b))]*d/ln
            constraints.append(([int(a),int(b)],[-1.,1.],t,w_square))
        # P_S: closest point on target surface.
        ps=_surface_project(x,target,parameterization)
        for i,p in enumerate(ps):
            constraints.append(([i],[1.],p,w_surface))
        new=_solve_constraints(len(x),3,constraints,anchor=x,anchor_weight=1e-9)
        step=float(np.linalg.norm(new-x)/max(math.sqrt(len(x)),1.))
        x=new
        records.append(_checkpoint(x,it+1,step))
        if step<1e-8: break
    iteration_history={
        "records":records,
        "solver":"optimize_paper_local_global_k3d",
        "backend":"paper-aligned Eq.(1) local projection + global least squares (authoritative K3D solver)",
        "weights":{"w1_planar":w_planar,"w2_square":w_square,"w3_surface":w_surface},
        "units":"mesh coordinate units (same units as the K3D vertex coordinates)",
        "planarity_definition":"distance from each quad vertex to that quad's best-fit plane (paper Sec. 6 definition)",
        "shadow_solve":False,
    }

    metrics=dict(getattr(mesh,"metrics",{}))
    metrics.update(
        k3d_solver_model="paper_aligned_projection_local_global",
        k3d_objective_terms="EAssembled = w1*EPlanar + w2*(ELength+EShape) + w3*ESurface",
        k3d_exactness_label="paper_aligned_not_reference_exact",
        k3d_local_global_iterations=max(0,len(records)-1),
        k3d_planarity_residual=records[-1]["EPlanar"] if records else 0.,
        k3d_surface_residual=records[-1]["ESurface"] if records else 0.,
        k3d_iteration_history=iteration_history,
        k3d_iteration_history_available=True,
        paper_alignment_note="Explicit PP/PQ/PS local projections and sparse global least-squares. Surface projection uses closest triangle among KD-tree candidates."
    )
    if not np.all(np.isfinite(x)):
        raise RuntimeError("Paper local/global K3D produced non-finite vertices")
    # Guard against the unconstrained global least-squares null mode collapsing
    # or exploding the whole K3D. Translation is fixed by recentering to M3D;
    # scale is not altered because edge/square constraints determine it.
    source_center=np.asarray(mesh.vertices,float).mean(axis=0)
    solved_center=x.mean(axis=0)
    x=x+(source_center-solved_center)
    # Recentering is a rigid translation of the solver output, so re-measure the
    # last checkpoint: the final history row must describe exactly the K3D that
    # this function returns.
    if records:
        last=records[-1]
        records[-1]=_checkpoint(x,last["iteration"],last["step"])
        metrics["k3d_planarity_residual"]=records[-1]["EPlanar"]
        metrics["k3d_surface_residual"]=records[-1]["ESurface"]
    span=np.ptp(x,axis=0)
    source_span=np.ptp(np.asarray(mesh.vertices,float),axis=0)
    if float(np.linalg.norm(span)) < 1e-10*max(float(np.linalg.norm(source_span)),1.0):
        raise RuntimeError("Paper local/global K3D collapsed to a near-point configuration")
    # Post-process the authoritative local/global K3D with a constrained
    # minimum-displacement planarization.  Downstream code receives this
    # polished K3D object, so K2D edge targets and T3D extrusion both derive
    # from exactly the same planarized geometry.
    x_pre_polish=x.copy()
    polish_tol=_env_float("ONESTRING_K3D_POLISH_TOLERANCE",1e-8)
    polish_iters=_env_int("ONESTRING_K3D_POLISH_MAX_ITERATIONS",300)
    polished,polish=minimum_displacement_planarity_polish(
        x_pre_polish,faces,tolerance=polish_tol,max_iterations=polish_iters
    )
    if polish["success"]:
        x=polished
    else:
        raise RuntimeError(
            "K3D minimum-displacement planarity polish failed: "
            f"{polish['message']} (planarity_max={polish['planarity_max']:.6g})"
        )
    metrics.update({
        "k3d_planarity_polish_applied":True,
        "k3d_planarity_polish_solver":"sparse trust-region continuation: minimum displacement + quad coplanarity penalty",
        "k3d_planarity_polish_reference":"pre-polish paper local/global K3D",
        "k3d_planarity_polish_success":bool(polish["success"]),
        "k3d_planarity_polish_iterations":int(polish["iterations"]),
        "k3d_planarity_polish_tolerance":float(polish["tolerance"]),
        "k3d_planarity_polish_max":float(polish["planarity_max"]),
        "k3d_planarity_polish_rms":float(polish["planarity_rms"]),
        "k3d_planarity_polish_displacement_rms":float(polish["displacement_rms"]),
        "k3d_planarity_polish_displacement_max":float(polish["displacement_max"]),
        "k3d_downstream_geometry":"minimum-displacement planarized K3D",
    })
    # Preserve the local/global history as the history of that solver; add an
    # explicit terminal polish record instead of pretending SLSQP was another
    # local/global iteration.
    iteration_history["polish"] = dict(polish)
    metrics["k3d_planarity_residual"]=float(polish["planarity_rms"]**2)
    # The final history row must describe the geometry actually returned to
    # downstream K2D/T3D, not the pre-polish local/global iterate.
    if records:
        last=records[-1]
        records[-1]=_checkpoint(x,last["iteration"],last["step"])
        metrics["k3d_surface_residual"]=records[-1]["ESurface"]
    span=np.ptp(x,axis=0)
    metrics["k3d_bbox_span"]=[float(v) for v in span]
    metrics["k3d_vertex_min"]=[float(v) for v in np.min(x,axis=0)]
    metrics["k3d_vertex_max"]=[float(v) for v in np.max(x,axis=0)]
    out=type(mesh)(x,faces.copy(),mesh.grid,"K3D",metrics,list(getattr(mesh,"split_lines",[])))
    report_type=getattr(pipeline,"StageReport",None)
    if report_type is None: raise RuntimeError("paper K3D requires StageReport")
    report=report_type(name="M3D -> K3D",objective=metrics["k3d_objective_terms"],
        before_error=0.,after_error=float(metrics["k3d_planarity_residual"]+metrics["k3d_surface_residual"]),
        constraint_violation=float(metrics["k3d_planarity_residual"]),computation_time=time.perf_counter()-start,
        counts={"vertices":len(x),"quads":len(faces)})
    print(f"[PAPER-LG-K3D] iterations={len(records)} seconds={report.computation_time:.3f}",flush=True)
    return out,report


def _cross2(a,b): return a[0]*b[1]-a[1]*b[0]


def _sat_projection(poly_a,poly_b,tol=1e-10):
    """Return minimum separating translation for A and -translation for B."""
    best_depth=np.inf; best_axis=None
    for poly in (poly_a,poly_b):
        for i in range(4):
            e=poly[(i+1)%4]-poly[i]
            axis=np.array([-e[1],e[0]],float); n=np.linalg.norm(axis)
            if n<tol: continue
            axis/=n
            pa=poly_a@axis; pb=poly_b@axis
            negative=pa.max()-pb.min()
            positive=pb.max()-pa.min()
            if min(negative,positive)<=tol: return None
            overlap=min(negative,positive)  # also separates contained polygons
            if overlap<best_depth:
                if positive<negative: axis=-axis
                best_depth=float(overlap); best_axis=axis
    return None if best_axis is None else (0.5*(best_depth+tol)*best_axis)


def optimize_paper_local_global_k2d(mesh_2d,mesh_3d,params,*,progress_callback=None,pipeline=None):
    """Paper-aligned Eq.(5): local projections + global sparse least squares."""
    start=time.perf_counter()
    topology,x=build_linkage_topology(mesh_2d)
    faces=topology.faces
    source3=np.asarray(mesh_3d.vertices,float)[topology.source_vertex_ids]
    edges=_edges(faces)
    target_len={(int(a),int(b)):float(np.linalg.norm(source3[b]-source3[a])) for a,b in edges}
    w_edge=_env_float("ONESTRING_EQ5_W_EDGE",1.)
    w_col=_env_float("ONESTRING_EQ5_W_COLLISION",1.)
    w_fab=_env_float("ONESTRING_EQ5_W_FAB",.001)
    theta=math.radians(_env_float("ONESTRING_EQ5_THETA_MIN_DEG",5.))
    iterations=_env_int("ONESTRING_EQ5_ITERATIONS",240)
    records=[]; snapshots=[dict(iteration=0,xy=x.copy(),initial=True)]

    for it in range(iterations):
        constraints=[]
        # P_E: project edge vector to K3D target length.
        for a,b in edges:
            d=x[b]-x[a]; ln=float(np.linalg.norm(d))
            if ln<1e-12: continue
            target=target_len[(int(a),int(b))]*d/ln
            constraints.append(([int(a),int(b)],[-1.,1.],target,w_edge))
        # P_F: exact angle-cone projection already used by current Eq.5 implementation.
        if len(topology.gaps):
            c=topology.gaps[:,0]; ia=topology.gaps[:,1]; ib=topology.gaps[:,2]
            va=x[ia]-x[c]; vb=x[ib]-x[c]
            pa,pb,_=project_angle_vectors(va,vb,theta)
            for j in range(len(c)):
                constraints.append(([int(c[j]),int(ia[j])],[-1.,1.],pa[j],w_fab))
                constraints.append(([int(c[j]),int(ib[j])],[-1.,1.],pb[j],w_fab))
        # P_C: non-penetration projection. SAT supplies the minimum separating
        # translation, but unlike the old solver this is a LOCAL PROJECTION,
        # not a SAT-depth penalty differentiated inside the objective.
        collision_pairs=[]
        polys=x[faces]
        lo=polys.min(axis=1); hi=polys.max(axis=1)
        for a in range(len(faces)):
            for b in range(a+1,len(faces)):
                if hi[a,0]<=lo[b,0] or hi[b,0]<=lo[a,0] or hi[a,1]<=lo[b,1] or hi[b,1]<=lo[a,1]: continue
                shift=_sat_projection(polys[a],polys[b])
                if shift is None: continue
                collision_pairs.append((a,b))
                for vid in faces[a]:
                    constraints.append(([int(vid)],[1.],x[int(vid)]-shift,w_col))
                for vid in faces[b]:
                    constraints.append(([int(vid)],[1.],x[int(vid)]+shift,w_col))
        new=_solve_constraints(len(x),2,constraints,anchor=x,anchor_weight=1e-8)
        step=float(np.linalg.norm(new-x))
        x=new
        # diagnostics
        fab=0; amin=180.;amax=0.
        for c,a,b in topology.gaps:
            u=x[a]-x[c];v=x[b]-x[c]
            ang=math.degrees(math.atan2(abs(_cross2(u,v)),np.dot(u,v)))
            amin=min(amin,ang);amax=max(amax,ang)
            fab+=int(ang<math.degrees(theta)-1e-5 or ang>90.+1e-5)
        edge_sq=0.0; edge_abs=[]
        for a,b in edges:
            err=float(np.linalg.norm(x[b]-x[a])-target_len[(int(a),int(b))])
            edge_sq+=err*err; edge_abs.append(abs(err))
        fab_energy=0.0
        if len(topology.gaps):
            cc=topology.gaps[:,0]; aa=topology.gaps[:,1]; bb=topology.gaps[:,2]
            va=x[aa]-x[cc]; vb=x[bb]-x[cc]
            pa,pb,_=project_angle_vectors(va,vb,theta)
            fab_energy=float(np.sum((va-pa)**2)+np.sum((vb-pb)**2))
        # ECollision is reported as the squared local projection displacement.
        collision_projection_energy=0.0
        collision_pairs=[]
        polys=x[faces]; lo=polys.min(axis=1); hi=polys.max(axis=1)
        for a in range(len(faces)):
            for b in range(a+1,len(faces)):
                if hi[a,0]<=lo[b,0] or hi[b,0]<=lo[a,0] or hi[a,1]<=lo[b,1] or hi[b,1]<=lo[a,1]: continue
                shift=_sat_projection(polys[a],polys[b])
                if shift is not None:
                    collision_pairs.append((a,b))
                    collision_projection_energy += 8.0*float(np.dot(shift,shift))
        eflat=w_edge*edge_sq+w_col*collision_projection_energy+w_fab*fab_energy
        row=dict(iteration=it+1,step=step,EFlat=eflat,EEdge=edge_sq,
                 ECollision=collision_projection_energy,EFab=fab_energy,
                 collisions=len(collision_pairs),fab_violations=fab,
                 gap_min_deg=amin if len(topology.gaps) else 0.,gap_max_deg=amax,
                 edge_rms=math.sqrt(edge_sq/max(len(edges),1)),
                 edge_mean=float(np.mean(edge_abs)) if edge_abs else 0.)
        records.append(row)
        if it==0 or (it+1)%10==0: snapshots.append(dict(row,xy=x.copy()))
        if progress_callback:
            progress_callback("Paper local/global K2D",(it+1)/iterations,f"iter {it+1}/{iterations}; collisions={len(collision_pairs)}; fab={fab}")
        if step<1e-9 and not collision_pairs and fab==0: break

    # Recount final collisions rather than using the pre-global local set.
    final_collisions=0
    polys=x[faces]; lo=polys.min(axis=1); hi=polys.max(axis=1)
    for a in range(len(faces)):
        for b in range(a+1,len(faces)):
            if hi[a,0]<=lo[b,0] or hi[b,0]<=lo[a,0] or hi[a,1]<=lo[b,1] or hi[b,1]<=lo[a,1]: continue
            final_collisions+=int(_sat_projection(polys[a],polys[b]) is not None)
    final=records[-1] if records else dict(iteration=0,step=0.,fab_violations=0,gap_min_deg=0.,gap_max_deg=0.)
    final=dict(final,collisions=final_collisions)
    snapshots.append(dict(final,xy=x.copy(),final=True))
    history=dict(faces=faces.copy(),snapshots=snapshots,records=records,initial_xy=np.asarray(mesh_2d.vertices,float)[topology.source_vertex_ids,:2].copy(),final_xy=x.copy(),solver_message="paper-aligned local/global")
    metrics=dict(getattr(mesh_2d,"metrics",{}))
    metrics.update(final,
        objective="EFlat = w1*EEdge + w2*ECollision + w3*EFab (projection local/global)",
        actual_backend="paper_aligned_projection_local_global",
        paper_collision_discretization="SAT minimum-separating-translation used as local non-penetration projection; not claimed identical to Konakovic reference implementation",
        paper_eq5_pairwise_linkage=True,paper_eq5_auxetic_topology=False,
        fabrication_feasible=(final_collisions==0 and final["fab_violations"]==0),
        theta_min_deg=math.degrees(theta),
        optimizer_iterations=len(records))
    out=type(mesh_2d)(np.column_stack((x,np.zeros(len(x)))),faces.copy(),mesh_2d.grid,"K2D",metrics,list(getattr(mesh_2d,"split_lines",[])))
    out.linkage_topology=topology; out.eq5_history=history
    report_type=getattr(pipeline,"StageReport",None)
    if report_type is None: raise RuntimeError("paper K2D requires StageReport")
    report=report_type(name="M2D -> K2D",objective=metrics["objective"],before_error=0.,after_error=0.,
        constraint_violation=float(final_collisions+final["fab_violations"]),computation_time=time.perf_counter()-start,
        counts={"vertices":len(x),"quads":len(faces),"hinges":len(topology.hinges)})
    print(f"[PAPER-LG-K2D] iterations={len(records)} collisions={final_collisions} fab={final['fab_violations']} seconds={report.computation_time:.3f}",flush=True)
    return out,report


__all__=["optimize_paper_local_global_k3d","optimize_paper_local_global_k2d","minimum_displacement_planarity_polish"]
