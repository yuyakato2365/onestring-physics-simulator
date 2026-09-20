"""Record and render the M3D -> K3D optimization trajectory.

This is a diagnostic-only patch for the 2026-09-20 paper-T3D launcher.  It keeps
the numerical K3D result unchanged: the original optimizer is still authoritative.
A shadow SciPy least-squares solve records the same CPU residual objective at
iteration/evaluation checkpoints so E_Assembled and geometric planarity can be
inspected in the UI, analogous to the Eq.(5) K2D history.
"""
from __future__ import annotations

import math
import numpy as np


def _components(pipeline, target, mesh, parameterization, params, vertices):
    v=np.asarray(vertices,float)
    rp=np.asarray(pipeline._planarity_residuals(v,mesh.faces),float)
    rs=np.asarray(pipeline._square_residuals(v,mesh.faces),float)
    closest=np.asarray(pipeline._closest_surface_vertices(v,parameterization.surface_vertices_3d),float)
    r_surface=(v-closest).ravel()
    e_planar=float(np.sum(rp*rp))
    e_square=float(np.sum(rs*rs))
    e_surface=float(np.sum(r_surface*r_surface))
    w1=float(params.w_planar);w2=float(params.w_square);w3=float(params.w_surface)
    # Geometric distance from each quad's fourth vertex to the plane through its
    # first three vertices.  This is the same physical quantity reported by the
    # existing K3D planarity metric, kept separate from the weighted energy.
    abs_planar=np.abs(rp)
    return {
        "EAssembled":w1*e_planar+w2*e_square+w3*e_surface,
        "w1EPlanar":w1*e_planar,
        "w2ESquare":w2*e_square,
        "w3ESurface":w3*e_surface,
        "EPlanar":e_planar,
        "ESquare":e_square,
        "ESurface":e_surface,
        "planarity_max":float(np.max(abs_planar)) if abs_planar.size else 0.0,
        "planarity_mean":float(np.mean(abs_planar)) if abs_planar.size else 0.0,
        "planarity_rms":float(np.sqrt(np.mean(abs_planar*abs_planar))) if abs_planar.size else 0.0,
    }


def _shadow_history(pipeline,target,mesh,parameterization,params):
    least_squares=getattr(pipeline,"least_squares",None)
    if least_squares is None:
        return None
    base=np.asarray(mesh.vertices,float).copy()
    records=[]
    eval_count=0
    last_x=None
    def residual(x):
        nonlocal eval_count,last_x
        v=np.asarray(x,float).reshape(-1,3)
        parts=[
            math.sqrt(float(params.w_planar))*np.asarray(pipeline._planarity_residuals(v,mesh.faces),float),
            math.sqrt(float(params.w_square))*np.asarray(pipeline._square_residuals(v,mesh.faces),float),
        ]
        closest=np.asarray(pipeline._closest_surface_vertices(v,parameterization.surface_vertices_3d),float)
        parts.append(math.sqrt(float(params.w_surface))*(v-closest).ravel())
        parts.append(math.sqrt(0.05)*(v[:,:2]-base[:,:2]).ravel())
        eval_count+=1
        # scipy.trf exposes function evaluations rather than a stable per-iteration
        # callback on all supported versions.  Record every objective evaluation;
        # collapse consecutive identical x values to avoid finite-difference spam.
        if last_x is None or not np.array_equal(v,last_x):
            rec=_components(pipeline,target,mesh,parameterization,params,v)
            rec["evaluation"]=eval_count
            records.append(rec)
            last_x=v.copy()
        return np.concatenate([p.ravel() for p in parts if p.size])
    try:
        opt=least_squares(residual,base.ravel(),max_nfev=max(5,int(params.max_3d_iterations)),method="trf")
    except Exception as exc:
        return {"records":[],"message":f"K3D history shadow solve failed: {exc}"}
    final=_components(pipeline,target,mesh,parameterization,params,opt.x.reshape(-1,3))
    final["evaluation"]=eval_count
    if not records or records[-1]["evaluation"]!=eval_count:
        records.append(final)
    return {
        "records":records,
        "message":str(getattr(opt,"message","")),
        "backend":"shadow scipy least_squares diagnostic",
        "note":"History mirrors the CPU K3D residual objective and does not replace the authoritative K3D result.",
    }


def install_k3d_iteration_history_patch(pipeline):
    if getattr(pipeline,"_onestring_k3d_history_installed",False):
        return
    original=pipeline._optimize_k3d
    def wrapped(target,mesh,parameterization,params):
        out,report=original(target,mesh,parameterization,params)
        history=None
        # The authoritative CPU path uses this exact least-squares objective.
        backend=str(getattr(out,"metrics",{}).get("actual_backend",""))
        if backend=="cpu" or backend=="scipy" or not backend:
            history=_shadow_history(pipeline,target,mesh,parameterization,params)
        if history and history.get("records"):
            out.metrics["k3d_iteration_history"]=history
            out.metrics["k3d_iteration_history_available"]=True
        else:
            out.metrics["k3d_iteration_history_available"]=False
            out.metrics["k3d_iteration_history_note"]=(
                "Per-step K3D history is currently recorded for the SciPy/CPU objective; "
                f"authoritative backend was {backend or 'unknown'}."
            )
        return out,report
    pipeline._optimize_k3d=wrapped
    # build_onestring_design is defined in the original module namespace.
    original_module=getattr(pipeline,"_original",None)
    if original_module is not None:
        original_module._optimize_k3d=wrapped
    pipeline._onestring_k3d_history_installed=True


__all__=["install_k3d_iteration_history_patch"]
