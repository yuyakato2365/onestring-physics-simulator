"""Persistent K2D checkpoints and a global all-hinge rigid-tile solver.

This module is deliberately independent from the spanning-tree kinematic solver.
Every tile owns an SE(2) pose (tx, ty, theta).  Every hinge is treated equally
in one global residual system; there are no privileged tree hinges.  History
checkpoints let the K2D experiment be rerun without OptCuts/M2D/K3D.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


def history_root() -> Path:
    root = Path(__file__).resolve().parents[2] / ".onestring_history" / "k2d"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_k2d_checkpoint(tiles: np.ndarray, constraints: list[tuple[int,int,int,int]], metrics: dict[str,Any] | None=None, *, label: str="pipeline") -> Path:
    x = np.asarray(tiles, dtype=float)
    c = np.asarray(constraints, dtype=np.int64).reshape((-1,4))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = history_root() / f"{stamp}_{label}.npz"
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "label": label, "tiles": int(len(x)), "hinges": int(len(c))}
    for k,v in dict(metrics or {}).items():
        if isinstance(v,(str,int,float,bool)) or v is None:
            meta[str(k)] = v
    np.savez_compressed(path, tiles=x, constraints=c, metadata=np.asarray(json.dumps(meta)))
    return path


def list_checkpoints() -> list[Path]:
    return sorted(history_root().glob("*.npz"), reverse=True)


def load_checkpoint(path: str | Path) -> tuple[np.ndarray,list[tuple[int,int,int,int]],dict[str,Any]]:
    with np.load(Path(path), allow_pickle=False) as z:
        tiles=np.asarray(z["tiles"],dtype=float)
        constraints=[tuple(map(int,row)) for row in np.asarray(z["constraints"],dtype=int)]
        meta=json.loads(str(np.asarray(z["metadata"]).item()))
    return tiles,constraints,meta


def _rotated(local: np.ndarray, theta: np.ndarray) -> np.ndarray:
    c=np.cos(theta)[:,None,None]; s=np.sin(theta)[:,None,None]
    out=np.empty_like(local)
    out[:,:,0]=local[:,:,0]*c[:,0,:]-local[:,:,1]*s[:,0,:]
    out[:,:,1]=local[:,:,0]*s[:,0,:]+local[:,:,1]*c[:,0,:]
    return out


def poses_to_tiles(local: np.ndarray, centres0: np.ndarray, z: np.ndarray) -> np.ndarray:
    n=len(local); q=np.asarray(z,dtype=float).reshape((n,3))
    return _rotated(local,q[:,2])+q[:,:2,None].transpose(0,2,1)


def global_hinge_solve(initial: np.ndarray, constraints: list[tuple[int,int,int,int]], *, max_nfev: int=300, anchor_weight: float=1e-5) -> tuple[np.ndarray,dict[str,Any]]:
    """Solve all point-hinge coincidences simultaneously over all tile SE(2) poses.

    Collision is intentionally *not* hidden in this feasibility phase.  The
    result answers the first question cleanly: can all physical hinges close at
    once while every tile stays rigid?  A second collision phase can then start
    from this result rather than mixing two failure causes.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    x0=np.asarray(initial,dtype=float); n=len(x0)
    centres=np.mean(x0,axis=1); local=x0-centres[:,None,:]
    z0=np.column_stack([centres,np.zeros(n)]).reshape(-1)
    scale=max(float(np.median(np.linalg.norm(x0[:,1]-x0[:,0],axis=1))),1e-12)

    # Gauge: tile 0 pose is weakly anchored; all hinges remain in the same global system.
    def residual(z: np.ndarray) -> np.ndarray:
        tiles=poses_to_tiles(local,centres,z)
        vals=[]
        for ia,ca,ib,cb in constraints:
            vals.extend(((tiles[ia,ca]-tiles[ib,cb])/scale).tolist())
        q=z.reshape((n,3))
        vals.extend((np.sqrt(anchor_weight)*(q[0,:2]-centres[0])/scale).tolist())
        vals.append(float(np.sqrt(anchor_weight)*q[0,2]))
        return np.asarray(vals,dtype=float)

    rows=2*len(constraints)+3; cols=3*n
    sp=lil_matrix((rows,cols),dtype=int); r=0
    for ia,_ca,ib,_cb in constraints:
        sp[r:r+2,3*ia:3*ia+3]=1; sp[r:r+2,3*ib:3*ib+3]=1; r+=2
    sp[r:r+3,0:3]=1
    result=least_squares(residual,z0,jac_sparsity=sp.tocsr(),max_nfev=int(max_nfev),xtol=1e-10,ftol=1e-10,gtol=1e-10,verbose=0)
    final=poses_to_tiles(local,centres,result.x)
    errs=np.asarray([np.linalg.norm(final[ia,ca]-final[ib,cb]) for ia,ca,ib,cb in constraints],dtype=float)
    tol=max(scale*5e-4,1e-9)
    metrics={
        "solver":"global_all_hinge_se2",
        "success":bool(result.success), "message":str(result.message), "nfev":int(result.nfev),
        "tile_count":n, "hinge_count":len(constraints), "tile_scale":scale,
        "hinge_tolerance":tol, "hinge_max_error":float(np.max(errs)) if errs.size else 0.0,
        "hinge_rms_error":float(np.sqrt(np.mean(errs*errs))) if errs.size else 0.0,
        "hinge_violations":int(np.sum(errs>tol)),
        "all_hinges_satisfied":bool(errs.size==0 or np.max(errs)<=tol),
    }
    return final,metrics


def install_k2d_history_recorder(pipeline: Any) -> None:
    """Record the authoritative K2D input/output every test run, without changing numerics."""
    from . import optcuts_test_k2d_relative_layout_patch as mod
    if getattr(mod,"_onestring_k2d_history_installed",False): return
    base=mod._build_rigid_k2d_layout
    def wrapped(pipeline_obj: Any, mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        tiles,metrics=base(pipeline_obj,mesh_2d,mesh_3d,params,progress_callback=progress_callback)
        try:
            faces=np.asarray(mesh_2d.faces,dtype=int)
            constraints=mod._hinge_constraints(pipeline_obj,faces)
            p=save_k2d_checkpoint(np.asarray(tiles,dtype=float),constraints,dict(metrics or {}),label="k2d")
            metrics=dict(metrics or {}); metrics["onestring_k2d_history_checkpoint"]=str(p)
            print(f"[OPTCUTS-K2D-HISTORY] saved={p}")
        except Exception as exc:
            print(f"[OPTCUTS-K2D-HISTORY-WARN] {type(exc).__name__}: {exc}")
        return tiles,metrics
    mod._build_rigid_k2d_layout=wrapped
    mod._onestring_k2d_history_installed=True

__all__=["history_root","save_k2d_checkpoint","list_checkpoints","load_checkpoint","global_hinge_solve","install_k2d_history_recorder"]
