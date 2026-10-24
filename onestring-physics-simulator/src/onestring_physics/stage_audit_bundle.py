"""Side-effect-free stage capture for repeatable OneString visual audits.

Opt-in only. Emits a single run_id, stable camera bounds, stage arrays and
machine-readable provenance. Does not modify any production geometry.
"""
from __future__ import annotations
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np


def _jsonable(value):
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value,float) and not np.isfinite(value):
        # Nonfinite metrics are EVIDENCE. Keep them explicitly as
        # strings, rather than failing to write a crash bundle.
        return ("NaN" if np.isnan(value) else
                "Infinity" if value>0 else "-Infinity")
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_jsonable(x) for x in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return str(value)


def _git_sha():
    try:
        return subprocess.check_output(
            ["git","rev-parse","HEAD"],text=True,
            stderr=subprocess.DEVNULL,timeout=3,
        ).strip()
    except (OSError,subprocess.SubprocessError):
        return "unavailable"


def _fixed_bounds(arrays):
    # One fixed framing for all stages: SOURCE geometry controls bounds.
    v=arrays.get("S_vertices")
    if v is None or not len(v):return np.array([[-1,1]]*3,float)
    lo=np.min(v,axis=0);hi=np.max(v,axis=0)
    span=max(float(np.max(hi-lo)),1e-9)
    center=.5*(lo+hi)
    return np.column_stack((center-.58*span,center+.58*span))


class StageAuditWriter:
    def __init__(self,base_dir,params,*,run_id=None,runtime_functions=None):
        self.run_id=str(run_id or (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            +"-"+uuid.uuid4().hex[:8]
        ))
        self.path=Path(base_dir)/self.run_id
        self.path.mkdir(parents=True,exist_ok=False)
        self.stages={}
        self.summary={
            "run_id":self.run_id,"git_sha":_git_sha(),
            "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "parameters":_jsonable(params),
            "runtime_functions":{
                str(k):{
                    "module":str(getattr(v,"__module__","unknown")),
                    "qualname":str(getattr(v,"__qualname__","unknown")),
                } for k,v in (runtime_functions or {}).items()
            },
            "stage_metrics":{},
            "status":"partial",
            "missing_artifacts":[],
        }

    def record(self,name,vertices,faces,metrics=None):
        v=np.asarray(vertices,float)
        f=np.asarray(faces,int)
        if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():
            raise ValueError(f"AUDIT_INVALID_STAGE_VERTICES: {name}")
        if f.ndim!=2 or f.size and (f.min()<0 or f.max()>=len(v)):
            raise ValueError(f"AUDIT_INVALID_STAGE_FACES: {name}")
        self.stages[name+"_vertices"]=v.copy()
        self.stages[name+"_faces"]=f.copy()
        self.summary["stage_metrics"][name]=_jsonable(metrics or {})
        self.flush()

    def record_uv(self,parameterization):
        for name,attribute in (
            ("source_uv","uv_vertices_2d"),
            ("source_uv_faces","uv_faces"),
            ("omega_boundary","omega_boundary"),
        ):
            value=getattr(parameterization,attribute,None)
            if value is not None:
                self.stages[name]=np.asarray(value).copy()
        self.summary["parameterization_metrics"]=_jsonable(
            getattr(parameterization,"metrics",{}) or {}
        )
        self.flush()

    def record_quad(self,name,mesh):
        self.record(
            name,mesh.vertices,mesh.faces,
            getattr(mesh,"metrics",{}) or {},
        )
        if name in ("M3D","K3D"):
            m=getattr(mesh,"metrics",{}) or {}
            rows=m.get("seam_exact_affine_rows",[])
            if rows:
                self.summary["seam_affine_rows"]=_jsonable(rows)
                self.summary["seam_stitch_mode"]=m.get("seam_stitch_mode")
                self.flush()

    def record_t3d(self,tiles):
        # TileAssembly has (n_quads,8,3) points, not a single indexed mesh.
        v=np.asarray(tiles.vertices,float)
        if v.ndim!=3 or v.shape[1:]!=(8,3):
            raise ValueError("AUDIT_INVALID_T3D_TILE_SHAPE")
        self.stages["T3D_tiles"]=v.copy()
        self.summary["stage_metrics"]["T3D"]=_jsonable(
            getattr(tiles,"metrics",{}) or {}
        )
        self.flush()

    def flush(self):
        self.stages["audit_fixed_bounds"]=_fixed_bounds(self.stages)
        # Write atomically, preserving a usable partial artifact on failure.
        temp=self.path/"stages.partial.npz"
        with temp.open("wb") as fp:
            np.savez_compressed(fp,**self.stages)
        os.replace(temp,self.path/"stages.npz")
        target=self.path/"metrics.json"
        candidate=self.path/"metrics.partial.json"
        candidate.write_text(
            json.dumps(_jsonable(self.summary),indent=2,
                       ensure_ascii=False,allow_nan=False),
            encoding="utf-8",
        )
        os.replace(candidate,target)

    def finish(self):
        self.summary["status"]="geometry_stages_complete"
        self.flush()

    def fail(self,stage,exc):
        self.summary["status"]="failed_before_complete"
        self.summary["failure_stage"]=stage
        self.summary["failure_type"]=type(exc).__name__
        self.summary["failure_message"]=str(exc)
        self.flush()
