"""Validity-preserving K3D wrapper for OptCuts paths, with M3D diagnostics."""
from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np

from .t3d_recovery import validate_top_quad


def _invalid_faces(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[int,str]]:
    verts=np.asarray(vertices,float); f=np.asarray(faces,int); invalid=[]
    for fi,face in enumerate(f):
        try: top=verts[np.asarray(face,int)]
        except Exception:
            invalid.append((int(fi),"invalid_face_index")); continue
        valid,reason=validate_top_quad(np.asarray(top,float))
        if not valid: invalid.append((int(fi),str(reason)))
    return invalid


def _diagnose_invalid(mesh: Any, invalid: list[tuple[int,str]], limit: int=12):
    sources=list(getattr(mesh,"_optcuts_test_face_sources",[]) or [])
    uv_faces=list(getattr(mesh,"_optcuts_test_face_uv",[]) or [])
    rows=[]
    for fi,reason in invalid[:limit]:
        row={"face":int(fi),"reason":str(reason)}
        if 0<=fi<len(sources): row["source"]=str(sources[fi])
        if 0<=fi<len(uv_faces):
            try: row["m2d_uv"]=np.round(np.asarray(uv_faces[fi],float),6).tolist()
            except Exception: pass
        rows.append(row)
    return rows


def _optcuts_active(pipeline: Any, mesh: Any|None=None) -> bool:
    if bool(getattr(pipeline,"_onestring_optcuts_active_run",False)): return True
    metrics=dict(getattr(mesh,"metrics",{}) or {}) if mesh is not None else {}
    return bool(metrics.get("optcuts_grid_seam_enabled",False) or metrics.get("optcuts_grid_seam_applied",False)
                or metrics.get("flattening_backend")=="official_optcuts_external"
                or metrics.get("omega_parameterization_mode") in ("optcuts","optcuts_test")
                or metrics.get("parameterization_method") in ("optcuts","optcuts_test"))


def install_optcuts_k3d_validity_patch(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_optcuts_k3d_validity_patch_installed",False): return
    base_optimize=pipeline._optimize_k3d

    def optimize_k3d_with_validity(target,mesh,parameterization,params):
        out,report=base_optimize(target,mesh,parameterization,params)
        if not _optcuts_active(pipeline,mesh) and str(getattr(parameterization,"method","")) not in ("optcuts","optcuts_test"):
            return out,report
        base_vertices=np.asarray(mesh.vertices,float); faces=np.asarray(out.faces,int); candidate=np.asarray(out.vertices,float)
        base_invalid=_invalid_faces(base_vertices,faces)
        if base_invalid:
            reasons=dict(Counter(r for _,r in base_invalid)); details=_diagnose_invalid(mesh,base_invalid)
            print(f"[OPTCUTS-M3D-INVALID] count={len(base_invalid)} details={details}")
            raise RuntimeError(
                "OPTCUTS_INVALID_M3D_BEFORE_K3D: lifted M3D already contains invalid quads; "
                f"reasons={reasons} details={details}"
            )

        candidate_invalid=_invalid_faces(candidate,faces)
        metrics=dict(getattr(out,"metrics",{}) or {})
        metrics.update({"optcuts_k3d_validity_guard_applied":True,
                        "optcuts_k3d_invalid_candidate_count":int(len(candidate_invalid)),
                        "optcuts_k3d_invalid_candidate_reason_counts":dict(Counter(r for _,r in candidate_invalid))})
        if not candidate_invalid:
            metrics.update({"optcuts_k3d_validity_backtracked":False,"optcuts_k3d_validity_step_alpha":1.0,"optcuts_k3d_invalid_final_count":0})
            out.metrics.update(metrics); print("[OPTCUTS-K3D-GUARD] candidate_valid=True alpha=1.000000 invalid_final=0"); return out,report

        displacement=candidate-base_vertices; low=0.0; high=1.0
        for _ in range(20):
            mid=.5*(low+high); trial=base_vertices+mid*displacement
            if _invalid_faces(trial,faces): high=mid
            else: low=mid
        alpha=max(0.0,min(1.0,low*.95)); repaired=base_vertices+alpha*displacement; final_invalid=_invalid_faces(repaired,faces)
        if final_invalid:
            alpha=0.0; repaired=base_vertices.copy(); final_invalid=_invalid_faces(repaired,faces)
        if final_invalid:
            raise RuntimeError(f"OPTCUTS_K3D_VALIDITY_GUARD_FAILED: examples={_diagnose_invalid(mesh,final_invalid)}")
        out.vertices=repaired
        metrics.update({"optcuts_k3d_validity_backtracked":True,"optcuts_k3d_validity_step_alpha":float(alpha),
                        "optcuts_k3d_validity_boundary_alpha":float(low),"optcuts_k3d_invalid_final_count":0,
                        "optcuts_k3d_validity_model":"global M3D->K3D displacement backtracking under validate_top_quad"})
        out.metrics.update(metrics)
        try: report.failed_constraints.append("optcuts_k3d_top_validity_backtrack")
        except Exception: pass
        print(f"[OPTCUTS-K3D-GUARD] candidate_valid=False candidate_invalid={len(candidate_invalid)} alpha={alpha:.6f} boundary_alpha={low:.6f} invalid_final=0")
        return out,report

    pipeline._optimize_k3d=optimize_k3d_with_validity
    original=getattr(pipeline,"_original",None)
    if original is not None: original._optimize_k3d=optimize_k3d_with_validity
    for fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict): glb["_optimize_k3d"]=optimize_k3d_with_validity
    pipeline._onestring_optcuts_k3d_validity_patch_installed=True


__all__=["install_optcuts_k3d_validity_patch","_invalid_faces"]
