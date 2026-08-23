"""Validity wrapper for OptCuts paths, with non-fatal diagnostics in optcuts_test."""
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


def _is_test_mode(mesh: Any, parameterization: Any) -> bool:
    if bool(getattr(mesh, "_optcuts_test_boundary_clipped", False)):
        return True
    metrics=dict(getattr(mesh,"metrics",{}) or {})
    if metrics.get("omega_parameterization_mode")=="optcuts_test" or metrics.get("parameterization_method")=="optcuts_test":
        return True
    return str(getattr(parameterization,"method",""))=="optcuts_test"


def _store_invalid(mesh: Any, invalid: list[tuple[int,str]], stage: str) -> None:
    ids=[int(fi) for fi,_ in invalid]
    reasons={int(fi):str(reason) for fi,reason in invalid}
    setattr(mesh,"_optcuts_invalid_face_ids",ids)
    setattr(mesh,"_optcuts_invalid_face_reasons",reasons)
    metrics=dict(getattr(mesh,"metrics",{}) or {})
    metrics.update({
        "optcuts_invalid_face_ids":ids,
        "optcuts_invalid_face_reasons":reasons,
        "optcuts_invalid_face_count":len(ids),
        "optcuts_invalid_face_stage":stage,
    })
    try: mesh.metrics.update(metrics)
    except Exception: pass


def install_optcuts_k3d_validity_patch(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_optcuts_k3d_validity_patch_installed",False): return
    base_optimize=pipeline._optimize_k3d

    # Install the visualization overlay regardless of whether the separate
    # OptCuts visualization compatibility patch has already run.  Either wrapper
    # order is safe because both wrap the current figure function.
    try:
        from .optcuts_invalid_visualization_patch import install_optcuts_invalid_visualization_patch
        install_optcuts_invalid_visualization_patch()
    except Exception as exc:
        print(f"[OPTCUTS-INVALID-VIZ] install skipped: {type(exc).__name__}: {exc}")

    def optimize_k3d_with_validity(target,mesh,parameterization,params):
        out,report=base_optimize(target,mesh,parameterization,params)
        if not _optcuts_active(pipeline,mesh) and str(getattr(parameterization,"method","")) not in ("optcuts","optcuts_test"):
            return out,report

        test_mode=_is_test_mode(mesh,parameterization)
        base_vertices=np.asarray(mesh.vertices,float); faces=np.asarray(out.faces,int); candidate=np.asarray(out.vertices,float)
        base_invalid=_invalid_faces(base_vertices,faces)
        candidate_invalid=_invalid_faces(candidate,faces)

        if base_invalid:
            reasons=dict(Counter(r for _,r in base_invalid)); details=_diagnose_invalid(mesh,base_invalid)
            _store_invalid(mesh,base_invalid,"M3D")
            print(f"[OPTCUTS-M3D-INVALID] count={len(base_invalid)} reasons={reasons} details={details}")
            if not test_mode:
                raise RuntimeError(
                    "OPTCUTS_INVALID_M3D_BEFORE_K3D: lifted M3D already contains invalid quads; "
                    f"reasons={reasons} details={details}"
                )
            # Diagnostic test mode: do not pretend alpha=0 is a valid endpoint.
            # Keep the K3D optimizer result, mark whatever remains invalid, and
            # continue so the complete state can be inspected.
            final_invalid=candidate_invalid
            _store_invalid(out,final_invalid,"K3D")
            out.metrics.update({
                "optcuts_k3d_validity_guard_applied":True,
                "optcuts_k3d_nonfatal_diagnostic_mode":True,
                "optcuts_m3d_invalid_initial_count":len(base_invalid),
                "optcuts_k3d_invalid_final_count":len(final_invalid),
                "optcuts_k3d_invalid_final_reason_counts":dict(Counter(r for _,r in final_invalid)),
            })
            try: report.failed_constraints.append("optcuts_test_invalid_panels_nonfatal")
            except Exception: pass
            print(
                f"[OPTCUTS-TEST-NONFATAL] M3D_invalid={len(base_invalid)} "
                f"K3D_invalid={len(final_invalid)}; continuing pipeline"
            )
            return out,report

        metrics=dict(getattr(out,"metrics",{}) or {})
        metrics.update({"optcuts_k3d_validity_guard_applied":True,
                        "optcuts_k3d_invalid_candidate_count":int(len(candidate_invalid)),
                        "optcuts_k3d_invalid_candidate_reason_counts":dict(Counter(r for _,r in candidate_invalid))})
        if not candidate_invalid:
            _store_invalid(mesh,[],"M3D")
            _store_invalid(out,[],"K3D")
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
            if test_mode:
                _store_invalid(out,final_invalid,"K3D")
                out.metrics.update({"optcuts_k3d_nonfatal_diagnostic_mode":True,"optcuts_k3d_invalid_final_count":len(final_invalid)})
                print(f"[OPTCUTS-TEST-NONFATAL] K3D guard could not repair {len(final_invalid)} faces; continuing")
                return out,report
            raise RuntimeError(f"OPTCUTS_K3D_VALIDITY_GUARD_FAILED: examples={_diagnose_invalid(mesh,final_invalid)}")
        out.vertices=repaired
        _store_invalid(out,[],"K3D")
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
