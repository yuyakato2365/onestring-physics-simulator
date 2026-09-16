"""2026-09-14 OptCuts variant with an exact LSCM K2D/flat-layout route.

This module also provides the capture-time K2D route used by the latest
"Paper Eq.5 K2D + latest Omega + latest K3D planarity" hybrid.

Important performance rule: numerical meaning must not change with tile count.
The latest hybrid therefore uses the same vectorized EEdge projection for every
mesh size.  It does not call the legacy K2D implementation's expensive
collision diagnostics/relaxation at 56%; ECollision and the real EFab gap-angle
term are handled by the unified independent-tile Eq.5 layout stage.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .lscm_latest_omega_hybrid_20260914_patch import install_deferred_hybrid_hook


VARIANT = "3"
VERSION_ID = "2026-09-14-paper-k2d-eq5-stage-separated"
PAPER_HYBRID_MODE = "lscm_latest_omega_hard_k3d"


def _active(params: Any) -> bool:
    explicit = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == VARIANT
    latest_ui = os.environ.get("ONESTRING_PAPER_K2D_20260914", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return bool((explicit or latest_ui) and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test")


def _paper_hybrid_active(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == PAPER_HYBRID_MODE


def _emit(pipeline: Any, callback: Any, stage: str, fraction: float, detail: str) -> None:
    fn = getattr(pipeline, "_emit_progress", None)
    if callable(fn):
        try:
            fn(callback, stage, fraction, detail); return
        except Exception: pass
    if callback is not None:
        try: callback(stage, fraction, detail)
        except Exception: pass


def _paper_edge_only_k2d(pipeline: Any, mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
    started = time.perf_counter()
    base_xy = np.asarray(mesh_2d.vertices[:, :2], dtype=float).copy()
    faces = np.asarray(mesh_2d.faces, dtype=int)
    edges_list = list(pipeline._unique_mesh_edges(faces))
    if edges_list:
        edge_idx = np.asarray(edges_list, dtype=int); aa=edge_idx[:,0]; bb=edge_idx[:,1]
        target_lengths=np.linalg.norm(np.asarray(mesh_3d.vertices,dtype=float)[aa]-np.asarray(mesh_3d.vertices,dtype=float)[bb],axis=1)
    else:
        edge_idx=np.zeros((0,2),dtype=int); aa=np.zeros(0,dtype=int); bb=np.zeros(0,dtype=int); target_lengths=np.zeros(0,dtype=float)
    def errors(xy):
        if len(edge_idx)==0:return 0.0,0.0
        err=np.abs(np.linalg.norm(xy[bb]-xy[aa],axis=1)-target_lengths); return float(np.mean(err)),float(np.max(err))
    before_mean,before_max=errors(base_xy); xy=base_xy.copy(); degree=np.zeros((len(xy),1),dtype=float)
    if len(edge_idx): np.add.at(degree,aa,1.0); np.add.at(degree,bb,1.0)
    degree=np.maximum(degree,1.0); iterations=max(160,int(getattr(params,"max_2d_iterations",40))*6)
    mean_target=float(np.mean(target_lengths)) if len(target_lengths) else 1.0; tolerance=max(1e-7,0.002*mean_target)
    base_centroid=np.mean(base_xy,axis=0,keepdims=True) if len(base_xy) else np.zeros((1,2)); iterations_done=0
    _emit(pipeline,progress_callback,"Paper K2D EEdge",0.02,f"vectorized solve: {len(faces)} tiles, {len(edge_idx)} edges")
    for it in range(iterations):
        if len(edge_idx):
            delta=xy[bb]-xy[aa]; lengths=np.linalg.norm(delta,axis=1); safe=np.maximum(lengths,1e-12)
            correction=((lengths-target_lengths)/safe)[:,None]*delta*0.5; accum=np.zeros_like(xy)
            np.add.at(accum,aa,correction); np.add.at(accum,bb,-correction); xy+=accum/degree
        if len(xy): xy+=base_centroid-np.mean(xy,axis=0,keepdims=True)
        iterations_done=it+1
        if (it+1)%10==0 or it+1==iterations:
            mean_err,max_err=errors(xy); _emit(pipeline,progress_callback,"Paper K2D EEdge",min(0.98,(it+1)/max(1,iterations)),f"iter {it+1}/{iterations}, mean={mean_err:.4g}, max={max_err:.4g}")
            if max_err<=tolerance: break
    after_mean,after_max=errors(xy); vertices=np.column_stack([xy,np.zeros(len(xy),dtype=float)])
    metrics=dict(getattr(mesh_2d,"metrics",{}) or {}); metrics.update({"version_id":"2026-09-14-paper-eq5-unified-k2d","objective":"Paper Eq.5 split: EEdge only in shared metric stage; ECollision + actual EFab in independent linkage stage","paper_eq5_EEdge_stage":True,"paper_eq5_edge_solver":"vectorized projective length constraints","paper_eq5_edge_solver_same_for_all_tile_counts":True,"paper_eq5_no_tile_count_threshold":True,"paper_eq5_no_legacy_collision_diagnostics_at_metric_stage":True,"paper_eq5_old_position_anchor_disabled":True,"paper_eq5_old_position_anchor_was_not_EFab":True,"paper_eq5_collision_fab_deferred_to_independent_linkage":True,"edge_matching_error":after_mean,"edge_matching_error_before":before_mean,"edge_matching_error_after":after_mean,"mean_edge_length_error_before":before_mean,"mean_edge_length_error_after":after_mean,"max_edge_length_error_before":before_max,"max_edge_length_error_after":after_max,"k2d_edge_iterations":iterations_done,"k2d_edge_tolerance":tolerance,"k2d_edge_elapsed_sec":time.perf_counter()-started,"k2d_z_abs_max":0.0,"actual_backend":"paper_vectorized_numpy"})
    out=type(mesh_2d)(vertices,faces.copy(),mesh_2d.grid,"K2D",metrics,list(getattr(mesh_2d,"split_lines",[])))
    report_cls=getattr(pipeline,"StageReport",None)
    if report_cls is None:
        class _Report: pass
        report=_Report(); report.name="M2D -> K2D"; report.objective=metrics["objective"]; report.before_error=before_mean; report.after_error=after_mean; report.constraint_violation=after_max; report.computation_time=time.perf_counter()-started; report.failed_constraints=[]; report.counts={}
    else:
        count_fn=getattr(pipeline,"_mesh_counts",None); counts=count_fn(out) if callable(count_fn) else {}
        report=report_cls(name="M2D -> K2D",objective=metrics["objective"],before_error=before_mean,after_error=after_mean,constraint_violation=after_max,computation_time=time.perf_counter()-started,counts=counts)
    _emit(pipeline,progress_callback,"Paper K2D EEdge",1.0,f"complete in {time.perf_counter()-started:.2f}s")
    print(f"[PAPER-EQ5-K2D-EDGE] unified vectorized EEdge tiles={len(faces)} edges={len(edge_idx)} iter={iterations_done} max_err={after_max:.6g} sec={time.perf_counter()-started:.3f}")
    return out,report


def _tag_lscm_equivalent_result(result: Any, report: Any):
    metrics=dict(getattr(result,"metrics",{}) or {}); metrics.update({"version_id":VERSION_ID,"k2d_solver_equivalent_to_lscm":True,"k2d_common_solver_authoritative":True,"k2d_optcuts_specific_rigid_tile_replacement":False,"k2d_optcuts_hard_sat_stage_in_k2d":False,"k2d_optcuts_global_se2_stage_in_k2d":False,"k2d_split_panel_centroid_realign_disabled":True,"k2d_split_panel_post_eq5_translation_applied":False,"paper_k2d_shared_vertex_authoritative":True,"paper_k2d_independent_rigid_tile_layout_in_k2d":False,"paper_k2d_hinge_alignment_in_k2d":False,"paper_k2d_hinge_layout_deferred_to_section_4_4":True,"paper_k2d_absolute_m2d_layout_reinjected_after_optimization":False})
    try: result.metrics.clear(); result.metrics.update(metrics)
    except Exception: pass
    return result,report


def _tag_exact_lscm_flat_layout(layout,mesh,base_make_layout): return layout

def _wire_k2d(pipeline,fn):
    pipeline._optimize_k2d=fn; original=getattr(pipeline,"_original",None)
    if original is not None: original._optimize_k2d=fn
    for build_fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(build_fn,"__globals__",None)
        if isinstance(glb,dict): glb["_optimize_k2d"]=fn

def _wire_flat_layout(pipeline,fn):
    pipeline._make_flat_tile_layout=fn; original=getattr(pipeline,"_original",None)
    if original is not None: original._make_flat_tile_layout=fn
    for build_fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(build_fn,"__globals__",None)
        if isinstance(glb,dict): glb["_make_flat_tile_layout"]=fn


def _install_simple_split_bypass():
    try: from . import simple_split_panel_patch as simple_split_module
    except Exception:return
    if getattr(simple_split_module,"_onestring_20260914_lscm_k2d_bypass_installed",False):return
    original_installer=simple_split_module.install_simple_split_panel_patch
    def install_with_20260914_lscm_route(pipeline_module,optimization_debug_module):
        original_installer(pipeline_module,optimization_debug_module); legacy_split_k2d=pipeline_module._optimize_k2d; legacy_flat_layout=pipeline_module._make_flat_tile_layout
        dated_solver=getattr(pipeline_module,"_onestring_20260914_lscm_k2d_solver",None) or legacy_split_k2d; dated_layout=getattr(pipeline_module,"_onestring_20260914_lscm_flat_layout",None) or legacy_flat_layout
        def k2d_dispatch(mesh_2d,mesh_3d,params,progress_callback=None):
            if not _active(params): return legacy_split_k2d(mesh_2d,mesh_3d,params,progress_callback=progress_callback)
            return dated_solver(mesh_2d,mesh_3d,params,progress_callback=progress_callback)
        def flat_layout_dispatch(mesh,params=None): return dated_layout(mesh,params) if params is not None and _active(params) else legacy_flat_layout(mesh,params)
        _wire_k2d(pipeline_module,k2d_dispatch); _wire_flat_layout(pipeline_module,flat_layout_dispatch)
    simple_split_module.install_simple_split_panel_patch=install_with_20260914_lscm_route; simple_split_module._onestring_20260914_lscm_k2d_bypass_installed=True


def install_optcuts_paper_k2d_20260914_patch(pipeline):
    install_deferred_hybrid_hook(pipeline)
    if getattr(pipeline,"_onestring_optcuts_paper_k2d_20260914_installed",False):return
    lscm_common_k2d=pipeline._optimize_k2d; lscm_common_flat_layout=pipeline._make_flat_tile_layout
    def optimize(mesh_2d,mesh_3d,params,progress_callback=None):
        if _active(params) or _paper_hybrid_active(params): return _paper_edge_only_k2d(pipeline,mesh_2d,mesh_3d,params,progress_callback)
        return lscm_common_k2d(mesh_2d,mesh_3d,params,progress_callback=progress_callback)
    def exact_lscm_flat_layout(mesh,params=None): return lscm_common_flat_layout(mesh,params)
    pipeline._onestring_20260914_lscm_k2d_solver=optimize; pipeline._onestring_20260914_lscm_flat_layout=exact_lscm_flat_layout; pipeline._onestring_20260914_lscm_flat_layout_base=lscm_common_flat_layout
    _wire_k2d(pipeline,optimize); _wire_flat_layout(pipeline,exact_lscm_flat_layout); _install_simple_split_bypass(); pipeline._onestring_optcuts_paper_k2d_20260914_installed=True

__all__=["install_optcuts_paper_k2d_20260914_patch","VERSION_ID"]
