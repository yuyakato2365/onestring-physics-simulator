"""Latest Omega/K3D hybrid with explicit auxetic paper Sec. 4.3 Eq. (5) K2D."""
from __future__ import annotations

import copy
from dataclasses import replace
import os
from typing import Any

from .paper_eq5_k2d_solver import optimize_paper_eq5

MODE = "lscm_latest_omega_hard_k3d"
VERSION_ID = "2026-09-17-paper-eq5-unified-k2d"
VERSION_LABEL = "2026-09-17 — auxetic paper Eq.5 K2D + latest Ω/K3D"
OMEGA_LABEL = "2026-09-17 | auxetic paper Eq.5 K2D + latest Ω/K3D"
VERSION_DESCRIPTION = (
    "M2Dに隣接タイルごとのcorner jointをencodeし、切れ目を保持してK2Dを最適化。"
    "EEdge・EFabの射影距離とSAT衝突項を解析勾配で最小化。論文の数値解法と衝突離散化には相違あり。"
)


def _active(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == MODE


def _clone_params(params: Any, **updates: Any) -> Any:
    try:
        return replace(params, **updates)
    except Exception:
        out = copy.copy(params)
        for key, value in updates.items():
            try: setattr(out, key, value)
            except Exception:
                try: object.__setattr__(out, key, value)
                except Exception: pass
        return out


def _wire(pipeline: Any, name: str, fn: Any) -> None:
    setattr(pipeline, name, fn)
    original = getattr(pipeline, "_original", None)
    if original is not None: setattr(original, name, fn)
    for build_fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(build_fn,"__globals__",None)
        if isinstance(glb,dict): glb[name]=fn


def _env_float(name: str, default: float) -> float:
    try: return float(os.environ.get(name,str(default)))
    except Exception: return float(default)


def _env_int(name: str, default: int) -> int:
    try: return int(float(os.environ.get(name,str(default))))
    except Exception: return int(default)


def _render_eq5_controls(st: Any) -> None:
    st.markdown("### K2D Flat Configuration")
    st.markdown(
        "<div style='padding:14px 16px;border:1px solid rgba(70,110,160,.25);border-radius:14px;background:rgba(240,246,253,.72);margin:6px 0 12px'>"
        "<div style='display:flex;justify-content:space-between'><b>Flat objective</b><span style='opacity:.65'>Paper Sec. 4.3 · Eq. (5)</span></div>"
        "<div style='font-family:Georgia,serif;font-size:20px;margin:10px 0'>E<sub>Flat</sub>(v) = ω₁E<sub>Edge</sub> + ω₂E<sub>Collision</sub> + ω₃E<sub>Fab</sub></div>"
        "<div style='font-size:12px;opacity:.72'>M₂D → pairwise corner linkage → K₂D。EFabはタイル間の開口角。衝突はSAT近似、数値解法はL-BFGS。</div>"
        "</div>", unsafe_allow_html=True)
    w1=st.number_input("ω1 / EEdge",min_value=0.0,max_value=10000.0,value=max(0.0,_env_float("ONESTRING_EQ5_W_EDGE",1.0)),step=0.1,format="%.4f",key="onestring_eq5_w_edge",help="各tile edgeを対応K3D edge lengthへ合わせる項。")
    w2=st.number_input("ω2 / ECollision",min_value=0.0,max_value=10000.0,value=max(0.0,_env_float("ONESTRING_EQ5_W_COLLISION",1.0)),step=0.1,format="%.4f",key="onestring_eq5_w_collision",help="独立tile同士の2D overlapを避ける項。")
    w3=st.number_input("ω3 / EFab",min_value=0.0,max_value=10000.0,value=max(0.0,_env_float("ONESTRING_EQ5_W_FAB",0.001)),step=0.001,format="%.6f",key="onestring_eq5_w_fab",help="同じhinge周りで隣接する異なるtileのray間gap angleを制御する項。")
    theta=st.number_input("θmin / EFab gap angle [deg]",min_value=0.0,max_value=90.0,value=min(90.0,max(0.0,_env_float("ONESTRING_EQ5_THETA_MIN_DEG",5.0))),step=1.0,format="%.2f",key="onestring_eq5_theta_min_deg")
    iterations=st.number_input("Eq.5 K2D iterations",min_value=1,max_value=10000,value=max(1,_env_int("ONESTRING_EQ5_ITERATIONS",240)),step=20,key="onestring_eq5_iterations")
    os.environ["ONESTRING_EQ5_W_EDGE"]=str(float(w1)); os.environ["ONESTRING_EQ5_W_COLLISION"]=str(float(w2)); os.environ["ONESTRING_EQ5_W_FAB"]=str(float(w3)); os.environ["ONESTRING_EQ5_THETA_MIN_DEG"]=str(float(theta)); os.environ["ONESTRING_EQ5_ITERATIONS"]=str(int(iterations))
    st.caption("各ヒンジは2枚のタイルだけで共有します。全反復の目的関数を評価し、制約が残る場合は未充足と表示します。")


def _install_selector_patch() -> None:
    try: import streamlit as st
    except Exception: return
    if getattr(st,"_onestring_lscm_latest_omega_hybrid_selector_installed",False): return
    original_selectbox=st.selectbox
    def selectbox_with_hybrid(*args: Any, **kwargs: Any) -> Any:
        label=args[0] if args else kwargs.get("label")
        if label=="version":
            if len(args)>=2:
                options=list(args[1])
                if not any(isinstance(v,dict) and v.get("id")==VERSION_ID for v in options): options.append({"id":VERSION_ID,"label":VERSION_LABEL,"description":VERSION_DESCRIPTION})
                kwargs={**kwargs,"index":len(options)-1}; args=(args[0],options,*args[2:])
            elif "options" in kwargs:
                options=list(kwargs["options"])
                if not any(isinstance(v,dict) and v.get("id")==VERSION_ID for v in options): options.append({"id":VERSION_ID,"label":VERSION_LABEL,"description":VERSION_DESCRIPTION})
                kwargs={**kwargs,"options":options,"index":len(options)-1}
        if label=="Omega parameterization mode":
            if len(args)>=2:
                options=list(args[1])
                if OMEGA_LABEL not in options: options.append(OMEGA_LABEL)
                kwargs={**kwargs,"index":options.index(OMEGA_LABEL)}; args=(args[0],options,*args[2:])
            elif "options" in kwargs:
                options=list(kwargs["options"])
                if OMEGA_LABEL not in options: options.append(OMEGA_LABEL)
                kwargs={**kwargs,"options":options,"index":options.index(OMEGA_LABEL)}
        selected=original_selectbox(*args,**kwargs)
        if label=="Omega parameterization mode" and selected==OMEGA_LABEL:
            _render_eq5_controls(st)
            return MODE
        return selected
    st.selectbox=selectbox_with_hybrid; st._onestring_lscm_latest_omega_hybrid_selector_installed=True


def install_lscm_latest_omega_hybrid_patch(pipeline: Any, *, lscm_build_m2d: Any, lscm_optimize_k2d: Any, lscm_make_flat_tile_layout: Any) -> None:
    if getattr(pipeline,"_onestring_lscm_latest_omega_hybrid_installed",False): return
    latest_parameterization=pipeline._build_surface_parameterization; latest_k3d=pipeline._optimize_k3d
    def make_dispatches(*,fallback_parameterization:Any,fallback_m2d:Any,fallback_k3d:Any,fallback_k2d:Any,fallback_flat_layout:Any):
        def parameterization_dispatch(surface,target,grid,params):
            if not _active(params): return fallback_parameterization(surface,target,grid,params)
            latest_params=_clone_params(params,omega_parameterization_mode="optcuts_test"); result=latest_parameterization(surface,target,grid,latest_params)
            try:
                result.method=MODE; result.metrics.update({"version_id":VERSION_ID,"hybrid_stage_s_to_omega":"current latest OptCuts-test Omega","hybrid_stage_m2d":"common shared-vertex M2D","hybrid_stage_k3d":"current latest OptCuts-test2 K3D stack","hybrid_stage_k2d":"pairwise corner linkage + directly minimized Eq.5 terms"})
            except Exception: pass
            return result
        def m2d_dispatch(grid,domain,params=None):
            if params is None or not _active(params): return fallback_m2d(grid,domain,params)
            return lscm_build_m2d(grid,domain,params)
        def k3d_dispatch(target,mesh,parameterization,params):
            if not _active(params): return fallback_k3d(target,mesh,parameterization,params)
            previous=os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT")
            try:
                os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"]="2"; return latest_k3d(target,mesh,parameterization,_clone_params(params,omega_parameterization_mode="optcuts_test"))
            finally:
                if previous is None: os.environ.pop("ONESTRING_OPTCUTS_TEST_VARIANT",None)
                else: os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"]=previous
        def k2d_dispatch(mesh_2d,mesh_3d,params,progress_callback=None):
            if not _active(params): return fallback_k2d(mesh_2d,mesh_3d,params,progress_callback=progress_callback)
            result, report = optimize_paper_eq5(mesh_2d,mesh_3d,params,progress_callback=progress_callback,pipeline=pipeline)
            from .eq5_iteration_history_view import render_completed_k2d
            render_completed_k2d(result.eq5_history)
            return result, report
        def flat_layout_dispatch(mesh,params=None): return fallback_flat_layout(mesh,params)
        return {"_build_surface_parameterization":parameterization_dispatch,"_build_m2d":m2d_dispatch,"_optimize_k3d":k3d_dispatch,"_optimize_k2d":k2d_dispatch,"_make_flat_tile_layout":flat_layout_dispatch}
    def install_routes(target_pipeline):
        fallbacks={"fallback_parameterization":target_pipeline._build_surface_parameterization,"fallback_m2d":target_pipeline._build_m2d,"fallback_k3d":target_pipeline._optimize_k3d,"fallback_k2d":target_pipeline._optimize_k2d,"fallback_flat_layout":target_pipeline._make_flat_tile_layout}
        for name,fn in make_dispatches(**fallbacks).items(): _wire(target_pipeline,name,fn)
    install_routes(pipeline)
    try:
        from . import simple_split_panel_patch as simple_split_module
        if not getattr(simple_split_module,"_onestring_lscm_hybrid_rewire_installed",False):
            original_installer=simple_split_module.install_simple_split_panel_patch
            def install_then_rewire(pipeline_module,optimization_debug_module):
                original_installer(pipeline_module,optimization_debug_module); install_routes(pipeline_module); print("[PAPER-EQ5-ROUTE] auxetic Eq.5 routing reinstalled after Simple Split")
            simple_split_module.install_simple_split_panel_patch=install_then_rewire; simple_split_module._onestring_lscm_hybrid_rewire_installed=True
    except Exception: pass
    pipeline._onestring_lscm_latest_omega_hybrid_installed=True


def install_deferred_hybrid_hook(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_lscm_hybrid_deferred_hook_installed",False): return
    lscm_build_m2d=pipeline._build_m2d; lscm_optimize_k2d=pipeline._optimize_k2d; lscm_make_flat_tile_layout=pipeline._make_flat_tile_layout; _install_selector_patch()
    try: from . import optcuts_test2_acceleration_patch as acceleration_module
    except Exception: return
    original_installer=acceleration_module.install_optcuts_test2_acceleration_patch
    def install_acceleration_then_hybrid(target_pipeline):
        original_installer(target_pipeline); install_lscm_latest_omega_hybrid_patch(target_pipeline,lscm_build_m2d=lscm_build_m2d,lscm_optimize_k2d=lscm_optimize_k2d,lscm_make_flat_tile_layout=lscm_make_flat_tile_layout); print("[PAPER-EQ5-INSTALL] explicit auxetic Eq.5 K2D installed")
    acceleration_module.install_optcuts_test2_acceleration_patch=install_acceleration_then_hybrid; pipeline._onestring_lscm_hybrid_deferred_hook_installed=True


__all__=["MODE","OMEGA_LABEL","VERSION_ID","VERSION_LABEL","install_deferred_hybrid_hook","install_lscm_latest_omega_hybrid_patch"]
