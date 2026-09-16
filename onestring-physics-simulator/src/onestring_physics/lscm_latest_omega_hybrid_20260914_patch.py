"""Latest Omega/K3D hybrid with paper Sec. 4.3 Eq. (5) K2D.

Unlike the historical captured-LSCM and vectorized-EEdge routes, this mode keeps
EEdge, ECollision and EFab in one K2D local/global solve.  Hinge optimization
remains a later Sec. 4.4 concern.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import os
from typing import Any

from .paper_eq5_k2d_solver import optimize_paper_eq5

MODE = "lscm_latest_omega_hard_k3d"
VERSION_ID = "2026-09-17-paper-eq5-unified-k2d"
VERSION_LABEL = "2026-09-17 — paper Eq.5 K2D + latest Ω/K3D"
OMEGA_LABEL = "2026-09-17 | paper Eq.5 K2D + latest Ω/K3D"
VERSION_DESCRIPTION = (
    "M2D→K2Dを原論文Sec.4.3 Eq.(5)のEEdge+ECollision+EFab統合local/global solveで計算。"
    "EFabはSupplement Appendix Aのgap-angle projection。hingeはSec.4.4へ分離。"
)


def _active(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == MODE


def _clone_params(params: Any, **updates: Any) -> Any:
    try:
        return replace(params, **updates)
    except Exception:
        out = copy.copy(params)
        for key, value in updates.items():
            try:
                setattr(out, key, value)
            except Exception:
                try:
                    object.__setattr__(out, key, value)
                except Exception:
                    pass
        return out


def _wire(pipeline: Any, name: str, fn: Any) -> None:
    setattr(pipeline, name, fn)
    original = getattr(pipeline, "_original", None)
    if original is not None:
        setattr(original, name, fn)
    for build_fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb[name] = fn


def _install_selector_patch() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_lscm_latest_omega_hybrid_selector_installed", False):
        return
    original_selectbox = st.selectbox

    def selectbox_with_hybrid(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label == "version":
            if len(args) >= 2:
                options = list(args[1])
                if not any(isinstance(v, dict) and v.get("id") == VERSION_ID for v in options):
                    options.append({"id": VERSION_ID, "label": VERSION_LABEL, "description": VERSION_DESCRIPTION})
                kwargs = {**kwargs, "index": len(options) - 1}
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if not any(isinstance(v, dict) and v.get("id") == VERSION_ID for v in options):
                    options.append({"id": VERSION_ID, "label": VERSION_LABEL, "description": VERSION_DESCRIPTION})
                kwargs = {**kwargs, "options": options, "index": len(options) - 1}
        if label == "Omega parameterization mode":
            if len(args) >= 2:
                options = list(args[1])
                if OMEGA_LABEL not in options:
                    options.append(OMEGA_LABEL)
                kwargs = {**kwargs, "index": options.index(OMEGA_LABEL)}
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if OMEGA_LABEL not in options:
                    options.append(OMEGA_LABEL)
                kwargs = {**kwargs, "options": options, "index": options.index(OMEGA_LABEL)}
        selected = original_selectbox(*args, **kwargs)
        if label == "Omega parameterization mode" and selected == OMEGA_LABEL:
            try:
                st.caption("Paper Sec.4.3 Eq.(5): EEdge + ECollision + EFab in one K2D solve; Appendix-A angle projection.")
            except Exception:
                pass
            return MODE
        return selected

    st.selectbox = selectbox_with_hybrid
    st._onestring_lscm_latest_omega_hybrid_selector_installed = True


def install_lscm_latest_omega_hybrid_patch(
    pipeline: Any,
    *,
    lscm_build_m2d: Any,
    lscm_optimize_k2d: Any,
    lscm_make_flat_tile_layout: Any,
) -> None:
    if getattr(pipeline, "_onestring_lscm_latest_omega_hybrid_installed", False):
        return

    latest_parameterization = pipeline._build_surface_parameterization
    latest_k3d = pipeline._optimize_k3d

    def make_dispatches(*, fallback_parameterization: Any, fallback_m2d: Any, fallback_k3d: Any, fallback_k2d: Any, fallback_flat_layout: Any):
        def parameterization_dispatch(surface: Any, target: Any, grid: Any, params: Any):
            if not _active(params):
                return fallback_parameterization(surface, target, grid, params)
            latest_params = _clone_params(params, omega_parameterization_mode="optcuts_test")
            result = latest_parameterization(surface, target, grid, latest_params)
            try:
                result.method = MODE
                result.metrics.update({
                    "version_id": VERSION_ID,
                    "hybrid_stage_s_to_omega": "current latest OptCuts-test Omega",
                    "hybrid_stage_m2d": "common M2D topology",
                    "hybrid_stage_k3d": "current latest OptCuts-test2 K3D stack",
                    "hybrid_stage_k2d": "paper Sec.4.3 Eq.5 unified EEdge+ECollision+EFab",
                })
            except Exception:
                pass
            return result

        def m2d_dispatch(grid: Any, domain: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_m2d(grid, domain, params)
            return lscm_build_m2d(grid, domain, params)

        def k3d_dispatch(target: Any, mesh: Any, parameterization: Any, params: Any):
            if not _active(params):
                return fallback_k3d(target, mesh, parameterization, params)
            previous_variant = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT")
            try:
                os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "2"
                latest_params = _clone_params(params, omega_parameterization_mode="optcuts_test")
                return latest_k3d(target, mesh, parameterization, latest_params)
            finally:
                if previous_variant is None:
                    os.environ.pop("ONESTRING_OPTCUTS_TEST_VARIANT", None)
                else:
                    os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = previous_variant

        def k2d_dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return fallback_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
            return optimize_paper_eq5(
                mesh_2d,
                mesh_3d,
                params,
                progress_callback=progress_callback,
                pipeline=pipeline,
            )

        def flat_layout_dispatch(mesh: Any, params: Any = None):
            # This function belongs to the following K2D->T2D construction.  Eq.5
            # has already finished here; do not use it to claim collision/EFab
            # satisfaction in K2D.
            return fallback_flat_layout(mesh, params)

        return {
            "_build_surface_parameterization": parameterization_dispatch,
            "_build_m2d": m2d_dispatch,
            "_optimize_k3d": k3d_dispatch,
            "_optimize_k2d": k2d_dispatch,
            "_make_flat_tile_layout": flat_layout_dispatch,
        }

    def install_routes(target_pipeline: Any) -> None:
        fallbacks = {
            "fallback_parameterization": target_pipeline._build_surface_parameterization,
            "fallback_m2d": target_pipeline._build_m2d,
            "fallback_k3d": target_pipeline._optimize_k3d,
            "fallback_k2d": target_pipeline._optimize_k2d,
            "fallback_flat_layout": target_pipeline._make_flat_tile_layout,
        }
        for name, fn in make_dispatches(**fallbacks).items():
            _wire(target_pipeline, name, fn)

    install_routes(pipeline)

    try:
        from . import simple_split_panel_patch as simple_split_module
        if not getattr(simple_split_module, "_onestring_lscm_hybrid_rewire_installed", False):
            original_installer = simple_split_module.install_simple_split_panel_patch

            def install_then_rewire(pipeline_module: Any, optimization_debug_module: Any) -> None:
                original_installer(pipeline_module, optimization_debug_module)
                install_routes(pipeline_module)
                print("[PAPER-EQ5-ROUTE] unified Eq.5 routing reinstalled after Simple Split")

            simple_split_module.install_simple_split_panel_patch = install_then_rewire
            simple_split_module._onestring_lscm_hybrid_rewire_installed = True
    except Exception:
        pass

    pipeline._onestring_lscm_latest_omega_hybrid_installed = True


def install_deferred_hybrid_hook(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_lscm_hybrid_deferred_hook_installed", False):
        return
    lscm_build_m2d = pipeline._build_m2d
    lscm_optimize_k2d = pipeline._optimize_k2d
    lscm_make_flat_tile_layout = pipeline._make_flat_tile_layout
    _install_selector_patch()
    try:
        from . import optcuts_test2_acceleration_patch as acceleration_module
    except Exception:
        return
    original_installer = acceleration_module.install_optcuts_test2_acceleration_patch

    def install_acceleration_then_hybrid(target_pipeline: Any) -> None:
        original_installer(target_pipeline)
        install_lscm_latest_omega_hybrid_patch(
            target_pipeline,
            lscm_build_m2d=lscm_build_m2d,
            lscm_optimize_k2d=lscm_optimize_k2d,
            lscm_make_flat_tile_layout=lscm_make_flat_tile_layout,
        )
        print("[PAPER-EQ5-INSTALL] unified EEdge+ECollision+EFab K2D installed")

    acceleration_module.install_optcuts_test2_acceleration_patch = install_acceleration_then_hybrid
    pipeline._onestring_lscm_hybrid_deferred_hook_installed = True


__all__ = [
    "MODE",
    "OMEGA_LABEL",
    "VERSION_ID",
    "VERSION_LABEL",
    "install_deferred_hybrid_hook",
    "install_lscm_latest_omega_hybrid_patch",
]
