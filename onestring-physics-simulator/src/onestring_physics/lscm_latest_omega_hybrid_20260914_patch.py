"""2026-09-14 diagnostic hybrid: LSCM baseline + latest Omega + latest K3D planarity.

This variant starts from the ordinary LSCM pipeline and replaces only:

1. S -> Omega with the current latest OptCuts-test Omega route.
2. K3D with the current latest OptCuts-test2 K3D/hard-planarity route.

M2D, K2D, and K2D -> independent flat-tile placement are routed directly to
function objects captured before app_optcuts installs OptCuts-specific wrappers.
The install is deliberately deferred until app_optcuts finishes stacking its
current OptCuts/K3D wrappers, so the two replacement stages really are the
current latest implementations.
"""
from __future__ import annotations

from dataclasses import replace
import os
from typing import Any


MODE = "lscm_latest_omega_hard_k3d"
VERSION_ID = "2026-09-14-lscm-baseline-latest-omega-k3d-planarity"
VERSION_LABEL = "2026-09-14 — LSCM baseline + latest Ω + latest K3D planarity"
OMEGA_LABEL = "2026-09-14 | LSCM baseline + latest Ω + latest K3D planarity"
VERSION_DESCRIPTION = (
    "比較用のクリーンなhybrid。LSCMを土台にして、S→Ωだけ現在最新版OptCuts、"
    "K3Dだけ現在最新版のhard-planarity stackへ置換する。M2D/K2D/flat panel placementは"
    "OptCuts専用wrapperを通さず、起動時に保存したLSCM関数を直接使用する。"
)


def _active(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == MODE


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
                    options.append(
                        {
                            "id": VERSION_ID,
                            "label": VERSION_LABEL,
                            "description": VERSION_DESCRIPTION,
                        }
                    )
                kwargs = {**kwargs, "index": len(options) - 1}
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if not any(isinstance(v, dict) and v.get("id") == VERSION_ID for v in options):
                    options.append(
                        {
                            "id": VERSION_ID,
                            "label": VERSION_LABEL,
                            "description": VERSION_DESCRIPTION,
                        }
                    )
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
                st.caption(
                    "LSCM baseline hybrid: only S→Ω and K3D hard-planarity are replaced; "
                    "M2D/K2D/flat panel placement use the captured LSCM implementation."
                )
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

    # Capture the completed current OptCuts/K3D stack.  Only these two routes are
    # intentionally borrowed by the hybrid.
    latest_parameterization = pipeline._build_surface_parameterization
    latest_k3d = pipeline._optimize_k3d

    def make_dispatches(
        *,
        fallback_parameterization: Any,
        fallback_m2d: Any,
        fallback_k3d: Any,
        fallback_k2d: Any,
        fallback_flat_layout: Any,
    ) -> dict[str, Any]:
        def parameterization_dispatch(surface: Any, target: Any, grid: Any, params: Any):
            if not _active(params):
                return fallback_parameterization(surface, target, grid, params)
            latest_params = replace(params, omega_parameterization_mode="optcuts_test")
            result = latest_parameterization(surface, target, grid, latest_params)
            try:
                result.method = MODE
                result.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "hybrid_lscm_baseline": True,
                        "hybrid_stage_s_to_omega": "current latest OptCuts-test Omega",
                        "hybrid_stage_m2d": "captured ordinary LSCM M2D",
                        "hybrid_stage_k3d": "current latest OptCuts-test2 K3D/hard-planarity stack",
                        "hybrid_stage_k2d": "captured ordinary LSCM K2D",
                        "hybrid_stage_flat_layout": "captured ordinary LSCM whole-mesh flat-tile placement",
                    }
                )
            except Exception:
                pass
            print("[LSCM-HYBRID-OMEGA] latest OptCuts Omega used")
            return result

        def m2d_dispatch(grid: Any, domain: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_m2d(grid, domain, params)
            result = lscm_build_m2d(grid, domain, params)
            try:
                result.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "hybrid_m2d_exact_lscm_function": True,
                        "hybrid_optcuts_m2d_wrappers_bypassed": True,
                    }
                )
            except Exception:
                pass
            print(
                "[LSCM-HYBRID-M2D] exact captured LSCM M2D used "
                f"faces={len(getattr(result, 'faces', []))}"
            )
            return result

        def k3d_dispatch(target: Any, mesh: Any, parameterization: Any, params: Any):
            if not _active(params):
                return fallback_k3d(target, mesh, parameterization, params)

            previous_variant = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT")
            try:
                # Match the current latest visible K3D numerical route.
                os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "2"
                latest_params = replace(params, omega_parameterization_mode="optcuts_test")
                result, report = latest_k3d(target, mesh, parameterization, latest_params)
            finally:
                if previous_variant is None:
                    os.environ.pop("ONESTRING_OPTCUTS_TEST_VARIANT", None)
                else:
                    os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = previous_variant
            try:
                result.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "hybrid_k3d_latest_planarity_stack": True,
                        "hybrid_k3d_source": "current latest OptCuts-test2 K3D route",
                    }
                )
            except Exception:
                pass
            print("[LSCM-HYBRID-K3D] current latest K3D/hard-planarity route used")
            return result, report

        def k2d_dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return fallback_k2d(
                    mesh_2d,
                    mesh_3d,
                    params,
                    progress_callback=progress_callback,
                )
            result, report = lscm_optimize_k2d(
                mesh_2d,
                mesh_3d,
                params,
                progress_callback=progress_callback,
            )
            try:
                result.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "hybrid_k2d_exact_lscm_function": True,
                        "hybrid_optcuts_k2d_wrappers_bypassed": True,
                    }
                )
            except Exception:
                pass
            print("[LSCM-HYBRID-K2D] exact captured LSCM _optimize_k2d used")
            return result, report

        def flat_layout_dispatch(mesh: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_flat_layout(mesh, params)
            layout = lscm_make_flat_tile_layout(mesh, params)
            try:
                layout.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "hybrid_flat_layout_exact_lscm_function": True,
                        "hybrid_flat_layout_whole_mesh_call": True,
                        "hybrid_componentwise_layout": False,
                        "hybrid_parallel_component_layout": False,
                        "hybrid_flat_layout_input_faces": int(len(getattr(mesh, "faces", []))),
                    }
                )
            except Exception:
                pass
            print(
                "[LSCM-HYBRID-FLAT-LAYOUT] exact captured LSCM whole-mesh panel placement used "
                f"faces={len(getattr(mesh, 'faces', []))}"
            )
            return layout

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
        dispatches = make_dispatches(**fallbacks)
        for name, fn in dispatches.items():
            _wire(target_pipeline, name, fn)

    install_routes(pipeline)

    # app_split_panels installs Simple Split after app_optcuts.  Preserve that
    # installer for every other mode, then wrap its resulting functions with a
    # fresh hybrid dispatcher whose non-hybrid fallbacks are the just-installed
    # Simple Split routes.
    try:
        from . import simple_split_panel_patch as simple_split_module

        if not getattr(simple_split_module, "_onestring_lscm_hybrid_rewire_installed", False):
            original_installer = simple_split_module.install_simple_split_panel_patch

            def install_then_rewire(pipeline_module: Any, optimization_debug_module: Any) -> None:
                original_installer(pipeline_module, optimization_debug_module)
                install_routes(pipeline_module)
                print("[LSCM-HYBRID-ROUTE] hybrid routing reinstalled after Simple Split")

            simple_split_module.install_simple_split_panel_patch = install_then_rewire
            simple_split_module._onestring_lscm_hybrid_rewire_installed = True
    except Exception:
        pass

    pipeline._onestring_lscm_latest_omega_hybrid_installed = True


def install_deferred_hybrid_hook(pipeline: Any) -> None:
    """Capture LSCM functions now and install the hybrid after OptCuts setup."""
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
        print("[LSCM-HYBRID-INSTALL] latest OptCuts/K3D stack captured; LSCM downstream functions fixed")

    acceleration_module.install_optcuts_test2_acceleration_patch = install_acceleration_then_hybrid
    pipeline._onestring_lscm_hybrid_deferred_hook_installed = True


__all__ = [
    "MODE",
    "VERSION_ID",
    "VERSION_LABEL",
    "OMEGA_LABEL",
    "VERSION_DESCRIPTION",
    "install_deferred_hybrid_hook",
    "install_lscm_latest_omega_hybrid_patch",
]
