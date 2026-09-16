"""2026-09-14 OptCuts variant with an exact LSCM K2D/flat-layout route.

For this dated OptCuts mode, S->Omega/M2D/K3D may differ from LSCM, but the
K2D numerical solve and the subsequent K2D -> independent flat-tile placement
must use the exact same common functions as ordinary LSCM.

In particular, this module MUST NOT split the K2D mesh into connected components
before calling ``_make_flat_tile_layout``.  Doing so changes the common layout
function's own branch conditions (for example its whole-mesh large-layout
threshold) and therefore is not LSCM-equivalent even if the same Python function
is called per component.

OptCuts seams remain represented only by the topology already present in
``mesh.faces``.  No 2026-09-14-specific panel-placement optimization, component
merge, parallel component solve, hard-SAT K2D replacement, all-tile SE(2) K2D
replacement, or post-K2D M2D-centroid realignment is applied here.
"""
from __future__ import annotations

import os
from typing import Any

from .lscm_latest_omega_hybrid_20260914_patch import install_deferred_hybrid_hook


VARIANT = "3"
VERSION_ID = "2026-09-14-paper-k2d-eq5-stage-separated"


def _active(params: Any) -> bool:
    explicit = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == VARIANT
    latest_ui = os.environ.get("ONESTRING_PAPER_K2D_20260914", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return bool(
        (explicit or latest_ui)
        and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"
    )


def _tag_lscm_equivalent_result(result: Any, report: Any) -> tuple[Any, Any]:
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics.update(
        {
            "version_id": VERSION_ID,
            "k2d_solver_equivalent_to_lscm": True,
            "k2d_common_solver_authoritative": True,
            "k2d_optcuts_specific_rigid_tile_replacement": False,
            "k2d_optcuts_hard_sat_stage_in_k2d": False,
            "k2d_optcuts_global_se2_stage_in_k2d": False,
            "k2d_split_panel_centroid_realign_disabled": True,
            "k2d_split_panel_post_eq5_translation_applied": False,
            "paper_k2d_shared_vertex_authoritative": True,
            "paper_k2d_independent_rigid_tile_layout_in_k2d": False,
            "paper_k2d_hinge_alignment_in_k2d": False,
            "paper_k2d_hinge_layout_deferred_to_section_4_4": True,
            "paper_k2d_absolute_m2d_layout_reinjected_after_optimization": False,
            "k2d_20260914_policy": (
                "Use the exact same whole-mesh common _optimize_k2d solver path as LSCM; "
                "OptCuts-specific rigid/hard/global K2D replacement is disabled."
            ),
        }
    )
    try:
        result.metrics.clear()
        result.metrics.update(metrics)
    except Exception:
        pass
    try:
        report.objective = (
            "Common LSCM/OneString K2D optimization from M2D and K3D; "
            "OptCuts-specific K2D replacement disabled."
        )
    except Exception:
        pass
    return result, report


def _tag_exact_lscm_flat_layout(layout: Any, mesh: Any, base_make_layout: Any) -> Any:
    metrics = dict(getattr(layout, "metrics", {}) or {})
    metrics.update(
        {
            "version_id": VERSION_ID,
            "k2d_flat_layout_solver_equivalent_to_lscm": True,
            "k2d_flat_layout_exact_same_captured_function": True,
            "k2d_flat_layout_whole_mesh_call": True,
            "k2d_flat_layout_componentwise_override": False,
            "k2d_flat_layout_parallel_override": False,
            "k2d_flat_layout_total_input_face_count": int(len(getattr(mesh, "faces", []))),
            "k2d_flat_layout_lscm_function_module": str(getattr(base_make_layout, "__module__", "")),
            "k2d_flat_layout_lscm_function_name": str(getattr(base_make_layout, "__name__", "")),
            "k2d_flat_layout_20260914_policy": (
                "Call the exact captured LSCM _make_flat_tile_layout once on the complete K2D mesh. "
                "OptCuts seam topology is carried only by mesh.faces; no dated panel-placement override."
            ),
        }
    )
    try:
        layout.metrics.clear()
        layout.metrics.update(metrics)
    except Exception:
        pass
    return layout


def _wire_k2d(pipeline: Any, fn: Any) -> None:
    pipeline._optimize_k2d = fn
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k2d = fn
    for build_fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k2d"] = fn


def _wire_flat_layout(pipeline: Any, fn: Any) -> None:
    pipeline._make_flat_tile_layout = fn
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._make_flat_tile_layout = fn
    for build_fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_make_flat_tile_layout"] = fn


def _install_simple_split_bypass() -> None:
    """Keep 09-14 on the captured LSCM K2D and flat-layout functions.

    Simple Split installs wrappers later in app_split_panels.  For the dated
    active mode we bypass those outer K2D/layout wrappers and route directly to
    the functions captured before OptCuts-only replacements are stacked.
    """
    try:
        from . import simple_split_panel_patch as simple_split_module
    except Exception:
        return

    if getattr(simple_split_module, "_onestring_20260914_lscm_k2d_bypass_installed", False):
        return

    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_with_20260914_lscm_route(pipeline_module: Any, optimization_debug_module: Any) -> None:
        original_installer(pipeline_module, optimization_debug_module)
        legacy_split_k2d = pipeline_module._optimize_k2d
        legacy_flat_layout = pipeline_module._make_flat_tile_layout

        dated_solver = getattr(pipeline_module, "_onestring_20260914_lscm_k2d_solver", None)
        dated_layout = getattr(pipeline_module, "_onestring_20260914_lscm_flat_layout", None)
        if not callable(dated_solver):
            dated_solver = legacy_split_k2d
        if not callable(dated_layout):
            dated_layout = legacy_flat_layout

        def k2d_dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return legacy_split_k2d(
                    mesh_2d,
                    mesh_3d,
                    params,
                    progress_callback=progress_callback,
                )

            result, report = dated_solver(
                mesh_2d,
                mesh_3d,
                params,
                progress_callback=progress_callback,
            )
            result, report = _tag_lscm_equivalent_result(result, report)

            copy_attrs = getattr(simple_split_module, "_copy_attrs", None)
            if callable(copy_attrs):
                try:
                    copy_attrs(mesh_2d, result)
                except Exception:
                    pass

            print(
                "[2026-09-14-K2D-LSCM-EQUIVALENT] "
                "common whole-mesh LSCM K2D solver used; OptCuts rigid/hard/global K2D wrappers bypassed; "
                "post-K2D M2D-centroid realignment disabled"
            )
            return result, report

        def flat_layout_dispatch(mesh: Any, params: Any = None):
            if params is not None and _active(params):
                return dated_layout(mesh, params)
            return legacy_flat_layout(mesh, params)

        _wire_k2d(pipeline_module, k2d_dispatch)
        _wire_flat_layout(pipeline_module, flat_layout_dispatch)

    simple_split_module.install_simple_split_panel_patch = install_with_20260914_lscm_route
    simple_split_module._onestring_20260914_lscm_k2d_bypass_installed = True


def install_optcuts_paper_k2d_20260914_patch(pipeline: Any) -> None:
    # Register the clean diagnostic hybrid while the LSCM downstream functions
    # are still available, before app_optcuts stacks its OptCuts-specific routes.
    install_deferred_hybrid_hook(pipeline)

    if getattr(pipeline, "_onestring_optcuts_paper_k2d_20260914_installed", False):
        return

    # Capture the exact common functions that ordinary LSCM uses before the
    # OptCuts-only wrapper stack is installed later by app_optcuts/app_split_panels.
    lscm_common_k2d = pipeline._optimize_k2d
    lscm_common_flat_layout = pipeline._make_flat_tile_layout

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        result, report = lscm_common_k2d(
            mesh_2d,
            mesh_3d,
            params,
            progress_callback=progress_callback,
        )
        if not _active(params):
            return result, report
        result, report = _tag_lscm_equivalent_result(result, report)
        print(
            "[2026-09-14-K2D-BASE] used exact captured LSCM _optimize_k2d once on the whole mesh; "
            "no 2026-09-14 extra K2D refinement"
        )
        return result, report

    def exact_lscm_flat_layout(mesh: Any, params: Any = None):
        # Deliberately one call on the original complete mesh.  Do not extract
        # connected components here: doing so changes the common LSCM function's
        # tile-count-dependent branch selection and is therefore not equivalent.
        layout = lscm_common_flat_layout(mesh, params)
        if params is None or not _active(params):
            return layout
        layout = _tag_exact_lscm_flat_layout(layout, mesh, lscm_common_flat_layout)
        fast_path = bool(getattr(layout, "metrics", {}).get("k2d_independent_fast_large_layout", False))
        threshold = getattr(layout, "metrics", {}).get("k2d_independent_fast_tile_threshold", "n/a")
        print(
            "[2026-09-14-FLAT-LAYOUT-LSCM-EXACT] "
            f"faces={len(getattr(mesh, 'faces', []))} whole_mesh_call=True "
            f"componentwise_override=False parallel_override=False "
            f"common_fast_path={fast_path} common_fast_threshold={threshold}"
        )
        return layout

    pipeline._onestring_20260914_lscm_k2d_solver = optimize
    pipeline._onestring_20260914_lscm_flat_layout = exact_lscm_flat_layout
    pipeline._onestring_20260914_lscm_flat_layout_base = lscm_common_flat_layout

    _wire_k2d(pipeline, optimize)
    _wire_flat_layout(pipeline, exact_lscm_flat_layout)
    _install_simple_split_bypass()
    pipeline._onestring_optcuts_paper_k2d_20260914_installed = True


__all__ = [
    "install_optcuts_paper_k2d_20260914_patch",
    "VERSION_ID",
]
