"""2026-09-14 OptCuts variant with an LSCM-equivalent K2D stage.

The dated OptCuts mode is allowed to differ from LSCM before K2D (Omega, seams,
M2D topology, and K3D), but K2D itself must use the same common OneString K2D
solver that the ordinary LSCM path uses.  In particular, OptCuts-specific rigid
per-tile K2D replacement, hard-SAT K2D feasibility, all-tile SE(2) hinge solves,
and post-K2D M2D-centroid re-alignment are not part of this dated K2D stage.

The normal common pipeline may still convert the resulting K2D mesh into the
standard independent flat-tile representation afterwards, exactly as it does for
LSCM.  Physical hinge/T2D processing therefore remains downstream rather than
silently replacing the authoritative K2D mesh.
"""
from __future__ import annotations

import os
from typing import Any


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
            "paper_k2d_relative_layout_preserved_after_split": True,
            "paper_k2d_absolute_m2d_layout_reinjected_after_optimization": False,
            "k2d_20260914_policy": (
                "Use the exact same common _optimize_k2d solver path as LSCM; "
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
            "OptCuts-specific rigid-tile K2D replacement disabled."
        )
    except Exception:
        pass
    return result, report


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


def _install_simple_split_bypass() -> None:
    """Ensure Simple Split cannot put an OptCuts-only K2D wrapper back on 09-14."""
    try:
        from . import simple_split_panel_patch as simple_split_module
    except Exception:
        return

    if getattr(simple_split_module, "_onestring_20260914_lscm_k2d_bypass_installed", False):
        return

    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_with_20260914_lscm_k2d(pipeline_module: Any, optimization_debug_module: Any) -> None:
        original_installer(pipeline_module, optimization_debug_module)
        legacy_split_k2d = pipeline_module._optimize_k2d
        dated_solver = getattr(pipeline_module, "_onestring_20260914_lscm_k2d_solver", None)
        if not callable(dated_solver):
            dated_solver = legacy_split_k2d

        def dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return legacy_split_k2d(
                    mesh_2d,
                    mesh_3d,
                    params,
                    progress_callback=progress_callback,
                )

            # Critical: call the stored dated solver directly.  Do not call the
            # current outer OptCuts-test K2D wrapper stack, because that stack can
            # contain rigid-tile, hard-SAT, and all-tile-SE2 replacement stages.
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
                "common LSCM K2D solver used; OptCuts rigid/hard/global K2D wrappers bypassed; "
                "post-K2D M2D-centroid realignment disabled"
            )
            return result, report

        _wire_k2d(pipeline_module, dispatch)

    simple_split_module.install_simple_split_panel_patch = install_with_20260914_lscm_k2d
    simple_split_module._onestring_20260914_lscm_k2d_bypass_installed = True


def install_optcuts_paper_k2d_20260914_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_paper_k2d_20260914_installed", False):
        return

    # Capture the common K2D function NOW, before app_optcuts installs its
    # OptCuts-test-specific K2D wrappers.  This is the same function used by the
    # ordinary LSCM mode at this point in the common pipeline.
    lscm_common_k2d = pipeline._optimize_k2d

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        result, report = lscm_common_k2d(
            mesh_2d,
            mesh_3d,
            params,
            progress_callback=progress_callback,
        )
        if not _active(params):
            return result, report

        # Do not add a second dated refinement here.  The point of this mode is a
        # controlled comparison: only the upstream parameterization/M2D/K3D may
        # differ; the K2D solver itself is exactly the common LSCM path.
        result, report = _tag_lscm_equivalent_result(result, report)
        print(
            "[2026-09-14-K2D-BASE] "
            "used common LSCM _optimize_k2d; no 2026-09-14 extra K2D refinement"
        )
        return result, report

    pipeline._onestring_20260914_lscm_k2d_solver = optimize
    _wire_k2d(pipeline, optimize)
    _install_simple_split_bypass()
    pipeline._onestring_optcuts_paper_k2d_20260914_installed = True


__all__ = ["install_optcuts_paper_k2d_20260914_patch", "VERSION_ID"]
