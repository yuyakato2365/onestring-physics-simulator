"""Bridge measured OptCuts CSF splits into the actual M2D grid topology.

The CSF diagnostic/planner runs at S -> Omega time and stores complete row/column
cut requests on the flattened domain. The Simple Split topology patch decides
whether to cut from ``mesh.metrics['csf_split_applied']``. This module propagates
those requests and also forces the final runtime M2D route to remove clipped
boundary panels after all diagnostic wrappers are installed.
"""
from __future__ import annotations

import os
from typing import Any

from .optcuts_strict_boundary_grid_crop_patch import install_optcuts_strict_boundary_grid_crop_patch


DEFAULT_MAX_SPLITS = 128


def _domain_value(domain: Any, name: str, default: Any = None) -> Any:
    if hasattr(domain, name):
        try:
            return getattr(domain, name)
        except Exception:
            pass
    metrics = getattr(domain, "metrics", None)
    if isinstance(metrics, dict):
        return metrics.get(name, default)
    return default


def install_optcuts_csf_grid_split_apply_patch(pipeline: Any) -> None:
    """Propagate the CSF split plan to M2D so Simple Split performs real cuts."""
    if getattr(pipeline, "_onestring_optcuts_csf_grid_split_apply_patch_installed", False):
        return

    os.environ.setdefault("ONESTRING_OPTCUTS_CSF_MAX_SPLITS", str(DEFAULT_MAX_SPLITS))
    base_build = pipeline._build_m2d

    def build_m2d_with_csf_split_metadata(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        mode = str(getattr(params, "omega_parameterization_mode", "")) if params is not None else ""
        if mode != "optcuts_test":
            return mesh

        split_lines = list(_domain_value(domain, "split_lines", []) or [])
        if not split_lines:
            split_lines = list(_domain_value(domain, "localized_split_segments", []) or [])
        csf_before = _domain_value(domain, "csf_before", float("nan"))
        csf_after = _domain_value(domain, "csf_after_split", float("nan"))
        threshold = _domain_value(domain, "csf_split_threshold", 2.0)

        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update(
            {
                "csf_split_applied": bool(split_lines),
                "split_locations": [tuple(line[:2]) for line in split_lines],
                "csf_split_lines": [tuple(line[:2]) for line in split_lines],
                "csf_split_threshold": float(threshold),
                "csf_before": float(csf_before),
                "csf_after_split": float(csf_after),
                "max_csf_before_split": float(csf_before),
                "max_csf_after_split": float(csf_after),
                "paper_style_grid_split_requested": bool(split_lines),
                "paper_style_grid_split_request_count": int(len(split_lines)),
                "paper_style_grid_split_rule": "complete row/column cuts when component scale factor > 2",
            }
        )
        mesh.metrics.update(metrics)
        try:
            mesh.split_lines = list(split_lines)
        except Exception:
            pass

        print(
            "[OPTCUTS-CSF-GRID-SPLIT-REQUEST] "
            f"active={bool(split_lines)} count={len(split_lines)} "
            f"before={float(csf_before):.9g} planned_after={float(csf_after):.9g} "
            f"bound={float(threshold):.9g}"
        )
        return mesh

    pipeline._build_m2d = build_m2d_with_csf_split_metadata
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_m2d_with_csf_split_metadata

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_with_csf_split_metadata

    pipeline._onestring_optcuts_strict_boundary_grid_crop_patch_installed = False
    install_optcuts_strict_boundary_grid_crop_patch(pipeline)
    pipeline._onestring_optcuts_csf_grid_split_apply_patch_installed = True


def _install_post_diagnostic_grid_policy_hook() -> None:
    """Make whole-cell cleanup the outermost runtime M2D step.

    app_split_panels installs split_diagnostics after package initialization. That
    changes the active _build_m2d function. Hook the diagnostics installer itself
    so the final policy is installed *after* diagnostic_build_m2d and therefore
    cannot be bypassed by the launcher wiring.
    """
    try:
        from . import split_diagnostics as diagnostics
        from .final_m2d_grid_policy_patch import install_final_m2d_grid_policy_patch
    except Exception:
        return

    if getattr(diagnostics, "_onestring_final_grid_policy_hooked", False):
        return
    original_installer = diagnostics.install_split_diagnostics

    def install_and_finalize(pipeline_module: Any, final_module: Any, project_root: Any):
        result = original_installer(pipeline_module, final_module, project_root)
        pipeline_module._final_m2d_grid_policy_patch_installed = False
        install_final_m2d_grid_policy_patch(pipeline_module)
        active = pipeline_module._build_m2d
        try:
            pipeline_module._original._build_m2d = active
        except Exception:
            pass
        for fn in (
            getattr(pipeline_module, "build_onestring_design", None),
            getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
            getattr(getattr(pipeline_module, "_original", None), "build_onestring_design", None),
        ):
            glb = getattr(fn, "__globals__", None)
            if isinstance(glb, dict):
                glb["_build_m2d"] = active
        print("[OPTCUTS-FINAL-M2D-GRID-ROUTE] installed after split diagnostics")
        return result

    diagnostics.install_split_diagnostics = install_and_finalize
    diagnostics._onestring_final_grid_policy_hooked = True


_install_post_diagnostic_grid_policy_hook()


__all__ = ["DEFAULT_MAX_SPLITS", "install_optcuts_csf_grid_split_apply_patch"]
