"""Record metadata for the final surface-constrained K3D experiment result."""
from __future__ import annotations

import os
from typing import Any


def install_optcuts_test2_surface_result_metadata_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test2_surface_result_metadata_installed", False):
        return

    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        out, report = base(target, mesh, parameterization, params)
        metrics = dict(getattr(out, "metrics", {}) or {})
        active = (
            str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"
            and os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0") == "2"
            and bool(metrics.get("k3d_surface_constrained_authoritative_result", False))
        )
        if not active:
            return out, report

        try:
            metrics["k3d_surface_constrained_upstream_al_applied"] = bool(
                metrics.get("k3d_augmented_lagrangian_applied", False)
            )
            metrics["k3d_surface_constrained_upstream_hard_planarity_tolerance"] = float(
                metrics.get("k3d_hard_planarity_tolerance", 0.0)
            )
            metrics["k3d_augmented_lagrangian_applied"] = False
            metrics["k3d_al_authoritative_result"] = False
            metrics["k3d_final_planarity_acceptance_mode"] = (
                "surface-constrained diagnostic final geometry"
            )
            out.metrics.update(metrics)
        except Exception:
            pass
        return out, report

    pipeline._optimize_k3d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = optimize
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize

    pipeline._onestring_optcuts_test2_surface_result_metadata_installed = True


__all__ = ["install_optcuts_test2_surface_result_metadata_patch"]
