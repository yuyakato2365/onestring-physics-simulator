"""Carry the active OptCuts parameterization object onto the Omega domain.

The OptCuts M2D seam adapter uses this only for a pre-lift validity check: each
candidate quad is inverse-mapped to S and rejected if its four corners would
collapse to duplicate/degenerate 3D top vertices. Non-OptCuts modes are
untouched.
"""
from __future__ import annotations

from typing import Any


def install_optcuts_parameterization_reference_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_parameterization_reference_patch_installed", False):
        return
    base_flatten = pipeline._flatten_to_domain

    def flatten_with_parameterization_reference(parameterization: Any, grid: Any, params: Any = None):
        domain = base_flatten(parameterization, grid, params)
        if str(getattr(parameterization, "method", "")) == "optcuts":
            try:
                setattr(domain, "_optcuts_parameterization", parameterization)
            except Exception:
                pass
        return domain

    pipeline._flatten_to_domain = flatten_with_parameterization_reference
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = flatten_with_parameterization_reference
    pipeline._onestring_optcuts_parameterization_reference_patch_installed = True


__all__ = ["install_optcuts_parameterization_reference_patch"]
