"""Bridge OptCuts_test seam metadata into the existing M2D seam projection path."""
from __future__ import annotations

from typing import Any

from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust
from .optcuts_test_harmonic_extension_patch import install_optcuts_test_harmonic_extension_patch


def install_optcuts_test_seam_metadata_bridge(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_seam_metadata_bridge_installed", False):
        return

    # Replace only optcuts_test's continuation routine.  This must happen after
    # optcuts_test_boundary_reparameterization_patch has been imported, but before
    # the first design run.  Ordinary optcuts / optcuts_grid remain untouched.
    install_optcuts_test_harmonic_extension_patch()

    base_flatten = pipeline._flatten_to_domain

    def flatten_with_test_seam(parameterization: Any, grid: Any, params: Any = None):
        domain = base_flatten(parameterization, grid, params)
        if str(getattr(parameterization, "method", "")) != "optcuts_test":
            return domain
        payload = extract_connected_seam_payload_robust(parameterization)
        setattr(domain, "_optcuts_grid_seam_payload", payload)
        previous = list(getattr(domain, "split_lines", []) or [])
        setattr(domain, "_optcuts_suppressed_legacy_split_lines", previous)
        try:
            domain.split_lines = []
        except Exception:
            pass
        setattr(domain, "_optcuts_test_grid_outline", getattr(parameterization, "_optcuts_test_grid_outline", None))
        return domain

    pipeline._flatten_to_domain = flatten_with_test_seam
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._flatten_to_domain = flatten_with_test_seam
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = flatten_with_test_seam
    pipeline._onestring_optcuts_test_seam_metadata_bridge_installed = True


__all__ = ["install_optcuts_test_seam_metadata_bridge"]
