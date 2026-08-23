"""Bridge ``optcuts_test`` into the simplified seam-smoothing / clipped-M2D path."""
from __future__ import annotations

from typing import Any

from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust
from .optcuts_test_simple_pipeline_patch import install_optcuts_test_simple_pipeline_patch
from .optcuts_test_boundary_clip_m2d_patch import install_optcuts_test_boundary_clip_m2d_patch


def install_optcuts_test_seam_metadata_bridge(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_seam_metadata_bridge_installed", False):
        return

    # The launcher still imports the historical experimental test wrapper for
    # compatibility, but this later wrapper deliberately supersedes its
    # grid-outline forcing.  optcuts_test is now:
    # official OptCuts -> seam/boundary smoothing -> harmonic Omega regeneration.
    install_optcuts_test_simple_pipeline_patch(pipeline)

    # Install the M2D clipper before Simple Split is installed.  Simple Split will
    # wrap the current _build_m2d and therefore retains this implementation below
    # it.  Ordinary modes no-op through the clipper.
    install_optcuts_test_boundary_clip_m2d_patch(pipeline)

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
        # New requested behavior: boundary-crossing grid cells are clipped to
        # Omega instead of being discarded wholesale.
        setattr(domain, "_optcuts_test_clip_boundary", True)
        setattr(domain, "_optcuts_test_smoothed_seam", True)
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
