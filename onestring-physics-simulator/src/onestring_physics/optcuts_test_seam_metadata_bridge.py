"""Bridge ``optcuts_test`` into seam smoothing and polygon-clipped M2D.

The source OptCuts seam is retained only as diagnostic/source metadata.  It must
not be published as ``_optcuts_grid_seam_payload`` because that attribute is the
trigger for the old rectilinear/grid seam adapter, which replaced the smoothed
OptCuts seam by horizontal/vertical fabrication-grid paths.
"""
from __future__ import annotations

from typing import Any

from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust
from .optcuts_test_simple_pipeline_patch import install_optcuts_test_simple_pipeline_patch
from .optcuts_test_boundary_clip_m2d_patch import install_optcuts_test_boundary_clip_m2d_patch
from .optcuts_test_polygon_visualization_patch import install_optcuts_test_polygon_visualization_patch
from .optcuts_test_k2d_relative_layout_patch import install_optcuts_test_k2d_relative_layout_patch
from .optcuts_test_k3d_pre_al_validity_patch import install_optcuts_test_k3d_pre_al_validity_patch
from .optcuts_test_k3d_augmented_lagrangian_patch import install_optcuts_test_k3d_augmented_lagrangian_patch
from .optcuts_test_k3d_slsqp_polish_patch import install_optcuts_test_k3d_slsqp_polish_patch


def install_optcuts_test_seam_metadata_bridge(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_seam_metadata_bridge_installed", False):
        return

    # This wrapper is installed after the older grid-outline experiment. For
    # optcuts_test it deliberately calls ordinary official OptCuts and then runs
    # equal-arclength + Taubin smoothing + harmonic Omega regeneration, so the
    # grid-outline forcing path is bypassed.
    install_optcuts_test_simple_pipeline_patch(pipeline)
    install_optcuts_test_boundary_clip_m2d_patch(pipeline)

    # K3D wrapper order is intentional:
    # ordinary K3D -> validity repair -> Augmented Lagrangian hard planarity
    # -> explicit SLSQP equality polishing.
    # The generic outer validity guard installed by app_optcuts.py then performs
    # assertion-only checks on the polished result and is forbidden to backtrack it.
    install_optcuts_test_k3d_pre_al_validity_patch(pipeline)
    install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline)
    install_optcuts_test_k3d_slsqp_polish_patch(pipeline)

    install_optcuts_test_k2d_relative_layout_patch(pipeline)
    install_optcuts_test_polygon_visualization_patch()

    base_flatten = pipeline._flatten_to_domain

    def flatten_with_test_seam(parameterization: Any, grid: Any, params: Any = None):
        domain = base_flatten(parameterization, grid, params)
        if str(getattr(parameterization, "method", "")) != "optcuts_test":
            return domain

        # Keep the actual OptCuts source seam for diagnostics/visualization only.
        # DO NOT expose it through _optcuts_grid_seam_payload: that is the legacy
        # trigger which converts it into rectilinear H/V grid paths.
        payload = extract_connected_seam_payload_robust(parameterization)
        setattr(domain, "_optcuts_test_source_seam_payload", payload)
        setattr(parameterization, "_optcuts_test_source_seam_payload", payload)
        try:
            if hasattr(domain, "_optcuts_grid_seam_payload"):
                delattr(domain, "_optcuts_grid_seam_payload")
        except Exception:
            pass

        previous = list(getattr(domain, "split_lines", []) or [])
        setattr(domain, "_optcuts_suppressed_legacy_split_lines", previous)
        try:
            domain.split_lines = []
        except Exception:
            pass
        setattr(domain, "_optcuts_test_clip_boundary", True)
        setattr(domain, "_optcuts_test_smoothed_seam", True)
        setattr(domain, "_optcuts_test_rectilinear_seam_disabled", True)
        try:
            parameterization.metrics.update(
                {
                    "optcuts_test_rectilinear_seam_disabled": True,
                    "optcuts_test_source_seam_kept_for_diagnostics_only": True,
                }
            )
        except Exception:
            pass
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
