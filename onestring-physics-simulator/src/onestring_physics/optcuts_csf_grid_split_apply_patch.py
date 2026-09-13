"""Bridge measured OptCuts CSF splits into the actual M2D grid topology.

The CSF diagnostic/planner runs at S -> Omega time and stores complete row/column
cut requests on the flattened domain.  The Simple Split topology patch, however,
decides whether to cut from ``mesh.metrics['csf_split_applied']``.  Without this
bridge the requests are visible in logs but never reach the fabricated M2D mesh.

This module makes the paper-style behavior explicit:

* measure the OptCuts scale factor;
* if a part still needs scale factor > 2, request a complete row/column cut;
* snap that request to an existing M2D fabrication-grid line (performed by
  ``simple_split_panel_patch``);
* duplicate interface vertices so the two sides are topologically disconnected;
* discard any fabrication-grid cell crossed by the Omega outer boundary instead
  of keeping a clipped partial tile.

No fake clamping of CSF values is performed.  The measured/planned residual is
carried through for diagnostics.
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

    # The previous diagnostic default of 16 was intentionally small.  For the
    # paper-style route, keep splitting until sigma<=2 in normal Bunny-scale
    # examples, while still retaining a finite emergency budget.
    os.environ.setdefault("ONESTRING_OPTCUTS_CSF_MAX_SPLITS", str(DEFAULT_MAX_SPLITS))

    base_build = pipeline._build_m2d

    def build_m2d_with_csf_split_metadata(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        mode = str(getattr(params, "omega_parameterization_mode", "")) if params is not None else ""
        if mode != "optcuts_test":
            return mesh

        split_lines = list(_domain_value(domain, "split_lines", []) or [])
        csf_before = _domain_value(domain, "csf_before", float("nan"))
        csf_after = _domain_value(domain, "csf_after_split", float("nan"))
        threshold = _domain_value(domain, "csf_split_threshold", 2.0)

        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update(
            {
                # Canonical flag consumed by simple_split_panel_patch.
                "csf_split_applied": bool(split_lines),
                "split_locations": [tuple(line[:2]) for line in split_lines],
                "csf_split_lines": [tuple(line[:2]) for line in split_lines],
                "csf_split_threshold": float(threshold),
                "csf_before": float(csf_before),
                "csf_after_split": float(csf_after),
                # Compatibility keys still read by the legacy Streamlit UI.
                "max_csf_before_split": float(csf_before),
                "max_csf_after_split": float(csf_after),
                "paper_style_grid_split_requested": bool(split_lines),
                "paper_style_grid_split_request_count": int(len(split_lines)),
                "paper_style_grid_split_rule": "complete row/column cuts when component scale factor > 2",
            }
        )
        mesh.metrics.update(metrics)

        # Preserve the requests on the mesh as well.  This is useful for debug
        # visualizations and makes the hand-off independent of domain lifetime.
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

    # Wrap the metadata bridge with strict whole-cell cropping.  Installation is
    # here (rather than another __init__ hook) so reloads preserve the ordering:
    # legacy M2D -> CSF metadata -> remove boundary-crossing cells -> Simple Split.
    pipeline._onestring_optcuts_strict_boundary_grid_crop_patch_installed = False
    install_optcuts_strict_boundary_grid_crop_patch(pipeline)

    pipeline._onestring_optcuts_csf_grid_split_apply_patch_installed = True


__all__ = ["DEFAULT_MAX_SPLITS", "install_optcuts_csf_grid_split_apply_patch"]
