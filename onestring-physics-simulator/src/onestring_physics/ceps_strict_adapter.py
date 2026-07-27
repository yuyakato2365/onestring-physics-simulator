"""Strict OneString downstream handling for the official CEPS chart.

A CEPS M2D point is valid only when it lies in an actual stitched CEPS UV
triangle. The old adapter accepted every point inside a convex hull and then
used nearest-triangle/nearest-vertex fallback, which could bridge the input
surface boundary and visually cap an open bottom. This module rejects that
behavior for CEPS while leaving the other parameterization modes unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _is_ceps(parameterization: Any) -> bool:
    metrics = getattr(parameterization, "metrics", {}) or {}
    return bool(
        str(getattr(parameterization, "method", "")) == "ceps"
        or metrics.get("ceps_backend_used") == "official_ceps_cli"
    )


def install_ceps_strict_adapter(module: Any) -> None:
    if getattr(module, "_CEPS_STRICT_ADAPTER_INSTALLED", False):
        return

    legacy_inverse = module.inverse_map_uv_to_surface
    legacy_clip = module._clip_m2d_faces_to_omega_boundary

    def strict_inverse(uv_point: np.ndarray, parameterization: Any):
        point, triangle_id, outside = legacy_inverse(uv_point, parameterization)
        if _is_ceps(parameterization) and (int(triangle_id) < 0 or bool(outside)):
            raise RuntimeError(
                "CEPS inverse map left the actual stitched UV triangle domain. "
                "Nearest-triangle/nearest-vertex fallback is disabled because it "
                "can bridge the physical boundary and cap an open surface."
            )
        return point, triangle_id, outside

    def strict_clip(mesh: Any, domain: Any, params: Any = None):
        faces, metrics = legacy_clip(mesh, domain, params)
        parameterization = getattr(domain, "parameterization", None)
        if parameterization is None or not _is_ceps(parameterization):
            return faces, metrics

        vertices = np.asarray(mesh.vertices, dtype=float)
        kept: list[np.ndarray] = []
        rejected = 0
        for face in np.asarray(faces, dtype=int):
            quad = vertices[np.asarray(face, dtype=int), :2]
            samples = np.vstack(
                [
                    quad,
                    0.5 * (quad + np.roll(quad, -1, axis=0)),
                    np.mean(quad, axis=0, keepdims=True),
                ]
            )
            valid = True
            for sample in samples:
                _point, triangle_id, outside = legacy_inverse(sample, parameterization)
                if int(triangle_id) < 0 or bool(outside):
                    valid = False
                    break
            if valid:
                kept.append(np.asarray(face, dtype=int).copy())
            else:
                rejected += 1

        if not kept:
            raise RuntimeError(
                "CEPS actual-triangle clipping removed all M2D quads. The adapter "
                "will not replace missing UV coverage with a convex hull or nearest fallback."
            )

        output_metrics = dict(metrics)
        output_metrics.update(
            {
                "ceps_m2d_actual_uv_triangle_clipping_used": True,
                "ceps_m2d_actual_uv_triangle_sample_count_per_quad": 9,
                "ceps_m2d_quad_rejected_outside_actual_uv_union_count": int(rejected),
                "ceps_m2d_quad_kept_inside_actual_uv_union_count": int(len(kept)),
                "ceps_nearest_inverse_fallback_allowed": False,
            }
        )
        return np.asarray(kept, dtype=int), output_metrics

    module.inverse_map_uv_to_surface = strict_inverse
    module._clip_m2d_faces_to_omega_boundary = strict_clip
    module._CEPS_STRICT_ADAPTER_INSTALLED = True
