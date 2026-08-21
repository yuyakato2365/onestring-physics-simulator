"""Visualization compatibility for parameterizations with duplicated UV seam vertices.

Legacy visual helpers assumed ``len(uv_vertices_2d) == len(surface_vertices_3d)``.
That is false for every real cut parameterization: one surface vertex can have
multiple UV copies on different seam sides.  This patch derives a 3D position
per UV id from the paired ``uv_faces`` / ``surface_faces`` corner correspondence.
It changes visualization only.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _xyz_per_uv(parameterization: Any) -> np.ndarray:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    if len(uv) == len(xyz):
        return xyz.copy()
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    if len(uf) != len(sf):
        return np.zeros((len(uv), 3), dtype=float)
    out = np.zeros((len(uv), 3), dtype=float)
    seen = np.zeros(len(uv), dtype=bool)
    for fuv, fs in zip(uf, sf):
        for uid, sid in zip(fuv, fs):
            uid, sid = int(uid), int(sid)
            if 0 <= uid < len(out) and 0 <= sid < len(xyz):
                out[uid] = xyz[sid]
                seen[uid] = True
    if np.any(~seen) and len(xyz):
        # Unreferenced UV ids should not normally exist in OptCuts output.  Keep
        # visualization total rather than failing an otherwise valid simulation.
        out[~seen] = np.mean(xyz, axis=0)
    return out


def install_optcuts_visualization_compat_patch() -> None:
    from . import visualization as viz

    if getattr(viz, "_onestring_optcuts_visualization_compat_installed", False):
        return

    def high_csf_vertices(state: Any):
        p = state.surface_parameterization
        uv = np.asarray(p.uv_vertices_2d, dtype=float)
        xyz = _xyz_per_uv(p)
        csf = np.asarray(getattr(state.conformal_domain, "csf_values", np.zeros(0)), dtype=float)
        threshold = float(state.mesh_2d_initial.metrics.get("csf_split_threshold", 2.0))
        if len(csf) != len(uv):
            return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)
        mask = csf > threshold
        return uv[mask], xyz[mask], csf[mask]

    def residual_high_csf_vertices(state: Any):
        p = state.surface_parameterization
        uv = np.asarray(p.uv_vertices_2d, dtype=float)
        xyz = _xyz_per_uv(p)
        csf = np.asarray(getattr(state.conformal_domain, "csf_values", np.zeros(0)), dtype=float)
        raw = state.mesh_2d_initial.metrics.get("csf_split_residual_high_vertex_indices_after_all", [])
        try:
            ids = np.asarray([int(x) for x in raw], dtype=int)
        except Exception:
            ids = np.zeros(0, dtype=int)
        ids = ids[(ids >= 0) & (ids < len(uv)) & (ids < len(csf))]
        if len(ids) == 0:
            return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros(0)
        return uv[ids], xyz[ids], csf[ids]

    def surface_peak_markers(state: Any):
        p = state.surface_parameterization
        peak_uv = viz._surface_peak_uvs(p)
        uv = np.asarray(p.uv_vertices_2d, dtype=float)
        xyz_uv = _xyz_per_uv(p)
        if len(peak_uv) == 0 or len(uv) == 0:
            return np.zeros((0, 2)), np.zeros((0, 3))
        peak_xyz = []
        for point in peak_uv:
            idx = int(np.argmin(np.linalg.norm(uv - point, axis=1)))
            peak_xyz.append(xyz_uv[idx])
        return peak_uv, np.asarray(peak_xyz, dtype=float)

    viz._high_csf_vertices = high_csf_vertices
    viz._residual_high_csf_vertices = residual_high_csf_vertices
    viz._surface_peak_markers = surface_peak_markers
    viz._onestring_optcuts_visualization_compat_installed = True


__all__ = ["install_optcuts_visualization_compat_patch", "_xyz_per_uv"]
