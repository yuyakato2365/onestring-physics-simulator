"""Visualization compatibility for OptCuts parameterizations with split UV indices.

OptCuts may duplicate UV vertices along seams, so ``uv_vertices_2d`` and
``surface_vertices_3d`` are not required to have the same length.  The paired
``uv_faces`` / ``surface_faces`` corner correspondence is the authoritative map
from a UV vertex to its 3D surface vertex.

This patch changes only visualization helpers.  Numerical pipeline data is not
modified.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _uv_to_surface_vertex_ids(parameterization: Any) -> np.ndarray:
    """Return one surface-vertex id for every UV vertex, or -1 if unresolved."""
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)
    surface_faces = np.asarray(parameterization.surface_faces, dtype=int)
    mapping = np.full(len(uv), -1, dtype=int)
    if uv_faces.ndim != 2 or surface_faces.ndim != 2 or len(uv_faces) != len(surface_faces):
        return mapping
    corners = min(uv_faces.shape[1], surface_faces.shape[1])
    for fi in range(len(uv_faces)):
        for ci in range(corners):
            uvid = int(uv_faces[fi, ci])
            svid = int(surface_faces[fi, ci])
            if 0 <= uvid < len(mapping):
                if mapping[uvid] < 0:
                    mapping[uvid] = svid
                # If OptCuts duplicated the 3D index too, different ids may still
                # represent the same geometric point.  The first corner mapping is
                # sufficient for visualization and preserves face correspondence.
    return mapping


def _uv_aligned_surface_xyz(parameterization: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return xyz aligned one-to-one with UV vertices and a validity mask."""
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    mapping = _uv_to_surface_vertex_ids(parameterization)
    valid = (mapping >= 0) & (mapping < len(surface_xyz))
    xyz = np.full((len(uv), 3), np.nan, dtype=float)
    if np.any(valid):
        xyz[valid] = surface_xyz[mapping[valid]]
    return xyz, valid


def _uv_aligned_csf(state: Any, mapping: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align CSF values to UV vertices for both UV- and surface-indexed data."""
    parameterization = state.surface_parameterization
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    raw = np.asarray(getattr(state.conformal_domain, "csf_values", np.zeros(0)), dtype=float)
    if len(raw) == len(uv):
        return raw.copy(), np.ones(len(uv), dtype=bool)
    if len(raw) == len(surface_xyz):
        valid = (mapping >= 0) & (mapping < len(raw))
        out = np.full(len(uv), np.nan, dtype=float)
        out[valid] = raw[mapping[valid]]
        return out, valid
    return np.full(len(uv), np.nan, dtype=float), np.zeros(len(uv), dtype=bool)


def install_optcuts_visualization_patch(visualization_module: Any) -> None:
    if getattr(visualization_module, "_onestring_optcuts_visualization_patch_installed", False):
        return

    def high_csf_vertices(state: Any):
        parameterization = state.surface_parameterization
        uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
        mapping = _uv_to_surface_vertex_ids(parameterization)
        xyz, xyz_valid = _uv_aligned_surface_xyz(parameterization)
        csf, csf_valid = _uv_aligned_csf(state, mapping)
        threshold = float(state.mesh_2d_initial.metrics.get("csf_split_threshold", 2.0))
        valid = xyz_valid & csf_valid & np.isfinite(csf)
        mask = valid & (csf > threshold)
        return uv[mask], xyz[mask], csf[mask]

    def residual_high_csf_vertices(state: Any):
        parameterization = state.surface_parameterization
        uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
        mapping = _uv_to_surface_vertex_ids(parameterization)
        xyz, xyz_valid = _uv_aligned_surface_xyz(parameterization)
        csf, csf_valid = _uv_aligned_csf(state, mapping)
        indices = state.mesh_2d_initial.metrics.get("csf_split_residual_high_vertex_indices_after_all", [])
        try:
            ids = np.asarray([int(i) for i in indices], dtype=int)
        except Exception:
            ids = np.zeros(0, dtype=int)
        ids = ids[(ids >= 0) & (ids < len(uv))]
        if len(ids):
            keep = xyz_valid[ids] & csf_valid[ids] & np.isfinite(csf[ids])
            ids = ids[keep]
        if len(ids) == 0:
            return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float)
        return uv[ids], xyz[ids], csf[ids]

    def surface_peak_markers(state: Any):
        parameterization = state.surface_parameterization
        peak_uv = visualization_module._surface_peak_uvs(parameterization)
        uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
        xyz, xyz_valid = _uv_aligned_surface_xyz(parameterization)
        if len(peak_uv) == 0 or len(uv) == 0 or not np.any(xyz_valid):
            return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)
        valid_ids = np.flatnonzero(xyz_valid)
        valid_uv = uv[valid_ids]
        peak_xyz = []
        for point in peak_uv:
            local = int(np.argmin(np.linalg.norm(valid_uv - np.asarray(point, dtype=float), axis=1)))
            peak_xyz.append(xyz[int(valid_ids[local])])
        return np.asarray(peak_uv, dtype=float), np.asarray(peak_xyz, dtype=float)

    visualization_module._high_csf_vertices = high_csf_vertices
    visualization_module._residual_high_csf_vertices = residual_high_csf_vertices
    visualization_module._surface_peak_markers = surface_peak_markers
    visualization_module._onestring_optcuts_visualization_patch_installed = True


__all__ = [
    "install_optcuts_visualization_patch",
    "_uv_to_surface_vertex_ids",
    "_uv_aligned_surface_xyz",
]
