"""Visualization compatibility and seam highlighting for OptCuts.

OptCuts parameterizations duplicate UV vertices along cuts, so legacy helpers
cannot assume ``len(uv_vertices_2d) == len(surface_vertices_3d)``.  This module
keeps those visualizations compatible and, for the ordinary OptCuts path, draws
its final fabrication seam explicitly:

* Omega / M2D: seam edges are overlaid as solid red lines.
* M3D: every quad panel touching either side of a seam is overlaid in red.

This module changes visualization only.  It does not alter OptCuts, Split, M2D,
or M3D numerics/topology.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


SEAM_RED = "#ef4444"


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
        out[~seen] = np.mean(xyz, axis=0)
    return out


def _edge_coord_key(a: np.ndarray, b: np.ndarray, decimals: int = 8) -> tuple[tuple[float, float], tuple[float, float]]:
    pa = tuple(np.round(np.asarray(a, dtype=float)[:2], decimals).tolist())
    pb = tuple(np.round(np.asarray(b, dtype=float)[:2], decimals).tolist())
    return (pa, pb) if pa <= pb else (pb, pa)


def _optcuts_seam_segments_2d(state: Any) -> np.ndarray:
    """Return final fabrication-grid seam segments from the stored OptCuts paths."""
    mesh = state.mesh_2d_initial
    vertices = np.asarray(mesh.vertices, dtype=float)
    paths = getattr(mesh, "_optcuts_grid_seam_paths", None)
    if not paths or len(vertices) == 0:
        return np.zeros((0, 2, 2), dtype=float)

    segments: list[np.ndarray] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for path in paths:
        ids = [int(v) for v in path]
        for a, b in zip(ids[:-1], ids[1:]):
            if not (0 <= a < len(vertices) and 0 <= b < len(vertices)):
                continue
            pa = vertices[a, :2]
            pb = vertices[b, :2]
            if np.linalg.norm(pa - pb) <= 1e-12:
                continue
            key = _edge_coord_key(pa, pb)
            if key in seen:
                continue
            seen.add(key)
            segments.append(np.asarray([pa, pb], dtype=float))
    if not segments:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(segments, dtype=float)


def _seam_adjacent_m2d_face_ids(state: Any) -> np.ndarray:
    """Find both panels adjacent to every seam edge using geometry, not vertex ids.

    Split duplicates seam-side vertices.  Therefore an id-based test would find
    only one side.  Coordinate edge keys are identical on the two zero-width cut
    copies and reliably identify both adjacent panels.
    """
    seam_segments = _optcuts_seam_segments_2d(state)
    if len(seam_segments) == 0:
        return np.zeros(0, dtype=int)

    seam_keys = {_edge_coord_key(segment[0], segment[1]) for segment in seam_segments}
    mesh = state.mesh_2d_initial
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    hit: list[int] = []
    for face_id, face in enumerate(faces):
        ids = [int(v) for v in np.asarray(face, dtype=int).reshape(-1)]
        if len(ids) < 2 or any(v < 0 or v >= len(vertices) for v in ids):
            continue
        matched = False
        for i, a in enumerate(ids):
            b = ids[(i + 1) % len(ids)]
            if _edge_coord_key(vertices[a], vertices[b]) in seam_keys:
                matched = True
                break
        if matched:
            hit.append(int(face_id))
    return np.asarray(hit, dtype=int)


def _add_seam_lines_to_domain_figure(fig: go.Figure, state: Any) -> None:
    segments = _optcuts_seam_segments_2d(state)
    if len(segments) == 0:
        return
    x: list[float | None] = []
    y: list[float | None] = []
    for segment in segments:
        x.extend([float(segment[0, 0]), float(segment[1, 0]), None])
        y.extend([float(segment[0, 1]), float(segment[1, 1]), None])
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=SEAM_RED, width=5),
            name="OptCuts seam",
            hoverinfo="skip",
        )
    )


def _add_seam_panels_to_m3d_figure(fig: go.Figure, state: Any) -> None:
    face_ids = _seam_adjacent_m2d_face_ids(state)
    m3d = state.mesh_3d_initial
    vertices = np.asarray(m3d.vertices, dtype=float)
    faces = np.asarray(m3d.faces, dtype=int)
    face_ids = face_ids[(face_ids >= 0) & (face_ids < len(faces))]
    if len(face_ids) == 0 or len(vertices) == 0:
        return

    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    ii: list[int] = []
    jj: list[int] = []
    kk: list[int] = []
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []

    for face_id in face_ids.tolist():
        face = [int(v) for v in np.asarray(faces[face_id], dtype=int).reshape(-1)]
        if len(face) < 3 or any(v < 0 or v >= len(vertices) for v in face):
            continue
        pts = vertices[np.asarray(face, dtype=int), :3]
        base = len(x)
        x.extend(pts[:, 0].tolist())
        y.extend(pts[:, 1].tolist())
        z.extend(pts[:, 2].tolist())
        for local in range(1, len(face) - 1):
            ii.append(base)
            jj.append(base + local)
            kk.append(base + local + 1)
        closed = np.vstack([pts, pts[0]])
        edge_x.extend([*closed[:, 0].tolist(), None])
        edge_y.extend([*closed[:, 1].tolist(), None])
        edge_z.extend([*closed[:, 2].tolist(), None])

    if not ii:
        return
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=ii,
            j=jj,
            k=kk,
            color=SEAM_RED,
            opacity=0.9,
            flatshading=True,
            lighting=dict(ambient=0.9, diffuse=0.25, specular=0.0, roughness=1.0, fresnel=0.0),
            name=f"Seam-adjacent M3D panels ({len(face_ids)})",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="#991b1b", width=3),
            name="Seam panel edges",
            showlegend=False,
            hoverinfo="skip",
        )
    )


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

    original_figure_domain = viz.figure_domain
    original_figure_m3d_overlay = viz.figure_m3d_overlay

    def figure_domain_with_optcuts_seam(state: Any) -> go.Figure:
        fig = original_figure_domain(state)
        _add_seam_lines_to_domain_figure(fig, state)
        return fig

    def figure_m3d_overlay_with_optcuts_seam_panels(state: Any) -> go.Figure:
        fig = original_figure_m3d_overlay(state)
        _add_seam_panels_to_m3d_figure(fig, state)
        return fig

    viz._high_csf_vertices = high_csf_vertices
    viz._residual_high_csf_vertices = residual_high_csf_vertices
    viz._surface_peak_markers = surface_peak_markers
    viz.figure_domain = figure_domain_with_optcuts_seam
    viz.figure_m3d_overlay = figure_m3d_overlay_with_optcuts_seam_panels
    viz._onestring_optcuts_visualization_compat_installed = True


__all__ = [
    "install_optcuts_visualization_compat_patch",
    "_xyz_per_uv",
    "_optcuts_seam_segments_2d",
    "_seam_adjacent_m2d_face_ids",
]
