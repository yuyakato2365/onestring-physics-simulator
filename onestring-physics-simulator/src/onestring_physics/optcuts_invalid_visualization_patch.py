"""Visualization overlay for invalid OptCuts-test panels.

The diagnostic test mode deliberately keeps running when a lifted M3D/K3D face
fails ``validate_top_quad``.  Those face ids are stored on the mesh and painted
in orange so the rest of the pipeline can still be inspected.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import plotly.graph_objects as go

INVALID_ORANGE = "#f59e0b"
INVALID_EDGE = "#7c2d12"


def _add_invalid_faces(fig: go.Figure, mesh: Any, label: str) -> None:
    raw = getattr(mesh, "_optcuts_invalid_face_ids", None)
    if raw is None:
        raw = (getattr(mesh, "metrics", {}) or {}).get("optcuts_invalid_face_ids", [])
    try:
        face_ids = sorted({int(v) for v in (raw or [])})
    except Exception:
        return
    vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=float)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 4), dtype=int)), dtype=int)
    face_ids = [fi for fi in face_ids if 0 <= fi < len(faces)]
    if not face_ids or len(vertices) == 0:
        return

    x=[]; y=[]; z=[]; ii=[]; jj=[]; kk=[]
    ex=[]; ey=[]; ez=[]
    for fi in face_ids:
        ids=[int(v) for v in np.asarray(faces[fi], dtype=int).reshape(-1)]
        if len(ids) < 3 or any(v < 0 or v >= len(vertices) for v in ids):
            continue
        pts=vertices[np.asarray(ids, dtype=int), :3]
        base=len(x)
        x.extend(pts[:,0].tolist()); y.extend(pts[:,1].tolist()); z.extend(pts[:,2].tolist())
        for j in range(1, len(ids)-1):
            ii.append(base); jj.append(base+j); kk.append(base+j+1)
        closed=np.vstack([pts, pts[0]])
        ex.extend([*closed[:,0].tolist(), None]); ey.extend([*closed[:,1].tolist(), None]); ez.extend([*closed[:,2].tolist(), None])
    if not ii:
        return
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=ii, j=jj, k=kk,
        color=INVALID_ORANGE, opacity=0.95, flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.15, specular=0.0, roughness=1.0, fresnel=0.0),
        name=f"Invalid {label} panels ({len(face_ids)})",
    ))
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color=INVALID_EDGE, width=5),
        name=f"Invalid {label} edges", showlegend=False, hoverinfo="skip",
    ))


def install_optcuts_invalid_visualization_patch() -> None:
    from . import visualization as viz
    if getattr(viz, "_onestring_optcuts_invalid_visualization_installed", False):
        return
    base = viz.figure_m3d_overlay

    def figure_m3d_overlay_with_invalid(state: Any) -> go.Figure:
        fig = base(state)
        m3d = getattr(state, "mesh_3d_initial", None)
        if m3d is not None:
            _add_invalid_faces(fig, m3d, "M3D")
        k3d = getattr(state, "mesh_3d_optimized", None)
        if k3d is not None:
            _add_invalid_faces(fig, k3d, "K3D")
        return fig

    viz.figure_m3d_overlay = figure_m3d_overlay_with_invalid
    viz._onestring_optcuts_invalid_visualization_installed = True


__all__=["install_optcuts_invalid_visualization_patch"]
