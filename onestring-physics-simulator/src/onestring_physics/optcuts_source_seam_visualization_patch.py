"""Show the original OptCuts surface-mesh seam on the Omega view.

A cut parameterization duplicates each seam edge in UV, so the faithful Omega
visualization is the pair of UV edge copies, not the midpoint between them.
This module changes visualization only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import plotly.graph_objects as go


def source_seam_uv_copies(parameterization: Any) -> np.ndarray:
    """Return both UV copies of every original surface-mesh seam edge.

    Shape is ``(2 * seam_edge_count, 2, 2)``.  A surface edge is considered a
    seam when its two incident triangles map the same 3D edge to geometrically
    separated UV endpoint positions.
    """
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    if len(xyz) == 0 or len(uv) == 0 or len(sf) == 0 or len(sf) != len(uf):
        return np.zeros((0, 2, 2), dtype=float)

    span3 = max(float(np.nanmax(xyz) - np.nanmin(xyz)), 1.0)
    tol3 = max(1e-10 * span3, 1e-12)
    canonical_lookup: dict[tuple[int, int, int], int] = {}
    canonical_of = np.empty(len(xyz), dtype=int)
    for vi, p in enumerate(xyz[:, :3]):
        key = tuple(np.rint(p / tol3).astype(np.int64).tolist())
        cid = canonical_lookup.get(key)
        if cid is None:
            cid = len(canonical_lookup)
            canonical_lookup[key] = cid
        canonical_of[vi] = int(cid)

    incidence: dict[tuple[int, int], list[dict[int, np.ndarray]]] = defaultdict(list)
    for face3, face2 in zip(sf, uf):
        face3 = np.asarray(face3, dtype=int)
        face2 = np.asarray(face2, dtype=int)
        if len(face3) < 3 or len(face2) < 3:
            continue
        cface = canonical_of[face3[:3]]
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            ca, cb = int(cface[ia]), int(cface[ib])
            if ca == cb:
                continue
            key = (ca, cb) if ca < cb else (cb, ca)
            incidence[key].append(
                {
                    ca: np.asarray(uv[int(face2[ia])], dtype=float),
                    cb: np.asarray(uv[int(face2[ib])], dtype=float),
                }
            )

    uv_span = max(float(np.nanmax(uv) - np.nanmin(uv)), 1.0)
    uv_tol = max(1e-8 * uv_span, 1e-10)
    copies: list[np.ndarray] = []
    for (a, b), incident in incidence.items():
        if len(incident) != 2:
            continue
        c0, c1 = incident
        if a not in c0 or b not in c0 or a not in c1 or b not in c1:
            continue
        if (
            float(np.linalg.norm(c0[a] - c1[a])) <= uv_tol
            and float(np.linalg.norm(c0[b] - c1[b])) <= uv_tol
        ):
            continue
        copies.append(np.asarray([c0[a], c0[b]], dtype=float))
        copies.append(np.asarray([c1[a], c1[b]], dtype=float))

    if not copies:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(copies, dtype=float)


def install_optcuts_source_seam_visualization_patch() -> None:
    from . import visualization as viz

    if getattr(viz, "_onestring_optcuts_source_seam_visualization_installed", False):
        return

    original_figure_domain = viz.figure_domain

    def figure_domain_with_source_seam(state: Any) -> go.Figure:
        fig = original_figure_domain(state)
        segments = source_seam_uv_copies(state.surface_parameterization)
        if len(segments):
            xs: list[float | None] = []
            ys: list[float | None] = []
            for segment in segments:
                xs.extend([float(segment[0, 0]), float(segment[1, 0]), None])
                ys.extend([float(segment[0, 1]), float(segment[1, 1]), None])
            fig.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="#f97316", width=4, dash="dash"),
                    name="OptCuts source seam (surface mesh; both UV sides)",
                    hoverinfo="skip",
                )
            )
        return fig

    viz.figure_domain = figure_domain_with_source_seam
    viz._onestring_optcuts_source_seam_visualization_installed = True


__all__ = [
    "source_seam_uv_copies",
    "install_optcuts_source_seam_visualization_patch",
]
