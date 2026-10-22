"""Display only: which four boundary points became the rectangle corners, and
Omega drawn with its long side vertical, as in the paper's Figure 5.

Nothing here changes the computation. Corners come from the recorded corner
positions of the rectangular BFF (Paper mode) or, for any other rectangular
Omega, from the four boundary vertices at the corners of the UV rectangle.
"""
from __future__ import annotations

import numpy as np

LABELS = ("C1", "C2", "C3", "C4")
COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")


def omega_corner_info(state):
    from .paper_mesh_splitting import chart_geometry
    from .paper_split_reflatten import _boundary_loop

    sp = state.surface_parameterization
    xyz, faces, _ = chart_geometry(sp)
    xyz = np.asarray(xyz, float)
    faces = np.asarray(faces, int)
    uv = np.asarray(sp.uv_vertices_2d, float)
    loop = _boundary_loop(faces)
    B = uv[loop]
    lo, hi = B.min(axis=0), B.max(axis=0)
    size = np.maximum(hi - lo, 1e-12)
    side_gap = np.minimum.reduce([np.abs(B[:, 0] - lo[0]), np.abs(B[:, 0] - hi[0]),
                                  np.abs(B[:, 1] - lo[1]), np.abs(B[:, 1] - hi[1])])
    is_rectangle = bool(side_gap.max() <= 0.02 * size.max())
    metrics = dict(getattr(sp, "metrics", {}) or {})
    recorded = metrics.get("bff_rectangle_corner_positions")
    if recorded is not None and len(recorded) == 4:
        P = xyz[loop]
        ids = [int(loop[int(np.argmin(np.linalg.norm(P - np.asarray(c, float), axis=1)))]) for c in recorded]
        source = "BFFに渡した隅（記録値）"
    else:
        targets = (lo, np.array([hi[0], lo[1]]), hi, np.array([lo[0], hi[1]]))
        ids = [int(loop[int(np.argmin(np.linalg.norm(B - t, axis=1)))]) for t in targets]
        source = "Ωの矩形の四隅に写った境界点"
    # label C1..C4 in counter-clockwise order around the Omega rectangle
    centre = uv[ids].mean(axis=0)
    order = np.argsort(np.arctan2(uv[ids, 1] - centre[1], uv[ids, 0] - centre[0]))
    ids = [ids[k] for k in order]
    return dict(xyz=xyz, uv=uv, loop=loop, corner_ids=ids, width=float(size[0]), height=float(size[1]),
                is_rectangle=is_rectangle, source=source,
                feature_count=len(metrics.get("bff_rectangle_feature_angles") or []))


def add_corners_to_surface_figure(fig, info):
    """Add the boundary and the four labelled corners to an existing 3D figure of S."""
    import plotly.graph_objects as go

    B = info["xyz"][list(info["loop"]) + [info["loop"][0]]]
    fig.add_trace(go.Scatter3d(x=B[:, 0], y=B[:, 1], z=B[:, 2], mode="lines",
                               line=dict(color="#1b2a31", width=4), name="Sの境界", hoverinfo="skip"))
    for k, vid in enumerate(info["corner_ids"]):
        p = info["xyz"][vid]
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", text=[LABELS[k]],
                                   textposition="top center", textfont=dict(size=16, color=COLORS[k]),
                                   marker=dict(size=9, color=COLORS[k], line=dict(color="white", width=1)),
                                   name=f"矩形Ωの隅 {LABELS[k]}"))
    return fig


def _upright(points, info):
    """Rotate Omega by +90 degrees when it is wider than tall (display only)."""
    P = np.asarray(points, float)
    if info["width"] > info["height"]:
        P = np.column_stack([-P[..., 1], P[..., 0]]) if P.ndim == 2 else np.stack([-P[..., 1], P[..., 0]], axis=-1)
    return P


def figure_omega_upright(state, info):
    """Omega boundary, M2D tiles (coloured by piece) and the corner labels, long side vertical."""
    import plotly.graph_objects as go

    fig = go.Figure()
    m2 = state.mesh_2d_initial
    canon = np.asarray(getattr(m2, "_split_panel_source_vertices", m2.vertices), float)[:, :2]
    faces = np.asarray(m2.faces, int)
    comps = getattr(m2, "_split_panel_face_components", None) or [list(range(len(faces)))]
    palette = ("#4f7cac", "#e2873a", "#59a14f", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7", "#9c755f")
    for c_index, ids in enumerate(sorted(comps, key=len, reverse=True)):
        xs, ys = [], []
        for f in faces[np.asarray(ids, int)]:
            q = _upright(canon[list(f) + [f[0]]], info)
            xs += list(q[:, 0]) + [None]; ys += list(q[:, 1]) + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                                 fillcolor=palette[c_index % len(palette)], opacity=0.55,
                                 line=dict(color="rgba(20,30,40,0.6)", width=0.8),
                                 name=f"M2D ピース{c_index + 1}（{len(ids)}枚）", hoverinfo="skip"))
    B = _upright(info["uv"][list(info["loop"]) + [info["loop"][0]]], info)
    fig.add_trace(go.Scatter(x=B[:, 0], y=B[:, 1], mode="lines", line=dict(color="#1b2a31", width=2),
                             name="Ωの境界", hoverinfo="skip"))
    C = _upright(info["uv"][info["corner_ids"]], info)
    for k in range(4):
        fig.add_trace(go.Scatter(x=[C[k, 0]], y=[C[k, 1]], mode="markers+text", text=[LABELS[k]],
                                 textposition="top right", textfont=dict(size=15, color=COLORS[k]),
                                 marker=dict(size=12, color=COLORS[k], line=dict(color="white", width=1)),
                                 name=LABELS[k]))
    long_side, short_side = max(info["width"], info["height"]), min(info["width"], info["height"])
    shape = "矩形" if info["is_rectangle"] else "矩形ではない"
    fig.update_layout(
        title=dict(text=f"Ω（図5の向き：長辺を縦に表示・表示のみ） {shape}、縦横比 1 : {long_side / short_side:.2f}", x=0.01),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="white", plot_bgcolor="white", height=640,
        legend=dict(orientation="h", y=-0.02))
    return fig
