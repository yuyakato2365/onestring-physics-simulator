"""Experimental Split -> Panel debug patch.

Goals:
- Treat CSF split candidates as full grid-line cuts (not localized slits) so a
  split produces explicit disconnected panel components during debugging.
- Visualize M2D -> K2D with those panels packed apart, while leaving the actual
  M2D/K2D optimization coordinates untouched.

This module is intentionally installed by app_split_panels.py only. It does not
change the default app.py path until the behavior is validated.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _face_components_by_edge(faces: np.ndarray) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    if len(f) == 0:
        return []
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            a, b = ids[i], ids[(i + 1) % len(ids)]
            edge_to_faces[tuple(sorted((a, b)))].append(fi)
    adj: list[set[int]] = [set() for _ in range(len(f))]
    for touching in edge_to_faces.values():
        if len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = touching[i], touching[j]
                adj[a].add(b)
                adj[b].add(a)
    unseen = set(range(len(f)))
    out: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        q = deque([start])
        group = [start]
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    q.append(nxt)
                    group.append(nxt)
        out.append(np.asarray(group, dtype=int))
    out.sort(key=len, reverse=True)
    return out


def _component_vertex_ids(faces: np.ndarray, components: list[np.ndarray]) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    return [np.unique(f[c].reshape(-1)) for c in components]


def _pack_components(vertices: np.ndarray, component_vertices: list[np.ndarray], gap_fraction: float = 0.12) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return display-only packed coordinates and per-panel offsets.

    Actual optimization coordinates are never modified.
    """
    pts = np.asarray(vertices, dtype=float)
    if not len(pts) or not component_vertices:
        return pts.copy(), []
    global_span = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
    scale = max(float(np.nanmax(global_span)), 1.0e-9)
    gap = max(float(gap_fraction) * scale, 1.0e-6)

    boxes: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for ids in component_vertices:
        p = pts[ids]
        lo = np.nanmin(p, axis=0)
        hi = np.nanmax(p, axis=0)
        boxes.append((lo, hi, float(hi[0] - lo[0]), float(hi[1] - lo[1])))

    target_row_width = max(2.5 * scale, max((b[2] for b in boxes), default=scale))
    offsets: list[np.ndarray] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0
    for lo, hi, width, height in boxes:
        if cursor_x > 0.0 and cursor_x + width > target_row_width:
            cursor_x = 0.0
            cursor_y -= row_h + gap
            row_h = 0.0
        target_lo = np.asarray([cursor_x, cursor_y - height], dtype=float)
        offset = target_lo - lo
        offsets.append(offset)
        cursor_x += width + gap
        row_h = max(row_h, height)

    packed = pts.copy()
    owner = np.full(len(pts), -1, dtype=int)
    for panel_id, (ids, offset) in enumerate(zip(component_vertices, offsets)):
        # Full grid-line cuts should ensure distinct vertex IDs across panels.
        # If a point-only junction remains shared, keep the first ownership so
        # the debug view exposes that topology rather than silently duplicating.
        free = ids[owner[ids] < 0]
        packed[free] += offset
        owner[free] = panel_id
    return packed, offsets


def _unique_quad_edges(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=int)
    edges = []
    for face in f:
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edges.append(tuple(sorted((ids[i], ids[(i + 1) % len(ids)]))))
    return np.asarray(sorted(set(edges)), dtype=int) if edges else np.zeros((0, 2), dtype=int)


def _edge_xy(vertices: np.ndarray, edges: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    v = np.asarray(vertices, dtype=float)
    xs: list[float | None] = []
    ys: list[float | None] = []
    for a, b in np.asarray(edges, dtype=int):
        xs.extend([float(v[a, 0]), float(v[b, 0]), None])
        ys.extend([float(v[a, 1]), float(v[b, 1]), None])
    return xs, ys


def render_split_panel_correspondence(mesh_2d: Any, mesh_k2d: Any, mesh_k3d: Any | None = None) -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except Exception:
        return

    start = np.asarray(mesh_2d.vertices, dtype=float)
    final = np.asarray(mesh_k2d.vertices, dtype=float)
    faces = np.asarray(mesh_2d.faces, dtype=int)
    if start.ndim != 2 or final.ndim != 2 or len(start) != len(final) or not len(faces):
        return

    components = _face_components_by_edge(faces)
    panel_vertices = _component_vertex_ids(faces, components)
    start_packed, _ = _pack_components(start, panel_vertices)
    final_packed, _ = _pack_components(final, panel_vertices)

    metrics = getattr(mesh_2d, "metrics", {}) or {}
    duplicated = int(metrics.get("csf_split_duplicated_vertex_count", 0) or 0)
    split_applied = bool(metrics.get("csf_split_applied", False))
    st.subheader("Split / Panel layout debug")
    st.caption(
        f"Split applied={split_applied} | panels={len(components)} | "
        f"duplicated split vertices={duplicated}. "
        "Panel separation below is display-only; optimization coordinates are unchanged."
    )
    if split_applied and len(components) <= 1:
        st.warning(
            "Split vertices were duplicated, but the M2D topology is still one edge-connected panel. "
            "This means the cut did not fully separate panel groups."
        )

    fig = go.Figure()
    for panel_id, face_ids in enumerate(components):
        local_faces = faces[face_ids]
        edges = _unique_quad_edges(local_faces)
        x, y = _edge_xy(start_packed, edges)
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=f"Panel {panel_id} ({len(face_ids)} quads)"))
        ids = panel_vertices[panel_id]
        c = np.mean(start_packed[ids], axis=0)
        fig.add_annotation(x=float(c[0]), y=float(c[1]), text=f"P{panel_id}", showarrow=False)
    fig.update_layout(
        title="M2D after Split — panels packed apart",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
        height=700,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Packed correspondence morph. Each panel remains visually separated for the
    # entire animation, making cross-panel identity mistakes immediately visible.
    frame_count = 31
    panel_edges = [_unique_quad_edges(faces[c]) for c in components]
    initial_data = []
    for panel_id, edges in enumerate(panel_edges):
        x, y = _edge_xy(start_packed, edges)
        initial_data.append(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=4), name=f"Panel {panel_id}"))
    anim = go.Figure(data=initial_data)
    frames = []
    for k in range(frame_count):
        a = k / max(1, frame_count - 1)
        xy = (1.0 - a) * start_packed + a * final_packed
        data = []
        for panel_id, edges in enumerate(panel_edges):
            x, y = _edge_xy(xy, edges)
            data.append(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=4), name=f"Panel {panel_id}"))
        frames.append(go.Frame(data=data, name=str(k)))
    anim.frames = frames
    anim.update_layout(
        title="Separated panels: M2D → K2D correspondence morph",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
        height=760,
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]),
            dict(label="Reset", method="animate", args=[["0"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
        ])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=str(k)) for k in range(frame_count)])],
    )
    st.plotly_chart(anim, use_container_width=True)


def install_split_panel_debug(pipeline_module: Any, optimization_debug_module: Any) -> None:
    """Install the experimental full-cut + separated-panel debug behavior."""
    if getattr(pipeline_module, "_onestring_split_panel_debug_installed", False):
        return

    # Before Seam work, make Split semantically explicit: a candidate row/column
    # becomes a complete cut across the current M2D grid. This intentionally
    # disables the newer localized slit heuristic in this debug path.
    def full_grid_split_segments(parameterization: Any, csf: np.ndarray, threshold: float, split_lines: list[tuple], params: Any = None) -> list[tuple]:
        return [tuple(line[:2]) for line in (split_lines or [])]

    pipeline_module._localized_csf_split_segments = full_grid_split_segments
    optimization_debug_module.render_k2d_correspondence_morph = render_split_panel_correspondence
    pipeline_module._onestring_split_panel_debug_installed = True
