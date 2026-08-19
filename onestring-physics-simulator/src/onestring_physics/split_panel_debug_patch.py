"""Experimental but semantic Split -> Panel patch.

This patch makes Split visible in the actual M2D/K2D geometry, not only in a
separate debug drawing.

Behavior:
- CSF split candidates become complete grid-line cuts.
- Split vertices are duplicated by the normal pipeline, producing disconnected
  face components.
- Those components are explicit Panels.
- M2D panel coordinates are translated into a non-overlapping packed layout.
- The pre-layout Omega coordinates are retained privately and are used for the
  M2D -> M3D inverse parameterization, so separating fabrication panels does not
  corrupt the c^{-1}: Omega -> S lookup.
- K2D starts from the separated M2D and is packed again by Panel after the
  optimizer, guaranteeing a clearly open Split boundary in the returned K2D.

Installed only by app_split_panels.py while the behavior is being validated.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _face_components_by_edge(faces: np.ndarray) -> list[np.ndarray]:
    """Return face components using shared *edges* only.

    Point contacts do not glue panels together. This is intentional: after a
    Split, two panels that merely meet at a duplicated/coincident corner remain
    separate fabrication pieces.
    """
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


def _panel_gap(vertices: np.ndarray, grid: Any | None) -> float:
    pts = np.asarray(vertices, dtype=float)
    span = np.ptp(pts[:, :2], axis=0) if len(pts) else np.asarray([1.0, 1.0])
    scale = max(float(np.max(span)), 1.0e-9)
    tile_size = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 0.0) if grid is not None else 0.0
    gap_size = max(float(getattr(grid, "gap_size", 0.0) or 0.0), 0.0) if grid is not None else 0.0
    # Deliberately obvious Split gap: at least two tile widths, and also visible
    # for normalized/small test meshes.
    return max(2.0 * tile_size, 8.0 * gap_size, 0.10 * scale, 1.0e-5)


def _pack_components(
    vertices: np.ndarray,
    component_vertices: list[np.ndarray],
    *,
    grid: Any | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Translate complete panels into a guaranteed non-overlapping layout.

    Translation is rigid per panel, so all within-panel edge lengths and quad
    shapes are unchanged. The z coordinate (if present) is untouched.
    """
    pts = np.asarray(vertices, dtype=float)
    if not len(pts) or not component_vertices:
        return pts.copy(), []

    xy = pts[:, :2]
    gap = _panel_gap(pts, grid)
    boxes: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for ids in component_vertices:
        p = xy[ids]
        lo = np.nanmin(p, axis=0)
        hi = np.nanmax(p, axis=0)
        boxes.append((lo, hi, float(hi[0] - lo[0]), float(hi[1] - lo[1])))

    total_area = sum(max(width, gap) * max(height, gap) for _, _, width, height in boxes)
    widest = max((width for _, _, width, _ in boxes), default=gap)
    target_row_width = max(2.5 * widest, np.sqrt(max(total_area, gap * gap)) * 1.8)

    offsets: list[np.ndarray] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0
    for lo, _hi, width, height in boxes:
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
        # A proper Split should duplicate all boundary vertices needed to make
        # components vertex-disjoint. If a point is still shared, leave its first
        # ownership in place so metrics expose that incomplete cut.
        free = ids[owner[ids] < 0]
        packed[free, :2] += offset[None, :]
        owner[free] = panel_id
    return packed, offsets


def _panel_metadata(faces: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    components = _face_components_by_edge(faces)
    vertices = _component_vertex_ids(faces, components)
    face_panel = np.full(len(np.asarray(faces, dtype=int)), -1, dtype=int)
    for panel_id, face_ids in enumerate(components):
        face_panel[face_ids] = panel_id
    return components, vertices, face_panel


def _copy_panel_attrs(source: Any, target: Any) -> None:
    for name in (
        "_split_panel_source_vertices",
        "_split_panel_face_components",
        "_split_panel_vertex_components",
        "_split_panel_face_ids",
        "_split_panel_offsets",
    ):
        if hasattr(source, name):
            try:
                setattr(target, name, getattr(source, name))
            except Exception:
                pass


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
    """Render the *actual* separated M2D/K2D coordinates panel by panel."""
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

    components, panel_vertices, _face_panel = _panel_metadata(faces)
    metrics = getattr(mesh_2d, "metrics", {}) or {}
    duplicated = int(metrics.get("csf_split_duplicated_vertex_count", 0) or 0)
    split_applied = bool(metrics.get("csf_split_applied", False))

    st.subheader("Split / Panel geometry")
    st.caption(
        f"Split applied={split_applied} | panels={len(components)} | "
        f"duplicated split vertices={duplicated}. "
        "These are the actual M2D/K2D coordinates used downstream, not display-only offsets."
    )
    if split_applied and len(components) <= 1:
        st.error(
            "Split was requested, but the resulting M2D still has only one edge-connected panel. "
            "The cut is incomplete and should not be treated as a valid panel split."
        )

    fig = go.Figure()
    for panel_id, face_ids in enumerate(components):
        edges = _unique_quad_edges(faces[face_ids])
        x, y = _edge_xy(start, edges)
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=3), name=f"Panel {panel_id} ({len(face_ids)} quads)"))
        ids = panel_vertices[panel_id]
        c = np.mean(start[ids, :2], axis=0)
        fig.add_annotation(x=float(c[0]), y=float(c[1]), text=f"P{panel_id}", showarrow=False)
    fig.update_layout(
        title="M2D after Split — actual separated panel geometry",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
        height=700,
        showlegend=True,
    )
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

    frame_count = 31
    panel_edges = [_unique_quad_edges(faces[c]) for c in components]
    initial_data = []
    for panel_id, edges in enumerate(panel_edges):
        x, y = _edge_xy(start, edges)
        initial_data.append(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=4), name=f"Panel {panel_id}"))
    anim = go.Figure(data=initial_data)
    frames = []
    for k in range(frame_count):
        a = k / max(1, frame_count - 1)
        xy = (1.0 - a) * start + a * final
        data = []
        for panel_id, edges in enumerate(panel_edges):
            x, y = _edge_xy(xy, edges)
            data.append(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=4), name=f"Panel {panel_id}"))
        frames.append(go.Frame(data=data, name=str(k)))
    anim.frames = frames
    anim.update_layout(
        title="Actual separated panels: M2D → K2D correspondence morph",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(constrain="domain"),
        height=760,
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]),
            dict(label="Reset", method="animate", args=[["0"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
        ])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=str(k)) for k in range(frame_count)])],
    )
    try:
        st.plotly_chart(anim, use_container_width=True)
    except TypeError:
        st.plotly_chart(anim)


def install_split_panel_debug(pipeline_module: Any, optimization_debug_module: Any) -> None:
    """Install full Split -> separated M2D/K2D panel semantics."""
    if getattr(pipeline_module, "_onestring_split_panel_debug_installed", False):
        return

    # 1) A Split is a full cut for this validation path. Localized slits can be
    # useful later for Seam design, but they do not necessarily create Panels.
    def full_grid_split_segments(
        parameterization: Any,
        csf: np.ndarray,
        threshold: float,
        split_lines: list[tuple],
        params: Any = None,
    ) -> list[tuple]:
        return [tuple(line[:2]) for line in (split_lines or [])]

    pipeline_module._localized_csf_split_segments = full_grid_split_segments

    # 2) Wrap M2D construction. The legacy function performs the topological
    # cut (vertex duplication). We then rigidly translate the resulting Panels.
    original_build_m2d = pipeline_module._build_m2d

    def build_m2d_with_real_panel_separation(grid: Any, domain: Any, params: Any = None):
        mesh = original_build_m2d(grid, domain, params)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        split_applied = bool(metrics.get("csf_split_applied", False))
        components, panel_vertices, face_panel = _panel_metadata(np.asarray(mesh.faces, dtype=int))
        if not split_applied or len(components) <= 1:
            metrics.update({
                "split_panel_geometry_separated": False,
                "split_panel_count": int(len(components)),
                "split_panel_separation_reason": "no effective multi-panel split",
            })
            mesh.metrics.update(metrics)
            return mesh

        source_vertices = np.asarray(mesh.vertices, dtype=float).copy()
        separated, offsets = _pack_components(source_vertices, panel_vertices, grid=getattr(mesh, "grid", grid))
        metrics.update({
            "split_panel_geometry_separated": True,
            "split_panel_count": int(len(components)),
            "split_panel_face_counts": [int(len(c)) for c in components],
            "split_panel_vertex_counts": [int(len(v)) for v in panel_vertices],
            "split_panel_offsets_xy": [[float(x) for x in off] for off in offsets],
            "split_panel_gap": float(_panel_gap(source_vertices, getattr(mesh, "grid", grid))),
            "split_panel_layout_model": "rigid per-panel translation after topological full-grid cut",
            "split_panel_source_uv_preserved_for_m3d": True,
        })
        out = pipeline_module._original.QuadMesh(
            separated,
            np.asarray(mesh.faces, dtype=int).copy(),
            mesh.grid,
            mesh.stage,
            metrics,
            list(getattr(mesh, "split_lines", [])),
        )
        setattr(out, "_split_panel_source_vertices", source_vertices)
        setattr(out, "_split_panel_face_components", components)
        setattr(out, "_split_panel_vertex_components", panel_vertices)
        setattr(out, "_split_panel_face_ids", face_panel)
        setattr(out, "_split_panel_offsets", offsets)
        return out

    pipeline_module._build_m2d = build_m2d_with_real_panel_separation

    # 3) M2D positions have now been translated for fabrication/layout. The
    # inverse parameterization must still query Omega using the pre-translation
    # coordinates. Build a temporary canonical M2D only for c^{-1}.
    original_lift_m2d_to_m3d = pipeline_module._lift_m2d_to_m3d

    def lift_m2d_to_m3d_with_canonical_uv(
        target: Any,
        mesh: Any,
        parameterization: Any,
        params: Any,
    ):
        source_vertices = getattr(mesh, "_split_panel_source_vertices", None)
        if source_vertices is None:
            return original_lift_m2d_to_m3d(target, mesh, parameterization, params)
        canonical = pipeline_module._original.QuadMesh(
            np.asarray(source_vertices, dtype=float).copy(),
            np.asarray(mesh.faces, dtype=int).copy(),
            mesh.grid,
            mesh.stage,
            dict(getattr(mesh, "metrics", {}) or {}),
            list(getattr(mesh, "split_lines", [])),
        )
        out, report = original_lift_m2d_to_m3d(target, canonical, parameterization, params)
        _copy_panel_attrs(mesh, out)
        out.metrics.update({
            "m3d_used_pre_panel_layout_uv": True,
            "m3d_panel_layout_translation_ignored_for_inverse_map": True,
        })
        return out, report

    pipeline_module._lift_m2d_to_m3d = lift_m2d_to_m3d_with_canonical_uv

    # 4) K2D receives the separated M2D. After optimization, apply only rigid
    # per-panel translations again so no optimizer regularizer can visually
    # re-close a Split boundary. No within-panel length/shape is changed.
    original_optimize_k2d = pipeline_module._optimize_k2d

    def optimize_k2d_with_real_panel_separation(
        mesh_2d: Any,
        mesh_3d: Any,
        params: Any,
        progress_callback: Any = None,
    ):
        result, report = original_optimize_k2d(
            mesh_2d,
            mesh_3d,
            params,
            progress_callback=progress_callback,
        )
        components, panel_vertices, face_panel = _panel_metadata(np.asarray(result.faces, dtype=int))
        if bool(getattr(mesh_2d, "metrics", {}).get("csf_split_applied", False)) and len(components) > 1:
            raw = np.asarray(result.vertices, dtype=float).copy()
            separated, offsets = _pack_components(raw, panel_vertices, grid=getattr(result, "grid", None))
            metrics = dict(getattr(result, "metrics", {}) or {})
            metrics.update({
                "k2d_split_panel_geometry_separated": True,
                "k2d_split_panel_count": int(len(components)),
                "k2d_split_panel_face_counts": [int(len(c)) for c in components],
                "k2d_split_panel_offsets_xy": [[float(x) for x in off] for off in offsets],
                "k2d_split_panel_layout_model": "rigid per-panel post-optimization translation",
            })
            rebuilt = pipeline_module._original.QuadMesh(
                separated,
                np.asarray(result.faces, dtype=int).copy(),
                result.grid,
                result.stage,
                metrics,
                list(getattr(result, "split_lines", [])),
            )
            setattr(rebuilt, "_split_panel_source_vertices", getattr(mesh_2d, "_split_panel_source_vertices", None))
            setattr(rebuilt, "_split_panel_face_components", components)
            setattr(rebuilt, "_split_panel_vertex_components", panel_vertices)
            setattr(rebuilt, "_split_panel_face_ids", face_panel)
            setattr(rebuilt, "_split_panel_offsets", offsets)
            result = rebuilt
        try:
            render_split_panel_correspondence(mesh_2d, result, mesh_3d)
        except Exception:
            pass
        return result, report

    pipeline_module._optimize_k2d = optimize_k2d_with_real_panel_separation

    # Disable the older correspondence wrapper's display-only renderer by
    # replacing its renderer with the actual-geometry renderer. If that wrapper
    # is already installed, it will call this function after our K2D wrapper.
    optimization_debug_module.render_k2d_correspondence_morph = render_split_panel_correspondence
    pipeline_module._onestring_split_panel_debug_installed = True


__all__ = [
    "install_split_panel_debug",
    "render_split_panel_correspondence",
]
