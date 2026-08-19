"""Semantic Split -> Panel patch used by app_split_panels.py.

In this validation path a Split is a true topological panel cut:
- faces are partitioned by each requested row/column split;
- every vertex id shared by the two sides is duplicated on one side;
- therefore the two sides cannot remain edge-connected by routing around a
  geometric split-line endpoint;
- resulting face components are explicit Panels;
- M2D/K2D panel geometry is rigidly packed apart for an obvious split gap;
- original pre-pack M2D UV coordinates are retained for M2D -> M3D inverse map.
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
            edge_to_faces[tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))].append(fi)
    adjacency: list[set[int]] = [set() for _ in range(len(f))]
    for touching in edge_to_faces.values():
        if len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = touching[i], touching[j]
                adjacency[a].add(b)
                adjacency[b].add(a)
    unseen = set(range(len(f)))
    components: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        group = [start]
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    group.append(nxt)
        components.append(np.asarray(group, dtype=int))
    components.sort(key=len, reverse=True)
    return components


def _component_vertex_ids(faces: np.ndarray, components: list[np.ndarray]) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    return [np.unique(f[face_ids].reshape(-1)) for face_ids in components]


def _panel_metadata(faces: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    components = _face_components_by_edge(faces)
    vertices = _component_vertex_ids(faces, components)
    face_panel = np.full(len(np.asarray(faces, dtype=int)), -1, dtype=int)
    for panel_id, face_ids in enumerate(components):
        face_panel[face_ids] = panel_id
    return components, vertices, face_panel


def _panel_gap(vertices: np.ndarray, grid: Any | None) -> float:
    pts = np.asarray(vertices, dtype=float)
    span = np.ptp(pts[:, :2], axis=0) if len(pts) else np.asarray([1.0, 1.0])
    scale = max(float(np.max(span)), 1.0e-9)
    tile_size = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 0.0) if grid is not None else 0.0
    gap_size = max(float(getattr(grid, "gap_size", 0.0) or 0.0), 0.0) if grid is not None else 0.0
    return max(2.0 * tile_size, 8.0 * gap_size, 0.10 * scale, 1.0e-5)


def _pack_components(
    vertices: np.ndarray,
    component_vertices: list[np.ndarray],
    *,
    grid: Any | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    pts = np.asarray(vertices, dtype=float).copy()
    if not len(pts) or not component_vertices:
        return pts, []
    gap = _panel_gap(pts, grid)
    boxes: list[tuple[np.ndarray, float, float]] = []
    for ids in component_vertices:
        local = pts[ids, :2]
        lo = np.min(local, axis=0)
        hi = np.max(local, axis=0)
        boxes.append((lo, float(hi[0] - lo[0]), float(hi[1] - lo[1])))
    total_area = sum(max(w, gap) * max(h, gap) for _, w, h in boxes)
    widest = max((w for _, w, _ in boxes), default=gap)
    target_row_width = max(2.5 * widest, np.sqrt(max(total_area, gap * gap)) * 1.8)
    offsets: list[np.ndarray] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for lo, width, height in boxes:
        if cursor_x > 0.0 and cursor_x + width > target_row_width:
            cursor_x = 0.0
            cursor_y -= row_height + gap
            row_height = 0.0
        target_lo = np.asarray([cursor_x, cursor_y - height], dtype=float)
        offsets.append(target_lo - lo)
        cursor_x += width + gap
        row_height = max(row_height, height)
    for ids, offset in zip(component_vertices, offsets):
        pts[ids, :2] += offset[None, :]
    return pts, offsets


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
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edges.add(tuple(sorted((ids[i], ids[(i + 1) % len(ids)]))))
    return np.asarray(sorted(edges), dtype=int) if edges else np.zeros((0, 2), dtype=int)


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
    components, panel_vertices, _ = _panel_metadata(faces)
    metrics = getattr(mesh_2d, "metrics", {}) or {}
    st.subheader("Split / Panel geometry")
    st.caption(
        f"Split applied={bool(metrics.get('csf_split_applied', False))} | "
        f"panels={len(components)} | "
        f"duplicated split vertices={int(metrics.get('csf_split_duplicated_vertex_count', 0) or 0)}. "
        "These are actual M2D/K2D coordinates, not display-only offsets."
    )
    if bool(metrics.get("csf_split_applied", False)) and len(components) <= 1:
        st.error("Split is still topologically incomplete: expected at least two edge-connected panels.")
    else:
        st.success(f"Topological Split verified: {len(components)} disconnected panels.")

    fig = go.Figure()
    for panel_id, face_ids in enumerate(components):
        edges = _unique_quad_edges(faces[face_ids])
        x, y = _edge_xy(start, edges)
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=3), name=f"Panel {panel_id} ({len(face_ids)} quads)"))
        center = np.mean(start[panel_vertices[panel_id], :2], axis=0)
        fig.add_annotation(x=float(center[0]), y=float(center[1]), text=f"P{panel_id}", showarrow=False)
    fig.update_layout(title="M2D after Split — actual separated panels", xaxis=dict(scaleanchor="y", scaleratio=1), height=700)
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

    panel_edges = [_unique_quad_edges(faces[c]) for c in components]
    frames = []
    for k in range(31):
        a = k / 30.0
        xy = (1.0 - a) * start + a * final
        frame_data = []
        for panel_id, edges in enumerate(panel_edges):
            x, y = _edge_xy(xy, edges)
            frame_data.append(go.Scattergl(x=x, y=y, mode="lines+markers", marker=dict(size=4), name=f"Panel {panel_id}"))
        frames.append(go.Frame(data=frame_data, name=str(k)))
    initial = frames[0].data if frames else []
    anim = go.Figure(data=initial, frames=frames)
    anim.update_layout(
        title="Separated panels: M2D → K2D correspondence morph",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=760,
        updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}])])],
    )
    try:
        st.plotly_chart(anim, use_container_width=True)
    except TypeError:
        st.plotly_chart(anim)


def install_split_panel_debug(pipeline_module: Any, optimization_debug_module: Any) -> None:
    if getattr(pipeline_module, "_onestring_split_panel_debug_installed", False):
        return

    # A Split candidate is treated as a complete row/column partition in this
    # validation path. Localized slit semantics belong to later Seam work.
    def full_grid_split_segments(parameterization: Any, csf: np.ndarray, threshold: float, split_lines: list[tuple], params: Any = None) -> list[tuple]:
        return [tuple(line[:2]) for line in (split_lines or [])]

    pipeline_module._localized_csf_split_segments = full_grid_split_segments

    # Critical change: do NOT only duplicate vertices geometrically lying on a
    # line. Partition faces by the split and duplicate every vertex id shared by
    # both face sets. The two sets are then vertex-disjoint, hence they cannot
    # remain edge-connected by going around the end of the geometric line.
    def guaranteed_panel_cut(vertices: np.ndarray, faces: np.ndarray, split_line: Any):
        verts = np.asarray(vertices, dtype=float).copy()
        out_faces = np.asarray(faces, dtype=int).copy()
        if len(out_faces) == 0:
            return verts, out_faces, 0
        snapped = pipeline_module._snap_split_line_to_mesh(verts, split_line)
        if snapped is None:
            return verts, out_faces, 0
        axis, value = pipeline_module._split_line_axis_value(snapped)
        coord = 1 if str(axis) == "row" else 0
        centroids = np.mean(verts[out_faces][:, :, coord], axis=1)
        negative_faces = np.flatnonzero(centroids < float(value))
        positive_faces = np.flatnonzero(centroids > float(value))
        if len(negative_faces) == 0 or len(positive_faces) == 0:
            return verts, out_faces, 0
        negative_vertices = set(int(v) for v in out_faces[negative_faces].reshape(-1))
        positive_vertices = set(int(v) for v in out_faces[positive_faces].reshape(-1))
        interface = sorted(negative_vertices & positive_vertices)
        if not interface:
            return verts, out_faces, 0
        duplicate_for: dict[int, int] = {}
        for vertex_id in interface:
            duplicate_for[vertex_id] = len(verts)
            verts = np.vstack([verts, verts[vertex_id]])
        for face_id in positive_faces:
            for local_id, vertex_id in enumerate(out_faces[face_id]):
                replacement = duplicate_for.get(int(vertex_id))
                if replacement is not None:
                    out_faces[face_id, local_id] = replacement
        return verts, out_faces, len(duplicate_for)

    pipeline_module._split_m2d_along_existing_grid_line = guaranteed_panel_cut

    original_build_m2d = pipeline_module._build_m2d

    def build_m2d_with_real_panel_separation(grid: Any, domain: Any, params: Any = None):
        mesh = original_build_m2d(grid, domain, params)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        components, panel_vertices, face_panel = _panel_metadata(np.asarray(mesh.faces, dtype=int))
        split_applied = bool(metrics.get("csf_split_applied", False))
        if not split_applied:
            metrics.update({"split_panel_geometry_separated": False, "split_panel_count": int(len(components)), "split_panel_separation_reason": "no split applied"})
            mesh.metrics.update(metrics)
            return mesh
        if len(components) <= 1:
            metrics.update({"split_panel_geometry_separated": False, "split_panel_count": 1, "split_panel_separation_reason": "topological cut unexpectedly remained connected"})
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
            "split_panel_layout_model": "guaranteed face-partition cut + rigid panel packing",
            "split_panel_source_uv_preserved_for_m3d": True,
        })
        out = pipeline_module._original.QuadMesh(separated, np.asarray(mesh.faces, dtype=int).copy(), mesh.grid, mesh.stage, metrics, list(getattr(mesh, "split_lines", [])))
        setattr(out, "_split_panel_source_vertices", source_vertices)
        setattr(out, "_split_panel_face_components", components)
        setattr(out, "_split_panel_vertex_components", panel_vertices)
        setattr(out, "_split_panel_face_ids", face_panel)
        setattr(out, "_split_panel_offsets", offsets)
        return out

    pipeline_module._build_m2d = build_m2d_with_real_panel_separation

    original_lift_m2d_to_m3d = pipeline_module._lift_m2d_to_m3d

    def lift_m2d_to_m3d_with_canonical_uv(target: Any, mesh: Any, parameterization: Any, params: Any):
        source_vertices = getattr(mesh, "_split_panel_source_vertices", None)
        if source_vertices is None:
            return original_lift_m2d_to_m3d(target, mesh, parameterization, params)
        canonical = pipeline_module._original.QuadMesh(np.asarray(source_vertices, dtype=float).copy(), np.asarray(mesh.faces, dtype=int).copy(), mesh.grid, mesh.stage, dict(getattr(mesh, "metrics", {}) or {}), list(getattr(mesh, "split_lines", [])))
        out, report = original_lift_m2d_to_m3d(target, canonical, parameterization, params)
        _copy_panel_attrs(mesh, out)
        out.metrics.update({"m3d_used_pre_panel_layout_uv": True, "m3d_panel_layout_translation_ignored_for_inverse_map": True})
        return out, report

    pipeline_module._lift_m2d_to_m3d = lift_m2d_to_m3d_with_canonical_uv

    original_optimize_k2d = pipeline_module._optimize_k2d

    def optimize_k2d_with_real_panel_separation(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
        result, report = original_optimize_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        components, panel_vertices, face_panel = _panel_metadata(np.asarray(result.faces, dtype=int))
        if bool(getattr(mesh_2d, "metrics", {}).get("csf_split_applied", False)) and len(components) > 1:
            separated, offsets = _pack_components(np.asarray(result.vertices, dtype=float), panel_vertices, grid=getattr(result, "grid", None))
            metrics = dict(getattr(result, "metrics", {}) or {})
            metrics.update({"k2d_split_panel_geometry_separated": True, "k2d_split_panel_count": int(len(components)), "k2d_split_panel_face_counts": [int(len(c)) for c in components], "k2d_split_panel_offsets_xy": [[float(x) for x in off] for off in offsets]})
            rebuilt = pipeline_module._original.QuadMesh(separated, np.asarray(result.faces, dtype=int).copy(), result.grid, result.stage, metrics, list(getattr(result, "split_lines", [])))
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

    # build_onestring_design in the wrapper ultimately executes functions from
    # the legacy module namespace, so patch those function slots too.
    pipeline_module._original._build_m2d = build_m2d_with_real_panel_separation
    pipeline_module._original._lift_m2d_to_m3d = lift_m2d_to_m3d_with_canonical_uv
    pipeline_module._original._optimize_k2d = optimize_k2d_with_real_panel_separation

    optimization_debug_module.render_k2d_correspondence_morph = render_split_panel_correspondence
    pipeline_module._onestring_split_panel_debug_installed = True


__all__ = ["install_split_panel_debug", "render_split_panel_correspondence"]
