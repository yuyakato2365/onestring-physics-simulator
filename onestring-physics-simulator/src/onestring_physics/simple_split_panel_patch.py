"""Simple deterministic Split -> Panel behavior for the validation app.

A Split does only three things:
1. snap a requested row/column to an existing grid coordinate;
2. duplicate the interface vertex ids on one side so the topology disconnects;
3. translate the resulting rigid panels by +/- gap/2 across each cut.

There is deliberately no bin-packing, panel reordering, or secondary topology
reconstruction in this module.  The pre-gap UV coordinates are retained for the
M2D -> M3D inverse map.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _normalize_line(line: Any) -> tuple[str, float] | None:
    try:
        axis = str(line[0]).strip().lower()
        value = float(line[1])
    except Exception:
        return None
    aliases = {"col": "column", "column": "column", "row": "row"}
    axis = aliases.get(axis)
    if axis is None or not np.isfinite(value):
        return None
    return axis, value


def _edge_components(faces: np.ndarray) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    if len(f) == 0:
        return []
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edge = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            edge_to_faces[edge].append(fi)
    adjacency: list[set[int]] = [set() for _ in range(len(f))]
    for touching in edge_to_faces.values():
        if len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = int(touching[i]), int(touching[j])
                adjacency[a].add(b)
                adjacency[b].add(a)
    unseen = set(range(len(f)))
    components: list[np.ndarray] = []
    while unseen:
        root = unseen.pop()
        queue = deque([root])
        group = [root]
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    group.append(nxt)
        components.append(np.asarray(group, dtype=int))
    components.sort(key=lambda c: int(np.min(c)) if len(c) else -1)
    return components


def _component_vertices(faces: np.ndarray, components: list[np.ndarray]) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    return [np.unique(f[c].reshape(-1)) for c in components]


def _snap_line(vertices: np.ndarray, faces: np.ndarray, line: Any) -> tuple[str, float] | None:
    parsed = _normalize_line(line)
    if parsed is None:
        return None
    axis, requested = parsed
    coord = 1 if axis == "row" else 0
    used = np.unique(np.asarray(faces, dtype=int).reshape(-1))
    values = np.unique(np.round(np.asarray(vertices, dtype=float)[used, coord], 12))
    if len(values) < 2:
        return None
    # End coordinates cannot separate faces, so prefer an internal grid value.
    candidates = values[1:-1] if len(values) > 2 else values
    if len(candidates) == 0:
        return None
    snapped = float(candidates[int(np.argmin(np.abs(candidates - requested)))])
    return axis, snapped


def _cut_once(vertices: np.ndarray, faces: np.ndarray, line: Any):
    """Disconnect the two face sets separated by one snapped grid line."""
    verts = np.asarray(vertices, dtype=float).copy()
    out_faces = np.asarray(faces, dtype=int).copy()
    snapped = _snap_line(verts, out_faces, line)
    if snapped is None or len(out_faces) == 0:
        return verts, out_faces, None
    axis, value = snapped
    coord = 1 if axis == "row" else 0
    span = max(float(np.ptp(verts[:, coord])), 1.0)
    tol = max(1e-10 * span, 1e-12)
    centroids = np.mean(verts[out_faces][:, :, coord], axis=1)
    negative = np.flatnonzero(centroids < value - tol)
    positive = np.flatnonzero(centroids > value + tol)
    if len(negative) == 0 or len(positive) == 0:
        return verts, out_faces, None

    neg_vertices = set(map(int, out_faces[negative].reshape(-1)))
    pos_vertices = set(map(int, out_faces[positive].reshape(-1)))
    interface = sorted(neg_vertices & pos_vertices)
    if not interface:
        return verts, out_faces, None
    if not np.all(np.abs(verts[interface, coord]-value) <= tol):
        return verts, out_faces, None

    before = len(_edge_components(out_faces))
    replacement: dict[int, int] = {}
    for old_id in interface:
        replacement[old_id] = len(verts)
        verts = np.vstack([verts, verts[old_id]])
    for fi in positive:
        for li, old_id in enumerate(out_faces[int(fi)]):
            new_id = replacement.get(int(old_id))
            if new_id is not None:
                out_faces[int(fi), li] = new_id
    after = len(_edge_components(out_faces))
    if after <= before:
        # Never return a fake Split.
        return np.asarray(vertices, dtype=float).copy(), np.asarray(faces, dtype=int).copy(), None

    return verts, out_faces, {
        "axis": axis,
        "requested_value": float(_normalize_line(line)[1]),
        "snapped_value": float(value),
        "duplicated_vertices": int(len(interface)),
        "components_before": int(before),
        "components_after": int(after),
    }


def _panel_gap(vertices: np.ndarray, grid: Any | None) -> float:
    pts = np.asarray(vertices, dtype=float)
    span = np.ptp(pts[:, :2], axis=0) if len(pts) else np.asarray([1.0, 1.0])
    scale = max(float(np.max(span)), 1e-9)
    tile_size = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 0.0) if grid is not None else 0.0
    gap_size = max(float(getattr(grid, "gap_size", 0.0) or 0.0), 0.0) if grid is not None else 0.0
    # Visible but modest: panels retain their original layout and only open at seams.
    return max(1.25 * tile_size, 4.0 * gap_size, 0.035 * scale, 1e-5)


def _open_split_gaps(
    canonical: np.ndarray,
    faces: np.ndarray,
    split_records: list[dict[str, Any]],
    grid: Any | None,
):
    """Rigidly translate each panel; never pack/reorder panels."""
    source = np.asarray(canonical, dtype=float)
    out = source.copy()
    components = _edge_components(faces)
    panel_vertices = _component_vertices(faces, components)
    gap = _panel_gap(source, grid)
    offsets: list[np.ndarray] = []
    for ids in panel_vertices:
        center = np.mean(source[ids, :2], axis=0)
        offset = np.zeros(2, dtype=float)
        for record in split_records:
            affected = record.get('face_ids')
            if affected is not None and not np.intersect1d(ids, np.unique(faces[affected])).size:
                continue
            axis = str(record["axis"])
            value = float(record["snapped_value"])
            coord = 1 if axis == "row" else 0
            if center[coord] < value:
                offset[coord] -= 0.5 * gap
            elif center[coord] > value:
                offset[coord] += 0.5 * gap
        out[ids, :2] += offset[None, :]
        offsets.append(offset)
    return out, components, panel_vertices, offsets, gap


def apply_paper_split(pipeline_module, mesh, domain, params=None):
    """The only cut owner: final cropped M2D -> measured hierarchical parts."""
    from .paper_mesh_splitting import split_mesh
    canonical, faces, records, measured = split_mesh(mesh.vertices, mesh.faces, domain, params)
    metrics = dict(mesh.metrics)
    for key in list(metrics):
        if key.startswith(('csf_split_', 'split_', 'simple_split_', 'max_csf_', 'number_of_splits', 'raw_split_')):
            metrics.pop(key)
    metrics.update(measured)
    from .split_geometry_constraints import paired_vertices
    metrics['split_vertex_pairs'] = list(mesh.metrics.get('split_vertex_pairs', [])) + paired_vertices(canonical, faces, measured['split_boundary_pairs'])
    metrics.update(
        csf_split_parameterization_method=str(getattr(domain.parameterization, 'method', 'unknown')),
        m2d_connected_component_count_before_csf_split=len(_edge_components(mesh.faces)),
        split_diagnostics=dict(domain.paper_split_diagnostics),
    )
    if getattr(domain, 'reference_mode', False):
        # The reference-BFF builder also publishes this legacy report. Keep
        # it in sync instead of leaving a pending/zero-split result behind.
        domain.reference_split_diagnostics = dict(domain.paper_split_diagnostics)
    # Work in grid coordinates when opening a rotated grid's seams.
    angle = np.deg2rad(float(getattr(domain, 'reference_grid_rotation_degrees', 0.)))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    grid_vertices = canonical.copy(); grid_vertices[:, :2] = canonical[:, :2] @ rotation
    separated, components, panel_vertices, offsets, gap = _open_split_gaps(grid_vertices, faces, records, mesh.grid)
    separated[:, :2] = separated[:, :2] @ rotation.T
    offsets = [offset @ rotation.T for offset in offsets]
    metrics.update(paper_split_finalized=True, simple_split_active=True, simple_split_records=records,
                   split_panel_count=len(components), split_panel_geometry_separated=len(components)>1,
                   split_panel_face_counts=[len(c) for c in components], split_panel_offsets_xy=[o.tolist() for o in offsets],
                   split_panel_gap=gap if records else 0., split_panel_layout_model='component-local seam gaps',
                   final_split_panel_count=len(components), paper_style_complete_split=bool(records))
    if hasattr(pipeline_module, '_m2d_audit_metrics'):
        metrics.update(pipeline_module._m2d_audit_metrics(separated, faces, suffix='after_split'))
    out = _make_quadmesh(pipeline_module, mesh, separated, faces, metrics)
    _copy_attrs(mesh, out)
    for name in ('_optcuts_test_boundary_clipped', '_optcuts_test_face_sources', '_optcuts_test_face_uv'):
        if hasattr(mesh, name):
            setattr(out, name, getattr(mesh, name))
    if hasattr(mesh, '_polygon_faces'):
        out._polygon_faces = faces.tolist()
    out.split_lines = list(domain.split_lines)
    out._split_panel_source_vertices = canonical
    out._split_panel_face_components = components
    out._split_panel_vertex_components = panel_vertices
    out._split_panel_offsets = offsets
    out._split_panel_records = records
    return out


def _copy_attrs(source: Any, target: Any) -> None:
    for name in (
        "_split_panel_source_vertices",
        "_split_panel_face_components",
        "_split_panel_vertex_components",
        "_split_panel_offsets",
        "_split_panel_records",
    ):
        if hasattr(source, name):
            try:
                setattr(target, name, getattr(source, name))
            except Exception:
                pass


def _make_quadmesh(pipeline_module: Any, mesh: Any, vertices: np.ndarray, faces: np.ndarray, metrics: dict[str, Any]):
    cls = getattr(getattr(pipeline_module, "_original", None), "QuadMesh", None)
    if cls is None:
        cls = type(mesh)
    return cls(
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
        mesh.grid,
        mesh.stage,
        metrics,
        list(getattr(mesh, "split_lines", [])),
    )


def render_split_panel_correspondence(mesh_2d: Any, mesh_k2d: Any, mesh_k3d: Any | None = None) -> None:
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except Exception:
        return
    faces = np.asarray(mesh_2d.faces, dtype=int)
    start = np.asarray(mesh_2d.vertices, dtype=float)
    final = np.asarray(mesh_k2d.vertices, dtype=float)
    components = _edge_components(faces)
    panel_vertices = _component_vertices(faces, components)
    metrics = getattr(mesh_2d, "metrics", {}) or {}
    st.subheader("Split / Panel geometry")
    st.caption(
        f"Split applied={bool(metrics.get('csf_split_applied', False))} | "
        f"panels={len(components)} | "
        f"simple cut records={len(metrics.get('simple_split_records', []) or [])}. "
        "Panels keep their original arrangement; only seam gaps are opened."
    )
    if bool(metrics.get("csf_split_applied", False)) and len(components) <= 1:
        st.error("Split is topologically incomplete: no disconnected panel was produced.")
    elif bool(metrics.get("csf_split_applied", False)):
        st.success(f"Topological Split verified: {len(components)} disconnected panels.")

    fig = go.Figure()
    for panel_id, ids in enumerate(panel_vertices):
        # Draw all quad edges belonging to this panel.
        face_ids = components[panel_id]
        xs: list[float | None] = []
        ys: list[float | None] = []
        seen: set[tuple[int, int]] = set()
        for face in faces[face_ids]:
            q = [int(v) for v in face]
            for i in range(len(q)):
                edge = tuple(sorted((q[i], q[(i + 1) % len(q)])))
                if edge in seen:
                    continue
                seen.add(edge)
                a, b = edge
                xs.extend([float(start[a, 0]), float(start[b, 0]), None])
                ys.extend([float(start[a, 1]), float(start[b, 1]), None])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines+markers", marker=dict(size=3), name=f"Panel {panel_id}"))
        center = np.mean(start[ids, :2], axis=0)
        fig.add_annotation(x=float(center[0]), y=float(center[1]), text=f"P{panel_id}", showarrow=False)
    fig.update_layout(title="M2D after Split — original layout + seam gaps", xaxis=dict(scaleanchor="y", scaleratio=1), height=700)
    try:
        st.plotly_chart(fig, config={"responsive": True})
    except Exception:
        st.plotly_chart(fig)


def install_simple_split_panel_patch(pipeline_module: Any, optimization_debug_module: Any) -> None:
    if getattr(pipeline_module, "_simple_split_panel_patch_installed", False):
        return

    # Treat generated Split lines as complete row/column cuts, not localized slits.
    def full_grid_segments(parameterization: Any, csf: np.ndarray, threshold: float, split_lines: list[tuple], params: Any = None):
        return [tuple(line[:2]) for line in (split_lines or [])]

    pipeline_module._localized_csf_split_segments = full_grid_segments

    def upstream_cut(vertices: np.ndarray, faces: np.ndarray, split_line: Any):
        v2, f2, record = _cut_once(vertices, faces, split_line)
        return v2, f2, int(record["duplicated_vertices"]) if record is not None else 0

    pipeline_module._split_m2d_along_existing_grid_line = upstream_cut
    base_build = pipeline_module._build_m2d

    def build_m2d_simple_split(grid: Any, domain: Any, params: Any = None):
        import copy
        uncut = copy.copy(domain)
        uncut.split_lines = []
        uncut.localized_split_segments = []
        uncut._paper_split_deferred = True
        mesh = base_build(grid, uncut, params)
        if getattr(domain, "paper_split_policy", False):
            return apply_paper_split(pipeline_module, mesh, domain, params)
        # Compatibility for callers supplying explicit split lines without a
        # parameterization. Keep one cut owner and never re-weld source seams.
        vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
        records = []
        for line in list(getattr(domain, "split_lines", []) or []):
            vertices, faces, record = _cut_once(vertices, faces, line)
            if record is not None:
                records.append(record)
        canonical = vertices.copy()
        vertices, components, panel_vertices, offsets, gap = _open_split_gaps(vertices, faces, records, mesh.grid)
        metrics = dict(mesh.metrics)
        metrics.update(csf_split_applied=bool(records), number_of_splits=len(records),
                       split_locations=[[r['axis'], r['snapped_value']] for r in records],
                       simple_split_records=records, split_panel_count=len(components),
                       csf_split_duplicated_vertex_count=sum(r['duplicated_vertices'] for r in records))
        out = _make_quadmesh(pipeline_module, mesh, vertices, faces, metrics)
        out._split_panel_source_vertices = canonical
        out._split_panel_face_components = components
        out._split_panel_vertex_components = panel_vertices
        out._split_panel_offsets = offsets
        out._split_panel_records = records
        return out

    pipeline_module._build_m2d = build_m2d_simple_split

    base_lift = pipeline_module._lift_m2d_to_m3d

    def lift_with_canonical_uv(target: Any, mesh: Any, parameterization: Any, params: Any):
        from .split_geometry_constraints import propagate
        canonical = getattr(mesh, "_split_panel_source_vertices", None)
        if canonical is None:
            out, report = base_lift(target, mesh, parameterization, params)
            propagate(mesh, out)
            return out, report
        canonical_mesh = _make_quadmesh(
            pipeline_module,
            mesh,
            np.asarray(canonical, dtype=float).copy(),
            np.asarray(mesh.faces, dtype=int).copy(),
            dict(getattr(mesh, "metrics", {}) or {}),
        )
        out, report = base_lift(target, canonical_mesh, parameterization, params)
        _copy_attrs(mesh, out)
        propagate(mesh, out)
        out.metrics.update({"m3d_used_pre_panel_layout_uv": True, "m3d_panel_gap_ignored_for_inverse_map": True})
        return out, report

    pipeline_module._lift_m2d_to_m3d = lift_with_canonical_uv

    base_k2d = pipeline_module._optimize_k2d

    def optimize_k2d_keep_panel_layout(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
        result, report = base_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        components = _edge_components(np.asarray(result.faces, dtype=int))
        panel_vertices = _component_vertices(np.asarray(result.faces, dtype=int), components)
        desired_components = getattr(mesh_2d, "_split_panel_face_components", None)
        desired_vertices = getattr(mesh_2d, "_split_panel_vertex_components", None)
        if len(components) > 1 and desired_components is not None and desired_vertices is not None and len(components) == len(desired_vertices):
            verts = np.asarray(result.vertices, dtype=float).copy()
            desired_xy = np.asarray(mesh_2d.vertices, dtype=float)
            # Keep each K2D panel's optimized local shape, but translate it so its
            # centroid matches the corresponding M2D panel. No bin-packing.
            for ids_result, ids_desired in zip(panel_vertices, desired_vertices):
                current_center = np.mean(verts[ids_result, :2], axis=0)
                desired_center = np.mean(desired_xy[ids_desired, :2], axis=0)
                verts[ids_result, :2] += (desired_center - current_center)[None, :]
            metrics = dict(getattr(result, "metrics", {}) or {})
            metrics.update({
                "k2d_split_panel_geometry_separated": True,
                "k2d_split_panel_count": int(len(components)),
                "k2d_split_panel_layout_model": "panel centroids aligned to M2D seam-gap layout",
            })
            result = _make_quadmesh(pipeline_module, result, verts, np.asarray(result.faces, dtype=int).copy(), metrics)
            _copy_attrs(mesh_2d, result)
        try:
            render_split_panel_correspondence(mesh_2d, result, mesh_3d)
        except Exception:
            pass
        return result, report

    pipeline_module._optimize_k2d = optimize_k2d_keep_panel_layout

    # Patch the exact legacy build globals, but do not permanently stack wrappers
    # on the backup module across reloads. The validation launcher freezes the
    # pipeline reload before the legacy app executes.
    build_candidates = (
        getattr(pipeline_module, "build_onestring_design", None),
        getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline_module, "_original", None), "build_onestring_design", None),
    )
    for fn in build_candidates:
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_simple_split
            glb["_lift_m2d_to_m3d"] = lift_with_canonical_uv
            glb["_optimize_k2d"] = optimize_k2d_keep_panel_layout

    optimization_debug_module.render_k2d_correspondence_morph = render_split_panel_correspondence
    pipeline_module._simple_split_panel_patch_installed = True


__all__ = ["install_simple_split_panel_patch", "render_split_panel_correspondence"]
