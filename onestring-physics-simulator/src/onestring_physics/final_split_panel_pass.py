"""Paper-style final Split -> Panel topology pass.

The earlier experimental Split code can leave local slits or centroid-based
partitions in M2D.  This pass deliberately rebuilds the unsplit coincident-grid
topology, then applies each requested Split as a *complete* internal row/column
seam on exactly one current edge-connected component.

A valid Split must:
- snap to an existing internal grid coordinate;
- be a connected chain of existing quad edges;
- run boundary-to-boundary through the selected component;
- duplicate only vertices on that seam on one side;
- increase the edge-connected component count by exactly one.

This matches the OneString paper's complete grid-direction bisection semantics
much more closely than a localized slit.  The pre-pack coordinates are retained
on ``_split_panel_source_vertices`` for the existing M2D -> M3D inverse map.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _split_axis_value(line: Any) -> tuple[str, float] | None:
    try:
        axis = str(line[0])
        value = float(line[1])
    except Exception:
        return None
    if axis not in {"row", "column"} or not np.isfinite(value):
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
    comps: list[np.ndarray] = []
    while unseen:
        root = unseen.pop()
        q = deque([root])
        group = [root]
        while q:
            cur = q.popleft()
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    q.append(nxt)
                    group.append(nxt)
        comps.append(np.asarray(group, dtype=int))
    comps.sort(key=len, reverse=True)
    return comps


def _weld_exact_coincident_vertices(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Undo earlier Split-only duplicate IDs while preserving grid geometry."""
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int).copy()
    if len(verts) == 0:
        return verts.copy(), f, 0
    scale = max(float(np.max(np.ptp(verts[:, :2], axis=0))), 1.0)
    tol = max(1e-11 * scale, 1e-12)
    key_to_new: dict[tuple[int, ...], int] = {}
    old_to_new = np.empty(len(verts), dtype=int)
    new_vertices: list[np.ndarray] = []
    for old_id, p in enumerate(verts):
        key = tuple(np.rint(np.asarray(p, dtype=float) / tol).astype(np.int64).tolist())
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(new_vertices)
            key_to_new[key] = new_id
            new_vertices.append(np.asarray(p, dtype=float).copy())
        old_to_new[old_id] = new_id
    f = old_to_new[f]
    return np.asarray(new_vertices, dtype=float), f, int(len(verts) - len(new_vertices))


def _component_edge_data(faces: np.ndarray, face_ids: np.ndarray):
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in np.asarray(face_ids, dtype=int):
        ids = [int(v) for v in faces[int(fi)]]
        for i in range(len(ids)):
            edge_to_faces[tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))].append(int(fi))
    boundary_edges = {e for e, inc in edge_to_faces.items() if len(inc) == 1}
    boundary_vertices = {v for e in boundary_edges for v in e}
    return edge_to_faces, boundary_edges, boundary_vertices


def _internal_grid_values(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, coord: int) -> np.ndarray:
    vids = np.unique(np.asarray(faces, dtype=int)[np.asarray(face_ids, dtype=int)].reshape(-1))
    vals = np.unique(np.round(np.asarray(vertices, dtype=float)[vids, coord], 12))
    if len(vals) <= 2:
        return np.zeros(0, dtype=float)
    return vals[1:-1].astype(float)


def _complete_cut_candidate(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, axis: str, requested: float):
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    comp = np.asarray(face_ids, dtype=int)
    coord = 1 if axis == "row" else 0
    internal = _internal_grid_values(verts, f, comp, coord)
    if len(internal) == 0:
        return None
    value = float(internal[int(np.argmin(np.abs(internal - float(requested))))])
    span = max(float(np.ptp(verts[:, coord])), 1.0)
    tol = max(1e-9 * span, 1e-11)

    centroids = np.mean(verts[f[comp]][:, :, coord], axis=1)
    neg = comp[centroids < value - tol]
    pos = comp[centroids > value + tol]
    if len(neg) == 0 or len(pos) == 0:
        return None

    edge_to_faces, _boundary_edges, boundary_vertices = _component_edge_data(f, comp)
    neg_set, pos_set = set(map(int, neg)), set(map(int, pos))
    seam_edges: list[tuple[int, int]] = []
    for edge, inc in edge_to_faces.items():
        if len(inc) != 2:
            continue
        a, b = edge
        if not (abs(float(verts[a, coord]) - value) <= tol and abs(float(verts[b, coord]) - value) <= tol):
            continue
        sides = {(-1 if fi in neg_set else 1 if fi in pos_set else 0) for fi in inc}
        if -1 in sides and 1 in sides:
            seam_edges.append(edge)
    if not seam_edges:
        return None

    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in seam_edges:
        graph[a].add(b)
        graph[b].add(a)
    start = next(iter(graph))
    seen = {start}
    q = deque([start])
    while q:
        v = q.popleft()
        for n in graph[v]:
            if n not in seen:
                seen.add(n)
                q.append(n)
    if len(seen) != len(graph):
        return None
    degrees = {v: len(n) for v, n in graph.items()}
    endpoints = [v for v, d in degrees.items() if d == 1]
    if len(endpoints) != 2 or any(d not in {1, 2} for d in degrees.values()):
        return None
    if not all(v in boundary_vertices for v in endpoints):
        return None

    seam_vertices = sorted(graph.keys())
    neg_vertices = set(map(int, f[neg].reshape(-1)))
    pos_vertices = set(map(int, f[pos].reshape(-1)))
    shared = neg_vertices & pos_vertices
    if shared != set(seam_vertices):
        return None

    score = (abs(value - float(requested)), -len(comp))
    return {
        "axis": axis,
        "requested": float(requested),
        "value": value,
        "face_ids": comp,
        "negative_faces": neg,
        "positive_faces": pos,
        "seam_edges": seam_edges,
        "seam_vertices": seam_vertices,
        "endpoints": endpoints,
        "score": score,
    }


def _apply_complete_cut(vertices: np.ndarray, faces: np.ndarray, candidate: dict[str, Any]):
    verts = np.asarray(vertices, dtype=float).copy()
    out_faces = np.asarray(faces, dtype=int).copy()
    replacement: dict[int, int] = {}
    for vid in candidate["seam_vertices"]:
        replacement[int(vid)] = len(verts)
        verts = np.vstack([verts, verts[int(vid)]])
    for fi in np.asarray(candidate["positive_faces"], dtype=int):
        for li, vid in enumerate(out_faces[int(fi)]):
            if int(vid) in replacement:
                out_faces[int(fi), li] = replacement[int(vid)]
    return verts, out_faces, len(replacement)


def _complete_cut_once(vertices: np.ndarray, faces: np.ndarray, axis: str, requested: float):
    before_components = _edge_components(faces)
    candidates = []
    for comp in before_components:
        cand = _complete_cut_candidate(vertices, faces, comp, axis, requested)
        if cand is not None:
            candidates.append(cand)
    if not candidates:
        return np.asarray(vertices, dtype=float).copy(), np.asarray(faces, dtype=int).copy(), None
    candidates.sort(key=lambda c: c["score"])
    cand = candidates[0]
    verts2, faces2, added = _apply_complete_cut(vertices, faces, cand)
    after_components = _edge_components(faces2)
    if len(after_components) != len(before_components) + 1:
        return np.asarray(vertices, dtype=float).copy(), np.asarray(faces, dtype=int).copy(), None
    record = {
        "axis": str(axis),
        "requested_value": float(requested),
        "snapped_value": float(cand["value"]),
        "duplicated_vertices": int(added),
        "seam_edge_count": int(len(cand["seam_edges"])),
        "seam_vertex_count": int(len(cand["seam_vertices"])),
        "boundary_endpoints": [int(v) for v in cand["endpoints"]],
        "components_before": int(len(before_components)),
        "components_after": int(len(after_components)),
        "complete_boundary_to_boundary": True,
    }
    return verts2, faces2, record


def _panel_vertices(faces: np.ndarray, components: list[np.ndarray]) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    return [np.unique(f[c].reshape(-1)) for c in components]


def _pack_panels(vertices: np.ndarray, panel_vertices: list[np.ndarray], grid: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    pts = np.asarray(vertices, dtype=float).copy()
    if len(panel_vertices) <= 1:
        return pts, [np.zeros(2, dtype=float)] if panel_vertices else []
    xy = pts[:, :2]
    span = np.ptp(xy, axis=0) if len(xy) else np.asarray([1.0, 1.0])
    scale = max(float(np.max(span)), 1e-9)
    tile_size = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 0.0)
    gap_size = max(float(getattr(grid, "gap_size", 0.0) or 0.0), 0.0)
    gap = max(2.0 * tile_size, 8.0 * gap_size, 0.12 * scale, 1e-5)
    boxes = []
    for ids in panel_vertices:
        p = xy[ids]
        lo, hi = np.min(p, axis=0), np.max(p, axis=0)
        boxes.append((lo, float(hi[0] - lo[0]), float(hi[1] - lo[1])))
    max_width = max((b[1] for b in boxes), default=scale)
    target_width = max(2.5 * max_width, 2.0 * scale)
    cursor_x = cursor_y = row_h = 0.0
    offsets = []
    for lo, width, height in boxes:
        if cursor_x > 0.0 and cursor_x + width > target_width:
            cursor_x = 0.0
            cursor_y -= row_h + gap
            row_h = 0.0
        target_lo = np.asarray([cursor_x, cursor_y - height], dtype=float)
        off = target_lo - lo
        offsets.append(off)
        cursor_x += width + gap
        row_h = max(row_h, height)
    for ids, off in zip(panel_vertices, offsets):
        pts[ids, :2] += off[None, :]
    return pts, offsets


def install_final_split_panel_pass(pipeline_module: Any) -> None:
    if getattr(pipeline_module, "_final_split_panel_pass_installed", False):
        return
    previous_build = pipeline_module._build_m2d

    def build_m2d_final_split(grid: Any, domain: Any, params: Any = None):
        mesh = previous_build(grid, domain, params)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        if not bool(metrics.get("csf_split_applied", False)):
            metrics["final_split_panel_pass_applied"] = False
            mesh.metrics.update(metrics)
            return mesh

        canonical = getattr(mesh, "_split_panel_source_vertices", None)
        if canonical is None:
            canonical = np.asarray(mesh.vertices, dtype=float).copy()
        else:
            canonical = np.asarray(canonical, dtype=float).copy()
        faces = np.asarray(mesh.faces, dtype=int).copy()

        # Remove any earlier local/centroid Split duplicate IDs, then recut from
        # the clean regular-grid topology using complete seams only.
        canonical, faces, rewelded = _weld_exact_coincident_vertices(canonical, faces)
        initial_components = _edge_components(faces)

        raw_lines = list(metrics.get("split_locations", []) or [])
        if not raw_lines:
            raw_lines = list(getattr(mesh, "split_lines", []) or [])
        parsed = [p for p in (_split_axis_value(line) for line in raw_lines) if p is not None]

        per_line = []
        rejected = []
        for axis, value in parsed:
            canonical2, faces2, record = _complete_cut_once(canonical, faces, axis, value)
            if record is None:
                rejected.append({"axis": axis, "requested_value": float(value), "reason": "no valid boundary-to-boundary internal grid seam"})
                continue
            canonical, faces = canonical2, faces2
            per_line.append(record)

        components = _edge_components(faces)
        if parsed and not per_line:
            raise RuntimeError(
                "PAPER_STYLE_SPLIT_FAILED: split lines were requested but no complete boundary-to-boundary grid seam could be built. "
                f"requested={parsed}, rejected={rejected}"
            )
        if len(components) <= len(initial_components) and parsed:
            raise RuntimeError("PAPER_STYLE_SPLIT_FAILED: final topology did not gain a disconnected panel.")

        pverts = _panel_vertices(faces, components)
        packed, offsets = _pack_panels(canonical, pverts, getattr(mesh, "grid", grid))
        metrics.update({
            "final_split_panel_pass_applied": bool(per_line),
            "paper_style_complete_split": True,
            "paper_style_localized_split_disabled_in_final_topology": True,
            "paper_style_rewelded_earlier_split_duplicates": int(rewelded),
            "final_split_panel_count": int(len(components)),
            "final_split_panel_face_counts": [int(len(c)) for c in components],
            "final_split_panel_vertex_counts": [int(len(v)) for v in pverts],
            "final_split_panel_added_duplicate_vertices": int(sum(r["duplicated_vertices"] for r in per_line)),
            "final_split_panel_per_line": per_line,
            "final_split_panel_rejected_lines": rejected,
            "split_panel_geometry_separated": bool(len(components) > len(initial_components)),
            "split_panel_count": int(len(components)),
            "split_panel_offsets_xy": [[float(x) for x in off] for off in offsets],
            "split_panel_layout_model": "paper-style complete boundary-to-boundary grid seam + rigid panel packing",
            "split_panel_source_uv_preserved_for_m3d": True,
        })

        out = pipeline_module._original.QuadMesh(packed, faces, mesh.grid, mesh.stage, metrics, list(getattr(mesh, "split_lines", [])))
        setattr(out, "_split_panel_source_vertices", canonical.copy())
        setattr(out, "_split_panel_face_components", components)
        setattr(out, "_split_panel_vertex_components", pverts)
        face_panel = np.full(len(faces), -1, dtype=int)
        for pid, face_ids in enumerate(components):
            face_panel[face_ids] = int(pid)
        setattr(out, "_split_panel_face_ids", face_panel)
        setattr(out, "_split_panel_offsets", offsets)
        return out

    pipeline_module._build_m2d = build_m2d_final_split
    pipeline_module._original._build_m2d = build_m2d_final_split
    for fn in (
        getattr(pipeline_module, "build_onestring_design", None),
        getattr(pipeline_module._original, "build_onestring_design", None),
        getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_final_split
    pipeline_module._final_split_panel_pass_installed = True


__all__ = ["install_final_split_panel_pass"]
