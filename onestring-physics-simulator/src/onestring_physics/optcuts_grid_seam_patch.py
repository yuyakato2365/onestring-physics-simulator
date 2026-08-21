"""Convert OptCuts UV seams into continuous zero-width cuts on the M2D grid.

This module is opt-in and is installed only by ``app_optcuts.py``.  It does not
change the stable row/column Simple Split implementation.

The adapter has two stages:
1. recover paired internal OptCuts seam edges from the 3D-face / UV-face index
   correspondence and carry a representative connected seam graph through the
   S -> Omega -> M2D boundary;
2. on M2D, heal only the narrow strip of grid cells removed around that seam,
   snap the connected seam graph to the existing grid-edge graph, and disconnect
   topology by duplicating shared vertex ids.  No cells are deleted to create
   seam width and no geometric gap is opened.
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
import heapq
from typing import Any

import numpy as np


def _surface_edge_uv_copies(parameterization: Any):
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    if len(sf) != len(uf):
        return []
    incidences: dict[tuple[int, int], list[dict[int, int]]] = defaultdict(list)
    for face3, face2 in zip(sf, uf):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va, vb = int(face3[a]), int(face3[b])
            key = tuple(sorted((va, vb)))
            incidences[key].append({va: int(face2[a]), vb: int(face2[b])})

    seam_edges = []
    for edge, copies in incidences.items():
        # An internal manifold edge has two incident triangles.  It is a cut if
        # the two triangles use different UV vertex ids for either 3D endpoint.
        if len(copies) != 2:
            continue
        a, b = edge
        c0, c1 = copies
        if a not in c0 or b not in c0 or a not in c1 or b not in c1:
            continue
        if c0[a] == c1[a] and c0[b] == c1[b]:
            continue
        pa = 0.5 * (uv[c0[a]] + uv[c1[a]])
        pb = 0.5 * (uv[c0[b]] + uv[c1[b]])
        seam_edges.append((int(a), int(b), np.asarray(pa, float), np.asarray(pb, float)))
    return seam_edges


def _extract_connected_seam_payload(parameterization: Any) -> dict[str, Any]:
    seam_edges = _surface_edge_uv_copies(parameterization)
    if not seam_edges:
        return {"segments": np.zeros((0, 2, 2), dtype=float), "nodes": {}, "edges": []}

    accum: dict[int, list[np.ndarray]] = defaultdict(list)
    edges: list[tuple[int, int]] = []
    for a, b, pa, pb in seam_edges:
        accum[a].append(pa)
        accum[b].append(pb)
        edges.append((a, b))
    nodes = {int(k): np.mean(np.asarray(v, dtype=float), axis=0) for k, v in accum.items()}
    segments = np.asarray([[nodes[a], nodes[b]] for a, b in edges], dtype=float)
    return {"segments": segments, "nodes": nodes, "edges": edges}


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-20:
        return np.linalg.norm(pts - a[None, :], axis=1)
    t = np.clip(((pts - a[None, :]) @ ab) / denom, 0.0, 1.0)
    q = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - q, axis=1)


def _edge_components_with_cuts(faces: np.ndarray, cut_edges: set[tuple[int, int]]) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edge = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            edge_to_faces[edge].append(fi)
    adjacency = [set() for _ in range(len(f))]
    for edge, touching in edge_to_faces.items():
        if edge in cut_edges or len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = int(touching[i]), int(touching[j])
                adjacency[a].add(b)
                adjacency[b].add(a)
    unseen = set(range(len(f)))
    out: list[np.ndarray] = []
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
        out.append(np.asarray(group, dtype=int))
    out.sort(key=lambda x: int(np.min(x)) if len(x) else -1)
    return out


def _duplicate_vertices_by_face_component(vertices: np.ndarray, faces: np.ndarray, components: list[np.ndarray]):
    verts = np.asarray(vertices, dtype=float).copy()
    out = np.asarray(faces, dtype=int).copy()
    vertex_components: dict[int, list[int]] = defaultdict(list)
    for ci, face_ids in enumerate(components):
        for vid in np.unique(out[face_ids].reshape(-1)):
            vertex_components[int(vid)].append(ci)
    replacements: dict[tuple[int, int], int] = {}
    for vid, comps in vertex_components.items():
        for ci in comps[1:]:
            replacements[(ci, vid)] = len(verts)
            verts = np.vstack([verts, verts[vid]])
    for ci, face_ids in enumerate(components):
        if ci == 0:
            continue
        for fi in face_ids:
            for li, vid in enumerate(out[int(fi)]):
                new_id = replacements.get((ci, int(vid)))
                if new_id is not None:
                    out[int(fi), li] = new_id
    return verts, out


def _grid_graph(vertices: np.ndarray, faces: np.ndarray):
    verts = np.asarray(vertices, dtype=float)
    adjacency: dict[int, list[int]] = defaultdict(list)
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            a, b = ids[i], ids[(i + 1) % len(ids)]
            e = tuple(sorted((a, b)))
            if e in edges:
                continue
            edges.add(e)
            adjacency[a].append(b)
            adjacency[b].append(a)
    return adjacency, edges


def _nearest_used_vertex(vertices: np.ndarray, used: np.ndarray, point: np.ndarray) -> int:
    pts = np.asarray(vertices, dtype=float)[used, :2]
    return int(used[int(np.argmin(np.linalg.norm(pts - np.asarray(point, dtype=float)[None, :], axis=1)))])


def _direction_code(delta: np.ndarray) -> int:
    d = np.asarray(delta, dtype=float)
    if abs(float(d[0])) >= abs(float(d[1])):
        return 0 if d[0] >= 0 else 1
    return 2 if d[1] >= 0 else 3


def _snap_segment_path(
    vertices: np.ndarray,
    adjacency: dict[int, list[int]],
    start: int,
    goal: int,
    segment_a: np.ndarray,
    segment_b: np.ndarray,
    tile_scale: float,
) -> list[int]:
    if start == goal:
        return [start]
    verts = np.asarray(vertices, dtype=float)
    turn_weight = 0.35 * tile_scale
    distance_weight = 2.5
    # State includes incoming grid direction so a turn penalty suppresses noisy
    # left-right staircasing while still allowing diagonal seams to be traced.
    heap: list[tuple[float, int, int]] = [(0.0, start, -1)]
    dist: dict[tuple[int, int], float] = {(start, -1): 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(start, -1): None}
    best_goal: tuple[int, int] | None = None
    while heap:
        cost, node, prev_dir = heapq.heappop(heap)
        state = (node, prev_dir)
        if cost != dist.get(state):
            continue
        if node == goal:
            best_goal = state
            break
        p = verts[node, :2]
        for nxt in adjacency.get(node, []):
            q = verts[nxt, :2]
            direction = _direction_code(q - p)
            mid = 0.5 * (p + q)
            dseg = float(_segment_distance(mid[None, :], segment_a, segment_b)[0])
            step = float(np.linalg.norm(q - p)) + distance_weight * dseg
            if prev_dir >= 0 and direction != prev_dir:
                step += turn_weight
            ns = (int(nxt), direction)
            nc = cost + step
            if nc < dist.get(ns, float("inf")):
                dist[ns] = nc
                parent[ns] = state
                heapq.heappush(heap, (nc, int(nxt), direction))
    if best_goal is None:
        return []
    rev: list[int] = []
    state: tuple[int, int] | None = best_goal
    while state is not None:
        rev.append(int(state[0]))
        state = parent.get(state)
    return list(reversed(rev))


def _heal_narrow_seam_strip(mesh: Any, segments: np.ndarray, grid: Any):
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=float)
    all_faces = np.asarray([tile.vertex_ids for tile in (getattr(mesh.grid, "tiles", None) or [])], dtype=int)
    if len(all_faces) == 0 or len(segments) == 0:
        return faces, 0
    current = {tuple(sorted(map(int, face))) for face in faces}
    missing = [face for face in all_faces if tuple(sorted(map(int, face))) not in current]
    if not missing:
        return faces, 0
    missing_arr = np.asarray(missing, dtype=int)
    centers = np.mean(vertices[missing_arr, :2], axis=1)
    tile_size = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 1e-8)
    threshold = 0.80 * tile_size
    near = np.zeros(len(missing_arr), dtype=bool)
    for seg in np.asarray(segments, dtype=float):
        near |= _segment_distance(centers, seg[0], seg[1]) <= threshold
    # Require adjacency to the already-kept domain.  This prevents a seam endpoint
    # near the outer boundary from restoring arbitrary cells outside Omega.
    kept_vertices = set(map(int, faces.reshape(-1))) if len(faces) else set()
    adjacent = np.asarray([len(kept_vertices.intersection(map(int, f))) >= 2 for f in missing_arr], dtype=bool)
    restore = missing_arr[near & adjacent]
    if len(restore) == 0:
        return faces, 0
    return np.vstack([faces, restore]), int(len(restore))


def _make_quadmesh(pipeline: Any, mesh: Any, vertices: np.ndarray, faces: np.ndarray, metrics: dict[str, Any]):
    cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
    return cls(
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
        mesh.grid,
        mesh.stage,
        metrics,
        list(getattr(mesh, "split_lines", [])),
    )


def install_optcuts_seam_metadata_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_seam_metadata_patch_installed", False):
        return
    base_flatten = pipeline._flatten_to_domain

    def flatten_with_optcuts_seams(parameterization: Any, grid: Any, params: Any = None):
        domain = base_flatten(parameterization, grid, params)
        if str(getattr(parameterization, "method", "")) != "optcuts":
            return domain
        payload = _extract_connected_seam_payload(parameterization)
        setattr(domain, "_optcuts_grid_seam_payload", payload)
        # OptCuts already chose the distortion-relieving cuts.  Suppress the
        # legacy CSF row/column heuristic in this mode so the two policies do not
        # compete.  Preserve it for diagnostics and for all non-OptCuts modes.
        previous = list(getattr(domain, "split_lines", []) or [])
        setattr(domain, "_optcuts_suppressed_legacy_split_lines", previous)
        try:
            domain.split_lines = []
        except Exception:
            pass
        return domain

    pipeline._flatten_to_domain = flatten_with_optcuts_seams
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = flatten_with_optcuts_seams
    pipeline._onestring_optcuts_seam_metadata_patch_installed = True


def install_optcuts_grid_seam_m2d_patch(pipeline: Any) -> None:
    """Install after Simple Split so this is the final M2D topology adapter."""
    if getattr(pipeline, "_onestring_optcuts_grid_seam_m2d_patch_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_m2d_with_optcuts_grid_seam(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        payload = getattr(domain, "_optcuts_grid_seam_payload", None)
        if not isinstance(payload, dict):
            return mesh
        segments = np.asarray(payload.get("segments", np.zeros((0, 2, 2))), dtype=float)
        nodes = dict(payload.get("nodes", {}) or {})
        source_edges = list(payload.get("edges", []) or [])
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update({
            "optcuts_grid_seam_enabled": True,
            "optcuts_source_seam_edge_count": int(len(source_edges)),
            "optcuts_legacy_split_suppressed": True,
            "optcuts_suppressed_legacy_split_lines": list(getattr(domain, "_optcuts_suppressed_legacy_split_lines", []) or []),
        })
        if len(segments) == 0 or not source_edges:
            metrics.update({"optcuts_grid_seam_applied": False, "optcuts_grid_seam_reason": "no_internal_optcuts_seam"})
            mesh.metrics.update(metrics)
            return mesh

        healed_faces, restored = _heal_narrow_seam_strip(mesh, segments, grid)
        vertices = np.asarray(mesh.vertices, dtype=float).copy()
        adjacency, _ = _grid_graph(vertices, healed_faces)
        used = np.asarray(sorted(adjacency), dtype=int)
        if len(used) == 0:
            metrics.update({"optcuts_grid_seam_applied": False, "optcuts_grid_seam_reason": "empty_grid_graph"})
            mesh.metrics.update(metrics)
            return mesh

        tile_scale = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 1e-8)
        node_to_grid: dict[int, int] = {}
        for node_id, point in nodes.items():
            node_to_grid[int(node_id)] = _nearest_used_vertex(vertices, used, np.asarray(point, dtype=float))

        cut_edges: set[tuple[int, int]] = set()
        snapped_paths: list[list[int]] = []
        failed = 0
        for a, b in source_edges:
            if int(a) not in node_to_grid or int(b) not in node_to_grid:
                failed += 1
                continue
            pa = np.asarray(nodes[int(a)], dtype=float)
            pb = np.asarray(nodes[int(b)], dtype=float)
            path = _snap_segment_path(
                vertices,
                adjacency,
                node_to_grid[int(a)],
                node_to_grid[int(b)],
                pa,
                pb,
                tile_scale,
            )
            if len(path) < 2:
                failed += 1
                continue
            snapped_paths.append(path)
            for u, v in zip(path[:-1], path[1:]):
                cut_edges.add(tuple(sorted((int(u), int(v)))))

        components = _edge_components_with_cuts(healed_faces, cut_edges)
        if not cut_edges or len(components) <= 1:
            metrics.update({
                "optcuts_grid_seam_applied": False,
                "optcuts_grid_seam_reason": "snapped_path_did_not_disconnect_grid",
                "optcuts_grid_seam_failed_source_edges": int(failed),
                "optcuts_grid_seam_restored_face_count": int(restored),
            })
            mesh.metrics.update(metrics)
            return mesh

        canonical_vertices, cut_faces = _duplicate_vertices_by_face_component(vertices, healed_faces, components)
        # canonical_vertices already contains the duplicate ids at exactly the same
        # coordinates: the seam is zero-width numerically and only topological.
        out_vertices = canonical_vertices.copy()
        final_components = _edge_components_with_cuts(cut_faces, set())
        panel_vertices = [np.unique(cut_faces[c].reshape(-1)) for c in final_components]
        metrics.update({
            "optcuts_grid_seam_applied": True,
            "optcuts_grid_seam_model": "connected grid-edge snap + zero-width vertex-duplication cut",
            "optcuts_grid_seam_restored_face_count": int(restored),
            "optcuts_grid_seam_snapped_path_count": int(len(snapped_paths)),
            "optcuts_grid_seam_cut_edge_count": int(len(cut_edges)),
            "optcuts_grid_seam_failed_source_edges": int(failed),
            "optcuts_grid_seam_panel_count": int(len(final_components)),
            "optcuts_grid_seam_panel_face_counts": [int(len(c)) for c in final_components],
            "optcuts_grid_seam_gap": 0.0,
            "split_panel_geometry_separated": bool(len(final_components) > 1),
            "split_panel_count": int(len(final_components)),
            "split_panel_gap": 0.0,
            "split_panel_layout_model": "OptCuts zero-width seam; no cell deletion gap",
        })
        out = _make_quadmesh(pipeline, mesh, out_vertices, cut_faces, metrics)
        setattr(out, "_split_panel_source_vertices", canonical_vertices.copy())
        setattr(out, "_split_panel_face_components", final_components)
        setattr(out, "_split_panel_vertex_components", panel_vertices)
        setattr(out, "_split_panel_offsets", [np.zeros(2, dtype=float) for _ in final_components])
        setattr(out, "_optcuts_grid_seam_cut_edges", sorted(cut_edges))
        setattr(out, "_optcuts_grid_seam_paths", snapped_paths)
        return out

    pipeline._build_m2d = build_m2d_with_optcuts_grid_seam
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_with_optcuts_grid_seam
    pipeline._onestring_optcuts_grid_seam_m2d_patch_installed = True


__all__ = [
    "install_optcuts_seam_metadata_patch",
    "install_optcuts_grid_seam_m2d_patch",
    "_extract_connected_seam_payload",
]
