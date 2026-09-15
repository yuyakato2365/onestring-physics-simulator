"""Rectilinear OptCuts-to-M2D seam adapter for OneString.

OptCuts is kept as the distortion-aware seam proposer.  The final OneString seam
is not the arbitrary OptCuts polyline: the OptCuts seam graph is compressed into
chains and embedded on the already chosen M2D fabrication grid.  Every final
seam segment is horizontal or vertical, every endpoint/bend is a grid vertex,
and the grid minimum unit is the existing ``tile_size`` chosen before the run.

This intentionally reverses the previous policy: we do not approximate every
small OptCuts source edge independently.  Degree-2 source vertices are collapsed
into one chain first; each chain is represented by a straight or one-bend L path
when possible, with a strongly turn-penalized rectilinear fallback only when the
cropped Omega domain prevents such an L path.
"""
from __future__ import annotations

from collections import defaultdict, deque
import heapq
from typing import Any

import numpy as np

from .optcuts_grid_seam_patch import _grid_graph, _make_quadmesh, _nearest_used_vertex
from .optcuts_grid_seam_topology_patch import (
    _duplicate_vertices_along_cut_edges,
    _face_components,
    _mapped_quad_is_valid,
)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _connected_components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    unseen = set(adjacency)
    comps: list[list[int]] = []
    while unseen:
        root = unseen.pop()
        q = deque([root])
        group = [root]
        while q:
            cur = q.popleft()
            for nxt in adjacency.get(cur, set()):
                if nxt in unseen:
                    unseen.remove(nxt)
                    q.append(nxt)
                    group.append(nxt)
        comps.append(group)
    return comps


def _extract_source_chains(nodes: dict[int, np.ndarray], edges: list[tuple[int, int]]) -> list[list[int]]:
    """Collapse degree-2 OptCuts seam vertices into graph chains.

    Endpoints and junctions are preserved.  A closed degree-2 loop is split into
    two chains at approximately opposite vertices so it can still be embedded on
    a rectilinear grid without mapping every tiny source edge separately.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    valid_edges: list[tuple[int, int]] = []
    for a, b in edges:
        a, b = int(a), int(b)
        if a == b or a not in nodes or b not in nodes:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        valid_edges.append((a, b))
    if not valid_edges:
        return []

    visited: set[tuple[int, int]] = set()
    chains: list[list[int]] = []
    terminals = {v for v, nbrs in adjacency.items() if len(nbrs) != 2}

    def walk(start: int, nxt: int) -> list[int]:
        chain = [int(start), int(nxt)]
        visited.add(_edge_key(start, nxt))
        prev, cur = int(start), int(nxt)
        while cur not in terminals:
            choices = [x for x in adjacency[cur] if x != prev and _edge_key(cur, x) not in visited]
            if not choices:
                break
            following = int(choices[0])
            visited.add(_edge_key(cur, following))
            chain.append(following)
            prev, cur = cur, following
        return chain

    for terminal in sorted(terminals):
        for nbr in sorted(adjacency[terminal]):
            if _edge_key(terminal, nbr) in visited:
                continue
            chains.append(walk(terminal, nbr))

    # Handle all-degree-2 closed components (or any residual edge) explicitly.
    for comp in _connected_components(adjacency):
        residual = [
            (a, b)
            for a in comp
            for b in adjacency[a]
            if a < b and _edge_key(a, b) not in visited
        ]
        while residual:
            a, b = residual[0]
            chain = [int(a), int(b)]
            visited.add(_edge_key(a, b))
            prev, cur = int(a), int(b)
            while True:
                choices = [x for x in adjacency[cur] if x != prev and _edge_key(cur, x) not in visited]
                if not choices:
                    break
                following = int(choices[0])
                visited.add(_edge_key(cur, following))
                chain.append(following)
                prev, cur = cur, following
                if cur == chain[0]:
                    break
            # For a loop, split at the farthest source-space vertex from the start.
            if len(chain) > 4 and chain[-1] == chain[0]:
                unique = chain[:-1]
                p0 = np.asarray(nodes[unique[0]], float)
                far_i = int(np.argmax([np.linalg.norm(np.asarray(nodes[v], float) - p0) for v in unique]))
                far_i = max(1, min(far_i, len(unique) - 1))
                chains.append(unique[: far_i + 1])
                chains.append(unique[far_i:] + [unique[0]])
            else:
                chains.append(chain)
            residual = [
                (x, y)
                for x in comp
                for y in adjacency[x]
                if x < y and _edge_key(x, y) not in visited
            ]
    return [c for c in chains if len(c) >= 2]


def _nearest_vertex_to_coordinate(
    vertices: np.ndarray,
    used: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> int | None:
    pts = np.asarray(vertices, float)[used, :2]
    distances = np.linalg.norm(pts - np.asarray(target, float)[None, :], axis=1)
    idx = int(np.argmin(distances))
    return int(used[idx]) if float(distances[idx]) <= float(tolerance) else None


def _straight_grid_path(
    vertices: np.ndarray,
    adjacency: dict[int, list[int]],
    start: int,
    goal: int,
    tile_size: float,
) -> list[int]:
    if start == goal:
        return [int(start)]
    pts = np.asarray(vertices, float)
    a = pts[int(start), :2]
    b = pts[int(goal), :2]
    tol = max(1e-8, 0.08 * float(tile_size))
    horizontal = abs(float(a[1] - b[1])) <= tol
    vertical = abs(float(a[0] - b[0])) <= tol
    if not (horizontal or vertical):
        return []
    varying = 0 if horizontal else 1
    fixed = 1 - varying
    current = int(start)
    path = [current]
    seen = {current}
    for _ in range(max(8, len(adjacency) + 2)):
        if current == int(goal):
            return path
        p = pts[current, :2]
        candidates: list[tuple[float, int]] = []
        for nxt in adjacency.get(current, []):
            q = pts[int(nxt), :2]
            if abs(float(q[fixed] - a[fixed])) > tol:
                continue
            # Move monotonically toward the goal; never backtrack along a straight seam.
            old = abs(float(p[varying] - b[varying]))
            new = abs(float(q[varying] - b[varying]))
            if new >= old - 1e-10:
                continue
            candidates.append((new, int(nxt)))
        if not candidates:
            return []
        _, current = min(candidates)
        if current in seen:
            return []
        seen.add(current)
        path.append(current)
    return []


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    denom = float(np.dot(d, d))
    if denom <= 1e-20:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, d) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * d)))


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    poly = np.asarray(polyline, float)
    if len(poly) == 0:
        return 0.0
    if len(poly) == 1:
        return float(np.linalg.norm(np.asarray(point, float) - poly[0]))
    return min(_point_segment_distance(point, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def _path_cost(vertices: np.ndarray, path: list[int], source_polyline: np.ndarray, tile_size: float) -> float:
    if len(path) < 2:
        return float("inf")
    pts = np.asarray(vertices, float)[np.asarray(path, int), :2]
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    mids = 0.5 * (pts[:-1] + pts[1:])
    deviation = float(sum(_distance_to_polyline(mid, source_polyline) for mid in mids))
    directions = []
    for delta in np.diff(pts, axis=0):
        directions.append(0 if abs(float(delta[0])) >= abs(float(delta[1])) else 1)
    turns = sum(int(a != b) for a, b in zip(directions[:-1], directions[1:]))
    return length + 3.0 * deviation + 1.5 * float(tile_size) * turns


def _one_bend_rectilinear_path(
    vertices: np.ndarray,
    adjacency: dict[int, list[int]],
    used: np.ndarray,
    start: int,
    goal: int,
    source_polyline: np.ndarray,
    tile_size: float,
) -> list[int]:
    pts = np.asarray(vertices, float)
    a = pts[int(start), :2]
    b = pts[int(goal), :2]
    direct = _straight_grid_path(vertices, adjacency, start, goal, tile_size)
    candidates: list[list[int]] = [direct] if len(direct) >= 2 else []
    tol = max(1e-8, 0.25 * float(tile_size))
    for bend_xy in (np.asarray([b[0], a[1]]), np.asarray([a[0], b[1]])):
        bend = _nearest_vertex_to_coordinate(vertices, used, bend_xy, tol)
        if bend is None:
            continue
        first = _straight_grid_path(vertices, adjacency, start, bend, tile_size)
        second = _straight_grid_path(vertices, adjacency, bend, goal, tile_size)
        if first and second:
            candidate = first + second[1:]
            if len(candidate) >= 2:
                candidates.append(candidate)
    if not candidates:
        return []
    return min(candidates, key=lambda p: _path_cost(vertices, p, source_polyline, tile_size))


def _rectilinear_fallback(
    vertices: np.ndarray,
    adjacency: dict[int, list[int]],
    start: int,
    goal: int,
    source_polyline: np.ndarray,
    tile_size: float,
) -> list[int]:
    """Grid-only fallback with a large turn penalty; every segment remains H/V."""
    if start == goal:
        return [int(start)]
    pts = np.asarray(vertices, float)
    heap: list[tuple[float, int, int]] = [(0.0, int(start), -1)]
    dist: dict[tuple[int, int], float] = {(int(start), -1): 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(int(start), -1): None}
    target_state: tuple[int, int] | None = None
    while heap:
        cost, node, prev_axis = heapq.heappop(heap)
        state = (int(node), int(prev_axis))
        if cost != dist.get(state):
            continue
        if int(node) == int(goal):
            target_state = state
            break
        p = pts[int(node), :2]
        for nxt in adjacency.get(int(node), []):
            q = pts[int(nxt), :2]
            delta = q - p
            axis = 0 if abs(float(delta[0])) >= abs(float(delta[1])) else 1
            step = float(np.linalg.norm(delta))
            mid = 0.5 * (p + q)
            step += 3.0 * _distance_to_polyline(mid, source_polyline)
            if prev_axis >= 0 and axis != prev_axis:
                step += 3.0 * float(tile_size)
            ns = (int(nxt), int(axis))
            nc = float(cost + step)
            if nc < dist.get(ns, float("inf")):
                dist[ns] = nc
                parent[ns] = state
                heapq.heappush(heap, (nc, int(nxt), int(axis)))
    if target_state is None:
        return []
    rev: list[int] = []
    state: tuple[int, int] | None = target_state
    while state is not None:
        rev.append(int(state[0]))
        state = parent.get(state)
    return list(reversed(rev))


def _build_rectilinear_cut_network(
    vertices: np.ndarray,
    faces: np.ndarray,
    nodes: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    tile_size: float,
) -> tuple[set[tuple[int, int]], list[list[int]], dict[str, int]]:
    adjacency, _grid_edges = _grid_graph(vertices, faces)
    used = np.asarray(sorted(adjacency), dtype=int)
    if len(used) == 0:
        return set(), [], {"chain_count": 0, "fallback_count": 0, "collapsed_count": 0}
    chains = _extract_source_chains(nodes, edges)
    # Shared source graph nodes map once, so chain junctions remain exactly connected.
    node_to_grid = {
        int(node_id): _nearest_used_vertex(vertices, used, np.asarray(point, float))
        for node_id, point in nodes.items()
    }
    cut_edges: set[tuple[int, int]] = set()
    paths: list[list[int]] = []
    fallback_count = 0
    collapsed = 0
    for chain in chains:
        start_node, goal_node = int(chain[0]), int(chain[-1])
        start = node_to_grid.get(start_node)
        goal = node_to_grid.get(goal_node)
        if start is None or goal is None or int(start) == int(goal):
            collapsed += 1
            continue
        source_polyline = np.asarray([nodes[int(v)] for v in chain], float)
        path = _one_bend_rectilinear_path(
            vertices, adjacency, used, int(start), int(goal), source_polyline, tile_size
        )
        if len(path) < 2:
            path = _rectilinear_fallback(
                vertices, adjacency, int(start), int(goal), source_polyline, tile_size
            )
            if len(path) >= 2:
                fallback_count += 1
        if len(path) < 2:
            collapsed += 1
            continue
        paths.append(path)
        for a, b in zip(path[:-1], path[1:]):
            cut_edges.add(_edge_key(int(a), int(b)))
    return cut_edges, paths, {
        "chain_count": int(len(chains)),
        "fallback_count": int(fallback_count),
        "collapsed_count": int(collapsed),
    }


def install_optcuts_rectilinear_seam_patch(pipeline: Any) -> None:
    """Install the fixed-grid rectilinear OptCuts seam policy as final M2D adapter."""
    if getattr(pipeline, "_onestring_optcuts_rectilinear_seam_patch_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_m2d_with_rectilinear_optcuts_seam(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        payload = getattr(domain, "_optcuts_grid_seam_payload", None)
        if not isinstance(payload, dict):
            return mesh
        nodes = {int(k): np.asarray(v, float) for k, v in dict(payload.get("nodes", {}) or {}).items()}
        source_edges = [(int(a), int(b)) for a, b in list(payload.get("edges", []) or [])]
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        tile_size = max(float(getattr(grid, "tile_size", 0.0) or getattr(params, "tile_size", 0.0) or 0.0), 1e-8)
        metrics.update({
            "optcuts_grid_seam_enabled": True,
            "optcuts_rectilinear_fusion_enabled": True,
            "optcuts_rectilinear_minimum_unit": float(tile_size),
            "optcuts_rectilinear_allowed_axes": ["grid_u", "grid_v"],
            "optcuts_legacy_split_suppressed": True,
            "optcuts_source_seam_edge_count": int(len(source_edges)),
        })
        if not source_edges or not nodes:
            metrics.update({"optcuts_grid_seam_applied": False, "optcuts_grid_seam_reason": "no_internal_optcuts_seam"})
            mesh.metrics.update(metrics)
            return mesh

        vertices = np.asarray(mesh.vertices, float).copy()
        working_faces = np.asarray(mesh.faces, int).copy()
        cut_edges, paths, rect_stats = _build_rectilinear_cut_network(
            vertices, working_faces, nodes, source_edges, tile_size
        )
        if not cut_edges:
            metrics.update({
                "optcuts_grid_seam_applied": False,
                "optcuts_grid_seam_reason": "rectilinear_network_empty",
                "optcuts_rectilinear_chain_count": int(rect_stats["chain_count"]),
                "optcuts_rectilinear_collapsed_chain_count": int(rect_stats["collapsed_count"]),
            })
            mesh.metrics.update(metrics)
            print("[OPTCUTS-RECT] no usable rectilinear seam network")
            return mesh

        cut_vertices, cut_faces, duplicated = _duplicate_vertices_along_cut_edges(
            vertices, working_faces, cut_edges
        )

        # Keep the existing chart-safety guard: a grid-valid quad is still rejected
        # if its four corners do not map to one valid 3D tile under the OptCuts UV map.
        parameterization = getattr(domain, "_optcuts_parameterization", None)
        invalid_face_ids: list[int] = []
        if parameterization is not None and len(cut_faces):
            for fi, face in enumerate(cut_faces):
                if not _mapped_quad_is_valid(
                    pipeline,
                    cut_vertices[np.asarray(face, int)],
                    parameterization,
                    tile_size,
                ):
                    invalid_face_ids.append(int(fi))
            if invalid_face_ids:
                keep = np.ones(len(cut_faces), dtype=bool)
                keep[np.asarray(invalid_face_ids, dtype=int)] = False
                cut_faces = cut_faces[keep]

        components = _face_components(cut_faces)
        panel_vertices = [np.unique(cut_faces[c].reshape(-1)) for c in components]
        metrics.update({
            "optcuts_grid_seam_applied": bool(duplicated > 0),
            "optcuts_grid_seam_model": "OptCuts-guided fixed-unit rectilinear seam network",
            "optcuts_rectilinear_chain_count": int(rect_stats["chain_count"]),
            "optcuts_rectilinear_path_count": int(len(paths)),
            "optcuts_rectilinear_fallback_path_count": int(rect_stats["fallback_count"]),
            "optcuts_rectilinear_collapsed_chain_count": int(rect_stats["collapsed_count"]),
            "optcuts_grid_seam_cut_edge_count": int(len(cut_edges)),
            "optcuts_grid_seam_duplicated_vertex_count": int(duplicated),
            "optcuts_grid_seam_invalid_mapped_quad_count": int(len(invalid_face_ids)),
            "optcuts_grid_seam_component_count": int(len(components)),
            "optcuts_grid_seam_gap": 0.0,
            "split_panel_geometry_separated": bool(duplicated > 0),
            "split_panel_count": int(len(components)),
            "split_panel_gap": 0.0,
            "split_panel_layout_model": "OptCuts-guided orthogonal grid seams; zero-width topology cut",
        })
        out = _make_quadmesh(pipeline, mesh, cut_vertices, cut_faces, metrics)
        setattr(out, "_split_panel_source_vertices", cut_vertices.copy())
        setattr(out, "_split_panel_face_components", components)
        setattr(out, "_split_panel_vertex_components", panel_vertices)
        setattr(out, "_split_panel_offsets", [np.zeros(2, dtype=float) for _ in components])
        setattr(out, "_optcuts_grid_seam_cut_edges", sorted(cut_edges))
        setattr(out, "_optcuts_grid_seam_paths", paths)
        print(
            "[OPTCUTS-RECT] applied "
            f"source_edges={len(source_edges)} chains={rect_stats['chain_count']} "
            f"paths={len(paths)} fallback={rect_stats['fallback_count']} "
            f"cut_edges={len(cut_edges)} duplicated_vertices={duplicated} "
            f"invalid_quads_removed={len(invalid_face_ids)} components={len(components)} "
            f"unit={tile_size:g}"
        )
        return out

    pipeline._build_m2d = build_m2d_with_rectilinear_optcuts_seam
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_with_rectilinear_optcuts_seam
    pipeline._onestring_optcuts_rectilinear_seam_patch_installed = True


__all__ = [
    "install_optcuts_rectilinear_seam_patch",
    "_extract_source_chains",
    "_build_rectilinear_cut_network",
]
