"""Robust M2D topology cut for snapped OptCuts grid seams.

Unlike a component-only cutter, this duplicates a grid vertex separately for
connected sectors of incident faces around that vertex. Therefore open slits
and branched seam graphs are represented correctly even when cutting the seam
does not split the whole mesh into multiple connected components.

Important: cells removed by the original Omega clipping are NOT restored here.
Restoring a quad that crosses an OptCuts UV seam can make two of its corners map
to the same 3D point, yielding T3D_FAILED_INVALID_TOP/duplicate_top_vertex.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from .optcuts_grid_seam_patch import (
    _grid_graph,
    _make_quadmesh,
    _nearest_used_vertex,
    _snap_segment_path,
)


def _face_components(faces: np.ndarray) -> list[np.ndarray]:
    f = np.asarray(faces, dtype=int)
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            e = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            edge_to_faces[e].append(fi)
    adj = [set() for _ in range(len(f))]
    for touching in edge_to_faces.values():
        if len(touching) < 2:
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = int(touching[i]), int(touching[j])
                adj[a].add(b)
                adj[b].add(a)
    unseen = set(range(len(f)))
    out: list[np.ndarray] = []
    while unseen:
        root = unseen.pop()
        q = deque([root])
        group = [root]
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    q.append(nxt)
                    group.append(nxt)
        out.append(np.asarray(group, dtype=int))
    out.sort(key=lambda x: int(np.min(x)) if len(x) else -1)
    return out


def _duplicate_vertices_along_cut_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    cut_edges: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Duplicate vertex ids by local incident-face sectors around each seam node."""
    verts = np.asarray(vertices, dtype=float).copy()
    out = np.asarray(faces, dtype=int).copy()
    if not cut_edges or len(out) == 0:
        return verts, out, 0

    incident: dict[int, list[int]] = defaultdict(list)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(out):
        ids = [int(v) for v in face]
        for v in ids:
            incident[v].append(fi)
        for i in range(len(ids)):
            e = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            edge_faces[e].append(fi)

    seam_vertices = sorted({v for edge in cut_edges for v in edge})
    duplicate_count = 0
    for vertex in seam_vertices:
        face_ids = sorted(set(incident.get(int(vertex), [])))
        if len(face_ids) <= 1:
            continue
        local = {fi: set() for fi in face_ids}
        face_set = set(face_ids)
        for edge, touching in edge_faces.items():
            if int(vertex) not in edge or edge in cut_edges:
                continue
            local_touch = [int(fi) for fi in touching if int(fi) in face_set]
            for i in range(len(local_touch)):
                for j in range(i + 1, len(local_touch)):
                    a, b = local_touch[i], local_touch[j]
                    local[a].add(b)
                    local[b].add(a)

        unseen = set(face_ids)
        sectors: list[list[int]] = []
        while unseen:
            root = unseen.pop()
            q = deque([root])
            sector = [root]
            while q:
                cur = q.popleft()
                for nxt in local[cur]:
                    if nxt in unseen:
                        unseen.remove(nxt)
                        q.append(nxt)
                        sector.append(nxt)
            sectors.append(sector)
        if len(sectors) <= 1:
            continue
        for sector in sectors[1:]:
            new_id = len(verts)
            verts = np.vstack([verts, verts[int(vertex)]])
            duplicate_count += 1
            for fi in sector:
                mask = out[int(fi)] == int(vertex)
                out[int(fi), mask] = new_id
    return verts, out, int(duplicate_count)


def _mapped_quad_is_valid(
    pipeline: Any,
    points_2d: np.ndarray,
    parameterization: Any,
    tile_scale: float,
) -> bool:
    """Reject a quad that would become a duplicate/degenerate top in M3D/K3D."""
    mapped: list[np.ndarray] = []
    inverse = getattr(pipeline, "inverse_map_uv_to_surface", None)
    if inverse is None:
        return True
    try:
        for point in np.asarray(points_2d, dtype=float):
            value = inverse(np.asarray(point[:2], dtype=float), parameterization)
            xyz = np.asarray(value[0] if isinstance(value, tuple) else value, dtype=float).reshape(-1)[:3]
            if len(xyz) != 3 or not np.all(np.isfinite(xyz)):
                return False
            mapped.append(xyz)
    except Exception:
        return False
    pts = np.asarray(mapped, dtype=float)
    if pts.shape != (4, 3):
        return False
    scale = max(float(np.max(np.ptp(pts, axis=0))), float(tile_scale), 1e-9)
    tol = max(1e-9, 1e-7 * scale)
    pair_min = min(
        float(np.linalg.norm(pts[i] - pts[j]))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    if pair_min <= tol:
        return False
    area = 0.5 * (
        float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
        + float(np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[0])))
    )
    return bool(np.isfinite(area) and area > max(1e-14, tol * tol))


def install_optcuts_grid_seam_topology_patch(pipeline: Any) -> None:
    """Install after Simple Split; affect only domains carrying OptCuts seam data."""
    if getattr(pipeline, "_onestring_optcuts_grid_seam_topology_patch_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_m2d_with_connected_optcuts_seam(grid: Any, domain: Any, params: Any = None):
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
            "optcuts_suppressed_legacy_split_lines": list(
                getattr(domain, "_optcuts_suppressed_legacy_split_lines", []) or []
            ),
        })
        if len(segments) == 0 or not source_edges:
            metrics.update({
                "optcuts_grid_seam_applied": False,
                "optcuts_grid_seam_reason": "no_internal_optcuts_seam",
            })
            mesh.metrics.update(metrics)
            print("[OPTCUTS-SEAM] no internal seam detected; M2D unchanged")
            return mesh

        # Do not heal/restore cells removed by Omega clipping. A restored quad may
        # straddle two UV charts and map two corners to the same 3D point.
        vertices = np.asarray(mesh.vertices, dtype=float).copy()
        working_faces = np.asarray(mesh.faces, dtype=int).copy()
        restored = 0
        adjacency, _grid_edges = _grid_graph(vertices, working_faces)
        used = np.asarray(sorted(adjacency), dtype=int)
        if len(used) == 0:
            metrics.update({
                "optcuts_grid_seam_applied": False,
                "optcuts_grid_seam_reason": "empty_grid_graph",
            })
            mesh.metrics.update(metrics)
            return mesh

        tile_scale = max(float(getattr(grid, "tile_size", 0.0) or 0.0), 1e-8)
        node_to_grid = {
            int(node_id): _nearest_used_vertex(vertices, used, np.asarray(point, dtype=float))
            for node_id, point in nodes.items()
        }
        cut_edges: set[tuple[int, int]] = set()
        snapped_paths: list[list[int]] = []
        collapsed_or_failed = 0
        for a, b in source_edges:
            a, b = int(a), int(b)
            if a not in node_to_grid or b not in node_to_grid:
                collapsed_or_failed += 1
                continue
            path = _snap_segment_path(
                vertices,
                adjacency,
                node_to_grid[a],
                node_to_grid[b],
                np.asarray(nodes[a], dtype=float),
                np.asarray(nodes[b], dtype=float),
                tile_scale,
            )
            if len(path) < 2:
                collapsed_or_failed += 1
                continue
            snapped_paths.append(path)
            for u, v in zip(path[:-1], path[1:]):
                cut_edges.add(tuple(sorted((int(u), int(v)))))

        if not cut_edges:
            metrics.update({
                "optcuts_grid_seam_applied": False,
                "optcuts_grid_seam_reason": "all_source_edges_collapsed_during_grid_snap",
                "optcuts_grid_seam_restored_face_count": 0,
            })
            mesh.metrics.update(metrics)
            print("[OPTCUTS-SEAM] all seam segments collapsed during grid snap")
            return mesh

        cut_vertices, cut_faces, duplicated = _duplicate_vertices_along_cut_edges(
            vertices, working_faces, cut_edges
        )

        # Guard against exactly the invalid topology seen at T3D: evaluate each
        # candidate M2D quad through the same inverse map before K2D/T3D.
        parameterization = getattr(domain, "_optcuts_parameterization", None)
        invalid_face_ids: list[int] = []
        if parameterization is not None and len(cut_faces):
            for fi, face in enumerate(cut_faces):
                if not _mapped_quad_is_valid(
                    pipeline,
                    cut_vertices[np.asarray(face, dtype=int)],
                    parameterization,
                    tile_scale,
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
            "optcuts_grid_seam_model": (
                "connected grid-edge snap + zero-width local face-sector vertex duplication"
            ),
            "optcuts_grid_seam_restored_face_count": 0,
            "optcuts_grid_seam_restoration_disabled_reason": "avoid_uv_chart_crossing_quads",
            "optcuts_grid_seam_invalid_mapped_quad_count": int(len(invalid_face_ids)),
            "optcuts_grid_seam_snapped_path_count": int(len(snapped_paths)),
            "optcuts_grid_seam_cut_edge_count": int(len(cut_edges)),
            "optcuts_grid_seam_collapsed_or_failed_source_edges": int(collapsed_or_failed),
            "optcuts_grid_seam_duplicated_vertex_count": int(duplicated),
            "optcuts_grid_seam_component_count": int(len(components)),
            "optcuts_grid_seam_component_face_counts": [int(len(c)) for c in components],
            "optcuts_grid_seam_gap": 0.0,
            "split_panel_geometry_separated": bool(duplicated > 0),
            "split_panel_count": int(len(components)),
            "split_panel_gap": 0.0,
            "split_panel_layout_model": "OptCuts zero-width seam; no seam-strip cell restoration",
        })
        out = _make_quadmesh(pipeline, mesh, cut_vertices, cut_faces, metrics)
        setattr(out, "_split_panel_source_vertices", cut_vertices.copy())
        setattr(out, "_split_panel_face_components", components)
        setattr(out, "_split_panel_vertex_components", panel_vertices)
        setattr(out, "_split_panel_offsets", [np.zeros(2, dtype=float) for _ in components])
        setattr(out, "_optcuts_grid_seam_cut_edges", sorted(cut_edges))
        setattr(out, "_optcuts_grid_seam_paths", snapped_paths)
        print(
            "[OPTCUTS-SEAM] applied "
            f"source_edges={len(source_edges)} grid_paths={len(snapped_paths)} "
            f"cut_edges={len(cut_edges)} duplicated_vertices={duplicated} "
            f"invalid_mapped_quads_removed={len(invalid_face_ids)} "
            f"components={len(components)}"
        )
        return out

    pipeline._build_m2d = build_m2d_with_connected_optcuts_seam
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_with_connected_optcuts_seam
    pipeline._onestring_optcuts_grid_seam_topology_patch_installed = True


__all__ = [
    "install_optcuts_grid_seam_topology_patch",
    "_duplicate_vertices_along_cut_edges",
    "_mapped_quad_is_valid",
]
