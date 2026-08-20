"""Final, execution-path Split -> Panel topology pass.

This module is intentionally small and independent of the earlier experimental
Split patch.  It runs *after* the normal M2D builder and guarantees that a
reported CSF Split is reflected in the returned M2D topology.

For every split line, faces are partitioned by face-centroid side.  Every vertex
ID shared by both partitions is duplicated on the positive side.  Therefore no
shared edge can remain across the cut.  The resulting edge-connected face
components are then treated as Panels and rigidly packed apart in M2D.

The pre-pack coordinates are stored on ``_split_panel_source_vertices`` so the
existing M2D -> M3D wrapper can still evaluate c^{-1} in the original Omega
coordinates.
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
            edge = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            edge_to_faces[edge].append(fi)
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
        ids = [root]
        while q:
            cur = q.popleft()
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    q.append(nxt)
                    ids.append(nxt)
        comps.append(np.asarray(ids, dtype=int))
    comps.sort(key=len, reverse=True)
    return comps


def _force_cut_once(
    vertices: np.ndarray,
    faces: np.ndarray,
    axis: str,
    value: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Disconnect the two face sets separated by one geometric split line."""
    verts = np.asarray(vertices, dtype=float).copy()
    out_faces = np.asarray(faces, dtype=int).copy()
    if len(verts) == 0 or len(out_faces) == 0:
        return verts, out_faces, 0

    coord = 1 if axis == "row" else 0
    centroids = np.mean(verts[out_faces][:, :, coord], axis=1)
    neg = np.flatnonzero(centroids < float(value) - 1e-12)
    pos = np.flatnonzero(centroids > float(value) + 1e-12)

    # Faces whose centroid lands exactly on the split line are assigned by a
    # tiny deterministic perturbation of the threshold.  On a regular quad grid
    # this should be rare, but leaving them unassigned could reconnect panels.
    mid = np.flatnonzero(np.abs(centroids - float(value)) <= 1e-12)
    if len(mid):
        for fi in mid:
            local = verts[out_faces[int(fi)], coord]
            if float(np.mean(local)) >= float(value):
                pos = np.append(pos, int(fi))
            else:
                neg = np.append(neg, int(fi))

    neg = np.unique(neg.astype(int))
    pos = np.unique(pos.astype(int))
    if len(neg) == 0 or len(pos) == 0:
        return verts, out_faces, 0

    neg_vertices = set(int(v) for v in out_faces[neg].reshape(-1))
    pos_vertices = set(int(v) for v in out_faces[pos].reshape(-1))
    interface = sorted(neg_vertices & pos_vertices)
    if not interface:
        # This line may already have been cut by the earlier Split pass.
        return verts, out_faces, 0

    replacement: dict[int, int] = {}
    for vid in interface:
        replacement[vid] = len(verts)
        verts = np.vstack([verts, verts[vid]])
    for fi in pos:
        for li, vid in enumerate(out_faces[int(fi)]):
            new_id = replacement.get(int(vid))
            if new_id is not None:
                out_faces[int(fi), li] = new_id
    return verts, out_faces, len(replacement)


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

    boxes: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for ids in panel_vertices:
        p = xy[ids]
        lo = np.min(p, axis=0)
        hi = np.max(p, axis=0)
        boxes.append((lo, hi, float(hi[0] - lo[0]), float(hi[1] - lo[1])))

    max_width = max((b[2] for b in boxes), default=scale)
    target_width = max(2.5 * max_width, 2.0 * scale)
    cursor_x = 0.0
    cursor_y = 0.0
    row_h = 0.0
    offsets: list[np.ndarray] = []
    for lo, _hi, width, height in boxes:
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
    """Wrap the current M2D builder and wire it into the actual execution globals."""
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

        # Always use the canonical/pre-pack coordinates when an earlier wrapper
        # already stored them; otherwise use the returned M2D coordinates.
        canonical = getattr(mesh, "_split_panel_source_vertices", None)
        if canonical is None:
            canonical = np.asarray(mesh.vertices, dtype=float).copy()
        else:
            canonical = np.asarray(canonical, dtype=float).copy()
        faces = np.asarray(mesh.faces, dtype=int).copy()

        raw_lines = list(metrics.get("split_locations", []) or [])
        if not raw_lines:
            raw_lines = list(getattr(mesh, "split_lines", []) or [])
        parsed = [p for p in (_split_axis_value(line) for line in raw_lines) if p is not None]

        duplicated_final = 0
        per_line: list[dict[str, Any]] = []
        for axis, value in parsed:
            before = len(_edge_components(faces))
            canonical, faces, added = _force_cut_once(canonical, faces, axis, value)
            after = len(_edge_components(faces))
            duplicated_final += int(added)
            per_line.append({
                "axis": axis,
                "value": float(value),
                "duplicated_vertices": int(added),
                "components_before": int(before),
                "components_after": int(after),
            })

        components = _edge_components(faces)
        if len(components) <= 1 and parsed:
            # Fallback: partition directly using the first split line even if an
            # earlier pass altered connectivity in an unexpected way. Rebuild
            # from the canonical face geometry and duplicate every vertex shared
            # by the two centroid-side sets.
            axis, value = parsed[0]
            canonical, faces, added = _force_cut_once(canonical, faces, axis, value)
            duplicated_final += int(added)
            components = _edge_components(faces)

        if len(components) <= 1:
            raise RuntimeError(
                "FINAL_SPLIT_PANEL_PASS_FAILED: Split was reported but final M2D topology still has one edge-connected component. "
                f"lines={parsed}, existing_duplicated={metrics.get('csf_split_duplicated_vertex_count', 0)}, final_duplicated={duplicated_final}"
            )

        pverts = _panel_vertices(faces, components)
        packed, offsets = _pack_panels(canonical, pverts, getattr(mesh, "grid", grid))
        metrics.update({
            "final_split_panel_pass_applied": True,
            "final_split_panel_count": int(len(components)),
            "final_split_panel_face_counts": [int(len(c)) for c in components],
            "final_split_panel_vertex_counts": [int(len(v)) for v in pverts],
            "final_split_panel_added_duplicate_vertices": int(duplicated_final),
            "final_split_panel_per_line": per_line,
            "split_panel_geometry_separated": True,
            "split_panel_count": int(len(components)),
            "split_panel_offsets_xy": [[float(x) for x in off] for off in offsets],
            "split_panel_layout_model": "FINAL face-partition topology cut + rigid panel packing",
            "split_panel_source_uv_preserved_for_m3d": True,
        })

        out = pipeline_module._original.QuadMesh(
            packed,
            faces,
            mesh.grid,
            mesh.stage,
            metrics,
            list(getattr(mesh, "split_lines", [])),
        )
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

    # Critical: build_onestring_design is a function object whose globals may
    # point at a backed-up module dictionary. Patch that dictionary directly.
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
