"""Compact OptCuts chart-aware M2D vertices after face filtering.

Chart-aware filtering removes quads that cross immutable OptCuts UV boundaries.
The underlying regular overlay still contains vertices that are no longer
referenced by any kept face.  Those vertices intentionally have no chart id and
must not be sent through the chart-restricted M2D->M3D inverse map.

This patch removes unused vertices, remaps every face index, and remaps the
per-vertex chart ids.  It is OptCuts-only and must be installed outside the
chart-aware M2D builder.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _copy_runtime_attrs(source: Any, target: Any) -> None:
    for name in (
        "_optcuts_face_chart_ids",
        "_split_panel_source_vertices",
        "_split_panel_face_components",
        "_split_panel_vertex_components",
        "_split_panel_offsets",
        "_optcuts_grid_seam_cut_edges",
        "_optcuts_grid_seam_paths",
    ):
        if hasattr(source, name):
            try:
                setattr(target, name, getattr(source, name))
            except Exception:
                pass


def _compact_optcuts_mesh(pipeline: Any, mesh: Any) -> Any:
    faces = np.asarray(mesh.faces, dtype=int)
    vertices = np.asarray(mesh.vertices, dtype=float)
    vertex_charts = getattr(mesh, "_optcuts_vertex_chart_ids", None)
    if vertex_charts is None or len(faces) == 0:
        return mesh

    used = np.unique(faces.reshape(-1))
    if len(used) == 0:
        return mesh
    if np.min(used) < 0 or np.max(used) >= len(vertices):
        raise RuntimeError(
            "OPTCUTS_COMPACTION_INVALID_FACE_INDEX: "
            f"vertex_count={len(vertices)} min={int(np.min(used))} max={int(np.max(used))}"
        )

    charts = np.asarray(vertex_charts, dtype=int)
    if len(charts) != len(vertices):
        raise RuntimeError(
            "OPTCUTS_COMPACTION_CHART_LENGTH_MISMATCH: "
            f"vertices={len(vertices)} chart_ids={len(charts)}"
        )

    # Only referenced vertices matter.  A referenced vertex without a chart is a
    # real chart-assignment bug and should remain a hard error.
    bad_used = used[charts[used] < 0]
    if len(bad_used):
        raise RuntimeError(
            "OPTCUTS_REFERENCED_VERTEX_WITHOUT_CHART: "
            f"count={len(bad_used)} examples={bad_used[:32].astype(int).tolist()}"
        )

    if len(used) == len(vertices) and np.array_equal(used, np.arange(len(vertices), dtype=int)):
        mesh.metrics["optcuts_unused_vertex_compaction_applied"] = False
        mesh.metrics["optcuts_unused_vertex_count_removed"] = 0
        return mesh

    old_to_new = np.full(len(vertices), -1, dtype=int)
    old_to_new[used] = np.arange(len(used), dtype=int)
    remapped_faces = old_to_new[faces]
    compact_vertices = vertices[used].copy()
    compact_charts = charts[used].copy()

    cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
    metrics = dict(getattr(mesh, "metrics", {}) or {})
    metrics.update({
        "optcuts_unused_vertex_compaction_applied": True,
        "optcuts_vertex_count_before_compaction": int(len(vertices)),
        "optcuts_vertex_count_after_compaction": int(len(compact_vertices)),
        "optcuts_unused_vertex_count_removed": int(len(vertices) - len(compact_vertices)),
    })
    out = cls(
        compact_vertices,
        remapped_faces,
        mesh.grid,
        mesh.stage,
        metrics,
        list(getattr(mesh, "split_lines", [])),
    )
    _copy_runtime_attrs(mesh, out)
    setattr(out, "_optcuts_vertex_chart_ids", compact_charts)

    # Runtime topology attributes that store vertex ids must be remapped when
    # present.  Face-component arrays index faces, so they remain unchanged.
    if hasattr(mesh, "_optcuts_grid_seam_cut_edges"):
        remapped_edges = []
        for edge in getattr(mesh, "_optcuts_grid_seam_cut_edges", []) or []:
            a, b = int(edge[0]), int(edge[1])
            if 0 <= a < len(old_to_new) and 0 <= b < len(old_to_new):
                na, nb = int(old_to_new[a]), int(old_to_new[b])
                if na >= 0 and nb >= 0:
                    remapped_edges.append(tuple(sorted((na, nb))))
        setattr(out, "_optcuts_grid_seam_cut_edges", sorted(set(remapped_edges)))
    if hasattr(mesh, "_optcuts_grid_seam_paths"):
        remapped_paths = []
        for path in getattr(mesh, "_optcuts_grid_seam_paths", []) or []:
            new_path = []
            for vid in path:
                vid = int(vid)
                if 0 <= vid < len(old_to_new) and int(old_to_new[vid]) >= 0:
                    new_path.append(int(old_to_new[vid]))
            if len(new_path) >= 2:
                remapped_paths.append(new_path)
        setattr(out, "_optcuts_grid_seam_paths", remapped_paths)

    print(
        "[OPTCUTS-COMPACT] "
        f"vertices_before={len(vertices)} vertices_after={len(compact_vertices)} "
        f"unused_removed={len(vertices) - len(compact_vertices)}"
    )
    return out


def install_optcuts_chart_vertex_compaction_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_chart_vertex_compaction_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_m2d_compacted(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        if getattr(mesh, "_optcuts_vertex_chart_ids", None) is None:
            return mesh
        return _compact_optcuts_mesh(pipeline, mesh)

    pipeline._build_m2d = build_m2d_compacted
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_compacted
    pipeline._onestring_optcuts_chart_vertex_compaction_installed = True


__all__ = [
    "install_optcuts_chart_vertex_compaction_patch",
    "_compact_optcuts_mesh",
]
