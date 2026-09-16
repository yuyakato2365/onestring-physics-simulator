"""Manifold guard for the OptCuts -> M2D seam adapter.

T3D construction requires every topological M2D/K3D edge to have at most two
incident tiles.  The OptCuts seam adapter can temporarily re-introduce regular
grid faces near a seam before duplicating seam vertices.  This guard removes
only exact duplicate quads first, then rejects any remaining excess face on an
edge that would otherwise have three or more incidents.

The guard is opt-in: it only acts on meshes carrying ``optcuts_grid_seam_enabled``.
It never changes the stable non-OptCuts Split path.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(np.asarray(faces, dtype=int)):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edge = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            out[edge].append(int(fi))
    return out


def _dedupe_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    f = np.asarray(faces, dtype=int)
    keep: list[int] = []
    seen: set[tuple[int, ...]] = set()
    removed = 0
    for fi, face in enumerate(f):
        # For the regular quad grid, the sorted vertex set uniquely identifies a
        # tile independent of winding/start corner.
        key = tuple(sorted(map(int, face)))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        keep.append(int(fi))
    return f[np.asarray(keep, dtype=int)] if keep else np.zeros((0, f.shape[1]), dtype=int), int(removed)


def _drop_excess_nonmanifold_faces(faces: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Greedily keep faces only when none of their edges would exceed degree 2."""
    f = np.asarray(faces, dtype=int)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    keep: list[int] = []
    removed = 0
    offending: set[tuple[int, int]] = set()
    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        edges = [tuple(sorted((ids[i], ids[(i + 1) % len(ids)]))) for i in range(len(ids))]
        bad = [e for e in edges if edge_count[e] >= 2]
        if bad:
            removed += 1
            offending.update(bad)
            continue
        keep.append(int(fi))
        for e in edges:
            edge_count[e] += 1
    kept = f[np.asarray(keep, dtype=int)] if keep else np.zeros((0, f.shape[1]), dtype=int)
    return kept, int(removed), sorted(offending)


def _copy_quadmesh_with_faces(pipeline: Any, mesh: Any, faces: np.ndarray, metrics: dict[str, Any]):
    cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
    out = cls(
        np.asarray(mesh.vertices, dtype=float).copy(),
        np.asarray(faces, dtype=int).copy(),
        mesh.grid,
        mesh.stage,
        metrics,
        list(getattr(mesh, "split_lines", [])),
    )
    # Preserve seam/canonical metadata consumed by lift/K2D compatibility code.
    for name in (
        "_split_panel_source_vertices",
        "_split_panel_face_components",
        "_split_panel_vertex_components",
        "_split_panel_offsets",
        "_optcuts_grid_seam_cut_edges",
        "_optcuts_grid_seam_paths",
    ):
        if hasattr(mesh, name):
            try:
                setattr(out, name, getattr(mesh, name))
            except Exception:
                pass
    return out


def install_optcuts_manifold_guard_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_manifold_guard_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_m2d_manifold_guard(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        if not bool(metrics.get("optcuts_grid_seam_enabled", False)):
            return mesh

        original_faces = np.asarray(mesh.faces, dtype=int)
        faces, duplicate_removed = _dedupe_faces(original_faces)
        incidence = _edge_incidence(faces)
        before_bad = {edge: ids for edge, ids in incidence.items() if len(ids) > 2}
        excess_removed = 0
        offending_edges: list[tuple[int, int]] = []
        if before_bad:
            faces, excess_removed, offending_edges = _drop_excess_nonmanifold_faces(faces)

        final_incidence = _edge_incidence(faces)
        after_bad = {edge: ids for edge, ids in final_incidence.items() if len(ids) > 2}
        metrics.update({
            "optcuts_manifold_guard_applied": True,
            "optcuts_duplicate_face_removed_count": int(duplicate_removed),
            "optcuts_nonmanifold_edge_count_before_guard": int(len(before_bad)),
            "optcuts_nonmanifold_excess_face_removed_count": int(excess_removed),
            "optcuts_nonmanifold_edge_count_after_guard": int(len(after_bad)),
            "optcuts_nonmanifold_offending_edges": [list(map(int, e)) for e in offending_edges[:64]],
        })
        if after_bad:
            raise RuntimeError(
                "OPTCUTS_M2D_MANIFOLD_GUARD_FAILED: edges with >2 incident faces remain: "
                f"{list(after_bad.items())[:8]}"
            )

        print(
            "[OPTCUTS-MANIFOLD] "
            f"duplicate_faces_removed={duplicate_removed} "
            f"nonmanifold_edges_before={len(before_bad)} "
            f"excess_faces_removed={excess_removed} "
            f"nonmanifold_edges_after={len(after_bad)}"
        )
        if len(faces) == len(original_faces) and duplicate_removed == 0 and excess_removed == 0:
            mesh.metrics.update(metrics)
            return mesh
        return _copy_quadmesh_with_faces(pipeline, mesh, faces, metrics)

    pipeline._build_m2d = build_m2d_manifold_guard
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_manifold_guard
    pipeline._onestring_optcuts_manifold_guard_installed = True


__all__ = ["install_optcuts_manifold_guard_patch", "_edge_incidence"]
