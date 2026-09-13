"""Strictly crop OptCuts M2D to whole fabrication-grid cells.

For the dated OptCuts route, boundary-crossing cells are not clipped into partial
polygons.  Instead, any M2D face touched by the Omega boundary is removed.  This
keeps only whole interior grid cells, which is closer to the intended fabrication
model and avoids tiny/irregular boundary tiles contaminating K3D/K2D.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)[:2]
    a = np.asarray(a, dtype=float)[:2]
    b = np.asarray(b, dtype=float)[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-30:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_on_boundary(p: np.ndarray, boundary: np.ndarray, tol: float) -> bool:
    poly = np.asarray(boundary, dtype=float)
    if len(poly) < 2:
        return False
    for i in range(len(poly)):
        if _point_segment_distance(p, poly[i], poly[(i + 1) % len(poly)]) <= tol:
            return True
    return False


def _strict_interior_face_mask(vertices: np.ndarray, faces: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Keep only whole quad cells that do not touch the Omega boundary."""
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    poly = np.asarray(boundary, dtype=float)
    if len(f) == 0:
        return np.zeros(0, dtype=bool)
    if len(poly) < 3:
        return np.ones(len(f), dtype=bool)

    span = max(float(np.max(np.ptp(poly[:, :2], axis=0))), 1.0)
    tol = max(1e-8 * span, 1e-10)
    keep = np.ones(len(f), dtype=bool)

    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        # Boundary clipping often creates triangles encoded with repeated ids or
        # irregular polygons converted to quads.  A fabrication cell must retain
        # four distinct corners.
        if len(set(ids)) != 4:
            keep[fi] = False
            continue
        pts = verts[np.asarray(ids, dtype=int), :2]
        # If Omega passes through the grid cell, clipping creates one or more
        # vertices on the domain boundary.  Remove the entire cell instead of
        # retaining that partial tile.
        if any(_point_on_boundary(p, poly, tol) for p in pts):
            keep[fi] = False
            continue
    return keep


def install_optcuts_strict_boundary_grid_crop_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_strict_boundary_grid_crop_patch_installed", False):
        return

    base_build = pipeline._build_m2d

    def build_m2d_drop_boundary_cells(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        mode = str(getattr(params, "omega_parameterization_mode", "")) if params is not None else ""
        if mode != "optcuts_test":
            return mesh

        boundary = np.asarray(getattr(domain, "boundary", np.empty((0, 2))), dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        vertices = np.asarray(mesh.vertices, dtype=float)
        mask = _strict_interior_face_mask(vertices, faces, boundary)
        kept_faces = faces[mask].copy()

        metrics = dict(getattr(mesh, "metrics", {}) or {})
        removed = int(len(faces) - len(kept_faces))
        metrics.update(
            {
                "strict_boundary_grid_crop": True,
                "strict_boundary_grid_crop_policy": "drop any whole grid cell touched by Omega boundary",
                "strict_boundary_grid_faces_before": int(len(faces)),
                "strict_boundary_grid_faces_after": int(len(kept_faces)),
                "strict_boundary_grid_faces_removed": removed,
            }
        )

        cls = type(mesh)
        out = cls(
            vertices.copy(),
            kept_faces,
            mesh.grid,
            mesh.stage,
            metrics,
            list(getattr(mesh, "split_lines", [])),
        )
        # Preserve ad-hoc metadata/attributes used by downstream patches.
        try:
            for name, value in vars(mesh).items():
                if name not in {"vertices", "faces", "grid", "stage", "metrics", "split_lines"}:
                    setattr(out, name, value)
        except Exception:
            pass

        print(
            "[OPTCUTS-STRICT-BOUNDARY-GRID] "
            f"before={len(faces)} removed={removed} after={len(kept_faces)} "
            "rule=drop-boundary-crossing-cell"
        )
        return out

    pipeline._build_m2d = build_m2d_drop_boundary_cells
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_m2d_drop_boundary_cells

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_m2d_drop_boundary_cells

    pipeline._onestring_optcuts_strict_boundary_grid_crop_patch_installed = True


__all__ = ["_strict_interior_face_mask", "install_optcuts_strict_boundary_grid_crop_patch"]
