"""Final M2D policy for the OptCuts validation route.

This patch is intentionally installed *after* all historical/diagnostic M2D wrappers.
It therefore acts on the mesh that will actually be sent to M3D.

Policy:
1. Boundary clipping is not allowed to create fabrication panels. Keep only genuine
   whole axis-aligned grid cells (four distinct corners forming a rectangle).
2. Apply requested CSF row/column splits to that cleaned mesh by duplicating
   interface vertices along existing grid lines.
3. Preserve pre-gap UV coordinates for the M2D -> M3D inverse map.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import simple_split_panel_patch as _simple


def _is_whole_grid_quad(points: np.ndarray, tol: float) -> bool:
    pts = np.asarray(points, dtype=float)[:, :2]
    if pts.shape != (4, 2):
        return False

    # Four geometrically distinct corners.
    keys = {tuple(np.rint(p / tol).astype(np.int64).tolist()) for p in pts}
    if len(keys) != 4:
        return False

    xs = np.sort(np.unique(np.rint(pts[:, 0] / tol).astype(np.int64)))
    ys = np.sort(np.unique(np.rint(pts[:, 1] / tol).astype(np.int64)))
    if len(xs) != 2 or len(ys) != 2:
        return False

    # A genuine grid cell contains exactly the Cartesian product of its two x/y
    # coordinates. Any clipped triangle/trapezoid/partial quad fails this test.
    expected = {(int(x), int(y)) for x in xs for y in ys}
    actual = {
        (int(round(float(p[0]) / tol)), int(round(float(p[1]) / tol)))
        for p in pts
    }
    if actual != expected:
        return False

    # Reject degenerate slivers.
    width = float((xs[1] - xs[0]) * tol)
    height = float((ys[1] - ys[0]) * tol)
    return width > 10.0 * tol and height > 10.0 * tol


def _whole_grid_mask(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if len(f) == 0:
        return np.zeros(0, dtype=bool)
    span = max(float(np.max(np.ptp(verts[:, :2], axis=0))), 1.0)
    tol = max(1.0e-9 * span, 1.0e-11)
    keep = np.zeros(len(f), dtype=bool)
    for fi, face in enumerate(f):
        ids = np.asarray(face, dtype=int)
        if len(ids) != 4:
            continue
        keep[fi] = _is_whole_grid_quad(verts[ids], tol)
    return keep


def _copy_extra_attrs(src: Any, dst: Any) -> None:
    try:
        for name, value in vars(src).items():
            if name not in {"vertices", "faces", "grid", "stage", "metrics", "split_lines"}:
                setattr(dst, name, value)
    except Exception:
        pass


def install_final_m2d_grid_policy_patch(pipeline: Any) -> None:
    """Install as the outermost/final M2D wrapper."""
    if getattr(pipeline, "_final_m2d_grid_policy_patch_installed", False):
        return

    base_build = pipeline._build_m2d

    def final_grid_build(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        mode = str(getattr(params, "omega_parameterization_mode", "")) if params is not None else ""
        if mode != "optcuts_test":
            return mesh

        vertices = np.asarray(mesh.vertices, dtype=float).copy()
        faces = np.asarray(mesh.faces, dtype=int).copy()
        before = int(len(faces))

        # First remove every boundary-created partial panel. This is geometry based,
        # so it does not depend on legacy source labels such as boundary_triangle.
        keep = _whole_grid_mask(vertices, faces)
        faces = faces[keep].copy()
        removed_partial = int(before - len(faces))

        # Use the CSF planner's complete row/column requests. Prefer the domain,
        # because older wrappers may not have copied them into mesh.metrics.
        raw_lines = list(getattr(domain, "split_lines", []) or [])
        if not raw_lines:
            raw_lines = list(getattr(domain, "localized_split_segments", []) or [])
        if not raw_lines:
            raw_lines = list((getattr(mesh, "metrics", {}) or {}).get("split_locations", []) or [])

        records: list[dict[str, Any]] = []
        rejected: list[Any] = []
        for line in raw_lines:
            v2, f2, record = _simple._cut_once(vertices, faces, line)
            if record is None:
                rejected.append(tuple(line[:2]) if isinstance(line, (list, tuple)) else repr(line))
                continue
            vertices, faces = v2, f2
            records.append(record)

        canonical = vertices.copy()
        separated, components, panel_vertices, offsets, gap = _simple._open_split_gaps(
            canonical, faces, records, getattr(mesh, "grid", grid)
        )

        metrics = dict(getattr(mesh, "metrics", {}) or {})
        csf_before = float(getattr(domain, "csf_before", metrics.get("csf_before", float("nan"))))
        csf_after = float(getattr(domain, "csf_after_split", metrics.get("csf_after_split", float("nan"))))
        metrics.update({
            "strict_boundary_grid_crop": True,
            "strict_boundary_grid_crop_policy": "keep only complete axis-aligned four-corner grid cells",
            "strict_boundary_grid_faces_before": before,
            "strict_boundary_grid_faces_removed": removed_partial,
            "strict_boundary_grid_faces_after": int(len(faces)),
            "csf_split_applied": bool(records),
            "simple_split_active": bool(records),
            "simple_split_records": records,
            "simple_split_rejected_lines": rejected,
            "split_panel_geometry_separated": bool(len(components) > 1),
            "split_panel_count": int(len(components)),
            "split_panel_gap": float(gap),
            "paper_style_complete_split": bool(records),
            "max_csf_before_split": csf_before,
            "max_csf_after_split": csf_after,
        })

        cls = type(mesh)
        out = cls(
            np.asarray(separated, dtype=float),
            np.asarray(faces, dtype=int),
            mesh.grid,
            mesh.stage,
            metrics,
            list(raw_lines),
        )
        _copy_extra_attrs(mesh, out)
        setattr(out, "_split_panel_source_vertices", canonical)
        setattr(out, "_split_panel_face_components", components)
        setattr(out, "_split_panel_vertex_components", panel_vertices)
        setattr(out, "_split_panel_offsets", offsets)
        setattr(out, "_split_panel_records", records)

        print(
            "[OPTCUTS-FINAL-M2D-GRID] "
            f"before={before} removed_partial={removed_partial} kept={len(faces)} "
            f"split_requested={len(raw_lines)} split_applied={len(records)} "
            f"split_rejected={len(rejected)} components={len(components)}"
        )
        return out

    pipeline._build_m2d = final_grid_build
    pipeline._final_m2d_grid_policy_patch_installed = True


__all__ = ["_is_whole_grid_quad", "_whole_grid_mask", "install_final_m2d_grid_policy_patch"]
