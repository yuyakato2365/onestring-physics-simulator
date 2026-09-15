"""Paper-aligned Omega -> M2D behavior for ``optcuts_test`` / ``optcuts_test2``.

The One String paper uses a one-way construction:

    S -> Omega -> regular quad grid -> crop cells not fully contained in Omega.

This patch keeps the OptCuts-produced parameterization/cut topology as Omega and
removes the former feedback loop that rebuilt Omega from an intermediate M2D
footprint.  M2D is then filtered so that every retained quad is fully contained
in the fixed planar domain.

Only the Omega -> M2D relation is changed here.  OptCuts remains the selected
parameterization backend, and downstream CSF split / K3D / K2D patches remain
unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def _is_optcuts_test(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"


def _clean_polygon(boundary: Any) -> np.ndarray:
    poly = np.asarray(boundary, dtype=float)
    if poly.ndim != 2 or poly.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    poly = poly[:, :2]
    poly = poly[np.all(np.isfinite(poly), axis=1)]
    if len(poly) == 0:
        return poly

    # Remove consecutive duplicate points.  Keep the polygon open internally;
    # edge iteration closes it with np.roll.
    scale = max(float(np.max(np.ptp(poly, axis=0))), 1.0)
    tol = max(1e-12, 1e-11 * scale)
    kept = [poly[0]]
    for p in poly[1:]:
        if float(np.linalg.norm(p - kept[-1])) > tol:
            kept.append(p)
    out = np.asarray(kept, dtype=float)
    if len(out) >= 2 and float(np.linalg.norm(out[0] - out[-1])) <= tol:
        out = out[:-1]
    return out


def _geom_tolerance(poly: np.ndarray, quad: np.ndarray | None = None) -> float:
    arrays = [np.asarray(poly, dtype=float)]
    if quad is not None:
        arrays.append(np.asarray(quad, dtype=float))
    values = np.vstack([a[:, :2] for a in arrays if a.size])
    scale = max(float(np.max(np.ptp(values, axis=0))) if len(values) else 1.0, 1.0)
    return max(1e-12, 1e-10 * scale)


def _point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    ap = p - a
    cross = float(ab[0] * ap[1] - ab[1] * ap[0])
    if abs(cross) > tol * max(float(np.linalg.norm(ab)), 1.0):
        return False
    dot = float(np.dot(ap, ab))
    if dot < -tol:
        return False
    if dot > float(np.dot(ab, ab)) + tol:
        return False
    return True


def _point_in_polygon_or_boundary(point: np.ndarray, polygon: np.ndarray, tol: float) -> bool:
    """Even-odd point-in-polygon test with the boundary counted as inside."""
    p = np.asarray(point, dtype=float)[:2]
    poly = np.asarray(polygon, dtype=float)[:, :2]
    n = len(poly)
    if n < 3:
        return False

    inside = False
    x, y = float(p[0]), float(p[1])
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if _point_on_segment(p, a, b, tol):
            return True
        y0, y1 = float(a[1]), float(b[1])
        if (y0 > y) == (y1 > y):
            continue
        denom = y1 - y0
        if abs(denom) <= tol:
            continue
        x_hit = float(a[0]) + (y - y0) * float(b[0] - a[0]) / denom
        if x_hit > x + tol:
            inside = not inside
    return inside


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _proper_segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tol: float,
) -> bool:
    """Return True only for an interior crossing; touching/overlap is allowed."""
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    return (
        ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol))
        and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol))
    )


def _strictly_inside_quad(point: np.ndarray, quad: np.ndarray, tol: float) -> bool:
    """Strict point-in-quad test used to catch a concave boundary entering a cell."""
    q = np.asarray(quad, dtype=float)[:, :2]
    p = np.asarray(point, dtype=float)[:2]
    if any(_point_on_segment(p, q[i], q[(i + 1) % len(q)], tol) for i in range(len(q))):
        return False
    return _point_in_polygon_or_boundary(p, q, tol)


def _quad_fully_contained(quad: np.ndarray, polygon: np.ndarray) -> bool:
    """Geometric full-cell containment, not center-based crop.

    Conditions:
    1. all quad corners are inside or on the Omega boundary;
    2. no quad edge properly crosses the Omega boundary;
    3. no Omega boundary vertex lies strictly inside the quad (concave notch / hole guard).
    """
    q = np.asarray(quad, dtype=float)[:, :2]
    poly = np.asarray(polygon, dtype=float)[:, :2]
    if len(q) < 3 or len(poly) < 3:
        return False
    tol = _geom_tolerance(poly, q)

    if not all(_point_in_polygon_or_boundary(p, poly, tol) for p in q):
        return False

    for i in range(len(q)):
        qa, qb = q[i], q[(i + 1) % len(q)]
        for j in range(len(poly)):
            pa, pb = poly[j], poly[(j + 1) % len(poly)]
            if _proper_segments_intersect(qa, qb, pa, pb, tol):
                return False

    if any(_strictly_inside_quad(p, q, tol) for p in poly):
        return False
    return True


def _compact_quad_mesh(pipeline: Any, mesh: Any, keep: np.ndarray, metrics: dict[str, Any]) -> Any:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)[np.asarray(keep, dtype=bool)]
    if len(faces) == 0:
        raise RuntimeError(
            "OPTCUTS_TEST_PAPER_CROP_EMPTY: no regular quad cell is fully contained in Omega. "
            "Reduce tile size or inspect the OptCuts planar domain."
        )

    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=int)
    remap[used] = np.arange(len(used), dtype=int)
    compact_vertices = vertices[used]
    compact_faces = remap[faces]

    cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
    try:
        out = cls(
            compact_vertices,
            compact_faces,
            mesh.grid,
            mesh.stage,
            metrics,
            list(getattr(mesh, "split_lines", [])),
        )
    except TypeError:
        out = cls(
            vertices=compact_vertices,
            faces=compact_faces,
            grid=mesh.grid,
            stage=mesh.stage,
            metrics=metrics,
            split_lines=list(getattr(mesh, "split_lines", [])),
        )
    return out


def install_optcuts_test_boundary_reparameterization_patch(pipeline: Any) -> None:
    """Install paper-style one-way Omega -> M2D semantics for optcuts_test."""
    if getattr(pipeline, "_onestring_optcuts_test_boundary_patch_installed", False):
        return

    base_parameterization = pipeline._build_surface_parameterization
    base_m2d = pipeline._build_m2d

    def paper_parameterization(surface: Any, target: Any, grid: Any, params: Any):
        if not _is_optcuts_test(params):
            return base_parameterization(surface, target, grid, params)

        # Route through the ordinary official OptCuts backend, but DO NOT build
        # an intermediate M2D and DO NOT move/re-solve Omega from its footprint.
        ordinary_params = replace(params, omega_parameterization_mode="optcuts")
        parameterization = base_parameterization(surface, target, grid, ordinary_params)
        parameterization.method = "optcuts_test"
        parameterization.metrics.update(
            {
                "omega_parameterization_mode": "optcuts_test",
                "requested_omega_parameterization_mode": "optcuts_test",
                "optcuts_test_enabled": True,
                "optcuts_test_model": (
                    "official OptCuts Omega/cut topology kept fixed -> regular quad grid -> "
                    "strict full-cell crop; no M2D-to-Omega feedback"
                ),
                "optcuts_test_paper_one_way_omega_to_m2d": True,
                "optcuts_test_grid_outline_reparameterization": False,
                "optcuts_test_omega_modified_from_m2d": False,
                "optcuts_test_seam_topology_preserved": True,
            }
        )
        print(
            "[OPTCUTS-TEST-PAPER-OMEGA] fixed OptCuts Omega; "
            "disabled M2D-footprint boundary reparameterization"
        )
        return parameterization

    def paper_m2d(grid: Any, domain: Any, params: Any = None):
        mesh = base_m2d(grid, domain, params)
        if params is None or not _is_optcuts_test(params):
            return mesh

        polygon = _clean_polygon(getattr(domain, "boundary", []))
        if len(polygon) < 3:
            raise RuntimeError("OPTCUTS_TEST_PAPER_CROP_INVALID_OMEGA_BOUNDARY")

        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        keep = np.asarray(
            [_quad_fully_contained(vertices[face, :2], polygon) for face in faces],
            dtype=bool,
        )
        retained = int(np.count_nonzero(keep))
        rejected = int(len(faces) - retained)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update(
            {
                "paper_m2d_one_way_from_fixed_omega": True,
                "paper_m2d_strict_full_cell_crop": True,
                "paper_m2d_crop_rule": "retain quad iff the complete cell is contained in Omega",
                "paper_m2d_candidate_face_count": int(len(faces)),
                "paper_m2d_retained_face_count": retained,
                "paper_m2d_rejected_partial_face_count": rejected,
                "paper_m2d_center_crop_authoritative": False,
            }
        )
        print(
            "[OPTCUTS-TEST-PAPER-M2D] "
            f"candidates={len(faces)} retained={retained} rejected_partial={rejected} "
            "rule=full-cell-contained-in-fixed-Omega"
        )
        return _compact_quad_mesh(pipeline, mesh, keep, metrics)

    pipeline._build_surface_parameterization = paper_parameterization
    pipeline._build_m2d = paper_m2d
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = paper_parameterization
        original._build_m2d = paper_m2d

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = paper_parameterization
            glb["_build_m2d"] = paper_m2d

    pipeline._onestring_optcuts_test_boundary_patch_installed = True


__all__ = [
    "install_optcuts_test_boundary_reparameterization_patch",
    "_clean_polygon",
    "_point_in_polygon_or_boundary",
    "_quad_fully_contained",
]
