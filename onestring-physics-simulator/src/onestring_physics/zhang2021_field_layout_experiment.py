"""Experimental, actually generating, coarse quad-layout backend.

This implements an INDEPENDENT paper-INSPIRED approach:
conservative boundary straightening, four-RoSy field, geometric patch pairing,
conforming quad templates, local quality optimization and strict OneString audit.

It is NOT a verified transcription of Zhang et al., CAD 2021. The actual
authors' segmentation, patch optimization, C++ equations and code are not
available here. Calling this the reference algorithm would be misleading.
No Gmsh, no masked regular-grid fallback, no silent geometric repair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class FieldLayoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class FieldLayoutResult:
    uv: np.ndarray
    quads: np.ndarray
    metrics: dict[str, Any]


def _ccw(points):
    p = np.asarray(points, float)
    cross = np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                   - p[:, 1] * np.roll(p[:, 0], -1))
    return p if cross > 0 else p[::-1].copy()


def _convex(points, tolerance):
    p = np.asarray(points, float)
    e = np.roll(p, -1, axis=0) - p
    return bool(np.all(np.cross(e, np.roll(e, -1, axis=0)) > tolerance))


def _boundary_candidate(original, requested, target):
    area = float(original.area)
    span = max(original.bounds[2] - original.bounds[0],
               original.bounds[3] - original.bounds[1])
    h = np.sqrt(area / max(target, 4))
    tolerances = (min(.24 * h, .07 * span), min(.12 * h, .035 * span),
                  min(.06 * h, .018 * span), min(.02 * h, .006 * span), 0.)
    for tol in tolerances:
        p = original.simplify(tol, preserve_topology=True) if tol else original
        if (p.geom_type == "Polygon" and p.is_valid and p.area > 0
                and len(p.exterior.coords) >= 4
                and p.difference(original).area <= area * 1e-11
                and original.difference(p).area <= area * requested * .5):
            return p, float(tol)
    return original, 0.


def _triangles(polygon):
    import shapely
    from shapely.ops import triangulate
    if hasattr(shapely, "constrained_delaunay_triangles"):
        geom = shapely.constrained_delaunay_triangles(polygon)
        triangles = list(geom.geoms)
    else:
        # For Shapely 2.0, never accept nonconforming triangulation or holes.
        triangles = [t for t in triangulate(polygon) if polygon.covers(t)]
    if (not triangles
            or any(len(t.exterior.coords) != 4 for t in triangles)
            or abs(sum(t.area for t in triangles) - polygon.area)
               > 1e-8 * polygon.area):
        raise FieldLayoutError("CONSTRAINED_DOMAIN_TRIANGULATION_FAILED")
    return triangles


def _four_rosy_angle(boundary):
    edges = np.roll(boundary, -1, axis=0) - boundary
    lengths = np.linalg.norm(edges, axis=1)
    theta = np.arctan2(edges[:, 1], edges[:, 0])
    z = np.sum(lengths * np.exp(4j * theta))
    return float(np.angle(z) / 4) if abs(z) > 1e-12 else 0.


def _patch_layout(tris, field, scale):
    """Find adjacent convex four-sided patches using field alignment.

    The triangle-pair segmentation is an original approximation, NOT Zhang's
    optimized polygon segmentation.
    """
    edges: dict[tuple, list[int]] = {}
    for i, t in enumerate(tris):
        tri = np.asarray(t.exterior.coords, float)[:-1, :2]
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            key = tuple(sorted((tuple(np.round(a / scale, 11)),
                                tuple(np.round(b / scale, 11)))))
            edges.setdefault(key, []).append(i)
    candidates = []
    tol = 1e-12 * scale * scale
    for owners in edges.values():
        if len(owners) != 2:
            continue
        i, j = owners
        combined = tris[i].union(tris[j])
        if combined.geom_type != "Polygon" or len(combined.exterior.coords) != 5:
            continue
        q = _ccw(np.asarray(combined.exterior.coords, float)[:-1, :2])
        if not _convex(q, tol):
            continue
        e = np.roll(q, -1, axis=0) - q
        length = np.linalg.norm(e, axis=1)
        theta = np.arctan2(e[:, 1], e[:, 0])
        directions = e / length[:, None]
        orthogonality = np.mean(np.abs(
            np.sum(directions * np.roll(directions, 1, axis=0), axis=1)))
        weight = (10 - 3 * orthogonality - np.std(np.log(length))
                  + .5 * np.mean(np.cos(4 * (theta - field))))
        candidates.append((float(weight), min(i, j), max(i, j), q))
    # Deterministic weighted pairing: 2 triangles -> coarse four-sided patch.
    # Global matching is not claimed; unmatched triangles receive an exact
    # three-quad template with conforming edge midpoints.
    used = set()
    patches = []
    matches = 0
    for _, i, j, q in sorted(candidates, key=lambda row: (-row[0], row[1], row[2])):
        if i not in used and j not in used:
            patches.append(q)
            used.update((i, j))
            matches += 1
    for i, t in enumerate(tris):
        if i not in used:
            patches.append(_ccw(np.asarray(t.exterior.coords, float)[:-1, :2]))
    return patches, matches


def _templates(patches, scale):
    """Conforming all-quad template: 4 quads / convex quad; 3 / triangle."""
    vertex = []
    lookup = {}
    faces = []
    center_groups = []
    def insert(p):
        p = np.asarray(p, float)
        key = tuple(np.rint(p / (scale * 1e-10)).astype(np.int64))
        if key in lookup:
            k = lookup[key]
            if np.linalg.norm(vertex[k] - p) > 1e-9 * scale:
                raise FieldLayoutError("VERTEX_QUANTIZATION_COLLISION")
            return k
        k = len(vertex)
        lookup[key] = k
        vertex.append(p)
        return k
    for p in patches:
        n = len(p)
        if n not in (3, 4):
            raise FieldLayoutError("INVALID_COARSE_PATCH")
        v = [insert(x) for x in p]
        mid = [insert(.5 * (p[i] + p[(i + 1) % n])) for i in range(n)]
        center = insert(np.mean(p, axis=0))
        group = []
        for i in range(n):
            group.append(len(faces))
            faces.append([v[i], mid[i], center, mid[(i - 1) % n]])
        center_groups.append((center, group))
    return np.asarray(vertex, float), np.asarray(faces, int), center_groups


def _optimize_centers(uv, quads, center_groups, scale):
    """Conservative quality optimization; all patch boundaries stay fixed."""
    uv = uv.copy()
    def objective(ids):
        p = uv[quads[np.asarray(ids, int)]]
        e = np.roll(p, -1, axis=1) - p
        length = np.maximum(np.linalg.norm(e, axis=2), 1e-14)
        cos = np.sum(e * np.roll(e, 1, axis=1), axis=2) / (
            length * np.roll(length, 1, axis=1))
        return float(np.sum(np.mean(cos ** 2, axis=1)
                            + .2 * np.var(np.log(length), axis=1)))
    before = sum(objective(indices) for _, indices in center_groups)
    changed = 0
    for _ in range(2):
        for center, faces in center_groups:
            original = uv[center].copy()
            neighbor = np.unique(quads[np.asarray(faces, int)])
            neighbor = neighbor[neighbor != center]
            direction = uv[neighbor].mean(axis=0) - original
            prev = objective(faces)
            for t in (1., .5, .25):
                uv[center] = original + t * direction
                if (all(_convex(uv[quads[i]], 1e-12 * scale ** 2)
                        for i in faces)
                        and objective(faces) < prev - 1e-12):
                    changed += 1
                    break
                uv[center] = original
    after = sum(objective(indices) for _, indices in center_groups)
    if after > before + 1e-8:
        raise FieldLayoutError("QUAD_OPTIMIZATION_REGRESSED")
    return uv, {"quality_objective_before": before,
                "quality_objective_after": after, "center_moves": changed}


def generate_field_layout(source: Any, target_quads: int = 35, *,
                          max_missing_fraction: float = .02,
                          allow_outline_simplification: bool = True,
                          require_seam_correspondence: bool = True) -> FieldLayoutResult:
    """Generate actual all-quad candidate; reject invalid geometry and seams.

    Does not promise exactly target_quads or the paper's exact algorithm.
    """
    from shapely.geometry import Polygon
    from .zhang2021_reference_bridge import audit_reference_quads
    from .optcuts_planar_quad_20261026 import _quad_quality
    try:
        if target_quads < 4 or not 0 <= max_missing_fraction < .5:
            raise FieldLayoutError("INVALID_TARGET_OR_AREA_TOLERANCE")
        boundary = np.asarray(source.omega_boundary, float)[:, :2]
        original = Polygon(boundary)
        if not original.is_valid or original.area <= 0 or original.interiors:
            raise FieldLayoutError("INVALID_SOURCE_OMEGA")
        outline, tolerance = (
            _boundary_candidate(original, max_missing_fraction, target_quads)
            if allow_outline_simplification else (original, 0.))
        tris = _triangles(outline)
        scale = max(np.ptp(boundary, axis=0).max(), 1e-8)
        direction = _four_rosy_angle(
            np.asarray(outline.exterior.coords, float)[:-1, :2])
        patches, matches = _patch_layout(tris, direction, scale)
        uv, quads, centers = _templates(patches, scale)
        uv, energy = _optimize_centers(uv, quads, centers, scale)
        audit = audit_reference_quads(
            source, uv, quads, max_undercoverage_fraction=max_missing_fraction,
            max_overcoverage_fraction=1e-7,
            require_seam_correspondence=require_seam_correspondence)
        q2 = _quad_quality(np.column_stack((uv, np.zeros(len(uv)))), quads)
        q3 = _quad_quality(audit.vertices_3d, quads)
        metrics = dict(audit.diagnostics)
        metrics.update(energy)
        metrics.update(
            planar_quad_engine="experimental_coarse_layout; NOT Zhang 2021 reference",
            planar_quad_method="field_guided_patch_templates_experiment",
            planar_quad_faces=int(len(quads)),
            planar_quad_target_faces=int(target_quads),
            planar_quad_field_angle_degrees=float(np.degrees(direction)),
            planar_quad_boundary_simplification_tolerance=float(tolerance),
            planar_quad_source_triangles=int(len(tris)),
            planar_quad_coarse_patch_count=int(len(patches)),
            planar_quad_paired_triangle_patches=int(matches),
            planar_quad_omega_undercoverage_fraction=float(
                audit.diagnostics["omega_undercoverage_fraction"]),
            planar_quad_omega_overcoverage_fraction=float(
                audit.diagnostics["omega_overcoverage_fraction"]),
            planar_quad_uv_inverse_exact_piecewise_linear=True,
            planar_quad_m3d_quality_optimized=bool(energy["center_moves"]),
            planar_quad_refine_3d_moves_accepted=int(energy["center_moves"]),
            planar_quad_optimization_model="constrained patch-center quality improvement",
            planar_quad_m3d_quality_policy="only valid non-inverted patch centers",
        )
        for k, v in q2.items():
            metrics["planar_quad_m2d_" + k] = v
        for k, v in q3.items():
            metrics["planar_quad_m3d_" + k] = v
        return FieldLayoutResult(uv=uv, quads=quads, metrics=metrics)
    except FieldLayoutError:
        raise
    except Exception as exc:
        raise FieldLayoutError(f"FIELD_LAYOUT_FAILED: {exc}") from exc
