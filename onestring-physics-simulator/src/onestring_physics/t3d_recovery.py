"""Variable-topology T3D solid construction and recovery utilities.

The authoritative geometry in this module is a convex polyhedron.  The legacy
eight-vertex tile remains useful as a T2D/deployment compatibility proxy, but it
is not used to decide whether a recoverable T3D solid exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Iterable

import numpy as np


T3D_OK_NOMINAL_FRUSTUM = "T3D_OK_NOMINAL_FRUSTUM"
T3D_RECOVERED_CAPPED_FRUSTUM = "T3D_RECOVERED_CAPPED_FRUSTUM"
T3D_RECOVERED_HALFSPACE_CLIP = "T3D_RECOVERED_HALFSPACE_CLIP"
T3D_RECOVERED_WEDGE = "T3D_RECOVERED_WEDGE"
T3D_RECOVERED_PYRAMID = "T3D_RECOVERED_PYRAMID"
T3D_RECOVERED_LOCAL_THICKNESS = "T3D_RECOVERED_LOCAL_THICKNESS"
T3D_RECOVERED_SYNCHRONIZED_PAIR = "T3D_RECOVERED_SYNCHRONIZED_PAIR"
T3D_RECOVERED_JUNCTION_CAP = "T3D_RECOVERED_JUNCTION_CAP"
T3D_RECOVERED_GLOBAL_CLIP = "T3D_RECOVERED_GLOBAL_CLIP"
T3D_RECOVERED_MESH_CLEANUP = "T3D_RECOVERED_MESH_CLEANUP"
T3D_RECOVERED_LEGACY_EMERGENCY_PRISM = "T3D_RECOVERED_LEGACY_EMERGENCY_PRISM"
T3D_FAILED_INVALID_TOP = "T3D_FAILED_INVALID_TOP"
T3D_FAILED_TOP_SURFACE_INTERSECTION = "T3D_FAILED_TOP_SURFACE_INTERSECTION"
T3D_FAILED_EMPTY_FEASIBLE_REGION = "T3D_FAILED_EMPTY_FEASIBLE_REGION"
T3D_FAILED_BELOW_MANUFACTURING_LIMIT = "T3D_FAILED_BELOW_MANUFACTURING_LIMIT"
T3D_FAILED_NONMANIFOLD_CONTACT = "T3D_FAILED_NONMANIFOLD_CONTACT"
T3D_FAILED_NONORIENTABLE_COMPONENT = "T3D_FAILED_NONORIENTABLE_COMPONENT"
T3D_FAILED_GLOBAL_COLLISION_INFEASIBLE = "T3D_FAILED_GLOBAL_COLLISION_INFEASIBLE"
T3D_FAILED_NUMERICAL_IMPLEMENTATION = "T3D_FAILED_NUMERICAL_IMPLEMENTATION"


class T3DConstructionError(RuntimeError):
    def __init__(self, status: str, message: str, tile_ids: Iterable[int] = ()):  # noqa: D401
        super().__init__(f"{status}: {message}")
        self.status = status
        self.tile_ids = tuple(int(v) for v in tile_ids)


@dataclass
class ConvexTileSolid:
    vertices: np.ndarray
    faces: list[list[int]]
    top_face_ids: list[int]
    contact_face_by_edge: dict[int, int]
    recovery_status: str
    recovery_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)


def _normalize(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(arr))
    if np.isfinite(norm) and norm > 1e-14:
        return arr / norm
    if fallback is None:
        return np.zeros_like(arr)
    return _normalize(np.asarray(fallback, dtype=float), None)


def _face_normal(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return np.zeros(3, dtype=float)
    accum = np.zeros(3, dtype=float)
    for idx in range(len(pts)):
        accum += np.cross(pts[idx], pts[(idx + 1) % len(pts)])
    return _normalize(accum)


def _polygon_area_3d(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    origin = pts[0]
    return float(
        0.5
        * sum(
            np.linalg.norm(np.cross(pts[idx] - origin, pts[idx + 1] - origin))
            for idx in range(1, len(pts) - 1)
        )
    )


def _dedupe_polygon(points: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for point in points:
        p = np.asarray(point, dtype=float)
        if not out or float(np.linalg.norm(p - out[-1])) > tolerance:
            out.append(p)
    if len(out) > 1 and float(np.linalg.norm(out[0] - out[-1])) <= tolerance:
        out.pop()
    return out


def cleanup_polyhedron(
    vertices: np.ndarray,
    faces: list[list[int]],
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, list[list[int]], dict[str, int]]:
    """Merge near vertices, drop degenerate faces, and orient faces outward."""
    source = np.asarray(vertices, dtype=float)
    merged: list[np.ndarray] = []
    remap: dict[int, int] = {}
    for old_id, point in enumerate(source):
        new_id = next(
            (idx for idx, existing in enumerate(merged) if float(np.linalg.norm(point - existing)) <= tolerance),
            None,
        )
        if new_id is None:
            new_id = len(merged)
            merged.append(np.asarray(point, dtype=float))
        remap[old_id] = int(new_id)

    clean_faces: list[list[int]] = []
    for face in faces:
        mapped: list[int] = []
        for value in face:
            idx = remap[int(value)]
            if not mapped or mapped[-1] != idx:
                mapped.append(idx)
        if len(mapped) > 1 and mapped[0] == mapped[-1]:
            mapped.pop()
        if len(set(mapped)) < 3:
            continue
        pts = np.asarray([merged[idx] for idx in mapped], dtype=float)
        if _polygon_area_3d(pts) <= tolerance * tolerance:
            continue
        clean_faces.append(mapped)

    if not merged or not clean_faces:
        return np.zeros((0, 3), dtype=float), [], {
            "merged_vertex_count": int(len(source)),
            "removed_face_count": int(len(faces)),
        }

    used = sorted({idx for face in clean_faces for idx in face})
    compact_map = {old: new for new, old in enumerate(used)}
    compact = np.asarray([merged[idx] for idx in used], dtype=float)
    compact_faces = [[compact_map[idx] for idx in face] for face in clean_faces]
    centroid = np.mean(compact, axis=0)
    for idx, face in enumerate(compact_faces):
        pts = compact[np.asarray(face, dtype=int)]
        normal = _face_normal(pts)
        if float(np.dot(normal, np.mean(pts, axis=0) - centroid)) < 0.0:
            compact_faces[idx] = list(reversed(face))
    return compact, compact_faces, {
        "merged_vertex_count": int(len(source) - len(compact)),
        "removed_face_count": int(len(faces) - len(compact_faces)),
    }


def clip_convex_polyhedron(
    vertices: np.ndarray,
    faces: list[list[int]],
    plane_normal: np.ndarray,
    plane_offset: float,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, list[list[int]], dict[str, object]]:
    """Clip a closed convex polyhedron to ``n dot x <= d``."""
    verts = np.asarray(vertices, dtype=float)
    raw_normal = np.asarray(plane_normal, dtype=float)
    normal_scale = float(np.linalg.norm(raw_normal))
    n = raw_normal / normal_scale if np.isfinite(normal_scale) and normal_scale > 1e-14 else np.zeros(3, dtype=float)
    normalized_offset = float(plane_offset) / normal_scale if normal_scale > 1e-14 else float(plane_offset)
    if len(verts) == 0 or not faces or float(np.linalg.norm(n)) <= 1e-14:
        return np.zeros((0, 3), dtype=float), [], {"empty": True, "cap_face_id": None}

    clipped_polygons: list[list[np.ndarray]] = []
    intersections: list[np.ndarray] = []
    for face in faces:
        polygon = [verts[int(idx)] for idx in face]
        output: list[np.ndarray] = []
        previous = polygon[-1]
        previous_distance = float(np.dot(n, previous) - normalized_offset)
        previous_inside = previous_distance <= tolerance
        for current in polygon:
            current_distance = float(np.dot(n, current) - normalized_offset)
            current_inside = current_distance <= tolerance
            if current_inside != previous_inside:
                denominator = previous_distance - current_distance
                alpha = 0.5 if abs(denominator) <= 1e-15 else previous_distance / denominator
                point = previous + float(np.clip(alpha, 0.0, 1.0)) * (current - previous)
                output.append(point)
                intersections.append(point)
            if current_inside:
                output.append(current)
            previous = current
            previous_distance = current_distance
            previous_inside = current_inside
        output = _dedupe_polygon(output, tolerance)
        if len(output) >= 3 and _polygon_area_3d(np.asarray(output)) > tolerance * tolerance:
            clipped_polygons.append(output)

    unique_intersections: list[np.ndarray] = []
    for point in intersections:
        if not any(float(np.linalg.norm(point - existing)) <= tolerance for existing in unique_intersections):
            unique_intersections.append(np.asarray(point, dtype=float))
    cap_added = False
    if len(unique_intersections) >= 3:
        center = np.mean(unique_intersections, axis=0)
        helper = np.asarray([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, n))) > 0.85:
            helper = np.asarray([0.0, 1.0, 0.0])
        u = _normalize(helper - n * float(np.dot(helper, n)))
        v = _normalize(np.cross(n, u))
        ordered = sorted(
            unique_intersections,
            key=lambda p: float(np.arctan2(np.dot(p - center, v), np.dot(p - center, u))),
        )
        if float(np.dot(_face_normal(np.asarray(ordered)), n)) < 0.0:
            ordered.reverse()
        clipped_polygons.append(ordered)
        cap_added = True

    if not clipped_polygons:
        return np.zeros((0, 3), dtype=float), [], {"empty": True, "cap_face_id": None}

    flat_vertices: list[np.ndarray] = []
    indexed_faces: list[list[int]] = []
    for polygon in clipped_polygons:
        face_ids: list[int] = []
        for point in polygon:
            vertex_id = next(
                (idx for idx, existing in enumerate(flat_vertices) if float(np.linalg.norm(point - existing)) <= tolerance),
                None,
            )
            if vertex_id is None:
                vertex_id = len(flat_vertices)
                flat_vertices.append(np.asarray(point, dtype=float))
            face_ids.append(int(vertex_id))
        indexed_faces.append(face_ids)
    clean_vertices, clean_faces, cleanup = cleanup_polyhedron(
        np.asarray(flat_vertices, dtype=float), indexed_faces, tolerance
    )
    cap_face_id = None
    if cap_added and len(clean_faces):
        candidate_ids = [
            idx
            for idx, face in enumerate(clean_faces)
            if max(abs(float(np.dot(n, clean_vertices[v]) - normalized_offset)) for v in face) <= tolerance * 5.0
        ]
        cap_face_id = candidate_ids[-1] if candidate_ids else None
    return clean_vertices, clean_faces, {
        "empty": len(clean_vertices) == 0 or len(clean_faces) == 0,
        "cap_face_id": cap_face_id,
        "cap_added": bool(cap_added),
        **cleanup,
    }


def signed_volume(vertices: np.ndarray, faces: list[list[int]]) -> float:
    verts = np.asarray(vertices, dtype=float)
    volume = 0.0
    for face in faces:
        if len(face) < 3:
            continue
        p0 = verts[face[0]]
        for idx in range(1, len(face) - 1):
            volume += float(np.dot(p0, np.cross(verts[face[idx]], verts[face[idx + 1]]))) / 6.0
    return float(volume)


def polyhedron_validation(
    vertices: np.ndarray,
    faces: list[list[int]],
    minimum_volume: float,
    minimum_feature_size: float,
) -> dict[str, object]:
    verts = np.asarray(vertices, dtype=float)
    edge_counts: dict[tuple[int, int], int] = {}
    edge_lengths: list[float] = []
    for face in faces:
        for idx, a in enumerate(face):
            b = face[(idx + 1) % len(face)]
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    for a, b in edge_counts:
        edge_lengths.append(float(np.linalg.norm(verts[a] - verts[b])))
    volume = signed_volume(verts, faces) if len(verts) else 0.0
    watertight = bool(edge_counts) and all(value == 2 for value in edge_counts.values())
    manifold = watertight
    minimum_edge = min(edge_lengths, default=0.0)
    finite = bool(np.all(np.isfinite(verts)))
    return {
        "finite": finite,
        "volume": float(volume),
        "minimum_feature_size": float(minimum_edge),
        "watertight": watertight,
        "manifold": manifold,
        "valid": bool(
            finite
            and watertight
            and manifold
            and volume >= float(minimum_volume)
            and minimum_edge >= float(minimum_feature_size)
        ),
    }


def validate_top_quad(top: np.ndarray, area_epsilon: float = 1e-12) -> tuple[bool, str]:
    points = np.asarray(top, dtype=float)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        return False, "nonfinite_or_wrong_shape"
    if min(float(np.linalg.norm(points[i] - points[j])) for i, j in itertools.combinations(range(4), 2)) <= 1e-10:
        return False, "duplicate_top_vertex"
    normal = _face_normal(points)
    if float(np.linalg.norm(normal)) <= 1e-12 or _polygon_area_3d(points) <= area_epsilon:
        return False, "degenerate_top"
    # Project to the dominant plane and reject bow-tie ordering.
    drop = int(np.argmax(np.abs(normal)))
    xy = np.delete(points, drop, axis=1)

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
        return orient(a, b, c) * orient(a, b, d) < 0.0 and orient(c, d, a) * orient(c, d, b) < 0.0

    if crosses(xy[0], xy[1], xy[2], xy[3]) or crosses(xy[1], xy[2], xy[3], xy[0]):
        return False, "self_intersecting_top"
    return True, ""


def _initial_local_box(top: np.ndarray, normal: np.ndarray, thickness: float, lateral_radius: float) -> tuple[np.ndarray, list[list[int]]]:
    center = np.mean(top, axis=0)
    n = _normalize(normal)
    edge = top[1] - top[0]
    u = _normalize(edge - n * float(np.dot(edge, n)), np.asarray([1.0, 0.0, 0.0]))
    v = _normalize(np.cross(n, u), np.asarray([0.0, 1.0, 0.0]))
    r = float(lateral_radius)
    upper = [center + sx * r * u + sy * r * v for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    lower = [point - float(thickness) * n for point in upper]
    vertices = np.asarray([*upper, *lower], dtype=float)
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    vertices, faces, _ = cleanup_polyhedron(vertices, faces)
    return vertices, faces


def build_tile_polyhedron(
    *,
    tile_id: int,
    top: np.ndarray,
    normal: np.ndarray,
    side_planes: list[tuple[np.ndarray, float]],
    requested_thickness: float,
    minimum_thickness: float,
    minimum_volume: float,
    minimum_feature_size: float,
    miter_jump_limit: float,
    tolerance: float = 1e-9,
) -> ConvexTileSolid:
    """Build one tile by half-space clipping with cap/thickness recovery."""
    valid_top, top_reason = validate_top_quad(top)
    if not valid_top:
        raise T3DConstructionError(T3D_FAILED_INVALID_TOP, top_reason, [tile_id])
    top = np.asarray(top, dtype=float)
    n = _normalize(normal)
    top_center = np.mean(top, axis=0)
    span = max(float(np.linalg.norm(top[(idx + 1) % 4] - top[idx])) for idx in range(4))

    def attempt(local_thickness: float, bounded: bool) -> tuple[ConvexTileSolid | None, dict[str, object]]:
        radius = max(span * 20.0, local_thickness * 40.0, 1.0)
        vertices, faces = _initial_local_box(top, n, local_thickness, radius)
        cap_count = 0
        cleanup_used = False
        planes: list[tuple[np.ndarray, float, str, int | None]] = [
            (n, float(np.dot(n, top_center)), "top", None),
            (-n, float(-np.dot(n, top_center) + local_thickness), "bottom", None),
        ]
        planes.extend((np.asarray(pn, dtype=float), float(pd), "contact", edge_id) for edge_id, (pn, pd) in enumerate(side_planes))
        for plane_normal, plane_offset, _kind, _edge_id in planes:
            vertices, faces, info = clip_convex_polyhedron(vertices, faces, plane_normal, plane_offset, tolerance)
            if bool(info.get("empty", False)):
                return None, {"reason": "empty_halfspace_intersection"}
            cleanup_used = cleanup_used or int(info.get("merged_vertex_count", 0)) > 0 or int(info.get("removed_face_count", 0)) > 0

        reasons: list[str] = []
        status = T3D_OK_NOMINAL_FRUSTUM
        if bounded:
            edge = top[1] - top[0]
            u = _normalize(edge - n * float(np.dot(edge, n)), np.asarray([1.0, 0.0, 0.0]))
            v = _normalize(np.cross(n, u), np.asarray([0.0, 1.0, 0.0]))
            projected_top = np.column_stack([(top - top_center) @ u, (top - top_center) @ v])
            max_u = float(np.max(np.abs(projected_top[:, 0]))) + float(miter_jump_limit) * local_thickness
            max_v = float(np.max(np.abs(projected_top[:, 1]))) + float(miter_jump_limit) * local_thickness
            cap_planes = [
                (u, float(np.dot(u, top_center) + max_u)),
                (-u, float(np.dot(-u, top_center) + max_u)),
                (v, float(np.dot(v, top_center) + max_v)),
                (-v, float(np.dot(-v, top_center) + max_v)),
            ]
            for plane_normal, plane_offset in cap_planes:
                if len(vertices) and float(np.max(vertices @ plane_normal - plane_offset)) > tolerance:
                    vertices, faces, info = clip_convex_polyhedron(vertices, faces, plane_normal, plane_offset, tolerance)
                    if bool(info.get("empty", False)):
                        return None, {"reason": "cap_emptied_polyhedron"}
                    if bool(info.get("cap_added", False)):
                        cap_count += 1
            if cap_count:
                status = T3D_RECOVERED_CAPPED_FRUSTUM
                reasons.append("large_miter_jump")

        validation = polyhedron_validation(vertices, faces, minimum_volume, minimum_feature_size)
        if not bool(validation["valid"]):
            return None, {"reason": "manufacturing_validation", **validation}
        depths = (top_center[None, :] - vertices) @ n
        bottom_ids = np.flatnonzero(np.abs(depths - local_thickness) <= max(tolerance * 10.0, local_thickness * 1e-7))
        bottom_vertex_count = int(len(bottom_ids))
        if bottom_vertex_count <= 1:
            status = T3D_RECOVERED_PYRAMID
            reasons.append("bottom_face_collapsed_to_point")
        elif bottom_vertex_count <= 3:
            status = T3D_RECOVERED_WEDGE
            reasons.append("bottom_face_reduced")
        actual_max_depth = float(max(0.0, np.max(depths)))
        if actual_max_depth < local_thickness * (1.0 - 1e-7):
            status = T3D_RECOVERED_LOCAL_THICKNESS
            reasons.append("feasible_solid_terminates_before_requested_bottom_plane")
        elif local_thickness < requested_thickness * (1.0 - 1e-8):
            status = T3D_RECOVERED_LOCAL_THICKNESS
            reasons.append("nominal_thickness_infeasible")
        elif cleanup_used and status == T3D_OK_NOMINAL_FRUSTUM:
            # Cleanup is routine for clipping; report it only when topology changed.
            if len(vertices) != 8 or len(faces) != 6:
                status = T3D_RECOVERED_MESH_CLEANUP
                reasons.append("sliver_cleanup")

        top_face_ids = [
            face_id
            for face_id, face in enumerate(faces)
            if all(abs(float(np.dot(n, vertices[idx] - top_center))) <= tolerance * 20.0 for idx in face)
        ]
        contact_face_by_edge: dict[int, int] = {}
        for edge_id, (plane_normal, plane_offset) in enumerate(side_planes):
            pn = _normalize(plane_normal)
            candidates = [
                face_id
                for face_id, face in enumerate(faces)
                if all(abs(float(np.dot(pn, vertices[idx]) - plane_offset)) <= tolerance * 20.0 for idx in face)
            ]
            if candidates:
                contact_face_by_edge[edge_id] = int(candidates[0])
        metrics: dict[str, object] = {
            "tile_id": int(tile_id),
            "status": status,
            "triggered_conditions": list(reasons),
            "recovery_steps": list(reasons),
            "requested_thickness": float(requested_thickness),
            "actual_min_depth": float(max(0.0, np.min(depths))),
            "actual_max_depth": actual_max_depth,
            "local_thickness_ratio": float(actual_max_depth / max(requested_thickness, 1e-15)),
            "volume": float(validation["volume"]),
            "minimum_width": float(validation["minimum_feature_size"]),
            "minimum_feature_size": float(validation["minimum_feature_size"]),
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "cap_face_count": int(cap_count),
            "collision_count_before": 0,
            "collision_count_after": 0,
            "shared_plane_error": 0.0,
            "watertight": bool(validation["watertight"]),
            "manifold": bool(validation["manifold"]),
        }
        return ConvexTileSolid(vertices, faces, top_face_ids, contact_face_by_edge, status, reasons, metrics), validation

    nominal, nominal_info = attempt(float(requested_thickness), False)
    if nominal is not None:
        # Far intersections are valid mathematically but not acceptable geometry.
        lateral = nominal.vertices - top_center[None, :]
        lateral -= np.outer(lateral @ n, n)
        if float(np.max(np.linalg.norm(lateral, axis=1))) <= span + miter_jump_limit * requested_thickness:
            return nominal
    capped, capped_info = attempt(float(requested_thickness), True)
    if capped is not None:
        return capped

    low = float(minimum_thickness)
    high = float(requested_thickness)
    best: ConvexTileSolid | None = None
    for _ in range(18):
        candidate_h = low if best is None and high == requested_thickness else 0.5 * (low + high)
        candidate, _info = attempt(candidate_h, True)
        if candidate is not None:
            best = candidate
            low = candidate_h
        else:
            high = candidate_h
    if best is not None:
        best.recovery_status = T3D_RECOVERED_LOCAL_THICKNESS
        best.recovery_reasons.append("nominal_thickness_infeasible")
        best.metrics["status"] = best.recovery_status
        best.metrics["recovery_steps"] = list(best.recovery_reasons)
        best.metrics["local_thickness_ratio"] = float(best.metrics["actual_max_depth"]) / max(requested_thickness, 1e-15)
        return best
    failure_status = (
        T3D_FAILED_BELOW_MANUFACTURING_LIMIT
        if nominal_info.get("reason") == "manufacturing_validation" or capped_info.get("reason") == "manufacturing_validation"
        else T3D_FAILED_EMPTY_FEASIBLE_REGION
    )
    raise T3DConstructionError(failure_status, "all nominal/cap/local-thickness recovery tiers failed", [tile_id])


def triangulate_solid(solid: ConvexTileSolid) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for face in solid.faces:
        for idx in range(1, len(face) - 1):
            triangles.append((int(face[0]), int(face[idx]), int(face[idx + 1])))
    return triangles


def build_emergency_normal_prism(
    *,
    tile_id: int,
    top: np.ndarray,
    normal: np.ndarray,
    thickness: float,
    triggering_status: str,
    triggering_reason: str,
) -> ConvexTileSolid:
    """Build the last-resort one-sided prism used only for visible recovery.

    Invalid K3D top faces are deliberately rejected: this fallback may replace
    an infeasible thickness/contact construction, but it must not invent a new
    mandatory top polygon.
    """
    valid, reason = validate_top_quad(top)
    if not valid:
        raise T3DConstructionError(T3D_FAILED_INVALID_TOP, reason, [tile_id])
    top = np.asarray(top, dtype=float)
    n = _normalize(np.asarray(normal, dtype=float), _face_normal(top))
    bottom = top - float(thickness) * n[None, :]
    vertices = np.vstack([top, bottom])
    faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    vertices, faces, _cleanup = cleanup_polyhedron(vertices, faces)
    quality = polyhedron_validation(vertices, faces, 0.0, 0.0)
    reasons = [
        "all_geometric_recovery_tiers_exhausted",
        f"trigger_status={triggering_status}",
        triggering_reason,
    ]
    metrics: dict[str, object] = {
        "tile_id": int(tile_id),
        "status": T3D_RECOVERED_LEGACY_EMERGENCY_PRISM,
        "triggered_conditions": [triggering_status, triggering_reason],
        "recovery_steps": reasons,
        "requested_thickness": float(thickness),
        "actual_min_depth": 0.0,
        "actual_max_depth": float(thickness),
        "local_thickness_ratio": 1.0,
        "volume": float(quality["volume"]),
        "minimum_width": float(quality["minimum_feature_size"]),
        "minimum_feature_size": float(quality["minimum_feature_size"]),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "cap_face_count": 0,
        "collision_count_before": 0,
        "collision_count_after": 0,
        "shared_plane_error": None,
        "watertight": bool(quality["watertight"]),
        "manifold": bool(quality["manifold"]),
        "legacy_emergency_fallback": True,
        "manufacturing_authoritative": False,
        "display_color": "gray",
    }
    return ConvexTileSolid(
        vertices=vertices,
        faces=faces,
        top_face_ids=[0],
        contact_face_by_edge={},
        recovery_status=T3D_RECOVERED_LEGACY_EMERGENCY_PRISM,
        recovery_reasons=reasons,
        metrics=metrics,
    )


def aabb_overlap(a: ConvexTileSolid, b: ConvexTileSolid, tolerance: float = 1e-9) -> bool:
    a_min, a_max = np.min(a.vertices, axis=0), np.max(a.vertices, axis=0)
    b_min, b_max = np.min(b.vertices, axis=0), np.max(b.vertices, axis=0)
    return bool(np.all(a_max > b_min + tolerance) and np.all(b_max > a_min + tolerance))


def convex_sat_overlap(a: ConvexTileSolid, b: ConvexTileSolid, tolerance: float = 1e-9) -> bool:
    """Conservative convex SAT using face normals and edge cross products."""
    axes: list[np.ndarray] = []
    edge_vectors: list[list[np.ndarray]] = [[], []]
    for solid_index, solid in enumerate((a, b)):
        for face in solid.faces:
            normal = _face_normal(solid.vertices[np.asarray(face, dtype=int)])
            if float(np.linalg.norm(normal)) > 1e-12:
                axes.append(normal)
            for idx, vertex_id in enumerate(face):
                edge = solid.vertices[face[(idx + 1) % len(face)]] - solid.vertices[vertex_id]
                if float(np.linalg.norm(edge)) > 1e-12:
                    edge_vectors[solid_index].append(_normalize(edge))
    for edge_a in edge_vectors[0]:
        for edge_b in edge_vectors[1]:
            axis = _normalize(np.cross(edge_a, edge_b))
            if float(np.linalg.norm(axis)) > 1e-10:
                axes.append(axis)
    for axis in axes:
        proj_a = a.vertices @ axis
        proj_b = b.vertices @ axis
        if float(np.max(proj_a)) <= float(np.min(proj_b)) + tolerance:
            return False
        if float(np.max(proj_b)) <= float(np.min(proj_a)) + tolerance:
            return False
    return True
