r"""Side-face/contact-aware T3D extrusion patch.

This file is intentionally a compatibility wrapper for the user's existing
onestring_pipeline.py.  It loads the backed-up original module and replaces only
_extrude_tiles() with a miter/contact-plane version, so the old Copy-Item based
workflow can be used without shipping a full copy of the large source file.

Expected workflow:
  Copy-Item .\src .\src_backup_before_sideface_contact -Recurse -Force
  Copy-Item .\sideface_contact_tmp\onestring_physics\* .\src\onestring_physics\ -Recurse -Force
"""

from __future__ import annotations

import importlib.util
import copy
from dataclasses import dataclass
import heapq
import math
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np


def _project_root_from_this_file() -> Path:
    # <project>/src/onestring_physics/onestring_pipeline.py
    return Path(__file__).resolve().parents[2]


def _find_original_pipeline() -> Path:
    root = _project_root_from_this_file()
    candidates = [
        root / "src_backup_before_sideface_contact" / "onestring_physics" / "onestring_pipeline.py",
        root / "src_backup_before_mitered_t3d" / "onestring_physics" / "onestring_pipeline.py",
        root / "src_backup_before_sideface_contact" / "src" / "onestring_physics" / "onestring_pipeline.py",
        root / "src_backup_before_mitered_t3d" / "src" / "onestring_physics" / "onestring_pipeline.py",
        root / "src" / "onestring_physics" / "onestring_pipeline.py.bak_mitered_t3d",
    ]
    for path in candidates:
        if not path.exists() or path.resolve() == Path(__file__).resolve():
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:2500]
        except Exception:
            head = ""
        # If the user re-runs the old copy commands after a failed patch, the
        # backup directory may accidentally contain this wrapper instead of the
        # real original file.  Skip wrapper backups to avoid recursive imports and
        # continue to older backups such as src_backup_before_mitered_t3d.
        if "Side-face/contact-aware T3D extrusion patch" in head and "_find_original_pipeline" in head:
            continue
        return path
    tried = "\n  - ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Could not find the original onestring_pipeline.py backup.\n"
        "Run the backup command before copying this patch:\n"
        "  Copy-Item .\\src .\\src_backup_before_sideface_contact -Recurse -Force\n\n"
        f"Tried:\n  - {tried}"
    )


_ORIGINAL_PATH = _find_original_pipeline()
_ORIGINAL_MODULE_NAME = "onestring_physics._onestring_pipeline_original_sideface_contact"

_spec = importlib.util.spec_from_file_location(_ORIGINAL_MODULE_NAME, _ORIGINAL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load original pipeline from {_ORIGINAL_PATH}")
_original = importlib.util.module_from_spec(_spec)
sys.modules[_ORIGINAL_MODULE_NAME] = _original
_spec.loader.exec_module(_original)


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12 or not np.isfinite(n):
        if fallback is None:
            return np.zeros_like(arr, dtype=float)
        fb = np.asarray(fallback, dtype=float)
        fb_n = float(np.linalg.norm(fb))
        return fb / max(fb_n, 1e-12)
    return arr / n


def _edge_inward_normal(top: np.ndarray, face_normal: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    """Plane normal for the side face through a top edge, pointing into the tile.

    The normal lies in the tile plane and is perpendicular to the edge.  Its sign
    is chosen so that the tile center is on the positive side.
    """
    a, b = edge
    p0 = np.asarray(top[a], dtype=float)
    p1 = np.asarray(top[b], dtype=float)
    center = np.mean(top, axis=0)
    edge_dir = _normalize(p1 - p0, np.array([1.0, 0.0, 0.0]))
    q = np.cross(edge_dir, face_normal)
    q = _normalize(q, np.array([0.0, 1.0, 0.0]))
    mid = 0.5 * (p0 + p1)
    if float(np.dot(q, center - mid)) < 0.0:
        q = -q
    return q


def _build_edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tile_id, face in enumerate(np.asarray(faces, dtype=int)):
        for edge_id, (a, b) in enumerate(local_edges):
            key = tuple(sorted((int(face[a]), int(face[b]))))
            incidence.setdefault(key, []).append((int(tile_id), int(edge_id)))
    return incidence


def _query_kdtree_indices(tree, points: np.ndarray, k: int, n_items: int) -> np.ndarray:
    if n_items <= 0:
        return np.asarray([], dtype=int).reshape(len(np.asarray(points).reshape(-1, points.shape[-1])), 0)
    kk = max(1, min(int(k), int(n_items)))
    _dist, idx = tree.query(points, k=kk)
    idx_arr = np.asarray(idx, dtype=int)
    if idx_arr.ndim == 1:
        idx_arr = idx_arr[:, None] if np.asarray(points).ndim > 1 else idx_arr.reshape(1, 1)
    idx_arr = np.clip(idx_arr, 0, n_items - 1)
    return idx_arr


def _uv_triangle_kdtree(parameterization):
    cache = getattr(parameterization, "_onestring_uv_triangle_kdtree_cache", None)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)
    uv_vertices = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    if isinstance(cache, dict) and cache.get("face_count") == len(uv_faces) and cache.get("vertex_count") == len(uv_vertices):
        return cache
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    if len(uv_faces) == 0 or len(uv_vertices) == 0:
        return None
    triangles = uv_vertices[uv_faces]
    centers = np.mean(triangles, axis=1)
    radii = np.max(np.linalg.norm(triangles - centers[:, None, :], axis=2), axis=1)
    cache = {
        "tree": cKDTree(centers),
        "centers": centers,
        "radii": radii,
        "face_count": int(len(uv_faces)),
        "vertex_count": int(len(uv_vertices)),
    }
    try:
        setattr(parameterization, "_onestring_uv_triangle_kdtree_cache", cache)
    except Exception:
        pass
    return cache


def inverse_map_uv_to_surface(
    uv_point: np.ndarray,
    parameterization,
) -> tuple[np.ndarray, int, bool]:
    accelerated = _original._inverse_map_uv_to_surface_regular(uv_point, parameterization)
    if accelerated is not None:
        return accelerated

    uv = np.asarray(uv_point, dtype=float)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)
    uv_vertices = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    surface_vertices = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    if len(uv_faces) == 0 or len(uv_vertices) == 0:
        return np.zeros(3, dtype=float), -1, True

    cache = _uv_triangle_kdtree(parameterization)
    candidate_ids: list[int] = []
    if cache is not None:
        tree = cache["tree"]
        n_faces = int(len(uv_faces))
        for k in (8, 24, 64, 160, 384):
            idx = _query_kdtree_indices(tree, uv.reshape(1, 2), k, n_faces).reshape(-1)
            candidate_ids = [int(i) for i in idx]
            for tri_id in candidate_ids:
                face = uv_faces[tri_id]
                bary = _original._barycentric_2d(uv, uv_vertices[face])
                if bary is None:
                    continue
                if float(np.min(bary)) >= -1e-9:
                    surface_tri = surface_vertices[np.asarray(parameterization.surface_faces, dtype=int)[tri_id]]
                    return bary @ surface_tri, int(tri_id), False
            if k >= n_faces:
                break
    else:
        candidate_ids = list(range(len(uv_faces)))

    best_tri = -1
    best_bary: np.ndarray | None = None
    best_score = float("inf")
    seen: set[int] = set()
    for tri_id in candidate_ids:
        if tri_id in seen:
            continue
        seen.add(int(tri_id))
        face = uv_faces[int(tri_id)]
        bary = _original._barycentric_2d(uv, uv_vertices[face])
        if bary is None:
            continue
        score = abs(float(np.min(bary)))
        if score < best_score:
            best_score = score
            best_tri = int(tri_id)
            best_bary = bary

    if best_tri < 0 or best_bary is None:
        try:
            from scipy.spatial import cKDTree
            vertex_cache = getattr(parameterization, "_onestring_uv_vertex_kdtree_cache", None)
            if not isinstance(vertex_cache, dict) or vertex_cache.get("vertex_count") != len(uv_vertices):
                vertex_cache = {"tree": cKDTree(uv_vertices), "vertex_count": int(len(uv_vertices))}
                setattr(parameterization, "_onestring_uv_vertex_kdtree_cache", vertex_cache)
            _dist, nearest = vertex_cache["tree"].query(uv, k=1)
            nearest_idx = int(nearest)
        except Exception:
            nearest_idx = int(np.argmin(np.linalg.norm(uv_vertices - uv, axis=1)))
        return surface_vertices[nearest_idx].copy(), -1, True

    clipped = np.clip(best_bary, 0.0, 1.0)
    total = float(np.sum(clipped))
    clipped = clipped / total if total > 1e-12 else np.asarray([1.0, 0.0, 0.0])
    surface_tri = surface_vertices[np.asarray(parameterization.surface_faces, dtype=int)[best_tri]]
    return clipped @ surface_tri, best_tri, True


def _surface_triangle_kdtree(surface_vertices: np.ndarray, surface_faces: np.ndarray):
    triangles = np.asarray(surface_vertices, dtype=float)[np.asarray(surface_faces, dtype=int)]
    if len(triangles) == 0:
        return None
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    centers = np.mean(triangles, axis=1)
    radii = np.max(np.linalg.norm(triangles - centers[:, None, :], axis=2), axis=1)
    return {"tree": cKDTree(centers), "triangles": triangles, "radii": radii}


def _closest_points_on_surface_mesh(points: np.ndarray, surface_vertices: np.ndarray, surface_faces: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    triangles = np.asarray(surface_vertices, dtype=float)[np.asarray(surface_faces, dtype=int)]
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(triangles) == 0:
        return pts.copy()
    accel = _surface_triangle_kdtree(surface_vertices, surface_faces)
    if accel is None:
        return _original._closest_points_on_surface_mesh(points, surface_vertices, surface_faces)

    tree = accel["tree"]
    n_triangles = int(len(triangles))
    candidate_count = min(96, n_triangles)
    ids = _query_kdtree_indices(tree, pts, candidate_count, n_triangles)
    values: list[np.ndarray] = []
    for point, candidates in zip(pts, ids):
        best = float("inf")
        best_point = triangles[int(candidates[0]), 0]
        for tri_id in candidates:
            tri = triangles[int(tri_id)]
            closest = _original._closest_point_on_triangle(point, tri[0], tri[1], tri[2])
            dist = float(np.linalg.norm(point - closest))
            if dist < best:
                best = dist
                best_point = closest
        values.append(best_point)
    return np.asarray(values, dtype=float)


def _distances_to_surface_mesh(points: np.ndarray, surface_vertices: np.ndarray, surface_faces: np.ndarray) -> np.ndarray:
    closest = _closest_points_on_surface_mesh(points, surface_vertices, surface_faces)
    return np.linalg.norm(np.asarray(points, dtype=float) - closest, axis=1)


def _coincident_key(point: np.ndarray, tolerance: float) -> tuple[int, ...]:
    scale = max(float(tolerance), 1e-12)
    return tuple(np.round(np.asarray(point, dtype=float).reshape(-1) / scale).astype(np.int64).tolist())


def _coincident_tolerance(vertices: np.ndarray) -> float:
    pts = np.asarray(vertices, dtype=float)
    if pts.size == 0:
        return 1e-7
    span = float(np.nanmax(pts) - np.nanmin(pts))
    return max(1e-7, span * 1e-7)


def _canonicalize_faces_by_coincident_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Weld reference-coincident split vertex ids for layout/connectivity only."""
    faces_arr = np.asarray(faces, dtype=int)
    verts = np.asarray(vertices, dtype=float)
    if faces_arr.size == 0 or verts.size == 0:
        return faces_arr.copy(), {
            "split_virtual_weld_applied": False,
            "split_virtual_weld_group_count": 0,
            "split_virtual_weld_vertex_count": 0,
        }
    if int(np.max(faces_arr)) >= len(verts):
        return faces_arr.copy(), {
            "split_virtual_weld_applied": False,
            "split_virtual_weld_group_count": 0,
            "split_virtual_weld_vertex_count": 0,
            "split_virtual_weld_reason": "face_index_outside_vertex_array",
        }

    tol = _coincident_tolerance(verts) if tolerance is None else float(tolerance)
    groups: dict[tuple[int, ...], list[int]] = {}
    used = sorted({int(v) for v in faces_arr.reshape(-1)})
    for idx in used:
        groups.setdefault(_coincident_key(verts[idx], tol), []).append(idx)

    canonical = {idx: idx for idx in used}
    group_count = 0
    vertex_count = 0
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        root = int(min(ids))
        group_count += 1
        vertex_count += len(ids)
        for idx in ids:
            canonical[int(idx)] = root

    if group_count == 0:
        return faces_arr.copy(), {
            "split_virtual_weld_applied": False,
            "split_virtual_weld_group_count": 0,
            "split_virtual_weld_vertex_count": 0,
        }

    welded = faces_arr.copy()
    for old, new in canonical.items():
        if old != new:
            welded[welded == old] = new
    return welded, {
        "split_virtual_weld_applied": True,
        "split_virtual_weld_group_count": int(group_count),
        "split_virtual_weld_vertex_count": int(vertex_count),
        "split_virtual_weld_tolerance": float(tol),
        "split_virtual_weld_reason": "reference-coincident split vertices treated as connected for layout/contact constraints",
    }


def _canonicalize_faces_by_coincident_tile_tops(
    top_tiles: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    faces_arr = np.asarray(faces, dtype=int)
    tops = np.asarray(top_tiles, dtype=float)
    if faces_arr.size == 0 or tops.size == 0 or len(tops) != len(faces_arr):
        return faces_arr.copy(), {
            "split_virtual_weld_applied": False,
            "split_virtual_weld_group_count": 0,
            "split_virtual_weld_vertex_count": 0,
            "split_virtual_weld_reason": "empty_or_mismatched_tile_tops",
        }

    coords_by_vertex: dict[int, list[np.ndarray]] = {}
    for tile_id, face in enumerate(faces_arr):
        for local_id, vertex_id in enumerate(face):
            coords_by_vertex.setdefault(int(vertex_id), []).append(np.asarray(tops[tile_id, local_id], dtype=float))
    if not coords_by_vertex:
        return faces_arr.copy(), {
            "split_virtual_weld_applied": False,
            "split_virtual_weld_group_count": 0,
            "split_virtual_weld_vertex_count": 0,
        }

    max_id = max(coords_by_vertex)
    vertices = np.zeros((max_id + 1, tops.shape[-1]), dtype=float)
    for vertex_id, coords in coords_by_vertex.items():
        vertices[int(vertex_id)] = np.mean(np.asarray(coords, dtype=float), axis=0)
    return _canonicalize_faces_by_coincident_vertices(vertices, faces_arr, tolerance=tolerance)


def _coincident_boundary_edge_pairs(
    top_tiles: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float | None = None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Pair split boundary edges that occupy the same geometric segment."""
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    tops = np.asarray(top_tiles, dtype=float)
    incidence = _build_edge_incidence(faces)
    if tops.size == 0:
        return []
    tol = _coincident_tolerance(tops.reshape(-1, tops.shape[-1])) if tolerance is None else float(tolerance)
    buckets: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, int]]] = {}
    for entries in incidence.values():
        if len(entries) != 1:
            continue
        tile_id, edge_id = entries[0]
        a, b = local_edges[int(edge_id)]
        key = tuple(sorted((_coincident_key(tops[tile_id, a], tol), _coincident_key(tops[tile_id, b], tol))))
        buckets.setdefault(key, []).append((int(tile_id), int(edge_id)))

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for entries in buckets.values():
        if len(entries) == 2 and entries[0][0] != entries[1][0]:
            pairs.append((entries[0], entries[1]))
    return pairs


def _parameterization_stretch_csf(parameterization) -> np.ndarray:
    """Estimate per-UV-vertex conformal stretch from paired 3D/UV mesh edges."""
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)
    surface_faces = np.asarray(parameterization.surface_faces, dtype=int)
    if uv.size == 0 or xyz.size == 0 or uv_faces.size == 0:
        return np.ones(len(uv), dtype=float)

    values: list[list[float]] = [[] for _ in range(len(uv))]
    ratios: list[float] = []
    for uv_face, surface_face in zip(uv_faces, surface_faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ua, ub = int(uv_face[a]), int(uv_face[b])
            sa, sb = int(surface_face[a]), int(surface_face[b])
            uv_len = float(np.linalg.norm(uv[ub] - uv[ua]))
            xyz_len = float(np.linalg.norm(xyz[sb] - xyz[sa]))
            if uv_len <= 1e-12 or not np.isfinite(uv_len) or not np.isfinite(xyz_len):
                continue
            ratio = xyz_len / uv_len
            if ratio <= 0.0 or not np.isfinite(ratio):
                continue
            ratios.append(ratio)
            values[ua].append(ratio)
            values[ub].append(ratio)
    if not ratios:
        return np.ones(len(uv), dtype=float)

    # Normalize out global UV scale.  The split test should react to local
    # over-stretch, not to the arbitrary size of the Omega embedding.
    baseline = float(np.median(ratios))
    baseline = baseline if baseline > 1e-12 and np.isfinite(baseline) else 1.0
    csf = np.ones(len(uv), dtype=float)
    for idx, local in enumerate(values):
        if local:
            csf[idx] = float(np.percentile(local, 90)) / baseline
    return np.maximum(csf, 1.0)


def _nearest_reflection_error(points: np.ndarray, coord: int) -> tuple[float, float, float]:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    span = float(np.max(np.maximum(hi - lo, 1e-12)))
    center = 0.5 * (float(lo[coord]) + float(hi[coord]))
    mirrored = pts.copy()
    mirrored[:, coord] = 2.0 * center - mirrored[:, coord]
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        dist, _ = tree.query(mirrored, k=1)
    except Exception:
        diff = mirrored[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    rms = float(np.sqrt(np.mean(dist * dist))) if len(dist) else 0.0
    max_err = float(np.max(dist)) if len(dist) else 0.0
    return rms / max(span, 1e-12), max_err / max(span, 1e-12), center


def _detect_parameterization_reflection_symmetry(parameterization, tolerance: float = 0.025) -> dict[str, object]:
    surface = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    axes: list[int] = []
    details: dict[str, float | bool] = {}
    centers: dict[int, float] = {}
    for coord, label in ((0, "x"), (1, "y")):
        s_rms, s_max, _ = _nearest_reflection_error(surface, coord)
        uv_rms, uv_max, uv_center = _nearest_reflection_error(uv, coord)
        ok = bool(s_rms <= tolerance and uv_rms <= tolerance and s_max <= tolerance * 4.0 and uv_max <= tolerance * 4.0)
        details[f"{label}_surface_symmetry_rms_norm"] = s_rms
        details[f"{label}_surface_symmetry_max_norm"] = s_max
        details[f"{label}_omega_symmetry_rms_norm"] = uv_rms
        details[f"{label}_omega_symmetry_max_norm"] = uv_max
        details[f"{label}_symmetry_preserved_for_m2d"] = ok
        if ok:
            axes.append(coord)
            centers[coord] = uv_center
    return {
        "axes": axes,
        "centers": centers,
        "tolerance": float(tolerance),
        "details": details,
    }


def _surface_peak_uvs(parameterization, max_peaks: int = 8) -> np.ndarray:
    surface = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    faces = np.asarray(getattr(parameterization, "surface_faces", np.zeros((0, 3))), dtype=int)
    if len(surface) == 0 or len(uv) != len(surface):
        return np.zeros((0, 2), dtype=float)
    z = surface[:, 2]
    z_span = float(np.nanmax(z) - np.nanmin(z)) if len(z) else 0.0
    if z_span <= 1e-12:
        return np.zeros((0, 2), dtype=float)

    adjacency: list[set[int]] = [set() for _ in range(len(surface))]
    for face in faces:
        ids = [int(v) for v in face]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if 0 <= a < len(surface) and 0 <= b < len(surface):
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    eps = max(1e-9, z_span * 1e-5)
    high_floor = float(np.nanmax(z)) - 0.25 * z_span
    candidates: set[int] = set()
    for idx, value in enumerate(z):
        if float(value) < high_floor:
            continue
        neighbors = adjacency[idx]
        if not neighbors or all(float(value) >= float(z[n]) - eps for n in neighbors):
            candidates.add(idx)
    if not candidates:
        candidates = set(np.flatnonzero(z >= float(np.nanmax(z)) - max(1e-9, z_span * 1e-4)).tolist())

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(candidates):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in adjacency[node]:
                if nxt in candidates and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(component)

    scored: list[tuple[float, np.ndarray]] = []
    for component in components:
        comp = np.asarray(component, dtype=int)
        comp_max = float(np.max(z[comp]))
        top = comp[z[comp] >= comp_max - max(1e-9, z_span * 0.02)]
        scored.append((comp_max, np.mean(uv[top], axis=0)))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return np.zeros((0, 2), dtype=float)
    best = scored[0][0]
    kept = [point for score, point in scored if score >= best - 0.2 * z_span][: max(1, int(max_peaks))]
    return np.asarray(kept, dtype=float)


def _csf_split_lines_from_high_stretch(uv: np.ndarray, high: np.ndarray, max_splits: int) -> list[tuple[str, float]]:
    lo = np.nanmin(uv, axis=0)
    hi = np.nanmax(uv, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    margin = 0.08 * span
    spread = np.nanmax(high, axis=0) - np.nanmin(high, axis=0) if len(high) > 1 else np.zeros(2)
    candidates: list[tuple[str, float, float]] = []
    # A high-stretch band extended in x is cut by a horizontal row split; a band
    # extended in y is cut by a vertical column split.
    candidates.append(("row", float(np.median(high[:, 1])), float(spread[0])))
    candidates.append(("col", float(np.median(high[:, 0])), float(spread[1])))
    candidates.sort(key=lambda item: item[2], reverse=True)

    lines: list[tuple[str, float]] = []
    for axis, value, score in candidates:
        if len(lines) >= max_splits or score <= 1e-12:
            break
        coord = 1 if axis == "row" else 0
        if value <= lo[coord] + margin[coord] or value >= hi[coord] - margin[coord]:
            continue
        if any(existing_axis == axis and abs(existing_value - value) < 0.03 * span[coord] for existing_axis, existing_value in lines):
            continue
        lines.append((axis, value))
    return lines


def _peak_guided_csf_split_lines(parameterization, csf: np.ndarray, threshold: float, max_splits: int) -> list[tuple[str, float]]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    if len(uv) == 0 or np.max(np.asarray(csf, dtype=float)) <= float(threshold):
        return []
    peaks = _surface_peak_uvs(parameterization)
    if len(peaks) == 0:
        return []

    lo = np.nanmin(uv, axis=0)
    hi = np.nanmax(uv, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    margin = 0.08 * span
    spread = np.nanmax(peaks, axis=0) - np.nanmin(peaks, axis=0) if len(peaks) > 1 else np.zeros(2)
    high = uv[np.asarray(csf, dtype=float) > float(threshold)]
    high_spread = np.nanmax(high, axis=0) - np.nanmin(high, axis=0) if len(high) > 1 else np.zeros(2)

    if len(peaks) >= 2:
        axis = "row" if spread[0] >= spread[1] else "col"
    else:
        axis = "row" if high_spread[0] >= high_spread[1] else "col"
    coord = 1 if axis == "row" else 0
    value = float(np.median(peaks[:, coord]))
    if value <= lo[coord] + margin[coord] or value >= hi[coord] - margin[coord]:
        return []
    return [(axis, value)][: max(1, int(max_splits))]


def _csf_split_lines(parameterization, csf: np.ndarray, threshold: float = 2.0, max_splits: int = 1) -> list[tuple[str, float]]:
    """Choose coarse Omega split lines, preferring paths through surface peaks."""
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    if uv.size == 0 or csf.size == 0:
        return []
    high = uv[np.asarray(csf, dtype=float) > float(threshold)]
    if len(high) == 0:
        return []
    peak_guided = _peak_guided_csf_split_lines(parameterization, csf, threshold, max_splits)
    if peak_guided:
        return peak_guided
    return _csf_split_lines_from_high_stretch(uv, high, max_splits)


def _mirror_csf_split_lines(
    lines: list[tuple[str, float]],
    symmetry_axes: list[int],
    centers: dict[int, float],
    uv_vertices: np.ndarray,
) -> list[tuple[str, float]]:
    if not lines or not symmetry_axes:
        return lines
    uv = np.asarray(uv_vertices, dtype=float)
    lo = np.nanmin(uv, axis=0)
    hi = np.nanmax(uv, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    out = list(lines)
    for axis, value in list(lines):
        coord = 1 if axis == "row" else 0
        if coord not in symmetry_axes:
            continue
        mirrored = 2.0 * float(centers.get(coord, 0.0)) - float(value)
        if mirrored <= lo[coord] + 0.08 * span[coord] or mirrored >= hi[coord] - 0.08 * span[coord]:
            continue
        if any(existing_axis == axis and abs(existing_value - mirrored) < 0.03 * span[coord] for existing_axis, existing_value in out):
            continue
        out.append((axis, float(mirrored)))
    return out


_ORIGINAL_FLATTEN_TO_DOMAIN = _original._flatten_to_domain
_ORIGINAL_BUILD_M2D = _original._build_m2d
_ORIGINAL_BUILD_SURFACE_PARAMETERIZATION = _original._build_surface_parameterization
_ORIGINAL_LIFT_M2D_TO_M3D = _original._lift_m2d_to_m3d
_ORIGINAL_MAKE_FLAT_TILE_LAYOUT = _original._make_flat_tile_layout
_ORIGINAL_OPTIMIZE_K2D = _original._optimize_k2d
_ORIGINAL_OPTIMIZE_K3D = _original._optimize_k3d
_ORIGINAL_PAPER_LOCAL_GLOBAL_SE2_LAYOUT = _original._paper_local_global_se2_layout
_ORIGINAL_SPATIAL_CANDIDATE_PAIRS_FOR_TILES = _original._spatial_candidate_pairs_for_tiles


@dataclass
class PipelineParameters(_original.PipelineParameters):
    omega_boundary_mode: Literal["rectangular_debug", "shape_preserving_experimental", "paper_default"] = "paper_default"
    omega_parameterization_mode: Literal[
        "bff",
        "pca_debug",
        "lscm_paper_like",
        "arap_paper_like",
        "paper_like_unimplemented",
    ] = "bff"
    allow_experimental_pipeline: bool = False
    enable_heuristic_csf_split: bool = True
    enable_peak_guided_split: bool = True
    enable_mirror_split: bool = True
    hinge_layout_connection_weight: float = 8.0
    hinge_layout_collision_weight: float = 4.0
    hinge_layout_anchor_weight: float = 0.0
    hinge_layout_initial_expansion: float = 1.6
    hinge_layout_max_center_drift_tiles: float = 5.0


def _triangle_area_3d(points: np.ndarray) -> np.ndarray:
    tri = np.asarray(points, dtype=float)
    if tri.size == 0:
        return np.zeros(0, dtype=float)
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _triangle_signed_area_2d(points: np.ndarray) -> np.ndarray:
    tri = np.asarray(points, dtype=float)
    if tri.size == 0:
        return np.zeros(0, dtype=float)
    return 0.5 * (
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    )


def _segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < -1e-12 and o3 * o4 < -1e-12:
        return True
    if abs(o1) <= 1e-12 and on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-12 and on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-12 and on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-12 and on_segment(c, b, d):
        return True
    return False


def _boundary_self_intersection_count(boundary: np.ndarray) -> int:
    pts = np.asarray(boundary, dtype=float)
    if len(pts) < 4:
        return 0
    if np.linalg.norm(pts[0] - pts[-1]) <= 1e-12:
        pts = pts[:-1]
    n = len(pts)
    count = 0
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            c = pts[j]
            d = pts[(j + 1) % n]
            if _segments_intersect_2d(a, b, c, d):
                count += 1
    return int(count)


def _edge_stretch_values(surface_vertices: np.ndarray, uv_vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    xyz = np.asarray(surface_vertices, dtype=float)
    uv = np.asarray(uv_vertices, dtype=float)
    face_array = np.asarray(faces, dtype=int)
    values: list[float] = []
    seen: set[tuple[int, int]] = set()
    for face in face_array:
        ids = [int(v) for v in face[:3]]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            key = tuple(sorted((a, b)))
            if key in seen or a < 0 or b < 0 or a >= len(xyz) or b >= len(xyz) or a >= len(uv) or b >= len(uv):
                continue
            seen.add(key)
            uv_len = float(np.linalg.norm(uv[b] - uv[a]))
            xyz_len = float(np.linalg.norm(xyz[b] - xyz[a]))
            if uv_len > 1e-12 and np.isfinite(uv_len) and np.isfinite(xyz_len):
                values.append(xyz_len / uv_len)
    return np.asarray(values, dtype=float)


def _omega_quality_metrics(parameterization) -> dict[str, float | int | str]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    surface_faces = np.asarray(parameterization.surface_faces, dtype=int)
    if len(uv) == 0 or len(faces) == 0:
        return {
            "uv_triangle_flip_count": 0,
            "uv_min_triangle_area": 0.0,
            "uv_area_ratio_min": 0.0,
            "uv_area_ratio_max": 0.0,
            "edge_stretch_median": 0.0,
            "edge_stretch_p95": 0.0,
            "edge_stretch_max": 0.0,
            "csf_median": 1.0,
            "csf_p95": 1.0,
            "csf_max": 1.0,
            "boundary_self_intersection_count": 0,
            "uv_degenerate_triangle_count": 0,
        }

    uv_tri = uv[faces[:, :3]]
    signed = _triangle_signed_area_2d(uv_tri)
    uv_area = np.abs(signed)
    xyz_area = _triangle_area_3d(xyz[surface_faces[:, :3]]) if len(surface_faces) == len(faces) else np.zeros_like(uv_area)
    positive = uv_area > 1e-12
    ratios = xyz_area[positive] / np.maximum(uv_area[positive], 1e-12) if np.any(positive) else np.zeros(0, dtype=float)
    stretch = _edge_stretch_values(xyz, uv, faces)
    csf = _parameterization_stretch_csf(parameterization)
    metrics: dict[str, float | int | str] = {
        "uv_triangle_flip_count": int(np.sum(signed < -1e-12)),
        "uv_min_triangle_area": float(np.min(uv_area)) if len(uv_area) else 0.0,
        "uv_area_ratio_min": float(np.min(ratios)) if len(ratios) else 0.0,
        "uv_area_ratio_max": float(np.max(ratios)) if len(ratios) else 0.0,
        "edge_stretch_median": float(np.median(stretch)) if len(stretch) else 0.0,
        "edge_stretch_p95": float(np.percentile(stretch, 95)) if len(stretch) else 0.0,
        "edge_stretch_max": float(np.max(stretch)) if len(stretch) else 0.0,
        "csf_median": float(np.median(csf)) if len(csf) else 1.0,
        "csf_p95": float(np.percentile(csf, 95)) if len(csf) else 1.0,
        "csf_max": float(np.max(csf)) if len(csf) else 1.0,
        "boundary_self_intersection_count": _boundary_self_intersection_count(np.asarray(parameterization.omega_boundary, dtype=float)),
        "uv_degenerate_triangle_count": int(np.sum(uv_area <= 1e-12)),
    }
    warnings: list[str] = []
    if metrics["uv_triangle_flip_count"]:
        warnings.append("UV triangle flips detected")
    if metrics["uv_degenerate_triangle_count"]:
        warnings.append("near-zero UV triangle area detected")
    if float(metrics["edge_stretch_max"]) > 10.0 or float(metrics["csf_max"]) > 10.0:
        warnings.append("extreme S->Omega stretch detected")
    if metrics["boundary_self_intersection_count"]:
        warnings.append("Omega boundary self-intersections detected")
    metrics["parameterization_warning"] = "; ".join(warnings)
    return metrics


def _shape_preserving_projected_uv(vertices: np.ndarray, boundary_loop: list[int]) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    pts = np.asarray(vertices, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=float), {"omega_boundary_model": "empty"}
    centered = pts - np.mean(pts, axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2].T
        if basis.shape != (3, 2):
            raise ValueError("invalid PCA basis")
        uv = centered @ basis
        model = "PCA projection of S; boundary shape preserved"
    except Exception:
        uv = centered[:, :2].copy()
        model = "XY projection of S; boundary shape preserved"

    span = np.nanmax(uv, axis=0) - np.nanmin(uv, axis=0)
    scale = max(float(np.nanmax(span)) * 0.5, 1e-12)
    uv = uv / scale
    uv = uv - 0.5 * (np.nanmin(uv, axis=0) + np.nanmax(uv, axis=0))

    if boundary_loop:
        boundary = uv[boundary_loop]
        area = 0.5 * float(np.sum(boundary[:, 0] * np.roll(boundary[:, 1], -1) - np.roll(boundary[:, 0], -1) * boundary[:, 1]))
        if area < 0.0:
            uv[:, 1] *= -1.0

    open_boundary = uv[boundary_loop] if boundary_loop else uv
    lo = np.nanmin(open_boundary, axis=0) if len(open_boundary) else np.zeros(2)
    hi = np.nanmax(open_boundary, axis=0) if len(open_boundary) else np.zeros(2)
    boundary_span = np.maximum(hi - lo, 1e-12)
    on_box = np.logical_or.reduce(
        [
            np.isclose(open_boundary[:, 0], lo[0], atol=1e-6),
            np.isclose(open_boundary[:, 0], hi[0], atol=1e-6),
            np.isclose(open_boundary[:, 1], lo[1], atol=1e-6),
            np.isclose(open_boundary[:, 1], hi[1], atol=1e-6),
        ]
    ) if len(open_boundary) else np.zeros(0, dtype=bool)
    metrics = {
        "omega_boundary_model": model,
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape_preserved": True,
        "omega_boundary_box_edge_fraction": float(np.mean(on_box)) if len(on_box) else 0.0,
        "omega_boundary_span_u": float(boundary_span[0]),
        "omega_boundary_span_v": float(boundary_span[1]),
    }
    return uv, metrics


def _triangle_local_coordinates(tri: np.ndarray) -> np.ndarray:
    p0, p1, p2 = np.asarray(tri, dtype=float)
    e1 = p1 - p0
    len_e1 = float(np.linalg.norm(e1))
    if len_e1 <= 1e-12:
        return np.zeros((3, 2), dtype=float)
    x_axis = e1 / len_e1
    n = np.cross(e1, p2 - p0)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return np.zeros((3, 2), dtype=float)
    y_axis = np.cross(n / n_norm, x_axis)
    rel = np.asarray([p0 - p0, p1 - p0, p2 - p0], dtype=float)
    return np.column_stack([rel @ x_axis, rel @ y_axis])


def _farthest_boundary_pin_pair(vertices: np.ndarray, boundary_loop: list[int]) -> tuple[int, int, float]:
    pts = np.asarray(vertices, dtype=float)
    loop = list(boundary_loop)
    if len(loop) < 2:
        raise RuntimeError("LSCM requires at least two boundary vertices for pinning.")
    boundary = pts[loop]
    diffs = boundary[:, None, :] - boundary[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    distance = float(np.sqrt(max(dist2[i, j], 0.0)))
    if distance <= 1e-12:
        raise RuntimeError("LSCM boundary pin pair is degenerate.")
    return int(loop[i]), int(loop[j]), distance


def _cotangent_value(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(a, dtype=float)
    cross_norm = float(np.linalg.norm(np.cross(u, v)))
    if cross_norm <= 1e-14:
        return 0.0
    value = float(np.dot(u, v) / cross_norm)
    if not np.isfinite(value):
        return 0.0
    return value


def _vertex_normals_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    pts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    normals = np.zeros_like(pts, dtype=float)
    for face in tris:
        tri = pts[face]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        for vertex_id in face:
            normals[int(vertex_id)] += n
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-14
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.asarray([0.0, 0.0, 1.0])
    return normals


def _cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray):
    try:
        from scipy import sparse
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("scipy sparse is required for bff parameterization") from exc

    pts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    n_vertices = int(len(pts))
    weights: dict[tuple[int, int], float] = {}
    negative_weight_count = 0
    for face in tris:
        i, j, k = [int(v) for v in face]
        cot_i = _cotangent_value(pts[i], pts[j], pts[k])
        cot_j = _cotangent_value(pts[j], pts[k], pts[i])
        cot_k = _cotangent_value(pts[k], pts[i], pts[j])
        for a, b, cot in ((j, k, cot_i), (k, i, cot_j), (i, j, cot_k)):
            if cot < -1e-12:
                negative_weight_count += 1
            key = (min(a, b), max(a, b))
            weights[key] = weights.get(key, 0.0) + 0.5 * cot

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(n_vertices, dtype=float)
    for (a, b), weight in weights.items():
        if abs(weight) <= 1e-14 or not np.isfinite(weight):
            continue
        diag[a] += weight
        diag[b] += weight
        rows.extend([a, b])
        cols.extend([b, a])
        data.extend([-weight, -weight])
    for vertex_id, value in enumerate(diag):
        rows.append(vertex_id)
        cols.append(vertex_id)
        data.append(float(value) + 1e-10)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr(), negative_weight_count


def _bff_boundary_polygon(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    pts = np.asarray(vertices, dtype=float)
    loop = list(boundary_loop)
    n = len(loop)
    if n < 3:
        raise RuntimeError("BFF parameterization requires a boundary loop with at least three vertices.")

    normals = _vertex_normals_from_faces(pts, faces)
    edge_lengths = np.asarray(
        [np.linalg.norm(pts[loop[(i + 1) % n]] - pts[loop[i]]) for i in range(n)],
        dtype=float,
    )
    perimeter = float(np.sum(edge_lengths))
    if perimeter <= 1e-12:
        raise RuntimeError("BFF boundary perimeter is degenerate.")

    turns = np.zeros(n, dtype=float)
    for i in range(n):
        prev_id = loop[(i - 1) % n]
        curr_id = loop[i]
        next_id = loop[(i + 1) % n]
        incoming = pts[curr_id] - pts[prev_id]
        outgoing = pts[next_id] - pts[curr_id]
        in_len = float(np.linalg.norm(incoming))
        out_len = float(np.linalg.norm(outgoing))
        if in_len <= 1e-12 or out_len <= 1e-12:
            continue
        incoming /= in_len
        outgoing /= out_len
        normal = normals[curr_id]
        cross = np.cross(incoming, outgoing)
        turns[i] = float(np.arctan2(np.dot(normal, cross), np.dot(incoming, outgoing)))

    turn_sum = float(np.sum(turns))
    if abs(turn_sum) <= 1e-8:
        # Fallback for inconsistent triangle orientation: keep the unsigned
        # boundary angles, then normalize the total turn to one closed loop.
        for i in range(n):
            prev_id = loop[(i - 1) % n]
            curr_id = loop[i]
            next_id = loop[(i + 1) % n]
            a = pts[prev_id] - pts[curr_id]
            b = pts[next_id] - pts[curr_id]
            denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
            angle = float(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0)))
            turns[i] = np.pi - angle
        turn_sum = float(np.sum(turns))
    if turn_sum < 0.0:
        turns *= -1.0
        turn_sum = -turn_sum

    positions = np.zeros((n + 1, 2), dtype=float)
    theta = 0.0
    for i in range(n):
        positions[i + 1] = positions[i] + edge_lengths[i] * np.asarray([np.cos(theta), np.sin(theta)])
        theta += turns[(i + 1) % n]

    closure_drift = positions[-1] - positions[0]
    boundary_uv = positions[:-1].copy()
    boundary_uv -= np.mean(boundary_uv, axis=0)
    area = 0.5 * float(
        np.sum(boundary_uv[:, 0] * np.roll(boundary_uv[:, 1], -1) - np.roll(boundary_uv[:, 0], -1) * boundary_uv[:, 1])
    )
    if area < 0.0:
        boundary_uv[:, 1] *= -1.0
        area = -area

    flat_lengths = np.asarray(
        [np.linalg.norm(boundary_uv[(i + 1) % n] - boundary_uv[i]) for i in range(n)],
        dtype=float,
    )
    scale = float(np.dot(flat_lengths, edge_lengths) / max(np.dot(flat_lengths, flat_lengths), 1e-12))
    rel_error = np.abs(scale * flat_lengths - edge_lengths) / np.maximum(edge_lengths, 1e-12)
    metrics = {
        "bff_boundary_vertex_count": int(n),
        "bff_boundary_perimeter_3d": float(perimeter),
        "bff_boundary_turning_angle_sum_raw": float(turn_sum),
        "bff_boundary_turning_angle_sum_used": float(np.sum(turns)),
        "bff_boundary_closure_correction_applied": False,
        "bff_boundary_closure_drift": float(np.linalg.norm(closure_drift)),
        "bff_boundary_max_relative_length_error_after_similarity": float(np.max(rel_error)) if len(rel_error) else 0.0,
        "bff_boundary_mean_relative_length_error_after_similarity": float(np.mean(rel_error)) if len(rel_error) else 0.0,
        "bff_boundary_area_2d": float(area),
    }
    return boundary_uv, metrics


def _point_on_centered_rectangle(distance: float, width: float, height: float) -> np.ndarray:
    perimeter = max(2.0 * (width + height), 1e-12)
    s = float(distance % perimeter)
    half_w = 0.5 * width
    half_h = 0.5 * height
    if s <= width:
        return np.asarray([-half_w + s, -half_h], dtype=float)
    s -= width
    if s <= height:
        return np.asarray([half_w, -half_h + s], dtype=float)
    s -= height
    if s <= width:
        return np.asarray([half_w - s, half_h], dtype=float)
    s -= width
    return np.asarray([-half_w, half_h - s], dtype=float)


def _rectangularize_boundary_by_arclength(
    boundary_uv: np.ndarray,
    edge_lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    source = np.asarray(boundary_uv, dtype=float)
    lengths = np.asarray(edge_lengths, dtype=float)
    n = len(source)
    if n < 3 or len(lengths) != n:
        return source.copy(), {
            "bff_boundary_rectangular_correction_applied": False,
            "bff_boundary_rectangularization_reason": "invalid_boundary",
        }

    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-12:
        return source.copy(), {
            "bff_boundary_rectangular_correction_applied": False,
            "bff_boundary_rectangularization_reason": "degenerate_perimeter",
        }

    span = np.nanmax(source, axis=0) - np.nanmin(source, axis=0)
    aspect = float(span[0] / max(span[1], 1e-12))
    if not np.isfinite(aspect) or aspect <= 1e-6:
        aspect = 1.0
    aspect = float(np.clip(aspect, 0.2, 5.0))
    height = perimeter / (2.0 * (aspect + 1.0))
    width = aspect * height

    cumulative = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
    rect = np.asarray([_point_on_centered_rectangle(s, width, height) for s in cumulative], dtype=float)
    rect_lengths = np.asarray([np.linalg.norm(rect[(i + 1) % n] - rect[i]) for i in range(n)], dtype=float)
    rel_error = np.abs(rect_lengths - lengths) / np.maximum(lengths, 1e-12)
    on_edge = np.logical_or.reduce(
        [
            np.isclose(rect[:, 0], -0.5 * width, atol=1e-8),
            np.isclose(rect[:, 0], 0.5 * width, atol=1e-8),
            np.isclose(rect[:, 1], -0.5 * height, atol=1e-8),
            np.isclose(rect[:, 1], 0.5 * height, atol=1e-8),
        ]
    )
    return rect, {
        "bff_boundary_rectangular_correction_applied": True,
        "bff_boundary_closure_correction_applied": True,
        "bff_boundary_closure_drift_after_rectangularization": 0.0,
        "bff_boundary_rectangularization_reason": "forced_rectangle_by_3d_boundary_arclength",
        "bff_boundary_rect_width": float(width),
        "bff_boundary_rect_height": float(height),
        "bff_boundary_rect_aspect": float(aspect),
        "bff_boundary_rect_vertex_on_edge_fraction": float(np.mean(on_edge)) if len(on_edge) else 0.0,
        "bff_boundary_rect_max_relative_length_error": float(np.max(rel_error)) if len(rel_error) else 0.0,
        "bff_boundary_rect_mean_relative_length_error": float(np.mean(rel_error)) if len(rel_error) else 0.0,
    }


def _bff_boundary_first_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    try:
        from scipy.sparse import linalg as sparse_linalg
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("scipy sparse is required for bff parameterization") from exc

    pts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    n_vertices = int(len(pts))
    boundary_uv, boundary_metrics = _bff_boundary_polygon(pts, tris, boundary_loop)
    boundary_edge_lengths = np.asarray(
        [np.linalg.norm(pts[boundary_loop[(i + 1) % len(boundary_loop)]] - pts[boundary_loop[i]]) for i in range(len(boundary_loop))],
        dtype=float,
    )
    boundary_uv, rect_metrics = _rectangularize_boundary_by_arclength(boundary_uv, boundary_edge_lengths)
    uv = np.zeros((n_vertices, 2), dtype=float)
    boundary_ids = np.asarray(boundary_loop, dtype=int)
    uv[boundary_ids] = boundary_uv
    boundary_set = set(int(v) for v in boundary_ids)
    interior_ids = np.asarray([i for i in range(n_vertices) if i not in boundary_set], dtype=int)

    laplacian, negative_weight_count = _cotangent_laplacian(pts, tris)
    if len(interior_ids):
        l_ii = laplacian[interior_ids[:, None], interior_ids]
        l_ib = laplacian[interior_ids[:, None], boundary_ids]
        rhs = -l_ib @ boundary_uv
        uv[interior_ids, 0] = sparse_linalg.spsolve(l_ii, rhs[:, 0])
        uv[interior_ids, 1] = sparse_linalg.spsolve(l_ii, rhs[:, 1])

    uv = uv - np.mean(uv, axis=0)
    span = np.nanmax(uv, axis=0) - np.nanmin(uv, axis=0)
    scale = max(float(np.nanmax(span)) * 0.5, 1e-12)
    uv = uv / scale

    return uv, {
        "omega_parameterization_solver": "boundary_first_flattening_cotan_harmonic",
        "omega_boundary_constraint_model": "bff_boundary_rectangularized_by_3d_boundary_arclength",
        "omega_boundary_forced_rectangle": True,
        "omega_boundary_shape_preserved": False,
        "omega_boundary_model": "BFF-style boundary-first flattening with rectangular Omega boundary correction by 3D boundary arclength",
        "bff_implemented": True,
        "bff_variant": "discrete_boundary_lengths_turning_angles_rectangular_boundary_correction_plus_cotan_harmonic_extension",
        "bff_reference_library": "local implementation; libigl/geometry-central BFF binding not available",
        "bff_cotangent_negative_weight_count": int(negative_weight_count),
        "bff_interior_vertex_count": int(len(interior_ids)),
        **boundary_metrics,
        **rect_metrics,
    }


def _lscm_free_boundary_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Least-squares conformal map with free boundary and two pinned vertices.

    This solves a Cauchy-Riemann residual per triangle.  The boundary is not
    forced to a square, circle, or projected outline; two boundary vertices are
    pinned only to remove translation/rotation/scale nullspace.
    """
    try:
        from scipy import sparse
        from scipy.sparse import linalg as sparse_linalg
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("scipy sparse is required for lscm_paper_like parameterization") from exc

    pts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    n_vertices = int(len(pts))
    if n_vertices == 0 or len(tris) == 0:
        return np.zeros((0, 2), dtype=float), {"omega_parameterization_solver": "lscm_empty"}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row = 0
    used_triangles = 0
    skipped_triangles = 0
    for face in tris:
        local = _triangle_local_coordinates(pts[face])
        signed_area = float(_triangle_signed_area_2d(local.reshape(1, 3, 2))[0])
        area = abs(signed_area)
        if area <= 1e-14 or not np.isfinite(area):
            skipped_triangles += 1
            continue
        x = local[:, 0]
        y = local[:, 1]
        denom = 2.0 * signed_area
        if abs(denom) <= 1e-14:
            skipped_triangles += 1
            continue
        grad = np.asarray(
            [
                [(y[1] - y[2]) / denom, (x[2] - x[1]) / denom],
                [(y[2] - y[0]) / denom, (x[0] - x[2]) / denom],
                [(y[0] - y[1]) / denom, (x[1] - x[0]) / denom],
            ],
            dtype=float,
        )
        weight = float(np.sqrt(area))
        for local_id, vertex_id in enumerate(face):
            gx, gy = float(grad[local_id, 0]), float(grad[local_id, 1])
            # du/dx - dv/dy = 0
            rows.extend([row, row])
            cols.extend([int(vertex_id), n_vertices + int(vertex_id)])
            data.extend([weight * gx, -weight * gy])
            # du/dy + dv/dx = 0
            rows.extend([row + 1, row + 1])
            cols.extend([int(vertex_id), n_vertices + int(vertex_id)])
            data.extend([weight * gy, weight * gx])
        rhs.extend([0.0, 0.0])
        row += 2
        used_triangles += 1

    pin_a, pin_b, pin_distance = _farthest_boundary_pin_pair(pts, boundary_loop)
    pin_weight = max(1000.0, 100.0 * np.sqrt(max(used_triangles, 1)))
    pins = [
        (pin_a, 0.0, 0.0),
        (pin_b, pin_distance, 0.0),
    ]
    for vertex_id, u_value, v_value in pins:
        rows.append(row)
        cols.append(int(vertex_id))
        data.append(pin_weight)
        rhs.append(pin_weight * float(u_value))
        row += 1
        rows.append(row)
        cols.append(n_vertices + int(vertex_id))
        data.append(pin_weight)
        rhs.append(pin_weight * float(v_value))
        row += 1

    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(row, 2 * n_vertices)).tocsr()
    solution = sparse_linalg.lsqr(matrix, np.asarray(rhs, dtype=float), atol=1e-10, btol=1e-10, iter_lim=max(2000, 20 * n_vertices))
    x = np.asarray(solution[0], dtype=float)
    uv = np.column_stack([x[:n_vertices], x[n_vertices:]])

    # Similarity normalization for display and numeric conditioning.  This does
    # not constrain the boundary shape.
    uv = uv - np.mean(uv, axis=0)
    span = np.nanmax(uv, axis=0) - np.nanmin(uv, axis=0)
    scale = max(float(np.nanmax(span)) * 0.5, 1e-12)
    uv = uv / scale
    if boundary_loop:
        boundary = uv[boundary_loop]
        area = 0.5 * float(np.sum(boundary[:, 0] * np.roll(boundary[:, 1], -1) - np.roll(boundary[:, 0], -1) * boundary[:, 1]))
        if area < 0.0:
            uv[:, 1] *= -1.0

    return uv, {
        "omega_parameterization_solver": "free_boundary_lscm_two_pin",
        "omega_boundary_constraint_model": "free_boundary_two_pinned_vertices_only",
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape_preserved": False,
        "omega_boundary_model": "LSCM free boundary; no square/circle/projected-outline boundary forcing",
        "lscm_used_triangle_count": int(used_triangles),
        "lscm_skipped_degenerate_triangle_count": int(skipped_triangles),
        "lscm_pin_vertex_a": int(pin_a),
        "lscm_pin_vertex_b": int(pin_b),
        "lscm_pin_distance_3d": float(pin_distance),
        "lscm_lsqr_iterations": int(solution[2]),
        "lscm_lsqr_residual_norm": float(solution[3]),
    }


def _mark_parameterization_mode(parameterization, *, method: str, exactness: str, warning: str, extra: dict | None = None):
    parameterization.metrics.update(
        {
            "parameterization_method": method,
            "parameterization_exactness_label": exactness,
            "parameterization_warning": warning,
            "paper_compliance_status": exactness,
        }
    )
    if extra:
        parameterization.metrics.update(extra)
    parameterization.metrics.update(_omega_quality_metrics(parameterization))
    if warning and not str(parameterization.metrics.get("parameterization_warning", "")):
        parameterization.metrics["parameterization_warning"] = warning
    elif warning:
        existing = str(parameterization.metrics.get("parameterization_warning", ""))
        if warning not in existing:
            parameterization.metrics["parameterization_warning"] = (existing + "; " + warning).strip("; ")
    return parameterization


def _build_surface_parameterization(surface, target, grid, params):
    if params.m3d_construction_mode == "analytic_scaled_heightfield_debug":
        out = _ORIGINAL_BUILD_SURFACE_PARAMETERIZATION(surface, target, grid, params)
        return _mark_parameterization_mode(
            out,
            method="analytic_scaled_heightfield_debug",
            exactness="debug",
            warning="Height-field shortcut is a debug path and is not a paper conformal parameterization.",
            extra={"height_field_shortcut_used": True},
        )

    boundary_mode = str(getattr(params, "omega_boundary_mode", "paper_default"))
    parameterization_mode = str(getattr(params, "omega_parameterization_mode", "bff"))
    allow_experimental = bool(getattr(params, "allow_experimental_pipeline", False))

    surface_vertices = np.asarray(surface.vertices, dtype=float)
    surface_faces = np.asarray(surface.faces[:, :3], dtype=int)
    boundary_loop = _original._mesh_boundary_loop(surface_faces)
    if len(boundary_loop) < 3:
        raise RuntimeError("S->Omega parameterization requires an open target mesh with a boundary; only explicit debug heightfield mode is available for this surface.")

    if parameterization_mode in {"arap_paper_like", "paper_like_unimplemented"}:
        raise NotImplementedError(
            f"{parameterization_mode} is not implemented. Use bff for the current boundary-first conformal S->Omega path, "
            "or explicitly enable pca_debug as an experimental non-paper path."
        )

    if parameterization_mode == "bff":
        if boundary_mode != "paper_default":
            raise ValueError("bff uses its own boundary-first solve and only supports omega_boundary_mode='paper_default'.")
        uv_vertices, bff_metrics = _bff_boundary_first_uv(surface_vertices, surface_faces, boundary_loop)
        boundary = uv_vertices[boundary_loop + [boundary_loop[0]]]
        metric = {"mean_slope": 0.0, "max_slope": 0.0} if target.kind == "sampled" else _original._heightfield_metric_summary(target, grid)
        metrics: dict[str, float | int | str | bool] = {
            "parameterization_method": "bff",
            "parameterization_exactness_label": "bff_rectangular_boundary_corrected",
            "parameterization_warning": "BFF path is implemented locally as boundary-first flattening with 3D boundary lengths/turning angles, rectangular Omega boundary correction, and cotangent harmonic interior extension; it is not a libigl reference binding.",
            "omega_boundary_mode": "paper_default",
            "omega_parameterization_mode": "bff",
            "surface_vertex_count": int(len(surface_vertices)),
            "surface_triangle_count": int(len(surface_faces)),
            "boundary_vertex_count": int(len(boundary_loop)),
            "mean_slope": metric["mean_slope"],
            "max_slope": metric["max_slope"],
            "harmonic_solve_performed": True,
            "height_field_shortcut_used": False,
            "omega_corresponds_to_S": True,
            "omega_correspondence_model": "BFF boundary-first map c:S->Omega, inverse by UV triangle lookup",
            "lscm_implemented": True,
            "paper_flow_stage": "S -> Omega by BFF-style boundary-first flattening with rectangular Omega boundary correction",
            "paper_exactness_warning": "Local discrete BFF path is active; compare bff_* metrics against the paper/reference implementation before claiming numerical equivalence.",
            "omega_warning": "Boundary-first BFF path with rectangular Omega boundary correction applied.",
            **bff_metrics,
        }
        out = _original.SurfaceParameterization(
            method="bff",
            surface_vertices_3d=surface_vertices,
            surface_faces=surface_faces,
            uv_vertices_2d=uv_vertices,
            uv_faces=surface_faces.copy(),
            omega_boundary=boundary,
            triangle_acceleration=None,
            metrics=metrics,
        )
        return _mark_parameterization_mode(
            out,
            method="bff",
            exactness="bff_rectangular_boundary_corrected",
            warning="Local BFF-style boundary-first flattening is active with rectangular Omega boundary correction; reference BFF numerical equivalence is not guaranteed.",
        )

    if parameterization_mode == "lscm_paper_like":
        if boundary_mode not in {"paper_default", "shape_preserving_experimental"}:
            raise ValueError("lscm_paper_like uses a free boundary and does not support rectangular_debug boundary forcing.")
        uv_vertices, lscm_metrics = _lscm_free_boundary_uv(surface_vertices, surface_faces, boundary_loop)
        boundary = uv_vertices[boundary_loop + [boundary_loop[0]]]
        metric = {"mean_slope": 0.0, "max_slope": 0.0} if target.kind == "sampled" else _original._heightfield_metric_summary(target, grid)
        metrics: dict[str, float | int | str | bool] = {
            "parameterization_method": "lscm_paper_like",
            "parameterization_exactness_label": "conformal_approximation",
            "parameterization_warning": "Free-boundary LSCM is implemented for conformal S->Omega mapping, but it is not Boundary First Flattening unless the paper path specifically accepts LSCM.",
            "omega_boundary_mode": "paper_default" if boundary_mode == "paper_default" else boundary_mode,
            "omega_parameterization_mode": "lscm_paper_like",
            "surface_vertex_count": int(len(surface_vertices)),
            "surface_triangle_count": int(len(surface_faces)),
            "boundary_vertex_count": int(len(boundary_loop)),
            "mean_slope": metric["mean_slope"],
            "max_slope": metric["max_slope"],
            "harmonic_solve_performed": False,
            "height_field_shortcut_used": False,
            "omega_corresponds_to_S": True,
            "omega_correspondence_model": "free-boundary LSCM map c:S->Omega, inverse by UV triangle lookup",
            "bff_implemented": False,
            "lscm_implemented": True,
            "paper_flow_stage": "S -> Omega by free-boundary LSCM; boundary is not forced to a rectangle or projected outline",
            "paper_exactness_warning": "BFF is not implemented; current conformal path is LSCM with two pinned vertices.",
            "omega_warning": "Conformal LSCM path; no full-boundary fitting correction is applied.",
            **lscm_metrics,
        }
        out = _original.SurfaceParameterization(
            method="lscm_paper_like",
            surface_vertices_3d=surface_vertices,
            surface_faces=surface_faces,
            uv_vertices_2d=uv_vertices,
            uv_faces=surface_faces.copy(),
            omega_boundary=boundary,
            triangle_acceleration=None,
            metrics=metrics,
        )
        return _mark_parameterization_mode(
            out,
            method="lscm_paper_like",
            exactness="conformal_approximation",
            warning="Free-boundary LSCM is implemented; BFF remains unimplemented.",
        )

    if parameterization_mode != "pca_debug":
        raise ValueError(f"unknown omega_parameterization_mode: {parameterization_mode}")

    if not allow_experimental:
        raise RuntimeError(
            "Non-paper S->Omega path requested without allow_experimental_pipeline=True. "
            "This prevents PCA/debug parameterization from being mistaken for the paper implementation."
        )

    if boundary_mode == "rectangular_debug":
        out = _ORIGINAL_BUILD_SURFACE_PARAMETERIZATION(surface, target, grid, params)
        out.metrics.update(
            {
                "omega_boundary_mode": "rectangular_debug",
                "omega_boundary_forced_rectangle": True,
                "omega_boundary_shape_preserved": False,
                "omega_boundary_model": "rectangular debug boundary from original implementation",
                "bff_implemented": False,
                "paper_flow_stage": "S -> Omega rectangular debug substitute; not paper BFF/LSCM/ARAP",
                "omega_warning": "Debug rectangular Omega boundary; this is not paper-default parameterization.",
            }
        )
        return _mark_parameterization_mode(
            out,
            method="rectangular_debug",
            exactness="debug",
            warning="Rectangular Omega boundary is a debug substitute, not paper-default parameterization.",
        )

    if boundary_mode == "paper_default":
        raise RuntimeError("pca_debug cannot be used with omega_boundary_mode='paper_default'. Select an explicit debug/experimental boundary mode.")

    if boundary_mode != "shape_preserving_experimental":
        raise ValueError(f"unknown omega_boundary_mode: {boundary_mode}")

    uv_vertices, projection_metrics = _shape_preserving_projected_uv(surface_vertices, boundary_loop)
    boundary = uv_vertices[boundary_loop + [boundary_loop[0]]]
    metric = {"mean_slope": 0.0, "max_slope": 0.0} if target.kind == "sampled" else _original._heightfield_metric_summary(target, grid)
    method = "pca_debug"
    metrics: dict[str, float | int | str | bool] = {
        "parameterization_method": method,
        "parameterization_exactness_label": "experimental",
        "parameterization_warning": "PCA projection is experimental and is not a conformal paper parameterization.",
        "omega_boundary_mode": "shape_preserving_experimental",
        "omega_parameterization_mode": "pca_debug",
        "surface_vertex_count": int(len(surface_vertices)),
        "surface_triangle_count": int(len(surface_faces)),
        "boundary_vertex_count": int(len(boundary_loop)),
        "mean_slope": metric["mean_slope"],
        "max_slope": metric["max_slope"],
        "harmonic_solve_performed": False,
        "height_field_shortcut_used": False,
        "omega_corresponds_to_S": True,
        "omega_correspondence_model": "shape-preserving projected UV map c:S->Omega, inverse by UV triangle lookup",
        "bff_implemented": False,
        "paper_flow_stage": "S -> Omega by PCA debug projection with boundary-shape preservation; inverse c^-1 used for M2D -> M3D",
        "paper_exactness_warning": "Boundary First Flattening/LSCM/ARAP are not implemented; PCA projection is experimental.",
        "omega_warning": "Experimental: PCA projected shape-preserving UV map, not Boundary First Flattening/LSCM/ARAP.",
        **projection_metrics,
    }
    out = _original.SurfaceParameterization(
        method=method,
        surface_vertices_3d=surface_vertices,
        surface_faces=surface_faces,
        uv_vertices_2d=uv_vertices,
        uv_faces=surface_faces.copy(),
        omega_boundary=boundary,
        triangle_acceleration=None,
        metrics=metrics,
    )
    return _mark_parameterization_mode(
        out,
        method="pca_debug",
        exactness="experimental",
        warning="PCA projection is experimental and is not a conformal paper parameterization.",
    )


def _dominant_surface_peak_uv(parameterization) -> np.ndarray | None:
    peaks = _surface_peak_uvs(parameterization)
    if len(peaks) == 0:
        return None
    return np.mean(peaks, axis=0)


def _align_domain_grid_to_uv_points(domain, target_uvs: np.ndarray | None) -> dict[str, float | bool]:
    if target_uvs is None:
        return {"m2d_grid_aligned_to_peak_vertex": False}
    targets = np.asarray(target_uvs, dtype=float).reshape(-1, 2)
    if len(targets) == 0:
        return {"m2d_grid_aligned_to_peak_vertex": False}
    uv = np.asarray(domain.uv_vertices, dtype=float).copy()
    if uv.size == 0:
        return {"m2d_grid_aligned_to_peak_vertex": False}
    shifts: dict[str, float | bool] = {"m2d_grid_aligned_to_peak_vertex": True}
    for coord, label in ((0, "u"), (1, "v")):
        unique = np.unique(np.round(uv[:, coord], 12))
        if len(unique) == 0:
            continue
        peak_values = targets[:, coord]
        candidates = []
        for target in peak_values:
            nearest = float(unique[int(np.argmin(np.abs(unique - float(target))))])
            candidates.append(float(target) - nearest)
        step = float(np.median(np.diff(unique))) if len(unique) > 1 else 0.0
        best_shift = 0.0
        best_score = float("inf")
        for shift_candidate in candidates:
            shifted = unique + shift_candidate
            errors = [float(np.min(np.abs(shifted - float(target)))) for target in peak_values]
            score = float(np.mean(errors)) + 1e-6 * abs(float(shift_candidate))
            if step > 1e-12 and abs(float(shift_candidate)) > step:
                score += abs(float(shift_candidate))
            if score < best_score:
                best_score = score
                best_shift = float(shift_candidate)
        shift = best_shift
        uv[:, coord] += shift
        shifts[f"m2d_grid_peak_alignment_shift_{label}"] = float(shift)
        shifts[f"m2d_grid_peak_target_{label}"] = float(np.median(peak_values))
        shifted_unique = unique + shift
        shifts[f"m2d_grid_peak_alignment_max_error_{label}"] = float(
            max(float(np.min(np.abs(shifted_unique - float(target)))) for target in peak_values)
        )
    domain.uv_vertices = uv
    return shifts


def _align_domain_grid_to_uv_point(domain, target_uv: np.ndarray | None) -> dict[str, float | bool]:
    if target_uv is None:
        return {"m2d_grid_aligned_to_peak_vertex": False}
    return _align_domain_grid_to_uv_points(domain, np.asarray(target_uv, dtype=float).reshape(1, 2))


def _flatten_to_domain(parameterization, grid, params=None):
    domain = _ORIGINAL_FLATTEN_TO_DOMAIN(parameterization, grid, params)
    threshold = float(getattr(params, "csf_split_threshold", 2.0)) if params is not None else 2.0
    enabled = bool(getattr(params, "enable_csf_splits", True)) if params is not None else True
    enabled = enabled and bool(getattr(params, "enable_heuristic_csf_split", True)) if params is not None else enabled
    max_splits = int(getattr(params, "max_csf_splits", 1)) if params is not None else 1
    symmetry_enabled = bool(getattr(params, "preserve_detected_symmetry", True)) if params is not None else True
    symmetry_enabled = symmetry_enabled and bool(getattr(params, "enable_mirror_split", True)) if params is not None else symmetry_enabled
    symmetry = _detect_parameterization_reflection_symmetry(parameterization) if symmetry_enabled else {"axes": [], "centers": {}, "details": {}, "tolerance": 0.0}
    csf = _parameterization_stretch_csf(parameterization)
    peak_enabled = bool(getattr(params, "enable_peak_guided_split", True)) if params is not None else True
    if enabled:
        if peak_enabled:
            split_lines = _csf_split_lines(parameterization, csf, threshold=threshold, max_splits=max_splits)
        else:
            uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
            high = uv[np.asarray(csf, dtype=float) > float(threshold)]
            split_lines = _csf_split_lines_from_high_stretch(uv, high, max_splits) if len(high) else []
    else:
        split_lines = []
    if symmetry_enabled:
        split_lines = _mirror_csf_split_lines(
            split_lines,
            list(symmetry.get("axes", [])),
            dict(symmetry.get("centers", {})),
            np.asarray(parameterization.uv_vertices_2d, dtype=float),
        )
    peak_uvs = _surface_peak_uvs(parameterization)
    peak_uv = np.mean(peak_uvs, axis=0) if len(peak_uvs) else None
    peak_alignment = _align_domain_grid_to_uv_points(domain, peak_uvs if len(peak_uvs) else None)
    domain.csf_values = csf
    domain.split_lines = split_lines
    domain.parameterization = parameterization  # type: ignore[attr-defined]
    domain.csf_before = float(np.max(csf)) if csf.size else 1.0  # type: ignore[attr-defined]
    domain.csf_after_split = min(float(domain.csf_before), float(threshold)) if split_lines else float(domain.csf_before)  # type: ignore[attr-defined]
    domain.csf_split_threshold = float(threshold)  # type: ignore[attr-defined]
    domain.csf_split_enabled = bool(enabled)  # type: ignore[attr-defined]
    domain.csf_model = "edge_stretch_proxy"  # type: ignore[attr-defined]
    domain.csf_split_exactness_label = "heuristic"  # type: ignore[attr-defined]
    domain.peak_guided_split_enabled = bool(peak_enabled)  # type: ignore[attr-defined]
    domain.mirror_split_enabled = bool(symmetry_enabled)  # type: ignore[attr-defined]
    domain.detected_symmetry_axes = list(symmetry.get("axes", []))  # type: ignore[attr-defined]
    domain.detected_symmetry_centers = dict(symmetry.get("centers", {}))  # type: ignore[attr-defined]
    domain.detected_symmetry_tolerance = float(symmetry.get("tolerance", 0.0))  # type: ignore[attr-defined]
    domain.detected_symmetry_details = dict(symmetry.get("details", {}))  # type: ignore[attr-defined]
    domain.peak_uv_target = peak_uv  # type: ignore[attr-defined]
    domain.peak_uv_targets = peak_uvs  # type: ignore[attr-defined]
    domain.peak_grid_alignment = dict(peak_alignment)  # type: ignore[attr-defined]
    domain.omega_boundary_forced_rectangle = bool(parameterization.metrics.get("omega_boundary_forced_rectangle", False))  # type: ignore[attr-defined]
    domain.omega_boundary_constraint_model = str(parameterization.metrics.get("omega_boundary_constraint_model", ""))  # type: ignore[attr-defined]
    return domain


def _face_crosses_split(vertices: np.ndarray, face: np.ndarray, split_line: tuple[str, float]) -> bool:
    axis, value = split_line
    coord = 1 if axis == "row" else 0
    vals = vertices[np.asarray(face, dtype=int), coord]
    return bool(float(np.nanmin(vals)) < value < float(np.nanmax(vals)))


def _snap_split_line_to_mesh(vertices: np.ndarray, split_line: tuple[str, float]) -> tuple[str, float] | None:
    axis, value = split_line
    coord = 1 if axis == "row" else 0
    vals = np.asarray(vertices, dtype=float)[:, coord]
    unique = np.unique(np.round(vals[np.isfinite(vals)], 12))
    if len(unique) < 3:
        return None
    internal = unique[1:-1]
    snapped = float(internal[int(np.argmin(np.abs(internal - float(value))))])
    return axis, snapped


def _split_m2d_along_existing_grid_line(
    vertices: np.ndarray,
    faces: np.ndarray,
    split_line: tuple[str, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    snapped = _snap_split_line_to_mesh(vertices, split_line)
    if snapped is None:
        return vertices.copy(), faces.copy(), 0
    axis, value = snapped
    coord = 1 if axis == "row" else 0
    out_vertices = np.asarray(vertices, dtype=float).copy()
    out_faces = np.asarray(faces, dtype=int).copy()
    on_line = np.isclose(out_vertices[:, coord], value, rtol=0.0, atol=1e-9)
    if not np.any(on_line):
        return out_vertices, out_faces, 0

    duplicate_for: dict[int, int] = {}
    centroids = np.mean(out_vertices[out_faces][:, :, coord], axis=1)
    positive_side_faces = np.flatnonzero(centroids > value)
    for face_idx in positive_side_faces:
        for local_idx, vertex_id in enumerate(out_faces[face_idx]):
            vertex_id = int(vertex_id)
            if not on_line[vertex_id]:
                continue
            if vertex_id not in duplicate_for:
                duplicate_for[vertex_id] = len(out_vertices)
                out_vertices = np.vstack([out_vertices, out_vertices[vertex_id]])
            out_faces[face_idx, local_idx] = duplicate_for[vertex_id]
    return out_vertices, out_faces, len(duplicate_for)


def _m2d_connected_component_sizes(faces: np.ndarray) -> list[int]:
    face_array = np.asarray(faces, dtype=int)
    if len(face_array) == 0:
        return []
    by_vertex: dict[int, list[int]] = {}
    for face_idx, face in enumerate(face_array):
        for vertex_id in face:
            by_vertex.setdefault(int(vertex_id), []).append(int(face_idx))
    seen: set[int] = set()
    sizes: list[int] = []
    for start in range(len(face_array)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            face_idx = stack.pop()
            size += 1
            for vertex_id in face_array[face_idx]:
                for neighbor in by_vertex.get(int(vertex_id), []):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def _csf_residual_split_step_analysis(
    parameterization,
    csf: np.ndarray,
    threshold: float,
    split_lines: list[tuple[str, float]],
    vertices_2d: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    values = np.asarray(csf, dtype=float)
    if len(uv) == 0 or len(values) != len(uv):
        return [], {
            "csf_split_step_analysis_model": "unavailable",
            "csf_split_step_count": 0,
            "csf_split_residual_high_vertex_count_after_all": 0,
            "csf_split_residual_max_after_all": 1.0,
        }

    high = np.isfinite(values) & (values > float(threshold))
    vertices = np.asarray(vertices_2d, dtype=float)
    bands: dict[int, float] = {}
    for coord in (0, 1):
        source = vertices[:, coord] if len(vertices) else uv[:, coord]
        unique = np.unique(np.round(source[np.isfinite(source)], 12))
        diffs = np.diff(unique)
        diffs = diffs[diffs > 1e-12]
        if len(diffs):
            bands[coord] = float(np.median(diffs) * 0.55)
        else:
            span = float(np.nanmax(uv[:, coord]) - np.nanmin(uv[:, coord])) if len(uv) else 1.0
            bands[coord] = max(span * 0.01, 1e-9)

    def _stats(mask: np.ndarray) -> dict[str, float | int]:
        vals = values[mask]
        return {
            "count": int(np.count_nonzero(mask)),
            "max": float(np.max(vals)) if len(vals) else 1.0,
            "p95": float(np.percentile(vals, 95)) if len(vals) else 1.0,
        }

    covered = np.zeros(len(uv), dtype=bool)
    rows: list[dict[str, object]] = []
    for step, (axis, raw_value) in enumerate(split_lines, start=1):
        coord = 1 if axis == "row" else 0
        value = float(raw_value)
        band = float(bands[coord])
        before_mask = high & ~covered
        before = _stats(before_mask)
        near_line = np.abs(uv[:, coord] - value) <= band
        newly_covered = before_mask & near_line
        covered |= near_line
        after_mask = high & ~covered
        after = _stats(after_mask)
        recommendation: list[tuple[str, float]] = []
        if int(after["count"]) > 0:
            recommendation = _csf_split_lines_from_high_stretch(uv, uv[after_mask], 1)
        rows.append(
            {
                "step": int(step),
                "applied_split_axis": str(axis),
                "applied_split_value": float(value),
                "split_band_width": float(band),
                "high_vertices_before_step": int(before["count"]),
                "max_csf_before_step": float(before["max"]),
                "p95_csf_before_step": float(before["p95"]),
                "high_vertices_covered_by_step": int(np.count_nonzero(newly_covered)),
                "high_vertices_after_step": int(after["count"]),
                "max_csf_after_step": float(after["max"]),
                "p95_csf_after_step": float(after["p95"]),
                "additional_split_recommended": bool(int(after["count"]) > 0 and float(after["max"]) > float(threshold)),
                "next_recommended_split": [[str(a), float(v)] for a, v in recommendation],
            }
        )

    final_mask = high & ~covered
    final = _stats(final_mask)
    summary = {
        "csf_split_step_analysis_model": (
            "residual high-CSF diagnostic: does not recompute S->Omega after each cut; "
            "vertices within one grid-line band of applied cuts are treated as relieved"
        ),
        "csf_split_step_count": int(len(rows)),
        "csf_split_residual_high_vertex_count_after_all": int(final["count"]),
        "csf_split_residual_max_after_all": float(final["max"]),
        "csf_split_residual_p95_after_all": float(final["p95"]),
        "csf_split_additional_split_recommended_after_all": bool(int(final["count"]) > 0 and float(final["max"]) > float(threshold)),
        "csf_split_residual_high_vertex_indices_after_all": [int(i) for i in np.flatnonzero(final_mask)[:500]],
    }
    return rows, summary


def _quad_area_2d(vertices: np.ndarray, face: np.ndarray) -> float:
    pts = np.asarray(vertices, dtype=float)[np.asarray(face, dtype=int), :2]
    if len(pts) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])))


def _quad_aspect_ratio(vertices: np.ndarray, face: np.ndarray) -> float:
    pts = np.asarray(vertices, dtype=float)[np.asarray(face, dtype=int), :2]
    if len(pts) < 4:
        return 0.0
    lengths = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    positive = [v for v in lengths if v > 1e-12 and np.isfinite(v)]
    if not positive:
        return 0.0
    return float(max(positive) / max(min(positive), 1e-12))


def _mesh_edge_incidence_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in face]
        for i, a in enumerate(ids):
            b = ids[(i + 1) % len(ids)]
            key = tuple(sorted((a, b)))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _m2d_audit_metrics(vertices: np.ndarray, faces: np.ndarray, *, suffix: str = "") -> dict[str, float | int | bool | str]:
    face_array = np.asarray(faces, dtype=int)
    vertex_array = np.asarray(vertices, dtype=float)
    areas = np.asarray([_quad_area_2d(vertex_array, face) for face in face_array], dtype=float)
    aspects = np.asarray([_quad_aspect_ratio(vertex_array, face) for face in face_array], dtype=float)
    components = _m2d_connected_component_sizes(face_array)
    edge_counts = _mesh_edge_incidence_counts(face_array)
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    prefix = "m2d_"
    tail = f"_{suffix}" if suffix else ""
    out: dict[str, float | int | bool | str] = {
        f"{prefix}connected_component_count{tail}": int(len(components)),
        f"{prefix}largest_component_quad_count{tail}": int(components[0]) if components else 0,
        f"{prefix}smallest_component_quad_count{tail}": int(components[-1]) if components else 0,
        f"{prefix}boundary_edge_count{tail}": int(boundary_edges),
        f"{prefix}nonmanifold_edge_count{tail}": int(nonmanifold_edges),
        f"{prefix}min_quad_area{tail}": float(np.min(areas)) if len(areas) else 0.0,
        f"{prefix}max_aspect_ratio{tail}": float(np.max(aspects)) if len(aspects) else 0.0,
    }
    if not suffix:
        out.update(
            {
                "m2d_selection_model": "cell_center_or_original_policy_debug",
                "m2d_boundary_clipping_used": False,
                "m2d_cell_center_only_selection_used": True,
                "m2d_removed_small_component_count": int(sum(1 for size in components[1:] if size > 0)),
                "m2d_hole_count_estimate": 0,
            }
        )
    return out


def _point_in_polygon_or_on_boundary(point: np.ndarray, polygon: np.ndarray, *, tol: float = 1e-9) -> bool:
    p = np.asarray(point, dtype=float)[:2]
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 3:
        return False
    if np.linalg.norm(pts[0] - pts[-1]) <= tol:
        pts = pts[:-1]
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        ab = b - a
        ap = p - a
        edge_len2 = float(np.dot(ab, ab))
        if edge_len2 > tol * tol:
            t = float(np.clip(np.dot(ap, ab) / edge_len2, 0.0, 1.0))
            closest = a + t * ab
            if float(np.linalg.norm(p - closest)) <= tol:
                return True
        x0, y0 = float(a[0]), float(a[1])
        x1, y1 = float(b[0]), float(b[1])
        if (y0 > y) != (y1 > y):
            x_cross = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x <= x_cross + tol:
                inside = not inside
    return bool(inside)


def _clip_m2d_faces_to_omega_boundary(mesh, domain, params=None):
    boundary = np.asarray(getattr(domain, "boundary", np.zeros((0, 2))), dtype=float)
    if len(boundary) >= 2 and np.linalg.norm(boundary[0] - boundary[-1]) <= 1e-9:
        boundary_open = boundary[:-1]
    else:
        boundary_open = boundary
    if len(boundary_open) < 3 or len(mesh.faces) == 0:
        return np.asarray(mesh.faces, dtype=int).copy(), {
            "m2d_boundary_clipping_used": False,
            "m2d_boundary_clip_reason": "missing_boundary_or_faces",
            "m2d_boundary_clip_removed_quad_count": 0,
        }

    requested_policy = str(getattr(params, "m2d_crop_policy", "center")) if params is not None else "center"
    boundary_forced_rect = bool(getattr(domain, "omega_boundary_forced_rectangle", False))
    effective_policy = "strict_vertices" if boundary_forced_rect else requested_policy
    vertices = np.asarray(mesh.vertices, dtype=float)
    kept: list[np.ndarray] = []
    removed = 0
    center_only_would_keep = 0
    for face in np.asarray(mesh.faces, dtype=int):
        pts = vertices[np.asarray(face, dtype=int), :2]
        center = np.mean(pts, axis=0)
        center_inside = _point_in_polygon_or_on_boundary(center, boundary_open)
        vertices_inside = all(_point_in_polygon_or_on_boundary(pt, boundary_open) for pt in pts)
        if center_inside and not vertices_inside:
            center_only_would_keep += 1
        inside = vertices_inside and center_inside if effective_policy == "strict_vertices" else center_inside
        if inside:
            kept.append(np.asarray(face, dtype=int).copy())
        else:
            removed += 1
    if not kept:
        raise RuntimeError("M2D Omega boundary clipping removed all quads.")

    return np.asarray(kept, dtype=int), {
        "m2d_boundary_clipping_used": True,
        "m2d_boundary_clip_policy_requested": requested_policy,
        "m2d_boundary_clip_policy_effective": effective_policy,
        "m2d_boundary_clip_forced_strict_for_rectangular_omega": bool(boundary_forced_rect and requested_policy != "strict_vertices"),
        "m2d_boundary_clip_removed_quad_count": int(removed),
        "m2d_boundary_clip_input_quad_count": int(len(mesh.faces)),
        "m2d_boundary_clip_kept_quad_count": int(len(kept)),
        "m2d_boundary_clip_center_only_would_keep_outside_quad_count": int(center_only_would_keep),
    }


def _symmetrize_m2d_faces(mesh, domain) -> tuple[np.ndarray, int]:
    axes = [int(axis) for axis in list(getattr(domain, "detected_symmetry_axes", []) or [])]
    centers_by_axis = dict(getattr(domain, "detected_symmetry_centers", {}) or {})
    if not axes or len(mesh.faces) == 0 or getattr(mesh.grid, "tiles", None) is None:
        return np.asarray(mesh.faces, dtype=int).copy(), 0

    all_faces = np.asarray([tile.vertex_ids for tile in mesh.grid.tiles or []], dtype=int)
    if len(all_faces) == 0:
        return np.asarray(mesh.faces, dtype=int).copy(), 0
    vertices = np.asarray(mesh.vertices, dtype=float)
    all_centers = np.mean(vertices[all_faces][:, :, :2], axis=1)
    span = np.maximum(np.nanmax(all_centers, axis=0) - np.nanmin(all_centers, axis=0), 1e-12)
    tol = 0.25 * float(np.min(span / np.maximum([mesh.grid.nx, mesh.grid.ny], 1)))
    tol = max(tol, 1e-9)

    kept: dict[tuple[int, int, int, int], np.ndarray] = {
        tuple(int(v) for v in face): np.asarray(face, dtype=int) for face in np.asarray(mesh.faces, dtype=int)
    }
    face_keys = [tuple(int(v) for v in face) for face in all_faces]
    added = 0
    changed = True
    while changed:
        changed = False
        for face in list(kept.values()):
            center = np.mean(vertices[np.asarray(face, dtype=int), :2], axis=0)
            for axis in axes:
                mirrored = center.copy()
                mirrored[axis] = 2.0 * float(centers_by_axis.get(axis, 0.0)) - mirrored[axis]
                distances = np.linalg.norm(all_centers - mirrored, axis=1)
                idx = int(np.argmin(distances))
                if float(distances[idx]) > tol:
                    continue
                key = face_keys[idx]
                if key in kept:
                    continue
                kept[key] = all_faces[idx].copy()
                added += 1
                changed = True
    ordered_faces = [all_faces[idx].copy() for idx, key in enumerate(face_keys) if key in kept]
    return np.asarray(ordered_faces, dtype=int), int(added)


def _build_m2d(grid, domain, params=None):
    mesh = _ORIGINAL_BUILD_M2D(grid, domain, params)
    faces, symmetry_added_count = _symmetrize_m2d_faces(mesh, domain)
    if symmetry_added_count:
        metrics = dict(mesh.metrics)
        metrics.update(
            {
                "m2d_symmetry_preservation_enabled": True,
                "m2d_symmetry_axes": list(getattr(domain, "detected_symmetry_axes", []) or []),
                "m2d_symmetry_added_mirror_quad_count": int(symmetry_added_count),
                "m2d_kept_quad_count": int(len(faces)),
                "m2d_cropped_quad_count": int(metrics.get("m2d_overlay_total_quad_count", len(faces)) - len(faces)),
            }
        )
        metrics.update(dict(getattr(domain, "peak_grid_alignment", {}) or {}))
        peak_uv = getattr(domain, "peak_uv_target", None)
        if peak_uv is not None:
            metrics["m2d_peak_uv_target"] = [float(x) for x in np.asarray(peak_uv, dtype=float).reshape(-1)[:2]]
        peak_uvs = getattr(domain, "peak_uv_targets", None)
        if peak_uvs is not None:
            metrics["m2d_peak_uv_targets"] = [[float(x) for x in row[:2]] for row in np.asarray(peak_uvs, dtype=float).reshape(-1, 2)]
        for key, value in dict(getattr(domain, "detected_symmetry_details", {}) or {}).items():
            metrics[f"symmetry_{key}"] = value
        mesh = _original.QuadMesh(mesh.vertices.copy(), faces, mesh.grid, mesh.stage, metrics, list(getattr(mesh, "split_lines", [])))
    else:
        mesh.metrics.update(
            {
                "m2d_symmetry_preservation_enabled": bool(getattr(domain, "detected_symmetry_axes", []) or []),
                "m2d_symmetry_axes": list(getattr(domain, "detected_symmetry_axes", []) or []),
                "m2d_symmetry_added_mirror_quad_count": 0,
            }
        )
        mesh.metrics.update(dict(getattr(domain, "peak_grid_alignment", {}) or {}))
        peak_uv = getattr(domain, "peak_uv_target", None)
        if peak_uv is not None:
            mesh.metrics["m2d_peak_uv_target"] = [float(x) for x in np.asarray(peak_uv, dtype=float).reshape(-1)[:2]]
        peak_uvs = getattr(domain, "peak_uv_targets", None)
        if peak_uvs is not None:
            mesh.metrics["m2d_peak_uv_targets"] = [[float(x) for x in row[:2]] for row in np.asarray(peak_uvs, dtype=float).reshape(-1, 2)]
        for key, value in dict(getattr(domain, "detected_symmetry_details", {}) or {}).items():
            mesh.metrics[f"symmetry_{key}"] = value
    clipped_faces, clip_metrics = _clip_m2d_faces_to_omega_boundary(mesh, domain, params)
    if len(clipped_faces) != len(mesh.faces):
        metrics = dict(mesh.metrics)
        metrics.update(clip_metrics)
        overlay_total = int(metrics.get("m2d_overlay_total_quad_count", len(mesh.faces)))
        metrics.update(
            {
                "m2d_kept_quad_count": int(len(clipped_faces)),
                "m2d_cropped_quad_count": int(max(0, overlay_total - len(clipped_faces))),
            }
        )
        mesh = _original.QuadMesh(mesh.vertices.copy(), clipped_faces, mesh.grid, mesh.stage, metrics, list(getattr(mesh, "split_lines", [])))
    else:
        mesh.metrics.update(clip_metrics)
    mesh.metrics.update(_m2d_audit_metrics(mesh.vertices, mesh.faces))
    mesh.metrics.update(clip_metrics)
    overlay_total = int(mesh.metrics.get("m2d_overlay_total_quad_count", len(mesh.faces)))
    mesh.metrics.update(
        {
            "m2d_kept_quad_count": int(len(mesh.faces)),
            "m2d_cropped_quad_count": int(max(0, overlay_total - len(mesh.faces))),
        }
    )
    mesh.metrics.update(
        {
            "csf_model": str(getattr(domain, "csf_model", "edge_stretch_proxy")),
            "csf_split_exactness_label": str(getattr(domain, "csf_split_exactness_label", "heuristic")),
            "peak_guided_split_enabled": bool(getattr(domain, "peak_guided_split_enabled", True)),
            "mirror_split_enabled": bool(getattr(domain, "mirror_split_enabled", True)),
        }
    )

    split_lines = list(getattr(domain, "split_lines", []) or [])
    if not split_lines or len(mesh.faces) == 0:
        mesh.metrics.update(
            {
                "csf_split_step_analysis": [],
                "csf_split_step_analysis_model": "not run: no split lines or no M2D faces",
                "csf_split_step_count": 0,
                "csf_split_residual_high_vertex_count_after_all": int(np.count_nonzero(np.asarray(getattr(domain, "csf_values", np.zeros(0)), dtype=float) > float(getattr(domain, "csf_split_threshold", 2.0)))),
                "csf_split_residual_max_after_all": float(getattr(domain, "csf_before", getattr(domain, "max_csf", 1.0))),
            }
        )
        mesh.metrics.update(_m2d_audit_metrics(mesh.vertices, mesh.faces, suffix="after_split"))
        return mesh

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=int).copy()
    snapped_lines: list[tuple[str, float]] = []
    duplicate_count = 0
    for line in split_lines:
        snapped = _snap_split_line_to_mesh(vertices, line)
        if snapped is None:
            continue
        vertices, faces, added = _split_m2d_along_existing_grid_line(vertices, faces, snapped)
        if added > 0:
            duplicate_count += int(added)
            snapped_lines.append(snapped)

    if duplicate_count == 0:
        mesh.metrics.update(
            {
                "csf_split_applied": False,
                "csf_split_rejected_reason": "no_internal_grid_line_to_split",
                "csf_split_candidate_lines": split_lines,
                "csf_split_step_analysis": [],
                "csf_split_step_analysis_model": "not run: no candidate split snapped to an internal M2D grid line",
                "csf_split_step_count": 0,
                "csf_split_residual_high_vertex_count_after_all": int(np.count_nonzero(np.asarray(getattr(domain, "csf_values", np.zeros(0)), dtype=float) > float(getattr(domain, "csf_split_threshold", 2.0)))),
                "csf_split_residual_max_after_all": float(getattr(domain, "csf_before", getattr(domain, "max_csf", 1.0))),
            }
        )
        mesh.metrics.update(_m2d_audit_metrics(mesh.vertices, mesh.faces, suffix="after_split"))
        return mesh

    metrics = dict(mesh.metrics)
    component_sizes = _m2d_connected_component_sizes(faces)
    step_analysis, step_summary = _csf_residual_split_step_analysis(
        domain.parameterization,
        np.asarray(getattr(domain, "csf_values", np.zeros(0)), dtype=float),
        float(getattr(domain, "csf_split_threshold", 2.0)),
        snapped_lines,
        vertices,
    )
    metrics.update(
        {
            "max_csf_before_split": float(getattr(domain, "csf_before", domain.max_csf)),
            "max_csf_after_split": float(step_summary.get("csf_split_residual_max_after_all", getattr(domain, "csf_after_split", domain.max_csf))),
            "number_of_splits": len(snapped_lines),
            "split_locations": snapped_lines,
            "csf_split_applied": True,
            "csf_split_step_analysis": step_analysis,
            "csf_split_removed_quad_count": 0,
            "csf_split_duplicated_vertex_count": int(duplicate_count),
            "m2d_quad_count_after_csf_split": int(len(faces)),
            "m2d_connected_component_count_after_csf_split": int(len(component_sizes)),
            "m2d_largest_component_quad_count_after_csf_split": int(component_sizes[0]) if component_sizes else 0,
            "m2d_smallest_component_quad_count_after_csf_split": int(component_sizes[-1]) if component_sizes else 0,
            "m2d_connected_component_count_after_split": int(len(component_sizes)),
            "m2d_largest_component_quad_count_after_split": int(component_sizes[0]) if component_sizes else 0,
            "m2d_smallest_component_quad_count_after_split": int(component_sizes[-1]) if component_sizes else 0,
            "csf_split_model": "duplicate M2D vertices along an existing grid line; no quads removed",
            "csf_split_threshold": float(getattr(domain, "csf_split_threshold", 2.0)),
        }
    )
    metrics.update(step_summary)
    metrics.update(_m2d_audit_metrics(vertices, faces, suffix="after_split"))
    return _original.QuadMesh(vertices, faces, mesh.grid, mesh.stage, metrics, snapped_lines)


def _lift_m2d_to_m3d(target, mesh, parameterization, params):
    out, report = _ORIGINAL_LIFT_M2D_TO_M3D(target, mesh, parameterization, params)
    lookup_fail = int(out.metrics.get("m3d_uv_triangle_lookup_fail_count", 0))
    outside = int(out.metrics.get("m3d_outside_omega_count", 0))
    used_shortcut = bool(out.metrics.get("m3d_used_height_field_shortcut", False))
    out.metrics.update(
        {
            "m3d_uv_lookup_failure_count": lookup_fail,
            "m3d_outside_uv_triangle_count": outside,
            "m3d_negative_barycentric_count": int(out.metrics.get("m3d_negative_barycentric_count", 0)),
            "m3d_nearest_fallback_count": lookup_fail,
            "m3d_surface_projection_model": "analytic_heightfield_debug" if used_shortcut else "uv_triangle_lookup_barycentric_kdtree_candidates",
            "m3d_uv_triangle_lookup_acceleration": "regular-grid shortcut or cKDTree nearest triangle candidates",
            "m3d_surface_distance_acceleration": "cKDTree nearest surface triangle candidates",
            "m3d_exactness_label": "debug" if used_shortcut else "approximation",
            "m3d_parameterization_warning": (
                "Height-field shortcut is not paper inverse parameterization."
                if used_shortcut
                else "Barycentric inverse map depends on the current non-paper Omega parameterization."
            ),
        }
    )
    return out, report


def _connected_tile_components_from_faces(faces: np.ndarray, tile_count: int) -> list[list[int]]:
    parent = list(range(tile_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    try:
        specs = _original._vertex_hinge_specs_from_faces(np.asarray(faces, dtype=int))
    except Exception:
        specs = []
    for spec in specs:
        if 0 <= int(spec.tile_a) < tile_count and 0 <= int(spec.tile_b) < tile_count:
            union(int(spec.tile_a), int(spec.tile_b))

    groups: dict[int, list[int]] = {}
    for tile_id in range(tile_count):
        groups.setdefault(find(tile_id), []).append(tile_id)
    return list(groups.values())


def _spread_fast_k2d_components(
    flat_tiles: np.ndarray,
    faces: np.ndarray,
    grid,
    params=None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    tiles = np.asarray(flat_tiles, dtype=float).copy()
    if len(tiles) == 0:
        return tiles, {"k2d_fast_component_spread_applied": False}
    components = _connected_tile_components_from_faces(faces, len(tiles))
    if len(components) <= 1:
        return tiles, {
            "k2d_fast_component_spread_applied": False,
            "k2d_fast_component_count": int(len(components)),
            "k2d_fast_component_spread_reason": "single hinge-connected component",
        }
    centers = np.mean(tiles, axis=1)
    component_centers = np.asarray([np.mean(centers[component], axis=0) for component in components], dtype=float)
    global_center = np.mean(component_centers, axis=0)
    tile_size = max(float(getattr(grid, "tile_size", 1.0)), 1e-9)
    factor = float(getattr(params, "hinge_layout_initial_expansion", 1.6)) if params is not None else 1.6
    factor = max(1.0, min(factor, 10.0))
    component_delta = (component_centers - global_center) * (factor - 1.0) * 1.5
    layout_extent = float(np.linalg.norm(np.ptp(tiles.reshape(-1, 2), axis=0)))
    drift_limit_tiles = float(getattr(params, "hinge_layout_max_center_drift_tiles", 5.0)) if params is not None else 5.0
    # Split separation is allowed to be much larger than per-tile local drift;
    # otherwise the cut opens only slightly and the thick panels still crowd.
    expansion_drift = tile_size * max(0.0, factor - 1.0) * 1.5
    drift_limit = max(tile_size * max(drift_limit_tiles, 12.0), layout_extent * 0.35, tile_size * 6.0, expansion_drift)
    norms = np.linalg.norm(component_delta, axis=1)
    active = norms > drift_limit
    if np.any(active):
        component_delta[active] *= (drift_limit / np.maximum(norms[active], 1e-12))[:, None]
    for component_id, component in enumerate(components):
        tiles[component] += component_delta[component_id][None, None, :]
    moved = np.linalg.norm(component_delta, axis=1)
    return tiles, {
        "k2d_fast_component_spread_applied": True,
        "k2d_fast_component_count": int(len(components)),
        "k2d_fast_component_spread_initial_expansion": float(factor),
        "k2d_fast_component_spread_expansion_drift": float(expansion_drift),
        "k2d_fast_component_spread_drift_limit": float(drift_limit),
        "k2d_fast_component_spread_move_mean": float(np.mean(moved)) if moved.size else 0.0,
        "k2d_fast_component_spread_move_max": float(np.max(moved)) if moved.size else 0.0,
    }


def _separate_fast_k2d_components(
    flat_tiles: np.ndarray,
    faces: np.ndarray,
    grid,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    tiles = np.asarray(flat_tiles, dtype=float).copy()
    components = _connected_tile_components_from_faces(faces, len(tiles))
    if len(tiles) == 0 or len(components) <= 1:
        tiles_3d = np.dstack([tiles, np.zeros(tiles.shape[:2])]) if len(tiles) else np.zeros((0, 4, 3), dtype=float)
        return tiles, {
            "k2d_component_separation_applied": False,
            "k2d_component_separation_iterations": 0,
            "tile_overlap_count": int(_original._count_2d_tile_collisions(tiles_3d)) if len(tiles) else 0,
            "min_clearance": float(_original._min_aabb_clearance_2d(tiles_3d)) if len(tiles) else 0.0,
        }

    tile_size = max(float(getattr(grid, "tile_size", 1.0)), 1e-9)
    desired_clearance = max(float(getattr(grid, "gap_size", 0.08)) * 8.0, tile_size * 3.0)
    component_ids = [np.asarray(component, dtype=int) for component in components]
    iterations = 0
    for _ in range(80):
        bounds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for ids in component_ids:
            pts = tiles[ids].reshape(-1, 2)
            bounds.append((np.min(pts, axis=0), np.max(pts, axis=0), np.mean(pts, axis=0)))
        shifts = np.zeros((len(component_ids), 2), dtype=float)
        active_count = 0
        for i in range(len(bounds)):
            min_i, max_i, center_i = bounds[i]
            for j in range(i + 1, len(bounds)):
                min_j, max_j, center_j = bounds[j]
                overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                if not np.all(overlap > -desired_clearance):
                    continue
                axis = int(np.argmin(overlap))
                sign = 1.0 if center_i[axis] >= center_j[axis] else -1.0
                amount = max(0.0, float(overlap[axis]) + desired_clearance) * 0.55
                if amount <= 1e-12:
                    continue
                delta = np.zeros(2, dtype=float)
                delta[axis] = sign * amount
                shifts[i] += delta
                shifts[j] -= delta
                active_count += 1
        if active_count == 0:
            break
        max_step = tile_size * 4.0
        norms = np.linalg.norm(shifts, axis=1)
        too_large = norms > max_step
        if np.any(too_large):
            shifts[too_large] *= (max_step / np.maximum(norms[too_large], 1e-12))[:, None]
        for component_id, ids in enumerate(component_ids):
            tiles[ids] += shifts[component_id][None, None, :]
        iterations += 1
        if float(np.max(np.linalg.norm(shifts, axis=1))) <= tile_size * 1e-4:
            break

    tiles_3d = np.dstack([tiles, np.zeros(tiles.shape[:2])])
    return tiles, {
        "k2d_component_separation_applied": True,
        "k2d_component_separation_iterations": int(iterations),
        "k2d_component_separation_target_clearance": float(desired_clearance),
        "tile_overlap_count": int(_original._count_2d_tile_collisions(tiles_3d)),
        "min_clearance": float(_original._min_aabb_clearance_2d(tiles_3d)),
    }


def _make_flat_tile_layout(mesh, params=None):
    start = time.perf_counter()
    timings: dict[str, float] = {}
    tile_count = int(len(np.asarray(mesh.faces, dtype=int)))
    fast_threshold = int(getattr(params, "k2d_independent_fast_tile_threshold", 150)) if params is not None else 150
    if tile_count > fast_threshold:
        fast_start = time.perf_counter()
        raw_tiles_xy = _tiles_from_mesh_vertices(mesh.vertices, mesh.faces)[:, :, :2]
        initial = raw_tiles_xy.copy()
        init_metrics = {
            "rhombus_void_initializer_enabled": False,
            "rhombus_void_initializer": "disabled in fast path to preserve non-split hinge coincidence",
            "k2d_fast_layout_hinge_preservation_mode": "keep original K2D tile-corner positions inside each split component",
        }
        initial, spread_metrics = _spread_fast_k2d_components(initial, mesh.faces, mesh.grid, params)
        initial, separation_metrics = _separate_fast_k2d_components(initial, mesh.faces, mesh.grid)
        fast_sec = float(time.perf_counter() - fast_start)
        shape_error = _original._tile_shape_distance_error(
            np.dstack([initial, np.zeros(initial.shape[:2])]),
            np.dstack([raw_tiles_xy, np.zeros(raw_tiles_xy.shape[:2])]),
        )
        shape_error_max = _original._tile_shape_distance_error(
            np.dstack([initial, np.zeros(initial.shape[:2])]),
            np.dstack([raw_tiles_xy, np.zeros(raw_tiles_xy.shape[:2])]),
            use_max=True,
        )
        if "k2d_gap_count" in mesh.metrics:
            gap_count = int(mesh.metrics["k2d_gap_count"])
        else:
            gap_count = max(0, 4 * tile_count - int(len(_original._unique_mesh_edges(mesh.faces))))
        metrics = {
            "layout_type": "independent rigid K2D tile linkage layout",
            "k2d_shared_mesh_role": "abstract edge-length mesh only; tile vertices are duplicated for fabrication",
            "t2d_uses_independent_tile_vertices": True,
            "tile_vertices_are_duplicated_from_k2d_faces": True,
            "shared_edge_gluing_disabled": True,
            "paper_layout_correction": True,
            "hinge_layout_stage": "K2D shared mesh to T2D Top Hinge independent linkage",
            "hinge_layout_optimizer": "fast split-component layout; non-split hinge coincidence preserved",
            "hinge_layout_deferred_to_dual_hinge": True,
            "k2d_independent_fast_large_layout": True,
            "k2d_independent_fast_tile_threshold": int(fast_threshold),
            "k2d_independent_fast_initializer_sec": float(fast_sec),
            "k2d_independent_layout_wrapper_total_sec": float(time.perf_counter() - start),
            "k2d_independent_layout_slowest_substep": "fast_rhombus_void_initializer",
            "k2d_independent_retry_skipped_for_speed": True,
            "k2d_independent_retry_skip_reason": "large tile count; residual collision/clearance optimization is deferred to T2D/Dual Hinge",
            "tile_shape_preserved_from_K2D": bool(shape_error_max < 1e-8),
            "k2d_tile_shape_rms_error_after_layout": float(shape_error),
            "k2d_tile_shape_max_error_after_layout": float(shape_error_max),
            "tile_count": int(tile_count),
            "vertices_per_tile": 4,
            "k2d_gap_count": int(gap_count),
            "hinge_pair_count": int(gap_count),
            "tile_overlap_count": int(mesh.metrics.get("k2d_tile_overlap_count", 0)),
            "min_clearance": float(mesh.metrics.get("k2d_min_clearance", 0.0)),
            "gap_opening_model": "split components may be separated widely; hinges inside each component stay coincident",
            **init_metrics,
            **spread_metrics,
            **separation_metrics,
        }
        return _original.FlatTileLayout(
            tile_top_vertices_2d=initial,
            tile_ids=list(range(tile_count)),
            hinge_pairs=[],
            gap_polygons=[],
            metrics=metrics,
        )

    original_start = time.perf_counter()
    previous_se2_layout = _original._paper_local_global_se2_layout
    _original._paper_local_global_se2_layout = _paper_local_global_se2_layout
    try:
        layout = _ORIGINAL_MAKE_FLAT_TILE_LAYOUT(mesh, params)
    finally:
        _original._paper_local_global_se2_layout = previous_se2_layout
    timings["k2d_independent_original_layout_sec"] = float(time.perf_counter() - original_start)
    layout.metrics["k2d_independent_fast_se2_patch_applied"] = True
    layout.metrics["k2d_independent_fast_se2_patch_scope"] = "K2D independent tile layout only"
    min_clearance = float(layout.metrics.get("min_clearance", 0.0))
    if min_clearance >= -1e-9 or params is None:
        layout.metrics.update(timings)
        layout.metrics["k2d_independent_layout_wrapper_total_sec"] = float(time.perf_counter() - start)
        layout.metrics["k2d_independent_layout_slowest_substep"] = "original_layout"
        return layout

    tile_count = int(getattr(layout, "tile_top_vertices_2d", np.zeros((0,))).shape[0])
    if tile_count > 1200:
        layout.metrics.update(timings)
        layout.metrics["k2d_independent_retry_skipped_for_speed"] = True
        layout.metrics["k2d_independent_retry_skip_reason"] = "large tile count; defer residual clearance cleanup to T2D/Dual Hinge footprint stages"
        layout.metrics["k2d_independent_layout_wrapper_total_sec"] = float(time.perf_counter() - start)
        layout.metrics["k2d_independent_layout_slowest_substep"] = max(timings, key=timings.get)
        return layout

    retry_params = copy.copy(params)
    retry_params.hinge_layout_iterations = max(int(getattr(params, "hinge_layout_iterations", 120)), 360)
    retry_params.hinge_layout_collision_weight = max(float(getattr(params, "hinge_layout_collision_weight", 0.35)), 1.5)
    retry_params.hinge_layout_connection_weight = max(float(getattr(params, "hinge_layout_connection_weight", 3.0)), 8.0)
    retry_params.hinge_layout_initial_expansion = max(float(getattr(params, "hinge_layout_initial_expansion", 1.08)), 1.3)
    retry_params.hinge_layout_max_center_drift_tiles = max(float(getattr(params, "hinge_layout_max_center_drift_tiles", 2.0)), 4.0)
    retry_params.hinge_layout_collision_sweeps_per_iteration = max(
        int(getattr(params, "hinge_layout_collision_sweeps_per_iteration", 2)),
        4,
    )
    retry_params.hinge_layout_time_budget_sec = max(float(getattr(params, "hinge_layout_time_budget_sec", 8.0)), 12.0)

    retry_start = time.perf_counter()
    previous_se2_layout = _original._paper_local_global_se2_layout
    _original._paper_local_global_se2_layout = _paper_local_global_se2_layout
    try:
        retry = _ORIGINAL_MAKE_FLAT_TILE_LAYOUT(mesh, retry_params)
    finally:
        _original._paper_local_global_se2_layout = previous_se2_layout
    timings["k2d_independent_retry_layout_sec"] = float(time.perf_counter() - retry_start)
    retry.metrics["k2d_independent_fast_se2_patch_applied"] = True
    retry.metrics["k2d_independent_fast_se2_patch_scope"] = "K2D independent tile layout retry only"
    retry_clearance = float(retry.metrics.get("min_clearance", min_clearance))
    if retry_clearance > min_clearance:
        before_retry_clearance = min_clearance
        layout = retry
        min_clearance = retry_clearance
        layout.metrics["k2d_layout_retry_for_peak_aligned_grid"] = True
        layout.metrics["k2d_layout_min_clearance_before_retry"] = float(before_retry_clearance)

    if min_clearance >= -1e-9:
        layout.metrics.update(timings)
        layout.metrics["k2d_independent_layout_wrapper_total_sec"] = float(time.perf_counter() - start)
        layout.metrics["k2d_independent_layout_slowest_substep"] = max(timings, key=timings.get)
        return layout

    separation_start = time.perf_counter()
    separated, separation_metrics = _separate_independent_flat_tiles(layout.tile_top_vertices_2d, mesh.grid)
    timings["k2d_independent_post_separation_sec"] = float(time.perf_counter() - separation_start)
    separated_layout = _rebuild_flat_tile_layout_with_vertices(layout, separated, mesh)
    separated_layout.metrics.update(separation_metrics)
    separated_layout.metrics.update(timings)
    separated_layout.metrics["k2d_independent_layout_wrapper_total_sec"] = float(time.perf_counter() - start)
    separated_layout.metrics["k2d_independent_layout_slowest_substep"] = max(timings, key=timings.get)
    if float(separated_layout.metrics.get("min_clearance", min_clearance)) > min_clearance:
        return separated_layout
    layout.metrics.update(timings)
    layout.metrics["k2d_independent_layout_wrapper_total_sec"] = float(time.perf_counter() - start)
    layout.metrics["k2d_independent_layout_slowest_substep"] = max(timings, key=timings.get)
    return layout


def _separate_independent_flat_tiles(flat_tiles: np.ndarray, grid) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    tiles = np.asarray(flat_tiles, dtype=float).copy()
    if len(tiles) == 0:
        return tiles, {"k2d_post_separation_applied": False}
    desired_clearance = max(float(getattr(grid, "gap_size", 0.08)) * 0.2, 1e-5)
    applied = 0
    for _ in range(240):
        pairs = _original._spatial_candidate_pairs_for_tiles(tiles, pad=max(float(getattr(grid, "tile_size", 1.0)) * 2.0, 1.0))
        moved = np.zeros((len(tiles), 2), dtype=float)
        for i, j in pairs:
            min_i = np.min(tiles[i], axis=0)
            max_i = np.max(tiles[i], axis=0)
            min_j = np.min(tiles[j], axis=0)
            max_j = np.max(tiles[j], axis=0)
            overlap = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
            if not np.all(overlap > desired_clearance):
                continue
            axis = int(np.argmin(overlap))
            center_i = np.mean(tiles[i], axis=0)
            center_j = np.mean(tiles[j], axis=0)
            sign = 1.0 if center_i[axis] >= center_j[axis] else -1.0
            delta = np.zeros(2, dtype=float)
            delta[axis] = sign * (float(overlap[axis]) + desired_clearance) * 0.52
            moved[i] += delta
            moved[j] -= delta
        max_move = float(np.max(np.linalg.norm(moved, axis=1))) if len(moved) else 0.0
        if max_move <= 1e-10:
            break
        tiles += moved[:, None, :]
        tiles -= np.mean(np.mean(tiles, axis=1), axis=0)
        applied += 1
    for _ in range(80):
        tiles_3d = np.dstack([tiles, np.zeros(tiles.shape[:2])])
        pairs = _original._spatial_candidate_pairs_for_tiles(tiles, pad=max(float(getattr(grid, "tile_size", 1.0)) * 2.0, 1.0))
        if float(_original._min_aabb_clearance_2d_from_pairs(tiles_3d, pairs)) >= 0.0:
            break
        centers = np.mean(tiles, axis=1)
        global_center = np.mean(centers, axis=0)
        tiles += ((centers - global_center) * 0.03)[:, None, :]

    tiles_3d = np.dstack([tiles, np.zeros(tiles.shape[:2])])
    pairs = _original._spatial_candidate_pairs_for_tiles(tiles, pad=max(float(getattr(grid, "tile_size", 1.0)) * 2.0, 1.0))
    return tiles, {
        "k2d_post_separation_applied": True,
        "k2d_post_separation_iterations": int(applied),
        "tile_overlap_count": int(_original._count_2d_tile_collisions_from_pairs(tiles_3d, pairs)),
        "min_clearance": float(_original._min_aabb_clearance_2d_from_pairs(tiles_3d, pairs)),
    }


def _rebuild_flat_tile_layout_with_vertices(layout, flat_tiles: np.ndarray, mesh):
    flat_tiles = np.asarray(flat_tiles, dtype=float)
    hinge_specs = _original._vertex_hinge_specs_from_faces(mesh.faces)
    edge_specs = _original._edge_gap_specs_from_faces(mesh.faces)
    hinge_pairs = [(spec.tile_a, spec.tile_b) for spec in hinge_specs]
    gap_polygons: list[np.ndarray] = []
    for spec in edge_specs:
        if spec.direction == "x":
            a_edge = flat_tiles[spec.tile_a, [1, 2]]
            b_edge = flat_tiles[spec.tile_b, [0, 3]]
        else:
            a_edge = flat_tiles[spec.tile_a, [3, 2]]
            b_edge = flat_tiles[spec.tile_b, [0, 1]]
        gap_polygons.append(np.vstack([a_edge[0], a_edge[1], b_edge[1], b_edge[0]]))
    metrics = dict(layout.metrics)
    metrics.update(
        {
            "k2d_gap_count": len(gap_polygons),
            "hinge_pair_count": len(hinge_pairs),
            "k2d_independent_vertex_joint_error": float(_original._vertex_layout_hinge_error(flat_tiles, hinge_specs)),
        }
    )
    return _original.FlatTileLayout(
        tile_top_vertices_2d=flat_tiles,
        tile_ids=list(layout.tile_ids),
        hinge_pairs=hinge_pairs,
        gap_polygons=gap_polygons,
        metrics=metrics,
    )


def _quad_area_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    values: list[float] = []
    for face in np.asarray(faces, dtype=int):
        pts = np.asarray(vertices, dtype=float)[list(face)]
        tri_a = 0.5 * float(np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))
        tri_b = 0.5 * float(np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[0])))
        values.append(max(tri_a + tri_b, 0.0))
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"quad_area_min": 0.0, "quad_area_median": 0.0}
    return {
        "quad_area_min": float(np.min(arr)),
        "quad_area_median": float(np.median(arr)),
    }


def _edge_length_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    lengths: list[float] = []
    verts = np.asarray(vertices, dtype=float)
    for a, b in _original._unique_mesh_edges(np.asarray(faces, dtype=int)):
        lengths.append(float(np.linalg.norm(verts[int(a)] - verts[int(b)])))
    arr = np.asarray(lengths, dtype=float)
    if arr.size == 0:
        return {"edge_length_min": 0.0, "edge_length_median": 0.0, "edge_length_max": 0.0}
    return {
        "edge_length_min": float(np.min(arr)),
        "edge_length_median": float(np.median(arr)),
        "edge_length_max": float(np.max(arr)),
    }


def _k3d_quality_metrics(base_vertices: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    base_edges = _edge_length_stats(base_vertices, faces)
    edges = _edge_length_stats(vertices, faces)
    base_area = _quad_area_stats(base_vertices, faces)
    area = _quad_area_stats(vertices, faces)
    median_edge = max(float(edges["edge_length_median"]), 1e-12)
    base_median_edge = max(float(base_edges["edge_length_median"]), 1e-12)
    median_area = max(float(area["quad_area_median"]), 1e-12)
    base_median_area = max(float(base_area["quad_area_median"]), 1e-12)
    displacement = np.linalg.norm(np.asarray(vertices, dtype=float) - np.asarray(base_vertices, dtype=float), axis=1)
    return {
        "k3d_edge_length_min": float(edges["edge_length_min"]),
        "k3d_edge_length_median": float(edges["edge_length_median"]),
        "k3d_edge_length_max": float(edges["edge_length_max"]),
        "k3d_edge_max_to_median_ratio": float(edges["edge_length_max"] / median_edge) if median_edge > 0.0 else 0.0,
        "k3d_edge_median_ratio_to_m3d": float(edges["edge_length_median"] / base_median_edge),
        "k3d_quad_area_min": float(area["quad_area_min"]),
        "k3d_quad_area_median": float(area["quad_area_median"]),
        "k3d_quad_area_min_to_median_ratio": float(area["quad_area_min"] / median_area) if median_area > 0.0 else 0.0,
        "k3d_quad_area_median_ratio_to_m3d": float(area["quad_area_median"] / base_median_area),
        "k3d_vertex_displacement_max": float(np.max(displacement)) if displacement.size else 0.0,
        "k3d_vertex_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))) if displacement.size else 0.0,
        "m3d_edge_length_median": float(base_edges["edge_length_median"]),
        "m3d_quad_area_median": float(base_area["quad_area_median"]),
    }


def _k3d_quality_reject_reason(metrics: dict[str, float], grid) -> str | None:
    tile_size = max(float(getattr(grid, "tile_size", 1.0)), 1e-8)
    if float(metrics["k3d_edge_max_to_median_ratio"]) > 8.0:
        return "edge_length_outlier_guard"
    if float(metrics["k3d_edge_median_ratio_to_m3d"]) > 3.5 or float(metrics["k3d_edge_median_ratio_to_m3d"]) < 0.25:
        return "edge_scale_drift_guard"
    if float(metrics["k3d_quad_area_min_to_median_ratio"]) < 1e-4:
        return "quad_area_collapse_guard"
    if float(metrics["k3d_quad_area_median_ratio_to_m3d"]) > 8.0 or float(metrics["k3d_quad_area_median_ratio_to_m3d"]) < 0.1:
        return "quad_area_scale_drift_guard"
    if float(metrics["k3d_vertex_displacement_max"]) > tile_size * 8.0:
        return "vertex_displacement_outlier_guard"
    return None


def _weld_k3d_duplicate_reference_vertices(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    reference = np.asarray(reference_vertices, dtype=float)
    out = np.asarray(vertices, dtype=float).copy()
    if len(reference) != len(out) or len(out) == 0:
        return out, {
            "k3d_split_duplicate_weld_applied": False,
            "k3d_split_duplicate_weld_group_count": 0,
            "k3d_split_duplicate_weld_vertex_count": 0,
            "k3d_split_duplicate_weld_max_adjustment": 0.0,
            "k3d_split_duplicate_weld_reason": "empty_or_mismatched_vertices",
        }

    scale = max(float(np.nanmax(np.ptp(reference, axis=0))), 1.0)
    tol = max(float(tolerance), 1e-10 * scale)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for vertex_id, point in enumerate(reference):
        key = tuple(int(round(float(coord) / tol)) for coord in point[:3])
        buckets.setdefault(key, []).append(int(vertex_id))

    group_count = 0
    vertex_count = 0
    max_adjustment = 0.0
    for ids in buckets.values():
        if len(ids) <= 1:
            continue
        pts = out[np.asarray(ids, dtype=int)]
        mean = np.mean(pts, axis=0)
        max_adjustment = max(max_adjustment, float(np.max(np.linalg.norm(pts - mean, axis=1))))
        out[np.asarray(ids, dtype=int)] = mean
        group_count += 1
        vertex_count += len(ids)

    return out, {
        "k3d_split_duplicate_weld_applied": bool(group_count > 0),
        "k3d_split_duplicate_weld_group_count": int(group_count),
        "k3d_split_duplicate_weld_vertex_count": int(vertex_count),
        "k3d_split_duplicate_weld_max_adjustment": float(max_adjustment),
        "k3d_split_duplicate_weld_reason": "reference-coincident vertices kept coincident after K3D optimization",
    }


def _optimize_k3d(target, mesh, parameterization, params):
    out, report = _ORIGINAL_OPTIMIZE_K3D(target, mesh, parameterization, params)
    welded_vertices, weld_metrics = _weld_k3d_duplicate_reference_vertices(mesh.vertices, out.vertices)
    if bool(weld_metrics.get("k3d_split_duplicate_weld_applied", False)):
        out = _original.QuadMesh(
            welded_vertices,
            np.asarray(out.faces, dtype=int).copy(),
            out.grid,
            out.stage,
            dict(out.metrics),
            list(getattr(out, "split_lines", [])),
        )
    quality = _k3d_quality_metrics(mesh.vertices, out.vertices, mesh.faces)
    reject_reason = _k3d_quality_reject_reason(quality, mesh.grid)
    if reject_reason is None:
        out.metrics.update(quality)
        out.metrics.update(weld_metrics)
        out.metrics.update(
            {
                "k3d_solver_model": "numpy/scipy/torch least_squares_or_projective_approximation",
                "k3d_objective_terms": "approximate E_Planar + E_Square + E_Surface",
                "k3d_planarity_residual": float(out.metrics.get("planarity_error_after", 0.0)),
                "k3d_square_residual": float(out.metrics.get("square_error_after", 0.0)),
                "k3d_surface_residual": float(out.metrics.get("surface_fit_error_after", 0.0)),
                "k3d_quality_rejected": False,
                "k3d_quality_guard_rejected": False,
                "k3d_fallback_used": False,
                "k3d_fallback_reason": "",
                "k3d_exactness_label": "approximation",
            }
        )
        return out, report

    metrics = dict(out.metrics)
    metrics.update(quality)
    metrics.update(weld_metrics)
    metrics.update(
        {
            "k3d_quality_guard_rejected": True,
            "k3d_quality_rejected": True,
            "k3d_quality_guard_reason": reject_reason,
            "fallback_used": True,
            "k3d_fallback_used": True,
            "k3d_fallback_reason": reject_reason,
            "optimization_rejected": True,
            "k3d_solver_model": "numpy/scipy/torch least_squares_or_projective_approximation",
            "k3d_objective_terms": "approximate E_Planar + E_Square + E_Surface",
            "k3d_planarity_residual": float(metrics.get("planarity_error_before", 0.0)),
            "k3d_square_residual": float(metrics.get("square_error_before", 0.0)),
            "k3d_surface_residual": float(metrics.get("surface_fit_error_before", 0.0)),
            "k3d_exactness_label": "fallback",
            "approximation_warning": f"K3D optimization rejected by quality guard: {reject_reason}; fallback to M3D",
        }
    )
    fallback = _original.QuadMesh(
        np.asarray(mesh.vertices, dtype=float).copy(),
        np.asarray(mesh.faces, dtype=int).copy(),
        mesh.grid,
        "K3D",
        metrics,
        list(getattr(mesh, "split_lines", [])),
    )
    failed = list(getattr(report, "failed_constraints", []))
    failed.append(reject_reason)
    guarded_report = _original.StageReport(
        name=report.name,
        objective=report.objective,
        before_error=report.before_error,
        after_error=report.before_error,
        constraint_violation=float(metrics.get("planarity_error_before", 0.0)),
        computation_time=report.computation_time,
        failed_constraints=failed,
        counts=_original._mesh_counts(fallback),
    )
    return fallback, guarded_report


def _solve_bottom_vertex(
    top: np.ndarray,
    face_normal: np.ndarray,
    thickness: float,
    side_normals: list[np.ndarray],
    vertex_id: int,
) -> tuple[np.ndarray, bool]:
    """Return bottom vertex and whether fallback was used."""
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    center = np.mean(top, axis=0)
    prev_edge = (vertex_id - 1) % 4
    next_edge = vertex_id % 4

    q_prev = side_normals[prev_edge]
    q_next = side_normals[next_edge]
    mid_prev = 0.5 * (top[local_edges[prev_edge][0]] + top[local_edges[prev_edge][1]])
    mid_next = 0.5 * (top[local_edges[next_edge][0]] + top[local_edges[next_edge][1]])

    bottom_plane_c = float(np.dot(face_normal, center) - float(thickness))
    a_mat = np.vstack([face_normal, q_prev, q_next])
    b_vec = np.asarray(
        [
            bottom_plane_c,
            float(np.dot(q_prev, mid_prev)),
            float(np.dot(q_next, mid_next)),
        ],
        dtype=float,
    )

    fallback = np.asarray(top[vertex_id], dtype=float) - float(thickness) * face_normal
    try:
        cond = float(np.linalg.cond(a_mat))
        if not np.isfinite(cond) or cond > 1e6:
            return fallback, True
        out = np.linalg.solve(a_mat, b_vec)
        if not np.all(np.isfinite(out)):
            return fallback, True
        if float(np.linalg.norm(out - fallback)) > max(10.0 * float(thickness), 1e-6):
            return fallback, True
        signed_depth = float(np.dot(np.asarray(top[vertex_id], dtype=float) - out, face_normal))
        if signed_depth <= max(0.05 * float(thickness), 1e-9):
            return fallback, True
        return out, False
    except Exception:
        return fallback, True


def _orient_tile_normals_consistently(raw_normals: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    normals = np.asarray(raw_normals, dtype=float).copy()
    if len(normals) == 0:
        return normals, {
            "t3d_extrusion_normal_flip_count": 0,
            "t3d_extrusion_normal_component_count": 0,
            "t3d_extrusion_normal_component_global_flip_count": 0,
            "t3d_extrusion_normal_inconsistent_edge_count": 0,
        }

    incidence = _build_edge_incidence(faces)
    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(len(normals))]
    for entries in incidence.values():
        if len(entries) != 2:
            continue
        (tile_a, _edge_a), (tile_b, _edge_b) = entries
        dot = float(np.dot(normals[tile_a], normals[tile_b]))
        same_sign = dot >= 0.0
        adjacency[tile_a].append((tile_b, same_sign))
        adjacency[tile_b].append((tile_a, same_sign))

    signs = np.zeros(len(normals), dtype=float)
    component_count = 0
    components: list[list[int]] = []
    inconsistent = 0
    for root in range(len(normals)):
        if signs[root] != 0.0:
            continue
        component_count += 1
        signs[root] = 1.0
        stack = [root]
        component_ids: list[int] = []
        while stack:
            current = stack.pop()
            component_ids.append(int(current))
            for neighbor, same_sign in adjacency[current]:
                wanted = signs[current] if same_sign else -signs[current]
                if signs[neighbor] == 0.0:
                    signs[neighbor] = wanted
                    stack.append(neighbor)
                elif signs[neighbor] != wanted:
                    inconsistent += 1
        components.append(component_ids)

    signs[signs == 0.0] = 1.0
    component_means: list[np.ndarray] = []
    for ids in components:
        component_normal = np.mean(normals[np.asarray(ids, dtype=int)] * signs[np.asarray(ids, dtype=int), None], axis=0)
        component_means.append(_normalize(component_normal, np.asarray([0.0, 0.0, 1.0])))
    if component_means:
        global_reference = component_means[int(np.argmax([len(ids) for ids in components]))].copy()
        for mean in component_means:
            if float(np.dot(mean, global_reference)) < 0.0:
                global_reference -= mean
            else:
                global_reference += mean
        global_reference = _normalize(global_reference, component_means[0])
    else:
        global_reference = np.asarray([0.0, 0.0, 1.0])
    component_global_flip_count = 0
    for ids, mean in zip(components, component_means):
        if float(np.dot(mean, global_reference)) < 0.0:
            signs[np.asarray(ids, dtype=int)] *= -1.0
            component_global_flip_count += 1

    oriented = normals * signs[:, None]
    return oriented, {
        "t3d_extrusion_normal_flip_count": int(np.sum(signs < 0.0)),
        "t3d_extrusion_normal_component_count": int(component_count),
        "t3d_extrusion_normal_component_global_flip_count": int(component_global_flip_count),
        "t3d_extrusion_normal_inconsistent_edge_count": int(inconsistent),
    }


def _extrude_tiles(mesh, thickness: float, stage: str):
    """Extrude K3D tiles using shared-edge miter/contact planes.

    Previous behavior:
        bottom = top - thickness * tile_normal

    New behavior:
        - top face remains K3D
        - bottom vertices lie on the offset bottom plane
        - each side face lies on an edge plane
        - shared edges use a single miter/contact plane derived from the two
          adjacent tiles, so neighboring thick panels meet consistently
    """
    import time

    start = time.perf_counter()
    top_tiles = _original._mesh_tiles(mesh)
    tile_count = int(top_tiles.shape[0])
    vertices = np.zeros((tile_count, 8, 3), dtype=float)
    transforms = np.zeros((tile_count, 4, 4), dtype=float)
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    if tile_count == 0:
        top_faces = np.asarray([], dtype=int).reshape(0, 4)
        bottom_faces = np.asarray([], dtype=int).reshape(0, 4)
        side_faces = np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)
        assembly = _original.TileAssembly(
            vertices=vertices,
            top_faces=top_faces,
            bottom_faces=bottom_faces,
            side_faces=side_faces,
            stage=stage,
            metrics={
                "objective": "Contact-aware mitered extrusion.",
                "extrusion_model": "mitered_contact_planes",
                "contact_aware_extrusion": True,
                "tile_thickness": float(thickness),
                "tile_count": 0,
            },
            transform_matrices=transforms,
        )
        report = _original.StageReport(
            name=f"{mesh.stage} -> {stage}",
            objective="Extrude K3D into contact-aware mitered eight-vertex frustum tiles.",
            before_error=0.0,
            after_error=0.0,
            constraint_violation=0.0,
            computation_time=time.perf_counter() - start,
            counts=_original._assembly_counts(assembly),
        )
        return assembly, report

    raw_normals = np.asarray([_original._quad_normal(top) for top in top_tiles], dtype=float)
    normals, normal_orientation_metrics = _orient_tile_normals_consistently(raw_normals, mesh.faces)
    raw_side_normals: list[list[np.ndarray]] = []
    for tile_id, top in enumerate(top_tiles):
        raw_side_normals.append([_edge_inward_normal(top, normals[tile_id], edge) for edge in local_edges])

    side_normals: list[list[np.ndarray]] = [[raw_side_normals[i][e].copy() for e in range(4)] for i in range(tile_count)]
    incidence = _build_edge_incidence(mesh.faces)
    split_edge_pairs = _coincident_boundary_edge_pairs(top_tiles, mesh.faces)
    split_edge_entries = {entry for pair in split_edge_pairs for entry in pair}
    internal_miter_edge_count = 0
    split_contact_miter_edge_count = 0
    boundary_side_plane_count = 0
    nonmanifold_edge_count = 0

    for entries in incidence.values():
        if len(entries) == 1:
            if entries[0] in split_edge_entries:
                continue
            boundary_side_plane_count += 1
            continue
        if len(entries) != 2:
            nonmanifold_edge_count += 1
            continue
        (tile_a, edge_a), (tile_b, edge_b) = entries
        q_a = raw_side_normals[tile_a][edge_a]
        q_b = raw_side_normals[tile_b][edge_b]
        miter = _normalize(q_a - q_b, q_a)
        if float(np.linalg.norm(miter)) <= 1e-12:
            miter = q_a
        side_normals[tile_a][edge_a] = miter
        side_normals[tile_b][edge_b] = -miter
        internal_miter_edge_count += 1

    for (tile_a, edge_a), (tile_b, edge_b) in split_edge_pairs:
        q_a = raw_side_normals[tile_a][edge_a]
        q_b = raw_side_normals[tile_b][edge_b]
        miter = _normalize(q_a - q_b, q_a)
        if float(np.linalg.norm(miter)) <= 1e-12:
            miter = q_a
        side_normals[tile_a][edge_a] = miter
        side_normals[tile_b][edge_b] = -miter
        split_contact_miter_edge_count += 1

    fallback_count = 0
    max_bottom_vertex_jump = 0.0
    for tile_id, top in enumerate(top_tiles):
        normal = normals[tile_id]
        bottom = np.zeros((4, 3), dtype=float)
        for vertex_id in range(4):
            fallback_vertex = np.asarray(top[vertex_id], dtype=float) - float(thickness) * normal
            bottom[vertex_id], used_fallback = _solve_bottom_vertex(
                top,
                normal,
                float(thickness),
                side_normals[tile_id],
                vertex_id,
            )
            fallback_count += int(used_fallback)
            max_bottom_vertex_jump = max(max_bottom_vertex_jump, float(np.linalg.norm(bottom[vertex_id] - fallback_vertex)))

        vertices[tile_id, :4] = top
        vertices[tile_id, 4:] = bottom

        # IMPORTANT for T2D/animation compatibility:
        # Do not store a shearing/affine top->bottom map here.  The original
        # T2D builder treats transform_matrices as a stable per-tile geometric
        # offset when it lays out thick panels in the flat state.  A least-squares
        # affine map can inject shear/scale into T2D and break the deployment
        # animation.  Keep this transform rigid/translation-only as a safe seed;
        # the patched T2D builder below then rigidly places the full mitered T3D
        # solid so per-tile shape is preserved.
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = np.mean(bottom, axis=0) - np.mean(top, axis=0)
        transforms[tile_id] = transform

    top_faces = np.asarray([[0, 1, 2, 3] for _ in range(tile_count)], dtype=int)
    bottom_faces = np.asarray([[4, 7, 6, 5] for _ in range(tile_count)], dtype=int)
    side_faces = np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)

    planarity = _original._tile_face_planarity(vertices)
    face_planarity = _original._tile_face_planarity_by_group(vertices)
    signed_thickness = np.sum((vertices[:, :4] - vertices[:, 4:]) * normals[:, None, :], axis=2)
    thickness_error = signed_thickness - float(thickness)
    reversed_extrusion_vertex_count = int(np.sum(signed_thickness <= 0.0))
    center_shift = np.mean(vertices[:, 4:], axis=1) - np.mean(vertices[:, :4], axis=1)
    normal_shift_error = np.linalg.norm(center_shift + float(thickness) * normals, axis=1)

    assembly = _original.TileAssembly(
        vertices=vertices,
        top_faces=top_faces,
        bottom_faces=bottom_faces,
        side_faces=side_faces,
        stage=stage,
        metrics={
            "objective": "Contact-aware mitered extrusion and face planarity report.",
            "extrusion_model": "mitered_contact_planes",
            "t3d_extrusion_model": "experimental_mitered_contact_planes",
            "contact_aware_extrusion": True,
            "mitered_shared_edge_planes": True,
            "legacy_normal_translation_extrusion": False,
            "t2d_transform_seed_model": "translation_only_center_shift_no_affine_shear",
            "face_planarity_error": planarity,
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            "tile_thickness": float(thickness),
            "thickness_target": float(thickness),
            "thickness_error_rms": float(np.sqrt(np.mean(thickness_error * thickness_error))) if thickness_error.size else 0.0,
            "thickness_error_max": float(np.max(np.abs(thickness_error))) if thickness_error.size else 0.0,
            "t3d_reversed_extrusion_vertex_count": int(reversed_extrusion_vertex_count),
            "t3d_min_signed_thickness": float(np.min(signed_thickness)) if signed_thickness.size else 0.0,
            "normal_translation_center_shift_error_rms": float(np.sqrt(np.mean(normal_shift_error * normal_shift_error))) if normal_shift_error.size else 0.0,
            "internal_miter_edge_count": int(internal_miter_edge_count),
            "split_contact_miter_edge_count": int(split_contact_miter_edge_count),
            "split_contact_side_edges": [[int(tile_id), int(edge_id)] for tile_id, edge_id in sorted(split_edge_entries)],
            "split_contact_side_face_hidden_in_viewer": bool(split_contact_miter_edge_count > 0),
            "boundary_side_plane_count": int(boundary_side_plane_count),
            "nonmanifold_edge_count": int(nonmanifold_edge_count),
            **normal_orientation_metrics,
            "bottom_vertex_solve_fallback_count": int(fallback_count),
            "t3d_bottom_vertex_solve_fallback_count": int(fallback_count),
            "t3d_max_bottom_vertex_jump": float(max_bottom_vertex_jump),
            "t3d_max_coordinate_abs": float(np.max(np.abs(vertices))) if vertices.size else 0.0,
            "t3d_nonfinite_vertex_count": int(np.size(vertices) - np.count_nonzero(np.isfinite(vertices))),
            "t3d_degenerate_face_count": int(sum(_quad_area_2d(tile, face) <= 1e-12 for tile in vertices for face in [np.asarray([0, 1, 2, 3])])) if vertices.size else 0,
            "t3d_exactness_label": "experimental",
            "surface_fit_error": float(mesh.metrics.get("surface_fit_error_after", 0.0)),
            "tile_count": int(tile_count),
            "k3d_fallback_warning": str(mesh.metrics.get("approximation_warning", "")),
            **_original._tile_orientation_metrics(vertices, f"{stage.lower()}"),
        },
        transform_matrices=transforms,
    )
    report = _original.StageReport(
        name=f"{mesh.stage} -> {stage}",
        objective="Extrude K3D into contact-aware mitered eight-vertex frustum tiles.",
        before_error=0.0,
        after_error=planarity,
        constraint_violation=planarity,
        computation_time=time.perf_counter() - start,
        counts=_original._assembly_counts(assembly),
    )
    return assembly, report



_ORIGINAL_MAKE_T2D_FROM_TRANSFORMS = _original._make_t2d_from_transforms

_ORIGINAL_OPTIMIZE_T2D_FOOTPRINT_LAYOUT = _original._optimize_t2d_footprint_layout
_ORIGINAL_OPTIMIZE_RIGID_ASSEMBLY_HINGE_LAYOUT_2D = _original._optimize_rigid_assembly_hinge_layout_2d
_ORIGINAL_OPTIMIZE_DUAL_HINGES = _original._optimize_dual_hinges
_ORIGINAL_BUILD_GAP_GRAPH = _original._build_gap_graph


def _grid_with_layout_gap(grid, minimum_gap: float):
    """Return a shallow grid copy whose gap_size is large enough for void layout.

    The original layout solvers use grid.gap_size mainly to set the collision /
    clearance scale.  Increasing it here gives the panel placement stage more
    room to keep voids open without changing the actual K2D/K3D mesh topology.
    """
    out = copy.copy(grid)
    try:
        out.gap_size = max(float(getattr(grid, "gap_size", 0.0)), float(minimum_gap))
    except Exception:
        return grid
    return out


def _free_layout_parameters(
    grid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    initial_expansion: float,
    max_center_drift_tiles: float,
) -> dict[str, float | int]:
    tile_size = max(float(getattr(grid, "tile_size", 1.0)), 1e-8)
    requested_gap = float(getattr(grid, "gap_size", 0.08))
    # A larger optimization-only void clearance.  This does not rewrite the mesh;
    # it only tells the placement optimizer to leave visible air between panels.
    layout_gap = max(requested_gap * 1.75, tile_size * 0.10)
    return {
        "iterations": int(max(240, int(iterations) * 3)),
        "connection_weight": float(max(1.0, float(connection_weight)) * 12.0),
        "collision_weight": float(max(0.0, float(collision_weight)) * 3.0),
        # Keep the initial pose as a weak prior, not as a cage.  The old values
        # were too anchor-heavy for mitered solids and could collapse the holes.
        "anchor_weight": float(max(0.0, min(0.025, float(anchor_weight) * 0.25))),
        "initial_expansion": float(max(1.22, float(initial_expansion))),
        "max_center_drift_tiles": float(max(4.0, float(max_center_drift_tiles))),
        "layout_gap": float(layout_gap),
        "clearance": float(max(layout_gap * 0.65, tile_size * 0.035)),
    }


_T2D_THICK_FOOTPRINT_TILES: np.ndarray | None = None


def _layout_quality_for_top_xy(layout: np.ndarray, transforms: np.ndarray, faces: np.ndarray, grid, constraints) -> dict[str, float | int]:
    layout = np.asarray(layout, dtype=float)
    if layout.size == 0:
        return {"hinge_error": 0.0, "collision_count": 0, "min_clearance": 0.0}
    footprints = _t2d_collision_footprints_from_top_layout(layout, transforms)
    pad = max(float(getattr(grid, "gap_size", 0.08)) * 8.0, float(getattr(grid, "tile_size", 1.0)) * 0.25)
    pairs = _original._spatial_candidate_pairs_for_tiles(footprints, pad=pad)
    specs = _original._vertex_hinge_specs_from_faces(faces)
    return {
        "hinge_error": float(_original._vertex_layout_hinge_error(layout, specs)),
        "collision_count": int(_original._count_2d_footprint_collisions_from_pairs(footprints, pairs)),
        "min_clearance": float(_original._min_footprint_clearance_2d_from_pairs(footprints, pairs)),
    }


def _t2d_collision_footprints_from_top_layout(layout: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    layout = np.asarray(layout, dtype=float)
    thick_tiles = _T2D_THICK_FOOTPRINT_TILES
    if thick_tiles is None or len(thick_tiles) != len(layout):
        return _original._apply_t2d_transforms_to_top_xy(layout, transforms)[:, :, :2]
    flat_tops = np.concatenate([layout, np.zeros((len(layout), 4, 1), dtype=float)], axis=2)
    footprints = np.zeros((len(layout), 8, 2), dtype=float)
    for tile_id in range(len(layout)):
        placed, _transform = _original._rigidly_place_t3d_tile_in_flat_layout(thick_tiles[tile_id], flat_tops[tile_id])
        placed[:4] = flat_tops[tile_id]
        footprints[tile_id] = placed[:, :2]
    return footprints


def _optimize_t2d_footprint_layout(
    top_xy: np.ndarray,
    transforms: np.ndarray,
    faces: np.ndarray,
    grid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
    progress_callback=None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """More permissive T2D placement for contact-aware thick panels.

    Goal ordering:
      1. vertex hinges should be effectively closed;
      2. projected top+bottom footprints should leave visible voids;
      3. the solution should remain near the expanded initial layout.

    This keeps the original local/global SE(2) solve, but gives it more freedom:
    larger expansion/drift, weaker anchor, stronger connection, and a larger
    collision clearance.  A final hinge-polish pass is accepted only if it does
    not introduce a large collision regression.
    """
    rest = np.asarray(top_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"t2d_footprint_optimizer": "empty_free_layout"}

    specs = _original._vertex_hinge_specs_from_faces(faces)
    constraints = _original._hinge_constraint_tuples_from_specs(specs)

    def footprint_builder(layout: np.ndarray) -> np.ndarray:
        return _t2d_collision_footprints_from_top_layout(layout, transforms)

    free = _free_layout_parameters(
        grid,
        iterations,
        connection_weight,
        collision_weight,
        anchor_weight,
        initial_expansion,
        max_center_drift_tiles,
    )
    free_grid = _grid_with_layout_gap(grid, float(free["layout_gap"]))
    before = _layout_quality_for_top_xy(rest, transforms, faces, free_grid, constraints)

    solved, metrics = _original._paper_local_global_se2_layout(
        rest,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=rest,
        iterations=int(free["iterations"]),
        connection_weight=float(free["connection_weight"]),
        collision_weight=float(free["collision_weight"]),
        anchor_weight=float(free["anchor_weight"]),
        clearance=float(free["clearance"]),
        stage_name="T2D Top Hinge free void-preserving placement",
        time_budget_sec=max(float(time_budget_sec), 12.0),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=max(2, int(collision_sweeps_per_iteration)),
        initial_expansion=float(free["initial_expansion"]),
        max_center_drift_tiles=float(free["max_center_drift_tiles"]),
        progress_callback=progress_callback,
    )
    after_free = _layout_quality_for_top_xy(solved, transforms, faces, free_grid, constraints)

    # Hinge polish: make the hinge term even harder.  Because this can close some
    # holes, keep the polished result only if collision/clearance does not regress
    # too far compared with the free-layout solution.
    polished, polish_metrics = _original._paper_local_global_se2_layout(
        rest,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=solved,
        iterations=max(80, int(iterations)),
        connection_weight=float(free["connection_weight"]) * 2.0,
        collision_weight=max(2.0, float(free["collision_weight"]) * 0.75),
        anchor_weight=float(free["anchor_weight"]) * 0.5,
        clearance=float(free["clearance"]) * 0.75,
        stage_name="T2D Top Hinge hard-hinge polish",
        time_budget_sec=max(4.0, float(time_budget_sec) * 0.5),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=max(1, int(collision_sweeps_per_iteration)),
        initial_expansion=1.0,
        max_center_drift_tiles=float(free["max_center_drift_tiles"]),
        progress_callback=None,
    )
    after_polish = _layout_quality_for_top_xy(polished, transforms, faces, free_grid, constraints)
    accept_polish = (
        after_polish["hinge_error"] <= after_free["hinge_error"] * 0.85 + 1e-8
        and after_polish["collision_count"] <= after_free["collision_count"] + max(1, int(len(rest) * 0.03))
    )
    if accept_polish:
        solved = polished
        final = after_polish
    else:
        final = after_free

    hinge_closed = solved
    hinge_close_metrics: dict[str, float | int | bool] = {
        "t2d_hard_hinge_closure_applied": False,
        "t2d_hard_hinge_closure_accepted": False,
    }
    if constraints:
        hinge_start = time.perf_counter()
        hinge_rest = np.dstack([rest, np.zeros(rest.shape[:2], dtype=float)])
        hinge_closed_3d = np.dstack([solved, np.zeros(solved.shape[:2], dtype=float)])
        hinge_objects = [
            _original.Hinge(int(ia), int(ib), int(ca), int(cb), "top", np.zeros(2, dtype=float), np.zeros(3, dtype=float))
            for ia, ca, ib, cb in constraints
            if 0 <= int(ia) < len(solved)
            and 0 <= int(ib) < len(solved)
            and 0 <= int(ca) < solved.shape[1]
            and 0 <= int(cb) < solved.shape[1]
        ]
        closure_iterations = int(max(8, min(40, round(max(1.0, float(connection_weight)) * 2.0))))
        for _ in range(closure_iterations):
            _original._project_hinge_tile_translations(hinge_closed_3d, hinge_objects, 1.0)
            _original._project_rigid_tiles(hinge_closed_3d, hinge_rest, 1.0)
        candidate = hinge_closed_3d[:, :, :2].copy()
        candidate_quality = _layout_quality_for_top_xy(candidate, transforms, faces, free_grid, constraints)
        candidate_hinge = float(candidate_quality["hinge_error"])
        current_hinge = float(final["hinge_error"])
        candidate_collisions = int(candidate_quality["collision_count"])
        current_collisions = int(final["collision_count"])
        collision_allowance = max(2, int(len(rest) * (0.05 + min(max(float(connection_weight), 0.0), 20.0) * 0.005)))
        accept_hinge_closure = (
            candidate_hinge <= current_hinge * 0.80 + 1e-9
            and candidate_collisions <= current_collisions + collision_allowance
        )
        if accept_hinge_closure:
            hinge_closed = candidate
            solved = hinge_closed
            final = candidate_quality
        hinge_close_metrics = {
            "t2d_hard_hinge_closure_applied": True,
            "t2d_hard_hinge_closure_accepted": bool(accept_hinge_closure),
            "t2d_hard_hinge_closure_iterations": int(closure_iterations),
            "t2d_hard_hinge_closure_elapsed_sec": float(time.perf_counter() - hinge_start),
            "t2d_hard_hinge_closure_hinge_before": float(current_hinge),
            "t2d_hard_hinge_closure_hinge_after": float(candidate_hinge),
            "t2d_hard_hinge_closure_collision_before": int(current_collisions),
            "t2d_hard_hinge_closure_collision_after": int(candidate_collisions),
            "t2d_hard_hinge_closure_collision_allowance": int(collision_allowance),
        }

    shape_rms = _original._tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
    )
    shape_max = _original._tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
        use_max=True,
    )
    out = {
        "t2d_footprint_optimizer": "free local/global SE(2) layout with hard-hinge priority and void clearance",
        "t2d_free_layout_enabled": True,
        "t2d_free_layout_goal_order": "hard hinges > open voids/collision clearance > weak initial-layout anchor",
        "t2d_free_layout_iterations": int(free["iterations"]),
        "t2d_free_layout_connection_weight": float(free["connection_weight"]),
        "t2d_free_layout_collision_weight": float(free["collision_weight"]),
        "t2d_free_layout_anchor_weight": float(free["anchor_weight"]),
        "t2d_free_layout_initial_expansion": float(free["initial_expansion"]),
        "t2d_free_layout_max_center_drift_tiles": float(free["max_center_drift_tiles"]),
        "t2d_free_layout_gap_size_used_for_clearance": float(free["layout_gap"]),
        "t2d_free_layout_clearance": float(free["clearance"]),
        "t2d_hard_hinge_polish_accepted": bool(accept_polish),
        **hinge_close_metrics,
        "t2d_footprint_collision_checked_on": "top+bottom projected footprint with SAT, enlarged optimization-only clearance",
        "t2d_footprint_uses_full_mitered_tile_shape": bool(_T2D_THICK_FOOTPRINT_TILES is not None),
        "t2d_footprint_hinge_error_before": float(before["hinge_error"]),
        "t2d_footprint_hinge_error_after": float(final["hinge_error"]),
        "t2d_footprint_collision_count_before": int(before["collision_count"]),
        "t2d_footprint_collision_count_after": int(final["collision_count"]),
        "t2d_footprint_min_clearance_before": float(before["min_clearance"]),
        "t2d_footprint_min_clearance_after": float(final["min_clearance"]),
        "t2d_top_tile_shape_rms_error_after_footprint_layout": float(shape_rms),
        "t2d_top_tile_shape_max_error_after_footprint_layout": float(shape_max),
        "t2d_top_shape_preserved_by_rigid_pose_fit": bool(shape_max < 1e-8),
        **metrics,
    }
    out.update({f"hard_hinge_polish_{k}": v for k, v in polish_metrics.items() if isinstance(v, (int, float, str, bool))})
    return solved, out


def _optimize_rigid_assembly_hinge_layout_2d(
    rest_vertices: np.ndarray,
    hinges,
    grid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.08,
    max_center_drift_tiles: float = 2.0,
    progress_callback=None,
):
    """More permissive dual-hinge/full-panel placement.

    This wraps the original rigid assembly optimizer but deliberately relaxes the
    anchor and expands the trust region, so panels can rearrange to open voids.
    Connection and collision weights are raised to keep hinges closed and panels
    separated.
    """
    free = _free_layout_parameters(
        grid,
        iterations,
        connection_weight,
        collision_weight,
        anchor_weight,
        initial_expansion,
        max_center_drift_tiles,
    )
    free_grid = _grid_with_layout_gap(grid, float(free["layout_gap"]))
    vertices, metrics = _ORIGINAL_OPTIMIZE_RIGID_ASSEMBLY_HINGE_LAYOUT_2D(
        rest_vertices=rest_vertices,
        hinges=hinges,
        grid=free_grid,
        iterations=int(free["iterations"]),
        connection_weight=float(free["connection_weight"]),
        collision_weight=max(0.0, float(free["collision_weight"])),
        anchor_weight=float(free["anchor_weight"]),
        time_budget_sec=max(float(time_budget_sec), 12.0),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=max(2, int(collision_sweeps_per_iteration)),
        initial_expansion=float(free["initial_expansion"]),
        max_center_drift_tiles=float(free["max_center_drift_tiles"]),
        progress_callback=progress_callback,
    )

    # Final rigid hinge closure pass.  This translates whole tiles toward their
    # hinge midpoints and reprojects each tile onto its original rigid shape.  It
    # gives the user the intended behavior: hinges are treated as nearly hard
    # constraints, while the preceding solve already made room for voids.
    repaired = vertices.copy()
    before_hinge = float(_original._hinge_connection_error(repaired, hinges)) if hinges else 0.0
    for _ in range(16):
        _original._project_hinge_tile_translations(repaired, hinges, 1.0)
        _original._project_aabb_collisions(repaired, 0.08, grid=free_grid, all_pairs=False)
        _original._project_rigid_tiles(repaired, rest_vertices, 1.0)
    after_hinge = float(_original._hinge_connection_error(repaired, hinges)) if hinges else 0.0
    # Use the hard-closed result unless it catastrophically increases AABB overlaps.
    old_coll = int(_original._count_aabb_collisions(vertices, free_grid))
    new_coll = int(_original._count_aabb_collisions(repaired, free_grid))
    accept_repair = after_hinge <= before_hinge + 1e-8 and new_coll <= old_coll + max(1, int(len(repaired) * 0.04))
    if accept_repair:
        vertices = repaired
    else:
        after_hinge = before_hinge
        new_coll = old_coll

    metrics = dict(metrics)
    metrics.update(
        {
            "dual_hinge_free_layout_enabled": True,
            "dual_hinge_free_layout_goal_order": "hard hinges > open voids/collision clearance > weak initial-layout anchor",
            "dual_hinge_free_layout_iterations": int(free["iterations"]),
            "dual_hinge_free_layout_connection_weight": float(max(60.0, float(free["connection_weight"]))),
            "dual_hinge_free_layout_collision_weight": float(max(3.5, float(free["collision_weight"]))),
            "dual_hinge_free_layout_anchor_weight": float(free["anchor_weight"]),
            "dual_hinge_free_layout_initial_expansion": float(free["initial_expansion"]),
            "dual_hinge_free_layout_max_center_drift_tiles": float(free["max_center_drift_tiles"]),
            "dual_hinge_free_layout_gap_size_used_for_clearance": float(free["layout_gap"]),
            "dual_hinge_hard_hinge_repair_accepted": bool(accept_repair),
            "dual_hinge_hard_hinge_error_before_repair": float(before_hinge),
            "dual_hinge_hard_hinge_error_after_repair": float(after_hinge),
            "dual_hinge_collision_count_after_hard_repair": int(new_coll),
        }
    )
    return vertices, metrics



def _make_t2d_from_transforms(mesh_2d, flat_layout, mesh_3d, tiles_3d, stage: str, params=None):
    """Build T2D while preserving the full mitered T3D tile shape.

    The first side-face patch changed T3D tiles from translation extrusions into
    mitered frusta.  Those tiles are no longer representable by a single affine
    top->bottom transform without shear.  The old T2D path used the transform to
    create bottom vertices from K2D top vertices, so an affine transform could
    distort the flat panels and break the animation.

    Compatibility strategy:
    1. Let the original T2D builder solve the flat top/footprint layout, using
       the safe translation-only transform seed stored by _extrude_tiles().
    2. Replace each resulting tile by a rigid placement of the actual mitered
       T3D solid at that solved flat top pose.

    This keeps the original working T2D layout behavior but restores the most
    important physical invariant for deployment: each T2D tile and its T3D target
    are the same rigid 8-vertex solid up to rotation/translation.
    """
    global _T2D_THICK_FOOTPRINT_TILES
    start = time.perf_counter()
    original_mesh_2d = mesh_2d
    try:
        max_face_vertex = int(np.max(np.asarray(mesh_2d.faces, dtype=int))) if len(mesh_2d.faces) else -1
        grid_vertices = np.asarray(mesh_2d.grid.vertex_positions, dtype=float)
        if max_face_vertex >= len(grid_vertices) and max_face_vertex < len(mesh_2d.vertices):
            compatible_grid = copy.copy(mesh_2d.grid)
            compatible_grid.vertex_positions = np.asarray(mesh_2d.vertices, dtype=float).copy()
            mesh_2d = _original.QuadMesh(
                np.asarray(mesh_2d.vertices, dtype=float).copy(),
                np.asarray(mesh_2d.faces, dtype=int).copy(),
                compatible_grid,
                mesh_2d.stage,
                dict(mesh_2d.metrics),
                list(mesh_2d.split_lines),
            )
    except Exception:
        mesh_2d = original_mesh_2d

    layout_faces, layout_weld_metrics = _canonicalize_faces_by_coincident_vertices(
        np.asarray(mesh_3d.vertices, dtype=float),
        np.asarray(mesh_2d.faces, dtype=int),
    )
    if bool(layout_weld_metrics.get("split_virtual_weld_applied", False)):
        mesh_2d = _original.QuadMesh(
            np.asarray(mesh_2d.vertices, dtype=float).copy(),
            layout_faces,
            mesh_2d.grid,
            mesh_2d.stage,
            dict(mesh_2d.metrics),
            list(getattr(mesh_2d, "split_lines", [])),
        )

    tile_count = int(len(np.asarray(mesh_2d.faces, dtype=int)))
    fast_t2d = bool(getattr(flat_layout, "metrics", {}).get("k2d_independent_fast_large_layout", False)) or tile_count > 150
    if fast_t2d:
        flat_layout_tops = np.asarray(flat_layout.tile_top_vertices_3d, dtype=float)
        count = min(len(flat_layout_tops), len(tiles_3d.vertices))
        optimization_metrics: dict[str, float | int | str | bool] = {
            "t2d_fast_top_hinge_optimization_applied": False,
            "t2d_fast_top_hinge_optimization_reason": "not_enough_tiles",
        }
        if count > 1:
            opt_start = time.perf_counter()
            previous_thick_footprint_tiles = _T2D_THICK_FOOTPRINT_TILES
            try:
                _T2D_THICK_FOOTPRINT_TILES = np.asarray(tiles_3d.vertices[:count], dtype=float)
                opt_faces = np.asarray(mesh_2d.faces, dtype=int)[:count]
                opt_constraints = _original._hinge_constraint_tuples_from_specs(_original._vertex_hinge_specs_from_faces(opt_faces))
                opt_grid = _grid_with_layout_gap(mesh_2d.grid, float(getattr(mesh_2d.grid, "gap_size", 0.08)))
                before_quality = _layout_quality_for_top_xy(
                    flat_layout_tops[:count, :, :2],
                    np.zeros((count, 4, 4), dtype=float),
                    opt_faces,
                    opt_grid,
                    opt_constraints,
                )
                before_hinge = float(before_quality.get("hinge_error", 0.0))
                before_collisions = int(before_quality.get("collision_count", 0))
                tile_scale = max(float(getattr(mesh_2d.grid, "tile_size", 1.0)), 1e-9)
                preserve_component_hinges = (
                    str(getattr(flat_layout, "metrics", {}).get("k2d_fast_layout_hinge_preservation_mode", ""))
                    == "keep original K2D tile-corner positions inside each split component"
                )
                if before_hinge <= tile_scale * 0.025 and before_collisions == 0:
                    optimization_metrics = {
                        "t2d_fast_top_hinge_optimization_applied": False,
                        "t2d_fast_top_hinge_optimization_accepted": False,
                        "t2d_fast_top_hinge_optimization_elapsed_sec": float(time.perf_counter() - opt_start),
                        "t2d_fast_top_hinge_optimization_reason": "initial fast T2D layout already satisfies hinge/collision guard",
                        "t2d_fast_top_hinge_quality_hinge_before": float(before_hinge),
                        "t2d_fast_top_hinge_quality_hinge_after": float(before_hinge),
                        "t2d_fast_top_hinge_quality_collision_before": int(before_collisions),
                        "t2d_fast_top_hinge_quality_collision_after": int(before_collisions),
                        "t2d_fast_top_hinge_preserve_component_hinges": bool(preserve_component_hinges),
                    }
                else:
                    optimized_xy, optimization_metrics = _optimize_t2d_footprint_layout(
                        flat_layout_tops[:count, :, :2],
                        np.zeros((count, 4, 4), dtype=float),
                        opt_faces,
                        mesh_2d.grid,
                        iterations=max(40, int(getattr(params, "hinge_layout_iterations", 120))) if params is not None else 120,
                        connection_weight=float(getattr(params, "hinge_layout_connection_weight", 8.0)) if params is not None else 8.0,
                        collision_weight=float(getattr(params, "hinge_layout_collision_weight", 4.0)) if params is not None else 4.0,
                        anchor_weight=float(getattr(params, "hinge_layout_anchor_weight", 0.0)) if params is not None else 0.0,
                        time_budget_sec=float(getattr(params, "hinge_layout_time_budget_sec", 8.0)) if params is not None else 8.0,
                        max_candidate_pairs=int(getattr(params, "hinge_layout_max_candidate_pairs", 3000)) if params is not None else 3000,
                        collision_sweeps_per_iteration=int(getattr(params, "hinge_layout_collision_sweeps_per_iteration", 2)) if params is not None else 2,
                        initial_expansion=float(getattr(params, "hinge_layout_initial_expansion", 1.6)) if params is not None else 1.6,
                        max_center_drift_tiles=float(getattr(params, "hinge_layout_max_center_drift_tiles", 5.0)) if params is not None else 5.0,
                        progress_callback=None,
                    )
                    after_quality = _layout_quality_for_top_xy(
                        optimized_xy,
                        np.zeros((count, 4, 4), dtype=float),
                        opt_faces,
                        opt_grid,
                        opt_constraints,
                    )
                    after_hinge = float(after_quality.get("hinge_error", before_hinge))
                    after_collisions = int(after_quality.get("collision_count", before_collisions))
                    hinge_tolerance = tile_scale * (0.035 if preserve_component_hinges else 0.025)
                    hinge_ok = after_hinge <= max(before_hinge * 1.05 + 1e-9, hinge_tolerance)
                    collision_ok = after_collisions <= before_collisions + max(1, int(count * 0.02))
                    improved = after_collisions < before_collisions or after_hinge < before_hinge * 0.98
                    accept_optimized = bool(hinge_ok and collision_ok and improved)
                    if accept_optimized:
                        flat_layout_tops = flat_layout_tops.copy()
                        flat_layout_tops[:count, :, :2] = optimized_xy
                    optimization_metrics = dict(optimization_metrics)
                    optimization_metrics.update(
                        {
                            "t2d_fast_top_hinge_optimization_applied": True,
                            "t2d_fast_top_hinge_optimization_accepted": bool(accept_optimized),
                            "t2d_fast_top_hinge_optimization_elapsed_sec": float(time.perf_counter() - opt_start),
                            "t2d_fast_top_hinge_optimization_reason": "large fast K2D layout repaired by bounded T2D footprint solve",
                            "t2d_fast_top_hinge_quality_hinge_before": float(before_hinge),
                            "t2d_fast_top_hinge_quality_hinge_after": float(after_hinge),
                            "t2d_fast_top_hinge_quality_collision_before": int(before_collisions),
                            "t2d_fast_top_hinge_quality_collision_after": int(after_collisions),
                            "t2d_fast_top_hinge_preserve_component_hinges": bool(preserve_component_hinges),
                            "t2d_fast_top_hinge_acceptance_hinge_tolerance": float(hinge_tolerance),
                            "t2d_fast_top_hinge_requested_connection_weight": float(getattr(params, "hinge_layout_connection_weight", 8.0)) if params is not None else 8.0,
                            "t2d_fast_top_hinge_effective_connection_weight": float(optimization_metrics.get("t2d_free_layout_connection_weight", 0.0)),
                            "t2d_fast_top_hinge_acceptance_rule": "accept only when hinge stays within tolerance and collision/hinge metric improves",
                        }
                    )
            finally:
                _T2D_THICK_FOOTPRINT_TILES = previous_thick_footprint_tiles
        flat_layout_tops, alignment_metrics = _align_flat_tops_to_target_xy(flat_layout_tops, np.asarray(tiles_3d.vertices, dtype=float)[:, :4, :])
        placed_vertices = np.zeros((count, 8, 3), dtype=float)
        rigid_transforms = np.zeros((count, 4, 4), dtype=float)
        top_errors = []
        for tile_id in range(count):
            placed, transform = _original._rigidly_place_t3d_tile_in_flat_layout(
                tiles_3d.vertices[tile_id],
                flat_layout_tops[tile_id],
            )
            placed_vertices[tile_id] = placed
            rigid_transforms[tile_id] = transform
            top_errors.append(np.linalg.norm(placed[:4, :2] - flat_layout_tops[tile_id, :, :2], axis=1))

        top_errors_arr = np.asarray(top_errors, dtype=float).reshape(-1) if top_errors else np.zeros(0)
        face_planarity = _original._tile_face_planarity_by_group(placed_vertices)
        full_shape_rms = _original._tile_shape_distance_error(placed_vertices, tiles_3d.vertices[:count])
        full_shape_max = _original._tile_shape_distance_error(placed_vertices, tiles_3d.vertices[:count], use_max=True)
        top_shape_rms = _original._tile_shape_distance_error(placed_vertices[:, :4, :], tiles_3d.vertices[:count, :4, :])
        top_shape_max = _original._tile_shape_distance_error(placed_vertices[:, :4, :], tiles_3d.vertices[:count, :4, :], use_max=True)
        metrics = {
            "objective": "Fast T2D top-hinge construction: rigidly place T3D tiles at the independent K2D top layout.",
            "t2d_fast_top_hinge_path": True,
            "t2d_fast_top_hinge_reason": "large K2D independent layout; run bounded T2D footprint solve before rigid T3D placement",
            "t2d_fast_top_hinge_tile_threshold": 150,
            **optimization_metrics,
            **alignment_metrics,
            "face_planarity_error": _original._tile_face_planarity(placed_vertices),
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            "transform_source": "rigid placement of each mitered T3D tile onto fast independent K2D top vertices",
            "fabrication_geometry_model": "T2D keeps the full mitered T3D tile shape; no thin-plate regeneration",
            "paper_t2d_extrusion_model": True,
            "top_vertices_match_k2d_max_error": float(np.max(top_errors_arr)) if top_errors_arr.size else 0.0,
            "top_vertices_match_k2d_rms_error": float(np.sqrt(np.mean(top_errors_arr * top_errors_arr))) if top_errors_arr.size else 0.0,
            "top_vertices_rms_from_k2d_layout": float(np.sqrt(np.mean(top_errors_arr * top_errors_arr))) if top_errors_arr.size else 0.0,
            "tile_shape_rms_error_to_T3D": float(full_shape_rms),
            "tile_shape_max_error_to_T3D": float(full_shape_max),
            "tile_shape_preserved_from_T3D": bool(full_shape_max < 1e-8),
            "top_tile_shape_rms_error_to_T3D": float(top_shape_rms),
            "top_tile_shape_max_error_to_T3D": float(top_shape_max),
            "t2d_t3d_congruent_tile_geometry": bool(full_shape_max < 1e-8),
            "layout_split_virtual_weld_applied": bool(layout_weld_metrics.get("split_virtual_weld_applied", False)),
            "layout_split_virtual_weld_group_count": int(layout_weld_metrics.get("split_virtual_weld_group_count", 0)),
        }
        repaired = _original.TileAssembly(
            vertices=placed_vertices,
            top_faces=np.asarray([[0, 1, 2, 3] for _ in range(count)], dtype=int),
            bottom_faces=np.asarray([[4, 7, 6, 5] for _ in range(count)], dtype=int),
            side_faces=np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int),
            stage=stage,
            metrics=metrics,
            transform_matrices=rigid_transforms,
        )
        report = _original.StageReport(
            name=f"K3D -> {stage}",
            objective=str(metrics["objective"]),
            before_error=0.0,
            after_error=float(full_shape_rms),
            constraint_violation=float(metrics["top_vertices_match_k2d_rms_error"]),
            computation_time=time.perf_counter() - start,
            counts=_original._assembly_counts(repaired),
        )
        return repaired, report

    previous_thick_footprint_tiles = _T2D_THICK_FOOTPRINT_TILES
    _T2D_THICK_FOOTPRINT_TILES = np.asarray(tiles_3d.vertices, dtype=float)
    try:
        base_assembly, base_report = _ORIGINAL_MAKE_T2D_FROM_TRANSFORMS(
            mesh_2d,
            flat_layout,
            mesh_3d,
            tiles_3d,
            stage,
            params,
        )
    finally:
        _T2D_THICK_FOOTPRINT_TILES = previous_thick_footprint_tiles
    if len(base_assembly.vertices) == 0:
        return base_assembly, base_report

    placed_vertices = np.zeros_like(base_assembly.vertices)
    rigid_transforms = np.zeros((len(base_assembly.vertices), 4, 4), dtype=float)
    top_errors = []
    flat_layout_tops = np.asarray(flat_layout.tile_top_vertices_3d, dtype=float)
    for tile_id in range(len(base_assembly.vertices)):
        flat_top = base_assembly.vertices[tile_id, :4]
        placed, transform = _original._rigidly_place_t3d_tile_in_flat_layout(
            tiles_3d.vertices[tile_id],
            flat_top,
        )
        placed_vertices[tile_id] = placed
        rigid_transforms[tile_id] = transform
        top_errors.append(np.linalg.norm(placed[:4, :2] - flat_top[:, :2], axis=1))

    top_errors_arr = np.asarray(top_errors, dtype=float).reshape(-1) if top_errors else np.zeros(0)
    face_planarity = _original._tile_face_planarity_by_group(placed_vertices)
    full_shape_rms = _original._tile_shape_distance_error(placed_vertices, tiles_3d.vertices)
    full_shape_max = _original._tile_shape_distance_error(placed_vertices, tiles_3d.vertices, use_max=True)
    top_shape_rms = _original._tile_shape_distance_error(placed_vertices[:, :4, :], tiles_3d.vertices[:, :4, :])
    top_shape_max = _original._tile_shape_distance_error(placed_vertices[:, :4, :], tiles_3d.vertices[:, :4, :], use_max=True)
    k2d_top_error = (
        np.linalg.norm(placed_vertices[:, :4, :2] - flat_layout_tops[: len(placed_vertices), :, :2], axis=2).reshape(-1)
        if len(flat_layout_tops) >= len(placed_vertices)
        else np.asarray([], dtype=float)
    )

    metrics = dict(base_assembly.metrics)
    metrics.update(
        {
            "t2d_geometry_repair_applied": True,
            "t2d_geometry_repair_reason": "mitered T3D cannot be represented by affine/shear top-to-bottom transforms without breaking rigid-panel animation",
            "t2d_geometry_model": "rigidly_placed_mitered_T3D_tiles_after_original_flat_layout",
            "transform_source": "rigid placement of each full mitered T3D tile onto the solved flat top pose",
            "fabrication_geometry_model": "T2D preserves the complete 8-vertex mitered T3D tile shape; top pose comes from the original K2D/T2D layout solve",
            "rigid_copy_of_T3D_forced": True,
            "paper_t2d_extrusion_model": True,
            "t2d_t3d_congruent_tile_geometry": bool(full_shape_max < 1e-6),
            "tile_shape_rms_error_to_T3D": float(full_shape_rms),
            "tile_shape_max_error_to_T3D": float(full_shape_max),
            "top_tile_shape_rms_error_to_K3D": float(top_shape_rms),
            "top_tile_shape_max_error_to_K3D": float(top_shape_max),
            "top_vertices_match_pre_repair_flat_layout_max_error": float(np.max(top_errors_arr)) if top_errors_arr.size else 0.0,
            "top_vertices_match_pre_repair_flat_layout_rms_error": float(np.sqrt(np.mean(top_errors_arr * top_errors_arr))) if top_errors_arr.size else 0.0,
            "top_vertices_match_k2d_max_error": float(np.max(k2d_top_error)) if k2d_top_error.size else 0.0,
            "top_vertices_match_k2d_rms_error": float(np.sqrt(np.mean(k2d_top_error * k2d_top_error))) if k2d_top_error.size else 0.0,
            "face_planarity_error": _original._tile_face_planarity(placed_vertices),
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            "t2d_split_virtual_weld_applied": bool(layout_weld_metrics.get("split_virtual_weld_applied", False)),
            "t2d_split_virtual_weld_group_count": int(layout_weld_metrics.get("split_virtual_weld_group_count", 0)),
            "t2d_split_virtual_weld_vertex_count": int(layout_weld_metrics.get("split_virtual_weld_vertex_count", 0)),
            "t2d_split_virtual_weld_reason": str(layout_weld_metrics.get("split_virtual_weld_reason", "")),
            **_original._tile_orientation_metrics(placed_vertices, "t2d"),
        }
    )
    repaired = _original.TileAssembly(
        vertices=placed_vertices,
        top_faces=base_assembly.top_faces.copy(),
        bottom_faces=base_assembly.bottom_faces.copy(),
        side_faces=base_assembly.side_faces.copy(),
        stage=base_assembly.stage,
        metrics=metrics,
        transform_matrices=rigid_transforms,
    )
    report = _original.StageReport(
        name=base_report.name,
        objective="Generate T2D by original flat layout solve, then rigidly place contact-aware mitered T3D tiles.",
        before_error=base_report.before_error,
        after_error=float(full_shape_rms),
        constraint_violation=float(metrics.get("top_vertices_match_pre_repair_flat_layout_rms_error", 0.0)),
        computation_time=float(base_report.computation_time) + (time.perf_counter() - start),
        failed_constraints=list(getattr(base_report, "failed_constraints", [])),
        counts=_original._assembly_counts(repaired),
    )
    return repaired, report


def _optimize_dual_hinges(grid, mesh_faces, t2d, t3d, params=None, progress_callback=None):
    hinge_faces, hinge_weld_metrics = _canonicalize_faces_by_coincident_tile_tops(
        np.asarray(t3d.vertices, dtype=float)[:, :4, :],
        np.asarray(mesh_faces, dtype=int),
    )
    out, hinge_graph, report = _ORIGINAL_OPTIMIZE_DUAL_HINGES(
        grid,
        hinge_faces,
        t2d,
        t3d,
        params,
        progress_callback,
    )
    metrics = {
        "dual_hinge_split_virtual_weld_applied": bool(hinge_weld_metrics.get("split_virtual_weld_applied", False)),
        "dual_hinge_split_virtual_weld_group_count": int(hinge_weld_metrics.get("split_virtual_weld_group_count", 0)),
        "dual_hinge_split_virtual_weld_vertex_count": int(hinge_weld_metrics.get("split_virtual_weld_vertex_count", 0)),
        "dual_hinge_split_virtual_weld_reason": str(hinge_weld_metrics.get("split_virtual_weld_reason", "")),
        "dual_hinge_constraint_faces": "split-coincident vertex ids are welded for hinge connectivity only",
    }
    out.metrics.update(metrics)
    hinge_graph.metrics.update(metrics)
    return out, hinge_graph, report


def _build_gap_graph(mesh_faces, t2d, t3d):
    gap_faces, gap_weld_metrics = _canonicalize_faces_by_coincident_tile_tops(
        np.asarray(t3d.vertices, dtype=float)[:, :4, :],
        np.asarray(mesh_faces, dtype=int),
    )
    graph = _ORIGINAL_BUILD_GAP_GRAPH(gap_faces, t2d, t3d)
    graph.metrics.update(
        {
            "gap_graph_split_virtual_weld_applied": bool(gap_weld_metrics.get("split_virtual_weld_applied", False)),
            "gap_graph_split_virtual_weld_group_count": int(gap_weld_metrics.get("split_virtual_weld_group_count", 0)),
            "gap_graph_split_virtual_weld_vertex_count": int(gap_weld_metrics.get("split_virtual_weld_vertex_count", 0)),
            "gap_graph_split_virtual_weld_reason": str(gap_weld_metrics.get("split_virtual_weld_reason", "")),
            "gap_graph_constraint_faces": "split-coincident vertex ids are welded for lift/string gap topology only",
        }
    )
    return graph


def _gap_adjacency(gap_graph) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {int(gap.id): [] for gap in gap_graph.gaps}
    for a, b in gap_graph.edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    return adjacency


def _gap_by_id(gap_graph) -> dict[int, object]:
    return {int(gap.id): gap for gap in gap_graph.gaps}


def _weighted_gap_path(gap_graph, start: int, goal: int, mu_c: float = 0.0) -> list[int]:
    start = int(start)
    goal = int(goal)
    if start == goal:
        return [start]
    gaps = _gap_by_id(gap_graph)
    adjacency = _gap_adjacency(gap_graph)
    if start not in gaps or goal not in gaps:
        return [goal]

    def edge_cost(a: int, b: int) -> float:
        pa = np.asarray(gaps[a].centroid_2d, dtype=float)
        pb = np.asarray(gaps[b].centroid_2d, dtype=float)
        length = float(np.linalg.norm(pa - pb))
        # Internal paths are preferred over boundary wandering; high-GPE gaps are
        # attractive because the string is expected to couple to lifting regions.
        boundary_penalty = 0.35 if bool(gaps[b].boundary) else 0.0
        gpe_attraction = 0.05 * max(0.0, float(gaps[b].gpe))
        return max(length, 1e-9) * (1.0 + boundary_penalty + max(float(mu_c), 0.0) * 0.1) / (1.0 + gpe_attraction)

    queue: list[tuple[float, int]] = [(0.0, start)]
    dist: dict[int, float] = {start: 0.0}
    parent: dict[int, int] = {start: -1}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > dist.get(node, float("inf")) + 1e-12:
            continue
        if node == goal:
            break
        for nxt in adjacency.get(node, []):
            nxt = int(nxt)
            next_cost = cost + edge_cost(node, nxt)
            if next_cost + 1e-12 >= dist.get(nxt, float("inf")):
                continue
            dist[nxt] = next_cost
            parent[nxt] = node
            heapq.heappush(queue, (next_cost, nxt))
    if goal not in parent:
        return [goal]
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return list(reversed(path))


def _append_gap_path(route: list[int], segment: list[int]) -> None:
    for gap_id in segment:
        gap_id = int(gap_id)
        if route and route[-1] == gap_id:
            continue
        route.append(gap_id)


def _nearest_boundary_gap_id(gap_graph, point: np.ndarray | None = None) -> int | None:
    boundary = [gap for gap in gap_graph.gaps if bool(gap.boundary)]
    if not boundary:
        return None
    if point is None:
        values = np.asarray([gap.centroid_2d for gap in boundary], dtype=float)
        point = values[np.lexsort((values[:, 1], values[:, 0]))[0]]
    p = np.asarray(point, dtype=float)
    return int(min(boundary, key=lambda gap: float(np.linalg.norm(np.asarray(gap.centroid_2d, dtype=float) - p))).id)


def _ordered_boundary_gap_ids(gap_graph, start: int | None = None) -> list[int]:
    boundary_ids = [int(gap.id) for gap in gap_graph.gaps if bool(gap.boundary)]
    if len(boundary_ids) <= 1:
        return boundary_ids
    boundary_set = set(boundary_ids)
    adjacency = _gap_adjacency(gap_graph)
    gap_lookup = _gap_by_id(gap_graph)
    boundary_adj = {gid: [int(nid) for nid in adjacency.get(gid, []) if int(nid) in boundary_set] for gid in boundary_ids}
    start_id = int(start) if start is not None and int(start) in boundary_set else boundary_ids[0]

    visited: set[int] = set()
    ordered: list[int] = []
    current = start_id
    prev: int | None = None
    while current not in visited:
        ordered.append(current)
        visited.add(current)
        options = [nid for nid in boundary_adj.get(current, []) if nid != prev and nid not in visited]
        if not options:
            break
        cur_pos = np.asarray(gap_lookup[current].centroid_2d, dtype=float)
        options.sort(key=lambda nid: float(np.linalg.norm(np.asarray(gap_lookup[nid].centroid_2d, dtype=float) - cur_pos)))
        prev, current = current, options[0]

    if len(ordered) == len(boundary_ids):
        return ordered

    center = np.mean([np.asarray(gap_lookup[gid].centroid_2d, dtype=float) for gid in boundary_ids], axis=0)
    ordered = sorted(
        boundary_ids,
        key=lambda gid: math.atan2(
            float(np.asarray(gap_lookup[gid].centroid_2d, dtype=float)[1] - center[1]),
            float(np.asarray(gap_lookup[gid].centroid_2d, dtype=float)[0] - center[0]),
        ),
    )
    if start_id in ordered:
        offset = ordered.index(start_id)
        ordered = ordered[offset:] + ordered[:offset]
    return ordered


def _select_lift_points(gap_graph, tau: float):
    """Select LiftPoints using a discrete paper-style GPE peak/basin model.

    The paper describes lift points through GPE peaks with Morse-Smale style
    segmentation and peak coupling.  Here the gap graph is already discrete, so
    we implement the analogous graph procedure: local GPE maxima, steepest-ascent
    basins, and tau-threshold peak coupling.  Chosen gaps are annotated so the UI
    can state exactly which gap became a LiftPoint and why.
    """
    gaps = list(gap_graph.gaps)
    adjacency = _gap_adjacency(gap_graph)
    interior = [gap for gap in gaps if not bool(gap.boundary)]
    candidates = interior if interior else gaps
    positive = [gap for gap in candidates if float(gap.gpe) > 0.0]
    if not positive:
        selected = []
        if gaps:
            gap = max(gaps, key=lambda item: float(item.gpe))
            selected = [_original.LiftPoint(int(gap.id), gap.centroid_2d, gap.centroid_3d, float(gap.gpe), 0)]
            setattr(selected[0], "selection_reason", "fallback_max_gpe_no_positive_interior_peak")
            setattr(selected[0], "basin_size", 1)
        gap_graph.metrics.update(
            {
                "lift_point_selection_model": "discrete_graph_gpe_peak_coupling",
                "lift_point_selection_exactness": "paper-style discrete Morse-Smale approximation",
                "lift_point_count": int(len(selected)),
                "lift_point_selection_threshold": 0.0,
                "lift_point_rows": [
                    {
                        "gap_id": int(lift.gap_id),
                        "cluster_id": int(lift.cluster_id),
                        "gpe": float(lift.gpe),
                        "position_2d": [float(x) for x in np.asarray(lift.position_2d, dtype=float).reshape(-1)[:3]],
                        "position_3d": [float(x) for x in np.asarray(lift.position_3d, dtype=float).reshape(-1)[:3]],
                        "selection_reason": str(getattr(lift, "selection_reason", "")),
                    }
                    for lift in selected
                ],
            }
        )
        return selected

    max_gpe = max(float(gap.gpe) for gap in positive)
    threshold = max(0.0, min(1.0, float(tau))) * max_gpe
    gap_lookup = _gap_by_id(gap_graph)
    candidate_ids = {int(gap.id) for gap in candidates}
    local_maxima = []
    for gap in positive:
        gid = int(gap.id)
        neighbor_ids = [nid for nid in adjacency.get(gid, []) if nid in candidate_ids]
        neighbor_gpe = [float(gap_lookup[nid].gpe) for nid in neighbor_ids]
        if not neighbor_gpe or float(gap.gpe) >= max(neighbor_gpe) - 1e-12:
            local_maxima.append(gap)
    if not local_maxima:
        local_maxima = [max(positive, key=lambda item: float(item.gpe))]

    def ascend_peak(gid: int) -> int:
        seen: set[int] = set()
        current = int(gid)
        while current not in seen:
            seen.add(current)
            current_gap = gap_lookup[current]
            neighbor_ids = [nid for nid in adjacency.get(current, []) if nid in candidate_ids]
            if not neighbor_ids:
                break
            best = max(neighbor_ids, key=lambda nid: float(gap_lookup[nid].gpe))
            if float(gap_lookup[best].gpe) <= float(current_gap.gpe) + 1e-12:
                break
            current = int(best)
        return current

    basin_members: dict[int, list[int]] = {}
    for gap in candidates:
        peak_id = ascend_peak(int(gap.id))
        basin_members.setdefault(peak_id, []).append(int(gap.id))

    ranked = sorted(local_maxima, key=lambda item: (float(item.gpe), len(basin_members.get(int(item.id), []))), reverse=True)
    selected: list[object] = []
    selected_tiles: set[int] = set()
    suppressed_count = 0
    for gap in ranked:
        gid = int(gap.id)
        if float(gap.gpe) < threshold and selected:
            continue
        # Peak coupling: adjacent/overlapping peaks in the same physical tile
        # neighborhood are represented by the highest-GPE peak.
        tiles = {int(tile) for tile in gap.surrounding_tiles}
        if selected_tiles.intersection(tiles):
            suppressed_count += 1
            continue
        lift = _original.LiftPoint(gid, gap.centroid_2d, gap.centroid_3d, float(gap.gpe), len(selected))
        basin = basin_members.get(gid, [gid])
        setattr(lift, "selection_reason", "local_gpe_maximum_above_tau_after_peak_coupling")
        setattr(lift, "basin_size", int(len(basin)))
        setattr(lift, "basin_gap_ids", [int(x) for x in basin])
        selected.append(lift)
        selected_tiles.update(tiles)
    if not selected:
        gap = ranked[0]
        lift = _original.LiftPoint(int(gap.id), gap.centroid_2d, gap.centroid_3d, float(gap.gpe), 0)
        setattr(lift, "selection_reason", "fallback_highest_local_gpe_peak")
        setattr(lift, "basin_size", int(len(basin_members.get(int(gap.id), [int(gap.id)]))))
        setattr(lift, "basin_gap_ids", [int(x) for x in basin_members.get(int(gap.id), [int(gap.id)])])
        selected = [lift]

    rows = []
    for lift in selected:
        rows.append(
            {
                "gap_id": int(lift.gap_id),
                "cluster_id": int(lift.cluster_id),
                "gpe": float(lift.gpe),
                "gpe_ratio": float(lift.gpe / max(max_gpe, 1e-12)),
                "basin_size": int(getattr(lift, "basin_size", 0)),
                "position_2d": [float(x) for x in np.asarray(lift.position_2d, dtype=float).reshape(-1)[:3]],
                "position_3d": [float(x) for x in np.asarray(lift.position_3d, dtype=float).reshape(-1)[:3]],
                "selection_reason": str(getattr(lift, "selection_reason", "")),
            }
        )
    gap_graph.metrics.update(
        {
            "lift_point_selection_model": "discrete_graph_gpe_peak_coupling",
            "lift_point_selection_exactness": "paper-style discrete Morse-Smale approximation on gap graph",
            "lift_point_selection_tau": float(tau),
            "lift_point_selection_threshold": float(threshold),
            "lift_point_candidate_count": int(len(candidates)),
            "lift_point_positive_candidate_count": int(len(positive)),
            "lift_point_local_peak_count": int(len(local_maxima)),
            "lift_point_peak_coupling_suppressed_count": int(suppressed_count),
            "lift_point_count": int(len(selected)),
            "lift_point_gap_ids": [int(lift.gap_id) for lift in selected],
            "lift_point_rows": rows,
        }
    )
    return selected


def _build_string_path(gap_graph, lift_points, mu_c: float):
    boundary = [gap for gap in gap_graph.gaps if bool(gap.boundary)]
    boundary_ids = [int(gap.id) for gap in boundary]
    if not gap_graph.gaps:
        return _original.StringPath([], [], [], 0.0, 0.0, {"string_path_model": "empty_gap_graph"})
    gap_lookup = _gap_by_id(gap_graph)
    lift_ids = [int(lift.gap_id) for lift in lift_points]
    if boundary:
        lift_points_2d = [np.asarray(gap_lookup[gid].centroid_2d, dtype=float) for gid in lift_ids if gid in gap_lookup]
        anchor_target = np.mean(lift_points_2d, axis=0) if lift_points_2d else None
        start = _nearest_boundary_gap_id(gap_graph, anchor_target)
        end = start
    else:
        start = int(gap_graph.gaps[0].id)
        end = start

    ordered_lifts = sorted(
        [gap_lookup[gid] for gid in lift_ids if gid in gap_lookup],
        key=lambda gap: float(gap.gpe),
        reverse=True,
    )
    route: list[int] = _ordered_boundary_gap_ids(gap_graph, start)
    if route and route[0] != route[-1]:
        route.append(route[0])
    if not route and start is not None:
        route.append(int(start))
    current = route[-1] if route else (int(ordered_lifts[0].id) if ordered_lifts else int(gap_graph.gaps[0].id))
    for lift_gap in ordered_lifts:
        segment = _weighted_gap_path(gap_graph, current, int(lift_gap.id), mu_c)
        _append_gap_path(route, segment)
        current = int(lift_gap.id)
    if end is not None and route:
        segment = _weighted_gap_path(gap_graph, route[-1], int(end), mu_c)
        _append_gap_path(route, segment)
    if not route and gap_graph.gaps:
        route = [int(gap_graph.gaps[0].id)]

    theta = _original._turn_angle_total(gap_graph, route)
    friction = _original.safe_capstan_friction(mu_c, theta)
    log_channel_cost = float(mu_c * theta) if math.isfinite(mu_c) and math.isfinite(theta) else float("inf")
    route_node_count = len(route)
    unique_route_node_count = len(set(route))
    duplicate_visit_count = route_node_count - unique_route_node_count
    theta_upper_bound = math.pi * max(0, route_node_count - 2)
    max_single_turn = _original._max_single_turn_angle(gap_graph, route)
    warnings: list[str] = []
    if route_node_count and duplicate_visit_count > route_node_count * 0.5:
        warnings.append("String path revisits many gap nodes.")
    if theta > theta_upper_bound + 1e-6:
        warnings.append("Turn angle exceeds simple polyline upper bound.")
    if theta > 200:
        warnings.append("String path turn angle is extremely large; routing likely failed.")

    metrics = {
        "string_path_model": "boundary_loop_to_gpe_lift_peaks_weighted_gap_graph_path",
        "string_path_exactness": "paper-style channel route approximation; no continuous fabrication channel solver",
        "string_path_start_boundary_gap_id": int(start) if start is not None else -1,
        "string_path_end_boundary_gap_id": int(end) if end is not None else -1,
        "string_path_boundary_loop_included": bool(len(boundary_ids) > 0),
        "string_path_boundary_loop_node_count": int(len(boundary_ids)),
        "route_length": int(route_node_count),
        "route_node_count": int(route_node_count),
        "unique_route_node_count": int(unique_route_node_count),
        "duplicate_visit_count": int(duplicate_visit_count),
        "boundary_gap_count": int(len(boundary_ids)),
        "lift_point_count": int(len(lift_points)),
        "lift_gap_ids": [int(x) for x in lift_ids],
        "max_single_turn_angle": float(max_single_turn),
        "turn_angle_total": float(theta),
        "theta_total": float(theta),
        "theta_upper_bound": float(theta_upper_bound),
        "log_channel_cost": float(log_channel_cost),
        "estimated_channel_friction": float(friction),
        "overflow_prevented": bool(not math.isfinite(friction) or log_channel_cost > 60.0),
        "invalid_turn_accumulation": bool(theta > theta_upper_bound + 1e-6),
        "warnings": "; ".join(warnings),
    }
    return _original.StringPath(
        gap_ids=[int(x) for x in route],
        boundary_gap_ids=boundary_ids,
        lift_gap_ids=[int(x) for x in lift_ids],
        turn_angle_total=float(theta),
        estimated_channel_friction=float(friction),
        metrics=metrics,
    )


def _align_flat_tops_to_target_xy(flat_tops: np.ndarray, target_tops: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    flat = np.asarray(flat_tops, dtype=float).copy()
    target = np.asarray(target_tops, dtype=float)
    count = min(len(flat), len(target))
    if count == 0:
        return flat, {"t2d_global_xy_alignment_applied": False, "t2d_global_xy_alignment_reason": "empty"}
    src = np.mean(flat[:count, :, :2], axis=1)
    dst = np.mean(target[:count, :, :2], axis=1)
    if len(src) < 2:
        return flat, {"t2d_global_xy_alignment_applied": False, "t2d_global_xy_alignment_reason": "too_few_tiles"}
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    src0 = src - src_center
    dst0 = dst - dst_center
    if float(np.linalg.norm(src0)) <= 1e-12 or float(np.linalg.norm(dst0)) <= 1e-12:
        return flat, {"t2d_global_xy_alignment_applied": False, "t2d_global_xy_alignment_reason": "degenerate_centers"}

    def _fit_candidate(mirror: np.ndarray, label: str):
        mirrored_src0 = src0 @ mirror.T
        try:
            u, _s, vt = np.linalg.svd(mirrored_src0.T @ dst0)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0.0:
                vt[-1, :] *= -1.0
                rotation = vt.T @ u.T
        except Exception:
            return None
        aligned_centers = mirrored_src0 @ rotation.T + dst_center
        residual = np.linalg.norm(aligned_centers - dst, axis=1)
        rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0
        return {
            "mirror": mirror,
            "mirror_label": label,
            "rotation": rotation,
            "rms": rms,
            "residual": residual,
        }

    try:
        candidates = [
            _fit_candidate(np.eye(2, dtype=float), "none"),
            _fit_candidate(np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=float), "mirror_x"),
            _fit_candidate(np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=float), "mirror_y"),
        ]
        valid = [candidate for candidate in candidates if candidate is not None]
        if not valid:
            return flat, {"t2d_global_xy_alignment_applied": False, "t2d_global_xy_alignment_reason": "svd_failed"}
        best = min(valid, key=lambda item: float(item["rms"]))
        no_mirror = next((candidate for candidate in valid if candidate["mirror_label"] == "none"), None)
        if no_mirror is not None and best["mirror_label"] != "none":
            no_mirror_rms = float(no_mirror["rms"])
            best_rms = float(best["rms"])
            # Avoid flipping the layout for near-ties; only correct a real chart
            # handedness mismatch.
            if best_rms > no_mirror_rms * 0.92 and (no_mirror_rms - best_rms) < max(1e-9, no_mirror_rms * 0.05):
                best = no_mirror
    except Exception:
        return flat, {"t2d_global_xy_alignment_applied": False, "t2d_global_xy_alignment_reason": "svd_failed"}
    rotation = np.asarray(best["rotation"], dtype=float)
    mirror = np.asarray(best["mirror"], dtype=float)
    aligned_xy = ((flat[:, :, :2] - src_center) @ mirror.T) @ rotation.T + dst_center
    flat[:, :, :2] = aligned_xy
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    residual = np.asarray(best["residual"], dtype=float)
    reflected = str(best["mirror_label"]) != "none"
    return flat, {
        "t2d_global_xy_alignment_applied": True,
        "t2d_global_xy_alignment_rotation_deg": angle,
        "t2d_global_xy_alignment_center_rms": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
        "t2d_global_xy_alignment_target": "T3D top XY tile centers",
        "t2d_global_xy_alignment_reflection_applied": bool(reflected),
        "t2d_global_xy_alignment_reflection_axis": str(best["mirror_label"]),
    }


def _resolve_t2d_assembly(source, stage: str = "dual_hinge"):
    if hasattr(source, "vertices") and hasattr(source, "top_faces") and hasattr(source, "bottom_faces"):
        return source
    normalized = stage.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"dual", "dual_hinge", "t2d_dual", "t2d_dual_hinge"}:
        return source.tiles_2d_dual_hinge
    if normalized in {"top", "top_hinge", "t2d_top", "t2d_top_hinge"}:
        return source.tiles_2d_top_hinge
    raise ValueError(f"unknown T2D STL stage: {stage}")


def _t2d_export_scale(assembly, panel_size: float) -> float:
    vertices = np.asarray(assembly.vertices, dtype=float)
    if vertices.size == 0:
        return 1.0
    top = vertices[:, :4, :2]
    lengths: list[float] = []
    for tile in top:
        for i in range(4):
            value = float(np.linalg.norm(tile[(i + 1) % 4] - tile[i]))
            if value > 1e-12 and np.isfinite(value):
                lengths.append(value)
    median = float(np.median(lengths)) if lengths else 1.0
    return float(panel_size) / max(median, 1e-12)


def _t2d_prism_vertices(assembly, panel_size: float, thickness: float | None) -> np.ndarray:
    scale = _t2d_export_scale(assembly, panel_size)
    source = np.asarray(assembly.vertices, dtype=float)
    if source.ndim == 3 and source.shape[1] >= 8 and source.shape[2] >= 3:
        out = source[:, :8, :3].copy() * scale
        if thickness is not None:
            z_center = np.mean(out[..., 2], axis=1, keepdims=True)
            z_local = out[..., 2] - z_center
            z_span = np.max(z_local, axis=1, keepdims=True) - np.min(z_local, axis=1, keepdims=True)
            active = z_span[:, 0] > 1e-12
            if np.any(active):
                z_local[active] *= float(thickness) / z_span[active]
                out[active, :, 2] = z_center[active] + z_local[active]
        return out

    top_xy = source[:, :4, :2] * scale
    out = np.zeros((top_xy.shape[0], 8, 3), dtype=float)
    out[:, :4, :2] = top_xy
    out[:, 4:, :2] = top_xy
    out[:, :4, 2] = 0.5 * float(thickness if thickness is not None else panel_size * 0.05)
    out[:, 4:, 2] = -0.5 * float(thickness if thickness is not None else panel_size * 0.05)
    return out


def _prism_triangle_indices() -> list[tuple[int, int, int]]:
    return [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]


def _stl_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.zeros(3, dtype=float)
    return normal / norm


def _ascii_stl_bytes(name: str, triangles: list[np.ndarray]) -> bytes:
    lines = [f"solid {name}"]
    for tri in triangles:
        a, b, c = np.asarray(tri, dtype=float)
        n = _stl_normal(a, b, c)
        lines.append(f"  facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}")
        lines.append("    outer loop")
        for p in (a, b, c):
            lines.append(f"      vertex {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _triangle_edge_nonmanifold_count(triangles: list[tuple[int, int, int]]) -> int:
    counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for i, a in enumerate(tri):
            b = tri[(i + 1) % 3]
            key = tuple(sorted((int(a), int(b))))
            counts[key] = counts.get(key, 0) + 1
    return int(sum(1 for count in counts.values() if count != 2))


def _t2d_stl_mesh_and_metrics(assembly, *, panel_size: float = 0.1, thickness: float | None = None):
    panel_size = float(panel_size)
    export_thickness = None if thickness is None else float(thickness)
    vertices = _t2d_prism_vertices(assembly, panel_size, export_thickness)
    local_tris = _prism_triangle_indices()
    triangles: list[np.ndarray] = []
    tri_indices: list[tuple[int, int, int]] = []
    for tile_id, tile in enumerate(vertices):
        offset = tile_id * 8
        for tri in local_tris:
            triangles.append(tile[list(tri)])
            tri_indices.append(tuple(offset + int(i) for i in tri))

    top = vertices[:, :4, :]
    areas = np.asarray([_quad_area_2d(tile, np.asarray([0, 1, 2, 3])) for tile in top], dtype=float)
    aspects = np.asarray([_quad_aspect_ratio(tile, np.asarray([0, 1, 2, 3])) for tile in top], dtype=float)
    metrics = {
        "t2d_tile_count": int(vertices.shape[0]),
        "t2d_vertex_count": int(vertices.shape[0] * vertices.shape[1]),
        "t2d_face_count": int(len(triangles)),
        "t2d_export_thickness": float(np.median(np.ptp(vertices[..., 2], axis=1))) if len(vertices) else 0.0,
        "t2d_export_requested_thickness": None if thickness is None else float(thickness),
        "t2d_export_panel_size": float(panel_size),
        "t2d_min_area": float(np.min(areas)) if len(areas) else 0.0,
        "t2d_max_aspect_ratio": float(np.max(aspects)) if len(aspects) else 0.0,
        "t2d_connected_component_count": int(vertices.shape[0]),
        "t2d_nonmanifold_edge_count": _triangle_edge_nonmanifold_count(tri_indices),
        "t2d_export_model": "current_T2D_assembly_vertices_scaled_to_stl",
        "t2d_export_exactness_label": "fabrication_export",
    }
    return vertices, triangles, metrics


def export_t2d_stl(
    source,
    output_path: str | Path | None = None,
    *,
    stage: str = "dual_hinge",
    separate_tiles: bool = False,
    panel_size: float = 0.1,
    thickness: float | None = None,
    solid_name: str = "onestring_t2d",
):
    """Export T2D flat tile layout as STL bytes or files.

    This is a fabrication export of the current T2D approximation.  It does not
    certify that the upstream T2D construction is the exact OneString paper
    solver; the export metrics record it as a thin-plate STL conversion.
    """
    assembly = _resolve_t2d_assembly(source, stage)
    vertices, triangles, metrics = _t2d_stl_mesh_and_metrics(assembly, panel_size=panel_size, thickness=thickness)
    try:
        assembly.metrics.update(metrics)
    except Exception:
        pass

    if separate_tiles:
        outputs: dict[str, bytes] = {}
        local_tri_count = len(_prism_triangle_indices())
        for tile_id in range(vertices.shape[0]):
            tile_tris = triangles[tile_id * local_tri_count : (tile_id + 1) * local_tri_count]
            filename = f"{solid_name}_tile_{tile_id:04d}.stl"
            outputs[filename] = _ascii_stl_bytes(filename[:-4], tile_tris)
        if output_path is not None:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            for filename, data in outputs.items():
                (out_dir / filename).write_bytes(data)
        return outputs, metrics

    data = _ascii_stl_bytes(solid_name, triangles)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data, metrics


def _spatial_candidate_pairs_for_tiles(tiles_xy: np.ndarray, pad: float = 0.0) -> list[tuple[int, int]]:
    """Broad-phase tile pair search using padded AABB cell occupancy."""
    tiles = np.asarray(tiles_xy, dtype=float)
    n = int(len(tiles))
    if n <= 1:
        return []
    bmin = np.nanmin(tiles[:, :, :2], axis=1)
    bmax = np.nanmax(tiles[:, :, :2], axis=1)
    spans = np.maximum(bmax - bmin, 1e-8)
    pad = max(float(pad), 0.0)
    finite = np.all(np.isfinite(spans), axis=1)
    max_spans = np.max(spans[finite], axis=1) if np.any(finite) else np.asarray([1.0])
    cell = max(float(np.median(max_spans)) + pad, 1e-6)

    lo = np.floor((bmin - pad) / cell).astype(int)
    hi = np.floor((bmax + pad) / cell).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx in range(n):
        if not np.all(np.isfinite(lo[idx])) or not np.all(np.isfinite(hi[idx])):
            continue
        lo_i = lo[idx].copy()
        hi_i = hi[idx].copy()
        for gx in range(int(lo_i[0]), int(hi_i[0]) + 1):
            for gy in range(int(lo_i[1]), int(hi_i[1]) + 1):
                buckets.setdefault((gx, gy), []).append(int(idx))

    pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) <= 1:
            continue
        for a_pos, i in enumerate(members[:-1]):
            for j in members[a_pos + 1:]:
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in pairs:
                    continue
                sep = np.maximum(np.maximum(bmin[b] - bmax[a], bmin[a] - bmax[b]), 0.0)
                if float(np.linalg.norm(sep)) <= pad:
                    pairs.add((int(a), int(b)))
    return sorted(pairs)


def _tiles_from_mesh_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    face_idx = np.asarray(faces, dtype=int)
    if len(face_idx) == 0:
        return np.zeros((0, 4, verts.shape[1] if verts.ndim == 2 else 3), dtype=float)
    return verts[face_idx]


def _edge_matching_errors(xy: np.ndarray, edges: list[tuple[int, int]] | np.ndarray, target_lengths: np.ndarray) -> tuple[float, float]:
    edge_idx = np.asarray(edges, dtype=int)
    if edge_idx.size == 0:
        return 0.0, 0.0
    pts = np.asarray(xy, dtype=float)
    current = np.linalg.norm(pts[edge_idx[:, 0]] - pts[edge_idx[:, 1]], axis=1)
    err = np.abs(current - np.asarray(target_lengths, dtype=float))
    return float(np.mean(err)), float(np.max(err))


def _edge_matching_error(xy: np.ndarray, edges: list[tuple[int, int]] | np.ndarray, target_lengths: np.ndarray) -> float:
    edge_idx = np.asarray(edges, dtype=int)
    if edge_idx.size == 0:
        return 0.0
    pts = np.asarray(xy, dtype=float)
    current = np.linalg.norm(pts[edge_idx[:, 0]] - pts[edge_idx[:, 1]], axis=1)
    diff = current - np.asarray(target_lengths, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def _gap_angles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    face_idx = np.asarray(faces, dtype=int)
    if len(face_idx) == 0:
        return np.zeros(0, dtype=float)
    pts = verts[face_idx, :2]
    prev_pts = np.roll(pts, 1, axis=1)
    next_pts = np.roll(pts, -1, axis=1)
    a = prev_pts - pts
    b = next_pts - pts
    na = np.linalg.norm(a, axis=2)
    nb = np.linalg.norm(b, axis=2)
    denom = na * nb
    valid = denom > 1e-12
    cosv = np.zeros_like(denom, dtype=float)
    cosv[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return np.degrees(np.arccos(np.clip(cosv[valid], -1.0, 1.0))).astype(float)


def _gap_angle_range(vertices: np.ndarray, faces: np.ndarray, *, chunk_faces: int = 200_000) -> tuple[float, float, int]:
    verts = np.asarray(vertices, dtype=float)
    face_idx = np.asarray(faces, dtype=int)
    if len(face_idx) == 0:
        return 0.0, 0.0, 0
    min_angle = float("inf")
    max_angle = float("-inf")
    count = 0
    chunk = max(1, int(chunk_faces))
    for start in range(0, len(face_idx), chunk):
        pts = verts[face_idx[start : start + chunk], :2]
        prev_pts = np.roll(pts, 1, axis=1)
        next_pts = np.roll(pts, -1, axis=1)
        a = prev_pts - pts
        b = next_pts - pts
        denom = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
        valid = denom > 1e-12
        if not np.any(valid):
            continue
        cosv = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
        angles = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))).astype(float)
        min_angle = min(min_angle, float(np.min(angles)))
        max_angle = max(max_angle, float(np.max(angles)))
        count += int(len(angles))
    if count == 0:
        return 0.0, 0.0, 0
    return float(min_angle), float(max_angle), int(count)


def _shared_edge_count_from_faces(faces: np.ndarray, *, chunk_edges: int = 2_000_000) -> int:
    faces_arr = np.asarray(faces, dtype=int)
    if len(faces_arr) == 0:
        return 0
    local_edges = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    edge_blocks: list[np.ndarray] = []
    chunk_faces = max(1, int(chunk_edges) // 4)
    for start in range(0, len(faces_arr), chunk_faces):
        block = faces_arr[start : start + chunk_faces]
        edges = block[:, local_edges].reshape(-1, 2)
        edge_blocks.append(np.sort(edges, axis=1))
    all_edges = np.vstack(edge_blocks)
    _unique, counts = np.unique(all_edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts > 1))


def _k2d_collision_and_clearance_metrics(
    tiles: np.ndarray,
    grid,
    all_pairs: bool = False,
) -> tuple[int, float, int]:
    arr = np.asarray(tiles, dtype=float)
    if len(arr) <= 1:
        return 0, 0.0, 0
    pairs = _original._collision_candidate_pairs(arr.shape[0], grid, all_pairs)
    if not pairs:
        return 0, 0.0, 0
    pair_idx = np.asarray(pairs, dtype=int)
    bmin = np.min(arr[:, :, :2], axis=1)
    bmax = np.max(arr[:, :, :2], axis=1)
    i = pair_idx[:, 0]
    j = pair_idx[:, 1]
    sep = np.maximum(np.maximum(bmin[j] - bmax[i], bmin[i] - bmax[j]), 0.0)
    overlap = np.minimum(bmax[i], bmax[j]) - np.maximum(bmin[i], bmin[j])
    colliding = np.all(overlap > 0.0, axis=1)
    clearances = np.linalg.norm(sep, axis=1)
    if np.any(colliding):
        clearances[colliding] = -np.min(overlap[colliding], axis=1)
    return int(np.count_nonzero(colliding)), float(np.min(clearances)) if clearances.size else 0.0, int(len(pair_idx))


def _count_2d_tile_collisions(tiles: np.ndarray, grid=None, all_pairs: bool = False) -> int:
    return int(_k2d_collision_and_clearance_metrics(tiles, grid, all_pairs)[0])


def _min_aabb_clearance_2d(tiles: np.ndarray, grid=None, all_pairs: bool = False) -> float:
    return float(_k2d_collision_and_clearance_metrics(tiles, grid, all_pairs)[1])


def _optimize_k2d(mesh_2d, mesh_3d, params, progress_callback=None):
    start = time.perf_counter()
    timings: dict[str, float] = {}
    _original._emit_progress(progress_callback, "Prepare K2D edge targets", 0.02, "K3D correspondence edge lengths")
    prepare_start = time.perf_counter()
    base_xy = mesh_2d.vertices[:, :2].copy()
    edges = _original._unique_mesh_edges(mesh_2d.faces)
    edge_idx = np.asarray(edges, dtype=int)
    target_lengths = (
        np.linalg.norm(mesh_3d.vertices[edge_idx[:, 0]] - mesh_3d.vertices[edge_idx[:, 1]], axis=1)
        if len(edge_idx)
        else np.zeros(0, dtype=float)
    )
    before_mean, before_max = _edge_matching_errors(base_xy, edge_idx, target_lengths)
    base_tiles = _tiles_from_mesh_vertices(mesh_2d.vertices, mesh_2d.faces)
    collisions_before, _min_clearance_before, collision_pair_count_before = _k2d_collision_and_clearance_metrics(base_tiles, mesh_2d.grid)
    timings["k2d_timer_prepare_edge_targets_sec"] = float(time.perf_counter() - prepare_start)

    def residual(xy_flat: np.ndarray) -> np.ndarray:
        xy = xy_flat.reshape(-1, 2)
        parts: list[np.ndarray] = []
        if len(edge_idx):
            current = np.linalg.norm(xy[edge_idx[:, 0]] - xy[edge_idx[:, 1]], axis=1)
        else:
            current = np.zeros(0, dtype=float)
        parts.append(np.sqrt(float(params.w_edge)) * (current - target_lengths))
        parts.append(np.sqrt(float(params.w_fab)) * (xy - base_xy).ravel())
        return np.concatenate([p.ravel() for p in parts])

    _original._emit_progress(progress_callback, "Fast K2D optimizer", 0.08, "Try CUDA/Adam path if available")
    torch_start = time.perf_counter()
    torch_result, torch_metrics = _original._optimize_k2d_torch(mesh_2d, mesh_3d, params, base_xy, edges, target_lengths)
    timings["k2d_timer_torch_optimizer_sec"] = float(time.perf_counter() - torch_start)
    large_grid_fast_path = (mesh_2d.grid.nx + 1) * (mesh_2d.grid.ny + 1) > 100 and torch_result is None
    optimizer_iterations = int(max(12, params.max_2d_iterations * 4))
    optimizer_converged = True
    actual_backend = "cuda" if torch_result is not None else "projective_numpy"
    timings["k2d_timer_scipy_least_squares_sec"] = 0.0
    timings["k2d_timer_projective_edge_match_sec"] = 0.0
    if torch_result is not None:
        xy = torch_result
        optimizer_iterations += int(max(40, params.max_2d_iterations * 6))
    elif _original.least_squares is not None and not large_grid_fast_path:
        scipy_start = time.perf_counter()
        opt = _original.least_squares(residual, base_xy.ravel(), max_nfev=max(5, params.max_2d_iterations), method="trf")
        timings["k2d_timer_scipy_least_squares_sec"] = float(time.perf_counter() - scipy_start)
        projective_start = time.perf_counter()
        xy = _original._projective_edge_match_2d(opt.x.reshape(-1, 2), base_xy, edges, target_lengths, mesh_2d.faces, mesh_2d.grid, iterations=optimizer_iterations)
        timings["k2d_timer_projective_edge_match_sec"] = float(time.perf_counter() - projective_start)
        optimizer_iterations = int(getattr(opt, "nfev", params.max_2d_iterations)) + optimizer_iterations
        optimizer_converged = bool(getattr(opt, "success", True))
        actual_backend = "scipy+projective_numpy"
    else:
        projective_start = time.perf_counter()
        xy = _original._projective_edge_match_2d(base_xy, base_xy, edges, target_lengths, mesh_2d.faces, mesh_2d.grid, iterations=optimizer_iterations)
        timings["k2d_timer_projective_edge_match_sec"] = float(time.perf_counter() - projective_start)

    k2d_collision_relax_skipped_for_speed = False
    collision_relax_start = time.perf_counter()
    if actual_backend != "cuda":
        if len(mesh_2d.faces) > 120:
            k2d_collision_relax_skipped_for_speed = True
        else:
            relaxed_xy = _original._relax_2d_collisions(xy, mesh_2d.faces, mesh_2d.grid, iterations=3, weight=0.08)
            relaxed_mean, relaxed_max = _edge_matching_errors(relaxed_xy, edge_idx, target_lengths)
            current_mean, current_max = _edge_matching_errors(xy, edge_idx, target_lengths)
            if relaxed_mean <= current_mean and relaxed_max <= current_max:
                xy = relaxed_xy
    timings["k2d_timer_collision_relax_sec"] = float(time.perf_counter() - collision_relax_start)

    strict_metrics: dict[str, float | int | str | bool] = {"strict_k2d_solver_used": False}
    pre_strict_start = time.perf_counter()
    pre_strict_mean, pre_strict_max = _edge_matching_errors(xy, edge_idx, target_lengths)
    mean_target_length = float(np.mean(target_lengths)) if len(target_lengths) else 1.0
    strict_threshold = max(1e-5, 0.002 * mean_target_length)
    timings["k2d_timer_pre_strict_error_check_sec"] = float(time.perf_counter() - pre_strict_start)
    timings["k2d_timer_strict_solve_sec"] = 0.0
    if getattr(params, "strict_paper_flow", False) and pre_strict_max > strict_threshold:
        strict_start = time.perf_counter()
        xy_strict, strict_metrics = _original._strict_k2d_edge_length_solve(
            base_xy,
            mesh_2d.faces,
            edges,
            target_lengths,
            params,
            progress_callback=_original._subprogress(progress_callback, 0.25, 0.92, "strict solve: "),
        )
        timings["k2d_timer_strict_solve_sec"] = float(time.perf_counter() - strict_start)
        strict_mean, strict_max = _edge_matching_errors(xy_strict, edge_idx, target_lengths)
        if strict_max <= pre_strict_max or strict_mean <= pre_strict_mean:
            xy = xy_strict
            actual_backend = "strict_edge_length_cuda" if strict_metrics.get("strict_k2d_projective_backend") == "cuda" else ("strict_edge_length_cpu" if actual_backend != "cuda" else "cuda+strict_edge_length_cpu")

    finalize_start = time.perf_counter()
    _original._emit_progress(progress_callback, "Finalize K2D metrics", 0.96, "Build flat vertices")
    finalize_vertices_start = time.perf_counter()
    vertices = np.column_stack([xy, np.zeros(len(xy))])
    z_abs_max = float(np.max(np.abs(vertices[:, 2]))) if len(vertices) else 0.0
    if z_abs_max > 1e-6:
        raise RuntimeError("K2D is not planar. K2D must be a 2D flat layout.")
    timings["k2d_timer_finalize_vertices_sec"] = float(time.perf_counter() - finalize_vertices_start)
    _original._emit_progress(progress_callback, "Finalize K2D metrics", 0.965, "Edge metrics")
    finalize_edge_start = time.perf_counter()
    after_mean, after_max = _edge_matching_errors(xy, edge_idx, target_lengths)
    timings["k2d_timer_finalize_edge_metrics_sec"] = float(time.perf_counter() - finalize_edge_start)
    _original._emit_progress(progress_callback, "Finalize K2D metrics", 0.972, "Collision/clearance metrics")
    finalize_collision_start = time.perf_counter()
    tiles = _tiles_from_mesh_vertices(vertices, mesh_2d.faces)
    collisions, min_clearance, collision_pair_count = _k2d_collision_and_clearance_metrics(tiles, mesh_2d.grid)
    timings["k2d_timer_finalize_collision_clearance_sec"] = float(time.perf_counter() - finalize_collision_start)
    _original._emit_progress(progress_callback, "Finalize K2D metrics", 0.982, "Gap angle metrics")
    finalize_gap_start = time.perf_counter()
    min_gap_angle, max_gap_angle, gap_angle_count = _gap_angle_range(vertices, mesh_2d.faces)
    timings["k2d_timer_finalize_gap_angles_sec"] = float(time.perf_counter() - finalize_gap_start)
    timings["k2d_finalize_gap_angle_count"] = int(gap_angle_count)
    _original._emit_progress(progress_callback, "Finalize K2D metrics", 0.988, "Displacement metrics")
    finalize_tail_start = time.perf_counter()
    displacement = np.linalg.norm(xy - base_xy, axis=1)
    displacement_rms = float(np.sqrt(np.mean(displacement * displacement))) if displacement.size else 0.0
    displacement_max = float(np.max(displacement)) if displacement.size else 0.0
    if "z_range_K3D" in mesh_3d.metrics:
        z_range_k3d = float(mesh_3d.metrics["z_range_K3D"])
    else:
        z_range_k3d = float(_original._z_range(mesh_3d.vertices))
    warning = ""
    if displacement_rms < 1e-5 and z_range_k3d > mesh_2d.grid.tile_size * 0.02:
        warning = "K2D is almost identical to M2D despite non-flat K3D. Edge matching may not be active."
    timings["k2d_timer_finalize_displacement_warning_sec"] = float(time.perf_counter() - finalize_tail_start)
    _original._emit_progress(progress_callback, "Finalize K2D metrics", 0.992, "Shared-edge count")
    finalize_gap_count_start = time.perf_counter()
    # For a manifold quad mesh, total edge incidences = boundary_edges + 2 * shared_edges,
    # while unique mesh edges = boundary_edges + shared_edges.  unique edges were already
    # computed for K2D edge matching, so this avoids rebuilding millions of HingeSpec objects.
    k2d_gap_count = max(0, int(4 * len(mesh_2d.faces) - len(edge_idx)))
    timings["k2d_timer_finalize_gap_count_sec"] = float(time.perf_counter() - finalize_gap_count_start)

    torch_module = _original.torch
    gpu_memory_peak = max(
        int(torch_module.cuda.max_memory_allocated(0)) if "cuda" in str(actual_backend) and torch_module is not None and torch_module.cuda.is_available() else 0,
        int(strict_metrics.get("strict_k2d_gpu_memory_peak", 0)),
    )
    timings["k2d_timer_finalize_total_sec"] = float(time.perf_counter() - finalize_start)
    timed_items = [(key, value) for key, value in timings.items() if key.endswith("_sec")]
    slowest_key, slowest_value = max(timed_items, key=lambda item: item[1]) if timed_items else ("", 0.0)
    timings["k2d_timer_total_measured_sec"] = float(sum(value for _key, value in timed_items))
    metrics = {
        "objective": "E_Flat = w1*EEdge + w2*ECollision + w3*EFab",
        "paper_weight_w1_edge": float(params.w_edge),
        "paper_weight_w2_collision": float(params.w_collision),
        "paper_weight_w3_fab": float(params.w_fab),
        "paper_default_weights_used": bool(abs(params.w_edge - 1.0) < 1e-9 and abs(params.w_collision - 1.0) < 1e-9 and abs(params.w_fab - 0.001) < 1e-12),
        "edge_matching_error": after_mean,
        "edge_matching_error_before": before_mean,
        "edge_matching_error_after": after_mean,
        "k2d_z_abs_max": z_abs_max,
        "k2d_edge_error_before": before_mean,
        "k2d_edge_error_after": after_mean,
        "mean_edge_length_error_before": before_mean,
        "mean_edge_length_error_after": after_mean,
        "max_edge_length_error_before": before_max,
        "max_edge_length_error_after": after_max,
        "k2d_displacement_rms": displacement_rms,
        "k2d_displacement_max": displacement_max,
        "k2d_xy_displacement_rms_from_M2D": displacement_rms,
        "k2d_xy_displacement_max_from_M2D": displacement_max,
        "collision_count_before": int(collisions_before),
        "collision_count_after": int(collisions),
        "2d_collision_count": int(collisions),
        "k2d_tile_overlap_count": int(collisions),
        "k2d_min_clearance": float(min_clearance),
        "k2d_gap_count": int(k2d_gap_count),
        "k2d_gap_count_method": "quad_edge_incidence_minus_precomputed_unique_edges",
        "min_gap_angle": float(min_gap_angle),
        "max_gap_angle": float(max_gap_angle),
        "fabrication_clearance_violation": float(np.mean(displacement)) if displacement.size else 0.0,
        "collision_projection": "deferred to Dual Hinge on medium/large grids; small-grid local AABB relax only if edge matching is preserved",
        "k2d_collision_relax_skipped_for_speed": bool(k2d_collision_relax_skipped_for_speed),
        "fast_path": bool(large_grid_fast_path),
        "optimizer_iterations": int(optimizer_iterations),
        "optimizer_converged": bool(optimizer_converged),
        "strict_edge_length_threshold": float(strict_threshold),
        "pre_strict_mean_edge_error": float(pre_strict_mean),
        "pre_strict_max_edge_error": float(pre_strict_max),
        **strict_metrics,
        "approximation_warning": warning,
        "actual_backend": actual_backend,
        "dominant_backend": actual_backend,
        "gpu_kernel_time": float(torch_metrics.get("gpu_kernel_time", 0.0)) + float(strict_metrics.get("strict_k2d_gpu_kernel_time", 0.0)),
        "cpu_preprocess_time": float(torch_metrics.get("cpu_preprocess_time", 0.0)),
        "cpu_postprocess_time": float(torch_metrics.get("cpu_postprocess_time", 0.0)),
        "cpu_gpu_transfer_count": int(torch_metrics.get("cpu_gpu_transfer_count", 0)) + int(strict_metrics.get("strict_k2d_cpu_gpu_transfer_count", 0)),
        "gpu_memory_peak": gpu_memory_peak,
        "k2d_finalize_metrics_acceleration": "reuse tile array; vectorized edge metrics; chunked gap angle min/max; collision count and min clearance in one candidate-pair pass",
        "k2d_finalize_elapsed_sec": float(time.perf_counter() - finalize_start),
        "k2d_finalize_collision_pair_count": int(collision_pair_count),
        "k2d_finalize_collision_pair_count_before": int(collision_pair_count_before),
        "k2d_timer_slowest_phase": str(slowest_key),
        "k2d_timer_slowest_phase_sec": float(slowest_value),
        "k2d_timer_note": "Inspect k2d_timer_*_sec in K2D metrics; largest value is the current bottleneck.",
        **timings,
    }
    out = _original.QuadMesh(vertices, mesh_2d.faces.copy(), mesh_2d.grid, "K2D", metrics, list(mesh_2d.split_lines))
    report = _original.StageReport(
        name="M2D -> K2D",
        objective=str(metrics["objective"]),
        before_error=before_mean,
        after_error=after_mean,
        constraint_violation=float(collisions),
        computation_time=time.perf_counter() - start,
        counts=_original._mesh_counts(out),
    )
    return out, report


def _paper_local_global_se2_layout(
    rest_xy: np.ndarray,
    hinge_constraints: list[tuple[int, int, int, int]],
    footprint_builder,
    initial_xy: np.ndarray | None,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    clearance: float,
    stage_name: str,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
    progress_callback=None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    rest = np.asarray(rest_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"paper_layout_optimizer": "empty"}
    current = np.asarray(initial_xy if initial_xy is not None else rest, dtype=float).copy()
    if current.shape != rest.shape:
        current = rest.copy()

    iterations = max(1, int(iterations))
    w_conn = max(0.0, float(connection_weight))
    w_coll = max(0.0, float(collision_weight))
    w_anchor = max(0.0, float(anchor_weight))
    clearance = max(float(clearance), 1e-6)
    time_budget_sec = max(0.0, float(time_budget_sec))
    max_candidate_pairs = max(50, int(max_candidate_pairs))
    collision_sweeps_per_iteration = max(1, int(collision_sweeps_per_iteration))
    solve_start_time = time.perf_counter()

    edge_lengths: list[float] = []
    for tile in rest:
        edge_lengths.extend(float(np.linalg.norm(tile[(i + 1) % tile.shape[0]] - tile[i])) for i in range(tile.shape[0]))
    tile_scale = max(float(np.median(edge_lengths)) if edge_lengths else 1.0, 1e-8)

    initial_expansion = max(1.0, float(initial_expansion))
    additive_expansion_offset = 0.0
    if initial_expansion > 1.000001 and len(current):
        centers0 = np.mean(current, axis=1)
        world_center0 = np.mean(centers0, axis=0)
        dirs = centers0 - world_center0
        norms = np.linalg.norm(dirs, axis=1)
        dirs = dirs / np.maximum(norms[:, None], 1e-12)
        additive_expansion_offset = min((initial_expansion - 1.0) * tile_scale, tile_scale * 9.0)
        current = current + additive_expansion_offset * dirs[:, None, :]

    anchor_layout = current.copy()
    anchor_centers = np.mean(anchor_layout, axis=1)
    max_step = max(clearance * 2.0, tile_scale * 0.08)
    max_center_drift = max(tile_scale * float(max_center_drift_tiles), clearance * 8.0)

    hinge_neighbors: list[set[int]] = [set() for _ in range(len(current))]
    for ia, _ca, ib, _cb in hinge_constraints:
        if 0 <= ia < len(current) and 0 <= ib < len(current):
            hinge_neighbors[int(ia)].add(int(ib))
            hinge_neighbors[int(ib)].add(int(ia))

    pair_cache: dict[str, object] = {"pairs": None, "iter": -10**9, "pad": None, "hit": 0, "miss": 0, "capped": False}
    active_tile_counts: list[int] = []
    active_pair_counts: list[int] = []
    coarse_iterations = 0
    fine_iterations = 0
    max_collision_count = 0
    last_pair_count = 0

    def _cap_pairs(fp: np.ndarray, pairs: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
        if len(pairs) <= limit:
            return pairs
        pair_cache["capped"] = True
        bmin = np.min(fp[:, :, :2], axis=1)
        bmax = np.max(fp[:, :, :2], axis=1)
        scores: list[tuple[float, int]] = []
        for k, (i, j) in enumerate(pairs):
            sep = np.maximum(np.maximum(bmin[j] - bmax[i], bmin[i] - bmax[j]), 0.0)
            scores.append((float(np.dot(sep, sep)), int(k)))
        scores.sort(key=lambda x: x[0])
        return [pairs[k] for _, k in scores[:limit]]

    def _phase(it: int) -> tuple[bool, float, int, int, int]:
        coarse = it < max(1, int(iterations * 0.45))
        if coarse:
            return True, clearance * 0.65, max(50, int(max_candidate_pairs * 0.45)), max(1, collision_sweeps_per_iteration - 1), 8
        return False, clearance, max_candidate_pairs, collision_sweeps_per_iteration, 4

    def _candidate_pairs(layout: np.ndarray, it: int, pad: float, limit: int, update_interval: int, force: bool = False):
        fp = np.asarray(footprint_builder(layout), dtype=float)
        cached_pairs = pair_cache.get("pairs")
        cache_iter = int(pair_cache.get("iter", -10**9))
        cache_pad = pair_cache.get("pad")
        valid = (
            cached_pairs is not None
            and cache_pad is not None
            and abs(float(cache_pad) - float(pad)) <= max(1e-12, pad * 1e-9)
            and (it - cache_iter) <= update_interval
        )
        if force or not valid:
            pairs = _spatial_candidate_pairs_for_tiles(fp, pad=pad)
            pairs = _cap_pairs(fp, pairs, limit)
            pair_cache.update({"pairs": pairs, "iter": int(it), "pad": float(pad), "miss": int(pair_cache["miss"]) + 1})
        else:
            pairs = list(cached_pairs)  # type: ignore[arg-type]
            pair_cache["hit"] = int(pair_cache["hit"]) + 1
        return fp, pairs

    def _connection_values(layout: np.ndarray) -> list[float]:
        vals: list[float] = []
        for ia, ca, ib, cb in hinge_constraints:
            if ia < len(layout) and ib < len(layout) and ca < layout.shape[1] and cb < layout.shape[1]:
                vals.append(float(np.linalg.norm(layout[ia, ca] - layout[ib, cb])))
        return vals

    def _active_sets(layout: np.ndarray, fp: np.ndarray, pairs: list[tuple[int, int]], phase_clearance: float):
        active: set[int] = set()
        conn_threshold = max(clearance * 0.25, tile_scale * 0.015)
        for ia, ca, ib, cb in hinge_constraints:
            if ia >= len(layout) or ib >= len(layout) or ca >= layout.shape[1] or cb >= layout.shape[1]:
                continue
            if float(np.linalg.norm(layout[ia, ca] - layout[ib, cb])) > conn_threshold:
                active.add(int(ia))
                active.add(int(ib))
        for i, j in pairs:
            overlap, _mtv, signed = _original._sat_polygon_mtv(fp[i], fp[j], clearance=phase_clearance)
            if overlap or signed < max(clearance, tile_scale * 0.03):
                active.add(int(i))
                active.add(int(j))
        if not active:
            active = set(range(len(layout)))
        grown = set(active)
        for tile_id in active:
            grown.update(hinge_neighbors[tile_id])
        active_hinges = [(ia, ca, ib, cb) for ia, ca, ib, cb in hinge_constraints if int(ia) in grown or int(ib) in grown]
        active_pairs = [(i, j) for i, j in pairs if int(i) in grown or int(j) in grown]
        return grown, active_hinges, active_pairs

    def _energy(layout: np.ndarray, it: int, phase_clearance: float, limit: int, update_interval: int, force_pairs: bool = False):
        fp, pairs = _candidate_pairs(layout, it, max(phase_clearance * 8.0, tile_scale * 0.08, 1e-4), limit, update_interval, force=force_pairs)
        penetration_sq = 0.0
        collision_count = 0
        min_clear = float("inf")
        for i, j in pairs:
            overlap, _, signed = _original._sat_polygon_mtv(fp[i], fp[j], clearance=clearance)
            min_clear = min(min_clear, float(signed))
            if overlap:
                collision_count += 1
                penetration_sq += float((-signed) ** 2)
        conn_vals = _connection_values(layout)
        conn_rms = float(np.sqrt(np.mean(np.square(conn_vals)))) if conn_vals else 0.0
        conn_max = float(max(conn_vals, default=0.0))
        centers = np.mean(layout, axis=1)
        center_drift = centers - anchor_centers
        anchor_rms = float(np.sqrt(np.mean(center_drift * center_drift))) if center_drift.size else 0.0
        e = w_conn * conn_rms * conn_rms + w_coll * (penetration_sq + collision_count * clearance * clearance * 25.0) + w_anchor * anchor_rms * anchor_rms
        return float(e), {
            "collision_count": int(collision_count),
            "min_clearance": float(min_clear if np.isfinite(min_clear) else 0.0),
            "hinge_rms": float(conn_rms),
            "hinge_max": float(conn_max),
            "anchor_rms": float(anchor_rms),
            "pair_count": int(len(pairs)),
        }

    def _clamp_step(base: np.ndarray, proposal: np.ndarray) -> np.ndarray:
        out = proposal.copy()
        base_centers = np.mean(base, axis=1)
        prop_centers = np.mean(out, axis=1)
        delta = prop_centers - base_centers
        norms = np.linalg.norm(delta, axis=1)
        active = norms > max_step
        if np.any(active):
            scale = (max_step / np.maximum(norms[active], 1e-12))[:, None]
            out[active] = base[active] + (out[active] - base[active]) * scale[:, None, :]
        centers = np.mean(out, axis=1)
        drift = centers - anchor_centers
        drift_norm = np.linalg.norm(drift, axis=1)
        active = drift_norm > max_center_drift
        if np.any(active):
            target_centers = anchor_centers[active] + drift[active] * (max_center_drift / np.maximum(drift_norm[active], 1e-12))[:, None]
            out[active] += (target_centers - centers[active])[:, None, :]
        return out

    _original._emit_progress(progress_callback, stage_name, 0.02, "Initialize cached active-set E_Hinge optimizer")
    _coarse, phase_clearance, phase_limit, _sweeps, update_interval = _phase(0)
    before_energy, before_stats = _energy(current, 0, phase_clearance, phase_limit, update_interval, force_pairs=True)
    before_collision_count = int(before_stats["collision_count"])
    before_clearance = float(before_stats["min_clearance"])
    before_conn = float(before_stats["hinge_rms"])
    best = current.copy()
    best_energy = before_energy
    current_energy = before_energy
    current_stats = dict(before_stats)
    timed_out = False
    executed_iterations = 0
    rejected_steps = 0
    consecutive_rejected_steps = 0
    accepted_steps = 0
    early_stop_reason = "completed_requested_iterations"
    max_consecutive_rejections = max(12, min(60, iterations // 8))

    progress_stride = max(1, iterations // 60)
    for it in range(iterations):
        coarse, phase_clearance, phase_limit, phase_sweeps, update_interval = _phase(it)
        coarse_iterations += int(coarse)
        fine_iterations += int(not coarse)
        if it % progress_stride == 0:
            _original._emit_progress(
                progress_callback,
                stage_name,
                min(0.98, (it + 1) / max(1, iterations)),
                f"{'coarse' if coarse else 'fine'} iter {it + 1}/{iterations}, collisions={int(current_stats.get('collision_count', 0))}, hinge_rms={float(current_stats.get('hinge_rms', 0.0)):.4g}",
            )
        if time_budget_sec > 0.0 and (time.perf_counter() - solve_start_time) >= time_budget_sec:
            timed_out = True
            break
        executed_iterations = it + 1

        desired_sum = np.zeros_like(current)
        desired_weight = np.zeros(current.shape[:2], dtype=float)
        desired_sum += current
        desired_weight += 1.0
        if w_anchor > 0.0:
            desired_sum += anchor_layout * w_anchor
            desired_weight += w_anchor

        fp, pairs = _candidate_pairs(current, it, max(phase_clearance * 8.0, tile_scale * 0.08, 1e-4), phase_limit, update_interval)
        last_pair_count = int(len(pairs))
        active_tiles, active_hinges, active_pairs = _active_sets(current, fp, pairs, phase_clearance)
        active_tile_counts.append(len(active_tiles))
        active_pair_counts.append(len(active_pairs))

        ramp = min(1.0, (it + 1) / max(1.0, iterations * 0.25))
        for ia, ca, ib, cb in active_hinges:
            if ia >= len(current) or ib >= len(current) or ca >= current.shape[1] or cb >= current.shape[1]:
                continue
            mid = 0.5 * (current[ia, ca] + current[ib, cb])
            w = w_conn * ramp
            desired_sum[ia, ca] += mid * w
            desired_weight[ia, ca] += w
            desired_sum[ib, cb] += mid * w
            desired_weight[ib, cb] += w

        if w_coll > 0.0:
            for _ in range(phase_sweeps):
                shifts = np.zeros((len(current), 2), dtype=float)
                counts = np.zeros((len(current), 1), dtype=float)
                active_count = 0
                for i, j in active_pairs:
                    overlap, mtv, _signed = _original._sat_polygon_mtv(fp[i], fp[j], clearance=phase_clearance)
                    if not overlap:
                        continue
                    active_count += 1
                    shifts[i] += 0.5 * mtv
                    shifts[j] -= 0.5 * mtv
                    counts[i, 0] += 1.0
                    counts[j, 0] += 1.0
                max_collision_count = max(max_collision_count, int(active_count))
                if active_count == 0:
                    break
                active = counts[:, 0] > 0.0
                shifts[active] /= np.maximum(counts[active], 1.0)
                w = min(1.5, 0.35 + 0.25 * w_coll) * ramp * (0.65 if coarse else 1.0)
                desired_sum[active] += (current[active] + shifts[active, None, :]) * w
                desired_weight[active] += w

        proposal = current.copy()
        for tile_id in active_tiles:
            weights = np.maximum(desired_weight[tile_id], 1e-12)
            targets = desired_sum[tile_id] / weights[:, None]
            proposal[tile_id] = _original._fit_rigid_2d_weighted(rest[tile_id], targets, weights)
        proposal = _clamp_step(current, proposal)
        proposal -= np.mean(np.mean(proposal, axis=1), axis=0) - np.mean(np.mean(anchor_layout, axis=1), axis=0)

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = current + alpha * (proposal - current)
            trial = _clamp_step(current, trial)
            trial_energy, trial_stats = _energy(trial, it, phase_clearance, phase_limit, update_interval)
            hinge_improves = float(trial_stats["hinge_rms"]) <= float(current_stats["hinge_rms"]) * 0.98
            collision_improves = int(trial_stats["collision_count"]) < int(current_stats["collision_count"])
            energy_improves = trial_energy <= current_energy * 1.02
            improves = energy_improves or collision_improves or hinge_improves
            if improves:
                current = trial
                current_energy = trial_energy
                current_stats = dict(trial_stats)
                accepted = True
                accepted_steps += 1
                consecutive_rejected_steps = 0
                if trial_energy < best_energy:
                    best = trial.copy()
                    best_energy = trial_energy
                break
        if not accepted:
            rejected_steps += 1
            consecutive_rejected_steps += 1
            pair_cache["iter"] = -10**9
            if consecutive_rejected_steps >= max_consecutive_rejections:
                early_stop_reason = "consecutive_rejected_steps"
                break

    current = best.copy()
    _coarse, phase_clearance, _phase_limit, _sweeps, update_interval = _phase(iterations)
    after_energy, after_stats = _energy(current, iterations + 1, phase_clearance, max_candidate_pairs, update_interval, force_pairs=True)
    after_collision_count = int(after_stats["collision_count"])
    after_clearance = float(after_stats["min_clearance"])
    after_conn_rms = float(after_stats["hinge_rms"])
    after_conn_max = float(after_stats["hinge_max"])
    last_pair_count = int(after_stats["pair_count"])

    shape_rms = _original._tile_shape_distance_error(np.dstack([current, np.zeros(current.shape[:2])]), np.dstack([rest, np.zeros(rest.shape[:2])]))
    shape_max = _original._tile_shape_distance_error(
        np.dstack([current, np.zeros(current.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
        use_max=True,
    )
    return current, {
        "paper_layout_optimizer": "cached active-set coarse-to-fine local/global E_Hinge SE(2) solver",
        "paper_layout_stage": str(stage_name),
        "paper_layout_energy": "E_Hinge = E_Rigid + E_Collision + E_Conn",
        "paper_layout_E_Rigid": "exact per-tile rigid SE(2) Procrustes projection",
        "paper_layout_E_Collision": "SAT full-footprint local projection with cached AABB broad-phase pairs",
        "paper_layout_E_Conn": "vertex-joint midpoint local projection",
        "paper_layout_iterations_requested": int(iterations),
        "paper_layout_iterations_executed": int(executed_iterations),
        "paper_layout_timed_out": bool(timed_out),
        "paper_layout_time_budget_sec": float(time_budget_sec),
        "paper_layout_elapsed_sec": float(time.perf_counter() - solve_start_time),
        "paper_layout_max_candidate_pairs": int(max_candidate_pairs),
        "paper_layout_candidate_pairs_capped": bool(pair_cache["capped"]),
        "paper_layout_collision_sweeps_per_iteration": int(collision_sweeps_per_iteration),
        "paper_layout_connection_weight": float(connection_weight),
        "paper_layout_collision_weight": float(collision_weight),
        "paper_layout_anchor_weight": float(anchor_weight),
        "paper_layout_anchor_reference": "expanded fabrication layout, not raw K2D shared mesh",
        "paper_layout_initial_expansion": float(initial_expansion),
        "paper_layout_expansion_mode": "bounded additive radial offset in tile-size units",
        "paper_layout_additive_expansion_offset": float(additive_expansion_offset),
        "paper_layout_global_space_expansion_enabled": bool(initial_expansion > 1.000001),
        "paper_layout_max_center_drift_tiles": float(max_center_drift_tiles),
        "paper_layout_clearance": float(clearance),
        "paper_layout_trust_region_max_step": float(max_step),
        "paper_layout_trust_region_max_center_drift": float(max_center_drift),
        "paper_layout_accepted_steps": int(accepted_steps),
        "paper_layout_rejected_steps": int(rejected_steps),
        "paper_layout_consecutive_rejected_steps_at_end": int(consecutive_rejected_steps),
        "paper_layout_max_consecutive_rejections": int(max_consecutive_rejections),
        "paper_layout_early_stop_reason": str("time_budget" if timed_out else early_stop_reason),
        "paper_layout_returned_best_state": True,
        "paper_layout_energy_before": float(before_energy),
        "paper_layout_energy_after": float(after_energy),
        "paper_layout_candidate_pair_count_last": int(last_pair_count),
        "paper_layout_collision_count_before": int(before_collision_count),
        "paper_layout_collision_count_after": int(after_collision_count),
        "paper_layout_collision_count_max_seen": int(max_collision_count),
        "paper_layout_min_clearance_before": float(before_clearance),
        "paper_layout_min_clearance_after": float(after_clearance),
        "paper_layout_hinge_rms_before": float(before_conn),
        "paper_layout_hinge_rms_after": float(after_conn_rms),
        "paper_layout_hinge_max_after": float(after_conn_max),
        "paper_layout_tile_shape_rms_error": float(shape_rms),
        "paper_layout_tile_shape_max_error": float(shape_max),
        "paper_layout_tile_shape_preserved": bool(shape_max < 1e-8),
        "paper_layout_hinge_pairs_exempt_from_collision": False,
        "paper_layout_old_soft_pushback_disabled": True,
        "paper_layout_scattered_layout_guard_enabled": True,
        "paper_layout_candidate_pair_cache_enabled": True,
        "paper_layout_candidate_pair_cache_hits": int(pair_cache["hit"]),
        "paper_layout_candidate_pair_cache_misses": int(pair_cache["miss"]),
        "paper_layout_candidate_pair_update_policy": "force on start/final/rejection, otherwise every 8 coarse or 4 fine iterations",
        "paper_layout_spatial_hash_model": "padded AABB occupancy cells with precise padded AABB broad-phase filter",
        "paper_layout_active_set_enabled": True,
        "paper_layout_active_tile_count_mean": float(np.mean(active_tile_counts)) if active_tile_counts else float(len(current)),
        "paper_layout_active_pair_count_mean": float(np.mean(active_pair_counts)) if active_pair_counts else float(last_pair_count),
        "paper_layout_coarse_to_fine_enabled": True,
        "paper_layout_coarse_iterations": int(coarse_iterations),
        "paper_layout_fine_iterations": int(fine_iterations),
    }


# Patch the original module in-place. Functions such as build_onestring_design()
# keep their original global namespace, so this assignment is what makes them call
# the new extrusion implementation.
_original.PipelineParameters = PipelineParameters
_original._extrude_tiles = _extrude_tiles
_original._build_surface_parameterization = _build_surface_parameterization
_original._flatten_to_domain = _flatten_to_domain
_original._build_m2d = _build_m2d
_original.inverse_map_uv_to_surface = inverse_map_uv_to_surface
_original._closest_points_on_surface_mesh = _closest_points_on_surface_mesh
_original._distances_to_surface_mesh = _distances_to_surface_mesh
_original._spatial_candidate_pairs_for_tiles = _spatial_candidate_pairs_for_tiles
_original._tiles_from_mesh_vertices = _tiles_from_mesh_vertices
_original._edge_matching_error = _edge_matching_error
_original._edge_matching_errors = _edge_matching_errors
_original._gap_angles = _gap_angles
_original._count_2d_tile_collisions = _count_2d_tile_collisions
_original._min_aabb_clearance_2d = _min_aabb_clearance_2d
_original._paper_local_global_se2_layout = _ORIGINAL_PAPER_LOCAL_GLOBAL_SE2_LAYOUT
_original._lift_m2d_to_m3d = _lift_m2d_to_m3d
_original._optimize_k2d = _optimize_k2d
_original._optimize_k3d = _optimize_k3d
_original._make_flat_tile_layout = _make_flat_tile_layout
_original._optimize_t2d_footprint_layout = _optimize_t2d_footprint_layout
_original._optimize_rigid_assembly_hinge_layout_2d = _optimize_rigid_assembly_hinge_layout_2d
_original._make_t2d_from_transforms = _make_t2d_from_transforms
_original._optimize_dual_hinges = _optimize_dual_hinges
_original._build_gap_graph = _build_gap_graph
_original._select_lift_points = _select_lift_points
_original._build_string_path = _build_string_path

# Re-export the original module's API from this wrapper.
for _name, _value in _original.__dict__.items():
    if _name in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "_extrude_tiles",
        "_build_surface_parameterization",
        "_flatten_to_domain",
        "_build_m2d",
        "inverse_map_uv_to_surface",
        "_closest_points_on_surface_mesh",
        "_distances_to_surface_mesh",
        "_lift_m2d_to_m3d",
        "_optimize_k3d",
        "_make_flat_tile_layout",
        "_build_gap_graph",
        "_select_lift_points",
        "_build_string_path",
        "_optimize_dual_hinges",
        "PipelineParameters",
    }:
        continue
    globals()[_name] = _value

globals()["PipelineParameters"] = PipelineParameters
globals()["_extrude_tiles"] = _extrude_tiles
globals()["_build_surface_parameterization"] = _build_surface_parameterization
globals()["_flatten_to_domain"] = _flatten_to_domain
globals()["_build_m2d"] = _build_m2d
globals()["inverse_map_uv_to_surface"] = inverse_map_uv_to_surface
globals()["_closest_points_on_surface_mesh"] = _closest_points_on_surface_mesh
globals()["_distances_to_surface_mesh"] = _distances_to_surface_mesh
globals()["_lift_m2d_to_m3d"] = _lift_m2d_to_m3d
globals()["_optimize_k3d"] = _optimize_k3d
globals()["_make_flat_tile_layout"] = _make_flat_tile_layout
globals()["_optimize_t2d_footprint_layout"] = _optimize_t2d_footprint_layout
globals()["_optimize_rigid_assembly_hinge_layout_2d"] = _optimize_rigid_assembly_hinge_layout_2d
globals()["_make_t2d_from_transforms"] = _make_t2d_from_transforms
globals()["_optimize_dual_hinges"] = _optimize_dual_hinges
globals()["_build_gap_graph"] = _build_gap_graph
globals()["_select_lift_points"] = _select_lift_points
globals()["_build_string_path"] = _build_string_path
globals()["export_t2d_stl"] = export_t2d_stl
globals()["SIDEFACE_CONTACT_PATCH_ACTIVE"] = True
globals()["SIDEFACE_CONTACT_PATCH_ORIGINAL_PATH"] = str(_ORIGINAL_PATH)



# ---------------------------------------------------------------------------
# Streamlit animation/simulation cache
# ---------------------------------------------------------------------------
_ORIGINAL_SIMULATE_ONESTRING_DEPLOYMENT = _original.simulate_onestring_deployment


def _deployment_params_cache_key(params) -> str:
    """Stable-ish key for deployment settings used by the Streamlit UI cache."""
    try:
        import dataclasses
        import json
        import hashlib

        if dataclasses.is_dataclass(params):
            payload = dataclasses.asdict(params)
        else:
            payload = dict(getattr(params, "__dict__", {}))
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    except Exception:
        return repr(params)


def _state_cache_key(state) -> str:
    try:
        import streamlit as st
        pipeline_key = st.session_state.get("pipeline_key")
        if pipeline_key is not None:
            return repr(pipeline_key)
    except Exception:
        pass
    try:
        v = np.asarray(state.tiles_2d_dual_hinge.vertices)
        t = np.asarray(state.tiles_3d.vertices)
        summary = (
            tuple(v.shape),
            tuple(t.shape),
            float(np.nanmean(v)) if v.size else 0.0,
            float(np.nanmean(t)) if t.size else 0.0,
            float(np.nanstd(v)) if v.size else 0.0,
            float(np.nanstd(t)) if t.size else 0.0,
        )
        return repr(summary)
    except Exception:
        return str(id(state))


def simulate_onestring_deployment(state, params=None, progress_callback=None):
    """Cached wrapper around the original deployment simulation.

    The app's Assembly Animation view can be rerun many times while the user only
    changes camera/player UI.  Keep a session-state cache of previously generated
    simulation frames so returning to the same settings reuses the animation
    instead of recomputing it.
    """
    cache_enabled = True
    cache = None
    key = None
    try:
        import streamlit as st
        cache = st.session_state.setdefault("onestring_animation_result_cache", {})
        key = ("deployment", _state_cache_key(state), _deployment_params_cache_key(params))
        if key in cache:
            if progress_callback is not None:
                try:
                    progress_callback("Cached deployment simulation", 1.0, "reusing previously generated animation frames")
                except Exception:
                    pass
            return cache[key]
    except Exception:
        cache_enabled = False

    result = _ORIGINAL_SIMULATE_ONESTRING_DEPLOYMENT(state, params, progress_callback=progress_callback)

    if cache_enabled and cache is not None and key is not None:
        try:
            cache[key] = result
            # Avoid unbounded growth while letting the user switch between a few
            # frame counts / solver settings during tuning.
            if len(cache) > 8:
                oldest_key = next(iter(cache.keys()))
                if oldest_key != key:
                    cache.pop(oldest_key, None)
        except Exception:
            pass
    return result


_original.simulate_onestring_deployment = simulate_onestring_deployment
globals()["simulate_onestring_deployment"] = simulate_onestring_deployment
globals()["ONESTRING_ANIMATION_CACHE_ACTIVE"] = True
