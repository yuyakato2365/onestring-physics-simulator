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
_ORIGINAL_OPTIMIZE_K3D = _original._optimize_k3d


@dataclass
class PipelineParameters(_original.PipelineParameters):
    omega_boundary_mode: Literal["rectangular_debug", "shape_preserving_experimental", "paper_default"] = "shape_preserving_experimental"
    omega_parameterization_mode: Literal[
        "pca_debug",
        "lscm_paper_like",
        "arap_paper_like",
        "paper_like_unimplemented",
    ] = "pca_debug"
    enable_heuristic_csf_split: bool = True
    enable_peak_guided_split: bool = True
    enable_mirror_split: bool = True


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

    boundary_mode = str(getattr(params, "omega_boundary_mode", "shape_preserving_experimental"))
    parameterization_mode = str(getattr(params, "omega_parameterization_mode", "pca_debug"))

    if parameterization_mode in {"lscm_paper_like", "arap_paper_like", "paper_like_unimplemented"} or boundary_mode == "paper_default":
        raise NotImplementedError(
            "Paper-default S->Omega parameterization is not implemented in this simulator. "
            "Use omega_parameterization_mode='pca_debug' with omega_boundary_mode='rectangular_debug' "
            "or 'shape_preserving_experimental' for explicit non-paper debug/experimental runs."
        )

    if parameterization_mode != "pca_debug":
        raise ValueError(f"unknown omega_parameterization_mode: {parameterization_mode}")

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

    if boundary_mode != "shape_preserving_experimental":
        raise ValueError(f"unknown omega_boundary_mode: {boundary_mode}")

    surface_vertices = np.asarray(surface.vertices, dtype=float)
    surface_faces = np.asarray(surface.faces[:, :3], dtype=int)
    boundary_loop = _original._mesh_boundary_loop(surface_faces)
    if len(boundary_loop) < 3:
        raise RuntimeError("Paper mode requires an open target mesh with a boundary; only debug heightfield mode is available for this surface.")

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
    mesh.metrics.update(_m2d_audit_metrics(mesh.vertices, mesh.faces))
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
            }
        )
        mesh.metrics.update(_m2d_audit_metrics(mesh.vertices, mesh.faces, suffix="after_split"))
        return mesh

    metrics = dict(mesh.metrics)
    component_sizes = _m2d_connected_component_sizes(faces)
    metrics.update(
        {
            "max_csf_before_split": float(getattr(domain, "csf_before", domain.max_csf)),
            "max_csf_after_split": float(getattr(domain, "csf_after_split", domain.max_csf)),
            "number_of_splits": len(snapped_lines),
            "split_locations": snapped_lines,
            "csf_split_applied": True,
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
            "m3d_surface_projection_model": "analytic_heightfield_debug" if used_shortcut else "uv_triangle_lookup_barycentric",
            "m3d_exactness_label": "debug" if used_shortcut else "approximation",
            "m3d_parameterization_warning": (
                "Height-field shortcut is not paper inverse parameterization."
                if used_shortcut
                else "Barycentric inverse map depends on the current non-paper Omega parameterization."
            ),
        }
    )
    return out, report


def _make_flat_tile_layout(mesh, params=None):
    layout = _ORIGINAL_MAKE_FLAT_TILE_LAYOUT(mesh, params)
    min_clearance = float(layout.metrics.get("min_clearance", 0.0))
    if min_clearance >= -1e-9 or params is None:
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

    retry = _ORIGINAL_MAKE_FLAT_TILE_LAYOUT(mesh, retry_params)
    retry_clearance = float(retry.metrics.get("min_clearance", min_clearance))
    if retry_clearance > min_clearance:
        before_retry_clearance = min_clearance
        layout = retry
        min_clearance = retry_clearance
        layout.metrics["k2d_layout_retry_for_peak_aligned_grid"] = True
        layout.metrics["k2d_layout_min_clearance_before_retry"] = float(before_retry_clearance)

    if min_clearance >= -1e-9:
        return layout

    separated, separation_metrics = _separate_independent_flat_tiles(layout.tile_top_vertices_2d, mesh.grid)
    separated_layout = _rebuild_flat_tile_layout_with_vertices(layout, separated, mesh)
    separated_layout.metrics.update(separation_metrics)
    if float(separated_layout.metrics.get("min_clearance", min_clearance)) > min_clearance:
        return separated_layout
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


def _optimize_k3d(target, mesh, parameterization, params):
    out, report = _ORIGINAL_OPTIMIZE_K3D(target, mesh, parameterization, params)
    quality = _k3d_quality_metrics(mesh.vertices, out.vertices, mesh.faces)
    reject_reason = _k3d_quality_reject_reason(quality, mesh.grid)
    if reject_reason is None:
        out.metrics.update(quality)
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
        return out, False
    except Exception:
        return fallback, True


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

    normals = np.asarray([_original._quad_normal(top) for top in top_tiles], dtype=float)
    raw_side_normals: list[list[np.ndarray]] = []
    for tile_id, top in enumerate(top_tiles):
        raw_side_normals.append([_edge_inward_normal(top, normals[tile_id], edge) for edge in local_edges])

    side_normals: list[list[np.ndarray]] = [[raw_side_normals[i][e].copy() for e in range(4)] for i in range(tile_count)]
    incidence = _build_edge_incidence(mesh.faces)
    internal_miter_edge_count = 0
    boundary_side_plane_count = 0
    nonmanifold_edge_count = 0

    for entries in incidence.values():
        if len(entries) == 1:
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
            "normal_translation_center_shift_error_rms": float(np.sqrt(np.mean(normal_shift_error * normal_shift_error))) if normal_shift_error.size else 0.0,
            "internal_miter_edge_count": int(internal_miter_edge_count),
            "boundary_side_plane_count": int(boundary_side_plane_count),
            "nonmanifold_edge_count": int(nonmanifold_edge_count),
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
        "connection_weight": float(max(40.0, float(connection_weight) * 12.0)),
        "collision_weight": float(max(3.0, float(collision_weight) * 3.0)),
        # Keep the initial pose as a weak prior, not as a cage.  The old values
        # were too anchor-heavy for mitered solids and could collapse the holes.
        "anchor_weight": float(max(0.003, min(0.025, float(anchor_weight) * 0.25))),
        "initial_expansion": float(max(1.22, float(initial_expansion))),
        "max_center_drift_tiles": float(max(4.0, float(max_center_drift_tiles))),
        "layout_gap": float(layout_gap),
        "clearance": float(max(layout_gap * 0.65, tile_size * 0.035)),
    }


def _layout_quality_for_top_xy(layout: np.ndarray, transforms: np.ndarray, faces: np.ndarray, grid, constraints) -> dict[str, float | int]:
    layout = np.asarray(layout, dtype=float)
    if layout.size == 0:
        return {"hinge_error": 0.0, "collision_count": 0, "min_clearance": 0.0}
    footprints = _original._apply_t2d_transforms_to_top_xy(layout, transforms)[:, :, :2]
    pad = max(float(getattr(grid, "gap_size", 0.08)) * 8.0, float(getattr(grid, "tile_size", 1.0)) * 0.25)
    pairs = _original._spatial_candidate_pairs_for_tiles(footprints, pad=pad)
    specs = _original._vertex_hinge_specs_from_faces(faces)
    return {
        "hinge_error": float(_original._vertex_layout_hinge_error(layout, specs)),
        "collision_count": int(_original._count_2d_footprint_collisions_from_pairs(footprints, pairs)),
        "min_clearance": float(_original._min_footprint_clearance_2d_from_pairs(footprints, pairs)),
    }


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
        return _original._apply_t2d_transforms_to_top_xy(layout, transforms)[:, :, :2]

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
        connection_weight=max(120.0, float(free["connection_weight"]) * 2.0),
        collision_weight=max(2.0, float(free["collision_weight"]) * 0.75),
        anchor_weight=max(0.002, float(free["anchor_weight"]) * 0.5),
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
        "t2d_footprint_collision_checked_on": "top+bottom projected footprint with SAT, enlarged optimization-only clearance",
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
        connection_weight=max(60.0, float(free["connection_weight"])),
        collision_weight=max(3.5, float(free["collision_weight"])),
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

    base_assembly, base_report = _ORIGINAL_MAKE_T2D_FROM_TRANSFORMS(
        mesh_2d,
        flat_layout,
        mesh_3d,
        tiles_3d,
        stage,
        params,
    )
    if len(base_assembly.vertices) == 0:
        return base_assembly, base_report

    placed_vertices = np.zeros_like(base_assembly.vertices)
    rigid_transforms = np.zeros((len(base_assembly.vertices), 4, 4), dtype=float)
    top_errors = []
    flat_layout_tops = np.asarray(flat_layout.tile_top_vertices_3d, dtype=float)
    for tile_id in range(len(base_assembly.vertices)):
        flat_top = flat_layout_tops[tile_id] if tile_id < len(flat_layout_tops) else base_assembly.vertices[tile_id, :4]
        placed, transform = _original._rigidly_place_t3d_tile_in_flat_layout(
            tiles_3d.vertices[tile_id],
            flat_top,
        )
        # T2D top hinges are defined on the solved K2D flat layout.  Keep those
        # four top vertices exact; only the bottom/side vertices inherit the
        # mitered T3D rigid placement.
        placed[:4] = flat_top
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


def _t2d_prism_vertices(assembly, panel_size: float, thickness: float) -> np.ndarray:
    scale = _t2d_export_scale(assembly, panel_size)
    top_xy = np.asarray(assembly.vertices, dtype=float)[:, :4, :2] * scale
    out = np.zeros((top_xy.shape[0], 8, 3), dtype=float)
    out[:, :4, :2] = top_xy
    out[:, 4:, :2] = top_xy
    out[:, :4, 2] = 0.5 * float(thickness)
    out[:, 4:, 2] = -0.5 * float(thickness)
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
    thickness = float(panel_size * 0.05 if thickness is None else thickness)
    vertices = _t2d_prism_vertices(assembly, panel_size, thickness)
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
        "t2d_export_thickness": float(thickness),
        "t2d_export_panel_size": float(panel_size),
        "t2d_min_area": float(np.min(areas)) if len(areas) else 0.0,
        "t2d_max_aspect_ratio": float(np.max(aspects)) if len(aspects) else 0.0,
        "t2d_connected_component_count": int(vertices.shape[0]),
        "t2d_nonmanifold_edge_count": _triangle_edge_nonmanifold_count(tri_indices),
        "t2d_export_model": "flat_2d_layout_extruded_as_thin_plate_stl",
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


# Patch the original module in-place. Functions such as build_onestring_design()
# keep their original global namespace, so this assignment is what makes them call
# the new extrusion implementation.
_original.PipelineParameters = PipelineParameters
_original._extrude_tiles = _extrude_tiles
_original._build_surface_parameterization = _build_surface_parameterization
_original._flatten_to_domain = _flatten_to_domain
_original._build_m2d = _build_m2d
_original._lift_m2d_to_m3d = _lift_m2d_to_m3d
_original._optimize_k3d = _optimize_k3d
_original._make_flat_tile_layout = _make_flat_tile_layout
_original._optimize_t2d_footprint_layout = _optimize_t2d_footprint_layout
_original._optimize_rigid_assembly_hinge_layout_2d = _optimize_rigid_assembly_hinge_layout_2d
_original._make_t2d_from_transforms = _make_t2d_from_transforms

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
        "_lift_m2d_to_m3d",
        "_optimize_k3d",
        "_make_flat_tile_layout",
        "PipelineParameters",
    }:
        continue
    globals()[_name] = _value

globals()["PipelineParameters"] = PipelineParameters
globals()["_extrude_tiles"] = _extrude_tiles
globals()["_build_surface_parameterization"] = _build_surface_parameterization
globals()["_flatten_to_domain"] = _flatten_to_domain
globals()["_build_m2d"] = _build_m2d
globals()["_lift_m2d_to_m3d"] = _lift_m2d_to_m3d
globals()["_optimize_k3d"] = _optimize_k3d
globals()["_make_flat_tile_layout"] = _make_flat_tile_layout
globals()["_optimize_t2d_footprint_layout"] = _optimize_t2d_footprint_layout
globals()["_optimize_rigid_assembly_hinge_layout_2d"] = _optimize_rigid_assembly_hinge_layout_2d
globals()["_make_t2d_from_transforms"] = _make_t2d_from_transforms
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
