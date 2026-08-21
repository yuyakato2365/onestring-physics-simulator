"""Positive-area UV triangle overlap audit for native Grid-OptCuts.

Native Grid-OptCuts V2 cannot use the authors' air-mesh scaffold because the two
copies of a fabrication seam are intentionally coincident.  Local orientation
checks therefore are not enough: two distant UV charts could overlap without
flipping any triangle.  This module performs one global broad-phase + exact
convex clipping audit after OptCuts finishes.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def _signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    )


def _line_intersection(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of segment p->q with infinite line a->b; caller brackets it."""
    r = q - p
    s = b - a
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) <= 1e-18:
        return 0.5 * (p + q)
    ap = a - p
    t = float((ap[0] * s[1] - ap[1] * s[0]) / denom)
    return p + np.clip(t, 0.0, 1.0) * r


def _clip_convex(subject: np.ndarray, clip_triangle: np.ndarray, tol: float) -> np.ndarray:
    output = [np.asarray(p, dtype=float) for p in np.asarray(subject, dtype=float)]
    clip = np.asarray(clip_triangle, dtype=float)
    orient = 1.0 if _signed_area(clip) >= 0.0 else -1.0

    for i in range(3):
        if not output:
            break
        a = clip[i]
        b = clip[(i + 1) % 3]

        def inside(x: np.ndarray) -> bool:
            cross = float((b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0]))
            return orient * cross >= -tol

        inp = output
        output = []
        prev = inp[-1]
        prev_inside = inside(prev)
        for cur in inp:
            cur_inside = inside(cur)
            if cur_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, cur, a, b))
                output.append(cur)
            elif prev_inside:
                output.append(_line_intersection(prev, cur, a, b))
            prev, prev_inside = cur, cur_inside
    return np.asarray(output, dtype=float) if output else np.zeros((0, 2), dtype=float)


def _triangle_overlap_area(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    clipped = _clip_convex(np.asarray(a, dtype=float), np.asarray(b, dtype=float), tol)
    return abs(_signed_area(clipped)) if len(clipped) >= 3 else 0.0


def positive_area_uv_overlaps(
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    *,
    max_examples: int = 20,
) -> tuple[int, list[tuple[int, int, float]]]:
    """Return number/examples of triangle pairs with positive-area UV overlap.

    Shared edges and coincident seam boundaries have zero area and are allowed.
    """
    uv = np.asarray(uv_vertices, dtype=float)
    faces = np.asarray(uv_faces, dtype=int)
    if len(faces) < 2:
        return 0, []
    tris = uv[faces]
    lo = np.min(tris, axis=1)
    hi = np.max(tris, axis=1)
    span = np.maximum(np.max(hi, axis=0) - np.min(lo, axis=0), 1e-12)
    total_area_scale = float(span[0] * span[1])
    area_tol = max(1e-14, 1e-12 * total_area_scale)
    geom_tol = max(1e-12, 1e-10 * float(np.max(span)))

    # Uniform-grid broad phase. Roughly sqrt(n) cells per axis keeps normal UV
    # meshes close to linear memory while still handling long triangles.
    bins_per_axis = max(1, min(256, int(math.ceil(math.sqrt(len(faces))))))
    global_lo = np.min(lo, axis=0)
    cell = span / bins_per_axis
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(len(faces)):
        a = np.floor((lo[fi] - global_lo) / cell).astype(int)
        b = np.floor((hi[fi] - global_lo) / cell).astype(int)
        a = np.clip(a, 0, bins_per_axis - 1)
        b = np.clip(b, 0, bins_per_axis - 1)
        for ix in range(int(a[0]), int(b[0]) + 1):
            for iy in range(int(a[1]), int(b[1]) + 1):
                buckets[(ix, iy)].append(fi)

    candidate_pairs: set[tuple[int, int]] = set()
    for ids in buckets.values():
        for pos, i in enumerate(ids):
            for j in ids[pos + 1 :]:
                if i == j:
                    continue
                candidate_pairs.add((i, j) if i < j else (j, i))

    count = 0
    examples: list[tuple[int, int, float]] = []
    for i, j in candidate_pairs:
        overlap_lo = np.maximum(lo[i], lo[j])
        overlap_hi = np.minimum(hi[i], hi[j])
        if np.any(overlap_hi - overlap_lo <= geom_tol):
            continue
        # Adjacent UV triangles that share a full indexed edge cannot have
        # positive overlap in a valid orientation; skipping them removes a large
        # amount of unnecessary clipping work.
        if len(set(map(int, faces[i])) & set(map(int, faces[j]))) >= 2:
            continue
        area = _triangle_overlap_area(tris[i], tris[j], geom_tol)
        if area > area_tol:
            count += 1
            if len(examples) < max_examples:
                examples.append((int(i), int(j), float(area)))
    return count, examples


__all__ = ["positive_area_uv_overlaps"]
