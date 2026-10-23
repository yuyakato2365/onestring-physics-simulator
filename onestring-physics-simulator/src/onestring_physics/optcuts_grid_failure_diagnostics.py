"""Failure-only, non-invasive evidence for experimental Grid-boundary UV fitting.

This module is deliberately separate from the optimization objective.  It
writes source/accepted/rejected/harmonic UV, original target grid, triangle
area ratios, local 270-degree corners and proper boundary segment crossings.
Interpretations are hypotheses, NOT proof of a topology obstruction.

Files are saved under logs/grid_boundary_diagnostics (or a user-specified
folder). Never alter UV, topology, exact 50% M2D mask or acceptance guards.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import re
import uuid

import numpy as np


def _signed_area(pts):
    xy = np.asarray(pts, dtype=float)
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                           - xy[:, 1] * np.roll(xy[:, 0], -1)))


def reflex_corner_indices(outline):
    """Interior angle > 180 degrees; independent of contour winding."""
    pts = np.asarray(outline, dtype=float)
    if len(pts) < 3:
        return np.empty(0, dtype=int)
    before = pts - np.roll(pts, 1, axis=0)
    after = np.roll(pts, -1, axis=0) - pts
    turn = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    winding = np.sign(_signed_area(pts))
    return np.flatnonzero(turn * winding < -1e-12)


def proper_boundary_crossings(points, limit=12):
    """Count non-adjacent proper edge crossings of a *given* UV boundary.

    Endpoints touching and collinear overlaps are not counted; those need
    separate exact-polygon analysis. This is a diagnostic, not injectivity
    verification. Vectorized O(B^2), invoked ONLY on saved failure cases.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4 or not np.isfinite(pts).all():
        return {"count": None, "examples": [], "reason": "nonfinite_or_too_short"}
    n = len(pts)
    nxt = np.roll(pts, -1, axis=0)
    edges = nxt - pts
    extent = max(float(np.ptp(pts, axis=0).max()), 1e-12)
    epsilon = 1e-13 * extent * extent
    count = 0
    examples = []
    for i in range(n):
        j = np.arange(i + 2, n, dtype=int)
        if i == 0:
            j = j[j != n - 1]
        if j.size == 0:
            continue
        ab = edges[i]
        cd = edges[j]
        ac = pts[j] - pts[i]
        ad = nxt[j] - pts[i]
        ca = pts[i] - pts[j]
        cb = nxt[i] - pts[j]
        lhs = ab[0] * ac[:, 1] - ab[1] * ac[:, 0]
        rhs = ab[0] * ad[:, 1] - ab[1] * ad[:, 0]
        other1 = cd[:, 0] * ca[:, 1] - cd[:, 1] * ca[:, 0]
        other2 = cd[:, 0] * cb[:, 1] - cd[:, 1] * cb[:, 0]
        crossings = ((lhs * rhs < 0) & (other1 * other2 < 0)
                     & (np.abs(lhs) > epsilon)
                     & (np.abs(rhs) > epsilon)
                     & (np.abs(other1) > epsilon)
                     & (np.abs(other2) > epsilon))
        matches = j[crossings]
        count += len(matches)
        for k in matches[:max(0, limit-len(examples))]:
            examples.append([int(i), int(k)])
    return {"count": int(count), "examples": examples,
            "note": "proper nonadjacent crossings only; not a full overlap audit"}


def analyze_failure(base, faces, boundary_indices, full_mapped_boundary,
                    last_accepted, last_rejected, harmonic_rejected,
                    grid_outline, step, fraction, last_valid_fraction):
    """Reproducible geometric diagnostics at a failed continuation step."""
    from .optcuts_grid_boundary_untangle import oriented_area_ratio

    base = np.asarray(base, dtype=float)
    faces = np.asarray(faces, dtype=int)
    bound = np.asarray(boundary_indices, dtype=int)
    mapped = np.asarray(full_mapped_boundary, dtype=float)
    accepted = np.asarray(last_accepted, dtype=float)
    rejected = np.asarray(last_rejected, dtype=float)
    harmonic = np.asarray(harmonic_rejected, dtype=float)
    outline = np.asarray(grid_outline, dtype=float)
    tile_step = np.asarray(step, dtype=float)
    if mapped.shape != base[bound].shape or rejected.shape != base.shape:
        raise ValueError("GRID_DIAGNOSTIC_ARRAY_SHAPES_INVALID")
    ratio = oriented_area_ratio(base, rejected, faces)
    accepted_ratio = oriented_area_ratio(base, accepted, faces)
    harmonic_ratio = oriented_area_ratio(base, harmonic, faces)
    centroids = np.mean(rejected[faces], axis=1)
    critical = np.argsort(ratio, kind="stable")[:min(len(faces), 100)]
    reflex_idx = reflex_corner_indices(outline)
    reflex = outline[reflex_idx]
    cell_unit = max(float(np.max(tile_step)), 1e-12)

    if len(reflex):
        # Chunk to keep memory bounded for large user-provided meshes.
        distances = np.empty(len(faces), dtype=float)
        for start in range(0, len(faces), 2048):
            local = centroids[start:start + 2048]
            distances[start:start + len(local)] = np.min(
                np.linalg.norm(local[:, None, :] - reflex[None, :, :],
                               axis=2), axis=1
            )
    else:
        distances = np.full(len(faces), float("inf"))
    threshold = 2.0 * cell_unit
    worst_size = min(len(faces), max(10, int(math.ceil(0.01 * len(faces)))))
    worst = np.argsort(ratio, kind="stable")[:worst_size]
    fraction_worst_near = float(np.mean(distances[worst] <= threshold))
    fraction_all_near = float(np.mean(distances <= threshold))
    enrichment = fraction_worst_near / max(fraction_all_near, 1e-12)

    target_at_failure = base[bound] + float(fraction) * (mapped-base[bound])
    last_valid_boundary = accepted[bound]
    intersections = {
        "target_at_failed_fraction": proper_boundary_crossings(target_at_failure),
        "last_accepted": proper_boundary_crossings(last_valid_boundary),
        "intended_full_boundary": proper_boundary_crossings(mapped),
    }
    worst_samples = []
    for face_id in critical[:40]:
        worst_samples.append({
            "triangle_id": int(face_id),
            "uv_face_vertex_ids": faces[face_id].tolist(),
            "signed_area_ratio": float(ratio[face_id]),
            "rejected_uv_centroid": centroids[face_id].tolist(),
            "original_uv_centroid": np.mean(base[faces[face_id]],
                                            axis=0).tolist(),
            "nearest_reflex_distance_cells": (
                float(distances[face_id] / cell_unit)
                if np.isfinite(distances[face_id]) else None
            ),
            "within_two_grid_cells_of_reflex": bool(
                distances[face_id] <= threshold),
        })
    if intersections["target_at_failed_fraction"]["count"]:
        interpretation = "intermediate_boundary_crossing_candidate"
        advice = (
            "Straight-line boundary homotopy crosses itself at the failed "
            "fraction. Test a non-linear boundary path or another vertex "
            "correspondence before modifying source seam topology."
        )
    elif len(reflex) and fraction_worst_near >= 0.5 and enrichment >= 2:
        interpretation = "reflex_corner_localized_compression_candidate"
        advice = (
            "Worst UV triangles cluster near 270-degree grid corners. "
            "Investigate corner-specific boundary assignment, curved chart "
            "envelope, or seam cuts near localized high-strain corners."
        )
    elif fraction_worst_near < 0.2:
        interpretation = "distributed_or_non_corner_compression_candidate"
        advice = (
            "Worst UV triangles are mostly away from reflex corners. "
            "Inspect harmonic initialization, area barrier, UV correspondence "
            "and global chart distortion before assuming a corner cut helps."
        )
    else:
        interpretation = "mixed_or_undetermined_compression"
        advice = (
            "Inspect the saved hotspot SVG, correspondence and source"
            " geometry; no single cause follows from the observed ratios."
        )

    result = {
        "schema": "onestring-grid-failure-diagnostic-v1",
        "boundary_fraction_rejected": float(fraction),
        "boundary_fraction_last_accepted": float(last_valid_fraction),
        "source_vertex_count": int(len(base)),
        "triangle_count": int(len(faces)),
        "boundary_vertex_count": int(len(bound)),
        "outline_corner_count": int(len(outline)),
        "reflex_corner_count": int(len(reflex)),
        "grid_step": tile_step.tolist(),
        "accepted_min_signed_area_ratio": float(np.min(accepted_ratio)),
        "rejected_min_signed_area_ratio": float(np.min(ratio)),
        "rejected_p01_signed_area_ratio": float(np.percentile(ratio, 1)),
        "rejected_p05_signed_area_ratio": float(np.percentile(ratio, 5)),
        "harmonic_rejected_min_signed_area_ratio": float(np.min(harmonic_ratio)),
        "rejected_flipped_triangle_count": int(np.count_nonzero(ratio <= 0)),
        "rejected_below_existing_0_005_area_guard_count": int(
            np.count_nonzero(ratio < .005)),
        "rejected_below_0_015_barrier_target_count": int(
            np.count_nonzero(ratio < .015)),
        "worst_one_percent_near_reflex_within_two_tiles_fraction":
            fraction_worst_near,
        "all_triangles_near_reflex_within_two_tiles_fraction":
            fraction_all_near,
        "reflex_hotspot_enrichment": float(enrichment),
        "proper_boundary_intersections": intersections,
        "ranked_worst_triangles": worst_samples,
        "diagnostic_hypothesis": interpretation,
        "suggested_investigation": advice,
        "limitations": (
            "A rejected step does NOT prove impossibility, nor identify a "
            "unique cause. Positive signed triangle areas alone cannot "
            "rule out nonlocal overlaps. Proper segment crossings ignore "
            "collinear/touching boundary contacts."
        ),
    }
    arrays = {
        "base_uv": base, "uv_faces": faces, "boundary_indices": bound,
        "target_full_boundary_uv": mapped,
        "accepted_uv": accepted, "rejected_uv": rejected,
        "rejected_harmonic_uv": harmonic,
        "grid_outline": outline, "grid_step": tile_step,
        "rejected_area_ratio": ratio,
        "rejected_triangle_reflex_distance": distances,
        "reflex_corner_indices": reflex_idx,
        "worst_triangle_ids": critical,
    }
    return result, arrays


def _svg_polyline(points, transform, stroke, width=1.4, dash=None, opacity=1):
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return ""
    coords = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in (transform(p) for p in pts)
    )
    stroke_dash = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{coords}" fill="none" '
            f'stroke="{stroke}" stroke-width="{width}" '
            f'opacity="{opacity}"{stroke_dash}/>')


def _svg_failure_view(arrays, analysis, *, focus=False):
    """Standalone SVG: topology, 270-degree corners, failed UV hotspots."""
    base = arrays["base_uv"]
    faces = arrays["uv_faces"]
    rejected = arrays["rejected_uv"]
    accepted = arrays["accepted_uv"]
    loop = arrays["boundary_indices"]
    grid = arrays["grid_outline"]
    reflex = grid[arrays["reflex_corner_indices"]]
    bad = arrays["rejected_area_ratio"]
    hot_ids = arrays["worst_triangle_ids"]
    if focus:
        center = np.mean(rejected[faces[int(hot_ids[0])]], axis=0)
        radius = max(3.5 * float(np.max(arrays["grid_step"])), 1e-8)
        lo, hi = center-radius, center+radius
        subject = "Worst triangle hotspot (zoom)"
    else:
        shown = np.vstack([base[loop], accepted[loop], rejected[loop], grid])
        lo = np.min(shown, axis=0)
        hi = np.max(shown, axis=0)
        subject = "Grid boundary continuation failure (overview)"
    span = np.maximum(hi-lo, 1e-9)
    padding = .06 * max(span)
    lo = lo-padding
    hi = hi+padding
    width, height = 960, 760
    def tr(p):
        q = (np.asarray(p, float)-lo)/np.maximum(hi-lo, 1e-12)
        return (50+860*float(q[0]), 685-610*float(q[1]))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="40" y="32" font-size="21" font-family="sans-serif">'
        f'{subject}</text>',
        '<text x="40" y="52" font-size="13" font-family="sans-serif">'
        'Gray: original outline; blue dashed: Grid target; '
        'green: last valid; red: rejected; orange: 270-degree corners'
        '</text>',
    ]
    # Source and target outlines are distinct from the actual cut graph.
    parts.append(_svg_polyline(np.vstack([base[loop], base[loop[0]]]),
                               tr, "#a3a3a3", 1., opacity=.75))
    parts.append(_svg_polyline(np.vstack([grid, grid[0]]]),
                               tr, "#3b82f6", 1.8, dash="6,4"))
    # Color only the 100 most compressed triangles (not the entire 16k mesh).
    for fid in hot_ids:
        tri = rejected[faces[int(fid)]]
        if focus and not np.any(np.all((tri>=lo)&(tri<=hi), axis=1)):
            continue
        coords = " ".join(f"{x:.2f},{y:.2f}"
                          for x, y in (tr(pt) for pt in tri))
        colour = "#c1121f" if bad[fid] < .005 else "#ff9f1c"
        parts.append(f'<polygon points="{coords}" fill="{colour}" '
                     f'fill-opacity=".55" stroke="{colour}" '
                     f'stroke-width=".35"/>')
    parts.append(_svg_polyline(
        np.vstack([accepted[loop], accepted[loop[0]]]),
        tr, "#15803d", 1.4, opacity=.85))
    parts.append(_svg_polyline(
        np.vstack([rejected[loop], rejected[loop[0]]]),
        tr, "#b91c1c", 1.7, opacity=.8))
    for pt in reflex:
        x, y = tr(pt)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" '
                     'fill="#f97316"/>')
    tri_id = int(hot_ids[0])
    x, y = tr(np.mean(rejected[faces[tri_id]], axis=0))
    parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" '
                 'fill="none" stroke="#991b1b" stroke-width="2"/>')
    parts.append(
        '<text x="42" y="730" font-family="monospace" font-size="13">'
        f'worst triangle #{tri_id}; signed ratio {bad[tri_id]:.5g}; '
        f'failed fraction {analysis["boundary_fraction_rejected"]:.5f}'
        '</text></svg>'
    )
    return "\n".join(parts)


def save_failure_bundle(*, base, faces, boundary_indices, mapped_boundary,
                        accepted, rejected, harmonic, grid_outline, grid_step,
                        fraction, last_valid_fraction, history,
                        metadata=None, grid_mask=None, grid_coverage=None,
                        output_dir=None):
    """Write lightweight JSON/SVG + full compressed arrays, return paths.

    The diagnostic directory can be overridden for tests or a separate
    experiment. Does NOT change the input arrays, solver or validity guards.
    """
    analysis, arrays = analyze_failure(
        base, faces, boundary_indices, mapped_boundary, accepted,
        rejected, harmonic, grid_outline, grid_step,
        fraction, last_valid_fraction,
    )
    if grid_mask is not None:
        arrays["grid_mask"] = np.asarray(grid_mask, dtype=bool)
    if grid_coverage is not None:
        arrays["grid_coverage"] = np.asarray(grid_coverage, dtype=float)
    if output_dir is None:
        output_dir = os.environ.get(
            "ONESTRING_GRID_DIAGNOSTICS_DIR",
            "logs/grid_boundary_diagnostics",
        )
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tag = re.sub(r"[^a-zA-Z0-9_-]", "_",
                 str((metadata or {}).get("attempt_tag", "fit")))[:70]
    name = f"{stamp}_{tag}_{uuid.uuid4().hex[:6]}"
    directory = destination / name
    directory.mkdir()
    data = dict(analysis)
    data["metadata"] = dict(metadata or {})
    data["history"] = list(history)
    data["created_utc"] = datetime.now(timezone.utc).isoformat()
    (directory/"diagnosis.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(directory/"reproduce_uv.npz", **arrays)
    (directory/"overview.svg").write_text(
        _svg_failure_view(arrays, analysis), encoding="utf-8",
    )
    (directory/"worst_triangle_zoom.svg").write_text(
        _svg_failure_view(arrays, analysis, focus=True), encoding="utf-8",
    )
    # Index makes it possible to compare different cut/rotation/corner fits
    # even if the enclosing Streamlit run ultimately raises an exception.
    record = {
        "path": str(directory),
        "created_utc": data["created_utc"],
        "metadata": data["metadata"],
        "last_valid_fraction": data["boundary_fraction_last_accepted"],
        "failed_fraction": data["boundary_fraction_rejected"],
        "worst_triangle_id": data["ranked_worst_triangles"][0]["triangle_id"],
        "min_signed_ratio": data["rejected_min_signed_area_ratio"],
        "reflex_hotspot_enrichment": data["reflex_hotspot_enrichment"],
        "diagnostic_hypothesis": data["diagnostic_hypothesis"],
        "boundary_crossings_at_failed_fraction":
            data["proper_boundary_intersections"][
                "target_at_failed_fraction"]["count"],
    }
    with (destination/"index.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False)+"\n")
    print("[GRID-DIAG] saved=" + str(directory), flush=True)
    print(
        "[GRID-DIAG] hypothesis=" + data["diagnostic_hypothesis"]
        + " worst_triangle="
        + str(data["ranked_worst_triangles"][0]["triangle_id"])
        + " reflex_enrichment="
        + f'{data["reflex_hotspot_enrichment"]:.3g}'
        + " boundary_crossings="
        + str(record["boundary_crossings_at_failed_fraction"]),
        flush=True,
    )
    return str(directory)
