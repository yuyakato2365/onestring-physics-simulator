"""Monotone nearest-boundary correspondence for ``optcuts_test``.

The first OptCuts_test prototype mapped the complete cut boundary to the grid
outline by uniform normalized arclength.  That preserves order, but it can force
a concave grid corner onto an unrelated location of the OptCuts boundary and
make a bijective continuation stall.

This patch keeps the cyclic order invariant while allowing boundary vertices to
slide along the target outline.  Each source boundary vertex first proposes its
nearest target arclength.  Those proposals are lifted to the appropriate target
lap using the source arclength fraction as a weak prior and then projected onto
the set of monotone sequences by weighted isotonic regression (PAVA).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import optcuts_test_boundary_reparameterization_patch as boundary_patch


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted non-decreasing isotonic regression using pool-adjacent violators."""
    y = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(y) != len(w):
        raise ValueError("PAVA value/weight size mismatch")
    if len(y) == 0:
        return y.copy()

    means: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, (value, mass) in enumerate(zip(y, w)):
        means.append(float(value))
        masses.append(max(float(mass), 1e-12))
        starts.append(i)
        ends.append(i + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            total = masses[-2] + masses[-1]
            merged = (masses[-2] * means[-2] + masses[-1] * means[-1]) / total
            means[-2] = float(merged)
            masses[-2] = float(total)
            ends[-2] = ends[-1]
            means.pop(); masses.pop(); starts.pop(); ends.pop()

    out = np.empty_like(y)
    for mean, start, end in zip(means, starts, ends):
        out[start:end] = mean
    return out


def _source_fractions(points_closed: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_closed, dtype=float)
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-20:
        return np.linspace(0.0, 1.0, len(pts))
    return cumulative / total


def _nearest_monotone_map(
    uv: np.ndarray,
    loop: list[int],
    outline: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    ids = [int(v) for v in loop]
    if ids[0] != ids[-1]:
        ids.append(ids[0])
    source = np.asarray(uv, dtype=float)[np.asarray(ids, dtype=int)]
    fractions = _source_fractions(source)

    _, _, perimeter = boundary_patch._closed_polyline_data(outline)
    if perimeter <= 1e-20:
        raise RuntimeError("OPTCUTS_TEST_DEGENERATE_GRID_OUTLINE")

    # Preserve orientation.  Work in a positive target-arclength coordinate q,
    # then convert back to the target outline's signed direction at the end.
    direction = 1.0 if boundary_patch._signed_area(source) * boundary_patch._signed_area(outline) >= 0.0 else -1.0
    start_s = boundary_patch._project_arclength(source[0], outline)

    projected = np.asarray(
        [boundary_patch._project_arclength(p, outline) for p in source[:-1]],
        dtype=float,
    )
    relative = np.mod(direction * (projected - start_s), perimeter)
    prior = fractions[:-1] * perimeter

    # Lift each nearest projection to the target lap closest to its source
    # arclength prior.  This prevents a spatially close concave branch on the
    # wrong side of the cyclic cut from stealing the correspondence.
    lifted = np.empty_like(relative)
    for i, (value, expected) in enumerate(zip(relative, prior)):
        candidates = np.asarray([value - perimeter, value, value + perimeter], dtype=float)
        lifted[i] = float(candidates[int(np.argmin(np.abs(candidates - expected)))])

    # Blend nearest-target evidence with a weaker uniform-arclength prior.
    # Endpoints are anchored strongly so the whole source loop traverses exactly
    # one target lap.  The final duplicated source endpoint is handled separately.
    nearest_weight = 4.0
    prior_weight = 1.0
    observations = (nearest_weight * lifted + prior_weight * prior) / (nearest_weight + prior_weight)
    weights = np.full(len(observations), nearest_weight + prior_weight, dtype=float)

    # Include virtual 0/T endpoints with large weight for a one-lap monotone map.
    augmented_y = np.concatenate([[0.0], observations[1:], [perimeter]])
    augmented_w = np.concatenate([[1.0e6], weights[1:], [1.0e6]])
    fitted_aug = _pava(augmented_y, augmented_w)
    fitted = np.empty_like(observations)
    fitted[0] = 0.0
    if len(fitted) > 1:
        fitted[1:] = fitted_aug[1:-1]

    # Keep strict order numerically.  Equal isotonic blocks are separated by a
    # tiny amount compared with one grid-edge length, avoiding boundary-edge
    # collapse without materially changing the optimized correspondence.
    if len(fitted) > 1:
        eps = max(1e-12, perimeter * 1e-10)
        for i in range(1, len(fitted)):
            fitted[i] = max(fitted[i], fitted[i - 1] + eps)
        if fitted[-1] >= perimeter:
            scale = (perimeter - eps) / max(fitted[-1], eps)
            fitted[1:] *= scale

    mapped: dict[int, np.ndarray] = {}
    squared = 0.0
    uniform_squared = 0.0
    for uid, q, p, frac in zip(ids[:-1], fitted, source[:-1], fractions[:-1]):
        target_point = boundary_patch._point_at_closed_arclength(
            outline, start_s + direction * float(q)
        )
        uniform_point = boundary_patch._point_at_closed_arclength(
            outline, start_s + direction * float(frac * perimeter)
        )
        mapped[int(uid)] = np.asarray(target_point, dtype=float)
        squared += float(np.sum((target_point - p) ** 2))
        uniform_squared += float(np.sum((uniform_point - p) ** 2))

    n = max(len(mapped), 1)
    diagnostics = {
        "boundary_correspondence_rms": float(np.sqrt(squared / n)),
        "uniform_arclength_rms": float(np.sqrt(uniform_squared / n)),
        "correspondence_improvement_ratio": float(squared / max(uniform_squared, 1e-30)),
        "target_perimeter": float(perimeter),
    }
    return mapped, diagnostics


def _build_test_targets_monotone(parameterization: Any, grid_outline: np.ndarray):
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    targets = np.full_like(uv, np.nan)
    loops = boundary_patch._optcuts_boundary_loops(parameterization)
    if not loops:
        raise RuntimeError("OPTCUTS_TEST_NO_UV_BOUNDARY_LOOP")

    primary = max(loops, key=len)
    mapped, diagnostics = _nearest_monotone_map(uv, primary, grid_outline)
    for uid, point in mapped.items():
        targets[int(uid)] = np.asarray(point, dtype=float)

    boundary_vertices = {
        v for edge in boundary_patch._uv_boundary_edges(parameterization) for v in edge
    }
    counts: dict[str, Any] = {
        "seam_vertex_count": int(boundary_patch._seam_vertex_count(parameterization)),
        "boundary_target_vertex_count": int(len(mapped)),
        "outer_boundary_vertex_count": int(len(mapped)),
        "outer_boundary_chain_count": 1,
        "uv_boundary_vertex_count": int(len(boundary_vertices)),
        "uv_boundary_loop_count": int(len(loops)),
        "boundary_correspondence_model": "nearest target arclength + source-arclength prior + cyclic weighted isotonic regression",
        **diagnostics,
    }
    return targets, counts


def install_optcuts_test_monotone_correspondence_patch() -> None:
    """Replace only the experimental optcuts_test boundary correspondence."""
    boundary_patch._build_test_targets = _build_test_targets_monotone


__all__ = [
    "install_optcuts_test_monotone_correspondence_patch",
    "_nearest_monotone_map",
    "_pava",
]
