"""Compatibility helpers for running ABD on pipeline-generated builtin shapes.

The normal ABD bridge uses the routed OneString gap path to create body-attached
string guides.  Some builtin/demo shapes can temporarily produce an empty,
partially invalid, or geometrically duplicated gap path even though the tile and
hinge assemblies are otherwise valid.  The ABD solver requires at least two
spatially distinct guide points, so those states previously failed before the
external solver was launched.

This module keeps the normal routed guides whenever they are usable.  Only when
they are missing or degenerate does it derive a deterministic fallback from the
same gap ownership, hinge graph, and finally the available tile order.  It does
not replace the ABD solver or silently fall back to the legacy physics backend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import abd_backend as _abd


_ORIGINAL_STRING_GUIDES = _abd._string_guides
_INSTALLED = False


def _guide_world(guide: dict[str, Any]) -> np.ndarray:
    return np.asarray(guide.get("initial_world_point", []), dtype=float).reshape(-1)


def _usable_guides(guides: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    """Remove invalid and coincident guides while preserving routed order."""

    usable: list[dict[str, Any]] = []
    for guide in guides:
        world = _guide_world(guide)
        if world.size < 3 or not np.all(np.isfinite(world[:3])):
            continue
        if usable and np.linalg.norm(world[:3] - _guide_world(usable[-1])[:3]) <= tolerance:
            continue
        usable.append(guide)

    if len(usable) >= 2:
        path_length = sum(
            float(np.linalg.norm(_guide_world(usable[index + 1])[:3] - _guide_world(usable[index])[:3]))
            for index in range(len(usable) - 1)
        )
        if path_length > tolerance:
            return usable
    return []


def _candidate_tile_ids(state: Any, tile_count: int) -> list[int]:
    """Return a stable tile order using path ownership before broad fallbacks."""

    ordered: list[int] = []

    def append(tile_id: Any) -> None:
        try:
            value = int(tile_id)
        except (TypeError, ValueError):
            return
        if 0 <= value < tile_count and value not in ordered:
            ordered.append(value)

    gap_by_id = {
        int(gap.id): gap
        for gap in getattr(getattr(state, "gap_graph", None), "gaps", [])
        if hasattr(gap, "id")
    }
    for gap_id in getattr(getattr(state, "string_path", None), "gap_ids", []):
        gap = gap_by_id.get(int(gap_id))
        for tile_id in getattr(gap, "surrounding_tiles", []) if gap is not None else []:
            append(tile_id)

    for hinge in getattr(getattr(state, "hinge_graph", None), "hinges", []):
        append(getattr(hinge, "tile_a", None))
        append(getattr(hinge, "tile_b", None))

    for tile_id in range(tile_count):
        append(tile_id)
    return ordered


def _fallback_guides(
    state: Any,
    centers: np.ndarray,
    initial_proxy: np.ndarray,
    tolerance: float,
) -> list[dict[str, Any]]:
    tile_count = int(len(initial_proxy))
    if tile_count == 0:
        raise _abd.ABDBackendError("ABD cannot create string guides because the tile assembly is empty")

    candidate_ids = _candidate_tile_ids(state, tile_count)
    guides: list[dict[str, Any]] = []

    for tile_id in candidate_ids:
        top = np.asarray(initial_proxy[tile_id, :4], dtype=float)
        if top.shape != (4, 3) or not np.all(np.isfinite(top)):
            continue

        # Prefer the corner farthest from the preceding guide.  This avoids
        # repeated guide positions on symmetric builtin shapes and keeps the
        # fallback deterministic.
        if guides:
            previous = _guide_world(guides[-1])[:3]
            vertex_id = int(np.argmax(np.linalg.norm(top - previous[None, :], axis=1)))
        else:
            vertex_id = 0
        world = top[vertex_id]
        if guides and np.linalg.norm(world - _guide_world(guides[-1])[:3]) <= tolerance:
            continue
        guides.append(
            {
                "gap_id": -1,
                "body_id": int(tile_id),
                "body_name": f"tile_{tile_id:04d}",
                "material_point": (world - centers[tile_id]).tolist(),
                "initial_world_point": world.tolist(),
                "source": "builtin_shape_fallback",
            }
        )
        if len(guides) >= 2:
            break

    # A one-tile diagnostic shape is still valid: route through two distinct
    # corners on that body rather than failing with a zero-length path.
    if len(guides) == 1:
        tile_id = int(guides[0]["body_id"])
        top = np.asarray(initial_proxy[tile_id, :4], dtype=float)
        first = _guide_world(guides[0])[:3]
        distances = np.linalg.norm(top - first[None, :], axis=1)
        vertex_id = int(np.argmax(distances))
        if float(distances[vertex_id]) > tolerance:
            world = top[vertex_id]
            guides.append(
                {
                    "gap_id": -1,
                    "body_id": tile_id,
                    "body_name": f"tile_{tile_id:04d}",
                    "material_point": (world - centers[tile_id]).tolist(),
                    "initial_world_point": world.tolist(),
                    "source": "builtin_shape_fallback",
                }
            )

    usable = _usable_guides(guides, tolerance)
    if len(usable) < 2:
        raise _abd.ABDBackendError(
            "ABD requires at least two spatially distinct string guides. "
            "The builtin-shape fallback could not derive them from the gap, hinge, or tile data."
        )
    return usable


def _builtin_compatible_string_guides(
    state: Any,
    centers: np.ndarray,
    initial_proxy: np.ndarray,
    guide_reference_proxy: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    scale = float(np.max(np.ptp(np.asarray(initial_proxy, dtype=float), axis=(0, 1)))) if np.size(initial_proxy) else 1.0
    tolerance = max(1.0e-10, scale * 1.0e-9)

    routed = _ORIGINAL_STRING_GUIDES(state, centers, initial_proxy, guide_reference_proxy)
    usable = _usable_guides(routed, tolerance)
    if usable:
        for guide in usable:
            guide.setdefault("source", "routed_gap_path")
        return usable
    return _fallback_guides(state, centers, initial_proxy, tolerance)


def install_builtin_shape_abd_compatibility() -> None:
    """Install the narrow guide-generation compatibility patch once."""

    global _INSTALLED
    if _INSTALLED:
        return
    _abd._string_guides = _builtin_compatible_string_guides
    _INSTALLED = True


__all__ = ["install_builtin_shape_abd_compatibility"]
