"""Bridge already-independent Eq.(5) K2D into the legacy FlatTileLayout API.

Eq.(5) already returns one 4-corner block per tile, so this bridge preserves
those coordinates instead of running the legacy independent-tile placement a
second time.  FlatTileLayout.hinge_pairs, however, is a TILE-pair API.  The
first bridge accidentally stored copied-vertex indices there; downstream Dual
Hinge then received ids up to 2623 for a 656-tile layout.  This version rebuilds
that compatibility field as deduplicated tile-id pairs while leaving K2D xy
unchanged.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _is_auxetic_k2d(mesh: Any) -> bool:
    return bool((getattr(mesh, "metrics", {}) or {}).get("paper_eq5_auxetic_topology", False))


def _polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def _layout_overlap_metrics(xy: np.ndarray) -> tuple[int, float]:
    def overlap(a: np.ndarray, b: np.ndarray) -> bool:
        for poly in (a, b):
            for i in range(len(poly)):
                edge = poly[(i + 1) % len(poly)] - poly[i]
                axis = np.array([-edge[1], edge[0]], dtype=float)
                n = float(np.linalg.norm(axis))
                if n <= 1e-12:
                    continue
                axis /= n
                pa = a @ axis
                pb = b @ axis
                if min(float(np.max(pa)), float(np.max(pb))) - max(float(np.min(pa)), float(np.min(pb))) <= 1e-10:
                    return False
        return True

    count = 0
    area_proxy = 0.0
    bounds_lo = np.min(xy, axis=1)
    bounds_hi = np.max(xy, axis=1)
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if np.any(bounds_hi[i] <= bounds_lo[j] + 1e-10) or np.any(bounds_hi[j] <= bounds_lo[i] + 1e-10):
                continue
            if overlap(xy[i], xy[j]):
                count += 1
                lo = np.maximum(bounds_lo[i], bounds_lo[j])
                hi = np.minimum(bounds_hi[i], bounds_hi[j])
                area_proxy += max(0.0, float(hi[0] - lo[0])) * max(0.0, float(hi[1] - lo[1]))
    return int(count), float(area_proxy)


def _tile_hinge_pairs_from_source_ids(source_ids: np.ndarray, tile_count: int) -> list[tuple[int, int]]:
    """Return FlatTileLayout-compatible TILE pairs, never copied-vertex ids."""
    incident: dict[int, set[int]] = {}
    for q, source in enumerate(source_ids.tolist()):
        tile_id = int(q // 4)
        if 0 <= tile_id < tile_count:
            incident.setdefault(int(source), set()).add(tile_id)

    pairs: set[tuple[int, int]] = set()
    for tiles in incident.values():
        ordered = sorted(tiles)
        # FlatTileLayout only carries pair connectivity, not local-corner ids.
        # Connect all tiles sharing the same original M2D vertex; downstream
        # hinge construction resolves local corners from the actual tile data.
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if a != b:
                    pairs.add((a, b))
    return sorted(pairs)


def _direct_layout(pipeline: Any, mesh: Any) -> Any:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise RuntimeError(f"Eq.5 direct flat layout expected quad faces, got {faces.shape}")
    if len(vertices) != 4 * len(faces):
        raise RuntimeError(
            "Eq.5 direct flat layout expected exactly four independent vertices "
            f"per tile, got vertices={len(vertices)} faces={len(faces)}"
        )
    expected = np.arange(4 * len(faces), dtype=int).reshape(-1, 4)
    if not np.array_equal(faces, expected):
        raise RuntimeError("Eq.5 direct flat layout expected sequential independent face indices")

    mesh_metrics = getattr(mesh, "metrics", {}) or {}
    source_ids = np.asarray(mesh_metrics.get("auxetic_source_vertex_ids", []), dtype=int)
    if len(source_ids) != len(vertices):
        raise RuntimeError(
            "Eq.5 direct flat layout is missing auxetic_source_vertex_ids; "
            f"got {len(source_ids)} for {len(vertices)} vertices"
        )

    hinge_pairs = _tile_hinge_pairs_from_source_ids(source_ids, len(faces))
    invalid_pairs = [(a, b) for a, b in hinge_pairs if a < 0 or b < 0 or a >= len(faces) or b >= len(faces)]
    if invalid_pairs:
        raise RuntimeError(f"Eq.5 bridge generated invalid tile hinge pairs: {invalid_pairs[:5]}")

    layout_type = getattr(pipeline, "FlatTileLayout", None)
    if layout_type is None:
        layout_type = getattr(getattr(pipeline, "_original", None), "FlatTileLayout", None)
    if layout_type is None:
        raise RuntimeError("Eq.5 direct flat layout could not resolve FlatTileLayout")

    xy = vertices[faces, :2].copy()
    overlap_count, overlap_area = _layout_overlap_metrics(xy)
    total_tile_area = float(sum(_polygon_area(tile) for tile in xy))
    gap_count = int(mesh_metrics.get("paper_eq5_physical_gap_count", mesh_metrics.get("physical_gap_count", 0)))
    metrics = {
        "source": "paper_eq5_already_independent_k2d",
        "direct_from_k2d": True,
        "legacy_second_independentization_bypassed": True,
        "tile_count": int(len(faces)),
        "hinge_pair_count": int(len(hinge_pairs)),
        "hinge_pair_index_space": "tile_ids",
        "hinge_pair_max_id": int(max((max(p) for p in hinge_pairs), default=-1)),
        "tile_overlap_count": int(overlap_count),
        "min_clearance": 0.0,
        "k2d_gap_count": int(gap_count),
        "layout_type": "paper_eq5_direct_independent_tiles",
        "tile_overlap_area": float(overlap_area),
        "tile_total_area": float(total_tile_area),
        "tile_overlap_area_ratio": float(overlap_area / max(total_tile_area, 1e-12)),
        "input_extent_x": float(np.ptp(vertices[:, 0])) if len(vertices) else 0.0,
        "input_extent_y": float(np.ptp(vertices[:, 1])) if len(vertices) else 0.0,
    }
    print(
        f"[PAPER-EQ5-FLAT-BRIDGE] direct=True tiles={len(faces)} vertices={len(vertices)} "
        f"hinge_pairs={len(hinge_pairs)} hinge_pair_max_id={metrics['hinge_pair_max_id']} "
        f"gaps={gap_count} overlaps={overlap_count} min_clearance=0 "
        f"extent={metrics['input_extent_x']:.6g}x{metrics['input_extent_y']:.6g}"
    )
    return layout_type(
        tile_top_vertices_2d=xy,
        tile_ids=list(range(len(faces))),
        hinge_pairs=hinge_pairs,
        gap_polygons=[],
        metrics=metrics,
    )


def install_eq5_flat_layout_bridge() -> None:
    from . import lscm_latest_omega_hybrid_20260914_patch as hybrid
    if getattr(hybrid, "_eq5_flat_layout_bridge_installed", False):
        return
    base_wire = hybrid._wire

    def wire_with_bridge(pipeline: Any, name: str, fn: Any) -> None:
        if name == "_make_flat_tile_layout":
            fallback = fn
            def direct_or_fallback(mesh: Any, params: Any = None) -> Any:
                if _is_auxetic_k2d(mesh):
                    return _direct_layout(pipeline, mesh)
                return fallback(mesh, params)
            fn = direct_or_fallback
        base_wire(pipeline, name, fn)

    hybrid._wire = wire_with_bridge
    hybrid._eq5_flat_layout_bridge_installed = True
    print("[PAPER-EQ5-FLAT-BRIDGE] installer armed")


__all__ = ["install_eq5_flat_layout_bridge"]