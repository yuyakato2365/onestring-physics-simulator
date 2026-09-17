"""Prevent the experimental Eq.(5) K2D from being independentized a second time.

The Eq.(5) solver already returns one independent 4-corner vertex block per tile.
The legacy _make_flat_tile_layout expects a shared-vertex K2D mesh and performs
another independent-tile placement solve. Feeding the Eq.(5) output into that
legacy path destroys the solver result. This bridge converts the already-
independent K2D to FlatTileLayout without moving it.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _is_auxetic_k2d(mesh: Any) -> bool:
    metrics = getattr(mesh, "metrics", {}) or {}
    return bool(metrics.get("paper_eq5_auxetic_topology", False))


def _polygon_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def _layout_overlap_metrics(xy: np.ndarray) -> tuple[int, float]:
    """Compute compatibility metrics consumed by legacy build_onestring_design."""
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

    source_ids = np.asarray((getattr(mesh, "metrics", {}) or {}).get("auxetic_source_vertex_ids", []), dtype=int)
    if len(source_ids) != len(vertices):
        raise RuntimeError(
            "Eq.5 direct flat layout is missing auxetic_source_vertex_ids; "
            f"got {len(source_ids)} for {len(vertices)} vertices"
        )

    groups: dict[int, list[int]] = {}
    for q, source in enumerate(source_ids.tolist()):
        groups.setdefault(int(source), []).append(int(q))
    hinge_pairs: list[tuple[int, int]] = []
    for copies in groups.values():
        if len(copies) < 2:
            continue
        anchor = copies[0]
        hinge_pairs.extend((anchor, q) for q in copies[1:])

    layout_type = getattr(pipeline, "FlatTileLayout", None)
    if layout_type is None:
        layout_type = getattr(getattr(pipeline, "_original", None), "FlatTileLayout", None)
    if layout_type is None:
        raise RuntimeError("Eq.5 direct flat layout could not resolve FlatTileLayout")

    xy = vertices[faces, :2].copy()
    overlap_count, overlap_area = _layout_overlap_metrics(xy)
    total_tile_area = float(sum(_polygon_area(tile) for tile in xy))
    metrics = {
        "source": "paper_eq5_already_independent_k2d",
        "direct_from_k2d": True,
        "legacy_second_independentization_bypassed": True,
        "tile_count": int(len(faces)),
        "hinge_pair_count": int(len(hinge_pairs)),
        "tile_overlap_count": int(overlap_count),
        "tile_overlap_area": float(overlap_area),
        "tile_total_area": float(total_tile_area),
        "tile_overlap_area_ratio": float(overlap_area / max(total_tile_area, 1e-12)),
        "input_extent_x": float(np.ptp(vertices[:, 0])) if len(vertices) else 0.0,
        "input_extent_y": float(np.ptp(vertices[:, 1])) if len(vertices) else 0.0,
    }
    print(
        f"[PAPER-EQ5-FLAT-BRIDGE] direct=True tiles={len(faces)} "
        f"vertices={len(vertices)} hinge_pairs={len(hinge_pairs)} "
        f"overlaps={overlap_count} extent={metrics['input_extent_x']:.6g}x{metrics['input_extent_y']:.6g}"
    )
    return layout_type(
        tile_top_vertices_2d=xy,
        tile_ids=list(range(len(faces))),
        hinge_pairs=hinge_pairs,
        gap_polygons=[],
        metrics=metrics,
    )


def install_eq5_flat_layout_bridge() -> None:
    """Wrap the hybrid patch's wiring so every rewire gets the direct bridge."""
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