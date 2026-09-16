"""Fast large-mesh T3D guard for interactive OneString experiments.

The legacy build pipeline reports 50% immediately after K3D and does not emit
56% until T3D extrusion finishes. The exact variable-topology T3D recovery has
repeated all-pairs solid collision audits, so large layouts can spend minutes
while the UI appears frozen at 50%.

Set ONESTRING_EXACT_T3D=1 to force the full variable-topology recovery. The
preview changes only K3D->T3D, not S->Omega or K3D.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Any

import numpy as np


def _truthy_environment(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _large_preview_threshold() -> int:
    raw = os.environ.get("ONESTRING_T3D_FAST_PREVIEW_TILE_THRESHOLD", "900")
    try:
        return max(1, int(raw))
    except Exception:
        return 900


def _aabb_sweep_candidates(points: np.ndarray, tolerance: float = 1e-9) -> list[tuple[int, int]]:
    """Conservative 3D AABB broad phase without enumerating all tile pairs."""
    values = np.asarray(points, dtype=float)
    if len(values) < 2:
        return []
    minimum = np.nanmin(values, axis=1)
    maximum = np.nanmax(values, axis=1)
    order = np.argsort(minimum[:, 0], kind="mergesort")
    active: list[int] = []
    pairs: list[tuple[int, int]] = []
    tol = float(max(tolerance, 0.0))
    for current_raw in order:
        current = int(current_raw)
        current_min_x = float(minimum[current, 0])
        if active:
            active = [other for other in active if float(maximum[other, 0]) >= current_min_x - tol]
        for other in active:
            if (
                maximum[other, 1] < minimum[current, 1] - tol
                or maximum[current, 1] < minimum[other, 1] - tol
                or maximum[other, 2] < minimum[current, 2] - tol
                or maximum[current, 2] < minimum[other, 2] - tol
            ):
                continue
            a, b = (other, current) if other < current else (current, other)
            pairs.append((a, b))
        active.append(current)
    return pairs


def _accelerated_top_surface_intersection_pairs(pipeline_module: Any, top_tiles: np.ndarray, faces: np.ndarray) -> list[tuple[int, int]]:
    """Use the old exact narrow phase after a conservative AABB broad phase."""
    tiles = np.asarray(top_tiles, dtype=float)
    adjacent = {
        tuple(sorted((entries[0][0], entries[1][0])))
        for entries in pipeline_module._build_edge_incidence(faces).values()
        if len(entries) == 2
    }
    intersections: list[tuple[int, int]] = []
    for tile_a, tile_b in _aabb_sweep_candidates(tiles):
        if (tile_a, tile_b) in adjacent:
            continue
        top_a = tiles[tile_a]
        top_b = tiles[tile_b]
        normal_a = pipeline_module._normalize(
            pipeline_module._original._quad_normal(top_a), np.asarray([0.0, 0.0, 1.0])
        )
        origin, u, v = pipeline_module._tile_plane_basis(top_a, normal_a)
        span = max(float(np.linalg.norm(top_a[(index + 1) % 4] - top_a[index])) for index in range(4))
        normal_b = pipeline_module._normalize(pipeline_module._original._quad_normal(top_b), normal_a)
        if abs(float(np.dot(normal_a, normal_b))) < 1.0 - 1e-5:
            continue
        plane_distance = np.max(np.abs((top_b - origin[None, :]) @ normal_a))
        if float(plane_distance) > max(span * 1e-7, 1e-9):
            continue
        poly_a = pipeline_module._project_to_basis(top_a, origin, u, v)
        poly_b = pipeline_module._project_to_basis(top_b, origin, u, v)
        overlap = pipeline_module._convex_polygon_clip(poly_a, poly_b)
        if pipeline_module._polygon_area_2d(overlap) > max(span * span * 1e-10, 1e-12):
            intersections.append((int(tile_a), int(tile_b)))
    return intersections


def install_fast_t3d_preview(pipeline_module: Any) -> None:
    """Install exact broad-phase acceleration plus a large-layout preview guard."""
    if getattr(pipeline_module, "_FAST_T3D_PREVIEW_PATCH_INSTALLED", False):
        return
    original_extrude = pipeline_module._extrude_tiles

    def accelerated_top_surface_intersections(top_tiles: np.ndarray, faces: np.ndarray):
        return _accelerated_top_surface_intersection_pairs(pipeline_module, top_tiles, faces)

    pipeline_module._top_surface_intersection_pairs = accelerated_top_surface_intersections

    def fast_extrude(mesh: Any, thickness: float, stage: str):
        variable = bool(getattr(mesh, "metrics", {}).get("t3d_variable_topology_enabled", False))
        tile_count = int(len(pipeline_module._original._mesh_tiles(mesh)))
        threshold = _large_preview_threshold()
        force_exact = _truthy_environment("ONESTRING_EXACT_T3D")
        if variable and tile_count > threshold and not force_exact:
            started = time.perf_counter()
            preview_mesh = copy.copy(mesh)
            preview_mesh.metrics = dict(getattr(mesh, "metrics", {}) or {})
            preview_mesh.metrics["t3d_variable_topology_enabled"] = False
            assembly, report = original_extrude(preview_mesh, thickness, stage)
            elapsed = time.perf_counter() - started
            assembly.metrics.update(
                {
                    "t3d_large_mesh_fast_preview": True,
                    "t3d_fast_preview_reason": "large variable-topology T3D skipped repeated quadratic solid collision audits",
                    "t3d_authoritative_variable_topology_skipped": True,
                    "t3d_fast_preview_tile_threshold": int(threshold),
                    "t3d_fast_preview_tile_count": int(tile_count),
                    "t3d_fast_preview_seconds": float(elapsed),
                    "t3d_fast_preview_opt_out": "set ONESTRING_EXACT_T3D=1 before starting Streamlit",
                    "t3d_fast_preview_scope": "K3D->T3D only; S->Omega and K3D are unchanged",
                }
            )
            try:
                report.objective = str(report.objective) + " Large-layout interactive preview used; exact variable-topology recovery was skipped."
            except Exception:
                pass
            return assembly, report
        return original_extrude(mesh, thickness, stage)

    # The backed-up build_onestring_design calls its own module-global
    # _extrude_tiles, so patch both views.
    pipeline_module._extrude_tiles = fast_extrude
    pipeline_module._original._extrude_tiles = fast_extrude
    pipeline_module._FAST_T3D_PREVIEW_PATCH_INSTALLED = True


__all__ = ["install_fast_t3d_preview", "_aabb_sweep_candidates"]
