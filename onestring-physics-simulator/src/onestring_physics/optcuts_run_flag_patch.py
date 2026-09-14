"""Persist the active OptCuts mode across stages that replace mesh metrics."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .optcuts_grid_orientation_patch import install_optcuts_grid_orientation_patch
from .optcuts_grid_seam_sidecar_patch import install_optcuts_grid_seam_sidecar_patch
from .optcuts_official_binary_patch import install_official_optcuts_binary_separation

HARD_COLLISION_VERSION_ID = "2026-09-14-paper-eq5-hard-nonpenetration"
HARD_COLLISION_VERSION_LABEL = "2026-09-14 — Paper Eq.5 + hard non-penetration"

install_official_optcuts_binary_separation()
install_optcuts_grid_orientation_patch()
install_optcuts_grid_seam_sidecar_patch()


def _enabled() -> bool:
    return os.environ.get("ONESTRING_HARD_NONPENETRATION", "0").lower() in {"1", "true", "yes", "on"}


def _install_optcuts_weight_ui_patch() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_optcuts_weight_ui_patch_installed", False):
        return

    original_number_input = st.number_input
    original_slider = st.slider
    original_selectbox = st.selectbox

    def patched_number_input(label: str, *args: Any, **kwargs: Any):
        if label == "w_planar / EPlanar":
            kwargs["max_value"] = 1_000_000.0
        elif label == "w_square / ESquare":
            kwargs["max_value"] = 1_000.0
        return original_number_input(label, *args, **kwargs)

    def patched_slider(label: str, *args: Any, **kwargs: Any):
        if label != "hinge connection weight":
            return original_slider(label, *args, **kwargs)
        min_value = float(args[0]) if len(args) > 0 else float(kwargs.pop("min_value", 0.1))
        value = float(args[2]) if len(args) > 2 else float(kwargs.pop("value", 8.0))
        step = float(args[3]) if len(args) > 3 else float(kwargs.pop("step", 0.1))
        help_text = kwargs.pop("help", "Weight for E_Conn: pairwise hinge vertex coincidence.")
        key = kwargs.pop("key", None)
        return original_number_input(
            label,
            min_value=min(0.0, min_value),
            max_value=1_000.0,
            value=min(1_000.0, max(0.0, value)),
            step=step,
            help=help_text,
            key=key,
            format="%.3f",
            **kwargs,
        )

    def patched_selectbox(label: str, options: Any, *args: Any, **kwargs: Any):
        if label != "version":
            return original_selectbox(label, options, *args, **kwargs)
        option_list = list(options)
        if not any(isinstance(v, dict) and v.get("id") == HARD_COLLISION_VERSION_ID for v in option_list):
            template = next(
                (dict(v) for v in reversed(option_list) if isinstance(v, dict) and v.get("id") == "2026-09-14-paper-eq5-unified-k2d"),
                {},
            )
            template.update({
                "id": HARD_COLLISION_VERSION_ID,
                "label": HARD_COLLISION_VERSION_LABEL,
                "description": (
                    "Paper Eq.5 K2D版。flat layoutだけでなくT2D Dual Hinge後の最終配置にも"
                    "非貫通をハード制約として要求し、重なりが1組でも残れば失敗する。"
                ),
            })
            option_list.append(template)
        selected = original_selectbox(label, option_list, *args, **kwargs)
        enabled = isinstance(selected, dict) and selected.get("id") == HARD_COLLISION_VERSION_ID
        os.environ["ONESTRING_HARD_NONPENETRATION"] = "1" if enabled else "0"
        if enabled:
            st.caption(
                "Hard non-penetration is checked again after T2D Dual Hinge. "
                "The final dual-hinge layout is accepted only when SAT overlap count is exactly zero."
            )
        return selected

    st.number_input = patched_number_input
    st.slider = patched_slider
    st.selectbox = patched_selectbox
    st._onestring_optcuts_weight_ui_patch_installed = True


_install_optcuts_weight_ui_patch()


def _sat_mtv(poly_a: np.ndarray, poly_b: np.ndarray, tolerance: float) -> np.ndarray | None:
    a = np.asarray(poly_a, dtype=float)[:, :2]
    b = np.asarray(poly_b, dtype=float)[:, :2]
    best_axis = None
    best_overlap = float("inf")
    for poly in (a, b):
        for edge in np.roll(poly, -1, axis=0) - poly:
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis /= norm
            pa = a @ axis
            pb = b @ axis
            overlap = min(float(np.max(pa)), float(np.max(pb))) - max(float(np.min(pa)), float(np.min(pb)))
            if overlap <= tolerance:
                return None
            if overlap < best_overlap:
                best_overlap = overlap
                best_axis = axis.copy()
    if best_axis is None or not np.isfinite(best_overlap):
        return None
    if float(np.dot(np.mean(a, axis=0) - np.mean(b, axis=0), best_axis)) < 0.0:
        best_axis = -best_axis
    return best_axis * (best_overlap + tolerance)


def _aabb_candidate_pairs(tiles: np.ndarray, tolerance: float):
    mins = np.min(tiles, axis=1)
    maxs = np.max(tiles, axis=1)
    for i in range(len(tiles) - 1):
        mask = (
            (mins[i, 0] < maxs[i + 1 :, 0] - tolerance)
            & (maxs[i, 0] > mins[i + 1 :, 0] + tolerance)
            & (mins[i, 1] < maxs[i + 1 :, 1] - tolerance)
            & (maxs[i, 1] > mins[i + 1 :, 1] + tolerance)
        )
        for rel in np.flatnonzero(mask):
            yield i, i + 1 + int(rel)


def _count_overlaps(tiles: np.ndarray, tolerance: float) -> tuple[int, float]:
    count = 0
    max_depth = 0.0
    for i, j in _aabb_candidate_pairs(tiles, tolerance):
        mtv = _sat_mtv(tiles[i], tiles[j], tolerance)
        if mtv is not None:
            count += 1
            max_depth = max(max_depth, float(np.linalg.norm(mtv)))
    return count, max_depth


def _solve_nonpenetrating_translations(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Gauss-Seidel feasibility projection using rigid XY translations only."""
    original = np.asarray(polygons, dtype=float)
    tiles = original.copy()
    if len(tiles) < 2:
        return tiles, np.zeros((len(tiles), 2), dtype=float), {"initial": 0, "final": 0, "sweeps": 0, "depth": 0.0}

    edge_lengths = np.linalg.norm(np.roll(tiles, -1, axis=1) - tiles, axis=2)
    positive = edge_lengths[edge_lengths > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    tolerance = max(1e-10, scale * 1e-7)
    initial_count, initial_depth = _count_overlaps(tiles, tolerance)
    max_sweeps = 500
    sweep = 0

    for sweep in range(1, max_sweeps + 1):
        moved = False
        # Rebuild candidates every sweep. Pair corrections are applied immediately,
        # so collision cannot be hidden by averaging mutually cancelling updates.
        for i, j in list(_aabb_candidate_pairs(tiles, tolerance)):
            mtv = _sat_mtv(tiles[i], tiles[j], tolerance)
            if mtv is None:
                continue
            tiles[i] += 0.5 * mtv
            tiles[j] -= 0.5 * mtv
            moved = True
        final_count, final_depth = _count_overlaps(tiles, tolerance)
        if final_count == 0:
            break
        if not moved:
            break

    final_count, final_depth = _count_overlaps(tiles, tolerance)
    if final_count != 0:
        raise RuntimeError(
            "HARD_NONPENETRATION_INFEASIBLE: "
            f"{final_count} overlapping tile pairs remain after {max_sweeps} final-stage sweeps "
            f"(max penetration proxy={final_depth:.6g})."
        )

    delta = np.mean(tiles - original, axis=1)
    return tiles, delta, {
        "initial": int(initial_count),
        "final": int(final_count),
        "sweeps": int(sweep),
        "depth": float(initial_depth),
    }


def _project_hard_nonpenetration(layout: Any) -> Any:
    tiles, _delta, stats = _solve_nonpenetrating_translations(np.asarray(layout.tile_top_vertices_2d, dtype=float))
    layout.tile_top_vertices_2d = tiles
    try:
        layout.metrics.update({
            "hard_nonpenetration_enabled": True,
            "hard_nonpenetration_stage": "flat_layout",
            "hard_nonpenetration_initial_overlap_pairs": stats["initial"],
            "hard_nonpenetration_final_overlap_pairs": stats["final"],
            "hard_nonpenetration_sweeps": stats["sweeps"],
            "hard_nonpenetration_acceptance": "final SAT overlap count must equal zero",
        })
    except Exception:
        pass
    print(f"[HARD-NONPENETRATION-FLAT] initial={stats['initial']} final=0 sweeps={stats['sweeps']}")
    return layout


def _project_dual_hinge_hard_nonpenetration(out: Any, hinge_graph: Any) -> None:
    vertices = np.asarray(out.vertices, dtype=float)
    if len(vertices) < 2:
        return
    _tops, delta, stats = _solve_nonpenetrating_translations(vertices[:, :4, :2])
    out.vertices[:, :, :2] += delta[:, None, :]
    transforms = getattr(out, "transform_matrices", None)
    if transforms is not None:
        transforms[:, 0, 3] += delta[:, 0]
        transforms[:, 1, 3] += delta[:, 1]

    # Hard projection happens after the dual-hinge solver, so refresh every
    # stored hinge point from the actual final geometry.
    for hinge in getattr(hinge_graph, "hinges", []):
        a = int(hinge.tile_a)
        b = int(hinge.tile_b)
        va = int(hinge.local_vertex_a)
        vb = int(hinge.local_vertex_b)
        hinge.rest_position_2d = 0.5 * (out.vertices[a, va] + out.vertices[b, vb])

    metrics = {
        "hard_nonpenetration_enabled": True,
        "hard_nonpenetration_stage": "post_dual_hinge_final",
        "hard_nonpenetration_initial_overlap_pairs": stats["initial"],
        "hard_nonpenetration_final_overlap_pairs": stats["final"],
        "hard_nonpenetration_sweeps": stats["sweeps"],
        "hard_nonpenetration_acceptance": "final Dual Hinge SAT overlap count must equal zero",
    }
    try:
        out.metrics.update(metrics)
    except Exception:
        pass
    try:
        hinge_graph.metrics.update(metrics)
    except Exception:
        pass
    print(f"[HARD-NONPENETRATION-DUAL] initial={stats['initial']} final=0 sweeps={stats['sweeps']}")


def _wire_dual_hard_wrapper(pipeline: Any) -> None:
    base = getattr(pipeline, "_optimize_dual_hinges", None)
    if not callable(base) or getattr(base, "_onestring_hard_nonpenetration_final_wrapper", False):
        return

    def dual_with_final_hard_collision(*args: Any, **kwargs: Any):
        out, hinge_graph, report = base(*args, **kwargs)
        if _enabled():
            _project_dual_hinge_hard_nonpenetration(out, hinge_graph)
        return out, hinge_graph, report

    dual_with_final_hard_collision._onestring_hard_nonpenetration_final_wrapper = True
    pipeline._optimize_dual_hinges = dual_with_final_hard_collision
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_dual_hinges = dual_with_final_hard_collision
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_dual_hinges"] = dual_with_final_hard_collision


def install_optcuts_run_flag_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_run_flag_patch_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    base_original_flat = getattr(pipeline, "_ORIGINAL_MAKE_FLAT_TILE_LAYOUT", None)
    if callable(base_original_flat) and not getattr(base_original_flat, "_onestring_hard_nonpenetration_wrapper", False):
        def original_flat_with_optional_hard_collision(mesh: Any, params: Any = None):
            layout = base_original_flat(mesh, params)
            return _project_hard_nonpenetration(layout) if _enabled() else layout
        original_flat_with_optional_hard_collision._onestring_hard_nonpenetration_wrapper = True
        pipeline._ORIGINAL_MAKE_FLAT_TILE_LAYOUT = original_flat_with_optional_hard_collision

    _wire_dual_hard_wrapper(pipeline)

    # Simple Split is installed later by app_split_panels.py. Re-wrap the final
    # dual-hinge entry point after that installer too, so no later routing can
    # silently bypass the hard final acceptance check.
    try:
        from . import simple_split_panel_patch as simple_split_module
        if not getattr(simple_split_module, "_onestring_hard_nonpenetration_rewire_installed", False):
            original_installer = simple_split_module.install_simple_split_panel_patch

            def install_then_hard_rewire(pipeline_module: Any, optimization_debug_module: Any) -> None:
                original_installer(pipeline_module, optimization_debug_module)
                _wire_dual_hard_wrapper(pipeline_module)
                print("[HARD-NONPENETRATION-ROUTE] final Dual Hinge wrapper reinstalled after Simple Split")

            simple_split_module.install_simple_split_panel_patch = install_then_hard_rewire
            simple_split_module._onestring_hard_nonpenetration_rewire_installed = True
    except Exception:
        pass

    def build_surface_parameterization_with_run_flag(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        active = mode in {"optcuts", "optcuts_grid"}
        grid_active = mode == "optcuts_grid"
        pipeline._onestring_optcuts_active_run = bool(active)
        pipeline._onestring_optcuts_grid_active_run = bool(grid_active)
        original_module = getattr(pipeline, "_original", None)
        if original_module is not None:
            try:
                original_module._onestring_optcuts_active_run = bool(active)
                original_module._onestring_optcuts_grid_active_run = bool(grid_active)
            except Exception:
                pass
        return base_builder(surface, target, grid, params)

    pipeline._build_surface_parameterization = build_surface_parameterization_with_run_flag
    original_module = getattr(pipeline, "_original", None)
    if original_module is not None:
        original_module._build_surface_parameterization = build_surface_parameterization_with_run_flag
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original_module, "build_onestring_design", None) if original_module is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = build_surface_parameterization_with_run_flag

    pipeline._onestring_optcuts_run_flag_patch_installed = True


__all__ = ["install_optcuts_run_flag_patch"]
