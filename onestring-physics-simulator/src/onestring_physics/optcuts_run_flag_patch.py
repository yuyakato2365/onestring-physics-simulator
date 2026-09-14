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


# Install before any OptCuts run. Ordinary ``optcuts`` must never resolve to the
# Grid-OptCuts executable; the latter is reserved for ``optcuts_grid`` only.
install_official_optcuts_binary_separation()

# Install before any Grid-OptCuts run. Orientation stabilizes the backend's
# global frame first; seam transfer then wraps the native pipeline call and
# applies the identical final rigid frame transform to the C++ cohE sidecar.
install_optcuts_grid_orientation_patch()
install_optcuts_grid_seam_sidecar_patch()


def _install_optcuts_weight_ui_patch() -> None:
    """Loosen K3D weight limits, numeric E_Conn, and a hard-collision version."""
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
            return original_number_input(label, *args, **kwargs)
        if label == "w_square / ESquare":
            kwargs["max_value"] = 1_000.0
            return original_number_input(label, *args, **kwargs)
        return original_number_input(label, *args, **kwargs)

    def patched_slider(label: str, *args: Any, **kwargs: Any):
        if label == "hinge connection weight":
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
        return original_slider(label, *args, **kwargs)

    def patched_selectbox(label: str, options: Any, *args: Any, **kwargs: Any):
        if label != "version":
            return original_selectbox(label, options, *args, **kwargs)
        option_list = list(options)
        if not any(isinstance(v, dict) and v.get("id") == HARD_COLLISION_VERSION_ID for v in option_list):
            template = next(
                (dict(v) for v in reversed(option_list) if isinstance(v, dict) and v.get("id") == "2026-09-14-paper-eq5-unified-k2d"),
                {},
            )
            template.update(
                {
                    "id": HARD_COLLISION_VERSION_ID,
                    "label": HARD_COLLISION_VERSION_LABEL,
                    "description": (
                        "Paper Eq.5 K2D版を維持し、flat linkageの最終解に非貫通をハード制約として要求する。"
                        "衝突が残る場合はsoft penaltyで妥協せず明示的に失敗する。"
                    ),
                }
            )
            option_list.append(template)
        selected = original_selectbox(label, option_list, *args, **kwargs)
        enabled = isinstance(selected, dict) and selected.get("id") == HARD_COLLISION_VERSION_ID
        os.environ["ONESTRING_HARD_NONPENETRATION"] = "1" if enabled else "0"
        if enabled:
            st.caption("Hard non-penetration: overlapping tile interiors are projected apart; a layout is accepted only when the final overlap count is zero.")
        return selected

    st.number_input = patched_number_input
    st.slider = patched_slider
    st.selectbox = patched_selectbox
    st._onestring_optcuts_weight_ui_patch_installed = True


_install_optcuts_weight_ui_patch()


def _sat_mtv(poly_a: np.ndarray, poly_b: np.ndarray, tolerance: float) -> np.ndarray | None:
    """Return the minimum translation vector that moves A out of convex B."""
    a = np.asarray(poly_a, dtype=float)[:, :2]
    b = np.asarray(poly_b, dtype=float)[:, :2]
    best_axis = None
    best_overlap = float("inf")
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for edge in edges:
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                continue
            axis /= norm
            pa = a @ axis
            pb = b @ axis
            overlap = min(float(np.max(pa)), float(np.max(pb))) - max(float(np.min(pa)), float(np.min(pb)))
            # Touching at a hinge/edge is feasible; only positive-area overlap is collision.
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
    n = len(tiles)
    for i in range(n - 1):
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


def _project_hard_nonpenetration(layout: Any) -> Any:
    """Rigidly translate tiles until no interior overlap remains, else fail.

    This is a feasibility projection, not a large collision penalty.  Tile shape
    is unchanged exactly because every correction is a rigid translation.  The
    existing E_Conn/E_Fab solution is used as the initial point; collision is the
    hard acceptance condition for this version.
    """
    tiles = np.asarray(layout.tile_top_vertices_2d, dtype=float).copy()
    if len(tiles) < 2:
        return layout
    edge_lengths = np.linalg.norm(np.roll(tiles, -1, axis=1) - tiles, axis=2)
    scale = float(np.median(edge_lengths[edge_lengths > 1e-12])) if np.any(edge_lengths > 1e-12) else 1.0
    tolerance = max(1e-10, scale * 1e-7)
    max_sweeps = 240
    initial_count, initial_depth = _count_overlaps(tiles, tolerance)
    sweep = 0
    for sweep in range(1, max_sweeps + 1):
        corrections = np.zeros((len(tiles), 2), dtype=float)
        hits = np.zeros(len(tiles), dtype=float)
        overlap_count = 0
        for i, j in _aabb_candidate_pairs(tiles, tolerance):
            mtv = _sat_mtv(tiles[i], tiles[j], tolerance)
            if mtv is None:
                continue
            overlap_count += 1
            corrections[i] += 0.5 * mtv
            corrections[j] -= 0.5 * mtv
            hits[i] += 1.0
            hits[j] += 1.0
        if overlap_count == 0:
            break
        active = hits > 0
        corrections[active] /= hits[active, None]
        # Damping avoids pairwise corrections overshooting each other in dense clusters.
        tiles[active] += 0.9 * corrections[active, None, :]
    final_count, final_depth = _count_overlaps(tiles, tolerance)
    if final_count != 0:
        raise RuntimeError(
            "HARD_NONPENETRATION_INFEASIBLE: "
            f"{final_count} overlapping tile pairs remain after {max_sweeps} feasibility sweeps "
            f"(max penetration proxy={final_depth:.6g})."
        )
    layout.tile_top_vertices_2d = tiles
    try:
        layout.metrics.update(
            {
                "hard_nonpenetration_enabled": True,
                "hard_nonpenetration_method": "SAT minimum-translation feasibility projection",
                "hard_nonpenetration_initial_overlap_pairs": int(initial_count),
                "hard_nonpenetration_initial_max_depth": float(initial_depth),
                "hard_nonpenetration_final_overlap_pairs": 0,
                "hard_nonpenetration_final_max_depth": float(final_depth),
                "hard_nonpenetration_sweeps": int(sweep),
                "hard_nonpenetration_acceptance": "final overlap count must equal zero",
            }
        )
    except Exception:
        pass
    print(
        f"[HARD-NONPENETRATION] initial={initial_count} final=0 sweeps={sweep} "
        f"initial_depth={initial_depth:.6g}"
    )
    return layout


def install_optcuts_run_flag_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_run_flag_patch_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    # The Sept-14 paper-faithful hybrid later captures this exact entry point as
    # its whole-layout solver.  Wrapping it here therefore adds the optional hard
    # feasibility stage without changing the successful base version.
    base_original_flat = getattr(pipeline, "_ORIGINAL_MAKE_FLAT_TILE_LAYOUT", None)
    if callable(base_original_flat) and not getattr(base_original_flat, "_onestring_hard_nonpenetration_wrapper", False):
        def original_flat_with_optional_hard_collision(mesh: Any, params: Any = None):
            layout = base_original_flat(mesh, params)
            if os.environ.get("ONESTRING_HARD_NONPENETRATION", "0").lower() in {"1", "true", "yes", "on"}:
                return _project_hard_nonpenetration(layout)
            return layout
        original_flat_with_optional_hard_collision._onestring_hard_nonpenetration_wrapper = True
        pipeline._ORIGINAL_MAKE_FLAT_TILE_LAYOUT = original_flat_with_optional_hard_collision

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
