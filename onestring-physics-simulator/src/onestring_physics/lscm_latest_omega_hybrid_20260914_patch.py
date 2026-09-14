"""2026-09-14 hybrid with paper-faithful flat K2D semantics.

The hybrid keeps the latest OptCuts Omega and hard-planarity K3D stages, but the
flat stage no longer changes its numerical model when the number of tiles grows.

Paper Section 4.3 is represented in two implementation substeps:
1. the shared metric K2D solve enforces EEdge (K2D edge lengths = K3D targets),
   with the old M2D-position term disabled because it was incorrectly labelled
   EFab;
2. the independent-tile K2D layout enforces collision avoidance and the actual
   fabrication gap-angle bound theta_min <= theta <= 90 deg while preserving the
   K2D tile metric exactly by rigid SE(2) tile motion.

The old >150-tile fast path is deliberately bypassed.  The same whole-layout
solver, broad phase and objective are used for every tile count.  Spatial broad
phase is retained only as an acceleration structure; it does not select a
separate numerical formulation or cap collision pairs.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math
import os
from typing import Any

import numpy as np


MODE = "lscm_latest_omega_hard_k3d"
VERSION_ID = "2026-09-14-paper-eq5-unified-k2d"
VERSION_LABEL = "2026-09-14 — Paper Eq.5 K2D + latest Ω + latest K3D planarity"
OMEGA_LABEL = "2026-09-14 | Paper Eq.5 K2D + latest Ω + latest K3D planarity"
VERSION_DESCRIPTION = (
    "最新版Ωとhard-planarity K3Dを維持し、K2Dは論文Sec.4.3の意味に合わせる。"
    "EEdgeはK3D辺長一致、ECollisionは全tile数で同じSAT処理、EFabは実際のgap opening angle制約。"
    "150枚超で別fast pathへ切り替える処理は使用しない。"
)


def _active(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == MODE


def _wire(pipeline: Any, name: str, fn: Any) -> None:
    setattr(pipeline, name, fn)
    original = getattr(pipeline, "_original", None)
    if original is not None:
        setattr(original, name, fn)
    for build_fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb[name] = fn


def _clone_params(params: Any, **updates: Any) -> Any:
    try:
        return replace(params, **updates)
    except Exception:
        cloned = copy.copy(params)
        for key, value in updates.items():
            try:
                setattr(cloned, key, value)
            except Exception:
                try:
                    object.__setattr__(cloned, key, value)
                except Exception:
                    pass
        return cloned


def _rotation_matrix(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _rotate_tile(tile: np.ndarray, pivot: np.ndarray, angle: float) -> np.ndarray:
    r = _rotation_matrix(angle)
    return (np.asarray(tile, dtype=float) - pivot) @ r.T + pivot


def _hinge_constraints(pipeline: Any, faces: np.ndarray) -> list[tuple[int, int, int, int]]:
    specs = pipeline._vertex_hinge_specs_from_faces(faces)
    return [
        (int(spec.tile_a), int(spec.corner_a0), int(spec.tile_b), int(spec.corner_b0))
        for spec in specs
    ]


def _gap_angle(tile_a: np.ndarray, ca: int, tile_b: np.ndarray, cb: int) -> tuple[float, float]:
    """Return the smallest boundary-ray opening angle and its orientation sign."""
    na = [((ca - 1) % 4), ((ca + 1) % 4)]
    nb = [((cb - 1) % 4), ((cb + 1) % 4)]
    pa = np.asarray(tile_a[ca], dtype=float)
    pb = np.asarray(tile_b[cb], dtype=float)
    best_angle = math.pi
    best_sign = 1.0
    for ia in na:
        va = np.asarray(tile_a[ia], dtype=float) - pa
        nva = float(np.linalg.norm(va))
        if nva <= 1e-12:
            continue
        va /= nva
        for ib in nb:
            vb = np.asarray(tile_b[ib], dtype=float) - pb
            nvb = float(np.linalg.norm(vb))
            if nvb <= 1e-12:
                continue
            vb /= nvb
            angle = math.acos(float(np.clip(np.dot(va, vb), -1.0, 1.0)))
            if angle < best_angle:
                best_angle = angle
                cross = float(va[0] * vb[1] - va[1] * vb[0])
                best_sign = 1.0 if cross >= 0.0 else -1.0
    return float(best_angle), float(best_sign)


def _paper_flat_energy(
    pipeline: Any,
    layout: np.ndarray,
    constraints: list[tuple[int, int, int, int]],
    *,
    tile_scale: float,
    theta_min: float,
    theta_max: float,
    w_collision: float,
    w_fab: float,
) -> tuple[float, dict[str, float | int]]:
    # Hinge coincidence is a topological constraint of the linkage rather than
    # one of Eq.(5)'s three soft terms.  Keep a dimensionless diagnostic/barrier
    # so a collision-free but disconnected layout is never accepted as "better".
    hinge_sq: list[float] = []
    angle_violation_sq: list[float] = []
    angles: list[float] = []
    for ia, ca, ib, cb in constraints:
        if ia >= len(layout) or ib >= len(layout):
            continue
        d = float(np.linalg.norm(layout[ia, ca] - layout[ib, cb])) / max(tile_scale, 1e-12)
        hinge_sq.append(d * d)
        angle, _ = _gap_angle(layout[ia], ca, layout[ib], cb)
        angles.append(angle)
        if angle < theta_min:
            angle_violation_sq.append((theta_min - angle) ** 2)
        elif angle > theta_max:
            angle_violation_sq.append((angle - theta_max) ** 2)
        else:
            angle_violation_sq.append(0.0)

    # Same collision model for every tile count.  Broad phase only removes pairs
    # that cannot interact; unlike the deleted fast path there is no pair cap and
    # no size-dependent change of objective.
    pairs = pipeline._spatial_candidate_pairs_for_tiles(
        np.asarray(layout, dtype=float),
        pad=max(tile_scale * 0.12, 1e-5),
    )
    penetration_sq: list[float] = []
    collision_count = 0
    min_clearance = float("inf")
    for i, j in pairs:
        overlap, _mtv, signed = pipeline._sat_polygon_mtv(layout[i], layout[j], clearance=0.0)
        min_clearance = min(min_clearance, float(signed))
        if overlap:
            collision_count += 1
            p = max(0.0, -float(signed)) / max(tile_scale, 1e-12)
            penetration_sq.append(p * p)

    e_conn = float(np.mean(hinge_sq)) if hinge_sq else 0.0
    e_collision = float(np.mean(penetration_sq)) if penetration_sq else 0.0
    e_fab = float(np.mean(angle_violation_sq)) if angle_violation_sq else 0.0
    # Large barrier is only for the exact linkage-connectivity constraint; the
    # three paper soft terms retain their requested weights.
    total = 100.0 * e_conn + float(w_collision) * e_collision + float(w_fab) * e_fab
    return float(total), {
        "hinge_rms_normalized": math.sqrt(max(e_conn, 0.0)),
        "collision_energy_normalized": e_collision,
        "fabrication_angle_energy": e_fab,
        "collision_count": int(collision_count),
        "candidate_pair_count": int(len(pairs)),
        "min_clearance": float(min_clearance if np.isfinite(min_clearance) else 0.0),
        "min_gap_angle_deg": float(math.degrees(min(angles))) if angles else 0.0,
        "max_gap_angle_deg": float(math.degrees(max(angles))) if angles else 0.0,
    }


def _refine_paper_flat_layout(pipeline: Any, mesh: Any, initial_layout: Any, params: Any) -> Any:
    rest = pipeline._tiles_from_mesh_vertices(mesh.vertices, mesh.faces)[:, :, :2].copy()
    current = np.asarray(initial_layout.tile_top_vertices_2d, dtype=float).copy()
    if current.shape != rest.shape:
        current = rest.copy()
    if len(current) == 0:
        return initial_layout

    constraints = _hinge_constraints(pipeline, mesh.faces)
    edge_lengths = []
    for tile in rest:
        for i in range(4):
            edge_lengths.append(float(np.linalg.norm(tile[(i + 1) % 4] - tile[i])))
    tile_scale = max(float(np.median(edge_lengths)) if edge_lengths else 1.0, 1e-8)

    theta_min_deg = float(getattr(params, "k2d_min_gap_angle_degrees", 5.0))
    theta_min_deg = float(np.clip(theta_min_deg, 0.0, 89.0))
    theta_min = math.radians(theta_min_deg)
    theta_max = math.pi / 2.0
    w_collision = max(0.0, float(getattr(params, "w_collision", 1.0)))
    w_fab = max(0.0, float(getattr(params, "w_fab", 0.001)))

    # Fixed numerical model and fixed iteration policy for every tile count.
    # No time budget, no tile-count threshold, no collision-pair cap.
    iterations = max(160, int(getattr(params, "max_2d_iterations", 40)) * 6)
    connection_projection_weight = 12.0
    fabrication_projection_weight = max(0.20, min(1.0, 200.0 * w_fab))
    collision_projection_weight = max(0.20, min(1.0, w_collision))

    anchor_centroid = np.mean(current.reshape(-1, 2), axis=0)
    best = current.copy()
    best_energy, best_stats = _paper_flat_energy(
        pipeline,
        best,
        constraints,
        tile_scale=tile_scale,
        theta_min=theta_min,
        theta_max=theta_max,
        w_collision=w_collision,
        w_fab=w_fab,
    )

    for _it in range(iterations):
        desired_sum = current.copy()
        desired_weight = np.ones(current.shape[:2], dtype=float)

        # Topological hinge coincidence.  This is intentionally strong and does
        # not ramp from zero: connectivity must not become weaker merely because
        # collision handling is active.
        for ia, ca, ib, cb in constraints:
            if ia >= len(current) or ib >= len(current):
                continue
            mid = 0.5 * (current[ia, ca] + current[ib, cb])
            desired_sum[ia, ca] += connection_projection_weight * mid
            desired_weight[ia, ca] += connection_projection_weight
            desired_sum[ib, cb] += connection_projection_weight * mid
            desired_weight[ib, cb] += connection_projection_weight

        # Actual paper EFab: penalize opening angles outside [theta_min, 90deg].
        for ia, ca, ib, cb in constraints:
            if ia >= len(current) or ib >= len(current):
                continue
            angle, sign = _gap_angle(current[ia], ca, current[ib], cb)
            if theta_min <= angle <= theta_max:
                continue
            target_angle = theta_min if angle < theta_min else theta_max
            delta = float(np.clip(target_angle - angle, -math.radians(12.0), math.radians(12.0)))
            pivot = 0.5 * (current[ia, ca] + current[ib, cb])
            target_a = _rotate_tile(current[ia], pivot, -0.5 * sign * delta)
            target_b = _rotate_tile(current[ib], pivot, +0.5 * sign * delta)
            desired_sum[ia] += fabrication_projection_weight * target_a
            desired_weight[ia] += fabrication_projection_weight
            desired_sum[ib] += fabrication_projection_weight * target_b
            desired_weight[ib] += fabrication_projection_weight

        # Paper ECollision with SAT.  The spatial broad phase is purely an
        # acceleration and is identical regardless of tile count.
        pairs = pipeline._spatial_candidate_pairs_for_tiles(
            current,
            pad=max(tile_scale * 0.12, 1e-5),
        )
        shifts = np.zeros((len(current), 2), dtype=float)
        counts = np.zeros(len(current), dtype=float)
        for i, j in pairs:
            overlap, mtv, _signed = pipeline._sat_polygon_mtv(current[i], current[j], clearance=0.0)
            if not overlap:
                continue
            shifts[i] += 0.5 * mtv
            shifts[j] -= 0.5 * mtv
            counts[i] += 1.0
            counts[j] += 1.0
        active = counts > 0.0
        if np.any(active):
            shifts[active] /= counts[active, None]
            desired_sum[active] += collision_projection_weight * (current[active] + shifts[active, None, :])
            desired_weight[active] += collision_projection_weight

        # Global rigid projection.  Because each rest K2D tile already carries
        # the K3D target edge lengths, rigid SE(2) fitting preserves EEdge exactly
        # during collision/fabrication optimization.
        proposal = current.copy()
        for tile_id in range(len(current)):
            weights = np.maximum(desired_weight[tile_id], 1e-12)
            targets = desired_sum[tile_id] / weights[:, None]
            proposal[tile_id] = pipeline._fit_rigid_2d_weighted(rest[tile_id], targets, weights)

        # Fix only the global translation gauge; there is no per-tile M2D anchor.
        proposal += anchor_centroid - np.mean(proposal.reshape(-1, 2), axis=0)

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            trial = current + alpha * (proposal - current)
            trial_energy, trial_stats = _paper_flat_energy(
                pipeline,
                trial,
                constraints,
                tile_scale=tile_scale,
                theta_min=theta_min,
                theta_max=theta_max,
                w_collision=w_collision,
                w_fab=w_fab,
            )
            if trial_energy <= best_energy * 1.0005:
                current = trial
                accepted = True
                if trial_energy < best_energy:
                    best = trial.copy()
                    best_energy = trial_energy
                    best_stats = trial_stats
                break
        if not accepted:
            # Do not switch algorithms.  Simply continue alternating projections
            # with the same objective; later constraints may make a step feasible.
            current = 0.75 * current + 0.25 * proposal

    current = best
    hinge_specs = pipeline._vertex_hinge_specs_from_faces(mesh.faces)
    edge_specs = pipeline._edge_gap_specs_from_faces(mesh.faces)
    hinge_pairs = [(int(spec.tile_a), int(spec.tile_b)) for spec in hinge_specs]
    gap_polygons: list[np.ndarray] = []
    for spec in edge_specs:
        if spec.direction == "x":
            a_edge = current[spec.tile_a, [1, 2]]
            b_edge = current[spec.tile_b, [0, 3]]
        else:
            a_edge = current[spec.tile_a, [3, 2]]
            b_edge = current[spec.tile_b, [0, 1]]
        gap_polygons.append(np.vstack([a_edge[0], a_edge[1], b_edge[1], b_edge[0]]))

    metrics = dict(getattr(initial_layout, "metrics", {}) or {})
    metrics.update(
        {
            "version_id": VERSION_ID,
            "layout_type": "paper Eq.5 independent K2D linkage",
            "paper_eq5_flat_layout": True,
            "paper_eq5_EEdge": "preserved exactly by rigid SE(2) motion of K2D tiles whose edges already match K3D",
            "paper_eq5_ECollision": "SAT non-penetration, same model for every tile count",
            "paper_eq5_EFab": "actual gap opening-angle bound",
            "paper_eq5_theta_min_deg": float(theta_min_deg),
            "paper_eq5_theta_max_deg": 90.0,
            "paper_eq5_no_m2d_position_fab_surrogate": True,
            "paper_eq5_no_tile_count_fast_path": True,
            "paper_eq5_no_pair_cap": True,
            "paper_eq5_no_time_budget": True,
            "paper_eq5_iterations": int(iterations),
            "paper_eq5_energy_after": float(best_energy),
            "paper_eq5_hinge_rms_normalized_after": float(best_stats.get("hinge_rms_normalized", 0.0)),
            "paper_eq5_collision_energy_after": float(best_stats.get("collision_energy_normalized", 0.0)),
            "paper_eq5_fabrication_angle_energy_after": float(best_stats.get("fabrication_angle_energy", 0.0)),
            "paper_eq5_collision_count_after": int(best_stats.get("collision_count", 0)),
            "paper_eq5_min_gap_angle_deg_after": float(best_stats.get("min_gap_angle_deg", 0.0)),
            "paper_eq5_max_gap_angle_deg_after": float(best_stats.get("max_gap_angle_deg", 0.0)),
            "tile_count": int(len(current)),
            "vertices_per_tile": 4,
            "k2d_gap_count": int(len(gap_polygons)),
            "hinge_pair_count": int(len(hinge_pairs)),
        }
    )
    return type(initial_layout)(
        tile_top_vertices_2d=current,
        tile_ids=list(range(len(current))),
        hinge_pairs=hinge_pairs,
        gap_polygons=gap_polygons,
        metrics=metrics,
    )


def _install_selector_patch() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_lscm_latest_omega_hybrid_selector_installed", False):
        return
    original_selectbox = st.selectbox

    def selectbox_with_hybrid(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label == "version":
            if len(args) >= 2:
                options = list(args[1])
                if not any(isinstance(v, dict) and v.get("id") == VERSION_ID for v in options):
                    options.append({"id": VERSION_ID, "label": VERSION_LABEL, "description": VERSION_DESCRIPTION})
                kwargs = {**kwargs, "index": len(options) - 1}
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if not any(isinstance(v, dict) and v.get("id") == VERSION_ID for v in options):
                    options.append({"id": VERSION_ID, "label": VERSION_LABEL, "description": VERSION_DESCRIPTION})
                kwargs = {**kwargs, "options": options, "index": len(options) - 1}
        if label == "Omega parameterization mode":
            if len(args) >= 2:
                options = list(args[1])
                if OMEGA_LABEL not in options:
                    options.append(OMEGA_LABEL)
                kwargs = {**kwargs, "index": options.index(OMEGA_LABEL)}
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if OMEGA_LABEL not in options:
                    options.append(OMEGA_LABEL)
                kwargs = {**kwargs, "options": options, "index": options.index(OMEGA_LABEL)}
        selected = original_selectbox(*args, **kwargs)
        if label == "Omega parameterization mode" and selected == OMEGA_LABEL:
            try:
                st.caption(
                    "Paper Eq.5 K2D: K3D edge matching + collision avoidance + actual gap-angle fabrication bound. "
                    "The same solver is used for all tile counts; the old >150 fast path is disabled."
                )
            except Exception:
                pass
            return MODE
        return selected

    st.selectbox = selectbox_with_hybrid
    st._onestring_lscm_latest_omega_hybrid_selector_installed = True


def install_lscm_latest_omega_hybrid_patch(
    pipeline: Any,
    *,
    lscm_build_m2d: Any,
    lscm_optimize_k2d: Any,
    lscm_make_flat_tile_layout: Any,
) -> None:
    if getattr(pipeline, "_onestring_lscm_latest_omega_hybrid_installed", False):
        return

    latest_parameterization = pipeline._build_surface_parameterization
    latest_k3d = pipeline._optimize_k3d
    # This is the pre-fast-path whole-layout implementation from the backed-up
    # ordinary pipeline.  It is used as an initializer for every tile count.
    paper_initializer = getattr(pipeline, "_ORIGINAL_MAKE_FLAT_TILE_LAYOUT", lscm_make_flat_tile_layout)

    def make_dispatches(
        *,
        fallback_parameterization: Any,
        fallback_m2d: Any,
        fallback_k3d: Any,
        fallback_k2d: Any,
        fallback_flat_layout: Any,
    ) -> dict[str, Any]:
        def parameterization_dispatch(surface: Any, target: Any, grid: Any, params: Any):
            if not _active(params):
                return fallback_parameterization(surface, target, grid, params)
            latest_params = _clone_params(params, omega_parameterization_mode="optcuts_test")
            result = latest_parameterization(surface, target, grid, latest_params)
            try:
                result.method = MODE
                result.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "hybrid_lscm_baseline": True,
                        "hybrid_stage_s_to_omega": "current latest OptCuts-test Omega",
                        "hybrid_stage_m2d": "captured ordinary LSCM M2D",
                        "hybrid_stage_k3d": "current latest OptCuts-test2 K3D/hard-planarity stack",
                        "hybrid_stage_k2d": "paper Eq.5 split implementation",
                    }
                )
            except Exception:
                pass
            print("[LSCM-HYBRID-OMEGA] latest OptCuts Omega used")
            return result

        def m2d_dispatch(grid: Any, domain: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_m2d(grid, domain, params)
            result = lscm_build_m2d(grid, domain, params)
            try:
                result.metrics.update({"version_id": VERSION_ID, "hybrid_m2d_exact_lscm_function": True})
            except Exception:
                pass
            return result

        def k3d_dispatch(target: Any, mesh: Any, parameterization: Any, params: Any):
            if not _active(params):
                return fallback_k3d(target, mesh, parameterization, params)
            previous_variant = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT")
            try:
                os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "2"
                latest_params = _clone_params(params, omega_parameterization_mode="optcuts_test")
                result, report = latest_k3d(target, mesh, parameterization, latest_params)
            finally:
                if previous_variant is None:
                    os.environ.pop("ONESTRING_OPTCUTS_TEST_VARIANT", None)
                else:
                    os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = previous_variant
            try:
                result.metrics.update({"version_id": VERSION_ID, "hybrid_k3d_latest_planarity_stack": True})
            except Exception:
                pass
            return result, report

        def k2d_dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return fallback_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
            # The old code used w_fab * ||x-M2D||^2.  That is not the paper's
            # fabrication term.  Disable it here; actual EFab is the gap-angle
            # term solved in the independent flat-layout substep below.
            metric_params = _clone_params(params, w_fab=0.0)
            result, report = lscm_optimize_k2d(
                mesh_2d,
                mesh_3d,
                metric_params,
                progress_callback=progress_callback,
            )
            try:
                result.metrics.update(
                    {
                        "version_id": VERSION_ID,
                        "objective": "Paper Eq.5 split: EEdge here; ECollision + actual EFab gap angle in independent K2D layout",
                        "paper_eq5_EEdge_stage": True,
                        "paper_eq5_old_position_anchor_disabled": True,
                        "paper_eq5_old_position_anchor_was_not_EFab": True,
                        "paper_eq5_collision_fab_deferred_to_independent_linkage": True,
                    }
                )
                report.objective = str(result.metrics["objective"])
            except Exception:
                pass
            print("[PAPER-EQ5-K2D] EEdge metric solve; old M2D-position 'EFab' disabled")
            return result, report

        def flat_layout_dispatch(mesh: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_flat_layout(mesh, params)
            # Use the same full solver as initializer for every tile count.  Make
            # its numerical guards non-size-dependent as well.
            init_params = _clone_params(
                params,
                hinge_layout_time_budget_sec=0.0,
                hinge_layout_max_candidate_pairs=1_000_000_000,
            )
            initial = paper_initializer(mesh, init_params)
            layout = _refine_paper_flat_layout(pipeline, mesh, initial, params)
            print(
                "[PAPER-EQ5-FLAT] unified all-tile solver; "
                f"faces={len(getattr(mesh, 'faces', []))}; tile-count fast path disabled"
            )
            return layout

        return {
            "_build_surface_parameterization": parameterization_dispatch,
            "_build_m2d": m2d_dispatch,
            "_optimize_k3d": k3d_dispatch,
            "_optimize_k2d": k2d_dispatch,
            "_make_flat_tile_layout": flat_layout_dispatch,
        }

    def install_routes(target_pipeline: Any) -> None:
        fallbacks = {
            "fallback_parameterization": target_pipeline._build_surface_parameterization,
            "fallback_m2d": target_pipeline._build_m2d,
            "fallback_k3d": target_pipeline._optimize_k3d,
            "fallback_k2d": target_pipeline._optimize_k2d,
            "fallback_flat_layout": target_pipeline._make_flat_tile_layout,
        }
        for name, fn in make_dispatches(**fallbacks).items():
            _wire(target_pipeline, name, fn)

    install_routes(pipeline)

    try:
        from . import simple_split_panel_patch as simple_split_module
        if not getattr(simple_split_module, "_onestring_lscm_hybrid_rewire_installed", False):
            original_installer = simple_split_module.install_simple_split_panel_patch

            def install_then_rewire(pipeline_module: Any, optimization_debug_module: Any) -> None:
                original_installer(pipeline_module, optimization_debug_module)
                install_routes(pipeline_module)
                print("[PAPER-EQ5-ROUTE] hybrid routing reinstalled after Simple Split")

            simple_split_module.install_simple_split_panel_patch = install_then_rewire
            simple_split_module._onestring_lscm_hybrid_rewire_installed = True
    except Exception:
        pass

    pipeline._onestring_lscm_latest_omega_hybrid_installed = True


def install_deferred_hybrid_hook(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_lscm_hybrid_deferred_hook_installed", False):
        return
    lscm_build_m2d = pipeline._build_m2d
    lscm_optimize_k2d = pipeline._optimize_k2d
    lscm_make_flat_tile_layout = pipeline._make_flat_tile_layout
    _install_selector_patch()
    try:
        from . import optcuts_test2_acceleration_patch as acceleration_module
    except Exception:
        return
    original_installer = acceleration_module.install_optcuts_test2_acceleration_patch

    def install_acceleration_then_hybrid(target_pipeline: Any) -> None:
        original_installer(target_pipeline)
        install_lscm_latest_omega_hybrid_patch(
            target_pipeline,
            lscm_build_m2d=lscm_build_m2d,
            lscm_optimize_k2d=lscm_optimize_k2d,
            lscm_make_flat_tile_layout=lscm_make_flat_tile_layout,
        )
        print("[PAPER-EQ5-INSTALL] unified paper-style K2D installed")

    acceleration_module.install_optcuts_test2_acceleration_patch = install_acceleration_then_hybrid
    pipeline._onestring_lscm_hybrid_deferred_hook_installed = True


__all__ = [
    "MODE",
    "VERSION_ID",
    "VERSION_LABEL",
    "OMEGA_LABEL",
    "VERSION_DESCRIPTION",
    "install_deferred_hybrid_hook",
    "install_lscm_latest_omega_hybrid_patch",
]
