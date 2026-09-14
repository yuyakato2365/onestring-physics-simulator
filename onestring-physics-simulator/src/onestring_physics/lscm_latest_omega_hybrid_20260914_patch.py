"""Paper-faithful 2026-09-14 hybrid: latest Omega/K3D + unified K2D.

Important invariants:
- no tile-count-dependent numerical branch;
- M2D->K2D metric solve is EEdge only and is vectorized O(E) per iteration;
- the historical w_fab*||x-M2D||^2 surrogate is not used as EFab;
- independent flat placement always uses the same pre-fast-path whole-layout solver;
- the active mode is wired after the OptCuts acceleration stack and again after
  Simple Split so stale outer wrappers cannot silently restore the old K2D path.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import os
from typing import Any

import numpy as np

MODE = "lscm_latest_omega_hard_k3d"
VERSION_ID = "2026-09-14-paper-eq5-unified-k2d"
VERSION_LABEL = "2026-09-14 — Paper Eq.5 K2D + latest Ω + latest K3D planarity"
OMEGA_LABEL = "2026-09-14 | Paper Eq.5 K2D + latest Ω + latest K3D planarity"
VERSION_DESCRIPTION = (
    "最新版Ωとhard-planarity K3Dを維持し、K2Dはtile数に依存しない同一経路を使う。"
    "56%のM2D→K2Dはvectorized EEdgeのみを解き、collision/fabricationはflat linkage側で扱う。"
)


def _active(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == MODE


def _clone_params(params: Any, **updates: Any) -> Any:
    try:
        return replace(params, **updates)
    except Exception:
        out = copy.copy(params)
        for key, value in updates.items():
            try:
                setattr(out, key, value)
            except Exception:
                try:
                    object.__setattr__(out, key, value)
                except Exception:
                    pass
        return out


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


def _emit(pipeline: Any, cb: Any, stage: str, progress: float, detail: str) -> None:
    fn = getattr(pipeline, "_emit_progress", None)
    if callable(fn):
        fn(cb, stage, progress, detail)
    elif cb is not None:
        try:
            cb(stage, progress, detail)
        except Exception:
            pass


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    e = np.concatenate(
        [f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 3]], f[:, [3, 0]]], axis=0
    )
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _paper_eedge_k2d(
    pipeline: Any,
    mesh_2d: Any,
    mesh_3d: Any,
    params: Any,
    progress_callback: Any = None,
):
    """Fast shared-mesh EEdge solve, identical for every tile count.

    This deliberately does *not* call the legacy _optimize_k2d because that
    function performs Python collision diagnostics before/after the solve and
    contains historical size-dependent branches.  Those were the 56% stall on
    gridSize=20.  The numerical objective here is exactly the metric part needed
    before the independent linkage solve:
        sum_e (||x_i-x_j|| - L_e(K3D))^2.
    """
    start_xy = np.asarray(mesh_2d.vertices[:, :2], dtype=np.float64)
    xy = start_xy.copy()
    edges = _unique_edges(mesh_2d.faces)
    if len(edges):
        a = edges[:, 0]
        b = edges[:, 1]
        target = np.linalg.norm(
            np.asarray(mesh_3d.vertices, dtype=np.float64)[a]
            - np.asarray(mesh_3d.vertices, dtype=np.float64)[b],
            axis=1,
        )
        degree = np.bincount(
            np.concatenate([a, b]), minlength=len(xy)
        ).astype(np.float64)[:, None]
        degree = np.maximum(degree, 1.0)
    else:
        a = b = np.zeros(0, dtype=np.int64)
        target = np.zeros(0, dtype=np.float64)
        degree = np.ones((len(xy), 1), dtype=np.float64)

    iterations = max(80, int(getattr(params, "max_2d_iterations", 40)) * 4)
    tol = max(1e-8, (float(np.mean(target)) if len(target) else 1.0) * 2e-3)
    center0 = np.mean(start_xy, axis=0, keepdims=True) if len(start_xy) else np.zeros((1, 2))
    before = (
        float(np.mean(np.abs(np.linalg.norm(xy[a] - xy[b], axis=1) - target)))
        if len(edges)
        else 0.0
    )

    _emit(pipeline, progress_callback, "Paper K2D EEdge", 0.02, f"{len(xy)} vertices, {len(edges)} edges")
    done = 0
    max_err = 0.0
    for it in range(iterations):
        if not len(edges):
            break
        delta = xy[b] - xy[a]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1e-12)
        correction = 0.5 * ((length - target) / safe)[:, None] * delta
        accum = np.zeros_like(xy)
        np.add.at(accum, a, correction)
        np.add.at(accum, b, -correction)
        xy += accum / degree
        xy += center0 - np.mean(xy, axis=0, keepdims=True)
        done = it + 1
        if done == 1 or done % 10 == 0 or done == iterations:
            err = np.abs(np.linalg.norm(xy[a] - xy[b], axis=1) - target)
            max_err = float(np.max(err)) if len(err) else 0.0
            _emit(
                pipeline,
                progress_callback,
                "Paper K2D EEdge",
                min(0.98, done / max(1, iterations)),
                f"iter {done}/{iterations}, max edge error={max_err:.4g}",
            )
            if max_err <= tol:
                break

    after_err = (
        np.abs(np.linalg.norm(xy[a] - xy[b], axis=1) - target)
        if len(edges)
        else np.zeros(0)
    )
    after = float(np.mean(after_err)) if len(after_err) else 0.0
    max_after = float(np.max(after_err)) if len(after_err) else 0.0
    vertices = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
    metrics = dict(getattr(mesh_2d, "metrics", {}) or {})
    metrics.update(
        {
            "version_id": VERSION_ID,
            "objective": "Paper Eq.5 split: EEdge metric solve; ECollision/EFab in independent flat linkage",
            "paper_eq5_EEdge_stage": True,
            "paper_eq5_old_position_anchor_disabled": True,
            "paper_eq5_no_tile_count_branch": True,
            "paper_eq5_eedge_solver": "vectorized Jacobi/projective edge-length projection",
            "paper_eq5_eedge_iterations": int(done),
            "edge_matching_error_before": float(before),
            "edge_matching_error_after": float(after),
            "edge_matching_error": float(after),
            "max_edge_length_error_after": float(max_after),
            "actual_backend": "vectorized_numpy_eedge",
            "k2d_collision_relax_skipped_for_speed": False,
            "collision_projection": "not part of shared metric solve; handled in independent flat linkage for every tile count",
        }
    )
    out = type(mesh_2d)(
        vertices,
        np.asarray(mesh_2d.faces, dtype=int).copy(),
        mesh_2d.grid,
        "K2D",
        metrics,
        list(getattr(mesh_2d, "split_lines", [])),
    )
    report_cls = getattr(pipeline, "StageReport", None)
    if report_cls is None:
        report = None
    else:
        counts_fn = getattr(pipeline, "_mesh_counts", None)
        counts = counts_fn(out) if callable(counts_fn) else {}
        report = report_cls(
            name="M2D -> K2D",
            objective=str(metrics["objective"]),
            before_error=float(before),
            after_error=float(after),
            constraint_violation=float(max_after),
            computation_time=0.0,
            counts=counts,
        )
    _emit(pipeline, progress_callback, "Paper K2D EEdge", 1.0, f"done; mean edge error={after:.4g}")
    print(
        f"[PAPER-EQ5-K2D] vectorized EEdge done vertices={len(xy)} edges={len(edges)} "
        f"iterations={done} mean_error={after:.6g} max_error={max_after:.6g}"
    )
    return out, report


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
                    "Paper Eq.5 K2D: vectorized EEdge at 56%, then the same whole-layout flat solver for every tile count."
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
    # Captured before the size-dependent wrapper was installed.  This is the
    # same whole-layout formulation regardless of tile count.
    paper_flat_layout = getattr(pipeline, "_ORIGINAL_MAKE_FLAT_TILE_LAYOUT", lscm_make_flat_tile_layout)

    def make_dispatches(*, fallback_parameterization: Any, fallback_m2d: Any, fallback_k3d: Any, fallback_k2d: Any, fallback_flat_layout: Any):
        def parameterization_dispatch(surface: Any, target: Any, grid: Any, params: Any):
            if not _active(params):
                return fallback_parameterization(surface, target, grid, params)
            latest_params = _clone_params(params, omega_parameterization_mode="optcuts_test")
            result = latest_parameterization(surface, target, grid, latest_params)
            try:
                result.method = MODE
                result.metrics.update({
                    "version_id": VERSION_ID,
                    "hybrid_stage_s_to_omega": "current latest OptCuts-test Omega",
                    "hybrid_stage_m2d": "captured ordinary LSCM M2D",
                    "hybrid_stage_k3d": "current latest OptCuts-test2 K3D/hard-planarity stack",
                    "hybrid_stage_k2d": "vectorized paper EEdge + unified independent flat layout",
                })
            except Exception:
                pass
            return result

        def m2d_dispatch(grid: Any, domain: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_m2d(grid, domain, params)
            return lscm_build_m2d(grid, domain, params)

        def k3d_dispatch(target: Any, mesh: Any, parameterization: Any, params: Any):
            if not _active(params):
                return fallback_k3d(target, mesh, parameterization, params)
            previous_variant = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT")
            try:
                os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = "2"
                latest_params = _clone_params(params, omega_parameterization_mode="optcuts_test")
                result = latest_k3d(target, mesh, parameterization, latest_params)
            finally:
                if previous_variant is None:
                    os.environ.pop("ONESTRING_OPTCUTS_TEST_VARIANT", None)
                else:
                    os.environ["ONESTRING_OPTCUTS_TEST_VARIANT"] = previous_variant
            return result

        def k2d_dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return fallback_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
            return _paper_eedge_k2d(pipeline, mesh_2d, mesh_3d, params, progress_callback)

        def flat_layout_dispatch(mesh: Any, params: Any = None):
            if params is None or not _active(params):
                return fallback_flat_layout(mesh, params)
            # Same function and same parameter policy for every tile count.  A
            # fixed time budget is allowed; it is not selected from tile count.
            same_params = _clone_params(
                params,
                k2d_independent_fast_tile_threshold=10**12,
                hinge_layout_time_budget_sec=max(12.0, float(getattr(params, "hinge_layout_time_budget_sec", 8.0))),
                hinge_layout_max_candidate_pairs=1_000_000_000,
            )
            layout = paper_flat_layout(mesh, same_params)
            try:
                layout.metrics.update({
                    "version_id": VERSION_ID,
                    "paper_eq5_no_tile_count_fast_path": True,
                    "paper_eq5_whole_layout_same_solver_all_counts": True,
                    "paper_eq5_shared_metric_solver": "vectorized_numpy_eedge",
                })
            except Exception:
                pass
            print(f"[PAPER-EQ5-FLAT] same whole-layout solver faces={len(getattr(mesh, 'faces', []))}")
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
        print("[PAPER-EQ5-INSTALL] active-branch vectorized K2D installed")

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
