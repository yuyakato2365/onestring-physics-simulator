"""2026-09-14 paper-stage K2D variant for the OptCuts experimental path.

Section 4.3 of One String defines K2D as a shared-vertex planar mesh optimized
from M2D with Eq. (5): EFlat = w1 EEdge + w2 ECollision + w3 EFab. Rigid
per-tile repositioning for hinge alignment belongs to the later T2D / Section
4.4 stage. This patch keeps K2D shared-vertex and collision-aware; it does not
insert SE(2) hinge placement into K2D.

The 2026-09-14 variant also treats the Eq. (5) K2D relative panel placement as
authoritative after Split. Older validation code recenters every disconnected
K2D component back onto its M2D seam-gap centroid after K2D optimization. That
silently re-injects the parameterization-dependent M2D absolute layout and makes
LSCM/OptCuts downstream comparisons non-equivalent. For this dated variant only,
that post-K2D centroid realignment is bypassed. Older variants retain the legacy
behavior for comparison.

The main paper states that EFab penalizes gap opening angles outside
[theta_min, 90 deg], but delegates the exact formula to Supplement Appendix A.
That supplement is not available in the current source bundle, so this variant
marks that part as not paper-exact instead of inventing a formula.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


VARIANT = "3"
VERSION_ID = "2026-09-14-paper-k2d-eq5-stage-separated"


def _active(params: Any) -> bool:
    explicit = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == VARIANT
    latest_ui = os.environ.get("ONESTRING_PAPER_K2D_20260914", "0").strip().lower() in {"1", "true", "yes", "on"}
    return bool((explicit or latest_ui) and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test")


def _fn(pipeline: Any, name: str):
    value = getattr(pipeline, name, None)
    if value is None:
        value = getattr(getattr(pipeline, "_original", None), name, None)
    return value


def _paper_collision_refine(
    pipeline: Any,
    mesh_2d: Any,
    mesh_3d: Any,
    xy0: np.ndarray,
    params: Any,
    progress_callback=None,
):
    unique_edges = _fn(pipeline, "_unique_mesh_edges")
    edge_errors = _fn(pipeline, "_edge_matching_errors")
    edge_project = _fn(pipeline, "_projective_edge_match_2d")
    collision_relax = _fn(pipeline, "_relax_2d_collisions")
    tiles_from_mesh = _fn(pipeline, "_tiles_from_mesh_vertices")
    count_collisions = _fn(pipeline, "_count_2d_tile_collisions")
    emit = _fn(pipeline, "_emit_progress")

    required = (unique_edges, edge_errors, edge_project, collision_relax, tiles_from_mesh, count_collisions)
    if not all(callable(fn) for fn in required):
        return np.asarray(xy0, float).copy(), {
            "paper_k2d_collision_refinement_applied": False,
            "paper_k2d_collision_refinement_reason": "required base helpers unavailable",
        }

    faces = np.asarray(mesh_2d.faces, dtype=int)
    xy = np.asarray(xy0, dtype=float).copy()
    base_xy = np.asarray(mesh_2d.vertices, dtype=float)[:, :2].copy()
    edges = unique_edges(faces)
    k3 = np.asarray(mesh_3d.vertices, dtype=float)
    target_lengths = np.asarray([np.linalg.norm(k3[a] - k3[b]) for a, b in edges], dtype=float)
    mean_target = float(np.mean(target_lengths)) if len(target_lengths) else 1.0
    edge_tol = max(1e-5, 0.004 * mean_target)
    budget = max(1.0, float(os.environ.get("ONESTRING_PAPER_K2D_TIME_BUDGET_SEC", "12.0")))
    outer_max = max(1, int(os.environ.get("ONESTRING_PAPER_K2D_OUTER", "12")))
    started = time.perf_counter()

    def collision_count(points: np.ndarray) -> int:
        verts3 = np.column_stack([points, np.zeros(len(points))])
        tiles = tiles_from_mesh(verts3, faces)
        return int(count_collisions(tiles, mesh_2d.grid))

    before_mean, before_max = edge_errors(xy, edges, target_lengths)
    before_collision = collision_count(xy)
    best = xy.copy()
    best_key = (before_collision, before_max, before_mean)
    used = 0

    # Keep ECollision active at K2D even for medium/large meshes. Each collision
    # step is followed by edge projection so the defining K3D edge targets remain
    # authoritative. This is still a lightweight approximation of the paper's
    # projection solver, not a claim of source-identical ShapeOp numerics.
    for outer in range(outer_max):
        candidate = edge_project(xy, base_xy, edges, target_lengths, faces, mesh_2d.grid, iterations=4)
        candidate = collision_relax(candidate, faces, mesh_2d.grid, iterations=1, weight=0.04)
        candidate = edge_project(candidate, base_xy, edges, target_lengths, faces, mesh_2d.grid, iterations=4)
        c_mean, c_max = edge_errors(candidate, edges, target_lengths)
        c_collision = collision_count(candidate)
        key = (c_collision, c_max, c_mean)
        if key <= best_key or c_max <= max(edge_tol, 1.05 * best_key[1]):
            xy = candidate
            if key < best_key:
                best = candidate.copy()
                best_key = key
        else:
            xy = 0.5 * (xy + candidate)
        used = outer + 1
        if callable(emit):
            emit(
                progress_callback,
                "Paper K2D Eq.(5) shared-vertex refinement",
                min(0.98, (outer + 1) / outer_max),
                f"outer {outer + 1}/{outer_max}; overlaps={c_collision}; max edge err={c_max:.4g}",
            )
        if best_key[0] == 0 and best_key[1] <= edge_tol:
            break
        if time.perf_counter() - started >= budget:
            break

    after_mean, after_max = edge_errors(best, edges, target_lengths)
    after_collision = collision_count(best)
    return best, {
        "paper_k2d_collision_refinement_applied": True,
        "paper_k2d_collision_refinement_model": "shared-vertex alternating K3D-edge projection + 2D non-penetration projection",
        "paper_k2d_collision_deferred_to_hinge_stage": False,
        "paper_k2d_collision_count_before_refinement": int(before_collision),
        "paper_k2d_collision_count_after_refinement": int(after_collision),
        "paper_k2d_edge_error_before_refinement_mean": float(before_mean),
        "paper_k2d_edge_error_before_refinement_max": float(before_max),
        "paper_k2d_edge_error_after_refinement_mean": float(after_mean),
        "paper_k2d_edge_error_after_refinement_max": float(after_max),
        "paper_k2d_refinement_outer_iterations": int(used),
        "paper_k2d_refinement_time_sec": float(time.perf_counter() - started),
        "paper_k2d_refinement_time_budget_sec": float(budget),
    }


def _install_split_post_k2d_realign_bypass() -> None:
    """Bypass legacy post-K2D M2D-centroid realignment only for 2026-09-14.

    ``app_split_panels`` installs Simple Split after the OptCuts launcher has
    already assembled the K2D wrapper stack. The legacy Simple Split installer
    wraps ``_optimize_k2d`` and, for multiple disconnected components, translates
    each optimized component so that its centroid matches the corresponding M2D
    seam-gap component. That changes the relative panel placement after Eq. (5).

    Capture the authoritative pre-Split K2D function before the legacy installer
    runs. After installation, dispatch 2026-09-14 runs directly to that captured
    function, while every older mode still uses the original Simple Split K2D
    wrapper unchanged.
    """
    try:
        from . import simple_split_panel_patch as simple_split_module
    except Exception:
        return
    if getattr(simple_split_module, "_onestring_20260914_k2d_realign_bypass_installed", False):
        return

    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_with_20260914_bypass(pipeline_module: Any, optimization_debug_module: Any) -> None:
        authoritative_k2d_before_split_wrapper = pipeline_module._optimize_k2d
        original_installer(pipeline_module, optimization_debug_module)
        legacy_split_k2d = pipeline_module._optimize_k2d

        def k2d_preserve_eq5_layout(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return legacy_split_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

            result, report = authoritative_k2d_before_split_wrapper(
                mesh_2d,
                mesh_3d,
                params,
                progress_callback=progress_callback,
            )
            metrics = dict(getattr(result, "metrics", {}) or {})
            metrics.update({
                "version_id": VERSION_ID,
                "k2d_split_panel_centroid_realign_disabled": True,
                "k2d_split_panel_post_eq5_translation_applied": False,
                "paper_k2d_relative_layout_preserved_after_split": True,
                "paper_k2d_absolute_m2d_layout_reinjected_after_optimization": False,
                "paper_k2d_split_layout_policy": "preserve authoritative Eq.(5) K2D coordinates; do not translate disconnected components back to M2D centroids",
            })
            try:
                result.metrics.clear()
                result.metrics.update(metrics)
            except Exception:
                pass

            copy_attrs = getattr(simple_split_module, "_copy_attrs", None)
            if callable(copy_attrs):
                try:
                    copy_attrs(mesh_2d, result)
                except Exception:
                    pass

            print(
                "[2026-09-14-PAPER-K2D-SPLIT] "
                "post-Eq5 M2D-centroid realignment disabled; relative K2D panel placement preserved"
            )
            return result, report

        pipeline_module._optimize_k2d = k2d_preserve_eq5_layout
        original = getattr(pipeline_module, "_original", None)
        if original is not None:
            original._optimize_k2d = k2d_preserve_eq5_layout
        for fn in (
            getattr(pipeline_module, "build_onestring_design", None),
            getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
            getattr(original, "build_onestring_design", None) if original is not None else None,
        ):
            glb = getattr(fn, "__globals__", None)
            if isinstance(glb, dict):
                glb["_optimize_k2d"] = k2d_preserve_eq5_layout

    simple_split_module.install_simple_split_panel_patch = install_with_20260914_bypass
    simple_split_module._onestring_20260914_k2d_realign_bypass_installed = True


def install_optcuts_paper_k2d_20260914_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_paper_k2d_20260914_installed", False):
        return

    # Must be installed before app_split_panels invokes Simple Split so the dated
    # variant can preserve the Eq. (5) K2D layout after topology separation.
    _install_split_post_k2d_realign_bypass()

    base = pipeline._optimize_k2d

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        if not _active(params):
            return base(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

        # Paper reproduction weights reported for Eq. (5).
        for name, value in (("w_edge", 1.0), ("w_collision", 1.0), ("w_fab", 0.001)):
            try:
                setattr(params, name, value)
            except Exception:
                try:
                    object.__setattr__(params, name, value)
                except Exception:
                    pass

        result, report = base(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        refined_xy, extra = _paper_collision_refine(
            pipeline,
            mesh_2d,
            mesh_3d,
            np.asarray(result.vertices, dtype=float)[:, :2],
            params,
            progress_callback=progress_callback,
        )
        result.vertices[:, :2] = refined_xy
        result.vertices[:, 2] = 0.0

        metrics = dict(getattr(result, "metrics", {}) or {})
        metrics.update(extra)
        metrics.update({
            "version_id": VERSION_ID,
            "paper_k2d_stage_separated": True,
            "paper_k2d_authoritative_representation": "shared-vertex planar QuadMesh K2D",
            "paper_k2d_shared_vertex_authoritative": True,
            "paper_k2d_independent_rigid_tile_layout_in_k2d": False,
            "paper_k2d_hinge_alignment_in_k2d": False,
            "paper_k2d_hinge_layout_deferred_to_section_4_4": True,
            "paper_k2d_eq5_policy": "EFlat = 1*EEdge + 1*ECollision + 0.001*EFab",
            "paper_k2d_edge_target": "corresponding K3D edge lengths",
            "paper_k2d_fabrication_policy": "opening angle should lie in [theta_min, 90 deg]",
            "paper_k2d_efab_exact_formula_available": False,
            "paper_k2d_efab_exactness_note": "main paper delegates exact EFab to Supplement Appendix A; current implementation does not invent the unavailable exact formula",
            "paper_k2d_post_split_centroid_realign_expected": False,
            "objective": "Paper-stage-separated K2D Eq.(5): shared vertices, K3D edge matching, collision active in K2D; Eq.(5) relative panel placement preserved after Split; hinge SE(2) deferred",
            "actual_backend": str(metrics.get("actual_backend", "base")) + " + paper shared-vertex collision refinement",
        })
        result.metrics.clear()
        result.metrics.update(metrics)
        try:
            report.objective = str(metrics["objective"])
            report.after_error = float(extra.get("paper_k2d_edge_error_after_refinement_mean", report.after_error))
            report.constraint_violation = float(extra.get("paper_k2d_collision_count_after_refinement", report.constraint_violation))
        except Exception:
            pass
        print(
            "[2026-09-14-PAPER-K2D] "
            f"overlaps={extra.get('paper_k2d_collision_count_before_refinement')}"
            f"->{extra.get('paper_k2d_collision_count_after_refinement')} "
            f"max_edge={extra.get('paper_k2d_edge_error_after_refinement_max')} "
            "hinge_layout_in_K2D=False post_split_centroid_realign=False"
        )
        return result, report

    pipeline._optimize_k2d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k2d = optimize
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k2d"] = optimize

    pipeline._onestring_optcuts_paper_k2d_20260914_installed = True


__all__ = ["install_optcuts_paper_k2d_20260914_patch", "VERSION_ID"]
