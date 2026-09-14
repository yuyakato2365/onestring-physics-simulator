"""2026-09-14 paper-stage K2D variant for the OptCuts experimental path.

The One String paper Section 4.3 defines K2D as a shared-vertex planar mesh
obtained by optimizing M2D with Eq. (5):

    EFlat = w_edge EEdge + w_collision ECollision + w_fab EFab.

Rigid per-tile repositioning for hinge placement belongs to the later T2D
Eq. (6) stage, not K2D.  This patch therefore keeps the shared-vertex K2D mesh
authoritative and performs a bounded collision-aware refinement after the
existing edge-length solve.  No SE(2) tile/hinge placement is introduced here.

The main paper specifies EFab as an opening-angle penalty with theta in
[theta_min, 90 deg] but delegates its exact formula to Supplement Appendix A.
Because that supplement is not part of the repository/source bundle, this patch
does not pretend to reproduce the unavailable exact EFab formula.  It preserves
the existing fabrication-angle diagnostics and marks this limitation explicitly.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


VARIANT = "3"
VERSION_ID = "2026-09-14-paper-k2d-eq5-stage-separated"


def _active(params: Any) -> bool:
    return (
        os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == VARIANT
        and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"
    )


def _fn(pipeline: Any, name: str):
    value = getattr(pipeline, name, None)
    if value is None:
        value = getattr(getattr(pipeline, "_original", None), name, None)
    return value


def _paper_collision_refine(pipeline: Any, mesh_2d: Any, mesh_3d: Any, xy0: np.ndarray, params: Any, progress_callback=None):
    unique_edges = _fn(pipeline, "_unique_mesh_edges")
    edge_errors = _fn(pipeline, "_edge_matching_errors")
    edge_project = _fn(pipeline, "_projective_edge_match_2d")
    collision_relax = _fn(pipeline, "_relax_2d_collisions")
    tiles_from_mesh = _fn(pipeline, "_tiles_from_mesh_vertices")
    count_collisions = _fn(pipeline, "_count_2d_tile_collisions")
    emit = _fn(pipeline, "_emit_progress")

    if not all(callable(fn) for fn in (unique_edges, edge_errors, edge_project, collision_relax, tiles_from_mesh, count_collisions)):
        return np.asarray(xy0, float).copy(), {
            "paper_k2d_collision_refinement_applied": False,
            "paper_k2d_collision_refinement_reason": "required base helpers unavailable",
        }

    faces = np.asarray(mesh_2d.faces, dtype=int)
    xy = np.asarray(xy0, dtype=float).copy()
    base_xy = np.asarray(mesh_2d.vertices, dtype=float)[:, :2].copy()
    edges = unique_edges(faces)
    target_lengths = np.asarray(
        [np.linalg.norm(np.asarray(mesh_3d.vertices)[a] - np.asarray(mesh_3d.vertices)[b]) for a, b in edges],
        dtype=float,
    )
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

    # Paper-style policy: edge projection and non-penetration projection act on
    # the same shared K2D vertices.  Unlike the legacy fast path, collision is
    # not silently deferred merely because the mesh has >120 faces.
    for outer in range(outer_max):
        candidate = edge_project(
            xy,
            base_xy,
            edges,
            target_lengths,
            faces,
            mesh_2d.grid,
            iterations=4,
        )
        candidate = collision_relax(candidate, faces, mesh_2d.grid, iterations=1, weight=0.04)
        candidate = edge_project(
            candidate,
            base_xy,
            edges,
            target_lengths,
            faces,
            mesh_2d.grid,
            iterations=4,
        )
        c_mean, c_max = edge_errors(candidate, edges, target_lengths)
        c_collision = collision_count(candidate)
        # Prefer fewer penetrations, then better K3D edge matching.  This keeps
        # ECollision active without sacrificing the defining EEdge objective.
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
        "paper_k2d_collision_refinement_model": "shared-vertex alternating edge projection + 2D non-penetration projection",
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


def install_optcuts_paper_k2d_20260914_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_paper_k2d_20260914_installed", False):
        return

    base = pipeline._optimize_k2d

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        if not _active(params):
            return base(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

        # Reproduction weights stated in the paper for Eq. (5).
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
            "paper_k2d_efab_exactness_note": "main paper delegates exact EFab to Supplement Appendix A; current implementation retains existing fabrication diagnostics rather than inventing that missing formula",
            "objective": "Paper-stage-separated K2D Eq.(5): shared vertices, K3D edge matching, collision active in K2D; hinge SE(2) deferred",
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
            "hinge_layout_in_K2D=False"
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
