"""Final explicit equality-constrained planarity polishing for optcuts_test K3D.

This wrapper is installed *after* the Augmented-Lagrangian K3D wrapper and
*before* the generic final validity guard.  The AL/restoration result is used as
the initial/reference shape, while SLSQP treats quad planarity as an explicit
equality constraint rather than another penalty term.

The objective is simply

    min 0.5 ||x - x_AL||^2
    s.t. c_q(x) = 0

where c_q is the same scale-normalized robust scalar-triple constraint used by
the AL patch.  A localized solve is attempted first on violating faces plus a
1-ring; if a material residual remains and the full problem is not too large, a
single global equality solve is attempted.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from .optcuts_test_k3d_augmented_lagrangian_patch import (
    _face_scales,
    _quad_constraint_values_and_gradient,
    _quad_plane_distances,
    _robust_constraint_faces,
)


def _faces_touching_vertices(faces: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    if len(vertex_ids) == 0:
        return np.zeros(0, dtype=int)
    mask = np.isin(np.asarray(faces, dtype=int)[:, :4], np.asarray(vertex_ids, dtype=int))
    return np.flatnonzero(np.any(mask, axis=1)).astype(int)


def _solve_subset_slsqp(
    start: np.ndarray,
    constraint_faces_all: np.ndarray,
    scales_all: np.ndarray,
    active_vertices: np.ndarray,
    constrained_face_ids: np.ndarray,
    *,
    maxiter: int,
    ftol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve explicit planarity equalities with all non-active vertices fixed."""
    x0_full = np.asarray(start, dtype=float)
    active = np.asarray(sorted(set(int(v) for v in active_vertices)), dtype=int)
    face_ids = np.asarray(sorted(set(int(i) for i in constrained_face_ids)), dtype=int)
    if minimize is None or len(active) == 0 or len(face_ids) == 0:
        return x0_full.copy(), {
            "success": False,
            "message": "SLSQP unavailable or empty active problem",
            "nit": 0,
            "active_vertices": int(len(active)),
            "constrained_faces": int(len(face_ids)),
        }

    cf = np.asarray(constraint_faces_all, dtype=int)[face_ids]
    sc = np.asarray(scales_all, dtype=float)[face_ids]
    vertex_to_local = {int(v): i for i, v in enumerate(active.tolist())}
    reference_active = x0_full[active].copy()

    def unpack(z: np.ndarray) -> np.ndarray:
        full = x0_full.copy()
        full[active] = np.asarray(z, dtype=float).reshape((-1, 3))
        return full

    def objective(z: np.ndarray) -> float:
        p = np.asarray(z, dtype=float).reshape((-1, 3))
        d = p - reference_active
        return 0.5 * float(np.sum(d * d))

    def objective_jac(z: np.ndarray) -> np.ndarray:
        p = np.asarray(z, dtype=float).reshape((-1, 3))
        return (p - reference_active).ravel()

    def constraint_fun(z: np.ndarray) -> np.ndarray:
        full = unpack(z)
        values, _ = _quad_constraint_values_and_gradient(full, cf, sc)
        return np.asarray(values, dtype=float)

    def constraint_jac(z: np.ndarray) -> np.ndarray:
        full = unpack(z)
        _, local_grad = _quad_constraint_values_and_gradient(full, cf, sc)
        jac = np.zeros((len(cf), 3 * len(active)), dtype=float)
        for fi, face in enumerate(cf):
            for corner in range(4):
                local_id = vertex_to_local.get(int(face[corner]))
                if local_id is not None:
                    jac[fi, 3 * local_id : 3 * local_id + 3] += local_grad[fi, corner]
        return jac

    result = minimize(
        fun=objective,
        x0=reference_active.ravel(),
        jac=objective_jac,
        method="SLSQP",
        constraints={"type": "eq", "fun": constraint_fun, "jac": constraint_jac},
        options={
            "maxiter": max(20, int(maxiter)),
            "ftol": float(ftol),
            "disp": False,
        },
    )
    out = unpack(np.asarray(result.x, dtype=float))
    values, _ = _quad_constraint_values_and_gradient(out, cf, sc)
    return out, {
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "nit": int(getattr(result, "nit", 0)),
        "active_vertices": int(len(active)),
        "constrained_faces": int(len(face_ids)),
        "max_normalized_constraint": float(np.max(np.abs(values))) if len(values) else 0.0,
    }


def _slsqp_planarity_polish(
    start: np.ndarray,
    faces: np.ndarray,
    *,
    plane_tolerance: float,
    local_trigger_relative: float = 5e-4,
    local_maxiter: int = 250,
    global_maxiter: int = 350,
    global_vertex_limit: int = 1400,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(start, dtype=float).copy()
    f4 = np.asarray(faces, dtype=int)[:, :4]
    constraint_faces, _ = _robust_constraint_faces(x, f4)
    scales = _face_scales(x, f4)
    tile_scale = float(np.median(scales)) if len(scales) else 1.0

    before = _quad_plane_distances(x, constraint_faces)
    max_before = float(np.max(before)) if len(before) else 0.0
    if max_before <= float(plane_tolerance):
        return x, {
            "k3d_slsqp_polish_applied": False,
            "k3d_slsqp_polish_reason": "already within tolerance",
            "k3d_slsqp_max_plane_distance_before": max_before,
            "k3d_slsqp_max_plane_distance_after": max_before,
        }

    # Local stage: faces with a fabrication-relevant residual, then include all
    # faces touching their vertices so neighboring planarity is not ignored.
    trigger = max(float(plane_tolerance) * 10.0, tile_scale * float(local_trigger_relative))
    violating = np.flatnonzero(before > trigger).astype(int)
    if len(violating) == 0:
        violating = np.asarray([int(np.argmax(before))], dtype=int)
    seed_vertices = np.unique(f4[violating].reshape(-1))
    one_ring_faces = _faces_touching_vertices(f4, seed_vertices)
    active_vertices = np.unique(f4[one_ring_faces].reshape(-1))
    local_constraint_faces = _faces_touching_vertices(f4, active_vertices)

    x_local, local_info = _solve_subset_slsqp(
        x,
        constraint_faces,
        scales,
        active_vertices,
        local_constraint_faces,
        maxiter=local_maxiter,
        ftol=1e-12,
    )
    local_dist = _quad_plane_distances(x_local, constraint_faces)
    local_max = float(np.max(local_dist)) if len(local_dist) else 0.0
    print(
        "[OPTCUTS-TEST-K3D-SLSQP-LOCAL] "
        f"before={max_before:.6g} after={local_max:.6g} tol={plane_tolerance:.6g} "
        f"active_vertices={local_info.get('active_vertices', 0)} "
        f"faces={local_info.get('constrained_faces', 0)} success={local_info.get('success', False)} "
        f"nit={local_info.get('nit', 0)} message={local_info.get('message', '')}"
    )
    x = x_local

    global_info: dict[str, Any] = {
        "success": False,
        "message": "global stage not attempted",
        "nit": 0,
        "active_vertices": 0,
        "constrained_faces": 0,
    }
    global_attempted = False

    # If localized polishing is insufficient, use all vertices only when the
    # problem size is still reasonable for dense SLSQP Jacobians.
    if local_max > float(plane_tolerance) and len(x) <= int(global_vertex_limit):
        global_attempted = True
        all_vertices = np.unique(f4.reshape(-1))
        all_faces = np.arange(len(f4), dtype=int)
        x_global, global_info = _solve_subset_slsqp(
            x,
            constraint_faces,
            scales,
            all_vertices,
            all_faces,
            maxiter=global_maxiter,
            ftol=1e-13,
        )
        global_dist = _quad_plane_distances(x_global, constraint_faces)
        global_max = float(np.max(global_dist)) if len(global_dist) else 0.0
        print(
            "[OPTCUTS-TEST-K3D-SLSQP-GLOBAL] "
            f"before={local_max:.6g} after={global_max:.6g} tol={plane_tolerance:.6g} "
            f"active_vertices={global_info.get('active_vertices', 0)} "
            f"faces={global_info.get('constrained_faces', 0)} success={global_info.get('success', False)} "
            f"nit={global_info.get('nit', 0)} message={global_info.get('message', '')}"
        )
        # Never accept a polishing stage that made the actual plane metric worse.
        if global_max <= local_max:
            x = x_global

    after = _quad_plane_distances(x, constraint_faces)
    max_after = float(np.max(after)) if len(after) else 0.0
    rms_after = float(np.sqrt(np.mean(after * after))) if len(after) else 0.0
    displacement = np.linalg.norm(x - np.asarray(start, dtype=float), axis=1)

    return x, {
        "k3d_slsqp_polish_applied": True,
        "k3d_slsqp_formulation": "min ||x-x_AL||^2 subject to robust quad planarity equalities",
        "k3d_slsqp_local_trigger": float(trigger),
        "k3d_slsqp_local_success": bool(local_info.get("success", False)),
        "k3d_slsqp_local_message": str(local_info.get("message", "")),
        "k3d_slsqp_local_iterations": int(local_info.get("nit", 0)),
        "k3d_slsqp_local_active_vertices": int(local_info.get("active_vertices", 0)),
        "k3d_slsqp_local_constrained_faces": int(local_info.get("constrained_faces", 0)),
        "k3d_slsqp_global_attempted": bool(global_attempted),
        "k3d_slsqp_global_success": bool(global_info.get("success", False)),
        "k3d_slsqp_global_message": str(global_info.get("message", "")),
        "k3d_slsqp_global_iterations": int(global_info.get("nit", 0)),
        "k3d_slsqp_max_plane_distance_before": float(max_before),
        "k3d_slsqp_max_plane_distance_after": float(max_after),
        "k3d_slsqp_rms_plane_distance_after": float(rms_after),
        "k3d_slsqp_constraint_satisfied": bool(max_after <= float(plane_tolerance)),
        "k3d_slsqp_vertex_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0,
        "k3d_slsqp_vertex_displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
    }


def install_optcuts_test_k3d_slsqp_polish_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_k3d_slsqp_polish_installed", False):
        return

    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        out, report = base(target, mesh, parameterization, params)
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return out, report
        metrics = dict(getattr(out, "metrics", {}) or {})
        if not bool(metrics.get("k3d_augmented_lagrangian_applied", False)):
            return out, report

        tol = float(metrics.get("k3d_hard_planarity_tolerance", 0.0))
        if tol <= 0.0:
            scales = _face_scales(np.asarray(out.vertices, dtype=float), np.asarray(out.faces, dtype=int))
            tile_scale = float(np.median(scales)) if len(scales) else 1.0
            tol = max(1e-10, tile_scale * 1e-6)

        solved, polish_metrics = _slsqp_planarity_polish(
            np.asarray(out.vertices, dtype=float),
            np.asarray(out.faces, dtype=int),
            plane_tolerance=tol,
            local_trigger_relative=float(getattr(params, "k3d_slsqp_local_trigger_relative", 5e-4)),
            local_maxiter=int(getattr(params, "k3d_slsqp_local_maxiter", 250)),
            global_maxiter=int(getattr(params, "k3d_slsqp_global_maxiter", 350)),
            global_vertex_limit=int(getattr(params, "k3d_slsqp_global_vertex_limit", 1400)),
        )
        out.vertices[:] = solved
        try:
            out.metrics.update(polish_metrics)
            out.metrics["k3d_planarity_mode"] = (
                "Augmented Lagrangian + feasibility restoration + explicit SLSQP equality polish"
            )
        except Exception:
            pass
        try:
            report.objective = str(getattr(report, "objective", "")) + " + explicit SLSQP planarity equalities"
            report.constraint_violation = float(polish_metrics.get("k3d_slsqp_max_plane_distance_after", 0.0))
        except Exception:
            pass
        return out, report

    pipeline._optimize_k3d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = optimize
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize

    pipeline._onestring_optcuts_test_k3d_slsqp_polish_installed = True


__all__ = ["install_optcuts_test_k3d_slsqp_polish_patch"]
