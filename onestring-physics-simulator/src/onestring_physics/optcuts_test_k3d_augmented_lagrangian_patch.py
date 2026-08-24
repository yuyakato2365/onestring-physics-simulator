"""OptCuts_test K3D hard planarity via an Augmented Lagrangian solve.

The ordinary K3D optimizer remains responsible for surface fitting and square
quality.  In ``optcuts_test`` only, this patch takes that K3D solution as a
reference and enforces one equality constraint per quadrilateral:

    g_q(x) = (x1-x0) dot ((x2-x0) cross (x3-x0)) = 0.

The scalar triple product vanishes iff the four vertices are coplanar (up to
degenerate cases).  We solve

    min  0.5*w_anchor*||x-x_ref||^2
    s.t. g_q(x)=0

with the augmented Lagrangian

    L_A = E + sum lambda_q c_q + 0.5*rho*sum c_q^2,

where c_q = g_q / s_q^3 is a scale-normalized constraint.  The existing K3D
solution therefore remains the preferred shape while planarity is driven to a
hard numerical tolerance without requiring an enormous soft ``w_planar``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


def _quad_constraint_values_and_gradient(
    vertices: np.ndarray,
    faces: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized scalar-triple constraints and per-corner gradients.

    gradients has shape (F, 4, 3), corresponding to the four face vertices.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)[:, :4]
    q = v[f]
    x0, x1, x2, x3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a = x1 - x0
    b = x2 - x0
    c = x3 - x0

    bc = np.cross(b, c)
    ca = np.cross(c, a)
    ab = np.cross(a, b)
    triple = np.einsum("ij,ij->i", a, bc)

    denom = np.maximum(np.asarray(scales, dtype=float) ** 3, 1e-18)
    values = triple / denom

    grad = np.zeros((len(f), 4, 3), dtype=float)
    grad[:, 1] = bc / denom[:, None]
    grad[:, 2] = ca / denom[:, None]
    grad[:, 3] = ab / denom[:, None]
    grad[:, 0] = -(grad[:, 1] + grad[:, 2] + grad[:, 3])
    return values, grad


def _quad_plane_distances(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Absolute fourth-point distance to the plane through the first 3 vertices."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)[:, :4]
    q = v[f]
    a = q[:, 1] - q[:, 0]
    b = q[:, 2] - q[:, 0]
    c = q[:, 3] - q[:, 0]
    n = np.cross(a, b)
    n_norm = np.linalg.norm(n, axis=1)
    triple = np.abs(np.einsum("ij,ij->i", c, n))
    return triple / np.maximum(n_norm, 1e-15)


def _face_scales(reference: np.ndarray, faces: np.ndarray) -> np.ndarray:
    q = np.asarray(reference, dtype=float)[np.asarray(faces, dtype=int)[:, :4]]
    vals: list[float] = []
    for tile in q:
        lengths = [
            float(np.linalg.norm(tile[(i + 1) % 4] - tile[i]))
            for i in range(4)
        ]
        vals.append(max(float(np.median(lengths)), 1e-8))
    return np.asarray(vals, dtype=float)


def _augmented_lagrangian_planarize(
    reference: np.ndarray,
    faces: np.ndarray,
    *,
    anchor_weight: float = 1.0,
    rho0: float = 10.0,
    outer_iterations: int = 10,
    inner_iterations: int = 35,
    relative_plane_tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    ref = np.asarray(reference, dtype=float)
    f = np.asarray(faces, dtype=int)
    if len(ref) == 0 or len(f) == 0:
        return ref.copy(), {"k3d_augmented_lagrangian_applied": False}
    if minimize is None:
        return ref.copy(), {
            "k3d_augmented_lagrangian_applied": False,
            "k3d_augmented_lagrangian_reason": "scipy.optimize.minimize unavailable",
        }

    f4 = f[:, :4]
    scales = _face_scales(ref, f4)
    tile_scale = float(np.median(scales)) if len(scales) else 1.0
    plane_tol = max(1e-10, tile_scale * float(relative_plane_tolerance))

    x = ref.copy()
    lambdas = np.zeros(len(f4), dtype=float)
    rho = max(float(rho0), 1e-6)
    w_anchor = max(float(anchor_weight), 1e-10)

    before_dist = _quad_plane_distances(x, f4)
    max_before = float(np.max(before_dist)) if len(before_dist) else 0.0
    rms_before = float(np.sqrt(np.mean(before_dist * before_dist))) if len(before_dist) else 0.0

    previous_violation = float("inf")
    inner_total = 0
    converged = False
    outer_done = 0

    def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
        verts = np.asarray(flat, dtype=float).reshape(ref.shape)
        delta = verts - ref
        energy = 0.5 * w_anchor * float(np.sum(delta * delta))
        grad_total = w_anchor * delta

        constraints, local_grad = _quad_constraint_values_and_gradient(verts, f4, scales)
        coeff = lambdas + rho * constraints
        energy += float(np.dot(lambdas, constraints) + 0.5 * rho * np.dot(constraints, constraints))

        contribution = coeff[:, None, None] * local_grad
        np.add.at(grad_total, f4[:, 0], contribution[:, 0])
        np.add.at(grad_total, f4[:, 1], contribution[:, 1])
        np.add.at(grad_total, f4[:, 2], contribution[:, 2])
        np.add.at(grad_total, f4[:, 3], contribution[:, 3])
        return energy, grad_total.ravel()

    for outer in range(max(1, int(outer_iterations))):
        outer_done = outer + 1
        result = minimize(
            fun=lambda z: objective_and_grad(z)[0],
            x0=x.ravel(),
            jac=lambda z: objective_and_grad(z)[1],
            method="L-BFGS-B",
            options={
                "maxiter": max(5, int(inner_iterations)),
                "ftol": 1e-12,
                "gtol": 1e-9,
                "maxls": 30,
            },
        )
        inner_total += int(getattr(result, "nit", 0))
        x = np.asarray(result.x, dtype=float).reshape(ref.shape)

        constraints, _ = _quad_constraint_values_and_gradient(x, f4, scales)
        lambdas = lambdas + rho * constraints

        distances = _quad_plane_distances(x, f4)
        max_dist = float(np.max(distances)) if len(distances) else 0.0
        normalized_violation = float(np.max(np.abs(constraints))) if len(constraints) else 0.0
        print(
            "[OPTCUTS-TEST-K3D-AL] "
            f"outer={outer_done} rho={rho:.6g} max_plane_dist={max_dist:.6g} "
            f"max_constraint={normalized_violation:.6g}"
        )
        if max_dist <= plane_tol:
            converged = True
            break

        # Standard AL policy: if the equality violation stalls, strengthen rho;
        # otherwise keep rho and let the multiplier update do the work.
        if normalized_violation > 0.5 * previous_violation:
            rho = min(rho * 5.0, 1e10)
        previous_violation = normalized_violation

    after_dist = _quad_plane_distances(x, f4)
    max_after = float(np.max(after_dist)) if len(after_dist) else 0.0
    rms_after = float(np.sqrt(np.mean(after_dist * after_dist))) if len(after_dist) else 0.0
    displacement = np.linalg.norm(x - ref, axis=1)

    return x, {
        "k3d_augmented_lagrangian_applied": True,
        "k3d_planarity_constraint_type": "scalar triple product equality per quad",
        "k3d_planarity_constraint": "(x1-x0) dot ((x2-x0) cross (x3-x0)) = 0",
        "k3d_augmented_lagrangian_reference": "ordinary K3D square+surface solution",
        "k3d_augmented_lagrangian_anchor_weight": float(w_anchor),
        "k3d_augmented_lagrangian_rho_initial": float(rho0),
        "k3d_augmented_lagrangian_rho_final": float(rho),
        "k3d_augmented_lagrangian_outer_iterations": int(outer_done),
        "k3d_augmented_lagrangian_inner_iterations_total": int(inner_total),
        "k3d_augmented_lagrangian_converged": bool(converged),
        "k3d_hard_planarity_tolerance": float(plane_tol),
        "k3d_hard_planarity_max_distance_before": float(max_before),
        "k3d_hard_planarity_rms_distance_before": float(rms_before),
        "k3d_hard_planarity_max_distance_after": float(max_after),
        "k3d_hard_planarity_rms_distance_after": float(rms_after),
        "k3d_hard_planarity_constraint_satisfied": bool(max_after <= plane_tol),
        "k3d_hard_planarity_vertex_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0,
        "k3d_hard_planarity_vertex_displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
    }


def install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_k3d_al_installed", False):
        return

    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        result = base(target, mesh, parameterization, params)
        k3d, report = result
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return result

        reference = np.asarray(k3d.vertices, dtype=float).copy()
        solved, al_metrics = _augmented_lagrangian_planarize(
            reference,
            np.asarray(k3d.faces, dtype=int),
            anchor_weight=float(getattr(params, "k3d_al_anchor_weight", 1.0)),
            rho0=float(getattr(params, "k3d_al_rho0", 10.0)),
            outer_iterations=int(getattr(params, "k3d_al_outer_iterations", 10)),
            inner_iterations=int(getattr(params, "k3d_al_inner_iterations", 35)),
            relative_plane_tolerance=float(getattr(params, "k3d_al_relative_planarity_tolerance", 1e-6)),
        )
        k3d.vertices[:] = solved
        try:
            k3d.metrics.update(al_metrics)
            k3d.metrics["k3d_planarity_mode"] = "hard equality via augmented Lagrangian"
            k3d.metrics["k3d_soft_w_planar_is_authoritative"] = False
        except Exception:
            pass
        try:
            report.objective = str(getattr(report, "objective", "")) + " + hard quad planarity (Augmented Lagrangian)"
            report.constraint_violation = float(al_metrics.get("k3d_hard_planarity_max_distance_after", 0.0))
        except Exception:
            pass
        return k3d, report

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

    pipeline._onestring_optcuts_test_k3d_al_installed = True


__all__ = ["install_optcuts_test_k3d_augmented_lagrangian_patch"]
