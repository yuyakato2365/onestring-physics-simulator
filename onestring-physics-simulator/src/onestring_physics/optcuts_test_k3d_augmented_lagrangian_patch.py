"""OptCuts_test K3D hard planarity via an Augmented Lagrangian solve.

The ordinary K3D optimizer remains responsible for surface fitting and square
quality. In ``optcuts_test`` only, this patch takes that K3D solution as a
reference and enforces one equality constraint per quadrilateral.

A naive scalar-triple constraint based on corners (0,1,2) is degenerate when
those three corners are collinear. This occurs for clipped boundary panels that
were promoted from a triangle to a four-corner representation by inserting an
edge midpoint. Therefore each face first selects the maximum-area triangle among
its four corners; the remaining corner is constrained to that triangle's plane.

For reordered corners (x0,x1,x2,x3),

    g_q(x) = (x1-x0) dot ((x2-x0) cross (x3-x0)) = 0.

We solve

    min  0.5*w_anchor*||x-x_ref||^2
    s.t. g_q(x)=0

with the augmented Lagrangian

    L_A = E + sum lambda_q c_q + 0.5*rho*sum c_q^2,

where c_q = g_q / s_q^3 is a scale-normalized constraint.

Because a large shared-vertex quad mesh can be poorly conditioned for a pure
L-BFGS augmented-Lagrangian solve, every outer iteration is followed by a
feasibility-restoration step.  Each quad is projected to its current best-fit
plane and the multiple projected positions proposed for a shared vertex are
averaged (consensus projection).  Repeating this projection drives the iterate
back toward the manifold where all quads are planar before the multiplier
update.  A final feasibility-polishing pass is applied before accepting K3D.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


def _robust_constraint_faces(reference: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder each quad so corners 0,1,2 form its maximum-area triangle.

    Returns (reordered_faces, base_triangle_double_areas). The fourth index is
    the remaining corner. This avoids the degenerate A-mid(A,B)-B base triangle
    that occurs for triangle-to-quad boundary surrogates.
    """
    ref = np.asarray(reference, dtype=float)
    f4 = np.asarray(faces, dtype=int)[:, :4]
    reordered = np.empty_like(f4)
    areas = np.zeros(len(f4), dtype=float)
    local_ids = (0, 1, 2, 3)
    for fi, face in enumerate(f4):
        pts = ref[face]
        best_combo = (0, 1, 2)
        best_area = -1.0
        for combo in combinations(local_ids, 3):
            a, b, c = pts[list(combo)]
            area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
            if area2 > best_area:
                best_area = area2
                best_combo = combo
        remaining = next(idx for idx in local_ids if idx not in best_combo)
        reordered[fi] = np.asarray(
            [face[best_combo[0]], face[best_combo[1]], face[best_combo[2]], face[remaining]],
            dtype=int,
        )
        areas[fi] = max(best_area, 0.0)
    return reordered, areas


def _quad_constraint_values_and_gradient(
    vertices: np.ndarray,
    constraint_faces: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized scalar-triple constraints and per-corner gradients."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(constraint_faces, dtype=int)[:, :4]
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


def _quad_plane_distances(vertices: np.ndarray, constraint_faces: np.ndarray) -> np.ndarray:
    """Distance of the remaining corner to a robust maximum-area base plane."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(constraint_faces, dtype=int)[:, :4]
    q = v[f]
    a = q[:, 1] - q[:, 0]
    b = q[:, 2] - q[:, 0]
    c = q[:, 3] - q[:, 0]
    n = np.cross(a, b)
    n_norm = np.linalg.norm(n, axis=1)
    triple = np.abs(np.einsum("ij,ij->i", c, n))
    out = np.full(len(f), np.inf, dtype=float)
    good = n_norm > 1e-12
    out[good] = triple[good] / n_norm[good]
    return out


def _face_scales(reference: np.ndarray, faces: np.ndarray) -> np.ndarray:
    q = np.asarray(reference, dtype=float)[np.asarray(faces, dtype=int)[:, :4]]
    vals: list[float] = []
    for tile in q:
        lengths = [
            float(np.linalg.norm(tile[(i + 1) % 4] - tile[i]))
            for i in range(4)
        ]
        positive = [length for length in lengths if length > 1e-10]
        vals.append(max(float(np.median(positive if positive else lengths)), 1e-8))
    return np.asarray(vals, dtype=float)


def _consensus_planarity_restore(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sweeps: int,
    relaxation: float = 1.0,
    tolerance: float | None = None,
    constraint_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    """Project all quads to best-fit planes and average shared-vertex proposals.

    The per-face projection is exact for that face. Shared vertices receive
    several proposals, so consensus averaging can reintroduce a small residual;
    repeated sweeps are therefore used. This is an alternating-projection style
    feasibility restoration, not a replacement for the AL objective.
    """
    x = np.asarray(vertices, dtype=float).copy()
    f4 = np.asarray(faces, dtype=int)[:, :4]
    if len(x) == 0 or len(f4) == 0:
        return x, 0, 0.0

    alpha = float(np.clip(relaxation, 0.05, 1.0))
    counts = np.zeros(len(x), dtype=float)
    flat_ids = f4.reshape(-1)
    np.add.at(counts, flat_ids, 1.0)
    active = counts > 0.0
    used = 0
    last_max = float("inf")

    for sweep in range(max(0, int(sweeps))):
        q = x[f4]
        centers = np.mean(q, axis=1)
        centered = q - centers[:, None, :]

        # The right singular vector associated with the smallest singular value
        # is the best-fit plane normal. numpy supports stacked SVD here.
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normals = vh[:, -1, :]
        except np.linalg.LinAlgError:
            # Rare fallback for an ill-conditioned batch.
            normals = np.empty((len(f4), 3), dtype=float)
            for fi, tile in enumerate(centered):
                _, _, vh_i = np.linalg.svd(tile, full_matrices=False)
                normals[fi] = vh_i[-1]

        n_norm = np.linalg.norm(normals, axis=1)
        good = n_norm > 1e-15
        normals[good] /= n_norm[good, None]
        normals[~good] = np.array([0.0, 0.0, 1.0])

        signed = np.einsum("fki,fi->fk", centered, normals)
        projected = q - signed[:, :, None] * normals[:, None, :]

        accum = np.zeros_like(x)
        for corner in range(4):
            np.add.at(accum, f4[:, corner], projected[:, corner])
        consensus = x.copy()
        consensus[active] = accum[active] / counts[active, None]
        x[active] = (1.0 - alpha) * x[active] + alpha * consensus[active]
        used = sweep + 1

        if tolerance is not None and constraint_faces is not None:
            distances = _quad_plane_distances(x, constraint_faces)
            last_max = float(np.max(distances)) if len(distances) else 0.0
            if last_max <= float(tolerance):
                break

    if constraint_faces is not None:
        distances = _quad_plane_distances(x, constraint_faces)
        last_max = float(np.max(distances)) if len(distances) else 0.0
    return x, used, float(last_max)


def _augmented_lagrangian_planarize(
    reference: np.ndarray,
    faces: np.ndarray,
    *,
    anchor_weight: float = 1.0,
    rho0: float = 10.0,
    outer_iterations: int = 16,
    inner_iterations: int = 70,
    relative_plane_tolerance: float = 1e-6,
    restoration_sweeps: int = 8,
    final_polish_sweeps: int = 400,
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
    constraint_faces, base_areas = _robust_constraint_faces(ref, f4)
    scales = _face_scales(ref, f4)
    tile_scale = float(np.median(scales)) if len(scales) else 1.0
    plane_tol = max(1e-10, tile_scale * float(relative_plane_tolerance))

    degenerate_base_count = int(np.sum(base_areas <= np.maximum(scales * scales * 1e-10, 1e-16)))
    if degenerate_base_count:
        raise RuntimeError(
            "OPTCUTS_TEST_K3D_PLANARITY_CONSTRAINT_DEGENERATE: "
            f"{degenerate_base_count} panels have no non-collinear triple among their four corners."
        )

    x = ref.copy()
    lambdas = np.zeros(len(f4), dtype=float)
    rho = max(float(rho0), 1e-6)
    w_anchor = max(float(anchor_weight), 1e-10)

    before_dist = _quad_plane_distances(x, constraint_faces)
    max_before = float(np.max(before_dist)) if len(before_dist) else 0.0
    rms_before = float(np.sqrt(np.mean(before_dist * before_dist))) if len(before_dist) else 0.0

    previous_violation = float("inf")
    inner_total = 0
    restoration_total = 0
    converged = False
    outer_done = 0
    last_inner_success = True
    last_inner_message = ""

    def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
        verts = np.asarray(flat, dtype=float).reshape(ref.shape)
        delta = verts - ref
        energy = 0.5 * w_anchor * float(np.sum(delta * delta))
        grad_total = w_anchor * delta

        constraints, local_grad = _quad_constraint_values_and_gradient(verts, constraint_faces, scales)
        coeff = lambdas + rho * constraints
        energy += float(np.dot(lambdas, constraints) + 0.5 * rho * np.dot(constraints, constraints))

        contribution = coeff[:, None, None] * local_grad
        np.add.at(grad_total, constraint_faces[:, 0], contribution[:, 0])
        np.add.at(grad_total, constraint_faces[:, 1], contribution[:, 1])
        np.add.at(grad_total, constraint_faces[:, 2], contribution[:, 2])
        np.add.at(grad_total, constraint_faces[:, 3], contribution[:, 3])
        return energy, grad_total.ravel()

    for outer in range(max(1, int(outer_iterations))):
        outer_done = outer + 1
        result = minimize(
            fun=lambda z: objective_and_grad(z)[0],
            x0=x.ravel(),
            jac=lambda z: objective_and_grad(z)[1],
            method="L-BFGS-B",
            options={
                "maxiter": max(10, int(inner_iterations)),
                "ftol": 1e-13,
                "gtol": 1e-10,
                "maxls": 50,
            },
        )
        inner_total += int(getattr(result, "nit", 0))
        last_inner_success = bool(getattr(result, "success", False))
        last_inner_message = str(getattr(result, "message", ""))
        x = np.asarray(result.x, dtype=float).reshape(ref.shape)

        before_restore = _quad_plane_distances(x, constraint_faces)
        max_before_restore = float(np.max(before_restore)) if len(before_restore) else 0.0
        x, used, max_after_restore = _consensus_planarity_restore(
            x,
            f4,
            sweeps=max(0, int(restoration_sweeps)),
            relaxation=1.0,
            tolerance=plane_tol,
            constraint_faces=constraint_faces,
        )
        restoration_total += int(used)

        # Update multipliers using the restored iterate. This makes lambda track
        # the residual of the actual candidate that will seed the next outer step.
        constraints, _ = _quad_constraint_values_and_gradient(x, constraint_faces, scales)
        lambdas = lambdas + rho * constraints

        distances = _quad_plane_distances(x, constraint_faces)
        max_dist = float(np.max(distances)) if len(distances) else 0.0
        normalized_violation = float(np.max(np.abs(constraints))) if len(constraints) else 0.0
        print(
            "[OPTCUTS-TEST-K3D-AL] "
            f"outer={outer_done} rho={rho:.6g} "
            f"plane_before_restore={max_before_restore:.6g} "
            f"max_plane_dist={max_dist:.6g} restore_sweeps={used} "
            f"max_constraint={normalized_violation:.6g} inner_success={last_inner_success}"
        )
        if max_dist <= plane_tol:
            converged = True
            break

        if normalized_violation > 0.5 * previous_violation:
            rho = min(rho * 5.0, 1e12)
        previous_violation = normalized_violation

    # Hard-feasibility polishing.  This stage does not change the target
    # objective; it only alternates exact per-face plane projections and shared
    # vertex consensus until the equality constraints are numerically satisfied.
    polish_used = 0
    pre_polish_dist = _quad_plane_distances(x, constraint_faces)
    pre_polish_max = float(np.max(pre_polish_dist)) if len(pre_polish_dist) else 0.0
    if pre_polish_max > plane_tol and int(final_polish_sweeps) > 0:
        x, polish_used, _ = _consensus_planarity_restore(
            x,
            f4,
            sweeps=max(0, int(final_polish_sweeps)),
            relaxation=1.0,
            tolerance=plane_tol,
            constraint_faces=constraint_faces,
        )
        restoration_total += int(polish_used)

    after_dist = _quad_plane_distances(x, constraint_faces)
    max_after = float(np.max(after_dist)) if len(after_dist) else 0.0
    rms_after = float(np.sqrt(np.mean(after_dist * after_dist))) if len(after_dist) else 0.0
    converged = bool(max_after <= plane_tol)
    displacement = np.linalg.norm(x - ref, axis=1)

    print(
        "[OPTCUTS-TEST-K3D-AL-FINAL] "
        f"pre_polish={pre_polish_max:.6g} max_plane_dist={max_after:.6g} "
        f"tol={plane_tol:.6g} polish_sweeps={polish_used} converged={converged}"
    )

    return x, {
        "k3d_augmented_lagrangian_applied": True,
        "k3d_planarity_constraint_type": "robust max-area scalar triple product equality per quad",
        "k3d_planarity_constraint": "max-area base triangle; remaining corner coplanar",
        "k3d_planarity_constraint_reordered_faces": True,
        "k3d_planarity_degenerate_base_count": int(degenerate_base_count),
        "k3d_augmented_lagrangian_reference": "validity-repaired ordinary K3D square+surface solution",
        "k3d_augmented_lagrangian_anchor_weight": float(w_anchor),
        "k3d_augmented_lagrangian_rho_initial": float(rho0),
        "k3d_augmented_lagrangian_rho_final": float(rho),
        "k3d_augmented_lagrangian_outer_iterations": int(outer_done),
        "k3d_augmented_lagrangian_inner_iterations_total": int(inner_total),
        "k3d_augmented_lagrangian_last_inner_success": bool(last_inner_success),
        "k3d_augmented_lagrangian_last_inner_message": str(last_inner_message),
        "k3d_augmented_lagrangian_converged": bool(converged),
        "k3d_planarity_restoration_sweeps_total": int(restoration_total),
        "k3d_planarity_final_polish_sweeps": int(polish_used),
        "k3d_planarity_pre_final_polish_max_distance": float(pre_polish_max),
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
            outer_iterations=int(getattr(params, "k3d_al_outer_iterations", 16)),
            inner_iterations=int(getattr(params, "k3d_al_inner_iterations", 70)),
            relative_plane_tolerance=float(getattr(params, "k3d_al_relative_planarity_tolerance", 1e-6)),
            restoration_sweeps=int(getattr(params, "k3d_al_restoration_sweeps", 8)),
            final_polish_sweeps=int(getattr(params, "k3d_al_final_polish_sweeps", 400)),
        )
        k3d.vertices[:] = solved
        try:
            k3d.metrics.update(al_metrics)
            k3d.metrics["k3d_planarity_mode"] = "hard equality via robust augmented Lagrangian + feasibility restoration"
            k3d.metrics["k3d_soft_w_planar_is_authoritative"] = False
            k3d.metrics["k3d_al_authoritative_result"] = True
        except Exception:
            pass
        try:
            report.objective = str(getattr(report, "objective", "")) + " + hard quad planarity (Augmented Lagrangian + feasibility restoration)"
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


__all__ = [
    "install_optcuts_test_k3d_augmented_lagrangian_patch",
    "_robust_constraint_faces",
    "_quad_plane_distances",
    "_consensus_planarity_restore",
]
