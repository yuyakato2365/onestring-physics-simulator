"""Orientation-preserving refinement for exact seam-constrained UV atlases.

The discrete seam equations are eliminated before calling this routine.  It
moves only the remaining root variables; it neither drops seam constraints
nor changes cuts, mesh faces, or the user grid. A penalty is a recovery
attempt, never a proof of injectivity or an unconditional acceptance rule.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def signed_areas(uv, faces):
    triangles = np.asarray(uv, float)[np.asarray(faces, int)]
    a, b = triangles[:, 0], triangles[:, 1]
    c = triangles[:, 2]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def refine_positive_uv(A, target, roots, *, component, matrices, offsets,
                       fixed, columns, faces, reference_uv, h,
                       accepted_ratio=1e-3, target_ratio=1e-2,
                       maxiter=150, penalties=(1., 16., 256., 4096., 65536.)):
    """Try untangling while *exactly* retaining eliminated lattice equations.

    Returns root variables and diagnostics. Caller MUST re-evaluate signed
    areas and seam residual and reject any still-invalid result. Nonconvex
    penalties cannot establish feasibility of a failed instance.
    """
    original = np.asarray(reference_uv, float)
    faces = np.asarray(faces, int)
    mats = np.asarray(matrices, float)
    comps = np.asarray(component, int)
    constant = np.asarray(offsets, float).copy()
    free_vertices = np.asarray([i for i, c in enumerate(comps)
                                if int(c) not in fixed], dtype=int)
    free_roots = np.asarray([int(columns[int(comps[i])] // 2)
                             for i in free_vertices], dtype=int)
    for i, c in enumerate(comps):
        if int(c) in fixed:
            constant[i] += mats[i] @ np.asarray(fixed[int(c)], float)
    free_matrices = mats[free_vertices]
    signed_old = signed_areas(original, faces)
    if np.any(~np.isfinite(signed_old)) or np.any(np.abs(signed_old) <= 1e-14):
        raise ValueError("Invalid source triangles for orientation-preserving refinement")
    orientations = np.sign(signed_old)
    weights = 1. / np.maximum(np.abs(signed_old), max(h * h * 1e-8, 1e-20))
    bounds = float(target_ratio) * np.abs(signed_old)
    target = np.asarray(target, float)
    roots = np.asarray(roots, float).copy()

    def positions(x):
        uv = constant.copy()
        uv[free_vertices] += np.einsum(
            'nij,nj->ni', free_matrices,
            x.reshape(-1, 2)[free_roots], optimize=True)
        return uv

    def score(x):
        ratios = signed_areas(positions(x), faces) / signed_old
        bad = ratios <= accepted_ratio
        return int(np.count_nonzero(bad)), float(np.min(ratios)), ratios

    history = []
    for penalty in penalties:
        def energy_and_gradient(x):
            residual = A @ x - target
            energy = float(np.dot(residual, residual))
            grad = 2. * np.asarray(A.T @ residual).reshape(-1)
            uv = positions(x)
            triangles = uv[faces]
            a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            signed_new = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                          - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            deficit = np.minimum(orientations * signed_new - bounds, 0.)
            if np.any(deficit):
                energy += float(penalty * np.dot(weights, deficit * deficit))
                dE_darea = 2. * penalty * weights * deficit * orientations
                grads = np.zeros_like(uv)
                corner0 = np.stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0]), axis=1)
                corner1 = np.stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0]), axis=1)
                corner2 = np.stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]), axis=1)
                for k, corner in enumerate((corner0, corner1, corner2)):
                    np.add.at(grads, faces[:, k], dE_darea[:, None] * corner)
                root_grads = np.einsum(
                    'nji,nj->ni', free_matrices,
                    grads[free_vertices], optimize=True)
                reduced = np.zeros((len(x) // 2, 2), float)
                np.add.at(reduced, free_roots, root_grads)
                grad += reduced.reshape(-1)
            return energy, grad

        result = minimize(energy_and_gradient, roots, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-12,
                                             'gtol': 1e-8, 'maxls': 40})
        if not np.all(np.isfinite(result.x)):
            history.append({'penalty': float(penalty), 'status': 'nonfinite'})
            break
        roots = np.asarray(result.x, float)
        count, minimum, _ = score(roots)
        history.append({'penalty': float(penalty), 'bad_faces': count,
                        'min_ratio': minimum, 'iterations': int(result.nit),
                        'optimizer_success': bool(result.success)})
        if count == 0:
            break
    bad_count, min_ratio, _ = score(roots)
    return roots, {'refinement_attempted': True,
                   'refinement_bad_faces': bad_count,
                   'refinement_min_signed_ratio': min_ratio,
                   'refinement_iterations': int(sum(i.get('iterations', 0) for i in history)),
                   'refinement_history': history}
