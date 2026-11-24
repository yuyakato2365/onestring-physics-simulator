"""Constrain the UVs on a candidate OptCuts disk while retaining its seam."""
import numpy as np
from scipy import sparse
from shapely.geometry import LinearRing


class ChartConstraintError(ValueError):
    """The constrained solve did not produce an admissible chart on this seam."""


def constrain_chart(xyz, faces, uv, uv_faces, boundary, epsilon, sd_bound):
    """Nearest UV update in the gradient norm, with one distortion cone per face."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('等角歪み上限は0以上の有限値で指定してください。')
    p = xyz[faces]
    e, f = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    length = np.linalg.norm(e, axis=1)
    area2 = np.linalg.norm(np.cross(e, f), axis=1)
    if not np.all(area2 > 0):
        raise ValueError('入力曲面に退化した三角形があります。')
    weights = area2 / area2.sum()
    # Reference coordinates of a face: (0,0), (length,0), (along,height).
    along = np.sum(e * f, axis=1) / length
    height = area2 / length
    def derivative(a, b):
        vals = np.column_stack((-a-b, a, b))
        return sparse.csr_matrix((vals.ravel(), (np.repeat(np.arange(len(faces)), 3),
                                 uv_faces.ravel())), shape=(len(faces), len(uv)))
    dx = derivative(1 / length, np.zeros_like(length))
    dy = derivative(-along / (length * height), 1 / height)
    # The complex derivatives alpha and beta of the piecewise linear map.
    matrices = [sparse.hstack(pair, format='csr') * .5 for pair in
                ((dx, dy), (-dy, dx), (dx, -dy), (dy, dx))]
    ar, ai, br, bi = matrices
    def evaluate(x):
        a, b, c, d = [m @ x for m in matrices]
        norm_a, norm_b = np.hypot(a, b), np.hypot(c, d)
        hi, lo = norm_a + norm_b, norm_a - norm_b
        if not np.all(lo > 0):
            raise ChartConstraintError('Ωに反転または退化した面があります。')
        distortion = (2 * norm_b)**2 / (hi * lo)
        sd = float(weights @ (hi**2 + lo**2 + hi**-2 + lo**-2))
        return distortion, sd, (a, b, c, d, hi, lo)

    x = np.asarray(uv, dtype=float).T.flatten()
    initial, initial_sd, _ = evaluate(x)
    # |beta| / |alpha| <= sqrt(epsilon / (4 + epsilon)) is exactly the
    # requested inequality. A fixed rotation gives a convex inner approximation.
    k = np.sqrt(epsilon / (4 + epsilon)) * (1 - 1e-5)
    for iteration in range(13):
        errors, sd, values = evaluate(x)
        candidate = x.reshape(2, -1).T.copy()
        if errors.max() <= epsilon + 1e-12 and sd <= sd_bound + 1e-10:
            # Positive local orientation and simple boundary on the OptCuts disk
            # imply global injectivity. Include the boundary in the final audit.
            ring = LinearRing(candidate[boundary])
            if not ring.is_valid or not ring.is_simple:
                raise ChartConstraintError('等角制約を満たした候補のΩ外周が自己交差しています。')
            return candidate, dict(conformal_limit=epsilon, conformal_max=float(errors.max()),
                                   conformal_initial_max=float(initial.max()), conformal_sd=sd,
                                   conformal_initial_sd=initial_sd, conformal_steps=iteration)
        if iteration == 12:
            break
        if iteration == 0:
            import scs
            lap = dx.T @ dx.multiply(weights[:, None]) + dy.T @ dy.multiply(weights[:, None])
            lap += sparse.eye(len(uv)) * (1e-6 / len(uv))
            hessian = sparse.triu(sparse.block_diag((lap, lap)), format='csc')
        a, b, c, d, hi, lo = values
        norm_a, norm_b = (hi + lo) / 2, np.hypot(c, d)
        rot_a, rot_b = a / norm_a, b / norm_a
        projection = ar.multiply(rot_a[:, None]) + ai.multiply(rot_b[:, None])
        # First-order SD bound, re-evaluated exactly on every new chart.
        slope_hi, slope_lo = 2 * (hi - hi**-3), 2 * (lo - lo**-3)
        da = weights * (slope_hi + slope_lo)
        db = weights * np.divide(slope_hi - slope_lo, norm_b,
                                  out=np.zeros_like(lo), where=norm_b > 1e-14)
        grad = projection.T @ da + br.T @ (db * c) + bi.T @ (db * d)
        n = len(faces)
        order = np.arange(3 * n).reshape(3, n).T.ravel()
        cones = sparse.vstack([k * projection, br, bi], format='csr')[order]
        # Fix the area gauge to first order. Renormalize exactly after the step.
        area_grad = 2 * (ar.T @ (weights*a) + ai.T @ (weights*b)
                         - br.T @ (weights*c) - bi.T @ (weights*d))
        system = sparse.vstack([sparse.csr_matrix(area_grad), sparse.csr_matrix(grad),
                                -cones], format='csc')
        rhs = np.r_[0., sd_bound - sd - 1e-7, np.column_stack((k * norm_a, c, d)).ravel()]
        solution = scs.solve(dict(P=hessian, A=system, b=rhs, c=np.zeros_like(x)),
                             dict(z=1, l=1, q=[3]*n), verbose=False,
                             eps_abs=1e-8, eps_rel=1e-8, max_iters=30000)
        if solution['info']['status_val'] not in (1, 2):
            break
        x += solution['x']
        _, _, trial = evaluate(x)
        x /= np.sqrt(weights @ (trial[-2] * trial[-1]))
    raise ChartConstraintError(f'現在のseamで制約を満たすΩが見つかりませんでした。'
                               f'等角歪み最大 {errors.max():.7g}（上限 {epsilon:g}）、'
                               f'対称Dirichlet歪み {sd:.7g}（上限 {sd_bound:g}）。')
