"""SE(2) rigid placement, signed hinge-face selection and nonpenetration projection."""

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve


def choose(solid, hinges):
    a, _, b, _ = hinges.T
    top = solid[:, :3]
    center = top.mean(1)
    normal = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    normal *= np.sign((normal * (center - solid[:, 7:10].mean(1))).sum(1))[:, None]
    return np.where(
        ((normal[b] - normal[a]) * (center[b] - center[a])).sum(1) < -1e-10, 7, 0
    )


def kagome_seed(flat, hinges, cells, up, angle):
    """Orient the specified hinge corners towards their lattice neighbours."""
    centers = cells.mean(1)
    pitch = np.linalg.norm(flat[:, 1] - flat[:, 0], axis=1).mean()
    target = np.zeros((len(flat), 3, 2))
    a, k, b, l = hinges.T
    target[a, k] = centers[b] - centers[a]
    target[b, l] = centers[a] - centers[b]
    low = flat[:, :3, :2]
    high = flat[:, 7:10, :2]
    relative = ((low - low.mean(1)[:, None]) + (high - high.mean(1)[:, None])) / 2
    theta = np.arctan2(
        (relative[:, :, 0] * target[:, :, 1] - relative[:, :, 1] * target[:, :, 0]).sum(1),
        (relative * target).sum((1, 2)),
    ) + np.radians(angle) * np.where(up, 1, -1)
    co, si = np.cos(theta), np.sin(theta)
    rotation = np.stack([co, si, -si, co], 1).reshape(-1, 2, 2)
    q = flat.copy()
    q[:, :, :2] = (q[:, :, :2] - low.mean(1)[:, None]) @ rotation + (centers * pitch * 1.97)[:, None]
    return q


def solve(flat, hinges, side):
    # Alternate rigid translations and rotations. The final LP enforces a separating plane for each nearby footprint.
    n = len(flat)
    a, k, b, l = hinges.T
    sel = [0, 1, 2, 7, 8, 9]
    base = flat[:, :3, :2].mean(1)
    rel = flat[:, :, :2] - base[:, None]
    radius = np.linalg.norm(rel[:, sel], axis=2).max(1)
    x = np.c_[base, np.zeros(n)]
    k = k + side
    l = l + side

    # Hull edges can connect different corners of the two triangle layers.
    axes = []
    for tile in rel[:, sel]:
        hull = tile[ConvexHull(tile).vertices]
        edges = np.roll(hull, -1, axis=0) - hull
        normals = edges[:, [1, 0]] * [-1, 1]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        axes.append(np.pad(normals, ((0, 6-len(normals)), (0, 0)), mode='edge'))
    axes = np.asarray(axes)

    def state(x):
        c, s = np.cos(x[:, 2]), np.sin(x[:, 2])
        rot = np.stack([c, s, -s, c], 1).reshape(-1, 2, 2)
        v = rel @ rot
        perp = v[:, :, [1, 0]] * [-1, 1]
        p = v + x[:, None, :2]
        return p, perp, rot

    def linear(x, weight):
        p, perp, rot = state(x)
        scope = cKDTree(x[:, :2]).query_pairs(
            2 * radius.max() + 1, output_type="ndarray"
        )
        i, j = scope.T
        near = (
            np.linalg.norm(x[i, :2] - x[j, :2], axis=1) <= radius[i] + radius[j] + 0.2
        )
        i, j = i[near], j[near]
        hull = p[:, sel]
        normals = axes @ rot
        ax = np.concatenate((normals[i], normals[j]), axis=1)
        u = ax @ hull[i].transpose(0, 2, 1)
        v = ax @ hull[j].transpose(0, 2, 1)
        f = v.min(2) - u.max(2)
        g = u.min(2) - v.max(2)
        both = np.concatenate((f, g), 1)
        key = both.argmax(1)
        rr = np.arange(len(i))
        gap = both[rr, key]
        act = gap < 0.025
        i, j = i[act], j[act]
        key = key[act]
        rr = rr[act]
        axis = ax[rr, key % 12] * np.where(key < 12, 1.0, -1.0)[:, None]
        ia = np.array(sel)[np.argmax(np.einsum("nki,ni->nk", hull[i], axis), axis=1)]
        jb = np.array(sel)[np.argmin(np.einsum("nki,ni->nk", hull[j], axis), axis=1)]
        delta = p[a, k] - p[b, l]
        z = flat[a, k, 2] - flat[b, l, 2]
        # Penalize only residuals beyond the fabrication tolerance more heavily.
        lengths = np.sqrt((delta**2).sum(1) + z * z)
        wt = np.sqrt(1 + 30 * np.maximum(lengths - 1.45, 0) ** 2)
        dif = np.vstack((delta[:, 0, None], delta[:, 1, None])).ravel()
        A = np.r_[a, a, i]
        B = np.r_[b, b, j]
        axis2 = np.vstack(
            (np.tile([1.0, 0.0], (len(a), 1)), np.tile([0.0, 1.0], (len(a), 1)), axis)
        )
        aa = np.r_[k, k, ia]
        bb = np.r_[l, l, jb]
        weights = np.r_[wt, wt, np.full(len(i), weight)]
        jac = (
            np.c_[
                axis2,
                (perp[A, aa] * axis2).sum(1),
                -axis2,
                -(perp[B, bb] * axis2).sum(1),
            ]
            * weights[:, None]
        )
        cols = np.c_[3 * A[:, None] + np.arange(3), 3 * B[:, None] + np.arange(3)]
        J = coo_matrix(
            (jac.ravel(), (np.repeat(np.arange(len(A)), 6), cols.ravel())),
            shape=(len(A), 3 * n),
        ).tocsr()
        err = np.r_[dif, 0.025 - gap[act]] * weights
        return J, err, lengths, gap

    for weight in [5, 20, 100, 500]:
        for it in range(60):
            J, e, le, g = linear(x, weight)
            loss = e @ e
            d = spsolve((J.T @ J + eye(3 * n) * 0.025).tocsc(), -J.T @ e).reshape(-1, 3)
            for rate in [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
                test = x + rate * d
                _, er, _, _ = linear(test, weight)
                if er @ er < loss:
                    x = test
                    break
            else:
                break
            if np.max(abs(rate * d)) < 1e-5:
                break
        print(weight, it, le.max(), (le > 2).sum(), g.min(), flush=True)
    out = flat.copy()
    out[:, :, :2] = state(x)[0]
    return out


import numpy as np
from shapely.geometry import MultiPoint
from shapely.strtree import STRtree
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def separate(p, h, side):
    a, k, b, l = h.T
    k = k + side
    l = l + side
    n = len(p)
    for it in range(12):
        polygons = [MultiPoint(t[:, :2]).convex_hull for t in p]
        tree = STRtree(polygons)
        ai, bi = tree.query([g.buffer(0.5) for g in polygons])
        rr = []
        cc = []
        vv = []
        limits = []

        def row(i, j, d, limit, slack=False):
            r = len(limits)
            rr.extend([r] * 4)
            cc.extend([2 * i, 2 * i + 1, 2 * j, 2 * j + 1])
            vv.extend([*d, *-d])
            limits.append(limit)
            if slack:
                rr.append(r)
                cc.append(2 * n)
                vv.append(-1)

        for i, j in zip(ai[ai < bi], bi[ai < bi]):
            A, B = [np.asarray(polygons[t].exterior.coords) for t in (i, j)]
            edges = np.vstack([np.diff(A, axis=0), np.diff(B, axis=0)])
            ax = edges[:, ::-1] * [-1, 1]
            ax /= np.linalg.norm(ax, axis=1)[:, None]
            ax = np.vstack([ax, -ax])
            gap = (B @ ax.T).min(0) - (A @ ax.T).max(0)
            m = gap.argmax()
            row(i, j, ax[m], gap[m] - 0.025)
        axes = np.c_[
            np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8)
        ]
        for i, ki, j, kj in zip(a, k, b, l):
            delta = p[i, ki] - p[j, kj]
            for d in axes:
                row(
                    i,
                    j,
                    d,
                    min(1.95, np.sqrt(max(0, 2**2 - delta[2] ** 2)))
                    * np.cos(np.pi / 16)
                    - delta[:2] @ d,
                    True,
                )
        M = coo_matrix((vv, (rr, cc)), shape=(len(limits), 2 * n + 1)).tocsr()
        res = linprog(
            np.r_[np.zeros(2 * n), 1],
            A_ub=M,
            b_ub=limits,
            bounds=[(-2, 2)] * (2 * n) + [(0, None)],
            method="highs",
        )
        if not res.success:
            raise ValueError(res.message)
        p[:, :, :2] += res.x[:-1].reshape(n, 1, 2)
        polys = [MultiPoint(t[:, :2]).convex_hull for t in p]
        ii, jj = STRtree(polys).query(polys, predicate="intersects")
        hit = sum(
            polys[i].intersection(polys[j]).area > 1e-8
            for i, j in zip(ii[ii < jj], jj[ii < jj])
        )
        span = np.linalg.norm(p[a, k] - p[b, l], axis=1)
        print("separation", it, "pairs", hit, "maxgap", span.max(), flush=True)
        if not hit:
            return p
    raise ValueError("交差を解消できません。")

def place_shell(base, hinges, side, cells, up, angles, refine=False):
    """Keep the closest collision-free rigid placement, including the input."""
    a, k, b, l = hinges.T

    def span(p):
        return np.linalg.norm(p[a, k + side] - p[b, l + side], axis=1).max()

    def separated(points):
        polygons = [MultiPoint(tile[:, :2]).convex_hull for tile in points]
        ii, jj = STRtree(polygons).query(polygons, predicate="intersects")
        return not (ii < jj).any()

    best = base if separated(base) else None
    best_gap = span(base) if best is not None else np.inf
    selected = None
    for angle in angles:
        if best_gap <= 2.000001:
            break
        seed = base if angle is None else kagome_seed(base, hinges, cells, up, angle)
        try:
            placed = separate(solve(seed, hinges, side), hinges, side)
        except ValueError as exc:
            print(f"平面配置 {angle}: {exc}", flush=True)
            continue
        gap = span(placed)
        print(f"平面配置 {angle}: 最大角間隔 {gap:.3f} mm", flush=True)
        if gap < best_gap:
            best, best_gap, selected = placed, gap, angle
    if best is None:
        raise ValueError("厚みを保ったパネルを平面上で分離できません。")
    if refine and best_gap > 2.000001:
        from separated_layout import separated_layout
        corners = best[:, [0, 1, 2, 7, 8, 9]]
        joints = hinges.copy()
        joints[:, [1, 3]] += 3 * (side != 0)[:, None]
        near = separated_layout(corners, joints, cells, up, 0., steps=60,
                                compact_from=corners, compact_mean_weight=.03)
        v = corners[:, 1, :2] - corners[:, 0, :2]
        w = near[:, 1, :2] - near[:, 0, :2]
        theta = np.arctan2(v[:, 0]*w[:, 1]-v[:, 1]*w[:, 0], (v*w).sum(1))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.stack([c, s, -s, c], 1).reshape(-1, 2, 2)
        candidate = best.copy()
        candidate[:, :, :2] = (best[:, :, :2]-best[:, :1, :2]) @ rot + near[:, :1, :2]
        if span(candidate) < best_gap and separated(candidate):
            best = candidate
            print(f"剛体配置の微調整: 最大角間隔 {span(best):.3f} mm", flush=True)
    return best, selected
