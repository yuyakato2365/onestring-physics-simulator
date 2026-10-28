"""Boundary-aligned integer-grid map for the 2026-10-28 structured-grid mode.

The logical polygon and the map Omega_1 -> polygon are solved together instead of one after the
other:

* directions of the boundary sides come from a smooth rotation field of the map (harmonic in
  Omega_1): side pairs that are close through the interior (Dirichlet-to-Neumann coupling of the
  cotangent Laplacian) are asked to rotate alike (``side_coupling``);
* unknown = 2D position of every Omega_1 vertex, energy = 3D symmetric Dirichlet (SLIM), linear
  constraints = every boundary side lies on an axis-parallel line, both copies of every cut
  satisfy x_B = R x_A + t (``aligned_slim``);
* corner positions are rounded to the integer lattice (one MILP: closure, equal cut copies,
  simple polygon), then the map is re-solved with the corners fixed and the boundary vertices
  sliding along their sides;
* the M2D grid is the pull-back of the integer lattice through that map (``place_grid_aligned``).
"""
from __future__ import annotations

import numpy as np

from .structured_grid_20261028 import (
    DIRS,
    StructuredGridError,
    _cyclic_positions,
    _lengths_for_directions,
    _locate_bary,
    _rot_between,
    _subside_pairs,
    _super_runs,
    place_grid,
)


# ------------------------------------------------------------------ smooth rotation field coupling
def side_coupling(src, feats: list[int], chart_uv: np.ndarray, chart_f2: np.ndarray, keep: int = 8) -> np.ndarray:
    """W[k, l] >= 0: how strongly sides k and l are tied through the interior of Omega_1.

    The rotation field theta of the map is harmonic; with theta given on the boundary its Dirichlet
    energy is  g^T S g = sum_{i<j} (-S_ij) (g_i - g_j)^2  (S = Schur complement of the cotangent
    Laplacian onto the boundary data vertices).  Summing -S_ij over the vertices of two sides gives
    the side coupling; every side keeps its ``keep`` strongest partners."""
    from scipy import sparse
    from scipy.sparse.linalg import splu
    cuv = np.asarray(chart_uv, float)
    cf = np.asarray(chart_f2, int)
    nV = len(cuv)
    I, J, Wt = [], [], []
    for tr in cf:
        for k in range(3):
            i, j, o = tr[k], tr[(k + 1) % 3], tr[(k + 2) % 3]
            u, v = cuv[i] - cuv[o], cuv[j] - cuv[o]
            cr = abs(u[0] * v[1] - u[1] * v[0])
            w = max(0.5 * float(u @ v) / max(cr, 1e-15), 1e-6)
            I += [i, j]; J += [j, i]; Wt += [w, w]
    A = sparse.coo_matrix((Wt, (I, J)), shape=(nV, nV)).tocsr()
    Lap = (sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A).tocsc()
    loop = np.asarray(src.loop, int)
    n = len(loop)
    m = len(feats)
    fset = set(feats)
    side_of = {}
    for k in range(m):
        for p in _cyclic_positions(n, feats[k], feats[(k + 1) % m])[1:-1]:
            side_of[int(loop[p])] = k
    D = np.array(sorted(side_of), int)
    if len(D) < 2:
        return np.zeros((m, m))
    Iset = np.setdiff1d(np.arange(nV), D)
    LII = Lap[Iset][:, Iset].tocsc()
    LID = Lap[Iset][:, D].toarray()
    X = splu(LII).solve(LID)
    S = Lap[D][:, D].toarray() - Lap[D][:, Iset] @ X
    S = np.asarray(S)
    w = np.maximum(-S, 0.0)
    np.fill_diagonal(w, 0.0)
    sid = np.array([side_of[int(v)] for v in D])
    W = np.zeros((m, m))
    np.add.at(W, (sid[:, None].repeat(len(D), 1), sid[None, :].repeat(len(D), 0)), w)
    np.fill_diagonal(W, 0.0)
    W = 0.5 * (W + W.T)
    out = np.zeros_like(W)
    for k in range(m):
        idx = np.argsort(-W[k])[:keep]
        out[k, idx] = W[k, idx]
    out = np.maximum(out, out.T)
    deg = out.sum(axis=1)
    scale = float(np.mean(deg[deg > 0])) if np.any(deg > 0) else 1.0
    del fset
    return out / max(scale, 1e-300)


# ------------------------------------------------------------------ dropping straight-through features
def merge_straight_features(src, dom: dict) -> dict:
    """Remove feature points where the logical boundary goes straight on (turn 0 on this copy and on
    the paired copy of a seam-cone candidate); OptCuts seam nodes always stay.  The polygon and the
    lattice are unchanged; the remaining features are the true corners."""
    from .zhang2021_onestring_seam_matching import seam_constraints
    feats, turn, dirs, n = dom["features"], dom["turn"], dom["dirs"], dom["lengths"]
    m = len(feats)
    partner = {}
    for ka, kb in dom.get("cones_idx", []):
        partner[ka] = kb
        partner[kb] = ka
    keep = []
    for k in range(m):
        node = bool(src.node_kind.get(feats[k]))
        straight = turn[k] == 0 and (k not in partner or turn[partner[k]] == 0)
        keep.append(node or not straight)
    if sum(keep) < 4:
        return dom
    kidx = [k for k in range(m) if keep[k]]
    new_feats = [feats[k] for k in kidx]
    new_turn = [turn[k] for k in kidx]
    new_dirs = [dirs[k] for k in kidx]
    new_n, new_L = [], []
    L = dom["target_lengths"]
    for r, k in enumerate(kidx):
        k2 = kidx[(r + 1) % len(kidx)]
        span = [(k + i) % m for i in range(((k2 - k) % m) or m)]
        new_n.append(int(sum(n[i] for i in span)))
        new_L.append(float(sum(L[i] for i in span)))
    pos = {k: r for r, k in enumerate(kidx)}
    cones = [(pos[a], pos[b]) for a, b in dom.get("cones_idx", []) if a in pos and b in pos]
    poly = np.asarray(dom["polygon"])[kidx]
    pairs = _subside_pairs(src, new_feats, seam_constraints(src))
    new = dict(dom)
    new.update({"features": new_feats, "turn": new_turn, "dirs": new_dirs, "lengths": new_n,
                "target_lengths": new_L, "polygon": poly, "cut_pairs": pairs, "cones_idx": cones,
                "super_runs": _super_runs(src, new_feats, new_turn, np.asarray(new_n), cones),
                "merged_from": m})
    return new


# ------------------------------------------------------------------ constrained SLIM
def _seam_groups(src, dom):
    feats, dirs = dom["features"], dom["dirs"]
    m = len(feats)
    loop = np.asarray(src.loop, int)
    n = len(loop)
    groups = []
    for ka, kb in dom["cut_pairs"]:
        PA = _cyclic_positions(n, feats[ka], feats[(ka + 1) % m])
        PB = _cyclic_positions(n, feats[kb], feats[(kb + 1) % m])
        posB = {int(src.xyz_id[j]): j for j in PB}
        prs = [(int(loop[a]), int(loop[posB[int(src.xyz_id[a])]])) for a in PA if int(src.xyz_id[a]) in posB]
        if len(prs) < 2:
            continue
        groups.append((_rot_between(DIRS[dirs[ka]], -DIRS[dirs[kb]]), prs))
    return groups


def _side_chains(src, dom):
    feats, dirs = dom["features"], dom["dirs"]
    m = len(feats)
    loop = np.asarray(src.loop, int)
    n = len(loop)
    chains = []
    for k in range(m):
        verts = [int(loop[p]) for p in _cyclic_positions(n, feats[k], feats[(k + 1) % m])]
        chains.append((k, verts, 1 if dirs[k] in (0, 2) else 0))
    return chains


def aligned_slim(prob, uv0, groups, chains, fixed, iters: int = 80, tol: float = 1e-7, area_weight: float = 0.0):
    """SLIM (3D symmetric Dirichlet) under linear constraints kept exactly at every iterate:
    seam transitions x_b = R x_a + t (groups), straight axis-parallel sides (chains: consecutive
    vertices share the coordinate ``ax``), fixed vertices.  Start must satisfy them; redundant
    constraints are tolerated (quasi-definite KKT)."""
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from .zhang2021_onestring_reparam import _max_feasible_step
    uv = np.asarray(uv0, float).copy()
    nV = len(uv)
    rows, cols, vals = [], [], []
    r = 0
    for R, prs in groups:
        a0, b0 = prs[0]
        for a, b in prs[1:]:
            for i in range(2):
                for vtx, sgn in ((b, 1.0), (b0, -1.0)):
                    rows.append(r + i); cols.append(i * nV + vtx); vals.append(sgn)
                for j in range(2):
                    if R[i][j]:
                        for vtx, sgn in ((a, -1.0), (a0, 1.0)):
                            rows.append(r + i); cols.append(j * nV + vtx); vals.append(sgn * R[i][j])
            r += 2
    for _k, verts, ax in chains:
        for a, b in zip(verts[:-1], verts[1:]):
            rows += [r, r]; cols += [ax * nV + a, ax * nV + b]; vals += [1.0, -1.0]; r += 1
    for v, ax in fixed:
        rows.append(r); cols.append(ax * nV + v); vals.append(1.0); r += 1
    C = sparse.csr_matrix((vals, (rows, cols)), shape=(r, 2 * nV))
    x0 = np.r_[uv[:, 0], uv[:, 1]]
    d = C @ x0
    wa = float(area_weight)
    wts = prob.weight / float(np.sum(prob.weight))

    def energy(P):
        """mean symmetric Dirichlet + area_weight * mean log^2 det J (area preservation)."""
        E_, J_, x_ = prob.energy(P)
        if wa > 0 and np.isfinite(E_):
            det = J_[:, 0, 0] * J_[:, 1, 1] - J_[:, 0, 1] * J_[:, 1, 0]
            if np.any(det <= 0):
                return np.inf, J_, x_
            E_ = E_ + wa * float(np.dot(wts, np.log(det) ** 2))
        return E_, J_, x_

    E, J, _ = energy(uv)
    if not np.isfinite(E):
        return uv, {"aligned_status": "start not injective"}
    E_start = E
    sw = np.sqrt(prob.area)
    Dx, Dy = prob.Dx, prob.Dy
    done = 0
    status = "converged"
    for _ in range(int(iters)):
        U, S, Vt = np.linalg.svd(J)
        Rt = U @ Vt
        sv = np.clip(S, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            w2 = np.where(np.abs(sv - 1) > 1e-8, (sv - sv ** -3) / (sv - 1), 4.0)
        w = np.sqrt(np.maximum(w2, 1e-12))
        W = np.einsum("tij,tj,tkj->tik", U, w, U)
        if wa > 0:
            # local target between the closest rotation (shape) and the closest unit-area map
            beta = wa / (1.0 + wa)
            detJ = np.clip(S[:, 0] * S[:, 1], 1e-12, None)
            Rt = (1.0 - beta) * Rt + beta * J / np.sqrt(detJ)[:, None, None]
        WR = W @ Rt
        blocks, rhs = [], []
        for rr in (0, 1):
            for c_, Dm in ((0, Dx), (1, Dy)):
                Au = sparse.diags(sw * W[:, rr, 0]) @ Dm
                Av = sparse.diags(sw * W[:, rr, 1]) @ Dm
                blocks.append(sparse.hstack([Au, Av]))
                rhs.append(sw * WR[:, rr, c_])
        A = sparse.vstack(blocks).tocsc()
        b = np.concatenate(rhs)
        H = (A.T @ A).tocsc()
        hd = float(H.diagonal().mean())
        H = H + sparse.identity(H.shape[0], format="csc") * (1e-10 * hd)
        K = sparse.bmat([[H, C.T], [C, -sparse.identity(r) * (1e-10 * hd)]], format="csc")
        try:
            sol = splu(K).solve(np.r_[A.T @ b, d])
        except RuntimeError as exc:
            status = f"KKT factorisation failed: {exc}"
            break
        xn = sol[:2 * nV]
        x = np.r_[uv[:, 0], uv[:, 1]]
        dx = xn - x
        duv = np.c_[dx[:nV], dx[nV:]]
        dJ = prob.jacobians(duv)
        amax = _max_feasible_step(J, dJ)
        alpha = min(1.0, 0.8 * amax)
        accepted = False
        while alpha > 1e-12:
            trial = uv + alpha * duv
            Et, Jt, _ = energy(trial)
            if np.isfinite(Et) and Et <= E:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            status = "line search stalled"
            break
        rel = (E - Et) / max(E, 1e-30)
        uv, E, J = trial, Et, Jt
        done += 1
        if rel < tol:
            break
    viol = float(np.max(np.abs(C @ np.r_[uv[:, 0], uv[:, 1]] - d))) if r else 0.0
    return uv, {"aligned_sd_before": float(E_start), "aligned_sd_after": float(E), "aligned_iterations": done,
                "aligned_constraint_violation": viol, "aligned_status": status}


def _start_map(src, dom, logical, quads, walk, chart, h):
    """Feasible start: the bijective chart -> polygon map of place_grid, rescaled to lattice unit h."""
    so: dict = {}
    uvm, f2, xyz, f3 = chart
    place_grid(src, dom, logical, quads, walk, chart_uv=uvm, chart_f2=f2, chart_xyz=xyz, chart_f3=f3,
               area_weight=0.0, seam_out=so)
    if "phi" not in so:
        return None, None
    return so["phi"] / so["scale"] * h, so["prob"]


def refine_domain_aligned(src, dom: dict, logical, quads, walk, chart, h: float, w_pos: float = 2.0,
                          min_3d_ratio: float = 0.9, under_weight: float = 3.0, area_weight: float = 0.0):
    """Continuous joint solve (free corners, straight axis-parallel sides, seamless cuts), then one
    MILP rounds the corner positions to the lattice (closure, equal cut copies, simple polygon)."""
    from shapely.geometry import Polygon
    start, prob = _start_map(src, dom, logical, quads, walk, chart, h)
    if start is None:
        return None, {"aligned_refine": "no bijective start map"}
    groups = _seam_groups(src, dom)
    chains = _side_chains(src, dom)
    star, info = aligned_slim(prob, start, groups, chains, fixed=[], iters=120, area_weight=area_weight)
    feats, dirs = dom["features"], dom["dirs"]
    m = len(feats)
    loop = np.asarray(src.loop, int)
    Tst = star[loop[np.asarray(feats)]] / h
    Tst = Tst - Tst[0]
    Lst = np.array([float((Tst[(k + 1) % m] - Tst[k]) @ DIRS[dirs[k]]) for k in range(m)])
    info["aligned_continuous_lengths"] = [round(float(x), 3) for x in Lst]
    info["_star"] = star
    # never fewer cells than the surface needs: every side gets at least ~its 3D length (protrusions
    # such as the ears keep as many panels across as a uniform grid would give them), shortfalls
    # cost ``under_weight`` times more than extra cells
    L3 = np.asarray(dom["target_lengths"], float)
    Lc = np.maximum(np.maximum(Lst, 0.3), min_3d_ratio * L3)
    nn = None
    for lo_ratio in (min_3d_ratio, 0.7 * min_3d_ratio, 0.0):
        n_lo = np.maximum(1, np.maximum(np.floor(lo_ratio * L3), np.floor(0.6 * np.maximum(Lst, 0.3)))).astype(int)
        nn, linfo = _lengths_for_directions(dirs, Lc, n_lo, dom["cut_pairs"], Tst, w_pos, under_weight=under_weight)
        info["aligned_rounding"] = linfo.get("status")
        info["aligned_min_3d_ratio_used"] = lo_ratio
        if nn is not None:
            break
    if nn is None:
        return None, info
    pts = [np.zeros(2, int)]
    for k in range(m):
        pts.append(pts[-1] + nn[k] * DIRS[dirs[k]])
    poly = np.asarray(pts[:-1])
    if np.any(pts[-1] != 0) or not Polygon(poly).is_valid:
        info["aligned_rounding"] = "not a simple closed polygon"
        return None, info
    new = dict(dom)
    new.update({"lengths": nn.tolist(), "polygon": poly,
                "super_runs": _super_runs(src, feats, dom["turn"], nn, dom.get("cones_idx", [])),
                "aligned_lengths_before": list(dom["lengths"])})
    return new, info


def place_grid_aligned(src, dom: dict, logical, quads, walk, chart, h: float, area_weight: float = 0.0):
    """M2D grid = pull-back of the integer lattice through the low-distortion map onto the integer
    polygon (corners fixed, boundary vertices sliding along their sides, cuts seamless)."""
    uvm, f2, xyz, f3 = chart
    start, prob = _start_map(src, dom, logical, quads, walk, chart, h)
    if start is None:
        raise StructuredGridError("SG_ALIGNED_NO_START_MAP")
    feats = dom["features"]
    loop = np.asarray(src.loop, int)
    groups = _seam_groups(src, dom)
    chains = _side_chains(src, dom)
    fixed = [(int(loop[j]), ax) for j in feats for ax in (0, 1)]
    cur, info = aligned_slim(prob, start, groups, chains, fixed, iters=150, area_weight=area_weight)
    e1 = cur[f2[:, 1]] - cur[f2[:, 0]]
    e2 = cur[f2[:, 2]] - cur[f2[:, 0]]
    if np.any(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0):
        raise StructuredGridError("SG_ALIGNED_MAP_NOT_INJECTIVE")
    lg = np.asarray(logical, float) * h
    uv = np.full((len(logical), 2), np.nan)
    # boundary lattice points: walk along the chart boundary chain of their side
    chain_of = {k: (verts, ax) for k, verts, ax in chains}
    dirs = dom["dirs"]
    for vtx, k, _j in walk:
        verts, _ax = chain_of[k]
        d = DIRS[dirs[k]].astype(float)
        s = cur[verts] @ d
        target = float(lg[vtx] @ d)
        i = int(np.clip(np.searchsorted(s, target) - 1, 0, len(s) - 2)) if np.all(np.diff(s) > 0) else \
            int(np.argmin(np.abs(s[:-1] - target)))
        lam = 0.0 if s[i + 1] == s[i] else float(np.clip((target - s[i]) / (s[i + 1] - s[i]), 0.0, 1.0))
        uv[vtx] = uvm[verts[i]] + lam * (uvm[verts[i + 1]] - uvm[verts[i]])
    bnd = np.array(sorted({w[0] for w in walk}))
    gfree = np.setdiff1d(np.arange(len(logical)), bnd)
    out = _locate_bary(cur, f2, lg[gfree], uvm)
    if out is None:
        raise StructuredGridError("SG_ALIGNED_LATTICE_POINT_OUTSIDE")
    uv[gfree] = out
    if np.any(np.isnan(uv)):
        raise StructuredGridError("SG_ALIGNED_GRID_INCOMPLETE")
    info["placement"] = "pull-back of the integer lattice through the boundary-aligned seamless SLIM map"
    place_grid_aligned.last = {"map": cur, "start": start, "dom": dom, "chart": chart, "logical": logical,
                               "quads": quads, "h": h}            # diagnostics only
    return uv, info


# ------------------------------------------------------------------ polysquare deformation
def polysquare_deform(prob, uv0, loop, iters: int = 40, w_start: float = 0.05, w_end: float = 20.0):
    """Deform Omega_1 so that every boundary edge turns to its nearest axis (2D PolySquare,
    rotation-driven as in PolyCube deformation, Gregson et al. 2011), while the interior stays as
    undistorted as possible (SLIM, flip-free).  Curved runs such as the round end of an ear become
    straight sides joined by clear corners, so the boundary headings read from the deformed chart
    give each lobe its full width instead of a column.

    Energy: symmetric Dirichlet + w * sum_e |e| |(P_j - P_i) - |P_j - P_i| a_e|^2, a_e = nearest axis
    of the current edge (re-snapped every iteration), w increased geometrically."""
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from .zhang2021_onestring_reparam import _max_feasible_step
    uv = np.asarray(uv0, float).copy()
    nV = len(uv)
    loop = np.asarray(loop, int)
    ei, ej = loop, np.roll(loop, -1)
    # global rotation that best aligns the boundary with the axes
    e = uv[ej] - uv[ei]
    L = np.linalg.norm(e, axis=1)
    th = float(np.angle(np.sum(L * np.exp(4j * np.arctan2(e[:, 1], e[:, 0])))) / 4.0)
    Rg = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    uv = uv @ Rg.T
    sw = np.sqrt(prob.area)
    Dx, Dy = prob.Dx, prob.Dy
    perim = float(np.sum(L))
    a_tot = float(np.sum(prob.area))
    E_sd, J, _ = prob.energy(uv)
    rowsB = np.arange(len(ei))
    DB = sparse.csr_matrix((np.r_[-np.ones(len(ei)), np.ones(len(ei))], (np.r_[rowsB, rowsB], np.r_[ei, ej])),
                           shape=(len(ei), nV))

    def align_energy(P, A_dirs, w):
        d = P[ej] - P[ei]
        ln = np.linalg.norm(d, axis=1)
        r = d - ln[:, None] * A_dirs
        return w * float(np.sum(ln * np.sum(r * r, axis=1))) / perim ** 2

    ws = np.geomspace(w_start, w_end, iters)
    for it, w in enumerate(ws):
        d = uv[ej] - uv[ei]
        ln = np.linalg.norm(d, axis=1)
        ang = np.arctan2(d[:, 1], d[:, 0])
        q = np.round(ang / (0.5 * np.pi))
        A_dirs = np.c_[np.cos(q * 0.5 * np.pi), np.sin(q * 0.5 * np.pi)]
        U, S, Vt = np.linalg.svd(J)
        Rt = U @ Vt
        sv = np.clip(S, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            w2 = np.where(np.abs(sv - 1) > 1e-8, (sv - sv ** -3) / (sv - 1), 4.0)
        Wl = np.einsum("tij,tj,tkj->tik", U, np.sqrt(np.maximum(w2, 1e-12)), U)
        WR = Wl @ Rt
        blocks, rhs = [], []
        for rr in (0, 1):
            for c_, Dm in ((0, Dx), (1, Dy)):
                blocks.append(sparse.hstack([sparse.diags(sw * Wl[:, rr, 0]) @ Dm, sparse.diags(sw * Wl[:, rr, 1]) @ Dm]))
                rhs.append(sw * WR[:, rr, c_])
        # boundary alignment rows (scaled like the area-weighted SD rows)
        sb = np.sqrt(w * ln * a_tot) / perim
        Z = sparse.csr_matrix((len(ei), nV))
        blocks.append(sparse.hstack([sparse.diags(sb) @ DB, Z]))
        rhs.append(sb * ln * A_dirs[:, 0])
        blocks.append(sparse.hstack([Z, sparse.diags(sb) @ DB]))
        rhs.append(sb * ln * A_dirs[:, 1])
        A = sparse.vstack(blocks).tocsc()
        b = np.concatenate(rhs)
        H = (A.T @ A).tocsc()
        H = H + sparse.identity(H.shape[0], format="csc") * (1e-9 * float(H.diagonal().mean()))
        x = splu(H).solve(A.T @ b)
        new = np.c_[x[:nV], x[nV:]]
        # keep the centroid (translation is free)
        new += uv.mean(axis=0) - new.mean(axis=0)
        duv = new - uv
        dJ = prob.jacobians(duv)
        amax = _max_feasible_step(J, dJ)
        alpha = min(1.0, 0.8 * amax)
        E0 = E_sd + align_energy(uv, A_dirs, w)
        while alpha > 1e-10:
            trial = uv + alpha * duv
            Et, Jt, _ = prob.energy(trial)
            if np.isfinite(Et) and Et + align_energy(trial, A_dirs, w) <= E0:
                break
            alpha *= 0.5
        else:
            continue
        uv, E_sd, J = trial, Et, Jt
    return uv, {"polysquare_sd": float(E_sd), "polysquare_iterations": len(ws)}


# ------------------------------------------------------------------ cone feedback
def cone_feedback(src, cone_pairs, star, chart, h: float, current: dict, radius: float = 2.0,
                  threshold: float = 0.35):
    """Seam cones asked for by the continuous map itself.

    Where the boundary-aligned map squeezes the surface (log2 area scale < -threshold within
    ``radius`` panels of a seam-cone candidate) the region carries Gaussian curvature that no cone
    releases (a cap such as an ear: too few cells); a +1 cone (3D valence 3) is added there.
    Where it stretches (> threshold) a -1 cone (valence 5).  Greedy, strongest first, at most one
    new cone per 2*radius panels; existing cones are kept."""
    uvm, f2, xyz, f3 = chart
    a2 = 0.5 * np.abs((star[f2[:, 1], 0] - star[f2[:, 0], 0]) * (star[f2[:, 2], 1] - star[f2[:, 0], 1])
                      - (star[f2[:, 1], 1] - star[f2[:, 0], 1]) * (star[f2[:, 2], 0] - star[f2[:, 0], 0]))
    X = np.asarray(xyz, float)[np.asarray(f3, int)]
    a3 = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
    ls = np.log2(np.maximum(a2, 1e-300) / np.maximum(a3, 1e-300))
    ls -= float(np.sum(ls * a3) / np.sum(a3))
    C3 = X.mean(axis=1)
    sites = np.array([np.asarray(xyz, float)[int(src.xyz_id[a])] for a, _b in cone_pairs]) if cone_pairs else np.zeros((0, 3))
    mean = np.zeros(len(sites))
    for q, p in enumerate(sites):
        w = a3 * (np.linalg.norm(C3 - p, axis=1) < radius * h)
        mean[q] = float(np.sum(w * ls) / np.sum(w)) if np.sum(w) > 0 else 0.0
    taken = [sites[q] for q, v in current.items() if v]
    out = dict(current)
    added = []
    for q in np.argsort(-np.abs(mean)):
        q = int(q)
        if abs(mean[q]) < threshold or q in out:
            continue
        if any(np.linalg.norm(sites[q] - t) < 2.0 * radius * h for t in taken):
            continue
        out[q] = 1 if mean[q] < 0 else -1
        taken.append(sites[q])
        added.append({"candidate": q, "xyz_id": int(src.xyz_id[cone_pairs[q][0]]), "log2_area_scale": round(mean[q], 3),
                      "valence_3d": 3 if mean[q] < 0 else 5})
    return out, added


def current_cones(src, cone_pairs, dom) -> dict:
    """Candidate index -> t_A + t_B of the cones in a logical domain."""
    by_xyz = {int(c["xyz_id"]): 4 - int(c["valence_3d"]) for c in dom.get("seam_cones", [])}
    return {q: by_xyz[int(src.xyz_id[a])] for q, (a, _b) in enumerate(cone_pairs) if int(src.xyz_id[a]) in by_xyz}


# ------------------------------------------------------------------ gluing back sub-panel cut branches
def prune_short_branches(xyz, f3, uv, f2, h: float, max_ratio: float = 1.5, sd_ratio_max: float = 1.15,
                         rounds: int = 3):
    """Sew back every leaf cut branch (a slit ending in a cut tip) whose 3D length is below
    ``max_ratio`` panels.  Such a branch is narrower than one panel, so the grid cannot follow
    it; it only leaves a spike / notch in Omega.  The released curvature is absorbed by the
    flattening (untangle + SLIM, boundary elsewhere fixed); rejected if the mean symmetric
    Dirichlet energy grows more than ``sd_ratio_max``."""
    from .zhang2021_onestring_boundary import build_source_boundary, uv_to_xyz_vertex_map, oriented_boundary_loops
    from .zhang2021_onestring_reparam import _SDProblem, _slim_iterations, signed_uv_areas
    from .zhang2021_onestring_seam_matching import seam_constraints
    from .structured_grid_20261028 import _untangle_triangles
    xyz = np.asarray(xyz, float); f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float).copy(); f2 = np.asarray(f2, int).copy()
    sa = signed_uv_areas(uv, f2)
    mirror = np.array([-1.0, 1.0]) if np.all(sa < 0) else np.array([1.0, 1.0])
    E_ref = float(_SDProblem(xyz, f3, f2, len(uv)).energy(uv * mirror)[0])
    report = []
    for _round in range(rounds):
        uvm = uv * mirror
        src = build_source_boundary(uvm, f2, uv_to_xyz_vertex_map(f3, f2, len(uv)), xyz)
        loop = np.asarray(src.loop, int)
        n = len(loop)
        tot3 = float(src.cum3[-1])
        merge = {}
        for con in seam_constraints(src):
            if con.kind != "slit":
                continue
            a0, a1 = con.run_a
            length = (float(src.cum3[a1]) - float(src.cum3[a0])) % tot3
            if length >= max_ratio * h:
                continue
            posB = {int(src.xyz_id[j]): j for j in _cyclic_positions(n, *con.run_b)}
            for p in _cyclic_positions(n, a0, a1):
                q = posB.get(int(src.xyz_id[p]))
                if q is not None and loop[q] != loop[p]:
                    merge[int(loop[q])] = int(loop[p])
            report.append({"round": _round, "tip_xyz": int(src.xyz_id[a1]), "length_panels": round(length / h, 3)})
        if not merge:
            break
        new_uv = uvm.copy()
        for b, a in merge.items():
            new_uv[a] = 0.5 * (uvm[a] + uvm[b])
        f2n = f2.copy()
        for b, a in merge.items():
            f2n[f2n == b] = a
        used = np.unique(f2n)
        remap = -np.ones(len(uv), int)
        remap[used] = np.arange(len(used))
        f2n = remap[f2n]
        P = new_uv[used]
        loops = oriented_boundary_loops(f2n)
        if len(loops) != 1:
            report.append({"round": _round, "rejected": f"{len(loops)} boundary loops after sewing"})
            break
        bset = np.asarray(loops[0], int)
        touched = set(remap[a] for a in merge.values())
        # free: everything within 3 rings of the sewn seam, never the outer boundary
        adj = {}
        for t in f2n:
            for i in range(3):
                adj.setdefault(int(t[i]), set()).update(int(x) for x in t)
        ring = set(touched)
        for _ in range(3):
            ring |= {w for v in ring for w in adj.get(v, ())}
        free_local = np.array(sorted(ring - set(bset.tolist())), int)
        P = _untangle_triangles(P, f2n, free_local, iters=800)
        e1 = P[f2n[:, 1]] - P[f2n[:, 0]]
        e2 = P[f2n[:, 2]] - P[f2n[:, 0]]
        if np.any(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] <= 0):
            report.append({"round": _round, "rejected": "flipped triangles after sewing"})
            break
        prob = _SDProblem(xyz, f3, f2n, len(P))
        free_all = np.setdiff1d(np.arange(len(P)), bset)
        P, E1, _it = _slim_iterations(prob, P, free_all, iters=200, tol=1e-7)
        if not np.isfinite(E1) or E1 / E_ref > sd_ratio_max:
            report.append({"round": _round, "rejected": f"SD mean x{E1 / E_ref:.3f}"})
            break
        uv, f2 = P * mirror, f2n
        report.append({"round": _round, "sewn_branches": sum(1 for r in report if r.get("round") == _round
                                                              and "tip_xyz" in r), "sd_ratio": round(E1 / E_ref, 4)})
    return uv, f2, report


__all__ = ["cone_feedback", "current_cones", "prune_short_branches", "side_coupling", "merge_straight_features", "aligned_slim", "refine_domain_aligned",
           "place_grid_aligned"]
