"""Cone-first seams for the 2026-10-28 structured-grid mode (OneString).

A regular quad grid whose seams close with quads is a seamless parameterisation;
its valence-3 / valence-5 vertices (cones) must sit where the surface's curvature
asks for them, and - because the 09-18 K2D linkage needs even valences inside
M2D - every cone must lie on a seam.  OptCuts places its cuts for distortion,
not for quads, so this module

1. computes the smoothest 4-direction (cross) field on the 3D surface
   (Knoeppel, Crane, Pinkall, Schroeder 2013, "Globally optimal direction
   fields": cotangent connection Laplacian, smallest eigenvector) and reads its
   singularities (index +1/4 -> valence 3, -1/4 -> valence 5);
2. adds short cut branches (3D shortest paths) from every +1/4 singularity that
   is not near the OptCuts cut to the cut (the new branch tip is the cone), and
   snaps singularities near the cut onto it (cones in the middle of a seam);
3. splits the OptCuts chart along the new branches and opens the slits with a
   free-boundary bijective SLIM (3D symmetric Dirichlet).

The cones found here are passed to the logical-domain MILP as forced seam
cones.  Everything is reported; nothing here stops the run.
"""
from __future__ import annotations

from collections import defaultdict
import heapq

import numpy as np


# ------------------------------------------------------------------ cross field
def _corner_angles(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    A = np.zeros(F.shape)
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, v = X[b] - X[a], X[c] - X[a]
        cs = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        A[:, k] = np.arccos(np.clip(cs, -1.0, 1.0))
    return A


def cross_field(X: np.ndarray, F: np.ndarray, n: int = 4, align_boundary: bool = True):
    """Smoothest n-direction field on vertices; returns (u (complex, |u|=1), info dict).

    align_boundary: the field is tangent/normal to the open boundary (the grid's boundary
    follows grid lines), which removes spurious singularities along it."""
    from scipy import sparse
    from scipy.sparse.linalg import splu

    X = np.asarray(X, float)
    F = np.asarray(F, int)
    nv = len(X)
    ang = _corner_angles(X, F)
    # boundary vertices
    ec = defaultdict(int)
    for f in F:
        for k in range(3):
            ec[tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))] += 1
    bnd = np.zeros(nv, bool)
    for e, c in ec.items():
        if c == 1:
            bnd[list(e)] = True
    angsum = np.zeros(nv)
    np.add.at(angsum, F.ravel(), ang.ravel())
    scale = np.where(bnd, np.pi, 2 * np.pi) / np.maximum(angsum, 1e-12)
    # ordered one-rings: outgoing half-edge v->b, next (CCW) is v->c for face (v,b,c)
    nxt = {}
    corner = {}
    has_prev = set()
    for fi, f in enumerate(F):
        for k in range(3):
            v, b, c = int(f[k]), int(f[(k + 1) % 3]), int(f[(k + 2) % 3])
            nxt[(v, b)] = c
            corner[(v, b)] = ang[fi, k]
            has_prev.add((v, c))
    out = defaultdict(list)
    for (v, b) in nxt:
        out[v].append(b)
    phi = {}
    for v in range(nv):
        if not out[v]:
            continue
        starts = [b for b in out[v] if (v, b) not in has_prev]
        b0 = starts[0] if starts else out[v][0]
        a = 0.0
        b = b0
        for _ in range(len(out[v]) + 2):
            phi[(v, b)] = a * scale[v]
            if (v, b) not in nxt:
                break
            a += corner[(v, b)]
            b = nxt[(v, b)]
            if b == b0:
                break
        # the last edge of a boundary fan (no next): give it its angle too
        if (v, b) not in phi:
            phi[(v, b)] = a * scale[v]
    # cotangent weights
    W = defaultdict(float)
    for fi, f in enumerate(F):
        for k in range(3):
            i, j, o = int(f[(k + 1) % 3]), int(f[(k + 2) % 3]), int(f[k])
            W[tuple(sorted((i, j)))] += 0.5 / np.tan(max(ang[fi, k], 1e-6))
    rows, cols, vals = [], [], []
    diag = np.zeros(nv)
    trans = {}
    for (i, j), w in W.items():
        if (i, j) not in phi or (j, i) not in phi:
            continue
        w = max(w, 1e-6)
        r = phi[(j, i)] - phi[(i, j)] + np.pi          # transport i -> j
        trans[(i, j)] = r
        trans[(j, i)] = -r
        z = np.exp(1j * n * r)
        rows += [j, i]; cols += [i, j]; vals += [-w * z, -w * np.conj(z)]
        diag[i] += w; diag[j] += w
    A = sparse.coo_matrix((np.r_[vals, diag], (np.r_[rows, np.arange(nv)], np.r_[cols, np.arange(nv)])),
                          shape=(nv, nv)).tocsc().astype(complex)
    area = np.zeros(nv)
    tri_area = 0.5 * np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    np.add.at(area, F.ravel(), np.repeat(tri_area / 3.0, 3))
    M = sparse.diags(area).tocsc().astype(complex)
    if align_boundary and np.any(bnd):
        # boundary frames start on a boundary edge (angle 0): u = 1 there; harmonic-smoothest inside
        B = np.flatnonzero(bnd)
        I = np.flatnonzero(~bnd)
        A = A.tocsr()
        AII = A[I][:, I].tocsc()
        AIB = A[I][:, B]
        u = np.ones(nv, complex)
        u[I] = splu(AII).solve(-(AIB @ u[B]))
    else:
        lu = splu((A + 1e-8 * M).tocsc())
        u = np.exp(1j * np.random.default_rng(0).uniform(0, 2 * np.pi, nv))
        for _ in range(60):
            u = lu.solve(M @ u)
            u /= np.sqrt(np.real(np.vdot(u, M @ u)))
    u = u / np.maximum(np.abs(u), 1e-300)
    # singularity index per face
    th = np.angle(u)
    sing = []
    for fi, f in enumerate(F):
        tot = 0.0
        hol = 0.0
        ok = True
        for k in range(3):
            i, j = int(f[k]), int(f[(k + 1) % 3])
            if (i, j) not in trans:
                ok = False
                break
            r = trans[(i, j)]
            hol += r
            d = np.angle(np.exp(1j * (th[j] - th[i] - n * r)))
            tot += d
        if not ok:
            continue
        H = np.angle(np.exp(1j * hol))                    # holonomy of the triangle, wrapped
        idx = (tot + n * H) / (2 * np.pi * n)
        m = int(np.rint(idx * n))
        if m != 0:
            sing.append((fi, m))                          # index = m / n
    return u, {"singular_faces": sing, "boundary_vertex": bnd, "angle_sum": angsum}


def _smoothed(X, F, iters: int):
    from scipy import sparse
    if iters <= 0:
        return X
    n = len(X)
    I = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
    J = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    A = sparse.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
    Y = np.asarray(X, float).copy()
    for _ in range(int(iters)):
        Y = 0.5 * Y + 0.5 * (A @ Y) / deg[:, None]
    return Y


# ------------------------------------------------------------------ cut graph
def chart_cut_edges(f3: np.ndarray, f2: np.ndarray):
    """3D edges that are OptCuts seams (two UV copies) and open-boundary edges (one)."""
    uv_e = defaultdict(int)
    for f in f2:
        for k in range(3):
            uv_e[tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))] += 1
    xyz_of = {}
    for a, b in zip(f2.ravel(), f3.ravel()):
        xyz_of[int(a)] = int(b)
    cnt = defaultdict(int)
    for e, c in uv_e.items():
        if c == 1:
            cnt[tuple(sorted((xyz_of[e[0]], xyz_of[e[1]])))] += 1
    cut = {e for e, c in cnt.items() if c == 2}
    open_ = {e for e, c in cnt.items() if c == 1}
    return cut, open_


def _dijkstra(adj, src: int, targets: set, banned: set):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist.get(v, np.inf):
            continue
        if v in targets and v != src:
            path = [v]
            while path[-1] != src:
                path.append(prev[path[-1]])
            return path[::-1], d
        for w, l in adj[v]:
            if w in banned and w not in targets:
                continue
            nd = d + l
            if nd < dist.get(w, np.inf):
                dist[w] = nd
                prev[w] = v
                heapq.heappush(pq, (nd, w))
    return None, np.inf


def plan_cone_cuts(X, F, cut: set, open_: set, h3d: float, snap_ratio: float = 0.9, max_branch_ratio: float = 6.0,
                   snap_negative: bool = True, snap_positive: bool = True, smooth_iters: int = 30):
    """Cones from the cross field; new cut branches for +1/4 cones away from the cut.

    Returns dict(forced={xyz_vertex: +1/-1 turn sum on the seam}, branches=[[v...]], report=...)."""
    X = np.asarray(X, float)
    F = np.asarray(F, int)
    # the field is computed on a lightly smoothed copy: mesh-scale bumps of the scan would
    # otherwise create many cancelling +/- singularities far below the panel size
    u, info = cross_field(_smoothed(X, F, smooth_iters), F)
    sing = info["singular_faces"]
    cutv = {v for e in cut for v in e}
    bndv = {v for e in open_ for v in e}
    adj = defaultdict(list)
    seen = set()
    for f in F:
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            e = tuple(sorted((a, b)))
            if e in seen:
                continue
            seen.add(e)
            l = float(np.linalg.norm(X[a] - X[b]))
            adj[a].append((b, l))
            adj[b].append((a, l))
    forced: dict[int, int] = {}
    branches = []
    report = []
    G = set(cutv)
    # cancel +1/4 / -1/4 pairs closer than h3d (discretisation noise, no net curvature)
    cen = {fi: X[F[fi]].mean(axis=0) for fi, _m in sing}
    alive = list(sing)
    cancelled = 0
    changed = True
    while changed:
        changed = False
        best = None
        for a in range(len(alive)):
            for b in range(a + 1, len(alive)):
                if alive[a][1] * alive[b][1] < 0:
                    dd = float(np.linalg.norm(cen[alive[a][0]] - cen[alive[b][0]]))
                    if dd < h3d and (best is None or dd < best[0]):
                        best = (dd, a, b)
        if best is not None:
            _, a, b = best
            alive = [t for k, t in enumerate(alive) if k not in (a, b)]
            cancelled += 1
            changed = True
    # strongest first: |index| then +1/4 before -1/4
    order = sorted(alive, key=lambda t: (-abs(t[1]), -t[1]))
    for fi, m in order:
        c = X[F[fi]].mean(axis=0)
        v = int(F[fi][np.argmin(np.linalg.norm(X[F[fi]] - c, axis=1))])
        # distance to the current cut graph (a seam vertex carries at most one cone)
        free_seam = (G - set(forced)) - bndv
        path, d = _dijkstra(adj, v, free_seam, set()) if free_seam else (None, np.inf)
        if v in free_seam:
            path, d = [v], 0.0
        _pb, d_bnd = _dijkstra(adj, v, bndv, set()) if bndv else (None, np.inf)
        if v in bndv:
            d_bnd = 0.0
        if d_bnd < d:
            report.append({"face": int(fi), "index": f"{m}/4", "vertex": v, "distance_to_seam": float(d),
                           "distance_to_open_boundary": float(d_bnd),
                           "action": "closer to the open boundary: absorbed by its corners"})
            continue
        entry = {"face": int(fi), "index": f"{m}/4", "vertex": v, "distance_to_seam": float(d)}
        if path is None:
            entry["action"] = "unreachable"
        elif d <= snap_ratio * h3d:
            tgt = path[-1]
            if tgt in bndv and tgt not in G:
                entry["action"] = "near open boundary: absorbed by the boundary corners"
            elif (m > 0 and not snap_positive) or (m < 0 and not snap_negative):
                entry["action"] = "near the seam: left to the logical-domain MILP"
            else:
                forced[tgt] = int(np.sign(m))
                entry.update({"action": "snapped onto the seam", "seam_vertex": int(tgt)})
        elif m > 0 and d <= max_branch_ratio * h3d:
            branches.append(path)                    # path[0] = cone (new tip), path[-1] on the cut
            G |= set(path)
            entry.update({"action": "new cut branch (cone = its tip)", "branch_length": float(d)})
        else:
            entry["action"] = ("negative cone away from the seam: absorbed by distortion" if m < 0
                               else "too far from the seam: absorbed by distortion")
        report.append(entry)
    return {"forced": forced, "branches": branches, "report": report, "field": u,
            "singularities_raw": len(sing), "pairs_cancelled": cancelled}


# ------------------------------------------------------------------ chart split
def split_chart(uv, f2, f3, branches_xyz):
    """Cut the UV chart along new 3D branches (interior to the chart); duplicates path vertices."""
    uv = np.asarray(uv, float).copy()
    f2 = np.asarray(f2, int).copy()
    f3 = np.asarray(f3, int)
    # 3D edge -> UV edge (interior edges of the chart are unique)
    e3_to_uv = {}
    for t in range(len(f2)):
        for k in range(3):
            a2, b2 = int(f2[t, k]), int(f2[t, (k + 1) % 3])
            a3, b3 = int(f3[t, k]), int(f3[t, (k + 1) % 3])
            e3_to_uv.setdefault(tuple(sorted((a3, b3))), set()).add(tuple(sorted((a2, b2))))
    cut_uv = set()
    for path in branches_xyz:
        for a, b in zip(path[:-1], path[1:]):
            es = e3_to_uv.get(tuple(sorted((a, b))), set())
            if len(es) != 1:
                raise ValueError("branch edge is already a seam or missing")
            cut_uv |= es
    # chart boundary edges count as separators too
    ec = defaultdict(int)
    for f in f2:
        for k in range(3):
            ec[tuple(sorted((int(f[k]), int(f[(k + 1) % 3]))))] += 1
    sep = cut_uv | {e for e, c in ec.items() if c == 1}
    vfaces = defaultdict(list)
    for t, f in enumerate(f2):
        for v in f:
            vfaces[int(v)].append(t)
    touched = {v for e in cut_uv for v in e}
    new_uv = [uv]
    nxt_id = len(uv)
    for v in touched:
        faces = vfaces[v]
        # union faces around v that share a non-separator edge through v
        parent = {t: t for t in faces}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        edge_owner = defaultdict(list)
        for t in faces:
            for w in f2[t]:
                w = int(w)
                if w != v:
                    edge_owner[tuple(sorted((v, w)))].append(t)
        for e, ts in edge_owner.items():
            if e in sep or len(ts) != 2:
                continue
            a, b = find(ts[0]), find(ts[1])
            if a != b:
                parent[a] = b
        comps = defaultdict(list)
        for t in faces:
            comps[find(t)].append(t)
        groups = list(comps.values())
        for g in groups[1:]:
            new_uv.append(uv[v:v + 1])
            for t in g:
                f2[t][f2[t] == v] = nxt_id
            nxt_id += 1
    split_chart.touched = sorted(set(touched) | set(range(len(uv), nxt_id)))
    return np.vstack(new_uv), f2


def open_slits(xyz, f3, uv, f2, iters: int = 400, touched=None):
    """Free-boundary bijective SLIM of a chart (opens zero-width slits).

    Every copy of a slit vertex is first pulled a little into its own fan of triangles, so the
    two sides of a new cut never coincide; SLIM then opens the slit as the 3D metric asks."""
    from .zhang2021_onestring_reparam import _SDProblem, signed_uv_areas
    from .structured_grid_20261028 import _seamless_slim
    uv = np.asarray(uv, float).copy()
    f2 = np.asarray(f2, int)
    if touched:
        vf = defaultdict(list)
        for t, f in enumerate(f2):
            for v in f:
                vf[int(v)].append(t)
        for v in touched:
            ts = vf.get(int(v), [])
            if not ts:
                continue
            cen = uv[f2[ts]].mean(axis=1).mean(axis=0)
            nb = np.unique(f2[ts])
            lmin = float(np.min(np.linalg.norm(uv[nb] - uv[v], axis=1)[nb != v])) if len(nb) > 1 else 0.0
            d = cen - uv[v]
            nd = float(np.linalg.norm(d))
            if nd > 0 and lmin > 0:
                uv[v] = uv[v] + d / nd * 0.15 * lmin
    s = np.sign(np.sum(signed_uv_areas(uv, f2)))
    mir = np.array([1.0, 1.0]) if s > 0 else np.array([-1.0, 1.0])
    prob = _SDProblem(np.asarray(xyz, float), np.asarray(f3, int), np.asarray(f2, int), len(uv))
    out, info = _seamless_slim(prob, uv * mir, [], gauge=0, iters=iters)
    return out * mir, info


def boundary_simple(uv, f2) -> bool:
    from shapely.geometry import Polygon
    from .zhang2021_onestring_boundary import oriented_boundary_loops
    loops = oriented_boundary_loops(np.asarray(f2, int))
    if len(loops) != 1:
        return False
    return bool(Polygon(np.asarray(uv)[loops[0]]).is_valid)


__all__ = ["cross_field", "chart_cut_edges", "plan_cone_cuts", "split_chart", "open_slits", "boundary_simple"]
