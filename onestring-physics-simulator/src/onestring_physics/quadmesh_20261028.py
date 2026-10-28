"""Comparison: unstructured quad meshing of Omega_1 with interior singularities allowed.

Boundary: every run between OptCuts seam nodes is split at equal 3D arc length into k pieces
(k from the panel size 2h; both copies of a cut get the same k at the same 3D points, so the seams
close exactly).  Gmsh (Frontal-Delaunay for quads + Blossom recombination) meshes the polygon at
size 2h; every element is then split into quads (quad -> 4, triangle -> 3), boundary edge
midpoints taken at the 3D arc-length midpoint of their run piece (again identical on both
copies).  Interior points are relaxed with the same Brackbill-Saltzman functional as the
structured prototype.  Interior vertices may have any valence (3, 5, 6 ...).
"""
from __future__ import annotations

import numpy as np


def _boundary_samples(src, h2: float):
    from .zhang2021_onestring_boundary import _param_from_fraction_3d, run_interval
    from .zhang2021_onestring_seam_matching import seam_constraints
    n = src.n
    nodes = sorted(set(int(j) for j in src.nodes))
    if len(nodes) < 2:
        nodes = sorted({0, n // 2} | set(nodes))
    tot3 = float(src.cum3[-1])
    runs = [(nodes[r], nodes[(r + 1) % len(nodes)]) for r in range(len(nodes))]
    L3 = {}
    for a, b in runs:
        L3[(a, b)] = ((float(src.cum3[b]) - float(src.cum3[a])) % tot3) or tot3
    k = {r: max(1, int(round(L3[r] / h2))) for r in runs}
    for con in seam_constraints(src):
        ra, rb = tuple(con.run_a), tuple(con.run_b)
        if ra in k and rb in k:
            kk = max(1, int(round(0.5 * (L3[ra] + L3[rb]) / h2)))
            k[ra] = k[rb] = kk
    P = src.perimeter
    pts, tag, tip = [], [], []
    for a, b in runs:
        lo, hi = run_interval(src, (a, b))
        kk = k[(a, b)]
        fr = [j / kk for j in range(kk)]
        # extra samples half a piece away from a cut tip (on both copies: same fractions) so that
        # the tip gets >= 3 panels around it
        if src.node_kind.get(a) == "tip":
            fr += [0.5 / kk]
        if src.node_kind.get(b) == "tip":
            fr += [1.0 - 0.5 / kk]
        fr = sorted(set(round(f, 12) for f in fr))
        for i, f in enumerate(fr):
            s_ = _param_from_fraction_3d(src, lo, hi, f) % P
            pts.append(src.point_at(s_)[2])
            tag.append(((a, b), f, fr[i + 1] if i + 1 < len(fr) else 1.0))
            tip.append(i == 0 and src.node_kind.get(a) == "tip")

    def midpoint(i):
        (a, b), f0, f1 = tag[i]
        lo, hi = run_interval(src, (a, b))
        s_ = _param_from_fraction_3d(src, lo, hi, 0.5 * (f0 + f1)) % P
        return src.point_at(s_)[2]

    return np.asarray(pts, float), midpoint, tip


def _gmsh_coarse(src, hc):
    import gmsh
    B, midpoint, is_tip = _boundary_samples(src, hc)
    nb = len(B)
    nb = len(B)
    gmsh.initialize(interruptible=False)          # no signal handler: runs inside the Streamlit thread
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("omega1")
        # cut tips get a finer size so that >= 3 panels surround them (3D valence >= 3)
        ptag = [gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, (0.5 if t else 1.0) * hc)
                for p, t in zip(B, is_tip)]
        ltag = [gmsh.model.geo.addLine(ptag[i], ptag[(i + 1) % nb]) for i in range(nb)]
        loop = gmsh.model.geo.addCurveLoop(ltag)
        surf = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        for lt in ltag:
            gmsh.model.mesh.setTransfiniteCurve(lt, 2)
        gmsh.model.mesh.setRecombine(2, surf)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        X = np.asarray(coords, float).reshape(-1, 3)[:, :2]
        index = {int(t): i for i, t in enumerate(tags)}
        etypes, _etags, enodes = gmsh.model.mesh.getElements(2)
        elems = []
        for et, nd in zip(etypes, enodes):
            nn = {2: 3, 3: 4}.get(int(et))
            if nn is None:
                continue
            for e in np.asarray(nd, int).reshape(-1, nn):
                elems.append([index[int(t)] for t in e])
    finally:
        gmsh.finalize()
    return B, midpoint, is_tip, X, elems


def quadmesh_omega(src, uvm, f2, xyz, f3, h: float, lam_o: float = 0.3, lam_v: float = 0.3, coarse_scale: float = 1.65,
                   target_count: float | None = None):
    import gmsh
    from .crop_bs_grid_20261028 import brackbill_saltzman
    target = None
    if target_count:
        target = float(target_count)
    hc = 2.0 * h * float(coarse_scale)                 # coarse size; every element is split in 3-4 quads
    B, midpoint, is_tip, X, elems = _gmsh_coarse(src, hc)
    if target:
        est = sum(len(e) for e in elems)
        if abs(est / target - 1.0) > 0.08:            # one calibration pass towards the requested count
            hc *= float(np.sqrt(est / target))
            B, midpoint, is_tip, X, elems = _gmsh_coarse(src, hc)
    nb = len(B)
    # boundary nodes: gmsh node on each geometric point (same coordinates)
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    d, bidx = tree.query(B)
    if np.max(d) > 1e-9 * max(1.0, float(np.ptp(B))):
        raise RuntimeError("QUADMESH_BOUNDARY_NODES_NOT_FOUND")
    bpos = {int(v): i for i, v in enumerate(bidx)}
    # orientation: CCW
    def area(e):
        P = X[e]
        return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    elems = [e if area(e) > 0 else e[::-1] for e in elems]
    # a cut tip needs >= 3 panels around it (3D valence >= 3): split coarse elements at the tip
    X = np.asarray(X, float)
    Xl = [p for p in X]
    tips = [int(bidx[i]) for i, t in enumerate(is_tip) if t]
    for t in tips:
        for _ in range(4):
            inc = [ei for ei, e in enumerate(elems) if t in e]
            if len(inc) >= 3:
                break
            # widest element at the tip
            def ang(e):
                i = e.index(t); a, b = np.asarray(Xl[e[i - 1]]), np.asarray(Xl[e[(i + 1) % len(e)]])
                u, v = a - Xl[t], b - Xl[t]
                return np.arctan2(abs(u[0] * v[1] - u[1] * v[0]), u @ v)
            ei = max(inc, key=lambda j: ang(elems[j]))
            e = elems[ei]
            i = e.index(t)
            k = len(e)
            if k >= 4:                                   # quad: diagonal from the tip
                opp = e[(i + 2) % k]
                e1 = [e[(i + j) % k] for j in range(3)]
                e2 = [e[(i + j) % k] for j in range(2, k)] + [t]
                elems[ei] = e1
                elems.append(e2)
                del opp
            else:                                        # triangle: midpoint of the opposite edge
                a, b = e[(i + 1) % 3], e[(i + 2) % 3]
                Xl.append(0.5 * (np.asarray(Xl[a]) + np.asarray(Xl[b])))
                y = len(Xl) - 1
                elems[ei] = [t, a, y]
                elems.append([t, y, b])
                for ej, f in enumerate(elems):           # neighbour across (a, b) gets the new node
                    if ej in (ei, len(elems) - 1) or a not in f or b not in f:
                        continue
                    ia, ib = f.index(a), f.index(b)
                    pos = max(ia, ib) if abs(ia - ib) == 1 else 0
                    elems[ej] = f[:pos] + [y] + f[pos:] if pos else f + [y]
                    break
    X = np.asarray(Xl, float)
    V = [p for p in X]
    n_tri = sum(1 for e in elems if len(e) == 3)
    n_quad = len(elems) - n_tri
    mid = {}

    def m(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            ia, ib = bpos.get(a), bpos.get(b)
            if ia is not None and ib is not None and ((ia + 1) % nb == ib or (ib + 1) % nb == ia):
                first = ia if (ia + 1) % nb == ib else ib
                V.append(np.asarray(midpoint(first), float))
            else:
                V.append(0.5 * (X[a] + X[b]))
            mid[key] = len(V) - 1
        return mid[key]

    quads = []
    for e in elems:
        c = len(V)
        V.append(X[e].mean(axis=0))
        k = len(e)
        for i in range(k):
            quads.append([e[i], m(e[i], e[(i + 1) % k]), c, m(e[(i - 1) % k], e[i])])
    V = np.asarray(V, float)
    quads = np.asarray(quads, int)
    bset = set(int(v) for v in bidx)
    bset |= {mid[(min(a, b), max(a, b))] for (a, b) in mid if a in bpos and b in bpos and
             ((bpos[a] + 1) % nb == bpos[b] or (bpos[b] + 1) % nb == bpos[a])}
    free = np.setdiff1d(np.arange(len(V)), sorted(bset))
    nbr = [[] for _ in range(len(V))]
    for q in quads:
        for i in range(4):
            nbr[q[i]] += [q[(i + 1) % 4], q[(i - 1) % 4]]

    def folded(P):
        a = P[quads[:, 1]] - P[quads[:, 0]]
        b = P[quads[:, 3]] - P[quads[:, 0]]
        c = P[quads[:, 3]] - P[quads[:, 2]]
        d = P[quads[:, 1]] - P[quads[:, 2]]
        return np.any(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] <= 0) or np.any(c[:, 0] * d[:, 1] - c[:, 1] * d[:, 0] <= 0)

    for _ in range(200):                      # boundary midpoints moved onto the curved boundary
        if not folded(V):
            break
        V[free] = np.asarray([V[nbr[v]].mean(axis=0) for v in free])
    G, info = brackbill_saltzman(V, quads, sorted(bset), uvm, f2, xyz, f3, h, lam_o=lam_o, lam_v=lam_v)
    from collections import Counter
    edges = Counter(tuple(sorted((int(q[i]), int(q[(i + 1) % 4])))) for q in quads for i in range(4))
    bnd = {v for e, c in edges.items() if c == 1 for v in e}
    val = Counter(int(v) for q in quads for v in q)
    irr = Counter(c for v, c in val.items() if v not in bnd and c != 4)
    info.update({"placement": "unstructured quad mesh (gmsh Frontal-Delaunay quads + Blossom, split to all quads) "
                              "+ Brackbill-Saltzman relaxation; interior singularities allowed",
                 "coarse_quads": n_quad, "coarse_triangles": n_tri, "quads": int(len(quads)),
                 "boundary_samples": nb, "interior_irregular_vertices": {int(k): int(v) for k, v in irr.items()}})
    return G, quads, info


__all__ = ["quadmesh_omega"]
