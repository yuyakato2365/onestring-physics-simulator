"""Shared mesh utilities: OptCuts OBJ reading, cut-mesh boundary, seam branches."""
from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter


def read_obj_uv(path):
    V, T, F = [], [], []
    for l in open(path):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("vt "):
            T.append([float(x) for x in l.split()[1:3]])
        elif l.startswith("f "):
            F.append([[int(t) - 1 for t in x.split("/")[:2]] for x in l.split()[1:4]])
    F = np.array(F)
    return np.array(V), np.array(T), F[:, :, 0], F[:, :, 1]


def read_dump(path):
    with open(path, "rb") as f:
        it = np.frombuffer(f.read(4), np.int32)[0]; lam = np.frombuffer(f.read(8), np.float64)[0]
        nV, nF = np.frombuffer(f.read(8), np.int32)
        vd = np.frombuffer(f.read(8 * 5 * nV), np.float64).reshape(nV, 5)
        fd = np.frombuffer(f.read(4 * 3 * nF), np.int32).reshape(nF, 3)
    return int(it), float(lam), vd[:, :2].copy(), vd[:, 2:].copy(), fd.astype(int)


def signed_areas(U, F):
    a = U[F[:, 1]] - U[F[:, 0]]; b = U[F[:, 2]] - U[F[:, 0]]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def boundary_loop(F):
    he = set()
    for f in F:
        for k in range(3):
            he.add((int(f[k]), int(f[(k + 1) % 3])))
    nxt = {a: b for (a, b) in he if (b, a) not in he}
    loops, seen = [], set()
    for s in nxt:
        if s in seen:
            continue
        L = [s]; seen.add(s); x = nxt[s]
        while x != s:
            L.append(x); seen.add(x); x = nxt[x]
        loops.append(L)
    loops.sort(key=len, reverse=True)
    return loops


class CutMesh:
    """UV (cut) mesh with, per UV vertex, its 3D vertex id; boundary loop ordered with the domain on the left."""

    def __init__(self, V3, U, FV, FT):
        self.V3 = np.asarray(V3, float); self.U = np.asarray(U, float)
        self.FV = np.asarray(FV, int); self.F = np.asarray(FT, int)
        n = len(U)
        self.v_of = np.full(n, -1, int)
        for fv, ft in zip(self.FV, self.F):
            self.v_of[ft] = fv
        self.X = self.V3[self.v_of]                       # 3D position of every UV vertex
        loops = boundary_loop(self.F)
        assert len(loops) == 1, f"expected one UV boundary loop, got {len(loops)}"
        self.loop = np.array(loops[0])
        if signed_area_poly(self.U[self.loop]) < 0:
            self.loop = self.loop[::-1]
        self.on_bnd = np.zeros(n, bool); self.on_bnd[self.loop] = True
        self.pos_in_loop = {int(v): i for i, v in enumerate(self.loop)}
        # boundary edges (consecutive loop vertices) -> 3D edge; seam edges appear twice
        L = self.loop; m = len(L)
        e3 = [tuple(sorted((int(self.v_of[L[i]]), int(self.v_of[L[(i + 1) % m]])))) for i in range(m)]
        cnt = Counter(e3)
        self.bedge3 = e3
        self.is_seam_edge = np.array([cnt[e] == 2 for e in e3])
        self.partner_edge = {}
        first = {}
        for i, e in enumerate(e3):
            if cnt[e] == 2:
                if e in first:
                    j = first[e]; self.partner_edge[i] = j; self.partner_edge[j] = i
                else:
                    first[e] = i
        self.copies = defaultdict(list)                   # 3D vertex -> UV copies
        for u, v in enumerate(self.v_of):
            self.copies[int(v)].append(u)

    def branches(self):
        """Seam branches: maximal runs of consecutive seam edges along the loop between seam nodes.
        A node is a loop position where the seam property changes, or where the 3D vertex has != 2 seam edges.
        Returns list of dict(edges=[loop edge indices], partner=[...]) pairing the two banks."""
        L = self.loop; m = len(L)
        seam_deg = Counter()
        for i in range(m):
            if self.is_seam_edge[i]:
                a, b = self.bedge3[i]
                seam_deg[a] += 1; seam_deg[b] += 1
        for k in seam_deg:
            seam_deg[k] //= 2                              # each 3D seam edge is listed twice
        rim_v = set()
        for i in range(m):
            if not self.is_seam_edge[i]:
                rim_v.update(self.bedge3[i])
        def is_node(pos):
            v3 = int(self.v_of[L[pos]])
            prev_e, next_e = (pos - 1) % m, pos
            if self.is_seam_edge[prev_e] != self.is_seam_edge[next_e]:
                return True
            if not self.is_seam_edge[next_e]:
                return False
            return seam_deg[v3] != 2 or v3 in rim_v
        nodes = [p for p in range(m) if is_node(p)]
        runs = []
        for a_i, p in enumerate(nodes):
            q = nodes[(a_i + 1) % len(nodes)]
            if not self.is_seam_edge[p]:
                continue
            edges = []; k = p
            while k != q:
                edges.append(k); k = (k + 1) % m
            runs.append(edges)
        # pair runs: run r is the partner of run s if their edges are partners (reversed)
        first_edge_run = {}
        for ri, r in enumerate(runs):
            for e in r:
                first_edge_run[e] = ri
        pairs, done = [], set()
        for ri, r in enumerate(runs):
            if ri in done:
                continue
            rj = first_edge_run[self.partner_edge[r[0]]]
            done |= {ri, rj}
            pairs.append((ri, rj))
        return nodes, runs, pairs


def signed_area_poly(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def outline_runs(U, L, H, min_run=0.5):
    """Classify outline edges as horizontal/vertical from a +-0.1 h chord, merge runs shorter than min_run * h.
    Returns (cls per edge, runs [(first edge, n edges)], per-vertex distance to its run line in units of h,
    with vertices within 0.2 h of a corner excluded (nan))."""
    m = len(L)
    seg = U[np.roll(L, -1)] - U[L]; slen = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(slen)]; Luv = cum[-1]
    def point_at(s):
        s = s % Luv
        k = min(int(np.searchsorted(cum, s, side="right") - 1), m - 1)
        return U[L[k]] + (s - cum[k]) / max(slen[k], 1e-30) * seg[k]
    cls = np.zeros(m, int)
    for k in range(m):
        sm = 0.5 * (cum[k] + cum[k + 1])
        dd = point_at(sm + 0.1 * H) - point_at(sm - 0.1 * H)
        cls[k] = 0 if abs(dd[0]) >= abs(dd[1]) else 1
    def runs_of(c):
        if not np.any(c != c[0]):
            return [(0, m)]
        start = int(np.argmax(c != np.roll(c, 1)))
        rs, k0 = [], start
        for i in range(1, m + 1):
            k = (start + i) % m
            if c[k] != c[(k - 1) % m] or i == m:
                rs.append((k0, (k - k0) % m or m)); k0 = k
        return rs
    def run_len(r):
        return sum(slen[(r[0] + i) % m] for i in range(r[1]))
    while True:
        rs = runs_of(cls)
        if len(rs) <= 4:
            break
        lens = [run_len(r) for r in rs]
        i = int(np.argmin(lens))
        if lens[i] >= min_run * H:
            break
        for j in range(rs[i][1]):
            cls[(rs[i][0] + j) % m] = 1 - cls[(rs[i][0] + j) % m]
    rs = runs_of(cls)
    dev = np.full(m, np.nan)
    for (k0, n) in rs:
        c = cls[k0]
        idx = [(k0 + i) % m for i in range(n + 1)]
        w = np.ones(len(idx))
        lvl = np.median(U[L[idx], 1 - c])
        for i, k in enumerate(idx):
            s_from = (cum[k] - cum[k0]) % Luv; s_to = (cum[(k0 + n) % m] - cum[k]) % Luv
            if min(s_from, s_to) < 0.2 * H:
                continue
            dev[k] = abs(U[L[k], 1 - c] - lvl) / H
    return cls, rs, dev


def split_boundary_chords(V3, U, FV, FT):
    """Split every interior edge whose two ends are both on the UV boundary (midpoint in 3D and in UV), until none is
    left.  Straightening the outline would otherwise squeeze the triangles between such an edge and the boundary
    (or a triangle with all three corners on the boundary) to zero area.  The surface itself is unchanged."""
    V3 = np.asarray(V3, float).copy(); U = np.asarray(U, float).copy()
    FV = [list(f) for f in FV]; FT = [list(f) for f in FT]
    n_split = 0
    while True:
        loops = boundary_loop(np.asarray(FT))
        onb = set(loops[0])
        own = defaultdict(list)
        for fi, f in enumerate(FT):
            for k in range(3):
                own[tuple(sorted((f[k], f[(k + 1) % 3])))].append(fi)
        cand = [e for e, fs in own.items() if len(fs) == 2 and e[0] in onb and e[1] in onb]
        if not cand:
            break
        done_faces = set()
        for (a, b) in cand:
            fs = own[(a, b)]
            if any(f in done_faces for f in fs):
                continue
            # new UV vertex and new 3D vertex at the midpoints
            fa = FT[fs[0]]; ka = fa.index(a); kb = fa.index(b)
            va, vb = FV[fs[0]][ka], FV[fs[0]][kb]
            V3 = np.vstack([V3, 0.5 * (V3[va] + V3[vb])]); nv = len(V3) - 1
            U = np.vstack([U, 0.5 * (U[a] + U[b])]); nu = len(U) - 1
            for fi in fs:
                ft, fv = FT[fi], FV[fi]
                i = ft.index(a); j = ft.index(b)
                # orient so that (p, q) is the split edge in face order, r the opposite corner
                if (i + 1) % 3 == j:
                    p_, q_ = i, j
                else:
                    p_, q_ = j, i
                r_ = 3 - p_ - q_
                t1 = [ft[p_], nu, ft[r_]]; t2 = [nu, ft[q_], ft[r_]]
                v1 = [fv[p_], nv, fv[r_]]; v2 = [nv, fv[q_], fv[r_]]
                FT[fi] = t1; FV[fi] = v1
                FT.append(t2); FV.append(v2)
                done_faces.add(fi); done_faces.add(len(FT) - 1)
            n_split += 1
    return V3, U, np.asarray(FV), np.asarray(FT), n_split
