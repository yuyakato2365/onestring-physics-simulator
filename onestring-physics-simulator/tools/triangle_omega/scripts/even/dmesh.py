"""Dynamic triangle mesh (dict based) with flip / split / collapse, plus reference-surface helpers."""
import numpy as np, igl
from collections import defaultdict

def ekey(a, b): return (a, b) if a < b else (b, a)

class Ref:
    """Reference surface: closest-point projection, normals, scalar interpolation, boundary polyline."""
    def __init__(self, V, F):
        self.V = np.asarray(V, float); self.F = np.asarray(F, int)
        self.FN = igl.per_face_normals(self.V, self.F, np.array([0., 0., 1.]))
        self.loop = np.asarray(igl.boundary_loop(self.F))
        self.bnd = np.zeros(len(V), bool); self.bnd[self.loop] = True
        from scipy.spatial import cKDTree
        self.ctree = cKDTree(self.V[self.F].mean(1))
    def closest(self, P):
        P = np.atleast_2d(P)
        d2, I, C = igl.point_mesh_squared_distance(P, self.V, self.F)
        return C, I.astype(int), d2
    def interp(self, P, vals):
        C, I, _ = self.closest(P)
        T = self.F[I]
        B = igl.barycentric_coordinates(C, self.V[T[:, 0]], self.V[T[:, 1]], self.V[T[:, 2]])
        return (B * vals[T]).sum(1)
    def normal(self, P):
        C, I, _ = self.closest(P); return self.FN[I]
    def _setup_fast(self):
        from scipy.spatial import cKDTree
        self.vtree = cKDTree(self.V); self.vfaces = [[] for _ in range(len(self.V))]
        for i, f in enumerate(self.F):
            for x in f: self.vfaces[x].append(i)
    def closest_fast(self, p):
        """closest point on the reference: triangles around the 8 nearest reference vertices"""
        if not hasattr(self, "vfaces"): self._setup_fast()
        _, nn = self.vtree.query(p, k=8)
        fs = np.unique(np.concatenate([self.vfaces[i] for i in nn]))
        A, B, C = self.V[self.F[fs, 0]], self.V[self.F[fs, 1]], self.V[self.F[fs, 2]]
        Q = _closest_on_tris(np.asarray(p, float), A, B, C)
        d = ((Q - p) ** 2).sum(1); k = int(np.argmin(d))
        return Q[k], int(fs[k])
    def interp_fast(self, p, vals):
        q, f = self.closest_fast(p); T = self.F[f]
        b = igl.barycentric_coordinates(q[None], self.V[T[0]][None], self.V[T[1]][None], self.V[T[2]][None])[0]
        return float(b @ vals[T])
    def fast_normal(self, p):
        return self.FN[self.ctree.query(p)[1]]
    def project_boundary(self, P):
        P = np.atleast_2d(P); A = self.V[self.loop]; B = self.V[np.roll(self.loop, -1)]
        out = []
        for p in P:
            ab = B - A; t = np.clip(((p - A) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
            Q = A + t[:, None] * ab; k = np.argmin(((Q - p) ** 2).sum(1)); out.append(Q[k])
        return np.array(out)

def _closest_on_tris(p, A, B, C):
    """closest points from p to many triangles (Ericson, Real-Time Collision Detection 5.1.5), vectorised"""
    ab = B - A; ac = C - A; ap = p - A
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = p - B; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = p - C; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / den; w = vc / den
    out = A + ab * v[:, None] + ac * w[:, None]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3); out[m] = (A + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6); out[m] = (A + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0); t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6))
    out[m] = (B + (C - B) * t[:, None])[m]
    out[(d1 <= 0) & (d2 <= 0)] = A[(d1 <= 0) & (d2 <= 0)]
    out[(d3 >= 0) & (d4 <= d3)] = B[(d3 >= 0) & (d4 <= d3)]
    out[(d6 >= 0) & (d5 <= d6)] = C[(d6 >= 0) & (d5 <= d6)]
    return out

class DMesh:
    def __init__(self, V, F):
        self.P = {i: np.array(v, float) for i, v in enumerate(V)}
        self.F = {i: list(map(int, f)) for i, f in enumerate(F)}
        self.nv = len(V); self.nf = len(F)
        self.vf = defaultdict(set); self.ef = defaultdict(set)
        for fid, f in self.F.items(): self._add_adj(fid, f)
        self.journal = None
    # ---- transactions ----
    def begin(self): self.journal = []
    def commit(self): self.journal = None
    def rollback(self):
        J = self.journal; self.journal = None
        for e in reversed(J):
            if e[0] == "addf": f = self.F.pop(e[1]); self._rm_adj(e[1], f)
            elif e[0] == "rmf": self.F[e[1]] = e[2]; self._add_adj(e[1], e[2])
            elif e[0] == "addv": self.P.pop(e[1], None); self.vf.pop(e[1], None)
            elif e[0] == "delv": self.P[e[1]] = e[2]
            elif e[0] == "setp": self.P[e[1]] = e[2]
    # ---- adjacency ----
    def _add_adj(self, fid, f):
        for i in range(3):
            self.vf[f[i]].add(fid); self.ef[ekey(f[i], f[(i + 1) % 3])].add(fid)
    def _rm_adj(self, fid, f):
        for i in range(3):
            self.vf[f[i]].discard(fid)
            k = ekey(f[i], f[(i + 1) % 3]); self.ef[k].discard(fid)
            if not self.ef[k]: del self.ef[k]
    def add_face(self, f):
        fid = self.nf; self.nf += 1; self.F[fid] = list(f); self._add_adj(fid, f)
        if self.journal is not None: self.journal.append(("addf", fid))
        return fid
    def rm_face(self, fid):
        f = self.F.pop(fid); self._rm_adj(fid, f)
        if self.journal is not None: self.journal.append(("rmf", fid, f))
    def add_vertex(self, p):
        v = self.nv; self.nv += 1; self.P[v] = np.array(p, float)
        if self.journal is not None: self.journal.append(("addv", v))
        return v
    def nbrs(self, v):
        s = set()
        for fid in self.vf[v]: s.update(self.F[fid])
        s.discard(v); return s
    def deg(self, v): return len(self.nbrs(v))
    def is_bedge(self, a, b): return len(self.ef.get(ekey(a, b), ())) == 1
    def is_bvert(self, v):
        return any(self.is_bedge(v, u) for u in self.nbrs(v))
    def edges(self): return list(self.ef.keys())
    def opp(self, a, b):
        return [next(x for x in self.F[fid] if x != a and x != b) for fid in self.ef[ekey(a, b)]]
    @staticmethod
    def has(f, a, b): return any(f[i] == a and f[(i + 1) % 3] == b for i in range(3))
    # ---- operations (return list of new face ids, or None if topologically invalid) ----
    def flip(self, a, b):
        fs = list(self.ef.get(ekey(a, b), ()))
        if len(fs) != 2: return None
        f0, f1 = self.F[fs[0]], self.F[fs[1]]
        if not self.has(f0, a, b): a, b = b, a
        c = next(x for x in f0 if x not in (a, b)); d = next(x for x in f1 if x not in (a, b))
        if c == d or ekey(c, d) in self.ef: return None
        if self.deg(a) <= 3 or self.deg(b) <= 3: return None
        self.rm_face(fs[0]); self.rm_face(fs[1])
        return [self.add_face([a, d, c]), self.add_face([d, b, c])]
    def split(self, a, b, p):
        fs = list(self.ef.get(ekey(a, b), ()))
        if not fs: return None
        m = self.add_vertex(p); new = []
        for fid in fs:
            f = self.F[fid]; x, y = (a, b) if self.has(f, a, b) else (b, a)
            c = next(t for t in f if t not in (a, b)); self.rm_face(fid)
            new += [self.add_face([x, m, c]), self.add_face([m, y, c])]
        return new, m
    def collapse_ok(self, a, b):
        k = ekey(a, b)
        if k not in self.ef: return False
        common = self.nbrs(a) & self.nbrs(b); opp = set(self.opp(a, b))
        if common != opp: return False
        be = self.is_bedge(a, b); ba, bb = self.is_bvert(a), self.is_bvert(b)
        if not be and ba and bb: return False
        for c in opp:
            if self.deg(c) <= 3: return False
        if len(self.nbrs(a) | self.nbrs(b)) - 2 < 3: return False
        return True
    def collapse(self, a, b, p):
        """merge b into a at position p; returns affected face ids"""
        if not self.collapse_ok(a, b): return None
        for fid in list(self.ef[ekey(a, b)]): self.rm_face(fid)
        aff = []
        for fid in list(self.vf[b]):
            f = [a if x == b else x for x in self.F[fid]]; self.rm_face(fid); aff.append(self.add_face(f))
        if self.journal is not None: self.journal += [("delv", b, self.P[b]), ("setp", a, self.P[a])]
        del self.P[b]; self.vf.pop(b, None)
        self.P[a] = np.array(p, float)
        return list(self.vf[a])
    # ---- export ----
    def arrays(self):
        ids = sorted(self.P); idx = {v: i for i, v in enumerate(ids)}
        V = np.array([self.P[v] for v in ids]); F = np.array([[idx[x] for x in f] for f in self.F.values()])
        return V, F, ids, idx
    def boundary_verts(self):
        s = set()
        for k, fs in self.ef.items():
            if len(fs) == 1: s.update(k)
        return s

def tri_angles(P):
    a, b, c = P
    def ang(x, y, z):
        u = y - x; v = z - x; return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30), -1, 1)))
    return ang(a, b, c), ang(b, c, a), ang(c, a, b)

def geom_ok(M, ref, fids, min_angle, max_ndev, normal_at=None):
    for fid in fids:
        if fid not in M.F: continue
        P = np.array([M.P[x] for x in M.F[fid]])
        n = np.cross(P[1] - P[0], P[2] - P[0]); ln = np.linalg.norm(n)
        if ln < 1e-14: return False
        if min(tri_angles(P)) < min_angle: return False
        nr = normal_at(P.mean(0)) if normal_at is not None else ref.fast_normal(P.mean(0))
        if n @ nr / ln < np.cos(np.radians(max_ndev)): return False
    return True


class SurfCheck:
    """Validity of faces against the reference surface: every corner's own reference normal must agree with the face normal,
    the face centroid must stay close to the surface (thin parts such as the ears cannot be bridged), minimum angle."""
    def __init__(self, M, ref):
        self.M = M; self.ref = ref; self.vn = {}
    def vnormal(self, v):
        if v not in self.vn:
            _, f = self.ref.closest_fast(self.M.P[v]); self.vn[v] = self.ref.FN[f]
        return self.vn[v]
    def forget(self, vs):
        for v in vs: self.vn.pop(v, None)
    def ok(self, fids, min_angle, max_ndev, ell=None, max_gap=0.3):
        cmax = np.cos(np.radians(max_ndev)); M = self.M
        for fid in fids:
            if fid not in M.F: continue
            f = M.F[fid]; P = np.array([M.P[x] for x in f])
            n = np.cross(P[1] - P[0], P[2] - P[0]); ln = np.linalg.norm(n)
            if ln < 1e-14: return False
            n = n / ln
            if min(tri_angles(P)) < min_angle: return False
            if any(n @ self.vnormal(x) < cmax for x in f): return False
            if ell is not None:
                if not hasattr(self.ref, "vtree"):
                    from scipy.spatial import cKDTree; self.ref.vtree = cKDTree(self.ref.V)
                if self.ref.vtree.query(P.mean(0))[0] > max_gap * np.mean([ell[x] for x in f]): return False
        return True
