"""OneString extension (NOT part of Zhang et al. 2021): seam-conforming chart.

Given the source OptCuts chart (paired 3D/UV triangles) and the Zhang 2021
quad mesh Q, re-parameterise the chart so that its boundary coincides with
Q's boundary polygon:

* every boundary vertex of Q becomes an exact source boundary vertex
  (boundary edges are split where needed, together with their twin copy on
  the other side of a cut so the 3D triangulation stays conforming; the 3D
  surface geometry is unchanged because a split point is the piecewise-linear
  image of the UV split point),
* interior edges joining two boundary vertices are split at their midpoint,
  so no triangle has three boundary vertices (otherwise a triangle could be
  flattened onto one straight Q boundary edge),
* source boundary vertices between two consecutive Q corners are placed on
  the straight Q edge (proportional UV arc length),
* interior UV vertices are solved with fixed boundary by a bijective,
  progressive symmetric-Dirichlet minimisation (SLIM local/global steps,
  Rabinovich et al. 2017, with a flip-preventing line search).

Q is not modified.  Afterwards every Q vertex lies in the closed chart, all
Q boundary edges are covered by collinear source boundary edges, and the
UV->3D lift of each Q boundary vertex is an exact point of the cut / open
boundary of S.  The distortion added to the chart is measured and reported;
any foldover is an error (never silently accepted).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .zhang2021_onestring_boundary import (
    BoundaryCorrespondenceError,
    Correspondence,
    SourceBoundary,
    build_source_boundary,
    finalize_correspondence,
    initial_correspondence,
    oriented_boundary_loops,
    uv_to_xyz_vertex_map,
)

__all__ = [
    "SeamConformingReparamError",
    "SplitResult",
    "ReparamResult",
    "split_for_correspondence",
    "chord_boundary_targets",
    "seam_conforming_reparameterization",
    "signed_uv_areas",
]


class SeamConformingReparamError(RuntimeError):
    pass


def signed_uv_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    P = np.asarray(uv, float)[np.asarray(faces, int)]
    return 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                  - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))


# ---------------------------------------------------------------- splitting
@dataclass
class SplitResult:
    xyz: np.ndarray
    f3: np.ndarray
    uv: np.ndarray
    f2: np.ndarray
    anchors: list[int]              # UV vertex id per Q boundary vertex (q_loop order)
    split_boundary_edges: int
    split_interior_chords: int
    centroid_fans: int
    new_uv_vertices: int
    new_xyz_vertices: int
    parent_face: np.ndarray         # original triangle id of each output triangle


def _apply_edge_splits(xyz, f3, uv, f2, uv_splits: dict[tuple[int, int], list[float]]):
    """Split UV edges at the given fractions (measured from the smaller id).

    The paired 3D edge is split at the same fraction; 3D split vertices are
    shared between the two UV copies of a cut edge (tolerance 1e-9 on the
    fraction), so the 3D triangulation remains conforming.  Triangles with
    one split edge are fanned from the opposite corner, triangles with two or
    three split edges from an added centroid (UV centroid <-> 3D centroid,
    i.e. barycentric (1/3,1/3,1/3) of the same piecewise-linear map).
    """
    xyz = np.asarray(xyz, float)
    uv = np.asarray(uv, float)
    f3 = np.asarray(f3, int)
    f2 = np.asarray(f2, int)
    n_uv0, n_xyz0 = len(uv), len(xyz)
    new_uv: list[np.ndarray] = []
    new_xyz: list[np.ndarray] = []
    # UV edge -> paired 3D endpoints in the same orientation (lo uv -> hi uv)
    uv_to_3d_edge: dict[tuple[int, int], tuple[int, int]] = {}
    for t2, t3 in zip(f2, f3):
        for i in range(3):
            a, b = int(t2[i]), int(t2[(i + 1) % 3])
            A, B = int(t3[i]), int(t3[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            pair = (A, B) if a < b else (B, A)
            old = uv_to_3d_edge.setdefault(key, pair)
            if old != pair:
                raise SeamConformingReparamError("UV edge paired with two different 3D edges")
    split3: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)

    def xyz_vertex(A: int, B: int, f: float) -> int:
        lo, hi = (A, B) if A < B else (B, A)
        g = f if A == lo else 1.0 - f
        for h, vid in split3[(lo, hi)]:
            if abs(h - g) <= 1e-9:
                return vid
        vid = n_xyz0 + len(new_xyz)
        new_xyz.append(xyz[lo] + g * (xyz[hi] - xyz[lo]))
        split3[(lo, hi)].append((g, vid))
        return vid

    chains: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    for key, fracs in uv_splits.items():
        a, b = key
        if a > b:
            raise ValueError("uv_splits keys must be (min, max)")
        if key not in uv_to_3d_edge:
            raise SeamConformingReparamError(f"split requested on a non-existing UV edge {key}")
        A, B = uv_to_3d_edge[key]
        fr: list[float] = []
        for f in sorted(float(x) for x in fracs):
            if not (0.0 < f < 1.0):
                raise SeamConformingReparamError("split fraction outside (0,1)")
            if fr and f - fr[-1] <= 1e-9:
                continue
            fr.append(f)
        items = []
        for f in fr:
            uid = n_uv0 + len(new_uv)
            new_uv.append(uv[a] + f * (uv[b] - uv[a]))
            items.append((f, uid, xyz_vertex(A, B, f)))
        chains[key] = items

    def chain(a: int, b: int, A: int, B: int):
        key = (a, b) if a < b else (b, a)
        items = chains.get(key)
        if not items:
            return [], []
        seq = items if a < b else items[::-1]
        return [it[1] for it in seq], [it[2] for it in seq]

    f2n: list[list[int]] = []
    f3n: list[list[int]] = []
    parent: list[int] = []
    fans = 0
    for ti, (t2, t3) in enumerate(zip(f2, f3)):
        t2 = [int(x) for x in t2]
        t3 = [int(x) for x in t3]
        ch = [chain(t2[c], t2[(c + 1) % 3], t3[c], t3[(c + 1) % 3]) for c in range(3)]
        nsplit = sum(1 for c in range(3) if ch[c][0])
        if nsplit == 0:
            f2n.append(t2)
            f3n.append(t3)
            parent.append(ti)
            continue
        if nsplit == 1:
            c = next(c for c in range(3) if ch[c][0])
            p2 = [t2[c]] + ch[c][0] + [t2[(c + 1) % 3]]
            p3 = [t3[c]] + ch[c][1] + [t3[(c + 1) % 3]]
            o2, o3 = t2[(c + 2) % 3], t3[(c + 2) % 3]
            for k in range(len(p2) - 1):
                f2n.append([p2[k], p2[k + 1], o2])
                f3n.append([p3[k], p3[k + 1], o3])
                parent.append(ti)
            continue
        cu = n_uv0 + len(new_uv)
        new_uv.append(uv[t2].mean(axis=0))
        cx = n_xyz0 + len(new_xyz)
        new_xyz.append(xyz[t3].mean(axis=0))
        poly2, poly3 = [], []
        for c in range(3):
            poly2.append(t2[c])
            poly3.append(t3[c])
            poly2 += ch[c][0]
            poly3 += ch[c][1]
        m = len(poly2)
        for k in range(m):
            f2n.append([cu, poly2[k], poly2[(k + 1) % m]])
            f3n.append([cx, poly3[k], poly3[(k + 1) % m]])
            parent.append(ti)
        fans += 1
    uv_out = np.concatenate([uv, np.asarray(new_uv).reshape(-1, 2)]) if new_uv else uv.copy()
    xyz_out = np.concatenate([xyz, np.asarray(new_xyz).reshape(-1, 3)]) if new_xyz else xyz.copy()
    return (xyz_out, np.asarray(f3n, int), uv_out, np.asarray(f2n, int),
            np.asarray(parent, int), chains, fans)


def split_for_correspondence(xyz, f3, uv, f2, src: SourceBoundary, corr: Correspondence,
                             *, rel_tol: float = 1e-7, split_boundary_chords: bool = True) -> SplitResult:
    """Realise the chosen boundary points as source vertices (see module doc)."""
    n = src.n
    uv_splits: dict[tuple[int, int], list[float]] = defaultdict(list)
    anchor_spec: list[tuple[int | None, tuple[int, int] | None, float]] = []
    n_boundary_requests = 0

    def request(j: int, t: float):
        a, b = int(src.loop[j]), int(src.loop[(j + 1) % n])
        key = (a, b) if a < b else (b, a)
        uv_splits[key].append(t if a < b else 1.0 - t)
        return key, (t if a < b else 1.0 - t)

    for s in corr.s:
        j, t, _ = src.point_at(s)
        if t <= rel_tol:
            anchor_spec.append((int(src.loop[j]), None, 0.0))
        elif t >= 1.0 - rel_tol:
            anchor_spec.append((int(src.loop[(j + 1) % n]), None, 0.0))
        else:
            key, f = request(j, t)
            anchor_spec.append((None, key, f))
            n_boundary_requests += 1
            if src.edge_kind[j] == "cut":
                request(int(src.twin[j]), 1.0 - t)
    split_boundary = len(uv_splits)
    chords = 0
    if split_boundary_chords:
        on_boundary = np.zeros(int(np.max(f2)) + 1, bool)
        on_boundary[np.asarray(src.loop, int)] = True
        count: dict[tuple[int, int], int] = defaultdict(int)
        for t in np.asarray(f2, int):
            for i in range(3):
                a, b = int(t[i]), int(t[(i + 1) % 3])
                count[(a, b) if a < b else (b, a)] += 1
        for key, c in count.items():
            if c == 2 and on_boundary[key[0]] and on_boundary[key[1]]:
                uv_splits[key].append(0.5)
                chords += 1
    xyz2, f32, uv2, f22, parent, chains, fans = _apply_edge_splits(xyz, f3, uv, f2, uv_splits)
    anchors: list[int] = []
    for vid, key, f in anchor_spec:
        if vid is not None:
            anchors.append(int(vid))
            continue
        best = min(chains[key], key=lambda it: abs(it[0] - f))
        if abs(best[0] - f) > 1e-9:
            raise SeamConformingReparamError("anchor split vertex not found")
        anchors.append(int(best[1]))
    if len(set(anchors)) != len(anchors):
        raise SeamConformingReparamError(
            "ZHANG2021_ONESTRING_TWO_QUAD_CORNERS_ON_ONE_SOURCE_VERTEX: two boundary vertices of Q "
            "were assigned the same point of dOmega")
    return SplitResult(xyz=xyz2, f3=f32, uv=uv2, f2=f22, anchors=anchors,
                       split_boundary_edges=int(split_boundary), split_interior_chords=int(chords),
                       centroid_fans=int(fans), new_uv_vertices=int(len(uv2) - len(uv)),
                       new_xyz_vertices=int(len(xyz2) - len(xyz)), parent_face=parent)


# -------------------------------------------------------- boundary targets
def chord_boundary_targets(uv: np.ndarray, loop: list[int], anchors: list[int],
                           quad_uv: np.ndarray, q_loop: list[int]) -> dict[int, np.ndarray]:
    """Place every source boundary vertex on Q's boundary polygon."""
    pos = {v: i for i, v in enumerate(loop)}
    try:
        idx = [pos[a] for a in anchors]
    except KeyError as exc:
        raise SeamConformingReparamError("anchor is not a boundary vertex of the split chart") from exc
    n = len(loop)
    P = np.asarray(uv, float)[loop]
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    targets: dict[int, np.ndarray] = {}
    k_count = len(anchors)
    covered = 0
    for k in range(k_count):
        i0 = idx[k]
        i1 = idx[(k + 1) % k_count]
        q0 = np.asarray(quad_uv[q_loop[k]], float)
        q1 = np.asarray(quad_uv[q_loop[(k + 1) % k_count]], float)
        span = (i1 - i0) % n
        if span == 0:
            raise SeamConformingReparamError("degenerate anchor span")
        covered += span
        lengths = [seg[(i0 + m) % n] for m in range(span)]
        total = float(np.sum(lengths))
        acc = 0.0
        targets[loop[i0]] = q0
        for m in range(1, span):
            acc += lengths[m - 1]
            f = acc / total if total > 0 else m / span
            targets[loop[(i0 + m) % n]] = q0 + f * (q1 - q0)
    if covered != n or len(targets) != n:
        raise SeamConformingReparamError(
            "ZHANG2021_ONESTRING_ANCHORS_NOT_CYCLIC: Q corners do not traverse dOmega exactly once")
    return targets


# ------------------------------------------------------------ SD machinery
class _SDProblem:
    """Area-weighted symmetric Dirichlet of the UV map of every 3D triangle."""

    def __init__(self, xyz, f3, f2, n_vertices: int):
        X = np.asarray(xyz, float)[np.asarray(f3, int)]
        e1 = X[:, 1] - X[:, 0]
        e2 = X[:, 2] - X[:, 0]
        l1 = np.linalg.norm(e1, axis=1)
        area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        if np.any(l1 <= 0) or np.any(area2 <= 1e-300):
            raise SeamConformingReparamError("degenerate 3D source triangle")
        x2 = np.einsum("ij,ij->i", e2, e1) / l1
        y2 = area2 / l1
        self.area = 0.5 * area2
        self.weight = self.area / self.area.sum()
        self.faces = np.asarray(f2, int)
        m = len(self.faces)
        # local frame p0=(0,0), p1=(l1,0), p2=(x2,y2); f linear: grad f = G @ (f0,f1,f2)
        G = np.zeros((m, 2, 3))
        G[:, 0, 0] = -1.0 / l1
        G[:, 0, 1] = 1.0 / l1
        G[:, 1, 0] = (x2 / l1 - 1.0) / y2
        G[:, 1, 1] = -x2 / l1 / y2
        G[:, 1, 2] = 1.0 / y2
        rows = np.repeat(np.arange(m), 3)
        cols = self.faces.reshape(-1)
        shape = (m, int(n_vertices))
        self.Dx = sparse.csr_matrix((G[:, 0, :].reshape(-1), (rows, cols)), shape=shape)
        self.Dy = sparse.csr_matrix((G[:, 1, :].reshape(-1), (rows, cols)), shape=shape)

    def jacobians(self, uv):
        u = uv[:, 0]
        v = uv[:, 1]
        J = np.empty((len(self.faces), 2, 2))
        J[:, 0, 0] = self.Dx @ u
        J[:, 0, 1] = self.Dy @ u
        J[:, 1, 0] = self.Dx @ v
        J[:, 1, 1] = self.Dy @ v
        return J

    @staticmethod
    def _det(J):
        return J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

    def energy(self, uv):
        J = self.jacobians(uv)
        det = self._det(J)
        if np.any(det <= 0):
            return np.inf, J, det
        fro = np.sum(J * J, axis=(1, 2))
        sd = fro + fro / (det * det)  # ||J||_F^2 + ||J^-1||_F^2
        return float(np.dot(self.weight, sd)), J, det

    def per_triangle_sd(self, uv):
        J = self.jacobians(uv)
        det = self._det(J)
        fro = np.sum(J * J, axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(det > 0, fro + fro / (det * det), np.inf)


def _max_feasible_step(J, dJ):
    """Largest alpha > 0 keeping det(J + alpha dJ) > 0 on every triangle."""
    a = dJ[:, 0, 0] * dJ[:, 1, 1] - dJ[:, 0, 1] * dJ[:, 1, 0]
    b = J[:, 0, 0] * dJ[:, 1, 1] + dJ[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * dJ[:, 1, 0] - dJ[:, 0, 1] * J[:, 1, 0]
    c = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    alpha = np.full(len(c), np.inf)
    scale = np.maximum(np.abs(a) + np.abs(b) + np.abs(c), 1e-300)
    lin = np.abs(a) <= 1e-14 * scale
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = -c[lin] / b[lin]
        alpha[lin] = np.where(r > 0, r, np.inf)
        quad = ~lin
        disc = b[quad] ** 2 - 4 * a[quad] * c[quad]
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        qq = -0.5 * (b[quad] + np.where(b[quad] >= 0, 1.0, -1.0) * sq)
        r1 = qq / a[quad]
        r2 = np.where(qq != 0, c[quad] / qq, np.inf)
        r1 = np.where(ok & (r1 > 0), r1, np.inf)
        r2 = np.where(ok & (r2 > 0), r2, np.inf)
        alpha[quad] = np.minimum(r1, r2)
    return float(np.min(alpha)) if len(alpha) else np.inf


def _slim_iterations(prob: _SDProblem, uv, free, iters=20, tol=1e-5):
    """SLIM (Rabinovich et al. 2017) for symmetric Dirichlet, fixed boundary.

    Requires a counter-clockwise (det J > 0) chart; the caller mirrors a
    clockwise chart first.  Never accepts a step that inverts a triangle.
    """
    uv = uv.copy()
    nV = len(uv)
    fixed = np.setdiff1d(np.arange(nV), free)
    E, J, _ = prob.energy(uv)
    if not np.isfinite(E):
        raise SeamConformingReparamError("SLIM started from a non-injective chart")
    sw = np.sqrt(prob.area)
    Dx, Dy = prob.Dx, prob.Dy
    idx_free = np.r_[free, free + nV]
    idx_fixed = np.r_[fixed, fixed + nV]
    done = 0
    for _ in range(int(iters)):
        U, S, Vt = np.linalg.svd(J)
        R = U @ Vt                                    # det(J) > 0  =>  rotation
        s = np.clip(S, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            w2 = np.where(np.abs(s - 1) > 1e-8, (s - s ** -3) / (s - 1), 4.0)
        w = np.sqrt(np.maximum(w2, 1e-12))
        W = np.einsum("tij,tj,tkj->tik", U, w, U)
        WR = W @ R
        blocks, rhs = [], []
        for r in (0, 1):
            for c_, D in ((0, Dx), (1, Dy)):
                Au = sparse.diags(sw * W[:, r, 0]) @ D
                Av = sparse.diags(sw * W[:, r, 1]) @ D
                blocks.append(sparse.hstack([Au, Av]))
                rhs.append(sw * WR[:, r, c_])
        A = sparse.vstack(blocks).tocsc()
        b = np.concatenate(rhs)
        x = np.r_[uv[:, 0], uv[:, 1]]
        Af = A[:, idx_free]
        rhs_f = b - A[:, idx_fixed] @ x[idx_fixed]
        M = (Af.T @ Af).tocsc()
        M = M + sparse.identity(M.shape[0], format="csc") * (1e-12 * float(M.diagonal().mean()))
        sol = splu(M).solve(Af.T @ rhs_f)
        d = np.zeros_like(x)
        d[idx_free] = sol - x[idx_free]
        duv = np.c_[d[:nV], d[nV:]]
        dJ = prob.jacobians(duv)
        amax = _max_feasible_step(J, dJ)
        alpha = min(1.0, 0.8 * amax)
        accepted = False
        while alpha > 1e-12:
            trial = uv + alpha * duv
            Et, Jt, _ = prob.energy(trial)
            if np.isfinite(Et) and Et <= E:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        rel = (E - Et) / max(E, 1e-30)
        uv, E, J = trial, Et, Jt
        done += 1
        if rel < tol:
            break
    return uv, E, done


def _uniform_laplacian(n, faces):
    f = np.asarray(faces, int)
    rows = np.r_[f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]]
    cols = np.r_[f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]]
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sparse.diags(deg) - A).tocsr()


@dataclass
class ReparamResult:
    xyz: np.ndarray              # source 3D vertices (+ split points on the same surface)
    f3: np.ndarray
    uv: np.ndarray               # re-parameterised chart, same frame/orientation as the input
    f2: np.ndarray
    parent_face: np.ndarray      # input triangle of every output triangle
    anchors: list[int]           # chart vertex of every Q boundary vertex (q_loop order)
    q_loop: list[int]
    quad_uv: np.ndarray          # final quad mesh (== input Q unless seam matching inserted chords)
    quads: np.ndarray
    metrics: dict = field(default_factory=dict)


def seam_node_valences(src: SourceBoundary, corr: Correspondence, quads: np.ndarray) -> list[dict]:
    """Quads around every OptCuts seam node in 3D (summed over its UV copies).

    An interior point of S (cut tip, junction) is surrounded by ~4 panels in a
    regular layout; a point on the open boundary of S by ~2.  A tip carried by
    2 quads gives ~180 degree panel corners in 3D, by 1 quad a folded panel.
    """
    P = src.perimeter
    val = np.bincount(np.asarray(quads, int).ravel())
    groups: dict[int, list] = defaultdict(list)
    s = np.asarray(corr.s, float)
    for j in src.nodes:
        ns = float(src.cum[j]) % P
        k = int(np.argmin(np.abs((s - ns + P / 2) % P - P / 2)))
        exact = abs((s[k] - ns + P / 2) % P - P / 2) <= 1e-9 * max(P, 1.0)
        v = int(corr.q_loop[k])
        groups[int(src.xyz_id[j])].append((src.node_kind.get(j, "?"), int(val[v]) if exact else 0))
    out = []
    for xid, items in sorted(groups.items()):
        kinds = sorted({k for k, _ in items})
        ideal = 2 if "transition" in kinds else 4
        out.append({"xyz_id": xid, "kinds": kinds, "copies": len(items),
                    "valences": [v for _, v in items], "valence_sum": int(sum(v for _, v in items)),
                    "ideal": ideal})
    return out


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    c = np.cumsum(weights[order])
    return float(values[order[min(len(order) - 1, int(np.searchsorted(c, q * c[-1])))]])


def progressive_boundary_slim(prob: "_SDProblem", uv: np.ndarray, faces: np.ndarray, bverts: np.ndarray,
                              b_target: np.ndarray, *, max_steps: int = 2000, step_slim_iters: int = 4,
                              final_iters: int = 80, error_prefix: str = "ZHANG2021_ONESTRING_REPARAM"):
    """Move the boundary vertices ``bverts`` to ``b_target`` without ever inverting a triangle.

    Counter-clockwise chart ``uv`` (det J > 0).  The boundary follows the straight path
    b(t) = b_start + t (b_target - b_start); each step moves along (boundary increment,
    harmonic extension inside), bounded by the exact largest flip-free step, and SLIM
    (symmetric Dirichlet w.r.t. the 3D triangles of ``prob``) restores the interior.
    Returns (uv, final symmetric Dirichlet energy, stats).
    """
    uv2 = np.asarray(uv, float)
    f22 = np.asarray(faces, int)
    bverts = np.asarray(bverts, int)
    b_target = np.asarray(b_target, float)
    free = np.setdiff1d(np.arange(len(uv2)), bverts)
    L = _uniform_laplacian(len(uv2), f22)
    lu = splu(L[free][:, free].tocsc())
    Lfb = L[free][:, bverts]
    cur = uv2.copy()
    b_start = cur[bverts].copy()
    # Boundary path b(t) = b_start + t (b_target - b_start), t in [0, 1].  Each
    # step moves along D = (boundary increment, harmonic extension inside),
    # with the step bounded by the exact largest flip-free step along D
    # (roots of det(J + a dJ)); SLIM then restores element quality before the
    # next step.  Nothing is ever accepted with an inverted triangle.
    t_done, steps, rejected, slim_total = 0.0, 0, 0, 0
    min_fraction = 1.0
    while t_done < 1.0 - 1e-12:
        if steps >= max_steps:
            raise SeamConformingReparamError(
                f"{error_prefix}_NO_PROGRESS: reached t={t_done:.4f} after {steps} steps")
        remaining = 1.0 - t_done
        delta_b = (b_target - cur[bverts])           # = remaining * (b_target - b_start)
        D = np.zeros_like(cur)
        D[bverts] = delta_b
        D[free] = -lu.solve(Lfb @ delta_b)
        _, J, _ = prob.energy(cur)
        amax = _max_feasible_step(J, prob.jacobians(D))
        frac = min(1.0, 0.5 * amax)
        steps += 1
        if frac < 1e-9:
            raise SeamConformingReparamError(
                f"{error_prefix}_FOLDOVER: the chart boundary cannot be moved beyond "
                f"t={t_done:.6f} of the way to the quad boundary without inverting a source triangle")
        trial = cur + frac * D
        if frac >= 1.0:
            trial[bverts] = b_target
        E, _, _ = prob.energy(trial)
        if not np.isfinite(E):
            rejected += 1
            frac *= 0.5
            trial = cur + frac * D
            E, _, _ = prob.energy(trial)
            if not np.isfinite(E):
                raise SeamConformingReparamError(f"{error_prefix}_STEP_BOUND_INCONSISTENT")
        min_fraction = min(min_fraction, frac)
        cur, _, k = _slim_iterations(prob, trial, free, iters=step_slim_iters)
        slim_total += k
        t_done = 1.0 if frac >= 1.0 else t_done + frac * remaining
    cur[bverts] = b_target                     # exact (removes round-off of the increments)
    E_chk = prob.energy(cur)[0]
    if not np.isfinite(E_chk):
        raise SeamConformingReparamError(f"{error_prefix}_FOLDOVER_AT_EXACT_BOUNDARY")
    cur, E1, k = _slim_iterations(prob, cur, free, iters=final_iters, tol=1e-7)
    slim_total += k
    return cur, E1, {"steps": int(steps), "rejected": int(rejected), "min_fraction": float(min_fraction),
                     "slim_iterations": int(slim_total)}


def seam_conforming_reparameterization(xyz, f3, uv, f2, quad_uv, quads, *,
                                       snap_nodes: bool = True, align_pairs: bool = True,
                                       seam_matching: bool = False, seam_matching_max_insertions: int = 24,
                                       max_steps: int = 2000, step_slim_iters: int = 4,
                                       final_iters: int = 80) -> ReparamResult:
    """OneString extension; see module docstring.  Raises on any failure."""
    started = time.perf_counter()
    xyz = np.asarray(xyz, float)
    f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float)
    f2 = np.asarray(f2, int)
    quad_uv = np.asarray(quad_uv, float)
    quads = np.asarray(quads, int)
    sa_in = signed_uv_areas(uv, f2)
    if not (np.all(sa_in > 0) or np.all(sa_in < 0)):
        raise SeamConformingReparamError("ZHANG2021_ONESTRING_SOURCE_CHART_NOT_INJECTIVE")
    u2x = uv_to_xyz_vertex_map(f3, f2, len(uv))
    src = build_source_boundary(uv, f2, u2x, xyz)
    init = initial_correspondence(src, quad_uv, quads, snap_nodes=snap_nodes)
    projection_distance = np.asarray(init.projection_distance, float)
    matching = None
    if seam_matching:
        from .zhang2021_onestring_seam_matching import match_seam_counts
        matching = match_seam_counts(src, quad_uv, quads, init, max_insertions=seam_matching_max_insertions)
        quad_uv, quads, init = matching.quad_uv, matching.quads, matching.init
    corr = finalize_correspondence(src, init, align_pairs=align_pairs)
    sp = split_for_correspondence(xyz, f3, uv, f2, src, corr)
    xyz2, f32, f22 = sp.xyz, sp.f3, sp.f2
    sa = signed_uv_areas(sp.uv, f22)
    if np.all(sa > 0):
        mirror = np.array([1.0, 1.0])
    elif np.all(sa < 0):
        mirror = np.array([-1.0, 1.0])     # solve in the CCW frame, mirror back at the end
    else:
        raise SeamConformingReparamError("ZHANG2021_ONESTRING_SOURCE_CHART_NOT_INJECTIVE_AFTER_SPLIT")
    uv2 = sp.uv * mirror
    qm = quad_uv * mirror
    loops = oriented_boundary_loops(f22)
    if len(loops) != 1:
        raise SeamConformingReparamError("split chart is not a disk")
    u2x2 = uv_to_xyz_vertex_map(f32, f22, len(uv2))
    src2 = build_source_boundary(sp.uv, f22, u2x2, xyz2)
    targets = chord_boundary_targets(uv2, src2.loop, sp.anchors, qm, corr.q_loop)
    bverts = np.asarray(src2.loop, int)
    b_target = np.asarray([targets[v] for v in bverts])
    free = np.setdiff1d(np.arange(len(uv2)), bverts)
    prob = _SDProblem(xyz2, f32, f22, len(uv2))
    E0 = prob.energy(uv2)[0]
    sd0 = prob.per_triangle_sd(uv2)
    if not np.isfinite(E0):
        raise SeamConformingReparamError("split chart is not injective")
    cur, E1, pstats = progressive_boundary_slim(prob, uv2, f22, bverts, b_target, max_steps=max_steps,
                                                step_slim_iters=step_slim_iters, final_iters=final_iters)
    steps, rejected = pstats["steps"], pstats["rejected"]
    min_fraction, slim_total = pstats["min_fraction"], pstats["slim_iterations"]
    sd1 = prob.per_triangle_sd(cur)
    if not np.all(np.isfinite(sd1)):
        raise SeamConformingReparamError("ZHANG2021_ONESTRING_REPARAM_FOLDOVER_FINAL")
    out_uv = cur * mirror
    sa_out = signed_uv_areas(out_uv, f22)
    if not (np.all(sa_out * np.sign(sa_in[0]) > 0)):
        raise SeamConformingReparamError("ZHANG2021_ONESTRING_REPARAM_ORIENTATION_CHANGED")
    w = prob.weight
    n_orig = len(uv)
    node_valence = seam_node_valences(src, corr, quads)
    metrics = {
        "zhang2021_onestring_reparam_model": (
            "OneString extension: seam-aware boundary correspondence + conforming edge splits + "
            "progressive bijective SLIM (symmetric Dirichlet), boundary fixed to Q's boundary polygon"),
        "zhang2021_onestring_snapped_seam_nodes": int(corr.snapped_nodes),
        "zhang2021_onestring_regularized_boundary_points": int(corr.regularized_points),
        "zhang2021_onestring_seam_nodes": int(len(src.nodes)),
        "zhang2021_onestring_seam_node_kinds": {k: sum(1 for v in src.node_kind.values() if v == k)
                                                for k in ("tip", "junction", "transition")},
        "zhang2021_onestring_unsnapped_seam_nodes": list(corr.unsnapped_nodes),
        "zhang2021_onestring_aligned_seam_vertex_pairs": int(corr.aligned_pairs),
        "zhang2021_onestring_soft_seam_branches": int(corr.misaligned_branches),
        "zhang2021_onestring_hard_seam_branches": int(sum(1 for e in corr.branch_report
                                                           if str(e.get("status", "")).startswith("hard"))),
        "zhang2021_onestring_seam_branch_report": corr.branch_report,
        "zhang2021_onestring_boundary_projection_max": float(np.max(projection_distance)),
        "zhang2021_onestring_boundary_projection_mean": float(np.mean(projection_distance)),
        "zhang2021_onestring_seam_matching_enabled": bool(seam_matching),
        "zhang2021_onestring_seam_matching_accepted": bool(matching.accepted) if matching else False,
        "zhang2021_onestring_seam_matching_rejection_reason": matching.rejection_reason if matching else "",
        "zhang2021_onestring_seam_matching_inserted_chords": int(matching.inserted_chords) if matching else 0,
        "zhang2021_onestring_seam_matching_added_quads": int(matching.added_quads) if matching else 0,
        "zhang2021_onestring_seam_matching_state_before": dict(matching.state_before) if matching else {},
        "zhang2021_onestring_seam_matching_state_after": dict(matching.state_after) if matching else {},
        "zhang2021_onestring_seam_constraints": list(matching.constraints) if matching else [],
        "zhang2021_onestring_seam_matching_log": list(matching.log) if matching else [],
        "zhang2021_onestring_split_boundary_edges": int(sp.split_boundary_edges),
        "zhang2021_onestring_split_interior_chords": int(sp.split_interior_chords),
        "zhang2021_onestring_centroid_fans": int(sp.centroid_fans),
        "zhang2021_onestring_added_uv_vertices": int(sp.new_uv_vertices),
        "zhang2021_onestring_added_xyz_vertices": int(sp.new_xyz_vertices),
        "zhang2021_onestring_sd_mean_before": float(E0),
        "zhang2021_onestring_sd_mean_after": float(E1),
        "zhang2021_onestring_sd_p95_before": _weighted_quantile(sd0, w, 0.95),
        "zhang2021_onestring_sd_p95_after": _weighted_quantile(sd1, w, 0.95),
        "zhang2021_onestring_sd_max_before": float(np.max(sd0)),
        "zhang2021_onestring_sd_max_after": float(np.max(sd1)),
        "zhang2021_onestring_reparam_steps": int(steps),
        "zhang2021_onestring_reparam_rejected_steps": int(rejected),
        "zhang2021_onestring_reparam_min_step_fraction": float(min_fraction),
        "zhang2021_onestring_reparam_slim_iterations": int(slim_total),
        "zhang2021_onestring_reparam_seconds": float(time.perf_counter() - started),
        "zhang2021_onestring_max_uv_displacement": float(np.max(np.linalg.norm(out_uv[:n_orig] - uv, axis=1))),
        "zhang2021_onestring_source_orientation": "ccw" if sa_in[0] > 0 else "cw",
        "zhang2021_onestring_seam_node_3d_valence": node_valence,
        "zhang2021_onestring_seam_nodes_low_3d_valence": int(sum(
            1 for v in node_valence if v["valence_sum"] < v["ideal"] - 1)),
    }
    return ReparamResult(xyz=xyz2, f3=f32, uv=out_uv, f2=f22, parent_face=sp.parent_face,
                         anchors=list(map(int, sp.anchors)), q_loop=list(map(int, corr.q_loop)),
                         quad_uv=np.asarray(quad_uv, float), quads=np.asarray(quads, int),
                         metrics=metrics)
