"""Hard OptCuts-seam constraints as an explicit linear variable reduction (OneString).

The certified source-seam constraints of ``optcuts_seam_exact_closure`` are
linear rows ``sum_i w_i x_i = 0`` (one per breakpoint of the two UV copies of
every physical cut edge).  For a seam whose two M2D sides are sampled at the
same 3D points (matched partition) the rows are vertex equalities.  If one
side is a refinement of the other (nested partition) the extra vertices are
T-vertices: ``x_b = (1 - s) x_a1 + s x_a2`` with ``a1, a2`` the endpoints of
the coarse side's edge.  Rows between two edge interiors are implied by these
(3D arc-length fractions are used on both sides).

Here every row is eliminated explicitly: one free variable per equality
group, T-vertices substituted by their host edge, all other vertices free::

    x = E z        (E sparse, n x m)

Every optimiser that works on ``z`` (the paper local/global K3D of the
2026-09-18 route, its planarity polish) therefore keeps the seam closed
**exactly** at every iterate - no penalty, no post-hoc welding.  An
interleaved partition (a T-vertex hosted by an edge that itself ends at a
T-vertex) would force whole seam branches to be straight; it is reported as
not eliminable and the seam is not accepted as hard.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse


class SeamReductionError(RuntimeError):
    pass


def _canonical(row) -> list[tuple[int, float]]:
    acc: dict[int, float] = {}
    for i, w in row:
        acc[int(i)] = acc.get(int(i), 0.0) + float(w)
    return [(i, w) for i, w in sorted(acc.items()) if abs(w) > 1e-12]


def classify_rows(rows) -> dict:
    """Split rows into equalities, T-vertex rows and implied (edge-edge) rows."""
    eq, trow, other = [], [], []
    for raw in rows:
        r = _canonical(raw)
        if not r:
            continue
        if len(r) == 2 and abs(r[0][1] + r[1][1]) <= 1e-9 * max(abs(r[0][1]), 1.0):
            eq.append((r[0][0], r[1][0]))
            continue
        if len(r) == 3:
            pos = [(i, w) for i, w in r if w > 0]
            neg = [(i, w) for i, w in r if w < 0]
            single, pair = (neg, pos) if len(neg) == 1 else (pos, neg) if len(pos) == 1 else (None, None)
            if single is not None and len(pair) == 2:
                b, wb = single[0]
                total = abs(pair[0][1]) + abs(pair[1][1])
                if abs(total - abs(wb)) <= 1e-7 * max(abs(wb), 1.0):
                    (a1, w1), (a2, w2) = pair
                    s = abs(w2) / total
                    trow.append((int(b), int(a1), int(a2), float(s)))
                    continue
        other.append(r)
    return {"equalities": eq, "t_rows": trow, "other": other}


@dataclass
class SeamReduction:
    n: int
    E: sparse.csr_matrix                 # (n, m) : x = E z
    free_of_vertex: np.ndarray           # column of a free vertex, -1 for a T-vertex
    t_vertices: list[tuple[int, int, int, float]] = field(default_factory=list)
    groups: list[list[int]] = field(default_factory=list)
    rows: list = field(default_factory=list)
    max_row_violation_of_basis: float = 0.0

    @property
    def m(self) -> int:
        return int(self.E.shape[1])

    def expand(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float).reshape(self.m, -1)
        return np.asarray(self.E @ z)

    def fit(self, x: np.ndarray) -> np.ndarray:
        """Least-squares reduced coordinates of x (exact if x is feasible)."""
        x = np.asarray(x, float).reshape(self.n, -1)
        EtE = (self.E.T @ self.E).tocsc()
        from scipy.sparse.linalg import splu
        lu = splu(EtE + sparse.identity(self.m, format="csc") * 1e-14)
        return lu.solve(np.asarray(self.E.T @ x))

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.expand(self.fit(x))

    def kron(self, dim: int) -> sparse.csr_matrix:
        return sparse.kron(self.E, sparse.identity(int(dim), format="csr"), format="csr")

    def residual(self, x: np.ndarray) -> float:
        if not self.rows:
            return 0.0
        from .optcuts_seam_soft_constraints import row_matrix
        C = row_matrix(self.n, self.rows)
        x = np.asarray(x, float).reshape(self.n, -1)
        return float(np.max(np.abs(C @ x))) if C.shape[0] else 0.0


def build_seam_reduction(n_vertices: int, groups, rows) -> SeamReduction:
    """Eliminate equality groups and T-vertices; verify every row against the basis."""
    n = int(n_vertices)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return int(i)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for g in groups or []:
        g = [int(v) for v in g]
        for v in g[1:]:
            union(g[0], v)
    cls = classify_rows(rows or [])
    for a, b in cls["equalities"]:
        union(a, b)
    t_rows = cls["t_rows"]
    t_set = {b for b, *_ in t_rows}
    # A T-vertex must be eliminated through a host edge whose endpoints are
    # NOT T-vertices themselves (nested partition); a T-vertex that is also
    # equal to another vertex is fine only if all its group members agree.
    by_t: dict[int, tuple[int, int, float]] = {}
    for b, a1, a2, s in t_rows:
        rb = find(b)
        key = rb
        if key in by_t:
            # Several rows for one T-group: they must describe the same point;
            # checked by the final row verification.
            continue
        by_t[key] = (a1, a2, s)
    t_roots = set(by_t)
    for key, (a1, a2, s) in by_t.items():
        if find(a1) in t_roots or find(a2) in t_roots:
            raise SeamReductionError(
                "ZHANG2021_SEAM_PARTITION_INTERLEAVED: T-vertex group "
                f"{key} is hosted by an edge ({a1},{a2}) ending at another T-vertex; the two UV "
                "sides are not nested, so exact closure would force the seam branch straight")
    roots = np.asarray([find(i) for i in range(n)])
    free_roots = [r for r in np.unique(roots) if int(r) not in t_roots]
    col = {int(r): k for k, r in enumerate(free_roots)}
    rr, cc, vv = [], [], []
    free_of_vertex = np.full(n, -1, int)
    for i in range(n):
        r = int(roots[i])
        if r in col:
            rr.append(i)
            cc.append(col[r])
            vv.append(1.0)
            free_of_vertex[i] = col[r]
        else:
            a1, a2, s = by_t[r]
            for a, w in ((a1, 1.0 - s), (a2, s)):
                ra = int(roots[a])
                if ra not in col:
                    raise SeamReductionError("ZHANG2021_SEAM_REDUCTION_INTERNAL: host not free")
                rr.append(i)
                cc.append(col[ra])
                vv.append(w)
    E = sparse.coo_matrix((vv, (rr, cc)), shape=(n, len(free_roots))).tocsr()
    E.sum_duplicates()
    red = SeamReduction(n=n, E=E, free_of_vertex=free_of_vertex,
                        t_vertices=[(int(b), int(a1), int(a2), float(s)) for b, a1, a2, s in t_rows],
                        groups=[sorted(int(v) for v in np.flatnonzero(roots == r)) for r in np.unique(roots)
                                if np.sum(roots == r) > 1],
                        rows=[list(map(tuple, _canonical(r))) for r in (rows or [])])
    if red.rows:
        from .optcuts_seam_soft_constraints import row_matrix
        C = row_matrix(n, red.rows)
        CE = (C @ E).tocoo()
        red.max_row_violation_of_basis = float(np.max(np.abs(CE.data))) if CE.nnz else 0.0
        if red.max_row_violation_of_basis > 1e-7:
            raise SeamReductionError(
                "ZHANG2021_SEAM_ROWS_INCONSISTENT_WITH_NESTED_BASIS: max |C E| = "
                f"{red.max_row_violation_of_basis:.3e} (an implied seam row is not satisfied by the "
                "eliminated variables; the two sides' breakpoints are not 3D-consistent)")
    return red


def nested_partition_audit(n_vertices: int, groups, rows) -> dict:
    """Is exact (hard) closure possible without forcing seam branches straight?"""
    cls = classify_rows(rows or [])
    out = {"equality_rows": len(cls["equalities"]), "t_vertex_rows": len(cls["t_rows"]),
           "implied_rows": len(cls["other"]), "eliminable": False, "reason": ""}
    try:
        red = build_seam_reduction(n_vertices, groups, rows)
    except SeamReductionError as exc:
        out["reason"] = str(exc)
        return out
    out.update(eliminable=True, reduced_unknowns=int(red.m), t_vertices=len({b for b, *_ in red.t_vertices}),
               max_row_violation_of_basis=red.max_row_violation_of_basis)
    return out


__all__ = ["SeamReduction", "SeamReductionError", "build_seam_reduction", "classify_rows",
           "nested_partition_audit"]
