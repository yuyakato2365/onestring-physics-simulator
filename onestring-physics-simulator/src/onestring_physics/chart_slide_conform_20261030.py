"""Keep the re-parameterised chart conforming after boundary grid vertices slide (2026-10-30).

The structured-grid chart is built so that its boundary IS the quad boundary polygon: every grid
boundary vertex is a chart vertex and the chart vertices between two grid boundary vertices lie on
the straight grid edge.  The downstream seam classifier relies on that (each grid boundary edge must
be covered by chart boundary edges, and the two sides of a cut are matched by their 3D samples).

When the panel optimiser slides a grid boundary vertex to a point inside a chart boundary edge, that
property breaks and the seam can no longer be closed in K3D.  This module restores it:

1. a chart vertex is inserted at the new position of every slid grid vertex, on the 3D boundary edge
   it lies on (the 3D edge is split once and every chart triangle using it is split, so both sides
   of a cut get a vertex at exactly the same 3D point);
2. the chart vertices between two grid boundary vertices whose run changed are moved onto the new
   straight grid edge, spaced by their 3D arc length (their 3D points do not move);
3. if a boundary triangle flips, the chart vertices near the boundary are relaxed (uniform Laplacian,
   boundary fixed) until no triangle is flipped; otherwise an error is raised.
"""
from __future__ import annotations

import numpy as np

from .zhang2021_onestring_boundary import oriented_boundary_loops, quad_boundary_loop


class ChartConformError(RuntimeError):
    pass


def _signed(uv, f):
    a, b, c = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def conform_chart_to_grid_boundary(rp, quv_old, quv_new, quads, tol_ratio=1e-9):
    """Return (new_rp, report).  ``rp`` is a ReparamResult-like object (xyz, f3, uv, f2, parent_face,
    anchors, q_loop); grid vertex ids of ``quads`` index ``quv_old`` / ``quv_new``."""
    import copy
    uv = np.asarray(rp.uv, float).copy(); f2 = np.asarray(rp.f2, int).copy()
    xyz = np.asarray(rp.xyz, float).copy(); f3 = np.asarray(rp.f3, int).copy()
    parent = np.asarray(rp.parent_face).copy()
    span = float(np.max(np.ptp(uv, axis=0)))
    tol = tol_ratio * max(span, 1e-12)
    qb = [int(v) for v in quad_boundary_loop(np.asarray(quads, int))]
    moved = [g for g in qb if np.linalg.norm(quv_new[g] - quv_old[g]) > tol]
    report = {"chart_conform_moved_grid_vertices": len(moved), "chart_conform_inserted_uv": 0,
              "chart_conform_inserted_xyz": 0, "chart_conform_relaxed_vertices": 0}
    if not moved:
        return rp, report
    anchor = {int(q): int(a) for q, a in zip(rp.q_loop, rp.anchors)}
    for g in qb:
        if g not in anchor or np.linalg.norm(uv[anchor[g]] - quv_old[g]) > 1e3 * tol:
            raise ChartConformError(f"CHART_CONFORM_ANCHOR_MISMATCH: grid vertex {g}")
    xid = np.zeros(len(uv), int); xid[f2.ravel()] = f3.ravel()
    split3 = {}                                   # (a3, b3, quantised point) -> new 3D vertex

    def boundary_edges():
        loop = oriented_boundary_loops(f2)
        if len(loop) != 1:
            raise ChartConformError("CHART_CONFORM_CHART_NOT_A_DISK")
        return [int(v) for v in loop[0]]

    def split_uv_edge(A, B, p_uv, n3):
        """split the (single) chart triangle that has the boundary uv edge A-B at p_uv; vertex -> n3."""
        nonlocal uv, f2, f3, parent, xid
        hit = None
        for i, t in enumerate(f2):
            for k in range(3):
                if {int(t[k]), int(t[(k + 1) % 3])} == {A, B}:
                    hit = (i, k)
                    break
            if hit:
                break
        if hit is None:
            raise ChartConformError("CHART_CONFORM_EDGE_NOT_FOUND")
        i, k = hit
        t2, t3 = f2[i].copy(), f3[i].copy()
        a, b, c = int(t2[k]), int(t2[(k + 1) % 3]), int(t2[(k + 2) % 3])
        a3, b3, c3 = int(t3[k]), int(t3[(k + 1) % 3]), int(t3[(k + 2) % 3])
        nu = len(uv)
        uv = np.vstack((uv, p_uv[None])); xid = np.r_[xid, n3]
        f2[i] = (a, nu, c); f3[i] = (a3, n3, c3)
        f2 = np.vstack((f2, [[nu, b, c]])); f3 = np.vstack((f3, [[n3, b3, c3]])); parent = np.r_[parent, parent[i]]
        report["chart_conform_inserted_uv"] += 1
        return nu

    new_anchor = dict(anchor)
    for g in moved:
        loop = boundary_edges(); nb = len(loop); pos = {v: i for i, v in enumerate(loop)}
        p = quv_new[g]
        best = None
        for i in range(nb):
            A, B = loop[i], loop[(i + 1) % nb]
            ab = uv[B] - uv[A]
            t = float(np.clip(np.dot(p - uv[A], ab) / max(float(np.dot(ab, ab)), 1e-300), 0.0, 1.0))
            d = float(np.linalg.norm(uv[A] + t * ab - p))
            if best is None or d < best[0]:
                best = (d, A, B, t)
        d, A, B, t = best
        if d > 1e3 * tol:
            raise ChartConformError(f"CHART_CONFORM_SLID_VERTEX_OFF_BOUNDARY: grid vertex {g} distance {d:.3g}")
        elen = float(np.linalg.norm(uv[B] - uv[A]))
        if t * elen <= 1e3 * tol:
            new_anchor[g] = A; continue
        if (1 - t) * elen <= 1e3 * tol:
            new_anchor[g] = B; continue
        a3, b3 = int(xid[A]), int(xid[B])
        P3 = (1 - t) * xyz[a3] + t * xyz[b3]
        key = (min(a3, b3), max(a3, b3))
        n3 = None
        for (kk, q3), v3 in split3.items():
            if kk == key and np.linalg.norm(np.asarray(q3) - P3) <= 1e-9 * max(1.0, float(np.ptp(xyz))):
                n3 = v3
        if n3 is None:
            n3 = len(xyz); xyz = np.vstack((xyz, P3[None])); split3[(key, tuple(P3))] = n3
            report["chart_conform_inserted_xyz"] += 1
        new_anchor[g] = split_uv_edge(A, B, p, n3)
    # every other chart triangle that still uses a split 3D edge gets the same split (other cut side)
    for (key, q3), n3 in split3.items():
        while True:
            loop = boundary_edges(); nb = len(loop)
            todo = None
            for i in range(nb):
                A, B = loop[i], loop[(i + 1) % nb]
                if {int(xid[A]), int(xid[B])} == set(key):
                    tt = float(np.dot(xyz[n3] - xyz[xid[A]], xyz[xid[B]] - xyz[xid[A]]) /
                               max(float(np.dot(xyz[xid[B]] - xyz[xid[A]], xyz[xid[B]] - xyz[xid[A]])), 1e-300))
                    todo = (A, B, tt)
                    break
            if todo is None:
                break
            A, B, tt = todo
            split_uv_edge(A, B, (1 - tt) * uv[A] + tt * uv[B], n3)
    # move the chart vertices of every changed run onto the new straight grid edge (3D arc spacing)
    loop = boundary_edges(); nb = len(loop); pos = {v: i for i, v in enumerate(loop)}
    inv = {}
    for g in qb:
        inv[new_anchor[g]] = g
    apos = sorted(pos[a] for a in inv)
    moved_set = set(moved)
    touched = set()
    for j in range(len(apos)):
        i0, i1 = apos[j], apos[(j + 1) % len(apos)]
        g0, g1 = inv[loop[i0]], inv[loop[i1]]
        if g0 not in moved_set and g1 not in moved_set:
            continue
        run = [loop[(i0 + s) % nb] for s in range(((i1 - i0) % nb) + 1)]
        P = xyz[xid[np.asarray(run)]]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        s = s / max(s[-1], 1e-300)
        for v, sv in zip(run[1:-1], s[1:-1]):
            uv[v] = (1 - sv) * quv_new[g0] + sv * quv_new[g1]
            touched.add(v)
        uv[run[0]] = quv_new[g0]; uv[run[-1]] = quv_new[g1]
    # relax the chart near the boundary if a triangle flipped
    sign = 1.0 if np.sum(_signed(uv, f2)) > 0 else -1.0
    bad = np.where(sign * _signed(uv, f2) <= 1e-14 * span * span)[0]
    if len(bad):
        isb = np.zeros(len(uv), bool); isb[loop] = True
        nbr = [set() for _ in range(len(uv))]
        for t in f2:
            for k in range(3):
                nbr[t[k]].update((int(t[(k + 1) % 3]), int(t[(k + 2) % 3])))
        region = set(int(v) for v in f2[bad].ravel())
        for _ in range(6):
            region |= {u for v in list(region) for u in nbr[v]}
        free = np.array(sorted(v for v in region if not isb[v]), int)
        for it in range(400):
            for v in free:
                uv[v] = np.mean(uv[list(nbr[v])], axis=0)
            if not np.any(sign * _signed(uv, f2) <= 1e-14 * span * span):
                break
        report["chart_conform_relaxed_vertices"] = int(len(free))
        if np.any(sign * _signed(uv, f2) <= 1e-14 * span * span):
            raise ChartConformError("CHART_CONFORM_FLIPPED_TRIANGLES_AFTER_RELAXATION")
    out = copy.copy(rp)
    out.uv, out.f2, out.xyz, out.f3, out.parent_face = uv, f2, xyz, f3, parent
    out.anchors = [int(new_anchor[int(q)]) for q in rp.q_loop]
    return out, report
