"""OneString Ω pre-conditioning BEFORE quad meshing (not part of Zhang et al. 2021).

Implements Step 2 of the 2026-10-26 design (``phi_1 = h o phi_0``):

    h : Omega_0 -> Omega_1   bijective, boundary simplified, interior re-parameterised

* Boundary (E_deviation / E_smooth / E_line of the design):
  every boundary run between two OptCuts seam nodes (cut tips, junctions,
  cut/open transitions -- kept fixed as mandatory corners) is replaced by a
  Douglas-Peucker polyline with tolerance ``tol``.  Straight segments give
  E_line = E_smooth = 0 inside a segment; corners are kept only where the
  boundary deviates from a line by more than ``tol``; the two-sided Hausdorff
  distance between the old and the new boundary is measured and must stay
  <= ``tol`` (hard constraint C5).  Boundary vertices keep their order and are
  re-distributed along their segment by arc-length fraction (monotone).
* Interior: progressive bijective SLIM (symmetric Dirichlet w.r.t. the 3D
  triangles of S), never accepting an inverted triangle (C1).  3D positions are
  untouched, so the inverse map to S, the cut topology and the seam
  correspondence of OptCuts are preserved exactly; only UV moves.
* Acceptance: mean and p95 symmetric Dirichlet must not grow by more than the
  given ratios (the design's 1.035 / 1.08 rule plus a tail check); otherwise the
  candidate is rejected with an explicit reason -- never silently used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .zhang2021_onestring_boundary import (
    BoundaryCorrespondenceError,
    build_source_boundary,
    uv_to_xyz_vertex_map,
)
from .zhang2021_onestring_reparam import (
    SeamConformingReparamError,
    _apply_edge_splits,
    _SDProblem,
    _weighted_quantile,
    progressive_boundary_slim,
    signed_uv_areas,
)
from .zhang2021_quad_metrics import two_sided_boundary_hausdorff


@dataclass
class PreconditionResult:
    tol_ratio: float
    accepted: bool
    xyz: np.ndarray                      # S (input vertices + chord split points on S)
    f3: np.ndarray
    uv: np.ndarray                       # Omega_1
    f2: np.ndarray
    reason: str = ""
    metrics: dict = field(default_factory=dict)


def split_boundary_chords(xyz, f3, uv, f2):
    """Split every interior edge joining two boundary vertices at its midpoint (UV and 3D).

    A straightened boundary segment would otherwise make such a chord collinear
    with the boundary and squeeze the triangles between them to zero area.
    The 3D split point lies on the same edge of S, so S is unchanged.
    """
    from collections import defaultdict
    from .zhang2021_onestring_boundary import oriented_boundary_loops
    f2 = np.asarray(f2, int)
    on_b = np.zeros(len(uv), bool)
    for loop in oriented_boundary_loops(f2):
        on_b[np.asarray(loop, int)] = True
    count: dict[tuple[int, int], int] = defaultdict(int)
    for t in f2:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            count[(a, b) if a < b else (b, a)] += 1
    splits = {k: [0.5] for k, c in count.items() if c == 2 and on_b[k[0]] and on_b[k[1]]}
    if not splits:
        return np.asarray(xyz, float), np.asarray(f3, int), np.asarray(uv, float), f2, 0
    x2, t3, u2, t2, _parent, _chains, _fans = _apply_edge_splits(xyz, f3, uv, f2, splits)
    return x2, t3, u2, t2, len(splits)


def _douglas_peucker(P: np.ndarray, tol: float) -> list[int]:
    """Indices (into P, endpoints included) of the simplified open polyline."""
    keep = {0, len(P) - 1}
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        d = b - a
        L = float(np.hypot(*d))
        Q = P[i + 1:j] - a
        if L <= 1e-300:
            dist = np.linalg.norm(Q, axis=1)
        else:
            dist = np.abs(Q[:, 0] * d[1] - Q[:, 1] * d[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep.add(m)
            stack += [(i, m), (m, j)]
    return sorted(keep)


def simplified_boundary(src, tol: float) -> tuple[np.ndarray, list[int]]:
    """Target position of every source loop vertex + the retained corner positions."""
    n = src.n
    P = src.uv
    mandatory = sorted(set(int(j) for j in src.nodes))
    if not mandatory:                                  # e.g. an open disk without cuts
        far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
        mandatory = sorted({0, far})
    if len(mandatory) == 1:
        far = int(np.argmax(np.linalg.norm(P - P[mandatory[0]], axis=1)))
        mandatory = sorted({mandatory[0], far})
    corners: set[int] = set(mandatory)
    for r, a in enumerate(mandatory):
        b = mandatory[(r + 1) % len(mandatory)]
        idx = [(a + k) % n for k in range(((b - a) % n) + 1)]
        for k in _douglas_peucker(P[idx], tol):
            corners.add(idx[k])
    corners_sorted = sorted(corners)
    targets = np.empty_like(P)
    for r, a in enumerate(corners_sorted):
        b = corners_sorted[(r + 1) % len(corners_sorted)]
        span = (b - a) % n or n
        ca = src.cum[a]
        length = (src.cum[b] - ca) % src.perimeter if span < n else src.perimeter
        for k in range(span):
            j = (a + k) % n
            t = 0.0 if length <= 0 else ((src.cum[j] - ca) % src.perimeter) / length
            targets[j] = P[a] + t * (P[b] - P[a])
    return targets, corners_sorted


def precondition_omega(xyz, f3, uv, f2, tol_ratio: float, *, sd_mean_ratio_max: float = 1.08,
                       sd_p95_ratio_max: float = 1.25, max_steps: int = 2000) -> PreconditionResult:
    xyz = np.asarray(xyz, float)
    f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float)
    f2 = np.asarray(f2, int)
    tol_ratio = float(tol_ratio)
    if tol_ratio <= 0.0:
        return PreconditionResult(0.0, True, xyz, f3, uv.copy(), f2, "identity (no pre-conditioning)",
                                  {"zhang2021_precondition_tol_ratio": 0.0,
                                   "zhang2021_precondition_identity": True})
    xyz0, f30, uv0, f20 = xyz, f3, uv, f2

    def reject(reason: str, m: dict | None = None) -> PreconditionResult:
        return PreconditionResult(tol_ratio, False, xyz0, f30, uv0.copy(), f20, reason, dict(m or {}))

    xyz, f3, uv, f2, n_chords = split_boundary_chords(xyz, f3, uv, f2)
    sa = signed_uv_areas(uv, f2)
    if np.all(sa > 0):
        mirror = np.array([1.0, 1.0])
    elif np.all(sa < 0):
        mirror = np.array([-1.0, 1.0])
    else:
        raise SeamConformingReparamError("ZHANG2021_PRECONDITION_SOURCE_CHART_NOT_INJECTIVE")
    uvm = uv * mirror
    u2x = uv_to_xyz_vertex_map(f3, f2, len(uv))
    try:
        src = build_source_boundary(uvm, f2, u2x, xyz)
    except BoundaryCorrespondenceError as exc:
        return reject(str(exc))
    diag = float(np.linalg.norm(np.ptp(uv, axis=0)))
    tol = tol_ratio * diag
    targets, corners = simplified_boundary(src, tol)
    m = {"zhang2021_precondition_tol_ratio": tol_ratio,
         "zhang2021_precondition_identity": False,
         "zhang2021_precondition_boundary_vertices": int(src.n),
         "zhang2021_precondition_corners": int(len(corners)),
         "zhang2021_precondition_mandatory_seam_nodes": int(len(src.nodes)),
         "zhang2021_precondition_split_chords": int(n_chords)}
    from shapely.geometry import LinearRing
    if not LinearRing(targets).is_simple:
        return reject("ZHANG2021_PRECONDITION_TARGET_BOUNDARY_SELF_INTERSECTS", m)
    loop = list(range(src.n))
    dh, _, _ = two_sided_boundary_hausdorff(src.uv, [loop], targets, [loop])
    m["zhang2021_precondition_boundary_hausdorff_percent_bbox"] = 100.0 * float(dh) / max(diag, 1e-300)
    if dh > tol * (1.0 + 1e-6) + 1e-12:
        return reject(f"ZHANG2021_PRECONDITION_HAUSDORFF_EXCEEDED: {dh:.4g} > {tol:.4g}", m)
    prob = _SDProblem(xyz, f3, f2, len(uv))
    E0 = float(prob.energy(uvm)[0])
    sd0 = prob.per_triangle_sd(uvm)
    bverts = np.asarray(src.loop, int)
    try:
        cur, E1, stats = progressive_boundary_slim(prob, uvm, f2, bverts, targets, max_steps=max_steps,
                                                   error_prefix="ZHANG2021_PRECONDITION")
    except SeamConformingReparamError as exc:
        return reject(str(exc), m)
    sd1 = prob.per_triangle_sd(cur)
    w = prob.weight
    p95_0, p95_1 = _weighted_quantile(sd0, w, 0.95), _weighted_quantile(sd1, w, 0.95)
    m.update({
        "zhang2021_precondition_sd_mean_before": E0, "zhang2021_precondition_sd_mean_after": float(E1),
        "zhang2021_precondition_sd_p95_before": p95_0, "zhang2021_precondition_sd_p95_after": p95_1,
        "zhang2021_precondition_sd_max_before": float(np.max(sd0)),
        "zhang2021_precondition_sd_max_after": float(np.max(sd1)),
        "zhang2021_precondition_sd_mean_ratio": float(E1) / E0,
        "zhang2021_precondition_sd_p95_ratio": p95_1 / p95_0,
        "zhang2021_precondition_slim_steps": stats["steps"],
        "zhang2021_precondition_max_uv_displacement_percent_bbox":
            100.0 * float(np.max(np.linalg.norm(cur - uvm, axis=1))) / max(diag, 1e-300),
    })
    out = cur * mirror
    if not np.all(np.sign(signed_uv_areas(out, f2)) == np.sign(sa[0])):
        return reject("ZHANG2021_PRECONDITION_ORIENTATION_CHANGED", m)
    if float(E1) / E0 > sd_mean_ratio_max or p95_1 / p95_0 > sd_p95_ratio_max:
        return reject(
            f"ZHANG2021_PRECONDITION_DISTORTION_REJECTED: SD mean x{float(E1) / E0:.3f} "
            f"(limit {sd_mean_ratio_max}), p95 x{p95_1 / p95_0:.3f} (limit {sd_p95_ratio_max})", m)
    return PreconditionResult(tol_ratio, True, xyz, f3, out, f2, "accepted", m)


__all__ = ["PreconditionResult", "precondition_omega", "simplified_boundary"]
