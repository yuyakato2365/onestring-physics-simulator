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


def _node_neighbourhood_corners(src, nodes: list[int], radius: float) -> set[int]:
    """Loop vertices at arc distance ~radius on both sides of every node.

    Keeping them as corners preserves the boundary direction on both sides of
    an OptCuts seam node at the panel scale.  Without them the simplified
    chords node->node can turn a reflex cut-tip corner (the 3D neighbourhood of
    a tip is a full turn) into a sharp convex corner, which leaves no room for
    the >= 3 panels a tip needs after hard seam closure.
    """
    if radius <= 0 or not nodes:
        return set()
    n, P = src.n, src.perimeter
    cum = np.asarray(src.cum[:-1], float)
    out: set[int] = set()
    ordered = sorted(nodes)
    for r, j in enumerate(ordered):
        nxt = ordered[(r + 1) % len(ordered)]
        prv = ordered[(r - 1) % len(ordered)]
        gap_next = (cum[nxt] - cum[j]) % P or P
        gap_prev = (cum[j] - cum[prv]) % P or P
        for direction, gap in ((+1, gap_next), (-1, gap_prev)):
            reach = min(float(radius), 0.45 * gap)
            k = j
            for _ in range(n):
                k2 = (k + direction) % n
                d = (cum[k2] - cum[j]) % P if direction > 0 else (cum[j] - cum[k2]) % P
                if d >= reach or k2 in (nxt, prv):
                    if k2 not in (nxt, prv):
                        out.add(int(k2))
                    break
                k = k2
    return out


def simplified_boundary(src, tol: float, *, node_keep_radius: float = 0.0) -> tuple[np.ndarray, list[int]]:
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
    mandatory = sorted(set(mandatory) | _node_neighbourhood_corners(src, list(src.nodes), node_keep_radius))
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


def boundary_features(src, scale: float, angle_deg: float = 35.0) -> list[int]:
    """Seam nodes + sharp corners of open-boundary runs (turning >= angle at the given scale).

    Cut runs carry no extra corners (both UV copies of a cut must stay
    parameterised identically); their only features are the seam nodes.
    """
    n, P = src.n, src.perimeter
    cum = np.asarray(src.cum[:-1], float)
    feats = set(int(j) for j in src.nodes)
    uv = src.uv

    def point_at(s):
        return src.point_at(s % P)[2]

    turning = np.zeros(n)
    for j in range(n):
        a = point_at(cum[j] - scale) - uv[j]
        b = point_at(cum[j] + scale) - uv[j]
        ang = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)) % 360.0   # interior angle (CCW)
        turning[j] = 180.0 - ang
    open_v = [j for j in range(n) if src.edge_kind[j] == "open" and src.edge_kind[(j - 1) % n] == "open"]
    cand = sorted((-abs(turning[j]), j) for j in open_v if abs(turning[j]) >= angle_deg)
    for _, j in cand:            # non-maximum suppression along the boundary
        if all(min(abs(cum[j] - cum[k]), P - abs(cum[j] - cum[k])) > scale for k in feats):
            feats.add(int(j))
    return sorted(feats)


def _fair_polyline(P: np.ndarray, tol: float) -> np.ndarray:
    """Discrete smoothing spline with fixed ends: min |D2 x|^2 + w |x - P|^2, max deviation <= tol.

    A run whose chord already stays within tol becomes a straight line.
    """
    m = len(P)
    if m <= 2:
        return P.copy()
    a, b = P[0], P[-1]
    t = np.linspace(0.0, 1.0, m)
    d = b - a
    L = float(np.linalg.norm(d))
    if L > 0:
        dist = np.abs((P[:, 0] - a[0]) * d[1] - (P[:, 1] - a[1]) * d[0]) / L
        if float(np.max(dist)) <= tol:
            seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            u = seg / max(seg[-1], 1e-300)
            return a + u[:, None] * d
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    D = sparse.diags([np.ones(m - 2), -2 * np.ones(m - 2), np.ones(m - 2)], [0, 1, 2], shape=(m - 2, m))
    DtD = (D.T @ D).tocsr()
    inner = np.arange(1, m - 1)

    def solve(w):
        A = (DtD + w * sparse.identity(m)).tocsr()
        rhs = w * P.copy()
        X = P.copy()
        Ai = A[inner][:, inner].tocsc()
        corr = A[inner][:, [0, m - 1]] @ P[[0, m - 1]]
        for c in range(2):
            X[inner, c] = spsolve(Ai, rhs[inner, c] - corr[:, c])
        return X

    lo, hi = 1e-8, 1e8          # w small = smooth, large = close to P
    best = P.copy()
    for _ in range(40):
        w = np.sqrt(lo * hi)
        X = solve(w)
        if float(np.max(np.linalg.norm(X - P, axis=1))) <= tol:
            best, hi = X, w
        else:
            lo = w
    return best


def faired_boundary(src, tol: float, *, corner_scale: float) -> tuple[np.ndarray, list[int]]:
    """Piecewise smooth boundary: per run between features a straight line or a smoothing spline."""
    feats = boundary_features(src, corner_scale)
    if len(feats) < 2:
        far = int(np.argmax(np.linalg.norm(src.uv - src.uv[0], axis=1)))
        feats = sorted({0, far} | set(feats))
    n = src.n
    targets = src.uv.copy()
    for r, a in enumerate(feats):
        b = feats[(r + 1) % len(feats)]
        idx = [(a + k) % n for k in range((b - a) % n + 1)] if b != a else list(range(n)) + [a]
        X = _fair_polyline(src.uv[idx], tol)
        targets[idx] = X
    return targets, feats


def _taubin_run(P: np.ndarray, scale: float, lam: float = 0.6, pass_ratio: float = 4.0,
                stop_ratio: float = 2.0) -> np.ndarray:
    """Non-shrinking (Taubin lambda|mu) low-pass of an open polyline with fixed end points.

    Pass band = wavelengths >= pass_ratio*scale (shapes of a few panels survive), wavelengths
    <= stop_ratio*scale (bumps about one panel wide) are damped to <= 10 %.  Only the normal component of each Laplacian step is applied,
    so vertices do not slide along the curve (the boundary <-> 3D correspondence is kept)."""
    m = len(P)
    if m <= 3 or scale <= 0:
        return P.copy()
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = float(np.mean(seg)) if np.mean(seg) > 0 else 1e-12
    k_of = lambda wl: 1.0 - np.cos(min(np.pi, 2.0 * np.pi * s / wl))      # noqa: E731
    k_pb = float(k_of(pass_ratio * scale))
    mu = 1.0 / (k_pb - 1.0 / lam)
    f_stop = abs((1.0 - lam * k_of(stop_ratio * scale)) * (1.0 - mu * k_of(stop_ratio * scale)))
    n_it = 5 if f_stop <= 0 or f_stop >= 1 else int(np.clip(np.ceil(np.log(0.1) / np.log(f_stop)), 5, 600))
    X = P.copy()

    def step(X, f):
        d_prev = X[1:-1] - X[:-2]
        d_next = X[2:] - X[1:-1]
        wp = 1.0 / np.maximum(np.linalg.norm(d_prev, axis=1), 1e-300)
        wn = 1.0 / np.maximum(np.linalg.norm(d_next, axis=1), 1e-300)
        lap = (wp[:, None] * (-d_prev) + wn[:, None] * d_next) / (wp + wn)[:, None]
        t = X[2:] - X[:-2]
        t /= np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        lap = np.sum(lap * nrm, axis=1)[:, None] * nrm
        Y = X.copy()
        Y[1:-1] += f * lap
        return Y

    for _ in range(n_it):
        X = step(step(X, lam), mu)
    return X


def _neck_spans(P: np.ndarray, width: float) -> list[tuple[int, int]]:
    """Thin notches / fingers of an open polyline: vertex pairs i < j closer than ``width`` while
    the curve between them is much longer (outermost spans, non-overlapping).  A low-pass moves
    the two walls of such a feature sideways into each other (self-intersection), and closing it
    would stretch the chart enormously (the walls are far apart in 3D), so it is kept as it is."""
    X = np.asarray(P, float)
    m = len(X)
    if m < 5 or width <= 0:
        return []
    from scipy.spatial import cKDTree
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    pairs = [(i, j) for i, j in cKDTree(X).query_pairs(width)
             if j - i >= 3 and 3.0 * np.linalg.norm(X[i] - X[j]) < seg[j] - seg[i] < 0.5 * seg[-1]]
    spans: list[tuple[int, int]] = []
    for i, j in sorted(pairs, key=lambda ij: -(seg[ij[1]] - seg[ij[0]])):
        if any(not (j < a or i > b) for a, b in spans):
            continue
        spans.append((i, j))
    return sorted(spans)


def taubin_boundary(src, scale: float, fix_tips: bool = False) -> tuple[np.ndarray, list[int]]:
    """Boundary low-pass at the panel scale; only the OptCuts seam junctions and cut/open
    transitions stay fixed (cut tips too when ``fix_tips``).  A cut tip is a single chart vertex
    between the two copies of its branch, so smoothing through it keeps the seam correspondence;
    its sharp spike is a sub-panel feature that the grid cannot represent anyway."""
    n = src.n
    fixed = sorted(set(int(j) for j in src.nodes if fix_tips or src.node_kind.get(j) != "tip"))
    if len(fixed) < 2:
        far = int(np.argmax(np.linalg.norm(src.uv - src.uv[fixed[0] if fixed else 0], axis=1)))
        fixed = sorted(set(fixed) | {fixed[0] if fixed else 0, far})
    extra: set = set()
    for _ in range(8):
        targets = _taubin_targets(src, scale, sorted(set(fixed) | extra))
        # a boundary edge that collapses (typically at a cut tip pulled back into its slit) would
        # leave a degenerate chart triangle: keep that vertex where it is and smooth again
        e0 = np.linalg.norm(np.roll(src.uv, -1, axis=0) - src.uv, axis=1)
        e1 = np.linalg.norm(np.roll(targets, -1, axis=0) - targets, axis=1)
        bad = [int(i) for i in np.flatnonzero(e1 < 0.25 * e0)]
        new = set()
        for i in bad:
            j = (i + 1) % n
            new.add(i if src.node_kind.get(i) == "tip" or src.node_kind.get(j) != "tip" else j)
        new -= set(fixed) | extra
        if not new:
            break
        extra |= new
    taubin_boundary.pinned = sorted(extra)
    return targets, sorted(set(fixed) | extra)


def _taubin_targets(src, scale: float, fixed: list[int]) -> np.ndarray:
    n = src.n
    targets = src.uv.copy()
    # thin necks (narrow cut slits including their tip, sub-panel notches) are found on the whole
    # loop and kept as they are; only the pieces between fixed nodes / neck ends are low-passed
    order = [(fixed[0] + k) % n for k in range(n + 1)]
    P = src.uv[order]
    spans = _neck_spans(P, 0.5 * scale)
    taubin_boundary.necks = len(spans)
    anchors = {k for k in range(n + 1) if order[k] in set(fixed)} | {k for sp in spans for k in sp}
    frozen = {k for i, j in spans for k in range(i + 1, j)}
    cuts = sorted(k for k in anchors if k not in frozen)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < 2 or (lo + 1) in frozen:
            continue
        targets[order[lo:hi + 1]] = _taubin_run(P[lo:hi + 1], scale)
    return targets


def precondition_omega(xyz, f3, uv, f2, tol_ratio: float, *, sd_mean_ratio_max: float = 1.08,
                       sd_p95_ratio_max: float = 1.25, max_steps: int = 2000,
                       node_keep_ratio: float = 0.0, boundary_model: str = "polyline",
                       smooth_scale: float = 0.0) -> PreconditionResult:
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
    if boundary_model == "taubin":
        # scale-based low-pass: the deviation bound is the panel size itself (not a fixed tolerance)
        taubin_boundary.necks = 0
        targets, corners = taubin_boundary(src, float(smooth_scale))
        tol = max(tol, 2.0 * float(smooth_scale))
    elif boundary_model == "spline":
        targets, corners = faired_boundary(src, tol, corner_scale=max(float(node_keep_ratio), 0.02) * diag)
    else:
        targets, corners = simplified_boundary(src, tol, node_keep_radius=float(node_keep_ratio) * diag)
    m = {"zhang2021_precondition_tol_ratio": tol_ratio,
         "zhang2021_precondition_node_keep_ratio": float(node_keep_ratio),
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


__all__ = ["PreconditionResult", "precondition_omega", "simplified_boundary", "faired_boundary", "boundary_features"]
