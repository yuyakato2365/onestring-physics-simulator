"""Mac print_model_20261105._hinge_fill; generalized only polygon corner count.
Original license: SOURCE_LICENSE.
"""
import numpy as np

def _seg_dist(p, a, b):
    ab = b - a
    t = 0.0 if ab @ ab < 1e-18 else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))

def _hinge_fill(qa, va, qb, vb, width, max_gap, seg=16):
    """membrane that joins tiles a and b at a hinge (qa, qb: their face corners (4, 2); va, vb: the hinge corners).
    Returns (polygon, gap between the two corners), or (None, gap) when the corners are farther apart than max_gap.

    Around the hinge corners the gap between the two tiles has an acute side and an obtuse side.  On the obtuse side
    nothing is added beyond the straight line between the two corners, so the membrane follows the tile corners.  On the
    acute side the gap is filled and the fill ends in a smooth curve that is tangent to both tile edges, so the gap
    keeps a rounded tip.  The curve is a cubic Bezier whose four points all lie on the two tile edges (from the point d
    along each edge, back half way towards the corner), so it always stays between the two edges - also when the
    corners do not quite touch and the edges are nearly parallel (the 2026-11-05 version used the point where the edge
    lines meet, which is extremely far away for nearly parallel edges and blew the preview up to millions of mm).
    When the corners do not quite touch, the same fill bridges the space between them.  The fill is lengthened along
    the edges until its narrowest part - from the obtuse-side line to the curve - is at least `width` (or until it
    reaches 45 % of an edge)."""
    a0, b0 = qa[va], qb[vb]
    gap = float(np.linalg.norm(b0 - a0))
    if gap > max_gap:
        return None, gap
    ua = [qa[(va - 1) % len(qa)] - a0, qa[(va + 1) % len(qa)] - a0]
    ub = [qb[(vb - 1) % len(qb)] - b0, qb[(vb + 1) % len(qb)] - b0]
    unit = lambda v: v / max(np.linalg.norm(v), 1e-12)
    # the acute side: the pair of edges (one per tile) with the smallest angle between them
    _, ea, eb = min(((np.arccos(np.clip(unit(x) @ unit(y), -1, 1)), x, y) for x in ua for y in ub), key=lambda z: z[0])
    theta = np.arccos(np.clip(unit(ea) @ unit(eb), -1, 1))
    cap = 0.45 * min(np.linalg.norm(ea), np.linalg.norm(eb))
    # for touching corners the curve's middle is 0.625 d cos(theta/2) from the corner
    d = min(1.6 * width / max(np.cos(theta / 2.0), 0.2), cap)
    u = np.linspace(0.0, 1.0, seg)[:, None]
    while True:
        p0, p1 = a0 + d * unit(ea), a0 + 0.5 * d * unit(ea)
        p2, p3 = b0 + 0.5 * d * unit(eb), b0 + d * unit(eb)
        curve = (1 - u) ** 3 * p0 + 3 * u * (1 - u) ** 2 * p1 + 3 * u ** 2 * (1 - u) * p2 + u ** 3 * p3
        narrowest = min(_seg_dist(c, a0, b0) for c in curve)
        if narrowest >= width or d >= cap:
            break
        d = min(d * 1.15, cap)
    pts = [a0] + list(curve) + ([b0] if gap > 1e-9 else [])
    return np.array(pts), gap
