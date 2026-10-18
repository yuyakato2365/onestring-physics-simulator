"""Independent, read-only Split research geometry and CSF metrics.

All UVs produced by this package are diagnostic and NEVER mutate the parent
Omega parameterization or the manufacturing grid. These are research metrics,
NOT the paper authors' undisclosed discrete implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

from onestring_physics.reference_bff import (
    run_official_bff, validate_reference_mesh,
)


@dataclass(frozen=True)
class ExperimentConfig:
    grid_n: int = 14
    corner_policy: str = "lobe_tips"
    grid_policy: str = "fit_center_min_side"
    curvature_rings: int = 3
    metric: str = "free_bff_weighted_tile_area"
    threshold: float = 2.0
    max_splits: int = 8


def load_triangle_obj(path):
    """Parse triangular v/f OBJ; fail rather than silently alter input."""
    vertices, faces = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tokens = line.strip().split()
        if not tokens or tokens[0] == "#":
            continue
        if tokens[0] == "v":
            if len(tokens) < 4:
                raise ValueError("Malformed OBJ vertex")
            vertices.append([float(x) for x in tokens[1:4]])
        elif tokens[0] == "f":
            indices = [int(s.split("/")[0]) - 1 for s in tokens[1:]]
            if len(indices) != 3:
                raise ValueError("Only triangle OBJ files are supported")
            faces.append(indices)
    xyz, tri = np.asarray(vertices, float), np.asarray(faces, int)
    validate_reference_mesh(xyz, tri)
    return xyz, tri


def mesh_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed_twice_area(p):
    return float(np.cross(p[1] - p[0], p[2] - p[0]))


def area_jacobians(xyz, uv, faces):
    """J = dA_3D/dA_UV for each triangle; orientation / degeneracy checked."""
    f = np.asarray(faces, int)
    x, u = np.asarray(xyz, float)[f], np.asarray(uv, float)[f]
    a3 = np.linalg.norm(np.cross(x[:, 1]-x[:, 0], x[:, 2]-x[:, 0]), axis=1)
    a2 = np.cross(u[:, 1]-u[:, 0], u[:, 2]-u[:, 0])
    if not len(a2) or not np.isfinite(a2).all() or not np.isfinite(a3).all():
        raise ValueError("Empty or nonfinite Jacobian samples")
    if np.any(a3 <= 1e-12 * max(float(np.median(a3)), 1e-20)):
        raise ValueError("Degenerate source triangles")
    if np.any(abs(a2) <= 1e-12 * max(float(np.median(abs(a2))), 1e-20)):
        raise ValueError("Degenerate chart triangles")
    if np.any(a2 < 0) and np.any(a2 > 0):
        raise ValueError("Folded UV chart")
    j = a3 / abs(a2)
    if not np.isfinite(j).all() or np.any(j <= 0):
        raise ValueError("Invalid area Jacobian")
    return j


def scale_span(values, *, units="area_ratio"):
    values = np.asarray(values, float)
    if not len(values) or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Incomplete or invalid expansion samples")
    ratio = float(values.max() / values.min())
    if units == "area_ratio":
        return ratio
    if units == "linear_ratio":
        return float(np.sqrt(ratio))
    raise ValueError("Unknown metric units: " + units)


def boundary_loop(xyz, faces):
    return np.asarray(validate_reference_mesh(xyz, faces)["boundary_loops"][0], int)


def lobe_tip_corners(xyz, faces):
    """Preset hypothesis: select 4 high turning-angle boundary peaks.

    PCA projects the source boundary to a plane. Circular 3-tap smoothing
    suppresses local turning-angle noise. Nonmaximum suppression uses a
    preregistered min boundary separation = perimeter/8, without Split data.
    """
    loop = boundary_loop(xyz, faces)
    p = xyz[loop]
    center = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p-center, full_matrices=False)
    uv = (p-center) @ vt[:2].T
    edge = np.roll(uv, -1, axis=0)-uv
    lengths = np.linalg.norm(edge, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Boundary projection contains collapsed edges")
    tang = edge / lengths[:, None]
    turning = np.abs(np.arctan2(
        tang[:, 0]*np.roll(tang[:, 1], 1)-tang[:, 1]*np.roll(tang[:, 0], 1),
        np.sum(tang*np.roll(tang, 1, axis=0), axis=1)))
    turning = (np.roll(turning, 1)+turning+np.roll(turning, -1))/3
    arc = np.r_[0., np.cumsum(lengths)]
    total = float(arc[-1])
    ranked = np.argsort(-turning, kind="stable")
    chosen = []
    for idx in ranked:
        distance = [min(abs(arc[idx]-arc[j]), total-abs(arc[idx]-arc[j]))
                    for j in chosen]
        if not distance or min(distance) >= total/8:
            chosen.append(int(idx))
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise ValueError("Cannot select four distinct boundary lobe tips")
    chosen.sort()
    return loop[chosen].astype(int).tolist()


def grid_rectangles(uv, n, policy="fit_center_min_side"):
    """Strict squares only. Each rectangle is [xmin,ymin,xmax,ymax]."""
    u = np.asarray(uv, float)
    lo, hi = np.min(u, axis=0), np.max(u, axis=0)
    dims = hi-lo
    if n < 2 or np.min(dims) <= 0:
        raise ValueError("Invalid grid")
    pitch = (np.min(dims) if policy == "fit_center_min_side"
             else np.max(dims)) / n
    if policy not in {"fit_center_min_side", "original_origin_max_side"}:
        raise ValueError("Unknown grid policy")
    start = (lo+hi - n*pitch)/2 if policy == "fit_center_min_side" else lo
    squares = []
    for iy in range(n):
        for ix in range(n):
            qlo = start + pitch * np.asarray([ix, iy])
            qhi = qlo + pitch
            if np.all(qlo >= lo-1e-9*dims) and np.all(qhi <= hi+1e-9*dims):
                squares.append((ix, iy, *qlo, *qhi))
    return squares, dict(pitch=float(pitch), offset=start.tolist(),
                         rect_min=lo.tolist(), rect_max=hi.tolist(),
                         eligible_tile_count=len(squares))


def clip_poly(poly, axis, bound, keep_greater):
    """Sutherland-Hodgman; vertex rows contain parent UV followed by XYZ."""
    out = []
    for a, b in zip(poly, poly[1:]+poly[:1]):
        da, db = a[axis]-bound, b[axis]-bound
        inside_a = da >= -1e-12 if keep_greater else da <= 1e-12
        inside_b = db >= -1e-12 if keep_greater else db <= 1e-12
        if inside_a != inside_b:
            t = da/(da-db)
            out.append(a+t*(b-a))
        if inside_b:
            out.append(b)
    return out


def clip_source_to_box(xyz, uv, faces, box):
    """Clip curved triangles in parent UV, interpolating 3D on edge crossings.

    A new independent disk mesh is built only for a diagnostic BFF call.
    The manufacturer-facing source UV / tile identifiers remain untouched.
    """
    xmin, ymin, xmax, ymax = [float(x) for x in box]
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Empty split box")
    points, triangles, keys = [], [], {}
    span = max(float(np.ptp(uv)), 1.)
    def key_for(v):
        # Quantization is relative to UV size; validate_reference_mesh catches
        # seam cracks or nonmanifold mistakes instead of accepting them.
        return tuple(np.round(v / span, 10))
    for face in np.asarray(faces, int):
        poly = [np.r_[uv[i], xyz[i]] for i in face]
        for axis, bound, greater in (
            (0, xmin, True), (0, xmax, False),
            (1, ymin, True), (1, ymax, False)
        ):
            poly = clip_poly(poly, axis, bound, greater)
            if len(poly) < 3:
                break
        if len(poly) < 3:
            continue
        idx = []
        for vertex in poly:
            k = key_for(vertex)
            if k not in keys:
                keys[k] = len(points)
                points.append(vertex)
            idx.append(keys[k])
        for j in range(1, len(idx)-1):
            tri = [idx[0], idx[j], idx[j+1]]
            if len(set(tri)) == 3:
                triangles.append(tri)
    if not triangles:
        raise ValueError("No triangles inside cut component")
    p, f = np.asarray(points, float), np.asarray(triangles, int)
    # Discard numerical zero-area triangles along clipping boundaries only.
    a = np.linalg.norm(np.cross(p[f[:, 1], 2:]-p[f[:, 0], 2:],
                                p[f[:, 2], 2:]-p[f[:, 0], 2:]), axis=1)
    f = f[a > 1e-12*max(float(np.median(a)), 1e-20)]
    if not len(f):
        raise ValueError("No valid triangles after clipping")
    used = np.unique(f)
    remap = np.full(len(p), -1, int)
    remap[used] = np.arange(len(used))
    p, f = p[used], remap[f]
    validate_reference_mesh(p[:, 2:], f)
    return p[:, 2:], p[:, :2], f


def poly_rectangle_area(triangle_uv, bounds):
    x0, y0, x1, y1 = bounds
    poly = [np.asarray(v, float) for v in triangle_uv]
    for axis, val, greater in ((0, x0, True), (0, x1, False),
                               (1, y0, True), (1, y1, False)):
        poly = clip_poly(poly, axis, val, greater)
        if len(poly) < 3:
            return 0.
    p = np.asarray(poly)
    return float(abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1) -
                            p[:, 1]*np.roll(p[:, 0], -1))) / 2)


def tile_jacobians(parent_uv, diagnostic_j, faces, tiles,
                   *, coverage_tolerance=0.005):
    """Explicit metric: each tile's overlap-area-weighted mean triangle J.

    The tile's *original Omega UV area* supplies overlap weights, whereas
    each J comes from a separately flattened chart for that component.
    It is not max/min *within* each tile, not author-certified, and is
    intentionally identified with a unique metric name.
    """
    u = np.asarray(parent_uv, float)
    f = np.asarray(faces, int)
    j = np.asarray(diagnostic_j, float)
    if len(j) != len(f):
        raise ValueError("Jacobian/face correspondence missing")
    values, coverage = [], []
    tri = u[f]
    tri_min, tri_max = tri.min(axis=1), tri.max(axis=1)
    for row in tiles:
        bounds = row[2:]
        x0,y0,x1,y1 = bounds
        hits = np.flatnonzero(
            (tri_max[:,0] >= x0) & (tri_min[:,0] <= x1) &
            (tri_max[:,1] >= y0) & (tri_min[:,1] <= y1))
        areas = np.asarray([poly_rectangle_area(tri[t], bounds) for t in hits])
        positive = areas > 0
        measured = float(np.sum(areas))
        tile_area = float((x1-x0)*(y1-y0))
        if measured < tile_area*(1-coverage_tolerance):
            raise ValueError("Unmeasured tile: area coverage < 99.5%")
        if measured > tile_area*(1+coverage_tolerance):
            raise ValueError("Chart triangle overlap / double coverage")
        values.append(float(np.dot(areas[positive], j[hits][positive])/measured))
        coverage.append(measured/tile_area)
    return np.asarray(values, float), np.asarray(coverage, float)


def curvature_field(xyz, faces, rings=3):
    xyz, f = np.asarray(xyz, float), np.asarray(faces, int)
    n = len(xyz)
    theta, area = np.zeros(n), np.zeros(n)
    neighbors = [set() for _ in range(n)]
    p = xyz[f]
    tri_area = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)/2
    for k in range(3):
        a, b = p[:, (k+1)%3]-p[:, k], p[:, (k+2)%3]-p[:, k]
        angles = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                            np.sum(a*b, axis=1))
        np.add.at(theta, f[:, k], angles)
        np.add.at(area, f[:, k], tri_area/3)
        for ids in f:
            neighbors[int(ids[k])].update((int(ids[(k+1)%3]), int(ids[(k+2)%3])))
    boundary = set(map(int, boundary_loop(xyz, f)))
    values = np.divide(2*np.pi-theta, area,
                       out=np.full(n, np.nan), where=area>0)
    values[list(boundary)] = np.nan
    if rings < 0:
        raise ValueError("Negative curvature smoothing rings")
    for _ in range(rings):
        values = np.asarray([
            float(np.mean([values[j] for j in (i, *sorted(neighbors[i]))
                           if np.isfinite(values[j])]))
            if any(np.isfinite(values[j]) for j in (i, *sorted(neighbors[i])))
            else np.nan for i in range(n)])
        values[list(boundary)] = np.nan
    return values


def nearest_curvature_grid_vertex(curvature, parent_uv, tiles, component_tiles):
    """Research cut selector; nearest valid source sample in parent UV."""
    from scipy.spatial import cKDTree
    chosen = [tiles[i] for i in component_tiles]
    xs = sorted({float(x) for t in chosen for x in (t[2], t[4])})
    ys = sorted({float(y) for t in chosen for y in (t[3], t[5])})
    interior = np.asarray([(x,y) for x in xs[1:-1] for y in ys[1:-1]])
    valid = np.isfinite(curvature)
    if not len(interior) or not np.any(valid):
        raise ValueError("No interior vertex / finite curvature")
    distances, indices = cKDTree(np.asarray(parent_uv)[valid]).query(interior)
    scores = curvature[valid][indices]
    winner = int(np.argmax(scores))
    return interior[winner], float(scores[winner]), float(distances[winner])


def split_tile_ids(tiles, tile_ids, axis, cut, tol=1e-8):
    negative, positive = [], []
    for i in tile_ids:
        t = tiles[i]
        lo, hi = (t[2],t[4]) if axis == "col" else (t[3],t[5])
        if hi <= cut + tol:
            negative.append(i)
        elif lo >= cut - tol:
            positive.append(i)
        else:
            raise ValueError("Candidate cut crosses a manufacturing tile")
    if not negative or not positive:
        raise ValueError("Empty child tile group")
    return negative, positive
