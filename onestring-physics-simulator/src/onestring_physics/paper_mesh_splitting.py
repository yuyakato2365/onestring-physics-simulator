"""Sec. 4.5: complete, component-local grid cuts on an existing Omega.

Supplement Fig. 18 bounds *area* expansion by two (linear expansion sqrt(2)).
We retain the selected chart and measure its piecewise-affine area Jacobian.
Each disconnected part has an independent similarity scale: its normalized
area range is max(area Jacobian)/min(area Jacobian). No cut-band relief or
post-cut conformal remeshing is assumed. Axis choice and grid snapping are
discretization choices, not a claim to reproduce the authors' implementation.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def chart_geometry(parameterization):
    """UV and surface index spaces can differ at OptCuts seams."""
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    xyz = np.asarray(parameterization.surface_vertices_3d, float)
    sf = np.asarray(parameterization.surface_faces, int)
    uf = np.asarray(getattr(parameterization, 'uv_faces', sf), int)
    if sf.shape != uf.shape:
        raise ValueError('UV/surface face correspondence differs')
    source = np.full(len(uv), -1, int)
    for ids_uv, ids_surface in zip(uf, sf):
        for uid, sid in zip(ids_uv, ids_surface):
            if source[uid] >= 0 and source[uid] != sid:
                raise ValueError('A UV vertex corresponds to multiple surface vertices')
            source[uid] = sid
    aligned = np.zeros((len(uv), 3))
    valid = source >= 0
    aligned[valid] = xyz[source[valid]]
    return aligned, uf, source


def area_field(parameterization):
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    xyz, faces, _ = chart_geometry(parameterization)
    tri, surface = uv[faces], xyz[faces]
    a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    signed = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    surface_area2 = np.linalg.norm(np.cross(surface[:, 1]-surface[:, 0], surface[:, 2]-surface[:, 0]), axis=1)
    if not len(faces) or not np.isfinite(tri).all() or not np.isfinite(surface).all():
        raise ValueError('Split CSF requires a finite, nonempty triangle correspondence')
    if np.any(np.abs(signed) <= np.finfo(float).tiny) or np.any(surface_area2 <= np.finfo(float).tiny):
        raise ValueError('Split CSF is undefined on a degenerate surface/UV triangle')
    if np.any(signed > 0) and np.any(signed < 0):
        raise ValueError('Split CSF cannot certify a folded UV map')
    raw = surface_area2 / np.abs(signed)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite area Jacobian in Split CSF')
    normalized = raw / raw.min()
    vertex = np.ones(len(uv))
    np.maximum.at(vertex, faces.ravel(), np.repeat(normalized, 3))
    return vertex, normalized, raw


def gaussian_curvature(parameterization):
    """Signed angle defect / barycentric dual area; exclude chart boundaries."""
    xyz = np.asarray(parameterization.surface_vertices_3d, float)
    faces = np.asarray(parameterization.surface_faces, int)
    points = xyz[faces]
    area = .5*np.linalg.norm(np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]), axis=1)
    angles, dual = np.zeros(len(xyz)), np.zeros(len(xyz))
    np.add.at(dual, faces.ravel(), np.repeat(area/3, 3))
    for k in range(3):
        a, b = points[:, (k+1)%3]-points[:, k], points[:, (k+2)%3]-points[:, k]
        theta = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))
        np.add.at(angles, faces[:, k], theta)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    boundary = np.unique(unique[count == 1])
    out = np.divide(2*np.pi-angles, dual, out=np.full(len(xyz), -np.inf), where=dual > 0)
    out[boundary] = -np.inf
    _, uf, source = chart_geometry(parameterization)
    mapped = np.full(len(source), -np.inf)
    valid = source >= 0
    mapped[valid] = out[source[valid]]
    uv_edges = np.sort(np.concatenate([uf[:, [0, 1]], uf[:, [1, 2]], uf[:, [2, 0]]]), axis=1)
    unique, count = np.unique(uv_edges, axis=0, return_counts=True)
    mapped[np.unique(unique[count == 1])] = -np.inf
    return mapped


def _cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _intersection_area(triangle, polygon):
    """Convex clipping; measure which source triangles actually cover a quad."""
    points = list(triangle)
    orientation = 1 if sum(_cross(polygon[i], polygon[(i+1)%4]) for i in range(4)) > 0 else -1
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        output = []
        if not points:
            return 0.
        prev = points[-1]
        dp = orientation*_cross(b-a, prev-a)
        for point in points:
            dc = orientation*_cross(b-a, point-a)
            if (dc >= 0) != (dp >= 0):
                output.append(prev + dp/(dp-dc)*(point-prev))
            if dc >= 0:
                output.append(point)
            prev, dp = point, dc
        points = output
    if len(points) < 3:
        return 0.
    # Translate before summing to avoid cancellation for shifted charts.
    points = np.asarray(points)-points[0]
    return abs(sum(_cross(a, b) for a, b in zip(points, np.roll(points, -1, axis=0))))*.5


def _face_samples(vertices, faces, parameterization):
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    sf = np.asarray(getattr(parameterization, 'uv_faces', parameterization.surface_faces), int)
    triangles = uv[sf]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    lo, hi, samples = [], [], []
    _, _, raw = area_field(parameterization)
    for face in faces:
        polygon = vertices[face, :2]
        candidates = np.flatnonzero(np.all(upper >= polygon.min(axis=0), axis=1) & np.all(lower <= polygon.max(axis=0), axis=1))
        area = abs(sum(_cross(a-polygon[0], b-polygon[0]) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))))*.5
        ids = [int(i) for i in candidates if _intersection_area(triangles[i], polygon) > area*1e-12]
        if not ids:
            raise ValueError('M2D quad has no source triangles for component CSF evaluation')
        lo.append(float(raw[ids].min())); hi.append(float(raw[ids].max()))
        samples.append(np.unique(sf[ids]))
    return np.asarray(lo), np.asarray(hi), samples


def split_mesh(vertices, faces, domain, params=None):
    from .simple_split_panel_patch import _edge_components, _cut_once

    vertices, faces = np.asarray(vertices, float).copy(), np.asarray(faces, int).copy()
    parameterization = domain.parameterization
    lo, hi, samples = _face_samples(vertices, faces, parameterization)
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    curvature = gaussian_curvature(parameterization)
    angle = np.deg2rad(float(getattr(domain, 'reference_grid_rotation_degrees', 0.)))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Cuts and snapping take place in grid coordinates, including rotated BFF grids.
    vertices[:, :2] = vertices[:, :2] @ rotation
    uv_grid = uv @ rotation
    threshold = float(getattr(domain, 'csf_split_threshold', 2.))
    budget = max(0, int(getattr(domain, 'max_csf_splits', 64)))
    enabled = bool(getattr(domain, 'csf_split_enabled', True))
    records, rejected, pairs, blocked = [], [], [], set()
    def ratio(component):
        return float(hi[component].max()/lo[component].min())
    initial_components = _edge_components(faces)
    before = max(map(ratio, initial_components), default=1.)
    while enabled and len(records) < budget:
        components = _edge_components(faces)
        bad = [c for c in components if ratio(c) > threshold+1e-10 and tuple(c) not in blocked]
        if not bad:
            break
        component = max(bad, key=ratio)
        ids = np.unique(np.concatenate([samples[i] for i in component]))
        # Boundary points have no intrinsic Gaussian estimate on this chart.
        ids = ids[np.isfinite(curvature[ids])]
        ids = ids[np.argsort(-curvature[ids], kind='stable')]
        component_before = ratio(component)
        selected = None
        used = np.unique(faces[component])
        bounds = vertices[used, :2]
        tested = set()
        for peak in ids:
            point = uv_grid[peak]
            if not np.all((point > bounds.min(axis=0)) & (point < bounds.max(axis=0))):
                continue
            # The peak must belong to this part, not just its bounding box.
            quads = vertices[faces[component], :2]
            edges = np.roll(quads, -1, axis=1)-quads
            delta = point-quads
            cross = edges[:, :, 0]*delta[:, :, 1]-edges[:, :, 1]*delta[:, :, 0]
            if not np.any(np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1)):
                continue
            candidates = []
            for axis, coord in [('col', 0), ('row', 1)]:
                values = np.unique(np.round(bounds[:, coord], 12))[1:-1]
                if not len(values):
                    continue
                snapped = float(values[np.argmin(abs(values-point[coord]))])
                if (axis, snapped) in tested:
                    continue
                tested.add((axis, snapped))
                v2, subset, record = _cut_once(vertices, faces[component], (axis, float(point[coord])))
                if record is None:
                    continue
                children = _edge_components(subset)
                if len(children) != 2:
                    continue  # complete bisection, not a multiway/topologically ambiguous cut
                imbalance = abs(len(children[0])-len(children[1]))
                candidates.append((imbalance, axis, v2, subset, record))
            if candidates:
                selected = min(candidates, key=lambda x: (x[0], x[1]))
                break  # highest signed Gaussian curvature with an admissible grid cut
        if selected is None:
            blocked.add(tuple(component))
            rejected.append({'face_ids': component.tolist(), 'sigma': component_before, 'reason': 'no_complete_grid_cut_through_interior_curvature_sample'})
            continue
        _, _, vertices, subset, record = selected
        # Store seam identity using stable tile/local-edge indices, never proximity.
        owners = defaultdict(list)
        for fi in component:
            for k in range(4):
                edge = tuple(sorted((int(faces[fi, k]), int(faces[fi, (k+1)%4]))))
                owners[edge].append((int(fi), k))
        updated = faces.copy(); updated[component] = subset
        for entries in owners.values():
            if len(entries) != 2:
                continue
            (a, ea), (b, eb) = entries
            if set(updated[a, [ea, (ea+1)%4]]) != set(updated[b, [eb, (eb+1)%4]]):
                pairs.append([a, ea, b, eb])
        faces = updated
        children = _edge_components(subset)
        record.update(face_ids=component.tolist(), curvature_vertex_id=int(peak), gaussian_curvature=float(curvature[peak]),
                      sigma_before=component_before, sigma_after=[ratio(component[c]) for c in children],
                      component_id=int(component.min()))
        records.append(record)
    final = _edge_components(faces)
    values = [ratio(c) for c in final]
    unresolved = sum(v > threshold+1e-10 for v in values)
    exhausted = bool(enabled and unresolved and len(records) >= budget)
    after = max(values, default=1.)
    face_sigma = np.ones(len(faces))
    for component in final:
        face_sigma[component] = hi[component]/lo[component].min()
    domain.csf_before, domain.csf_after_split = before, after
    domain.split_lines = [('row' if r['axis'] == 'row' else 'col', r['snapped_value']) for r in records]
    domain.localized_split_segments = list(domain.split_lines)
    metrics = dict(
        csf_model='normalized per-component area Jacobian; Supplement Fig. 18',
        csf_split_exactness_label='paper-rule discretization on retained piecewise-affine chart',
        csf_split_applied=bool(records), csf_split_threshold=threshold, max_csf_splits=budget,
        max_csf_before_split=before, max_csf_after_split=after, number_of_splits=len(records),
        split_locations=[list(x) for x in domain.split_lines],
        raw_split_locations=[[r['axis'], r['requested_value']] for r in records],
        csf_split_duplicated_vertex_count=sum(r['duplicated_vertices'] for r in records),
        csf_split_budget_exhausted=exhausted, csf_split_unresolved_component_count=unresolved,
        csf_split_status='satisfied' if not unresolved else ('disabled' if not enabled else 'budget_exhausted' if exhausted else 'no_admissible_cut'),
        csf_split_step_analysis=records, csf_split_step_count=len(records),
        csf_split_step_analysis_model='exact source-triangle area Jacobian extrema over each retained quad component; independent component similarity normalization; no reparameterization',
        csf_split_component_sigma=values, csf_split_component_area_scale=[float(lo[c].min()) for c in final],
        csf_split_face_sigma=face_sigma.tolist(),
        csf_split_residual_high_face_count=int(np.count_nonzero(face_sigma > threshold+1e-10)),
        csf_split_additional_split_recommended_after_all=bool(unresolved),
        csf_split_conformal_anisotropy_max=parameterization.metrics.get('split_conformal_anisotropy_max'),
        csf_split_area_bound_requires_conformal_map=True,
        csf_split_residual_max_after_all=after,
        csf_split_rejected_components=rejected,
        split_boundary_pairs=pairs,
        m2d_quad_count_after_csf_split=len(faces), m2d_connected_component_count_after_csf_split=len(final),
        m2d_largest_component_quad_count_after_csf_split=max(map(len, final), default=0),
        m2d_smallest_component_quad_count_after_csf_split=min(map(len, final), default=0),
        csf_split_direction_rule='highest signed area-normalized Gaussian curvature; nearest grid line; most balanced admissible axis',
        csf_split_grid_rotation_degrees=float(np.rad2deg(angle)),
    )
    vertices[:, :2] = vertices[:, :2] @ rotation.T
    domain.paper_split_diagnostics = dict(metrics, split_count=len(records), status=metrics['csf_split_status'])
    parameterization.metrics['split_diagnostics'] = dict(domain.paper_split_diagnostics)
    return vertices, faces, records, metrics
