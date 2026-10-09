"""Sec. 4.5: complete, component-local grid cuts on an existing Omega.

Supplement Fig. 18 bounds *area* expansion by two (linear expansion sqrt(2)).
We retain the selected chart and measure its piecewise-affine area Jacobian.
CSF is chart-global, fixed by S-to-Omega; cutting does not change its value.
Cuts isolate grid panels spanning above- and below-threshold samples. No
post-cut conformal remeshing or component-local rescaling is assumed. Axis choice and grid snapping are
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
    tiny = np.finfo(float).tiny
    uv_area = np.abs(signed)
    # A degenerate SOURCE triangle is an input defect and still refuses the run.
    xyz_scale = float(np.median(surface_area2)) if len(surface_area2) else 0.
    if np.any(surface_area2 <= max(tiny, 1e-12*xyz_scale)):
        raise ValueError('Split CSF is undefined: the source surface mesh has a degenerate triangle')
    # Preserve unknown values as NaN, while allowing the usable chart to reach
    # grid cropping. These faces must prevent certification, not abort every
    # chart before we even know which part the grid uses.
    uv_scale = float(np.median(uv_area[uv_area > tiny])) if np.any(uv_area > tiny) else 0.
    degenerate = uv_area <= max(tiny, 1e-12*uv_scale)
    usable = ~degenerate
    if not np.any(usable):
        raise ValueError(
            'Split CSF is undefined: every UV triangle is collapsed or numerically degenerate')
    if np.any(signed[usable] > 0) and np.any(signed[usable] < 0):
        raise ValueError('Split CSF cannot certify a folded UV map')
    raw = np.full(len(faces), np.nan)
    raw[usable] = surface_area2[usable]/uv_area[usable]
    if not np.isfinite(raw[usable]).all():
        raise ValueError('Nonfinite area Jacobian in Split CSF')
    normalized = raw/np.nanmin(raw)
    vertex = np.ones(len(uv))
    np.maximum.at(vertex, faces[usable].ravel(), np.repeat(normalized[usable], 3))
    q = [0., .1, 1., 5., 25., 50., 75., 95., 99., 99.9, 100.]
    finite_raw = raw[np.isfinite(raw)]
    uv_ok = uv_area[usable]
    parameterization.metrics.update(
        split_csf_raw_percentiles={f'p{x:g}': float(v) for x, v in zip(q, np.percentile(finite_raw, q))},
        split_csf_uv_area_percentiles={f'p{x:g}': float(v) for x, v in zip(q, np.percentile(uv_ok, q))},
        split_csf_xyz_area_percentiles={f'p{x:g}': float(v) for x, v in zip(q, np.percentile(surface_area2, q))},
        split_csf_raw_below_p1_count=int((finite_raw < np.percentile(finite_raw, 1.)).sum()),
        split_csf_normalized_if_anchor_p1=float(np.nanmax(raw)/np.percentile(finite_raw, 1.)),
        split_csf_normalized_if_anchor_median=float(np.nanmax(raw)/np.median(finite_raw)),
        split_csf_anchor_raw_min=float(np.nanmin(raw)),
        split_csf_degenerate_uv_face_count=int(degenerate.sum()),
        split_csf_degenerate_uv_face_fraction=float(degenerate.mean()),
        split_csf_degenerate_uv_face_ids=np.flatnonzero(degenerate).tolist(),
        split_csf_degenerate_uv_policy='NaN in source field; evaluate usable chart only; disable bound certification',
        split_csf_field_complete=not bool(np.any(degenerate)),
    )
    pr = {f'p{x:g}': float(v) for x, v in zip(q, np.percentile(finite_raw, q))}
    pu = {f'p{x:g}': float(v) for x, v in zip(q, np.percentile(uv_ok, q))}
    print(
        '[AREA-FIELD] faces=%d degenerate_uv=%d (%.3f%%)\n'
        '  raw      p0=%.4g p1=%.4g p5=%.4g p50=%.4g p95=%.4g p99=%.4g p100=%.4g\n'
        '  uv_area  p0=%.4g p1=%.4g p50=%.4g p100=%.4g\n'
        '  lambda_max  anchor=min:%.4g  anchor=p1:%.4g  anchor=median:%.4g'
        % (len(faces), int(degenerate.sum()), 100*degenerate.mean(),
           pr['p0'], pr['p1'], pr['p5'], pr['p50'], pr['p95'], pr['p99'], pr['p100'],
           pu['p0'], pu['p1'], pu['p50'], pu['p100'],
           float(np.nanmax(raw)/np.nanmin(raw)),
           float(np.nanmax(raw)/np.percentile(finite_raw, 1.)),
           float(np.nanmax(raw)/np.median(finite_raw))),
        flush=True)
    return vertex, normalized, raw

def gaussian_curvature(parameterization):
    """Intrinsic signed curvature; exclude physical, not artificial UV boundaries."""
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
    # OptCuts may duplicate an interior surface vertex along a chart seam.
    # That changes the UV boundary, not its intrinsic Gaussian curvature.
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


def _face_samples(vertices, faces, parameterization, threshold=2.):
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    sf = np.asarray(getattr(parameterization, 'uv_faces', parameterization.surface_faces), int)
    triangles = uv[sf]
    lower, upper = triangles.min(axis=1), triangles.max(axis=1)
    lo, hi, samples = [], [], []
    _, _, raw = area_field(parameterization)
    tiny = np.finfo(float).tiny
    # The active floor stays at 1e-12 of the quad area, i.e. "strictly positive overlap".
    # The sweep only reports what other floors would measure; it changes nothing.
    floors = (0., 1e-6, 1e-4, 1e-3, 1e-2, 5e-2)
    sweep = {f: ([], []) for f in floors}
    lo_cover, hi_cover = [], []
    for face in faces:
        polygon = vertices[face, :2]
        candidates = np.flatnonzero(np.all(upper >= polygon.min(axis=0), axis=1) & np.all(lower <= polygon.max(axis=0), axis=1))
        area = abs(sum(_cross(a-polygon[0], b-polygon[0]) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))))*.5
        overlap = [(int(i), _intersection_area(triangles[i], polygon)) for i in candidates if np.isfinite(raw[i])]
        overlap = [(i, c) for i, c in overlap if c > area*1e-12]
        if not overlap:
            raise ValueError('M2D quad has no non-degenerate source triangle for component CSF evaluation')
        ids = np.asarray([i for i, _ in overlap], int)
        cover = np.asarray([c for _, c in overlap], float)/max(area, tiny)
        values = raw[ids]
        lo.append(float(values.min())); hi.append(float(values.max()))
        samples.append(np.unique(sf[ids]))
        lo_cover.append(float(cover[int(values.argmin())])); hi_cover.append(float(cover[int(values.argmax())]))
        for f in floors:
            keep = cover >= f
            if not np.any(keep):
                # Never drop every source triangle: fall back to the widest-covering one.
                keep = cover >= cover.max()
            sweep[f][0].append(float(values[keep].min())); sweep[f][1].append(float(values[keep].max()))
    lo, hi = np.asarray(lo), np.asarray(hi)
    lo_cover, hi_cover = np.asarray(lo_cover), np.asarray(hi_cover)
    quad_sigma = hi/lo
    q = [0., 50., 90., 99., 100.]
    report = {}
    for f in floors:
        fl, fh = np.asarray(sweep[f][0]), np.asarray(sweep[f][1])
        report[f] = (float(fh.max()/fl.min()), int(np.count_nonzero(fh/fl > threshold+1e-10)), float((fh/fl).max()))
    parameterization.metrics.update(
        split_quad_count=int(len(faces)),
        split_quad_sigma_percentiles={f'p{x:g}': float(v) for x, v in zip(q, np.percentile(quad_sigma, q))},
        split_quad_irreducible_count=int(np.count_nonzero(quad_sigma > threshold+1e-10)),
        split_quad_lo_set_by_sliver_count=int(np.count_nonzero(lo_cover < .01)),
        split_quad_hi_set_by_sliver_count=int(np.count_nonzero(hi_cover < .01)),
        split_quad_coverage_floor_active=1e-12,
        split_quad_coverage_floor_sweep={f'{f:g}': {'chart_sigma': report[f][0], 'quads_over_threshold': report[f][1],
                                                    'worst_quad_sigma': report[f][2]} for f in floors},
    )
    print('[FACE-SAMPLES] quads=%d threshold=%.4g active_coverage_floor=1e-12 (unchanged)' % (len(faces), threshold), flush=True)
    print('  quad sigma hi/lo  ' + '  '.join('p%g=%.4g' % (x, v) for x, v in zip(q, np.percentile(quad_sigma, q)))
          + '   irreducible(>thr)=%d/%d' % (int(np.count_nonzero(quad_sigma > threshold+1e-10)), len(faces)), flush=True)
    print('  lo set by a triangle covering <1%% of the quad: %d quads  (<0.01%%: %d)'
          % (int(np.count_nonzero(lo_cover < .01)), int(np.count_nonzero(lo_cover < 1e-4))), flush=True)
    print('  hi set by a triangle covering <1%% of the quad: %d quads  (<0.01%%: %d)'
          % (int(np.count_nonzero(hi_cover < .01)), int(np.count_nonzero(hi_cover < 1e-4))), flush=True)
    print('  coverage-floor sweep (report only, nothing changed):', flush=True)
    for f in floors:
        s, n, w = report[f]
        print('    floor=%-8g chart_sigma=%-8.4g worst_quad_sigma=%-8.4g quads_over_threshold=%d' % (f, s, w, n), flush=True)
    return lo, hi, samples



def _parameterization_from_optcuts_result(template, result):
    """Clone the live parameterization shell around another official OptCuts UV."""
    import copy
    out=copy.copy(template)
    out.surface_vertices_3d=np.asarray(result.surface_vertices_3d,float)
    out.surface_faces=np.asarray(result.surface_faces,int)
    out.uv_vertices_2d=np.asarray(result.uv_vertices_2d,float)
    out.uv_faces=np.asarray(result.uv_faces,int)
    loop=[int(v) for v in result.boundary_loops[0]]
    out.omega_boundary=out.uv_vertices_2d[loop+[loop[0]]]
    out.metrics=dict(getattr(template,'metrics',{}) or {})
    out.metrics.update(result.metrics)
    out.metrics['boundary_loop']=loop
    return out


def _split_count_for_parameterization(vertices, faces, domain, params, parameterization):
    """Evaluate the real Split algorithm without mutating the caller's domain."""
    import copy
    trial=copy.copy(domain)
    trial.parameterization=parameterization
    # Avoid recursively evaluating OptCuts candidates inside the trial.
    if hasattr(trial.parameterization,'_onestring_optcuts_candidates'):
        trial.parameterization=copy.copy(trial.parameterization)
        try:
            delattr(trial.parameterization,'_onestring_optcuts_candidates')
        except AttributeError:
            pass
    _,_,records,metrics=_split_mesh_single(vertices,faces,trial,params)
    return len(records),metrics


def _select_optcuts_by_split_count(vertices, faces, domain, params):
    parameterization=domain.parameterization
    candidates=getattr(parameterization,'_onestring_optcuts_candidates',None)
    if not candidates or len(candidates)<2:
        return parameterization,None
    from .optcuts_quality_audit import quality, selection_score
    evaluated=[]
    for index,result in enumerate(candidates):
        p=_parameterization_from_optcuts_result(parameterization,result)
        count,split_metrics=_split_count_for_parameterization(vertices,faces,domain,params,p)
        q=quality(result)
        evaluated.append((selection_score(q,count),index,p,count,q,split_metrics))
        print('[OPTCUTS-SPLIT-CANDIDATE] index=%d initial_cut=%s splits=%d tail=%.6g csf_over2=%.6g csf_p95=%.6g csf_max=%.6g'
              % (index,result.metrics.get('optcuts_initial_cut_option'),count,q['csf_tail_penalty'],
                 q['csf_over_2_area_fraction'],q['csf_p95'],q['csf_max']),flush=True)
    best=min(evaluated,key=lambda x:x[0])
    _,index,p,count,q,split_metrics=best
    p.metrics.update(optcuts_split_count_selection_pending=False,
                     optcuts_split_count_selection_status='selected',
                     optcuts_split_count_selected_candidate=int(index),
                     optcuts_split_count_selected=int(count),
                     optcuts_split_count_candidates=[{'candidate':int(x[1]),'split_count':int(x[3]),
                         'score':list(x[0]),'quality':x[4]} for x in evaluated],
                     optcuts_csf_aware_candidate_selection=True,
                     optcuts_official_objective_modified=False)
    print('[OPTCUTS-SPLIT-SELECT] candidate=%d splits=%d' % (index,count),flush=True)
    return p,evaluated

def split_mesh(vertices, faces, domain, params=None):
    selected,evaluated=_select_optcuts_by_split_count(vertices,faces,domain,params)
    if evaluated is not None:
        domain.parameterization=selected
    return _split_mesh_single(vertices,faces,domain,params)


def _split_mesh_single(vertices, faces, domain, params=None):
    from .simple_split_panel_patch import _edge_components
    from .final_split_panel_pass import _complete_cut_once

    vertices, faces = np.asarray(vertices, float).copy(), np.asarray(faces, int).copy()
    parameterization = domain.parameterization
    threshold = float(getattr(domain, 'csf_split_threshold', 2.))
    lo, hi, samples = _face_samples(vertices, faces, parameterization, threshold)
    # Source-chart CSF is fixed; cuts cannot reduce its absolute maximum.
    # Use one chart-wide anchor, never renormalize independently per component.
    anchor = float(parameterization.metrics['split_csf_anchor_raw_min'])
    absolute_hi = hi / anchor
    absolute_lo = lo / anchor
    irreducible = absolute_hi.copy()
    uv = np.asarray(parameterization.uv_vertices_2d, float)
    curvature = gaussian_curvature(parameterization)
    angle = np.deg2rad(float(getattr(domain, 'reference_grid_rotation_degrees', 0.)))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Cuts and snapping take place in grid coordinates, including rotated BFF grids.
    vertices[:, :2] = vertices[:, :2] @ rotation
    uv_grid = uv @ rotation
    budget = max(0, int(getattr(domain, 'max_csf_splits', 64)))
    enabled = bool(getattr(domain, 'csf_split_enabled', True))
    records, rejected, pairs, blocked = [], [], [], set()
    def ratio(component):
        return float(absolute_hi[component].max())
    def mixed_edges(component):
        # Count connected quad edges straddling the fixed chart-global limit.
        edge_owner = defaultdict(list)
        for fi in component:
            for k in range(4):
                edge = tuple(sorted((int(faces[fi, k]), int(faces[fi, (k+1)%4]))))
                edge_owner[edge].append(int(fi))
        return sum(1 for owners in edge_owner.values() if len(owners) == 2
                   and (absolute_hi[owners[0]] > threshold + 1e-10)
                   != (absolute_hi[owners[1]] > threshold + 1e-10))
    initial_components = _edge_components(faces)
    before = max(map(ratio, initial_components), default=1.)
    while enabled and len(records) < budget:
        components = _edge_components(faces)
        # Only cut components that still connect high-CSF and low-CSF panels.
        # High-only components remain above the bound, but cutting them again
        # cannot change the fixed S-to-Omega CSF.
        bad = [c for c in components if mixed_edges(c) > 0 and tuple(c) not in blocked]
        if not bad:
            break
        # Round only the ordering key: rotating Omega can introduce last-bit
        # differences in equivalent area ratios. Keep all reported CSFs raw.
        component = max(bad, key=lambda c: (mixed_edges(c), round(ratio(c), 10)))
        ids = np.unique(np.concatenate([samples[i] for i in component]))
        # Intrinsic source curvature remains valid at artificial split seams.
        ids = ids[np.isfinite(curvature[ids])]
        ids = ids[np.argsort(-curvature[ids], kind='stable')]
        component_before = ratio(component)
        selected = None
        used = np.unique(faces[component])
        bounds = vertices[used, :2]
        tested = set()
        complete_candidate_seen = False
        for peak in ids:
            point = uv_grid[peak]
            # The peak must belong to this part, not just its bounding box.
            quads = vertices[faces[component], :2]
            edges = np.roll(quads, -1, axis=1)-quads
            delta = point-quads
            cross = edges[:, :, 0]*delta[:, :, 1]-edges[:, :, 1]*delta[:, :, 0]
            if not np.any(np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1)):
                continue
            candidates = []
            for axis, coord in [('col', 0), ('row', 1)]:
                # A peak on a previous cut cannot justify moving the same cut
                # inward by another whole grid row. The other axis may work.
                if not bounds[:, coord].min()+1e-12 < point[coord] < bounds[:, coord].max()-1e-12:
                    continue
                values = np.unique(np.round(bounds[:, coord], 12))[1:-1]
                if not len(values):
                    continue
                snapped = float(values[np.argmin(abs(values-point[coord]))])
                if (axis, snapped) in tested:
                    continue
                tested.add((axis, snapped))
                v2, subset, record = _complete_cut_once(vertices, faces[component], axis, float(point[coord]))
                if record is None:
                    continue
                children = _edge_components(subset)
                if len(children) != 2:
                    continue  # complete bisection, not a multiway/topologically ambiguous cut
                imbalance = abs(len(children[0])-len(children[1]))
                child_sigmas = [ratio(component[c]) for c in children]
                complete_candidate_seen = True
                # CSF itself must not improve when only connectivity changes.
                # Prefer cuts that separate violating panels from valid ones.
                # Paper leaves the choice between two grid directions open.
                # Compare their measured residuals before using balance as a tie-break.
                # Evaluate separation on the proposed subset, not old connectivity.
                old_faces = faces
                faces = faces.copy()
                faces[component] = subset
                child_mixed = sum(mixed_edges(component[c]) for c in children)
                faces = old_faces
                if child_mixed >= mixed_edges(component):
                    continue
                score = (child_mixed, imbalance, axis)
                candidates.append((score, v2, subset, record))
            if candidates:
                selected = min(candidates, key=lambda x: x[0])
                break  # highest signed Gaussian curvature with an admissible grid cut
        if selected is None:
            blocked.add(tuple(component))
            rejected.append({'face_ids': component.tolist(), 'sigma': component_before,
                             'reason': 'no_mixed_edge_reduction' if complete_candidate_seen else 'no_complete_grid_cut_near_source_curvature_sample'})
            continue
        _, vertices, subset, record = selected
        # Store seam identity using stable tile/local-edge indices, never proximity.
        owners = defaultdict(list)
        for fi in component:
            for k in range(4):
                edge = tuple(sorted((int(faces[fi, k]), int(faces[fi, (k+1)%4]))))
                owners[edge].append((int(fi), k))
        updated = faces.copy(); updated[component] = subset
        step_pairs = []
        for entries in owners.values():
            if len(entries) != 2:
                continue
            (a, ea), (b, eb) = entries
            if set(updated[a, [ea, (ea+1)%4]]) != set(updated[b, [eb, (eb+1)%4]]):
                pairs.append([a, ea, b, eb])
                step_pairs.append([a, ea, b, eb])
        faces = updated
        children = _edge_components(subset)
        record.update(face_ids=component.tolist(), curvature_vertex_id=int(peak), gaussian_curvature=float(curvature[peak]),
                      sigma_before=component_before, sigma_after=[ratio(component[c]) for c in children],
                      worst_sigma_reduction=component_before-max(ratio(component[c]) for c in children),
                      child_face_ids=[component[c].tolist() for c in children],
                      split_boundary_pairs=step_pairs,
                      grid_snap_distance=abs(record['snapped_value']-record['requested_value']),
                      component_id=int(component.min()))
        records.append(record)
    final = _edge_components(faces)
    invalid_uv_count = int(parameterization.metrics.get('split_csf_degenerate_uv_face_count', 0))
    values = [ratio(c) for c in final]
    unresolved = sum(v > threshold+1e-10 for v in values)
    unresolved_mixed = sum(mixed_edges(c) for c in final)
    exhausted = bool(enabled and unresolved_mixed and len(records) >= budget)
    after = max(values, default=1.)
    face_sigma = np.ones(len(faces))
    for component in final:
        face_sigma[component] = absolute_hi[component]
    domain.csf_before, domain.csf_after_split = before, after
    domain.split_lines = [('row' if r['axis'] == 'row' else 'col', r['snapped_value']) for r in records]
    domain.localized_split_segments = list(domain.split_lines)
    metrics = dict(
        csf_model='fixed chart-global area Jacobian from S-to-Omega (not per-component range)',
        csf_split_exactness_label='complete hierarchical grid cuts; retained-chart CSF estimator, not Fig. 6 numerical reproduction',
        csf_split_applied=bool(records), csf_split_threshold=threshold, max_csf_splits=budget,
        max_csf_before_split=before, max_csf_after_split=after, number_of_splits=len(records),
        split_locations=[list(x) for x in domain.split_lines],
        raw_split_locations=[[r['axis'], r['requested_value']] for r in records],
        csf_split_duplicated_vertex_count=sum(r['duplicated_vertices'] for r in records),
        csf_split_budget_exhausted=exhausted, csf_split_unresolved_component_count=unresolved, csf_split_unresolved_mixed_edges=unresolved_mixed,
        csf_split_status='uncertified_uv_degeneracy' if invalid_uv_count else 'satisfied' if not unresolved else ('isolated_high_csf_not_certified' if not unresolved_mixed else 'disabled' if not enabled else 'budget_exhausted' if exhausted else 'mixed_boundary_remaining'),
        csf_split_step_analysis=records, csf_split_step_count=len(records),
        csf_split_initial_components=[{'face_ids': c.tolist(), 'sigma': ratio(c)} for c in initial_components],
        csf_split_bound_certified=not bool(unresolved or invalid_uv_count),
        csf_split_degenerate_uv_face_count=invalid_uv_count,
        csf_split_parameterization_repair_required=bool(invalid_uv_count),
        csf_split_evaluation_scope='usable source triangles only' if invalid_uv_count else 'all source triangles',
        csf_split_step_analysis_model='chart-global source-triangle area Jacobian; cuts separate high and low CSF panels; no reparameterization',
        csf_split_component_sigma=values, csf_split_component_area_scale=[anchor for c in final],
        csf_split_component_irreducible_quad_sigma=[float(irreducible[c].max()) for c in final],
        csf_split_irreducible_quad_count=int(np.count_nonzero(irreducible > threshold+1e-10)),
        csf_split_quad_coverage_floor_sweep=parameterization.metrics.get('split_quad_coverage_floor_sweep'),
        csf_split_face_sigma=face_sigma.tolist(),
        csf_split_residual_high_face_count=int(np.count_nonzero(face_sigma > threshold+1e-10)),
        csf_split_additional_split_recommended_after_all=exhausted,
        csf_split_stopping_rule='accept cuts reducing high/low quad adjacency; retain fixed chart CSF and report residual violations',
        csf_split_conformal_anisotropy_max=parameterization.metrics.get('split_conformal_anisotropy_max'),
        csf_split_area_bound_requires_conformal_map=True,
        csf_split_residual_max_after_all=after,
        csf_split_rejected_components=rejected,
        split_boundary_pairs=pairs,
        m2d_quad_count_after_csf_split=len(faces), m2d_connected_component_count_after_csf_split=len(final),
        m2d_largest_component_quad_count_after_csf_split=max(map(len, final), default=0),
        m2d_smallest_component_quad_count_after_csf_split=min(map(len, final), default=0),
        csf_split_direction_rule='highest signed area-normalized source Gaussian curvature; nearest complete grid cut; minimize mixed high/low adjacency',
        csf_split_grid_rotation_degrees=float(np.rad2deg(angle)),
    )
    vertices[:, :2] = vertices[:, :2] @ rotation.T
    domain.csf_model = metrics['csf_model']
    domain.csf_split_exactness_label = metrics['csf_split_exactness_label']
    domain.paper_split_diagnostics = dict(metrics, split_count=len(records), status=metrics['csf_split_status'])
    parameterization.metrics['split_diagnostics'] = dict(domain.paper_split_diagnostics)
    return vertices, faces, records, metrics
