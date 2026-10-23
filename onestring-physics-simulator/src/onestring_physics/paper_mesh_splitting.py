"""Hierarchical full-grid Split, with separate research and legacy policies.

Zaman et al. 2025 Sec.4.5 and Fig.6 place complete grid cuts near high source
Gaussian curvature when the bounded auxetic expansion sigma exceeds two.
The paper does NOT specify a discrete cut candidate optimizer or reflattening
implementation. Paper mode assesses the *linear conformal scale-factor span*
sqrt(max(area Jacobian)/min(area Jacobian)) in each connected component.
The underlying Jacobian describes area scale, i.e. lambda^2, not lambda.
Only candidates improving the worst child scale factor are committed. This prevents unproductive budget-
exhausting cuts but is an explicitly labeled discretization, not authors' code.
No piecewise UV rescaling or post-cut conformal remeshing is performed, so an
area bound computed here is NOT a certified manufacturable mechanism.
OFF mode keeps its pre-existing fixed-chart linear-CSF behavior unchanged.
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


def _m2d_m3d_panel_area_field(vertices, faces, parameterization):
    """Effective area expansion of each actual M2D quad into its pre-Split M3D.

    Lift each *grid vertex* with exactly the same strict UV-triangle barycentric
    interpolation as the official-BFF M2D -> M3D stage. For every quad,
    sum the areas of triangles (0,1,2) and (0,2,3); divide by its
    2D polygon area. These values are NOT the input BFF triangle Jacobians
    and are NOT a pointwise conformal scale factor.

    Evaluate on the canonical, unseparated M2D before any seam-gap offsets.
    The values remain tied to face IDs through all later connectivity cuts.
    """
    from .reference_bff import strict_inverse_map_uv_to_surface

    grid_uv = np.asarray(vertices, dtype=float)[:, :2]
    quads = np.asarray(faces, dtype=int)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uv_faces = np.asarray(
        getattr(parameterization, "uv_faces", parameterization.surface_faces),
        dtype=int,
    )
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    xyz_faces = np.asarray(parameterization.surface_faces, dtype=int)
    if uv_faces.shape != xyz_faces.shape:
        raise ValueError("M2D/M3D panel CSF needs paired BFF UV and surface triangles")
    used = np.unique(quads.ravel())
    mapped = np.full((len(grid_uv), 3), np.nan, dtype=float)
    for vertex_id in used:
        pos, _, _, _ = strict_inverse_map_uv_to_surface(
            grid_uv[int(vertex_id)], uv, uv_faces, xyz, xyz_faces,
            tolerance=1e-10, vertex_id=int(vertex_id),
        )
        mapped[int(vertex_id)] = pos

    planar_quads = grid_uv[quads]
    spatial_quads = mapped[quads]
    dxy = np.roll(planar_quads, -1, axis=1)
    area2 = .5 * np.abs(np.sum(
        planar_quads[:, :, 0] * dxy[:, :, 1]
        - planar_quads[:, :, 1] * dxy[:, :, 0], axis=1
    ))
    t1 = spatial_quads[:, 1] - spatial_quads[:, 0]
    t2 = spatial_quads[:, 2] - spatial_quads[:, 0]
    t3 = spatial_quads[:, 3] - spatial_quads[:, 0]
    area3 = .5 * (
        np.linalg.norm(np.cross(t1, t2), axis=1)
        + np.linalg.norm(np.cross(t2, t3), axis=1)
    )
    scale2 = float(np.median(area2)) if len(area2) else 0.
    if (not len(quads) or
        not np.isfinite(area2).all() or not np.isfinite(area3).all() or
        np.any(area2 <= max(np.finfo(float).tiny, scale2 * 1e-12)) or
        np.any(area3 <= max(np.finfo(float).tiny, 1e-12 * float(np.median(area3))))):
        raise ValueError("M2D/M3D panel CSF undefined: zero/invalid quad area")
    expansion = area3 / area2
    if not np.isfinite(expansion).all() or np.any(expansion <= 0):
        raise ValueError("M2D/M3D panel CSF produced invalid area expansion")
    diagnostics = {
        "split_panel_area_source": "pre_split_M2D_M3D_strict_bff_inverse",
        "split_panel_area_definition": "area(M3D triangulation 012+023) / area(M2D polygon)",
        "split_panel_area_triangle_diagonal": "0-2",
        "split_panel_area_ratio_min": float(expansion.min()),
        "split_panel_area_ratio_max": float(expansion.max()),
        "split_panel_area_ratio_values": expansion.tolist(),
        "split_panel_area_m2d_values": area2.tolist(),
        "split_panel_area_m3d_values": area3.tolist(),
        "split_panel_m3d_uv_inverse_vertex_count": int(len(used)),
        "split_panel_csf_is_conformal_certificate": False,
    }
    return expansion, diagnostics


def _face_samples(vertices, faces, parameterization, threshold=2., linear_csf=False,
                  sampling_method="overlap_extrema"):
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
    if sampling_method == "interpolated_grid_vertices":
        from .paper_csf_discrete import interpolate_jacobian_to_grid_vertices
        raw_lo,raw_hi=lo.copy(),hi.copy()
        lo,hi,sampling_diagnostics = interpolate_jacobian_to_grid_vertices(
            uv,sf,raw,vertices,faces
        )
        parameterization.metrics.update(
            split_csf_sampling_diagnostics=sampling_diagnostics,
            split_quad_overlap_extrema_before_vertex_sampling=dict(
                worst_quad_linear_csf=float(np.max(np.sqrt(raw_hi/raw_lo))),
                chart_linear_csf=float(np.sqrt(raw_hi.max()/raw_lo.min())),
                min_extremum_coverage_below_one_percent=int((lo_cover<.01).sum()),
                max_extremum_coverage_below_one_percent=int((hi_cover<.01).sum()),
            ),
        )
    elif sampling_method == "m2d_m3d_panel_area":
        panel_jacobian, panel_diag = _m2d_m3d_panel_area_field(
            vertices, faces, parameterization
        )
        # One effective area scale per M2D/M3D quad. Cutting along grid
        # lines does not subdivide a quad; each split child reuses these
        # original face IDs and area values. No BFF triangle extrema are
        # fed into the candidate acceptance test in this mode.
        lo = panel_jacobian.copy()
        hi = panel_jacobian.copy()
        parameterization.metrics.update(split_csf_sampling_diagnostics=panel_diag)
    elif sampling_method != "overlap_extrema":
        raise ValueError(f"Unknown Paper CSF sampling method: {sampling_method}")
    parameterization.metrics["split_csf_sampling_method"] = sampling_method
    # Do not compare a raw area-Jacobian ratio to a *linear* CSF threshold.
    quad_sigma = np.sqrt(hi/lo) if linear_csf else hi/lo
    q = [0., 50., 90., 99., 100.]
    report = {}
    for f in floors:
        if sampling_method == "m2d_m3d_panel_area":
            # Coverage-floor sweep is not meaningful for panel-based samples.
            fl, fh = lo, hi
        else:
            fl, fh = np.asarray(sweep[f][0]), np.asarray(sweep[f][1])
        ratios = fh/fl
        chart_ratio = fh.max()/fl.min()
        if linear_csf:
            ratios = np.sqrt(ratios)
            chart_ratio = np.sqrt(chart_ratio)
        report[f] = (float(chart_ratio), int(np.count_nonzero(ratios > threshold+1e-10)), float(ratios.max()))
    parameterization.metrics.update(
        split_quad_count=int(len(faces)),
        split_quad_csf_units=('linear lambda=sqrt(area Jacobian ratio)' if linear_csf
                              else 'legacy area Jacobian ratio'),
        split_quad_sigma_percentiles={f'p{x:g}': float(v) for x, v in zip(q, np.percentile(quad_sigma, q))},
        split_quad_irreducible_count=int(np.count_nonzero(quad_sigma > threshold+1e-10)),
        split_quad_lo_set_by_sliver_count=int(np.count_nonzero(lo_cover < .01)),
        split_quad_hi_set_by_sliver_count=int(np.count_nonzero(hi_cover < .01)),
        split_quad_coverage_floor_active=1e-12,
        split_quad_coverage_floor_sweep={f'{f:g}': {'chart_sigma': report[f][0], 'quads_over_threshold': report[f][1],
                                                    'worst_quad_sigma': report[f][2]} for f in floors},
    )
    print('[FACE-SAMPLES] quads=%d threshold=%.4g sampling=%s active_coverage_floor=%s'
          % (len(faces), threshold, sampling_method,
             'none (M2D/M3D whole-panel area)' if sampling_method=='m2d_m3d_panel_area'
             else 'none (grid-vertex samples)' if sampling_method=='interpolated_grid_vertices'
             else '1e-12 (overlap extrema)'), flush=True)
    print('  quad sigma ' + ('sqrt(hi/lo)' if linear_csf else 'hi/lo') + '  ' + '  '.join('p%g=%.4g' % (x, v) for x, v in zip(q, np.percentile(quad_sigma, q)))
          + '   irreducible(>thr)=%d/%d' % (int(np.count_nonzero(quad_sigma > threshold+1e-10)), len(faces)), flush=True)
    print('  lo set by a triangle covering <1%% of the quad: %d quads  (<0.01%%: %d)'
          % (int(np.count_nonzero(lo_cover < .01)), int(np.count_nonzero(lo_cover < 1e-4))), flush=True)
    print('  hi set by a triangle covering <1%% of the quad: %d quads  (<0.01%%: %d)'
          % (int(np.count_nonzero(hi_cover < .01)), int(np.count_nonzero(hi_cover < 1e-4))), flush=True)
    print('  overlap-extrema coverage-floor sweep (diagnostic only; not the interpolated sampler):', flush=True)
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
    # OptCuts candidate selection evaluates CSF cuts only; do not accidentally
    # inject the outer T2D-feedback cut on every alternative UV chart.
    trial_params=copy.copy(params) if params is not None else None
    if trial_params is not None:
        trial_params.collision_split_forced_lines=[]
    _,_,records,metrics=_split_mesh_single(vertices,faces,trial,trial_params)
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
    # Opt-in Sec.4.5 research reconstruction. Do not alter the default
    # fixed-UV, OptCuts or collision-driven Split route.
    criterion = str(getattr(params, "paper_split_criterion", "fixed_uv"))
    if criterion in ("reflatten_tile_area", "fixed_chart_tile_area"):
        # The two modes share curvature peaks, complete grid cuts, area-span
        # scoring and physical M2D seam updates. The evaluation UV differs:
        # independent free-boundary BFF or inherited original Omega chart.
        from .paper_split_reflatten import (
            split_mesh_reflatten, split_mesh_fixed_chart_area,
        )
        if criterion == "reflatten_tile_area":
            return split_mesh_reflatten(vertices, faces, domain, params)
        return split_mesh_fixed_chart_area(vertices, faces, domain, params)
    selected,evaluated=_select_optcuts_by_split_count(vertices,faces,domain,params)
    if evaluated is not None:
        domain.parameterization=selected
    return _split_mesh_single(vertices,faces,domain,params)


def _component_area_span(lo, hi, component):
    """Minimum achievable max area expansion after independent UNIFORM rescale.

    For positive Jacobian determinants J over a separated component, every
    uniform 2D rescale changes J by the same factor. Normalize min(J)=1:
    max(J)/min(J) is the remaining expansion factor. This ratio is only a
    *potential bound* until the component's UV/tiles are actually rescaled;
    never label this alone as fabrication-certified.
    """
    lo=np.asarray(lo,float)
    hi=np.asarray(hi,float)
    c=np.asarray(component,int)
    if not len(c) or not np.isfinite(lo[c]).all() or not np.isfinite(hi[c]).all():
        raise ValueError("Cannot evaluate component CSF: empty/nonfinite area Jacobian")
    amin=float(lo[c].min())
    amax=float(hi[c].max())
    if amin<=0 or amax<amin:
        raise ValueError("Invalid Jacobian range in CSF component")
    return amax/amin


def _component_conformal_scale_span(lo, hi, component):
    """Minimum max/min *linear* conformal scale on a disconnected piece.

    A conformal map has J=lambda^2: a piece may be uniformly rescaled
    so min(lambda)=1 and the required max linear stretch becomes
    sqrt(Jmax/Jmin). The *area* ratio is its square. Report both metrics;
    this bound is conditional on an actual piecewise scaling operation.
    """
    return float(np.sqrt(_component_area_span(lo, hi, component)))


def _initial_single_cut_exhaustive_audit(vertices, faces, ratio, limit):
    """Read-only audit of every complete row/column split from the INITIAL M2D.

    This evaluates the same fixed UV/Jacobian field used by Split itself.
    No candidates are committed, no search policy or count is changed.
    Consequently a residual >limit is a proof that one fixed-chart
    full-grid cut CANNOT satisfy this discretized CSF model. It is
    NOT a claim about the authors' UV charts or post-cut remapping.
    """
    from .simple_split_panel_patch import _edge_components
    from .final_split_panel_pass import _complete_cut_once

    comps = _edge_components(faces)
    baseline = max((ratio(c) for c in comps), default=1.)
    floor_candidates = []
    scanned = 0
    for component in comps:
        bounds = vertices[np.unique(faces[component]), :2]
        for axis, coordinate in (("col", 0), ("row", 1)):
            for value in np.unique(np.round(bounds[:, coordinate], 12))[1:-1]:
                _, subfaces, record = _complete_cut_once(
                    vertices, faces[component], axis, float(value)
                )
                if record is None:
                    continue
                children = _edge_components(subfaces)
                if len(children) != 2:
                    continue
                child_csfs = [ratio(component[child]) for child in children]
                untouched = [ratio(other) for other in comps
                             if other is not component]
                worst = max(child_csfs + untouched)
                scanned += 1
                floor_candidates.append(dict(
                    axis=axis,
                    coordinate=float(record["snapped_value"]),
                    component_id=int(component.min()),
                    worst_linear_csf=float(worst),
                    child_linear_csf=[float(v) for v in child_csfs],
                    improvement=float(baseline-worst),
                    one_cut_below_threshold=bool(worst <= limit+1e-10),
                ))
    floor_candidates.sort(key=lambda c: (
        c["worst_linear_csf"], -c["improvement"],
        c["axis"], c["coordinate"]
    ))
    return dict(
        scope="initial M2D fixed UV; complete cuts in any grid row/column; "
              "potential per-component scale normalization only",
        initial_component_count=len(comps),
        initial_max_linear_csf=float(baseline),
        complete_cut_candidate_count=int(scanned),
        best_single_cut=floor_candidates[0] if floor_candidates else None,
        any_single_cut_below_threshold=any(
            c["one_cut_below_threshold"] for c in floor_candidates
        ),
        candidates=floor_candidates,
    )


def _split_mesh_single(vertices, faces, domain, params=None):
    from .simple_split_panel_patch import _edge_components
    from .final_split_panel_pass import _complete_cut_once

    vertices, faces = np.asarray(vertices, float).copy(), np.asarray(faces, int).copy()
    parameterization = domain.parameterization
    paper_fidelity = bool(getattr(params, "paper_fidelity_mode", False))
    # The theoretical Fig.18 area limit is 2; do not inherit arbitrary OFF
    # thresholds when called directly without the Streamlit Paper preset.
    area_limit_experiment = bool(
        paper_fidelity and getattr(params, "paper_split_area_limit_experiment", False)
    )
    # In a conformal map, the area Jacobian is lambda^2. An area
    # expansion-span limit of 2 corresponds to linear span sqrt(2).
    # This interpretation is OPT-IN: §4.5's threshold notation is ambiguous.
    # The default paper computational candidate still uses linear lambda=2.
    threshold = (float(np.sqrt(2.0)) if area_limit_experiment else
                 2.0 if paper_fidelity else
                 float(getattr(domain, 'csf_split_threshold', 2.)))
    # Paper Sec.4.5 describes hierarchical cuts, not a strict per-cut
    # decrease in the worst component CSF. A decrease may require a
    # sequence of cuts with an intermediate unchanged maximum.
    # Every accepted neutral cut must separate nonempty quad sets,
    # which ensures termination even when it does not improve CSF.
    allow_neutral_cuts = bool(
        paper_fidelity and getattr(params, "paper_split_allow_neutral_cuts", True)
    )
    sampling_method = (str(getattr(params, "paper_csf_sampling_method",
                                  "m2d_m3d_panel_area")) if paper_fidelity
                       else "overlap_extrema")
    # Curvature-candidate association still uses which source BFF triangles
    # overlap an M2D quad. The NEW dual-centroid area field, when selected,
    # is separately computed from M2D/M3D tile-centroid geometry and is the
    # ONLY field used for Split's CSF acceptance test.
    dual_mode = paper_fidelity and sampling_method == "m2d_m3d_dual_centroid"
    lo, hi, samples = _face_samples(
        vertices, faces, parameterization, threshold, paper_fidelity,
        "overlap_extrema" if dual_mode else sampling_method,
    )
    dual_field = None
    if dual_mode:
        from .paper_dual_csf import build_dual_centroid_field, component_dual_stats
        dual_field = build_dual_centroid_field(vertices, faces, parameterization)
        parameterization.metrics["split_csf_sampling_method"] = sampling_method
        parameterization.metrics["split_csf_sampling_diagnostics"] = dict(
            dual_field["diagnostics"]
        )
    # In Sec.4.5, lambda is the conformal scale factor of the map;
    # the area Jacobian is lambda^2. A disconnected piece can, in
    # principle, be uniformly rescaled, so its *linear* scale span is
    # sqrt(Jmax/Jmin), not the area ratio Jmax/Jmin. This change alone
    # should not be confused with actually applying per-piece UV rescale.
    # Legacy OFF mode retains its historical fixed-chart calculation.
    irreducible = np.sqrt(hi)
    paper_quad_floor = np.sqrt(hi / lo)
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
    def dual_stats(component, active_faces=None):
        return component_dual_stats(
            component, faces if active_faces is None else active_faces,
            dual_field,
        )

    def ratio(component, active_faces=None):
        if dual_mode:
            result = dual_stats(component, active_faces)
            # Display-only placeholder when there is no complete dual cell.
            # This is NEVER used to accept a child cut without coverage,
            # and the final status explicitly marks unmeasured components.
            return float(result["span"]) if result["measurable"] else 1.0
        if paper_fidelity:
            return _component_conformal_scale_span(lo, hi, component)
        # Exact original OFF calculation: global-min-normalized linear scale.
        return float(np.sqrt(hi[component].max()))
    initial_components = _edge_components(faces)
    before = max(map(ratio, initial_components), default=1.)
    initial_single_cut_audit = (
        _initial_single_cut_exhaustive_audit(vertices, faces, ratio, threshold)
        if paper_fidelity and not dual_mode else None
    )
    while enabled and len(records) < budget:
        components = _edge_components(faces)
        bad = [c for c in components
               if ratio(c) > threshold + 1e-10 and tuple(c) not in blocked]
        if not bad:
            break
        # Exactly the OFF mode's offending-component and peak order.
        component = max(bad, key=lambda c: (round(ratio(c), 10), len(c)))
        ids = np.unique(np.concatenate([samples[i] for i in component]))
        ids = ids[np.isfinite(curvature[ids])]
        ids = ids[np.argsort(-curvature[ids], kind='stable')]
        component_before = ratio(component)
        # Lower bound imposed by indivisible grid quads. In Paper mode this
        # is max over each quad's local J_max / J_min. Splitting cannot
        # reduce that distortion without changing the mesh/parameterization.
        # An M2D tile is not itself a dual linkage cell. There is no
        # per-quad distortion floor in the centroid-dual mode.
        component_floor = (1.0 if dual_mode else
                           float((paper_quad_floor if paper_fidelity
                                  else irreducible)[component].max()))
        if component_before <= component_floor * (1. + 1e-10):
            blocked.add(tuple(component))
            rejected.append({'face_ids': component.tolist(),
                             'sigma': component_before,
                             'reason': 'within_quad_distortion_floor',
                             'irreducible_sigma': component_floor})
            continue
        selected = None
        selected_peak = None
        used = np.unique(faces[component])
        bounds = vertices[used, :2]
        tested = set()
        complete_candidate_seen = False
        insufficient_dual_candidate = False
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
                # A high-curvature source point on a *previous* seam can
                # still guide the next grid cut in one of the resulting
                # subdomains. Snapping to its nearest interior grid line is
                # necessary for sequential plateau-then-improvement cuts.
                # Legacy strict mode retains its historical boundary guard.
                if (not allow_neutral_cuts
                    and not bounds[:, coord].min()+1e-12 < point[coord] < bounds[:, coord].max()-1e-12):
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
                complete_candidate_seen = True
                if dual_mode:
                    trial_faces = faces.copy()
                    trial_faces[component] = subset
                    # A cut can disconnect the four tiles around the central
                    # grid vertex. Those cross-seam dual cells cease to exist.
                    # A child with zero complete dual cells is NOT a valid
                    # CSF<=2 success; reject this candidate conservatively.
                    if not all(
                        dual_stats(component[c], trial_faces)["measurable"]
                        for c in children
                    ):
                        insufficient_dual_candidate = True
                        continue
                    child_sigmas = [
                        ratio(component[c], trial_faces) for c in children
                    ]
                else:
                    child_sigmas = [ratio(component[c]) for c in children]
                # A strict one-step improvement rule is NOT specified in
                # Sec.4.5 and rejects multi-cut solutions (e.g. [1,9,1,9]).
                # Every valid bisection has smaller nonempty child tile
                # sets, so neutral moves are finite, never an endless loop.
                worst_child = max(child_sigmas)
                tolerance = 1e-10 * max(1.0, component_before)
                if worst_child > component_before + tolerance:
                    continue
                if not allow_neutral_cuts and worst_child >= component_before - tolerance:
                    continue
                score = (worst_child, imbalance, axis)
                candidates.append((score, v2, subset, record))
            if candidates:
                selected = min(candidates, key=lambda x: x[0])
                selected_peak = int(peak)
                break  # highest-curvature peak admitting a complete grid cut
        if selected is None:
            blocked.add(tuple(component))
            rejected.append({'face_ids': component.tolist(), 'sigma': component_before,
                             'reason': ('all_candidate_cuts_leave_unmeasured_dual_regions'
                                        if dual_mode and insufficient_dual_candidate else
                                        'no_distortion_reducing_grid_cut'
                                        if complete_candidate_seen else
                                        'no_complete_grid_cut_near_source_curvature_sample'),
                             'irreducible_sigma': component_floor})
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
        record.update(face_ids=component.tolist(), curvature_vertex_id=int(selected_peak), gaussian_curvature=float(curvature[selected_peak]),
                      neutral_sigma_step=bool(
                          abs(component_before-max(ratio(component[c]) for c in children))
                          <=1e-10*max(1.,component_before)
                      ),
                      sigma_before=component_before, sigma_after=[ratio(component[c]) for c in children],
                      worst_sigma_reduction=component_before-max(ratio(component[c]) for c in children),
                      child_face_ids=[component[c].tolist() for c in children],
                      split_boundary_pairs=step_pairs,
                      grid_snap_distance=abs(record['snapped_value']-record['requested_value']),
                      component_id=int(component.min()))
        records.append(record)
    # The CSF cuts above are left unchanged. Collision-driven cuts are injected
    # ONE AT A TIME by the outer full-pipeline feedback loop. They modify the
    # real M2D vertex IDs before M3D/K3D/K2D/T3D/T2D are rebuilt.
    csf_count=len(records)
    collision_records=[]
    forced = list(getattr(params, 'collision_split_forced_lines', ()) or ())
    for entry in forced:
        axis=str(entry.get('axis',''))
        value=float(entry.get('value',float('nan')))
        anchor=int(entry.get('source_face_id',-1))
        if axis not in ('row','col') or not np.isfinite(value) or not (0 <= anchor < len(faces)):
            raise ValueError(f'Invalid collision Split candidate: {entry!r}')
        comp=next((c for c in _edge_components(faces) if anchor in c),None)
        if comp is None:
            raise ValueError('Collision Split source face no longer belongs to a component')
        # Exactly one current component and one complete internal grid line.
        old_owners=defaultdict(list)
        for fi in comp:
            for k in range(4):
                edge=tuple(sorted((int(faces[fi,k]),int(faces[fi,(k+1)%4]))))
                old_owners[edge].append((int(fi),k))
        new_vertices,subset,record=_complete_cut_once(
            vertices,faces[comp],axis,value
        )
        if record is None or abs(float(record['snapped_value'])-value) > 1e-7*max(1.,abs(value)):
            raise ValueError(
                'Collision Split candidate was not a valid complete seam '
                f'(axis={axis}, value={value}, face={anchor})'
            )
        updated=faces.copy();updated[comp]=subset
        step_pairs=[]
        for entries in old_owners.values():
            if len(entries)!=2:
                continue
            (a,ea),(b,eb)=entries
            if set(updated[a,[ea,(ea+1)%4]]) != set(updated[b,[eb,(eb+1)%4]]):
                pair=[a,ea,b,eb]
                step_pairs.append(pair);pairs.append(pair)
        if not step_pairs:
            raise ValueError('Collision Split produced no actual broken shared edges')
        vertices,faces=new_vertices,updated
        record.update(
            face_ids=comp.tolist(),
            collision_driven=True,
            collision_source_face_id=anchor,
            collision_score=float(entry.get('score',0.)),
            collision_boundary_distance_tiles=float(entry.get('boundary_distance_tiles',0.)),
            split_boundary_pairs=step_pairs,
            component_id=int(comp.min()),
        )
        records.append(record);collision_records.append(record)

    final = _edge_components(faces)
    invalid_uv_count = int(parameterization.metrics.get('split_csf_degenerate_uv_face_count', 0))
    values = [ratio(c) for c in final]
    final_dual_stats = [dual_stats(c) for c in final] if dual_mode else []
    unmeasured_dual_regions = int(sum(not s["measurable"] for s in final_dual_stats))
    unresolved = sum(v > threshold+1e-10 for v in values)
    unresolved_mixed = 0  # legacy metric retained for dashboard compatibility
    exhausted = bool(enabled and unresolved and csf_count >= budget)
    stopping_reason = (
        'disabled' if not enabled else
        'threshold_met' if not unresolved else
        'budget_exhausted' if exhausted else
        'no_admissible_cut_remaining'
    )
    after = max(values, default=1.)
    face_sigma = paper_quad_floor.copy() if paper_fidelity else np.sqrt(hi).copy()
    dual_uncovered_quad_count = 0
    if dual_mode:
        face_sigma = np.ones(len(faces), float)
        covered = np.zeros(len(faces), bool)
        intact_dual_ids = sorted({
            idx for summary in final_dual_stats
            for idx in summary["used_dual_ids"]
        })
        if intact_dual_ids:
            reference_min = float(np.min(dual_field["jacobian"][intact_dual_ids]))
            for idx in intact_dual_ids:
                member_ids = dual_field["face_ids"][idx]
                j = float(dual_field["jacobian"][idx])
                face_sigma[member_ids] = np.maximum(
                    face_sigma[member_ids], np.sqrt(j/reference_min)
                )
                covered[member_ids] = True
        dual_uncovered_quad_count = int((~covered).sum())
    domain.csf_before, domain.csf_after_split = before, after
    domain.split_lines = [('row' if r['axis'] == 'row' else 'col', r['snapped_value']) for r in records]
    domain.localized_split_segments = list(domain.split_lines)
    metrics = dict(
        csf_model=(
            ('Experimental four-tile centroid dual area span sqrt(max(A3/A2)/min(A3/A2))'
             if dual_mode else
             'Experimental M2D/M3D quad area span sqrt(max(A3/A2)/min(A3/A2))'
             if sampling_method == "m2d_m3d_panel_area"
             else 'Paper Sec.4.5 linear conformal scale span sqrt(max(J)/min(J))')
            if paper_fidelity else
            'local inverse length scale sqrt(dA_S/dA_Omega); retained UV chart'
        ),
        csf_split_exactness_label=(
            'Experimental four-tile centroid dual area span; '
            'not a closed-to-open auxetic expansion or manufacturability certificate'
            if dual_mode else
            ('EXPERIMENTAL paper Figure 18 area-span<=2 interpreted via '
             'sqrt(area Jacobian span)<=sqrt(2); no actual UV rescale/remap, '
             'reparameterization or fabrication certificate')
            if area_limit_experiment else
            'Paper candidate linear CSF only: sqrt(area Jacobian span) after hypothetical '
            'per-piece rescaling; no actual UV rescale/remap or fabrication certificate'
            if paper_fidelity and sampling_method not in (
                "m2d_m3d_panel_area", "m2d_m3d_dual_centroid"
            ) else
            'Experimental M2D/M3D dual-centroid area-span proxy; does not certify '
            'conformality or fabrication (no per-piece rescale/remap)'
            if paper_fidelity else
            'OFF-mode fixed-chart inverse-length CSF; no post-cut reparameterization'
        ),
        csf_split_applied=bool(csf_count), csf_split_threshold=threshold, max_csf_splits=budget,
        max_csf_before_split=before, max_csf_after_split=after, number_of_splits=len(records),
        split_locations=[list(x) for x in domain.split_lines],
        raw_split_locations=[[r['axis'], r['requested_value']] for r in records],
        csf_split_duplicated_vertex_count=sum(r['duplicated_vertices'] for r in records[:csf_count]),
        collision_split_applied=bool(collision_records),
        collision_split_count=len(collision_records),
        collision_split_step_analysis=collision_records,
        collision_split_duplicated_vertex_count=sum(r['duplicated_vertices'] for r in collision_records),
        collision_split_grid_seam_model='complete boundary-to-boundary M2D cut; full downstream rebuild',
        csf_split_budget_exhausted=exhausted, csf_split_unresolved_component_count=unresolved,
        csf_split_stopping_reason=stopping_reason,
        csf_split_unresolved_mixed_edges=unresolved_mixed,
        csf_split_status=(
            'uncertified_insufficient_dual_cells' if dual_mode and unmeasured_dual_regions else
            'uncertified_uv_degeneracy' if invalid_uv_count
            else 'satisfied' if not unresolved else
            ('disabled' if not enabled else 'budget_exhausted'
             if exhausted else 'no_admissible_curvature_guided_grid_cut')
        ),
        csf_split_step_analysis=records[:csf_count], csf_split_step_count=csf_count,
        csf_split_neutral_cut_policy="allow_hierarchical_plateau" if allow_neutral_cuts else "strict_one_step_improvement",
        csf_split_neutral_cut_count=int(sum(bool(r.get("neutral_sigma_step",False)) for r in records[:csf_count])),
        csf_split_locations=[['row' if r['axis']=='row' else 'col',r['snapped_value']]
                             for r in records[:csf_count]],
        csf_split_initial_components=[
            dict(face_ids=c.tolist(), sigma=ratio(c), **(
                {"dual_cell_count": dual_stats(c)["cell_count"],
                 "dual_measurable": dual_stats(c)["measurable"]}
                if dual_mode else {}
            ))
            for c in initial_components
        ],
        # This OFF-mode CSF is a *linear* proxy; no paper Fig.18 area
        # upper bound has been geometrically certified.
        csf_split_bound_certified=False,
        # Keep *area* and *linear* thresholds separate. The paper uses
        # lambda<=2, but Supplement Fig.18 calls 2 the area-expansion
        # limit; we must not silently certify both with one comparison.
        csf_split_candidate_area_bound_satisfied=bool(
            paper_fidelity and after*after <= 2.0 + 1e-10
            and not invalid_uv_count and not unmeasured_dual_regions
        ),
        csf_split_candidate_linear_bound_satisfied=bool(
            paper_fidelity and after <= 2.0 + 1e-10 and not invalid_uv_count
            and not unmeasured_dual_regions
        ),
        csf_split_area_limit_experiment=area_limit_experiment,
        csf_split_effective_linear_threshold=float(threshold),
        csf_split_requested_area_threshold=2.0 if area_limit_experiment else None,
        csf_split_component_area_ratio=[
            (dual_stats(c)["area_span"] if dual_mode else
             float(_component_area_span(lo, hi, c)))
            for c in final
        ] if paper_fidelity else [],
        csf_split_area_ratio_after=float(after*after) if paper_fidelity else None,
        csf_split_linear_csf_area_relation='lambda=sqrt(area Jacobian ratio)' if paper_fidelity else None,
        csf_split_component_rescale_applied=False,
        csf_split_paper_area_span_active=paper_fidelity,
        csf_split_sampling_method=sampling_method,
        csf_split_dual_diagnostics=dual_field["diagnostics"] if dual_mode else None,
        csf_split_dual_component_stats=final_dual_stats if dual_mode else None,
        csf_split_dual_unmeasured_region_count=unmeasured_dual_regions,
        csf_split_dual_uncovered_quad_count=dual_uncovered_quad_count,
        csf_split_dual_cross_seam_policy=(
            "require all four member quads and a common actual Grid vertex "
            "inside each connected Split region; reject unmeasurable cut children"
            if dual_mode else None
        ),
        csf_split_panel_area_diagnostics=(
            parameterization.metrics.get('split_csf_sampling_diagnostics')
            if sampling_method in ('m2d_m3d_panel_area', 'm2d_m3d_dual_centroid') else None
        ),
        csf_split_algorithm_same_as_normal=not paper_fidelity,
        csf_split_degenerate_uv_face_count=invalid_uv_count,
        csf_split_parameterization_repair_required=bool(invalid_uv_count),
        csf_split_evaluation_scope=('intact four-tile centroid dual cells after each cut' if dual_mode else
                                    'original M2D/M3D quads' if paper_fidelity and sampling_method == 'm2d_m3d_panel_area' else
                                    'usable source triangles only' if invalid_uv_count else 'all source triangles'),
        csf_split_step_analysis_model=(
            ('Paper-oriented experiment: four-tile centroid dual area span; crossing-cut cells excluded'
             if dual_mode else
             'Paper-oriented experiment: sqrt(M3D/M2D panel area-span) after strict inverse lift'
             if sampling_method == 'm2d_m3d_panel_area'
             else 'Paper: sqrt(component area-Jacobian span) candidate; no UV rescale/remap')
            if paper_fidelity else
            'fixed-chart inverse length scale; no conformal reparameterization'
        ),
        csf_split_component_sigma=values, csf_split_component_area_scale=[1.0 for _ in final],
        csf_split_component_irreducible_quad_sigma=[
            (1.0 if dual_mode else
             float((paper_quad_floor if paper_fidelity else irreducible)[c].max()))
            for c in final
        ],
        csf_split_initial_one_cut_exhaustive_audit=initial_single_cut_audit,
        csf_split_irreducible_quad_count=(0 if dual_mode else int(np.count_nonzero(
            (paper_quad_floor if paper_fidelity else irreducible) > threshold+1e-10
        ))),
        csf_split_quad_coverage_floor_sweep=parameterization.metrics.get('split_quad_coverage_floor_sweep'),
        csf_split_face_sigma=face_sigma.tolist(),
        csf_split_residual_high_face_count=int(np.count_nonzero(face_sigma > threshold+1e-10)),
        csf_split_additional_split_recommended_after_all=bool(
            exhausted and not all(r['reason'] == 'within_quad_distortion_floor' for r in rejected)
        ),
        csf_split_stopping_rule=(
            'Paper-oriented experiment: four-panel centroid dual areas M3D/M2D; '
            'evaluate only intact cells wholly inside connected Split component; '
            'reject child regions with no intact dual cell; no physical limit certificate'
            if dual_mode else
            'Paper-oriented experiment: per-quad area(M3D)/area(M2D), then '
            'sqrt(max/min) across disconnected component; '
            'source Gaussian-curvature priority; complete Grid cuts; '
            'no per-piece remap, rescale or conformity certificate'
            if paper_fidelity and sampling_method == 'm2d_m3d_panel_area' else
            'Paper Sec.4.5 approximate: use sqrt(component area-Jacobian span); '
            'highest eligible source Gaussian-curvature point; complete '
            'grid bisection with strict worst-child linear-CSF improvement; '
            'terminate at lambda=2, irreducible quad, no improving cut or budget; '
            'no physical UV rescale or remap performed'
            if paper_fidelity else
            'OFF-mode algorithm: max-CSF component; curvature-guided full grid '
            'bisections; worst-child CSF decrease required; stop at distortion '
            'floor, no improvement, threshold or budget'
        ),
        csf_split_conformal_anisotropy_max=parameterization.metrics.get('split_conformal_anisotropy_max'),
        csf_split_area_bound_requires_conformal_map=True,
        csf_split_residual_max_after_all=after,
        csf_split_rejected_components=rejected,
        split_boundary_pairs=pairs,
        m2d_quad_count_after_csf_split=len(faces), m2d_connected_component_count_after_csf_split=len(final),
        m2d_largest_component_quad_count_after_csf_split=max(map(len, final), default=0),
        m2d_smallest_component_quad_count_after_csf_split=min(map(len, final), default=0),
        csf_split_direction_rule=(
            'Paper: highest signed source Gaussian curvature admitting an improving '
            'full-grid cut; choose axis with least worst-child linear CSF'
            if paper_fidelity else
            'highest signed area-normalized source Gaussian curvature; nearest complete grid cut; minimize worst child distortion'
        ),
        csf_split_grid_rotation_degrees=float(np.rad2deg(angle)),
    )
    vertices[:, :2] = vertices[:, :2] @ rotation.T
    domain.csf_model = metrics['csf_model']
    domain.csf_split_exactness_label = metrics['csf_split_exactness_label']
    domain.paper_split_diagnostics = dict(metrics, split_count=len(records), status=metrics['csf_split_status'])
    parameterization.metrics['split_diagnostics'] = dict(domain.paper_split_diagnostics)
    return vertices, faces, records, metrics
