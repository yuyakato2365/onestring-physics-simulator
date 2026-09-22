"""2026-09-23 experiment: version-gated hooks and reproducible run records.

0000 performs diagnostics only. Nonzero supported masks constrain assembled
copies using source-surface barycentric provenance, never flat topology welding.
"""
from __future__ import annotations
import copy
import json
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
import time
import uuid
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from .extrusion_quality import evaluate_extrusion, quality_summary

VERSION = "2026-09-23-extrusion-aware"
FEATURES = ("use_principal_curvature_grid", "use_extrusion_aware_k3d",
            "use_curvature_adaptive_grid", "use_extrusion_aware_split")
ADAPTIVE_LIMITATION = (
    "Curvature-adaptive Grid Size is unavailable: QuadGrid row/column indexing, "
    "axis-aligned CSF cuts, native OptCuts fixed h/phase, and hinge placement assume "
    "a regular lattice. Local refinement needs a conforming quad remesher and explicit "
    "panel/hinge correspondence. No uniform-resolution substitute is applied."
)


def enabled(params):
    return getattr(params, 'model_version', '') == VERSION


def feature_mask(params):
    return ''.join('1' if getattr(params, k, False) else '0' for k in FEATURES)


def validate(params):
    if not enabled(params):
        return
    if params.use_curvature_adaptive_grid:
        raise NotImplementedError(ADAPTIVE_LIMITATION)
    for key in ('extrusion_weight', 'extrusion_split_weight'):
        value = float(getattr(params, key))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if params.thickness <= 0 or not np.isfinite(params.thickness):
        raise ValueError('Positive finite thickness required')
    if params.use_extrusion_aware_k3d and int(params.extrusion_max_nfev) < 1:
        raise ValueError('extrusion_max_nfev must be positive')


def prepare_parameterization(p, params):
    if not enabled(params) or not params.use_principal_curvature_grid:
        return
    if any(token in str(p.method) for token in ('grid_optcuts', 'optcuts_grid')) or any(
        bool(value) for key, value in p.metrics.items()
        if key in {'optcuts_grid_native', 'optcuts_grid_constrained_m2d'}
    ):
        raise NotImplementedError('Principal grid rotation cannot change a native OptCuts fixed seam lattice')
    from .extrusion_curvature import align_principal_grid
    align_principal_grid(p)


def split_cost_values(p, domain, grid, params, csf, pipeline):
    """Add provisional extrusion difficulty to the existing CSF candidate cost.

    The pilot is sampled before split selection on the same overlay, using the
    actual thickness and area-weighted quad normals. No offending panel is cut
    directly. The existing peak/residual candidate generator, budget, symmetry,
    snapping, and duplication stages still choose and construct the cuts.
    """
    if not enabled(params) or not params.use_extrusion_aware_split:
        return csf
    uv = np.asarray(domain.uv_vertices, float)
    nx, ny = int(getattr(domain, 'overlay_nx', grid.nx)), int(getattr(domain, 'overlay_ny', grid.ny))
    if len(uv) != (nx+1)*(ny+1):
        raise ValueError('Extrusion split pilot requires the existing structured overlay')
    mapped = [pipeline.inverse_map_uv_to_surface(point, p) for point in uv]
    xyz = np.asarray([m[0] for m in mapped])
    valid = np.asarray([not m[2] and m[1] >= 0 for m in mapped], bool)
    faces = []
    for row in range(ny):
        for col in range(nx):
            a = row*(nx+1)+col
            f = [a, a+1, a+nx+2, a+nx+1]
            if valid[f].all():
                faces.append(f)
    if not faces:
        p.metrics['extrusion_split_status'] = 'no fully mapped pilot quads; unchanged CSF candidates'
        return csf
    faces = np.asarray(faces, int)
    quality = evaluate_extrusion(xyz, faces, params.thickness)
    centers = uv[faces].mean(axis=1)
    _, idx = cKDTree(centers).query(p.uv_vertices_2d)
    difficulty = quality['difficulty'][idx]
    # D_existing = CSF, lambda has the corresponding dimensionless scale.
    augmented = np.asarray(csf)+float(params.extrusion_split_weight)*difficulty
    p.metrics.update(extrusion_split_status='shared evaluator added to existing CSF candidate cost',
                     extrusion_split_cost_max=float(np.max(augmented)),
                     extrusion_split_pilot_panel_count=len(faces),
                     extrusion_split_pilot_difficulty=quality['difficulty'].tolist(),
                     extrusion_split_cost_weight=float(params.extrusion_split_weight))
    return augmented


def select_split_candidates(p, csf, augmented, threshold, maximum, pipeline):
    """Rank the existing CSF peak/residual candidate pool with an additive cost."""
    if maximum <= 0:
        return []
    uv = np.asarray(p.uv_vertices_2d)
    candidates = []
    for values in (csf, augmented):
        for line in pipeline._csf_split_lines(p, values, threshold, max(2*maximum, 4)):
            pipeline._append_unique_split_line(candidates, line, uv, 4*maximum+8)
    remaining = np.ones(len(uv), bool)
    selected, records = [], []
    for _ in range(maximum):
        best = None
        for line in candidates:
            if line in selected:
                continue
            coord = 1 if line[0] == 'row' else 0
            band = max(float(np.ptp(uv[:, coord]))*.08, 1e-10)
            near = (np.abs(uv[:, coord]-line[1]) <= band) & remaining
            existing = float(np.sum(np.maximum(np.asarray(csf)[near]-threshold, 0)))
            extra = float(np.sum((np.asarray(augmented)-csf)[near]))
            score = existing+extra
            if best is None or score > best[0]:
                best = (score, line, near, existing, extra)
        if best is None or best[0] <= 0:
            break
        score, line, near, existing, extra = best
        selected.append(line)
        remaining[near] = False
        records.append(dict(line=list(line), existing_cost=existing, extrusion_cost=extra, combined_cost=score))
    p.metrics['extrusion_split_candidate_scores'] = records
    p.metrics['extrusion_split_candidate_pool_count'] = len(candidates)
    return selected


def source_equivalence(mesh2d, mesh3d, p, pipeline):
    """Identify duplicates by (original surface vertex IDs, barycentric weights).

    Coincidence alone never joins unrelated sheets. Flat canonical UV is used
    before the visual gaps introduced by Simple Split.
    """
    uv = np.asarray(getattr(mesh2d, '_split_panel_source_vertices', mesh2d.vertices))[:, :2]
    vertices = np.asarray(mesh3d.vertices)
    groups = {}
    for i, point in enumerate(uv):
        _, tid, outside = pipeline.inverse_map_uv_to_surface(point, p)
        key = ('canonical_uv', tuple(np.round(point, 10))) if getattr(mesh2d, 'split_lines', []) else ('independent', i)
        if not outside and 0 <= tid < len(p.uv_faces):
            tri = np.asarray(p.uv_vertices_2d)[p.uv_faces[tid]]
            b = np.linalg.lstsq((tri[1:]-tri[0]).T, point-tri[0], rcond=None)[0]
            weights = np.array([1-b.sum(), *b])
            if np.min(weights) >= -1e-8:
                key = tuple(sorted((int(vid), int(round(w*1e8))) for vid, w in
                                   zip(p.surface_faces[tid], weights) if abs(w) > 1e-8))
        groups.setdefault(key, []).append(i)
    inverse = np.arange(len(vertices))
    tol = max(np.linalg.norm(np.ptp(vertices, axis=0))*1e-7, 1e-10)
    for ids in groups.values():
        if len(ids) > 1 and np.max(np.linalg.norm(vertices[ids]-vertices[ids[0]], axis=1)) <= tol:
            inverse[ids] = ids[0]
    _, mapping = np.unique(inverse, return_inverse=True)
    return mapping


def geometry_view(mesh, mapping):
    view = copy.copy(mesh)
    n = int(np.max(mapping))+1
    v = np.zeros((n, 3))
    np.add.at(v, mapping, mesh.vertices)
    v /= np.bincount(mapping)[:, None]
    view.vertices, view.faces = v, mapping[np.asarray(mesh.faces)]
    view.metrics = dict(mesh.metrics)
    view.metrics.pop('assembled_geometry_map', None)
    return view


def optimize_assembled(base, target, mesh2d, mesh3d, p, params, pipeline):
    if not enabled(params) or feature_mask(params) == '0000':
        return base(target, mesh3d, p, params)
    mapping = source_equivalence(mesh2d, mesh3d, p, pipeline)
    reduced = geometry_view(mesh3d, mapping)
    result, report = base(target, reduced, p, params)
    if params.use_extrusion_aware_k3d and not result.metrics.get('extrusion_objective_applied'):
        raise RuntimeError('Extrusion-aware K3D requires the paper local/global K3D route; select the 2026-09-18 OptCuts Ω route')
    result.vertices = result.vertices[mapping].copy()
    result.faces = np.asarray(mesh3d.faces).copy()
    result.metrics['assembled_geometry_map'] = mapping.tolist()
    result.metrics['assembled_equivalence_group_count'] = int(np.count_nonzero(np.bincount(mapping) > 1))
    result.metrics['assembled_equivalence_policy'] = 'source barycentric identity; reduced geometry unknowns only; original flat topology retained'
    # History snapshots belong to the reduced solve, not the expanded mesh.
    result.metrics['assembled_geometry_reduced_vertex_count'] = len(reduced.vertices)
    return result, report


def refine_k3d(x, faces, source, target, parameterization, params, edge_target, extra_constraints=()):
    """Continue K3D with E_existing + w_e E_extrusion, sparse finite differences.

    Existing plane/square/surface projections and the original edge targets are
    reused. Called before the pre-existing hard-planarity/polish stages.
    """
    from .paper_local_global_solvers import _best_fit_plane_projection, _closest_square_projection, _surface_project
    edges = np.asarray(list(edge_target), int)
    lengths = np.asarray(list(edge_target.values()))
    scale = max(float(np.mean(lengths)), 1e-12)
    wp, wq, ws = (float(getattr(params, k)) for k in ('w_planar', 'w_square', 'w_surface'))
    we = float(params.extrusion_weight)
    def residual(flat):
        v = flat.reshape(-1, 3)
        planar = np.asarray([v[f]-_best_fit_plane_projection(v[f]) for f in faces]).ravel()*np.sqrt(wp)
        square = np.asarray([v[f]-_closest_square_projection(v[f]) for f in faces]).ravel()*np.sqrt(wq)
        d = v[edges[:, 1]]-v[edges[:, 0]]
        ln = np.linalg.norm(d, axis=1)
        edge = (d*(1-lengths/np.maximum(ln, 1e-12))[:, None]).ravel()*np.sqrt(wq)
        surface = (v-_surface_project(v, target, parameterization)).ravel()*np.sqrt(ws)
        quality = evaluate_extrusion(v, faces, params.thickness)['residuals'].ravel()*scale*np.sqrt(we)
        extra = [np.sqrt(w)*(np.sum(v[np.asarray(ids)]*np.asarray(coeffs)[:, None], axis=0)-t)
                 for ids, coeffs, t, w in extra_constraints]
        return np.concatenate((planar, square, edge, surface, quality, np.asarray(extra).ravel()))
    # Quad normal dependencies extend to all faces incident on its corners.
    incident = [set() for _ in x]
    for face in faces:
        for i in face:
            incident[i].update(int(j) for j in face)
    width = evaluate_extrusion(x, faces, params.thickness)['residuals'].shape[1]
    rows, cols = [], []
    row = 0
    blocks = [(12, f) for f in faces]*2 + [(3, e) for e in edges] + [(3, [i]) for i in range(len(x))]
    blocks += [(width, sorted(set().union(*(incident[i] for i in f)))) for f in faces]
    blocks += [(3, ids) for ids, _, _, _ in extra_constraints]
    for count, ids in blocks:
        for r in range(row, row+count):
            for i in ids:
                rows.extend([r]*3); cols.extend([3*i, 3*i+1, 3*i+2])
        row += count
    pattern = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(row, x.size)).tocsr()
    r0 = residual(x.ravel())
    if we == 0:
        return x, dict(extrusion_objective_applied=True, extrusion_objective_before=float(r0@r0),
                       extrusion_objective_after=float(r0@r0), extrusion_optimizer_nfev=0)
    solved = least_squares(residual, x.ravel(), jac_sparsity=pattern, method='trf',
                           max_nfev=int(params.extrusion_max_nfev), ftol=1e-6, xtol=1e-6, gtol=1e-6)
    r1 = residual(solved.x)
    accepted = np.isfinite(solved.x).all() and r1@r1 <= r0@r0+1e-12
    result = solved.x.reshape(-1, 3) if accepted else x
    return result, dict(extrusion_objective_applied=True,
                        extrusion_objective_before=float(r0@r0),
                        extrusion_objective_after=float(r1@r1) if accepted else float(r0@r0),
                        extrusion_optimizer_nfev=int(solved.nfev), extrusion_optimizer_success=bool(solved.success),
                        extrusion_optimizer_message=str(solved.message), extrusion_optimizer_accepted=bool(accepted),
                        extrusion_objective_definition='EPlanar + EShape + ELength + ESurface (existing weights) + w_e * mean_source_edge_length^2 * sum(panel difficulty^2)')


def make_extrusion_objective(vertices, faces, thickness, coefficient):
    """Sparse numerical extrusion gradient for the existing hard K3D solver.

    SciPy's same sparse finite-difference routine used internally by
    least_squares avoids one evaluator call per scalar unknown.
    """
    from scipy.optimize._numdiff import approx_derivative
    incident = [set() for _ in vertices]
    for f in faces:
        for i in f:
            incident[i].update(int(j) for j in f)
    width = evaluate_extrusion(vertices, faces, thickness)['residuals'].shape[1]
    rows, cols = [], []
    for fi, f in enumerate(faces):
        ids = sorted(set().union(*(incident[i] for i in f)))
        for r in range(fi*width, (fi+1)*width):
            for i in ids:
                rows.extend([r]*3); cols.extend([3*i, 3*i+1, 3*i+2])
    pattern = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(width*len(faces), np.size(vertices))).tocsr()
    def residual(flat):
        return evaluate_extrusion(flat.reshape(-1,3), faces, thickness)['residuals'].ravel()
    def energy(flat):
        r = residual(flat)
        return coefficient*float(r@r)
    def gradient(flat):
        r = residual(flat)
        jac = approx_derivative(residual, flat, method='2-point', sparsity=pattern, f0=r)
        return 2*coefficient*np.asarray(jac.T@r).ravel()
    return energy, gradient


def record_run(state, params, runtimes):
    from .paper_local_global_solvers import _best_fit_plane_projection, _closest_square_projection, _surface_project
    mesh = state.mesh_3d_optimized
    mapping = np.asarray(mesh.metrics.get('assembled_geometry_map', np.arange(len(mesh.vertices))), int)
    reduced = geometry_view(mesh, mapping)
    before = mesh.metrics.get('extrusion_pre_t3d', {})
    quality = evaluate_extrusion(reduced.vertices, reduced.faces, params.thickness)
    actual = evaluate_extrusion(mesh.vertices, mesh.faces, params.thickness, tiles=state.tiles_3d.vertices)
    v, f = mesh.vertices, mesh.faces
    planar = sum(float(np.sum((v[q]-_best_fit_plane_projection(v[q]))**2)) for q in f)
    square_shape = sum(float(np.sum((v[q]-_closest_square_projection(v[q]))**2)) for q in f)
    reference = np.asarray(state.mesh_3d_initial.vertices)
    edge_targets = {}
    for face in f:
        q = reference[face]
        mean_length = float(np.mean(np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)))
        for a, b in zip(face, np.roll(face, -1)):
            edge_targets.setdefault(tuple(sorted((int(a), int(b)))), []).append(mean_length)
    square_length = sum((float(np.linalg.norm(v[a]-v[b]))-float(np.mean(values)))**2 for (a,b), values in edge_targets.items())
    square = square_shape+square_length
    surface = float(np.sum((v-_surface_project(v, None, state.surface_parameterization))**2))
    m2d = state.mesh_2d_initial
    # Count paired boundary edges by source identity, including separate UV
    # seam charts where exact endpoint correspondence exists.
    from . import onestring_pipeline as pipeline
    canonical_ids = source_equivalence(m2d, state.mesh_3d_initial, state.surface_parameterization, pipeline)
    incidence, pairs = {}, {}
    for face in m2d.faces:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            incidence[key] = incidence.get(key, 0)+1
    for (a, b), count in incidence.items():
        if count == 1:
            key = tuple(sorted((int(canonical_ids[a]), int(canonical_ids[b]))))
            pairs.setdefault(key, []).append((a, b))
    cut_edges = [copies[0] for copies in pairs.values() if len(copies) > 1]
    split_length = sum(float(np.linalg.norm(state.mesh_3d_initial.vertices[a]-state.mesh_3d_initial.vertices[b])) for a, b in cut_edges)
    separation = 0.0
    for group in np.unique(canonical_ids):
        ids = np.flatnonzero(canonical_ids == group)
        if len(ids) > 1:
            separation = max(separation, float(np.max(np.linalg.norm(mesh.vertices[ids]-mesh.vertices[ids[0]], axis=1))))
    t2 = state.tiles_2d_dual_hinge.metrics
    t2_top = state.tiles_2d_top_hinge.metrics
    report = dict(version=VERSION, feature_mask=feature_mask(params),
                  parameters={k: getattr(params, k) for k in (*FEATURES, 'extrusion_weight', 'extrusion_split_weight', 'extrusion_max_nfev', 'thickness')},
                  panel_count=len(f), split_count=len(cut_edges), split_total_length=split_length,
                  assembled_split_boundary_separation_max=separation,
                  split_count_definition='paired source-barycentric boundary edge segments; exact corresponding samples only',
                  split_requested_line_count=len(m2d.split_lines),
                  k3d_planar_error=planar, k3d_square_error=square, k3d_surface_error=surface,
                  k3d_square_error_definition='EShape + ELength using M3D source mean incident edge targets',
                  k3d_square_shape_error=square_shape, k3d_square_length_error=square_length,
                  extrusion_pre_t3d=before, extrusion_final_k3d=quality_summary(quality),
                  **quality_summary(actual),
                  panel_extrusion_difficulty=quality['difficulty'].tolist(),
                  panel_t3d_difficulty=actual['difficulty'].tolist(),
                  t2d_collision_count=t2.get('flat_collision_count', t2.get('tile_overlap_count')),
                  t2d_top_collision_count=t2_top.get('flat_collision_count', t2_top.get('tile_overlap_count')),
                  fabrication_violation_count=t2.get('fabrication_gap_violations'),
                  k2d_collision_count=state.mesh_2d_optimized.metrics.get('collisions'),
                  runtime=runtimes,
                  pipeline_parameters=asdict(params),
                  solver_environment={k: val for k, val in os.environ.items() if k.startswith(('ONESTRING_K3D_', 'ONESTRING_PAPER_K3D_', 'ONESTRING_EQ5_', 'ONESTRING_OPTCUTS_DISTORTION'))},
                  source_surface_sha256=hashlib.sha256(np.asarray(state.target_surface.vertices).tobytes()+np.asarray(state.target_surface.faces).tobytes()).hexdigest(),
                  k3d_extrusion_optimizer={k: val for k,val in mesh.metrics.items() if k.startswith('extrusion_objective') or k.startswith('extrusion_optimizer')},
                  split_candidate_scores=state.surface_parameterization.metrics.get('extrusion_split_candidate_scores', []),
                  principal_grid=state.surface_parameterization.metrics.get('principal_grid_scope', 'disabled'),
                  split_status=state.surface_parameterization.metrics.get('extrusion_split_status', 'disabled'),
                  assembled_equivalence_group_count=mesh.metrics.get('assembled_equivalence_group_count', 0),
                  baseline_equivalence_policy='0000 retains old geometry; nonzero masks enforce source identity',
                  metric_units='length in mesh units, angles in degrees, difficulty dimensionless; K3D errors squared length sums')
    mesh.metrics['extrusion_difficulty_per_panel'] = quality['difficulty'].tolist()
    state.tiles_3d.metrics['extrusion_difficulty_per_panel'] = quality['difficulty'].tolist()
    state.extrusion_experiment = report
    path = Path(params.extrusion_results_dir)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / (time.strftime('%Y%m%d-%H%M%S')+'-'+feature_mask(params)+'-'+uuid.uuid4().hex[:8]+'.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    state.extrusion_experiment_path = str(destination.resolve())
    return report


def difficulty_colors(values):
    # Fixed 0..1 scale supports comparisons across runs; selected pink is applied
    # afterwards by the unchanged linked-panel viewer.
    v = np.clip(np.asarray(values, float), 0, 1)
    stops = np.array([[37, 99, 235], [250, 204, 21], [220, 38, 38]], float)
    return ['rgb(%d,%d,%d)' % tuple(np.rint((1-(x*2)%1)*stops[min(int(x*2), 1)]+((x*2)%1)*stops[min(int(x*2)+1, 2)]).astype(int))
            if x < 1 else 'rgb(220,38,38)' for x in v]
