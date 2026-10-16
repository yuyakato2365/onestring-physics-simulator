"""Curvature-driven tensor-product refinement for the common paper grid route.

Only whole grid lines are inserted, before crop/split or panel IDs exist. This
preserves the rectangular index space and bipartite quad adjacency. Refinement
propagates along a strip; it is deliberately not a local quadtree/remesher.
"""
from __future__ import annotations
import numpy as np
from .extrusion_curvature import principal_curvature

SCOPE = 'curvature-adaptive whole-row/column subdivision; conforming tensor-product quads'


def _axes(domain, grid):
    nx = int(getattr(domain, 'overlay_nx', grid.nx))
    ny = int(getattr(domain, 'overlay_ny', grid.ny))
    uv = np.asarray(domain.uv_vertices, float)
    if nx < 1 or ny < 1 or uv.shape != ((nx+1)*(ny+1), 2) or not np.isfinite(uv).all():
        raise ValueError('Adaptive grid needs a finite structured UV overlay')
    points = uv.reshape(ny+1, nx+1, 2)
    xs, ys = points[0, :, 0].copy(), points[:, 0, 1].copy()
    if (np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0) or
            not np.allclose(points[:, :, 0], xs[None, :], rtol=0, atol=1e-12) or
            not np.allclose(points[:, :, 1], ys[:, None], rtol=0, atol=1e-12)):
        raise ValueError('Adaptive grid requires ordered axis-aligned rows and columns')
    return xs, ys


def _subdivide(axis, factors):
    # Keep every original line exactly, not a reconstructed floating point copy.
    values, parents = [], []
    for i, factor in enumerate(factors):
        values.extend(np.linspace(axis[i], axis[i+1], int(factor)+1)[:-1])
        parents.extend([i]*int(factor))
    return np.asarray([*values, axis[-1]]), np.asarray(parents, int)


def refine_overlay(p, domain, grid, params):
    """Refine strips when estimated normal turn kappa * physical edge exceeds a target.

    Surface triangles supply the physical UV metric. Curvature magnitude uses
    fit reliability, NOT principal-direction anisotropy: umbilical/spherical
    regions can need refinement even though their directions are unreliable.
    """
    method = str(p.method)
    if ('optcuts_grid' in method or 'grid_optcuts' in method or
            method == 'paper_reference_bff' or getattr(domain, 'reference_mode', False)):
        raise NotImplementedError('Adaptive grid cannot alter a native OptCuts/BFF fixed lattice; use the common paper local/global route')
    if hasattr(domain, 'adaptive_grid'):
        raise ValueError('Adaptive refinement must run once before M2D construction')
    xs, ys = _axes(domain, grid)
    v, f = np.asarray(p.surface_vertices_3d, float), np.asarray(p.surface_faces, int)
    uv, uf = np.asarray(p.uv_vertices_2d, float), np.asarray(p.uv_faces, int)
    if f.ndim != 2 or f.shape[1] != 3 or f.shape != uf.shape:
        raise ValueError('Adaptive grid needs paired source/UV triangles')
    if not np.isfinite(v).all() or not np.isfinite(uv).all():
        raise ValueError('Adaptive grid received nonfinite parameterization')
    cached = getattr(p, '_extrusion_curvature_fit', None)
    if cached is None:
        k, _, _, _, reliability = principal_curvature(v, f, return_fit_confidence=True)
    else:
        k, reliability = cached
    curvature = np.max(np.abs(k), axis=1)*reliability
    turn_u, turn_v = np.zeros(len(xs)-1), np.zeros(len(ys)-1)
    skipped = 0
    for sf, tf in zip(f, uf):
        tri = uv[tf]
        duv = (tri[1:]-tri[0]).T
        singular = np.linalg.svd(duv, compute_uv=False)
        if singular[0] <= 0 or singular[-1] <= singular[0]*1e-10:
            skipped += 1
            continue
        metric = (v[sf[1:]]-v[sf[0]]).T @ np.linalg.inv(duv)
        kappa = float(np.max(curvature[sf]))
        if kappa == 0:
            continue
        for coord, axis, turns in ((0, xs, turn_u), (1, ys, turn_v)):
            low, high = float(tri[:, coord].min()), float(tri[:, coord].max())
            # Intersect open intervals, so a triangle ending on a grid line does
            # not refine the next unrelated strip just by touching its endpoint.
            first = max(0, int(np.searchsorted(axis, low, side='right'))-1)
            last = min(len(axis)-1, int(np.searchsorted(axis, high, side='left')))
            if last > first:
                estimate = kappa*np.linalg.norm(metric[:, coord])*np.diff(axis)[first:last]
                turns[first:last] = np.maximum(turns[first:last], estimate)
    tolerance = np.radians(float(params.adaptive_normal_turn_degrees))
    maximum = int(params.adaptive_max_subdivisions)
    factors = [np.clip(np.ceil(turns/tolerance-1e-12), 1, maximum).astype(int) for turns in (turn_u, turn_v)]
    # Preserve detected reflection symmetry only where the original axis itself
    # is symmetric. Whole strips, unlike local cells, remain conforming.
    for coord, axis in enumerate((xs, ys)):
        centers = getattr(domain, 'detected_symmetry_centers', {})
        if coord in getattr(domain, 'detected_symmetry_axes', []):
            center = float(centers.get(coord, 0.))
            if np.allclose(axis, 2*center-axis[::-1], rtol=0, atol=1e-9):
                factors[coord] = np.maximum(factors[coord], factors[coord][::-1])
    count = int(factors[0].sum()*factors[1].sum())
    if count > int(params.adaptive_max_cells):
        raise ValueError(f'Adaptive grid requires {count} overlay cells; budget is {params.adaptive_max_cells}. Increase the budget or normal-turn target.')
    new_x, parent_x = _subdivide(xs, factors[0])
    new_y, parent_y = _subdivide(ys, factors[1])
    nx, ny = len(new_x)-1, len(new_y)-1
    parent_cells = (parent_y[:, None]*(len(xs)-1)+parent_x[None, :]).ravel()
    metrics = dict(scope=SCOPE, base_nx=len(xs)-1, base_ny=len(ys)-1, nx=nx, ny=ny,
                   base_overlay_cells=(len(xs)-1)*(len(ys)-1), overlay_cells=count,
                   u_subdivisions=factors[0].tolist(), v_subdivisions=factors[1].tolist(),
                   u_lines=new_x.tolist(), v_lines=new_y.tolist(),
                   estimated_turn_u_deg=np.degrees(turn_u).tolist(),
                   estimated_turn_v_deg=np.degrees(turn_v).tolist(),
                   target_normal_turn_deg=float(params.adaptive_normal_turn_degrees),
                   capped_strip_count=sum(int(np.count_nonzero(t/tolerance > maximum+1e-12)) for t in (turn_u, turn_v)),
                   skipped_degenerate_uv_triangles=skipped,
                   curvature_fit_confidence_mean=float(np.mean(reliability)),
                   base_cell_per_overlay_panel=parent_cells.tolist())
    # Do not perturb a flat/low-curvature overlay even at floating point level.
    if np.any(factors[0] > 1) or np.any(factors[1] > 1):
        xx, yy = np.meshgrid(new_x, new_y, indexing='xy')
        domain.uv_vertices = np.column_stack((xx.ravel(), yy.ravel()))
    domain.overlay_nx, domain.overlay_ny = nx, ny
    domain.adaptive_grid = metrics
    domain.overlay_step_u = float(np.min(np.diff(new_x)))
    domain.overlay_step_v = float(np.min(np.diff(new_y)))
    # Explicit nonuniform coordinates are authoritative; scalar steps are only
    # conservative legacy diagnostic scales and must not rebuild the grid.
    p.metrics['adaptive_grid'] = metrics
    return metrics


def attach_panel_correspondence(mesh, domain):
    """Audit the actual cropped/split M2D, assigning parent IDs without reordering."""
    data = domain.adaptive_grid
    canonical = np.asarray(getattr(mesh, '_split_panel_source_vertices', mesh.vertices))[:, :2]
    faces = np.asarray(mesh.faces, int)
    xs, ys = np.asarray(data['u_lines']), np.asarray(data['v_lines'])
    q = canonical[faces]
    centers = q.mean(axis=1)
    cols = np.searchsorted(xs, centers[:, 0], side='right')-1
    rows = np.searchsorted(ys, centers[:, 1], side='right')-1
    if (np.any(cols < 0) or np.any(cols >= data['nx']) or np.any(rows < 0) or np.any(rows >= data['ny'])):
        raise ValueError('Adaptive M2D correspondence escaped the overlay')
    expected = np.stack((np.column_stack((xs[cols], ys[rows])),
                         np.column_stack((xs[cols+1], ys[rows])),
                         np.column_stack((xs[cols+1], ys[rows+1])),
                         np.column_stack((xs[cols], ys[rows+1]))), axis=1)
    if not np.allclose(q, expected, rtol=0, atol=max(np.ptp(canonical, axis=0).max()*1e-9, 1e-10)):
        raise ValueError('An M2D wrapper changed adaptive grid coordinates or quad ordering')
    overlay_ids = rows*data['nx']+cols
    if len(np.unique(overlay_ids)) != len(faces):
        raise ValueError('Adaptive M2D duplicated a panel')
    if mesh.grid.nx != data['nx'] or mesh.grid.ny != data['ny']:
        raise ValueError('Adaptive M2D grid indexing is inconsistent')
    # Keep the full uncut grid separate from duplicate cut vertex IDs. The
    # paper linkage builder uses mesh.faces for adjacency and the grid width
    # for checkerboard phase, not uniform spacing.
    mesh.grid.vertex_positions = np.column_stack((domain.uv_vertices, np.zeros(len(domain.uv_vertices))))
    mesh.metrics['adaptive_grid'] = data
    mesh.metrics['adaptive_overlay_panel_ids'] = overlay_ids.tolist()
    mesh.metrics['adaptive_base_cell_per_panel'] = np.asarray(data['base_cell_per_overlay_panel'])[overlay_ids].tolist()
