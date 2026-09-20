"""OneString Sec. 4.3/4.4 flat construction, with explicit approximation labels.

The paper specifies rigid top->bottom transforms but does not specify how to
fit them when the four corresponding points are not congruent (a general
frustum). We use proper least-squares rigid fits and report their residuals;
nonzero residuals are never claimed to reproduce the full T3D solid exactly.
"""
from __future__ import annotations

import time
import numpy as np


def rigid_fit(source, target):
    """Column-vector homogeneous transform; no scale, shear or reflection."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    r = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = r
    transform[:3, 3] = bc-r@ac
    return transform


def apply_transform(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def make_t2d(mesh, layout, k3d, t3d, stage, pipeline):
    started = time.perf_counter()
    topology = mesh.linkage_topology
    tops = np.asarray(mesh.vertices, float)[mesh.faces].copy()
    solids = np.asarray(t3d.vertices, float)
    n = len(tops)
    if solids.shape != (n, 8, 3) or not np.isfinite(solids).all() or not np.isfinite(tops).all():
        raise ValueError('T2D requires finite corresponding K2D quads and 8-vertex T3D tiles')
    if list(layout.tile_ids) != list(range(n)) or not np.array_equal(topology.source_faces, k3d.faces):
        raise ValueError('K2D/T3D tile identity or source-face correspondence changed')
    vertices = np.empty_like(solids)
    placements, bottom_transforms, fit_errors = [], [], []
    for i, solid in enumerate(solids):
        placement = rigid_fit(solid[:4], tops[i])
        top_to_bottom = rigid_fit(solid[:4], solid[4:])
        flat_transform = placement @ top_to_bottom @ np.linalg.inv(placement)
        vertices[i, :4] = tops[i]  # Sec. 4.3: retain the optimized K2D vertices.
        vertices[i, 4:] = apply_transform(tops[i], flat_transform)
        fit_errors.append(np.max(np.linalg.norm(apply_transform(solid[:4], top_to_bottom)-solid[4:], axis=1)))
        placements.append(placement)
        bottom_transforms.append(flat_transform)
    shape_error = pipeline._tile_shape_distance_error(vertices, solids, use_max=True)
    scale = float(np.median(np.linalg.norm(np.roll(tops, -1, axis=1)-tops, axis=2)))
    metrics = dict(
        objective='Sec. 4.3 top-to-bottom rigid transform in the K2D tile frame',
        section_4_3_linkage=True, tile_ids=list(range(n)),
        top_vertices_match_k2d_max_error=0., top_vertices_match_k2d_rms_error=0.,
        tile_shape_max_error_to_T3D=float(shape_error),
        tile_shape_rms_error_to_T3D=pipeline._tile_shape_distance_error(vertices, solids),
        t2d_t3d_congruent_tile_geometry=bool(shape_error <= scale*1e-6),
        top_to_bottom_rigid_fit_max_error=float(max(fit_errors, default=0.)),
        transform_matrices_semantics='T3D-to-K2D reference frames, not an exact whole-solid mapping when congruence residual is nonzero',
        face_planarity_error=pipeline._tile_face_planarity(vertices),
        t2d_layout_optimization_deferred_to_eq6=True,
        t2d_top_hinge_overlap_trimming_implemented=False,
        paper_classification={
            'K2D_top_vertices_and_tile_identity': 'Paper-exact',
            'top_to_bottom_rigid_transform_operation': 'Paper-exact',
            'least_squares_frame_alignment_and_fit_for_noncongruent_faces': 'Paper-consistent approximation',
            'signed_discrete_normal_curvature': 'Paper-consistent approximation',
            'rigid_SE2_and_convex_footprint_collision': 'Paper-consistent approximation',
            'untrimmed_top_hinge_preview': 'Project-specific extension',
        },
    )
    result = pipeline.TileAssembly(vertices, t3d.top_faces.copy(), t3d.bottom_faces.copy(),
                                   t3d.side_faces.copy(), stage, metrics, np.asarray(placements))
    result.linkage_topology = topology
    result.top_to_bottom_transforms = np.asarray(bottom_transforms)
    result.k2d_reference_tops = tops.copy()
    result.theta_min_deg = float(mesh.metrics.get('theta_min_deg', 5.))
    result.k2d_fabrication_feasible = bool(mesh.metrics.get('fabrication_feasible', False))
    result.metrics.update(validate_flat_linkage(result, t3d, pipeline))
    report = pipeline.StageReport(name='K2D -> T2D top hinge', objective=metrics['objective'],
                                  before_error=0., after_error=float(shape_error),
                                  constraint_violation=float(shape_error),
                                  computation_time=time.perf_counter()-started,
                                  counts=pipeline._assembly_counts(result))
    return result, report


def build_hinge_graph(t2d, t3d, pipeline, *, dual):
    """Keep the exact K2D corner pairs; choose surfaces from signed curvature."""
    vertices = np.asarray(t3d.vertices, float)
    centers = vertices[:, :4].mean(axis=1)
    normals = []
    for tile in vertices:
        _, _, vh = np.linalg.svd(tile[:4]-tile[:4].mean(axis=0))
        normal = vh[-1]
        # Outward normal points away from the extruded bottom, independent of winding.
        if normal @ (tile[:4].mean(axis=0)-tile[4:].mean(axis=0)) < 0:
            normal = -normal
        normals.append(normal)
    normals = np.asarray(normals)
    hinges, curvature = [], []
    for a, ca, b, cb in t2d.linkage_topology.hinges:
        direction = centers[b]-centers[a]
        value = float((normals[b]-normals[a]) @ direction / max(direction@direction, 1e-30))
        curvature.append(value)
        offset = 4 if dual and value < -1e-10 else 0
        va, vb = int(ca+offset), int(cb+offset)
        rest = .5*(t2d.vertices[a, va]+t2d.vertices[b, vb])
        target = .5*(t3d.vertices[a, va]+t3d.vertices[b, vb])
        hinges.append(pipeline.Hinge(int(a), int(b), va, vb, 'bottom' if offset else 'top', rest, target))
    return pipeline.HingeGraph(hinges, dict(
        hinge_topology='explicit K2D pairwise joints; no re-inference or split welding',
        hinge_selection_classification='Paper-consistent approximation',
        hinge_selection_model='signed normal variation in the tile-center direction; negative selects bottom',
        signed_normal_curvature=curvature,
        top_hinge_count=sum(h.surface == 'top' for h in hinges),
        bottom_hinge_count=sum(h.surface == 'bottom' for h in hinges),
    ))


def validate_flat_linkage(t2d, t3d, pipeline, graph=None):
    """Independent output checks: a finite penalty is not a feasibility certificate."""
    from scipy.spatial import ConvexHull
    from .paper_eq5_k2d_solver import collision_energy_gradient
    vertices = np.asarray(t2d.vertices, float)
    scale = float(np.median(np.linalg.norm(np.roll(vertices[:, :4], -1, axis=1)-vertices[:, :4], axis=2)))
    hulls = [ConvexHull(tile[:, :2]).vertices.tolist() for tile in vertices]
    width = max(map(len, hulls))
    faces = np.array([[8*i+j for j in h+[h[-1]]*(width-len(h))] for i, h in enumerate(hulls)])
    _, _, collisions = collision_energy_gradient(vertices[:, :, :2].reshape(-1, 2), faces, scale*1e-8)
    if graph is None:
        graph = build_hinge_graph(t2d, t3d, pipeline, dual=False)
    errors, angles = [], []
    source = t2d.linkage_topology.source_faces
    for h in graph.hinges:
        a, b = h.tile_a, h.tile_b
        ca, cb = h.local_vertex_a % 4, h.local_vertex_b % 4
        offset = 4 if h.surface == 'bottom' else 0
        errors.append(float(np.linalg.norm(vertices[a, ca+offset]-vertices[b, cb+offset])))
        shared = set(source[a]) & set(source[b])
        other = next(v for v in shared if v != source[a, ca])
        ia, ib = int(np.flatnonzero(source[a] == other)[0]), int(np.flatnonzero(source[b] == other)[0])
        u = vertices[a, ia+offset]-vertices[a, ca+offset]
        v = vertices[b, ib+offset]-vertices[b, cb+offset]
        angles.append(float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u,v)), u@v))))
    violations = sum(a < t2d.theta_min_deg-1e-5 or a > 90.+1e-5 for a in angles)
    error = max(errors, default=0.)
    shape_error = pipeline._tile_shape_distance_error(vertices, t3d.vertices, use_max=True)
    before = np.linalg.norm(np.roll(t2d.k2d_reference_tops, -1, axis=1)-t2d.k2d_reference_tops, axis=2)
    after = np.linalg.norm(np.roll(vertices[:, :4], -1, axis=1)-vertices[:, :4], axis=2)
    feasible = collisions == 0 and violations == 0 and error < scale*1e-4 and shape_error < scale*1e-6 and t2d.k2d_fabrication_feasible
    return dict(flat_collision_count=int(collisions), fabrication_gap_violations=int(violations),
                fabrication_gap_min_deg=min(angles, default=0.), fabrication_gap_max_deg=max(angles, default=0.),
                hinge_connection_max_error=float(error), t2d_edge_length_max_error_to_k2d=float(np.max(np.abs(before-after))),
                tile_shape_max_error_to_T3D=float(shape_error), fabrication_feasible=bool(feasible),
                fabrication_validation='conservative convex XY footprints; gap angles; hinge coincidence; congruence; K2D feasibility',
                fabrication_status='validated_flat_candidate' if feasible else 'constraints_unresolved')


def build_gap_graph(t2d, t3d, pipeline):
    """Trace actual void boundaries through the explicit corner joints.

    Each reverse tile edge has one successor: at a hinge continue on the
    partner tile, otherwise on the same tile. This partitions the outside
    half-edges into bounded voids and exterior boundaries, without inventing
    neighbours from tile numbering or welding coincident split vertices.
    Sec. 5.2's shared-tile adjacency is used by the existing routing interface.
    Label orientation and virtual boundary sampling remain approximations.
    """
    partners = {}
    for a, ca, b, cb in t2d.linkage_topology.hinges:
        partners[int(a), int(ca)] = (int(b), int(cb))
        partners[int(b), int(cb)] = (int(a), int(ca))
    unseen = {(a, c) for a in range(len(t2d.vertices)) for c in range(4)}
    cycles = []
    while unseen:
        start = min(unseen)
        dart, cycle = start, []
        while dart in unseen:
            unseen.remove(dart)
            cycle.append(dart)
            a, c = dart
            endpoint = (a, (c-1) % 4)
            dart = partners.get(endpoint, endpoint)
        if dart != start:
            raise ValueError('Invalid corner-joint boundary permutation')
        cycles.append(cycle)
    gaps, tile_to_gaps = [], {}
    degenerate = 0
    scale = float(np.ptp(t2d.vertices[:, :4, :2], axis=(0, 1)).max())
    def add(corners, boundary):
        tiles = sorted({a for a, _ in corners})
        p2 = np.array([t2d.vertices[a, c] for a, c in corners])
        p3 = np.array([t3d.vertices[a, c] for a, c in corners])
        if boundary:
            kind, label = 'virtual_boundary', -1
        else:
            # This geometric orientation convention is not the authors' code.
            extent = np.ptp(p2[:, :2], axis=0)
            kind, label = ('vertical', 0) if extent[1] >= extent[0] else ('horizontal', 1)
        gap = pipeline.Gap(len(gaps), tiles, p2.mean(axis=0), p3.mean(axis=0), kind, boundary, label)
        gaps.append(gap)
        for tile in tiles:
            tile_to_gaps.setdefault(tile, []).append(gap.id)
    for cycle in cycles:
        points = np.array([t2d.vertices[a, c, :2] for a, c in cycle])
        area = .5*np.sum(points[:, 0]*np.roll(points[:, 1], -1)-points[:, 1]*np.roll(points[:, 0], -1))
        if area > scale**2*1e-12:
            add(cycle, False)
        elif area < -scale**2*1e-12:
            for a, c in cycle:
                add([(a, c), (a, (c-1) % 4)], True)
        else:
            degenerate += 1
    edges = set()
    for incident in tile_to_gaps.values():
        for i, a in enumerate(incident):
            edges.update(tuple(sorted((a, b))) for b in incident[i+1:])
    adjacency = {i: set() for i in range(len(gaps))}
    for a, b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    pending = set(adjacency); components = 0
    while pending:
        components += 1
        stack = [pending.pop()]
        while stack:
            for b in adjacency[stack.pop()] & pending:
                pending.remove(b); stack.append(b)
    zmin = float(np.min(t3d.vertices[..., 2]))
    z = np.mean(t3d.vertices[:, :4, 2], axis=1)
    for gap in gaps:
        gap.gpe = float(.25*9.81*np.sum(z[gap.surrounding_tiles]-zmin))
    return pipeline.GapGraph(gaps, sorted(edges), dict(
        gap_count=len(gaps), edge_count=len(edges),
        boundary_gap_count=sum(g.boundary for g in gaps), split_boundary_gap_count=0,
        max_gpe=max((g.gpe for g in gaps), default=0.),
        gap_graph_algorithm='explicit corner-joint void boundary cycles; shared-tile adjacency (Sec. 5.2)',
        gap_graph_components=components, string_graph_connected=components == 1,
        degenerate_void_cycles=degenerate,
        gap_geometry_validated=bool(t2d.metrics.get('fabrication_feasible', False)) and degenerate == 0,
        gap_graph_split_virtual_weld_applied=False,
        paper_classification={
            'bounded_void_nodes_and_shared_tile_adjacency': 'Paper-exact',
            'boundary_cycle_extraction_and_orientation_labels': 'Paper-consistent approximation',
            'virtual_boundary_sampling_and_existing_string_route_heuristic': 'Project-specific extension',
        },
        gap_graph_limitations='split boundary labels (-2) and Sec. 5.3 routing MILP are not implemented; disconnected components are not bridged',
    ))
