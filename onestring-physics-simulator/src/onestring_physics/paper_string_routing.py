"""Sec. 5.3 closed route with geometric turn costs and split-entry constraints.

MILP is our discretization of the paper's stated optimization (the paper does
not prescribe a solver). A directed cycle uses each visited gap once, covers
every boundary tile and lift gap, and connects all selected nodes using a
single-commodity flow. Turn variables price predecessor/current/successor
triples, including the closing turn. Infeasibility never becomes a jump edge.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def build_string_path(graph, lift_points, mu_c, pipeline):
    gaps = {int(g.id): g for g in graph.gaps}
    boundary = sorted(i for i, g in gaps.items() if g.boundary)
    lifts = sorted({int(p.gap_id) for p in lift_points})
    metrics = dict(string_path_model='Sec. 5.3 minimum-turn closed-cycle MILP',
                   string_path_exactness='discrete graph optimization; solver status reported',
                   string_path_valid=False, lift_gap_ids=lifts, boundary_gap_count=len(boundary))
    def failed(reason):
        metrics.update(string_path_status=reason, route_node_count=0, warnings=reason)
        return pipeline.StringPath([], boundary, lifts, 0., 0., metrics)
    if not gaps:
        return failed('empty_gap_graph')
    if graph.metrics.get('split_boundary_missing_pairs'):
        return failed('split_boundary_correspondence_missing')
    if set(lifts)-set(gaps):
        return failed('unknown_lift_gap')
    edges = graph.metrics.get('routing_edges', graph.edges)
    adjacency = {i: set() for i in gaps}
    for a, b in edges:
        if a == b or a not in gaps or b not in gaps:
            continue
        # A split boundary may traverse another boundary or its matched seam,
        # but must never be used as an entrance to an interior gap.
        if (gaps[a].label == -2 and not gaps[b].boundary) or (gaps[b].label == -2 and not gaps[a].boundary):
            continue
        adjacency[a].add(b); adjacency[b].add(a)
    if not boundary:
        return failed('no_boundary_anchor')
    required_tiles = set(graph.metrics.get('boundary_tile_ids', []))
    if not required_tiles:
        required_tiles = {t for i in boundary for t in gaps[i].surrounding_tiles}
    nodes = sorted(gaps)
    arcs = [(i, j) for i in nodes for j in sorted(adjacency[i])]
    triples = [(i, j, k) for j in nodes for i in sorted(adjacency[j]) for k in sorted(adjacency[j]) if i != k]
    if not triples:
        return failed('no_closed_cycle')
    objective, integral, lower, upper = [], [], [], []
    def var(cost=0., binary=True, ub=1.):
        i = len(objective)
        objective.append(cost); integral.append(int(binary)); lower.append(0.); upper.append(ub)
        return i
    z = {i: var() for i in nodes}
    x = {e: var() for e in arcs}
    turns = {}
    for i, j, k in triples:
        a = np.asarray(gaps[j].centroid_2d)-gaps[i].centroid_2d
        b = np.asarray(gaps[k].centroid_2d)-gaps[j].centroid_2d
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        # Coincident gap centroids do not define a channel direction.
        angle = math.acos(float(np.clip(a@b/denom, -1., 1.))) if denom > 1e-15 else math.pi
        turns[i, j, k] = var(angle)
    flow = {e: var(binary=False, ub=len(nodes)) for e in arcs}
    anchor = {i: var() for i in boundary}
    supply = {i: var(binary=False, ub=len(nodes)) for i in boundary}
    rows, cols, vals, lbs, ubs = [], [], [], [], []
    def constraint(terms, lb=0., ub=0.):
        row = len(lbs)
        for index, value in terms:
            rows.append(row); cols.append(index); vals.append(value)
        lbs.append(lb); ubs.append(ub)
    for j in nodes:
        constraint([(x[i, j], 1.) for i in adjacency[j]]+[(z[j], -1.)])
        constraint([(x[j, k], 1.) for k in adjacency[j]]+[(z[j], -1.)])
        for i in adjacency[j]:
            constraint([(turns[i, j, k], 1.) for k in adjacency[j] if k != i]+[(x[i, j], -1.)])
        for k in adjacency[j]:
            constraint([(turns[i, j, k], 1.) for i in adjacency[j] if i != k]+[(x[j, k], -1.)])
        terms = [(flow[i, j], 1.) for i in adjacency[j]]+[(flow[j, k], -1.) for k in adjacency[j]]
        terms.append((z[j], -1.))
        if j in supply:
            terms.append((supply[j], 1.))
        constraint(terms)
    for e in arcs:
        constraint([(flow[e], 1.), (x[e], -len(nodes))], -np.inf, 0.)
    constraint([(index, 1.) for index in anchor.values()], 1., 1.)
    for i in boundary:
        constraint([(anchor[i], 1.), (z[i], -1.)], -np.inf, 0.)
        constraint([(supply[i], 1.), (anchor[i], -len(nodes))], -np.inf, 0.)
    for i in set(lifts):
        constraint([(z[i], 1.)], 1., 1.)
    for tile in sorted(required_tiles):
        constraint([(z[i], 1.) for i in nodes if gaps[i].boundary and tile in gaps[i].surrounding_tiles], 1., np.inf)
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lbs), len(objective))).tocsc()
    result = milp(np.asarray(objective), integrality=np.asarray(integral), bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, lbs, ubs), options={'time_limit': 30., 'mip_rel_gap': 0.})
    metrics.update(string_path_solver_status=int(result.status), string_path_solver_message=str(result.message),
                   string_path_optimal=bool(result.status == 0), string_path_solver='scipy.optimize.milp / HiGHS',
                   string_path_anchor_selection='optimized over boundary gaps')
    if result.x is None:
        return failed('route_infeasible' if result.status == 2 else 'solver_no_feasible_incumbent')
    root = next((i for i, index in anchor.items() if result.x[index] > .5), None)
    if root is None:
        return failed('invalid_solver_anchor')
    metrics['string_path_anchor_gap_id'] = root
    successor = {i: j for (i, j), index in x.items() if result.x[index] > .5}
    route = [root]
    for _ in range(len(nodes)):
        nxt = successor.get(route[-1])
        if nxt is None:
            return failed('invalid_solver_incumbent')
        route.append(nxt)
        if nxt == root:
            break
        if nxt in route[:-1]:
            return failed('invalid_subtour')
    visited = set(route[:-1])
    covered = {t for i in visited if gaps[i].boundary for t in gaps[i].surrounding_tiles}
    if route[-1] != root or len(visited) != len(successor) or not set(lifts) <= visited or not required_tiles <= covered:
        return failed('incomplete_route')
    cycle = route[:-1]
    theta = sum(objective[turns[cycle[k-1], cycle[k], cycle[(k+1)%len(cycle)]]] for k in range(len(cycle)))
    friction = pipeline.safe_capstan_friction(mu_c, theta)
    metrics.update(string_path_valid=True, string_path_status='optimal' if result.status == 0 else 'feasible_incumbent',
                   route_length=len(route), route_node_count=len(route), unique_route_node_count=len(cycle),
                   duplicate_visit_count=0, closure_repeat_count=1, turn_angle_total=theta, theta_total=theta,
                   log_channel_cost=mu_c*theta, estimated_channel_friction=friction,
                   boundary_tiles_covered=len(covered), boundary_tiles_required=len(required_tiles),
                   split_boundary_interior_entry_count=0, string_path_closed=True,
                   string_path_all_components_connected=True, warnings='')
    return pipeline.StringPath(route, boundary, lifts, float(theta), float(friction), metrics)
