"""Bijective free-boundary surface parameterization for one disk chart.

The implementation is deliberately independent from the existing BFF and CEPS
backends.  It is *Smith & Schaefer 2015-inspired*, not a claim of reproducing
their complete implementation.  It follows three ideas from the paper:

* start from a valid convex-boundary Tutte embedding;
* minimize an area-weighted symmetric Dirichlet/isometric distortion whose
  inverse term diverges as a triangle degenerates; and
* restrict every line-search step to remain before triangle degeneracy and
  moving boundary edge/vertex collisions.

Unlike the paper, this compact research implementation uses an O(m^2)
boundary search instead of a spatial hash and a small custom L-BFGS loop.  It
also validates global triangle overlap before accepting every update.  No seam,
cut, topology, BFF, CEPS, lambda, or OneString grid-loss optimization is added.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from .reference_bff import count_internal_triangle_overlaps, triangle_jacobian_diagnostics


_EPS = 1.0e-12


@dataclass(frozen=True)
class BijectiveFreeBoundaryConfig:
    """Numerical controls for the isolated free-boundary experiment."""

    max_iterations: int = 60
    gradient_tolerance: float = 1.0e-7
    relative_energy_tolerance: float = 1.0e-8
    line_search_max_steps: int = 20
    line_search_safety: float = 0.8
    boundary_barrier_weight: float = 1.0
    lbfgs_history_size: int = 8
    minimum_signed_double_area: float = 1.0e-12


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _extract_single_disk_boundary(
    faces: np.ndarray,
    vertex_count: int,
) -> tuple[list[int], dict[str, int]]:
    """Validate one connected disk and return its existing boundary loop."""

    tris = np.asarray(faces, dtype=int)
    if tris.ndim != 2 or tris.shape[1] < 3 or len(tris) < 2:
        raise RuntimeError("bijective free-boundary parameterization requires triangle faces")
    tris = tris[:, :3]
    edge_counts: dict[tuple[int, int], int] = {}
    directed_edges: list[tuple[int, int]] = []
    graph: list[set[int]] = [set() for _ in range(vertex_count)]
    active: set[int] = set()
    for face in tris:
        ids = [int(value) for value in face]
        if len(set(ids)) != 3:
            raise RuntimeError("bijective free-boundary parameterization rejects degenerate input faces")
        if any(value < 0 or value >= vertex_count for value in ids):
            raise RuntimeError("bijective free-boundary mesh contains an out-of-range vertex index")
        active.update(ids)
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            directed_edges.append((a, b))
            graph[a].add(b)
            graph[b].add(a)
    if len(active) != vertex_count:
        raise RuntimeError("bijective free-boundary parameterization rejects isolated vertices")
    if any(count > 2 for count in edge_counts.values()):
        raise RuntimeError("bijective free-boundary parameterization requires a manifold disk")

    stack = [min(active)]
    reached: set[int] = set()
    while stack:
        vertex = stack.pop()
        if vertex in reached:
            continue
        reached.add(vertex)
        stack.extend(graph[vertex] - reached)
    if reached != active:
        raise RuntimeError("bijective free-boundary parameterization requires one connected disk")

    boundary_edges = [
        (a, b)
        for a, b in directed_edges
        if edge_counts[(min(a, b), max(a, b))] == 1
    ]
    if len(boundary_edges) < 3:
        raise RuntimeError("bijective free-boundary parameterization requires one open boundary loop")
    boundary_vertices = {value for edge in boundary_edges for value in edge}
    adjacency: dict[int, list[int]] = {value: [] for value in boundary_vertices}
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("bijective free-boundary parameterization requires one simple boundary loop")

    directed_next: dict[int, int] = {}
    directed_valid = True
    for a, b in boundary_edges:
        if a in directed_next:
            directed_valid = False
            break
        directed_next[a] = b
    loop: list[int] = []
    start = min(boundary_vertices)
    if directed_valid and set(directed_next) == boundary_vertices:
        current = start
        for _ in range(len(boundary_vertices)):
            if current in loop:
                break
            loop.append(current)
            current = directed_next[current]
        if current != start or len(loop) != len(boundary_vertices):
            loop = []
    if not loop:
        previous = -1
        current = start
        for _ in range(len(boundary_vertices)):
            loop.append(current)
            candidates = [value for value in adjacency[current] if value != previous]
            if not candidates:
                break
            following = candidates[0]
            previous, current = current, following
        if current != start or len(loop) != len(boundary_vertices):
            raise RuntimeError("mesh has multiple boundary loops")

    edge_count = len(edge_counts)
    chi = len(active) - edge_count + len(tris)
    if chi != 1:
        raise RuntimeError(
            f"bijective free-boundary parameterization requires one connected disk; got chi={chi}"
        )
    return loop, {
        "topology_vertex_count": len(active),
        "topology_edge_count": edge_count,
        "topology_face_count": len(tris),
        "topology_boundary_count": 1,
        "topology_euler_characteristic": chi,
    }


def _tutte_embedding(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
) -> np.ndarray:
    """Create a valid convex-boundary embedding with positive uniform weights."""

    try:
        from scipy import sparse
        from scipy.sparse import linalg as spla
    except Exception as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("scipy sparse is required for the Tutte initialization") from exc

    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    n = len(xyz)
    loop = np.asarray(boundary_loop, dtype=int)
    next_loop = np.roll(loop, -1)
    lengths = np.linalg.norm(xyz[next_loop] - xyz[loop], axis=1)
    perimeter = float(np.sum(lengths))
    if not np.isfinite(perimeter) or perimeter <= _EPS:
        raise RuntimeError("the input boundary has zero 3D perimeter")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
    angles = 2.0 * np.pi * cumulative / perimeter
    radius = perimeter / (2.0 * np.pi)
    uv = np.zeros((n, 2), dtype=float)
    uv[loop] = radius * np.column_stack([np.cos(angles), np.sin(angles)])

    neighbors: list[set[int]] = [set() for _ in range(n)]
    for face in tris:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            neighbors[int(a)].add(int(b))
            neighbors[int(b)].add(int(a))
    boundary_set = set(int(value) for value in loop)
    interior = [value for value in range(n) if value not in boundary_set]
    if interior:
        local = {vertex: index for index, vertex in enumerate(interior)}
        matrix = sparse.lil_matrix((len(interior), len(interior)), dtype=float)
        rhs = np.zeros((len(interior), 2), dtype=float)
        for vertex in interior:
            row = local[vertex]
            degree = len(neighbors[vertex])
            if degree < 2:
                raise RuntimeError("Tutte initialization found an invalid interior vertex")
            matrix[row, row] = float(degree)
            for neighbor in neighbors[vertex]:
                if neighbor in local:
                    matrix[row, local[neighbor]] -= 1.0
                else:
                    rhs[row] += uv[neighbor]
        uv[np.asarray(interior, dtype=int)] = spla.spsolve(matrix.tocsr(), rhs)

    signed = _signed_double_areas(uv, tris)
    if np.all(signed < -1.0e-12):
        uv[:, 1] *= -1.0
        signed *= -1.0
    if np.any(signed <= 1.0e-12) or not np.all(np.isfinite(uv)):
        raise RuntimeError("Tutte initialization did not produce a strictly valid disk embedding")
    uv -= np.mean(uv, axis=0)
    return uv


def _surface_differentials(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    inverse_surface = np.zeros((len(tris), 2, 2), dtype=float)
    surface_areas = np.zeros(len(tris), dtype=float)
    for face_id, face in enumerate(tris):
        p0, p1, p2 = xyz[face]
        e1 = p1 - p0
        e2 = p2 - p0
        length = float(np.linalg.norm(e1))
        normal = np.cross(e1, e2)
        doubled_area = float(np.linalg.norm(normal))
        if length <= _EPS or doubled_area <= 2.0 * _EPS:
            raise RuntimeError("bijective free-boundary parameterization rejects degenerate 3D triangles")
        x2 = float(np.dot(e2, e1 / length))
        y2 = doubled_area / length
        d_surface = np.asarray([[length, x2], [0.0, y2]], dtype=float)
        inverse_surface[face_id] = np.linalg.inv(d_surface)
        surface_areas[face_id] = 0.5 * doubled_area
    return inverse_surface, surface_areas


def _point_segment_distance_gradient(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    edge = second - first
    squared = float(np.dot(edge, edge))
    if squared <= _EPS:
        delta = point - first
        distance = max(float(np.linalg.norm(delta)), _EPS)
        normal = delta / distance
        return distance, normal, -0.5 * normal, -0.5 * normal
    fraction = float(np.clip(np.dot(point - first, edge) / squared, 0.0, 1.0))
    closest = first + fraction * edge
    delta = point - closest
    distance = max(float(np.linalg.norm(delta)), _EPS)
    normal = delta / distance
    # The closest-point envelope theorem removes the derivative of fraction.
    return distance, normal, -(1.0 - fraction) * normal, -fraction * normal


def _energy_and_gradient(
    uv: np.ndarray,
    faces: np.ndarray,
    inverse_surface: np.ndarray,
    surface_areas: np.ndarray,
    boundary_loop: list[int],
    barrier_epsilon: float,
    barrier_weight: float,
) -> tuple[float, np.ndarray, float, float]:
    tris = np.asarray(faces, dtype=int)[:, :3]
    gradient = np.zeros_like(uv)
    distortion = 0.0
    for face_id, face in enumerate(tris):
        q0, q1, q2 = uv[face]
        d_uv = np.column_stack([q1 - q0, q2 - q0])
        jacobian = d_uv @ inverse_surface[face_id]
        determinant = float(np.linalg.det(jacobian))
        if determinant <= _EPS or not np.isfinite(determinant):
            return math.inf, gradient, math.inf, math.inf
        inverse_jacobian = np.linalg.inv(jacobian)
        area = float(surface_areas[face_id])
        face_energy = area * (
            float(np.sum(jacobian * jacobian))
            + float(np.sum(inverse_jacobian * inverse_jacobian))
        )
        distortion += face_energy
        jacobian_gradient = 2.0 * area * (
            jacobian
            - inverse_jacobian.T @ inverse_jacobian @ inverse_jacobian.T
        )
        uv_gradient = jacobian_gradient @ inverse_surface[face_id].T
        gradient[int(face[1])] += uv_gradient[:, 0]
        gradient[int(face[2])] += uv_gradient[:, 1]
        gradient[int(face[0])] -= uv_gradient[:, 0] + uv_gradient[:, 1]

    boundary_energy = 0.0
    loop = [int(value) for value in boundary_loop]
    for edge_index, first_id in enumerate(loop):
        second_id = loop[(edge_index + 1) % len(loop)]
        for point_id in loop:
            if point_id in (first_id, second_id):
                continue
            distance, grad_point, grad_first, grad_second = _point_segment_distance_gradient(
                uv[point_id], uv[first_id], uv[second_id]
            )
            if distance >= barrier_epsilon:
                continue
            ratio = barrier_epsilon / distance - 1.0
            value = ratio * ratio
            derivative = -2.0 * barrier_epsilon * ratio / (distance * distance)
            boundary_energy += value
            gradient[point_id] += barrier_weight * derivative * grad_point
            gradient[first_id] += barrier_weight * derivative * grad_first
            gradient[second_id] += barrier_weight * derivative * grad_second
    total = distortion + barrier_weight * boundary_energy
    return float(total), gradient, float(distortion), float(boundary_energy)


def _signed_double_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, dtype=float)[np.asarray(faces, dtype=int)[:, :3]]
    first = tri[:, 1] - tri[:, 0]
    second = tri[:, 2] - tri[:, 0]
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def _segments_intersect(
    first_a: np.ndarray,
    first_b: np.ndarray,
    second_a: np.ndarray,
    second_b: np.ndarray,
    tolerance: float = 1.0e-12,
) -> bool:
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return _cross2(b - a, c - a)

    def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
        return bool(
            min(a[0], b[0]) - tolerance <= p[0] <= max(a[0], b[0]) + tolerance
            and min(a[1], b[1]) - tolerance <= p[1] <= max(a[1], b[1]) + tolerance
        )

    o1 = orient(first_a, first_b, second_a)
    o2 = orient(first_a, first_b, second_b)
    o3 = orient(second_a, second_b, first_a)
    o4 = orient(second_a, second_b, first_b)
    if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
        return True
    return bool(
        (abs(o1) <= tolerance and on_segment(first_a, first_b, second_a))
        or (abs(o2) <= tolerance and on_segment(first_a, first_b, second_b))
        or (abs(o3) <= tolerance and on_segment(second_a, second_b, first_a))
        or (abs(o4) <= tolerance and on_segment(second_a, second_b, first_b))
    )


def boundary_self_intersection_count(uv: np.ndarray, boundary_loop: list[int]) -> int:
    loop = [int(value) for value in boundary_loop]
    count = 0
    for first in range(len(loop)):
        a0, a1 = loop[first], loop[(first + 1) % len(loop)]
        for second in range(first + 1, len(loop)):
            b0, b1 = loop[second], loop[(second + 1) % len(loop)]
            if {a0, a1}.intersection({b0, b1}):
                continue
            if _segments_intersect(uv[a0], uv[a1], uv[b0], uv[b1]):
                count += 1
    return int(count)


def _positive_quadratic_roots(c0: float, c1: float, c2: float) -> list[float]:
    scale = max(abs(c0), abs(c1), abs(c2), 1.0)
    tolerance = 1.0e-14 * scale
    roots: list[float] = []
    if abs(c2) <= tolerance:
        if abs(c1) > tolerance:
            value = -c0 / c1
            if value > 1.0e-12 and np.isfinite(value):
                roots.append(float(value))
        return roots
    discriminant = c1 * c1 - 4.0 * c2 * c0
    if discriminant < -tolerance:
        return roots
    square_root = math.sqrt(max(discriminant, 0.0))
    for value in ((-c1 - square_root) / (2.0 * c2), (-c1 + square_root) / (2.0 * c2)):
        if value > 1.0e-12 and np.isfinite(value):
            roots.append(float(value))
    return sorted(set(roots))


def _safe_step_limit(
    uv: np.ndarray,
    direction: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
) -> tuple[float, str]:
    """Approximate Section 3.3's first-singularity line-search bound."""

    limit = math.inf
    reason = "unbounded"
    for face in np.asarray(faces, dtype=int)[:, :3]:
        p0, p1, p2 = uv[face]
        d0, d1, d2 = direction[face]
        first = p1 - p0
        second = p2 - p0
        delta_first = d1 - d0
        delta_second = d2 - d0
        coefficients = (
            _cross2(first, second),
            _cross2(delta_first, second) + _cross2(first, delta_second),
            _cross2(delta_first, delta_second),
        )
        roots = _positive_quadratic_roots(*coefficients)
        if roots and roots[0] < limit:
            limit = roots[0]
            reason = "triangle_degeneracy"

    loop = [int(value) for value in boundary_loop]
    for edge_index, first_id in enumerate(loop):
        second_id = loop[(edge_index + 1) % len(loop)]
        first = uv[second_id] - uv[first_id]
        delta_first = direction[second_id] - direction[first_id]
        for point_id in loop:
            if point_id in (first_id, second_id):
                continue
            second = uv[point_id] - uv[first_id]
            delta_second = direction[point_id] - direction[first_id]
            roots = _positive_quadratic_roots(
                _cross2(first, second),
                _cross2(delta_first, second) + _cross2(first, delta_second),
                _cross2(delta_first, delta_second),
            )
            for root in roots:
                edge = (uv[second_id] + root * direction[second_id]) - (
                    uv[first_id] + root * direction[first_id]
                )
                relative = (uv[point_id] + root * direction[point_id]) - (
                    uv[first_id] + root * direction[first_id]
                )
                denominator = float(np.dot(edge, edge))
                if denominator <= _EPS:
                    continue
                fraction = float(np.dot(relative, edge) / denominator)
                if -1.0e-9 <= fraction <= 1.0 + 1.0e-9 and root < limit:
                    limit = root
                    reason = "boundary_edge_vertex_collision"
                    break
    return float(limit), reason


def _lbfgs_direction(
    gradient: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    vector = gradient.reshape(-1).copy()
    alphas: list[float] = []
    for step, change, rho in reversed(history):
        alpha = rho * float(np.dot(step, vector))
        alphas.append(alpha)
        vector -= alpha * change
    if history:
        step, change, _rho = history[-1]
        denominator = float(np.dot(change, change))
        scale = float(np.dot(step, change)) / denominator if denominator > _EPS else 1.0
        vector *= max(scale, 1.0e-8)
    for (step, change, rho), alpha in zip(history, reversed(alphas)):
        beta = rho * float(np.dot(change, vector))
        vector += step * (alpha - beta)
    return -vector.reshape(gradient.shape)


def _is_valid_embedding(
    uv: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
    minimum_signed_double_area: float,
) -> tuple[bool, int, int]:
    signed = _signed_double_areas(uv, faces)
    if not np.all(np.isfinite(uv)) or np.any(signed <= minimum_signed_double_area):
        return False, boundary_self_intersection_count(uv, boundary_loop), -1
    boundary_intersections = boundary_self_intersection_count(uv, boundary_loop)
    if boundary_intersections:
        return False, boundary_intersections, -1
    overlaps = int(count_internal_triangle_overlaps(uv, faces))
    return overlaps == 0, boundary_intersections, overlaps


def bijective_free_boundary_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: BijectiveFreeBoundaryConfig | None = None,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Flatten a single disk while leaving all boundary UV positions free."""

    started = time.perf_counter()
    settings = config or BijectiveFreeBoundaryConfig()
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4 or not np.all(np.isfinite(xyz)):
        raise RuntimeError("bijective free-boundary parameterization requires finite Vx3 vertices")
    loop, topology = _extract_single_disk_boundary(tris, len(xyz))
    uv = _tutte_embedding(xyz, tris, loop)
    inverse_surface, surface_areas = _surface_differentials(xyz, tris)
    boundary_xyz = xyz[np.asarray(loop, dtype=int)]
    boundary_next = np.roll(boundary_xyz, -1, axis=0)
    barrier_epsilon = max(
        0.25 * float(np.mean(np.linalg.norm(boundary_next - boundary_xyz, axis=1))),
        1.0e-8,
    )

    initial_valid, initial_boundary_intersections, initial_overlaps = _is_valid_embedding(
        uv, tris, loop, settings.minimum_signed_double_area
    )
    if not initial_valid:
        raise RuntimeError(
            "Tutte initialization was not bijective: "
            f"boundary_intersections={initial_boundary_intersections}, overlaps={initial_overlaps}"
        )
    initial_diagnostics = triangle_jacobian_diagnostics(xyz, uv, tris)
    energy, gradient, distortion_energy, boundary_energy = _energy_and_gradient(
        uv,
        tris,
        inverse_surface,
        surface_areas,
        loop,
        barrier_epsilon,
        settings.boundary_barrier_weight,
    )
    initial_energy = float(energy)
    initial_distortion_energy = float(distortion_energy)
    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    converged = False
    termination_reason = "maximum_iterations"
    accepted_iterations = 0
    rejected_line_search_steps = 0
    last_safe_step_reason = "unbounded"

    for _iteration in range(max(0, int(settings.max_iterations))):
        gradient -= np.mean(gradient, axis=0, keepdims=True)
        gradient_rms = float(np.linalg.norm(gradient) / math.sqrt(max(1, gradient.size)))
        if gradient_rms <= settings.gradient_tolerance:
            converged = True
            termination_reason = "gradient_tolerance"
            break
        direction = _lbfgs_direction(gradient, history)
        direction -= np.mean(direction, axis=0, keepdims=True)
        directional_derivative = float(np.sum(gradient * direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= -1.0e-14:
            history.clear()
            direction = -gradient
            direction -= np.mean(direction, axis=0, keepdims=True)
            directional_derivative = float(np.sum(gradient * direction))

        accepted = False
        candidate_data: tuple[np.ndarray, float, np.ndarray, float, float, float] | None = None
        for attempt in range(2):
            safe_limit, safe_reason = _safe_step_limit(uv, direction, tris, loop)
            last_safe_step_reason = safe_reason
            step = 1.0 if not np.isfinite(safe_limit) else min(1.0, settings.line_search_safety * safe_limit)
            for _line_search in range(max(1, int(settings.line_search_max_steps))):
                if step <= 1.0e-14:
                    break
                candidate = uv + step * direction
                candidate -= np.mean(candidate, axis=0, keepdims=True)
                valid, _boundary_count, _overlap_count = _is_valid_embedding(
                    candidate, tris, loop, settings.minimum_signed_double_area
                )
                if valid:
                    candidate_energy, candidate_gradient, candidate_distortion, candidate_boundary = _energy_and_gradient(
                        candidate,
                        tris,
                        inverse_surface,
                        surface_areas,
                        loop,
                        barrier_epsilon,
                        settings.boundary_barrier_weight,
                    )
                    armijo = energy + 1.0e-4 * step * directional_derivative
                    if np.isfinite(candidate_energy) and candidate_energy <= armijo:
                        candidate_data = (
                            candidate,
                            float(candidate_energy),
                            candidate_gradient,
                            float(candidate_distortion),
                            float(candidate_boundary),
                            float(step),
                        )
                        accepted = True
                        break
                rejected_line_search_steps += 1
                step *= 0.5
            if accepted:
                break
            if attempt == 0 and history:
                history.clear()
                direction = -gradient
                direction -= np.mean(direction, axis=0, keepdims=True)
                directional_derivative = float(np.sum(gradient * direction))
                continue
            break
        if not accepted or candidate_data is None:
            termination_reason = "valid_line_search_exhausted"
            break

        candidate, new_energy, new_gradient, new_distortion, new_boundary, _step = candidate_data
        step_vector = (candidate - uv).reshape(-1)
        gradient_change = (new_gradient - gradient).reshape(-1)
        curvature = float(np.dot(step_vector, gradient_change))
        if curvature > 1.0e-12:
            history.append((step_vector, gradient_change, 1.0 / curvature))
            if len(history) > max(1, int(settings.lbfgs_history_size)):
                history.pop(0)
        relative_change = abs(energy - new_energy) / max(abs(energy), 1.0)
        uv = candidate
        energy = new_energy
        gradient = new_gradient
        distortion_energy = new_distortion
        boundary_energy = new_boundary
        accepted_iterations += 1
        if relative_change <= settings.relative_energy_tolerance:
            converged = True
            termination_reason = "relative_energy_tolerance"
            break

    final_valid, boundary_intersections, overlaps = _is_valid_embedding(
        uv, tris, loop, settings.minimum_signed_double_area
    )
    if not final_valid:
        raise RuntimeError(
            "bijective free-boundary optimization lost validity: "
            f"boundary_intersections={boundary_intersections}, overlaps={overlaps}"
        )
    diagnostics = triangle_jacobian_diagnostics(xyz, uv, tris)
    lambda_values = np.asarray(diagnostics["raw_lambda_uv_to_surface_sigma_max"], dtype=float)
    log_lambda = np.log(lambda_values)
    anisotropy = np.asarray(diagnostics["anisotropy"], dtype=float)
    valid_lambda = lambda_values[np.isfinite(lambda_values) & (lambda_values > 0.0)]
    valid_log = log_lambda[np.isfinite(log_lambda)]
    valid_anisotropy = anisotropy[np.isfinite(anisotropy)]
    signed = _signed_double_areas(uv, tris)
    warning = "" if converged else f"Optimization stopped with {termination_reason}; the returned map remains bijective."
    metrics: dict[str, Any] = {
        **topology,
        "parameterization_method": "bijective_free_boundary",
        "parameterization_exactness_label": "smith_schaefer_2015_inspired_collision_aware_approximation",
        "parameterization_runtime_seconds": float(time.perf_counter() - started),
        "parameterization_warning": warning,
        "flattening_backend": "local_bijective_free_boundary_symmetric_dirichlet",
        "omega_parameterization_solver": "tutte_then_validity_preserving_lbfgs_symmetric_dirichlet",
        "omega_boundary_mode": "paper_default",
        "omega_parameterization_mode": "bijective_free_boundary",
        "requested_omega_parameterization_mode": "bijective_free_boundary",
        "omega_boundary_fixed": False,
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape": "free",
        "omega_boundary_model": "free boundary optimized with collision-aware interior-point line search",
        "omega_boundary_constraint_model": "existing boundary vertex set; unconstrained 2D positions",
        "boundary_loop": list(map(int, loop)),
        "boundary_vertex_count": len(loop),
        "uv_triangle_flip_count": int(diagnostics["uv_triangle_flip_count"]),
        "uv_degenerate_triangle_count": int(diagnostics["uv_degenerate_triangle_count"]),
        "uv_min_triangle_area": 0.5 * float(np.min(signed)),
        "uv_flip_triangle_ids": np.flatnonzero(signed < -settings.minimum_signed_double_area).astype(int).tolist(),
        "uv_degenerate_triangle_ids": np.flatnonzero(signed <= settings.minimum_signed_double_area).astype(int).tolist(),
        "internal_triangle_overlap_count": int(overlaps),
        "boundary_self_intersection_count": int(boundary_intersections),
        "initial_internal_triangle_overlap_count": int(initial_overlaps),
        "initial_uv_triangle_flip_count": int(initial_diagnostics["uv_triangle_flip_count"]),
        "initial_energy": initial_energy,
        "final_energy": float(energy),
        "initial_distortion_energy": initial_distortion_energy,
        "final_distortion_energy": float(distortion_energy),
        "final_boundary_barrier_energy": float(boundary_energy),
        "optimization_iteration_count": int(accepted_iterations),
        "optimization_converged": bool(converged),
        "optimization_succeeded": bool(accepted_iterations > 0 or converged),
        "optimization_termination_reason": termination_reason,
        "optimization_rejected_line_search_step_count": int(rejected_line_search_steps),
        "line_search_last_safe_step_reason": last_safe_step_reason,
        "boundary_barrier_epsilon": float(barrier_epsilon),
        "boundary_barrier_weight": float(settings.boundary_barrier_weight),
        "lambda_definition": "sigma_max of UV-to-surface Jacobian, matching reference_bff.triangle_jacobian_diagnostics",
        "mapping_direction": diagnostics["mapping_direction"],
        "lambda_min": float(np.min(valid_lambda)),
        "lambda_median": float(np.median(valid_lambda)),
        "lambda_max": float(np.max(valid_lambda)),
        "log_lambda_min": float(np.min(valid_log)),
        "log_lambda_median": float(np.median(valid_log)),
        "log_lambda_max": float(np.max(valid_log)),
        "anisotropy_mean": float(np.mean(valid_anisotropy)),
        "anisotropy_max": float(np.max(valid_anisotropy)),
        "per_triangle_lambda": lambda_values.tolist(),
        "per_triangle_log_lambda": log_lambda.tolist(),
        "per_triangle_anisotropy": anisotropy.tolist(),
        "smith_schaefer_2015_inspired": True,
        "smith_schaefer_components": [
            "valid Tutte initialization",
            "area-weighted symmetric isometric/Dirichlet barrier distortion",
            "first triangle-degeneracy step bound",
            "first moving boundary edge/vertex collision step bound",
            "validity-preserving backtracking line search",
        ],
        "implementation_simplifications": [
            "O(m^2) boundary candidate search instead of a spatial hash",
            "custom limited-memory BFGS loop instead of the paper implementation",
            "direct global-overlap validation before accepting each update",
        ],
        "onestring_grid_loss_used": False,
        "lambda_directly_optimized": False,
        "topology_modified": False,
        "seams_or_cuts_added": False,
    }
    return uv, loop, metrics


def install_bijective_free_boundary(pipeline_module: Any) -> None:
    """Install only the explicit ``bijective_free_boundary`` pipeline mode."""

    if getattr(pipeline_module, "_BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED", False):
        return
    legacy = pipeline_module._build_surface_parameterization

    def build(surface: Any, target: Any, grid: Any, params: Any) -> Any:
        mode = str(getattr(params, "omega_parameterization_mode", "bff"))
        if mode != "bijective_free_boundary":
            return legacy(surface, target, grid, params)
        if str(getattr(params, "omega_boundary_mode", "paper_default")) != "paper_default":
            raise ValueError("bijective_free_boundary requires omega_boundary_mode='paper_default'")
        vertices = np.asarray(surface.vertices, dtype=float)
        faces = np.asarray(surface.faces, dtype=int)[:, :3]
        config = BijectiveFreeBoundaryConfig(
            max_iterations=int(getattr(params, "bijective_free_boundary_max_iterations", 60)),
            gradient_tolerance=float(getattr(params, "bijective_free_boundary_gradient_tolerance", 1.0e-7)),
            relative_energy_tolerance=float(
                getattr(params, "bijective_free_boundary_energy_tolerance", 1.0e-8)
            ),
            line_search_max_steps=int(
                getattr(params, "bijective_free_boundary_line_search_max_steps", 20)
            ),
            line_search_safety=float(getattr(params, "bijective_free_boundary_line_search_safety", 0.8)),
            boundary_barrier_weight=float(
                getattr(params, "bijective_free_boundary_boundary_barrier_weight", 1.0)
            ),
        )
        uv, loop, metrics = bijective_free_boundary_parameterization(vertices, faces, config)
        slope = (
            {"mean_slope": 0.0, "max_slope": 0.0}
            if getattr(target, "kind", "") == "sampled"
            else pipeline_module._original._heightfield_metric_summary(target, grid)
        )
        metrics.update(
            {
                "surface_vertex_count": len(vertices),
                "surface_triangle_count": len(faces),
                "mean_slope": float(slope["mean_slope"]),
                "max_slope": float(slope["max_slope"]),
                "height_field_shortcut_used": False,
                "omega_corresponds_to_S": True,
                "omega_correspondence_model": "bijective free-boundary map c:S->Omega; inverse by UV triangle lookup",
                "paper_flow_stage": "S -> Omega by a Smith & Schaefer 2015-inspired free-boundary optimization",
                "paper_exactness_warning": (
                    "Collision-aware free-boundary approximation; not the authors' complete implementation."
                ),
                "omega_warning": str(metrics.get("parameterization_warning", "")),
            }
        )
        output = pipeline_module._original.SurfaceParameterization(
            method="bijective_free_boundary",
            surface_vertices_3d=vertices,
            surface_faces=faces,
            uv_vertices_2d=uv,
            uv_faces=faces.copy(),
            omega_boundary=uv[np.asarray(loop + [loop[0]], dtype=int)],
            triangle_acceleration=None,
            metrics=metrics,
        )
        marker = getattr(pipeline_module, "_mark_parameterization_mode", None)
        if callable(marker):
            output = marker(
                output,
                method="bijective_free_boundary",
                exactness="smith_schaefer_2015_inspired_collision_aware_approximation",
                warning=str(metrics.get("parameterization_warning", "")),
            )
        return output

    pipeline_module._build_surface_parameterization = build
    pipeline_module._original._build_surface_parameterization = build
    pipeline_module._BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED = True


__all__ = [
    "BijectiveFreeBoundaryConfig",
    "bijective_free_boundary_parameterization",
    "boundary_self_intersection_count",
    "install_bijective_free_boundary",
]
