"""Bijective free-boundary surface parameterization for one disk chart.

The implementation is deliberately independent from the existing BFF and CEPS
backends.  It follows the Smith & Schaefer 2015 algorithm while making no claim
of binary compatibility with the authors' implementation:

* start from a valid convex-boundary Floater embedding;
* minimize an area-weighted symmetric Dirichlet/isometric distortion whose
  inverse term diverges as a triangle degenerates; and
* restrict every line-search step to remain before triangle degeneracy and
  moving boundary edge/vertex collisions.

Unlike the paper, this compact research implementation uses an O(m^2)
boundary search instead of a spatial hash and a small custom L-BFGS loop.
Positive orientation plus a simple boundary supplies the per-step disk
injectivity condition; an explicit global overlap audit is retained at the
start and end.  No seam, cut, topology, BFF, CEPS, lambda, or OneString
grid-loss optimization is added.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from .reference_bff import count_internal_triangle_overlaps, triangle_jacobian_diagnostics


_EPS = 1.0e-12
ProgressCallback = Callable[[str, float, str], None]


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    fraction: float,
    detail: str = "",
) -> None:
    if callback is None:
        return
    try:
        callback(stage, max(0.0, min(1.0, float(fraction))), detail)
    except Exception:
        # Reporting must never change the numerical result of the optimizer.
        return


@dataclass(frozen=True)
class BijectiveFreeBoundaryConfig:
    """Numerical controls for the isolated free-boundary experiment."""

    max_iterations: int = 1000
    gradient_tolerance: float = 1.0e-7
    relative_energy_tolerance: float = 1.0e-8
    line_search_max_steps: int = 20
    line_search_safety: float = 0.9
    initial_step_scale: float = 3.0
    conformal_weight: float = 4.0
    boundary_barrier_weight: float = 1.0
    lbfgs_history_size: int = 8
    minimum_signed_double_area: float = 1.0e-12
    validate_global_overlap_each_step: bool = False
    low_frequency_metric_weight: float = 8.0
    initial_boundary_shape: str = "circle"


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _circle_fit_relative_rms(points: np.ndarray) -> float:
    """Return radial RMS residual after fitting the best algebraic circle."""

    values = np.asarray(points, dtype=float)
    design = np.column_stack([2.0 * values[:, 0], 2.0 * values[:, 1], np.ones(len(values))])
    rhs = np.sum(values * values, axis=1)
    center_x, center_y, constant = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = np.asarray([center_x, center_y], dtype=float)
    radius = math.sqrt(max(float(constant + np.dot(center, center)), _EPS))
    radial = np.linalg.norm(values - center, axis=1)
    return float(np.sqrt(np.mean((radial - radius) ** 2)) / radius)


def _boundary_nonsimilarity_change(
    initial_points: np.ndarray,
    final_points: np.ndarray,
) -> tuple[float, float]:
    """Measure boundary change remaining after optimal translation/rotation/scale."""

    initial = np.asarray(initial_points, dtype=float)
    final = np.asarray(final_points, dtype=float)
    initial_complex = (initial[:, 0] - np.mean(initial[:, 0])) + 1j * (
        initial[:, 1] - np.mean(initial[:, 1])
    )
    final_complex = (final[:, 0] - np.mean(final[:, 0])) + 1j * (
        final[:, 1] - np.mean(final[:, 1])
    )
    denominator = float(np.sum(np.abs(initial_complex) ** 2))
    scale_rotation = (
        np.sum(np.conjugate(initial_complex) * final_complex) / denominator
        if denominator > _EPS
        else 1.0 + 0.0j
    )
    residual = np.abs(final_complex - scale_rotation * initial_complex)
    rms = float(np.sqrt(np.mean(residual * residual)))
    reference_radius = float(np.sqrt(np.mean(np.abs(final_complex) ** 2)))
    return rms, float(rms / max(reference_radius, _EPS))


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
    boundary_shape: str = "circle",
) -> np.ndarray:
    """Create a valid convex-boundary Floater mean-value embedding.

    Smith and Schaefer permit either Tutte's or Floater's positive-weight
    embedding as the bijective starting point.  Mean-value weights are much
    better conditioned than uniform graph weights on irregular scanned meshes
    such as Bunny while retaining the convex-boundary bijectivity guarantee.
    """

    try:
        from scipy import sparse
        from scipy.sparse import linalg as spla
    except Exception as exc:  # pragma: no cover - dependency is declared by the package
        raise RuntimeError("scipy sparse is required for the Floater initialization") from exc

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
    triangle_xyz = xyz[tris]
    surface_area = 0.5 * float(
        np.sum(
            np.linalg.norm(
                np.cross(
                    triangle_xyz[:, 1] - triangle_xyz[:, 0],
                    triangle_xyz[:, 2] - triangle_xyz[:, 0],
                ),
                axis=1,
            )
        )
    )
    if not np.isfinite(surface_area) or surface_area <= _EPS:
        raise RuntimeError("the input surface has zero 3D area")
    # A convex starting boundary may have any scale, but perimeter scaling can
    # severely compress a disk whose seam is short relative to its area (the
    # open Bunny is a representative case).  Equal-area scaling starts the
    # symmetric Dirichlet solve near its globally preferred scale and avoids
    # letting inverse-Jacobian gradients from a few tiny triangles freeze the
    # otherwise free boundary.
    uv = np.zeros((n, 2), dtype=float)
    if boundary_shape == "circle":
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
        polygon_area = 0.5 * abs(
            float(
                np.sum(
                    boundary_uv[:, 0] * np.roll(boundary_uv[:, 1], -1)
                    - boundary_uv[:, 1] * np.roll(boundary_uv[:, 0], -1)
                )
            )
        )
        uv[loop] = boundary_uv * math.sqrt(surface_area / max(polygon_area, _EPS))
    elif boundary_shape == "rectangle":
        # A literal square has collinear boundary vertices and can create zero
        # area boundary triangles.  A p=8 superellipse is visibly rectangular
        # but strictly convex, preserving Floater's positive-area guarantee.
        exponent = 8.0
        dense_angle = np.linspace(0.0, 2.0 * np.pi, 4097)
        cosine = np.cos(dense_angle)
        sine = np.sin(dense_angle)
        dense_rectangle = np.column_stack(
            [
                np.sign(cosine) * np.abs(cosine) ** (2.0 / exponent),
                np.sign(sine) * np.abs(sine) ** (2.0 / exponent),
            ]
        )
        dense_arclength = np.concatenate(
            [
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(dense_rectangle, axis=0), axis=1)),
            ]
        )
        target_arclength = cumulative / perimeter * dense_arclength[-1]
        boundary_uv = np.column_stack(
            [
                np.interp(target_arclength, dense_arclength, dense_rectangle[:, 0]),
                np.interp(target_arclength, dense_arclength, dense_rectangle[:, 1]),
            ]
        )
        polygon_area = 0.5 * abs(
            float(
                np.sum(
                    boundary_uv[:, 0] * np.roll(boundary_uv[:, 1], -1)
                    - boundary_uv[:, 1] * np.roll(boundary_uv[:, 0], -1)
                )
            )
        )
        uv[loop] = boundary_uv * math.sqrt(surface_area / max(polygon_area, _EPS))
    else:
        raise ValueError(
            "initial_boundary_shape must be 'circle' or 'rectangle'; "
            f"got {boundary_shape!r}"
        )

    neighbors: list[set[int]] = [set() for _ in range(n)]
    mean_value_numerators: list[dict[int, float]] = [dict() for _ in range(n)]
    for face in tris:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            neighbors[int(a)].add(int(b))
            neighbors[int(b)].add(int(a))
        for local_index in range(3):
            center_id = int(face[local_index])
            first_id = int(face[(local_index + 1) % 3])
            second_id = int(face[(local_index + 2) % 3])
            first_vector = xyz[first_id] - xyz[center_id]
            second_vector = xyz[second_id] - xyz[center_id]
            first_length = float(np.linalg.norm(first_vector))
            second_length = float(np.linalg.norm(second_vector))
            if first_length <= _EPS or second_length <= _EPS:
                raise RuntimeError("Floater initialization found a zero-length surface edge")
            sine = float(np.linalg.norm(np.cross(first_vector, second_vector))) / (
                first_length * second_length
            )
            cosine = float(np.dot(first_vector, second_vector)) / (first_length * second_length)
            half_tangent = max(0.0, sine / max(1.0 + cosine, 1.0e-12))
            mean_value_numerators[center_id][first_id] = (
                mean_value_numerators[center_id].get(first_id, 0.0) + half_tangent
            )
            mean_value_numerators[center_id][second_id] = (
                mean_value_numerators[center_id].get(second_id, 0.0) + half_tangent
            )
    boundary_set = set(int(value) for value in loop)
    interior = [value for value in range(n) if value not in boundary_set]
    if interior:
        local = {vertex: index for index, vertex in enumerate(interior)}
        matrix = sparse.lil_matrix((len(interior), len(interior)), dtype=float)
        rhs = np.zeros((len(interior), 2), dtype=float)
        for vertex in interior:
            row = local[vertex]
            if len(neighbors[vertex]) < 2:
                raise RuntimeError("Floater initialization found an invalid interior vertex")
            weights: dict[int, float] = {}
            for neighbor in neighbors[vertex]:
                edge_length = float(np.linalg.norm(xyz[neighbor] - xyz[vertex]))
                numerator = float(mean_value_numerators[vertex].get(neighbor, 0.0))
                weight = numerator / max(edge_length, _EPS)
                if not np.isfinite(weight) or weight <= 0.0:
                    weight = 1.0 / max(edge_length, _EPS)
                weights[neighbor] = weight
            weight_sum = float(sum(weights.values()))
            if not np.isfinite(weight_sum) or weight_sum <= _EPS:
                raise RuntimeError("Floater initialization produced invalid positive weights")
            matrix[row, row] = weight_sum
            for neighbor, weight in weights.items():
                if neighbor in local:
                    matrix[row, local[neighbor]] -= weight
                else:
                    rhs[row] += weight * uv[neighbor]
        uv[np.asarray(interior, dtype=int)] = spla.spsolve(matrix.tocsr(), rhs)

    signed = _signed_double_areas(uv, tris)
    if np.all(signed < -1.0e-12):
        uv[:, 1] *= -1.0
        signed *= -1.0
    if np.any(signed <= 1.0e-12) or not np.all(np.isfinite(uv)):
        raise RuntimeError("Floater initialization did not produce a strictly valid disk embedding")
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


def _energy_and_gradient_bruteforce(
    uv: np.ndarray,
    faces: np.ndarray,
    inverse_surface: np.ndarray,
    surface_areas: np.ndarray,
    boundary_loop: list[int],
    barrier_epsilon: float,
    barrier_weight: float,
    conformal_weight: float = 0.0,
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
        frobenius_squared = float(np.sum(jacobian * jacobian))
        face_energy = area * (
            frobenius_squared
            + float(np.sum(inverse_jacobian * inverse_jacobian))
        )
        conformal_energy = area * (frobenius_squared / determinant - 2.0)
        face_energy += float(conformal_weight) * conformal_energy
        distortion += face_energy
        jacobian_gradient = 2.0 * area * (
            jacobian
            - inverse_jacobian.T @ inverse_jacobian @ inverse_jacobian.T
        )
        if conformal_weight:
            jacobian_gradient += float(conformal_weight) * area * (
                2.0 * jacobian / determinant
                - (frobenius_squared / determinant) * inverse_jacobian.T
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


def _energy_and_gradient(
    uv: np.ndarray,
    faces: np.ndarray,
    inverse_surface: np.ndarray,
    surface_areas: np.ndarray,
    boundary_loop: list[int],
    barrier_epsilon: float,
    barrier_weight: float,
    conformal_weight: float = 0.0,
) -> tuple[float, np.ndarray, float, float]:
    """Vectorized equivalent of the retained scalar reference implementation."""

    tris = np.asarray(faces, dtype=int)[:, :3]
    coordinates = np.asarray(uv, dtype=float)
    triangle_uv = coordinates[tris]
    d_uv = np.stack(
        [triangle_uv[:, 1] - triangle_uv[:, 0], triangle_uv[:, 2] - triangle_uv[:, 0]],
        axis=2,
    )
    jacobian = np.matmul(d_uv, inverse_surface)
    determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    if np.any(determinant <= _EPS) or not np.all(np.isfinite(determinant)):
        return math.inf, np.zeros_like(coordinates), math.inf, math.inf

    inverse_jacobian = np.empty_like(jacobian)
    inverse_jacobian[:, 0, 0] = jacobian[:, 1, 1] / determinant
    inverse_jacobian[:, 0, 1] = -jacobian[:, 0, 1] / determinant
    inverse_jacobian[:, 1, 0] = -jacobian[:, 1, 0] / determinant
    inverse_jacobian[:, 1, 1] = jacobian[:, 0, 0] / determinant
    frobenius_squared = np.sum(jacobian * jacobian, axis=(1, 2))
    symmetric_dirichlet = np.sum(inverse_jacobian * inverse_jacobian, axis=(1, 2))
    conformal_per_face = np.maximum(frobenius_squared / determinant - 2.0, 0.0)
    distortion = float(
        np.sum(
            surface_areas
            * (frobenius_squared + symmetric_dirichlet + float(conformal_weight) * conformal_per_face)
        )
    )
    inverse_transpose = np.swapaxes(inverse_jacobian, 1, 2)
    jacobian_gradient = 2.0 * surface_areas[:, None, None] * (
        jacobian - inverse_transpose @ inverse_jacobian @ inverse_transpose
    )
    if conformal_weight:
        jacobian_gradient += (
            float(conformal_weight)
            * surface_areas[:, None, None]
            * (
                2.0 * jacobian / determinant[:, None, None]
                - (frobenius_squared / determinant)[:, None, None] * inverse_transpose
            )
        )
    uv_gradient = jacobian_gradient @ np.swapaxes(inverse_surface, 1, 2)
    gradient = np.zeros_like(coordinates)
    np.add.at(gradient, tris[:, 1], uv_gradient[:, :, 0])
    np.add.at(gradient, tris[:, 2], uv_gradient[:, :, 1])
    np.add.at(gradient, tris[:, 0], -(uv_gradient[:, :, 0] + uv_gradient[:, :, 1]))

    loop = np.asarray(boundary_loop, dtype=int)
    boundary_energy = 0.0
    if len(loop):
        first_ids = loop
        second_ids = np.roll(loop, -1)
        first = coordinates[first_ids]
        second = coordinates[second_ids]
        points = coordinates[loop]
        edge = second - first
        squared = np.sum(edge * edge, axis=1)
        safe_squared = np.maximum(squared, _EPS)
        relative = points[None, :, :] - first[:, None, :]
        fraction = np.clip(
            np.sum(relative * edge[:, None, :], axis=2) / safe_squared[:, None],
            0.0,
            1.0,
        )
        fraction = np.where(squared[:, None] <= _EPS, 0.5, fraction)
        closest = first[:, None, :] + fraction[:, :, None] * edge[:, None, :]
        delta = points[None, :, :] - closest
        distance = np.maximum(np.linalg.norm(delta, axis=2), _EPS)
        normal = delta / distance[:, :, None]
        incident = np.eye(len(loop), dtype=bool) | np.roll(np.eye(len(loop), dtype=bool), 1, axis=1)
        active = (~incident) & (distance < barrier_epsilon)
        ratio = np.where(active, barrier_epsilon / distance - 1.0, 0.0)
        boundary_energy = float(np.sum(ratio * ratio))
        derivative = np.where(
            active,
            -2.0 * barrier_epsilon * ratio / (distance * distance),
            0.0,
        )
        weighted_normal = barrier_weight * derivative[:, :, None] * normal
        point_gradient = np.sum(weighted_normal, axis=0)
        first_gradient = np.sum(-(1.0 - fraction)[:, :, None] * weighted_normal, axis=1)
        second_gradient = np.sum(-fraction[:, :, None] * weighted_normal, axis=1)
        np.add.at(gradient, loop, point_gradient)
        np.add.at(gradient, first_ids, first_gradient)
        np.add.at(gradient, second_ids, second_gradient)

    total = distortion + barrier_weight * boundary_energy
    return float(total), gradient, distortion, float(boundary_energy)


def _conformal_energy(
    uv: np.ndarray,
    faces: np.ndarray,
    inverse_surface: np.ndarray,
    surface_areas: np.ndarray,
) -> float:
    """Scale-invariant conformal distortion; zero iff both singular values agree."""

    tris = np.asarray(faces, dtype=int)[:, :3]
    triangle_uv = np.asarray(uv, dtype=float)[tris]
    d_uv = np.stack(
        [triangle_uv[:, 1] - triangle_uv[:, 0], triangle_uv[:, 2] - triangle_uv[:, 0]],
        axis=2,
    )
    jacobian = np.matmul(d_uv, inverse_surface)
    determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    if np.any(determinant <= _EPS) or not np.all(np.isfinite(determinant)):
        return math.inf
    frobenius_squared = np.sum(jacobian * jacobian, axis=(1, 2))
    return float(np.sum(surface_areas * np.maximum(frobenius_squared / determinant - 2.0, 0.0)))


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
    loop = np.asarray(boundary_loop, dtype=int)
    edge_count = len(loop)
    if edge_count < 4:
        return 0

    first_edge, second_edge = np.triu_indices(edge_count, k=1)
    nonadjacent = (second_edge != first_edge + 1) & ~(
        (first_edge == 0) & (second_edge == edge_count - 1)
    )
    first_edge = first_edge[nonadjacent]
    second_edge = second_edge[nonadjacent]
    coordinates = np.asarray(uv, dtype=float)[loop]
    starts = coordinates
    ends = np.roll(coordinates, -1, axis=0)
    a = starts[first_edge]
    b = ends[first_edge]
    c = starts[second_edge]
    d = ends[second_edge]

    def orient(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> np.ndarray:
        edge = end - start
        relative = point - start
        return edge[:, 0] * relative[:, 1] - edge[:, 1] * relative[:, 0]

    def on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> np.ndarray:
        tolerance = 1.0e-12
        return (
            (np.minimum(start[:, 0], end[:, 0]) - tolerance <= point[:, 0])
            & (point[:, 0] <= np.maximum(start[:, 0], end[:, 0]) + tolerance)
            & (np.minimum(start[:, 1], end[:, 1]) - tolerance <= point[:, 1])
            & (point[:, 1] <= np.maximum(start[:, 1], end[:, 1]) + tolerance)
        )

    tolerance = 1.0e-12
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    proper = (o1 * o2 < -tolerance) & (o3 * o4 < -tolerance)
    touching = (
        ((np.abs(o1) <= tolerance) & on_segment(a, b, c))
        | ((np.abs(o2) <= tolerance) & on_segment(a, b, d))
        | ((np.abs(o3) <= tolerance) & on_segment(c, d, a))
        | ((np.abs(o4) <= tolerance) & on_segment(c, d, b))
    )
    return int(np.count_nonzero(proper | touching))


def _positive_quadratic_roots(c0: float, c1: float, c2: float) -> list[float]:
    scale = max(abs(c0), abs(c1), abs(c2), 1.0)
    tolerance = 1.0e-14 * scale
    roots: list[float] = []
    if abs(c2) <= tolerance:
        if abs(c1) > tolerance:
            value = -c0 / c1
            if value > np.finfo(float).eps and np.isfinite(value):
                roots.append(float(value))
        return roots
    discriminant = c1 * c1 - 4.0 * c2 * c0
    if discriminant < -tolerance:
        return roots
    square_root = math.sqrt(max(discriminant, 0.0))
    for value in ((-c1 - square_root) / (2.0 * c2), (-c1 + square_root) / (2.0 * c2)):
        if value > np.finfo(float).eps and np.isfinite(value):
            roots.append(float(value))
    return sorted(set(roots))


def _safe_step_limit_bruteforce(
    uv: np.ndarray,
    direction: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
    minimum_signed_double_area: float = 0.0,
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
            _cross2(first, second) - float(minimum_signed_double_area),
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


def _positive_quadratic_roots_array(
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
) -> np.ndarray:
    """Vectorized form of ``_positive_quadratic_roots`` with two sorted slots."""

    first, second, third = np.broadcast_arrays(
        np.asarray(c0, dtype=float),
        np.asarray(c1, dtype=float),
        np.asarray(c2, dtype=float),
    )
    scale = np.maximum.reduce([np.abs(first), np.abs(second), np.abs(third), np.ones_like(first)])
    tolerance = 1.0e-14 * scale
    linear = np.abs(third) <= tolerance
    roots = np.full(first.shape + (2,), math.inf, dtype=float)
    linear_root = np.divide(
        -first,
        second,
        out=np.full_like(first, math.inf),
        where=np.abs(second) > tolerance,
    )
    roots[..., 0] = np.where(
        linear
        & (np.abs(second) > tolerance)
        & (linear_root > np.finfo(float).eps)
        & np.isfinite(linear_root),
        linear_root,
        math.inf,
    )

    discriminant = second * second - 4.0 * third * first
    quadratic = (~linear) & (discriminant >= -tolerance)
    square_root = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = np.where(quadratic, 2.0 * third, 1.0)
    candidate_roots = np.stack(
        [(-second - square_root) / denominator, (-second + square_root) / denominator],
        axis=-1,
    )
    candidate_roots = np.where(
        quadratic[..., None]
        & (candidate_roots > np.finfo(float).eps)
        & np.isfinite(candidate_roots),
        candidate_roots,
        math.inf,
    )
    candidate_roots.sort(axis=-1)
    roots = np.where(quadratic[..., None], candidate_roots, roots)
    return roots


def _safe_step_limit(
    uv: np.ndarray,
    direction: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
    minimum_signed_double_area: float = 0.0,
) -> tuple[float, str]:
    """Vectorized equivalent of the retained first-singularity reference code."""

    coordinates = np.asarray(uv, dtype=float)
    movement = np.asarray(direction, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    triangle_uv = coordinates[tris]
    triangle_direction = movement[tris]
    first = triangle_uv[:, 1] - triangle_uv[:, 0]
    second = triangle_uv[:, 2] - triangle_uv[:, 0]
    delta_first = triangle_direction[:, 1] - triangle_direction[:, 0]
    delta_second = triangle_direction[:, 2] - triangle_direction[:, 0]
    triangle_roots = _positive_quadratic_roots_array(
        first[:, 0] * second[:, 1]
        - first[:, 1] * second[:, 0]
        - float(minimum_signed_double_area),
        delta_first[:, 0] * second[:, 1]
        - delta_first[:, 1] * second[:, 0]
        + first[:, 0] * delta_second[:, 1]
        - first[:, 1] * delta_second[:, 0],
        delta_first[:, 0] * delta_second[:, 1] - delta_first[:, 1] * delta_second[:, 0],
    )
    triangle_limit = float(np.min(triangle_roots[..., 0])) if len(tris) else math.inf
    limit = triangle_limit
    reason = "triangle_degeneracy" if np.isfinite(triangle_limit) else "unbounded"

    loop = np.asarray(boundary_loop, dtype=int)
    if len(loop):
        first_ids = loop
        second_ids = np.roll(loop, -1)
        edge = coordinates[second_ids] - coordinates[first_ids]
        delta_edge = movement[second_ids] - movement[first_ids]
        relative = coordinates[loop][None, :, :] - coordinates[first_ids][:, None, :]
        delta_relative = movement[loop][None, :, :] - movement[first_ids][:, None, :]
        boundary_roots = _positive_quadratic_roots_array(
            edge[:, None, 0] * relative[:, :, 1] - edge[:, None, 1] * relative[:, :, 0],
            delta_edge[:, None, 0] * relative[:, :, 1]
            - delta_edge[:, None, 1] * relative[:, :, 0]
            + edge[:, None, 0] * delta_relative[:, :, 1]
            - edge[:, None, 1] * delta_relative[:, :, 0],
            delta_edge[:, None, 0] * delta_relative[:, :, 1]
            - delta_edge[:, None, 1] * delta_relative[:, :, 0],
        )
        finite_roots = np.where(np.isfinite(boundary_roots), boundary_roots, 0.0)
        edge_at_root = (
            edge[:, None, None, :]
            + finite_roots[:, :, :, None] * delta_edge[:, None, None, :]
        )
        relative_at_root = (
            relative[:, :, None, :]
            + finite_roots[:, :, :, None] * delta_relative[:, :, None, :]
        )
        denominator = np.sum(edge_at_root * edge_at_root, axis=3)
        fraction = np.divide(
            np.sum(relative_at_root * edge_at_root, axis=3),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > _EPS,
        )
        incident = np.eye(len(loop), dtype=bool) | np.roll(np.eye(len(loop), dtype=bool), 1, axis=1)
        valid = (
            np.isfinite(boundary_roots)
            & (~incident[:, :, None])
            & (denominator > _EPS)
            & (fraction >= -1.0e-9)
            & (fraction <= 1.0 + 1.0e-9)
        )
        boundary_limit = float(np.min(np.where(valid, boundary_roots, math.inf)))
        if boundary_limit < limit:
            limit = boundary_limit
            reason = "boundary_edge_vertex_collision"
    return float(limit), reason


def _lbfgs_direction(
    gradient: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray, float]],
    coordinates: np.ndarray | None = None,
    low_frequency_weight: float = 8.0,
) -> np.ndarray:
    gradient_norm = np.linalg.norm(gradient, axis=1)
    active_norm = gradient_norm[np.isfinite(gradient_norm) & (gradient_norm > _EPS)]
    reference_norm = float(np.median(active_norm)) if len(active_norm) else 1.0
    diagonal_scale = reference_norm / np.maximum(gradient_norm, reference_norm * 1.0e-4)
    # Do not impose a coarse lower bound here: scanned meshes can differ by
    # many orders of magnitude in per-vertex curvature.  A 1e-4 floor still
    # allowed one skinny interior triangle to set the singularity step for the
    # entire chart and effectively froze the boundary variables.
    diagonal_scale = np.clip(diagonal_scale, 1.0e-12, 1.0e4)

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
    # A positive diagonal initial inverse-Hessian keeps the whole-chart L-BFGS
    # update from being dominated by a handful of skinny triangles.  Boundary
    # and interior variables remain in the same quasi-Newton solve and use the
    # same Smith-Schaefer energy and singularity-aware line search.
    local_metric_vector = vector.reshape(gradient.shape) * diagonal_scale[:, None]
    if coordinates is not None and len(coordinates) == len(gradient):
        centered = np.asarray(coordinates, dtype=float) - np.mean(coordinates, axis=0, keepdims=True)
        coordinate_scale = max(float(np.sqrt(np.mean(centered * centered))), _EPS)
        x = centered[:, 0] / coordinate_scale
        y = centered[:, 1] / coordinate_scale
        # Low-frequency chart modes let the same full-energy gradient express
        # coherent scale, shear, and non-circular boundary changes instead of
        # being consumed exclusively by high-frequency skinny-triangle modes.
        basis = np.column_stack(
            [np.ones(len(x)), x, y, x * x, x * y, y * y]
        )
        orthogonal_basis, _ = np.linalg.qr(basis, mode="reduced")
        low_frequency = orthogonal_basis @ (orthogonal_basis.T @ vector.reshape(gradient.shape))
        local_rms = float(np.linalg.norm(local_metric_vector) / math.sqrt(max(1, local_metric_vector.size)))
        low_frequency_rms = float(
            np.linalg.norm(low_frequency) / math.sqrt(max(1, low_frequency.size))
        )
        if local_rms > _EPS and low_frequency_rms > _EPS:
            local_metric_vector += (
                max(0.0, float(low_frequency_weight))
                * low_frequency
                * (local_rms / low_frequency_rms)
            )
    vector = local_metric_vector.reshape(-1)
    for (step, change, rho), alpha in zip(history, reversed(alphas)):
        beta = rho * float(np.dot(change, vector))
        vector += step * (alpha - beta)
    return -vector.reshape(gradient.shape)


def _check_local_validity(
    uv: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
    minimum_signed_double_area: float,
) -> tuple[bool, int]:
    """Check finite coordinates, positive triangle areas, and simple boundary."""

    signed = _signed_double_areas(uv, faces)
    if not np.all(np.isfinite(uv)) or np.any(signed <= minimum_signed_double_area):
        return False, boundary_self_intersection_count(uv, boundary_loop)
    boundary_intersections = boundary_self_intersection_count(uv, boundary_loop)
    return boundary_intersections == 0, boundary_intersections


def _check_global_overlap(
    uv: np.ndarray,
    faces: np.ndarray,
    *,
    stats: dict[str, int | float] | None = None,
) -> int:
    """Run the expensive global non-adjacent triangle overlap check."""

    return int(count_internal_triangle_overlaps(uv, faces, stats=stats))


def _is_valid_embedding(
    uv: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
    minimum_signed_double_area: float,
) -> tuple[bool, int, int]:
    """Compatibility helper combining the split local and global checks."""

    locally_valid, boundary_intersections = _check_local_validity(
        uv, faces, boundary_loop, minimum_signed_double_area
    )
    if not locally_valid:
        return False, boundary_intersections, -1
    overlaps = _check_global_overlap(uv, faces)
    return overlaps == 0, boundary_intersections, overlaps


def bijective_free_boundary_parameterization(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: BijectiveFreeBoundaryConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Flatten a single disk while leaving all boundary UV positions free."""

    started = time.perf_counter()
    settings = config or BijectiveFreeBoundaryConfig()
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4 or not np.all(np.isfinite(xyz)):
        raise RuntimeError("bijective free-boundary parameterization requires finite Vx3 vertices")
    _emit_progress(progress_callback, "Extract boundary", 0.02, f"V={len(xyz)}, F={len(tris)}")
    loop, topology = _extract_single_disk_boundary(tris, len(xyz))
    initialization_started = time.perf_counter()
    _emit_progress(progress_callback, "Floater initialization", 0.08, f"boundary vertices={len(loop)}")
    uv = _tutte_embedding(xyz, tris, loop, settings.initial_boundary_shape)
    initial_uv = uv.copy()
    initialization_seconds = time.perf_counter() - initialization_started
    surface_differentials_started = time.perf_counter()
    _emit_progress(progress_callback, "Surface differential precomputation", 0.14, f"triangles={len(tris)}")
    inverse_surface, surface_areas = _surface_differentials(xyz, tris)
    surface_differentials_seconds = time.perf_counter() - surface_differentials_started
    boundary_xyz = xyz[np.asarray(loop, dtype=int)]
    boundary_next = np.roll(boundary_xyz, -1, axis=0)
    barrier_epsilon = max(
        0.25 * float(np.mean(np.linalg.norm(boundary_next - boundary_xyz, axis=1))),
        1.0e-8,
    )

    overlap_check_call_count = 0
    overlap_check_total_seconds = 0.0
    overlap_check_max_seconds = 0.0
    overlap_broad_phase_candidate_pair_count_total = 0
    overlap_total_possible_pair_count_total = 0
    energy_gradient_call_count = 0
    energy_gradient_total_seconds = 0.0
    safe_step_call_count = 0
    safe_step_total_seconds = 0.0
    boundary_self_intersection_check_call_count = 0
    boundary_self_intersection_check_total_seconds = 0.0

    def timed_local_validity(candidate_uv: np.ndarray) -> tuple[bool, int]:
        nonlocal boundary_self_intersection_check_call_count
        nonlocal boundary_self_intersection_check_total_seconds
        signed = _signed_double_areas(candidate_uv, tris)
        finite_and_positive = bool(
            np.all(np.isfinite(candidate_uv))
            and np.all(signed > settings.minimum_signed_double_area)
        )
        check_started = time.perf_counter()
        boundary_count = boundary_self_intersection_count(candidate_uv, loop)
        boundary_self_intersection_check_total_seconds += time.perf_counter() - check_started
        boundary_self_intersection_check_call_count += 1
        return finite_and_positive and boundary_count == 0, boundary_count

    def timed_overlap_check(candidate_uv: np.ndarray) -> tuple[int, float]:
        nonlocal overlap_check_call_count, overlap_check_total_seconds, overlap_check_max_seconds
        nonlocal overlap_broad_phase_candidate_pair_count_total, overlap_total_possible_pair_count_total
        check_stats: dict[str, int | float] = {}
        check_started = time.perf_counter()
        count = _check_global_overlap(candidate_uv, tris, stats=check_stats)
        elapsed = time.perf_counter() - check_started
        overlap_check_call_count += 1
        overlap_check_total_seconds += elapsed
        overlap_check_max_seconds = max(overlap_check_max_seconds, elapsed)
        overlap_broad_phase_candidate_pair_count_total += int(
            check_stats.get("broad_phase_candidate_pair_count", 0)
        )
        overlap_total_possible_pair_count_total += int(check_stats.get("total_possible_pair_count", 0))
        return count, elapsed

    def timed_energy_gradient(candidate_uv: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        nonlocal energy_gradient_call_count, energy_gradient_total_seconds
        evaluation_started = time.perf_counter()
        result = _energy_and_gradient(
            candidate_uv,
            tris,
            inverse_surface,
            surface_areas,
            loop,
            barrier_epsilon,
            settings.boundary_barrier_weight,
            settings.conformal_weight,
        )
        energy_gradient_total_seconds += time.perf_counter() - evaluation_started
        energy_gradient_call_count += 1
        return result

    def timed_safe_step(candidate_uv: np.ndarray, direction: np.ndarray) -> tuple[float, str]:
        nonlocal safe_step_call_count, safe_step_total_seconds
        evaluation_started = time.perf_counter()
        result = _safe_step_limit(
            candidate_uv,
            direction,
            tris,
            loop,
            settings.minimum_signed_double_area,
        )
        safe_step_total_seconds += time.perf_counter() - evaluation_started
        safe_step_call_count += 1
        return result

    _emit_progress(progress_callback, "Initial validity check", 0.19, "local and global checks")
    initial_validity_started = time.perf_counter()
    initial_local_valid, initial_boundary_intersections = timed_local_validity(uv)
    initial_overlaps, initial_overlap_seconds = (
        timed_overlap_check(uv) if initial_local_valid else (-1, 0.0)
    )
    initial_valid = initial_local_valid and initial_overlaps == 0
    initial_validity_check_seconds = time.perf_counter() - initial_validity_started
    if not initial_valid:
        raise RuntimeError(
            "Floater initialization was not bijective: "
            f"boundary_intersections={initial_boundary_intersections}, overlaps={initial_overlaps}"
        )
    initial_diagnostics = triangle_jacobian_diagnostics(xyz, uv, tris)
    _emit_progress(progress_callback, "Initial energy evaluation", 0.24, "symmetric Dirichlet energy")
    energy, gradient, distortion_energy, boundary_energy = timed_energy_gradient(uv)
    initial_energy = float(energy)
    initial_distortion_energy = float(distortion_energy)
    initial_conformal_energy = _conformal_energy(uv, tris, inverse_surface, surface_areas)
    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    converged = False
    termination_reason = "maximum_iterations"
    accepted_iterations = 0
    rejected_line_search_steps = 0
    last_safe_step_reason = "unbounded"
    line_search_candidate_count = 0
    line_search_accepted_candidate_count = 0
    armijo_rejected_candidate_count = 0
    local_validity_rejected_candidate_count = 0
    global_overlap_rejected_candidate_count = 0
    optimization_iteration_log: list[dict[str, Any]] = []
    maximum_iterations = max(0, int(settings.max_iterations))
    boundary_mask = np.zeros(len(xyz), dtype=bool)
    boundary_mask[np.asarray(loop, dtype=int)] = True
    progress_iteration_stride = max(1, maximum_iterations // 200)

    for _iteration in range(maximum_iterations):
        iteration_started = time.perf_counter()
        iteration_overlap_started = overlap_check_total_seconds
        iteration_line_search_attempts = 0
        accepted_step: float | None = None
        last_rejection_reason = ""
        gradient -= np.mean(gradient, axis=0, keepdims=True)
        gradient_rms = float(np.linalg.norm(gradient) / math.sqrt(max(1, gradient.size)))
        iteration_fraction = 0.25 + 0.65 * ((_iteration + 1) / max(1, maximum_iterations))
        report_iteration_progress = (
            _iteration == 0
            or _iteration + 1 == maximum_iterations
            or _iteration % progress_iteration_stride == 0
        )
        if report_iteration_progress:
            _emit_progress(
                progress_callback,
                f"Optimization iteration {_iteration + 1} / {maximum_iterations}",
                iteration_fraction,
                (
                    f"energy={energy:.6g}, distortion={distortion_energy:.6g}, "
                    f"boundary={boundary_energy:.6g}, gradient RMS={gradient_rms:.3g}"
                ),
            )
        if gradient_rms <= settings.gradient_tolerance:
            converged = True
            termination_reason = "gradient_tolerance"
            optimization_iteration_log.append(
                {
                    "iteration": _iteration,
                    "energy": float(energy),
                    "distortion_energy": float(distortion_energy),
                    "boundary_energy": float(boundary_energy),
                    "gradient_rms": gradient_rms,
                    "safe_step_limit": None,
                    "safe_step_reason": "gradient_tolerance",
                    "line_search_attempts": 0,
                    "accepted_step": None,
                    "accepted": False,
                    "last_rejection_reason": "",
                    "overlap_check_seconds": 0.0,
                    "iteration_seconds": float(time.perf_counter() - iteration_started),
                }
            )
            break
        direction = _lbfgs_direction(
            gradient, history, uv, settings.low_frequency_metric_weight
        )
        direction -= np.mean(direction, axis=0, keepdims=True)
        boundary_gradient_rms = float(
            np.linalg.norm(gradient[boundary_mask])
            / math.sqrt(max(1, gradient[boundary_mask].size))
        )
        interior_gradient_rms = float(
            np.linalg.norm(gradient[~boundary_mask])
            / math.sqrt(max(1, gradient[~boundary_mask].size))
        )
        boundary_direction_rms = float(
            np.linalg.norm(direction[boundary_mask])
            / math.sqrt(max(1, direction[boundary_mask].size))
        )
        interior_direction_rms = float(
            np.linalg.norm(direction[~boundary_mask])
            / math.sqrt(max(1, direction[~boundary_mask].size))
        )
        directional_derivative = float(np.sum(gradient * direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= -1.0e-14:
            history.clear()
            direction = _lbfgs_direction(
                gradient, [], uv, settings.low_frequency_metric_weight
            )
            direction -= np.mean(direction, axis=0, keepdims=True)
            directional_derivative = float(np.sum(gradient * direction))

        accepted = False
        candidate_data: tuple[np.ndarray, float, np.ndarray, float, float, float] | None = None
        for attempt in range(2):
            safe_limit, safe_reason = timed_safe_step(uv, direction)
            last_safe_step_reason = safe_reason
            requested_step = max(float(settings.initial_step_scale), np.finfo(float).eps)
            step = (
                requested_step
                if not np.isfinite(safe_limit)
                else min(requested_step, settings.line_search_safety * safe_limit)
            )
            for _line_search in range(max(1, int(settings.line_search_max_steps))):
                # A very large gradient on a skinny valid triangle can require
                # a tiny scalar step while still producing a meaningful UV
                # displacement.  Do not discard such steps by an absolute
                # 1e-14 cutoff; the validity and Armijo checks below decide.
                if step <= np.finfo(float).eps:
                    break
                iteration_line_search_attempts += 1
                line_search_candidate_count += 1
                if report_iteration_progress or iteration_line_search_attempts > 1:
                    _emit_progress(
                        progress_callback,
                        f"Optimization iteration {_iteration + 1} / {maximum_iterations}",
                        iteration_fraction,
                        (
                            f"line-search candidate {iteration_line_search_attempts}; step={step:.3g}; "
                            f"safe limit={safe_limit:.3g}; reason={safe_reason}"
                        ),
                    )
                candidate = uv + step * direction
                candidate -= np.mean(candidate, axis=0, keepdims=True)
                locally_valid, _boundary_count = timed_local_validity(candidate)
                if not locally_valid:
                    local_validity_rejected_candidate_count += 1
                    last_rejection_reason = "local_validity"
                    rejected_line_search_steps += 1
                    step *= 0.5
                    continue
                candidate_energy, candidate_gradient, candidate_distortion, candidate_boundary = (
                    timed_energy_gradient(candidate)
                )
                armijo = energy + 1.0e-4 * step * directional_derivative
                if not np.isfinite(candidate_energy) or candidate_energy > armijo:
                    armijo_rejected_candidate_count += 1
                    last_rejection_reason = "armijo"
                    rejected_line_search_steps += 1
                    step *= 0.5
                    continue
                overlap_seconds = 0.0
                if settings.validate_global_overlap_each_step:
                    candidate_overlaps, overlap_seconds = timed_overlap_check(candidate)
                    if candidate_overlaps:
                        global_overlap_rejected_candidate_count += 1
                        last_rejection_reason = "global_overlap"
                        rejected_line_search_steps += 1
                        _emit_progress(
                            progress_callback,
                            f"Optimization iteration {_iteration + 1} / {maximum_iterations}",
                            iteration_fraction,
                            f"candidate rejected: overlaps={candidate_overlaps}; check={overlap_seconds:.3f}s",
                        )
                        step *= 0.5
                        continue
                candidate_data = (
                    candidate,
                    float(candidate_energy),
                    candidate_gradient,
                    float(candidate_distortion),
                    float(candidate_boundary),
                    float(step),
                )
                accepted = True
                accepted_step = float(step)
                line_search_accepted_candidate_count += 1
                if report_iteration_progress:
                    _emit_progress(
                        progress_callback,
                        f"Optimization iteration {_iteration + 1} / {maximum_iterations}",
                        iteration_fraction,
                        f"candidate accepted; overlap check={overlap_seconds:.3f}s",
                    )
                break
            if accepted:
                break
            if attempt == 0 and history:
                history.clear()
                direction = _lbfgs_direction(
                    gradient, [], uv, settings.low_frequency_metric_weight
                )
                direction -= np.mean(direction, axis=0, keepdims=True)
                directional_derivative = float(np.sum(gradient * direction))
                continue
            break
        if not accepted or candidate_data is None:
            termination_reason = "valid_line_search_exhausted"
            optimization_iteration_log.append(
                {
                    "iteration": _iteration,
                    "energy": float(energy),
                    "distortion_energy": float(distortion_energy),
                    "boundary_energy": float(boundary_energy),
                    "gradient_rms": gradient_rms,
                    "boundary_gradient_rms": boundary_gradient_rms,
                    "interior_gradient_rms": interior_gradient_rms,
                    "boundary_direction_rms": boundary_direction_rms,
                    "interior_direction_rms": interior_direction_rms,
                    "safe_step_limit": float(safe_limit),
                    "safe_step_reason": str(last_safe_step_reason),
                    "line_search_attempts": iteration_line_search_attempts,
                    "accepted_step": None,
                    "accepted": False,
                    "last_rejection_reason": last_rejection_reason,
                    "overlap_check_seconds": float(overlap_check_total_seconds - iteration_overlap_started),
                    "iteration_seconds": float(time.perf_counter() - iteration_started),
                }
            )
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
        optimization_iteration_log.append(
            {
                "iteration": _iteration,
                "energy": float(energy),
                "distortion_energy": float(distortion_energy),
                "boundary_energy": float(boundary_energy),
                "gradient_rms": gradient_rms,
                "boundary_gradient_rms": boundary_gradient_rms,
                "interior_gradient_rms": interior_gradient_rms,
                "boundary_direction_rms": boundary_direction_rms,
                "interior_direction_rms": interior_direction_rms,
                "safe_step_limit": float(safe_limit),
                "safe_step_reason": str(last_safe_step_reason),
                "line_search_attempts": iteration_line_search_attempts,
                "accepted_step": accepted_step,
                "accepted": True,
                "last_rejection_reason": last_rejection_reason,
                "overlap_check_seconds": float(overlap_check_total_seconds - iteration_overlap_started),
                "iteration_seconds": float(time.perf_counter() - iteration_started),
            }
        )
        if relative_change <= settings.relative_energy_tolerance:
            converged = True
            termination_reason = "relative_energy_tolerance"
            break

    _emit_progress(progress_callback, "Final validity check", 0.95, "local and global checks")
    final_validity_started = time.perf_counter()
    final_local_valid, boundary_intersections = timed_local_validity(uv)
    overlaps, final_overlap_seconds = timed_overlap_check(uv) if final_local_valid else (-1, 0.0)
    final_valid = final_local_valid and overlaps == 0
    final_validity_check_seconds = time.perf_counter() - final_validity_started
    if not final_valid:
        raise RuntimeError(
            "bijective free-boundary optimization lost validity: "
            f"boundary_intersections={boundary_intersections}, overlaps={overlaps}"
        )
    diagnostics = triangle_jacobian_diagnostics(xyz, uv, tris)
    final_conformal_energy = _conformal_energy(uv, tris, inverse_surface, surface_areas)
    lambda_values = np.asarray(diagnostics["raw_lambda_uv_to_surface_sigma_max"], dtype=float)
    log_lambda = np.log(lambda_values)
    anisotropy = np.asarray(diagnostics["anisotropy"], dtype=float)
    valid_lambda = lambda_values[np.isfinite(lambda_values) & (lambda_values > 0.0)]
    valid_log = log_lambda[np.isfinite(log_lambda)]
    valid_anisotropy = anisotropy[np.isfinite(anisotropy)]
    signed = _signed_double_areas(uv, tris)
    boundary_ids = np.asarray(loop, dtype=int)
    boundary_displacement = np.linalg.norm(uv[boundary_ids] - initial_uv[boundary_ids], axis=1)
    initial_boundary_radius = np.linalg.norm(
        initial_uv[boundary_ids] - np.mean(initial_uv[boundary_ids], axis=0, keepdims=True), axis=1
    )
    final_boundary_radius = np.linalg.norm(
        uv[boundary_ids] - np.mean(uv[boundary_ids], axis=0, keepdims=True), axis=1
    )
    boundary_nonsimilarity_rms, boundary_nonsimilarity_relative_rms = (
        _boundary_nonsimilarity_change(initial_uv[boundary_ids], uv[boundary_ids])
    )
    warning = "" if converged else f"Optimization stopped with {termination_reason}; the returned map remains bijective."
    metrics: dict[str, Any] = {
        **topology,
        "parameterization_method": "bijective_free_boundary",
        "parameterization_exactness_label": "smith_schaefer_2015_algorithm_independent_implementation",
        "parameterization_runtime_seconds": float(time.perf_counter() - started),
        "parameterization_warning": warning,
        "flattening_backend": "local_bijective_free_boundary_symmetric_dirichlet",
        "omega_parameterization_solver": "floater_then_validity_preserving_lbfgs_symmetric_dirichlet",
        "initialization_method": "floater_mean_value_equal_area_convex_boundary",
        "initialization_boundary_shape": str(settings.initial_boundary_shape),
        "lbfgs_initial_inverse_hessian": "positive_vertex_diagonal_plus_low_frequency_chart_metric",
        "lbfgs_low_frequency_metric_weight": float(settings.low_frequency_metric_weight),
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
        "surface_vertex_count": len(xyz),
        "surface_triangle_count": len(tris),
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
        "initial_conformal_energy": float(initial_conformal_energy),
        "final_conformal_energy": float(final_conformal_energy),
        "conformal_energy_definition": "area * (frobenius_norm(J)^2 / det(J) - 2); zero iff sigma1 == sigma2",
        "conformal_energy_weight": float(settings.conformal_weight),
        "conformal_constraint_enabled": bool(settings.conformal_weight > 0.0),
        "final_boundary_barrier_energy": float(boundary_energy),
        "boundary_displacement_rms": float(np.sqrt(np.mean(boundary_displacement**2))),
        "boundary_displacement_max": float(np.max(boundary_displacement)),
        "initial_boundary_circle_fit_relative_rms": _circle_fit_relative_rms(
            initial_uv[boundary_ids]
        ),
        "final_boundary_circle_fit_relative_rms": _circle_fit_relative_rms(uv[boundary_ids]),
        "boundary_nonsimilarity_change_rms": boundary_nonsimilarity_rms,
        "boundary_nonsimilarity_change_relative_rms": boundary_nonsimilarity_relative_rms,
        "initial_omega_boundary": initial_uv[
            np.asarray(loop + [loop[0]], dtype=int)
        ].tolist(),
        "initial_boundary_radius_cv": float(
            np.std(initial_boundary_radius) / max(np.mean(initial_boundary_radius), _EPS)
        ),
        "final_boundary_radius_cv": float(
            np.std(final_boundary_radius) / max(np.mean(final_boundary_radius), _EPS)
        ),
        "optimization_requested_max_iterations": int(maximum_iterations),
        "optimization_iteration_count": int(accepted_iterations),
        "optimization_converged": bool(converged),
        "optimization_succeeded": bool(accepted_iterations > 0 or converged),
        "optimization_termination_reason": termination_reason,
        "optimization_rejected_line_search_step_count": int(rejected_line_search_steps),
        "line_search_candidate_count": int(line_search_candidate_count),
        "line_search_accepted_candidate_count": int(line_search_accepted_candidate_count),
        "armijo_rejected_candidate_count": int(armijo_rejected_candidate_count),
        "local_validity_rejected_candidate_count": int(local_validity_rejected_candidate_count),
        "global_overlap_rejected_candidate_count": int(global_overlap_rejected_candidate_count),
        "global_overlap_validation_each_step": bool(settings.validate_global_overlap_each_step),
        "overlap_check_call_count": int(overlap_check_call_count),
        "overlap_check_total_seconds": float(overlap_check_total_seconds),
        "overlap_check_max_seconds": float(overlap_check_max_seconds),
        "overlap_check_mean_seconds": float(
            overlap_check_total_seconds / overlap_check_call_count if overlap_check_call_count else 0.0
        ),
        "overlap_broad_phase_candidate_pair_count_total": int(
            overlap_broad_phase_candidate_pair_count_total
        ),
        "overlap_total_possible_pair_count_total": int(overlap_total_possible_pair_count_total),
        "overlap_broad_phase_candidate_reduction_fraction": float(
            1.0 - overlap_broad_phase_candidate_pair_count_total / overlap_total_possible_pair_count_total
            if overlap_total_possible_pair_count_total
            else 0.0
        ),
        "energy_gradient_call_count": int(energy_gradient_call_count),
        "energy_gradient_total_seconds": float(energy_gradient_total_seconds),
        "safe_step_call_count": int(safe_step_call_count),
        "safe_step_total_seconds": float(safe_step_total_seconds),
        "boundary_self_intersection_check_call_count": int(
            boundary_self_intersection_check_call_count
        ),
        "boundary_self_intersection_check_total_seconds": float(
            boundary_self_intersection_check_total_seconds
        ),
        "floater_initialization_seconds": float(initialization_seconds),
        # Backward-compatible timing key for existing diagnostics.
        "tutte_initialization_seconds": float(initialization_seconds),
        "surface_differentials_seconds": float(surface_differentials_seconds),
        "initial_validity_check_seconds": float(initial_validity_check_seconds),
        "final_validity_check_seconds": float(final_validity_check_seconds),
        "initial_overlap_check_seconds": float(initial_overlap_seconds),
        "final_overlap_check_seconds": float(final_overlap_seconds),
        "optimization_iteration_log": optimization_iteration_log,
        "line_search_last_safe_step_reason": last_safe_step_reason,
        "boundary_barrier_epsilon": float(barrier_epsilon),
        "boundary_barrier_weight": float(settings.boundary_barrier_weight),
        "line_search_initial_step_scale": float(settings.initial_step_scale),
        "line_search_safety": float(settings.line_search_safety),
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
        "smith_schaefer_2015_algorithm_aligned": True,
        "smith_schaefer_components": [
            "valid Floater mean-value initialization",
            "area-weighted symmetric isometric/Dirichlet barrier distortion",
            "scale-invariant conformal distortion penalty",
            "whole-chart L-BFGS including unconstrained boundary vertices",
            "first triangle-degeneracy step bound",
            "first moving boundary edge/vertex collision step bound",
            "validity-preserving backtracking line search",
        ],
        "implementation_simplifications": [
            "O(m^2) boundary candidate search instead of a spatial hash",
            "custom limited-memory BFGS loop instead of the paper implementation",
            "exact global-overlap audit at initialization and completion",
        ],
        "onestring_grid_loss_used": False,
        "lambda_directly_optimized": False,
        "topology_modified": False,
        "seams_or_cuts_added": False,
    }
    _emit_progress(
        progress_callback,
        "S -> Omega complete",
        1.0,
        (
            f"iterations={accepted_iterations}; energy={energy:.6g}; "
            f"overlap checks={overlap_check_call_count}"
        ),
    )
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
            max_iterations=int(getattr(params, "bijective_free_boundary_max_iterations", 1000)),
            gradient_tolerance=float(getattr(params, "bijective_free_boundary_gradient_tolerance", 1.0e-7)),
            relative_energy_tolerance=float(
                getattr(params, "bijective_free_boundary_energy_tolerance", 1.0e-8)
            ),
            line_search_max_steps=int(
                getattr(params, "bijective_free_boundary_line_search_max_steps", 20)
            ),
            line_search_safety=float(getattr(params, "bijective_free_boundary_line_search_safety", 0.9)),
            initial_step_scale=float(
                getattr(params, "bijective_free_boundary_initial_step_scale", 3.0)
            ),
            conformal_weight=float(
                getattr(params, "bijective_free_boundary_conformal_weight", 4.0)
            ),
            boundary_barrier_weight=float(
                getattr(params, "bijective_free_boundary_boundary_barrier_weight", 1.0)
            ),
            initial_boundary_shape=str(
                getattr(params, "bijective_free_boundary_initial_boundary_shape", "circle")
            ),
        )
        uv, loop, metrics = bijective_free_boundary_parameterization(
            vertices,
            faces,
            config,
            progress_callback=getattr(params, "_bijective_free_boundary_progress_callback", None),
        )
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
