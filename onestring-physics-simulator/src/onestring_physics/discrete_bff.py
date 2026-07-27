"""Discrete Boundary First Flattening (BFF) for a single disk mesh.

The ``bff`` pipeline path uses the discrete Cherrier formula, a
Neumann-to-Dirichlet solve, the paper's best-fit closed boundary construction,
and harmonic extension.  It does not use LSCM.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from typing import Any

import numpy as np
from scipy import optimize, sparse
from scipy.sparse import linalg as spla

EPS = 1e-12


@dataclass(frozen=True)
class RectangleTarget:
    width: float
    height: float

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.width + self.height)

    @property
    def corners(self) -> np.ndarray:
        w, h = self.width / 2.0, self.height / 2.0
        return np.asarray([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=float)

    @property
    def side_fractions(self) -> np.ndarray:
        return np.asarray([self.width, self.height, self.width, self.height]) / self.perimeter


def _boundary_loop(faces: np.ndarray, vertex_count: int) -> tuple[list[int], dict[str, int]]:
    tris = np.asarray(faces, dtype=int)[:, :3]
    edge_counts: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    edges: set[tuple[int, int]] = set()
    graph: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in tris:
        if len(set(map(int, face))) != 3:
            raise RuntimeError("discrete BFF requires non-degenerate triangles")
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            if not (0 <= a < vertex_count and 0 <= b < vertex_count):
                raise RuntimeError("mesh contains an out-of-range vertex index")
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edges.add(key)
            directed.append((a, b))
            graph[a].add(b)
            graph[b].add(a)
    if any(count > 2 for count in edge_counts.values()):
        raise RuntimeError("discrete BFF requires a manifold disk")
    boundary_edges = [(a, b) for a, b in directed if edge_counts[(min(a, b), max(a, b))] == 1]
    if len(boundary_edges) < 3:
        raise RuntimeError("discrete BFF requires one open boundary loop")

    nxt: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for a, b in boundary_edges:
        nxt[a] = b
        incoming[b] = incoming.get(b, 0) + 1
    if len(nxt) == len(boundary_edges) and all(incoming.get(v, 0) == 1 for v in nxt):
        start = min(nxt)
        loop = [start]
        for _ in range(len(boundary_edges) - 1):
            value = nxt[loop[-1]]
            if value in loop:
                raise RuntimeError("mesh has multiple boundary loops")
            loop.append(value)
        if nxt[loop[-1]] != start:
            raise RuntimeError("boundary loop does not close")
    else:
        boundary_graph: dict[int, list[int]] = {}
        for a, b in boundary_edges:
            boundary_graph.setdefault(a, []).append(b)
            boundary_graph.setdefault(b, []).append(a)
        if any(len(values) != 2 for values in boundary_graph.values()):
            raise RuntimeError("boundary is not a simple cycle")
        start = min(boundary_graph)
        loop, previous, current = [start], -1, start
        while True:
            values = boundary_graph[current]
            value = values[0] if values[0] != previous else values[1]
            if value == start:
                break
            if value in loop:
                raise RuntimeError("mesh has multiple boundary loops")
            loop.append(value)
            previous, current = current, value
        if len(loop) != len(boundary_graph):
            raise RuntimeError("mesh has multiple boundary loops")

    active = set(map(int, tris.reshape(-1)))
    stack, visited = [next(iter(active))], set()
    while stack:
        value = stack.pop()
        if value in visited:
            continue
        visited.add(value)
        stack.extend(graph[value] - visited)
    chi = len(active) - len(edges) + len(tris)
    if visited != active or chi != 1:
        raise RuntimeError(f"discrete BFF requires one connected disk; got chi={chi}")
    return loop, {
        "bff_topology_vertex_count": len(active),
        "bff_topology_edge_count": len(edges),
        "bff_topology_face_count": len(tris),
        "bff_topology_boundary_count": 1,
        "bff_topology_euler_characteristic": chi,
    }


def _triangle_angles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(vertices, float)[np.asarray(faces, int)[:, :3]]
    result = np.zeros((len(tri), 3))
    for c in range(3):
        a = tri[:, (c + 1) % 3] - tri[:, c]
        b = tri[:, (c + 2) % 3] - tri[:, c]
        denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), EPS)
        result[:, c] = np.arccos(np.clip(np.sum(a * b, axis=1) / denom, -1.0, 1.0))
    return result


def _laplacian(vertices: np.ndarray, faces: np.ndarray) -> tuple[sparse.csr_matrix, dict[str, int]]:
    pts, tris = np.asarray(vertices, float), np.asarray(faces, int)[:, :3]
    weights: dict[tuple[int, int], float] = {}
    negative = degenerate = 0
    for face in tris:
        p = pts[face]
        for opposite, a, b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            u, v = p[a] - p[opposite], p[b] - p[opposite]
            cross = float(np.linalg.norm(np.cross(u, v)))
            if cross <= 1e-15:
                cot, degenerate = 0.0, degenerate + 1
            else:
                cot = float(np.dot(u, v) / cross)
            negative += int(cot < -1e-12)
            i, j = int(face[a]), int(face[b])
            key = (min(i, j), max(i, j))
            weights[key] = weights.get(key, 0.0) + 0.5 * cot
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diagonal = np.zeros(len(pts))
    for (i, j), weight in weights.items():
        if abs(weight) <= 1e-16 or not np.isfinite(weight):
            continue
        diagonal[i] += weight
        diagonal[j] += weight
        rows += [i, j]
        cols += [j, i]
        data += [-weight, -weight]
    for i, value in enumerate(diagonal):
        rows.append(i)
        cols.append(i)
        data.append(float(value))
    return sparse.coo_matrix((data, (rows, cols)), shape=(len(pts), len(pts))).tocsr(), {
        "bff_cotangent_negative_count": negative,
        "bff_cotangent_degenerate_contribution_count": degenerate,
    }


def _curvatures(vertices: np.ndarray, faces: np.ndarray, loop: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sums = np.zeros(len(vertices))
    for face, angles in zip(np.asarray(faces, int)[:, :3], _triangle_angles(vertices, faces)):
        for vertex, angle in zip(face, angles):
            sums[int(vertex)] += float(angle)
    boundary = set(loop)
    gaussian = np.asarray([0.0 if i in boundary else 2.0 * math.pi - sums[i] for i in range(len(vertices))])
    geodesic = np.asarray([math.pi - sums[i] for i in loop])
    gb = float(np.sum(gaussian) + np.sum(geodesic))
    return gaussian, geodesic, {
        "bff_gaussian_curvature_sum": float(np.sum(gaussian)),
        "bff_boundary_geodesic_curvature_sum": float(np.sum(geodesic)),
        "bff_gauss_bonnet_sum": gb,
        "bff_gauss_bonnet_error": abs(gb - 2.0 * math.pi),
    }


def _pca(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, float) - np.mean(points, axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    area = 0.5 * np.sum(projected[:, 0] * np.roll(projected[:, 1], -1) - np.roll(projected[:, 0], -1) * projected[:, 1])
    if area < 0:
        projected[:, 1] *= -1
    return projected


def _target(vertices: np.ndarray, loop: list[int], params: Any) -> tuple[RectangleTarget, dict[str, Any]]:
    projected = _pca(np.asarray(vertices, float)[loop])
    span = np.maximum(np.ptp(projected, axis=0), EPS)
    mode = str(getattr(params, "boundary_target_aspect_mode", "lscm_initial"))
    if mode == "fixed":
        raw = float(getattr(params, "boundary_target_aspect_ratio", 1.0))
        source = "fixed_parameter"
    elif mode in {"lscm_initial", "projected_initial"}:
        raw = float(span[0] / span[1])
        source = "best_fit_plane_PCA_projection_of_3D_boundary"
    else:
        raise ValueError(f"unknown boundary_target_aspect_mode: {mode}")
    lo = max(1e-3, float(getattr(params, "boundary_target_aspect_min", 0.2)))
    hi = max(lo, float(getattr(params, "boundary_target_aspect_max", 5.0)))
    aspect = float(np.clip(raw if np.isfinite(raw) and raw > 0 else 1.0, lo, hi))
    height = 2.0 / max(1.0, aspect)
    target = RectangleTarget(aspect * height, height)
    return target, {
        "boundary_target_shape": "rectangle",
        "boundary_target_aspect_mode": mode,
        "boundary_target_aspect_mode_legacy_lscm_name_reinterpreted": mode == "lscm_initial",
        "boundary_target_aspect_source": source,
        "boundary_target_aspect_ratio_raw": raw,
        "boundary_target_aspect_ratio": aspect,
        "boundary_target_width": target.width,
        "boundary_target_height": target.height,
    }


def _corners(vertices: np.ndarray, loop: list[int], target: RectangleTarget) -> tuple[list[int], list[int], dict[str, Any]]:
    points = _pca(np.asarray(vertices, float)[loop])
    normalized = 2.0 * (points - np.mean(points, axis=0)) / np.maximum(np.ptp(points, axis=0), EPS)
    directions = np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) / math.sqrt(2.0)
    scores = normalized @ directions.T
    count = min(10, len(loop))
    candidates = [np.argsort(scores[:, i])[-count:][::-1] for i in range(4)]
    best, best_score, n = None, -float("inf"), len(loop)
    for values in itertools.product(*candidates):
        p0, p1, p2, p3 = map(int, values)
        relative = np.asarray([0, (p1 - p0) % n, (p2 - p0) % n, (p3 - p0) % n])
        if not (0 < relative[1] < relative[2] < relative[3] < n):
            continue
        gaps = np.diff(np.r_[relative, n]) / n
        score = float(sum(scores[p, c] for c, p in enumerate(values))) - 1.5 * float(np.sum((gaps - target.side_fractions) ** 2))
        if score > best_score:
            best, best_score = (p0, p1, p2, p3), score
    if best is None:
        raise RuntimeError("could not select four ordered rectangle corners")
    p0, p1, p2, p3 = best
    positions = [0, (p1 - p0) % n, (p2 - p0) % n, (p3 - p0) % n]
    ordered = loop[p0:] + loop[:p0]
    return ordered, positions, {
        "boundary_corner_vertex_ids": [ordered[p] for p in positions],
        "boundary_corner_loop_positions": positions,
        "boundary_corner_selection_score": best_score,
        "boundary_corner_selection_model": "3D_boundary_PCA_extrema_with_cyclic_aspect_penalty",
    }


def _edge_lengths(points: np.ndarray, loop: list[int]) -> np.ndarray:
    pts = np.asarray(points, float)
    return np.asarray([np.linalg.norm(pts[loop[(i + 1) % len(loop)]] - pts[loop[i]]) for i in range(len(loop))])


def _rectangle_samples(target: RectangleTarget, lengths: np.ndarray, positions: list[int]) -> np.ndarray:
    result, n = np.zeros((len(lengths), 2)), len(lengths)
    corners = target.corners
    for side in range(4):
        start = positions[side]
        end = positions[side + 1] if side < 3 else n
        segment = lengths[start:end]
        if end <= start or float(np.sum(segment)) <= EPS:
            raise RuntimeError("every rectangle side needs at least one positive-length boundary edge")
        t = np.r_[0.0, np.cumsum(segment[:-1])] / np.sum(segment)
        result[start:end] = (1.0 - t[:, None]) * corners[side] + t[:, None] * corners[(side + 1) % 4]
    return result


def _target_angles(samples: np.ndarray, source_lengths: np.ndarray) -> tuple[np.ndarray, float]:
    incoming, outgoing = samples - np.roll(samples, 1, axis=0), np.roll(samples, -1, axis=0) - samples
    angles = np.arctan2(incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0], np.sum(incoming * outgoing, axis=1))
    if float(np.sum(angles)) < 0:
        angles *= -1
    defect = 2.0 * math.pi - float(np.sum(angles))
    dual = 0.5 * (source_lengths + np.roll(source_lengths, 1))
    angles += defect * dual / np.sum(dual)
    return angles, defect


def _neumann_to_dirichlet(
    laplacian: sparse.csr_matrix,
    gaussian: np.ndarray,
    loop: list[int],
    geodesic: np.ndarray,
    target_angles: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rhs = -np.asarray(gaussian, float)
    rhs[np.asarray(loop, int)] -= np.asarray(geodesic) - np.asarray(target_angles)
    compatibility = float(np.sum(rhs))
    rhs -= np.mean(rhs)
    system = laplacian + sparse.identity(len(rhs), format="csr") * 1e-8
    try:
        u, solver = spla.spsolve(system, rhs), "spsolve_regularized_zero_neumann"
    except Exception:
        u, solver = spla.lsqr(system, rhs, atol=1e-11, btol=1e-11, iter_lim=max(2000, 20 * len(rhs)))[0], "lsqr_regularized_zero_neumann"
    u = np.asarray(u, float)
    if not np.all(np.isfinite(u)):
        raise RuntimeError("BFF Neumann-to-Dirichlet solve returned non-finite values")
    u -= np.mean(u[np.asarray(loop, int)])
    return u, {
        "bff_neumann_to_dirichlet_solver": solver,
        "bff_neumann_regularization": 1e-8,
        "bff_neumann_compatibility_sum_before_projection": compatibility,
        "bff_neumann_rhs_sum_after_projection": float(np.sum(rhs)),
        "bff_neumann_residual_norm": float(np.linalg.norm(system @ u - rhs)),
    }


def _positive_qp(desired: np.ndarray, tangents: np.ndarray, source: np.ndarray, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    weights = 1.0 / np.maximum(source, EPS)
    fun = lambda x: 0.5 * float(np.dot(weights * (x - desired), x - desired))
    jac = lambda x: weights * (x - desired)
    result = optimize.minimize(
        fun,
        np.maximum(desired, 10.0 * epsilon),
        jac=jac,
        method="SLSQP",
        bounds=[(epsilon, None)] * len(desired),
        constraints={"type": "eq", "fun": lambda x: tangents @ x, "jac": lambda _x: tangents},
        options={"ftol": 1e-12, "maxiter": max(500, 10 * len(desired))},
    )
    corrected = np.asarray(result.x, float)
    closure = float(np.linalg.norm(tangents @ corrected))
    if not result.success or closure > 1e-8 * max(1.0, float(np.sum(corrected))) or np.min(corrected) <= 0:
        raise RuntimeError(f"positive BFF closure QP failed: {result.message}; closure={closure:.3e}")
    return corrected, {
        "bff_positive_length_qp_used": True,
        "bff_positive_length_qp_status": str(result.message),
        "bff_positive_length_qp_iterations": int(result.nit),
    }


def _best_fit_curve(desired: np.ndarray, angles: np.ndarray, source: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target_edge = np.roll(samples, -1, axis=0)[0] - samples[0]
    phi = math.atan2(float(target_edge[1]), float(target_edge[0])) - float(angles[0])
    tangents = np.zeros((2, len(desired)))
    for i, angle in enumerate(angles):
        phi += float(angle)
        tangents[:, i] = (math.cos(phi), math.sin(phi))
    diagonal = np.maximum(source, EPS)
    gram_inv = np.linalg.pinv((tangents * diagonal[None, :]) @ tangents.T, rcond=1e-13)
    unconstrained = desired - diagonal * (tangents.T @ (gram_inv @ (tangents @ desired)))
    corrected = unconstrained
    qp = {"bff_positive_length_qp_used": False, "bff_positive_length_qp_status": "not_needed", "bff_positive_length_qp_iterations": 0}
    epsilon = max(1e-12, 1e-10 * float(np.mean(np.maximum(desired, EPS))))
    if float(np.min(corrected)) <= epsilon:
        corrected, qp = _positive_qp(desired, tangents, source, epsilon)
    boundary = np.zeros((len(desired), 2))
    for i in range(1, len(desired)):
        boundary[i] = boundary[i - 1] + corrected[i - 1] * tangents[:, i - 1]
    relative = np.abs(corrected - desired) / np.maximum(desired, EPS)
    return boundary, corrected, {
        **qp,
        "bff_best_fit_closure_model": "paper_Eq20_weighted_minimum_length_correction_with_fixed_exterior_angles",
        "bff_best_fit_closure_error": float(np.linalg.norm(tangents @ corrected)),
        "bff_best_fit_min_corrected_length": float(np.min(corrected)),
        "bff_best_fit_max_relative_length_adjustment": float(np.max(relative)),
        "bff_best_fit_mean_relative_length_adjustment": float(np.mean(relative)),
        "bff_best_fit_negative_length_count_unconstrained": int(np.sum(unconstrained <= 0)),
    }


def _extend(laplacian: sparse.csr_matrix, loop: list[int], boundary_uv: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    boundary = np.asarray(loop, int)
    interior = np.asarray([i for i in range(laplacian.shape[0]) if i not in set(loop)], int)
    uv = np.zeros((laplacian.shape[0], 2))
    uv[boundary] = boundary_uv
    solver, residual = "boundary_only", 0.0
    if len(interior):
        aii, aib = laplacian[interior[:, None], interior], laplacian[interior[:, None], boundary]
        rhs = -aib @ uv[boundary]
        try:
            uv[interior, 0] = spla.spsolve(aii, np.asarray(rhs[:, 0]).ravel())
            uv[interior, 1] = spla.spsolve(aii, np.asarray(rhs[:, 1]).ravel())
            solver = "cotangent_harmonic_dirichlet_spsolve"
        except Exception:
            uv[interior, 0] = spla.lsqr(aii, np.asarray(rhs[:, 0]).ravel(), atol=1e-11, btol=1e-11)[0]
            uv[interior, 1] = spla.lsqr(aii, np.asarray(rhs[:, 1]).ravel(), atol=1e-11, btol=1e-11)[0]
            solver = "cotangent_harmonic_dirichlet_lsqr"
        residual = float(np.linalg.norm(aii @ uv[interior] - rhs))
    return uv, {
        "bff_extension_solver": solver,
        "bff_extension_model": "official_prescribed-angle_path_harmonic_extension_of_BFF-compatible_boundary",
        "bff_extension_interior_vertex_count": len(interior),
        "bff_extension_residual_norm": residual,
    }


def _normalize(uv: np.ndarray, loop: list[int], faces: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(uv, float).copy()
    out -= np.mean(out[np.asarray(loop, int)], axis=0)
    tri = np.asarray(faces, int)[:, :3]
    def areas(values: np.ndarray) -> np.ndarray:
        p = values[tri]
        return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    if float(np.median(areas(out))) < 0:
        out[:, 1] *= -1
    boundary = out[np.asarray(loop, int)]
    scale = 2.0 / max(float(np.max(np.ptp(boundary, axis=0))), EPS)
    out *= scale
    signed = areas(out)
    return out, {
        "bff_uv_normalization_scale": scale,
        "uv_triangle_flip_count": int(np.sum(signed < -1e-12)),
        "uv_degenerate_triangle_count": int(np.sum(np.abs(signed) <= 1e-12)),
        "uv_min_triangle_area": float(np.min(np.abs(signed))) if len(signed) else 0.0,
    }


def discrete_bff_rectangle(vertices: np.ndarray, faces: np.ndarray, params: Any) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    started = time.perf_counter()
    pts, tris = np.asarray(vertices, float), np.asarray(faces, int)[:, :3]
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 4 or len(tris) < 2 or not np.all(np.isfinite(pts)):
        raise RuntimeError("discrete BFF requires a finite Vx3 single-disk triangle mesh")
    loop, topology = _boundary_loop(tris, len(pts))
    target, target_metrics = _target(pts, loop, params)
    loop, corner_positions, corner_metrics = _corners(pts, loop, target)
    source_lengths = _edge_lengths(pts, loop)
    samples = _rectangle_samples(target, source_lengths, corner_positions)
    target_angles, angle_defect = _target_angles(samples, source_lengths)
    laplacian, laplace_metrics = _laplacian(pts, tris)
    gaussian, geodesic, curvature_metrics = _curvatures(pts, tris, loop)
    u, ntd_metrics = _neumann_to_dirichlet(laplacian, gaussian, loop, geodesic, target_angles)
    boundary = np.asarray(loop, int)
    desired = np.exp(0.5 * (u[boundary] + u[np.roll(boundary, -1)])) * source_lengths
    boundary_uv, corrected, closure_metrics = _best_fit_curve(desired, target_angles, source_lengths, samples)
    uv, extension_metrics = _extend(laplacian, loop, boundary_uv)
    uv, validity = _normalize(uv, loop, tris)
    final_span = np.maximum(np.ptp(uv[boundary], axis=0), EPS)
    achieved = float(max(final_span) / min(final_span))
    requested = float(target_metrics["boundary_target_aspect_ratio"])
    requested = max(requested, 1.0 / max(requested, EPS))
    metrics: dict[str, Any] = {
        **topology, **target_metrics, **corner_metrics, **laplace_metrics,
        **curvature_metrics, **ntd_metrics, **closure_metrics,
        **extension_metrics, **validity,
        "parameterization_method": "bff",
        "parameterization_exactness_label": "discrete_bff_rectangular_target",
        "flattening_backend": "local_discrete_bff_cherrier_ps_best_fit_curve",
        "omega_parameterization_solver": "discrete_BFF_NtD_BestFitCurve_harmonic_extension",
        "omega_boundary_constraint_model": "target_exterior_angles_plus_BFF_compatible_scale_factors_and_minimum_closure_correction",
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_fixed": False,
        "omega_boundary_shape": "BFF-compatible rectangle with minimally corrected edge lengths",
        "omega_boundary_model": "Discrete BFF rectangle: Cherrier NtD and Eq.20 best-fit closure",
        "bff_implemented": True,
        "bff_backend_used": "local_discrete_bff",
        "bff_reference_backend_available": False,
        "bff_reference_library": "local NumPy/SciPy implementation aligned with the paper and official C++ path",
        "bff_variant": "prescribed_rectangle_exterior_angles_discrete_cherrier_best_fit_curve",
        "bff_cherrier_formula_implemented": True,
        "bff_poincare_steklov_operator": "Neumann-to-Dirichlet via regularized cotangent Laplacian",
        "bff_best_fit_curve_implemented": True,
        "bff_uses_lscm": False,
        "lscm_implemented_in_this_path": False,
        "harmonic_solve_performed": True,
        "constrained_lscm_solve_performed": False,
        "boundary_loop": list(map(int, loop)),
        "boundary_target_corners": target.corners.tolist(),
        "boundary_target_exterior_angle_sum": float(np.sum(target_angles)),
        "boundary_target_curvature_closure_defect_corrected": angle_defect,
        "boundary_desired_length_sum": float(np.sum(desired)),
        "boundary_corrected_length_sum": float(np.sum(corrected)),
        "boundary_achieved_unoriented_aspect_ratio": achieved,
        "boundary_requested_unoriented_aspect_ratio": requested,
        "boundary_aspect_relative_error": abs(achieved - requested) / max(requested, EPS),
        "parameterization_runtime_seconds": time.perf_counter() - started,
        "parameterization_warning": "",
    }
    if validity["uv_triangle_flip_count"]:
        metrics["parameterization_warning"] = "Discrete BFF produced UV flips; no LSCM fallback was substituted."
    if validity["uv_degenerate_triangle_count"]:
        metrics["parameterization_warning"] = (metrics["parameterization_warning"] + "; degenerate UV triangles detected").strip("; ")
    return uv, loop, metrics


def install_discrete_bff(pipeline_module: Any) -> None:
    """Patch the compatibility wrapper so ``omega_parameterization_mode='bff'`` uses discrete BFF."""
    if getattr(pipeline_module, "_DISCRETE_BFF_PATCH_INSTALLED", False):
        return
    legacy = pipeline_module._build_surface_parameterization

    def build(surface: Any, target: Any, grid: Any, params: Any) -> Any:
        mode = str(getattr(params, "omega_parameterization_mode", "bff"))
        debug = str(getattr(params, "m3d_construction_mode", "mesh_harmonic")) == "analytic_scaled_heightfield_debug"
        if mode != "bff" or debug:
            return legacy(surface, target, grid, params)
        if str(getattr(params, "omega_boundary_mode", "paper_default")) != "paper_default":
            raise ValueError("discrete bff requires omega_boundary_mode='paper_default'")
        vertices = np.asarray(surface.vertices, float)
        faces = np.asarray(surface.faces[:, :3], int)
        uv, loop, bff_metrics = discrete_bff_rectangle(vertices, faces, params)
        slope = {"mean_slope": 0.0, "max_slope": 0.0} if getattr(target, "kind", "") == "sampled" else pipeline_module._original._heightfield_metric_summary(target, grid)
        metrics = {
            **bff_metrics,
            "omega_boundary_mode": "paper_default",
            "omega_parameterization_mode": "bff",
            "requested_omega_parameterization_mode": "bff",
            "surface_vertex_count": len(vertices),
            "surface_triangle_count": len(faces),
            "boundary_vertex_count": len(loop),
            "mean_slope": float(slope["mean_slope"]),
            "max_slope": float(slope["max_slope"]),
            "height_field_shortcut_used": False,
            "omega_corresponds_to_S": True,
            "omega_correspondence_model": "discrete BFF map c:S->Omega; inverse by UV triangle lookup",
            "paper_flow_stage": "S -> Omega by discrete BFF curvature prescription and best-fit closure",
            "paper_exactness_warning": "Local discrete implementation; validate numerically against official BFF before claiming reference equivalence.",
            "omega_warning": str(bff_metrics.get("parameterization_warning", "")),
        }
        out = pipeline_module._original.SurfaceParameterization(
            method="bff",
            surface_vertices_3d=vertices,
            surface_faces=faces,
            uv_vertices_2d=uv,
            uv_faces=faces.copy(),
            omega_boundary=uv[loop + [loop[0]]],
            triangle_acceleration=None,
            metrics=metrics,
        )
        marker = getattr(pipeline_module, "_mark_parameterization_mode", None)
        if callable(marker):
            out = marker(out, method="bff", exactness="discrete_bff_rectangular_target", warning=metrics["parameterization_warning"])
        return out

    pipeline_module._build_surface_parameterization = build
    pipeline_module._original._build_surface_parameterization = build
    pipeline_module._DISCRETE_BFF_PATCH_INSTALLED = True
