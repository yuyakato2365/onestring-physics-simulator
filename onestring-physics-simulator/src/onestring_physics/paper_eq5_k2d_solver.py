"""Paper-grounded K2D solver for One String to Pull Them All, Sec. 4.3 Eq. (5).

This module keeps EEdge, ECollision and EFab in the *same* K2D local/global
iteration.  In particular, EFab is the Appendix-A gap-angle projection, not an
anchor to M2D, and collision handling is not deferred to the hinge stage.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


def _unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        for i in range(len(face)):
            a, b = int(face[i]), int(face[(i + 1) % len(face)])
            out.add((a, b) if a < b else (b, a))
    return sorted(out)


def _rot(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(math.acos(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


def _fab_constraints(faces: np.ndarray) -> list[tuple[int, int, int]]:
    """(center, previous, next) linkage-angle triples from the quad topology."""
    triples: set[tuple[int, int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        n = len(face)
        for i in range(n):
            c, a, b = int(face[i]), int(face[(i - 1) % n]), int(face[(i + 1) % n])
            # orientation of the two rays is immaterial for the unsigned angle
            triples.add((c, min(a, b), max(a, b)))
    return sorted(triples)


def _project_fab(xy: np.ndarray, faces: np.ndarray, theta_min: float) -> tuple[np.ndarray, float, int]:
    """Appendix-A projection P_F onto theta_min <= theta <= pi/2.

    For fixed ray lengths, the nearest valid pair is obtained by rotating both
    rays in opposite directions.  The angle correction is split according to
    squared ray lengths, which minimizes ||x-v1||^2 + ||y-v2||^2 locally.
    """
    accum = np.zeros_like(xy)
    weight = np.zeros((len(xy), 1), dtype=float)
    energy = 0.0
    violations = 0
    for c, ia, ib in _fab_constraints(faces):
        va = xy[ia] - xy[c]
        vb = xy[ib] - xy[c]
        la2, lb2 = float(np.dot(va, va)), float(np.dot(vb, vb))
        if la2 <= 1e-16 or lb2 <= 1e-16:
            continue
        theta = _angle(va, vb)
        target = min(max(theta, theta_min), 0.5 * math.pi)
        delta = target - theta
        if abs(delta) <= 1e-10:
            continue
        violations += 1
        cross = float(va[0] * vb[1] - va[1] * vb[0])
        orient = 1.0 if cross >= 0.0 else -1.0
        total = la2 + lb2
        da = -orient * delta * (lb2 / total)
        db = orient * delta * (la2 / total)
        pa, pb = _rot(va, da), _rot(vb, db)
        qa, qb = xy[c] + pa, xy[c] + pb
        energy += float(np.dot(qa - xy[ia], qa - xy[ia]) + np.dot(qb - xy[ib], qb - xy[ib]))
        accum[ia] += qa - xy[ia]
        accum[ib] += qb - xy[ib]
        weight[ia, 0] += 1.0
        weight[ib, 0] += 1.0
    out = xy.copy()
    active = weight[:, 0] > 0
    out[active] += accum[active] / weight[active]
    return out, energy, violations


def _sat_mtv(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray | None:
    best_axis = None
    best_depth = float("inf")
    ca, cb = np.mean(poly_a, axis=0), np.mean(poly_b, axis=0)
    for poly in (poly_a, poly_b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.asarray([-e[1], e[0]], dtype=float)
            n = float(np.linalg.norm(axis))
            if n <= 1e-12:
                continue
            axis /= n
            aa, bb = poly_a @ axis, poly_b @ axis
            depth = min(float(np.max(aa)), float(np.max(bb))) - max(float(np.min(aa)), float(np.min(bb)))
            if depth <= 0.0:
                return None
            if depth < best_depth:
                if float(np.dot(cb - ca, axis)) < 0.0:
                    axis = -axis
                best_depth, best_axis = depth, axis
    return None if best_axis is None else best_axis * best_depth


def _project_collisions(xy: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float, int]:
    faces = np.asarray(faces, dtype=int)
    accum = np.zeros_like(xy)
    counts = np.zeros((len(xy), 1), dtype=float)
    energy = 0.0
    collisions = 0
    face_sets = [set(map(int, f)) for f in faces]
    polys = [xy[f] for f in faces]
    for i in range(len(faces)):
        amin, amax = np.min(polys[i], axis=0), np.max(polys[i], axis=0)
        for j in range(i + 1, len(faces)):
            # Topologically adjacent quads meet at linkage vertices/edges and are
            # not counted as penetrating one another.
            if face_sets[i].intersection(face_sets[j]):
                continue
            bmin, bmax = np.min(polys[j], axis=0), np.max(polys[j], axis=0)
            if np.any(np.minimum(amax, bmax) - np.maximum(amin, bmin) <= 0.0):
                continue
            mtv = _sat_mtv(polys[i], polys[j])
            if mtv is None:
                continue
            collisions += 1
            energy += float(np.dot(mtv, mtv))
            for v in faces[i]:
                accum[int(v)] -= 0.5 * mtv
                counts[int(v), 0] += 1.0
            for v in faces[j]:
                accum[int(v)] += 0.5 * mtv
                counts[int(v), 0] += 1.0
    out = xy.copy()
    active = counts[:, 0] > 0
    out[active] += accum[active] / counts[active]
    return out, energy, collisions


def optimize_paper_eq5(mesh_2d: Any, mesh_3d: Any, params: Any, *, progress_callback: Any = None, pipeline: Any = None):
    """Solve Eq. (5) by weighted projective local/global iterations."""
    start = time.perf_counter()
    xy0 = np.asarray(mesh_2d.vertices[:, :2], dtype=float)
    xy = xy0.copy()
    faces = np.asarray(mesh_2d.faces, dtype=int)
    edges = _unique_edges(faces)
    edge_idx = np.asarray(edges, dtype=int)
    aa, bb = edge_idx[:, 0], edge_idx[:, 1]
    targets = np.asarray([np.linalg.norm(mesh_3d.vertices[a] - mesh_3d.vertices[b]) for a, b in edges], dtype=float)
    degree = np.zeros((len(xy), 1), dtype=float)
    np.add.at(degree, aa, 1.0); np.add.at(degree, bb, 1.0)
    degree = np.maximum(degree, 1.0)
    w_edge = float(getattr(params, "w_edge", 1.0))
    w_collision = float(getattr(params, "w_collision", 1.0))
    w_fab = float(getattr(params, "w_fab", 0.001))
    theta_min = float(getattr(params, "theta_min", getattr(params, "min_gap_angle", math.radians(5.0))))
    if theta_min > math.pi:
        theta_min = math.radians(theta_min)
    theta_min = float(np.clip(theta_min, 0.0, 0.5 * math.pi))
    iterations = max(80, int(getattr(params, "max_2d_iterations", 40)) * 6)
    centroid0 = np.mean(xy0, axis=0, keepdims=True)
    final_collisions = 0
    final_fab_violations = 0
    for it in range(iterations):
        old = xy.copy()
        # EEdge local projection, Eq. (3)/(4) with K3D target lengths.
        d = xy[bb] - xy[aa]
        lengths = np.linalg.norm(d, axis=1)
        safe = np.maximum(lengths, 1e-12)
        corr = ((lengths - targets) / safe)[:, None] * d * 0.5
        acc = np.zeros_like(xy)
        np.add.at(acc, aa, corr); np.add.at(acc, bb, -corr)
        edge_candidate = xy + acc / degree
        # ECollision and EFab are projected in this same K2D iteration.
        collision_candidate, _, final_collisions = _project_collisions(xy, faces)
        fab_candidate, _, final_fab_violations = _project_fab(xy, faces, theta_min)
        total_w = max(w_edge + w_collision + w_fab, 1e-12)
        xy = (w_edge * edge_candidate + w_collision * collision_candidate + w_fab * fab_candidate) / total_w
        # Only remove the global translation gauge; M2D is not an EFab anchor.
        xy += centroid0 - np.mean(xy, axis=0, keepdims=True)
        if progress_callback is not None and (it % max(1, iterations // 30) == 0 or it + 1 == iterations):
            try:
                progress_callback("Paper Eq.5 K2D", (it + 1) / iterations, f"iter {it+1}/{iterations}; collisions={final_collisions}; fab={final_fab_violations}")
            except Exception:
                pass
        if float(np.max(np.linalg.norm(xy - old, axis=1))) < 1e-9:
            break
    vertices = np.column_stack([xy, np.zeros(len(xy))])
    edge_err = np.abs(np.linalg.norm(xy[bb] - xy[aa], axis=1) - targets)
    _, collision_energy, final_collisions = _project_collisions(xy, faces)
    _, fab_energy, final_fab_violations = _project_fab(xy, faces, theta_min)
    metrics = dict(getattr(mesh_2d, "metrics", {}) or {})
    metrics.update({
        "objective": "E_Flat = w1*EEdge + w2*ECollision + w3*EFab (paper Eq.5 unified local/global)",
        "paper_eq5_unified": True,
        "paper_eq5_collision_deferred": False,
        "paper_eq5_fab_is_appendix_a_angle_projection": True,
        "paper_eq5_fab_is_m2d_anchor": False,
        "paper_weight_w1_edge": w_edge,
        "paper_weight_w2_collision": w_collision,
        "paper_weight_w3_fab": w_fab,
        "paper_fab_theta_min_rad": theta_min,
        "paper_fab_theta_max_rad": 0.5 * math.pi,
        "edge_matching_error": float(np.mean(edge_err)) if len(edge_err) else 0.0,
        "max_edge_length_error_after": float(np.max(edge_err)) if len(edge_err) else 0.0,
        "collision_count_after": int(final_collisions),
        "2d_collision_count": int(final_collisions),
        "paper_collision_projection_energy": float(collision_energy),
        "paper_fab_projection_energy": float(fab_energy),
        "paper_fab_violation_count": int(final_fab_violations),
        "optimizer_iterations": int(it + 1),
        "actual_backend": "numpy_projective_local_global_eq5",
    })
    out = type(mesh_2d)(vertices, faces.copy(), mesh_2d.grid, "K2D", metrics, list(getattr(mesh_2d, "split_lines", [])))
    report_type = None
    if pipeline is not None:
        report_type = getattr(pipeline, "StageReport", None) or getattr(getattr(pipeline, "_original", None), "StageReport", None)
    if report_type is None:
        # Existing solver supplies the concrete report class; use it only as a constructor source.
        _, template = pipeline._optimize_k2d(mesh_2d, mesh_3d, params, progress_callback=None)
        report_type = type(template)
    report = report_type(
        name="M2D -> K2D",
        objective=str(metrics["objective"]),
        before_error=0.0,
        after_error=float(metrics["edge_matching_error"]),
        constraint_violation=float(final_collisions + final_fab_violations),
        computation_time=time.perf_counter() - start,
        counts={"vertices": int(len(vertices)), "quads": int(len(faces))},
    )
    print(f"[PAPER-EQ5-K2D] unified EEdge+ECollision+EFab; collisions={final_collisions} fab_violations={final_fab_violations} edge_mean={metrics['edge_matching_error']:.6g}")
    return out, report


__all__ = ["optimize_paper_eq5"]
