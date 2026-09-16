"""Experimental paper Eq.(5) K2D on an explicit auxetic tile topology.

The input M2D is the ordinary shared-vertex quad mesh used by the rest of the
pipeline.  For this dedicated 2026-09-17 mode we first duplicate every quad
corner, so K2D is an independent rigid-tile/linkage representation.  Copies
that came from the same M2D vertex form a hinge group.  EFab is then evaluated
on gaps between rays belonging to different incident tiles, never on a quad's
own interior corner angle.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(math.acos(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0)))


def _rot(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.environ.get(name, str(fallback)))
    except Exception:
        return float(fallback)


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(float(os.environ.get(name, str(fallback))))
    except Exception:
        return int(fallback)


def _build_auxetic_topology(mesh_2d: Any):
    """Duplicate every quad corner and retain its source-M2D vertex id.

    Returns independent tile vertices/faces plus hinge groups and EFab gap-ray
    pairs.  A gap-ray pair is (hinge_copy_a, neighbour_a, hinge_copy_b,
    neighbour_b).  The two rays always belong to different tiles.
    """
    src_xy = np.asarray(mesh_2d.vertices[:, :2], dtype=float)
    src_faces = np.asarray(mesh_2d.faces, dtype=int)
    n_faces = len(src_faces)
    xy = np.zeros((4 * n_faces, 2), dtype=float)
    faces = np.arange(4 * n_faces, dtype=int).reshape(n_faces, 4)
    source_ids = np.zeros(4 * n_faces, dtype=int)
    tile_ids = np.repeat(np.arange(n_faces, dtype=int), 4)
    local_ids = np.tile(np.arange(4, dtype=int), n_faces)
    groups: dict[int, list[int]] = {}

    for f, face in enumerate(src_faces):
        for local, source in enumerate(face):
            q = 4 * f + local
            xy[q] = src_xy[int(source)]
            source_ids[q] = int(source)
            groups.setdefault(int(source), []).append(q)

    # Each tile corner contributes two inward tile-edge rays.  Around a shared
    # M2D vertex, sort all rays by polar angle and pair consecutive rays only
    # when they come from different tiles.  These are the physical openings
    # between incident tiles; same-tile consecutive rays are the tile interior
    # angle and are deliberately excluded from EFab.
    gap_pairs: list[tuple[int, int, int, int]] = []
    for source, copies in groups.items():
        rays: list[tuple[float, int, int, int]] = []
        center = src_xy[source]
        for q in copies:
            f, local = int(tile_ids[q]), int(local_ids[q])
            for nl in ((local - 1) % 4, (local + 1) % 4):
                nb = 4 * f + nl
                vec = xy[nb] - center
                if float(np.linalg.norm(vec)) <= 1e-12:
                    continue
                rays.append((math.atan2(float(vec[1]), float(vec[0])), f, q, nb))
        rays.sort(key=lambda item: item[0])
        if len(rays) < 2:
            continue
        seen: set[tuple[int, int, int, int]] = set()
        for i in range(len(rays)):
            _, fa, ca, na = rays[i]
            _, fb, cb, nb = rays[(i + 1) % len(rays)]
            if fa == fb:
                continue
            key = (ca, na, cb, nb)
            rev = (cb, nb, ca, na)
            if key in seen or rev in seen:
                continue
            seen.add(key)
            gap_pairs.append(key)

    hinge_groups = [np.asarray(v, dtype=int) for v in groups.values() if len(v) > 1]
    return xy, faces, source_ids, hinge_groups, gap_pairs


def _project_hinges(xy: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    out = xy.copy()
    for ids in groups:
        p = np.mean(out[ids], axis=0)
        out[ids] = p
    return out


def _project_edges(xy: np.ndarray, faces: np.ndarray, source_ids: np.ndarray, mesh_3d: Any):
    out = xy.copy()
    acc = np.zeros_like(out)
    counts = np.zeros((len(out), 1), dtype=float)
    err = 0.0
    for face in faces:
        for k in range(4):
            a, b = int(face[k]), int(face[(k + 1) % 4])
            sa, sb = int(source_ids[a]), int(source_ids[b])
            target = float(np.linalg.norm(mesh_3d.vertices[sa] - mesh_3d.vertices[sb]))
            d = out[b] - out[a]
            length = float(np.linalg.norm(d))
            if length <= 1e-12:
                continue
            delta = 0.5 * (length - target) * d / length
            acc[a] += delta
            acc[b] -= delta
            counts[a, 0] += 1.0
            counts[b, 0] += 1.0
            err += (length - target) ** 2
    active = counts[:, 0] > 0
    out[active] += acc[active] / counts[active]
    return out, err


def _project_fab(xy: np.ndarray, gap_pairs: list[tuple[int, int, int, int]], theta_min: float):
    out = xy.copy()
    accum = np.zeros_like(out)
    counts = np.zeros((len(out), 1), dtype=float)
    energy = 0.0
    violations = 0
    angles: list[float] = []
    for ca, na, cb, nb in gap_pairs:
        # Hinge copies are coincident after _project_hinges.  Keep the two
        # centers explicit so the representation remains valid if a later
        # solver allows a small connection residual.
        va = out[na] - out[ca]
        vb = out[nb] - out[cb]
        la2, lb2 = float(np.dot(va, va)), float(np.dot(vb, vb))
        if la2 <= 1e-16 or lb2 <= 1e-16:
            continue
        theta = _angle(va, vb)
        # Consecutive polar rays define the smaller opening.  The paper's
        # fabrication interval is [theta_min, pi/2].
        target = min(max(theta, theta_min), 0.5 * math.pi)
        angles.append(theta)
        delta = target - theta
        if abs(delta) <= 1e-10:
            continue
        violations += 1
        cross = float(va[0] * vb[1] - va[1] * vb[0])
        orient = 1.0 if cross >= 0.0 else -1.0
        total = la2 + lb2
        da = -orient * delta * (lb2 / total)
        db = orient * delta * (la2 / total)
        qa = out[ca] + _rot(va, da)
        qb = out[cb] + _rot(vb, db)
        accum[na] += qa - out[na]
        accum[nb] += qb - out[nb]
        counts[na, 0] += 1.0
        counts[nb, 0] += 1.0
        energy += float(np.dot(qa - out[na], qa - out[na]) + np.dot(qb - out[nb], qb - out[nb]))
    active = counts[:, 0] > 0
    out[active] += accum[active] / counts[active]
    amin = min(angles) if angles else 0.0
    amax = max(angles) if angles else 0.0
    return out, energy, violations, amin, amax


def _sat_mtv(poly_a: np.ndarray, poly_b: np.ndarray):
    best_axis, best_depth = None, float("inf")
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
                best_axis, best_depth = axis, depth
    return None if best_axis is None else best_axis * best_depth


def _project_collisions(xy: np.ndarray, faces: np.ndarray):
    out = xy.copy()
    accum = np.zeros_like(out)
    counts = np.zeros((len(out), 1), dtype=float)
    collisions = 0
    energy = 0.0
    polys = [out[f] for f in faces]
    for i in range(len(faces)):
        amin, amax = np.min(polys[i], axis=0), np.max(polys[i], axis=0)
        for j in range(i + 1, len(faces)):
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
    active = counts[:, 0] > 0
    out[active] += accum[active] / counts[active]
    return out, energy, collisions


def optimize_paper_eq5(mesh_2d: Any, mesh_3d: Any, params: Any, *, progress_callback: Any = None, pipeline: Any = None):
    start = time.perf_counter()
    xy0, faces, source_ids, hinge_groups, gap_pairs = _build_auxetic_topology(mesh_2d)
    xy = xy0.copy()
    centroid0 = np.mean(xy0, axis=0, keepdims=True)

    w_edge = _env_float("ONESTRING_EQ5_W_EDGE", float(getattr(params, "w_edge", 1.0)))
    w_collision = _env_float("ONESTRING_EQ5_W_COLLISION", float(getattr(params, "w_collision", 1.0)))
    w_fab = _env_float("ONESTRING_EQ5_W_FAB", float(getattr(params, "w_fab", 0.001)))
    theta_deg = _env_float("ONESTRING_EQ5_THETA_MIN_DEG", 5.0)
    theta_min = float(np.clip(math.radians(theta_deg), 0.0, 0.5 * math.pi))
    iterations = max(1, _env_int("ONESTRING_EQ5_ITERATIONS", max(80, int(getattr(params, "max_2d_iterations", 40)) * 6)))
    print(f"[PAPER-EQ5-SETTINGS] w_edge={w_edge:g} w_collision={w_collision:g} w_fab={w_fab:g} theta_min_deg={theta_deg:g} iterations={iterations}")
    print(f"[PAPER-EQ5-AUXETIC] source_vertices={len(mesh_2d.vertices)} independent_vertices={len(xy)} tiles={len(faces)} hinge_groups={len(hinge_groups)} gap_angles={len(gap_pairs)}")

    final_collisions = final_fab_violations = 0
    amin = amax = 0.0
    for it in range(iterations):
        old = xy.copy()
        xy = _project_hinges(xy, hinge_groups)
        edge_candidate, _ = _project_edges(xy, faces, source_ids, mesh_3d)
        collision_candidate, _, final_collisions = _project_collisions(xy, faces)
        fab_candidate, _, final_fab_violations, amin, amax = _project_fab(xy, gap_pairs, theta_min)
        total = max(w_edge + w_collision + w_fab, 1e-12)
        xy = (w_edge * edge_candidate + w_collision * collision_candidate + w_fab * fab_candidate) / total
        xy = _project_hinges(xy, hinge_groups)
        xy += centroid0 - np.mean(xy, axis=0, keepdims=True)
        step = float(np.max(np.linalg.norm(xy - old, axis=1))) if len(xy) else 0.0
        if it == 0 or (it + 1) % max(1, iterations // 8) == 0 or it + 1 == iterations:
            print(f"[PAPER-EQ5-K2D-ITER] iter={it+1} collisions={final_collisions} fab_violations={final_fab_violations} gap_min_deg={math.degrees(amin):.4g} gap_max_deg={math.degrees(amax):.4g} step={step:.6g}")
        if progress_callback is not None and (it % max(1, iterations // 30) == 0 or it + 1 == iterations):
            try:
                progress_callback("Paper Eq.5 auxetic K2D", (it + 1) / iterations, f"iter {it+1}/{iterations}; gaps={len(gap_pairs)}; fab={final_fab_violations}")
            except Exception:
                pass
        if step < 1e-9:
            break

    _, edge_energy = _project_edges(xy, faces, source_ids, mesh_3d)
    _, collision_energy, final_collisions = _project_collisions(xy, faces)
    _, fab_energy, final_fab_violations, amin, amax = _project_fab(xy, gap_pairs, theta_min)
    edge_count = max(1, 4 * len(faces))
    edge_mean = math.sqrt(max(0.0, edge_energy) / edge_count)
    vertices = np.column_stack([xy, np.zeros(len(xy))])
    metrics = dict(getattr(mesh_2d, "metrics", {}) or {})
    metrics.update({
        "objective": "E_Flat = w1*EEdge + w2*ECollision + w3*EFab on explicit auxetic tile topology",
        "paper_eq5_unified": True,
        "paper_eq5_auxetic_topology": True,
        "paper_eq5_fab_topology_verified": True,
        "paper_weight_w1_edge": w_edge,
        "paper_weight_w2_collision": w_collision,
        "paper_weight_w3_fab": w_fab,
        "paper_fab_theta_min_rad": theta_min,
        "paper_fab_theta_max_rad": 0.5 * math.pi,
        "paper_fab_gap_constraint_count": int(len(gap_pairs)),
        "paper_fab_violation_count": int(final_fab_violations),
        "paper_fab_gap_min_deg": float(math.degrees(amin)),
        "paper_fab_gap_max_deg": float(math.degrees(amax)),
        "paper_fab_projection_energy": float(fab_energy),
        "paper_collision_projection_energy": float(collision_energy),
        "collision_count_after": int(final_collisions),
        "2d_collision_count": int(final_collisions),
        "edge_matching_error": float(edge_mean),
        "optimizer_iterations": int(it + 1),
        "actual_backend": "numpy_auxetic_projective_eq5",
        "auxetic_source_vertex_ids": source_ids.tolist(),
        "auxetic_source_face_count": int(len(mesh_2d.faces)),
    })
    out = type(mesh_2d)(vertices, faces.copy(), mesh_2d.grid, "K2D", metrics, list(getattr(mesh_2d, "split_lines", [])))

    report_type = getattr(pipeline, "StageReport", None) if pipeline is not None else None
    if report_type is None and pipeline is not None:
        report_type = getattr(getattr(pipeline, "_original", None), "StageReport", None)
    if report_type is None:
        raise RuntimeError("Paper Eq.5 auxetic solver could not resolve StageReport type")
    report = report_type(
        name="M2D -> auxetic K2D",
        objective=str(metrics["objective"]),
        before_error=0.0,
        after_error=float(edge_mean),
        constraint_violation=float(final_collisions + final_fab_violations),
        computation_time=time.perf_counter() - start,
        counts={"vertices": int(len(vertices)), "quads": int(len(faces))},
    )
    print(f"[PAPER-EQ5-K2D] auxetic=True collisions={final_collisions} fab_violations={final_fab_violations} gap_min_deg={math.degrees(amin):.4g} gap_max_deg={math.degrees(amax):.4g} edge_mean={edge_mean:.6g}")
    return out, report


__all__ = ["optimize_paper_eq5"]
