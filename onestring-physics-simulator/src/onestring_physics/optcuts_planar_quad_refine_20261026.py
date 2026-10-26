"""OneString-aware refinement of a *valid* planar all-quad M2D.

Fixed source OptCuts UV / 3D triangle pairings remain authoritative.  Mesh
interior UV vertices may move, but boundary vertices are pinned, each proposal
must have a valid strict barycentric inverse, and every incident quad must
retain positive area and reasonable angles. The goal is *manufacturing*
quality after lifting, not simply pretty 2D quads.

This is original experimental engineering code, not Zhang et al. (2021).
"""
from __future__ import annotations

import numpy as np


def boundary_vertices(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, int)
    owners: dict[tuple[int, int], int] = {}
    for face in f:
        for k in range(4):
            e = tuple(sorted((int(face[k]), int(face[(k+1) % 4]))))
            owners[e] = owners.get(e, 0) + 1
    return np.asarray(sorted({x for e, n in owners.items() if n == 1 for x in e}), int)


def validate_disk_mesh(uv, faces) -> dict[str, int]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    f = np.asarray(faces, int)
    v = np.asarray(uv, float)
    if not len(f) or f.ndim != 2 or f.shape[1] != 4:
        raise ValueError("PLANAR_QUAD_NOT_ALL_QUADS")
    if np.any(f < 0) or np.any(f >= len(v)) or any(len(set(face)) != 4 for face in f):
        raise ValueError("PLANAR_QUAD_FACE_INDEX_OR_REPEAT_INVALID")
    owners: dict[tuple[int, int], int] = {}
    for face in f:
        for k in range(4):
            e = tuple(sorted((int(face[k]), int(face[(k + 1) % 4]))))
            owners[e] = owners.get(e, 0) + 1
    if max(owners.values()) > 2:
        raise ValueError("PLANAR_QUAD_NONMANIFOLD_EDGE")
    polys = [Polygon(v[face]) for face in f]
    if not all(p.is_valid and p.area > 1e-14 for p in polys):
        raise ValueError("PLANAR_QUAD_INVALID_FACE")
    orientations = np.asarray([
        .5 * np.sum(v[f0, 0] * np.roll(v[f0, 1], -1)
                    - v[f0, 1] * np.roll(v[f0, 0], -1)) for f0 in f])
    if np.any(orientations * orientations[0] <= 0):
        raise ValueError("PLANAR_QUAD_WINDING_OR_COLLAPSE")
    union = unary_union(polys)
    area_sum = float(sum(p.area for p in polys))
    if area_sum - float(union.area) > 1e-8 * max(area_sum, 1.):
        raise ValueError("PLANAR_QUAD_FACE_OVERLAP")
    if union.geom_type != "Polygon" or len(union.interiors):
        raise ValueError("PLANAR_QUAD_DISCONNECTED_OR_HOLE")
    if len(boundary_vertices(f)) % 2:
        raise ValueError("PLANAR_QUAD_BOUNDARY_PARITY_INVALID")
    return {"quad_count": len(f), "boundary_vertices": len(boundary_vertices(f))}


def _quad_penalty(quads_uv, quads_xyz, h_uv, h_xyz):
    """Relative edge, angle and noncollapsed area (2D & 3D)."""
    e2 = np.roll(quads_uv, -1, axis=1) - quads_uv
    e3 = np.roll(quads_xyz, -1, axis=1) - quads_xyz
    l2 = np.maximum(np.linalg.norm(e2, axis=-1), 1e-14)
    l3 = np.maximum(np.linalg.norm(e3, axis=-1), 1e-14)
    # a scale-invariant square objective and bounded element-size term
    e3_shape = np.var(np.log(l3), axis=1)
    e3_size = np.mean(np.log(l3 / max(h_xyz, 1e-12)) ** 2, axis=1)
    e2_size = np.mean(np.log(l2 / max(h_uv, 1e-12)) ** 2, axis=1)
    cos3 = np.sum(e3 * np.roll(e3, 1, axis=1), axis=-1) / (
        l3 * np.roll(l3, 1, axis=1))
    angle = np.mean(cos3**2, axis=1)
    diag2 = np.linalg.norm(quads_uv[:, 2] - quads_uv[:, 0], axis=1)
    diag3 = np.linalg.norm(quads_uv[:, 3] - quads_uv[:, 1], axis=1)
    diag_reg = np.log(np.maximum(diag2, 1e-12) / np.maximum(diag3, 1e-12)) ** 2
    return 1.8 * e3_shape + .7 * angle + .4 * e3_size + .35 * e2_size + .15 * diag_reg


def _incident_uv_quad_is_valid(uv, faces, ids, sign, area_min):
    pts = uv[faces[ids]]
    e = np.roll(pts, -1, axis=1) - pts
    orient = np.cross(e, np.roll(e, -1, axis=1))
    if np.any(orient * sign <= area_min):
        return False
    # No folds; corners must be convex with a finite safety margin.
    lengths = np.linalg.norm(e, axis=-1)
    if np.any(lengths <= np.sqrt(area_min) * .01):
        return False
    return True


def relax_m3d_quality(
    uv: np.ndarray, faces: np.ndarray, parameterization,
    inverse_fn, *, sweeps: int = 2, max_moves: int = 2000,
    displacement_fraction: float = .16,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Conservative coordinate search; refuse every invalid mesh candidate.

    The boundary is pinned, as is the source UV triangulation (so S↔Omega
    bijectivity and original 3D seam correspondence never change).
    """
    from shapely.geometry import Polygon
    uv = np.asarray(uv, float).copy()
    faces = np.asarray(faces, int)
    validate_disk_mesh(uv, faces)
    nfaces = len(faces)
    if not 1 <= nfaces <= 20000:
        raise ValueError("PLANAR_QUAD_REFINEMENT_BAD_FACE_COUNT")
    xyz = inverse_fn(np.asarray(parameterization.uv_vertices_2d, float),
                     parameterization.uv_faces,
                     parameterization.surface_vertices_3d,
                     parameterization.surface_faces, uv)
    banned = set(boundary_vertices(faces).tolist())
    incident: list[list[int]] = [[] for _ in uv]
    neighbors: list[set[int]] = [set() for _ in uv]
    for fi, face in enumerate(faces):
        for i, idx in enumerate(face):
            incident[int(idx)].append(fi)
            neighbors[int(idx)].update(int(j) for j in face if j != idx)
    lens2 = np.linalg.norm(uv[faces] - np.roll(uv[faces], -1, axis=1), axis=2)
    lens3 = np.linalg.norm(xyz[faces] - np.roll(xyz[faces], -1, axis=1), axis=2)
    h2 = max(float(np.median(lens2)), 1e-9)
    h3 = max(float(np.median(lens3)), 1e-9)
    signed = np.sum(uv[faces, 0] * np.roll(uv[faces, 1], -1, axis=1) -
                    uv[faces, 1] * np.roll(uv[faces, 0], -1, axis=1), axis=1)
    sign = 1. if np.median(signed) > 0 else -1.
    min_area = 1e-8 * h2 * h2
    # Baseline energy includes mean loss and mild worst-panel emphasis.
    score = _quad_penalty(uv[faces], xyz[faces], h2, h3)
    baseline = float(np.mean(score) + .1 * np.max(score))
    accepted = 0
    candidates = [i for i in range(len(uv)) if i not in banned and len(incident[i]) > 0]
    moves_left = max(0, int(max_moves))
    for _sweep in range(max(0, int(sweeps))):
        improved = 0
        # Improve bad-panel neighborhoods first.
        order = sorted(candidates, key=lambda i: max(score[incident[i]]), reverse=True)
        for i in order:
            if moves_left <= 0:
                break
            faces_i = np.asarray(incident[i], int)
            if not len(faces_i):
                continue
            original = uv[i].copy()
            nearby = uv[np.asarray(sorted(neighbors[i]), int)]
            center = np.mean(nearby, axis=0)
            vecs = [center - original]
            eps = min(h2 * displacement_fraction, max(h2 * .01, np.linalg.norm(center - original)))
            vecs += [np.array([eps, 0.]), np.array([-eps, 0.]),
                     np.array([0., eps]), np.array([0., -eps])]
            penalty_old = float(np.sum(score[faces_i]))
            for d in vecs:
                moves_left -= 1
                if moves_left < 0:
                    break
                step = np.asarray(d, float)
                n = np.linalg.norm(step)
                if n < 1e-12:
                    continue
                if n > h2 * displacement_fraction:
                    step *= h2 * displacement_fraction / n
                uv[i] = original + step
                if not _incident_uv_quad_is_valid(uv, faces, faces_i, sign, min_area):
                    continue
                try:
                    xyz_trial = inverse_fn(
                        parameterization.uv_vertices_2d, parameterization.uv_faces,
                        parameterization.surface_vertices_3d,
                        parameterization.surface_faces, uv[i:i+1])[0]
                except Exception:
                    continue
                if not np.isfinite(xyz_trial).all():
                    continue
                hold = xyz[i].copy()
                xyz[i] = xyz_trial
                new_score = _quad_penalty(uv[faces[faces_i]], xyz[faces[faces_i]], h2, h3)
                if np.isfinite(new_score).all() and float(np.sum(new_score)) < penalty_old - 1e-9:
                    score[faces_i] = new_score
                    accepted += 1
                    improved += 1
                    original = uv[i].copy()
                    penalty_old = float(np.sum(new_score))
                else:
                    xyz[i] = hold
                    uv[i] = original
            uv[i] = original
        if not improved:
            break
    # Global post-audit; an off-branch update must never be mistaken as success.
    validate_disk_mesh(uv, faces)
    final = float(np.mean(score) + .1 * np.max(score))
    if final > baseline + 1e-8:
        raise ValueError("PLANAR_QUAD_3D_REFINEMENT_REGRESSED")
    return uv, {
        "planar_quad_refine_3d_moves_accepted": int(accepted),
        "planar_quad_refine_3d_energy_before": float(baseline),
        "planar_quad_refine_3d_energy_after": float(final),
        "planar_quad_m3d_quality_optimized": bool(accepted),
        "planar_quad_optimization_model": "interior-vertex constrained UV coordinate search; source OptCuts UV fixed",
    }
