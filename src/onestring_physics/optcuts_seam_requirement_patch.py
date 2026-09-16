"""Requirement-preserving OptCuts seam post-processing for OneString.

User-visible contract (do not silently replace this with another policy):
1. OptCuts chooses the seam topology.
2. Omega is re-parameterized so every maximal degree-2 seam chain is straight.
3. Each straight chain is aligned to the existing fabrication grid (axis + phase).
4. M2D may realize that chain only as one straight grid path; a stair/L fallback is
   treated as a requirement violation rather than an acceptable approximation.

The implementation keeps OptCuts' duplicated UV seam sides separated.  We move
all UV copies belonging to one canonical surface vertex by the same displacement;
therefore the seam *centre line* is constrained while the local cut opening is
preserved.  The displacement is propagated to the rest of Omega by a constrained
harmonic solve.  Seam targets are hard constraints and candidate maps that flip
or collapse any UV triangle are rejected.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math
import os
from typing import Any

import numpy as np

from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust
from .optcuts_rectilinear_seam_patch import _extract_source_chains


def _canonical_uv_copies(parameterization: Any) -> dict[int, list[int]]:
    """Return canonical 3D vertex -> all UV vertex ids used for that point."""
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    if len(xyz) == 0 or len(sf) == 0 or len(sf) != len(uf):
        return {}
    span = max(float(np.nanmax(xyz) - np.nanmin(xyz)), 1.0)
    tol = max(1e-10 * span, 1e-12)
    key_to_canonical: dict[tuple[int, int, int], int] = {}
    canonical_of = np.empty(len(xyz), dtype=int)
    for vi, p in enumerate(xyz[:, :3]):
        key = tuple(np.rint(p / tol).astype(np.int64).tolist())
        if key not in key_to_canonical:
            key_to_canonical[key] = len(key_to_canonical)
        canonical_of[vi] = int(key_to_canonical[key])

    out: dict[int, set[int]] = defaultdict(set)
    for face3, face2 in zip(sf, uf):
        for raw_vid, uv_vid in zip(np.asarray(face3, int), np.asarray(face2, int)):
            out[int(canonical_of[int(raw_vid)])].add(int(uv_vid))
    return {k: sorted(v) for k, v in out.items()}


def _chain_targets(
    nodes: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    *,
    grid_size: float | None = None,
    phase_u: float = 0.0,
    phase_v: float = 0.0,
    angle_degrees: float = 0.0,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build hard targets for seam-chain centres.

    With ``grid_size is None`` this is the straightening stage: each maximal
    degree-2 chain is projected to its best-fit line.  With ``grid_size`` set,
    the already-straight chain is expressed in grid coordinates and placed on
    the nearest horizontal/vertical lattice line without changing its coordinate
    along that chosen grid axis.
    """
    chains = _extract_source_chains(nodes, edges)
    proposals: dict[int, list[np.ndarray]] = defaultdict(list)
    chain_axes: list[str] = []
    theta = math.radians(float(angle_degrees))
    c, s = math.cos(theta), math.sin(theta)
    # world -> grid coordinates; inverse is transpose because R is orthonormal.
    world_from_grid = np.asarray([[c, -s], [s, c]], dtype=float)
    grid_from_world = world_from_grid.T

    for chain in chains:
        ids = [int(v) for v in chain if int(v) in nodes]
        if len(ids) < 2:
            continue
        pts = np.asarray([nodes[v] for v in ids], dtype=float)[:, :2]
        center = np.mean(pts, axis=0)
        centered = pts - center[None, :]
        if len(pts) == 2:
            direction = pts[1] - pts[0]
        else:
            _u, _sv, vh = np.linalg.svd(centered, full_matrices=False)
            direction = np.asarray(vh[0], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = np.asarray([1.0, 0.0], dtype=float)
        else:
            direction = direction / norm
        straight = center[None, :] + (centered @ direction)[:, None] * direction[None, :]

        if grid_size is None:
            targets = straight
            chain_axes.append("free")
        else:
            h = max(float(grid_size), 1e-12)
            q = (grid_from_world @ straight.T).T
            qdir = grid_from_world @ direction
            horizontal = abs(float(qdir[0])) >= abs(float(qdir[1]))
            if horizontal:
                lattice = float(phase_v) + round((float(np.mean(q[:, 1])) - float(phase_v)) / h) * h
                q[:, 1] = lattice
                chain_axes.append("grid_u")
            else:
                lattice = float(phase_u) + round((float(np.mean(q[:, 0])) - float(phase_u)) / h) * h
                q[:, 0] = lattice
                chain_axes.append("grid_v")
            targets = (world_from_grid @ q.T).T

        for vid, target in zip(ids, targets):
            proposals[int(vid)].append(np.asarray(target, dtype=float))

    # Junctions can belong to several chains.  A single UV position must satisfy
    # all incident chains, so reconcile only the shared endpoint by averaging the
    # independently proposed hard targets.  Degree-2 interior nodes have one target.
    merged = {vid: np.mean(np.asarray(vals, float), axis=0) for vid, vals in proposals.items()}
    return merged, {
        "chain_count": int(len(chains)),
        "chain_axes": chain_axes,
        "junction_target_count": int(sum(len(v) > 1 for v in proposals.values())),
    }


def _triangle_signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, float)[np.asarray(faces, int)]
    if tri.shape[1] != 3:
        return np.zeros(len(tri), dtype=float)
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _valid_same_orientation(reference_uv: np.ndarray, candidate_uv: np.ndarray, faces: np.ndarray) -> bool:
    before = _triangle_signed_areas(reference_uv, faces)
    after = _triangle_signed_areas(candidate_uv, faces)
    if len(before) == 0:
        return True
    scale = max(float(np.nanmax(np.abs(before))), 1.0)
    eps = max(1e-12, 1e-10 * scale)
    mask = np.abs(before) > eps
    if np.any(np.abs(after[mask]) <= eps):
        return False
    return bool(np.all(np.sign(after[mask]) == np.sign(before[mask])))


def _uv_graph(faces: np.ndarray, n_vertices: int) -> tuple[list[set[int]], set[int]]:
    adjacency = [set() for _ in range(int(n_vertices))]
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for face in np.asarray(faces, int):
        ids = [int(x) for x in face]
        for i in range(len(ids)):
            a, b = ids[i], ids[(i + 1) % len(ids)]
            if a == b:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_count[tuple(sorted((a, b)))] += 1
    boundary = {v for edge, count in edge_count.items() if count == 1 for v in edge}
    return adjacency, boundary


def _harmonic_deform_with_hard_targets(
    uv: np.ndarray,
    faces: np.ndarray,
    hard_targets: dict[int, np.ndarray],
) -> np.ndarray:
    """Propagate hard seam targets through Omega while preserving UV validity."""
    base = np.asarray(uv, dtype=float)
    n = len(base)
    if n == 0 or not hard_targets:
        return base.copy()
    adjacency, boundary = _uv_graph(faces, n)
    hard = {int(k): np.asarray(v, float) for k, v in hard_targets.items() if 0 <= int(k) < n}
    # Outer/cut boundary vertices not belonging to the seam are anchors.  This
    # keeps the global Omega placement stable while still allowing the interior
    # to absorb the seam straightening displacement.
    fixed: dict[int, np.ndarray] = {int(v): base[int(v)].copy() for v in boundary if int(v) not in hard}
    fixed.update(hard)
    free = [i for i in range(n) if i not in fixed]
    if not free:
        out = base.copy()
        for i, p in fixed.items():
            out[i] = p
        return out

    free_index = {v: i for i, v in enumerate(free)}
    # Try increasingly local regularization.  Hard seam targets never soften;
    # only the free-vertex response changes.  We accept only a non-flipped map.
    for mu in (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        m = len(free)
        A = np.zeros((m, m), dtype=float)
        B = np.zeros((m, 2), dtype=float)
        for row, v in enumerate(free):
            nbrs = adjacency[v]
            deg = max(len(nbrs), 1)
            A[row, row] = float(deg) + float(mu)
            B[row] += float(mu) * base[v]
            for w in nbrs:
                if w in free_index:
                    A[row, free_index[w]] -= 1.0
                else:
                    B[row] += fixed.get(int(w), base[int(w)])
        try:
            solved = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            solved, *_ = np.linalg.lstsq(A, B, rcond=None)
        out = base.copy()
        out[np.asarray(free, int)] = solved
        for i, p in fixed.items():
            out[int(i)] = p
        if _valid_same_orientation(base, out, faces):
            return out
    raise RuntimeError(
        "OPTCUTS_SEAM_OMEGA_CONSTRAINT_INFEASIBLE: hard straight/grid seam targets "
        "would flip or collapse UV triangles"
    )


def _apply_node_targets(parameterization: Any, node_targets: dict[int, np.ndarray]) -> Any:
    payload = extract_connected_seam_payload_robust(parameterization)
    nodes = {int(k): np.asarray(v, float) for k, v in dict(payload.get("nodes", {}) or {}).items()}
    copies = _canonical_uv_copies(parameterization)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    hard: dict[int, np.ndarray] = {}
    for canonical, target in node_targets.items():
        if int(canonical) not in nodes:
            continue
        delta = np.asarray(target, float) - np.asarray(nodes[int(canonical)], float)
        for uv_id in copies.get(int(canonical), []):
            hard[int(uv_id)] = uv[int(uv_id)] + delta
    if not hard:
        return parameterization
    new_uv = _harmonic_deform_with_hard_targets(uv, faces, hard)
    result = copy.copy(parameterization)
    try:
        result.uv_vertices_2d = new_uv
    except Exception:
        object.__setattr__(result, "uv_vertices_2d", new_uv)
    return result


def _max_chain_line_error(nodes: dict[int, np.ndarray], edges: list[tuple[int, int]]) -> float:
    worst = 0.0
    for chain in _extract_source_chains(nodes, edges):
        pts = np.asarray([nodes[int(v)] for v in chain if int(v) in nodes], float)
        if len(pts) <= 2:
            continue
        a, b = pts[0], pts[-1]
        d = b - a
        denom = float(np.dot(d, d))
        if denom <= 1e-20:
            worst = max(worst, float(np.max(np.linalg.norm(pts - a[None, :], axis=1))))
            continue
        t = ((pts - a[None, :]) @ d) / denom
        proj = a[None, :] + t[:, None] * d[None, :]
        worst = max(worst, float(np.max(np.linalg.norm(pts - proj, axis=1))))
    return float(worst)


def install_optcuts_seam_requirement_patch(pipeline: Any) -> None:
    """Install after OptCuts parameterization and seam-metadata support."""
    if getattr(pipeline, "_onestring_optcuts_seam_requirement_patch_installed", False):
        return
    base_flatten = pipeline._flatten_to_domain

    def flatten_with_required_seam_sequence(parameterization: Any, grid: Any, params: Any = None):
        if str(getattr(parameterization, "method", "")) != "optcuts":
            return base_flatten(parameterization, grid, params)
        payload0 = extract_connected_seam_payload_robust(parameterization)
        nodes0 = {int(k): np.asarray(v, float) for k, v in dict(payload0.get("nodes", {}) or {}).items()}
        edges0 = [(int(a), int(b)) for a, b in list(payload0.get("edges", []) or [])]
        if not nodes0 or not edges0:
            return base_flatten(parameterization, grid, params)

        # Stage 2: straighten the seam in Omega.
        straight_targets, straight_stats = _chain_targets(nodes0, edges0)
        straight_param = _apply_node_targets(parameterization, straight_targets)
        payload1 = extract_connected_seam_payload_robust(straight_param)
        nodes1 = {int(k): np.asarray(v, float) for k, v in dict(payload1.get("nodes", {}) or {}).items()}
        edges1 = [(int(a), int(b)) for a, b in list(payload1.get("edges", []) or [])]
        h = max(float(getattr(grid, "tile_size", 0.0) or getattr(params, "tile_size", 0.0) or 0.0), 1e-8)

        # Stage 3: align each now-straight chain to an actual fabrication-grid line.
        grid_targets, grid_stats = _chain_targets(
            nodes1,
            edges1,
            grid_size=h,
            phase_u=float(os.environ.get("ONESTRING_OPTCUTS_GRID_PHASE_U", "0") or 0.0),
            phase_v=float(os.environ.get("ONESTRING_OPTCUTS_GRID_PHASE_V", "0") or 0.0),
            angle_degrees=float(os.environ.get("ONESTRING_OPTCUTS_GRID_ANGLE_DEGREES", "0") or 0.0),
        )
        aligned_param = _apply_node_targets(straight_param, grid_targets)
        payload2 = extract_connected_seam_payload_robust(aligned_param)
        nodes2 = {int(k): np.asarray(v, float) for k, v in dict(payload2.get("nodes", {}) or {}).items()}
        edges2 = [(int(a), int(b)) for a, b in list(payload2.get("edges", []) or [])]
        line_error = _max_chain_line_error(nodes2, edges2)
        tol = max(1e-8, 1e-5 * h)
        if line_error > tol:
            raise RuntimeError(
                f"OPTCUTS_SEAM_REQUIREMENT_NOT_MET: seam is not straight after Omega/grid alignment "
                f"(max_error={line_error:.6g}, tolerance={tol:.6g})"
            )

        domain = base_flatten(aligned_param, grid, params)
        setattr(domain, "_optcuts_requirement_parameterization", aligned_param)
        setattr(domain, "_optcuts_requirement_sequence", "seam->omega_straight->grid_align")
        setattr(domain, "_optcuts_requirement_metrics", {
            "straight_chain_count": int(straight_stats["chain_count"]),
            "grid_chain_count": int(grid_stats["chain_count"]),
            "grid_chain_axes": list(grid_stats["chain_axes"]),
            "max_final_chain_line_error": float(line_error),
            "line_tolerance": float(tol),
            "tile_size": float(h),
        })
        return domain

    pipeline._flatten_to_domain = flatten_with_required_seam_sequence
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = flatten_with_required_seam_sequence
    pipeline._onestring_optcuts_seam_requirement_patch_installed = True


def install_strict_straight_grid_seam_verifier(pipeline: Any) -> None:
    """Reject M2D output if a required seam became an L/stair grid path."""
    if getattr(pipeline, "_onestring_strict_straight_grid_seam_verifier_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_and_verify(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        if getattr(domain, "_optcuts_requirement_sequence", None) != "seam->omega_straight->grid_align":
            return mesh
        paths = list(getattr(mesh, "_optcuts_grid_seam_paths", []) or [])
        verts = np.asarray(mesh.vertices, float)
        tol = max(1e-8, 1e-5 * max(float(getattr(grid, "tile_size", 0.0) or 0.0), 1e-8))
        nonstraight = 0
        for path in paths:
            pts = verts[np.asarray(path, int), :2]
            if len(pts) <= 2:
                continue
            dx = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            dy = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            if min(dx, dy) > tol:
                nonstraight += 1
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update(dict(getattr(domain, "_optcuts_requirement_metrics", {}) or {}))
        metrics.update({
            "optcuts_requirement_sequence": "seam->omega_straight->grid_align",
            "optcuts_requirement_grid_path_count": int(len(paths)),
            "optcuts_requirement_nonstraight_grid_path_count": int(nonstraight),
        })
        mesh.metrics.update(metrics)
        if nonstraight:
            raise RuntimeError(
                f"OPTCUTS_GRID_ALIGNMENT_REQUIREMENT_NOT_MET: {nonstraight} seam path(s) "
                "required an L/stair route after Omega straightening"
            )
        return mesh

    pipeline._build_m2d = build_and_verify
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_and_verify
    pipeline._onestring_strict_straight_grid_seam_verifier_installed = True


__all__ = [
    "install_optcuts_seam_requirement_patch",
    "install_strict_straight_grid_seam_verifier",
    "_chain_targets",
]
