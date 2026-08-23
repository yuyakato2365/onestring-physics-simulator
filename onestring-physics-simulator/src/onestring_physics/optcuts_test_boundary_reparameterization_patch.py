"""Experimental OptCuts_test boundary reparameterization.

Flow
----
1. Run ordinary official OptCuts and keep its cut topology.
2. Build the same reference M2D footprint used by OneString from the initial Omega.
3. Extract the outer boundary of the retained quad-cell union.
4. Reparameterize the *same cut mesh* with that grid-cell outline as the outer
   boundary target while preserving all OptCuts seam UV copies as hard boundary
   constraints.

No seam topology is recomputed here. UV vertex ids / UV faces are unchanged.
The OptCuts seam is therefore the same cut; only the remaining embedding is
re-solved around the grid-derived outline.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np

from .optcuts_grid_constrained_parameterization_patch import (
    _optimize_constrained_uv,
    _surface_seam_records,
)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _quad_union_boundary(mesh: Any) -> np.ndarray:
    """Return the longest ordered outer loop of a retained quad-cell union."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    incidence: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        ids = [int(v) for v in face]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            incidence[_edge_key(a, b)] += 1
    boundary_edges = [edge for edge, count in incidence.items() if count == 1]
    if not boundary_edges:
        return np.zeros((0, 2), dtype=float)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    unused = {_edge_key(a, b) for a, b in boundary_edges}
    loops: list[list[int]] = []
    while unused:
        first = next(iter(unused))
        start, nxt = int(first[0]), int(first[1])
        loop = [start, nxt]
        unused.discard(_edge_key(start, nxt))
        cur = nxt
        for _ in range(len(boundary_edges) + 4):
            candidates = [v for v in adjacency[cur] if _edge_key(cur, v) in unused]
            if not candidates:
                break
            following = int(candidates[0])
            unused.discard(_edge_key(cur, following))
            if following == loop[0]:
                loop.append(following)
                break
            loop.append(following)
            cur = following
        if len(loop) >= 4:
            loops.append(loop)
    if not loops:
        return np.zeros((0, 2), dtype=float)

    def perimeter(loop: list[int]) -> float:
        pts = vertices[np.asarray(loop, dtype=int), :2]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    chosen = max(loops, key=perimeter)
    pts = vertices[np.asarray(chosen, dtype=int), :2]
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-10:
        pts = np.vstack([pts, pts[0]])
    return np.asarray(pts, dtype=float)


def _nearest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polyline, dtype=float)
    best = None
    best_d2 = float("inf")
    for a, b in zip(poly[:-1], poly[1:]):
        d = b - a
        denom = float(np.dot(d, d))
        t = 0.0 if denom <= 1e-20 else float(np.clip(np.dot(p - a, d) / denom, 0.0, 1.0))
        q = a + t * d
        d2 = float(np.dot(p - q, p - q))
        if d2 < best_d2:
            best_d2 = d2
            best = q
    return p.copy() if best is None else np.asarray(best, dtype=float)


def _uv_boundary_vertices(parameterization: Any) -> set[int]:
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    incidence: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        ids = [int(v) for v in face]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            incidence[_edge_key(a, b)] += 1
    out: set[int] = set()
    for (a, b), count in incidence.items():
        if count == 1:
            out.add(int(a))
            out.add(int(b))
    return out


def _seam_uv_vertices(parameterization: Any) -> set[int]:
    seam: set[int] = set()
    for record in _surface_seam_records(parameterization):
        for copy in record["copies"]:
            seam.update(int(v) for v in copy.values())
    return seam


def _build_test_targets(parameterization: Any, grid_outline: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    targets = np.full_like(uv, np.nan)
    seam_ids = _seam_uv_vertices(parameterization)
    boundary_ids = _uv_boundary_vertices(parameterization)
    outer_ids = sorted(boundary_ids - seam_ids)

    # Keep the existing OptCuts seam boundary coordinates exactly during this
    # experimental re-solve. This preserves both cut topology and seam geometry.
    for uid in seam_ids:
        if 0 <= uid < len(uv):
            targets[uid] = uv[uid]

    for uid in outer_ids:
        if 0 <= uid < len(uv):
            targets[uid] = _nearest_point_on_polyline(uv[uid], grid_outline)

    return targets, {
        "seam_fixed_vertex_count": int(len(seam_ids)),
        "outer_boundary_vertex_count": int(len(outer_ids)),
        "uv_boundary_vertex_count": int(len(boundary_ids)),
    }


def _reference_domain_without_invalid_optcuts_boundary_diagnostic(
    pipeline: Any,
    parameterization: Any,
    grid: Any,
    params: Any,
):
    """Build the regular overlay while suppressing one incompatible diagnostic.

    ``_reference_flatten_to_domain`` uses ``metrics['boundary_loop']`` as surface
    vertex ids only for its Gaussian-curvature diagnostic.  In an OptCuts result,
    boundary_loop contains UV ids and can include duplicated seam vertices beyond
    the original surface-vertex count.  The overlay construction itself is valid.
    Temporarily hiding that loop therefore avoids the invalid UV-id -> surface-id
    indexing without changing UV geometry, seam topology, or the resulting grid.
    """
    metrics = parameterization.metrics
    had_loop = "boundary_loop" in metrics
    saved_loop = metrics.get("boundary_loop")
    metrics["boundary_loop"] = []
    try:
        return pipeline._reference_flatten_to_domain(parameterization, grid, params)
    finally:
        if had_loop:
            metrics["boundary_loop"] = saved_loop
        else:
            metrics.pop("boundary_loop", None)


def install_optcuts_test_boundary_reparameterization_patch(pipeline: Any) -> None:
    """Add ``optcuts_test`` without changing ordinary ``optcuts`` behavior."""
    if getattr(pipeline, "_onestring_optcuts_test_boundary_patch_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def test_builder(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts_test":
            return base_builder(surface, target, grid, params)

        # Reuse the known-good ordinary OptCuts path to generate the cut topology.
        ordinary_params = replace(params, omega_parameterization_mode="optcuts")
        parameterization = base_builder(surface, target, grid, ordinary_params)
        parameterization.method = "optcuts_test"
        parameterization.metrics["omega_parameterization_mode"] = "optcuts_test"
        parameterization.metrics["requested_omega_parameterization_mode"] = "optcuts_test"
        parameterization.metrics["optcuts_test_initial_omega_boundary"] = np.asarray(
            parameterization.omega_boundary, dtype=float
        ).tolist()

        # Produce the footprint using the regular Omega overlay.  OptCuts may have
        # more UV ids than surface vertices because seam copies are duplicated, so
        # suppress only the reference function's incompatible Gaussian-boundary
        # diagnostic; all geometry used to construct the overlay stays unchanged.
        domain = _reference_domain_without_invalid_optcuts_boundary_diagnostic(
            pipeline, parameterization, grid, ordinary_params
        )
        footprint = pipeline._build_reference_m2d(grid, domain, ordinary_params)
        outline = _quad_union_boundary(footprint)
        if len(outline) < 4:
            raise RuntimeError("OPTCUTS_TEST_GRID_OUTLINE_EMPTY")

        hard_targets, counts = _build_test_targets(parameterization, outline)
        if counts["outer_boundary_vertex_count"] == 0:
            raise RuntimeError(
                "OPTCUTS_TEST_NO_NON_SEAM_OUTER_BOUNDARY: the current OptCuts cut mesh has no "
                "separate outer-boundary UV vertices to fit to the M2D grid outline"
            )

        iterations = 220
        uv_final, opt_info = _optimize_constrained_uv(
            parameterization,
            np.asarray(parameterization.uv_vertices_2d, dtype=float),
            hard_targets,
            iterations,
        )
        parameterization.uv_vertices_2d = np.asarray(uv_final, dtype=float)
        parameterization.omega_boundary = np.asarray(outline, dtype=float)
        parameterization.metrics.update({
            "optcuts_test_enabled": True,
            "optcuts_test_model": (
                "official OptCuts cut topology -> initial Omega -> reference M2D cell footprint -> "
                "grid-cell outer boundary -> constrained UV reparameterization with OptCuts seam fixed"
            ),
            "optcuts_test_seam_topology_preserved": True,
            "optcuts_test_seam_geometry_fixed_during_resolve": True,
            "optcuts_test_grid_outline_vertex_count": int(len(outline) - 1),
            "optcuts_test_grid_outline": np.asarray(outline, dtype=float).tolist(),
            "optcuts_test_reference_m2d_face_count": int(len(np.asarray(footprint.faces))),
            **counts,
            **{f"optcuts_test_{k}": v for k, v in opt_info.items()},
        })
        setattr(parameterization, "_optcuts_test_grid_outline", np.asarray(outline, dtype=float))
        print(
            "[OPTCUTS-TEST] "
            f"footprint_quads={len(np.asarray(footprint.faces))} outline={len(outline)-1} "
            f"outer_fixed={counts['outer_boundary_vertex_count']} seam_fixed={counts['seam_fixed_vertex_count']}"
        )
        return parameterization

    pipeline._build_surface_parameterization = test_builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = test_builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = test_builder
    pipeline._onestring_optcuts_test_boundary_patch_installed = True


__all__ = [
    "install_optcuts_test_boundary_reparameterization_patch",
    "_quad_union_boundary",
    "_build_test_targets",
    "_reference_domain_without_invalid_optcuts_boundary_diagnostic",
]
