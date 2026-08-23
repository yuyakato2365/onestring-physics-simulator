"""Experimental OptCuts_test boundary reparameterization.

Flow
----
1. Run ordinary official OptCuts and keep its cut topology.
2. Build the same reference M2D footprint used by OneString from the initial Omega.
3. Extract the outer boundary of the retained quad-cell union.
4. Reparameterize the *same cut mesh* with that grid-cell outline as the outer
   boundary target while preserving the OptCuts seam as a cut boundary.

The important invariant is topological, not positional: UV vertex ids / UV faces
and the two boundary copies created by every OptCuts seam are unchanged.  Seam
vertices may move during the re-solve; they are not incorrectly frozen in their
old 2D locations.  Only the true non-seam outer boundary is hard-constrained to
the grid-cell outline.
"""
from __future__ import annotations

from collections import defaultdict, deque
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


def _uv_boundary_edges(parameterization: Any) -> set[tuple[int, int]]:
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    incidence: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        ids = [int(v) for v in face]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            incidence[_edge_key(a, b)] += 1
    return {edge for edge, count in incidence.items() if count == 1}


def _seam_uv_edges(parameterization: Any) -> set[tuple[int, int]]:
    """Return the UV boundary edges that are the two copies of OptCuts cuts."""
    out: set[tuple[int, int]] = set()
    for record in _surface_seam_records(parameterization):
        sa, sb = [int(x) for x in record["surface_edge"]]
        for copy in record["copies"]:
            if sa in copy and sb in copy:
                out.add(_edge_key(int(copy[sa]), int(copy[sb])))
    return out


def _ordered_edge_components(edges: set[tuple[int, int]]) -> list[list[int]]:
    """Trace non-branching boundary edge components as ordered chains/loops."""
    if not edges:
        return []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    unused = set(edges)
    chains: list[list[int]] = []
    while unused:
        component_vertices: set[int] = set()
        seed = next(iter(unused))
        q = deque(seed)
        while q:
            v = int(q.popleft())
            if v in component_vertices:
                continue
            component_vertices.add(v)
            for n in adjacency[v]:
                if n not in component_vertices:
                    q.append(n)
        endpoints = sorted(v for v in component_vertices if len(adjacency[v]) == 1)
        start = endpoints[0] if endpoints else min(component_vertices)
        chain = [int(start)]
        prev = -1
        cur = int(start)
        for _ in range(len(component_vertices) + 3):
            candidates = [n for n in adjacency[cur] if n != prev and _edge_key(cur, n) in unused]
            if not candidates:
                break
            nxt = int(candidates[0])
            unused.discard(_edge_key(cur, nxt))
            chain.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def _outer_boundary_chains(parameterization: Any) -> list[list[int]]:
    """Return only original/external boundary chains, excluding OptCuts seam copies."""
    return _ordered_edge_components(_uv_boundary_edges(parameterization) - _seam_uv_edges(parameterization))


def _closed_polyline_arclength(polyline: np.ndarray) -> tuple[np.ndarray, float]:
    poly = np.asarray(polyline, dtype=float)
    if len(poly) < 2:
        return np.zeros(len(poly), dtype=float), 0.0
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-10:
        poly = np.vstack([poly, poly[0]])
    lengths = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return cumulative, float(cumulative[-1])


def _project_arclength_on_closed_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(polyline, dtype=float)
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-10:
        poly = np.vstack([poly, poly[0]])
    cumulative, _ = _closed_polyline_arclength(poly)
    best_s = 0.0
    best_d2 = float("inf")
    for i, (a, b) in enumerate(zip(poly[:-1], poly[1:])):
        d = b - a
        denom = float(np.dot(d, d))
        t = 0.0 if denom <= 1e-20 else float(np.clip(np.dot(p - a, d) / denom, 0.0, 1.0))
        q = a + t * d
        d2 = float(np.dot(p - q, p - q))
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(cumulative[i] + t * np.linalg.norm(d))
    return best_s


def _point_at_closed_arclength(polyline: np.ndarray, s: float) -> np.ndarray:
    poly = np.asarray(polyline, dtype=float)
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-10:
        poly = np.vstack([poly, poly[0]])
    cumulative, total = _closed_polyline_arclength(poly)
    if total <= 1e-20:
        return poly[0].copy()
    value = float(s % total)
    i = int(np.searchsorted(cumulative, value, side="right") - 1)
    i = max(0, min(i, len(poly) - 2))
    span = float(cumulative[i + 1] - cumulative[i])
    t = 0.0 if span <= 1e-20 else (value - float(cumulative[i])) / span
    return (1.0 - t) * poly[i] + t * poly[i + 1]


def _chain_fraction(points: np.ndarray, closed: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.zeros(len(pts), dtype=float)
    use = pts[:-1] if closed and np.linalg.norm(pts[0] - pts[-1]) <= 1e-10 else pts
    if len(use) <= 1:
        return np.zeros(len(pts), dtype=float)
    seg = np.linalg.norm(np.diff(use, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    frac = np.linspace(0.0, 1.0, len(use)) if total <= 1e-20 else cumulative / total
    if len(use) != len(pts):
        frac = np.concatenate([frac, [1.0]])
    return frac


def _map_outer_chain_to_outline(uv: np.ndarray, chain: list[int], outline: np.ndarray) -> dict[int, np.ndarray]:
    """Map one outer-boundary chain monotonically to the grid outline.

    This deliberately does not independently nearest-project vertices: doing so
    collapses neighboring vertices onto the same corner and creates flips.
    """
    ids = [int(x) for x in chain]
    pts = uv[np.asarray(ids, dtype=int)]
    closed = ids[0] == ids[-1]
    fractions = _chain_fraction(pts, closed)
    _, perimeter = _closed_polyline_arclength(outline)
    if perimeter <= 1e-20:
        return {}

    start_s = _project_arclength_on_closed_polyline(pts[0], outline)
    if closed:
        # Preserve orientation when wrapping the complete target outline.
        source_area = 0.5 * float(np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]))
        target = np.asarray(outline, dtype=float)
        target_area = 0.5 * float(np.sum(target[:-1, 0] * target[1:, 1] - target[1:, 0] * target[:-1, 1]))
        direction = 1.0 if source_area * target_area >= 0.0 else -1.0
        values = [start_s + direction * float(t) * perimeter for t in fractions]
    else:
        end_s = _project_arclength_on_closed_polyline(pts[-1], outline)
        forward = float((end_s - start_s) % perimeter)
        backward = forward - perimeter
        # Choose the orientation whose arclength interpolation stays closer to
        # the current Omega chain. This preserves locality as well as ordering.
        candidates = []
        for delta in (forward, backward):
            mapped = np.asarray(
                [_point_at_closed_arclength(outline, start_s + float(t) * delta) for t in fractions],
                dtype=float,
            )
            candidates.append((float(np.sum((mapped - pts) ** 2)), delta, mapped))
        _, _, mapped = min(candidates, key=lambda item: item[0])
        return {uid: mapped[i] for i, uid in enumerate(ids)}

    mapped = np.asarray([_point_at_closed_arclength(outline, s) for s in values], dtype=float)
    return {uid: mapped[i] for i, uid in enumerate(ids)}


def _build_test_targets(parameterization: Any, grid_outline: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    targets = np.full_like(uv, np.nan)
    seam_edges = _seam_uv_edges(parameterization)
    seam_ids = {v for edge in seam_edges for v in edge}
    chains = _outer_boundary_chains(parameterization)

    assigned: dict[int, list[np.ndarray]] = defaultdict(list)
    for chain in chains:
        for uid, point in _map_outer_chain_to_outline(uv, chain, grid_outline).items():
            assigned[int(uid)].append(np.asarray(point, dtype=float))
    for uid, points in assigned.items():
        if 0 <= uid < len(uv):
            targets[uid] = np.mean(np.asarray(points, dtype=float), axis=0)

    constrained = set(assigned)
    return targets, {
        "seam_vertex_count": int(len(seam_ids)),
        "seam_fixed_vertex_count": int(len(seam_ids & constrained)),
        "seam_free_boundary_vertex_count": int(len(seam_ids - constrained)),
        "outer_boundary_vertex_count": int(len(constrained)),
        "outer_boundary_chain_count": int(len(chains)),
        "uv_boundary_vertex_count": int(len({v for edge in _uv_boundary_edges(parameterization) for v in edge})),
    }


def _reference_domain_without_invalid_optcuts_boundary_diagnostic(
    pipeline: Any,
    parameterization: Any,
    grid: Any,
    params: Any,
):
    """Build the regular overlay while suppressing one incompatible diagnostic."""
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


def _staged_boundary_resolve(parameterization: Any, final_targets: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Move the outer boundary to the grid outline progressively.

    A single hard jump from the OptCuts boundary to a non-convex staircase can
    invert triangles before the interior has time to adapt.  Continuation keeps
    every accepted stage bijective and adaptively halves the boundary step when a
    requested stage is too aggressive.
    """
    initial = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    fixed = np.all(np.isfinite(final_targets), axis=1)
    if not np.any(fixed):
        raise RuntimeError("OPTCUTS_TEST_NO_OUTER_BOUNDARY_TARGETS")
    source_targets = initial.copy()
    current = initial.copy()
    alpha = 0.0
    step = 0.20
    minimum_step = 0.025
    accepted = 0
    retries = 0
    last_info: dict[str, Any] = {}
    while alpha < 1.0 - 1e-12:
        proposed = min(1.0, alpha + step)
        stage_targets = np.full_like(final_targets, np.nan)
        stage_targets[fixed] = (
            (1.0 - proposed) * source_targets[fixed] + proposed * final_targets[fixed]
        )
        try:
            candidate, info = _optimize_constrained_uv(
                parameterization,
                current,
                stage_targets,
                80,
            )
        except RuntimeError as exc:
            if "OPTCUTS_GRID_CONSTRAINT_INFEASIBLE" not in str(exc) or step <= minimum_step + 1e-12:
                raise
            step *= 0.5
            retries += 1
            continue
        current = np.asarray(candidate, dtype=float)
        alpha = proposed
        accepted += 1
        last_info = dict(info)
        if step < 0.20:
            step = min(0.20, step * 1.5)
    return current, {
        **last_info,
        "continuation_accepted_stage_count": int(accepted),
        "continuation_retry_count": int(retries),
        "continuation_final_alpha": float(alpha),
    }


def install_optcuts_test_boundary_reparameterization_patch(pipeline: Any) -> None:
    """Add ``optcuts_test`` without changing ordinary ``optcuts`` behavior."""
    if getattr(pipeline, "_onestring_optcuts_test_boundary_patch_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def test_builder(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts_test":
            return base_builder(surface, target, grid, params)

        ordinary_params = replace(params, omega_parameterization_mode="optcuts")
        parameterization = base_builder(surface, target, grid, ordinary_params)
        parameterization.method = "optcuts_test"
        parameterization.metrics["omega_parameterization_mode"] = "optcuts_test"
        parameterization.metrics["requested_omega_parameterization_mode"] = "optcuts_test"
        parameterization.metrics["optcuts_test_initial_omega_boundary"] = np.asarray(
            parameterization.omega_boundary, dtype=float
        ).tolist()

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
                "OPTCUTS_TEST_NO_NON_SEAM_OUTER_BOUNDARY: no external UV boundary chain remained "
                "after excluding the two OptCuts seam copies"
            )

        uv_final, opt_info = _staged_boundary_resolve(parameterization, hard_targets)
        parameterization.uv_vertices_2d = np.asarray(uv_final, dtype=float)
        parameterization.omega_boundary = np.asarray(outline, dtype=float)
        parameterization.metrics.update({
            "optcuts_test_enabled": True,
            "optcuts_test_model": (
                "official OptCuts cut topology -> initial Omega -> reference M2D cell footprint -> "
                "grid-cell outer boundary -> order-preserving staged UV reparameterization; "
                "OptCuts seam remains a free cut boundary"
            ),
            "optcuts_test_seam_topology_preserved": True,
            "optcuts_test_seam_geometry_fixed_during_resolve": False,
            "optcuts_test_outer_boundary_mapping": "ordered boundary chains -> monotone grid-outline arclength",
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
            f"outer_fixed={counts['outer_boundary_vertex_count']} "
            f"outer_chains={counts['outer_boundary_chain_count']} "
            f"seam_vertices={counts['seam_vertex_count']} seam_outer_endpoints={counts['seam_fixed_vertex_count']} "
            f"stages={opt_info.get('continuation_accepted_stage_count', 0)} "
            f"retries={opt_info.get('continuation_retry_count', 0)}"
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
    "_outer_boundary_chains",
    "_build_test_targets",
    "_staged_boundary_resolve",
    "_reference_domain_without_invalid_optcuts_boundary_diagnostic",
]
