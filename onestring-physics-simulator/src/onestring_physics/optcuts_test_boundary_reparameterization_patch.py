"""Experimental ``optcuts_test`` grid-outline reparameterization.

The key topological fact is that after OptCuts cuts a closed surface, the two UV
copies of each physical seam are themselves part of the planar domain boundary.
They must not be treated as a second boundary independent of the Omega outline.

Flow
----
1. Run the known-good ordinary OptCuts backend. Keep its UV ids, UV faces and cut
   topology unchanged.
2. Build the ordinary OneString quad-cell footprint from the initial Omega.
3. Extract the ordered outer boundary of that retained quad-cell union.
4. Take the complete ordered OptCuts UV boundary loop (including both copies of
   every seam) and map it monotonically, in the same cyclic order, onto that grid
   outline.
5. Move that boundary by continuation. At every stage, first verify that moving
   the boundary alone is orientation preserving, then relax only interior UV
   vertices with a flip-safe Symmetric-Dirichlet solver whose line search rejects
   every step that would invert or degenerate a triangle.

Thus the seam remains a boundary because it is the same cut boundary topology;
its 2D coordinates are allowed to move during the new parameterization.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .optcuts_grid_constrained_parameterization_patch import (
    _source_inverse_matrices,
    _surface_seam_records,
)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _quad_union_boundary(mesh: Any) -> np.ndarray:
    """Return the longest ordered outer loop of the retained quad-cell union."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    incidence: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        ids = [int(v) for v in face]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            incidence[_edge_key(a, b)] += 1
    boundary_edges = {edge for edge, count in incidence.items() if count == 1}
    if not boundary_edges:
        return np.zeros((0, 2), dtype=float)

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    unused = set(boundary_edges)
    loops: list[list[int]] = []
    while unused:
        a, b = next(iter(unused))
        start, cur = int(a), int(b)
        loop = [start, cur]
        unused.discard(_edge_key(start, cur))
        prev = start
        for _ in range(len(boundary_edges) + 4):
            candidates = [
                int(v) for v in adjacency[cur]
                if int(v) != prev and _edge_key(cur, int(v)) in unused
            ]
            if not candidates:
                if start in adjacency[cur] and _edge_key(cur, start) in unused:
                    unused.discard(_edge_key(cur, start))
                    loop.append(start)
                break
            nxt = candidates[0]
            unused.discard(_edge_key(cur, nxt))
            loop.append(nxt)
            if nxt == start:
                break
            prev, cur = cur, nxt
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


def _ordered_edge_components(edges: set[tuple[int, int]]) -> list[list[int]]:
    if not edges:
        return []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    unseen_vertices = set(adjacency)
    components: list[list[int]] = []
    while unseen_vertices:
        seed = unseen_vertices.pop()
        verts = {seed}
        q = deque([seed])
        while q:
            cur = q.popleft()
            for nxt in adjacency[cur]:
                if nxt not in verts:
                    verts.add(nxt)
                    unseen_vertices.discard(nxt)
                    q.append(nxt)

        endpoints = [v for v in verts if len(adjacency[v]) == 1]
        start = min(endpoints) if endpoints else min(verts)
        chain = [int(start)]
        prev = -1
        cur = int(start)
        used: set[tuple[int, int]] = set()
        for _ in range(max(4, len(verts) + 3)):
            choices = [
                int(v) for v in adjacency[cur]
                if int(v) != prev and _edge_key(cur, int(v)) not in used
            ]
            if not choices:
                break
            nxt = choices[0]
            used.add(_edge_key(cur, nxt))
            chain.append(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(chain) >= 2:
            components.append(chain)
    return components


def _optcuts_boundary_loops(parameterization: Any) -> list[list[int]]:
    """Return complete ordered UV boundary loops, including OptCuts seam copies."""
    uv_count = len(np.asarray(parameterization.uv_vertices_2d))
    stored = [int(v) for v in (parameterization.metrics.get("boundary_loop", []) or [])]
    if stored and all(0 <= v < uv_count for v in stored):
        loop = stored[:]
        if loop[0] != loop[-1]:
            loop.append(loop[0])
        return [loop]
    return _ordered_edge_components(_uv_boundary_edges(parameterization))


def _outer_boundary_chains(parameterization: Any) -> list[list[int]]:
    """Compatibility alias: in optcuts_test the entire cut boundary is the outline."""
    return _optcuts_boundary_loops(parameterization)


def _closed_polyline_data(polyline: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    poly = np.asarray(polyline, dtype=float)
    if len(poly) == 0:
        return poly, np.zeros(0), 0.0
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-10:
        poly = np.vstack([poly, poly[0]])
    lengths = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return poly, cumulative, float(cumulative[-1])


def _point_at_closed_arclength(polyline: np.ndarray, s: float) -> np.ndarray:
    poly, cumulative, total = _closed_polyline_data(polyline)
    if len(poly) == 0:
        return np.zeros(2, dtype=float)
    if total <= 1e-20:
        return poly[0].copy()
    value = float(s % total)
    i = int(np.searchsorted(cumulative, value, side="right") - 1)
    i = max(0, min(i, len(poly) - 2))
    span = float(cumulative[i + 1] - cumulative[i])
    t = 0.0 if span <= 1e-20 else (value - float(cumulative[i])) / span
    return (1.0 - t) * poly[i] + t * poly[i + 1]


def _project_arclength(point: np.ndarray, polyline: np.ndarray) -> float:
    p = np.asarray(point, dtype=float)
    poly, cumulative, _ = _closed_polyline_data(polyline)
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


def _signed_area(polyline: np.ndarray) -> float:
    poly, _, _ = _closed_polyline_data(polyline)
    if len(poly) < 4:
        return 0.0
    return 0.5 * float(np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1]))


def _map_closed_loop_to_outline(
    uv: np.ndarray,
    loop: list[int],
    outline: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map a complete UV boundary loop monotonically onto the grid outline."""
    ids = [int(v) for v in loop]
    if ids[0] != ids[-1]:
        ids.append(ids[0])
    source = uv[np.asarray(ids, dtype=int)]
    source_lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    source_cumulative = np.concatenate([[0.0], np.cumsum(source_lengths)])
    source_total = float(source_cumulative[-1])
    if source_total <= 1e-20:
        raise RuntimeError("OPTCUTS_TEST_DEGENERATE_SOURCE_BOUNDARY")

    _, _, target_total = _closed_polyline_data(outline)
    if target_total <= 1e-20:
        raise RuntimeError("OPTCUTS_TEST_DEGENERATE_GRID_OUTLINE")

    start_s = _project_arclength(source[0], outline)
    direction = 1.0 if _signed_area(source) * _signed_area(outline) >= 0.0 else -1.0
    mapped: dict[int, np.ndarray] = {}
    for uid, source_s in zip(ids[:-1], source_cumulative[:-1]):
        fraction = float(source_s / source_total)
        mapped[int(uid)] = _point_at_closed_arclength(
            outline, start_s + direction * fraction * target_total
        )
    return mapped


def _seam_vertex_count(parameterization: Any) -> int:
    vertices: set[int] = set()
    for record in _surface_seam_records(parameterization):
        for copy in record["copies"]:
            vertices.update(int(v) for v in copy.values())
    return len(vertices)


def _build_test_targets(parameterization: Any, grid_outline: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    targets = np.full_like(uv, np.nan)
    loops = _optcuts_boundary_loops(parameterization)
    if not loops:
        raise RuntimeError("OPTCUTS_TEST_NO_UV_BOUNDARY_LOOP")

    primary = max(loops, key=len)
    mapped = _map_closed_loop_to_outline(uv, primary, grid_outline)
    for uid, point in mapped.items():
        targets[int(uid)] = np.asarray(point, dtype=float)

    boundary_vertices = {v for edge in _uv_boundary_edges(parameterization) for v in edge}
    return targets, {
        "seam_vertex_count": int(_seam_vertex_count(parameterization)),
        "boundary_target_vertex_count": int(len(mapped)),
        "outer_boundary_vertex_count": int(len(mapped)),
        "outer_boundary_chain_count": 1,
        "uv_boundary_vertex_count": int(len(boundary_vertices)),
        "uv_boundary_loop_count": int(len(loops)),
    }


def _reference_domain_without_invalid_optcuts_boundary_diagnostic(
    pipeline: Any,
    parameterization: Any,
    grid: Any,
    params: Any,
):
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


def _triangle_signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, dtype=float)[np.asarray(faces, dtype=int)]
    return 0.5 * (
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    )


def _orientation_signature(uv: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    areas = _triangle_signed_areas(uv, faces)
    nonzero = areas[np.abs(areas) > 1e-15]
    sign = 1.0 if len(nonzero) == 0 or float(np.median(nonzero)) >= 0.0 else -1.0
    scale = max(float(np.median(np.abs(nonzero))) if len(nonzero) else 1.0, 1e-12)
    return sign, scale


def _invalid_triangle_ids(
    uv: np.ndarray,
    faces: np.ndarray,
    sign: float,
    area_scale: float,
) -> np.ndarray:
    areas = sign * _triangle_signed_areas(uv, faces)
    eps = max(1e-12, 1e-8 * area_scale)
    return np.flatnonzero(~np.isfinite(areas) | (areas <= eps))


def _flip_safe_relax(
    parameterization: Any,
    start_uv: np.ndarray,
    hard_targets: np.ndarray,
    iterations: int = 80,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Relax free vertices while *never* accepting an orientation-changing step.

    This solver is intentionally conservative.  The hard boundary is applied
    first and validated.  Interior gradients use Symmetric Dirichlet plus a
    logarithmic determinant barrier.  Every gradient step is backtracked until
    all triangles keep the orientation of the original OptCuts embedding.
    """
    if torch is None:
        raise RuntimeError("OPTCUTS_TEST_REQUIRES_TORCH")

    faces_np = np.asarray(parameterization.uv_faces, dtype=int)
    initial_reference = np.asarray(parameterization.metrics.get("optcuts_test_initial_uv", []), dtype=float)
    if initial_reference.shape != np.asarray(start_uv).shape:
        initial_reference = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    sign, area_scale = _orientation_signature(initial_reference, faces_np)

    fixed = np.all(np.isfinite(hard_targets), axis=1)
    current = np.asarray(start_uv, dtype=float).copy()
    current[fixed] = hard_targets[fixed]

    invalid_boundary = _invalid_triangle_ids(current, faces_np, sign, area_scale)
    if len(invalid_boundary):
        raise RuntimeError(
            "OPTCUTS_TEST_BOUNDARY_STEP_INFEASIBLE: moving only the boundary already flips/degenerates "
            f"{len(invalid_boundary)} triangles; examples={invalid_boundary[:16].tolist()}"
        )

    inv_source_np = _source_inverse_matrices(parameterization)
    device = torch.device("cpu")
    dtype = torch.float64
    faces = torch.tensor(faces_np, device=device, dtype=torch.long)
    inv_source = torch.tensor(inv_source_np, device=device, dtype=dtype)
    fixed_t = torch.tensor(fixed, device=device, dtype=torch.bool)
    anchor = torch.tensor(current, device=device, dtype=dtype)
    variable = torch.tensor(current, device=device, dtype=dtype, requires_grad=True)

    accepted_steps = 0
    rejected_steps = 0
    last_energy = float("inf")
    base_lr = 2.0e-3

    def energy_of(x: torch.Tensor) -> torch.Tensor:
        tri = x[faces]
        b0 = tri[:, 1] - tri[:, 0]
        b1 = tri[:, 2] - tri[:, 0]
        B = torch.stack([b0, b1], dim=2)
        J = torch.bmm(B, inv_source)
        a, b = J[:, 0, 0], J[:, 0, 1]
        c, d = J[:, 1, 0], J[:, 1, 1]
        det = a * d - b * c
        det_safe = torch.clamp(det, min=1e-12)
        frob = a * a + b * b + c * c + d * d
        sd = frob + frob / (det_safe * det_safe)
        barrier = -2.0e-3 * torch.log(det_safe)
        free_delta = x[~fixed_t] - anchor[~fixed_t]
        anchor_term = 1.0e-6 * torch.mean(free_delta * free_delta) if torch.any(~fixed_t) else 0.0
        return torch.mean(sd + barrier) + anchor_term

    for _ in range(max(1, int(iterations))):
        if variable.grad is not None:
            variable.grad.zero_()
        loss = energy_of(variable)
        if not torch.isfinite(loss):
            break
        loss.backward()
        grad = variable.grad.detach().clone()
        grad[fixed_t] = 0.0
        grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu())
        if not np.isfinite(grad_norm) or grad_norm < 1e-9:
            last_energy = float(loss.detach().cpu())
            break

        direction = grad / max(grad_norm, 1e-20)
        step = base_lr
        old_value = float(loss.detach().cpu())
        accepted = False
        old_np = variable.detach().cpu().numpy()
        for _line in range(24):
            candidate_t = variable.detach() - step * direction
            candidate_t[fixed_t] = torch.tensor(hard_targets[fixed], dtype=dtype, device=device)
            candidate_np = candidate_t.cpu().numpy()
            invalid = _invalid_triangle_ids(candidate_np, faces_np, sign, area_scale)
            if len(invalid):
                step *= 0.5
                rejected_steps += 1
                continue
            with torch.no_grad():
                candidate_energy = float(energy_of(candidate_t).detach().cpu())
            if np.isfinite(candidate_energy) and candidate_energy <= old_value + 1e-10:
                variable = candidate_t.detach().clone().requires_grad_(True)
                accepted_steps += 1
                last_energy = candidate_energy
                accepted = True
                break
            step *= 0.5
            rejected_steps += 1
        if not accepted:
            # The current iterate is already legal. Failure to find a descent
            # step is convergence/stagnation, not a reason to return a flipped UV.
            variable = torch.tensor(old_np, dtype=dtype, device=device, requires_grad=True)
            last_energy = old_value
            break

    result = variable.detach().cpu().numpy()
    result[fixed] = hard_targets[fixed]
    invalid_final = _invalid_triangle_ids(result, faces_np, sign, area_scale)
    if len(invalid_final):
        raise RuntimeError(
            "OPTCUTS_TEST_INTERNAL_SOLVER_BUG: flip-safe solver returned invalid triangles "
            f"{invalid_final[:16].tolist()}"
        )
    return result, {
        "optimizer": "flip_safe_symmetric_dirichlet_backtracking",
        "iterations_requested": int(iterations),
        "accepted_gradient_steps": int(accepted_steps),
        "rejected_line_search_steps": int(rejected_steps),
        "final_energy": float(last_energy),
        "boundary_only_invalid_triangle_count": 0,
        "flip_count": 0,
    }


def _staged_boundary_resolve(parameterization: Any, final_targets: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Progressively map the complete cut boundary onto the grid outline."""
    initial = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    parameterization.metrics["optcuts_test_initial_uv"] = initial.tolist()
    fixed = np.all(np.isfinite(final_targets), axis=1)
    if not np.any(fixed):
        raise RuntimeError("OPTCUTS_TEST_NO_BOUNDARY_TARGETS")

    current = initial.copy()
    alpha = 0.0
    step = 0.05
    minimum_step = 0.00025
    accepted = 0
    retries = 0
    boundary_rejections = 0
    last_info: dict[str, Any] = {}

    while alpha < 1.0 - 1e-12:
        proposed = min(1.0, alpha + step)
        stage_targets = np.full_like(final_targets, np.nan)
        stage_targets[fixed] = (
            (1.0 - proposed) * initial[fixed] + proposed * final_targets[fixed]
        )
        try:
            candidate, info = _flip_safe_relax(
                parameterization,
                current,
                stage_targets,
                80,
            )
        except RuntimeError as exc:
            text = str(exc)
            if "OPTCUTS_TEST_BOUNDARY_STEP_INFEASIBLE" not in text:
                raise
            boundary_rejections += 1
            if step <= minimum_step + 1e-15:
                raise RuntimeError(
                    "OPTCUTS_TEST_GRID_OUTLINE_GEOMETRICALLY_INFEASIBLE_AT_CURRENT_CORRESPONDENCE: "
                    "even moving only the ordered OptCuts boundary by a continuation step of "
                    f"{step:.6g} causes triangle inversion before interior optimization. Original: {exc}"
                ) from exc
            step *= 0.5
            retries += 1
            continue

        current = np.asarray(candidate, dtype=float)
        alpha = proposed
        accepted += 1
        last_info = dict(info)
        if step < 0.05:
            step = min(0.05, step * 1.35)

    return current, {
        **last_info,
        "continuation_accepted_stage_count": int(accepted),
        "continuation_retry_count": int(retries),
        "continuation_boundary_rejection_count": int(boundary_rejections),
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

        boundary_targets, counts = _build_test_targets(parameterization, outline)
        uv_final, opt_info = _staged_boundary_resolve(parameterization, boundary_targets)
        parameterization.uv_vertices_2d = np.asarray(uv_final, dtype=float)
        parameterization.omega_boundary = np.asarray(outline, dtype=float)
        parameterization.metrics.update({
            "optcuts_test_enabled": True,
            "optcuts_test_model": (
                "official OptCuts cut topology -> initial Omega -> M2D quad footprint -> "
                "grid-cell outline -> full OptCuts cut-boundary loop mapped monotonically to outline -> "
                "flip-safe staged Symmetric Dirichlet interior resolve"
            ),
            "optcuts_test_seam_topology_preserved": True,
            "optcuts_test_seam_geometry_fixed_during_resolve": False,
            "optcuts_test_boundary_semantics": (
                "OptCuts seam copies are part of the Omega boundary and are re-mapped with that boundary; "
                "they are not treated as an independent inner boundary"
            ),
            "optcuts_test_outer_boundary_mapping": "full ordered cut boundary -> grid-outline arclength",
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
            f"boundary_targets={counts['boundary_target_vertex_count']} "
            f"seam_vertices={counts['seam_vertex_count']} "
            f"stages={opt_info.get('continuation_accepted_stage_count', 0)} "
            f"retries={opt_info.get('continuation_retry_count', 0)} "
            f"boundary_rejections={opt_info.get('continuation_boundary_rejection_count', 0)}"
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
    "_flip_safe_relax",
    "_reference_domain_without_invalid_optcuts_boundary_diagnostic",
]
