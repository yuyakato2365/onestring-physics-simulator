"""Flip-safe harmonic continuation for ``optcuts_test``.

The previous experimental continuation tested a boundary move while every
interior UV vertex was frozen.  That is too strict: a perfectly feasible domain
motion can invert a very small boundary-adjacent triangle unless its interior
vertex moves at the same time.

This patch replaces only the ``optcuts_test`` continuation routine.  At each
stage it extends the prescribed boundary displacement harmonically through the
cut UV mesh, then backtracks the *whole* displacement until every triangle keeps
its original orientation.  The already-existing flip-safe Symmetric Dirichlet
relaxation is then allowed to improve the legal candidate without ever accepting
an inverted step.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import factorized
except Exception:  # pragma: no cover
    coo_matrix = None
    factorized = None

from . import optcuts_test_boundary_reparameterization_patch as test_module


def _uniform_laplacian_solver(
    vertex_count: int,
    faces: np.ndarray,
    fixed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray] | None, Any]:
    """Build the Dirichlet combinatorial-Laplacian system once for continuation."""
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in face]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))

    fixed_ids = np.flatnonzero(fixed)
    free_ids = np.flatnonzero(~fixed)
    if len(free_ids) == 0:
        return free_ids, fixed_ids, None, None

    if coo_matrix is None or factorized is None:
        return free_ids, fixed_ids, None, edges

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros(vertex_count, dtype=float)
    for a, b in edges:
        rows.extend([a, b])
        cols.extend([b, a])
        data.extend([-1.0, -1.0])
        degree[a] += 1.0
        degree[b] += 1.0
    rows.extend(range(vertex_count))
    cols.extend(range(vertex_count))
    data.extend(degree.tolist())
    L = coo_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count)).tocsc()
    Lii = L[free_ids][:, free_ids].tocsc()
    Lib = L[free_ids][:, fixed_ids].tocsc()
    try:
        solve = factorized(Lii)
    except Exception:
        return free_ids, fixed_ids, None, edges
    return free_ids, fixed_ids, solve, Lib


def _iterative_harmonic_extension(
    vertex_count: int,
    edges: set[tuple[int, int]],
    fixed: np.ndarray,
    boundary_delta: np.ndarray,
) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    delta = np.zeros((vertex_count, 2), dtype=float)
    delta[fixed] = boundary_delta[fixed]
    free_ids = np.flatnonzero(~fixed)
    for _ in range(600):
        previous = delta.copy()
        for vid in free_ids:
            nbrs = adjacency[int(vid)]
            if nbrs:
                delta[int(vid)] = np.mean(previous[np.asarray(nbrs, dtype=int)], axis=0)
        delta[fixed] = boundary_delta[fixed]
        if float(np.max(np.linalg.norm(delta - previous, axis=1))) < 1e-10:
            break
    return delta


def _harmonic_extension(
    current: np.ndarray,
    target_boundary: np.ndarray,
    fixed: np.ndarray,
    system: tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray] | None, Any],
    faces: np.ndarray,
) -> np.ndarray:
    """Return a full-mesh displacement whose boundary equals the requested move."""
    free_ids, fixed_ids, solve, auxiliary = system
    boundary_delta = np.zeros_like(current)
    boundary_delta[fixed] = target_boundary[fixed] - current[fixed]
    full = np.zeros_like(current)
    full[fixed] = boundary_delta[fixed]
    if len(free_ids) == 0:
        return full

    if solve is not None:
        Lib = auxiliary
        rhs = -(Lib @ boundary_delta[fixed_ids])
        try:
            full[free_ids, 0] = np.asarray(solve(np.asarray(rhs[:, 0]).reshape(-1)), dtype=float)
            full[free_ids, 1] = np.asarray(solve(np.asarray(rhs[:, 1]).reshape(-1)), dtype=float)
        except Exception:
            solve = None
    if solve is None:
        edges = auxiliary
        if not isinstance(edges, set):
            edges = set()
            for face in np.asarray(faces, dtype=int):
                ids = [int(v) for v in face]
                for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                    edges.add((a, b) if a < b else (b, a))
        full = _iterative_harmonic_extension(len(current), edges, fixed, boundary_delta)
    if not np.all(np.isfinite(full)):
        raise RuntimeError("OPTCUTS_TEST_HARMONIC_EXTENSION_NONFINITE")
    return full


def _staged_boundary_resolve_harmonic(
    parameterization: Any,
    final_targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    initial = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    parameterization.metrics["optcuts_test_initial_uv"] = initial.tolist()
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    fixed = np.all(np.isfinite(final_targets), axis=1)
    if not np.any(fixed):
        raise RuntimeError("OPTCUTS_TEST_NO_BOUNDARY_TARGETS")

    sign, area_scale = test_module._orientation_signature(initial, faces)
    initial_invalid = test_module._invalid_triangle_ids(initial, faces, sign, area_scale)
    if len(initial_invalid):
        raise RuntimeError(
            "OPTCUTS_TEST_INITIAL_OPTCUTS_UV_INVALID: official OptCuts input already contains "
            f"{len(initial_invalid)} flipped/degenerate triangles; examples={initial_invalid[:16].tolist()}"
        )

    system = _uniform_laplacian_solver(len(initial), faces, fixed)
    current = initial.copy()
    alpha = 0.0
    nominal_step = 0.05
    minimum_step = 1.0e-5
    accepted_stages = 0
    continuation_retries = 0
    harmonic_backtracks = 0
    relaxation_steps = 0
    last_info: dict[str, Any] = {}

    while alpha < 1.0 - 1e-12:
        requested = min(1.0, alpha + nominal_step)
        requested_boundary = np.full_like(final_targets, np.nan)
        requested_boundary[fixed] = (
            (1.0 - requested) * initial[fixed] + requested * final_targets[fixed]
        )
        displacement = _harmonic_extension(current, requested_boundary, fixed, system, faces)

        beta = 1.0
        accepted_candidate: np.ndarray | None = None
        accepted_alpha = alpha
        for _ in range(32):
            candidate = current + beta * displacement
            # Harmonic extension is linear; enforce the same beta on boundary
            # explicitly to remove sparse-solver roundoff.
            candidate[fixed] = current[fixed] + beta * (requested_boundary[fixed] - current[fixed])
            invalid = test_module._invalid_triangle_ids(candidate, faces, sign, area_scale)
            if len(invalid) == 0:
                accepted_candidate = candidate
                accepted_alpha = alpha + beta * (requested - alpha)
                break
            beta *= 0.5
            harmonic_backtracks += 1

        if accepted_candidate is None or accepted_alpha <= alpha + minimum_step * 0.25:
            if nominal_step <= minimum_step + 1e-15:
                raise RuntimeError(
                    "OPTCUTS_TEST_HARMONIC_CONTINUATION_STALLED: even simultaneous harmonic motion "
                    f"of boundary and interior cannot advance beyond alpha={alpha:.9f}. "
                    "This now indicates the current boundary correspondence/path is the likely issue, "
                    "not the previous boundary-only test."
                )
            nominal_step *= 0.5
            continuation_retries += 1
            continue

        # The candidate is already legal.  Improve distortion while keeping the
        # newly reached boundary fixed; the existing relaxation has strict
        # orientation-preserving line search.
        stage_targets = np.full_like(final_targets, np.nan)
        stage_targets[fixed] = accepted_candidate[fixed]
        relaxed, info = test_module._flip_safe_relax(
            parameterization,
            accepted_candidate,
            stage_targets,
            50,
        )
        current = np.asarray(relaxed, dtype=float)
        alpha = float(accepted_alpha)
        accepted_stages += 1
        relaxation_steps += int(info.get("accepted_gradient_steps", 0))
        last_info = dict(info)

        if beta < 0.999:
            nominal_step = max(minimum_step, nominal_step * 0.8)
        elif nominal_step < 0.05:
            nominal_step = min(0.05, nominal_step * 1.25)

    final_invalid = test_module._invalid_triangle_ids(current, faces, sign, area_scale)
    if len(final_invalid):
        raise RuntimeError(
            "OPTCUTS_TEST_HARMONIC_INTERNAL_BUG: final continuation contains invalid triangles; "
            f"examples={final_invalid[:16].tolist()}"
        )
    current[fixed] = final_targets[fixed]
    return current, {
        **last_info,
        "optimizer": "harmonic_displacement_continuation_plus_flip_safe_symmetric_dirichlet",
        "continuation_accepted_stage_count": int(accepted_stages),
        "continuation_retry_count": int(continuation_retries),
        "harmonic_backtracking_count": int(harmonic_backtracks),
        "relaxation_accepted_gradient_steps_total": int(relaxation_steps),
        "continuation_final_alpha": float(alpha),
        "flip_count": 0,
    }


def install_optcuts_test_harmonic_extension_patch() -> None:
    test_module._staged_boundary_resolve = _staged_boundary_resolve_harmonic


__all__ = [
    "install_optcuts_test_harmonic_extension_patch",
    "_staged_boundary_resolve_harmonic",
]
