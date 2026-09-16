"""Robust fallback initialization for the bijective free-boundary solver.

The existing Floater mean-value initialization intentionally parameterizes the
convex boundary by 3D boundary arc length.  On sampled meshes with extremely
short boundary edges this can place consecutive UV boundary vertices almost on
top of one another, producing a numerically degenerate initial triangle even
though the mesh is topologically a valid disk.

This patch preserves the original initializer as the first choice.  Only when
that initializer fails its strict signed-area check do we fall back to a
uniformly spaced convex boundary and normalized positive Tutte weights.  This
changes only the valid starting point; the subsequent free-boundary objective,
validity-aware line search, and optimization are unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1.0e-12


def _uniform_convex_boundary(
    xyz: np.ndarray,
    tris: np.ndarray,
    loop: np.ndarray,
    boundary_shape: str,
) -> np.ndarray:
    """Build a strictly convex, uniformly spaced, equal-area boundary."""
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
        raise RuntimeError("robust Floater fallback found zero 3D surface area")

    count = len(loop)
    if count < 3:
        raise RuntimeError("robust Floater fallback requires at least three boundary vertices")

    fractions = np.arange(count, dtype=float) / float(count)
    if boundary_shape == "circle":
        angles = 2.0 * np.pi * fractions
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
    elif boundary_shape == "rectangle":
        # Same strictly-convex p=8 superellipse semantics as the primary path,
        # but sample it uniformly by 2D boundary arc length rather than by the
        # possibly pathological 3D input boundary edge lengths.
        exponent = 8.0
        dense_angle = np.linspace(0.0, 2.0 * np.pi, 8193)
        cosine = np.cos(dense_angle)
        sine = np.sin(dense_angle)
        dense = np.column_stack(
            [
                np.sign(cosine) * np.abs(cosine) ** (2.0 / exponent),
                np.sign(sine) * np.abs(sine) ** (2.0 / exponent),
            ]
        )
        arclength = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        )
        targets = fractions * arclength[-1]
        boundary_uv = np.column_stack(
            [
                np.interp(targets, arclength, dense[:, 0]),
                np.interp(targets, arclength, dense[:, 1]),
            ]
        )
    else:
        raise ValueError(
            "initial_boundary_shape must be 'circle' or 'rectangle'; "
            f"got {boundary_shape!r}"
        )

    polygon_area = 0.5 * abs(
        float(
            np.sum(
                boundary_uv[:, 0] * np.roll(boundary_uv[:, 1], -1)
                - boundary_uv[:, 1] * np.roll(boundary_uv[:, 0], -1)
            )
        )
    )
    if not np.isfinite(polygon_area) or polygon_area <= _EPS:
        raise RuntimeError("robust Floater fallback produced zero boundary area")
    return boundary_uv * math.sqrt(surface_area / polygon_area)


def _normalized_uniform_tutte(
    base_module: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_loop: list[int],
    boundary_shape: str,
) -> np.ndarray:
    """Solve a normalized positive-weight Tutte embedding on a uniform boundary."""
    try:
        from scipy import sparse
        from scipy.sparse import linalg as spla
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy sparse is required for robust Floater fallback") from exc

    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    n = len(xyz)
    loop = np.asarray(boundary_loop, dtype=int)
    boundary_set = set(int(v) for v in loop)

    uv = np.zeros((n, 2), dtype=float)
    uv[loop] = _uniform_convex_boundary(xyz, tris, loop, boundary_shape)

    neighbors: list[set[int]] = [set() for _ in range(n)]
    for face in tris:
        a, b, c = [int(v) for v in face]
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))

    interior = [v for v in range(n) if v not in boundary_set]
    if interior:
        local = {vertex: row for row, vertex in enumerate(interior)}
        matrix = sparse.lil_matrix((len(interior), len(interior)), dtype=float)
        rhs = np.zeros((len(interior), 2), dtype=float)

        for vertex in interior:
            row = local[vertex]
            nbrs = sorted(neighbors[vertex])
            if len(nbrs) < 2:
                raise RuntimeError("robust Floater fallback found an invalid interior vertex")

            # Normalized positive weights make every row O(1), avoiding the
            # enormous dynamic range that can arise from skinny-triangle
            # mean-value weights.  Tutte's convex-combination guarantee only
            # requires positivity; uniform weights are therefore a robust seed.
            weight = 1.0 / float(len(nbrs))
            matrix[row, row] = 1.0
            for neighbor in nbrs:
                if neighbor in local:
                    matrix[row, local[neighbor]] -= weight
                else:
                    rhs[row] += weight * uv[neighbor]

        solved = spla.spsolve(matrix.tocsr(), rhs)
        uv[np.asarray(interior, dtype=int)] = np.asarray(solved, dtype=float)

    if not np.all(np.isfinite(uv)):
        raise RuntimeError("robust Floater fallback produced non-finite UV coordinates")

    signed = np.asarray(base_module._signed_double_areas(uv, tris), dtype=float)
    if np.all(signed < -_EPS):
        uv[:, 1] *= -1.0
        signed *= -1.0

    # Mixed signs after a positive-weight convex-boundary solve usually point
    # to inconsistent input face winding rather than to the numerical solve.
    positive = int(np.count_nonzero(signed > _EPS))
    negative = int(np.count_nonzero(signed < -_EPS))
    tiny = int(len(signed) - positive - negative)
    if negative or tiny:
        minimum = float(np.min(signed)) if len(signed) else 0.0
        raise RuntimeError(
            "Robust Floater fallback still could not create a strictly valid disk embedding: "
            f"positive={positive}, negative={negative}, near_zero={tiny}, min_signed_area={minimum:.3e}. "
            "If negative faces remain, check/repair inconsistent triangle winding; if only near_zero faces remain, "
            "the disk connectivity or boundary discretization is numerically degenerate."
        )

    uv -= np.mean(uv, axis=0, keepdims=True)
    return uv


def install_robust_floater_fallback(base_module: Any) -> None:
    """Wrap ``base._tutte_embedding`` with a robust positive-weight fallback."""
    if getattr(base_module, "_ROBUST_FLOATER_FALLBACK_INSTALLED", False):
        return

    original = base_module._tutte_embedding

    def robust_tutte_embedding(vertices, faces, boundary_loop, boundary_shape="circle"):
        try:
            uv = original(vertices, faces, boundary_loop, boundary_shape)
            base_module._LAST_FLOATER_INITIALIZATION = {
                "mode": "mean_value_arc_length",
                "fallback_used": False,
                "primary_error": "",
            }
            return uv
        except RuntimeError as primary_error:
            message = str(primary_error)
            # Do not conceal genuine malformed-mesh errors such as zero-length
            # surface edges.  The fallback is specifically for a numerically
            # invalid 2D embedding produced after an otherwise valid solve.
            if "did not produce a strictly valid disk embedding" not in message:
                raise
            try:
                uv = _normalized_uniform_tutte(
                    base_module,
                    np.asarray(vertices, dtype=float),
                    np.asarray(faces, dtype=int)[:, :3],
                    list(boundary_loop),
                    str(boundary_shape),
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Primary Floater initialization failed: {primary_error}. "
                    f"Uniform-boundary Tutte fallback also failed: {fallback_error}"
                ) from fallback_error

            base_module._LAST_FLOATER_INITIALIZATION = {
                "mode": "uniform_boundary_uniform_tutte",
                "fallback_used": True,
                "primary_error": message,
            }
            return uv

    base_module._tutte_embedding = robust_tutte_embedding
    base_module._ROBUST_FLOATER_FALLBACK_INSTALLED = True


__all__ = ["install_robust_floater_fallback"]
