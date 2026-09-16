"""Surface-constrained K3D planarization for the 2026-09-13 experiment.

The existing optcuts_test2 stack first computes the ordinary K3D result and its
hard-planarity polish.  This patch then alternates between

1. consensus projection of every quad to its best-fit plane, and
2. closest-point projection of every K3D vertex back to the original target
   surface mesh.

The experiment intentionally ends on the surface-projection step.  Therefore the
accepted vertices lie on the input surface (up to closest-point numerical error),
while the remaining planarity residual measures whether the two constraint sets
are actually compatible.  This is diagnostic as well as constructive: if the
residual cannot be driven small, a later optimizer cannot create an exact
surface-constrained planar solution without changing the formulation.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .optcuts_test_k3d_augmented_lagrangian_patch import (
    _consensus_planarity_restore,
    _quad_plane_distances,
    _robust_constraint_faces,
)


def _is_surface_constrained_variant(params: Any) -> bool:
    return (
        str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"
        and os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0") == "2"
    )


def _surface_vertices_faces(parameterization: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(getattr(parameterization, "surface_vertices_3d", []), dtype=float)
    faces = np.asarray(getattr(parameterization, "surface_faces", []), dtype=int)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise RuntimeError("SURFACE_CONSTRAINED_K3D_MISSING_SURFACE_VERTICES")
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise RuntimeError("SURFACE_CONSTRAINED_K3D_MISSING_SURFACE_FACES")
    return vertices, faces[:, :3]


def _project_to_surface(
    points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    pipeline: Any,
) -> np.ndarray:
    projector = getattr(pipeline, "_closest_points_on_surface_mesh", None)
    if projector is None:
        original = getattr(pipeline, "_original", None)
        projector = getattr(original, "_closest_points_on_surface_mesh", None) if original is not None else None
    if projector is None:
        raise RuntimeError("SURFACE_CONSTRAINED_K3D_CLOSEST_POINT_PROJECTOR_UNAVAILABLE")
    projected = np.asarray(projector(points, surface_vertices, surface_faces), dtype=float)
    if projected.shape != np.asarray(points).shape or not np.all(np.isfinite(projected)):
        raise RuntimeError("SURFACE_CONSTRAINED_K3D_INVALID_SURFACE_PROJECTION")
    return projected


def _alternate_planarity_and_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    pipeline: Any,
    *,
    cycles: int = 24,
    planarity_sweeps_per_cycle: int = 8,
    relative_plane_tolerance: float = 5e-4,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    x = np.asarray(vertices, dtype=float).copy()
    f4 = np.asarray(faces, dtype=int)[:, :4]
    if len(x) == 0 or len(f4) == 0:
        return x, {"k3d_surface_constrained_applied": False}

    constraint_faces, _ = _robust_constraint_faces(x, f4)
    q = x[f4]
    edge_lengths = np.linalg.norm(np.roll(q, -1, axis=1) - q, axis=2).reshape(-1)
    positive = edge_lengths[edge_lengths > 1e-10]
    tile_scale = float(np.median(positive)) if len(positive) else 1.0
    plane_tol = max(1e-10, tile_scale * float(relative_plane_tolerance))

    before_plane = _quad_plane_distances(x, constraint_faces)
    max_before = float(np.max(before_plane)) if len(before_plane) else 0.0
    rms_before = float(np.sqrt(np.mean(before_plane * before_plane))) if len(before_plane) else 0.0

    initial = x.copy()
    previous = float("inf")
    stagnant = 0
    used_cycles = 0
    last_surface_correction = 0.0

    print(
        "[OPTCUTS-2026-09-13-K3D-SURFACE-START] "
        f"vertices={len(x)} faces={len(f4)} cycles={int(cycles)} "
        f"plane_tol={plane_tol:.6g} model=planarity<->target-surface"
    )

    for cycle in range(max(1, int(cycles))):
        # Step A: move toward the shared hard-planarity manifold.
        planar, _used, _ = _consensus_planarity_restore(
            x,
            f4,
            sweeps=max(1, int(planarity_sweeps_per_cycle)),
            relaxation=1.0,
            tolerance=None,
            constraint_faces=constraint_faces,
        )

        # Step B: return every vertex to the original target surface.  Ending
        # each cycle here makes the surface constraint authoritative.
        projected = _project_to_surface(planar, surface_vertices, surface_faces, pipeline)
        corrections = np.linalg.norm(projected - planar, axis=1)
        last_surface_correction = float(np.max(corrections)) if len(corrections) else 0.0
        x = projected
        used_cycles = cycle + 1

        plane = _quad_plane_distances(x, constraint_faces)
        max_plane = float(np.max(plane)) if len(plane) else 0.0
        rms_plane = float(np.sqrt(np.mean(plane * plane))) if len(plane) else 0.0
        print(
            "[OPTCUTS-2026-09-13-K3D-SURFACE] "
            f"cycle={used_cycles} max_plane_dist={max_plane:.6g} "
            f"rms_plane_dist={rms_plane:.6g} max_surface_return={last_surface_correction:.6g}"
        )

        # This is an alternating-projection feasibility experiment.  Stop when
        # the surface-constrained iterate is planar enough, or when improvement
        # has effectively stalled for several cycles.
        if max_plane <= plane_tol:
            break
        improvement = previous - max_plane
        if improvement <= max(plane_tol * 1e-3, abs(previous) * 1e-6 if np.isfinite(previous) else 0.0):
            stagnant += 1
        else:
            stagnant = 0
        previous = max_plane
        if stagnant >= 5:
            break

    after_plane = _quad_plane_distances(x, constraint_faces)
    max_after = float(np.max(after_plane)) if len(after_plane) else 0.0
    rms_after = float(np.sqrt(np.mean(after_plane * after_plane))) if len(after_plane) else 0.0

    # Because x is the direct output of closest-point projection, a second
    # projection should move it only by numerical noise.  Record that residual
    # instead of claiming exactness without measurement.
    reproj = _project_to_surface(x, surface_vertices, surface_faces, pipeline)
    surface_residual = np.linalg.norm(reproj - x, axis=1)
    surface_max = float(np.max(surface_residual)) if len(surface_residual) else 0.0
    surface_rms = float(np.sqrt(np.mean(surface_residual * surface_residual))) if len(surface_residual) else 0.0
    displacement = np.linalg.norm(x - initial, axis=1)

    feasible = bool(max_after <= plane_tol)
    print(
        "[OPTCUTS-2026-09-13-K3D-SURFACE-FINAL] "
        f"cycles={used_cycles} max_plane_dist={max_after:.6g} rms_plane_dist={rms_after:.6g} "
        f"tol={plane_tol:.6g} surface_max={surface_max:.6g} compatible={feasible}"
    )

    return x, {
        "k3d_surface_constrained_applied": True,
        "k3d_surface_constraint": "closest point on original target triangle mesh after every planarity step",
        "k3d_surface_planarity_algorithm": "alternating consensus-planarity and closest-surface projection",
        "k3d_surface_projection_authoritative": True,
        "k3d_surface_projection_cycles": int(used_cycles),
        "k3d_surface_planarity_sweeps_per_cycle": int(planarity_sweeps_per_cycle),
        "k3d_surface_planarity_tolerance": float(plane_tol),
        "k3d_surface_planarity_max_before": float(max_before),
        "k3d_surface_planarity_rms_before": float(rms_before),
        "k3d_surface_planarity_max_after": float(max_after),
        "k3d_surface_planarity_rms_after": float(rms_after),
        "k3d_surface_planarity_compatible": bool(feasible),
        "k3d_surface_projection_max_residual": float(surface_max),
        "k3d_surface_projection_rms_residual": float(surface_rms),
        "k3d_surface_last_return_distance_max": float(last_surface_correction),
        "k3d_surface_constrained_vertex_displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
        "k3d_surface_constrained_vertex_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0,
    }


def install_optcuts_test2_surface_constrained_k3d_patch(pipeline: Any) -> None:
    """Install the 2026-09-13 surface-constrained K3D experiment."""
    if getattr(pipeline, "_onestring_optcuts_test2_surface_k3d_installed", False):
        return

    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        result = base(target, mesh, parameterization, params)
        if not _is_surface_constrained_variant(params):
            return result

        k3d, report = result
        surface_vertices, surface_faces = _surface_vertices_faces(parameterization)
        solved, metrics = _alternate_planarity_and_surface(
            np.asarray(k3d.vertices, dtype=float),
            np.asarray(k3d.faces, dtype=int),
            surface_vertices,
            surface_faces,
            pipeline,
            cycles=int(os.environ.get("ONESTRING_K3D_SURFACE_PROJECTION_CYCLES", "24")),
            planarity_sweeps_per_cycle=int(os.environ.get("ONESTRING_K3D_SURFACE_PLANARITY_SWEEPS", "8")),
            relative_plane_tolerance=float(os.environ.get("ONESTRING_K3D_SURFACE_RELATIVE_PLANE_TOL", "5e-4")),
        )
        k3d.vertices[:] = solved
        try:
            k3d.metrics.update(metrics)
            k3d.metrics["k3d_planarity_mode"] = "2026-09-13 target-surface constrained alternating projection"
            k3d.metrics["k3d_surface_constrained_authoritative_result"] = True
        except Exception:
            pass
        try:
            report.objective = str(getattr(report, "objective", "")) + " + target-surface constrained planarity"
            report.constraint_violation = float(metrics.get("k3d_surface_planarity_max_after", 0.0))
        except Exception:
            pass
        return k3d, report

    pipeline._optimize_k3d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = optimize
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize

    pipeline._onestring_optcuts_test2_surface_k3d_installed = True


__all__ = [
    "install_optcuts_test2_surface_constrained_k3d_patch",
    "_alternate_planarity_and_surface",
]
