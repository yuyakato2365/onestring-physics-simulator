"""Measure whether Omega optimization changes propagate through K3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from onestring_physics import PipelineParameters, create_builtin_shape  # noqa: E402
from onestring_physics.input_shape import load_target_shape  # noqa: E402
from onestring_physics import onestring_pipeline as pipeline  # noqa: E402


def _array_delta(first: np.ndarray, second: np.ndarray) -> dict[str, float | bool | list[int]]:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        return {"same_shape": False, "first_shape": list(a.shape), "second_shape": list(b.shape)}
    distance = np.linalg.norm((b - a).reshape(-1, a.shape[-1]), axis=1)
    return {
        "same_shape": True,
        "rms": float(np.sqrt(np.mean(distance**2))) if len(distance) else 0.0,
        "max": float(np.max(distance)) if len(distance) else 0.0,
        "changed_fraction_gt_1e-9": float(np.mean(distance > 1.0e-9)) if len(distance) else 0.0,
    }


def _run(iterations: int, args: argparse.Namespace):
    if args.mesh:
        target = load_target_shape(args.mesh)
    else:
        radius = max(1.5, args.grid_size * args.tile_size * 0.7)
        target = create_builtin_shape(
            args.target,
            {"amplitude": args.amplitude, "radius": radius, "sigma": radius * 0.45},
        )
    params = PipelineParameters(
        nx=args.grid_size,
        ny=args.grid_size,
        tile_size=args.tile_size,
        gap_size=0.08,
        thickness=0.08,
        max_3d_iterations=args.max_3d_iterations,
        max_2d_iterations=args.max_2d_iterations,
        m3d_construction_mode="mesh_harmonic",
        surface_mesh_subdivisions=args.surface_mesh_subdivisions,
        omega_overlay_margin=1,
        m2d_crop_policy="strict_vertices",
        omega_boundary_mode="paper_default",
        omega_parameterization_mode="bijective_free_boundary",
        bijective_free_boundary_max_iterations=iterations,
        bijective_free_boundary_line_search_max_steps=20,
        allow_experimental_pipeline=True,
    )
    started = time.perf_counter()
    params.ny = params.nx if params.ny is None else params.ny
    grid = pipeline.create_quad_grid(params.nx, params.ny, params.tile_size, params.gap_size)
    surface = pipeline._original._build_surface_mesh(target, grid, params.surface_mesh_subdivisions)
    parameterization = pipeline._build_surface_parameterization(surface, target, grid, params)
    domain = pipeline._flatten_to_domain(parameterization, grid, params)
    mesh_2d_initial = pipeline._build_m2d(grid, domain, params)
    mesh_3d_initial, _ = pipeline._lift_m2d_to_m3d(
        target, mesh_2d_initial, parameterization, params
    )
    mesh_3d_optimized, _ = pipeline._optimize_k3d(
        target, mesh_3d_initial, parameterization, params
    )
    state = SimpleNamespace(
        surface_parameterization=parameterization,
        conformal_domain=domain,
        mesh_2d_initial=mesh_2d_initial,
        mesh_3d_initial=mesh_3d_initial,
        mesh_3d_optimized=mesh_3d_optimized,
    )
    elapsed = time.perf_counter() - started
    metrics = state.surface_parameterization.metrics
    summary = {
        "requested_iterations": iterations,
        "accepted_iterations": metrics.get("optimization_iteration_count"),
        "termination_reason": metrics.get("optimization_termination_reason"),
        "line_search_candidate_count": metrics.get("line_search_candidate_count"),
        "line_search_accepted_candidate_count": metrics.get("line_search_accepted_candidate_count"),
        "armijo_rejected_candidate_count": metrics.get("armijo_rejected_candidate_count"),
        "local_validity_rejected_candidate_count": metrics.get("local_validity_rejected_candidate_count"),
        "global_overlap_rejected_candidate_count": metrics.get("global_overlap_rejected_candidate_count"),
        "last_safe_step_reason": metrics.get("line_search_last_safe_step_reason"),
        "first_iteration_log": (metrics.get("optimization_iteration_log") or [None])[0],
        "initial_energy": metrics.get("initial_energy"),
        "final_energy": metrics.get("final_energy"),
        "energy_reduction_fraction": (
            1.0 - float(metrics["final_energy"]) / float(metrics["initial_energy"])
            if float(metrics.get("initial_energy", 0.0)) > 0.0
            else None
        ),
        "boundary_displacement_rms": metrics.get("boundary_displacement_rms"),
        "boundary_displacement_max": metrics.get("boundary_displacement_max"),
        "boundary_vertex_count": metrics.get("boundary_vertex_count"),
        "initial_boundary_circle_fit_relative_rms": metrics.get(
            "initial_boundary_circle_fit_relative_rms"
        ),
        "final_boundary_circle_fit_relative_rms": metrics.get(
            "final_boundary_circle_fit_relative_rms"
        ),
        "boundary_nonsimilarity_change_rms": metrics.get(
            "boundary_nonsimilarity_change_rms"
        ),
        "boundary_nonsimilarity_change_relative_rms": metrics.get(
            "boundary_nonsimilarity_change_relative_rms"
        ),
        "initial_boundary_radius_cv": metrics.get("initial_boundary_radius_cv"),
        "final_boundary_radius_cv": metrics.get("final_boundary_radius_cv"),
        "uv_flip_count": metrics.get("uv_triangle_flip_count"),
        "uv_overlap_count": metrics.get("internal_triangle_overlap_count"),
        "m2d_vertex_count": len(state.mesh_2d_initial.vertices),
        "m2d_face_count": len(state.mesh_2d_initial.faces),
        "k3d_vertex_count": len(state.mesh_3d_optimized.vertices),
        "k3d_face_count": len(state.mesh_3d_optimized.faces),
        "m3d_to_k3d_delta": _array_delta(
            state.mesh_3d_initial.vertices, state.mesh_3d_optimized.vertices
        ),
        "k3d_metrics": {
            key: state.mesh_3d_optimized.metrics.get(key)
            for key in (
                "compute_backend",
                "optimization_rejected",
                "fallback_used",
                "approximation_warning",
                "planarity_error_before",
                "planarity_error_after",
                "square_error_before",
                "square_error_after",
                "surface_fit_error_before",
                "surface_fit_error_after",
            )
        },
        "pipeline_seconds": float(elapsed),
    }
    return state, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, nargs="+", default=[60, 200])
    parser.add_argument("--target", default="dome")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--tile-size", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=0.75)
    parser.add_argument("--surface-mesh-subdivisions", type=int, default=2)
    parser.add_argument("--max-3d-iterations", type=int, default=40)
    parser.add_argument("--max-2d-iterations", type=int, default=40)
    args = parser.parse_args()

    states_and_summaries = [_run(iterations, args) for iterations in args.iterations]
    states = [item[0] for item in states_and_summaries]
    summaries = [item[1] for item in states_and_summaries]
    comparison = None
    if len(states) >= 2:
        first_state, second_state = states[0], states[1]
        comparison = {
        "omega_uv_delta": _array_delta(
            first_state.surface_parameterization.uv_vertices_2d,
            second_state.surface_parameterization.uv_vertices_2d,
        ),
        "omega_boundary_delta": _array_delta(
            first_state.surface_parameterization.omega_boundary,
            second_state.surface_parameterization.omega_boundary,
        ),
        "m2d_delta": _array_delta(first_state.mesh_2d_initial.vertices, second_state.mesh_2d_initial.vertices),
        "m3d_delta": _array_delta(first_state.mesh_3d_initial.vertices, second_state.mesh_3d_initial.vertices),
        "k3d_delta": _array_delta(first_state.mesh_3d_optimized.vertices, second_state.mesh_3d_optimized.vertices),
        "m2d_faces_identical": bool(
            np.array_equal(first_state.mesh_2d_initial.faces, second_state.mesh_2d_initial.faces)
        ),
        "k3d_faces_identical": bool(
            np.array_equal(first_state.mesh_3d_optimized.faces, second_state.mesh_3d_optimized.faces)
        ),
        }
    print(json.dumps({"runs": summaries, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
