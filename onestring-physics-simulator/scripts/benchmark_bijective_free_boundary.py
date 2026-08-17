"""Benchmark the accelerated overlap checker and free-boundary optimizer.

The large case has 5,550 vertices.  Its brute-force overlap time is estimated
from the medium measurement by default because an O(F^2) run is not practical
for an interactive benchmark; pass ``--run-large-bruteforce`` to measure it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from onestring_physics.bijective_free_boundary import (  # noqa: E402
    BijectiveFreeBoundaryConfig,
    bijective_free_boundary_parameterization,
)
from onestring_physics.reference_bff import (  # noqa: E402
    count_internal_triangle_overlaps,
    count_internal_triangle_overlaps_bruteforce,
)


def _curved_grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.8, 0.8, ny)
    vertices = np.asarray(
        [[x, y, 0.25 * np.sin(1.4 * x) * np.cos(1.2 * y)] for y in ys for x in xs],
        dtype=float,
    )
    faces: list[list[int]] = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    boundary_count = 2 * (nx - 1) + 2 * (ny - 1)
    return vertices, np.asarray(faces, dtype=int), np.asarray([boundary_count], dtype=int)


def _timed_overlap(
    function: Any,
    uv: np.ndarray,
    faces: np.ndarray,
    **kwargs: Any,
) -> tuple[int, float, dict[str, int | float]]:
    stats: dict[str, int | float] = {}
    started = time.perf_counter()
    if function is count_internal_triangle_overlaps:
        count = function(uv, faces, stats=stats, **kwargs)
    else:
        count = function(uv, faces, **kwargs)
    return int(count), float(time.perf_counter() - started), stats


def _load_baseline_parameterizer(commit: str) -> tuple[Any, Any]:
    repository_root = PROJECT_ROOT.parent
    repository_path = repository_root.as_posix()
    source = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repository_path}",
            "-C",
            str(repository_root),
            "show",
            f"{commit}:onestring-physics-simulator/src/onestring_physics/bijective_free_boundary.py",
        ],
        text=True,
        encoding="utf-8",
    )
    module_name = "onestring_physics._benchmark_baseline_bijective_free_boundary"
    module = types.ModuleType(module_name)
    module.__file__ = f"git:{commit}:bijective_free_boundary.py"
    module.__package__ = "onestring_physics"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.count_internal_triangle_overlaps = count_internal_triangle_overlaps_bruteforce
    return module.bijective_free_boundary_parameterization, module.BijectiveFreeBoundaryConfig


def run_benchmark(
    *,
    run_large_bruteforce: bool,
    large_iterations: int,
    baseline_commit: str | None,
) -> dict[str, Any]:
    cases = [
        ("small", 12, 10, 10),
        ("medium", 32, 28, 5),
        ("large_5550_vertices", 75, 74, max(1, int(large_iterations))),
    ]
    rows: list[dict[str, Any]] = []
    medium_bruteforce_seconds: float | None = None
    medium_face_count: int | None = None
    medium_baseline_seconds: float | None = None
    baseline_parameterizer = None
    baseline_config = None
    if baseline_commit:
        baseline_parameterizer, baseline_config = _load_baseline_parameterizer(baseline_commit)
    for name, nx, ny, iterations in cases:
        vertices, faces, boundary = _curved_grid(nx, ny)
        uv = vertices[:, :2].copy()
        accelerated_count, accelerated_seconds, broad_phase = _timed_overlap(
            count_internal_triangle_overlaps, uv, faces
        )
        brute_count: int | None = None
        brute_seconds: float | None = None
        brute_estimated_seconds: float | None = None
        brute_is_estimate = False
        if name != "large_5550_vertices" or run_large_bruteforce:
            brute_count, brute_seconds, _ = _timed_overlap(
                count_internal_triangle_overlaps_bruteforce, uv, faces
            )
            if name == "medium":
                medium_bruteforce_seconds = brute_seconds
                medium_face_count = len(faces)
        elif medium_bruteforce_seconds is not None and medium_face_count is not None:
            brute_estimated_seconds = medium_bruteforce_seconds * (len(faces) / medium_face_count) ** 2
            brute_is_estimate = True

        baseline_seconds: float | None = None
        baseline_estimated_seconds: float | None = None
        baseline_is_estimate = False
        baseline_metrics: dict[str, Any] | None = None
        if baseline_parameterizer is not None and name != "large_5550_vertices":
            baseline_started = time.perf_counter()
            _baseline_uv, _baseline_loop, baseline_metrics = baseline_parameterizer(
                vertices,
                faces,
                baseline_config(max_iterations=iterations),
            )
            baseline_seconds = float(time.perf_counter() - baseline_started)
            if name == "medium":
                medium_baseline_seconds = baseline_seconds
        _uv, loop, metrics = bijective_free_boundary_parameterization(
            vertices,
            faces,
            BijectiveFreeBoundaryConfig(max_iterations=iterations),
        )
        if (
            baseline_parameterizer is not None
            and name == "large_5550_vertices"
            and brute_estimated_seconds is not None
        ):
            # The old ordering ran a brute-force global check for every line-search
            # candidate plus the initial and final UV.  Reuse the measured new run's
            # candidate count and non-overlap work, and label the result as estimated.
            old_overlap_calls = int(metrics["line_search_candidate_count"]) + 2
            non_overlap_seconds = max(
                0.0,
                float(metrics["parameterization_runtime_seconds"])
                - float(metrics["overlap_check_total_seconds"]),
            )
            baseline_estimated_seconds = brute_estimated_seconds * old_overlap_calls + non_overlap_seconds
            baseline_is_estimate = True
        rows.append(
            {
                "case": name,
                "V": len(vertices),
                "F": len(faces),
                "B": len(loop),
                "requested_iterations": iterations,
                "old_bruteforce_overlap_count": brute_count,
                "old_bruteforce_overlap_seconds": brute_seconds,
                "old_bruteforce_estimated_seconds": brute_estimated_seconds,
                "old_bruteforce_is_estimate": brute_is_estimate,
                "old_total_parameterization_seconds": baseline_seconds,
                "old_total_parameterization_estimated_seconds": baseline_estimated_seconds,
                "old_total_parameterization_is_estimate": baseline_is_estimate,
                "old_optimization_iterations": (
                    baseline_metrics.get("optimization_iteration_count") if baseline_metrics else None
                ),
                "new_overlap_count": accelerated_count,
                "new_overlap_seconds": accelerated_seconds,
                "old_new_overlap_count_match": brute_count == accelerated_count if brute_count is not None else None,
                "broad_phase_candidate_pairs": broad_phase.get("broad_phase_candidate_pair_count"),
                "total_possible_pairs": broad_phase.get("total_possible_pair_count"),
                "tutte_initialization_seconds": metrics["tutte_initialization_seconds"],
                "optimization_iterations": metrics["optimization_iteration_count"],
                "line_search_candidates": metrics["line_search_candidate_count"],
                "global_overlap_check_calls": metrics["overlap_check_call_count"],
                "energy_gradient_total_seconds": metrics["energy_gradient_total_seconds"],
                "safe_step_total_seconds": metrics["safe_step_total_seconds"],
                "overlap_total_seconds": metrics["overlap_check_total_seconds"],
                "total_parameterization_seconds": metrics["parameterization_runtime_seconds"],
                "final_energy": metrics["final_energy"],
                "uv_triangle_flip_count": metrics["uv_triangle_flip_count"],
                "uv_degenerate_triangle_count": metrics["uv_degenerate_triangle_count"],
                "boundary_self_intersection_count": metrics["boundary_self_intersection_count"],
                "internal_triangle_overlap_count": metrics["internal_triangle_overlap_count"],
                "optimization_converged": metrics["optimization_converged"],
                "termination_reason": metrics["optimization_termination_reason"],
            }
        )
    return {"cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-large-bruteforce", action="store_true")
    parser.add_argument("--large-iterations", type=int, default=1)
    parser.add_argument(
        "--baseline-commit",
        default=None,
        help="Load the old optimizer from this git commit and force its retained brute-force overlap checker.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        run_large_bruteforce=args.run_large_bruteforce,
        large_iterations=args.large_iterations,
        baseline_commit=args.baseline_commit,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
