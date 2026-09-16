"""Compare the existing discrete BFF and the free-boundary experiment.

Both methods use ``triangle_jacobian_diagnostics`` so lambda always means the
maximum singular value of the UV-to-surface Jacobian.  This script reports the
result; it intentionally does not assert that either method improves lambda.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from onestring_physics.bijective_free_boundary import (  # noqa: E402
    bijective_free_boundary_parameterization,
    boundary_self_intersection_count,
)
from onestring_physics.discrete_bff import discrete_bff_rectangle  # noqa: E402
from onestring_physics.reference_bff import (  # noqa: E402
    count_internal_triangle_overlaps,
    triangle_jacobian_diagnostics,
)


class _RectangleParameters:
    boundary_target_aspect_mode = "fixed"
    boundary_target_aspect_ratio = 1.5
    boundary_target_aspect_min = 0.2
    boundary_target_aspect_max = 5.0


def _curved_grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.8, 0.8, ny)
    vertices = np.asarray(
        [
            [x, y, 0.25 * np.sin(1.4 * x) * np.cos(1.2 * y)]
            for y in ys
            for x in xs
        ],
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
    return vertices, np.asarray(faces, dtype=int)


def _shared_summary(
    method: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    boundary_loop: list[int],
    runtime: float,
) -> dict[str, Any]:
    diagnostics = triangle_jacobian_diagnostics(vertices, uv, faces)
    lambda_values = np.asarray(diagnostics["raw_lambda_uv_to_surface_sigma_max"], dtype=float)
    log_lambda = np.log(lambda_values)
    anisotropy = np.asarray(diagnostics["anisotropy"], dtype=float)
    return {
        "method": method,
        "runtime_seconds": float(runtime),
        "flip_count": int(diagnostics["uv_triangle_flip_count"]),
        "overlap_count": int(count_internal_triangle_overlaps(uv, faces)),
        "boundary_self_intersection_count": boundary_self_intersection_count(uv, boundary_loop),
        "lambda_definition": "sigma_max of UV-to-surface Jacobian",
        "lambda_min": float(np.nanmin(lambda_values)),
        "lambda_median": float(np.nanmedian(lambda_values)),
        "lambda_max": float(np.nanmax(lambda_values)),
        "log_lambda_range": [float(np.nanmin(log_lambda)), float(np.nanmax(log_lambda))],
        "anisotropy_mean": float(np.nanmean(anisotropy)),
        "anisotropy_max": float(np.nanmax(anisotropy)),
    }


def compare(nx: int = 8, ny: int = 7) -> dict[str, dict[str, Any]]:
    vertices, faces = _curved_grid(nx, ny)
    bff_uv, bff_loop, bff_metrics = discrete_bff_rectangle(vertices, faces, _RectangleParameters())
    free_uv, free_loop, free_metrics = bijective_free_boundary_parameterization(vertices, faces)
    return {
        "bff": _shared_summary(
            "bff",
            vertices,
            faces,
            bff_uv,
            bff_loop,
            float(bff_metrics["parameterization_runtime_seconds"]),
        ),
        "bijective_free_boundary": _shared_summary(
            "bijective_free_boundary",
            vertices,
            faces,
            free_uv,
            free_loop,
            float(free_metrics["parameterization_runtime_seconds"]),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=7)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = compare(args.nx, args.ny)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
