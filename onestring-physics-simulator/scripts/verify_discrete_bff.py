from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from onestring_physics.discrete_bff import discrete_bff_rectangle


class _Parameters:
    boundary_target_aspect_mode = "fixed"
    boundary_target_aspect_ratio = 1.5
    boundary_target_aspect_min = 0.2
    boundary_target_aspect_max = 5.0


def _grid_mesh(nx: int, ny: int, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.8, 0.8, ny)
    vertices = []
    for y in ys:
        for x in xs:
            z = amplitude * np.sin(1.4 * x) * np.cos(1.2 * y)
            vertices.append([x, y, z])

    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an isolated discrete-BFF smoke test.")
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=7)
    parser.add_argument("--amplitude", type=float, default=0.25)
    args = parser.parse_args()

    vertices, faces = _grid_mesh(max(3, args.nx), max(3, args.ny), args.amplitude)
    uv, boundary_loop, metrics = discrete_bff_rectangle(vertices, faces, _Parameters())

    selected = {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "boundary_vertex_count": int(len(boundary_loop)),
        "uv_finite": bool(np.isfinite(uv).all()),
        "bff_implemented": bool(metrics.get("bff_implemented")),
        "bff_cherrier_formula_implemented": bool(metrics.get("bff_cherrier_formula_implemented")),
        "bff_best_fit_curve_implemented": bool(metrics.get("bff_best_fit_curve_implemented")),
        "bff_uses_lscm": bool(metrics.get("bff_uses_lscm")),
        "bff_gauss_bonnet_error": float(metrics.get("bff_gauss_bonnet_error", float("nan"))),
        "bff_neumann_rhs_sum_after_projection": float(metrics.get("bff_neumann_rhs_sum_after_projection", float("nan"))),
        "bff_best_fit_closure_error": float(metrics.get("bff_best_fit_closure_error", float("nan"))),
        "bff_best_fit_min_corrected_length": float(metrics.get("bff_best_fit_min_corrected_length", float("nan"))),
        "bff_positive_length_qp_used": bool(metrics.get("bff_positive_length_qp_used")),
        "uv_triangle_flip_count": int(metrics.get("uv_triangle_flip_count", -1)),
        "uv_degenerate_triangle_count": int(metrics.get("uv_degenerate_triangle_count", -1)),
    }
    print(json.dumps(selected, ensure_ascii=False, indent=2))

    failures = []
    if not selected["uv_finite"]:
        failures.append("UV contains non-finite values")
    if not selected["bff_implemented"]:
        failures.append("bff_implemented is false")
    if not selected["bff_cherrier_formula_implemented"]:
        failures.append("Cherrier solve is not active")
    if not selected["bff_best_fit_curve_implemented"]:
        failures.append("BestFitCurve is not active")
    if selected["bff_uses_lscm"]:
        failures.append("BFF path unexpectedly used LSCM")
    if selected["bff_best_fit_closure_error"] >= 1e-8:
        failures.append("boundary closure error is too large")
    if selected["bff_best_fit_min_corrected_length"] <= 0.0:
        failures.append("corrected boundary contains a non-positive edge")
    if selected["uv_triangle_flip_count"] != 0:
        failures.append("UV flips were detected")
    if selected["uv_degenerate_triangle_count"] != 0:
        failures.append("degenerate UV triangles were detected")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nPASS: discrete BFF is active, closed, positive-length, and flip-free on the smoke mesh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
