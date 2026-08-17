from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from onestring_physics.official_ceps import official_ceps_rectangle


def _grid_mesh(nx: int = 8, ny: int = 7) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    for y in np.linspace(-0.8, 0.8, ny):
        for x in np.linspace(-1.0, 1.0, nx):
            vertices.append([x, y, 0.25 * np.sin(1.4 * x) * np.cos(1.2 * y)])
    faces = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([[a, b, d], [a, d, c]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def main() -> int:
    vertices, faces = _grid_mesh()
    params = SimpleNamespace(
        boundary_target_aspect_mode="fixed",
        boundary_target_aspect_ratio=1.5,
        boundary_target_aspect_min=0.2,
        boundary_target_aspect_max=5.0,
        ceps_timeout_seconds=600.0,
        ceps_keep_temporary_files=False,
    )
    result, boundary, metrics = official_ceps_rectangle(vertices, faces, params)
    summary = {
        "input_vertex_count": len(vertices),
        "input_triangle_count": len(faces),
        "common_refinement_vertex_count": len(result.surface_vertices),
        "common_refinement_triangle_count": len(result.surface_faces),
        "uv_vertex_count": len(result.uv_vertices),
        "omega_boundary_vertex_count": len(boundary) - 1,
        "ceps_backend_used": metrics["ceps_backend_used"],
        "ceps_reference_backend": metrics["ceps_reference_backend"],
        "ceps_common_refinement_used": metrics["ceps_common_refinement_used"],
        "ceps_texture_interpolation": metrics["ceps_texture_interpolation"],
        "ceps_prescribed_boundary_curvature": metrics["ceps_prescribed_boundary_curvature"],
        "uv_triangle_flip_count": metrics["uv_triangle_flip_count"],
        "uv_degenerate_triangle_count": metrics["uv_degenerate_triangle_count"],
        "boundary_aspect_relative_error": metrics["boundary_aspect_relative_error"],
        "runtime_seconds": metrics["parameterization_runtime_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["ceps_backend_used"] != "official_ceps_cli":
        raise RuntimeError("official CEPS backend was not used")
    if summary["uv_triangle_flip_count"] or summary["uv_degenerate_triangle_count"]:
        raise RuntimeError("official CEPS smoke mesh produced flipped or degenerate ordinary-UV triangles")
    print("\nPASS: official CEPS CLI, common refinement, and OneString UV import are active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
