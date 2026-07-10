"""Fail-fast verification of the official BFF backend and golden plane result."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from onestring_physics.reference_bff import run_official_bff, strict_inverse_map_uv_to_surface


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "official_bff_plane_windows_v1_6.json"


def _similarity_rms(actual: np.ndarray, expected: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float) - np.mean(actual, axis=0)
    b = np.asarray(expected, dtype=float) - np.mean(expected, axis=0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    rotation = u @ vt
    aligned = a @ rotation
    scale = float(np.sum(aligned * b) / max(np.sum(aligned * aligned), 1e-300))
    residual = scale * aligned - b
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def main() -> None:
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    vertices = np.asarray(fixture["vertices"], dtype=float)
    faces = np.asarray(fixture["faces"], dtype=int)
    expected = np.asarray(fixture["official_uv"], dtype=float)
    result = run_official_bff(vertices, faces, boundary_policy="boundary_scale_zero")
    rms = _similarity_rms(result.uv_vertices, expected)
    if rms > 1e-10:
        raise SystemExit(f"official BFF golden comparison failed: similarity RMS={rms:.3e}")
    round_trip: list[float] = []
    for vertex_id, uv in enumerate(result.uv_vertices):
        mapped, _triangle_id, _barycentric, error = strict_inverse_map_uv_to_surface(
            uv,
            result.uv_vertices,
            faces,
            vertices,
            faces,
            vertex_id=vertex_id,
        )
        round_trip.append(error)
        if float(np.min(np.linalg.norm(vertices - mapped, axis=1))) > 1e-10:
            raise SystemExit(f"inverse map did not return a plane vertex for UV vertex {vertex_id}")
    print(f"official_bff_backend={result.metrics['parameterization_backend_name']}")
    print(f"official_bff_version={result.metrics['parameterization_backend_version']}")
    print(f"official_bff_sha256={result.metrics['parameterization_backend_sha256']}")
    print(f"golden_similarity_rms={rms:.3e}")
    print(f"round_trip_max={max(round_trip, default=0.0):.3e}")


if __name__ == "__main__":
    main()

