from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onestring_physics.input_shape import create_builtin_shape, load_target_shape  # noqa: E402
import onestring_physics.onestring_pipeline as pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OneString S->Omega parameterization modes.")
    parser.add_argument("--shape", default="dome", help="Built-in target shape when --mesh is omitted.")
    parser.add_argument("--mesh", type=Path, help="Optional OBJ/STL/PLY target mesh, for example a Bunny fixture.")
    parser.add_argument("--nx", type=int, default=3, help="M2D comparison grid size.")
    parser.add_argument("--surface-subdivisions", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = load_target_shape(args.mesh) if args.mesh else create_builtin_shape(args.shape, {"amplitude": 0.45, "radius": 2.0})
    grid = pipeline.create_quad_grid(args.nx, args.nx, 1.0, 0.08)
    surface = pipeline._build_surface_mesh(target, grid, args.surface_subdivisions)
    rows: list[dict[str, object]] = []
    for mode in ("bff", "lscm_paper_like", "boundary_sliding_lscm"):
        params = pipeline.PipelineParameters(
            nx=args.nx,
            ny=args.nx,
            surface_mesh_subdivisions=args.surface_subdivisions,
            omega_parameterization_mode=mode,
            omega_boundary_mode="paper_default",
            m2d_crop_policy="center",
            enable_csf_splits=False,
        )
        started = time.perf_counter()
        try:
            parameterization = pipeline._build_surface_parameterization(surface, target, grid, params)
            domain = pipeline._flatten_to_domain(parameterization, grid, params)
            mesh_2d = pipeline._build_m2d(grid, domain, params)
            mesh_3d, _report = pipeline._lift_m2d_to_m3d(target, mesh_2d, parameterization, params)
            metrics = parameterization.metrics
            rows.append(
                {
                    "mode": mode,
                    "uv_flip_count": int(metrics.get("uv_triangle_flip_count", 0)),
                    "uv_degenerate_count": int(metrics.get("uv_degenerate_triangle_count", 0)),
                    "angle_distortion_mean_deg": float(metrics.get("angle_distortion_mean_deg", 0.0)),
                    "angle_distortion_max_deg": float(metrics.get("angle_distortion_max_deg", 0.0)),
                    "csf_median": float(metrics.get("csf_median", 0.0)),
                    "csf_p95": float(metrics.get("csf_p95", 0.0)),
                    "csf_max": float(metrics.get("csf_max", 0.0)),
                    "boundary_self_intersection_count": int(metrics.get("boundary_self_intersection_count", 0)),
                    "lscm_energy_final": float(metrics.get("lscm_energy_final", 0.0)),
                    "m3d_uv_lookup_failure_count": int(mesh_3d.metrics.get("m3d_uv_lookup_failure_count", 0)),
                    "m3d_surface_triangle_hit_fraction": float(mesh_3d.metrics.get("m3d_surface_triangle_hit_fraction", 0.0)),
                    "runtime_seconds": float(time.perf_counter() - started),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_seconds": float(time.perf_counter() - started),
                }
            )
    print(json.dumps({"target": str(args.mesh) if args.mesh else args.shape, "rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
