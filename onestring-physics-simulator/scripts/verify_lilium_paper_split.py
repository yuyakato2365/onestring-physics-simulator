#!/usr/bin/env python3
"""Real-mesh Split comparison using the current official rectangular BFF backend.

This is a validation harness, not an alternate/fixed-count Split solver.
Use the same mesh, BFF boundary/rectangle and grid setup as the paper
to compare the reported cut count and locations against its Lilium figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from onestring_physics.input_shape import load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, build_paper_reference_initialization,
)
from onestring_physics.paper_fidelity_mode import configure_paper_fidelity


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh", type=Path, help="Original Lilium OBJ (no shape-specific behavior)")
    ap.add_argument("--grid", type=int, default=14, help="M2D grid resolution (default 14; NOT a confirmed author setting)")
    ap.add_argument("--margin", type=int, default=1, help="M2D grid padding")
    ap.add_argument("--rotation", type=float, default=0., help="Grid angle degrees")
    ap.add_argument("--report", type=Path, default=Path("output/lilium_paper_split_comparison.json"))
    ap.add_argument("--expect-splits", type=int, default=None, help="Optional benchmark assertion; never changes computation")
    args = ap.parse_args()
    source = args.mesh.expanduser().resolve()
    if not source.is_file():
        ap.error(f"Mesh does not exist: {source}")
    if args.grid < 2:
        ap.error("--grid must be >= 2")

    target = load_target_shape(source)
    # Verify only S->Omega->M2D->M3D to avoid unrelated K3D quality guards
    # masking the actual Split behavior.
    p = PipelineParameters(
        nx=args.grid,
        ny=args.grid,
        omega_overlay_margin=args.margin,
        reference_grid_rotation_degrees=args.rotation,
        omega_parameterization_mode="paper_reference_bff",
        bff_boundary_policy="target_rectangle",
        max_csf_splits=64,
        csf_split_threshold=2.,
        paper_fidelity_mode=True,
    )
    p, _ = configure_paper_fidelity(p)
    p.reference_diagnostics_path = str(args.report.parent / "lilium_reference_initialization.json")
    result = build_paper_reference_initialization(target, p)
    m = result.mesh_2d_initial.metrics
    para = result.surface_parameterization.metrics
    records = m.get("csf_split_step_analysis", [])
    cuts = []
    for rec in records:
        cuts.append({
            "axis": rec.get("axis"),
            "coordinate": rec.get("snapped_value"),
            "requested_coordinate": rec.get("requested_value"),
            "source_peak_vertex": rec.get("curvature_vertex_id"),
            "gaussian_curvature": rec.get("gaussian_curvature"),
            "linear_csf_before": rec.get("sigma_before"),
            "linear_csf_children": rec.get("sigma_after"),
            "linear_csf_worst_child_improvement": rec.get("worst_sigma_reduction"),
            "severed_edge_pairs": len(rec.get("split_boundary_pairs", [])),
        })
    report = {
        "mesh_file": str(source),
        "mesh_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "paper_reference_bff": {
            k: v for k, v in para.items()
            if k.startswith("bff_") or k.startswith("parameterization_backend")
        },
        "grid_request": {"n": args.grid, "margin": args.margin, "rotation": args.rotation},
        "actual_m2d_quads": len(result.mesh_2d_initial.faces),
        "number_of_splits": len(records),
        "one_split_picture_count_matches": len(records) == 1,
        "linear_csf_before": m.get("max_csf_before_split"),
        "linear_csf_after": m.get("max_csf_after_split"),
        "area_jacobian_span_after": m.get("csf_split_area_ratio_after"),
        "linear_csf_limit_two_satisfied": m.get("csf_split_candidate_linear_bound_satisfied"),
        "area_expansion_limit_two_satisfied": m.get("csf_split_candidate_area_bound_satisfied"),
        "fabrication_bound_certified": m.get("csf_split_bound_certified"),
        "stopping_reason": m.get("csf_split_stopping_reason"),
        "split_lines": m.get("csf_split_locations"),
        "cuts": cuts,
        "rejected_regions": m.get("csf_split_rejected_components"),
        "quad_sliver_diagnostics": {
            k: para.get(k) for k in (
                "split_quad_irreducible_count",
                "split_quad_lo_set_by_sliver_count",
                "split_quad_hi_set_by_sliver_count",
                "split_quad_coverage_floor_sweep",
            )
        },
        "limitations": [
            "Matching a single cut is not sufficient to match the published cut position.",
            "The authors' original BFF rectangle corners and grid orientation are unknown.",
            "The actual per-piece UV rescaling and fabrication certificate are not implemented.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print("mesh_sha256:", report["mesh_sha256"])
    print("quads:", report["actual_m2d_quads"])
    print("splits:", report["number_of_splits"])
    print("cut lines:", report["split_lines"])
    print("linear CSF:", report["linear_csf_before"], "->", report["linear_csf_after"])
    print("area ratio:", report["area_jacobian_span_after"])
    print("stopping reason:", report["stopping_reason"])
    print("report:", args.report.resolve())
    if args.expect_splits is not None and len(records) != args.expect_splits:
        print("MISMATCH: split count differs from the requested paper benchmark", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
