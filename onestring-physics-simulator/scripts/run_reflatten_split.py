#!/usr/bin/env python3
"""Run experimental Paper Sec.4.5 reflatten Split on arbitrary open OBJ/STL/PLY.

This is an S -> Omega -> M2D -> M3D geometry/CSF diagnostic, not a full
K3D/T2D assembly. It does not force any Split count or modify the input mesh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onestring_physics.input_shape import create_builtin_shape, load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, build_paper_reference_initialization,
)
from onestring_physics.reference_bff import validate_reference_mesh


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--mesh", type=Path, help="Any manifold disk-type OBJ, STL or PLY")
    inputs.add_argument(
        "--builtin", choices=("flat", "dome", "gaussian", "saddle", "wave", "half_gourd"),
        help="Evaluate an internal open surface without importing a mesh",
    )
    p.add_argument("--grid", type=int, default=14, help="Reference grid resolution (NOT a paper author setting)")
    p.add_argument("--max-splits", type=int, default=64, help="Maximum number of hierarchical CSF splits")
    p.add_argument("--rings", type=int, default=3, help="Gaussian curvature smoothing neighborhood (1..10)")
    p.add_argument("--corner-mode", choices=("auto", "lobe_tips", "source_boundary_shape_fit", "fixed_phase"),
                   default="auto", help="auto: Lilium reference -> lobe_tips; otherwise generic source boundary fit")
    p.add_argument("--rotation", type=float, default=0.0, help="M2D square grid rotation in degrees")
    p.add_argument("--bff-executable", type=Path, default=None, help="Official free-boundary BFF executable")
    p.add_argument("--bff-rectangle-executable", type=Path, default=None, help="Official rectangular BFF adapter")
    p.add_argument("--report", type=Path, default=Path("output/reflatten_split_report.json"))
    p.add_argument("--expect-splits", type=int, default=None,
                   help="Optional regression assertion, never a constraint on the solver")
    return p


def run(args):
    if args.grid < 2 or args.max_splits < 0 or not 1 <= args.rings <= 10:
        raise ValueError("Require grid >= 2, max-splits >= 0 and 1 <= rings <= 10")
    if args.mesh is not None:
        path = args.mesh.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Target mesh does not exist: {path}")
        if path.suffix.lower() not in {".obj", ".stl", ".ply"}:
            raise ValueError("The mesh input must be OBJ, STL, or PLY")
        target = load_target_shape(path)
        # Official BFF and the current source cutting implementation require
        # one consistently oriented manifold disk. Reject unsupported closed
        # or multiply connected meshes explicitly, NOT by silently removing faces.
        topology = validate_reference_mesh(target.points, target.faces)
        identity = str(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        is_lilium = path.stem.lower().startswith("lilium")
    else:
        target = create_builtin_shape(
            args.builtin,
            {"amplitude": 0.25, "radius": max(2.5, args.grid * .6), "sigma": max(.8, args.grid * .2)},
        )
        topology = None  # The pipeline generates/validates the triangulation.
        identity = f"builtin:{args.builtin}"
        digest = None
        is_lilium = False

    mode = args.corner_mode
    if mode == "auto":
        mode = "lobe_tips" if is_lilium else "source_boundary_shape_fit"
    exe = (str(args.bff_executable) if args.bff_executable is not None
           else os.environ.get("ONESTRING_BFF_EXECUTABLE") or None)
    rect_exe = (str(args.bff_rectangle_executable)
                if args.bff_rectangle_executable is not None
                else os.environ.get("ONESTRING_BFF_RECTANGLE_EXECUTABLE") or None)

    p = PipelineParameters(
        nx=args.grid, ny=args.grid,
        omega_parameterization_mode="paper_reference_bff",
        omega_boundary_mode="paper_default",
        bff_boundary_policy="target_rectangle",
        bff_executable=exe,
        bff_rectangle_executable=rect_exe,
        bff_rectangle_selection_mode=mode,
        paper_split_criterion="reflatten_tile_area",
        paper_split_curvature_rings=args.rings,
        reference_grid_fit_rectangle=True,
        reference_grid_rotation_degrees=args.rotation,
        m2d_crop_policy="strict_vertices",
        enable_csf_splits=True, csf_split_threshold=2.,
        max_csf_splits=args.max_splits,
    )
    state=build_paper_reference_initialization(
        target, p,
        progress_callback=lambda name, fraction, detail="":
            print(f"[{fraction:5.1%}] {name}: {detail}", flush=True),
    )
    metrics=state.mesh_2d_initial.metrics
    source_report=metrics.get("reflatten_split_report", {})
    result={
        "input": identity,
        "input_sha256": digest,
        "input_topology": (
            {k:topology[k] for k in
             ("vertex_count","triangle_count","disk_topology","boundary_loop_count")}
            if topology else "generated_builtin_surface"
        ),
        "grid": args.grid,
        "corner_mode": mode,
        "criterion": metrics.get("csf_split_criterion"),
        "quad_count": len(state.mesh_2d_initial.faces),
        "split_count": metrics.get("csf_split_step_count"),
        "before_area_span": metrics.get("max_csf_before_split"),
        "after_area_span": metrics.get("max_csf_after_split"),
        "status": metrics.get("csf_split_status"),
        "stop_reason": metrics.get("csf_split_stopping_reason"),
        "steps": source_report.get("steps", []),
        "leaves": source_report.get("leaves", []),
        "author_algorithm_verified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ("steps","leaves")},
                     ensure_ascii=False, indent=2))
    print(f"Detailed report: {args.report}")
    if args.expect_splits is not None and result["split_count"] != args.expect_splits:
        raise AssertionError(
            f"Expected {args.expect_splits} Split(s); got {result['split_count']}"
        )
    return result


def main():
    p=make_parser()
    args=p.parse_args()
    try:
        run(args)
    except Exception as exc:
        p.exit(1, f"Reflatten Split failed: {type(exc).__name__}: {exc}\n")


if __name__=="__main__":
    main()
