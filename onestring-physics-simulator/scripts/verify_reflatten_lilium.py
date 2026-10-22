#!/usr/bin/env python3
"""Independent Lilium reproduction audit through M3D (no expensive K3D/T2D).

Requires official bff-command-line, bff-rect-command-line and lilium.obj.
Never treats the OneString authors' algorithm as verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from onestring_physics.input_shape import load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, build_paper_reference_initialization
)

LILIUM_SHA256 = "4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh",type=Path,help="Path to original libigl lilium.obj")
    parser.add_argument("--grid",type=int,default=14)
    parser.add_argument("--max-splits",type=int,default=64)
    parser.add_argument("--assert-zip-lilium-result",action="store_true",
        help="Require 196 quads and 1 Split (reference from uploaded handoff, not author proof)")
    parser.add_argument("--report",type=Path,default=Path("output/reflatten_lilium_report.json"))
    args=parser.parse_args()
    digest=hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    if digest!=LILIUM_SHA256:
        print(f"WARNING: lilium.obj SHA256 differs: {digest}",file=sys.stderr)
    params=PipelineParameters(
        nx=args.grid,ny=args.grid,
        omega_parameterization_mode="paper_reference_bff",
        omega_boundary_mode="paper_default",
        bff_boundary_policy="target_rectangle",
        bff_executable=os.environ.get("ONESTRING_BFF_EXECUTABLE") or None,
        bff_rectangle_executable=os.environ.get("ONESTRING_BFF_RECTANGLE_EXECUTABLE") or None,
        bff_rectangle_selection_mode="lobe_tips",
        paper_split_criterion="reflatten_tile_area",
        paper_split_curvature_rings=3,
        reference_grid_fit_rectangle=True,
        csf_split_threshold=2.0,
        max_csf_splits=args.max_splits,
        m2d_crop_policy="strict_vertices",
        enable_csf_splits=True,
    )
    state=build_paper_reference_initialization(load_target_shape(str(args.mesh)),params,
        progress_callback=lambda stage,frac,detail="":print(f"[{frac:5.1%}] {stage}: {detail}",flush=True))
    mesh=state.mesh_2d_initial
    data=dict(
        input_sha256=digest,grid=args.grid,quad_count=int(len(mesh.faces)),
        split_count=int(mesh.metrics.get("csf_split_step_count",0)),
        criterion=mesh.metrics.get("csf_split_criterion"),
        before=mesh.metrics.get("max_csf_before_split"),
        after=mesh.metrics.get("max_csf_after_split"),
        stop_reason=mesh.metrics.get("csf_split_stopping_reason"),
        steps=mesh.metrics.get("reflatten_split_report",{}).get("steps",[]),
        leaves=mesh.metrics.get("reflatten_split_report",{}).get("leaves",[]),
        is_oneString_authors_code=False,
    )
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in data.items() if k not in ("steps","leaves")},
                     indent=2,ensure_ascii=False))
    print(f"FULL REPORT: {args.report}")
    if args.assert_zip_lilium_result:
        assert data["quad_count"]==196, f"expected 196, got {data['quad_count']}"
        assert data["split_count"]==1, f"expected 1 Split, got {data['split_count']}"
        assert data["stop_reason"]=="all components <= 2",data["stop_reason"]


if __name__=="__main__":main()
