#!/usr/bin/env python3
"""Run the REAL OneString S->Omega->M2D->M3D->K3D->T3D and K2D->T2D.

Unlike verify_lilium_paper_split.py, this calls build_onestring_design,
reports every stage and exports actual geometry artifacts. Failed quality
guards remain errors and can never be relabeled as paper reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from onestring_physics.input_shape import load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, ComputeConfig, build_onestring_design
)
from onestring_physics.paper_fidelity_mode import (
    configure_paper_fidelity, check_paper_fidelity
)


def dump_obj(path,vertices,faces):
    """Write 2D or 3D quad top surfaces as transparent actual OBJ triangles."""
    vertices=np.asarray(vertices,dtype=float)
    faces=np.asarray(faces,dtype=int)
    if vertices.ndim!=2 or vertices.shape[1] not in (2,3) or faces.ndim!=2:
        raise ValueError("Invalid geometry")
    if vertices.shape[1]==2:
        vertices=np.column_stack([vertices,np.zeros(len(vertices))])
    with path.open("w",encoding="ascii") as out:
        out.write("# Actual OneString full-pipeline geometry\n")
        for v in vertices:
            out.write(f"v {v[0]:.12g} {v[1]:.12g} {v[2]:.12g}\n")
        for f in faces:
            out.write("f "+" ".join(str(int(i)+1) for i in f)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mesh",type=Path)
    ap.add_argument("--grid",type=int,default=7)
    ap.add_argument("--output",type=Path,default=Path("output/paper_full_pipeline_smoke"))
    ap.add_argument("--corner-selection",choices=("fixed_phase","min_whole_source_area_span_4phase"),
                    default="fixed_phase")
    ap.add_argument("--area-limit-two",action="store_true")
    ap.add_argument("--max-3d-iterations",type=int,default=10)
    ap.add_argument("--max-2d-iterations",type=int,default=10)
    args=ap.parse_args()
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    report=dict(
        source=str(args.mesh),sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        grid=args.grid,
        corner_selection=args.corner_selection,
        area_limit_two=bool(args.area_limit_two),
        stages=[],
        status="running",
    )
    def save():
        (output/"full_pipeline_result.json").write_text(json.dumps(report,indent=2,default=str))
    def progress(stage,fraction,detail=""):
        message=f"PIPELINE {fraction*100:5.1f}% {stage}: {detail}"
        print(message,flush=True)
        report["stages"].append(dict(stage=stage,fraction=float(fraction),detail=detail))
        save()
    p=PipelineParameters(
        nx=args.grid,ny=args.grid,
        omega_parameterization_mode="paper_reference_bff",
        max_3d_iterations=args.max_3d_iterations,
        max_2d_iterations=args.max_2d_iterations,
        max_csf_splits=64,
        paper_fidelity_mode=True,
        compute=ComputeConfig(backend="cpu",dtype="float64"),
    )
    p,_=configure_paper_fidelity(p)
    p.bff_rectangle_selection_mode=args.corner_selection
    p.paper_split_area_limit_experiment=bool(args.area_limit_two)
    p.paper_csf_sampling_method="interpolated_grid_vertices"
    p.reference_diagnostics_path=str(output/"paper_full_reference_diagnostics.json")
    tick=time.perf_counter()
    try:
        state=build_onestring_design(load_target_shape(args.mesh),p,progress_callback=progress)
        print("PIPELINE complete all stages",flush=True)
        stages=[
            ("M2D",state.mesh_2d_initial),("M3D",state.mesh_3d_initial),
            ("K3D",state.mesh_3d_optimized),("K2D",state.mesh_2d_optimized),
        ]
        for label,mesh in stages:
            v=np.asarray(mesh.vertices)
            f=np.asarray(mesh.faces)
            dump_obj(output/(label+".obj"),v,f)
            report[label]=dict(vertices=len(v),faces=len(f))
        tiles=np.asarray(state.tiles_3d.vertices)
        if tiles.ndim==3 and tiles.shape[1]>=4:
            dump_obj(output/"T3D_tile_tops.obj",tiles[:,:4,:].reshape(-1,3),
                     np.arange(tiles.shape[0]*4).reshape(-1,4))
            report["T3D_tile_count"]=int(len(tiles))
        t2d=np.asarray(state.tiles_2d_dual_hinge.vertices)
        if t2d.ndim==3 and t2d.shape[1]>=4:
            dump_obj(output/"T2D_tile_tops.obj",t2d[:,:4,:].reshape(-1,3),
                     np.arange(t2d.shape[0]*4).reshape(-1,4))
            report["T2D_tile_count"]=int(len(t2d))
        fidelity=check_paper_fidelity(state,p)
        report["paper_fidelity"]=dict(verified=fidelity.verified,
            stages=fidelity.stages,reasons=fidelity.reasons)
        report["final_split"]=dict(
            count=int(state.mesh_2d_initial.metrics.get("csf_split_count",0)),
            area_span=float(state.mesh_2d_initial.metrics.get("csf_split_area_ratio_after",float("nan"))),
            stop_reason=state.mesh_2d_initial.metrics.get("csf_split_stopping_reason"),
        )
        report["status"]="completed_unverified" if not fidelity.verified else "completed_verified"
    except Exception as error:
        report["status"]="FAILED"
        report["error"]=str(error)
        report["exception"]=type(error).__name__
        report["traceback"]=traceback.format_exc()
        print("PIPELINE FAILED",report["exception"],report["error"],flush=True)
        save()
        raise
    finally:
        report["elapsed_sec"]=time.perf_counter()-tick
        save()
        print("PIPELINE report:",str((output/"full_pipeline_result.json").resolve()),flush=True)


if __name__=="__main__":
    main()
