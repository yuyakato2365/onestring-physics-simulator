#!/usr/bin/env python3
"""Sweep published Zhang QMSim numerical constraints, original algorithm ONLY."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from measure_zhang2021_original_panel_count import source_chart


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--exe-root",type=Path,required=True)
    parser.add_argument("--independent-eigen-solver-dll",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--hausdorff-bounds",nargs="+",type=float,
                        default=[.01,.05,.10,.25,.50])
    parser.add_argument("--sheet-widths",nargs="+",type=float,default=[.05,.15])
    parser.add_argument("--target-quads",type=int,default=35)
    a=parser.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    src=a.output_dir/"original_uv_source.npz"
    source_chart(7,src)
    records=[]
    for hb in a.hausdorff_bounds:
        for width in a.sheet_widths:
            if hb<=0 or width<=0: raise ValueError("Invalid author parameter")
            key=f"haus_{hb:.4f}_sheet_{width:.4f}".replace(".","p")
            quad=a.output_dir/f"{key}_native.npz"
            cmd=[sys.executable,str(Path(__file__).with_name(
                 "zhang2021_authors_native_adapter.py")),
                "--input",str(src),"--output",str(quad),
                "--exe-root",str(a.exe_root),
                "--independent-eigen-solver-dll",str(a.independent_eigen_solver_dll),
                "--target-quads",str(a.target_quads),
                "--hausdorff-bound",str(hb),
                "--sheet-width",str(width),
                "--timeout-seconds","80"]
            start=time.monotonic()
            result=subprocess.run(cmd,capture_output=True,text=True,
                                  errors="replace",timeout=110)
            record={"input_uv_grid":7,"source_triangles":72,
                    "target_quad_count":a.target_quads,
                    "hausdorff_bound":hb,"sheet_width":width,
                    "exit_code":result.returncode,
                    "runtime_seconds":round(time.monotonic()-start,2)}
            if result.returncode:
                record["failure"]=(result.stdout+"\n"+result.stderr)[-2400:]
            else:
                with np.load(quad,allow_pickle=False) as d:
                    record["actual_quads"]=len(d["faces"])
                    record["vertices_uv"]=len(d["vertices_uv"])
                # The exact original Omega coverage will be checked by the
                # separate strict OneString audit; no relaxed output accepted.
            records.append(record)
            print("ORIGINAL_ZHANG_HAUSDORFF_SWEEP",
                  json.dumps(record,ensure_ascii=False),flush=True)
            (a.output_dir/"original_hausdorff_sweep.json").write_text(
                json.dumps(records,indent=2,ensure_ascii=False)+"\n")
    if not any("actual_quads" in x for x in records):
        raise RuntimeError("Original author algorithm produced no usable run")


if __name__=="__main__":
    main()
