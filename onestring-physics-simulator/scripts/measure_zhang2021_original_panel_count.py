#!/usr/bin/env python3
"""Measure ORIGINAL authors' actual quad count vs input UV triangulation density.

Only uses published Zhang Straighten/Segment/QMSim executables and optionally
the separately licensed numerical Eigen backend. Does not implement a
different mesher, crop Omega or silently accept invalid candidate.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np


def source_chart(n:int,dest:Path):
    uv=np.array([(i/(n-1),j/(n-1)) for j in range(n) for i in range(n)],float)
    faces=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i
            faces.extend(((a,a+1,a+n+1),(a,a+n+1,a+n)))
    triangles=np.asarray(faces,dtype=np.int32)
    xyz=np.column_stack((uv,.14*np.sin(math.pi*uv[:,0])*np.sin(math.pi*uv[:,1])))
    boundary=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
    np.savez_compressed(dest,source_uv_vertices=uv,source_uv_faces=triangles,
        source_xyz_vertices=xyz,source_xyz_faces=triangles,
        omega_boundary=boundary)


def calibrate(exe_root:Path,numeric_dll:Path,output_dir:Path,
              grid_sizes:tuple[int,...],targets:tuple[int,...]):
    output_dir.mkdir(parents=True,exist_ok=True)
    measurements=[]
    for n in grid_sizes:
        source=output_dir/f"input_n{n}.npz"
        source_chart(n,source)
        for target in targets:
            output=output_dir/f"original_n{n}_target{target}.npz"
            cmd=[
                sys.executable, str(Path(__file__).with_name(
                    "zhang2021_authors_native_adapter.py")),
                "--input",str(source),"--output",str(output),
                "--exe-root",str(exe_root),
                "--independent-eigen-solver-dll",str(numeric_dll),
                "--target-quads",str(target),"--timeout-seconds","90",
            ]
            start=time.monotonic()
            process=subprocess.run(
                cmd,capture_output=True,text=True,errors="replace",
                check=False,timeout=120)
            record={"grid_n":n,"source_triangles":2*(n-1)**2,
                    "requested_target":target,
                    "runtime_seconds":round(time.monotonic()-start,3),
                    "exit_code":process.returncode}
            if process.returncode!=0:
                record["failure"]=(process.stderr+"\n"+process.stdout)[-2400:]
            else:
                with np.load(output,allow_pickle=False) as q:
                    record["actual_quads"]=int(len(q["faces"]))
                record["native_stages"]=3
            measurements.append(record)
            print("ORIGINAL_ZHANG_COUNT_EXPERIMENT",
                  json.dumps(record,ensure_ascii=False),flush=True)
            (output_dir/"panel_count_measurements.json").write_text(
                json.dumps(measurements,indent=2,ensure_ascii=False)+"\n",
                encoding="utf-8")
    return measurements


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--exe-root",type=Path,required=True)
    parser.add_argument("--independent-eigen-solver-dll",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--grid-sizes",nargs="+",type=int,default=[5,7,12])
    parser.add_argument("--targets",nargs="+",type=int,default=[8,20,35])
    args=parser.parse_args()
    results=calibrate(args.exe_root,args.independent_eigen_solver_dll,
        args.output_dir,tuple(args.grid_sizes),tuple(args.targets))
    if not any("actual_quads" in r for r in results):
        raise RuntimeError("Original authors' mesher never produced output")


if __name__=="__main__":
    main()
