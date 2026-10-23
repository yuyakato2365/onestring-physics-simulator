#!/usr/bin/env python3
"""Replay *only* fixed-boundary ear feasibility from a saved diagnostic ZIP.

NO Streamlit, native OptCuts binary or expensive UV optimization is needed.
Usage:
    python3 scripts/replay_grid_ear_correspondence.py \\
        logs/grid_boundary_diagnostics/grid_failure_evidence.zip

For each saved attempt, checks whether its original full-grid boundary would
force any 3-boundary-vertex triangle below the existing 0.005 positive area
guard. Re-generates all corner correspondences with the new ear-aware repair.
A zero count proves only the necessary ear condition, NOT global bijectivity.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from onestring_physics.optcuts_area_grid_boundary import (
    _boundary_target_candidates,
)
from onestring_physics.optcuts_grid_ear_correspondence import (
    constrained_boundary_faces, fixed_boundary_ear_report,
)


def replay(path):
    path=Path(path)
    reports=[]
    with ZipFile(path) as archive:
        uvfiles=sorted(x for x in archive.namelist()
                       if x.endswith("/reproduce_uv.npz"))
        if not uvfiles:
            raise ValueError("ZIP has no snapshots/*/reproduce_uv.npz")
        for name in uvfiles:
            with np.load(io.BytesIO(archive.read(name))) as arrays:
                base=arrays["base_uv"].copy()
                faces=arrays["uv_faces"].copy()
                loop=arrays["boundary_indices"].copy()
                outline=arrays["grid_outline"].copy()
                old_target=arrays["target_full_boundary_uv"].copy()
            ear_ids, _=constrained_boundary_faces(faces,loop,len(base))
            original=fixed_boundary_ear_report(
                base,faces,loop,old_target)
            repaired=_boundary_target_candidates(
                base,loop,outline,faces=faces)
            options=[]
            for score,phase,target,method in repaired:
                checked=fixed_boundary_ear_report(
                    base,faces,loop,target)
                options.append({
                    "phase":int(phase), "method":method,
                    "cost":float(score),
                    "forced_bad_ear_count":len(checked["bad_ids"]),
                    "min_boundary_ear_area_ratio":
                        checked["minimum_fixed_ratio"],
                })
            report={
                "snapshot":name,
                "boundary_only_triangle_ids":ear_ids.tolist(),
                "original_failed_correspondence_bad_ear_ids":
                    original["bad_ids"].tolist(),
                "original_fixed_min_ear_ratio":
                    original["minimum_fixed_ratio"],
                "new_correspondence_count":len(options),
                "new_ear_safe_candidates":
                    sum(x["forced_bad_ear_count"]==0 for x in options),
                "best_option":options[0] if options else None,
                "note":"This test is necessary, never sufficient for "
                       "global UV bijectivity or distortion acceptance.",
            }
            reports.append(report)
            print("[GRID-EAR-REPLAY] snapshot="+name)
            print(
                "[GRID-EAR-REPLAY] original forced-ear failures="
                +str(report["original_failed_correspondence_bad_ear_ids"])
            )
            print(
                "[GRID-EAR-REPLAY] repaired candidates="
                f"{report['new_ear_safe_candidates']}; "
                f"best={report['best_option']}"
            )
    output=path.parent/"grid_ear_replay_result.json"
    output.write_text(json.dumps(reports,indent=2,ensure_ascii=False,
                                 allow_nan=False),encoding="utf-8")
    print("[GRID-EAR-REPLAY] saved="+str(output))
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path,
                        help="grid_failure_evidence.zip")
    args=parser.parse_args()
    replay(args.zip_path)


if __name__=="__main__":
    main()
