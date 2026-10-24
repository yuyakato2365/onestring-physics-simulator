#!/usr/bin/env python3
"""Phase-0 A/B stage video comparator. Does NOT regenerate any geometry.

Both inputs must have the identical source S vertices and faces, fixed
source-derived camera bounds, and a unique run_id. Reject silent
comparison of different source meshes or mismatched stage camera data.

Examples:
 python scripts/compare_stage_audits.py \
   --baseline artifacts/old-run-id \
   --candidate artifacts/new-run-id \
   --out artifacts/comparison-old-vs-new
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _meta(folder):
    path=Path(folder)
    return json.loads((path/"metrics.json").read_text(encoding="utf-8"))


def validate_runs(old,new):
    first=_meta(old)
    second=_meta(new)
    if first["run_id"]==second["run_id"]:
        raise ValueError("Cannot compare the SAME run_id as old and new")
    a=np.load(Path(old)/"stages.npz",allow_pickle=False)
    b=np.load(Path(new)/"stages.npz",allow_pickle=False)
    try:
        for key in ("S_vertices","S_faces","audit_fixed_bounds"):
            if key not in a or key not in b:
                raise ValueError(f"Source audit missing required key: {key}")
            if a[key].shape!=b[key].shape or not np.allclose(
                a[key],b[key],rtol=0,atol=1e-12
            ):
                raise ValueError(
                    "STAGE_AUDIT_DIFFERENT_SOURCE_OR_CAMERA: "+key
                )
    finally:
        a.close();b.close()
    for stage in ("S","M3D","K3D","T3D"):
        if stage in (first.get("rendered_stages") or []) and stage in (
            second.get("rendered_stages") or []
        ):
            continue
    for m in (first,second):
        protocol=m.get("render_protocol",{})
        if not (protocol.get("same_camera") and
                protocol.get("backface_culling") is False):
            raise ValueError(
                "STAGE_AUDIT_CAMERA_OR_CULLING_NOT_VERIFIED: "
                +str(m.get("run_id"))
            )
    return first,second


def compare_png(old,new,out,stage,view):
    from PIL import Image,ImageDraw
    a=Path(old)/f"{stage}_{view}.png"
    b=Path(new)/f"{stage}_{view}.png"
    if not a.exists() or not b.exists():
        raise FileNotFoundError(
            "Missing fixed-view PNG; independently render both runs: "
            f"{a} / {b}"
        )
    left=Image.open(a).convert("RGB")
    right=Image.open(b).convert("RGB")
    if left.size!=right.size:
        raise ValueError("STAGE_AUDIT_FRAME_SIZE_MISMATCH")
    panel=Image.new("RGB",(2*left.width,left.height+40),"#a8a8a8")
    panel.paste(left,(0,40));panel.paste(right,(left.width,40))
    ImageDraw.Draw(panel).text((12,12),"BASELINE",fill="#000000")
    ImageDraw.Draw(panel).text((left.width+12,12),"CANDIDATE",fill="#000000")
    panel.save(out/f"{stage}_{view}_comparison.png")


def compare_video(old,new,out,stage,fps):
    import imageio.v2 as imageio
    baseline=Path(old)/f"{stage}_turntable.mp4"
    candidate=Path(new)/f"{stage}_turntable.mp4"
    if not baseline.exists() or not candidate.exists():
        raise FileNotFoundError(
            "STAGE_AUDIT_MISSING_BOTH_RUN_VIDEOS: "
            f"{baseline} | {candidate}"
        )
    reader_a=imageio.get_reader(str(baseline))
    reader_b=imageio.get_reader(str(candidate))
    video=out/f"{stage}_before_after.mp4"
    gif=out/f"{stage}_before_after.gif"
    writer=imageio.get_writer(str(video),fps=fps,codec="libx264",quality=6)
    gif_writer=imageio.get_writer(str(gif),fps=fps,loop=0)
    count=0
    try:
        for a,b in zip(reader_a,reader_b):
            a=np.asarray(a);b=np.asarray(b)
            if a.shape!=b.shape:
                raise ValueError("STAGE_AUDIT_FRAME_SIZE_MISMATCH")
            joint=np.concatenate((a[:,:,:3],b[:,:,:3]),axis=1)
            writer.append_data(joint)
            gif_writer.append_data(joint)
            count+=1
    finally:
        writer.close();gif_writer.close()
        reader_a.close();reader_b.close()
    if count==0:
        raise RuntimeError("STAGE_AUDIT_NO_COMPARISON_FRAMES")
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--no-video",action="store_true")
    parser.add_argument("--fps",type=int,default=30)
    args=parser.parse_args()
    left,right=validate_runs(args.baseline,args.candidate)
    args.out.mkdir(parents=True,exist_ok=True)
    report={
        "baseline_run_id":left["run_id"],
        "candidate_run_id":right["run_id"],
        "baseline_git_sha":left.get("git_sha"),
        "candidate_git_sha":right.get("git_sha"),
        "source_vertices_identical":True,
        "fixed_camera_identical":True,
        "comparisons":{},
    }
    for stage in ("S","M3D","K3D","T3D"):
        if stage not in (left.get("rendered_stages") or []) or stage not in (
            right.get("rendered_stages") or []
        ):
            report["comparisons"][stage]="UNAVAILABLE_IN_SAME_RUN"
            continue
        for view in ("front","side","back","ears"):
            compare_png(args.baseline,args.candidate,args.out,stage,view)
        result={"png_views":4}
        if not args.no_video:
            result["video_frames"]=compare_video(
                args.baseline,args.candidate,args.out,stage,args.fps
            )
        report["comparisons"][stage]=result
    (args.out/"compare_metrics.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
