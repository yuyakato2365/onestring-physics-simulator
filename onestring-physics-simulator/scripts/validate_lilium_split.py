#!/usr/bin/env python3
"""Run a falsifiable, manufacturing-grid-preserving Lilium Split experiment.

This never invokes or rewrites the regular OneString pipeline. Model B0 is a
diagnostic fixed-UV proxy; for authoritative preexisting Paper mode run
scripts/run_paper_full_pipeline.py separately on the reference branch.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import zlib
import struct
from dataclasses import asdict

import numpy as np

from onestring_physics.reference_bff import run_official_bff
from onestring_physics.split_validation.core import (
    ExperimentConfig, load_triangle_obj, mesh_digest, lobe_tip_corners,
    grid_rectangles, area_jacobians, scale_span, clip_source_to_box,
    tile_jacobians, curvature_field, nearest_curvature_grid_vertex,
    split_tile_ids,
)

CASES = {
    "B0": ("fixed_phase", "original_origin_max_side", 0, "fixed_uv_weighted_tile_area"),
    "B1": ("lobe_tips", "original_origin_max_side", 0, "fixed_uv_weighted_tile_area"),
    "B2": ("fixed_phase", "fit_center_min_side", 0, "fixed_uv_weighted_tile_area"),
    "B3": ("fixed_phase", "original_origin_max_side", 3, "fixed_uv_weighted_tile_area"),
    "B4": ("fixed_phase", "original_origin_max_side", 0, "free_bff_weighted_tile_area"),
    "B5": ("lobe_tips", "fit_center_min_side", 3, "fixed_uv_weighted_tile_area"),
    "B6": ("lobe_tips", "fit_center_min_side", 3, "free_bff_weighted_tile_area"),
    "B7": ("lobe_tips", "fit_center_min_side", 3, "rectangle_child_bff_weighted_tile_area"),
}
EXPECTED_SHA256 = "4f62ff741b41bd629ccbeaeb0bb2e44ac060cdff88f090129911bede790e8ab3"


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v,(list,dict))
                             else v for k,v in row.items()})


def check_tile_group_connectivity(tiles, ids):
    cells = {(int(tiles[i][0]), int(tiles[i][1])) for i in ids}
    if not cells:
        return False
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        x,y = stack.pop()
        for nxt in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == cells


def write_png(path, tiles, groups, cut_records):
    """Dependency-free overview: original grid, child labels, and cut lines."""
    width = height = 720
    buf = bytearray([255]*(width*height*3))
    def rectangle(x0,y0,x1,y1,color):
        for y in range(max(0,y0),min(height,y1)):
            k = 3*(y*width+max(0,x0))
            for x in range(max(0,x0),min(width,x1)):
                buf[k:k+3] = bytes(color)
                k += 3
    n = max(max(t[0],t[1]) for t in tiles)+1
    palette = [(132,172,206),(244,177,128),(157,199,148),
               (191,171,208),(227,203,107)]
    assignment = {}
    for ci,g in enumerate(groups):
        for idx in g["tile_ids"]:
            assignment[idx] = ci
    origin, scale = 50, 610/n
    for i,t in enumerate(tiles):
        x,y = int(t[0]), int(t[1])
        px = int(origin+x*scale)
        py = int(origin+(n-1-y)*scale)
        rectangle(px,py,int(origin+(x+1)*scale-2),
                  int(origin+(n-y)*scale-2),
                  palette[assignment.get(i,0)%len(palette)])
    for cut in cut_records:
        axis, value = cut["axis"], cut["coordinate"]
        coords = sorted({t[2] if axis=="col" else t[3] for t in tiles})
        idx = int(np.argmin(abs(np.asarray(coords)-value)))
        if axis=="col":
            px = int(origin+idx*scale)
            rectangle(px-3,origin,px+3,int(origin+n*scale),(160,30,40))
        else:
            py = int(origin+(n-idx)*scale)
            rectangle(origin,py-3,int(origin+n*scale),py+3,(160,30,40))
    rows = b"".join(b"\0"+bytes(buf[width*y*3:width*(y+1)*3])
                    for y in range(height))
    def chunk(name,data):
        return struct.pack(">I",len(data))+name+data+struct.pack(
            ">I",zlib.crc32(name+data)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+
        chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+
        chunk(b"IDAT",zlib.compress(rows,level=6))+chunk(b"IEND",b""))


def run_case(xyz, faces, case_id, base, grid_n, max_splits,
             curvature_override=None, fixed_cut_sweep=False):
    cp, gp, rings, metric = CASES[case_id]
    if curvature_override is not None:
        rings = curvature_override
    if cp == "lobe_tips":
        corners = lobe_tip_corners(xyz, faces)
        init = run_official_bff(xyz, faces, boundary_policy="target_rectangle",
                                rectangle_corner_positions=xyz[corners])
    else:
        corners = None
        init = run_official_bff(xyz, faces, boundary_policy="target_rectangle")
    source_uv = np.asarray(init.uv_vertices, float)
    tiles, grid_meta = grid_rectangles(source_uv, grid_n, gp)
    if not tiles:
        raise ValueError("No fully enclosed grid tiles")
    curvature = curvature_field(xyz, faces, rings)
    box = (*np.min(source_uv,axis=0), *np.max(source_uv,axis=0))
    allids = list(range(len(tiles)))
    cache = {}

    def evaluate(bounds, ids):
        key = (tuple(float(v) for v in bounds), tuple(ids), metric)
        if key in cache:
            return cache[key]
        childxyz, parent_uv, childfaces = clip_source_to_box(
            xyz, source_uv, faces, bounds)
        if metric == "fixed_uv_weighted_tile_area":
            chart_uv = parent_uv
            backend = "initial_rectangle_bff_uv_restricted"
        elif metric == "free_bff_weighted_tile_area":
            bff = run_official_bff(childxyz, childfaces,
                                   boundary_policy="boundary_scale_zero")
            chart_uv = bff.uv_vertices
            backend = bff.metrics.get("parameterization_backend_sha256")
        elif metric == "rectangle_child_bff_weighted_tile_area":
            bff = run_official_bff(childxyz, childfaces,
                                   boundary_policy="target_rectangle")
            chart_uv = bff.uv_vertices
            backend = bff.metrics.get("parameterization_backend_sha256")
        else:
            raise ValueError("Unknown metric")
        jac = area_jacobians(childxyz, chart_uv, childfaces)
        values, cov = tile_jacobians(parent_uv, jac, childfaces,
                                   [tiles[i] for i in ids])
        record = dict(
            sigma=scale_span(values, units="area_ratio"),
            min_j=float(np.min(values)), max_j=float(np.max(values)),
            valid_tile_count=len(values), min_coverage=float(np.min(cov)),
            source_triangle_count=len(childfaces),
            boundary_policy=metric, backend=backend,
            units="area_ratio", tile_sample_values=values.tolist(),
            bbox=list(map(float,bounds)),
        )
        cache[key] = record
        return record

    def split_info(bounds, ids, axis, cut):
        left,right = split_tile_ids(tiles, ids, axis, cut)
        bb1, bb2 = list(bounds), list(bounds)
        idx = 0 if axis == "col" else 1
        bb1[idx+2] = cut
        bb2[idx] = cut
        if not check_tile_group_connectivity(tiles,left):
            raise ValueError("Disconnected left child tiles")
        if not check_tile_group_connectivity(tiles,right):
            raise ValueError("Disconnected right child tiles")
        d1,d2 = evaluate(bb1,left),evaluate(bb2,right)
        return dict(axis=axis,coordinate=float(cut),
                    children=[{"bbox":bb1,"tile_ids":left,"metric":d1},
                              {"bbox":bb2,"tile_ids":right,"metric":d2}],
                    worst_child_sigma=max(d1["sigma"],d2["sigma"]),
                    child_counts=[len(left),len(right)])

    initial = evaluate(box, allids)
    components = [{"bbox":list(map(float,box)), "tile_ids":allids, "metric":initial}]
    cuts, candidates, stops = [], [], []
    while len(cuts) < max_splits:
        bad = [i for i,c in enumerate(components)
               if c["metric"]["sigma"] > base.threshold + 1e-10]
        if not bad:
            break
        comp_idx = max(bad,key=lambda i:components[i]["metric"]["sigma"])
        comp = components[comp_idx]
        bounds, ids = comp["bbox"], comp["tile_ids"]
        try:
            peak, peak_k, projection_distance = nearest_curvature_grid_vertex(
                curvature, source_uv, tiles, ids)
        except ValueError as e:
            stops.append(str(e))
            break
        trial = []
        for axis, val in (("col",peak[0]),("row",peak[1])):
            try:
                cut = split_info(bounds,ids,axis,float(val))
                cut.update(component_id=comp_idx,curvature_grid_point=peak.tolist(),
                           curvature_score=peak_k,
                           curvature_projection_distance=projection_distance)
                trial.append(cut)
                candidates.append({k:v for k,v in cut.items() if k!="children"})
            except (ValueError,RuntimeError) as e:
                candidates.append(dict(axis=axis, coordinate=float(val),
                            rejected_reason=str(e), component_id=comp_idx))
        if not trial:
            stops.append("No admissible candidate at curvature peak")
            break
        best = min(trial,key=lambda c:(c["worst_child_sigma"],c["axis"]))
        cuts.append({k:v for k,v in best.items() if k!="children"})
        components.pop(comp_idx)
        components.extend(best["children"])
    if len(cuts)==max_splits and any(c["metric"]["sigma"]>base.threshold
                                   for c in components):
        reason="budget_exhausted"
    elif all(c["metric"]["sigma"]<=base.threshold for c in components):
        reason="all_components_within_threshold"
    else:
        reason="no_admissible_cut"
    if fixed_cut_sweep:
        for axis in ("col","row"):
            coord = 0 if axis=="col" else 1
            vals = sorted({float(t[2+coord]) for t in tiles} |
                          {float(t[4+coord]) for t in tiles})
            for v in vals[1:-1]:
                try:
                    d = split_info(box,allids,axis,v)
                    candidates.append(dict(axis=axis,coordinate=v,
                        worst_child_sigma=d["worst_child_sigma"],
                        child_counts=d["child_counts"], origin="all_cuts_sweep"))
                except (ValueError,RuntimeError) as e:
                    candidates.append(dict(axis=axis,coordinate=v,
                         origin="all_cuts_sweep",rejected_reason=str(e)))
    return dict(
        case=case_id, description=dict(corners=cp,grid=gp,rings=rings,metric=metric),
        metric_id=metric+"_v1", units="area_ratio", grid=grid_meta,
        bff_initial=init.metrics.get("parameterization_backend_sha256"),
        rectangle_corners_source_vertex_ids=corners,
        original_source_uv_sha256=hashlib.sha256(source_uv.tobytes()).hexdigest(),
        initial=initial, initial_tile_count=len(tiles),
        cuts=cuts, components=components, split_count=len(cuts),
        stopping_reason=reason, rejections=stops, cut_candidates=candidates,
        manufacturing_grid_held_constant=True,
        free_bff_only_for_diagnostic=True,
        physical_mechanism_certified=False,
    ), tiles


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh",type=Path)
    p.add_argument("--grid",type=int,default=14)
    p.add_argument("--cases",default="B0,B6")
    p.add_argument("--max-splits",type=int,default=8)
    p.add_argument("--threshold",type=float,default=2.)
    p.add_argument("--sweep",action="store_true",
                   help="Expensive: evaluate every first-cut Grid line")
    p.add_argument("--sensitivity",action="store_true",
                   help="Expensive: 20 Grid/smoothing pairs for B6")
    p.add_argument("--require-lilium-sha",action="store_true")
    p.add_argument("--output",type=Path,default=Path("output/split_validation"))
    args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"figures").mkdir(exist_ok=True)
    sha=mesh_digest(args.mesh)
    if args.require_lilium_sha and sha!=EXPECTED_SHA256:
        raise ValueError("Lilium input SHA differs from the registered libigl mesh")
    xyz,faces=load_triangle_obj(args.mesh)
    bff_env={k:os.environ.get(k) for k in
             ("ONESTRING_BFF_EXECUTABLE","ONESTRING_BFF_RECTANGLE_EXECUTABLE")}
    environment=dict(python=sys.version,os=platform.platform(),numpy=np.__version__,
                     bff_environment=bff_env,input_sha256=sha)
    (args.output/"environment.json").write_text(json.dumps(environment,indent=2))
    config=vars(args).copy()
    config["mesh"]=str(args.mesh)
    config["output"]=str(args.output)
    (args.output/"run_config.json").write_text(json.dumps(config,indent=2))
    cfg=ExperimentConfig(grid_n=args.grid,threshold=args.threshold,
                         max_splits=args.max_splits)
    outputs, rows, candidates, sensitivity = {}, [], [], []
    for name in args.cases.split(","):
        case=name.strip().upper()
        if case not in CASES:
            raise ValueError("Unknown case "+case)
        try:
            print("Running",case,flush=True)
            result,tiles=run_case(xyz,faces,case,cfg,args.grid,
                                  args.max_splits,fixed_cut_sweep=args.sweep)
            outputs[case]=result
            rows.append(dict(case=case,metric=result["metric_id"],
                        initial_tile_count=result["initial_tile_count"],
                        initial_sigma=result["initial"]["sigma"],
                        split_count=result["split_count"],
                        cut_lines=[(c["axis"],c["coordinate"]) for c in result["cuts"]],
                        children_sigma=[c["metric"]["sigma"] for c in result["components"]],
                        stop_reason=result["stopping_reason"]))
            candidates.extend(dict(case=case,**c) for c in result["cut_candidates"])
            write_png(args.output/"figures"/(case+"_cut_map.png"),
                      tiles,result["components"],result["cuts"])
        except Exception as e:
            outputs[case]=dict(status="error",error=str(e),
                               traceback=traceback.format_exc())
            rows.append(dict(case=case,status="error",error=str(e)))
            print("ERROR",case,str(e),flush=True)
    if args.sensitivity:
        for n in (12,13,14,15,16):
            for ring in (2,3,5,8):
                try:
                    result,_=run_case(xyz,faces,"B6",cfg,n,args.max_splits,
                                      curvature_override=ring)
                    sensitivity.append(dict(grid=n,rings=ring,
                        split_count=result["split_count"],
                        stop_reason=result["stopping_reason"],
                        child_sigma=[c["metric"]["sigma"] for c in result["components"]]))
                except Exception as e:
                    sensitivity.append(dict(grid=n,rings=ring,error=str(e)))
    payload=dict(
        input_sha256=sha, configuration=asdict(cfg),
        warning="Diagnostic reflattened-tile metric; not author-certified CSF or fabrication",
        cases=outputs)
    (args.output/"run_results.json").write_text(
        json.dumps(payload,indent=2,allow_nan=False,default=str))
    write_csv(args.output/"ablation.csv",rows)
    write_csv(args.output/"cut_candidates.csv",candidates)
    write_csv(args.output/"sensitivity.csv",sensitivity)
    html="<html><meta charset='utf-8'><title>Split validation</title><body>"
    for case in outputs:
        html+="<h2>"+case+"</h2><img width='650' src='figures/"+case+"_cut_map.png'>"
        html+="<pre>"+json.dumps(rows[list(outputs).index(case)],indent=2)+"</pre>"
    html+="</body></html>"
    (args.output/"index.html").write_text(html,encoding="utf-8")
    print(json.dumps(rows,indent=2,ensure_ascii=False),flush=True)
    if any(r.get("status")=="error" for r in rows):
        return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
