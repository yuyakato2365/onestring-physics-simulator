#!/usr/bin/env python3
"""Falsifiable GENERIC post-cut re-BFF experiment, NOT a reproduction claim.

For a source disk S with a known official BFF chart c:S->Omega, clip the
triangles along one line in Omega (duplicating vertices across a seam).
Apply official BFF independently to each now-disk patch. Contrast
full-patch area-Jacobian spans with fixed original chart.

The OneString authors did NOT specify that this procedure occurs in
their Sec.4.5. It is a controlled research hypothesis and does not
change the production Split algorithm, grid, or cut acceptance.

Usage:
 python scripts/test_lilium_postcut_bff_hypothesis.py ~/Downloads/lilium.obj \
     --original-line row:0.378065792089
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from onestring_physics.input_shape import load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, build_paper_reference_initialization,
)
from onestring_physics.paper_fidelity_mode import configure_paper_fidelity
from onestring_physics.reference_bff import (
    run_official_bff, validate_reference_mesh,
)


def clip_piece(xyz, uv, triangles, axis, value, positive):
    """Clip connected triangular disk with exact seam-edge reuse."""
    values=np.asarray(uv)[:,axis]
    coords=np.asarray(xyz,float)
    mapped=np.asarray(uv,float)
    by_key={}
    out_xyz=[]
    out_uv=[]
    out_faces=[]
    def remember(v):
        k=v[0]
        if k not in by_key:
            by_key[k]=len(out_xyz)
            out_xyz.append(v[1])
            out_uv.append(v[2])
        return by_key[k]
    def keep(v):
        d=float(v[2][axis]-value)
        return d>=-1e-11 if positive else d<=1e-11
    def cut(a,b):
        da=float(a[2][axis]-value)
        db=float(b[2][axis]-value)
        t=-da/(db-da)
        if t<1e-12: return a
        if t>1-1e-12: return b
        ka,kb=a[0],b[0]
        if ka[0]=="v" and kb[0]=="v":
            k=("e",min(ka[1],kb[1]),max(ka[1],kb[1]))
        else:
            raise ValueError("Clipping should intersect original triangle edges")
        return (k,a[1]*(1-t)+b[1]*t,a[2]*(1-t)+b[2]*t)
    for ids in np.asarray(triangles,int):
        poly=[(("v",int(i)),coords[int(i)],mapped[int(i)]) for i in ids]
        result=[]
        for i in range(3):
            a,b=poly[i],poly[(i+1)%3]
            ia,ib=keep(a),keep(b)
            if ia and ib: result.append(b)
            elif ia and not ib: result.append(cut(a,b))
            elif not ia and ib:
                result.append(cut(a,b))
                result.append(b)
        # Deduplicate on-cut original vertices to avoid zero-area wedges.
        dedup=[]
        for v in result:
            if not dedup or dedup[-1][0]!=v[0]:
                dedup.append(v)
        if len(dedup)>1 and dedup[0][0]==dedup[-1][0]: dedup.pop()
        if len(dedup)<3: continue
        new=[remember(v) for v in dedup]
        for k in range(1,len(new)-1):
            tri=[new[0],new[k],new[k+1]]
            pt=np.asarray([out_uv[t] for t in tri])
            a,b=pt[1]-pt[0],pt[2]-pt[0]
            area=abs(a[0]*b[1]-a[1]*b[0])
            if area>1e-16: out_faces.append(tri)
    xyz_new=np.asarray(out_xyz,float)
    uv_new=np.asarray(out_uv,float)
    face_new=np.asarray(out_faces,int)
    topo=validate_reference_mesh(xyz_new,face_new)
    return xyz_new,uv_new,face_new,topo


def field(xyz,uv,faces):
    xyztri=np.asarray(xyz)[faces]
    uvtri=np.asarray(uv)[faces]
    area3=np.linalg.norm(np.cross(xyztri[:,1]-xyztri[:,0],
                                  xyztri[:,2]-xyztri[:,0]),axis=1)
    du=uvtri[:,1]-uvtri[:,0]
    dv=uvtri[:,2]-uvtri[:,0]
    area2=np.abs(du[:,0]*dv[:,1]-du[:,1]*dv[:,0])
    good=(area2>1e-16)&(area3>1e-16)
    jac=area3[good]/area2[good]
    if not len(jac): raise ValueError("No valid Jacobian triangles")
    return dict(
        faces=int(len(faces)),
        linear_span=float(np.sqrt(jac.max()/jac.min())),
        area_span=float(jac.max()/jac.min()),
        jac_min=float(jac.min()),
        jac_max=float(jac.max()),
        linear_span_pct1_99=float(np.sqrt(np.percentile(jac,99)/np.percentile(jac,1))),
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mesh",type=Path)
    p.add_argument("--grid",type=int,default=14)
    p.add_argument("--lines",nargs="+",default=["row:0.378065792089","row:0.084014620464"])
    p.add_argument("--all-initial-grid-cuts",action="store_true",
                   help="Test every complete 1-cut candidate from the actual generic M2D audit; no forced split count.")
    p.add_argument("--free-only",action="store_true",
                   help="Evaluate only standard, non-rectangle official BFF after splitting.")
    p.add_argument("--bff-executable",type=Path,required=True)
    p.add_argument("--report",type=Path,default=Path("output/lilium_postcut_bff_hypothesis.json"))
    args=p.parse_args()
    params=PipelineParameters(nx=args.grid,ny=args.grid,
        omega_parameterization_mode="paper_reference_bff",
        bff_boundary_policy="target_rectangle",
        paper_fidelity_mode=True)
    params,_=configure_paper_fidelity(params)
    state=build_paper_reference_initialization(load_target_shape(args.mesh),params)
    param=state.surface_parameterization
    xyz=np.asarray(param.surface_vertices_3d,float)
    uv=np.asarray(param.uv_vertices_2d,float)
    faces=np.asarray(param.surface_faces,int)
    data=dict(mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
              source_triangle_count=int(len(faces)),
              initial_chart=field(xyz,uv,faces),
              original_bff_boundary_policy="target_rectangle",
              tested_lines={},
              caution="This is a research hypothesis: Sec 4.5 does not state "
                      "post-cut BFF is run. Full source-patch extrema are NOT "
                      "the same as retained-grid M2D CSF.")
    labels=list(args.lines)
    if args.all_initial_grid_cuts:
        audit=state.mesh_2d_initial.metrics.get("csf_split_initial_one_cut_exhaustive_audit") or {}
        candidates=audit.get("candidates") or []
        if not candidates:
            raise RuntimeError("No initial-grid full cuts available for exhaustive audit")
        labels=[f"{c['axis']}:{c['coordinate']:.12g}" for c in candidates]
        data["candidate_provenance"]="All complete cuts generated by the original initial-grid generic audit"
        data["candidate_count"]=len(labels)
    data["all_cut_summary"]=[]
    for label in labels:
        axis,val=label.split(":")
        dim={"col":0,"row":1}[axis]
        val=float(val)
        halves=[]
        policies=(["automatic_reference"] if args.free_only else
                  ["target_rectangle","automatic_reference"])
        try:
            for positive in [False,True]:
                xyz2,uv2,faces2,topo=clip_piece(xyz,uv,faces,dim,val,positive)
                entry=dict(side=("positive" if positive else "negative"),
                           num_surface_triangles=len(faces2),
                           disk=bool(topo["disk_topology"]),
                           source_fixed_chart=field(xyz2,uv2,faces2),
                           reflattened={})
                for policy in policies:
                    res=run_official_bff(
                        xyz2,faces2,executable=args.bff_executable,
                        boundary_policy=policy,
                    )
                    entry["reflattened"][policy]=field(xyz2,res.uv_vertices,faces2)
                halves.append(entry)
        except (ValueError,RuntimeError) as exc:
            data["tested_lines"][label]={"error":str(exc)}
            data["all_cut_summary"].append({"label":label,"error":str(exc)})
            print("POSTCUT FAILED",label,str(exc),flush=True)
            if not args.all_initial_grid_cuts:
                raise
            continue
        data["tested_lines"][label]=halves
        worst_area=max(x["reflattened"]["automatic_reference"]["area_span"] for x in halves)
        worst_linear=max(x["reflattened"]["automatic_reference"]["linear_span"] for x in halves)
        data["all_cut_summary"].append({
            "label":label,
            "worst_area_span_after_free_bff":worst_area,
            "worst_linear_span_after_free_bff":worst_linear,
            "area_limit_two_satisfied":bool(worst_area<=2.+1e-10),
            "linear_limit_two_satisfied":bool(worst_linear<=2.+1e-10),
        })
        print("POSTCUT",label,"fixed/free scale spans",
              [(x["source_fixed_chart"]["linear_span"],
                x["reflattened"]["automatic_reference"]["linear_span"]) for x in halves],
              "area bound satisfied",worst_area<=2.,flush=True)
    if args.all_initial_grid_cuts:
        vals=[x for x in data["all_cut_summary"]
              if "worst_area_span_after_free_bff" in x]
        sorted_vals=sorted(vals,key=lambda d:d["worst_area_span_after_free_bff"])
        data["best_free_bff_cut_by_area_span"]=sorted_vals[0] if sorted_vals else None
        data["all_candidates_area_limit_two_count"]=sum(x["area_limit_two_satisfied"] for x in vals)
        data["all_candidates_linear_limit_two_count"]=sum(x["linear_limit_two_satisfied"] for x in vals)
        print("ALL FULL INITIAL GRID CUTS:",len(labels),
              "successful",len(vals),
              "area limit 2 pass",data["all_candidates_area_limit_two_count"],
              "linear limit 2 pass",data["all_candidates_linear_limit_two_count"],
              "best",data["best_free_bff_cut_by_area_span"],flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(data,indent=2,default=str))
    print("wrote",args.report.resolve(),flush=True)


if __name__=="__main__":
    main()
