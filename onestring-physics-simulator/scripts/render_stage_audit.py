#!/usr/bin/env python3
"""Phase 0: read-only stage audit (same fixed camera + run_id for every stage).

Usage:
  python scripts/render_stage_audit.py \
    --bundle artifacts/<run_id>/stages.npz \
    --out artifacts/<run_id>/

Requires numpy matplotlib imageio imageio-ffmpeg (for MP4). No changes to
the simulation or to the geometry being audited. All videos and PNGs
bear the exact run_id embedded in the same source bundle.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


VIEWS={
    "front":(16,-72),
    "side":(12,18),
    "back":(14,108),
    "ears":(32,-55),
}


def _triangles(v,faces):
    f=np.asarray(faces,int)
    if len(f)==0:return np.empty((0,3,3),float)
    if f.shape[1]==3:return np.asarray(v,float)[f]
    if f.shape[1]==4:
        return np.concatenate((
            np.asarray(v,float)[f[:,[0,1,2]]],
            np.asarray(v,float)[f[:,[0,2,3]]],
        ),axis=0)
    raise ValueError(f"Only triangles/quads accepted; got {f.shape}")


def _quad_area_ratios(current,reference,faces):
    f=np.asarray(faces,int)
    if len(f)==0 or f.shape[1]!=4:return None
    x=np.asarray(current,float)
    r=np.asarray(reference,float)
    if len(x)!=len(r):return None
    ref=_triangles(r,f)
    tri=_triangles(x,f)
    n0=np.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0])
    n1=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    denom=np.maximum(np.einsum("ij,ij->i",n0,n0),1e-30)
    signed=np.einsum("ij,ij->i",n1,n0)/denom
    # One quad contains 2 triangles; use minimum signed ratio.
    return np.minimum(signed[:len(f)],signed[len(f):])


def _mesh_for_stage(data,stage):
    if stage=="T3D":
        if "T3D_tiles" not in data:return None
        tiles=np.asarray(data["T3D_tiles"],float)
        # Render both exterior sheets as independent quads, no invented
        # interpolation across panels or seams.
        vertex=np.concatenate((tiles[:,:4].reshape(-1,3),
                               tiles[:,4:].reshape(-1,3)),axis=0)
        count=len(tiles)
        face=np.concatenate((
            np.arange(4*count).reshape(count,4),
            np.arange(4*count,8*count).reshape(count,4)[:,::-1],
        ))
        return vertex,face
    if stage+"_vertices" not in data:return None
    return np.asarray(data[stage+"_vertices"],float),np.asarray(
        data[stage+"_faces"],int
    )


def _make_plot(data,stage,view,azim=None):
    mesh=_mesh_for_stage(data,stage)
    if mesh is None:return None
    xyz,faces=mesh
    azim=VIEWS[view][1] if azim is None else azim
    elev=VIEWS[view][0]
    bounds=np.asarray(data["audit_fixed_bounds"],float)
    fig=plt.figure(figsize=(8.8,7.2),dpi=100,facecolor="#a8a8a8")
    ax=fig.add_subplot(111,projection="3d",computed_zorder=False)
    ax.set_facecolor("#a8a8a8")
    tris=_triangles(xyz,faces)
    if len(tris):
        normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
        # Approximate camera direction with a fixed azim/elevation; all
        # triangles are still drawn even if back-facing (culling OFF).
        elevrad=math.radians(elev)
        ar=math.radians(azim)
        eye=np.array([
            math.cos(elevrad)*math.cos(ar),
            math.cos(elevrad)*math.sin(ar),
            math.sin(elevrad),
        ])
        visible=np.einsum("ij,j->i",normals,eye)>=0.
        colors=np.tile(np.array([.07,.39,.76,.95]),(len(tris),1))
        colors[~visible]=[.71,.05,.50,.94]  # reverse face = MAGENTA
        poly=Poly3DCollection(
            tris,facecolors=colors,edgecolors="#17222e",
            linewidths=.18,alpha=.95,zsort="average",
        )
        ax.add_collection3d(poly)
    for i in range(3):
        pass
    ax.set_xlim(*bounds[0]);ax.set_ylim(*bounds[1]);ax.set_zlim(*bounds[2])
    if hasattr(ax,"set_box_aspect"):ax.set_box_aspect([1,1,1])
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x");ax.set_ylabel("y");ax.set_zlabel("z")
    ax.set_title(f"{RUN_ID} · {stage} · {view}",fontsize=11)
    fig.text(.03,.03,"Front: blue | Reverse: magenta | Flat + wireframe",
             fontsize=9,color="#181818")
    return fig


def _render(fig,path):
    fig.savefig(path,facecolor=fig.get_facecolor(),dpi=100)
    plt.close(fig)


def _overlay_seam_ribbon(data,out,stage):
    mesh=_mesh_for_stage(data,stage)
    if mesh is None or stage=="T3D":return
    xyz,_=mesh
    rows=SUMMARY.get("seam_affine_rows") or []
    if not rows:return
    bounds=np.asarray(data["audit_fixed_bounds"],float)
    fig=plt.figure(figsize=(8.8,7.2),dpi=100,facecolor="#a8a8a8")
    ax=fig.add_subplot(111,projection="3d")
    ax.set_facecolor("#a8a8a8")
    segments=[]
    lengths=[]
    for row in rows:
        if any(int(i)>=len(xyz) for i,w in row):
            continue
        pos=[(i,float(w)) for i,w in row if w>1e-12]
        neg=[(i,-float(w)) for i,w in row if w< -1e-12]
        if not pos or not neg:continue
        a=sum(w*xyz[int(i)] for i,w in pos)/sum(w for i,w in pos)
        b=sum(w*xyz[int(i)] for i,w in neg)/sum(w for i,w in neg)
        segments.append((a,b));lengths.append(float(np.linalg.norm(a-b)))
    if segments:
        maximum=max(max(lengths),1e-10)
        from matplotlib import cm,colors
        for (a,b),distance in zip(segments,lengths):
            ax.scatter(*a,color="#d62728",s=9)
            ax.scatter(*b,color="#185ccb",s=9)
            ax.plot(*np.stack([a,b]).T,color=cm.inferno(distance/maximum),
                    lw=1.3,alpha=.9)
        sm=cm.ScalarMappable(norm=colors.Normalize(0,maximum),cmap="inferno")
        fig.colorbar(sm,ax=ax,shrink=.53,label="Seam matching gap (3D units)")
    for i,fn in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):
        fn(*bounds[i])
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=VIEWS["front"][0],azim=VIEWS["front"][1])
    ax.set_title(f"{RUN_ID} · {stage} · physical Seam ribbon")
    _render(fig,out/f"{stage}_seam_ribbon.png")


def _degeneracy(data,out,stage):
    mesh=_mesh_for_stage(data,stage)
    ref=_mesh_for_stage(data,"M3D")
    if mesh is None or ref is None or stage not in ("K3D","M3D"):return
    v,f=mesh
    ratio=_quad_area_ratios(v,ref[0],f)
    if ratio is None:return
    SUMMARY.setdefault("render_audit",{})[stage]={
        "signed_area_ratio_min":float(np.min(ratio)),
        "signed_area_ratio_p5":float(np.percentile(ratio,5)),
        "signed_area_lt_0p30_count":int(np.count_nonzero(ratio<.30)),
        "inverted_quad_count":int(np.count_nonzero(ratio<=0.)),
    }
    base=_make_plot(data,stage,"front")
    if base is None:return
    plt.close(base)
    bounds=np.asarray(data["audit_fixed_bounds"],float)
    fig=plt.figure(figsize=(8.8,7.2),dpi=100,facecolor="#a8a8a8")
    ax=fig.add_subplot(111,projection="3d")
    ax.set_facecolor("#a8a8a8")
    quads=v[np.asarray(f,int)]
    colors=np.tile(np.array([.05,.40,.75,.90]),(len(f),1))
    colors[ratio<.30]=[.95,.10,.10,.99]
    colors[ratio<=0]=[.02,.02,.02,1]
    poly=Poly3DCollection(
        quads,facecolors=colors,edgecolors="#222222",linewidths=.3
    )
    ax.add_collection3d(poly)
    ax.set_xlim(*bounds[0]);ax.set_ylim(*bounds[1]);ax.set_zlim(*bounds[2])
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=16,azim=-72)
    ax.set_title(f"{RUN_ID} · {stage} · signed area vs M3D")
    fig.text(.04,.04,"Red: ratio < 0.30 | Black: inverted",
             fontsize=10)
    _render(fig,out/f"{stage}_degenerate_quads.png")


def _frames(data,stage,out,frames,fps):
    try:
        import imageio.v2 as imageio
    except ImportError:
        SUMMARY.setdefault("missing_artifacts",[]).append(
            f"{stage} MP4/GIF: imageio unavailable"
        )
        return
    import io
    outputs=[]
    for extension in ("mp4","gif"):
        try:
            kwargs=(
                {"fps":fps,"codec":"libx264","quality":6}
                if extension=="mp4" else {"fps":fps,"loop":0}
            )
            writer=imageio.get_writer(
                str(out/f"{stage}_turntable.{extension}"),**kwargs
            )
        except Exception as exc:
            SUMMARY.setdefault("missing_artifacts",[]).append(
                f"{stage} {extension}: {exc}"
            )
            continue
        try:
            for index in range(frames):
                fig=_make_plot(
                    data,stage,"front",azim=-72+360*index/frames,
                )
                if fig is None:break
                buffer=io.BytesIO()
                fig.savefig(buffer,format="png",dpi=100,
                            facecolor=fig.get_facecolor())
                plt.close(fig)
                buffer.seek(0)
                writer.append_data(imageio.imread(buffer))
            outputs.append(extension)
        except Exception as exc:
            SUMMARY.setdefault("missing_artifacts",[]).append(
                f"{stage} {extension} during render: {exc}"
            )
        finally:
            writer.close()
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--frames",type=int,default=300)
    parser.add_argument("--fps",type=int,default=30)
    parser.add_argument("--no-videos",action="store_true")
    opts=parser.parse_args()
    if opts.frames<1 or opts.fps<1:
        parser.error("frames and fps must be positive")
    opts.out.mkdir(parents=True,exist_ok=True)
    summary_file=opts.bundle.parent/"metrics.json"
    global RUN_ID,SUMMARY
    SUMMARY=json.loads(summary_file.read_text(encoding="utf-8"))
    RUN_ID=SUMMARY["run_id"]
    if opts.out.resolve()!=opts.bundle.parent.resolve():
        raise SystemExit("Audit output must be the source bundle run_id folder")
    with np.load(opts.bundle,allow_pickle=False) as bundle:
        data={k:bundle[k] for k in bundle.files}
    available=[]
    for stage in ("S","M3D","K3D","T3D"):
        if _mesh_for_stage(data,stage) is None:
            SUMMARY.setdefault("missing_artifacts",[]).append(
                f"{stage}: stage not present in current run"
            )
            continue
        available.append(stage)
        for view in VIEWS:
            fig=_make_plot(data,stage,view)
            _render(fig,opts.out/f"{stage}_{view}.png")
        _overlay_seam_ribbon(data,opts.out,stage)
        _degeneracy(data,opts.out,stage)
        if not opts.no_videos:
            _frames(data,stage,opts.out,opts.frames,opts.fps)
    SUMMARY["rendered_stages"]=available
    SUMMARY["render_protocol"]={
        "frames":opts.frames,"fps":opts.fps,
        "same_camera":True,"backface_culling":False,
        "backface_color":"magenta","wireframe":True,
        "fixed_bounds_source":"S",
        "source_distance_coverage_gate":"UNMEASURED",
        "chord_error_gate":"UNMEASURED",
        "video_view":"turntable",
    }
    (opts.out/"metrics.json").write_text(
        json.dumps(SUMMARY,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )
    print(f"[STAGE-AUDIT] {RUN_ID} rendered={available} "
          f"missing={SUMMARY.get('missing_artifacts',[])}")


if __name__=="__main__":
    main()
