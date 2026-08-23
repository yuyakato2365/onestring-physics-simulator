"""True 3--5 sided boundary panels for ``optcuts_test`` M2D.

The visible/geometric M2D keeps clipped boundary polygons exactly (simplified to
at most five vertices).  A separate four-point perimeter surrogate is retained
only because the legacy K3D/T3D numerical pipeline still assumes quads.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .quad_grid import create_quad_grid


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x,y = [float(v) for v in point[:2]]; poly=np.asarray(polygon,float); inside=False
    for i in range(len(poly)):
        x0,y0=poly[i]; x1,y1=poly[(i+1)%len(poly)]
        if (y0>y)!=(y1>y):
            den=float(y1-y0)
            if abs(den)>1e-20 and x < float(x0+(x1-x0)*(y-y0)/den): inside=not inside
    return inside


def _segment_intersection(a,b,c,d):
    a=np.asarray(a,float); b=np.asarray(b,float); c=np.asarray(c,float); d=np.asarray(d,float)
    r=b-a; s=d-c; cross=float(r[0]*s[1]-r[1]*s[0])
    if abs(cross)<=1e-12: return None
    q=c-a; t=float((q[0]*s[1]-q[1]*s[0])/cross); u=float((q[0]*r[1]-q[1]*r[0])/cross)
    if -1e-9<=t<=1+1e-9 and -1e-9<=u<=1+1e-9: return a+np.clip(t,0,1)*r
    return None


def _inside_cell(p, cell):
    lo=np.min(cell,axis=0)-1e-10; hi=np.max(cell,axis=0)+1e-10
    return bool(np.all(np.asarray(p,float)>=lo) and np.all(np.asarray(p,float)<=hi))


def _dedupe(points, tol=1e-9):
    out=[]
    for p in points:
        p=np.asarray(p,float)
        if not any(float(np.linalg.norm(p-q))<=tol for q in out): out.append(p)
    return out


def _polygon_area(poly):
    p=np.asarray(poly,float)
    if len(p)<3: return 0.0
    return 0.5*abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])))


def _intersection_polygon(cell, boundary):
    cell=np.asarray(cell,float); boundary=np.asarray(boundary,float)
    if len(boundary)>1 and np.linalg.norm(boundary[0]-boundary[-1])<1e-10: boundary=boundary[:-1]
    pts=[]
    for p in cell:
        if _point_in_polygon(p,boundary): pts.append(p)
    for p in boundary:
        if _inside_cell(p,cell): pts.append(p)
    ce=[(cell[i],cell[(i+1)%4]) for i in range(4)]
    be=[(boundary[i],boundary[(i+1)%len(boundary)]) for i in range(len(boundary))]
    for a,b in ce:
        for c,d in be:
            q=_segment_intersection(a,b,c,d)
            if q is not None: pts.append(q)
    pts=_dedupe(pts)
    if len(pts)<3: return np.zeros((0,2),float)
    arr=np.asarray(pts,float); center=np.mean(arr,axis=0)
    arr=arr[np.argsort(np.arctan2(arr[:,1]-center[1],arr[:,0]-center[0]))]
    return arr


def _simplify_to_five(poly):
    p=np.asarray(poly,float).copy()
    while len(p)>5:
        scores=[]
        for i in range(len(p)):
            a=p[(i-1)%len(p)]; b=p[i]; c=p[(i+1)%len(p)]
            ac=c-a; den=float(np.dot(ac,ac))
            t=0.0 if den<=1e-20 else float(np.clip(np.dot(b-a,ac)/den,0,1))
            scores.append(float(np.linalg.norm(b-(a+t*ac))))
        p=np.delete(p,int(np.argmin(scores)),axis=0)
    return p


def _perimeter_sample(poly, fractions=(0.0,0.25,0.5,0.75)):
    p=np.asarray(poly,float); closed=np.vstack([p,p[0]])
    seg=np.linalg.norm(np.diff(closed,axis=0),axis=1); cum=np.concatenate([[0.0],np.cumsum(seg)]); total=float(cum[-1])
    if total<=1e-14: return np.repeat(p[:1],4,axis=0)
    out=[]
    for f in fractions:
        s=float(f)*total; i=int(np.searchsorted(cum,s,side='right')-1); i=max(0,min(i,len(p)-1))
        span=float(cum[i+1]-cum[i]); t=0.0 if span<=1e-14 else (s-float(cum[i]))/span
        out.append((1-t)*closed[i]+t*closed[i+1])
    return np.asarray(out,float)


def install_optcuts_test_boundary_clip_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_optcuts_test_boundary_clip_installed",False): return
    base=pipeline._build_m2d
    base_lift=pipeline._lift_m2d_to_m3d

    def build_m2d(grid: Any, domain: Any, params: Any=None):
        if not bool(getattr(domain,"_optcuts_test_clip_boundary",False)): return base(grid,domain,params)
        nx=int(getattr(domain,"overlay_nx",grid.nx)); ny=int(getattr(domain,"overlay_ny",grid.ny))
        overlay=create_quad_grid(nx,ny,grid.tile_size,grid.gap_size)
        vertices=[np.asarray([p[0],p[1],0.0],float) for p in np.asarray(domain.uv_vertices,float)]
        original=np.asarray(vertices,float); boundary=np.asarray(domain.boundary,float)
        if len(boundary)>1 and np.linalg.norm(boundary[0]-boundary[-1])<1e-10: boundary=boundary[:-1]
        vertex_map={tuple(np.round(v[:2],10)):i for i,v in enumerate(vertices)}
        def get_id(p):
            key=tuple(np.round(np.asarray(p,float)[:2],10))
            if key in vertex_map: return vertex_map[key]
            idx=len(vertices); vertex_map[key]=idx; vertices.append(np.asarray([p[0],p[1],0.0],float)); return idx

        surrogate_faces=[]; polygon_faces=[]; full=clipped=removed=simplified=0; degree_counts={3:0,4:0,5:0}
        for tile in overlay.tiles or []:
            ids=[int(v) for v in tile.vertex_ids]; cell=original[np.asarray(ids,int),:2]
            if all(_point_in_polygon(p,boundary) for p in cell):
                surrogate_faces.append(tuple(ids)); polygon_faces.append(list(ids)); full+=1; degree_counts[4]+=1; continue
            poly=_intersection_polygon(cell,boundary)
            if len(poly)<3 or _polygon_area(poly)<=1e-12: removed+=1; continue
            if len(poly)>5: poly=_simplify_to_five(poly); simplified+=1
            if len(poly)<3 or len(poly)>5: removed+=1; continue
            true_ids=[get_id(p) for p in poly]; polygon_faces.append(true_ids); degree_counts[len(true_ids)]=degree_counts.get(len(true_ids),0)+1
            q=_perimeter_sample(poly); qids=[get_id(p) for p in q]; surrogate_faces.append(tuple(qids)); clipped+=1

        if not surrogate_faces: raise RuntimeError("OPTCUTS_TEST_M2D_CLIP_EMPTY")
        verts=np.asarray(vertices,float); faces=np.asarray(surrogate_faces,int)
        cls=getattr(getattr(pipeline,"_original",None),"QuadMesh",None)
        if cls is None: cls=type(base(grid,domain,params))
        metrics={
            "m2d_grid_overlay":"regular grid clipped against smooth Omega; true boundary polygons stored",
            "m2d_crop_policy":"polygon_clip_3_to_5",
            "m2d_full_panel_count":int(full),"m2d_boundary_clipped_panel_count":int(clipped),"m2d_removed_fully_outside_panel_count":int(removed),
            "m2d_boundary_polygon_simplified_to_five_count":int(simplified),
            "m2d_boundary_triangle_count":int(degree_counts.get(3,0)),"m2d_boundary_quad_count":int(degree_counts.get(4,0)),"m2d_boundary_pentagon_count":int(degree_counts.get(5,0)),
            "m2d_true_polygon_geometry":True,"m2d_legacy_quad_surrogate_for_downstream":True,
            "number_of_splits":len(getattr(domain,"split_lines",[]) or []),"split_locations":list(getattr(domain,"split_lines",[]) or []),
        }
        out=cls(verts,faces,overlay,"M2D",metrics,list(getattr(domain,"split_lines",[]) or []))
        setattr(out,"_polygon_faces",[list(map(int,f)) for f in polygon_faces]); setattr(out,"_optcuts_test_boundary_clipped",True)
        print(f"[OPTCUTS-TEST-M2D] full={full} clipped={clipped} outside_removed={removed} deg3={degree_counts.get(3,0)} deg4={degree_counts.get(4,0)} deg5={degree_counts.get(5,0)} simplified_gt5={simplified}")
        return out

    def lift(target: Any, mesh: Any, parameterization: Any, params: Any):
        lifted, report=base_lift(target,mesh,parameterization,params)
        if hasattr(mesh,"_polygon_faces"):
            setattr(lifted,"_polygon_faces",[list(f) for f in getattr(mesh,"_polygon_faces")])
            setattr(lifted,"_optcuts_test_boundary_clipped",True)
        return lifted,report

    pipeline._build_m2d=build_m2d; pipeline._lift_m2d_to_m3d=lift
    original=getattr(pipeline,"_original",None)
    if original is not None:
        original._build_m2d=build_m2d; original._lift_m2d_to_m3d=lift
    for fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict): glb["_build_m2d"]=build_m2d; glb["_lift_m2d_to_m3d"]=lift
    pipeline._onestring_optcuts_test_boundary_clip_installed=True


__all__=["install_optcuts_test_boundary_clip_m2d_patch"]
