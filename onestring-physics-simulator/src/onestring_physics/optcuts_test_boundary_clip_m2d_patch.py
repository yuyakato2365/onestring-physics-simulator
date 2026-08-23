"""Conservative boundary cutting for ``optcuts_test`` M2D.

Every generated panel must lie inside Omega.  Visible panels are triangles or
quads only.  Per-face provenance and M2D coordinates are retained through the
M2D -> M3D lift so validity failures can be traced to their exact source.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .quad_grid import create_quad_grid


def _point_in_polygon(point, polygon):
    p=np.asarray(point,float)[:2]; x,y=float(p[0]),float(p[1]); poly=np.asarray(polygon,float)
    for i in range(len(poly)):
        a=poly[i]; b=poly[(i+1)%len(poly)]; ab=b-a; den=float(np.dot(ab,ab))
        if den>1e-24:
            t=float(np.clip(np.dot(p-a,ab)/den,0,1))
            if float(np.linalg.norm(p-(a+t*ab)))<=1e-9: return True
    inside=False
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


def _dedupe(points,tol=1e-9):
    out=[]
    for p in points:
        q=np.asarray(p,float)
        if not any(float(np.linalg.norm(q-r))<=tol for r in out): out.append(q)
    return out


def _polygon_area(poly):
    p=np.asarray(poly,float)
    if len(p)<3: return 0.0
    return 0.5*abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])))


def _convex_hull(points):
    pts=sorted({(float(p[0]),float(p[1])) for p in np.asarray(points,float)})
    if len(pts)<=2: return np.asarray(pts,float)
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[]
    for p in pts:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=1e-14: lo.pop()
        lo.append(p)
    hi=[]
    for p in reversed(pts):
        while len(hi)>=2 and cross(hi[-2],hi[-1],p)<=1e-14: hi.pop()
        hi.append(p)
    return np.asarray(lo[:-1]+hi[:-1],float)


def _boundary_cell_intersections(cell,boundary):
    hits=[]; cell=np.asarray(cell,float); boundary=np.asarray(boundary,float)
    for i in range(4):
        for j in range(len(boundary)):
            q=_segment_intersection(cell[i],cell[(i+1)%4],boundary[j],boundary[(j+1)%len(boundary)])
            if q is not None: hits.append(q)
    return _dedupe(hits)


def _farthest_pair(points):
    pts=[np.asarray(p,float) for p in points]
    if len(pts)<2: return None
    best=None; dist=-1.0
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            d=float(np.linalg.norm(pts[i]-pts[j]))
            if d>dist: best=(pts[i],pts[j]); dist=d
    return best


def _panel_inside_omega(poly,boundary,samples_per_edge=13):
    p=np.asarray(poly,float)
    if len(p)<3 or _polygon_area(p)<=1e-12: return False
    for v in p:
        if not _point_in_polygon(v,boundary): return False
    for i in range(len(p)):
        a=p[i]; b=p[(i+1)%len(p)]
        for t in np.linspace(0,1,max(3,int(samples_per_edge))):
            if not _point_in_polygon((1-t)*a+t*b,boundary): return False
    center=np.mean(p,axis=0)
    if not _point_in_polygon(center,boundary): return False
    for v in p:
        for t in (0.2,0.4,0.6,0.8):
            if not _point_in_polygon((1-t)*center+t*v,boundary): return False
    return True


def _simple_cut_polygon(cell,boundary):
    cell=np.asarray(cell,float)
    inside=[np.asarray(p,float) for p in cell if _point_in_polygon(p,boundary)]
    pair=_farthest_pair(_boundary_cell_intersections(cell,boundary))
    if pair is None:
        hull=_convex_hull(inside) if len(inside)>=3 else np.zeros((0,2),float)
        return hull if _panel_inside_omega(hull,boundary) else np.zeros((0,2),float)
    if inside: anchor=np.mean(np.asarray(inside,float),axis=0)
    else:
        anchor=0.5*(pair[0]+pair[1])
        if not _point_in_polygon(anchor,boundary): return np.zeros((0,2),float)
    for alpha in (1.0,.98,.95,.90,.82,.72,.60,.45,.30,.15,.08):
        q0=anchor+alpha*(pair[0]-anchor); q1=anchor+alpha*(pair[1]-anchor)
        hull=_convex_hull(_dedupe(inside+[q0,q1]))
        if len(hull)>=3 and _panel_inside_omega(hull,boundary): return hull
    return np.zeros((0,2),float)


def _split_to_max_four(poly,boundary):
    p=np.asarray(poly,float)
    if len(p)<=4: return [p] if _panel_inside_omega(p,boundary) else []
    if len(p)==5:
        options=[(p[[0,1,2]],p[[0,2,3,4]]),(p[[1,2,3]],p[[1,3,4,0]]),(p[[2,3,4]],p[[2,4,0,1]])]
        for pieces in options:
            if all(_panel_inside_omega(x,boundary) for x in pieces): return list(pieces)
        return []
    pieces=[p[[0,i,i+1]] for i in range(1,len(p)-1)]
    return pieces if all(_panel_inside_omega(x,boundary) for x in pieces) else []


def _triangle_as_quad(poly):
    p=np.asarray(poly,float); lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1); e=int(np.argmax(lengths))
    mid=.5*(p[e]+p[(e+1)%3]); out=[]
    for i in range(3):
        out.append(p[i])
        if i==e: out.append(mid)
    return np.asarray(out,float)


def install_optcuts_test_boundary_clip_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline,"_onestring_optcuts_test_boundary_clip_installed",False): return
    base=pipeline._build_m2d; base_lift=pipeline._lift_m2d_to_m3d

    def build_m2d(grid,domain,params=None):
        if not bool(getattr(domain,"_optcuts_test_clip_boundary",False)): return base(grid,domain,params)
        nx=int(getattr(domain,"overlay_nx",grid.nx)); ny=int(getattr(domain,"overlay_ny",grid.ny))
        overlay=create_quad_grid(nx,ny,grid.tile_size,grid.gap_size)
        vertices=[np.asarray([p[0],p[1],0.0],float) for p in np.asarray(domain.uv_vertices,float)]
        original=np.asarray(vertices,float); boundary=np.asarray(domain.boundary,float)
        if len(boundary)>1 and np.linalg.norm(boundary[0]-boundary[-1])<1e-10: boundary=boundary[:-1]
        vmap={tuple(np.round(v[:2],10)):i for i,v in enumerate(vertices)}
        def get_id(p):
            key=tuple(np.round(np.asarray(p,float)[:2],10))
            if key in vmap: return vmap[key]
            idx=len(vertices); vmap[key]=idx; vertices.append(np.asarray([p[0],p[1],0.0],float)); return idx

        surrogate=[]; polygons=[]; sources=[]; uv_faces=[]
        full=clipped=removed=split5=tri=quad=rejected=0
        for tile in overlay.tiles or []:
            ids=[int(v) for v in tile.vertex_ids]; cell=original[np.asarray(ids,int),:2]
            if _panel_inside_omega(cell,boundary):
                surrogate.append(tuple(ids)); polygons.append(list(ids)); sources.append("full_grid_quad"); uv_faces.append(cell.copy())
                full+=1; quad+=1; continue
            cut=_simple_cut_polygon(cell,boundary)
            if len(cut)<3:
                removed+=1; rejected+=1; continue
            pieces=_split_to_max_four(cut,boundary)
            if not pieces:
                removed+=1; rejected+=1; continue
            is_split5=len(cut)==5
            if is_split5: split5+=1
            clipped+=1
            for piece in pieces:
                true_ids=[get_id(p) for p in piece]; polygons.append(true_ids)
                if len(piece)==3:
                    tri+=1; q=_triangle_as_quad(piece); surrogate.append(tuple(get_id(p) for p in q)); uv_faces.append(q.copy())
                    sources.append("boundary_pentagon_split_triangle" if is_split5 else "boundary_triangle")
                elif len(piece)==4:
                    quad+=1; surrogate.append(tuple(true_ids)); uv_faces.append(np.asarray(piece,float).copy())
                    sources.append("boundary_pentagon_split_quad" if is_split5 else "boundary_quad")

        if not surrogate: raise RuntimeError("OPTCUTS_TEST_M2D_CLIP_EMPTY")
        verts=np.asarray(vertices,float); faces=np.asarray(surrogate,int)
        cls=getattr(getattr(pipeline,"_original",None),"QuadMesh",None) or type(base(grid,domain,params))
        metrics={"m2d_crop_policy":"strict_inside_omega_max4","m2d_full_cell_count":full,"m2d_boundary_clipped_cell_count":clipped,
                 "m2d_removed_cell_count":removed,"m2d_boundary_pentagon_split_count":split5,"m2d_triangle_panel_count":tri,
                 "m2d_quad_panel_count":quad,"m2d_rejected_outside_omega_count":rejected,"m2d_max_visible_panel_degree":4,
                 "m2d_panel_subset_of_omega_enforced":True,"m2d_true_polygon_geometry":True,"m2d_legacy_quad_surrogate_for_downstream":True,
                 "number_of_splits":len(getattr(domain,"split_lines",[]) or []),"split_locations":list(getattr(domain,"split_lines",[]) or [])}
        out=cls(verts,faces,overlay,"M2D",metrics,list(getattr(domain,"split_lines",[]) or []))
        setattr(out,"_polygon_faces",[list(map(int,f)) for f in polygons]); setattr(out,"_optcuts_test_boundary_clipped",True)
        setattr(out,"_optcuts_test_face_sources",list(sources)); setattr(out,"_optcuts_test_face_uv",[np.asarray(x,float) for x in uv_faces])
        print(f"[OPTCUTS-TEST-M2D] full={full} clipped={clipped} removed={removed} rejected_outside={rejected} triangles={tri} quads={quad} pentagons_split={split5} subset_of_omega=True")
        return out

    def lift(target,mesh,parameterization,params):
        lifted,report=base_lift(target,mesh,parameterization,params)
        for attr in ("_polygon_faces","_optcuts_test_boundary_clipped","_optcuts_test_face_sources","_optcuts_test_face_uv"):
            if hasattr(mesh,attr): setattr(lifted,attr,getattr(mesh,attr))
        return lifted,report

    pipeline._build_m2d=build_m2d; pipeline._lift_m2d_to_m3d=lift
    original=getattr(pipeline,"_original",None)
    if original is not None: original._build_m2d=build_m2d; original._lift_m2d_to_m3d=lift
    for fn in (getattr(pipeline,"build_onestring_design",None),getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict): glb["_build_m2d"]=build_m2d; glb["_lift_m2d_to_m3d"]=lift
    pipeline._onestring_optcuts_test_boundary_clip_installed=True


__all__=["install_optcuts_test_boundary_clip_m2d_patch"]
