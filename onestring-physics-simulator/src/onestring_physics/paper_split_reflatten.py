"""Experimental Sec.4.5 reflatten/area-span Split reconstruction.

Adapted from work_branch_reflatten_split.patch (2026-10-20).
Not the OneString authors' verified implementation. Source 3D patches are
reflattened with OFFICIAL free-boundary BFF for a tile-area-span decision.
The final fabrication tiles retain their ORIGINAL Omega/grid coordinates.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

REFLATTEN_CRITERION = "reflatten_tile_area"
_CSF_MODEL = ("Reconstructed Sec.4.5: free-boundary official BFF per source "
              "component; max(A3D/A2D)/min(A3D/A2D) <= 2 (tile areas)")


def _tri_angles(V, F):
    P = np.asarray(V, float)[np.asarray(F, int)]
    out = np.zeros((len(F), 3))
    for i in range(3):
        a = P[:, (i+1)%3] - P[:, i]
        b = P[:, (i+2)%3] - P[:, i]
        d = np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1), 1e-300)
        out[:, i] = np.arccos(np.clip(np.einsum("ij,ij->i",a,b)/d,-1.,1.))
    return out


def _tri_areas3(V,F):
    P = np.asarray(V, float)[np.asarray(F, int)]
    return .5*np.linalg.norm(np.cross(P[:,1]-P[:,0],P[:,2]-P[:,0]),axis=1)


def _boundary_loop(F):
    F=np.asarray(F,int)
    counts=defaultdict(int)
    for tri in F:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            counts[(min(a,b),max(a,b))]+=1
    following={}
    for tri in F:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            if counts[(min(a,b),max(a,b))]==1:
                following[int(a)]=int(b)
    if not following:
        raise ValueError("source surface is closed: boundary BFF requires an open disk")
    loops=[];seen=set()
    for start in following:
        if start in seen:
            continue
        loop=[start]; seen.add(start); cur=following[start]
        while cur != start:
            if cur in seen or cur not in following:
                raise ValueError("source boundary is not a simple oriented cycle")
            loop.append(cur);seen.add(cur);cur=following[cur]
        loops.append(loop)
    return max(loops,key=len)


def lobe_tip_corner_positions(vertices, faces, window=9, snap_to_corners=False):
    """Four well-spaced boundary corners for the rectangular BFF.

    Default: the four well-spaced extrema of smoothed geodesic turning (Lilium:
    the petal tips). With ``snap_to_corners`` (Paper-oriented mode), boundaries
    with four genuine corners use those vertices and flat maxima use their
    centre, so straight-edged surfaces map to a valid rectangle. Lilium's four
    corners are identical either way.
    """
    V=np.asarray(vertices,float);F=np.asarray(faces,int)[:,:3]
    loop=_boundary_loop(F)
    n=len(loop)
    if n<16: raise ValueError("boundary too short for lobe-tip selection")
    angsum=np.bincount(F.ravel(),weights=_tri_angles(V,F).ravel(),minlength=len(V))
    turning=np.pi-angsum[loop]
    smoothed=np.array([turning[np.arange(i-window,i+window+1)%n].sum()
                       for i in range(n)])
    chosen=[]
    for index in np.argsort(-smoothed,kind="stable"):
        i=int(index)
        if all(min(abs(i-j),n-abs(i-j)) > n//8 for j in chosen):
            chosen.append(i)
        if len(chosen)==4: break
    if len(chosen)!=4: raise ValueError("unable to locate four separated BFF corners")
    if snap_to_corners:
        sharp=_sharp_corner_picks(turning,smoothed,n)
        chosen=sharp if sharp is not None else _refine_corner_picks(chosen,turning,smoothed,window)
    return V[[loop[i] for i in sorted(chosen)]].copy()


def _sharp_corner_picks(turning,smoothed,n):
    """Four genuine boundary corners, if the boundary has them.

    A corner is a single vertex whose geodesic turning is at least pi/8 and is
    not cancelled by its neighbours (a jagged boundary alternates +/- turning,
    so its window sum stays small). Height fields over a square have four such
    vertices (pi/2 on a flat sheet, about 0.65 rad on a saddle) while the
    largest smoothed turning of a saddle lies at its edge midpoints. Lilium's
    largest single-vertex turning is 0.25 rad, so it keeps the lobe tips.
    """
    picks=[]
    for index in np.argsort(-turning,kind="stable"):
        i=int(index)
        if turning[i]<np.pi/8: break
        if smoothed[i]<0.5*turning[i]: continue
        if all(min(abs(i-j),n-abs(i-j)) > n//8 for j in picks):
            picks.append(i)
        if len(picks)==4: return picks
    return None


def _refine_corner_picks(chosen,turning,smoothed,window):
    """Put each pick on the actual corner when the boundary HAS a corner there.

    Straight-edged boundaries (height fields over a square, polygons) concentrate
    the turning in single vertices, so every window that contains the corner has
    the same smoothed sum. The stable sort then returns the FIRST vertex of that
    plateau, ``window`` vertices before the corner: the rectangle corner lands on
    a straight edge while the real 90-degree corner must be flattened, which
    collapses its UV triangles (BFF output rejected as degenerate). Two cases are
    moved; anything else is left exactly as before (Lilium: raw turning <= 0.25
    rad and distinct smoothed maxima, so its four corners are unchanged).

    1. a vertex with turning >= pi/4 inside the window (a genuine sharp corner)
       -> that vertex;
    2. a flat maximum of the smoothed sum -> the centre of that plateau.
    """
    n=len(turning);refined=[]
    for i in chosen:
        span=np.arange(i-window,i+window+1)%n
        sharp=int(span[int(np.argmax(turning[span]))])
        if turning[sharp]>=np.pi/4:
            refined.append(sharp);continue
        tol=1e-9*max(1.0,abs(float(smoothed[i])))
        lo=hi=0
        while lo<n and abs(smoothed[(i-lo-1)%n]-smoothed[i])<=tol: lo+=1
        while hi<n and abs(smoothed[(i+hi+1)%n]-smoothed[i])<=tol: hi+=1
        refined.append(int((i+(hi-lo)//2)%n) if lo+hi<n else i)
    if len(set(refined))!=len(refined):
        return chosen
    return refined


def _smoothed_gaussian_curvature(V,F,rings):
    V=np.asarray(V,float);F=np.asarray(F,int)
    n=len(V);areas=_tri_areas3(V,F)
    angle_sum=np.bincount(F.ravel(),weights=_tri_angles(V,F).ravel(),minlength=n)
    vertex_areas=np.bincount(F.ravel(),weights=np.repeat(areas/3,3),minlength=n)
    boundary=np.zeros(n,bool);boundary[_boundary_loop(F)]=True
    k=np.where(boundary,0.,(2*np.pi-angle_sum)/np.maximum(vertex_areas,1e-300))
    if rings<=1:return k
    I=np.concatenate([F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0]])
    J=np.concatenate([F[:,1],F[:,2],F[:,0],F[:,0],F[:,1],F[:,2]])
    adj=sparse.csr_matrix((np.ones(len(I)),(I,J)),shape=(n,n))+sparse.identity(n)
    adj.data[:]=1.;reach=adj.copy()
    for _ in range(rings-1):
        reach=reach@adj;reach.data[:]=1.
    return np.asarray((reach@(k*vertex_areas))/np.maximum(reach@vertex_areas,1e-300)).ravel()


def _locate(uv,faces,points):
    """Barycentric lookup in old cut chart (KD-tree broad phase)."""
    from scipy.spatial import cKDTree
    UV=np.asarray(uv,float);F=np.asarray(faces,int);P=np.asarray(points,float)
    coords=UV[F]
    tree=cKDTree(coords.mean(axis=1))
    tri=np.full(len(P),-1,dtype=int);bary=np.full((len(P),3),np.nan)
    remaining=np.arange(len(P));k=min(12,len(F))
    if not k:raise ValueError("empty source triangulation")
    while remaining.size:
        _,ids=tree.query(P[remaining],k=k)
        ids=np.asarray(ids).reshape(len(remaining),-1)
        a,b,c=coords[ids,0],coords[ids,1],coords[ids,2]
        u=b-a;v=c-a;w=P[remaining,None,:]-a
        d00=(u*u).sum(-1);d01=(u*v).sum(-1);d11=(v*v).sum(-1)
        d20=(w*u).sum(-1);d21=(w*v).sum(-1)
        den=d00*d11-d01*d01
        with np.errstate(divide="ignore",invalid="ignore"):
            l1=(d11*d20-d01*d21)/den;l2=(d00*d21-d01*d20)/den
        lam=np.stack([1-l1-l2,l1,l2],axis=-1)
        inside=np.isfinite(lam).all(-1)&(lam>=-1e-10).all(-1)
        found=inside.any(axis=1)
        rows=np.flatnonzero(found)
        first=inside[rows].argmax(axis=1)
        tri[remaining[rows]]=ids[rows,first];bary[remaining[rows]]=lam[rows,first]
        remaining=remaining[~found]
        if k>=len(F):break
        k=min(len(F),k*4)
    return tri,bary


def _face_components(F):
    first={};rows=[];cols=[]
    for index,tri in enumerate(F):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            key=(min(a,b),max(a,b))
            if key in first:
                rows.append(first[key]);cols.append(index)
            else:first[key]=index
    graph=sparse.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(F),len(F)))
    count,labels=connected_components(graph,directed=False)
    return [np.flatnonzero(labels==i) for i in range(count)]


def _clip(V,F,UV,coord,value):
    s=UV[:,coord]-value
    s=np.where(np.abs(s)<1e-12,1e-12,s)
    Vnew=list(V);UVnew=list(UV);cache={}
    def crossing(a,b):
        key=(min(a,b),max(a,b))
        if key not in cache:
            t=s[a]/(s[a]-s[b])
            Vnew.append(V[a]+t*(V[b]-V[a]))
            UVnew.append(UV[a]+t*(UV[b]-UV[a]))
            cache[key]=len(Vnew)-1
        return cache[key]
    minus=[];plus=[]
    for tri in F:
        signs=s[tri]>0
        if signs.all():plus.append(tri);continue
        if not signs.any():minus.append(tri);continue
        lone=[i for i in range(3)
              if signs[i]!=signs[(i+1)%3] and signs[i]!=signs[(i+2)%3]][0]
        a,b,c=tri[lone],tri[(lone+1)%3],tri[(lone+2)%3]
        ab,ac=crossing(a,b),crossing(a,c)
        small=[a,ab,ac];large=[[ab,b,c],[ab,c,ac]]
        if signs[lone]:plus.append(small);minus+=large
        else:minus.append(small);plus+=large
    return np.asarray(Vnew),np.asarray(UVnew),np.asarray(minus,int),np.asarray(plus,int)


def _split_piece(piece,coord,value,pitch):
    V,F,UV=piece
    for shift in (0.,1e-3,-1e-3,3e-3,-3e-3,6e-3,-6e-3):
        vn,un,minus,plus=_clip(V,F,UV,coord,value+shift*pitch)
        out={}
        for side,ff in (("minus",minus),("plus",plus)):
            if not len(ff) or len(_face_components(ff))!=1:break
            used=np.unique(ff);remap=-np.ones(len(vn),int)
            remap[used]=np.arange(len(used));fp=remap[ff]
            if _tri_angles(vn[used],fp).min()<1e-3:break
            out[side]=(vn[used],fp,un[used])
        if len(out)==2:return out
    raise RuntimeError("could not cut 3D source piece cleanly along fixed grid line")


def _quad_area3(P):
    a=.5*np.linalg.norm(np.cross(P[:,1]-P[:,0],P[:,2]-P[:,0]),axis=-1)
    a+=.5*np.linalg.norm(np.cross(P[:,2]-P[:,0],P[:,3]-P[:,0]),axis=-1)
    b=.5*np.linalg.norm(np.cross(P[:,1]-P[:,0],P[:,3]-P[:,0]),axis=-1)
    b+=.5*np.linalg.norm(np.cross(P[:,2]-P[:,1],P[:,3]-P[:,1]),axis=-1)
    return .5*(a+b)


def tile_area_ratios(piece,quad_xy,new_uv,shrink=2e-3):
    V,F,UV=piece;quad_xy=np.asarray(quad_xy,float)
    centre=quad_xy.mean(axis=1,keepdims=True)
    pts=(centre+(1-shrink)*(quad_xy-centre)).reshape(-1,2)
    tri,bary=_locate(UV,F,pts)
    if (tri<0).any():raise RuntimeError("M2D tile corner lies outside clipped source piece")
    corners=F[tri]
    P3=(bary[:,:,None]*V[corners]).sum(1).reshape(-1,4,3)
    P2=(bary[:,:,None]*new_uv[corners]).sum(1).reshape(-1,4,2)
    x,y=P2[...,0],P2[...,1]
    a2=.5*np.abs((x*np.roll(y,-1,axis=1)-y*np.roll(x,-1,axis=1)).sum(1))
    if np.any(a2<=1e-14) or not np.isfinite(a2).all():
        raise RuntimeError("degenerate free-BFF tile chart")
    ratios=_quad_area3(P3)/a2
    if (ratios<=0).any() or not np.isfinite(ratios).all():
        raise RuntimeError("invalid 3D/2D tile area ratios")
    return ratios


def plan_reflatten_cuts(grid_xy,quads,xyz,tri_faces,uv,*,flatten,
                        rings=3,threshold=2.,max_cuts=64):
    """Assess a full-grid cut using fresh BFF for both 3D child patches."""
    grid_xy=np.asarray(grid_xy,float);quads=np.asarray(quads,int)
    quad_xy=grid_xy[quads];centres=quad_xy.mean(axis=1)
    pitch=float(np.median(np.linalg.norm(quad_xy[:,1]-quad_xy[:,0],axis=1)))
    tol=1e-6*max(pitch,1e-12)
    curvature=_smoothed_gaussian_curvature(xyz,tri_faces,int(rings))
    tri,bary=_locate(uv,tri_faces,grid_xy)
    vertex_k=np.full(len(grid_xy),-np.inf)
    good=tri>=0
    vertex_k[good]=(bary[good]*curvature[tri_faces[tri[good]]]).sum(1)
    def sigma(piece,ids):
        J=tile_area_ratios(piece,quad_xy[ids],flatten(piece[0],piece[1]))
        return float(np.max(J)/np.min(J)),J
    def interior(ids):
        edge_count=defaultdict(int)
        for quad in quads[ids]:
            for i in range(4):
                a,b=int(quad[i]),int(quad[(i+1)%4])
                edge_count[(min(a,b),max(a,b))]+=1
        boundary={v for edge,c in edge_count.items() if c==1 for v in edge}
        return sorted(set(quads[ids].ravel().tolist())-boundary)
    report=dict(model=_CSF_MODEL,threshold=float(threshold),curvature_rings=int(rings),
                pitch=pitch,steps=[],leaves=[])
    started=time.perf_counter()
    root=(np.asarray(xyz,float),np.asarray(tri_faces,int),np.asarray(uv,float))
    all_ids=np.arange(len(quads));root_sigma,root_J=sigma(root,all_ids)
    report["initial_sigma"]=root_sigma
    report["initial_tile_area_ratio"]=root_J.tolist()
    stack=[(root,all_ids,root_sigma,root_J,0)]
    cuts=[]
    while stack:
        piece,ids,parent_sigma,parent_J,depth=stack.pop()
        candidates=interior(ids)
        if parent_sigma<=threshold or not candidates or len(cuts)>=max_cuts:
            report["leaves"].append(dict(face_ids=ids.tolist(),sigma=parent_sigma,
                    satisfied=bool(parent_sigma<=threshold),tile_area_ratio=parent_J.tolist()))
            continue
        peak=max(candidates,key=lambda v:vertex_k[v])
        best=None;evaluated=[]
        for coord,axis in ((0,"col"),(1,"row")):
            value=float(grid_xy[peak,coord])
            minus=ids[centres[ids,coord]<value-tol]
            plus=ids[centres[ids,coord]>value+tol]
            if not len(minus) or not len(plus):continue
            try:halves=_split_piece(piece,coord,value,pitch)
            except RuntimeError:continue
            children=[]
            try:
                for side,child_ids in (("minus",minus),("plus",plus)):
                    score,J=sigma(halves[side],child_ids)
                    children.append((halves[side],child_ids,score,J))
            except (RuntimeError,ValueError):
                continue
            worst=max(child[2] for child in children)
            evaluated.append(dict(axis=axis,value=value,
                child_tiles=[int(len(c[1])) for c in children],
                child_sigma=[float(c[2]) for c in children],worst_child_sigma=float(worst)))
            if best is None or worst<best[0]:best=(worst,axis,value,children)
        if best is None:
            report["leaves"].append(dict(face_ids=ids.tolist(),sigma=parent_sigma,
                satisfied=False,reason="no_complete_grid_cut",tile_area_ratio=parent_J.tolist()))
            continue
        worst,axis,value,children=best
        cuts.append(dict(axis=axis,value=float(value),source_face_id=int(ids[0])))
        report["steps"].append(dict(depth=depth,parent_tiles=int(len(ids)),
                parent_sigma=float(parent_sigma),curvature_vertex=int(peak),
                curvature_xy=grid_xy[peak].tolist(),curvature=float(vertex_k[peak]),
                axis=axis,value=value,parent_face_ids=ids.tolist(),
                parent_tile_area_ratio=parent_J.tolist(),
                child_tiles=[int(len(c[1])) for c in children],
                child_face_ids=[c[1].tolist() for c in children],
                child_sigma=[float(c[2]) for c in children],candidates=evaluated))
        for child in children:stack.append((*child,depth+1))
    report["seconds"]=time.perf_counter()-started
    report["final_max_sigma"]=max((l["sigma"] for l in report["leaves"]),default=1.)
    report["satisfied"]=all(l["satisfied"] for l in report["leaves"])
    return cuts,report


def split_mesh_reflatten(vertices,faces,domain,params=None):
    """Use present COMPLETE source-M2D cut logic, replacing only the CSF decision."""
    from .paper_mesh_splitting import _split_mesh_single,chart_geometry
    from .reference_bff import run_official_bff

    p=domain.parameterization
    xyz,tri_faces,_=chart_geometry(p)
    uv=np.asarray(p.uv_vertices_2d,float)
    angle=np.deg2rad(float(getattr(domain,"reference_grid_rotation_degrees",0.)))
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    grid_xy=np.asarray(vertices,float)[:,:2]@R
    if params is None: raise ValueError("reflatten Split requires PipelineParameters")
    exe=getattr(params,"bff_executable",None)
    timeout=float(getattr(params,"bff_timeout_seconds",120.))
    def flatten(V,F):
        result=run_official_bff(V,F,executable=exe,
            boundary_policy="automatic_reference",timeout_seconds=timeout)
        return np.asarray(result.uv_vertices,float)
    cuts,report=plan_reflatten_cuts(
        grid_xy,faces,xyz,tri_faces,uv@R,flatten=flatten,
        rings=int(getattr(params,"paper_split_curvature_rings",3)),
        threshold=2.,max_cuts=int(getattr(domain,"max_csf_splits",64)))
    # Preserve the original forced-cut topology implementation and ALL fields.
    # Use a defensive copy of params so changing the decision cannot leak
    # user-requested collision-driven Split state back into the UI.
    trial=copy.copy(params)
    saved_budget=domain.max_csf_splits
    existing_forced=list(getattr(trial,"collision_split_forced_lines",()) or ())
    domain.max_csf_splits=0
    # Paper CSF seams are committed first, followed by any independently
    # accepted collision-driven feedback seams. This preserves the OFF-mode
    # outer rebuild loop instead of erasing or rejecting its topology.
    trial.collision_split_forced_lines=tuple(
        [dict(c,score=0.) for c in cuts] + existing_forced
    )
    try:
        out_v,out_f,records,metrics=_split_mesh_single(vertices,faces,domain,trial)
    finally:
        domain.max_csf_splits=saved_budget
    if len(records)!=len(cuts)+len(existing_forced):
        raise RuntimeError("Reflatten or collision Split cut plan was not completely executed")
    csf_records=records[:len(cuts)]
    collision_records=records[len(cuts):]
    for record,step in zip(csf_records,report["steps"]):
        record.update(collision_driven=False,reflatten_csf_cut=True,
            face_ids=step["parent_face_ids"],
            child_face_ids=step["child_face_ids"],
            sigma_before=step["parent_sigma"],sigma_after=step["child_sigma"],
            curvature_vertex_id=step["curvature_vertex"],
            gaussian_curvature=step["curvature"],candidate_cuts=step["candidates"])
    metrics.update(
        csf_model=_CSF_MODEL,
        csf_split_exactness_label="RECONSTRUCTION (not author code): per-piece official free-boundary BFF used for decision only",
        csf_split_criterion=REFLATTEN_CRITERION,csf_split_threshold=2.,
        csf_split_applied=bool(csf_records),number_of_splits=len(records),
        csf_split_step_analysis=csf_records,csf_split_step_count=len(csf_records),
        csf_split_locations=[[r["axis"],r["snapped_value"]] for r in csf_records],
        collision_split_applied=bool(collision_records),
        collision_split_count=len(collision_records),
        collision_split_step_analysis=collision_records,
        collision_split_duplicated_vertex_count=sum(r["duplicated_vertices"] for r in collision_records),
        csf_split_duplicated_vertex_count=sum(r["duplicated_vertices"] for r in csf_records),
        max_csf_before_split=report["initial_sigma"],
        max_csf_after_split=report["final_max_sigma"],
        csf_split_unresolved_component_count=sum(not leaf["satisfied"] for leaf in report["leaves"]),
        csf_split_initial_components=[dict(face_ids=list(range(len(faces))),sigma=report["initial_sigma"])],
        csf_split_component_sigma=[leaf["sigma"] for leaf in report["leaves"]],
        csf_split_step_analysis_model=_CSF_MODEL,
        csf_split_status="satisfied" if report["satisfied"] else "no_admissible_curvature_guided_grid_cut",
        csf_split_stopping_reason="all components <= 2" if report["satisfied"] else "unresolved",
        csf_split_stopping_rule="new official free-boundary BFF for each 3D source patch; per-tile 3D/free-BFF area span <= 2; actual M2D seams preserved",
        csf_split_direction_rule="maximum area-smoothed Gaussian curvature interior grid vertex; compare full row/column cuts by worst reflattened child area span",
        csf_split_budget_exhausted=bool(not report["satisfied"] and len(cuts)>=saved_budget),
        csf_split_area_ratio_after=report["final_max_sigma"],
        csf_split_component_rescale_applied=False,
        csf_split_linear_csf_area_relation="RECONSTRUCTION uses area span directly, not sqrt(area span)",
        reflatten_split_report=report,
    )
    # Do NOT certify global deployability or panel fabrication from a CSF proxy.
    metrics["csf_split_bound_certified"]=False
    domain.csf_model=metrics["csf_model"]
    domain.csf_before=report["initial_sigma"]
    domain.csf_after_split=report["final_max_sigma"]
    domain.csf_split_exactness_label=metrics["csf_split_exactness_label"]
    domain.paper_split_diagnostics=dict(metrics,split_count=len(records),status=metrics["csf_split_status"])
    p.metrics["split_diagnostics"]=dict(domain.paper_split_diagnostics)
    return out_v,out_f,records,metrics
