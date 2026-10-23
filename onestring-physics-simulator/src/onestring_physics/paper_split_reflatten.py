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
FIXED_CHART_CRITERION = "fixed_chart_tile_area"
_CSF_MODEL = ("Reconstructed Sec.4.5: free-boundary official BFF per source "
              "component; max(A3D/A2D)/min(A3D/A2D) <= 2 (tile areas)")
_FIXED_CHART_CSF_MODEL = ("Experimental baseline: inherited original Omega UV per "
                          "3D piece; max(A3D/A2D)/min(A3D/A2D) <= 2 (tile areas); "
                          "NO per-piece BFF")


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
        if sharp is not None:
            chosen=sharp
        else:
            shoulders=_shoulder_corner_picks(V[loop],min_elongation=1.3)
            chosen=shoulders if shoulders is not None else _refine_corner_picks(chosen,turning,smoothed,window)
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
        # zig-zag test on the immediate neighbours (+/-2): a tooth such as
        # +1.76/-0.97/-0.97 rad cancels out, a real corner does not. (A wide
        # window also swallows the gently flaring sides next to a real corner.)
        if turning[np.arange(i-2,i+3)%n].sum()<0.5*turning[i]: continue
        if all(min(abs(i-j),n-abs(i-j)) > n//8 for j in picks):
            picks.append(i)
        if len(picks)==4: return picks
    return None


def boundary_feature_angles(vertices, faces, corner_positions, threshold=np.pi/8):
    """Sharp non-corner boundary features to keep in the rectangular BFF target.

    The rectangle prescribes exterior angle pi/2 at the four corners and 0
    elsewhere. A small tooth on an otherwise straight side (e.g. a 0.01-long
    spike on a vessel rim, +1.76/-0.97/-0.97 rad) would have to be flattened,
    which collapses its UV triangle and makes the official BFF output invalid.
    Such clusters keep their own exterior angles, shifted to zero net turning,
    so the four sides stay straight on both sides of the feature and the total
    stays 2*pi. Returns [(x, y, z, angle), ...]; empty for smooth boundaries
    (Lilium, height fields) so their target is unchanged.
    """
    V=np.asarray(vertices,float);F=np.asarray(faces,int)[:,:3]
    loop=_boundary_loop(F);n=len(loop)
    angsum=np.bincount(F.ravel(),weights=_tri_angles(V,F).ravel(),minlength=len(V))
    turning=np.pi-angsum[loop]
    corner_pos=set()
    for c in np.asarray(corner_positions,float):
        corner_pos.add(int(np.argmin(np.linalg.norm(V[loop]-c,axis=1))))
    near_corner=lambda i: any(min(abs(i-c),n-abs(i-c))<=1 for c in corner_pos)
    flagged=[i for i in range(n) if abs(turning[i])>=threshold and not near_corner(i)]
    # each sharp vertex together with its two neighbours (which absorb the
    # opposite turning of a tooth or notch), merged when they touch
    members=sorted({(i+d)%n for i in flagged for d in (-1,0,1) if not near_corner((i+d)%n)})
    clusters=[]
    for i in members:
        if clusters and min(abs(i-clusters[-1][-1]),n-abs(i-clusters[-1][-1]))<=1:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    if len(clusters)>1 and min(abs(clusters[0][0]-clusters[-1][-1]),n-abs(clusters[0][0]-clusters[-1][-1]))<=1:
        clusters[0]=clusters.pop()+clusters[0]
    out=[]
    for cluster in clusters:
        if len(cluster)<2:
            continue
        values=turning[cluster]-turning[cluster].mean()
        for i,a in zip(cluster,values):
            x,y,z=V[loop[i]];out.append((float(x),float(y),float(z),float(a)))
    return out


def _shoulder_corner_picks(P,min_elongation=1.3):
    """Four corners for a smooth, ELONGATED outline: the shoulders of both ends.

    Without genuine corners the largest turning sits at the two end tips and at
    the side bulges, so the corners form a diamond (end, side, end, side) and
    every rectangle side takes a quarter of the outline: Omega comes out almost
    square however long the shape is. Here the corners are the boundary points
    furthest in the four directions +/-long axis +/-short axis (of the best-fit
    plane of the boundary), so the short sides are the two ends and the long
    sides are the two flanks. Outlines with length/width below
    ``min_elongation`` (Lilium: about 1) keep the previous rule.
    """
    P=np.asarray(P,float);n=len(P)
    centred=P-P.mean(axis=0)
    axes=np.linalg.svd(centred,full_matrices=False)[2]
    s=centred@axes[0];t=centred@axes[1]
    if np.ptp(t)<=1e-12 or np.ptp(s)/np.ptp(t)<float(min_elongation):
        return None
    picks=[]
    for a in (1.,-1.):
        for b in (1.,-1.):
            score=a*s/np.abs(s).max()+b*t/np.abs(t).max()
            picks.append(int(np.argmax(score)))
    if len(set(picks))!=4 or any(min(abs(i-j),n-abs(i-j))<=n//16 for k,i in enumerate(picks) for j in picks[k+1:]):
        return None
    return picks


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


class CornerSelectionRequest(BaseException):
    """Paper mode with manual corners: stop before the rectangular BFF and let
    the user pick four boundary points of S (BaseException, like below)."""
    def __init__(self, info):
        super().__init__("rectangle corners requested")
        self.info = info


class SplitDirectionQuestion(BaseException):
    """Paper mode asks the user for the direction of the next Split.

    A BaseException so wrapper layers that catch ``Exception`` let it reach the
    app, which shows the candidates and re-runs with the answer appended to
    ``paper_split_direction_choices``.
    """
    def __init__(self, info):
        super().__init__("split direction requested")
        self.info = info


def plan_reflatten_cuts(grid_xy,quads,xyz,tri_faces,uv,*,flatten=None,
                        rings=3,threshold=2.,max_cuts=64,direction_choices=(),ask=False,
                        curvature_tie_tolerance=0.02,force_one_cut=False,reflatten=True):
    """Assess complete grid cuts using per-piece BFF or inherited original UV.

    The opt-in validation cut exercises the topology even when sigma <= 2.
    ReBFF-OFF never invokes the BFF backend during Split evaluation.
    """
    if reflatten and not callable(flatten):
        raise ValueError("reBFF Split requires a callable BFF evaluator")
    grid_xy=np.asarray(grid_xy,float);quads=np.asarray(quads,int)
    quad_xy=grid_xy[quads];centres=quad_xy.mean(axis=1)
    pitch=float(np.median(np.linalg.norm(quad_xy[:,1]-quad_xy[:,0],axis=1)))
    tol=1e-6*max(pitch,1e-12)
    curvature=_smoothed_gaussian_curvature(xyz,tri_faces,int(rings))
    tri,bary=_locate(uv,tri_faces,grid_xy)
    vertex_k=np.full(len(grid_xy),-np.inf)
    good=tri>=0
    vertex_k[good]=(bary[good]*curvature[tri_faces[tri[good]]]).sum(1)
    # Height (+z of the input) of every grid vertex on S, for the tie rule below.
    vertex_z=np.full(len(grid_xy),-np.inf)
    vertex_z[good]=(bary[good]*np.asarray(xyz,float)[tri_faces[tri[good]],2]).sum(1)
    def sigma(piece,ids):
        # OFF retains the exact original S->Omega UV inherited through
        # clipping, with no per-piece reparameterization or rescaling.
        chart_uv=flatten(piece[0],piece[1]) if reflatten else piece[2]
        J=tile_area_ratios(piece,quad_xy[ids],chart_uv)
        return float(np.max(J)/np.min(J)),J
    def interior(ids):
        edge_count=defaultdict(int)
        for quad in quads[ids]:
            for i in range(4):
                a,b=int(quad[i]),int(quad[(i+1)%4])
                edge_count[(min(a,b),max(a,b))]+=1
        boundary={v for edge,c in edge_count.items() if c==1 for v in edge}
        return sorted(set(quads[ids].ravel().tolist())-boundary)
    report=dict(model=_CSF_MODEL if reflatten else _FIXED_CHART_CSF_MODEL,
                reflatten_per_piece=bool(reflatten),
                evaluation_uv="per_piece_official_free_bff" if reflatten else "original_omega_fixed_chart",
                threshold=float(threshold),curvature_rings=int(rings),
                pitch=pitch,steps=[],leaves=[],
                validation_force_one_cut_requested=bool(force_one_cut),
                validation_cut_generated=False)
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
        validation_forced=bool(force_one_cut and depth==0 and not cuts
                               and parent_sigma<=threshold)
        if ((parent_sigma<=threshold and not validation_forced)
                or not candidates or len(cuts)>=max_cuts):
            report["leaves"].append(dict(face_ids=ids.tolist(),sigma=parent_sigma,
                    satisfied=bool(parent_sigma<=threshold),tile_area_ratio=parent_J.tolist()))
            continue
        # Highest smoothed Gaussian curvature; when several interior vertices are
        # within ``curvature_tie_tolerance`` (relative) of the maximum - e.g. an
        # almost spherical bulb where the maximum is decided by mesh noise -
        # take the HIGHEST of them (largest z of the input surface).
        k_max=max(vertex_k[v] for v in candidates)
        tie_band=abs(k_max)*min(max(float(curvature_tie_tolerance),0.),0.99)
        ties=[v for v in candidates if vertex_k[v]>=k_max-tie_band]
        peak=max(ties,key=lambda v:(vertex_z[v],vertex_k[v])) if len(ties)>1 else max(candidates,key=lambda v:vertex_k[v])
        peak_rule="highest_among_near_equal_curvature" if len(ties)>1 else "max_curvature"
        best=None;evaluated=[];options={}
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
            options[axis]=(worst,axis,value,children)
            if best is None or worst<best[0]:best=(worst,axis,value,children)
        # Optional user choice for this cut (cuts are numbered in planning
        # order): "col" = vertical line in M2D, "row" = horizontal line.
        # "auto" or an unusable direction keeps the automatic choice.
        if ask and len(options)>1 and len(cuts)>=len(direction_choices):
            raise SplitDirectionQuestion(dict(
                step=len(cuts),parent_face_ids=ids.tolist(),parent_sigma=float(parent_sigma),
                candidates=evaluated,auto_axis=best[1],
                previous=[dict(axis=c["axis"],value=c["value"],parent_face_ids=st["parent_face_ids"])
                          for c,st in zip(cuts,report["steps"])],
                reflatten_per_piece=bool(reflatten),validation_forced=validation_forced,
                grid_xy=grid_xy.tolist(),quads=quads.tolist()))
        choice=str(direction_choices[len(cuts)]) if len(cuts)<len(direction_choices) else "auto"
        if choice=="stop":
            # The user ended the Split here: this and every remaining piece
            # become leaves as they are (sigma may stay above the threshold).
            max_cuts=len(cuts);report["stopped_by_user"]=True
            report["leaves"].append(dict(face_ids=ids.tolist(),sigma=parent_sigma,satisfied=False,
                    reason="stopped_by_user",tile_area_ratio=parent_J.tolist()))
            continue
        chosen_by="auto"
        if best is not None and choice in options:
            best=options[choice];chosen_by="user"
        if best is None:
            report["leaves"].append(dict(face_ids=ids.tolist(),sigma=parent_sigma,
                satisfied=False,reason="no_complete_grid_cut",tile_area_ratio=parent_J.tolist()))
            continue
        worst,axis,value,children=best
        cuts.append(dict(axis=axis,value=float(value),source_face_id=int(ids[0]),
                         validation_forced=validation_forced))
        report["steps"].append(dict(depth=depth,parent_tiles=int(len(ids)),
                validation_forced=validation_forced,
                parent_sigma=float(parent_sigma),curvature_vertex=int(peak),
                curvature_xy=grid_xy[peak].tolist(),curvature=float(vertex_k[peak]),
                peak_rule=peak_rule,near_equal_curvature_vertices=int(len(ties)),height=float(vertex_z[peak]),
                axis=axis,value=value,parent_face_ids=ids.tolist(),
                parent_tile_area_ratio=parent_J.tolist(),
                child_tiles=[int(len(c[1])) for c in children],
                child_face_ids=[c[1].tolist() for c in children],
                child_sigma=[float(c[2]) for c in children],candidates=evaluated,
                requested_direction=choice,chosen_by=chosen_by))
        for child in children:stack.append((*child,depth+1))
    report["seconds"]=time.perf_counter()-started
    report["validation_cut_generated"]=any(c.get("validation_forced",False) for c in cuts)
    report["final_max_sigma"]=max((l["sigma"] for l in report["leaves"]),default=1.)
    report["satisfied"]=all(l["satisfied"] for l in report["leaves"])
    return cuts,report


def split_mesh_reflatten(vertices,faces,domain,params=None,*,reflatten=True):
    """Use the exact same M2D seam-cut owner with either area-span evaluation."""
    from .paper_mesh_splitting import _split_mesh_single,chart_geometry

    p=domain.parameterization
    xyz,tri_faces,_=chart_geometry(p)
    uv=np.asarray(p.uv_vertices_2d,float)
    # The original Omega may be BFF, harmonic or LSCM. Both modes require
    # a valid chart; never silently repair flipped/degenerate triangles.
    from .reference_bff import triangle_jacobian_diagnostics
    chart_check=triangle_jacobian_diagnostics(xyz,uv,tri_faces)
    if (not np.isfinite(uv).all()
            or chart_check["uv_degenerate_triangle_count"]
            or chart_check["uv_triangle_flip_count"]
            or chart_check["surface_degenerate_triangle_count"]):
        raise ValueError(
            "Area-span Split needs a finite, non-flipped, non-degenerate "
            "initial S->Omega chart (including normal-flow harmonic/LSCM). "
            "Fix the input parameterization before CSF Split."
        )
    source_method=str(getattr(p,"method","unknown"))
    angle=np.deg2rad(float(getattr(domain,"reference_grid_rotation_degrees",0.)))
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    grid_xy=np.asarray(vertices,float)[:,:2]@R
    if params is None: raise ValueError("area-span Split requires PipelineParameters")
    flatten=None
    if reflatten:
        from .reference_bff import run_official_bff
        exe=getattr(params,"bff_executable",None)
        timeout=float(getattr(params,"bff_timeout_seconds",120.))
        def flatten(V,F):
            result=run_official_bff(V,F,executable=exe,
                boundary_policy="automatic_reference",timeout_seconds=timeout)
            return np.asarray(result.uv_vertices,float)
    cuts,report=plan_reflatten_cuts(
        grid_xy,faces,xyz,tri_faces,uv@R,flatten=flatten,reflatten=reflatten,
        rings=int(getattr(params,"paper_split_curvature_rings",3)),
        threshold=2.,max_cuts=int(getattr(domain,"max_csf_splits",64)),
        direction_choices=tuple(getattr(params,"paper_split_direction_choices",()) or ()),
        ask=bool(getattr(params,"paper_split_ask_direction",False)),
        curvature_tie_tolerance=float(getattr(params,"paper_split_curvature_tie_tolerance",0.02)),
        force_one_cut=bool(getattr(params,"paper_split_validation_force_one_cut",False)))
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
        record.update(collision_driven=False,reflatten_csf_cut=bool(reflatten),
            fixed_chart_area_cut=not bool(reflatten),
            validation_forced_cut=bool(step.get("validation_forced",False)),
            face_ids=step["parent_face_ids"],
            child_face_ids=step["child_face_ids"],
            sigma_before=step["parent_sigma"],sigma_after=step["child_sigma"],
            curvature_vertex_id=step["curvature_vertex"],
            gaussian_curvature=step["curvature"],candidate_cuts=step["candidates"])
    metrics.update(
        csf_model=_CSF_MODEL if reflatten else _FIXED_CHART_CSF_MODEL,
        csf_split_bff_evaluation_enabled=bool(reflatten),
        csf_split_evaluation_uv=report.get(
            "evaluation_uv",
            "per_piece_official_free_bff" if reflatten else "original_omega_fixed_chart"),
        csf_split_initial_omega_method=source_method,
        csf_split_initial_omega_is_official_bff=source_method=="paper_reference_bff",
        csf_split_normal_flow_enabled=not bool(getattr(params,"paper_fidelity_mode",False)),
        csf_split_exactness_label=(
            (("RECONSTRUCTION (not author code): per-piece official free-boundary BFF "
              "used for the decision only; initial S->Omega method=")
             if reflatten else
             ("EXPERIMENTAL FIXED CHART (not author code): no per-piece BFF; "
              "initial S->Omega UV used for the decision; method="))
            +source_method
        ),
        csf_split_criterion=REFLATTEN_CRITERION if reflatten else FIXED_CHART_CRITERION,
        csf_split_threshold=2.,
        csf_split_validation_force_one_cut_requested=bool(report["validation_force_one_cut_requested"]),
        csf_split_validation_forced_count=int(report["validation_cut_generated"]),
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
        csf_split_step_analysis_model=_CSF_MODEL if reflatten else _FIXED_CHART_CSF_MODEL,
        csf_split_status="satisfied" if report["satisfied"] else "no_admissible_curvature_guided_grid_cut",
        csf_split_stopping_reason="all components <= 2" if report["satisfied"] else "unresolved",
        csf_split_stopping_rule=(
            "new official free-boundary BFF per 3D patch; 3D/re-BFF-2D tile area span <= 2"
            if reflatten else
            "original Omega chart inherited by each 3D patch; 3D/fixed-UV tile area span <= 2"
        ),
        csf_split_direction_rule=(
            "maximum smoothed Gaussian curvature interior grid vertex; "
            "compare row/column cuts by worst child tile area span"
        ),
        csf_split_budget_exhausted=bool(not report["satisfied"] and len(cuts)>=saved_budget),
        csf_split_area_ratio_after=report["final_max_sigma"],
        csf_split_component_rescale_applied=False,
        csf_split_linear_csf_area_relation="RECONSTRUCTION uses area span directly, not sqrt(area span)",
        reflatten_split_report=report,  # Backward compatible UI/report key.
        tile_area_split_report=report,
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


def split_mesh_fixed_chart_area(vertices,faces,domain,params=None):
    """Area-span baseline: original S->Omega chart, no per-piece reBFF."""
    return split_mesh_reflatten(vertices,faces,domain,params,reflatten=False)
