"""Exact paired-vertex / paired-edge seam constraints for optcuts_test area Grid.

This is deliberately FAIL-CLOSED. A *cut* has two different UV border
copies of the same 3D surface edge. Grid polygon edges on both sides can
only form the same polyline without adding vertices when their 3D samples
have identical edge endpoints (up to source-geometric classification).
If the two Grid samplings are nonconforming, a quad-preserving remesh or
new cut/UV/Grid solve is needed; nearest-point welding is not valid.

Only boundary samples proven to originate on both UV copies of an
original 3D OptCuts cut are grouped; incidental 3D-near tiles are not.
"""
from collections import defaultdict

import numpy as np


class SeamTopologyMismatch(RuntimeError):
    pass


def _uv_boundary_edges(quad_faces):
    edges=defaultdict(list)
    for face_idx, face in enumerate(np.asarray(quad_faces,int)):
        face=list(map(int,face))
        for i in range(len(face)):
            a,b=face[i],face[(i+1)%len(face)]
            edges[tuple(sorted((a,b)))].append((face_idx,a,b))
    return [(item[0][1],item[0][2]) for item in edges.values() if len(item)==1]


def _cut_uv_segments(parameterization):
    """(UV line, 3D line, physical edge key, side number) for real cuts."""
    xyz=np.asarray(parameterization.surface_vertices_3d,float)
    sf=np.asarray(parameterization.surface_faces,int)
    uv=np.asarray(parameterization.uv_vertices_2d,float)
    uf=np.asarray(parameterization.uv_faces,int)
    if len(sf)!=len(uf) or sf.shape!=uf.shape:
        raise SeamTopologyMismatch("SEAM_SOURCE_FACE_UV_CORRESPONDENCE_MISSING")
    if len(xyz)==0 or len(uv)==0:
        raise SeamTopologyMismatch("SEAM_SOURCE_EMPTY")
    scale=max(float(np.ptp(xyz,axis=0).max()),1e-9)
    tol=max(scale*1e-9,1e-12)
    # Physical IDs, not UV IDs. Coincident source coordinates are canonical
    # only for discovering *original source edges*, never for K3D welding.
    rounded=np.rint(xyz/tol).astype(np.int64)
    canonical={}
    ids=[]
    for p in rounded:
        key=tuple(map(int,p))
        if key not in canonical:canonical[key]=len(canonical)
        ids.append(canonical[key])
    physical=np.asarray(ids,int)
    incident=defaultdict(list)
    for tri3,triuv in zip(sf,uf):
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=int(tri3[i]),int(tri3[j])
            ca,cb=int(physical[a]),int(physical[b])
            if ca==cb:continue
            key=tuple(sorted((ca,cb)))
            if key[0]==ca:
                incident[key].append((uv[int(triuv[i])],uv[int(triuv[j])],
                                      xyz[a],xyz[b]))
            else:
                incident[key].append((uv[int(triuv[j])],uv[int(triuv[i])],
                                      xyz[b],xyz[a]))
    uvscale=max(float(np.ptp(uv,axis=0).max()),1e-9)
    uvtol=max(uvscale*1e-8,1e-10)
    segments=[]
    for key,sides in incident.items():
        if len(sides)!=2:continue
        s0,s1=sides
        if (np.linalg.norm(s0[0]-s1[0])<=uvtol and
                np.linalg.norm(s0[1]-s1[1])<=uvtol):continue
        for side,seg in enumerate(sides):
            segments.append((
                np.stack(seg[:2]),np.stack(seg[2:]),
                key,side,
            ))
    return segments


def _source_seam_sample(uvp,segments,tol):
    """Source-mesh physical 3D sample for this Grid boundary UV coordinate."""
    p=np.asarray(uvp,float)
    matched=[]
    for uvline,xyzline,key,side in segments:
        v=uvline[1]-uvline[0]
        dd=float(np.dot(v,v))
        if dd<=1e-20:continue
        t=float(np.clip(np.dot(p-uvline[0],v)/dd,0,1))
        distance=float(np.linalg.norm(p-(uvline[0]+t*v)))
        if distance<=tol:
            matched.append((distance,xyzline[0]+t*(xyzline[1]-xyzline[0]),
                            key,side))
    if not matched:return None
    # At original seam corners, two adjacent physical seam edges may be
    # equally valid; both interpolate to the same original physical vertex.
    matched.sort(key=lambda x:x[0])
    if any(np.linalg.norm(x[1]-matched[0][1])>tol*100 for x in matched):
        raise SeamTopologyMismatch("SEAM_GRID_SAMPLE_AMBIGUOUS_SOURCE_SEAM")
    return matched[0][1]


def discover_exact_seam_topology(parameterization, uv_vertices, quad_faces):
    """Return pairs and counts using original UV->source-seam provenance.

    Exact paired edges require both edge endpoints to be sampled at the
    SAME 3D positions on both cut sides. A count of unmatched edges means
    the two polygonal boundaries do not admit exact stitching without
    changing the quad topology. Never silently report those as welded.
    """
    uv=np.asarray(uv_vertices,float)
    faces=np.asarray(quad_faces,int)
    segments=_cut_uv_segments(parameterization)
    if not segments:
        return {"status":"no_original_cut","groups":[],"paired_edges":[],
                "unmatched_edges":0,"seam_grid_edges":0}
    uvspan=max(float(np.ptp(uv,axis=0).max()),1.)
    tol=max(1e-8*uvspan,1e-9)
    xyz=np.asarray(parameterization.surface_vertices_3d,float)
    xyzspan=max(float(np.ptp(xyz,axis=0).max()),1.)
    qtol=max(1e-7*xyzspan,1e-9)
    edges=_uv_boundary_edges(faces)
    sampled={}
    for a,b in edges:
        for v in (a,b):
            if v not in sampled:
                sampled[v]=_source_seam_sample(uv[v],segments,tol)
    # Every edge on the true source seam must be matched on the other side.
    seam_edges=[(a,b) for a,b in edges
                if sampled[a] is not None and sampled[b] is not None]
    vertex_keys={}
    for v,p in sampled.items():
        if p is not None:
            vertex_keys[v]=tuple(np.rint(np.asarray(p)/qtol).astype(np.int64))
    by_sample=defaultdict(list)
    for v,key in vertex_keys.items():
        by_sample[key].append(v)
    groups=[ids for ids in by_sample.values() if len(ids)>1]
    edge_keys=defaultdict(list)
    for a,b in seam_edges:
        ka,kb=vertex_keys[a],vertex_keys[b]
        if ka==kb:
            raise SeamTopologyMismatch("SEAM_DEGENERATE_GRID_BOUNDARY_EDGE")
        edge_keys[tuple(sorted((ka,kb)))].append((a,b))
    paired=[]
    unmatched=[]
    for key,copies in edge_keys.items():
        # Two distinct UV sides must have distinct vertex IDs.
        if len(copies)==2 and set(copies[0]).isdisjoint(copies[1]):
            paired.append(copies)
        else:
            unmatched.extend(copies)
    return {
        "status":("conforming" if not unmatched else "nonconforming"),
        "groups":[sorted(x) for x in groups],
        "paired_edges":paired,
        "unmatched_edges":len(unmatched),
        "seam_grid_edges":len(seam_edges),
        "seam_grid_vertices":len(vertex_keys),
        "original_cut_segments":len(segments)//2,
        "source_xyz_tolerance":qtol,
    }


def enforce_shared_positions(vertices,groups):
    out=np.asarray(vertices,float).copy()
    for group in groups:
        ids=np.asarray(group,int)
        mean=out[ids].mean(axis=0)
        out[ids]=mean
    return out


def max_seam_gap(vertices,groups):
    points=np.asarray(vertices,float)
    return max((
        float(np.max(np.linalg.norm(points[np.asarray(g,int)]-
                        points[int(g[0])],axis=1)))
        for g in groups),default=0.)


def fit_k3d_with_exact_pairs(vertices,quad_faces,groups,*,iterations=12):
    """Restore planar quads while enforcing exact vertex equality each step.

    Uses the same local best-plane / global least squares model as the
    existing paper T3D stage, with equality groups eliminated from linear
    unknowns. If hard planarity cannot coexist with exact closed seam,
    fail rather than changing the fabrication QuadMesh or masking gaps.
    """
    from .paper_extrusion import _best_fit_plane, _planarity_error, _solve
    v=np.asarray(vertices,float).copy()
    faces=np.asarray(quad_faces,int)
    v=enforce_shared_positions(v,groups)
    initial=v.copy()
    before=_planarity_error(v,faces)
    for _ in range(max(0,int(iterations))):
        cons=[(f,_best_fit_plane(v[f]),1.) for f in faces]
        v=_solve(len(v),cons,initial,1e-2,
                 equality_groups=groups)
    after=_planarity_error(v,faces)
    gap=max_seam_gap(v,groups)
    if not np.isfinite(v).all() or gap>1e-10:
        raise SeamTopologyMismatch("SEAM_K3D_HARD_EQUALITY_NUMERICAL_FAILURE")
    return v,{"seam_k3d_gap_before":max_seam_gap(vertices,groups),
              "seam_k3d_gap_after":gap,
              "seam_k3d_planarity_before":before,
              "seam_k3d_planarity_after":after}


def install_optcuts_exact_seam_closure(pipeline):
    """Install LAST, outside K3D solver and M3D lift wrappers."""
    if getattr(pipeline,"_onestring_exact_seam_closure_installed",False):
        return
    old_lift=pipeline._lift_m2d_to_m3d
    old_optimize=pipeline._optimize_k3d

    def lift(target,mesh,parameterization,params):
        out,report=old_lift(target,mesh,parameterization,params)
        if (str(getattr(parameterization,"method",""))=="optcuts_test"
                and bool(parameterization.metrics.get(
                    "optcuts_area_grid_boundary_enabled",False))):
            out._seam_original_m2d_uv=np.asarray(mesh.vertices,float)[:,:2].copy()
        return out,report

    def optimize(target,mesh,parameterization,params):
        out,report=old_optimize(target,mesh,parameterization,params)
        uv=getattr(mesh,"_seam_original_m2d_uv",None)
        if uv is None:
            return out,report
        if not np.array_equal(np.asarray(mesh.faces,int),
                              np.asarray(out.faces,int)):
            raise SeamTopologyMismatch("SEAM_K3D_FACE_TOPOLOGY_CHANGED")
        topo=discover_exact_seam_topology(parameterization,uv,mesh.faces)
        log={k:v for k,v in topo.items() if k not in ("groups","paired_edges")}
        out.metrics.update({f"seam_closure_{k}":v for k,v in log.items()})
        if topo["status"]=="no_original_cut":
            return out,report
        if topo["unmatched_edges"]:
            raise SeamTopologyMismatch(
                "SEAM_K3D_NONCONFORMING_GRID_EDGE_SAMPLING: "
                f"unmatched_grid_boundary_edges={topo['unmatched_edges']} "
                f"of {topo['seam_grid_edges']}; "
                "the two UV seam sides have different Grid edge partitions. "
                "Exact K3D/T3D boundary correspondence requires conforming "
                "seam subdivision BEFORE M3D, not nearest-neighbor welding."
            )
        groups=topo["groups"]
        if not groups:
            raise SeamTopologyMismatch(
                "SEAM_K3D_NO_PAIRED_BOUNDARY_VERTICES: "
                "real original OptCuts cut has no coincident Grid samples"
            )
        result,diag=fit_k3d_with_exact_pairs(
            out.vertices,out.faces,groups
        )
        out.vertices[:]=result
        out.metrics.update(diag)
        out.metrics["seam_exact_vertex_groups"]=[
            list(map(int,g)) for g in groups]
        out.metrics["seam_exact_paired_grid_edges"]=len(topo["paired_edges"])
        out.metrics["seam_exact_closure_enabled"]=True
        print(
            "[SEAM-K3D] exact_groups="
            f"{len(groups)} matched_edges={len(topo['paired_edges'])} "
            f"gap_before={diag['seam_k3d_gap_before']:.6g} "
            f"gap_after={diag['seam_k3d_gap_after']:.6g} "
            f"quad_planarity_after={diag['seam_k3d_planarity_after']:.6g}",
            flush=True,
        )
        return out,report

    pipeline._lift_m2d_to_m3d=lift
    pipeline._optimize_k3d=optimize
    original=getattr(pipeline,"_original",None)
    if original is not None:
        original._lift_m2d_to_m3d=lift
        original._optimize_k3d=optimize
    for fn in (
        getattr(pipeline,"build_onestring_design",None),
        getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),
        getattr(original,"build_onestring_design",None) if original is not None else None,
    ):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict):
            glb["_lift_m2d_to_m3d"]=lift
            glb["_optimize_k3d"]=optimize
    pipeline._onestring_exact_seam_closure_installed=True


__all__=["SeamTopologyMismatch","discover_exact_seam_topology",
         "enforce_shared_positions","fit_k3d_with_exact_pairs",
         "install_optcuts_exact_seam_closure"]
