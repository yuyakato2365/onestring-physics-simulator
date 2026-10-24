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


def _cut_uv_segments(parameterization, *, include_open=False):
    """Extract *topological UV-chart boundary* edges with 3D provenance.

    The previous method iterated over physical edges and called an edge a
    seam only if the two 3D incident triangles mapped to numerically
    different UV line segments. That loses topological cuts when two
    distinct UV boundary copies are coincident (or very close) in UV.

    Instead first find UV-chart boundary edges by UV vertex indices, and
    only then classify each edge by its physical 3D incident count.
    Two UV border copies of the same 3D edge are a seam even if their
    UV coordinates happen to be equal. A one-sided physical edge is an
    actual open-source boundary. Three-plus incidences are ambiguous.
    """
    xyz=np.asarray(parameterization.surface_vertices_3d,float)
    sf=np.asarray(parameterization.surface_faces,int)
    uv=np.asarray(parameterization.uv_vertices_2d,float)
    uf=np.asarray(parameterization.uv_faces,int)
    if len(sf)!=len(uf) or sf.shape!=uf.shape:
        raise SeamTopologyMismatch("SEAM_SOURCE_FACE_UV_CORRESPONDENCE_MISSING")
    if len(xyz)==0 or len(uv)==0 or sf.ndim!=2 or sf.shape[1]!=3:
        raise SeamTopologyMismatch("SEAM_SOURCE_EMPTY_OR_INVALID_TRIANGULATION")
    scale=max(float(np.ptp(xyz,axis=0).max()),1e-9)
    tol=max(scale*1e-9,1e-12)
    rounded=np.rint(xyz/tol).astype(np.int64)
    canonical={}
    ids=[]
    for p in rounded:
        key=tuple(map(int,p))
        if key not in canonical:canonical[key]=len(canonical)
        ids.append(canonical[key])
    physical=np.asarray(ids,int)
    # Map each *source* UV boundary edge to a paired 3D edge.
    uv_edge_incidences=defaultdict(list)
    physical_incidences=defaultdict(list)
    for fi,(tri3,triuv) in enumerate(zip(sf,uf)):
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=int(tri3[i]),int(tri3[j])
            u,v=int(triuv[i]),int(triuv[j])
            ca,cb=int(physical[a]),int(physical[b])
            if ca==cb:
                raise SeamTopologyMismatch(
                    "SEAM_SOURCE_DEGENERATE_3D_EDGE: "
                    f"face={fi} original_vertices=({a},{b})"
                )
            key=(min(ca,cb),max(ca,cb))
            if key[0]==ca:
                uv_segment=np.array([uv[u],uv[v]],float)
                xyz_segment=np.array([xyz[a],xyz[b]],float)
            else:
                uv_segment=np.array([uv[v],uv[u]],float)
                xyz_segment=np.array([xyz[b],xyz[a]],float)
            record=(uv_segment,xyz_segment,key,fi)
            uv_edge_incidences[tuple(sorted((u,v)))].append(record)
            physical_incidences[key].append(record)
    segments=[]
    open_boundary=[]
    nonmanifold=[]
    inconsistent=[]
    for uvkey,side_records in uv_edge_incidences.items():
        if len(side_records)!=1:
            continue
        uvline,xyzline,key,fi=side_records[0]
        count=len(physical_incidences[key])
        if count==1:
            open_boundary.append(uvline)
        elif count==2:
            # Check that the *other* physical incidence is also a UV
            # border edge, as required of a genuine chart cut.
            others=physical_incidences[key]
            border_count=sum(
                len(uv_edge_incidences[tuple(sorted((
                    int(uf[r[3],i]),int(uf[r[3],j])
                )))])==1
                for r in others
                for i,j in ((0,1),(1,2),(2,0))
                if tuple(sorted((
                    int(physical[sf[r[3],i]]),
                    int(physical[sf[r[3],j]])
                )))==key
            )
            if border_count!=2:
                inconsistent.append((key,fi,border_count))
                continue
            # Pair via physical source edge ID, never by UV closeness.
            segments.append((uvline,xyzline,key,0))
        else:
            nonmanifold.append((key,count))
    if nonmanifold:
        raise SeamTopologyMismatch(
            "SEAM_SOURCE_NONMANIFOLD_BOUNDARY_PROVENANCE: "
            f"count={len(nonmanifold)} sample={nonmanifold[:5]}"
        )
    if inconsistent:
        raise SeamTopologyMismatch(
            "SEAM_SOURCE_UV_BOUNDARY_INCIDENCE_MISMATCH: "
            f"count={len(inconsistent)} sample={inconsistent[:5]}"
        )
    # Assign both topological UV copies of every physical cut edge.
    by_edge=defaultdict(list)
    for i,seg in enumerate(segments):
        by_edge[seg[2]].append(i)
    for key,indices in by_edge.items():
        if len(indices)!=2:
            raise SeamTopologyMismatch(
                "SEAM_SOURCE_PHYSICAL_CUT_NOT_TWO_SIDED: "
                f"edge={key} copies={len(indices)}"
            )
        for side,i in enumerate(sorted(indices)):
            uvline,xyzline,physical,_=segments[i]
            segments[i]=(uvline,xyzline,physical,side)
    return (segments,open_boundary) if include_open else segments

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


def _covered_fraction_on_source_boundary(a,b,lines,tol):
    """Total fraction of a Grid edge covered by collinear UV source edges.

    Endpoint-only classification falsely accepts a chord whose endpoints
    touch a source boundary but whose interior cuts across the chart.
    Intersect the real collinear segment intervals instead. This does not
    relax the UV proximity threshold or geometrically snap any panels.
    """
    a=np.asarray(a,float)
    b=np.asarray(b,float)
    d=b-a
    denom=float(np.dot(d,d))
    if denom<=1e-24:
        raise SeamTopologyMismatch("SEAM_M2D_ZERO_LENGTH_BOUNDARY_EDGE")
    length=denom**.5
    eps=max(float(tol)/length,1e-12)
    intervals=[]
    for line in lines:
        edge=np.asarray(line,float)
        # Source boundary segments must be entirely on the Grid edge's
        # supporting line, with nonzero interval overlap.
        e0=edge[0]-a
        e1=edge[1]-a
        off0=abs(float(d[0]*e0[1]-d[1]*e0[0]))/length
        off1=abs(float(d[0]*e1[1]-d[1]*e1[0]))/length
        if max(off0,off1)>tol:
            continue
        tt=np.dot(edge-a,d)/denom
        low=max(0.,float(min(tt)))
        high=min(1.,float(max(tt)))
        if high>low:
            intervals.append((low,high))
    if not intervals:
        return 0.
    intervals.sort()
    total=0.
    left,right=intervals[0]
    for lo,hi in intervals[1:]:
        if lo<=right+eps:
            right=max(right,hi)
        else:
            total+=right-left
            left,right=lo,hi
    total+=right-left
    return float(np.clip(total,0.,1.))


def _distance_to_uv_segments(p,lines):
    p=np.asarray(p,float)
    best=float("inf")
    for line in lines:
        line=np.asarray(line,float)
        d=line[1]-line[0]
        denom=float(np.dot(d,d))
        if denom<=1e-24:
            continue
        t=float(np.clip(np.dot(p-line[0],d)/denom,0.,1.))
        best=min(best,float(np.linalg.norm(p-(line[0]+t*d))))
    return best if np.isfinite(best) else None


def discover_exact_seam_topology(parameterization, uv_vertices, quad_faces):
    """Return pairs and counts using original UV->source-seam provenance.

    Exact paired edges require both edge endpoints to be sampled at the
    SAME 3D positions on both cut sides. A count of unmatched edges means
    the two polygonal boundaries do not admit exact stitching without
    changing the quad topology. Never silently report those as welded.
    """
    uv=np.asarray(uv_vertices,float)
    faces=np.asarray(quad_faces,int)
    segments,open_uv_edges=_cut_uv_segments(
        parameterization,include_open=True
    )
    if not segments:
        # The old implementation silently returned "no_original_cut",
        # claiming nothing and proceeding to K3D even when a CLOSED input
        # had a visible M2D boundary. That is not a legitimate closure.
        edges=_uv_boundary_edges(faces)
        if not edges:
            return {
                "status":"no_boundary","groups":[],"paired_edges":[],
                "unmatched_edges":0,"unclassified_boundary_edges":0,
                "seam_grid_edges":0,"source_open_grid_edges":0,
                "boundary_grid_edges":0,
            }
        if not open_uv_edges:
            raise SeamTopologyMismatch(
                "SEAM_K3D_NO_SOURCE_CUT_FOR_GRID_BOUNDARY: "
                f"grid_boundary_edges={len(edges)}; "
                "the chart has a nonempty Grid boundary but the paired "
                "3D/UV source has neither a two-sided cut nor a genuine "
                "open 3D edge. Source seam provenance/UV topology is "
                "missing: SmoothMax cannot generate missing seam pairs."
            )
        tol=max(float(np.ptp(uv,axis=0).max())*1e-8,1e-9)
        uncovered=[]
        for i,j in edges:
            fraction=_covered_fraction_on_source_boundary(
                uv[i],uv[j],open_uv_edges,tol
            )
            if fraction<1.-tol/max(
                float(np.linalg.norm(uv[i]-uv[j])),1e-12
            ):
                uncovered.append((int(i),int(j),float(fraction)))
        if uncovered:
            raise SeamTopologyMismatch(
                "SEAM_K3D_GRID_NOT_ON_SOURCE_OPEN_BOUNDARY: "
                f"uncovered_edges={len(uncovered)} sample={uncovered[:6]}"
            )
        return {
            "status":"source_open_boundary_only","groups":[],
            "paired_edges":[],"unmatched_edges":0,
            "unclassified_boundary_edges":0,"seam_grid_edges":0,
            "source_open_grid_edges":len(edges),
            "boundary_grid_edges":len(edges),
            "source_open_boundary_edge_count":len(open_uv_edges),
        }
    uvspan=max(float(np.ptp(uv,axis=0).max()),1.)
    tol=max(1e-8*uvspan,1e-9)
    xyz=np.asarray(parameterization.surface_vertices_3d,float)
    xyzspan=max(float(np.ptp(xyz,axis=0).max()),1.)
    qtol=max(1e-7*xyzspan,1e-9)
    edges=_uv_boundary_edges(faces)
    sampled={}
    on_open={}
    for a,b in edges:
        for v in (a,b):
            if v not in sampled:
                sampled[v]=_source_seam_sample(uv[v],segments,tol)
                on_open[v]=any(
                    np.linalg.norm(uv[v]-(line[0]+
                        np.clip(np.dot(uv[v]-line[0],line[1]-line[0])/
                        max(np.dot(line[1]-line[0],line[1]-line[0]),1e-20),
                        0.,1.)*(line[1]-line[0])))<=tol
                    for line in open_uv_edges
                )
    seam_edges=[]
    source_open_grid_edges=0
    mixed_cut_open_edges=[]
    unclassified=[]
    diagnostics=[]
    cut_lines=[record[0] for record in segments]
    full_source_lines=cut_lines+open_uv_edges
    # Verify each COMPLETE Grid boundary edge, not only its two endpoints:
    # both endpoint coordinates can land on a source curve even when their
    # straight connecting segment cuts across the true chart boundary.
    for a,b in edges:
        seam_covered=_covered_fraction_on_source_boundary(
            uv[a],uv[b],cut_lines,tol
        )
        open_covered=_covered_fraction_on_source_boundary(
            uv[a],uv[b],open_uv_edges,tol
        )
        full_covered=_covered_fraction_on_source_boundary(
            uv[a],uv[b],full_source_lines,tol
        )
        edge_tol=max(tol/max(float(np.linalg.norm(uv[b]-uv[a])),1e-12),
                     1e-9)
        if (seam_covered>=1.-edge_tol and
                sampled[a] is not None and sampled[b] is not None):
            seam_edges.append((a,b))
        elif open_covered>=1.-edge_tol:
            # An actual open input surface stays physically open.
            source_open_grid_edges+=1
        elif (full_covered>=1.-edge_tol and
                seam_covered>edge_tol and open_covered>edge_tol):
            # The physical source cut ends inside this ONE Grid edge.
            # Retain genuine open remainder, constrain only cut part.
            mixed_cut_open_edges.append((int(a),int(b),
                                         seam_covered,open_covered))
        else:
            unclassified.append((a,b))
            midpoint=(uv[a]+uv[b])*.5
            diagnostics.append({
                "edge":[int(a),int(b)],
                "uv_a":np.round(uv[a],9).tolist(),
                "uv_b":np.round(uv[b],9).tolist(),
                "cut_coverage":round(seam_covered,8),
                "open_coverage":round(open_covered,8),
                "all_source_boundary_coverage":round(full_covered,8),
                "seam_endpoint_ids":[
                    bool(sampled[a] is not None),
                    bool(sampled[b] is not None),
                ],
                "nearest_cut_distances":[
                    _distance_to_uv_segments(uv[a],cut_lines),
                    _distance_to_uv_segments(midpoint,cut_lines),
                    _distance_to_uv_segments(uv[b],cut_lines),
                ],
                "nearest_open_distances":[
                    _distance_to_uv_segments(uv[a],open_uv_edges),
                    _distance_to_uv_segments(midpoint,open_uv_edges),
                    _distance_to_uv_segments(uv[b],open_uv_edges),
                ],
                "likely_reason":(
                    "M2D_grid_edge_not_on_actual_omega_boundary"
                    if full_covered<1.-edge_tol else
                    "grid_edge_crosses_seam_to_open_source_boundary"
                    if (seam_covered>0 and open_covered>0) else
                    "uv_boundary_correspondence_or_source_topology"
                ),
            })
    # Fail closed if even one cell boundary is not proven to correspond
    # to a real source cut (two-sided) or true source open boundary.
    # In particular, don't weld an edge that only PARTLY lies on a cut.
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
    affine_rows=[]
    affine_metrics={}
    if not unclassified:
        from .optcuts_seam_affine_constraints import (
            source_cut_affine_rows,
        )
        affine_rows,affine_metrics=source_cut_affine_rows(
            uv,faces,segments,absolute_tolerance=tol
        )
    can_stitch=bool(not unclassified and affine_rows)
    from .optcuts_seam_soft_constraints import seam_matrix_rank_diagnostics
    rank_audit=seam_matrix_rank_diagnostics(affine_rows,len(uv))
    # Strict source-edge equations may geometrically force a collapse
    # when independent Grid partitions assign different breakpoints.
    # Hard weld is safe only with fully matching Grid edges and NO mixed
    # source-cut/original-open edge. Preserve source cut provenance in
    # both modes; do not confuse valid source boundary coverage with
    # nondegenerate *hard* solvability.
    fully_matching=bool(
        can_stitch and len(unmatched)==0
        and len(mixed_cut_open_edges)==0
        and affine_metrics.get("source_cut_different_grid_partition_count",1)==0
        and len(paired)>0
    )
    mode="hard" if fully_matching else "soft" if can_stitch else "unavailable"
    return {
        "status":("conforming" if can_stitch else "nonconforming"),
        "groups":[sorted(x) for x in groups],
        "affine_rows":affine_rows,
        "affine_metrics":affine_metrics,
        "affine_exact_stitchable":bool(fully_matching),
        "seam_stitch_mode":mode,
        "affine_rank_diagnostics":rank_audit,
        "paired_edges":paired,
        "unmatched_edges":0 if can_stitch else len(unmatched),
        "full_grid_edge_unmatched_before_affine":len(unmatched),
        "mixed_seam_open_grid_edges":len(mixed_cut_open_edges),
        "mixed_grid_edge_sample":[
            [i,j,float(c),float(o)] for i,j,c,o
            in mixed_cut_open_edges[:12]
        ],
        "unclassified_boundary_edges":len(unclassified),
        "unclassified_boundary_edge_ids_sample":[
            [int(a),int(b)] for a,b in unclassified[:12]
        ],
        "unclassified_boundary_diagnostics":diagnostics[:12],
        "source_open_boundary_edge_count":len(open_uv_edges),
        "source_cut_boundary_edge_count":len(segments),
        "source_open_grid_edges":int(source_open_grid_edges),
        "boundary_grid_edges":int(len(edges)),
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


def _polish_seam_closed_smoothmax(vertices, faces, groups, *, weight, maxiter=70):
    """Optimize real SmoothMax on exact-equivalence-reduced unknowns.

    A feasible seam-closed, planarity-restored geometry is ALWAYS retained
    as a rollback. An iterate is accepted only when the *actual* SmoothMax
    energy decreases while the original quad-planarity tolerance is not
    worsened; no approximate weld or relaxed planarity is accepted.
    """
    from scipy.optimize import minimize
    from .optcuts_test_k3d_augmented_lagrangian_patch import (
        _square_energy_and_gradient, _face_scales,
        _robust_constraint_faces, _quad_constraint_values_and_gradient,
        _quad_plane_distances,
    )
    baseline=np.asarray(vertices,float).copy()
    quads=np.asarray(faces,int)
    scales=_face_scales(baseline,quads)
    robust,_=_robust_constraint_faces(baseline,quads)
    dist0=_quad_plane_distances(baseline,robust)
    ref_max=float(np.max(dist0)) if len(dist0) else 0.
    square_before=float(_square_energy_and_gradient(
        baseline,quads,scales,weight)[0])
    edge_scale=float(np.median(scales)) if len(scales) else 1.
    max_planarity=max(ref_max*1.005,edge_scale*1e-6,1e-10)
    # Exact equality is enforced through a quotient of the vertex-index set.
    # Each original pair of seam copies is a single optimizer unknown.
    parent=np.arange(len(baseline),dtype=int)
    def find(i):
        i=int(i)
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=int(parent[i])
        return i
    for group in groups:
        root=find(group[0])
        for i in group[1:]:
            parent[find(i)]=root
    roots=np.asarray([find(i) for i in range(len(baseline))],int)
    _,inverse=np.unique(roots,return_inverse=True)
    n=int(np.max(inverse)+1)
    def unpack(z):
        return np.asarray(z,float).reshape(n,3)[inverse]
    def pack_grad(grad):
        out=np.zeros((n,3),float)
        np.add.at(out,inverse,grad)
        return out.ravel()
    x0=np.zeros((n,3),float)
    np.add.at(x0,inverse,baseline)
    counts=np.bincount(inverse,minlength=n).astype(float)
    x0/=counts[:,None]
    # The source is an already seam-closed and almost planar K3D; penalize
    # normalized triple-products strongly and retain a weak K3D reference.
    planar_weight=max(1000.,100.*float(weight))
    anchor_weight=.05*float(weight)
    nf=max(len(quads),1)
    def objective(z):
        v=unpack(z)
        energy,grad,_=_square_energy_and_gradient(v,quads,scales,weight)
        c,cg=_quad_constraint_values_and_gradient(v,robust,scales)
        energy+=.5*planar_weight*float(np.dot(c,c))/nf
        g=planar_weight*c[:,None,None]*cg/nf
        for local in range(4):
            np.add.at(grad,robust[:,local],g[:,local])
        delta=v-baseline
        normalizer=max(edge_scale*edge_scale,1e-18)
        energy+=.5*anchor_weight*float(np.sum(delta*delta))/normalizer/nf
        grad+=anchor_weight*delta/normalizer/nf
        return float(energy),pack_grad(grad)
    # The solver objective penalizes infeasible intermediate states, but
    # only the final fully audited geometry is allowed to replace baseline.
    result=minimize(
        objective,x0.ravel(),method="L-BFGS-B",jac=True,
        options={"maxiter":int(maxiter),"ftol":1e-11,"gtol":1e-8,"maxls":35},
    )
    trial=unpack(result.x)
    trial_max=float(np.max(_quad_plane_distances(trial,robust)))
    square_after=float(_square_energy_and_gradient(trial,quads,scales,weight)[0])
    accepted=bool(np.isfinite(trial).all()
                  and trial_max<=max_planarity
                  and square_after<square_before-1e-10
                  and max_seam_gap(trial,groups)<=1e-12)
    chosen=trial if accepted else baseline
    return chosen,{
        "k3d_seam_smoothmax_attempted":True,
        "k3d_seam_smoothmax_accepted":accepted,
        "k3d_seam_smoothmax_weight":float(weight),
        "k3d_seam_smoothmax_energy_before":square_before,
        "k3d_seam_smoothmax_energy_trial":square_after,
        "k3d_seam_smoothmax_energy_final":(
            square_after if accepted else square_before
        ),
        "k3d_seam_smoothmax_planarity_limit":max_planarity,
        "k3d_seam_smoothmax_planarity_trial":trial_max,
        "k3d_seam_smoothmax_iterations":int(result.nit),
        "k3d_seam_smoothmax_solver_success":bool(result.success),
    }


def fit_k3d_with_exact_pairs(
    vertices,quad_faces,groups,*,iterations=12,square_weight=0.
):
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
    diagnostics={"seam_k3d_gap_before":max_seam_gap(vertices,groups),
                 "seam_k3d_planarity_before":before}
    if float(square_weight)>0.:
        v,post_metrics=_polish_seam_closed_smoothmax(
            v,faces,groups,weight=float(square_weight)
        )
        diagnostics.update(post_metrics)
        after=_planarity_error(v,faces)
        gap=max_seam_gap(v,groups)
    diagnostics.update({
        "seam_k3d_gap_after":gap,
        "seam_k3d_planarity_after":after,
    })
    return v,diagnostics


def close_k3d_seam_after_full_vertex_restore(
    mesh, out, parameterization, params
):
    """Close source seams ONLY with full original M2D/M3D vertex IDs.

    A reduced extrusion-aware solve has a different vertex index space
    from the UV grid. It is invalid to interpret reduced face indices
    as original full-grid UV vertex indices. Always call this after
    the reduced K3D has been expanded to the original M3D vertex IDs.
    """
    uv=getattr(mesh,"_seam_original_m2d_uv",None)
    if uv is None:
        return out
    if len(uv)!=len(mesh.vertices) or len(uv)!=len(out.vertices):
        raise SeamTopologyMismatch(
            "SEAM_K3D_VERTEX_INDEX_SPACE_MISMATCH: "
            f"UV={len(uv)}, M3D={len(mesh.vertices)}, "
            f"K3D={len(out.vertices)}; "
            "must restore the full M2D/M3D vertex space before seam closure"
        )
    if not np.array_equal(np.asarray(mesh.faces,int),
                          np.asarray(out.faces,int)):
        raise SeamTopologyMismatch("SEAM_K3D_FACE_TOPOLOGY_CHANGED")
    topo=discover_exact_seam_topology(parameterization,uv,mesh.faces)
    log={k:v for k,v in topo.items() if k not in
         ("groups","paired_edges","affine_rows")}
    out.metrics.update({f"seam_closure_{k}":v for k,v in log.items()})
    if topo["status"] in ("no_boundary","source_open_boundary_only"):
        out.metrics["seam_exact_closure_enabled"]=False
        out.metrics["seam_exact_closure_reason"]=topo["status"]
        out.metrics["seam_exact_source_open_edge_count"]=int(
            topo.get("source_open_grid_edges",0)
        )
        print(
            "[SEAM-K3D] no_two_sided_cut "
            f"status={topo['status']} "
            f"genuine_source_open_grid_edges="
            f"{topo.get('source_open_grid_edges',0)}; "
            "there is NO corresponding side to weld. "
            "A genuine input hole cannot be closed by SmoothMax.",
            flush=True,
        )
        return out
    if topo["unclassified_boundary_edges"]:
        raise SeamTopologyMismatch(
            "SEAM_K3D_UNCLASSIFIED_GRID_BOUNDARY: "
            f"edges={topo['unclassified_boundary_edges']} "
            f"sample={topo['unclassified_boundary_edge_ids_sample']}; "
            f"diagnostics={topo['unclassified_boundary_diagnostics'][:4]}; "
            "the Grid edge must lie ENTIRELY on a paired source cut "
            "or true original open boundary. Do not weld/unclassify "
            "without restoring the Omega-M2D correspondence."
        )
    if topo["unmatched_edges"]:
        raise SeamTopologyMismatch(
            "SEAM_K3D_NONCONFORMING_GRID_EDGE_SAMPLING: "
            f"unmatched_grid_boundary_edges={topo['unmatched_edges']} "
            f"of {topo['seam_grid_edges']}; "
            "the two UV seam sides have different Grid edge partitions. "
            "This implementation requires matching source seam samples "
            "on both sides. Remesh/realign the Grid seam subdivisions "
            "before M3D; do not apply nearest-neighbor welding."
        )
    if topo.get("seam_stitch_mode")=="soft":
        from .optcuts_seam_soft_constraints import residual_stats
        seam_rows=topo.get("affine_rows",[])
        stats=residual_stats(out.vertices,seam_rows)
        # The objective was already applied inside K3D Augmented
        # Lagrangian. Do not project, weld, or restart the optimizer:
        # for incompatible Grid samples ker(C) may only contain a
        # zero-length Seam, making exact closure structurally impossible.
        out.metrics["seam_exact_closure_enabled"]=False
        out.metrics["seam_exact_closure_reason"]="incompatible_grid_partition_soft_fit"
        out.metrics["seam_stitch_mode"]="soft"
        out.metrics["seam_soft_row_count"]=len(seam_rows)
        out.metrics["seam_soft_gap_rms"]=stats["rms"]
        out.metrics["seam_soft_gap_max"]=stats["max"]
        out.metrics["seam_exact_affine_rows"]=[
            [[int(i),float(w)] for i,w in row]
            for row in seam_rows
        ]
        out.metrics["seam_exact_vertex_groups"]=[]
        out.metrics["seam_exact_full_shape_closed"]=False
        print(
            "[SEAM-K3D-SOFT] "
            f"rows={stats['row_count']} "
            f"rms={stats['rms']:.8g} max={stats['max']:.8g} "
            f"nullity={topo['affine_rank_diagnostics'].get('seam_nullity')} "
            "no hard projection; physical original-open regions retained",
            flush=True,
        )
        return out
    groups=topo["groups"]
    if not groups and not topo.get("affine_rows"):
        raise SeamTopologyMismatch(
            "SEAM_K3D_NO_PAIRED_BOUNDARY_VERTICES: "
            "real original OptCuts cut has no coincident Grid samples"
        )
    # Arbitrary mixed cut/open Grid edges have linear vertex-to-edge
    # constraints. They MUST survive the original SmoothMax optimization
    # and cannot be replaced by the older independent vertex-pair
    # planarity/averaging projection, which would reopen mixed edges.
    affine_rows=topo.get("affine_rows",[])
    from .optcuts_seam_affine_constraints import AffineSeamProjector
    affine=AffineSeamProjector(len(out.vertices),affine_rows)
    norm_scale=max(1.,float(np.max(np.abs(out.vertices))))
    residual_before=float(affine.residual(out.vertices))
    al_shared=bool(out.metrics.get("k3d_al_source_seam_hard_equality",False))
    measured=float("inf")
    plane_tol=float(out.metrics.get("k3d_hard_planarity_tolerance",0.))
    if al_shared:
        from .optcuts_test_k3d_augmented_lagrangian_patch import (
            _robust_constraint_faces,_quad_plane_distances,
        )
        cf,_=_robust_constraint_faces(out.vertices,out.faces)
        measured=float(np.max(_quad_plane_distances(out.vertices,cf)))
    al_gap=max_seam_gap(out.vertices,groups)
    from .optcuts_test_k3d_augmented_lagrangian_patch import (
        _quad_triangle_area_diagnostics,
    )
    checked_geometry=_quad_triangle_area_diagnostics(
        np.asarray(mesh.vertices,float),
        np.asarray(out.vertices,float),
        np.asarray(out.faces,int)
    )
    already_feasible=bool(
        al_shared and residual_before<=1e-9*norm_scale
        and al_gap<=1e-9*norm_scale and plane_tol>0
        and measured<=max(plane_tol,1e-10)
        and checked_geometry["collapsed_or_inverted_triangle_count"]==0
    )
    if already_feasible:
        result=np.asarray(out.vertices,float).copy()
        from .paper_extrusion import _planarity_error
        diag={
            "seam_k3d_gap_before":float(al_gap),
            "seam_k3d_gap_after":float(al_gap),
            "seam_k3d_planarity_before":_planarity_error(
                result,np.asarray(out.faces,int)
            ),
            "seam_k3d_planarity_after":_planarity_error(
                result,np.asarray(out.faces,int)
            ),
            "k3d_seam_smoothmax_attempted":False,
            "k3d_seam_post_projection_skipped":True,
            "k3d_seam_closed_in_single_AL_optimization":True,
        }
        print(
            "[SEAM-K3D] unified_affine_SmoothMax_AL_feasible "
            f"affine_rows={len(affine_rows)} "
            f"max_cut_gap_residual={residual_before:.9g} "
            "post_projection=SKIPPED",flush=True,
        )
    elif affine_rows:
        # A local AL can stop early with incomplete hard planarity. A
        # nearest-vertex weld must NEVER masquerade as the solution.
        # Continue within the same hard affine subspace instead.
        from .optcuts_test_k3d_augmented_lagrangian_patch import (
            _augmented_lagrangian_planarize,
            _quad_plane_distances,_robust_constraint_faces,
        )
        from .optcuts_test_k3d_augmented_lagrangian_patch import (
            _quad_triangle_area_diagnostics,
        )
        baseline=np.asarray(mesh.vertices,float)
        prior=np.asarray(out.vertices,float)
        before_guard=_quad_triangle_area_diagnostics(
            baseline,prior,np.asarray(out.faces,int)
        )
        # Never restart continuation from a collapsed result: this
        # would recompute face scales / the preservation reference
        # from near-zero panels, making SmoothMax collapse self-validating.
        if (before_guard["collapsed_or_inverted_triangle_count"]
                or before_guard["min_oriented_area_ratio"]<.08):
            restart=baseline.copy()
            restart_source="original_noncollapsed_M3D"
            print(
                "[SEAM-K3D-RECOVER] invalid_post_AL_geometry "
                f"collapsed_triangles={before_guard['collapsed_or_inverted_triangle_count']} "
                f"affected_faces={before_guard['collapsed_or_inverted_face_ids']} "
                "restart_from=M3D (not corrupted K3D)",
                flush=True,
            )
        else:
            restart=prior.copy()
            restart_source="optimized_K3D"
        result,refine_metrics=_augmented_lagrangian_planarize(
            restart,np.asarray(out.faces,int),
            square_weight=max(0.,float(getattr(params,"w_square",10.))),
            anchor_weight=float(getattr(params,"k3d_al_anchor_weight",1.)),
            rho0=float(getattr(params,"k3d_al_rho0",10.)),
            outer_iterations=max(4,int(getattr(
                params,"k3d_al_outer_iterations",8
            ))),
            inner_iterations=max(60,int(getattr(
                params,"k3d_al_inner_iterations",70
            ))),
            relative_plane_tolerance=float(getattr(
                params,"k3d_al_relative_planarity_tolerance",1e-6
            )),
            restoration_sweeps=int(getattr(
                params,"k3d_al_restoration_sweeps",8
            )),
            final_polish_sweeps=int(getattr(
                params,"k3d_al_final_polish_sweeps",400
            )),
            equality_groups=groups,affine_rows=affine_rows,
        )
        projection_gap=affine.residual(result)
        # A vanishing reference-plane normal makes the historical
        # plane-distance diagnostic +inf. Never misreport it as an
        # excessively large but otherwise valid planar quad.
        source_guard=_quad_triangle_area_diagnostics(
            baseline,result,np.asarray(out.faces,int)
        )
        bad_count=int(source_guard["collapsed_or_inverted_triangle_count"])
        cf,_=_robust_constraint_faces(result,out.faces)
        final_plane=float(np.max(_quad_plane_distances(result,cf)))
        allowed=float(refine_metrics.get("k3d_hard_planarity_tolerance",0.))
        if (bad_count or not np.isfinite(final_plane)):
            raise SeamTopologyMismatch(
                "SEAM_K3D_COLLAPSED_QUADS_IN_JOINT_OPTIMIZATION: "
                f"invalid_reference_triangles={bad_count}, "
                f"affected_quad_ids={source_guard['collapsed_or_inverted_face_ids']}, "
                f"min_signed_area_ratio={source_guard['min_oriented_area_ratio']:.9g}, "
                f"source_cut_gap={projection_gap:.9g}, "
                f"plane_distance={final_plane}, "
                f"restart_from={restart_source}. "
                "A zero-area quad has undefined plane normal. "
                "SmoothMax edge-length equality does not prevent a "
                "collapsed quad; exact source Seam plus local shape "
                "preservation may require revising panel topology."
            )
        if (projection_gap>1e-9*norm_scale or
                final_plane>max(allowed,1e-10)):
            raise SeamTopologyMismatch(
                "SEAM_K3D_AFFINE_SMOOTHMAX_PLANARITY_NOT_CONVERGED: "
                f"source_cut_gap={projection_gap:.9g}, "
                f"plane_distance={final_plane:.9g}, "
                f"plane_tol={allowed:.9g}, "
                f"restart_from={restart_source}, "
                f"min_signed_area_ratio={source_guard['min_oriented_area_ratio']:.9g}. "
                "Non-degenerate K3D still violates hard planarity under "
                "the present exact source-Seam and shape constraints. "
                "Do not relax the hard requirements without investigation."
            )
        diag={
            "seam_k3d_gap_before":float(al_gap),
            "seam_k3d_gap_after":max_seam_gap(result,groups),
            "seam_k3d_planarity_before":measured,
            "seam_k3d_planarity_after":final_plane,
            "k3d_seam_closed_in_single_AL_optimization":False,
            "k3d_seam_affine_al_continuation":True,
            "k3d_seam_smoothmax_attempted":True,
            "k3d_seam_post_projection_skipped":True,
        }
        out.metrics.update(refine_metrics)
    else:
        result,diag=fit_k3d_with_exact_pairs(
            out.vertices,out.faces,groups,
            square_weight=max(0.,float(getattr(params,"w_square",10.))),
        )
        diag["k3d_seam_post_projection_skipped"]=False
        diag["k3d_seam_closed_in_single_AL_optimization"]=False
    exact_gap=affine.residual(result)
    if exact_gap>1e-9*norm_scale:
        raise SeamTopologyMismatch(
            "SEAM_K3D_PARAMETRIC_CUT_GAP_NOT_ZERO: "
            f"max_constraint_residual={exact_gap:.9g}"
        )
    diag["seam_k3d_affine_row_count"]=len(affine_rows)
    diag["seam_k3d_affine_constraint_residual"]=float(exact_gap)
    out.vertices[:]=result
    out.metrics.update(diag)
    out.metrics["seam_exact_vertex_groups"]=[
        list(map(int,g)) for g in groups]
    out.metrics["seam_exact_affine_rows"]=[
        [[int(i),float(w)] for i,w in row]
        for row in topo.get("affine_rows",[])
    ]
    out.metrics["seam_exact_affine_constraint_count"]=len(
        topo.get("affine_rows",[])
    )
    out.metrics["seam_exact_paired_grid_edges"]=len(topo["paired_edges"])
    # Audit actual 3D boundary *segments*, not just independently welded
    # vertices. If both endpoint groups are shared, both entire straight
    # mesh edges must coincide under the same or reversed orientation.
    actual=np.asarray(out.vertices,float)
    edge_gap=0.
    for pair in topo["paired_edges"]:
        (a,b),(c,d)=pair
        p,q=actual[[int(a),int(b)]],actual[[int(c),int(d)]]
        gap_forward=max(float(np.linalg.norm(p-q)),
                        float(np.linalg.norm(actual[b]-actual[d])))
        gap_reverse=max(float(np.linalg.norm(actual[a]-actual[d])),
                        float(np.linalg.norm(actual[b]-actual[c])))
        edge_gap=max(edge_gap,min(gap_forward,gap_reverse))
    out.metrics["seam_exact_paired_edge_gap_max"]=float(edge_gap)
    if edge_gap>1e-10:
        raise SeamTopologyMismatch(
            "SEAM_K3D_PAIRED_EDGE_NOT_CLOSED_AFTER_SMOOTHMAX: "
            f"max_edge_endpoint_gap={edge_gap:.9g}"
        )
    out.metrics["seam_exact_closure_enabled"]=True
    out.metrics["seam_exact_source_open_edge_count"]=int(
        topo.get("source_open_grid_edges",0)
    )
    out.metrics["seam_exact_full_shape_closed"]=bool(
        topo.get("source_open_grid_edges",0)==0 and
        topo.get("mixed_seam_open_grid_edges",0)==0
    )
    out.metrics["seam_exact_closure_vertex_space"]="full_original_M2D_grid"
    out.metrics["seam_exact_closure_executed_after_reduction"]=bool(
        getattr(mesh,"_seam_deferred_from_reduced_K3D",False)
    )
    print(
        "[SEAM-K3D] exact_groups="
        f"{len(groups)} matched_edges={len(topo['paired_edges'])} "
        f"gap_before={diag['seam_k3d_gap_before']:.6g} "
        f"gap_after={diag['seam_k3d_gap_after']:.6g} "
        f"quad_planarity_after={diag['seam_k3d_planarity_after']:.6g} "
        f"smoothmax_accepted={diag.get('k3d_seam_smoothmax_accepted',False)} "
        f"smoothmax_before={diag.get('k3d_seam_smoothmax_energy_before',0.):.6g} "
        f"smoothmax_final={diag.get('k3d_seam_smoothmax_energy_final',0.):.6g}",
        flush=True,
    )
    return out


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
            uv=np.asarray(mesh.vertices,float)[:,:2].copy()
            out._seam_original_m2d_uv=uv
            # Certify the FULL M2D boundary topology *before* K3D. A
            # conforming source-Seam pair can be constrained during the
            # actual AL/SmoothMax solve, not merely welded afterward.
            topo=discover_exact_seam_topology(
                parameterization,uv,np.asarray(mesh.faces,int)
            )
            if (topo["status"]=="conforming" and
                    topo.get("affine_rows")):
                stitch_mode=topo.get("seam_stitch_mode","soft")
                out.metrics["seam_stitch_mode"]=stitch_mode
                out.metrics["seam_exact_vertex_groups"]=[
                    list(map(int,g)) for g in (
                        topo["groups"] if stitch_mode=="hard" else []
                    )
                ]
                out.metrics["seam_exact_preflight_certified"]=True
                out.metrics["seam_exact_affine_rows"]=[
                    [[int(i),float(w)] for i,w in row]
                    for row in topo["affine_rows"]
                ]
                out.metrics["seam_exact_preflight_paired_edges"]=len(
                    topo["paired_edges"]
                )
                out.metrics["seam_exact_preflight_mixed_edges"]=int(
                    topo["mixed_seam_open_grid_edges"]
                )
                print(
                    "[SEAM-K3D-PREFLIGHT] "
                    f"groups={len(topo['groups'])} "
                    f"matched_edges={len(topo['paired_edges'])} "
                    f"affine_rows={len(topo['affine_rows'])} "
                    f"mixed_seam_open_edges={topo['mixed_seam_open_grid_edges']} "
                    f"mode={stitch_mode} "
                    f"rank={topo['affine_rank_diagnostics'].get('seam_rank')} "
                    f"nullity={topo['affine_rank_diagnostics'].get('seam_nullity')} "
                    f"AL_hard_equality_ready={stitch_mode=='hard'}",
                    flush=True,
                )
            else:
                # Do NOT pass uncertified vertex groups to SmoothMax AL.
                # The existing strict topology audit after K3D still
                # rejects unclassified or nonconforming boundaries.
                out.metrics["seam_exact_preflight_certified"]=False
                print(
                    "[SEAM-K3D-PREFLIGHT] "
                    f"status={topo['status']} "
                    f"unmatched={topo.get('unmatched_edges',0)} "
                    f"unclassified={topo.get('unclassified_boundary_edges',0)} "
                    "AL_hard_equality_ready=False",
                    flush=True,
                )
        return out,report

    def optimize(target,mesh,parameterization,params):
        out,report=old_optimize(target,mesh,parameterization,params)
        if getattr(mesh,"_seam_defer_until_full_mesh",False):
            # extrusion_aware.optimize_assembled() maps vertices to a
            # reduced index space, solves K3D, then expands its result.
            # Defer source-Seam classification/welding until expansion:
            # the retained UV lattice still refers to FULL vertex IDs.
            out.metrics["seam_exact_closure_deferred"]=True
            print(
                "[SEAM-K3D] deferred_for_full_vertex_restore "
                f"reduced_vertices={len(mesh.vertices)}",
                flush=True,
            )
            return out,report
        return close_k3d_seam_after_full_vertex_restore(
            mesh,out,parameterization,params
        ),report

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
         "install_optcuts_exact_seam_closure",
         "close_k3d_seam_after_full_vertex_restore"]
