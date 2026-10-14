"""Rigid registration of T2D islands along the *physical* source-S Split seam.

Post-cut rectangular BFF produces disconnected UV charts. Simply packing
their rectangles side by side does not match the physical cut. This module
identifies vertices that were duplicated by the 3D triangle clipper, maps
those seam samples onto the closest corresponding M3D tile corners, then
fits rigid SE(2) placement of whole T2D components. Tile geometry and
internal hinge connections are never edited.

This is not a substitute for seam-compatible retiling; report the residual
and never mark an unmatched cut as geometrically welded.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def _boundary_point_groups(patches):
    """Find positive-length physical source seams by 3D boundary-edge overlap.

    Merely touching at one 3D vertex is NOT a split seam. Conversely, after
    recursive cuts one patch can have a boundary vertex in the middle of an
    adjacent patch's longer edge: counting equal vertex IDs would miss that
    real, positive-length seam. Intersect boundary *segments* geometrically
    and supply the overlap endpoints, including points interior to edges.
    No UV-space proximity is used to invent source adjacency.
    """
    from onestring_physics.reference_bff import validate_reference_mesh

    xyz_all=np.concatenate([np.asarray(p["xyz"],float) for p in patches])
    scale=max(1.,float(np.max(np.ptp(xyz_all,axis=0))))
    tol=1e-8*scale
    segments=[]
    for patch in patches:
        xyz=np.asarray(patch["xyz"],float)
        faces=np.asarray(patch["faces"],int)
        loop=np.asarray(validate_reference_mesh(xyz,faces)["boundary_loops"][0],int)
        start=xyz[loop]
        end=xyz[np.roll(loop,-1)]
        valid=np.linalg.norm(end-start,axis=1)>4*tol
        start,end=start[valid],end[valid]
        segments.append((
            start,end,
            np.minimum(start,end),np.maximum(start,end),
        ))

    by_pair={}
    for i in range(len(patches)):
        aa,ab,alo,ahi=segments[i]
        for j in range(i+1,len(patches)):
            ba,bb,blo,bhi=segments[j]
            if not len(aa) or not len(ba):
                continue
            if np.any(np.min(alo,axis=0)>np.max(bhi,axis=0)+tol) or np.any(
                np.min(blo,axis=0)>np.max(ahi,axis=0)+tol
            ):
                continue
            common=[]
            for a,b,low,high in zip(aa,ab,alo,ahi):
                # Only boundary edges with intersecting spatial bounding boxes
                # can overlap; avoid a Python O(N*M) geometric loop.
                possible=np.flatnonzero(np.all(
                    (bhi>=low-tol)&(blo<=high+tol),axis=1
                ))
                length=float(np.linalg.norm(b-a))
                unit=(b-a)/length
                for idx in possible:
                    p,q=ba[idx],bb[idx]
                    # Both endpoints must lie on the same infinite 3D line
                    # to tolerance, with a positive common interval.
                    dp=p-a
                    dq=q-a
                    tp=float(np.dot(dp,unit))
                    tq=float(np.dot(dq,unit))
                    if (np.linalg.norm(dp-tp*unit)>tol
                        or np.linalg.norm(dq-tq*unit)>tol):
                        continue
                    lo=max(0.,min(tp,tq))
                    hi=min(length,max(tp,tq))
                    if hi-lo<=4*tol:
                        continue
                    common.extend((a+unit*lo,a+unit*hi))
            if not common:
                continue
            # Shared endpoints can be generated repeatedly from subdivided
            # source triangles; preserve only geometrically distinct samples.
            distinct=[]
            for point in common:
                if all(np.linalg.norm(point-prev)>2*tol for prev in distinct):
                    distinct.append(point)
            if len(distinct)>=2:
                by_pair[(i,j)]=np.asarray(distinct,float)
    return by_pair


def _rigid_2d(source, destination):
    """SE(2) fit, no reflection or nonuniform scale; points are rows."""
    x=np.asarray(source,float)
    y=np.asarray(destination,float)
    if len(x)<2 or x.shape != y.shape or x.shape[1]!=2:
        raise ValueError("Need 2+ matching XY points for rigid seam fit")
    xc=np.mean(x,axis=0)
    yc=np.mean(y,axis=0)
    u,_,vt=np.linalg.svd((x-xc).T@(y-yc))
    diag=np.eye(2)
    diag[-1,-1] = np.linalg.det(u@vt)
    rotation=u@diag@vt
    shift=yc-xc@rotation
    rms=float(np.sqrt(np.mean(np.sum((x@rotation+shift-y)**2,axis=1))))
    return rotation,shift,rms


def _matching_shared_grid_corner_anchors(state, labels, reference, child, tol):
    """Use one-to-one exact square-grid seam vertices, not nearest 3D tiles.

    The two rectangular BFF charts were aligned along the physical S seam
    BEFORE retiling. All common 2D positions at the interface are therefore
    known correspondences at the M2D stage. No arbitrary nearest-neighbor
    seam mismatch is introduced.
    """
    uv=np.asarray(state.mesh_2d_initial.vertices,float)[:,:2]
    faces=np.asarray(state.mesh_2d_initial.faces,int)
    tile=np.asarray(state.tiles_2d_dual_hinge.vertices,float)
    collections=[]
    for patch_id in (reference,child):
        mapping={}
        for tile_id in np.flatnonzero(labels==patch_id):
            for corner,vid in enumerate(faces[tile_id]):
                key=tuple(np.rint(uv[int(vid)]/tol).astype(np.int64))
                mapping.setdefault(key,(int(tile_id),corner))
        collections.append(mapping)
    common=sorted(set(collections[0]).intersection(collections[1]))
    if len(common)<2:
        return None
    a=np.asarray([
        tile[collections[0][key][0],collections[0][key][1],:2]
        for key in common
    ],float)
    b=np.asarray([
        tile[collections[1][key][0],collections[1][key][1],:2]
        for key in common
    ],float)
    if np.linalg.norm(np.ptp(a,axis=0))<1e-8 or np.linalg.norm(np.ptp(b,axis=0))<1e-8:
        return None
    return a,b,len(common)


def _source_matching_anchors(seam_xyz, corner_xyz_a, corner_xyz_b, cutoff):
    """Nearest SAME-3D-source-seam sample for both patches, no fabricated IDs."""
    from scipy.spatial import cKDTree
    tree_a=cKDTree(corner_xyz_a)
    tree_b=cKDTree(corner_xyz_b)
    da,ia=tree_a.query(seam_xyz)
    db,ib=tree_b.query(seam_xyz)
    matches={}
    for distance_a,distance_b,idx_a,idx_b in zip(da,db,ia,ib):
        if max(float(distance_a),float(distance_b))>cutoff:
            continue
        pair=(int(idx_a),int(idx_b))
        cost=float(distance_a+distance_b)
        if pair not in matches or matches[pair]>cost:
            matches[pair]=cost
    # A nearest tile vertex can appear in multiple pairs; keep a unique
    # one-to-one correspondence, sorted by 3D closeness to the seam.
    used_a,used_b=set(),set()
    kept=[]
    for (ia,ib),cost in sorted(matches.items(),key=lambda item:item[1]):
        if ia in used_a or ib in used_b:
            continue
        kept.append((ia,ib,cost))
        used_a.add(ia);used_b.add(ib)
    return kept


def _translate_rotate_island(vertices, tile_ids, rotation, translation):
    """Move entire 8-corner rigid tiles together in XY."""
    moved=vertices[tile_ids].copy()
    moved[:,:,:2]=moved[:,:,:2]@rotation+translation
    vertices[tile_ids]=moved


def align_t2d_at_source_split_seams(
    state, patches, tile_patch_ids, *, max_anchor_distance_factor=1.7
):
    """Align source-adjacent T2D components; report if exact alignment fails.

    Input M3D top corners preserve the physical S mapping; they are used only
    to identify matching seam neighbors, NOT to change tile geometry.
    A spanning tree from the largest tile component anchors all other islands.
    """
    labels=np.asarray(tile_patch_ids,int)
    if labels.shape != (len(state.mesh_2d_initial.faces),):
        raise ValueError("Lost original chart owner correspondence")
    if len(labels)!=len(state.tiles_2d_dual_hinge.vertices):
        raise ValueError("T2D tiles do not match M2D owner IDs")
    if len(patches)<2:
        return {"split_seam_alignment":"not_required","split_seam_pair_count":0,
                "split_seam_anchor_pair_count":0,
                "split_seam_unresolved_patch_count":0,
                "split_seam_matching_certified":True}

    seams=_boundary_point_groups(patches)
    if not seams:
        raise RuntimeError("Cannot align re-BFF T2D: Split S patches have no matching duplicated 3D boundary vertices")
    all_m3d=np.asarray(state.mesh_3d_initial.vertices,float)
    all_faces=np.asarray(state.mesh_3d_initial.faces,int)
    tops=np.asarray(state.tiles_2d_dual_hinge.vertices,float)
    top_hinge=np.asarray(state.tiles_2d_top_hinge.vertices,float)
    if all_faces.shape!=(len(labels),4):
        raise ValueError("M3D requires 4-corner tile faces to locate seam panel correspondence")

    # Local M3D vertex-to-T2D tile mapping: shared M3D vertices may have
    # multiple per-tile rigid poses; keep the tile-local corner index.
    tile_source=all_m3d[all_faces]
    edge_lengths=np.linalg.norm(
        tile_source[:,[1,2,3,0]]-tile_source[:,[0,1,2,3]],axis=2)
    typical=float(np.median(edge_lengths[edge_lengths>1e-10]))
    if not np.isfinite(typical):
        raise RuntimeError("No nondegenerate 3D tile edges for seam registration")
    cutoff=max_anchor_distance_factor*typical
    ids_by_piece={k:np.flatnonzero(labels==k) for k in range(len(patches))}
    for i,ids in ids_by_piece.items():
        if not len(ids):
            raise RuntimeError(f"ReBFF piece {i} lost all tiles before seam alignment")
    # Start with the largest physical piece, and register every adjacent
    # component against that already-placed component.
    root=max(ids_by_piece,key=lambda key:len(ids_by_piece[key]))
    placed={root}
    history=[]
    while len(placed)<len(patches):
        candidates=[]
        for (a,b),xyz in seams.items():
            if (a in placed)==(b in placed):
                continue
            reference,child=(a,b) if a in placed else (b,a)
            ra,rb=ids_by_piece[reference],ids_by_piece[child]
            seam_grid_verified=bool(
                state.mesh_2d_initial.metrics.get(
                    "postcut_seam_shared_grid_vertex_count",0
                )>=2
            ) if hasattr(state.mesh_2d_initial,"metrics") else False
            corners=(
                _matching_shared_grid_corner_anchors(
                    state,labels,reference,child,tol=max(1e-10,typical*1e-8)
                ) if seam_grid_verified else None
            )
            if corners is not None:
                anchor_ref,anchor_child,matching_count=corners
                correspondence_method="shared_original_grid_seam_vertices"
            else:
                matches=_source_matching_anchors(
                    xyz,tile_source[ra].reshape(-1,3),
                    tile_source[rb].reshape(-1,3),cutoff)
                if len(matches)<2:
                    continue
                xy_r=tops[ra,:4,:2].reshape(-1,2)
                xy_b=tops[rb,:4,:2].reshape(-1,2)
                anchor_ref=np.asarray([xy_r[i] for i,j,_ in matches])
                anchor_child=np.asarray([xy_b[j] for i,j,_ in matches])
                matching_count=len(matches)
                correspondence_method="closest_3D_source_tile_corners"
            if np.linalg.norm(np.ptp(anchor_ref,axis=0))<1e-8 or np.linalg.norm(np.ptp(anchor_child,axis=0))<1e-8:
                continue
            rot,delta,rms=_rigid_2d(anchor_child,anchor_ref)
            # Prefer many genuinely corresponding physical seam samples and
            # low residual. No artificial scaling or deformation is allowed.
            candidates.append((matching_count,-rms,reference,child,rot,delta,rms,
                               correspondence_method))
        if not candidates:
            raise RuntimeError(
                "Cannot align re-BFF T2D at the physical Split seam: "
                f"only {len(placed)}/{len(patches)} regions have reliable 3D "
                "boundary/panel correspondences. Independent retiling can "
                "produce incompatible seam panels; exact seam-preserving "
                "M2D retopology is required rather than moving a picture."
            )
        count,_,reference,child,rotation,translation,rms,method=max(
            candidates,key=lambda item:(item[0],item[1]))
        children=ids_by_piece[child]
        for assembly in (state.tiles_2d_dual_hinge,state.tiles_2d_top_hinge):
            _translate_rotate_island(assembly.vertices,children,rotation,translation)
            transforms=getattr(assembly,"transform_matrices",None)
            if transforms is not None:
                for tile_id in children:
                    if transforms.shape[1:] == (4,4):
                        # Compose world XY row-vector operation x'=x@R+t.
                        left=np.eye(4)
                        left[:2,:2]=rotation.T
                        left[:2,3]=translation
                        transforms[tile_id]=left@transforms[tile_id]
        placed.add(child)
        history.append({"reference_piece":int(reference),"moved_piece":int(child),
                        "anchor_count":int(count),
                        "correspondence_method":method,
                        "residual_rms":float(rms)})
    for h in state.hinge_graph.hinges:
        a,b=int(h.tile_a),int(h.tile_b)
        va,vb=int(h.local_vertex_a),int(h.local_vertex_b)
        h.rest_position_2d=.5*(tops[a,va]+tops[b,vb])
    max_rms=max(row["residual_rms"] for row in history) if history else 0.
    # Conformally reBFFed, independently retiled panels have no exact guarantee
    # of a shared seam-grid; never certify based solely on rigid registration.
    return {
        "split_seam_alignment":(
            "shared_source_seam_grid_rigid_SE2"
            if history and all(
                h["correspondence_method"]=="shared_original_grid_seam_vertices"
                for h in history
            ) else "source_surface_3d_matched_rigid_SE2"
        ),
        "split_seam_pair_count":int(len(history)),
        "split_seam_anchor_pair_count":int(sum(row["anchor_count"] for row in history)),
        "split_seam_fit_rms_max":float(max_rms),
        "split_seam_fit_rms_relative_to_tile":float(max_rms/typical),
        "split_seam_matching_certified":False,
        "split_seam_match_limitation":
            "Rigid component alignment only; independent BFF remeshing cannot "
            "guarantee one-to-one matching seam panel vertices or joint topology.",
        "split_seam_registration_steps":history,
    }



def align_rectangular_bff_charts_on_source_seam(patches):
    """Map both independently re-BFFed rectangles onto one shared seam lattice.

    The parent grid Split duplicated the SAME source-S seam in 3D. Each child
    BFF was explicitly given its two physical seam endpoints as adjacent
    rectangle corners. By a conformal similarity (uniform scale + rotation +
    translation), fit these endpoints exactly without changing the CSF ratio.
    Root chart and child then share an axis-aligned seam with identical UV
    grid coordinates. A reflection or per-vertex deformation is prohibited.
    """
    if len(patches) < 2:
        return {
            "postcut_seam_uv_registration":"not_required",
            "postcut_seam_uv_pair_count":0,
        }
    if any(not p.get("seam_rectangle_corners_constrained",False) for p in patches):
        raise RuntimeError(
            "Cannot preserve a common rectangular seam grid: child BFF did "
            "not constrain BOTH split endpoints to rectangle corners."
        )
    seams=_boundary_point_groups(patches)
    if not seams:
        raise RuntimeError("No matching source-S seam between BFF rectangles")
    root=int(np.argmax([len(p["faces"]) for p in patches]))
    raw=[np.asarray(p["new_uv"],float).copy() for p in patches]
    transformed={root:raw[root].copy()}
    used={root}
    history=[]

    # A recursive T-junction can terminate inside an older boundary EDGE.
    # Interpolate the physical point on that source edge instead of snapping
    # to an unrelated nearest vertex of the older re-BFF chart.
    boundary_cache={}
    def get_uv_at_source(patch_index,point,values=None):
        xyz=np.asarray(patches[patch_index]["xyz"],float)
        if patch_index not in boundary_cache:
            from onestring_physics.reference_bff import validate_reference_mesh
            boundary_cache[patch_index]=np.asarray(
                validate_reference_mesh(
                    xyz,np.asarray(patches[patch_index]["faces"],int)
                )["boundary_loops"][0],int
            )
        loop=boundary_cache[patch_index]
        starts=xyz[loop]
        ends=xyz[np.roll(loop,-1)]
        direction=ends-starts
        denom=np.sum(direction*direction,axis=1)
        t=np.clip(np.sum((np.asarray(point)-starts)*direction,axis=1)/
                  np.maximum(denom,1e-30),0.,1.)
        candidate=starts+t[:,None]*direction
        residuals=np.linalg.norm(candidate-np.asarray(point),axis=1)
        nearest=int(np.argmin(residuals))
        tol=1e-7*max(1.,float(np.max(np.ptp(xyz,axis=0))))
        if residuals[nearest]>tol:
            raise RuntimeError(
                "Physical Split seam point is absent from this chart's "
                f"3D boundary (distance={residuals[nearest]:.6g})"
            )
        u=raw[patch_index] if values is None else np.asarray(values,float)
        v0=int(loop[nearest])
        v1=int(loop[(nearest+1)%len(loop)])
        return (1.-t[nearest])*u[v0]+t[nearest]*u[v1]

    while len(used)<len(patches):
        active=[]
        for pair,seam in seams.items():
            a,b=pair
            if (a in used)==(b in used) or len(seam)<2:
                continue
            base,child=(a,b) if a in used else (b,a)
            # Farthest projected UV points delimit the shared RECTANGLE-edge
            # interval, even when its 3D source curve bends across triangles.
            base_seam_uv=np.asarray([
                get_uv_at_source(base,point,transformed[base]) for point in seam
            ])
            distances=np.sum(
                (base_seam_uv[:,None,:]-base_seam_uv[None,:,:])**2,axis=2
            )
            i,j=np.unravel_index(np.argmax(distances),distances.shape)
            if i==j:
                continue
            p,q=seam[i],seam[j]
            target=base_seam_uv[[i,j]]
            source=np.asarray([
                get_uv_at_source(child,p),get_uv_at_source(child,q)
            ])
            va=target[1]-target[0]
            vb=source[1]-source[0]
            na=float(np.linalg.norm(va))
            nb=float(np.linalg.norm(vb))
            if min(na,nb)<=1e-10:
                continue
            c=float(np.dot(vb,va)/(nb*na))
            si=float((vb[0]*va[1]-vb[1]*va[0])/(nb*na))
            R=np.array([[c,si],[-si,c]],float)
            scale=na/nb
            mapped=(raw[child]-source[0])@R*scale+target[0]
            # Test the ACTUAL physical boundary correspondence, including
            # any virtual seam endpoint inside a T-junction boundary edge.
            seam_uv=np.asarray([
                get_uv_at_source(child,point) for point in seam
            ])
            d=np.abs(np.cross(
                source[1]-source[0],seam_uv-source[0]
            ))/nb
            relative=float(np.max(d)/nb)
            if relative>1e-5:
                raise RuntimeError(
                    "BFF did not map the shared source seam onto one rectangle "
                    f"edge (relative deviation={relative:.4g})."
                )
            # Preserve orientation: both rectangular patches must occupy
            # opposite sides of the common edge, otherwise a reflection
            # would be necessary and orientation-sensitive mapping changes.
            tangent=va/na
            root_centroid=np.mean(transformed[base],axis=0)-target[0]
            child_centroid=np.mean(mapped,axis=0)-target[0]
            base_side=float(np.cross(tangent,root_centroid))
            child_side=float(np.cross(tangent,child_centroid))
            if abs(base_side)<1e-10 or abs(child_side)<1e-10:
                raise RuntimeError("Degenerate seam-aligned rectangle width")
            if base_side*child_side>0:
                raise RuntimeError(
                    "Rectangular BFF child charts lie on the same side of "
                    "the source seam; cannot join without an orientation flip"
                )
            active.append((
                len(seam),base,child,mapped,scale,relative,
                target.copy(),
            ))
        if not active:
            raise RuntimeError(
                "Cannot assemble a connected chain of seam-compatible "
                "rectangular BFF charts"
            )
        _,base,child,mapped,scale,deviation,landmarks=max(
            active,key=lambda c:c[0]
        )
        transformed[child]=mapped
        used.add(child)
        history.append({
            "reference_piece":int(base),
            "moving_piece":int(child),
            "uniform_scale_ratio":float(scale),
            "seam_uv_line_deviation_relative":float(deviation),
        })

    # Put the first shared seam at coordinate zero. Both charts share the
    # same absolute square-grid lattice after this common global shift.
    first_pair=tuple(sorted((history[0]["reference_piece"],
                              history[0]["moving_piece"])))
    seam=seams[first_pair]
    end0=seam[0]
    uvzero=get_uv_at_source(root,end0,transformed[root])
    shift=-uvzero
    for i,patch in enumerate(patches):
        patch["new_uv_before_seam_alignment"]=raw[i]
        patch["new_uv"]=transformed[i]+shift
    # Endpoint equality does NOT ensure that the TWO independent BFF solvers
    # assign the same parameter to every internal source-S seam vertex.
    # Record the actual seam-parameter mismatch; the quad meshes can have
    # identical UV boundary edges while their inverse lifts differ in 3D.
    seam_interior_mismatch = {}
    global_max=0.
    for (a,b),points in seams.items():
        ua=np.asarray(patches[a]["new_uv"],float)
        ub=np.asarray(patches[b]["new_uv"],float)
        deviations=[]
        for source_point in points:
            pa=get_uv_at_source(a,source_point,ua)
            pb=get_uv_at_source(b,source_point,ub)
            deviations.append(float(np.linalg.norm(pa-pb)))
        maximum=float(max(deviations)) if deviations else 0.
        global_max=max(global_max,maximum)
        seam_interior_mismatch[f"{a}-{b}"]={
            "sample_count":int(len(deviations)),
            "max_uv_difference":maximum,
            "mean_uv_difference":float(np.mean(deviations)) if deviations else 0.,
        }
    return {
        "postcut_seam_uv_registration":"physical_seam_ends_uniform_BFF_similarity",
        "postcut_seam_interior_correspondence_error_max":global_max,
        "postcut_seam_interior_correspondence_by_pair":seam_interior_mismatch,
        "postcut_seam_uv_full_physical_correspondence_verified":bool(
            global_max <= 1e-8*max(1.,max(
                float(np.max(np.ptp(np.asarray(p["new_uv"],float),axis=0)))
                for p in patches
            ))
        ),
        "postcut_seam_uv_pair_count":len(history),
        "postcut_seam_uv_transform_steps":history,
        "postcut_seam_uv_global_grid_origin":[0.,0.],
        "postcut_seam_uv_preserves_csf_ratio":True,
    }
