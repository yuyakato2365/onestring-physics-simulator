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
    """Source 3D seam vertices with a copy in >=2 *distinct* chart pieces."""
    all_x = np.concatenate([np.asarray(p["xyz"], float) for p in patches])
    scale = max(1., float(np.max(np.ptp(all_x, axis=0))))
    tol = 1e-7 * scale
    buckets = defaultdict(list)
    from onestring_physics.reference_bff import validate_reference_mesh
    for pid, patch in enumerate(patches):
        xyz = np.asarray(patch["xyz"], float)
        faces = np.asarray(patch["faces"], int)
        topology = validate_reference_mesh(xyz, faces)
        for vid in topology["boundary_loops"][0]:
            point = xyz[int(vid)]
            key = tuple(np.rint(point / tol).astype(np.int64))
            buckets[key].append((pid,point))
    by_pair = defaultdict(list)
    for entry in buckets.values():
        ids = {pid for pid,_ in entry}
        if len(ids) < 2:
            continue
        center = np.mean([point for _,point in entry], axis=0)
        if any(np.linalg.norm(point-center)>2*tol for _,point in entry):
            continue  # spatial-hash collision, never invent a seam
        for a in sorted(ids):
            for b in sorted(ids):
                if a<b:
                    by_pair[(a,b)].append(center)
    return {pair:np.asarray(points,float) for pair,points in by_pair.items()}


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
            matches=_source_matching_anchors(
                xyz,tile_source[ra].reshape(-1,3),
                tile_source[rb].reshape(-1,3),cutoff)
            if len(matches)<2:
                continue
            # Use the same tile-local corner identity in 3D and 2D.
            xy_r=tops[ra].reshape(-1,8,3)[:,:4,:2].reshape(-1,2)
            xy_b=tops[rb].reshape(-1,8,3)[:,:4,:2].reshape(-1,2)
            anchor_ref=np.asarray([xy_r[i] for i,j,_ in matches])
            anchor_child=np.asarray([xy_b[j] for i,j,_ in matches])
            if np.linalg.norm(np.ptp(anchor_ref,axis=0))<1e-8 or np.linalg.norm(np.ptp(anchor_child,axis=0))<1e-8:
                continue
            rot,delta,rms=_rigid_2d(anchor_child,anchor_ref)
            # Prefer many genuinely corresponding physical seam samples and
            # low residual. No artificial scaling or deformation is allowed.
            candidates.append((len(matches),-rms,reference,child,rot,delta,rms))
        if not candidates:
            raise RuntimeError(
                "Cannot align re-BFF T2D at the physical Split seam: "
                f"only {len(placed)}/{len(patches)} regions have reliable 3D "
                "boundary/panel correspondences. Independent retiling can "
                "produce incompatible seam panels; exact seam-preserving "
                "M2D retopology is required rather than moving a picture."
            )
        _,_,reference,child,rotation,translation,rms=max(
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
                        "anchor_count":int(_source_matching_anchors(
                            seams[tuple(sorted((reference,child)))],
                            tile_source[ids_by_piece[reference]].reshape(-1,3),
                            tile_source[ids_by_piece[child]].reshape(-1,3),cutoff
                        ).__len__()),
                        "residual_rms":float(rms)})
    for h in state.hinge_graph.hinges:
        a,b=int(h.tile_a),int(h.tile_b)
        va,vb=int(h.local_vertex_a),int(h.local_vertex_b)
        h.rest_position_2d=.5*(tops[a,va]+tops[b,vb])
    max_rms=max(row["residual_rms"] for row in history) if history else 0.
    # Conformally reBFFed, independently retiled panels have no exact guarantee
    # of a shared seam-grid; never certify based solely on rigid registration.
    return {
        "split_seam_alignment":"source_surface_3d_matched_rigid_SE2",
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
