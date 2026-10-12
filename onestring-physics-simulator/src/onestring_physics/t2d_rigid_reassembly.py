"""Diagnostics: rigidly place the FABRICATED T2D tiles at designed T3D poses.

This is inverse/kinematic reassembly for geometrical feasibility analysis,
NOT a motion trajectory and NOT a physical demonstration that a string can
actually assemble the object. Each tile is transformed by a proper SE(3)
rigid transform fitted to corresponding T3D's EIGHT vertices. Original
T2D solid shape is never replaced with the T3D geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def proper_rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Transform all source vertices with the optimal det(R)=+1 Kabsch map."""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.shape != (8, 3) or b.shape != (8, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Rigid reassembly requires finite corresponding eight-vertex 3D tiles")
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ma).T @ (b - mb), full_matrices=True)
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    placed = (a - ma) @ r + mb
    return placed, float(np.sqrt(np.mean(np.sum((placed - b)**2, axis=1))))


def _axes(tile: np.ndarray) -> np.ndarray:
    x = tile[1] - tile[0]
    y = tile[3] - tile[0]
    xnorm = np.linalg.norm(x)
    if xnorm < 1e-12:
        return np.eye(3)
    x /= xnorm
    y = y - x * np.dot(x, y)
    ynorm = np.linalg.norm(y)
    if ynorm < 1e-12:
        return np.eye(3)
    y /= ynorm
    z = np.cross(x, y)
    return np.stack((x, y, z))


def obb_overlap_3d_counts(solids: np.ndarray, *, threshold: float = 1e-8) -> dict:
    """Approximate 3D pair contact by 15-axis SAT (not exact mesh collision).

    Hinge-connected pairs are not discarded: this deliberately reports any
    intersecting thick solids, even when the involved tiles share a hinge.
    """
    v = np.asarray(solids, dtype=float)
    if v.ndim != 3 or v.shape[1:] != (8, 3):
        raise ValueError("Expected (n,8,3) solid prism vertices")
    if not len(v):
        return {"pairs":0, "max_penetration":0., "participants":[], "pairs_list":[]}
    centers = v.mean(axis=1)
    radii = np.max(np.linalg.norm(v - centers[:,None,:], axis=-1), axis=1)
    if len(v) < 2:
        return {"pairs":0, "max_penetration":0., "participants":[], "pairs_list":[]}
    candidates = cKDTree(centers).query_pairs(float(2*np.max(radii)+threshold), output_type="ndarray")
    axes = np.stack([_axes(t[:4].copy()) for t in v])
    participants = set()
    collisions = []
    maxdepth = 0.
    for ia,ib in candidates:
        a,b = int(ia),int(ib)
        if np.linalg.norm(centers[a]-centers[b]) > radii[a]+radii[b]+threshold:
            continue
        directions=[*axes[a], *axes[b]]
        directions.extend(np.cross(x,y) for x in axes[a] for y in axes[b])
        separated = False
        mindepth = float("inf")
        for raw in directions:
            length = np.linalg.norm(raw)
            if length < 1e-9:
                continue
            unit = raw / length
            proj_a,proj_b = v[a] @ unit, v[b] @ unit
            depth = min(proj_a.max(),proj_b.max())-max(proj_a.min(),proj_b.min())
            if depth <= threshold:
                separated=True
                break
            mindepth=min(mindepth,float(depth))
        if separated or not np.isfinite(mindepth):
            continue
        maxdepth=max(maxdepth,mindepth)
        participants.update((a,b))
        collisions.append((a,b,round(mindepth,9)))
    return {"pairs":len(collisions), "max_penetration":float(maxdepth),
            "participants":sorted(participants), "pairs_list":collisions}


def rigid_reassemble_t2d(state) -> tuple[np.ndarray, dict]:
    """Return posed T2D solids and diagnostics using *only* T3D tile poses.

    Never modifies any state mesh or physical simulation result. Hinge gaps,
    T3D shape mismatch, and approximate solid collisions are all reported.
    """
    rest=np.asarray(state.tiles_2d_dual_hinge.vertices,dtype=float)
    target=np.asarray(state.tiles_3d.vertices,dtype=float)
    if rest.shape!=target.shape or rest.ndim!=3 or rest.shape[1:]!=(8,3):
        raise ValueError("T2D/T3D tiles must have matching (N,8,3) topology")
    aligned=[]
    tile_fit=[]
    for r,t in zip(rest,target):
        mapped,rms=proper_rigid_fit(r,t)
        aligned.append(mapped)
        tile_fit.append(rms)
    placed=np.asarray(aligned).reshape(rest.shape)
    hinge_errors=[]
    for hinge in getattr(getattr(state,"hinge_graph",None),"hinges",[]):
        a,b=int(hinge.tile_a),int(hinge.tile_b)
        va,vb=int(hinge.local_vertex_a),int(hinge.local_vertex_b)
        if str(getattr(hinge,"surface","top"))=="bottom":
            if va<4: va+=4
            if vb<4: vb+=4
        if (0<=a<len(placed) and 0<=b<len(placed)
                and 0<=va<8 and 0<=vb<8):
            hinge_errors.append(float(np.linalg.norm(placed[a,va]-placed[b,vb])))
    contact=obb_overlap_3d_counts(placed)
    initial_contact=obb_overlap_3d_counts(rest)
    tgt_rms=float(np.sqrt(np.mean(np.sum((placed-target)**2,axis=-1)))) if len(rest) else 0.
    # Confirm invariant Euclidean distances for ALL vertex pairs of each
    # tile: T2D solids are rigidly moved, never replaced by T3D geometry.
    old_dist=np.linalg.norm(rest[:,:,None,:]-rest[:,None,:,:],axis=-1)
    new_dist=np.linalg.norm(placed[:,:,None,:]-placed[:,None,:,:],axis=-1)
    max_internal_distance_change=float(np.max(np.abs(new_dist-old_dist))) if len(rest) else 0.
    return placed,{
        "tile_count":int(len(rest)),
        "per_tile_target_fit_rms":float(np.sqrt(np.mean(np.square(tile_fit)))) if tile_fit else 0.,
        "max_tile_target_fit_rms":float(max(tile_fit)) if tile_fit else 0.,
        "total_target_vertex_rms":tgt_rms,
        "hinge_rms":float(np.sqrt(np.mean(np.square(hinge_errors)))) if hinge_errors else 0.,
        "hinge_max":float(max(hinge_errors)) if hinge_errors else 0.,
        "hinge_count":len(hinge_errors),
        "initial_t2d_3d_obb_pairs":initial_contact["pairs"],
        "reassembled_3d_obb_pairs":contact["pairs"],
        "reassembled_3d_obb_max_depth":contact["max_penetration"],
        "reassembled_3d_obb_participants":contact["participants"],
        "rigid_panels_preserved":bool(max_internal_distance_change<=1e-8),
        "max_internal_distance_change":max_internal_distance_change,
        "kinematic_only":True,
        "contact_method":"approximate 15-axis OBB SAT, not exact polyhedral intersection",
    }
