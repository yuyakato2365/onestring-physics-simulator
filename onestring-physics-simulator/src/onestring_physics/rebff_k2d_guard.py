"""Conservative K2D fold-over guard for experimental reBFF manufacturing M2D.

Never modify the CSF/Split decision. K2D's unconstrained edge-length solver
may fold/bow-tie squares when its reference M2D coordinates were reBFF-warped,
especially on large meshes where collision relaxation was skipped.
Backtrack K2D displacements toward the valid M2D chart rather than
silently accepting an inverted or mutually intersecting panel layout.
Report both the original solver error and guarded edge error.
"""
from __future__ import annotations

import numpy as np


def _face_turns(xy, faces):
    p=np.asarray(xy,float)[np.asarray(faces,int)]
    a=np.roll(p,-1,axis=1)-p
    b=np.roll(p,-2,axis=1)-np.roll(p,-1,axis=1)
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _areas(xy, faces):
    p=np.asarray(xy,float)[np.asarray(faces,int)]
    x,y=p[...,0],p[...,1]
    return .5*(x*np.roll(y,-1,axis=1)-y*np.roll(x,-1,axis=1)).sum(axis=1)


def _convex_and_oriented(xy, faces, reference_signs, eps):
    area=_areas(xy,faces)
    turns=_face_turns(xy,faces)
    signs=reference_signs[:,None]
    return bool(np.isfinite(xy).all() and
                np.all(area*reference_signs>eps) and
                np.all(turns*signs>eps))


def _sat_positive_overlap(a,b,eps):
    # Both faces must already be strictly convex. Shared-edge contacts do
    # not register: the projected positive overlap must exist on every axis.
    for p in (a,b):
        e=np.roll(p,-1,axis=0)-p
        axes=np.stack((-e[:,1],e[:,0]),axis=1)
        for axis in axes:
            n=float(np.linalg.norm(axis))
            if n<=1e-15:
                return True
            proj_a=a@(axis/n)
            proj_b=b@(axis/n)
            overlap=min(float(proj_a.max()),float(proj_b.max()))-max(
                float(proj_a.min()),float(proj_b.min()))
            if overlap<=eps:
                return False
    return True


def count_overlaps(xy, faces, eps=1e-9, stop_after=None):
    """Count positive-area intersections between any two distinct quads.

    Vectorized AABB broadphase avoids exact SAT for distant panels. Adjacent
    quads only sharing an edge or vertex are not counted.
    """
    polygons=np.asarray(xy,float)[np.asarray(faces,int)]
    low=polygons.min(axis=1)
    high=polygons.max(axis=1)
    count=0
    for i in range(len(polygons)-1):
        indexes=np.flatnonzero(
            np.all(low[i+1:] < high[i]-eps,axis=1) &
            np.all(high[i+1:] > low[i]+eps,axis=1)
        )+i+1
        for j in indexes:
            if _sat_positive_overlap(polygons[i],polygons[j],eps):
                count+=1
                if stop_after is not None and count>=stop_after:
                    return count
    return count


def _edges(faces):
    edges=set()
    for face in np.asarray(faces,int):
        for i in range(len(face)):
            edges.add(tuple(sorted((int(face[i]),int(face[(i+1)%len(face)])))))
    return np.asarray(sorted(edges),int).reshape(-1,2)


def _edge_errors(xy, edge_ids, targets):
    if not len(edge_ids):
        return 0.,0.
    differences=np.abs(np.linalg.norm(
        xy[edge_ids[:,0]]-xy[edge_ids[:,1]],axis=1)-targets)
    return float(differences.mean()),float(differences.max())


def guard_rebff_k2d(mesh_2d, mesh_3d, k2d, *, minimum_fraction=1./512.):
    """Return guarded XY and transparent diagnostics, or hard error.

    This is invoked only when mesh_2d.metrics.m2d_postcut_rebff_applied.
    The original M2D must already be convex/non-overlapping; otherwise the
    bad geometry originates BEFORE K2D and must not be hidden by K2D.
    """
    original=np.asarray(mesh_2d.vertices,float)[:,:2]
    final=np.asarray(k2d.vertices,float)[:,:2]
    faces=np.asarray(mesh_2d.faces,int)
    if original.shape!=final.shape or not np.array_equal(faces,k2d.faces):
        raise RuntimeError("K2D_REBFF_TOPOLOGY_MISMATCH: optimizer changed source vertices/faces")
    if not len(faces):
        return final.copy(),dict(k2d_rebff_guard_applied=False)
    areas=_areas(original,faces)
    signs=np.sign(areas)
    tile_scale=max(float(np.median(np.sqrt(np.abs(areas)))),1e-9)
    eps=max(1e-12,1e-7*tile_scale*tile_scale)
    if not _convex_and_oriented(original,faces,signs,eps):
        raise RuntimeError(
            "K2D_REBFF_INPUT_INVALID: reBFF M2D has a non-convex, "
            "flipped or degenerate quad; K2D cannot repair its input automatically")
    base_overlap=count_overlaps(original,faces,eps=1e-7*tile_scale,stop_after=1)
    if base_overlap:
        raise RuntimeError(
            "K2D_REBFF_INPUT_OVERLAP: reBFF M2D panels already overlap before K2D; "
            "fix per-piece BFF alignment or panel packing first")
    edge_ids=_edges(faces)
    target=np.linalg.norm(
        np.asarray(mesh_3d.vertices,float)[edge_ids[:,0]]-
        np.asarray(mesh_3d.vertices,float)[edge_ids[:,1]],axis=1)
    starting_error,starting_max=_edge_errors(original,edge_ids,target)
    solver_error,solver_max=_edge_errors(final,edge_ids,target)
    original_overlaps=int(count_overlaps(final,faces,eps=1e-7*tile_scale,stop_after=2)) if (
        _convex_and_oriented(final,faces,signs,eps)) else -1

    def acceptable(alpha):
        trial=original+alpha*(final-original)
        if not _convex_and_oriented(trial,faces,signs,eps):
            return False,None
        if count_overlaps(trial,faces,eps=1e-7*tile_scale,stop_after=1):
            return False,None
        mean_error,max_error=_edge_errors(trial,edge_ids,target)
        if mean_error>starting_error+max(1e-9,1e-5*starting_error):
            return False,None
        return True,(trial,mean_error,max_error)

    selected_fraction=None
    selected=None
    alpha=1.
    while alpha>=minimum_fraction-1e-12:
        ok,data=acceptable(alpha)
        if ok:
            selected_fraction=alpha
            selected=data
            break
        alpha*=.5
    if selected is None:
        if starting_error<=1e-8:
            selected_fraction=0.
            selected=(original.copy(),starting_error,starting_max)
        else:
            raise RuntimeError(
                "K2D_REBFF_GEOMETRY_INFEASIBLE: no fold-free, overlap-free "
                "interpolation from reBFF M2D to the K2D solver output "
                "improves the K3D edge error. ReBFF M2D is not compatible "
                "with the current K2D solver/targets.")
    result,guarded_mean,guarded_max=selected
    return result,dict(
        k2d_rebff_guard_applied=True,
        k2d_rebff_guard_step_fraction=float(selected_fraction),
        k2d_rebff_guard_clipped=bool(selected_fraction<1.),
        k2d_rebff_guard_original_solver_overlap_count=original_overlaps,
        k2d_rebff_guard_start_mean_edge_error=float(starting_error),
        k2d_rebff_guard_start_max_edge_error=float(starting_max),
        k2d_rebff_guard_solver_mean_edge_error=float(solver_error),
        k2d_rebff_guard_solver_max_edge_error=float(solver_max),
        k2d_rebff_guard_final_mean_edge_error=float(guarded_mean),
        k2d_rebff_guard_final_max_edge_error=float(guarded_max),
        k2d_rebff_guard_overlap_count=0,
        k2d_rebff_guard_geometry_valid=True,
        k2d_rebff_guard_certifies_manufacturing=False,
    )
