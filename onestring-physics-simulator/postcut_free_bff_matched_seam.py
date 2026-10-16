"""One-cut FREE-boundary official BFF with EXACT shared source-seam UV.

Research hypothesis, NOT a method stated in OneString 2025. Independently
computed free-boundary BFF UVs cannot generally be joined by a 2-D rigid
transform. After one full initial-Grid cut, fit the two *whole* charts with
a conformal similarity, then solve a Dirichlet harmonic correction with
IDENTICAL UV at all shared physical seam vertices. Non-seam boundaries are
free BFF boundaries, not forced rectangles.

Every post-warp triangle and CSF must be checked. A matched UV seam does
not imply matched fabricated T2D panels or a retained immutable grid.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from onestring_physics.reference_bff import validate_reference_mesh


def _twice_area(uv, faces):
    t = np.asarray(uv, float)[np.asarray(faces, int)]
    d = t[:, 1] - t[:, 0]
    e = t[:, 2] - t[:, 0]
    return d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]


def _physical_seam_samples(patches, axis, coordinate):
    dim = {"col": 0, "row": 1}[axis]
    sample_indices = []
    source_uv = []
    source_xyz = []
    for patch in patches:
        xyz = np.asarray(patch["xyz"], float)
        inherited = np.asarray(patch["original_uv"], float)
        loop = np.asarray(
            validate_reference_mesh(xyz, np.asarray(patch["faces"], int))
            ["boundary_loops"][0], int
        )
        scale = max(1., float(np.ptp(inherited, axis=0).max()))
        ids = loop[np.abs(inherited[loop, dim]-coordinate) <= 1e-8*scale]
        if len(ids) < 2:
            raise ValueError("Free BFF source seam has fewer than 2 boundary vertices")
        ids = ids[np.argsort(inherited[ids, 1-dim], kind="stable")]
        sample_indices.append(ids)
        source_uv.append(inherited[ids])
        source_xyz.append(xyz[ids])
    if len(sample_indices[0]) != len(sample_indices[1]):
        raise ValueError(
            "Free BFF seam has unmatched source vertex subdivision; "
            "insert matching seam vertices instead of fabricating 2D correspondence"
        )
    scale = max(1.,float(np.max(np.ptp(np.concatenate(source_xyz), axis=0))))
    mismatch = np.linalg.norm(source_xyz[0]-source_xyz[1], axis=1)
    if np.max(mismatch) > 1e-8*scale:
        raise ValueError(
            f"Free BFF seam source vertices differ by {np.max(mismatch):.6g}; "
            "hard UV matching requires the same physical points"
        )
    return sample_indices, float(np.max(mismatch))


def _chart_with_parent_orientation(patch):
    uv = np.asarray(patch["new_uv"], float).copy()
    f = np.asarray(patch["faces"], int)
    parent_area = _twice_area(patch["original_uv"], f)
    raw_area = _twice_area(uv, f)
    if np.any(np.abs(raw_area) < 1e-16):
        raise ValueError("Free BFF source chart contains degenerate UV triangles")
    parent_sign = np.sign(np.median(parent_area))
    raw_sign = np.sign(np.median(raw_area))
    if not parent_sign or not raw_sign:
        raise ValueError("Free BFF chart orientation is undefined")
    if raw_sign != parent_sign:
        uv[:, 1] *= -1.
    signed = _twice_area(uv,f)
    if np.any(signed * parent_sign <= 0):
        raise ValueError("Free BFF chart contains flipped UV faces before seam fitting")
    return uv


def _fit_similarity(source, target):
    src=np.asarray(source,float); dst=np.asarray(target,float)
    x=src-src.mean(axis=0)
    y=dst-dst.mean(axis=0)
    u,d,vt=np.linalg.svd(x.T@y)
    m=np.eye(2)
    m[-1,-1]=np.linalg.det(u@vt)
    rot=u@m@vt
    scale=float(np.sum(d*np.diag(m))/max(float(np.sum(x*x)),1e-30))
    if scale<=0 or not np.isfinite(scale):
        raise ValueError("Free BFF seam similarity is singular")
    return scale*rot, dst.mean(axis=0)-src.mean(axis=0)@(scale*rot)


def _harmonic_seam_warp(xyz, faces, aligned, seam_ids, targets):
    """Dirichlet seam corrections, with all other boundary vertices fixed."""
    xyz=np.asarray(xyz,float)
    f=np.asarray(faces,int)
    uv=np.asarray(aligned,float)
    seam_ids=np.asarray(seam_ids,int)
    loop=np.asarray(validate_reference_mesh(xyz,f)["boundary_loops"][0],int)
    fixed=np.unique(loop)
    interior=np.setdiff1d(np.arange(len(xyz)),fixed)
    deltas=np.zeros_like(uv)
    deltas[seam_ids]=np.asarray(targets,float)-uv[seam_ids]
    if len(interior):
        # Intrinsic, positive graph weights. The correction is harmonic;
        # free-BFF geometry is retained away from the constrained seam.
        edges={}
        for tri in f:
            for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                a,b=sorted((int(a),int(b)))
                d=float(np.linalg.norm(xyz[a]-xyz[b]))
                if d < 1e-14:
                    raise ValueError("Degenerate source edge in seam warp")
                edges[(a,b)]=1./d
        rows=[]; cols=[]; values=[]
        diag=np.zeros(len(xyz))
        for (a,b),w in edges.items():
            diag[a]+=w;diag[b]+=w
            rows.extend([a,b]);cols.extend([b,a]);values.extend([-w,-w])
        rows.extend(range(len(xyz)))
        cols.extend(range(len(xyz)))
        values.extend(diag.tolist())
        lap=coo_matrix((values,(rows,cols)),shape=(len(xyz),len(xyz))).tocsr()
        rhs=-lap[interior][:,fixed] @ deltas[fixed]
        solution=np.column_stack([
            spsolve(lap[interior][:,interior],rhs[:,axis])
            for axis in (0,1)
        ])
        if not np.isfinite(solution).all():
            raise ValueError("Harmonic free-BFF seam correction did not converge")
        deltas[interior]=solution
    result=uv+deltas
    result[seam_ids]=targets  # pin exact common UV coordinates
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite free-BFF shared-seam solution")
    before=_twice_area(uv,f)
    after=_twice_area(result,f)
    # A flipped/collapsed triangle is NOT accepted even if its area was
    # silently omitted by the original diagnostic field() epsilon cutoff.
    if np.any(after*before <= 0):
        raise ValueError("Seam-constrained free BFF inverted UV triangles")
    if np.any(np.abs(after) < 1e-7*np.abs(before)):
        raise ValueError("Seam-constrained free BFF collapsed UV triangles")
    return result, float(np.max(np.linalg.norm(deltas,axis=1)))


def match_free_bff_split_children(children, axis, coordinate, *, evaluate):
    """Match BOTH children's full physical cut, minimizing WORST actual CSF.

    The independent BFF UVs are left intact as independent_bff_uv and
    independent_bff_post. Only new_uv/post are replaced by the seam-aligned
    atlas. This experimental correction is NOT an independent BFF result.
    """
    if len(children)!=2:
        raise ValueError("Seam-matched free BFF supports one 2-child cut")
    seam_ids, source_error = _physical_seam_samples(
        children, axis, float(coordinate)
    )
    aligned=[_chart_with_parent_orientation(p) for p in children]
    transform,shift=_fit_similarity(
        aligned[1][seam_ids[1]], aligned[0][seam_ids[0]]
    )
    aligned[1]=aligned[1]@transform+shift
    pts_a=aligned[0][seam_ids[0]]
    pts_b=aligned[1][seam_ids[1]]
    raw_rms=float(np.sqrt(np.mean(np.sum((pts_a-pts_b)**2,axis=1))))
    best=None
    rejected=[]
    # Try several common seam shapes: moving ONLY one side is frequently
    # worse than sharing the deformation across the two UV charts.
    for fraction in (0.5,0.25,0.75,0.,1.):
        target=(1.-fraction)*pts_a+fraction*pts_b
        trial=[]
        try:
            for p,uv,idx in zip(children,aligned,seam_ids):
                mapped,max_shift=_harmonic_seam_warp(
                    p["xyz"],p["faces"],uv,idx,target
                )
                post=evaluate(p["xyz"],mapped,p["faces"])
                if not np.isfinite(post["linear_span"]):
                    raise ValueError("Nonfinite seam-constrained CSF")
                trial.append((mapped,post,max_shift))
        except (ValueError,RuntimeError) as exc:
            rejected.append(f"{fraction:.2f}: {type(exc).__name__}: {exc}")
            continue
        score=max(float(row[1]["linear_span"]) for row in trial)
        if best is None or score<best[0]:
            best=(score,float(fraction),trial)
    if best is None:
        raise ValueError(
            "Cannot match physical free BFF seams without inverted or "
            f"collapsed UV triangles. Attempts: {rejected[:5]}"
        )
    worst, fraction, trial=best
    max_seam_error=float(np.max(np.linalg.norm(
        trial[0][0][seam_ids[0]]-trial[1][0][seam_ids[1]],axis=1
    )))
    if max_seam_error>1e-10*max(1.,float(np.max(np.ptp(pts_a,axis=0)))):
        raise AssertionError("Physical seam UV is not matched after correction")
    for p,(uv,post,max_shift) in zip(children,trial):
        p["new_uv"]=uv
        p["post"]=post
        p["postcut_free_seam_aligned"]=True
        p["postcut_free_seam_audit"]={
            "mode":"free_boundary_BFF_shared_physical_seam_dirichlet",
            "seam_sample_count":int(len(seam_ids[0])),
            "max_source_seam_xyz_error":float(source_error),
            "seam_uv_error_max":float(max_seam_error),
            "similarity_fitted_seam_rms_before_warp":raw_rms,
            "chosen_shared_seam_fraction":fraction,
            "max_uv_displacement":max_shift,
            "raw_independent_csf":float(p["independent_bff_post"]["linear_span"]),
            "constrained_csf":float(post["linear_span"]),
            "no_uv_triangle_flips":True,
            "preserves_original_global_grid":False,
            "physical_T2D_seam_alignment_verified":False,
        }
    return children
