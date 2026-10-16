"""Controlled Lilium experiment: multiple exact-free-BFF-seam repair models.

This is an EXPLORATORY sensitivity test, not the OneString authors' method.
One initial official rectangular BFF, one curvature-grid Split, and two
independent official free-boundary BFFs are provided as input. We compare:

* keeping the original shared-parent UV (no independent free BFF);
* independent free BFF with rigid/similarity registration (seam NOT exact);
* fixing all other boundary vertices and harmonically correcting the seam;
* allowing the other boundary vertices to slide, with different Laplacian
  regularization (seam exact);
* independently chosen seam fractions and a physically straight parent seam;
* least-squares affine versus conformal similarity chart registration.

Never claim a physical auxetic bound from Jacobian area span. The test refuses
UV face flips and reports separate mesh and grid compatibility limitations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

from postcut_free_bff_matched_seam import (
    _chart_with_parent_orientation, _fit_similarity, _physical_seam_samples,
    _twice_area,
)
from onestring_physics.reference_bff import validate_reference_mesh


def _affine_match(source, target):
    x=np.column_stack((np.asarray(source, float), np.ones(len(source))))
    y=np.asarray(target, float)
    fit=np.linalg.lstsq(x, y, rcond=None)[0]
    linear=fit[:2]
    if np.linalg.det(linear)<=1e-10:
        raise ValueError("Affine seam fit must be nonsingular, orientation preserving")
    return linear, fit[2]


def _build_laplacian(xyz, faces):
    xyz=np.asarray(xyz,float)
    f=np.asarray(faces,int)
    diag=np.zeros(len(xyz),float)
    edges={}
    for tri in f:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            a,b=sorted((int(a),int(b)))
            length=float(np.linalg.norm(xyz[a]-xyz[b]))
            if length<=1e-14:
                raise ValueError("Degenerate 3D mesh edge")
            edges[(a,b)]=1./length
    row=[];col=[];data=[]
    for (a,b),w in edges.items():
        diag[a]+=w; diag[b]+=w
        row.extend([a,b]);col.extend([b,a]);data.extend([-w,-w])
    row.extend(range(len(xyz)));col.extend(range(len(xyz)))
    data.extend(diag.tolist())
    mat=coo_matrix((data,(row,col)),shape=(len(xyz),len(xyz))).tocsr()
    return mat,float(np.median(diag))


class SeamHarmonicCorrector:
    def __init__(self, patch, seam_ids, policy, regularizer=0.0):
        self.xyz=np.asarray(patch["xyz"],float)
        self.faces=np.asarray(patch["faces"],int)
        self.seam_ids=np.asarray(seam_ids,int)
        self.policy=policy
        boundary=np.unique(np.asarray(validate_reference_mesh(
            self.xyz,self.faces
        )["boundary_loops"][0],int))
        self.fixed=(
            boundary if policy=="fixed_outer_boundary" else self.seam_ids
        )
        self.unknown=np.setdiff1d(np.arange(len(self.xyz)),self.fixed)
        lap,typical=_build_laplacian(self.xyz,self.faces)
        # Positive mass anchors displacement toward unwarped free BFF.
        # This is a convex proxy, not a distortion-minimizing ARAP solver.
        strength=float(regularizer)*typical
        matrix=lap+eye(len(self.xyz),format="csr")*strength
        self.edge=matrix[self.unknown][:,self.fixed].tocsc()
        self.factor=(splu(matrix[self.unknown][:,self.unknown].tocsc())
                     if len(self.unknown) else None)

    def warp(self, aligned, target):
        uv=np.asarray(aligned,float)
        target=np.asarray(target,float)
        delta=np.zeros_like(uv)
        delta[self.seam_ids]=target-uv[self.seam_ids]
        if len(self.unknown):
            rhs=-self.edge@delta[self.fixed]
            delta[self.unknown]=np.column_stack((
                self.factor.solve(rhs[:,0]),self.factor.solve(rhs[:,1])
            ))
        mapped=uv+delta
        mapped[self.seam_ids]=target
        before=_twice_area(uv,self.faces)
        after=_twice_area(mapped,self.faces)
        invalid=~np.isfinite(after)
        sign_bad=(after*before<=0)
        low=np.abs(after)/np.maximum(np.abs(before),1e-30)
        if np.any(invalid | sign_bad) or np.min(low)<1e-7:
            return None, dict(
                reason="flip_or_tiny_UV_triangles",
                num_flipped=int(np.count_nonzero(sign_bad)),
                min_signed_triangle_area_ratio=float(np.min(low)),
            )
        return mapped, dict(
            max_displacement=float(np.linalg.norm(delta,axis=1).max()),
            min_signed_triangle_area_ratio=float(np.min(low)),
            num_flipped=0,
        )


def _stats(uv, patch, evaluate):
    report=evaluate(patch["xyz"],uv,patch["faces"])
    return {
        "csf":float(report["linear_span"]),
        "csf_pct_1_99":float(report["linear_span_pct1_99"]),
        "area_span":float(report["area_span"]),
        "jac_min":float(report["jac_min"]),
        "jac_max":float(report["jac_max"]),
    }


def _seam_residual(uvs, ids):
    a=np.asarray(uvs[0],float)[ids[0]]
    b=np.asarray(uvs[1],float)[ids[1]]
    lengths=np.linalg.norm(a-b,axis=1)
    return float(np.max(lengths)),float(np.sqrt(np.mean(lengths**2)))


def _straight_seam(points):
    uv=np.asarray(points,float)
    # Straight edge parameterized by cumulative *physical* UV distances
    # on the inherited parent, so it does not assume equidistant sampling.
    distances=np.linalg.norm(np.diff(uv,axis=0),axis=1)
    cumulative=np.r_[0.,np.cumsum(distances)]
    ratio=cumulative/max(float(cumulative[-1]),1e-20)
    return uv[0]+ratio[:,None]*(uv[-1]-uv[0])


def sweep_free_bff_shared_seam(patches, axis, coordinate, *, evaluate):
    """Return data-only comparison; does NOT modify patches or app globals."""
    if len(patches)!=2:
        raise ValueError("One curvature Grid Split must yield exactly two patches")
    seam_ids,source_error=_physical_seam_samples(patches,axis,coordinate)
    original=[_chart_with_parent_orientation(p) for p in patches]
    raw=[p["independent_bff_post"] for p in patches]
    records=[]

    def record(name, uvs, *, exact, warp_meta=None, registration=None):
        if uvs is None:
            records.append(dict(name=name,valid=False,seam_exact=False,
                                reason=str(warp_meta)))
            return
        maxerr,rmserr=_seam_residual(uvs,seam_ids)
        result=[_stats(uv,p,evaluate) for uv,p in zip(uvs,patches)]
        worst=max(q["csf"] for q in result)
        records.append(dict(
            name=name,valid=True,seam_exact=bool(maxerr<1e-9),
            uv_seam_max_error=maxerr,uv_seam_rms_error=rmserr,
            csf_max=worst,csf_A=result[0]["csf"],csf_B=result[1]["csf"],
            csf_robust_max=max(q["csf_pct_1_99"] for q in result),
            component_metrics=result,warps=warp_meta,
            registration=registration,
            csf_linear_threshold_2_pass=bool(worst<=2+1e-10),
            physical_grid_compatibility_proven=False,
        ))

    record("parent_original_UV_exact_seam",
           [p["original_uv"] for p in patches],exact=True)
    record("independent_free_BFF_unregistered",original,exact=False)
    correctors={}
    for registration in ("similarity","affine"):
        a=original[0].copy()
        t,s=(_fit_similarity(original[1][seam_ids[1]],
                             a[seam_ids[0]]) if registration=="similarity"
             else _affine_match(original[1][seam_ids[1]],
                                a[seam_ids[0]]))
        b=original[1]@t+s
        if registration=="affine":
            # A general affine map changes singular values, so its CSF
            # must be measured on transformed coordinates too.
            before=_twice_area(b,patches[1]["faces"])
            if np.any(before*_twice_area(original[1],patches[1]["faces"])<=0):
                record("affine_registration_invalid",None,exact=False,
                       warp_meta="orientation_flip")
                continue
        aligned=[a,b]
        record(registration+"_only_seam_not_exact",aligned,exact=False,
               registration=registration)
        pts_a=a[seam_ids[0]]
        pts_b=b[seam_ids[1]]
        seam_targets={
            "avg0":pts_a,
            "avg025":0.75*pts_a+0.25*pts_b,
            "avg05":0.5*(pts_a+pts_b),
            "avg075":0.25*pts_a+0.75*pts_b,
            "avg1":pts_b,
        }
        # A straight seam is the original-grid hypothesis; it is NOT the
        # raw free-BFF parameterization any more.
        line=_straight_seam(np.asarray(patches[0]["original_uv"])[seam_ids[0]])
        m,c=_fit_similarity(line,0.5*(pts_a+pts_b))
        seam_targets["straight_parent_grid"] = line@m+c

        # Fixed boundary (previous experiment), and boundary-sliding
        # Laplacian correction with varying mass regularization strength.
        policies=[
            ("fixed_outer_boundary",0.0),
            ("sliding_outer_boundary",0.0),
            ("sliding_outer_boundary",0.01),
            ("sliding_outer_boundary",0.1),
            ("sliding_outer_boundary",1.0),
        ]
        for policy,mu in policies:
            for i,p in enumerate(patches):
                key=(i,policy,mu)
                if key not in correctors:
                    correctors[key]=SeamHarmonicCorrector(
                        p,seam_ids[i],policy,mu
                    )
            for target_label, target in seam_targets.items():
                uvs=[]
                info=[]
                for i in range(2):
                    corrected,meta=correctors[(i,policy,mu)].warp(
                        aligned[i],target
                    )
                    info.append(meta)
                    if corrected is None:
                        break
                    uvs.append(corrected)
                name=f"{registration}|{policy}|mu={mu:.3g}|{target_label}"
                if len(uvs)!=2:
                    record(name,None,exact=True,warp_meta=info,
                           registration=registration)
                else:
                    record(name,uvs,exact=True,warp_meta=info,
                           registration=registration)
    valid=[r for r in records if r["valid"] and r["seam_exact"]]
    valid.sort(key=lambda r:r["csf_max"])
    return dict(
        experiment="one_initial_grid_cut_free_BFF_seam_alignment_hypotheses",
        source_seam_points=int(len(seam_ids[0])),
        source_xyz_max_error=float(source_error),
        candidates_count=len(records),
        valid_exact_seam_count=len(valid),
        count_below_linear_csf_2=sum(r["csf_max"]<=2.+1e-10 for r in valid),
        best_exact_seam=valid[0] if valid else None,
        best_five_exact_seam=valid[:5],
        all_candidates=records,
        notes=[
            "Free-boundary BFF was solved independently for both patches.",
            "Exact seam means identical UV at duplicated 3D seam samples.",
            "Any UV after seam warping is not an independent conformal BFF.",
            "Maximum CSF is sqrt(max(area3/area2)/min(area3/area2)), not the "
            "quad auxetic closed/open area expansion bound of Figure 18.",
            "No initial Omega Grid or actual final T2D hinge-seam matching proven.",
            "Geometric UV overlap between patch interiors not certified.",
        ],
    )


def write_sweep_json(sweep, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(sweep,indent=2,ensure_ascii=False))
    return str(path)
