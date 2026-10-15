"""Experimental seam-constrained rectangular atlas from independently solved BFFs.

Independent rectangular BFF cannot, in general, give identical source-seam
parameters in every recursively cut patch. This adapter keeps each child's
EXACT inherited global rectangular boundary coordinates (including the seam
samples), then minimizes interior distortion toward a re-BFF target with a
fixed-boundary sparse harmonic solve. This is NOT an independent conformal BFF;
its ACTUAL mapped 3D/UV area-distortion (CSF) must be measured separately.

The sewn boundary must never be moved merely to make two UV pictures touch.
"""
from __future__ import annotations

import numpy as np


def seam_constrained_rectangular_atlas(xyz, faces, inherited_uv, raw_bff_uv):
    """Return continuous atlas UV, with every child boundary vertex fixed.

    inherited_uv is the parent's shared chart clipped by the straight-grid
    split. Consequently both children (and later descendants) inherit exactly
    the SAME UV at each physical Split-seam point, even at a T-junction.
    Independent BFF provides an interior target; all seam points are fixed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from onestring_physics.reference_bff import validate_reference_mesh

    xyz=np.asarray(xyz,float)
    faces=np.asarray(faces,int)
    old=np.asarray(inherited_uv,float)
    raw=np.asarray(raw_bff_uv,float)
    if old.shape!=raw.shape or old.shape!=(len(xyz),2):
        raise ValueError("Joint atlas requires identical source vertices in the raw BFF")
    if not np.isfinite(old).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite UV in independent BFF or inherited atlas")

    loop=np.asarray(validate_reference_mesh(xyz,faces)["boundary_loops"][0],int)
    boundary=np.unique(loop)
    interior=np.setdiff1d(np.arange(len(xyz)),boundary)
    # Rotate/scale (and, if required, anisotropically fit) the BFF rectangle
    # against all inherited boundary landmarks before harmonic extension.
    bff_hom=np.column_stack((raw,np.ones(len(raw))))
    linear=np.linalg.lstsq(bff_hom[boundary],old[boundary],rcond=None)[0]
    fitted=bff_hom@linear

    adjusted=old.copy()
    if len(interior):
        weights={}
        for a,b,c in faces:
            p,q,r=xyz[[a,b,c]]
            double_area=float(np.linalg.norm(np.cross(q-p,r-p)))
            if double_area<1e-20:
                raise ValueError("Degenerate source triangle in joint atlas")
            # Intrinsic positive cotangent weights; clipping prevents a
            # non-Delaunay source triangle from creating a non-M-matrix.
            for u,v,s,t in ((b,c,q-p,r-p),
                            (c,a,r-q,p-q),
                            (a,b,p-r,q-r)):
                cot=float(np.dot(s,t)/double_area)
                key=(min(int(u),int(v)),max(int(u),int(v)))
                weights[key]=weights.get(key,0.)+max(cot*0.5,1e-5)
        row=[];col=[];data=[]
        diagonal=np.zeros(len(xyz))
        for (a,b),w in weights.items():
            diagonal[a]+=w
            diagonal[b]+=w
            row.extend((a,b));col.extend((b,a));data.extend((-w,-w))
        row.extend(range(len(xyz)));col.extend(range(len(xyz)))
        data.extend(diagonal.tolist())
        lap=coo_matrix((data,(row,col)),shape=(len(xyz),len(xyz))).tocsr()
        delta=old[boundary]-fitted[boundary]
        rhs=-lap[interior][:,boundary]@delta
        correction=np.column_stack([
            spsolve(lap[interior][:,interior],rhs[:,axis])
            for axis in range(2)
        ])
        if not np.isfinite(correction).all():
            raise ValueError("Joint atlas harmonic interior solve diverged")
        trial=fitted.copy()
        trial[boundary]=old[boundary]
        trial[interior]=fitted[interior]+correction
        # Maintain triangle orientation. A raw BFF with inverted or
        # extreme interior vertices cannot be allowed to fold the atlas.
        def twice_area(uv):
            a,b,c=uv[faces[:,0]],uv[faces[:,1]],uv[faces[:,2]]
            u=b-a;v=c-a
            return u[:,0]*v[:,1]-u[:,1]*v[:,0]
        reference=twice_area(old)
        if np.any(abs(reference)<1e-15*max(1.,float(np.ptp(old,axis=0).max())**2)):
            raise ValueError("Degenerate inherited UV triangle in joint atlas")
        blend=1.
        for _ in range(21):
            candidate=old+blend*(trial-old)
            actual=twice_area(candidate)
            if np.all(actual*reference>=0.001*reference**2):
                adjusted=candidate
                break
            blend*=0.5
        else:
            blend=0.
            adjusted=old.copy()
    else:
        blend=0.

    if not np.array_equal(adjusted[boundary],old[boundary]):
        raise AssertionError("Physical seam boundary moved during registration")
    info={
        "postcut_joint_atlas":True,
        "postcut_joint_atlas_policy":"fixed_parent_rectangular_boundary_harmonic_BFF_interior",
        "postcut_joint_atlas_bff_interior_blend":float(blend),
        "postcut_joint_atlas_max_shift_from_parent":float(
            np.max(np.linalg.norm(adjusted-old,axis=1))
        ),
        "postcut_joint_atlas_boundary_sample_count":int(len(boundary)),
        "postcut_joint_atlas_conformality_preserved":False,
    }
    return adjusted,info


def verify_joint_source_seams(patches, *, relative_tolerance=1e-7):
    """Verify ALL true physical seams, including partial T-junction overlaps."""
    from onestring_physics.reference_bff import validate_reference_mesh
    from postcut_rebff_seam_registration import _boundary_point_groups

    seams=_boundary_point_groups(patches)
    overall=0.
    evidence={}
    for (a,b),physical in seams.items():
        uv_at=[]
        for index in (a,b):
            patch=patches[index]
            xyz=np.asarray(patch["xyz"],float)
            uv=np.asarray(patch["new_uv"],float)
            loop=np.asarray(
                validate_reference_mesh(xyz,np.asarray(patch["faces"],int))
                ["boundary_loops"][0],int
            )
            edge_a=xyz[loop];edge_b=xyz[np.roll(loop,-1)]
            directions=edge_b-edge_a
            denom=np.sum(directions**2,axis=1)
            values=[]
            for p in physical:
                fractions=np.clip(
                    np.sum((p-edge_a)*directions,axis=1)/
                    np.maximum(denom,1e-30),0.,1.,
                )
                projected=edge_a+fractions[:,None]*directions
                distance=np.linalg.norm(projected-p,axis=1)
                nearest=int(np.argmin(distance))
                scale=max(1.,float(np.max(np.ptp(xyz,axis=0))))
                if distance[nearest]>relative_tolerance*scale:
                    raise RuntimeError(
                        f"Source split seam {a}-{b} has no matching 3D boundary segment"
                    )
                uv0=uv[loop[nearest]]
                uv1=uv[loop[(nearest+1)%len(loop)]]
                values.append(uv0+(uv1-uv0)*fractions[nearest])
            uv_at.append(np.asarray(values,float))
        difference=np.linalg.norm(uv_at[0]-uv_at[1],axis=1)
        maximum=float(np.max(difference))
        overall=max(overall,maximum)
        evidence[f"{a}-{b}"]={
            "source_seam_samples":int(len(physical)),
            "max_uv_difference":maximum,
        }
    span=max(1.,max(float(np.max(np.ptp(np.asarray(p["new_uv"],float),axis=0)))
                     for p in patches))
    verified=overall<=1e-8*span
    if not verified:
        raise RuntimeError(
            "Joint seam-constrained atlas itself failed source-to-UV "
            f"correspondence: {evidence}"
        )
    fixed = all(p.get("fixed_initial_bff", False) for p in patches)
    return {
        "postcut_seam_uv_registration":("initial_BFF_restriction" if fixed else "joint_parent_atlas_fixed_boundary"),
        "postcut_seam_uv_full_physical_correspondence_verified":True,
        "postcut_seam_interior_correspondence_error_max":float(overall),
        "postcut_seam_interior_correspondence_by_pair":evidence,
        "postcut_seam_uv_pair_count":len(seams),
        "postcut_joint_atlas_active":True,
        "postcut_seam_uv_preserves_csf_ratio":fixed,
        "postcut_joint_atlas_csf_must_be_rechecked":True,
    }
