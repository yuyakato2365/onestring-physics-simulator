"""Generic sampled BFF conformal scale field on an M2D quad mesh.

Paper Sec.4.1 defines the continuous conformal map c:S->Omega and Sec.4.5
bounds its scale. Interpolating log(area Jacobian) at inverse-mapped M2D
vertices is an explicit finite-element DISCRETIZATION HYPOTHESIS, not an
author-provided method. This module never special-cases an input shape.

The piecewise-linear vertex field is positive; extrema on a quad's
four vertices are valid for its bilinear log-field interpolation.
It cannot certify distortion of the continuous BFF map between vertices.
"""
from __future__ import annotations
import numpy as np


def interpolate_jacobian_to_grid_vertices(
    chart_uv, chart_faces, face_jacobian, grid_vertices, grid_faces,
    *, tolerance=1e-9,
):
    """UV-area-weighted source-vertex log(J) -> containing-triangle grid log(J).

    Returns (quad_lo,quad_hi,diagnostics) in *area* Jacobian units. The caller
    converts to linear CSF via sqrt(max(J)/min(J)). No overlap area floor.
    Strictly errors on unlocatable grid vertices or undefined face samples.
    """
    uv=np.asarray(chart_uv,float)
    tri=np.asarray(chart_faces,int)
    J=np.asarray(face_jacobian,float)
    gv=np.asarray(grid_vertices,float)[:, :2]
    qf=np.asarray(grid_faces,int)
    if tri.ndim!=2 or tri.shape[1]!=3 or qf.ndim!=2 or qf.shape[1]!=4:
        raise ValueError("Expected source triangles and M2D quads")
    if len(J)!=len(tri):
        raise ValueError("Area Jacobian must contain one value per source triangle")
    xy=uv[tri]
    twice_area=np.abs((xy[:,1,0]-xy[:,0,0])*(xy[:,2,1]-xy[:,0,1])
                   -(xy[:,1,1]-xy[:,0,1])*(xy[:,2,0]-xy[:,0,0]))
    usable=(np.isfinite(J)&(J>0)&(twice_area>1e-18))
    if not np.any(usable):
        raise ValueError("No usable conformal Jacobian triangles")
    sums=np.zeros(len(uv),float)
    weights=np.zeros(len(uv),float)
    w=twice_area[usable]
    np.add.at(sums,tri[usable].ravel(),np.repeat(w*np.log(J[usable]),3))
    np.add.at(weights,tri[usable].ravel(),np.repeat(w,3))
    vertex_log=np.divide(sums,weights,out=np.full(len(uv),np.nan),where=weights>0)

    used=np.unique(qf)
    computed=np.full(len(gv),np.nan,float)
    lower=xy.min(axis=1)
    upper=xy.max(axis=1)
    worst_margin=float("inf")
    boundary_count=0
    for vi in used:
        q=gv[int(vi)]
        possible=np.flatnonzero(
            usable
            & np.all(lower<=q+tolerance,axis=1)
            & np.all(upper>=q-tolerance,axis=1)
        )
        if not len(possible):
            raise ValueError(f"M2D vertex {vi} is not covered by a usable UV triangle")
        t=xy[possible]
        a=t[:,0]
        u=t[:,1]-a
        v=t[:,2]-a
        d=q-a
        det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        good=np.abs(det)>1e-18
        if not np.any(good):
            raise ValueError(f"All overlapping triangles degenerate near M2D vertex {vi}")
        possible=possible[good];u=u[good];v=v[good];d=d[good];det=det[good]
        b1=(d[:,0]*v[:,1]-d[:,1]*v[:,0])/det
        b2=(u[:,0]*d[:,1]-u[:,1]*d[:,0])/det
        bary=np.column_stack([1.-b1-b2,b1,b2])
        margins=bary.min(axis=1)
        candidates=np.flatnonzero(margins>=-tolerance)
        if not len(candidates):
            raise ValueError(
                f"M2D vertex {vi} is not inside a UV triangle: best margin={margins.max():.4g}"
            )
        j=int(candidates[np.argmax(margins[candidates])])
        if not np.isfinite(vertex_log[tri[possible[j]]]).all():
            raise ValueError("Undefined interpolated source-vertex Jacobian")
        computed[int(vi)]=float(np.exp(
            np.dot(bary[j],vertex_log[tri[possible[j]]])
        ))
        worst_margin=min(worst_margin,float(margins[j]))
        if margins[j]<1e-7: boundary_count+=1

    face_values=computed[qf]
    if not np.isfinite(face_values).all() or not np.all(face_values>0):
        raise ValueError("Interpolated M2D Jacobian is invalid")
    lo=face_values.min(axis=1)
    hi=face_values.max(axis=1)
    return lo,hi,dict(
        source="source UV-area-weighted vertex logJ; "
               "barycentric interpolation at M2D grid vertices; "
               "bilinear quad logJ",
        m2d_samples=int(len(used)),
        triangles_used=int(usable.sum()),
        mean_jacobian=float(np.mean(face_values)),
        max_quad_linear_csf=float(np.max(np.sqrt(hi/lo))),
        max_chart_linear_csf=float(np.sqrt(hi.max()/lo.min())),
        min_triangle_barycentric=float(worst_margin),
        near_uv_triangle_edge_sample_count=int(boundary_count),
        certified_continuous_bff_field=False,
    )
