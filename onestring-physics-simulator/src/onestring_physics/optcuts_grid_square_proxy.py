"""Grid-basis square-panel quality proxy for OptCuts Ω candidate selection.

For each paired surface/UV triangle compute the differential of the inverse
parameterization (UV -> 3D). A locally square input grid maps to a locally
square 3D quad if the two columns of J are perpendicular and have equal norm.
This is a *local differential predictor*, not a substitute for computing
real M3D/K3D quad corner angles. The latter require explicit panel construction.
"""
from __future__ import annotations

import numpy as np


def square_grid_local_quality(source_xyz, source_faces, uv, uv_faces):
    xyz=np.asarray(source_xyz,float)
    xyz_faces=np.asarray(source_faces,int)
    points=np.asarray(uv,float)
    faces=np.asarray(uv_faces,int)
    if (len(faces)==0 or faces.shape!=xyz_faces.shape or
            faces.shape[1]!=3):
        raise ValueError("GRID_SQUARE_PROXY_PAIRED_FACES_REQUIRED")
    a=points[faces]
    b=xyz[xyz_faces]
    du=a[:,1,:]-a[:,0,:]
    dv=a[:,2,:]-a[:,0,:]
    x=b[:,1,:]-b[:,0,:]
    y=b[:,2,:]-b[:,0,:]
    det=du[:,0]*dv[:,1]-du[:,1]*dv[:,0]
    if not np.isfinite(det).all() or np.min(np.abs(det))<1e-15:
        raise ValueError("GRID_SQUARE_PROXY_UV_DEGENERATE")
    # J = [X1-X0, X2-X0] * inverse([u1-u0, u2-u0]).
    ju=(dv[:,1,None]*x-du[:,1,None]*y)/det[:,None]
    jv=(-dv[:,0,None]*x+du[:,0,None]*y)/det[:,None]
    nu=np.linalg.norm(ju,axis=1)
    nv=np.linalg.norm(jv,axis=1)
    if np.any(nu<1e-15) or np.any(nv<1e-15):
        raise ValueError("GRID_SQUARE_PROXY_3D_DEGENERATE")
    cosine=np.sum(ju*jv,axis=1)/(nu*nv)
    log_aspect=np.log(nu/nv)
    # Squared orthogonality and equal-side deviation. Under uniform UV/3D
    # scaling, 0 is perfect and higher is worse.
    score=np.square(cosine)+np.square(log_aspect)
    area=.5*np.linalg.norm(np.cross(x,y),axis=1)
    if not np.isfinite(score).all() or np.any(area<=0):
        raise ValueError("GRID_SQUARE_PROXY_NONFINITE_GEOMETRY")
    weight=area/area.sum()
    order=np.argsort(score)
    percentile=float(score[order[min(
        len(order)-1, np.searchsorted(np.cumsum(weight[order]),.95))]])
    return {
        "square_grid_proxy_mean":float(weight@score),
        "square_grid_proxy_p95":percentile,
        "square_grid_proxy_orthogonality_mean":float(weight@np.square(cosine)),
        "square_grid_proxy_side_ratio_mean":float(weight@np.abs(log_aspect)),
    }


def score_parameterization(chart, *, grid_score=0., square_weight=.5):
    metrics=square_grid_local_quality(
        chart.surface_vertices_3d,chart.surface_faces,
        chart.uv_vertices_2d,chart.uv_faces)
    combined=float(grid_score) + float(square_weight)*(
        metrics["square_grid_proxy_mean"]+.5*metrics["square_grid_proxy_p95"])
    return combined,metrics
