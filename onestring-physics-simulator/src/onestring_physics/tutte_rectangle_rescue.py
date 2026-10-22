"""Explicit NON-PAPER convex-combination rectangle fallback for failed BFF.

A positive uniform graph Laplacian with a monotone rectangular boundary
constructs a different (generally nonconformal) parameterization. This is
NOT an official BFF result, and must never silently certify Paper mode.
Input 3D positions/triangulation are left unchanged.

Intended ONLY for the UI's explicitly opted-in emergency export/experiment.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .reference_bff import (
    BFFRunResult, ReferenceBFFError, validate_rectangular_boundary,
    validate_reference_mesh, triangle_jacobian_diagnostics,
)


def rectangle_tutte_rescue(vertices, faces) -> BFFRunResult:
    """Injective-candidate rectangular graph-harmonic map for a valid mesh disk.

    The convex boundary is subdivided into four positive-length arclength
    segments. Interior vertex positions are averages of all graph neighbors.
    The resulting chart is ACCEPTED only if no triangle is flipped, degenerate,
    or overlapping. Disk topology is a prerequisite, not a repair step.
    """
    xyz = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    info = validate_reference_mesh(xyz, f)
    ids = np.asarray(info["boundary_loops"][0], dtype=int)
    n = len(ids)
    if n < 4:
        raise ReferenceBFFError("Tutte rectangle rescue requires >=4 boundary vertices")
    boundary_points = xyz[ids]
    lengths = np.linalg.norm(np.roll(boundary_points, -1, axis=0) - boundary_points, axis=1)
    if not np.all(np.isfinite(lengths)) or np.min(lengths) <= 1e-12:
        raise ReferenceBFFError("Tutte rectangle rescue: zero-length 3D boundary edge")
    arc = np.concatenate(([0.], np.cumsum(lengths)))
    total = float(arc[-1])
    corners = np.searchsorted(arc[:-1], total * np.arange(4) / 4., side="left")
    if len(np.unique(corners)) != 4:
        # Never collapse a rectangle corner: use distinct loop slots instead.
        corners = np.floor(n*np.arange(4)/4).astype(int)
    corners = np.asarray(corners, dtype=int)
    arclengths = [float(arc[corners[i+1]]-arc[corners[i]]) if i < 3
                  else float(total-arc[corners[3]]+arc[corners[0]]) for i in range(4)]
    if min(arclengths) <= 1e-12:
        raise ReferenceBFFError("Tutte rectangle rescue: collapsed boundary side")
    width = (arclengths[0]+arclengths[2])/2.
    height = (arclengths[1]+arclengths[3])/2.
    corners_uv = np.array([[0.,0.],[width,0.],[width,height],[0.,height]])
    boundary_uv = np.empty((n,2), dtype=float)
    for i in range(4):
        start=int(corners[i])
        end=int(corners[i+1]) if i<3 else n
        samples=np.arange(start,end,dtype=int)
        if not len(samples):
            raise ReferenceBFFError("Tutte rectangle rescue: empty boundary side")
        progress=(arc[samples]-arc[start])/arclengths[i]
        boundary_uv[samples]=(
            (1-progress[:,None])*corners_uv[i]
            + progress[:,None]*corners_uv[(i+1)%4]
        )

    boundary_mask = np.zeros(len(xyz), dtype=bool)
    boundary_mask[ids] = True
    interior = np.flatnonzero(~boundary_mask)
    uv = np.zeros((len(xyz),2), dtype=float)
    uv[ids] = boundary_uv
    if len(interior):
        # Uniform positive neighbor weights; no negative cotangent weights
        # that might flip skinny triangles in a Tutte embedding.
        edges = np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
        directed = np.vstack((edges,edges[:,::-1]))
        graph = sparse.coo_matrix(
            (np.ones(len(directed)),(directed[:,0],directed[:,1])),
            shape=(len(xyz),len(xyz))
        ).tocsr()
        graph.data[:]=1.
        degree=np.asarray(graph.sum(axis=1)).ravel()
        lap = sparse.diags(degree) - graph
        a = lap[interior,:][:,interior].tocsc()
        rhs = -(lap[interior,:][:,ids] @ boundary_uv)
        try:
            uv[interior] = np.asarray(spsolve(a,rhs), dtype=float).reshape(-1,2)
        except (ValueError,RuntimeError) as exc:
            raise ReferenceBFFError(f"Tutte rectangle rescue linear solve failed: {exc}") from exc

    if not np.all(np.isfinite(uv)):
        raise ReferenceBFFError("Tutte rectangle rescue produced non-finite UV")
    triangles=uv[f]
    uv_e1=triangles[:,1]-triangles[:,0]
    uv_e2=triangles[:,2]-triangles[:,0]
    signed=uv_e1[:,0]*uv_e2[:,1]-uv_e1[:,1]*uv_e2[:,0]
    epsilon=2e-14
    if np.any(np.abs(signed)<=epsilon):
        raise ReferenceBFFError(
            "Tutte rectangle rescue failed: triangle UV area degenerate "
            f"({int(np.count_nonzero(np.abs(signed)<=epsilon))} triangles); "
            "do not silently accept an invalid chart"
        )
    if np.min(signed) < -epsilon and np.max(signed) > epsilon:
        raise ReferenceBFFError(
            "Tutte rectangle rescue failed: inconsistent UV orientation "
            "(a non-injective chart remains)"
        )
    diagnostics=triangle_jacobian_diagnostics(xyz,uv,f)
    if diagnostics["uv_triangle_flip_count"] or diagnostics["uv_degenerate_triangle_count"]:
        raise ReferenceBFFError("Tutte rectangle rescue: differential validation failed")
    rect=validate_rectangular_boundary(uv,ids.tolist())
    return BFFRunResult(
        uv_vertices=uv,
        metrics={
            **rect,
            "parameterization_backend_name": "experimental_uniform_tutte_rectangle_NOT_BFF",
            "parameterization_backend_repository": "OneString local experimental rescue",
            "parameterization_backend_commit_sha": "NOT_APPLICABLE",
            "parameterization_backend_executable": "scipy_sparse_spsolve",
            "bff_boundary_policy_requested": "target_rectangle",
            "bff_boundary_policy_effective": "target_rectangle",
            "bff_rectangle_corner_boundary_wedge_indices": corners.tolist(),
            "bff_rectangle_corner_mode": "arclength_four_side_tutte",
            "nonpaper_rectangle_rescue_used": True,
            "paper_conformality_certified": False,
            "paper_reference_equivalent": False,
            "experimental_parameterization_warning": (
                "THE CHART IS A POSITIVE-WEIGHT GRAPH HARMONIC TUTTE "
                "RECTANGLE, NOT CONFORMAL AND NOT THE OFFICIAL BFF SOLUTION"
            ),
            "fallbacks_used": ["explicit_nonconformal_tutte_rectangle_rescue"],
            "input_topology": info,
        },
    )
