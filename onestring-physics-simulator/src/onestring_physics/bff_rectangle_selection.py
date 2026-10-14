"""Optional geometry-driven rectangle corner selection, NOT author-proven.

The authors' Lilium BFF corner IDs are unavailable. A fixed 3D arclength
quartile phase can create severe distortion. Here we run the unmodified
official BFF rectangle adapter at four uniformly prespecified corner
phases and choose the smallest *whole-source-mesh* area-Jacobian span.
We do NOT inspect split count, source filename, grid resolution or any
post-Split metric when choosing corners. No fallback to arbitrary UVs.
"""
from __future__ import annotations

import numpy as np

from .reference_bff import BFFRunResult, run_official_bff


PHASES = (0.0, 0.0625, 0.125, 0.1875)


def whole_surface_area_span(vertices, uv, faces):
    xyz=np.asarray(vertices,dtype=float)[np.asarray(faces,dtype=int)]
    uvtri=np.asarray(uv,dtype=float)[np.asarray(faces,dtype=int)]
    a=xyz[:,1]-xyz[:,0]
    b=xyz[:,2]-xyz[:,0]
    e=uvtri[:,1]-uvtri[:,0]
    f=uvtri[:,2]-uvtri[:,0]
    twice_3d=np.linalg.norm(np.cross(a,b),axis=1)
    twice_uv=np.abs(e[:,0]*f[:,1]-e[:,1]*f[:,0])
    if len(twice_uv)==0 or not np.isfinite(twice_3d).all() or not np.isfinite(twice_uv).all():
        raise ValueError("No finite source triangles for rectangular BFF selection")
    if np.any(twice_3d<=1e-16) or np.any(twice_uv<=1e-16):
        raise ValueError("One candidate has a degenerate source or UV triangle")
    jac=twice_3d/twice_uv
    if not np.isfinite(jac).all() or np.any(jac<=0):
        raise ValueError("Invalid source UV area Jacobian")
    return float(np.max(jac)/np.min(jac))


def choose_rectangle_by_min_area_span(
    vertices, faces, *, rectangle_executable=None, timeout_seconds=120.,
    runner=run_official_bff,
):
    """Return official BFF result with lowest maximum area distortion span.

    Evaluate ALL predetermined phases. Any candidate failure aborts the
    experimental comparison rather than silently substituting a solver.
    The true mechanical feasibility is NOT certified by this proxy.
    """
    attempts=[]
    best=None
    for phase in PHASES:
        result=runner(
            vertices,faces,
            rectangle_executable=rectangle_executable,
            rectangle_corner_phase_fraction=phase,
            boundary_policy="target_rectangle",
            timeout_seconds=timeout_seconds,
        )
        area=whole_surface_area_span(vertices,result.uv_vertices,faces)
        row=dict(
            boundary_arclength_phase=float(phase),
            whole_source_area_jacobian_span=float(area),
            whole_source_linear_span=float(np.sqrt(area)),
            corner_boundary_wedge_indices=result.metrics.get(
                "bff_rectangle_corner_boundary_wedge_indices"
            ),
        )
        attempts.append(row)
        if best is None or area < best[0]:
            best=(area,phase,result)
    assert best is not None
    _,phase,result=best
    return BFFRunResult(
        uv_vertices=result.uv_vertices,
        metrics={
            **result.metrics,
            "bff_corner_selection_method":"min_whole_source_area_jacobian_span_4phase_EXPERIMENTAL",
            "bff_corner_selection_evidence":"Whole source conformal Jacobian span, no Split-count optimization",
            "bff_corner_selection_attempts":attempts,
            "bff_corner_selection_selected_phase":float(phase),
            "bff_corner_selection_certifies_area_limit":False,
            "bff_corner_selection_author_method_verified":False,
        }
    )
