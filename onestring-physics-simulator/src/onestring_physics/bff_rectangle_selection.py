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



def _boundary_pca_projection(vertices: np.ndarray, loop: np.ndarray):
    """Arclength-weighted best-fit 2D plane of the *input S boundary*."""
    points = np.asarray(vertices, float)[loop]
    lengths = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("Input source boundary contains degenerate edges")
    weights = .5*(lengths+np.roll(lengths, 1))
    weights /= np.sum(weights)
    center = np.sum(points*weights[:,None], axis=0)
    centered = points-center
    covariance = (centered*weights[:,None]).T @ centered
    vals, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, np.argsort(vals)[-2:][::-1]]
    projected = centered @ axes
    nonplanarity = float(np.sqrt(max(vals[0],0.) / max(float(np.sum(vals)),1e-30)))
    if float(np.ptp(projected[:,0])) < 1e-10 or float(np.ptp(projected[:,1])) < 1e-10:
        raise ValueError("The input boundary is nearly collinear and cannot define a stable rectangle")
    return projected, weights, nonplanarity


def _corner_candidates(points: np.ndarray, projected: np.ndarray,
                       boundary_loop: np.ndarray) -> list[dict]:
    """Corner sets near oriented bounding-rectangle extrema of the 3D boundary."""
    n = len(projected)
    arc = np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    cumulative = np.concatenate(([0.], np.cumsum(arc)))
    result = []
    seen = set()
    for degrees in np.arange(0, 180, 15):
        angle = float(np.deg2rad(degrees))
        rotation = np.array([[np.cos(angle),-np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        rotated = projected @ rotation
        minimum, maximum = np.min(rotated,axis=0), np.max(rotated,axis=0)
        span = np.maximum(maximum-minimum,1e-12)
        unit = (rotated-minimum)/span
        targets = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        selected = []
        for target in targets:
            order = np.argsort(np.sum((unit-target)**2,axis=1),kind="stable")
            idx = next((int(i) for i in order if int(i) not in selected), None)
            if idx is None:
                break
            selected.append(idx)
        if len(selected)!=4:
            continue
        ordered = sorted(selected)
        arcs = [float(cumulative[ordered[(i+1)%4]] -
                      cumulative[ordered[i]]) if i<3 else
                float(cumulative[-1]-cumulative[ordered[-1]]+cumulative[ordered[0]])
                for i in range(4)]
        # A rectangle with one almost-collapsed side leads to severe BFF
        # distortion. These are genuinely invalid geometric corner sets.
        if min(arcs) < 0.025*float(cumulative[-1]):
            continue
        key = tuple(ordered)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "method":"source_boundary_pca_bounding_rectangle",
            "angle_degrees":float(degrees),
            "loop_corner_indices":ordered,
            "source_vertex_ids":[int(boundary_loop[i]) for i in ordered],
            "source_xyz":points[ordered].tolist(),
            "min_boundary_arc_fraction":float(min(arcs)/cumulative[-1]),
        })
    return result


def _boundary_similarity_error(target2d: np.ndarray, uv2d: np.ndarray,
                               weights: np.ndarray) -> float:
    """Weighted best-fit similarity registration; never affine-stretch UV.

    Reflections are permitted only in the *comparison*, not applied to BFF.
    """
    target=np.asarray(target2d,float)
    uv=np.asarray(uv2d,float)
    w=np.asarray(weights,float)
    a=target-np.sum(target*w[:,None],axis=0)
    b=uv-np.sum(uv*w[:,None],axis=0)
    norm_a=np.sqrt(np.sum(w*np.sum(a*a,axis=1)))
    norm_b=np.sqrt(np.sum(w*np.sum(b*b,axis=1)))
    if min(norm_a,norm_b)<=1e-12:
        raise ValueError("Degenerate shape in boundary fit comparison")
    a/=norm_a
    b/=norm_b
    u,_,vt=np.linalg.svd((b*w[:,None]).T@a)
    aligned=b@(u@vt)
    return float(np.sqrt(np.sum(w*np.sum((a-aligned)**2,axis=1))))


def choose_rectangle_by_source_boundary_fit(
    vertices, faces, *, rectangle_executable=None, timeout_seconds=120.,
    runner=run_official_bff,
):
    """Fit initial Omega using source-boundary geometry and the official BFF.

    1. Analyze the original 3D boundary (not its initial BFF chart).
    2. Try corner positions from PCA oriented bounding rectangles, plus
       arclength-quartile baselines.
    3. Solve official rectangular BFF for EACH candidate.
    4. Compare its entire mapped boundary to the source-boundary best-fit
       plane, under SIMILARITY ONLY, and penalize excessive area stretching.
    5. Choose lowest measured score. BFF UVs remain conformal, not warped.

    This is an explicit experimental heuristic, not the authors' unpublished
    corner-selection rule. Failed candidates are recorded, not silently used.
    """
    from .reference_bff import validate_reference_mesh, ReferenceBFFError
    x=np.asarray(vertices,dtype=float)
    f=np.asarray(faces,dtype=int)
    topology=validate_reference_mesh(x,f)
    loop=np.asarray(topology["boundary_loops"][0],dtype=int)
    projected,w,nonplanarity=_boundary_pca_projection(x,loop)
    candidates=_corner_candidates(x[loop],projected,loop)
    for phase in (0.,0.0625,0.125,0.1875):
        candidates.append(dict(method="boundary_arclength_quartiles",
                               phase=float(phase)))
    if not candidates:
        raise ReferenceBFFError("No rectangle corner candidates could be derived from source boundary")

    attempts=[]
    winning=None
    for candidate in candidates:
        kw={"rectangle_executable":rectangle_executable,
            "boundary_policy":"target_rectangle",
            "timeout_seconds":timeout_seconds}
        if "source_xyz" in candidate:
            kw["rectangle_corner_positions"]=np.asarray(candidate["source_xyz"],float)
        else:
            kw["rectangle_corner_phase_fraction"]=candidate["phase"]
        try:
            output=runner(x,f,**kw)
            if output.metrics.get("bff_rectangle_verified") is not True:
                raise ValueError("BFF output boundary failed rectangle verification")
            uvfaces=np.asarray(output.uv_vertices,float)[f]
            e=uvfaces[:,1]-uvfaces[:,0]
            g=uvfaces[:,2]-uvfaces[:,0]
            det=e[:,0]*g[:,1]-e[:,1]*g[:,0]
            if np.any(det>1e-14) and np.any(det < -1e-14):
                raise ValueError("Rectangular BFF candidate has mixed UV triangle orientations")
            span=whole_surface_area_span(x,output.uv_vertices,f)
            error=_boundary_similarity_error(projected,np.asarray(output.uv_vertices)[loop],w)
            # Source PCA bounding rectangle is a weak aspect target. Keep this
            # term small; direct boundary fidelity is the primary metric.
            expected=np.ptp(projected,axis=0)
            actual=np.ptp(np.asarray(output.uv_vertices,float)[loop],axis=0)
            aspect_source=float(np.max(expected)/max(np.min(expected),1e-15))
            aspect_uv=float(np.max(actual)/max(np.min(actual),1e-15))
            aspect_error=abs(float(np.log(aspect_uv/aspect_source)))
            # sqrt(span) is the same linear-CSF ratio measured later.
            score=error + .10*aspect_error + .06*float(np.log(np.sqrt(span)))
            row={
                "source_corner_method":candidate["method"],
                "source_pca_angle_degrees":candidate.get("angle_degrees"),
                "source_corner_vertex_ids":candidate.get("source_vertex_ids"),
                "boundary_arclength_phase":candidate.get("phase"),
                "normalized_boundary_fit_rmse":error,
                "aspect_log_error":aspect_error,
                "whole_source_linear_csf_span":float(np.sqrt(span)),
                "total_boundary_fit_score":float(score),
                "rectangle_aspect_ratio":aspect_uv,
                "corner_boundary_wedge_indices":output.metrics.get(
                    "bff_rectangle_corner_boundary_wedge_indices"),
            }
            attempts.append(row)
            if winning is None or (score,span)<winning[0]:
                winning=((score,span),candidate,output,row)
        except (ValueError,RuntimeError) as exc:
            attempts.append({
                "source_corner_method":candidate["method"],
                "source_pca_angle_degrees":candidate.get("angle_degrees"),
                "source_corner_vertex_ids":candidate.get("source_vertex_ids"),
                "boundary_arclength_phase":candidate.get("phase"),
                "failed_reason":f"{type(exc).__name__}: {str(exc)[:350]}",
            })
    if winning is None:
        # The general boundary-fit presets are only a heuristic. A valid disk
        # may still have a non-injective rectangle BFF chart for every PCA
        # corner choice and every coarse quartile phase. Broaden the phase
        # search ONLY in that case; do not change the Lilium lobe-tip path or
        # accept any degenerate/flipped chart merely to suppress an exception.
        fallback_candidates = [
            dict(method="boundary_arclength_quartiles_dense", phase=i/64.)
            for i in range(1, 16)
            if i % 4 != 0
        ]
        # Additional source-boundary maxima provide a qualitatively different
        # four-corner arrangement on non-rectangular boundaries. The optional
        # heuristic is still validated by the exact SAME UV checks.
        from .paper_split_reflatten import lobe_tip_corner_positions
        for window in (3, 7, 11):
            try:
                fallback_candidates.append(dict(
                    method="boundary_curvature_extrema",
                    corner_window=window,
                    source_xyz=lobe_tip_corner_positions(x, f, window=window).tolist(),
                ))
            except ValueError as exc:
                attempts.append(dict(
                    source_corner_method="boundary_curvature_extrema",
                    source_corner_window=window,
                    failed_reason=f"{type(exc).__name__}: {str(exc)[:350]}",
                ))
        for candidate in fallback_candidates:
            kw = dict(
                rectangle_executable=rectangle_executable,
                boundary_policy="target_rectangle",
                timeout_seconds=timeout_seconds,
            )
            if "source_xyz" in candidate:
                kw["rectangle_corner_positions"] = np.asarray(candidate["source_xyz"], float)
            else:
                kw["rectangle_corner_phase_fraction"] = candidate["phase"]
            try:
                output = runner(x, f, **kw)
                if output.metrics.get("bff_rectangle_verified") is not True:
                    raise ValueError("BFF output boundary failed rectangle verification")
                uvfaces = np.asarray(output.uv_vertices, float)[f]
                e = uvfaces[:, 1] - uvfaces[:, 0]
                g = uvfaces[:, 2] - uvfaces[:, 0]
                det = e[:, 0]*g[:, 1] - e[:, 1]*g[:, 0]
                if np.any(det > 1e-14) and np.any(det < -1e-14):
                    raise ValueError("Rectangular BFF candidate has mixed UV triangle orientations")
                span = whole_surface_area_span(x, output.uv_vertices, f)
                error = _boundary_similarity_error(
                    projected, np.asarray(output.uv_vertices)[loop], w
                )
                expected = np.ptp(projected, axis=0)
                actual = np.ptp(np.asarray(output.uv_vertices, float)[loop], axis=0)
                aspect_source = float(np.max(expected)/max(np.min(expected), 1e-15))
                aspect_uv = float(np.max(actual)/max(np.min(actual), 1e-15))
                aspect_error = abs(float(np.log(aspect_uv/aspect_source)))
                score = error + .10*aspect_error + .06*float(np.log(np.sqrt(span)))
                row = dict(
                    source_corner_method=candidate["method"],
                    source_corner_window=candidate.get("corner_window"),
                    source_pca_angle_degrees=None,
                    source_corner_vertex_ids=None,
                    boundary_arclength_phase=candidate.get("phase"),
                    normalized_boundary_fit_rmse=error,
                    aspect_log_error=aspect_error,
                    whole_source_linear_csf_span=float(np.sqrt(span)),
                    total_boundary_fit_score=float(score),
                    rectangle_aspect_ratio=aspect_uv,
                    corner_boundary_wedge_indices=output.metrics.get(
                        "bff_rectangle_corner_boundary_wedge_indices"
                    ),
                )
                attempts.append(row)
                if winning is None or (score, span) < winning[0]:
                    winning = ((score, span), candidate, output, row)
            except (ValueError, RuntimeError) as exc:
                attempts.append(dict(
                    source_corner_method=candidate["method"],
                    source_corner_window=candidate.get("corner_window"),
                    source_pca_angle_degrees=None,
                    source_corner_vertex_ids=None,
                    boundary_arclength_phase=candidate.get("phase"),
                    failed_reason=f"{type(exc).__name__}: {str(exc)[:350]}",
                ))
    if winning is None:
        from collections import Counter
        failures = Counter(
            item["failed_reason"].split(":")[0]
            for item in attempts if "failed_reason" in item
        )
        raise ReferenceBFFError(
            f"All {len(attempts)} original + adaptive rectangular BFF corner "
            f"candidates failed (reasons={dict(failures)}). "
            "No geometrically valid and consistently oriented rectangular "
            "BFF chart was found; the input was NOT altered or accepted "
            "with a foldover. Inspect short/skinny triangles and boundary "
            "shape, try a different explicit corner mode, or remesh the "
            "surface without changing its topology. "
            f"Last five candidate diagnostics: {attempts[-5:]}"
        )
    _,chosen,result,entry=winning
    return BFFRunResult(
        uv_vertices=result.uv_vertices,
        metrics={
            **result.metrics,
            "bff_corner_selection_method":"source_boundary_pca_fit_EXPERIMENTAL",
            "bff_corner_selection_evidence":"3D boundary PCA, whole-boundary similarity, aspect ratio and area Jacobian",
            "bff_corner_selection_attempts":attempts,
            "bff_corner_selection_attempt_count":len(attempts),
            "bff_corner_selection_valid_count":sum("total_boundary_fit_score" in x for x in attempts),
            "bff_corner_selection_adaptive_search_used":bool(
                entry["source_corner_method"] in (
                    "boundary_arclength_quartiles_dense", "boundary_curvature_extrema"
                )
            ),
            "bff_corner_selection_rejected_count":sum(
                "failed_reason" in row for row in attempts
            ),
            "bff_corner_selection_selected_score":float(entry["total_boundary_fit_score"]),
            "bff_corner_selection_selected_source_xyz":chosen.get("source_xyz"),
            "bff_corner_selection_selected_source_vertex_ids":chosen.get("source_vertex_ids"),
            "bff_corner_selection_selected_phase":chosen.get("phase"),
            "bff_corner_selection_selected_angle_degrees":chosen.get("angle_degrees"),
            "bff_corner_selection_source_boundary_pca_nonplanarity":float(nonplanarity),
            "bff_corner_selection_certifies_area_limit":False,
            "bff_corner_selection_author_method_verified":False,
        },
    )
