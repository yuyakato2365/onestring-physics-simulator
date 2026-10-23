"""Rank *actual official OptCuts seam/cut candidates* by real M2D grid geometry.

Only the experimental area-grid Omega mode imports this module.  It does NOT
invent a second lattice, edit OptCuts C++, relax the selected-cell threshold,
or move seams after the OptCuts solve.  The available topology search consists
of the official initial-cut alternatives (normally two for a closed genus-0
surface).  Their resulting UV charts may have different real production
lattices.  Each candidate is evaluated on its own ORIGINAL production grid;
the chosen lattice is frozen before any boundary deformation.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .optcuts_area_grid_boundary import construct_grid_target


def grid_outline_complexity(target):
    """Simple-polygon corner and concavity diagnostics in grid-cell units."""
    xy = np.asarray(target["grid_outline"], float)
    if len(xy) < 4:
        raise ValueError("GRID_SEAM_BAD_OUTLINE")
    edges = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(lengths) <= 0:
        raise ValueError("GRID_SEAM_ZERO_LENGTH_BOUNDARY_EDGE")
    prior = np.roll(edges, 1, axis=0)
    turn = prior[:, 0] * edges[:, 1] - prior[:, 1] * edges[:, 0]
    signed_area = 0.5 * float(np.sum(
        xy[:, 0] * np.roll(xy[:, 1], -1)
        - xy[:, 1] * np.roll(xy[:, 0], -1)))
    if abs(signed_area) < 1e-14:
        raise ValueError("GRID_SEAM_ZERO_AREA_OUTLINE")
    reflex = int(np.count_nonzero(turn * np.sign(signed_area) < -1e-12))
    cell_step = np.asarray(target["step"], float)
    perimeter_grid = float(np.sum(np.linalg.norm(edges / cell_step, axis=1)))
    cells = int(np.count_nonzero(target["mask"]))
    return {
        "corners": int(len(xy)),
        "reflex_corners": reflex,
        "perimeter_grid_units": perimeter_grid,
        "retained_cells": cells,
    }


def grid_seam_selection_score(complexity, audit, *, reference_sd=1.,
                              reference_seam=1.):
    """Smaller is preferred; penalize concavities, turns, and bad UV quality.

    Counts are divided by sqrt(area) to compare charts with differently sized
    original OptCuts UV footprints.  Distortion and seam-length penalties are
    dimensionless relative to the valid candidate pool.  This is a heuristic
    ranking, not a guarantee of a bijective final reparameterization.
    """
    size = math.sqrt(max(1, complexity["retained_cells"]))
    grid_cost = (
        2.5 * complexity["reflex_corners"]
        + 0.4 * complexity["corners"]
        + 0.08 * complexity["perimeter_grid_units"]
    ) / size
    sd = max(float(audit["distortion_area_mean"]), 1e-8)
    seam = max(float(audit.get("seam_length", 0.)), 0.)
    tail = float(audit.get("csf_over_2_area_fraction", 0.))
    quality_cost = (
        1.5 * max(0., math.log(sd / max(reference_sd, 1e-8)))
        + 2. * max(0., tail)
        + 0.20 * max(0., math.log(max(seam, 1e-8) /
                                  max(reference_seam, 1e-8)))
    )
    return float(grid_cost + quality_cost)


def _make_chart(original, optcuts_result, rank_index):
    """Copy a full paired UV/3D chart; never combine incompatible face IDs."""
    chart = copy.copy(original)
    chart.metrics = dict(getattr(original, "metrics", {}) or {})
    if optcuts_result is not None:
        chart.surface_vertices_3d = np.asarray(
            optcuts_result.surface_vertices_3d, float).copy()
        chart.surface_faces = np.asarray(
            optcuts_result.surface_faces, int).copy()
        chart.uv_vertices_2d = np.asarray(
            optcuts_result.uv_vertices_2d, float).copy()
        chart.uv_faces = np.asarray(optcuts_result.uv_faces, int).copy()
        from .optcuts_backend import _boundary_loops
        loops = _boundary_loops(chart.uv_faces)
        if len(loops) != 1:
            raise ValueError(f"GRID_SEAM_CANDIDATE_NOT_DISK: loops={len(loops)}")
        loop = [int(v) for v in loops[0]]
        chart.omega_boundary = chart.uv_vertices_2d[loop + [loop[0]]]
        chart.metrics.update(dict(getattr(optcuts_result, "metrics", {}) or {}))
        chart.metrics["boundary_loop"] = loop
    # Original OptCuts candidates must not silently replace this chart when
    # an intermediate domain is constructed, nor after the Grid fit.
    chart._onestring_optcuts_candidates = []
    chart._onestring_optcuts_candidate_qualities = []
    chart.metrics["optcuts_split_count_selection_pending"] = False
    chart.metrics["optcuts_area_grid_seam_start_index"] = int(rank_index)
    chart.triangle_acceleration = None
    return chart


def rank_grid_aware_seam_candidates(parameterization, grid, params, domain_builder):
    """Evaluate official seam candidates against their *actual* M2D overlays.

    Returns (score-ranked candidates, serializable audit).  Invalid topology,
    UV overlap, or invalid selected-cell polygon are explicitly rejected.
    This function does not call the expensive boundary-fitting solver.
    """
    from .optcuts_quality_audit import quality
    from .optcuts_backend import OptCutsResult
    from .optcuts_area_grid_boundary import GridBoundaryReparameterizationError

    alternatives = list(getattr(parameterization,
                                "_onestring_optcuts_candidates", []) or [])
    audits = list(getattr(parameterization,
                          "_onestring_optcuts_candidate_qualities", []) or [])
    # Synthetic fixtures and open meshes may have just one official result.
    sources = alternatives if alternatives else [None]
    evaluated = []
    report = []
    for index, source in enumerate(sources):
        summary = {"candidate_index": int(index),
                   "source": "official_initial_cut" if source is not None
                   else "single_official_chart"}
        try:
            chart = _make_chart(parameterization, source, index)
            audit = (dict(audits[index]) if index < len(audits)
                     else quality(OptCutsResult(
                         chart.surface_vertices_3d, chart.surface_faces,
                         chart.uv_vertices_2d, chart.uv_faces,
                         [], "grid-seam-selection", {})))
            bad = ("flipped_triangles", "degenerate_triangles",
                   "injectivity_overlap_pairs")
            if any(int(audit.get(key, 0)) for key in bad):
                raise ValueError("GRID_SEAM_SOURCE_UV_INVALID")
            if (not np.isfinite(float(audit["distortion_area_mean"]))
                    or float(audit["distortion_area_mean"]) <= 0):
                raise ValueError("GRID_SEAM_BAD_SOURCE_SD")
            domain = domain_builder(chart, grid, params)
            from .optcuts_backend import _boundary_loops
            loops = _boundary_loops(np.asarray(chart.uv_faces, int))
            if len(loops) != 1:
                raise ValueError("GRID_SEAM_CHART_NOT_SINGLE_DISK")
            target = construct_grid_target(
                chart.uv_vertices_2d, chart.uv_faces, loops[0],
                nx=grid.nx, ny=grid.ny,
                threshold=float(getattr(params,
                    "optcuts_area_grid_coverage_threshold", .5)),
                domain=domain)
            shape = grid_outline_complexity(target)
            summary.update(shape)
            summary.update({
                "source_sd_mean": float(audit["distortion_area_mean"]),
                "source_csf_over_2_area_fraction": float(
                    audit.get("csf_over_2_area_fraction", 0.)),
                "source_seam_length": float(audit.get("seam_length", 0.)),
                "real_grid_origin": target["origin"].tolist(),
                "real_grid_step": target["step"].tolist(),
                "status": "scored",
            })
            evaluated.append((chart, domain, summary, audit))
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            summary.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        report.append(summary)
    if not evaluated:
        raise GridBoundaryReparameterizationError(
            "GRID_SEAM_NO_VALID_CANDIDATE: "
            + " | ".join(f"start={r['candidate_index']}: {r.get('error')}"
                         for r in report))
    ref_sd = min(float(a["distortion_area_mean"])
                 for _c, _d, _r, a in evaluated)
    seam_lengths = [float(a.get("seam_length", 0.))
                    for _c, _d, _r, a in evaluated
                    if float(a.get("seam_length", 0.)) > 0]
    ref_seam = min(seam_lengths) if seam_lengths else 1.
    ranked = []
    for chart, domain, summary, audit in evaluated:
        # A more rectilinear seam must not win only by destroying the
        # pre-existing OptCuts low-distortion guarantee.
        sd = float(audit["distortion_area_mean"])
        if sd > max(1.5 * ref_sd, ref_sd + .25):
            summary.update(status="rejected_uv_quality",
                           error="source SD exceeds 1.5x reference guard")
            continue
        summary["score"] = grid_seam_selection_score(
            summary, audit, reference_sd=ref_sd,
            reference_seam=ref_seam)
        ranked.append((chart, domain, summary))
    if not ranked:
        raise GridBoundaryReparameterizationError(
            "GRID_SEAM_ALL_SOURCE_QUALITY_GUARDS_FAILED"
        )
    ranked.sort(key=lambda item: (item[2]["score"],
                                    item[2]["reflex_corners"],
                                    item[2]["candidate_index"]))
    for rank, (_chart, _domain, summary) in enumerate(ranked):
        summary["rank"] = int(rank + 1)
    return ranked, report


def copy_selected_chart_to_original(original, selected):
    """Keep pipeline's existing SurfaceParameterization reference valid."""
    fields = ("surface_vertices_3d", "surface_faces", "uv_vertices_2d",
              "uv_faces", "omega_boundary", "triangle_acceleration",
              "metrics")
    for name in fields:
        setattr(original, name, getattr(selected, name))
    for name in ("_optcuts_area_grid_mask", "_optcuts_area_grid_coverage",
                 "_optcuts_area_grid_source_boundary",
                 "_optcuts_area_grid_target_outline"):
        if hasattr(selected, name):
            setattr(original, name, getattr(selected, name))
    original._onestring_optcuts_candidates = []
    original._onestring_optcuts_candidate_qualities = []
    return original
