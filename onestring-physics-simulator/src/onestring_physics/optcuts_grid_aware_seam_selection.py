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
import time

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


def _make_chart(original, optcuts_result, rank_index, *, rotation_degrees=0.):
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
    chart.metrics["optcuts_area_grid_rotation_degrees"] = float(rotation_degrees)
    if rotation_degrees:
        # Rigidity keeps face geometry, UV distortion and seam topology
        # invariant, but changes the relative orientation of the ORIGINAL
        # UV domain to the as-yet-unconstructed production M2D grid.
        # This is NOT a new seam and is named accordingly in diagnostics.
        angle = math.radians(float(rotation_degrees))
        c, sn = math.cos(angle), math.sin(angle)
        r = np.array([[c, -sn], [sn, c]], dtype=float)
        uv = np.asarray(chart.uv_vertices_2d, dtype=float)
        centre = np.mean(uv, axis=0)
        chart.uv_vertices_2d = (uv - centre) @ r.T + centre
        boundary_ids = chart.metrics.get("boundary_loop")
        if not boundary_ids:
            from .optcuts_backend import _boundary_loops
            loops = _boundary_loops(chart.uv_faces)
            if len(loops) != 1:
                raise ValueError("GRID_SEAM_ROTATED_CHART_NOT_DISK")
            boundary_ids = [int(v) for v in loops[0]]
        loop = [int(i) for i in boundary_ids]
        chart.omega_boundary = chart.uv_vertices_2d[loop + [loop[0]]]
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
    # Each actual official seam topology is evaluated unrotated first.
    # If it cannot produce a feasible polygon, additional RIGID UV rotations
    # may choose a more grid-compatible Ω footprint without changing seams.
    # Rotation is allowed only BEFORE constructing each production lattice.
    sources = alternatives if alternatives else [None]
    rotations = (0., 22.5, 45., 67.5) if bool(
        getattr(params, "optcuts_area_grid_orientation_search_enabled",
                True)) else (0.,)
    tasks = ([(i, source, 0.) for i, source in enumerate(sources)] +
             [(i, source, degrees)
              for degrees in rotations if degrees != 0.
              for i, source in enumerate(sources)])
    evaluated = []
    report = []
    # Rotation is rigid: SD, area Jacobian, seam length and overlap counts
    # are invariant, so do not re-run expensive source audits per orientation.
    quality_by_source = {}
    started = time.perf_counter()
    print(
        "[GRID-SEAM-SEARCH] started; official_cut_sources="
        f"{len(sources)}; orientations={list(rotations)}; "
        f"grid_variants={len(tasks)}",
        flush=True,
    )
    for index, (source_index, source, rotation) in enumerate(tasks):
        iteration_started = time.perf_counter()
        print(
            f"[GRID-SEAM-SEARCH] evaluate={index+1}/{len(tasks)} "
            f"cut={source_index} rotation={rotation:g}deg "
            f"elapsed={iteration_started-started:.1f}s",
            flush=True,
        )
        summary = {
            "candidate_index": int(index),
            "source_start_index": int(source_index),
            "uv_rotation_degrees": float(rotation),
            "source": ("official_initial_cut" if source is not None
                       else "single_official_chart"),
        }
        try:
            chart = _make_chart(parameterization, source, source_index,
                                rotation_degrees=rotation)
            if source_index not in quality_by_source:
                quality_by_source[source_index] = (
                    dict(audits[source_index])
                    if source_index < len(audits)
                    else quality(OptCutsResult(
                        chart.surface_vertices_3d, chart.surface_faces,
                        chart.uv_vertices_2d, chart.uv_faces,
                        [], "grid-seam-selection", {}))
                )
            audit = quality_by_source[source_index]
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
            print(
                f"[GRID-SEAM-SEARCH] scored={index+1}/{len(tasks)} "
                f"tiles={shape['retained_cells']} "
                f"corners={shape['corners']} "
                f"reflex={shape['reflex_corners']} "
                f"duration={time.perf_counter()-iteration_started:.1f}s",
                flush=True,
            )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            summary.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
            print(
                f"[GRID-SEAM-SEARCH] invalid={index+1}/{len(tasks)} "
                f"reason={type(exc).__name__}: {str(exc)[:180]} "
                f"duration={time.perf_counter()-iteration_started:.1f}s",
                flush=True,
            )
        report.append(summary)
    if not evaluated:
        raise GridBoundaryReparameterizationError(
            "GRID_SEAM_NO_VALID_CANDIDATE: "
            + " | ".join(
                f"cut={r['source_start_index']} rotation={r['uv_rotation_degrees']}: "
                f"{r.get('error')}" for r in report))
    ref_sd = min(float(a["distortion_area_mean"])
                 for _c, _d, _r, a in evaluated)
    seam_lengths = [float(a.get("seam_length", 0.))
                    for _c, _d, _r, a in evaluated
                    if float(a.get("seam_length", 0.)) > 0]
    ref_seam = min(seam_lengths) if seam_lengths else 1.
    ranked = []
    for chart, domain, summary, audit in evaluated:
        # A source candidate is accepted only when its *actual* UV is
        # locally nonflipped, nonoverlapping and finite. Retain other valid
        # OptCuts seams even when SD exceeds the former arbitrary 1.5x
        # relative guard: score penalizes it, while the FINAL fitting step
        # separately enforces its distortion/injectivity hard checks.
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
    print(
        f"[GRID-SEAM-SEARCH] ranked={len(ranked)}/{len(tasks)} "
        f"elapsed={time.perf_counter()-started:.1f}s; "
        "starting bijective boundary fitting",
        flush=True,
    )
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
