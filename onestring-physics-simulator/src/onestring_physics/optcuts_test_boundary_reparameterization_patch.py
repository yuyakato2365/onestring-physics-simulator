"""Paper-aligned Omega -> M2D behavior for ``optcuts_test`` / ``optcuts_test2``.

The One String paper uses a one-way construction:

    S -> Omega -> regular quad grid -> crop cells not fully contained in Omega.

This patch keeps the OptCuts-produced parameterization/cut topology as Omega and
removes the former feedback loop that rebuilt Omega from an intermediate M2D
footprint.  M2D is then filtered so that every retained quad is fully contained
in the fixed planar domain.

Only the Omega -> M2D relation is changed here.  OptCuts remains the selected
parameterization backend, and downstream CSF split / K3D / K2D patches remain
unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def _is_optcuts_test(params: Any) -> bool:
    return str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"


def _clean_polygon(boundary: Any) -> np.ndarray:
    poly = np.asarray(boundary, dtype=float)
    if poly.ndim != 2 or poly.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    poly = poly[:, :2]
    poly = poly[np.all(np.isfinite(poly), axis=1)]
    if len(poly) == 0:
        return poly

    # Remove consecutive duplicate points.  Keep the polygon open internally;
    # edge iteration closes it with np.roll.
    scale = max(float(np.max(np.ptp(poly, axis=0))), 1.0)
    tol = max(1e-12, 1e-11 * scale)
    kept = [poly[0]]
    for p in poly[1:]:
        if float(np.linalg.norm(p - kept[-1])) > tol:
            kept.append(p)
    out = np.asarray(kept, dtype=float)
    if len(out) >= 2 and float(np.linalg.norm(out[0] - out[-1])) <= tol:
        out = out[:-1]
    return out


def _geom_tolerance(poly: np.ndarray, quad: np.ndarray | None = None) -> float:
    arrays = [np.asarray(poly, dtype=float)]
    if quad is not None:
        arrays.append(np.asarray(quad, dtype=float))
    values = np.vstack([a[:, :2] for a in arrays if a.size])
    scale = max(float(np.max(np.ptp(values, axis=0))) if len(values) else 1.0, 1.0)
    return max(1e-12, 1e-10 * scale)


def _point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    ap = p - a
    cross = float(ab[0] * ap[1] - ab[1] * ap[0])
    if abs(cross) > tol * max(float(np.linalg.norm(ab)), 1.0):
        return False
    dot = float(np.dot(ap, ab))
    if dot < -tol:
        return False
    if dot > float(np.dot(ab, ab)) + tol:
        return False
    return True


def _point_in_polygon_or_boundary(point: np.ndarray, polygon: np.ndarray, tol: float) -> bool:
    """Even-odd point-in-polygon test with the boundary counted as inside."""
    p = np.asarray(point, dtype=float)[:2]
    poly = np.asarray(polygon, dtype=float)[:, :2]
    n = len(poly)
    if n < 3:
        return False

    inside = False
    x, y = float(p[0]), float(p[1])
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if _point_on_segment(p, a, b, tol):
            return True
        y0, y1 = float(a[1]), float(b[1])
        if (y0 > y) == (y1 > y):
            continue
        denom = y1 - y0
        if abs(denom) <= tol:
            continue
        x_hit = float(a[0]) + (y - y0) * float(b[0] - a[0]) / denom
        if x_hit > x + tol:
            inside = not inside
    return inside


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _proper_segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    tol: float,
) -> bool:
    """Return True only for an interior crossing; touching/overlap is allowed."""
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    return (
        ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol))
        and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol))
    )


def _strictly_inside_quad(point: np.ndarray, quad: np.ndarray, tol: float) -> bool:
    """Strict point-in-quad test used to catch a concave boundary entering a cell."""
    q = np.asarray(quad, dtype=float)[:, :2]
    p = np.asarray(point, dtype=float)[:2]
    if any(_point_on_segment(p, q[i], q[(i + 1) % len(q)], tol) for i in range(len(q))):
        return False
    return _point_in_polygon_or_boundary(p, q, tol)


def _quad_fully_contained(quad: np.ndarray, polygon: np.ndarray) -> bool:
    """Geometric full-cell containment, not center-based crop.

    Conditions:
    1. all quad corners are inside or on the Omega boundary;
    2. no quad edge properly crosses the Omega boundary;
    3. no Omega boundary vertex lies strictly inside the quad (concave notch / hole guard).
    """
    q = np.asarray(quad, dtype=float)[:, :2]
    poly = np.asarray(polygon, dtype=float)[:, :2]
    if len(q) < 3 or len(poly) < 3:
        return False
    tol = _geom_tolerance(poly, q)

    if not all(_point_in_polygon_or_boundary(p, poly, tol) for p in q):
        return False

    for i in range(len(q)):
        qa, qb = q[i], q[(i + 1) % len(q)]
        for j in range(len(poly)):
            pa, pb = poly[j], poly[(j + 1) % len(poly)]
            if _proper_segments_intersect(qa, qb, pa, pb, tol):
                return False

    if any(_strictly_inside_quad(p, q, tol) for p in poly):
        return False
    return True


def _compact_quad_mesh(pipeline: Any, mesh: Any, keep: np.ndarray, metrics: dict[str, Any]) -> Any:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)[np.asarray(keep, dtype=bool)]
    if len(faces) == 0:
        raise RuntimeError(
            "OPTCUTS_TEST_PAPER_CROP_EMPTY: no regular quad cell is fully contained in Omega. "
            "Reduce tile size or inspect the OptCuts planar domain."
        )

    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=int)
    remap[used] = np.arange(len(used), dtype=int)
    compact_vertices = vertices[used]
    compact_faces = remap[faces]

    cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
    try:
        out = cls(
            compact_vertices,
            compact_faces,
            mesh.grid,
            mesh.stage,
            metrics,
            list(getattr(mesh, "split_lines", [])),
        )
    except TypeError:
        out = cls(
            vertices=compact_vertices,
            faces=compact_faces,
            grid=mesh.grid,
            stage=mesh.stage,
            metrics=metrics,
            split_lines=list(getattr(mesh, "split_lines", [])),
        )
    return out


def install_optcuts_test_boundary_reparameterization_patch(pipeline: Any) -> None:
    """Install paper-style one-way Omega -> M2D semantics for optcuts_test."""
    if getattr(pipeline, "_onestring_optcuts_test_boundary_patch_installed", False):
        return

    base_parameterization = pipeline._build_surface_parameterization
    base_domain = pipeline._flatten_to_domain
    base_m2d = pipeline._build_m2d

    def paper_parameterization(surface: Any, target: Any, grid: Any, params: Any):
        if not _is_optcuts_test(params):
            return base_parameterization(surface, target, grid, params)

        # Route through the ordinary official OptCuts backend, but DO NOT build
        # an intermediate M2D and DO NOT move/re-solve Omega from its footprint.
        ordinary_params = replace(params, omega_parameterization_mode="optcuts", optcuts_try_alternative_initial_cut=True)
        parameterization = base_parameterization(surface, target, grid, ordinary_params)
        # In the area-boundary mode, fit only AFTER the original OptCuts
        # Ω has constructed its ACTUAL production M2D lattice. The previous
        # code invented a different grid here and was not the user's method.
        parameterization.method = "optcuts_test"
        parameterization.metrics.update(
            {
                "omega_parameterization_mode": "optcuts_test",
                # Expose the user-visible Omega choice without altering the
                # stable internal optcuts_test carrier and its K3D/K2D patches.
                "requested_omega_parameterization_mode": (
                    ("optcuts_grid_seam_native"
                     if bool(getattr(params,"optcuts_grid_seam_native_enabled",False))
                     else "optcuts_area_grid_combined"
                     if all(bool(getattr(params,f,False)) for f in (
                         "optcuts_area_grid_phase_search_enabled",
                         "optcuts_area_grid_tail_sd_enabled",
                         "optcuts_area_grid_adaptive_remesh_enabled"))
                     else "optcuts_area_grid_phase_search"
                     if bool(getattr(params,"optcuts_area_grid_phase_search_enabled",False))
                     else "optcuts_area_grid_tail_sd"
                     if bool(getattr(params,"optcuts_area_grid_tail_sd_enabled",False))
                     else "optcuts_area_grid_adaptive_remesh"
                     if bool(getattr(params,"optcuts_area_grid_adaptive_remesh_enabled",False))
                     else "optcuts_area_grid_boundary")
                    if bool(getattr(params,"optcuts_area_grid_boundary_enabled",False))
                    else ("optcuts_grid_aware" if bool(getattr(
                        params,"optcuts_grid_aware_enabled",False))
                    else "optcuts_test")
                ),
                "optcuts_grid_aware_selected_as_omega_mode": bool(
                    getattr(params,"optcuts_grid_aware_enabled",False)
                ),
                "optcuts_area_grid_boundary_selected_as_omega_mode": bool(
                    getattr(params,"optcuts_area_grid_boundary_enabled",False)
                ),
                "optcuts_grid_seam_native_enabled": bool(getattr(
                    params,"optcuts_grid_seam_native_enabled",False)),
                "optcuts_test_enabled": True,
                "optcuts_test_model": (
                    ("patched native OptCuts Grid-Seam C++ cut topology -> "
                     "actual production grid mask -> exact-boundary UV fit")
                    if bool(getattr(params,"optcuts_grid_seam_native_enabled",False))
                    else ("official OptCuts Omega/cut topology kept fixed -> regular quad grid -> "
                          "strict full-cell crop; no M2D-to-Omega feedback")
                ),
                "optcuts_test_paper_one_way_omega_to_m2d": True,
                "optcuts_test_grid_outline_reparameterization": bool(
                    getattr(params, "optcuts_area_grid_boundary_enabled", False)
                ),
                "optcuts_test_omega_modified_from_m2d": bool(
                    getattr(params, "optcuts_area_grid_boundary_enabled", False)
                ),
                "optcuts_test_seam_topology_preserved": True,
            }
        )
        print(
            "[OPTCUTS-TEST-PAPER-OMEGA] "
            +("area-threshold grid footprint -> fixed-boundary harmonic UV"
              if bool(getattr(params,"optcuts_area_grid_boundary_enabled",False)) else
              "fixed OptCuts Omega; disabled M2D-footprint boundary reparameterization")
        )
        return parameterization

    def area_boundary_domain(parameterization: Any, grid: Any, params: Any = None):
        area_mode = bool(
            params is not None and _is_optcuts_test(params)
            and getattr(params, "optcuts_area_grid_boundary_enabled", False)
        )
        if not area_mode:
            return base_domain(parameterization, grid, params)

        from .optcuts_area_grid_boundary import (
            GridBoundaryReparameterizationError,
            apply_area_grid_boundary, fix_domain_lattice,
        )
        from .optcuts_grid_aware_seam_selection import (
            rank_grid_aware_seam_candidates, copy_selected_chart_to_original,
        )

        # This marker belongs to the CURRENT solve, not any previous
        # collection of diagnostic ZIP files.
        from .optcuts_grid_failure_diagnostics import begin_diagnostic_run
        parameterization.metrics["optcuts_area_grid_diagnostic_run_id"] = (
            begin_diagnostic_run()
        )

        # A: Choose among ACTUAL official OptCuts initial-cut/seam
        # topologies, using each candidate's own REAL production M2D grid.
        # No independently invented lattice or silent change in the 50% mask.
        source_count = len(getattr(
            parameterization, "_onestring_optcuts_candidates", []) or []) or 1
        source_status = str(parameterization.metrics.get(
            "optcuts_multistart_status", "not_reported"))
        source_error = str(parameterization.metrics.get(
            "optcuts_multistart_candidate_error", ""))
        ranked, report = rank_grid_aware_seam_candidates(
            parameterization, grid, params, base_domain,
        )
        failures = []
        selected = None
        selected_report = None
        for candidate_rank, (chart, candidate_domain, row) in enumerate(ranked, 1):
            print(
                "[GRID-SEAM-FIT] "
                f"candidate={candidate_rank}/{len(ranked)} "
                f"cut={row['source_start_index']} "
                f"rotation={row['uv_rotation_degrees']:g}deg "
                f"phase=({row.get('grid_phase_u',0):g},"
                f"{row.get('grid_phase_v',0):g}) "
                f"corners={row['corners']} "
                f"reflex={row['reflex_corners']} "
                f"score={row['score']:.5g}",
                flush=True,
            )
            try:
                # Attempt each seam candidate in grid-complexity order.
                # Only a successfully fitted candidate can be selected;
                # a simpler staircase alone is not sufficient.
                apply_area_grid_boundary(
                    chart, grid, params, domain=candidate_domain,
                )
            except GridBoundaryReparameterizationError as exc:
                print(
                    "[GRID-SEAM-FIT] candidate_failed "
                    f"rank={candidate_rank}/{len(ranked)} "
                    f"reason={str(exc)[:180]}",
                    flush=True,
                )
                row["status"] = "boundary_fit_failed"
                row["error"] = str(exc)
                failures.append(
                    f"cut={row['source_start_index']},"
                    f"uv_rotation={row['uv_rotation_degrees']:g}deg,"
                    f"score={row['score']:.5g}: {str(exc)[:210]}"
                )
                continue
            row["status"] = "boundary_fit_accepted"
            print(
                "[GRID-SEAM-FIT] candidate_accepted "
                f"rank={candidate_rank}/{len(ranked)}",
                flush=True,
            )
            selected, selected_report = chart, row
            break

        if selected is None:
            skipped = [
                f"cut={r['source_start_index']},"
                f"rotation={r['uv_rotation_degrees']:g}deg: "
                f"{r.get('status')}: {str(r.get('error',''))[:130]}"
                for r in report if r.get("status") == "invalid"
            ]
            raise GridBoundaryReparameterizationError(
                "GRID_SEAM_ALL_GRID_ORIENTATION_CANDIDATES_FAILED: "
                f"official_cut_sources={source_count}; "
                f"official_multistart_status={source_status}; "
                f"official_multistart_error={source_error[:160]}; "
                f"evaluated_uv_orientations={len(report)}; "
                f"valid_grid_candidates={len(ranked)}; "
                + " | ".join(failures + skipped)
                + "; Rotation variants do NOT generate new seams. "
                  "No exact-grid fit was silently accepted."
            )

        selected.metrics.update(
            optcuts_area_grid_seam_selection_enabled=True,
            optcuts_area_grid_seam_candidate_count=len(report),
            optcuts_area_grid_seam_ranking=report,
            optcuts_area_grid_seam_selected_start=int(
                selected_report["source_start_index"]),
            optcuts_area_grid_seam_selected_rotation_degrees=float(
                selected_report["uv_rotation_degrees"]),
            optcuts_area_grid_seam_selected_phase_u=float(
                selected_report.get("grid_phase_u",0.)),
            optcuts_area_grid_seam_selected_phase_v=float(
                selected_report.get("grid_phase_v",0.)),
            optcuts_area_grid_seam_official_source_count=source_count,
            optcuts_area_grid_seam_selected_score=float(
                selected_report["score"]),
            optcuts_area_grid_seam_selection_model=(
                "official OptCuts initial cuts plus rigid pre-grid "
                "UV orientation variants ranked on each original production "
                "M2D grid by corners, concavity, perimeter and UV quality; "
                "NO new cut topology from rotations; exact fit required"
            ),
            optcuts_area_grid_multistart_downstream_swap_enabled=False,
            optcuts_split_count_selection_pending=False,
            optcuts_area_grid_multistart_note=(
                "Selected official seam start fitted to its own real "
                "M2D footprint. Unfitted alternate charts cannot overwrite it."
            ),
        )
        # Downstream OneString retains the ORIGINAL SurfaceParameterization
        # object, so copy all paired chart data into that reference. Merely
        # returning a new chart would silently leave the old UV in the state.
        copy_selected_chart_to_original(parameterization, selected)
        # Recompute downstream CSF on the chosen final UV, but restore the
        # exact M2D grid that was evaluated BEFORE boundary deformation.
        domain = base_domain(parameterization, grid, params)
        domain = fix_domain_lattice(domain, parameterization)
        domain.parameterization = parameterization
        return domain

    def paper_m2d(grid: Any, domain: Any, params: Any = None):
        area_mode=bool(
            params is not None and _is_optcuts_test(params)
            and getattr(params,"optcuts_area_grid_boundary_enabled",False)
        )
        if area_mode:
            # Never first call base_m2d(): it would already crop partial
            # cells and can discard the very >=50% tiles we must KEEP.
            from .optcuts_area_grid_boundary import keep_cells_on_fixed_lattice
            count_x=int(domain.overlay_nx);count_y=int(domain.overlay_ny)
            original=getattr(pipeline,"_original",None)
            if original is not None and hasattr(original,"create_quad_grid"):
                full_grid=original.create_quad_grid(
                    count_x,count_y,grid.tile_size,grid.gap_size)
                full_faces=np.asarray(
                    [tile.vertex_ids for tile in (full_grid.tiles or [])],int
                ).reshape(-1,4)
                mesh_cls=original.QuadMesh
                verts=np.column_stack([
                    np.asarray(domain.uv_vertices,float),
                    np.zeros(len(domain.uv_vertices)),
                ])
                metrics={
                    "max_csf_before_split":float(
                        getattr(domain,"csf_before",getattr(domain,"max_csf",1.))),
                    "max_csf_after_split":float(
                        getattr(domain,"csf_after_split",getattr(domain,"max_csf",1.))),
                    "number_of_splits":len(domain.split_lines),
                    "m2d_overlay_total_quad_count":len(full_faces),
                    "m2d_overlay_grid_nx":count_x,
                    "m2d_overlay_grid_ny":count_y,
                    "m2d_crop_policy":"original_omega_overlap_area",
                }
                mesh=mesh_cls(
                    verts,full_faces,full_grid,"M2D",metrics,
                    list(getattr(domain,"split_lines",[])),
                )
            else:
                # Unit-test lightweight pipeline (also supplies the complete
                # overlay). Native app always takes the branch above.
                mesh=base_m2d(grid,domain,params)
            from .optcuts_area_grid_boundary import keep_cells_on_fixed_lattice
            keep=keep_cells_on_fixed_lattice(mesh,domain)
        else:
            mesh=base_m2d(grid, domain, params)
            if params is None or not _is_optcuts_test(params):
                return mesh
            polygon = _clean_polygon(getattr(domain, "boundary", []))
            if len(polygon) < 3:
                raise RuntimeError("OPTCUTS_TEST_PAPER_CROP_INVALID_OMEGA_BOUNDARY")
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=int)
            keep = np.asarray(
                [_quad_fully_contained(vertices[face, :2], polygon) for face in faces],
                dtype=bool,
            )
        faces = np.asarray(mesh.faces, dtype=int)
        retained = int(np.count_nonzero(keep))
        rejected = int(len(faces) - retained)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update(
            {
                "paper_m2d_one_way_from_fixed_omega": True,
                "paper_m2d_strict_full_cell_crop": not area_mode,
                "paper_m2d_crop_rule": (
                    "retain exactly the original-grid cells with overlap >= configured threshold"
                    if area_mode else
                    "retain quad iff the complete cell is contained in Omega"
                ),
                "optcuts_area_grid_boundary_applied_to_m2d":bool(area_mode),
                "optcuts_area_grid_coverage_threshold": (
                    float(getattr(params,"optcuts_area_grid_coverage_threshold",.5))
                    if area_mode else None
                ),
                "paper_m2d_candidate_face_count": int(len(faces)),
                "paper_m2d_retained_face_count": retained,
                "paper_m2d_rejected_partial_face_count": rejected,
                "paper_m2d_center_crop_authoritative": False,
            }
        )
        print(
            "[OPTCUTS-TEST-PAPER-M2D] "
            f"candidates={len(faces)} retained={retained} rejected_partial={rejected} "
            "rule=" + ("real-M2D-area-overlap" if area_mode else "full-cell-contained-in-fixed-Omega")
        )
        result=_compact_quad_mesh(pipeline, mesh, keep, metrics)
        if (area_mode and getattr(domain,"paper_split_policy",False)
                and not getattr(domain,"_paper_split_deferred",False)):
            # Keep the SAME downstream source-M2D Split stage used by
            # ordinary OptCuts; do not accidentally skip it when we
            # deliberately bypass its premature M2D crop.
            from .simple_split_panel_patch import apply_paper_split
            return apply_paper_split(pipeline,result,domain,params)
        return result

    pipeline._build_surface_parameterization = paper_parameterization
    pipeline._flatten_to_domain = area_boundary_domain
    pipeline._build_m2d = paper_m2d
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = paper_parameterization
        original._flatten_to_domain = area_boundary_domain
        original._build_m2d = paper_m2d

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = paper_parameterization
            glb["_flatten_to_domain"] = area_boundary_domain
            glb["_build_m2d"] = paper_m2d

    pipeline._onestring_optcuts_test_boundary_patch_installed = True


__all__ = [
    "install_optcuts_test_boundary_reparameterization_patch",
    "_clean_polygon",
    "_point_in_polygon_or_boundary",
    "_quad_fully_contained",
]
