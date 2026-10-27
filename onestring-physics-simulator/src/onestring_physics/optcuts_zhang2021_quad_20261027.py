"""2026-10-27 Omega mode: OptCuts -> Zhang et al. (2021) planar quad M2D.

    S -> official OptCuts (3D cut + paired UV chart, unchanged)
      -> [paper]     Zhang, Chai, Liu, Fu, "Quad meshing with coarse layouts
                     for planar domains", CAD 140 (2021) 103084: the authors'
                     reference programs Straighten / Segment / QMSim, run
                     exactly as their WholePipeline.bat (zhang2021_quad_reference)
      -> [OneString] seam-aware boundary correspondence + conforming boundary
                     splits + bijective SLIM re-parameterisation so that the
                     chart boundary equals the quad boundary polygon
                     (zhang2021_onestring_boundary / _reparam)
      -> M2D (= the paper's quad mesh Q, never modified), exact barycentric
         M3D lift, existing K3D / T3D / K2D / T2D.

The paper stage and the OneString stages are reported under separate metric
prefixes (``zhang2021_*`` vs ``zhang2021_onestring_*``).  Nothing falls back
to Gmsh or to the cropped regular grid: every failure raises
:class:`Zhang2021QuadModeError` with an explicit code.

This module does not modify the 2026-10-26 Gmsh prototype
(``optcuts_planar_quad_20261026``); it reuses its unstructured-grid container,
quality metrics and downstream adapters, which are format adapters only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
import time

import numpy as np

from .optcuts_planar_quad_20261026 import UnstructuredQuadGrid, _quad_quality
from .zhang2021_onestring_boundary import BoundaryCorrespondenceError, oriented_boundary_loops, quad_boundary_loop
from .zhang2021_onestring_reparam import (
    SeamConformingReparamError,
    seam_conforming_reparameterization,
    signed_uv_areas,
)
from .zhang2021_quad_metrics import scaled_jacobians, two_sided_boundary_hausdorff
from .zhang2021_quad_reference import (
    PAPER_DEFAULT_COLLAPSE_WIDTH_RATIO,
    PAPER_DOI,
    Zhang2021Error,
    Zhang2021Parameters,
    run_reference_pipeline,
)

MODE = "optcuts_zhang2021_quad_20261027"
MODE_LABEL = ("2026-10-27 | OptCuts → Zhang et al. 2021 Quad M2D "
              "(authors' reference code) + OneString seam/3D adapter")
FLAG = "optcuts_zhang2021_quad_20261027_enabled"

# OneString defaults.  N_tar = 35 is OneString's panel target, NOT the paper's
# default (4000).  eps = 0.05 relaxes the paper's 0.01 bbox-diagonal Hausdorff
# bound, without which a coarse target cannot be approached at all (the
# paper's Hausdorff constraint dominates N_tar); both are reported.
DEFAULT_TARGET_QUADS = 35
DEFAULT_HAUSDORFF_RATIO = 0.05


class Zhang2021QuadModeError(RuntimeError):
    """Explicit failure of the 2026-10-27 mode (never a silent fallback)."""


def mode_enabled(params: Any) -> bool:
    return bool(getattr(params, FLAG, False))


def _configure_initial_csf_split(domain: Any, params: Any) -> bool:
    """Temporarily skip legacy Omega CSF cuts in the Zhang2021 route.

    OptCuts topology cuts, Zhang quad meshing, boundary seam conformity and
    any *later* collision-driven splitting are not controlled by this switch.
    Both the local opt-in and the global CSF switch must be enabled to run
    the legacy initial CSF Split.  The default is deliberately OFF.
    """
    enabled = (
        bool(getattr(params, "zhang2021_initial_csf_split_enabled", False))
        and bool(getattr(params, "enable_csf_splits", True))
    )
    domain.paper_split_policy = enabled
    domain.csf_split_enabled = enabled
    domain.zhang2021_initial_csf_split_enabled = enabled
    return enabled



def zhang2021_parameters_from(params: Any) -> Zhang2021Parameters:
    zp = Zhang2021Parameters(
        target_quads=int(getattr(params, "zhang2021_target_quads", DEFAULT_TARGET_QUADS)),
        hausdorff_ratio=float(getattr(params, "zhang2021_hausdorff_ratio", DEFAULT_HAUSDORFF_RATIO)),
        collapse_width_ratio=float(getattr(params, "zhang2021_collapse_width_ratio",
                                           PAPER_DEFAULT_COLLAPSE_WIDTH_RATIO)),
        keep_features=bool(getattr(params, "zhang2021_keep_features", True)),
        timeout_seconds=float(getattr(params, "zhang2021_timeout_seconds", 3600.0)),
    )
    zp.validate()
    return zp


# ---------------------------------------------------------- downstream
# Which EXISTING OneString K3D/K2D/T2D stack consumes the Zhang M2D.  This is
# a OneString choice; neither route is part of Zhang et al. (2021).
#   optcuts_test2           : AL K3D with the certified OptCuts Seam closure
#                             (hard/soft), kinematic K2D, legacy Dual Hinge T2D.
#   paper_local_global_0918 : the SAME solvers as the 2026-09-18 Grid route
#                             (Eq.(1) local/global K3D, Eq.(5) local/global K2D
#                             on the Sec. 4.3 corner linkage, paper Dual Hinge
#                             T2D with hard non-penetration).  Like the Grid
#                             route it does NOT close OptCuts seams, and its
#                             alternating linkage needs a bipartite quad
#                             adjacency (every interior vertex of even valence).
ROUTE_OPTCUTS_TEST2 = "optcuts_test2"
ROUTE_PAPER_LOCAL_GLOBAL = "paper_local_global_0918"
DOWNSTREAM_ROUTES = (ROUTE_OPTCUTS_TEST2, ROUTE_PAPER_LOCAL_GLOBAL)


def downstream_route(params: Any) -> str:
    route = str(getattr(params, "zhang2021_downstream_route", ROUTE_OPTCUTS_TEST2) or ROUTE_OPTCUTS_TEST2)
    if route not in DOWNSTREAM_ROUTES:
        raise Zhang2021QuadModeError(
            f"ZHANG2021_DOWNSTREAM_ROUTE_INVALID: {route!r}; expected one of {DOWNSTREAM_ROUTES}")
    return route


def _route_active(params: Any, route: str) -> bool:
    return params is not None and mode_enabled(params) and downstream_route(params) == route


def odd_valence_interior_vertices(faces: np.ndarray) -> list[int]:
    """Interior vertices with an odd number of incident quads.

    For a disk-topology quad mesh the face-adjacency graph is bipartite iff
    this list is empty (the dual cycle space is generated by the vertex
    stars).  The Sec. 4.3 alternating corner linkage needs that 2-colouring.
    """
    faces = np.asarray(faces, int)
    if not len(faces):
        return []
    edge_count: dict[tuple[int, int], int] = {}
    for f in faces:
        for k in range(4):
            a, b = int(f[k]), int(f[(k + 1) % 4])
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = {v for e, c in edge_count.items() if c == 1 for v in e}
    valence = np.bincount(faces.ravel(), minlength=int(faces.max()) + 1)
    return [int(v) for v in np.unique(faces) if int(v) not in boundary and valence[v] % 2 == 1]


def _route_seam_report(vertices: np.ndarray, metrics: dict) -> dict:
    """Measured (not enforced) OptCuts Seam discrepancy of a K3D result."""
    rows = list(metrics.get("seam_exact_affine_rows") or [])
    report = {"zhang2021_route_seam_rows": int(len(rows)),
              "zhang2021_route_seam_groups": int(len(metrics.get("seam_exact_vertex_groups") or []))}
    if rows:
        from .optcuts_seam_soft_constraints import residual_stats
        stats = residual_stats(np.asarray(vertices, float), rows)
        report["zhang2021_route_seam_gap_max"] = float(stats["max"])
        report["zhang2021_route_seam_gap_rms"] = float(stats["rms"])
    return report


def _paper_local_global_k3d(pipeline: Any, target: Any, mesh: Any, parameterization: Any, params: Any):
    from .paper_local_global_solvers import optimize_paper_local_global_k3d

    print("[ZHANG2021-20261027] downstream=paper_local_global_0918: Eq.(1) local/global K3D "
          "(the 2026-09-18 Grid-route solver); OptCuts seams are NOT closed in this route", flush=True)
    out, report = optimize_paper_local_global_k3d(target, mesh, parameterization, params, pipeline=pipeline)
    seam = _route_seam_report(out.vertices, dict(getattr(mesh, "metrics", {}) or {}))
    # The 09-18 route has no Seam closure.  Remove the inherited M3D Seam
    # constraints so T3D does not enforce equalities K3D never satisfied; the
    # remaining discrepancy is reported, not hidden.
    out.metrics["seam_exact_vertex_groups"] = []
    out.metrics["seam_exact_affine_rows"] = []
    out.metrics["seam_stitch_mode"] = "none (paper_local_global_0918 route)"
    out.metrics.update(seam)
    out.metrics.update({
        "zhang2021_downstream_route": ROUTE_PAPER_LOCAL_GLOBAL,
        "zhang2021_route_k3d_solver": "paper_local_global_solvers.optimize_paper_local_global_k3d",
        "zhang2021_route_seam_closure_enforced": False,
    })
    if seam.get("zhang2021_route_seam_rows"):
        print(f"[ZHANG2021-20261027] route seam discrepancy (measured, not enforced): "
              f"max={seam['zhang2021_route_seam_gap_max']:.4g} rms={seam['zhang2021_route_seam_gap_rms']:.4g} "
              f"rows={seam['zhang2021_route_seam_rows']}", flush=True)
    return out, report


def _paper_local_global_k2d(pipeline: Any, mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
    from .paper_local_global_solvers import optimize_paper_local_global_k2d

    odd = odd_valence_interior_vertices(np.asarray(mesh_2d.faces, int))
    try:
        result, report = optimize_paper_local_global_k2d(
            mesh_2d, mesh_3d, params, progress_callback=progress_callback, pipeline=pipeline)
    except ValueError as exc:
        code = ("ZHANG2021_PAPER_LG_LINKAGE_NON_BIPARTITE" if "bipartite" in str(exc).lower()
                else "ZHANG2021_PAPER_LG_LINKAGE_FAILED")
        raise Zhang2021QuadModeError(
            f"{code}: {exc}; interior vertices of odd quad valence={len(odd)} "
            f"(first ids {odd[:12]}). The Sec. 4.3 alternating corner linkage of the "
            "paper_local_global_0918 route needs every interior vertex to have even valence; "
            "Zhang's coarse layouts contain valence-3/5 singular vertices. Use the "
            "optcuts_test2 downstream route instead (no silent fallback).") from exc
    try:
        from .eq5_iteration_history_view import render_completed_k2d
        render_completed_k2d(result.eq5_history)
    except Exception:  # noqa: BLE001 - UI rendering only
        pass
    result.metrics.update({
        "zhang2021_downstream_route": ROUTE_PAPER_LOCAL_GLOBAL,
        "zhang2021_route_k2d_solver": "paper_local_global_solvers.optimize_paper_local_global_k2d",
        "zhang2021_route_odd_valence_interior_vertices": int(len(odd)),
    })
    return result, report


def _finalize_split(mesh: Any, domain: Any, params: Any) -> Any:
    """Rebuild the unstructured grid and audit the CSF bound after the ONE Split."""
    metrics = mesh.metrics
    if metrics.get("zhang2021_split_finalized"):
        return mesh
    certified = bool(metrics.get("csf_split_bound_certified", False))
    metrics["zhang2021_csf_split_certified"] = certified
    post_faces = np.asarray(mesh.faces, int)
    post_uv = np.asarray(mesh.vertices, float)[:, :2]
    grid = mesh.grid
    grid_faces = np.asarray([t.vertex_ids for t in getattr(grid, "tiles", [])], int).reshape(-1, 4)
    grid_n = len(getattr(grid, "vertex_positions", []))
    changed = grid_n != len(post_uv) or not np.array_equal(grid_faces, post_faces)
    if changed:
        mesh.grid = UnstructuredQuadGrid(
            post_uv, post_faces,
            float(metrics.get("zhang2021_effective_uv_tile_size", getattr(grid, "tile_size", 1.0))),
            float(metrics.get("zhang2021_effective_uv_gap_size", getattr(grid, "gap_size", 0.08))))
    metrics["zhang2021_post_split_grid_reconstructed"] = bool(changed)
    metrics["zhang2021_post_split_face_count"] = int(len(post_faces))
    metrics["zhang2021_split_finalized"] = True
    if (bool(getattr(params, "enable_csf_splits", True))
            and float(getattr(domain, "csf_before", 1.0)) > float(getattr(params, "csf_split_threshold", 2.0))
            and not certified):
        raise Zhang2021QuadModeError(
            "ZHANG2021_CSF_SPLIT_UNCERTIFIED: "
            f"source_lambda_max={float(domain.csf_before):.6g} "
            f"threshold={float(getattr(params, 'csf_split_threshold', 2.0)):.6g}")
    return mesh


# ------------------------------------------------------------- geometry
def _locate(uv: np.ndarray, faces: np.ndarray, points: np.ndarray, tol: float):
    """Containing triangle + barycentric coordinates for every point (strict)."""
    from scipy.spatial import cKDTree

    uv = np.asarray(uv, float)
    faces = np.asarray(faces, int)
    P = uv[faces]
    centers = P.mean(axis=1)
    tree = cKDTree(centers)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    e1, e2 = b - a, c - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    tri_ids = np.empty(len(points), int)
    bary = np.empty((len(points), 3))
    n_faces = len(faces)
    for i, p in enumerate(np.asarray(points, float)):
        found = False
        for k in (8, 32, 128, 512, n_faces):
            k = min(k, n_faces)
            _, cand = tree.query(p, k=k)
            cand = np.atleast_1d(cand)
            d = p - a[cand]
            s = (d[:, 0] * e2[cand, 1] - d[:, 1] * e2[cand, 0]) / det[cand]
            t = (e1[cand, 0] * d[:, 1] - e1[cand, 1] * d[:, 0]) / det[cand]
            w = np.c_[1.0 - s - t, s, t]
            ok = np.flatnonzero(np.min(w, axis=1) >= -tol)
            if len(ok):
                j = ok[np.argmax(np.min(w[ok], axis=1))]
                tri_ids[i] = int(cand[j])
                bary[i] = np.clip(w[j], 0.0, 1.0) / np.sum(np.clip(w[j], 0.0, 1.0))
                found = True
                break
            if k >= n_faces:
                break
        if not found:
            raise Zhang2021QuadModeError(
                f"ZHANG2021_ONESTRING_QUAD_VERTEX_OUTSIDE_CHART: vertex={i} uv={np.round(p, 9).tolist()}")
    return tri_ids, bary


def _polygon_metrics(source_loop_uv: np.ndarray, quad_loop_uv: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        from shapely.geometry import Polygon
    except ImportError:          # optional: shapely is only used for this report
        out["zhang2021_onestring_domain_area_metrics"] = "unavailable (shapely not installed)"  # type: ignore[assignment]
        return out
    omega = Polygon(source_loop_uv)
    q = Polygon(quad_loop_uv)
    if not omega.is_valid:
        omega = omega.buffer(0)
    if not q.is_valid:
        raise Zhang2021QuadModeError("ZHANG2021_QUAD_BOUNDARY_SELF_INTERSECTS")
    area = float(omega.area)
    out.update({
        "zhang2021_omega_area": area,
        "zhang2021_quad_region_area": float(q.area),
        "zhang2021_omega_covered_fraction": float(q.intersection(omega).area / area),
        "zhang2021_omega_uncovered_fraction": float(omega.difference(q).area / area),
        "zhang2021_quad_outside_omega_fraction": float(q.difference(omega).area / area),
    })
    return out


@dataclass
class Zhang2021M2DPlan:
    quad_uv: np.ndarray            # M2D vertices (UV of the final chart == Q of the paper)
    quads: np.ndarray              # counter-clockwise in UV (OneString grid convention)
    m3d_preview: np.ndarray        # exact barycentric lift through the final chart
    chart: Any                     # ReparamResult (final chart)
    source: SimpleNamespace        # original OptCuts chart
    metrics: dict = field(default_factory=dict)


def plan_zhang2021_m2d(xyz, f3, uv, f2, params: Any) -> Zhang2021M2DPlan:
    """Paper stage + OneString adapter on one OptCuts chart (raises on failure)."""
    started = time.perf_counter()
    xyz = np.asarray(xyz, float)
    f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float)
    f2 = np.asarray(f2, int)
    zp = zhang2021_parameters_from(params)
    bin_dir = str(getattr(params, "zhang2021_bin_dir", "") or "").strip() or None
    try:
        res = run_reference_pipeline(xyz, f3, uv, f2, zp, bin_dir=bin_dir)
    except Zhang2021Error as exc:
        raise Zhang2021QuadModeError(str(exc)) from exc
    quv = np.asarray(res.quad_vertices, float)
    quads = np.asarray(res.quads, int)
    if len(quads) == 0:
        raise Zhang2021QuadModeError("ZHANG2021_EMPTY_QUAD_MESH")
    if int(res.metrics.get("zhang2021_flipped_quads", 0)) > 0:
        raise Zhang2021QuadModeError(
            f"ZHANG2021_OUTPUT_HAS_FLIPPED_QUADS: {res.metrics['zhang2021_flipped_quads']} "
            "(the paper's output is rejected, not repaired)")
    if int(res.metrics.get("zhang2021_quad_boundary_loops", 1)) != 1:
        raise Zhang2021QuadModeError(
            f"ZHANG2021_QUAD_MESH_NOT_A_DISK: boundary loops={res.metrics.get('zhang2021_quad_boundary_loops')}")
    try:
        rp = seam_conforming_reparameterization(
            xyz, f3, uv, f2, quv, quads,
            snap_nodes=bool(getattr(params, "zhang2021_snap_seam_nodes", True)),
            align_pairs=bool(getattr(params, "zhang2021_align_seam_pairs", True)),
            seam_matching=bool(getattr(params, "zhang2021_seam_matching", True)))
    except (SeamConformingReparamError, BoundaryCorrespondenceError) as exc:
        raise Zhang2021QuadModeError(str(exc)) from exc
    paper_quads = int(len(quads))
    # The paper's Q unless OneString seam matching inserted poly-chords.
    quv = np.asarray(rp.quad_uv, float)
    quads = np.asarray(rp.quads, int)
    sj = scaled_jacobians(quv, quads)
    # The final chart's boundary IS the quad boundary polygon: verify areas.
    q_loop = quad_boundary_loop(quads)
    q_poly = quv[q_loop]
    poly_area = 0.5 * abs(float(np.sum(q_poly[:, 0] * np.roll(q_poly[:, 1], -1)
                                       - q_poly[:, 1] * np.roll(q_poly[:, 0], -1))))
    chart_area = float(np.sum(np.abs(signed_uv_areas(rp.uv, rp.f2))))
    quad_area = 0.0
    for q in quads:
        pts = quv[q]
        quad_area += 0.5 * abs(float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))))
    rel = max(abs(chart_area - poly_area), abs(quad_area - poly_area)) / max(poly_area, 1e-300)
    if rel > 1e-8:
        raise Zhang2021QuadModeError(
            f"ZHANG2021_ONESTRING_CHART_DOES_NOT_TILE_QUAD_REGION: chart={chart_area:.12g} "
            f"quads={quad_area:.12g} polygon={poly_area:.12g}")
    span = float(np.max(np.ptp(rp.uv, axis=0)))
    tri, bary = _locate(rp.uv, rp.f2, quv, 1e-9)
    m3d = np.einsum("ij,ijk->ik", bary, rp.xyz[rp.f3[tri]])
    # OneString's M2D convention is counter-clockwise quads in UV (as QuadGrid).
    orient = float(np.sum(signed_uv_areas(quv, quads[:, [0, 1, 2]])) + np.sum(signed_uv_areas(quv, quads[:, [0, 2, 3]])))
    faces_ccw = quads[:, ::-1].copy() if orient < 0 else quads.copy()
    q2 = _quad_quality(np.column_stack((quv, np.zeros(len(quv)))), faces_ccw)
    q3 = _quad_quality(m3d, faces_ccw)
    sj3 = _scaled_jacobian_3d(m3d, faces_ccw)
    src_loop = oriented_boundary_loops(f2)
    if len(src_loop) != 1:
        raise Zhang2021QuadModeError("ZHANG2021_SOURCE_CHART_NOT_DISK")
    src_loop_uv = uv[src_loop[0]]
    dh, _dh_q_to_src, _dh_src_to_q = two_sided_boundary_hausdorff(
        src_loop_uv, [list(range(len(src_loop_uv)))], q_poly, [list(range(len(q_poly)))])
    bbox_diag = float(np.linalg.norm(np.ptp(uv, axis=0)))
    metrics: dict[str, Any] = {
        "requested_omega_parameterization_mode": MODE,
        "zhang2021_quad_20261027_enabled": True,
        "zhang2021_mode_label": MODE_LABEL,
        "zhang2021_paper_doi": PAPER_DOI,
        **{k: v for k, v in res.metrics.items()},
        "zhang2021_requested_vs_actual_quads": (
            f"requested {zp.target_quads}; paper output {paper_quads}; final M2D {len(quads)}"),
        "zhang2021_target_reached": bool(paper_quads == zp.target_quads),
        "zhang2021_paper_output_quads": paper_quads,
        "zhang2021_final_m2d_quads": int(len(quads)),
        "zhang2021_m2d_is_unmodified_paper_output": bool(len(quads) == paper_quads),
        "zhang2021_m2d_scaled_jacobian_min": float(np.min(sj)),
        "zhang2021_m2d_scaled_jacobian_mean": float(np.mean(sj)),
        "zhang2021_m3d_scaled_jacobian_min": float(np.min(sj3)),
        "zhang2021_m3d_scaled_jacobian_mean": float(np.mean(sj3)),
        "zhang2021_boundary_hausdorff_vs_source_omega": float(dh),
        "zhang2021_boundary_hausdorff_vs_source_omega_percent_bbox": 100.0 * float(dh) / max(bbox_diag, 1e-300),
        **_polygon_metrics(src_loop_uv, q_poly),
        **rp.metrics,
        "zhang2021_onestring_final_chart_equals_quad_region": True,
        "zhang2021_onestring_final_chart_area_relative_error": float(rel),
        "zhang2021_onestring_surface_coverage": (
            "complete: the final chart is a bijection onto all of S and its boundary is the quad "
            "boundary polygon, so the lifted M2D covers S without gaps or outside area"),
        "zhang2021_onestring_m2d_faces_reversed_to_ccw": bool(orient < 0),
        "zhang2021_onestring_lift_model": "strict containing triangle of the final chart + barycentric",
        **{"zhang2021_m2d_" + k: (int(v) if k.endswith("_face_index") else float(v)) for k, v in q2.items()},
        **{"zhang2021_m3d_" + k: (int(v) if k.endswith("_face_index") else float(v)) for k, v in q3.items()},
        "zhang2021_total_seconds": float(time.perf_counter() - started),
        "zhang2021_uv_span": span,
    }
    source = SimpleNamespace(xyz=xyz.copy(), f3=f3.copy(), uv=uv.copy(), f2=f2.copy())
    return Zhang2021M2DPlan(quad_uv=quv, quads=faces_ccw, m3d_preview=m3d, chart=rp,
                            source=source, metrics=metrics)


# ------------------------------------------- coordinated Omega + quad search
# OneString extension implementing the 2026-10-26 design order
#   OptCuts -> [h: boundary simplification + bijective SLIM of Omega]
#           -> Zhang et al. 2021 quad meshing of Omega_1
#           -> seam/boundary adapter -> M2D/M3D quality evaluation
# Omega and Q are coordinated by an explicit candidate search (pre-conditioning
# tolerance x Zhang epsilon) scored with M2D/M3D and downstream criteria; every
# candidate and its score is reported.  No candidate is accepted that violates a
# hard constraint (inverted UV, unaccepted distortion, failed adapter).
SELECTION_WEIGHTS = {
    "count": 1.0,          # |ln(actual / target)|
    "odd_valence": 1.0,    # per odd-valence interior vertex (Sec. 4.3 linkage, C6)
    "low_tip_valence": 0.5,  # per seam node below its ideal 3D valence (C7)
    "m3d_angle": 1.0,      # max(0, 45 deg - min M3D angle) / 45 deg
    "j_min": 0.5,          # 1 - min scaled Jacobian of Q
    "distortion": 5.0,     # final chart SD mean / OptCuts SD mean - 1
}


def _float_list(text: Any, default: list[float]) -> list[float]:
    if text is None or (isinstance(text, str) and not text.strip()):
        return list(default)
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    return [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]


def plan_zhang2021_m2d_coordinated(xyz, f3, uv, f2, params: Any) -> Zhang2021M2DPlan:
    """Pre-condition Omega (h) BEFORE quad meshing, then pick the best (h, Q) candidate."""
    from dataclasses import replace as _dc_replace
    from .zhang2021_onestring_precondition import precondition_omega
    from .zhang2021_onestring_reparam import _SDProblem

    started = time.perf_counter()
    xyz = np.asarray(xyz, float)
    f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float)
    f2 = np.asarray(f2, int)
    tols = _float_list(getattr(params, "zhang2021_precondition_tolerances", None), [0.0, 0.02, 0.04, 0.06, 0.1])
    epss = _float_list(getattr(params, "zhang2021_candidate_hausdorff_ratios", None),
                       [float(getattr(params, "zhang2021_hausdorff_ratio", DEFAULT_HAUSDORFF_RATIO))])
    mean_lim = float(getattr(params, "zhang2021_precondition_sd_mean_ratio_max", 1.08))
    p95_lim = float(getattr(params, "zhang2021_precondition_sd_p95_ratio_max", 1.30))
    target = int(getattr(params, "zhang2021_target_quads", DEFAULT_TARGET_QUADS))
    route = downstream_route(params)
    sa = signed_uv_areas(uv, f2)
    mirror = np.array([-1.0, 1.0]) if np.all(sa < 0) else np.array([1.0, 1.0])
    sd0_mean = float(_SDProblem(xyz, f3, f2, len(uv)).energy(uv * mirror)[0])
    rows: list[dict] = []
    best: tuple[float, Zhang2021M2DPlan, dict] | None = None
    for tol in tols:
        pre = precondition_omega(xyz, f3, uv, f2, tol, sd_mean_ratio_max=mean_lim, sd_p95_ratio_max=p95_lim)
        if not pre.accepted:
            rows.append({"precondition_tol": tol, "status": "precondition_rejected", "reason": pre.reason,
                         **{k: v for k, v in pre.metrics.items() if isinstance(v, (int, float))}})
            continue
        for eps in epss:
            row: dict[str, Any] = {"precondition_tol": tol, "zhang_eps": eps}
            row.update({k: v for k, v in pre.metrics.items() if isinstance(v, (int, float))})
            try:
                p = _dc_replace(params, zhang2021_hausdorff_ratio=eps) if hasattr(params, "__dataclass_fields__") \
                    else SimpleNamespace(**{**vars(params), "zhang2021_hausdorff_ratio": eps})
                plan = plan_zhang2021_m2d(pre.xyz, pre.f3, pre.uv, pre.f2, p)
            except Zhang2021QuadModeError as exc:
                row.update(status="failed", reason=str(exc)[:300])
                rows.append(row)
                continue
            m = plan.metrics
            n = int(len(plan.quads))
            odd = len(odd_valence_interior_vertices(plan.quads))
            low_tip = int(m.get("zhang2021_onestring_seam_nodes_low_3d_valence", 0))
            ang3 = float(m.get("zhang2021_m3d_angle_min_degrees", 0.0))
            jmin = float(m.get("zhang2021_j_min", 0.0))
            sd_ratio = float(m.get("zhang2021_onestring_sd_mean_after", sd0_mean)) / max(sd0_mean, 1e-300)
            W = SELECTION_WEIGHTS
            terms = {
                "count": W["count"] * abs(float(np.log(max(n, 1) / max(target, 1)))),
                "odd_valence": W["odd_valence"] * odd,
                "low_tip_valence": W["low_tip_valence"] * low_tip,
                "m3d_angle": W["m3d_angle"] * max(0.0, 45.0 - ang3) / 45.0,
                "j_min": W["j_min"] * max(0.0, 1.0 - jmin),
                "distortion": W["distortion"] * max(0.0, sd_ratio - 1.0),
            }
            score = float(sum(terms.values()))
            feasible = not (route == ROUTE_PAPER_LOCAL_GLOBAL and odd > 0)
            row.update(status="ok" if feasible else "infeasible_for_route", quads=n, patches=m.get("zhang2021_n_patches"),
                       odd_valence_interior=odd, low_tip_valence=low_tip, m3d_angle_min=ang3, j_min=jmin,
                       final_sd_mean_ratio_vs_optcuts=sd_ratio, score=score,
                       **{"score_" + k: float(v) for k, v in terms.items()})
            rows.append(row)
            print(f"[ZHANG2021-COORD] tol={tol} eps={eps}: quads={n} odd={odd} lowtip={low_tip} "
                  f"M3Dmin={ang3:.1f} SDx{sd_ratio:.3f} score={score:.3f} {'' if feasible else '(infeasible)'}",
                  flush=True)
            if feasible and (best is None or score < best[0]):
                best = (score, plan, {"tol": tol, "eps": eps, "pre": pre.metrics})
    if best is None:
        raise Zhang2021QuadModeError(
            "ZHANG2021_NO_FEASIBLE_COORDINATED_CANDIDATE: " +
            "; ".join(f"tol={r.get('precondition_tol')} eps={r.get('zhang_eps')}: {r.get('status')} "
                      f"{str(r.get('reason', ''))[:80]}" for r in rows))
    _, plan, chosen = best
    # Report against the ORIGINAL OptCuts chart Omega_0 (not Omega_1).
    src_loop = oriented_boundary_loops(f2)[0]
    q_loop = quad_boundary_loop(plan.quads)
    dh, _, _ = two_sided_boundary_hausdorff(uv[src_loop], [list(range(len(src_loop)))],
                                            plan.quad_uv[q_loop], [list(range(len(q_loop)))])
    diag = float(np.linalg.norm(np.ptp(uv, axis=0)))
    plan.metrics.update({
        "zhang2021_coordinated_search": True,
        "zhang2021_coordinated_model": (
            "OneString (not Zhang 2021): Omega_1 = h(Omega_0) by boundary simplification "
            "(Douglas-Peucker per seam run, Hausdorff <= tol, seam nodes fixed) + bijective SLIM; "
            "Zhang 2021 meshes Omega_1; candidates (tol x eps) scored on count / C6 / C7 / M3D / distortion"),
        "zhang2021_coordinated_candidates": rows,
        "zhang2021_coordinated_selection_weights": dict(SELECTION_WEIGHTS),
        "zhang2021_coordinated_chosen_precondition_tol": float(chosen["tol"]),
        "zhang2021_coordinated_chosen_hausdorff_ratio": float(chosen["eps"]),
        "zhang2021_coordinated_seconds": float(time.perf_counter() - started),
        "zhang2021_boundary_hausdorff_vs_preconditioned_omega_percent_bbox":
            plan.metrics.get("zhang2021_boundary_hausdorff_vs_source_omega_percent_bbox"),
        "zhang2021_boundary_hausdorff_vs_source_omega": float(dh),
        "zhang2021_boundary_hausdorff_vs_source_omega_percent_bbox": 100.0 * float(dh) / max(diag, 1e-300),
        "zhang2021_optcuts_sd_mean": sd0_mean,
        **{k: v for k, v in chosen["pre"].items()},
    })
    print(f"[ZHANG2021-COORD] chosen tol={chosen['tol']} eps={chosen['eps']} quads={len(plan.quads)} "
          f"(target {target}); {len(rows)} candidates", flush=True)
    plan.source = SimpleNamespace(xyz=xyz.copy(), f3=f3.copy(), uv=uv.copy(), f2=f2.copy())
    return plan


def _scaled_jacobian_3d(xyz: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Minimum corner sine of each (non-planar) 3D quad w.r.t. its average normal."""
    P = np.asarray(xyz, float)[np.asarray(quads, int)]
    n = np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    out = np.full(len(P), np.inf)
    for k in range(4):
        a = P[:, (k + 1) % 4] - P[:, k]
        b = P[:, (k - 1) % 4] - P[:, k]
        cr = np.einsum("ij,ij->i", np.cross(a, b), n)
        s = cr / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        out = np.minimum(out, s)
    return out


def _drop_uv_caches(result: Any) -> None:
    for name in ("_onestring_uv_triangle_kdtree_cache", "_onestring_uv_vertex_kdtree_cache"):
        if hasattr(result, name):
            try:
                delattr(result, name)
            except Exception:
                setattr(result, name, None)
    if hasattr(result, "triangle_acceleration"):
        result.triangle_acceleration = None


# ---------------------------------------------------------------- hooks
def install_zhang2021_quad_20261027(pipeline: Any) -> None:
    """Outermost hook; inactive unless ``optcuts_zhang2021_quad_20261027_enabled``."""
    if getattr(pipeline, "_zhang2021_quad_20261027_installed", False):
        return
    base_parameterization = pipeline._build_surface_parameterization
    base_domain = pipeline._flatten_to_domain
    base_m2d = pipeline._build_m2d

    def with_parameterization(surface, target, grid, params):
        result = base_parameterization(surface, target, grid, params)
        if not mode_enabled(params):
            return result
        if str(getattr(result, "method", "")) not in ("optcuts_test", "optcuts"):
            raise Zhang2021QuadModeError(
                "ZHANG2021_MODE_REQUIRES_OPTCUTS_CHART: parameterization method="
                f"{getattr(result, 'method', None)!r}")
        route = downstream_route(params)
        planner = (plan_zhang2021_m2d_coordinated if bool(getattr(params, "zhang2021_precondition_enabled", True))
                   else plan_zhang2021_m2d)
        plan = planner(result.surface_vertices_3d, result.surface_faces,
                       result.uv_vertices_2d, result.uv_faces, params)
        rp = plan.chart
        # OptCuts multi-start (closed genus-0 inputs) attaches alternative
        # charts that the Split stage may swap in by Split count.  The Zhang
        # M2D and the re-parameterised chart belong to THIS result only, so a
        # later swap would silently desynchronise M2D and M3D: disable it.
        dropped = len(getattr(result, "_onestring_optcuts_candidates", None) or [])
        if dropped:
            result._onestring_optcuts_candidates = []
            result._onestring_optcuts_candidate_qualities = []
            result.metrics["optcuts_split_count_selection_status"] = (
                "disabled: the Zhang2021 M2D and re-parameterised chart belong to this OptCuts result")
        result.metrics["optcuts_split_count_selection_pending"] = False
        result.metrics["zhang2021_optcuts_split_count_candidates_dropped"] = int(dropped)
        result.metrics["zhang2021_downstream_route"] = route
        result._zhang2021_plan = plan
        result._zhang2021_source_chart = plan.source
        result.surface_vertices_3d = rp.xyz
        result.surface_faces = rp.f3
        result.uv_vertices_2d = rp.uv
        result.uv_faces = rp.f2
        loop = oriented_boundary_loops(rp.f2)[0]
        result.omega_boundary = np.vstack((rp.uv[loop], rp.uv[loop[0]]))
        _drop_uv_caches(result)
        result.metrics.update(plan.metrics)
        result.metrics["zhang2021_source_chart_vertices"] = int(len(plan.source.uv))
        result.metrics["zhang2021_final_chart_vertices"] = int(len(rp.uv))
        print(f"[ZHANG2021-20261027] paper quads={len(plan.quads)} "
              f"(requested {plan.metrics['zhang2021_target_quads']}) patches={plan.metrics['zhang2021_n_patches']} "
              f"J_min={plan.metrics['zhang2021_j_min']:.3f} dH={plan.metrics['zhang2021_d_h_percent_bbox']:.3f}% "
              f"| OneString chart SD {plan.metrics['zhang2021_onestring_sd_mean_before']:.4f}->"
              f"{plan.metrics['zhang2021_onestring_sd_mean_after']:.4f} "
              f"hard/soft seam branches={plan.metrics['zhang2021_onestring_hard_seam_branches']}/"
              f"{plan.metrics['zhang2021_onestring_soft_seam_branches']}", flush=True)
        return result

    def with_domain(parameterization, grid, params=None):
        if not mode_enabled(params):
            return base_domain(parameterization, grid, params)
        plan = getattr(parameterization, "_zhang2021_plan", None)
        if plan is None:
            raise Zhang2021QuadModeError("ZHANG2021_DOMAIN_WITHOUT_PLAN")
        original = getattr(pipeline, "_original", None)
        cls = getattr(original, "PlanarDomain", None)
        area_metrics_fn = getattr(pipeline, "_paper_lambda_vertices_for_existing_omega", None)
        if cls is None or area_metrics_fn is None:
            raise Zhang2021QuadModeError("ZHANG2021_DOMAIN_DEPENDENCY_UNAVAILABLE: source area field")
        vertex, triangle, _differential = area_metrics_fn(parameterization)
        vertex = np.asarray(vertex, float)
        triangle = np.asarray(triangle, float)
        if (len(vertex) != len(parameterization.uv_vertices_2d) or not len(triangle)
                or not np.isfinite(triangle).all() or not np.isfinite(vertex).all()):
            raise Zhang2021QuadModeError("ZHANG2021_DOMAIN_CSF_INVALID")
        threshold = float(getattr(params, "csf_split_threshold", 2.0))
        if not np.isfinite(threshold) or threshold < 1.0:
            raise Zhang2021QuadModeError("ZHANG2021_DOMAIN_CSF_THRESHOLD_INVALID")
        result = cls(np.asarray(parameterization.omega_boundary, float).copy(),
                     np.asarray(parameterization.uv_vertices_2d, float).copy(),
                     f"{parameterization.method} + Zhang2021 quad M2D (OneString chart adapter)",
                     vertex.copy(), [])
        result.parameterization = parameterization
        result._zhang2021_plan = plan
        initial_split_enabled = _configure_initial_csf_split(result, params)
        result.csf_before = float(np.max(triangle))
        result.csf_after_split = result.csf_before
        result.csf_split_threshold = threshold
        result.max_csf_splits = max(0, int(getattr(params, "max_csf_splits", 64)))
        result.csf_model = "final (re-parameterised) paired UV/3D triangle area Jacobian"
        result.csf_split_exactness_label = (
            "pending unstructured M2D Split audit" if initial_split_enabled
            else "initial Omega CSF Split disabled; no distortion bound certified"
        )
        result.paper_split_diagnostics = {
            "status": ("pending_unstructured_m2d" if initial_split_enabled
                       else "disabled_in_zhang2021_mode"),
            "area_expansion_bound": threshold,
            "lambda_max": result.csf_before,
            "split_rule": ("complete existing quad-edge-chain only" if initial_split_enabled
                           else "not run: retain the unsplit Zhang2021 M2D"),
        }
        result.localize_csf_splits = False
        result.localized_split_segments = []
        result.reference_mode = False
        parameterization.metrics["split_diagnostics"] = dict(result.paper_split_diagnostics)
        parameterization.metrics["zhang2021_initial_csf_split_enabled"] = bool(initial_split_enabled)
        parameterization.metrics["zhang2021_independent_domain"] = True
        parameterization.metrics["zhang2021_legacy_regular_overlay_bypassed"] = True
        return result

    def with_m2d(grid, domain, params=None):
        if not mode_enabled(params):
            return base_m2d(grid, domain, params)
        plan = getattr(domain, "_zhang2021_plan", None)
        parameterization = getattr(domain, "parameterization", None)
        if plan is None or parameterization is None:
            raise Zhang2021QuadModeError("ZHANG2021_M2D_WITHOUT_PLAN")
        uv = np.asarray(plan.quad_uv, float)
        faces = np.asarray(plan.quads, int)
        from .optcuts_planar_quad_refine_20261026 import validate_disk_mesh
        try:
            validate_disk_mesh(uv, faces)
        except ValueError as exc:
            raise Zhang2021QuadModeError("ZHANG2021_M2D_TOPOLOGY_INVALID: " + str(exc)) from exc
        edges = np.linalg.norm(np.roll(uv[faces], -1, axis=1) - uv[faces], axis=2)
        effective_size = float(np.median(edges))
        original_size = max(float(getattr(grid, "tile_size", 1.0)), 1e-12)
        effective_gap = float(getattr(grid, "gap_size", 0.08)) * effective_size / original_size
        irregular = UnstructuredQuadGrid(uv, faces, effective_size, effective_gap)
        report = {k: v for k, v in plan.metrics.items() if not isinstance(v, (list, dict))}
        report.update({
            "zhang2021_m2d_faces": int(len(faces)),
            "zhang2021_effective_uv_tile_size": effective_size,
            "zhang2021_effective_uv_gap_size": effective_gap,
            "planar_quad_engine": "Zhang et al. 2021 reference implementation (authors' code); NOT Gmsh",
        })
        xyz = np.column_stack((uv, np.zeros(len(uv))))
        mesh = pipeline._original.QuadMesh(xyz, faces.copy(), irregular, "M2D", report, [])
        from .optcuts_seam_exact_closure import SeamTopologyMismatch, discover_exact_seam_topology
        try:
            seam = discover_exact_seam_topology(parameterization, uv, faces)
        except SeamTopologyMismatch as exc:
            raise Zhang2021QuadModeError("ZHANG2021_SEAM_PREFLIGHT_FAILED: " + str(exc)) from exc
        mesh.metrics.update({
            "zhang2021_seam_preflight_status": str(seam["status"]),
            "zhang2021_seam_stitch_mode": str(seam.get("seam_stitch_mode", "none")),
            "zhang2021_seam_paired_edges": int(len(seam.get("paired_edges", []))),
            "zhang2021_seam_grid_edges": int(seam.get("seam_grid_edges", 0)),
            "zhang2021_seam_unclassified_edges": int(seam.get("unclassified_boundary_edges", 0)),
            "zhang2021_seam_mixed_cut_open_edges": int(seam.get("mixed_seam_open_grid_edges", 0)),
            "zhang2021_seam_affine_rows": int(len(seam.get("affine_rows", []))),
        })
        if str(seam["status"]) == "nonconforming":
            raise Zhang2021QuadModeError(
                "ZHANG2021_SEAM_NOT_CONFORMING: M2D boundary edges are not covered by the "
                f"re-parameterised source boundary; unclassified={seam.get('unclassified_boundary_edges')}")
        print(f"[ZHANG2021-20261027] M2D {len(faces)} quads; seam={seam['status']}/"
              f"{seam.get('seam_stitch_mode')}; min M2D angle={report['zhang2021_m2d_angle_min_degrees']:.2f}; "
              f"min M3D angle={report['zhang2021_m3d_angle_min_degrees']:.2f}", flush=True)
        if bool(getattr(domain, "paper_split_policy", False)):
            if bool(getattr(domain, "_paper_split_deferred", False)):
                # The Simple Split wrapper outside this hook is the single
                # Split owner (it calls apply_paper_split on our output).
                # Splitting here as well would split twice and overwrite the
                # Split report; the outer post-split hook audits the result.
                mesh.metrics["zhang2021_csf_split_owner"] = "simple_split_wrapper"
            else:
                from .simple_split_panel_patch import apply_paper_split
                mesh = apply_paper_split(pipeline, mesh, domain, params)
                mesh.metrics["zhang2021_csf_split_owner"] = "zhang2021_m2d_hook"
                mesh = _finalize_split(mesh, domain, params)
        else:
            # Preserve Zhang's M2D exactly: the outer Simple Split wrapper
            # sees paper_split_policy=False and an empty split_lines list,
            # so it cannot run the legacy curvature/CSF cutting algorithm.
            mesh.metrics.update({
                "zhang2021_csf_split_owner": "disabled_before_m2d",
                "zhang2021_initial_csf_split_enabled": False,
                "zhang2021_csf_split_certified": False,
                "zhang2021_csf_split_status": "disabled_in_zhang2021_mode",
            })
        return mesh

    _wire(pipeline, "_build_surface_parameterization", with_parameterization)
    _wire(pipeline, "_flatten_to_domain", with_domain)
    _wire(pipeline, "_build_m2d", with_m2d)
    _install_route_hooks(pipeline)
    _register_post_simple_split_hooks()
    pipeline._zhang2021_quad_20261027_installed = True


def _wire(pipeline: Any, name: str, fn: Any) -> None:
    setattr(pipeline, name, fn)
    original = getattr(pipeline, "_original", None)
    if original is not None:
        setattr(original, name, fn)
    for build_fn in (getattr(pipeline, "build_onestring_design", None),
                     getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
                     getattr(original, "build_onestring_design", None) if original is not None else None):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb[name] = fn


def _install_route_hooks(pipeline: Any) -> None:
    """K3D/K2D route dispatch; a no-op unless this mode selects the 09-18 route."""
    inner_k3d = pipeline._optimize_k3d
    inner_k2d = pipeline._optimize_k2d

    def zhang2021_route_k3d(target, mesh, parameterization, params):
        if _route_active(params, ROUTE_PAPER_LOCAL_GLOBAL):
            return _paper_local_global_k3d(pipeline, target, mesh, parameterization, params)
        return inner_k3d(target, mesh, parameterization, params)

    def zhang2021_route_k2d(mesh_2d, mesh_3d, params, progress_callback=None):
        if _route_active(params, ROUTE_PAPER_LOCAL_GLOBAL):
            return _paper_local_global_k2d(pipeline, mesh_2d, mesh_3d, params, progress_callback)
        return inner_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

    _wire(pipeline, "_optimize_k3d", zhang2021_route_k3d)
    _wire(pipeline, "_optimize_k2d", zhang2021_route_k2d)


def _install_post_simple_split_hooks(pipeline: Any) -> None:
    """Outermost hooks, installed after the Simple Split launcher wrapped M2D/K2D.

    * M2D: when Simple Split owned the Split (``_paper_split_deferred``), audit
      its single result here (grid rebuild + CSF certification).
    * K3D/K2D: the 09-18 route is dispatched OUTSIDE the Simple Split wrappers,
      exactly like the 2026-09-18 hybrid route dispatch, so both routes run
      the same solver calls.
    """
    if getattr(pipeline, "_zhang2021_post_simple_split_hooks_installed", False):
        return
    inner_m2d = pipeline._build_m2d

    def zhang2021_post_split_m2d(grid, domain, params=None):
        mesh = inner_m2d(grid, domain, params)
        if (params is not None and mode_enabled(params)
                and mesh.metrics.get("zhang2021_csf_split_owner") == "simple_split_wrapper"):
            mesh = _finalize_split(mesh, domain, params)
        return mesh

    _wire(pipeline, "_build_m2d", zhang2021_post_split_m2d)
    _install_route_hooks(pipeline)
    pipeline._zhang2021_post_simple_split_hooks_installed = True


def _register_post_simple_split_hooks() -> None:
    try:
        from . import simple_split_panel_patch as simple_split_module
    except Exception:  # noqa: BLE001 - launcher without Simple Split
        return
    if getattr(simple_split_module, "_zhang2021_post_split_hook_registered", False):
        return
    previous = simple_split_module.install_simple_split_panel_patch

    def install_simple_split_then_zhang2021(pipeline_module, optimization_debug_module):
        previous(pipeline_module, optimization_debug_module)
        _install_post_simple_split_hooks(pipeline_module)

    simple_split_module.install_simple_split_panel_patch = install_simple_split_then_zhang2021
    simple_split_module._zhang2021_post_split_hook_registered = True


__all__ = [
    "MODE", "MODE_LABEL", "FLAG", "DEFAULT_TARGET_QUADS", "DEFAULT_HAUSDORFF_RATIO",
    "Zhang2021QuadModeError", "Zhang2021M2DPlan", "mode_enabled", "zhang2021_parameters_from",
    "plan_zhang2021_m2d", "plan_zhang2021_m2d_coordinated", "install_zhang2021_quad_20261027",
]
