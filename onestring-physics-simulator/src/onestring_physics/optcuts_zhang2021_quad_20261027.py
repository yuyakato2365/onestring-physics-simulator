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

# OneString defaults.  N_tar is ONLY the paper's density input (its target edge
# length l_tar = sqrt(area / N_tar); the paper's own default is 4000).  It is
# not a panel-count goal: no selection term depends on the resulting count,
# several densities are tried (``zhang2021_candidate_density_factors``) and the
# final count is whatever the quality-ranked feasible candidate has.
DEFAULT_DENSITY_N_TAR = 35
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
        target_quads=int(getattr(params, "zhang2021_target_quads", DEFAULT_DENSITY_N_TAR)),
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
#   paper_local_global_0918 : (default) the solvers of the 2026-09-18 Grid
#                             route (Eq.(1) local/global K3D, Eq.(5) K2D on the
#                             Sec. 4.3 corner linkage, paper Dual Hinge T2D with
#                             hard non-penetration).  In this mode the OptCuts
#                             seams are exact K3D variables (shared vertices /
#                             T-vertex substitution, zhang2021_seam_reduction);
#                             the linkage needs every interior vertex of even
#                             valence (Phase-1 constraint C7).
#   optcuts_test2           : AL K3D with the hard OptCuts seam projector,
#                             kinematic K2D, legacy Dual Hinge T2D.
# Both routes are followed by the K3D gate (quality + exact seam closure).
ROUTE_OPTCUTS_TEST2 = "optcuts_test2"
ROUTE_PAPER_LOCAL_GLOBAL = "paper_local_global_0918"
DOWNSTREAM_ROUTES = (ROUTE_OPTCUTS_TEST2, ROUTE_PAPER_LOCAL_GLOBAL)


DEFAULT_DOWNSTREAM_ROUTE = ROUTE_PAPER_LOCAL_GLOBAL


def downstream_route(params: Any) -> str:
    route = str(getattr(params, "zhang2021_downstream_route", DEFAULT_DOWNSTREAM_ROUTE) or DEFAULT_DOWNSTREAM_ROUTE)
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


def _seam_reduction_from_metrics(n_vertices: int, metrics: dict):
    """Exact OptCuts seam constraints of the M3D (certified at the lift) as x = E z."""
    from .zhang2021_seam_reduction import SeamReductionError, build_seam_reduction

    rows = list(metrics.get("seam_exact_affine_rows") or [])
    groups = list(metrics.get("seam_exact_vertex_groups") or [])
    mode = str(metrics.get("seam_stitch_mode", "none"))
    if not rows and not groups:
        return None
    if mode != "hard":
        raise Zhang2021QuadModeError(
            f"ZHANG2021_SEAM_NOT_HARD_CLOSABLE: the M3D seam preflight is {mode!r}; the Zhang2021 mode never "
            "optimises K3D with a soft or missing OptCuts seam")
    try:
        return build_seam_reduction(n_vertices, groups, rows)
    except SeamReductionError as exc:
        raise Zhang2021QuadModeError(str(exc)) from exc


def _paper_local_global_k3d(pipeline: Any, target: Any, mesh: Any, parameterization: Any, params: Any):
    from .paper_local_global_solvers import optimize_paper_local_global_k3d

    metrics_in = dict(getattr(mesh, "metrics", {}) or {})
    reduction = _seam_reduction_from_metrics(len(mesh.vertices), metrics_in)
    print("[ZHANG2021-20261027] downstream=paper_local_global_0918: Eq.(1) local/global K3D (the 2026-09-18 "
          "Grid-route solver) with the OptCuts seams as exact shared / T-vertex variables "
          f"({'none' if reduction is None else f'{reduction.n} -> {reduction.m} unknowns'})", flush=True)
    out, report = optimize_paper_local_global_k3d(target, mesh, parameterization, params, pipeline=pipeline,
                                                  seam_reduction=reduction)
    out.metrics.update({
        "zhang2021_downstream_route": ROUTE_PAPER_LOCAL_GLOBAL,
        "zhang2021_route_k3d_solver": ("paper_local_global_solvers.optimize_paper_local_global_k3d "
                                       "(seam_reduction: exact OptCuts seam variables)"),
        "zhang2021_route_seam_closure_enforced": reduction is not None,
    })
    if reduction is not None:
        out.metrics["seam_exact_vertex_groups"] = [list(map(int, g)) for g in metrics_in.get("seam_exact_vertex_groups", [])]
        out.metrics["seam_exact_affine_rows"] = list(metrics_in.get("seam_exact_affine_rows", []))
        out.metrics["seam_stitch_mode"] = "hard"
        out.metrics["seam_exact_closure_enabled"] = True
        out.metrics["seam_exact_closure_model"] = "eliminated variables inside every local/global iterate"
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
    quad_uv: np.ndarray            # M2D vertices (UV of the final chart)
    quads: np.ndarray              # counter-clockwise in UV (OneString grid convention)
    m3d_preview: np.ndarray        # exact barycentric lift through the final chart
    chart: Any                     # ReparamResult (final chart)
    source: SimpleNamespace        # original OptCuts chart
    metrics: dict = field(default_factory=dict)
    m2d_report: Any = None         # PanelQualityReport (M2D, UV)
    m3d_report: Any = None         # PanelQualityReport (M3D, lifted)


_MODULE_PLAN_CACHE: dict = {}


class Zhang2021CandidatesExhausted(Zhang2021QuadModeError):
    """No further feasible candidate for the requested rank (Phase-3 loop end)."""


def stage_thresholds(params: Any, stage: str):
    """Scale-free acceptance limits of one stage ('m2d', 'm3d', 'k3d') from the params text."""
    from .zhang2021_panel_quality import (
        DEFAULT_K3D_THRESHOLDS, DEFAULT_M2D_THRESHOLDS, DEFAULT_M3D_THRESHOLDS, PanelQualityThresholds)
    defaults = {"m2d": DEFAULT_M2D_THRESHOLDS, "m3d": DEFAULT_M3D_THRESHOLDS, "k3d": DEFAULT_K3D_THRESHOLDS}[stage]
    text = getattr(params, f"zhang2021_{stage}_gate", "") if params is not None else ""
    try:
        return PanelQualityThresholds.from_text(text, **defaults)
    except ValueError as exc:
        raise Zhang2021QuadModeError(f"ZHANG2021_{stage.upper()}_GATE_INVALID: {exc}") from exc


def _seam_preflight(rp: Any, quad_uv: np.ndarray, faces: np.ndarray) -> dict:
    """The downstream source-seam classifier on this candidate (before any K3D)."""
    from .optcuts_seam_exact_closure import SeamTopologyMismatch, discover_exact_seam_topology

    param_like = SimpleNamespace(surface_vertices_3d=rp.xyz, surface_faces=rp.f3, uv_vertices_2d=rp.uv,
                                 uv_faces=rp.f2, metrics={"zhang2021_quad_20261027_enabled": True})
    try:
        topo = discover_exact_seam_topology(param_like, quad_uv, faces)
    except SeamTopologyMismatch as exc:
        return {"zhang2021_seam_preflight_status": "error", "zhang2021_seam_preflight_mode": "unavailable",
                "zhang2021_seam_preflight_reason": str(exc)[:400]}
    status = str(topo.get("status"))
    mode = str(topo.get("seam_stitch_mode", "none"))
    if status in ("no_boundary", "source_open_boundary_only"):
        mode = "no_cut"
    audit = topo.get("nested_partition_audit") or {}
    return {
        "zhang2021_seam_preflight_status": status,
        "zhang2021_seam_preflight_mode": mode,
        "zhang2021_seam_preflight_reason": ("" if mode in ("hard", "no_cut") else
                                            f"stitch mode {mode}: {audit.get('reason', '')}"),
        "zhang2021_seam_preflight_vertex_groups": int(len(topo.get("groups", []))),
        "zhang2021_seam_preflight_paired_edges": int(len(topo.get("paired_edges", []))),
        "zhang2021_seam_preflight_affine_rows": int(len(topo.get("affine_rows", []))),
        "zhang2021_seam_preflight_t_vertices": int(audit.get("t_vertices", 0) or 0),
        "zhang2021_seam_preflight_open_boundary_edges": int(topo.get("source_open_grid_edges", 0)),
        "zhang2021_seam_preflight_rank": dict(topo.get("affine_rank_diagnostics") or {}),
    }


def plan_zhang2021_m2d(xyz, f3, uv, f2, params: Any, *, zhang_result: Any = None) -> Zhang2021M2DPlan:
    """Paper stage + OneString adapter on one chart (raises on failure).

    1. [paper]     Zhang et al. 2021 reference programs -> Q (``zhang_result`` may
                   pass a precomputed run of exactly this chart and parameters).
    2. [OneString] seam-node snapping, cut-tip / junction valence repair, matched /
                   nested 3D sampling of every cut branch (exact closure), conforming
                   splits + bijective SLIM chart re-parameterisation.
    3. [OneString] 3D panel-quality optimisation of the interior M2D vertices
                   inside the final chart (boundary = exact seam samples, fixed).
    4. M2D (UV validity) and M3D (size/shape uniformity) gates, seam preflight.
    """
    from .optcuts_planar_quad_refine_20261026 import validate_disk_mesh
    from .zhang2021_panel_optimizer import (
        ChartLift, PanelOptimizationError, PanelOptimizerWeights, optimize_panel_quality, quad_validity)
    from .zhang2021_panel_quality import panel_quality

    started = time.perf_counter()
    xyz = np.asarray(xyz, float)
    f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float)
    f2 = np.asarray(f2, int)
    zp = zhang2021_parameters_from(params)
    bin_dir = str(getattr(params, "zhang2021_bin_dir", "") or "").strip() or None
    res = zhang_result
    if res is None:
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
    hard = bool(getattr(params, "zhang2021_hard_seams", True))
    try:
        rp = seam_conforming_reparameterization(
            xyz, f3, uv, f2, quv, quads,
            snap_nodes=bool(getattr(params, "zhang2021_snap_seam_nodes", True)) or hard,
            align_pairs=bool(getattr(params, "zhang2021_align_seam_pairs", True)),
            seam_matching=bool(getattr(params, "zhang2021_seam_matching", False)),
            hard_seams=hard,
            repair_node_valence=bool(getattr(params, "zhang2021_repair_node_valence", True)))
    except (SeamConformingReparamError, BoundaryCorrespondenceError) as exc:
        raise Zhang2021QuadModeError(str(exc)) from exc
    paper_quads = int(len(quads))
    quv = np.asarray(rp.quad_uv, float)
    quads = np.asarray(rp.quads, int)
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
    # OneString's M2D convention is counter-clockwise quads in UV (as QuadGrid).
    orient = float(np.sum(signed_uv_areas(quv, quads[:, [0, 1, 2]])) + np.sum(signed_uv_areas(quv, quads[:, [0, 2, 3]])))
    faces_ccw = quads[:, ::-1].copy() if orient < 0 else quads.copy()
    try:
        validate_disk_mesh(quv, faces_ccw)
        topology_ok, topology_detail = True, "all-quad disk"
    except ValueError as exc:
        topology_ok, topology_detail = False, str(exc)[:300]
    # [OneString] 3D panel-quality optimisation of the interior vertices.
    lift = ChartLift(rp.uv, rp.f2, rp.xyz, rp.f3)
    fixed = np.zeros(len(quv), bool)
    fixed[quad_boundary_loop(faces_ccw)] = True
    quv_in = quv.copy()
    opt_report: dict = {"panel_optimizer_enabled": False}
    if bool(getattr(params, "zhang2021_panel_optimization", True)):
        try:
            quv, m3d, opt_report = optimize_panel_quality(
                lift, quv, faces_ccw, fixed, weights=PanelOptimizerWeights(),
                max_iter=int(getattr(params, "zhang2021_panel_optimization_iterations", 300)))
        except PanelOptimizationError as exc:
            raise Zhang2021QuadModeError(str(exc)) from exc
        opt_report["panel_optimizer_enabled"] = True
    else:
        tri = lift.locate(quv)
        if np.any(tri < 0):
            raise Zhang2021QuadModeError(
                f"ZHANG2021_ONESTRING_QUAD_VERTEX_OUTSIDE_CHART: {int(np.sum(tri < 0))} vertices")
        m3d = lift.lift(quv, tri)
    validity = quad_validity(quv, faces_ccw)[0]
    q2 = panel_quality(quv, faces_ccw, stage="M2D", thresholds=stage_thresholds(params, "m2d"),
                       planar_2d_orientation=1.0)
    q3 = panel_quality(m3d, faces_ccw, stage="M3D", thresholds=stage_thresholds(params, "m3d"))
    legacy2 = _quad_quality(np.column_stack((quv, np.zeros(len(quv)))), faces_ccw)
    legacy3 = _quad_quality(m3d, faces_ccw)
    sj = scaled_jacobians(quv, faces_ccw)
    sj3 = _scaled_jacobian_3d(m3d, faces_ccw)
    src_loop = oriented_boundary_loops(f2)
    if len(src_loop) != 1:
        raise Zhang2021QuadModeError("ZHANG2021_SOURCE_CHART_NOT_DISK")
    src_loop_uv = uv[src_loop[0]]
    dh, _dh_q_to_src, _dh_src_to_q = two_sided_boundary_hausdorff(
        src_loop_uv, [list(range(len(src_loop_uv)))], q_poly, [list(range(len(q_poly)))])
    bbox_diag = float(np.linalg.norm(np.ptp(uv, axis=0)))
    seam = _seam_preflight(rp, quv, faces_ccw)
    odd = odd_valence_interior_vertices(faces_ccw)
    moved = float(np.max(np.linalg.norm(quv - quv_in, axis=1))) if len(quv) else 0.0
    metrics: dict[str, Any] = {
        "requested_omega_parameterization_mode": MODE,
        "zhang2021_quad_20261027_enabled": True,
        "zhang2021_mode_label": MODE_LABEL,
        "zhang2021_paper_doi": PAPER_DOI,
        **{k: v for k, v in res.metrics.items()},
        "zhang2021_density_input_n_tar": int(zp.target_quads),
        "zhang2021_density_input_note": (
            "N_tar is only the paper's density input (l_tar = sqrt(area / N_tar)); it is not a panel-count "
            "goal and no selection term depends on it"),
        "zhang2021_paper_output_quads": paper_quads,
        "zhang2021_final_m2d_quads": int(len(faces_ccw)),
        "zhang2021_m2d_is_unmodified_paper_output": bool(len(faces_ccw) == paper_quads and moved == 0.0
                                                         and not rp.metrics.get("zhang2021_onestring_node_valence_repairs")),
        "zhang2021_onestring_m2d_topology_modified": bool(len(faces_ccw) != paper_quads
                                                          or rp.metrics.get("zhang2021_onestring_node_valence_repairs")),
        "zhang2021_onestring_m2d_interior_max_move": moved,
        "zhang2021_m2d_scaled_jacobian_min": float(np.min(sj)),
        "zhang2021_m2d_scaled_jacobian_mean": float(np.mean(sj)),
        "zhang2021_m3d_scaled_jacobian_min": float(np.min(sj3)),
        "zhang2021_m3d_scaled_jacobian_mean": float(np.mean(sj3)),
        "zhang2021_boundary_hausdorff_vs_source_omega": float(dh),
        "zhang2021_boundary_hausdorff_vs_source_omega_percent_bbox": 100.0 * float(dh) / max(bbox_diag, 1e-300),
        **_polygon_metrics(src_loop_uv, q_poly),
        **rp.metrics,
        **{k: v for k, v in opt_report.items()},
        "zhang2021_onestring_chart_injective": True,
        "zhang2021_m2d_topology_valid": bool(topology_ok),
        "zhang2021_m2d_topology_detail": topology_detail,
        "zhang2021_m2d_invalid_quads": int(np.sum(validity <= 0)),
        "zhang2021_m2d_quads_valid": bool(np.all(validity > 0)),
        "zhang2021_onestring_final_chart_equals_quad_region": True,
        "zhang2021_onestring_final_chart_area_relative_error": float(rel),
        "zhang2021_onestring_surface_coverage": (
            "complete: the final chart is a bijection onto all of S and its boundary is the quad "
            "boundary polygon, so the lifted M2D covers S without gaps or outside area"),
        "zhang2021_onestring_m2d_faces_reversed_to_ccw": bool(orient < 0),
        "zhang2021_onestring_lift_model": "strict containing triangle of the final chart + barycentric",
        **{"zhang2021_m2d_" + k: (int(v) if k.endswith("_face_index") else float(v)) for k, v in legacy2.items()},
        **{"zhang2021_m3d_" + k: (int(v) if k.endswith("_face_index") else float(v)) for k, v in legacy3.items()},
        **q2.prefixed("zhang2021_m2d"),
        **q3.prefixed("zhang2021_m3d"),
        **seam,
        "zhang2021_odd_valence_interior_vertices": int(len(odd)),
        "zhang2021_total_seconds": float(time.perf_counter() - started),
        "zhang2021_uv_span": span,
    }
    source = SimpleNamespace(xyz=xyz.copy(), f3=f3.copy(), uv=uv.copy(), f2=f2.copy())
    return Zhang2021M2DPlan(quad_uv=quv, quads=faces_ccw, m3d_preview=m3d, chart=rp,
                            source=source, metrics=metrics, m2d_report=q2, m3d_report=q3)


# ------------------------------------------- coordinated Omega + quad search
# OneString extension (2026-10-26 design order, revised 2026-10-27):
#   OptCuts -> [h: boundary simplification preserving seam-node directions +
#               bijective SLIM of Omega] -> Zhang et al. 2021 quad meshing of
#   Omega_1 (density N_tar and Hausdorff eps are candidate parameters) ->
#   hard seam adapter -> 3D panel optimisation -> phased selection
#   (zhang2021_candidate_selection).  There is NO panel-count term anywhere.
def _float_list(text: Any, default: list[float]) -> list[float]:
    if text is None or (isinstance(text, str) and not text.strip()):
        return list(default)
    if isinstance(text, (list, tuple)):
        return [float(x) for x in text]
    return [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]


def candidate_densities(params: Any, surface_area: float) -> list[int]:
    """Density inputs N_tar of the paper (never a goal).

    ``zhang2021_target_edge_length`` > 0 converts a desired 3D panel edge length
    into the paper's density input N_tar = area(S) / l^2; otherwise
    ``zhang2021_target_quads`` is used.  ``zhang2021_candidate_density_factors``
    multiplies it to explore several densities; quality decides.
    """
    l_tar = float(getattr(params, "zhang2021_target_edge_length", 0.0) or 0.0)
    if l_tar > 0:
        base = max(4, int(round(float(surface_area) / (l_tar * l_tar))))
    else:
        base = int(getattr(params, "zhang2021_target_quads", DEFAULT_DENSITY_N_TAR))
    factors = _float_list(getattr(params, "zhang2021_candidate_density_factors", None), [1.0])
    out: list[int] = []
    for f in factors:
        if not np.isfinite(f) or f <= 0:
            raise Zhang2021QuadModeError(f"ZHANG2021_CANDIDATE_DENSITY_FACTOR_INVALID: {f}")
        n = max(4, int(round(base * f)))
        if n not in out:
            out.append(n)
    return out


def _params_with(params: Any, **changes) -> Any:
    from dataclasses import is_dataclass, replace as _dc_replace
    if is_dataclass(params):
        try:
            return _dc_replace(params, **changes)
        except TypeError:
            pass
    import copy as _copy
    out = _copy.copy(params)
    for k, v in changes.items():
        setattr(out, k, v)
    return out


def _planner_cache_key(xyz, f3, uv, f2, params) -> str:
    import hashlib
    h = hashlib.sha1()
    for a in (xyz, f3, uv, f2):
        h.update(np.ascontiguousarray(a).tobytes())
    for name in sorted(n for n in dir(params) if n.startswith("zhang2021_") and n not in (
            "zhang2021_candidate_rank", "zhang2021_verification_attempts", "zhang2021_compare_verified_candidates")):
        try:
            h.update(f"{name}={getattr(params, name)!r};".encode())
        except Exception:  # noqa: BLE001 - only hashing
            pass
    return h.hexdigest()


def plan_zhang2021_m2d_coordinated(xyz, f3, uv, f2, params: Any, *, cache: dict | None = None) -> Zhang2021M2DPlan:
    """Phases 1 + 2: evaluate every (tol, eps, N_tar) candidate, rank the feasible ones.

    Returns the candidate of rank ``zhang2021_candidate_rank`` (0 = best quality
    score); Phase 3 (downstream verification) asks for the next rank when a
    candidate fails K3D / T3D / K2D / T2D.  Raises with the full candidate table
    when no candidate satisfies the hard constraints.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .zhang2021_candidate_selection import candidate_score, feasibility, parse_weights, summarize_row
    from .zhang2021_onestring_precondition import precondition_omega
    from .zhang2021_onestring_reparam import _SDProblem

    started = time.perf_counter()
    xyz = np.asarray(xyz, float)
    f3 = np.asarray(f3, int)
    uv = np.asarray(uv, float)
    f2 = np.asarray(f2, int)
    route = downstream_route(params)
    key = _planner_cache_key(xyz, f3, uv, f2, params)
    entry = cache.get(key) if cache is not None else None
    if entry is None:
        weights = parse_weights(getattr(params, "zhang2021_selection_weights", ""))
        tols = (_float_list(getattr(params, "zhang2021_precondition_tolerances", None), [0.0, 0.02])
                if bool(getattr(params, "zhang2021_precondition_enabled", True)) else [0.0])
        epss = _float_list(getattr(params, "zhang2021_candidate_hausdorff_ratios", None),
                           [float(getattr(params, "zhang2021_hausdorff_ratio", DEFAULT_HAUSDORFF_RATIO))])
        area3d = float(0.5 * np.sum(np.linalg.norm(np.cross(xyz[f3[:, 1]] - xyz[f3[:, 0]],
                                                             xyz[f3[:, 2]] - xyz[f3[:, 0]]), axis=1)))
        densities = candidate_densities(params, area3d)
        keep = float(getattr(params, "zhang2021_precondition_node_keep_ratio", 0.06))
        mean_lim = float(getattr(params, "zhang2021_precondition_sd_mean_ratio_max", 1.08))
        p95_lim = float(getattr(params, "zhang2021_precondition_sd_p95_ratio_max", 1.30))
        sa = signed_uv_areas(uv, f2)
        mirror = np.array([-1.0, 1.0]) if np.all(sa < 0) else np.array([1.0, 1.0])
        sd0_mean = float(_SDProblem(xyz, f3, f2, len(uv)).energy(uv * mirror)[0])
        diag = float(np.linalg.norm(np.ptp(uv, axis=0)))
        src_loop = oriented_boundary_loops(f2)[0]
        rows: list[dict] = []
        jobs = []
        for tol in tols:
            pre = precondition_omega(xyz, f3, uv, f2, tol, sd_mean_ratio_max=mean_lim, sd_p95_ratio_max=p95_lim,
                                     node_keep_ratio=keep)
            if not pre.accepted:
                rows.append({"precondition_tol": tol, "zhang_eps": None, "n_tar": None,
                             "status": "precondition_rejected", "reason": pre.reason,
                             "rejections": ["h: " + pre.reason]})
                continue
            for eps in epss:
                for n_tar in densities:
                    p = _params_with(params, zhang2021_hausdorff_ratio=float(eps), zhang2021_target_quads=int(n_tar))
                    jobs.append((tol, eps, n_tar, pre, p))
        bin_dir = str(getattr(params, "zhang2021_bin_dir", "") or "").strip() or None

        def run_job(job):
            tol, eps, n_tar, pre, p = job
            try:
                return run_reference_pipeline(pre.xyz, pre.f3, pre.uv, pre.f2, zhang2021_parameters_from(p),
                                              bin_dir=bin_dir)
            except Exception as exc:  # noqa: BLE001 - recorded per candidate
                return exc

        workers = max(1, int(getattr(params, "zhang2021_parallel_workers", 0) or 0)) or None
        if workers is None:
            import os
            workers = max(1, min(4, (os.cpu_count() or 2)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_job, jobs))
        feasible: list[tuple[float, int, Zhang2021M2DPlan]] = []
        for (tol, eps, n_tar, pre, p), zres in zip(jobs, results):
            row: dict[str, Any] = {"precondition_tol": tol, "zhang_eps": eps, "n_tar": n_tar}
            row.update({k: v for k, v in pre.metrics.items() if isinstance(v, (int, float)) and "sd_" in k})
            if isinstance(zres, Exception):
                row.update(status="failed", reason=str(zres)[:300], rejections=["paper stage: " + str(zres)[:200]])
                rows.append(row)
                continue
            try:
                plan = plan_zhang2021_m2d(pre.xyz, pre.f3, pre.uv, pre.f2, p, zhang_result=zres)
            except Zhang2021QuadModeError as exc:
                row.update(status="failed", reason=str(exc)[:300], paper_quads=int(len(zres.quads)),
                           rejections=["adapter: " + str(exc)[:240]])
                rows.append(row)
                continue
            m = plan.metrics
            checks = feasibility(m, route_needs_even_valence=(route == ROUTE_PAPER_LOCAL_GLOBAL))
            q_loop = quad_boundary_loop(plan.quads)
            dh, _, _ = two_sided_boundary_hausdorff(uv[src_loop], [list(range(len(src_loop)))],
                                                    plan.quad_uv[q_loop], [list(range(len(q_loop)))])
            distortion = float(m.get("zhang2021_onestring_sd_mean_after", sd0_mean)) / max(sd0_mean, 1e-300) - 1.0
            fit = float(dh) / max(diag, 1e-300)
            score, terms = candidate_score(plan.m3d_report.metrics, plan.m2d_report.metrics, distortion, fit, weights)
            ok = all(c["ok"] for c in checks)
            row.update(
                status="feasible" if ok else "infeasible", quads=int(len(plan.quads)),
                paper_quads=int(m.get("zhang2021_paper_output_quads", 0)), score=score,
                **{"score_" + k: float(v) for k, v in terms.items()},
                rejections=[f"{c['constraint']}: {c['detail']}" for c in checks if not c["ok"]],
                m3d_angle_min=float(plan.m3d_report.metrics["angle_min_deg"]),
                m3d_angle_max=float(plan.m3d_report.metrics["angle_max_deg"]),
                m3d_edge_ratio_max=float(plan.m3d_report.metrics["edge_ratio_max"]),
                m3d_e_size=float(plan.m3d_report.metrics["e_size"]),
                m3d_e_angle=float(plan.m3d_report.metrics["e_angle"]),
                m3d_size_ratio=float(plan.m3d_report.metrics["size_ratio_max_min"]),
                seam_mode=str(m.get("zhang2021_seam_preflight_mode")),
                seam_t_vertices=int(m.get("zhang2021_onestring_seam_t_vertices", 0)),
                node_repairs=int(m.get("zhang2021_onestring_node_valence_repairs", 0)),
                odd_valence_interior=int(m.get("zhang2021_odd_valence_interior_vertices", 0)),
                final_sd_mean_ratio_vs_optcuts=distortion + 1.0, boundary_fit=fit)
            rows.append(row)
            plan.metrics["zhang2021_candidate_row"] = dict(row)
            if ok:
                feasible.append((score, len(rows) - 1, plan))
            else:
                # keep the best infeasible plan for the diagnostics figure
                row["_plan"] = plan
            print("[ZHANG2021-CAND] " + summarize_row(row), flush=True)
        feasible.sort(key=lambda t: (t[0], t[1]))
        for rank, (_, idx, _plan) in enumerate(feasible):
            rows[idx]["rank"] = rank
        entry = {"rows": rows, "feasible": feasible, "weights": weights, "sd0_mean": sd0_mean,
                 "densities": densities, "tols": tols, "epss": epss,
                 "seconds": float(time.perf_counter() - started)}
        if cache is not None:
            if cache is _MODULE_PLAN_CACHE and len(cache) >= 4:
                cache.clear()
            cache[key] = entry
    rows, feasible = entry["rows"], entry["feasible"]
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    rank = int(getattr(params, "zhang2021_candidate_rank", 0) or 0)
    if not feasible:
        from .zhang2021_diagnostics import write_candidate_diagnostics
        best_bad = min((r for r in rows if r.get("_plan") is not None), key=lambda r: r.get("score", np.inf),
                       default=None)
        where = write_candidate_diagnostics(public_rows, best_bad.get("_plan") if best_bad else None, params)
        raise Zhang2021QuadModeError(
            "ZHANG2021_NO_FEASIBLE_CANDIDATE: no (tol, eps, N_tar) candidate satisfies the hard constraints "
            "(UV bijective, quad topology, coverage, hard OptCuts seam closure, M2D/M3D quality gates, route "
            f"linkage). Diagnostics: {where}\n" + "\n".join(summarize_row(r) for r in public_rows))
    if rank >= len(feasible):
        raise Zhang2021CandidatesExhausted(
            f"ZHANG2021_CANDIDATES_EXHAUSTED: rank {rank} requested, {len(feasible)} feasible candidate(s)")
    import copy as _copy
    score, idx, plan = feasible[rank]
    plan = _copy.deepcopy(plan)
    chosen = rows[idx]
    plan.metrics.update({
        "zhang2021_coordinated_search": True,
        "zhang2021_coordinated_model": (
            "OneString (not Zhang 2021): Omega_1 = h(Omega_0) (Douglas-Peucker per seam run with seam-node "
            "directions kept, Hausdorff <= tol, bijective SLIM) -> Zhang 2021 with density input N_tar and "
            "Hausdorff eps -> hard-seam adapter -> 3D panel optimisation; Phase 1 hard constraints, Phase 2 "
            "quality score without any panel-count term"),
        "zhang2021_coordinated_candidates": public_rows,
        "zhang2021_coordinated_feasible_candidates": int(len(feasible)),
        "zhang2021_coordinated_selection_weights": dict(entry["weights"]),
        "zhang2021_coordinated_chosen_rank": int(rank),
        "zhang2021_coordinated_chosen_score": float(score),
        "zhang2021_coordinated_chosen_precondition_tol": float(chosen["precondition_tol"]),
        "zhang2021_coordinated_chosen_hausdorff_ratio": float(chosen["zhang_eps"]),
        "zhang2021_coordinated_chosen_density_n_tar": int(chosen["n_tar"]),
        "zhang2021_coordinated_seconds": float(entry["seconds"]),
        "zhang2021_optcuts_sd_mean": float(entry["sd0_mean"]),
    })
    print(f"[ZHANG2021-COORD] rank {rank}/{len(feasible)} chosen: tol={chosen['precondition_tol']} "
          f"eps={chosen['zhang_eps']} N_tar(density)={chosen['n_tar']} -> {len(plan.quads)} quads, "
          f"E={score:.4g}; {len(rows)} candidates evaluated", flush=True)
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
        if not mode_enabled(params):
            return base_parameterization(surface, target, grid, params)
        import copy as _copy
        run_cache = getattr(params, "_zhang2021_run_cache", None)
        cached = run_cache.get("optcuts_parameterization") if isinstance(run_cache, dict) else None
        if cached is not None:
            # Phase-3 retry with the next candidate: the OptCuts chart of this
            # surface is unchanged, reuse it instead of re-running OptCuts.
            result = _copy.deepcopy(cached)
            print("[ZHANG2021-20261027] reusing the OptCuts chart of this run (candidate retry)", flush=True)
        else:
            result = base_parameterization(surface, target, grid, params)
            if isinstance(run_cache, dict):
                try:
                    run_cache["optcuts_parameterization"] = _copy.deepcopy(result)
                except Exception:  # noqa: BLE001 - caching is only an optimisation
                    pass
        if str(getattr(result, "method", "")) not in ("optcuts_test", "optcuts"):
            raise Zhang2021QuadModeError(
                "ZHANG2021_MODE_REQUIRES_OPTCUTS_CHART: parameterization method="
                f"{getattr(result, 'method', None)!r}")
        route = downstream_route(params)
        if not isinstance(run_cache, dict):
            run_cache = _MODULE_PLAN_CACHE
        plan = plan_zhang2021_m2d_coordinated(result.surface_vertices_3d, result.surface_faces,
                                              result.uv_vertices_2d, result.uv_faces, params, cache=run_cache)
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
        print(f"[ZHANG2021-20261027] paper output {plan.metrics['zhang2021_paper_output_quads']} quads "
              f"(density input N_tar={plan.metrics['zhang2021_density_input_n_tar']}, not a goal) -> M2D "
              f"{len(plan.quads)} quads, patches={plan.metrics['zhang2021_n_patches']} "
              f"J_min={plan.metrics['zhang2021_j_min']:.3f} dH={plan.metrics['zhang2021_d_h_percent_bbox']:.3f}% "
              f"| chart SD {plan.metrics['zhang2021_onestring_sd_mean_before']:.4f}->"
              f"{plan.metrics['zhang2021_onestring_sd_mean_after']:.4f} | seam preflight "
              f"{plan.metrics.get('zhang2021_seam_preflight_mode')} (matched/nested branches "
              f"{plan.metrics.get('zhang2021_onestring_seam_matched_branches')}/"
              f"{plan.metrics.get('zhang2021_onestring_seam_nested_branches')}, T-vertices "
              f"{plan.metrics.get('zhang2021_onestring_seam_t_vertices')}) | M3D gate "
              f"{'passed' if plan.metrics.get('zhang2021_m3d_quality_passed') else 'FAILED'}", flush=True)
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
        if str(seam["status"]) == "conforming" and str(seam.get("seam_stitch_mode")) != "hard":
            raise Zhang2021QuadModeError(
                "ZHANG2021_SEAM_NOT_HARD_CLOSABLE: the M2D samples the OptCuts cut as a "
                f"{seam.get('seam_stitch_mode')!r} seam; only exact (hard) closure is accepted")
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
    from .zhang2021_verification import install_zhang2021_verification_loop
    install_zhang2021_verification_loop(pipeline)
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
        if params is None or not mode_enabled(params):
            return inner_k3d(target, mesh, parameterization, params)
        route = downstream_route(params)
        if route == ROUTE_PAPER_LOCAL_GLOBAL:
            out, report = _paper_local_global_k3d(pipeline, target, mesh, parameterization, params)
        else:
            if str((getattr(mesh, "metrics", {}) or {}).get("seam_stitch_mode", "none")) == "soft":
                raise Zhang2021QuadModeError(
                    "ZHANG2021_SEAM_NOT_HARD_CLOSABLE: the M3D seam preflight is 'soft'; the optcuts_test2 K3D "
                    "would only approximate the OptCuts seam, which the Zhang2021 mode never accepts")
            out, report = inner_k3d(target, mesh, parameterization, params)
        from .zhang2021_verification import k3d_gate
        out.metrics["zhang2021_seam_preflight_mode"] = str(
            (getattr(parameterization, "metrics", {}) or {}).get("zhang2021_seam_preflight_mode", ""))
        k3d_gate(out, params, route=route)
        return out, report

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
    "MODE", "MODE_LABEL", "FLAG", "DEFAULT_DENSITY_N_TAR", "DEFAULT_HAUSDORFF_RATIO",
    "Zhang2021QuadModeError", "Zhang2021M2DPlan", "mode_enabled", "zhang2021_parameters_from",
    "plan_zhang2021_m2d", "plan_zhang2021_m2d_coordinated", "install_zhang2021_quad_20261027",
]
