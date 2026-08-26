"""OneString-style rigid-tile K2D for ``optcuts_test``.

The old OptCuts_test patch treated K2D as a shared-vertex metric embedding and
then tried to pull scattered vertices back toward M2D.  That is not the flat
linkage we want.  This patch keeps the shared QuadMesh only as a topology carrier
for legacy downstream code, but makes the *authoritative* K2D layout an array of
independent rigid tiles.  Each K3D quad is flattened to its intrinsic best-fit
plane, rigidly aligned to the corresponding M2D tile, and then optimized only by
one SE(2) pose per tile with OneString-style hinge/collision/anchor terms.

For ``optcuts_test`` collision avoidance is now an acceptance constraint, not
only a soft energy.  After the ordinary OneString-style SE(2) solve, an SAT-based
rigid feasibility projection translates penetrating tiles apart by their minimum
translation vectors (MTVs) until no positive-area overlap remains.  Point/edge
contact at a hinge is allowed; actual penetration is not.  If any penetration
remains after the hard projection, K2D is rejected rather than silently passed
downstream.

The resulting independent layout is injected into ``_make_flat_tile_layout`` so
K2D visualization and T2D construction use the rigid linkage, not the legacy
shared-vertex embedding.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _rigid_align_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best orientation-preserving 2D rigid alignment of source to target."""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    cs = np.mean(src, axis=0)
    cd = np.mean(dst, axis=0)
    a = src - cs
    b = dst - cd
    try:
        u, _s, vt = np.linalg.svd(a.T @ b)
        r = u @ vt
        if np.linalg.det(r) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vt
        return a @ r + cd
    except Exception:
        return a + cd


def _flatten_k3d_tile(points_3d: np.ndarray, m2d_reference: np.ndarray) -> np.ndarray:
    """Flatten one K3D quad without changing its in-plane rigid geometry."""
    p = np.asarray(points_3d, dtype=float)
    c = np.mean(p, axis=0)
    q = p - c
    try:
        _u, _s, vh = np.linalg.svd(q, full_matrices=False)
        basis = np.asarray(vh[:2], dtype=float).T
        flat = q @ basis
    except Exception:
        flat = q[:, :2].copy()
    ref = np.asarray(m2d_reference, dtype=float)
    if _signed_area(flat) * _signed_area(ref) < 0.0:
        flat[:, 1] *= -1.0
    return _rigid_align_2d(flat, ref)


def _tile_shape_error_to_k3d(flat_tiles: np.ndarray, k3d_tiles: np.ndarray) -> tuple[float, float]:
    if len(flat_tiles) == 0:
        return 0.0, 0.0
    f3 = np.dstack([np.asarray(flat_tiles, float), np.zeros(np.asarray(flat_tiles).shape[:2])])
    k3 = np.asarray(k3d_tiles, float)
    df = np.linalg.norm(f3[:, :, None, :] - f3[:, None, :, :], axis=-1)
    dk = np.linalg.norm(k3[:, :, None, :] - k3[:, None, :, :], axis=-1)
    err = np.abs(df - dk)
    return float(np.sqrt(np.mean(err * err))), float(np.max(err))


def _hinge_constraints(pipeline: Any, faces: np.ndarray) -> list[tuple[int, int, int, int]]:
    owner = getattr(pipeline, "_original", None)
    specs_fn = getattr(owner, "_vertex_hinge_specs_from_faces", None) or getattr(pipeline, "_vertex_hinge_specs_from_faces", None)
    if specs_fn is None:
        return []
    specs = specs_fn(np.asarray(faces, dtype=int))
    return [
        (int(s.tile_a), int(s.corner_a0), int(s.tile_b), int(s.corner_b0))
        for s in specs
    ]


def _hinge_rms(layout: np.ndarray, constraints: list[tuple[int, int, int, int]]) -> float:
    vals: list[float] = []
    for ia, ca, ib, cb in constraints:
        if ia < len(layout) and ib < len(layout) and ca < layout.shape[1] and cb < layout.shape[1]:
            vals.append(float(np.linalg.norm(layout[ia, ca] - layout[ib, cb])))
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0


def _relative_center_rms(layout: np.ndarray, reference: np.ndarray, constraints: list[tuple[int, int, int, int]]) -> float:
    if not constraints:
        return 0.0
    pairs = sorted({(min(a, b), max(a, b)) for a, _ca, b, _cb in constraints if a != b})
    if not pairs:
        return 0.0
    c = np.mean(np.asarray(layout, float), axis=1)
    r = np.mean(np.asarray(reference, float), axis=1)
    vals = np.asarray([(c[j] - c[i]) - (r[j] - r[i]) for i, j in pairs], dtype=float)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def _collision_backend(pipeline: Any):
    owner = getattr(pipeline, "_original", None)
    pair_fn = getattr(owner, "_spatial_candidate_pairs_for_tiles", None) or getattr(pipeline, "_spatial_candidate_pairs_for_tiles", None)
    sat_fn = getattr(owner, "_sat_polygon_mtv", None) or getattr(pipeline, "_sat_polygon_mtv", None)
    return pair_fn, sat_fn


def _tile_scale(tiles: np.ndarray) -> float:
    t = np.asarray(tiles, dtype=float)
    if not t.size:
        return 1.0
    lengths = np.linalg.norm(np.roll(t, -1, axis=1) - t, axis=2)
    positive = lengths[lengths > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def _penetrating_pairs(pipeline: Any, tiles: np.ndarray, penetration_tolerance: float) -> list[tuple[int, int, np.ndarray, float]]:
    """Return only positive-area penetrations; hinge point/edge contact is allowed."""
    t = np.asarray(tiles, dtype=float)
    if len(t) < 2:
        return []
    pair_fn, sat_fn = _collision_backend(pipeline)
    if pair_fn is None or sat_fn is None:
        raise RuntimeError("OPTCUTS_TEST_K2D_HARD_COLLISION_BACKEND_UNAVAILABLE")
    scale = _tile_scale(t)
    pairs = pair_fn(t, pad=max(scale * 0.08, 1e-5))
    bad: list[tuple[int, int, np.ndarray, float]] = []
    tol = max(float(penetration_tolerance), scale * 1e-12, 1e-12)
    for i, j in pairs:
        overlap, mtv, signed = sat_fn(t[i], t[j], clearance=0.0)
        mtv = np.asarray(mtv, dtype=float)
        depth = float(np.linalg.norm(mtv)) if overlap else 0.0
        # _sat_polygon_mtv reports touching polygons as overlap with zero MTV.
        # That contact is intentional at vertex hinges and is not penetration.
        if overlap and depth > tol and float(signed) < -tol:
            bad.append((int(i), int(j), mtv, depth))
    return bad


def _hard_nonoverlap_project(
    pipeline: Any,
    tiles: np.ndarray,
    *,
    max_sweeps: int,
    penetration_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigidly translate penetrating tiles until SAT penetration is zero.

    Each correction is a pure translation, so per-tile K3D intrinsic geometry is
    preserved exactly.  We use the SAT minimum translation vector and split it
    equally between the two tiles.  The global centroid is therefore preserved.
    """
    x = np.asarray(tiles, dtype=float).copy()
    before = _penetrating_pairs(pipeline, x, penetration_tolerance)
    initial_count = len(before)
    max_depth_before = max((item[3] for item in before), default=0.0)
    sweeps_used = 0

    for sweep in range(max(1, int(max_sweeps))):
        bad = _penetrating_pairs(pipeline, x, penetration_tolerance)
        if not bad:
            break
        shifts = np.zeros((len(x), 2), dtype=float)
        counts = np.zeros(len(x), dtype=float)
        for i, j, mtv, _depth in bad:
            # mtv points from j toward i; half on each tile separates the pair.
            shifts[i] += 0.5 * mtv
            shifts[j] -= 0.5 * mtv
            counts[i] += 1.0
            counts[j] += 1.0
        active = counts > 0.0
        if not np.any(active):
            break
        shifts[active] /= counts[active, None]
        x[active] += shifts[active, None, :]
        sweeps_used = sweep + 1

    after = _penetrating_pairs(pipeline, x, penetration_tolerance)
    max_depth_after = max((item[3] for item in after), default=0.0)
    return x, {
        "onestring_k2d_hard_nonoverlap_applied": True,
        "onestring_k2d_hard_nonoverlap_model": "SAT MTV rigid-translation feasibility projection",
        "onestring_k2d_hard_nonoverlap_touching_allowed": True,
        "onestring_k2d_hard_nonoverlap_initial_penetration_count": int(initial_count),
        "onestring_k2d_hard_nonoverlap_final_penetration_count": int(len(after)),
        "onestring_k2d_hard_nonoverlap_max_depth_before": float(max_depth_before),
        "onestring_k2d_hard_nonoverlap_max_depth_after": float(max_depth_after),
        "onestring_k2d_hard_nonoverlap_sweeps": int(sweeps_used),
        "onestring_k2d_hard_nonoverlap_satisfied": bool(len(after) == 0),
    }


def _build_rigid_k2d_layout(pipeline: Any, mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
    faces = np.asarray(mesh_2d.faces, dtype=int)
    if len(faces) == 0:
        return np.zeros((0, 4, 2), dtype=float), {}

    m2d_vertices = np.asarray(mesh_2d.vertices, dtype=float)
    k3d_vertices = np.asarray(mesh_3d.vertices, dtype=float)
    m2d_tiles = m2d_vertices[faces, :2]
    k3d_tiles = k3d_vertices[faces, :3]

    rest = np.asarray(
        [_flatten_k3d_tile(k3d_tiles[i], m2d_tiles[i]) for i in range(len(faces))],
        dtype=float,
    )
    constraints = _hinge_constraints(pipeline, faces)
    shape_rms_before, shape_max_before = _tile_shape_error_to_k3d(rest, k3d_tiles)

    owner = getattr(pipeline, "_original", None)
    solver = getattr(owner, "_paper_local_global_se2_layout", None) or getattr(pipeline, "_paper_local_global_se2_layout", None)
    if solver is None:
        return rest, {
            "onestring_k2d_rigid_layout_applied": False,
            "onestring_k2d_rigid_layout_reason": "paper SE2 solver unavailable",
        }

    tile_size = max(float(getattr(mesh_2d.grid, "tile_size", 1.0)), 1e-8)
    gap_size = max(float(getattr(mesh_2d.grid, "gap_size", 0.0)), 0.0)
    clearance = max(gap_size * 0.20, tile_size * 0.008, 1e-6)
    iterations = max(80, int(getattr(params, "hinge_layout_iterations", 120)))
    connection_weight = max(8.0, float(getattr(params, "hinge_layout_connection_weight", 3.0)) * 2.5)
    collision_weight = max(1.5, float(getattr(params, "hinge_layout_collision_weight", 0.35)) * 3.0)
    anchor_weight = max(0.75, float(getattr(params, "hinge_layout_anchor_weight", 0.03)) * 12.0)
    max_drift = min(0.75, max(0.25, float(getattr(params, "hinge_layout_max_center_drift_tiles", 2.0))))

    before_hinge = _hinge_rms(rest, constraints)
    before_rel = _relative_center_rms(rest, rest, constraints)
    solved, solver_metrics = solver(
        rest,
        constraints,
        footprint_builder=lambda layout: np.asarray(layout, dtype=float),
        initial_xy=rest,
        iterations=iterations,
        connection_weight=connection_weight,
        collision_weight=collision_weight,
        anchor_weight=anchor_weight,
        clearance=clearance,
        stage_name="K2D OneString rigid-tile vertex-hinge layout",
        time_budget_sec=float(getattr(params, "hinge_layout_time_budget_sec", 8.0)),
        max_candidate_pairs=int(getattr(params, "hinge_layout_max_candidate_pairs", 3000)),
        collision_sweeps_per_iteration=int(getattr(params, "hinge_layout_collision_sweeps_per_iteration", 2)),
        initial_expansion=1.0,
        max_center_drift_tiles=max_drift,
        progress_callback=progress_callback,
    )
    solved = np.asarray(solved, dtype=float)
    shape_rms_after, shape_max_after = _tile_shape_error_to_k3d(solved, k3d_tiles)

    if shape_max_after > max(shape_max_before + 1e-7, 1e-5 * tile_size):
        solved = rest.copy()
        shape_rms_after, shape_max_after = shape_rms_before, shape_max_before
        fallback = True
    else:
        fallback = False

    penetration_tol = max(tile_size * float(getattr(params, "k2d_hard_nonoverlap_relative_tolerance", 1e-9)), 1e-12)
    solved, hard_metrics = _hard_nonoverlap_project(
        pipeline,
        solved,
        max_sweeps=int(getattr(params, "k2d_hard_nonoverlap_max_sweeps", 400)),
        penetration_tolerance=penetration_tol,
    )
    after_hinge = _hinge_rms(solved, constraints)
    after_rel = _relative_center_rms(solved, rest, constraints)
    shape_rms_after, shape_max_after = _tile_shape_error_to_k3d(solved, k3d_tiles)

    if int(hard_metrics.get("onestring_k2d_hard_nonoverlap_final_penetration_count", 0)) > 0:
        raise RuntimeError(
            "OPTCUTS_TEST_K2D_HARD_NONOVERLAP_FAILED: rigid K2D still contains positive-area "
            f"tile penetrations after hard SAT projection; count={hard_metrics.get('onestring_k2d_hard_nonoverlap_final_penetration_count')} "
            f"max_depth={hard_metrics.get('onestring_k2d_hard_nonoverlap_max_depth_after'):.9g}."
        )

    metrics = {
        "onestring_k2d_rigid_layout_applied": True,
        "onestring_k2d_authoritative_representation": "independent rigid tiles; shared QuadMesh retained only for topology compatibility",
        "onestring_k2d_model": "K3D intrinsic quad -> per-tile flatten -> SE(2) soft solve -> hard SAT non-overlap projection",
        "onestring_k2d_shared_vertex_metric_embedding_authoritative": False,
        "onestring_k2d_tile_pose_dof": "SE(2): theta, tx, ty per tile; hard projection uses rigid translation only",
        "onestring_k2d_hinge_model": "one alternating vertex joint per neighboring tile pair; hinge proximity remains soft, non-overlap is hard",
        "onestring_k2d_hinge_constraint_count": int(len(constraints)),
        "onestring_k2d_clearance": float(clearance),
        "onestring_k2d_connection_weight": float(connection_weight),
        "onestring_k2d_collision_weight": float(collision_weight),
        "onestring_k2d_anchor_weight": float(anchor_weight),
        "onestring_k2d_initial_expansion": 1.0,
        "onestring_k2d_max_center_drift_tiles": float(max_drift),
        "onestring_k2d_hinge_rms_before": float(before_hinge),
        "onestring_k2d_hinge_rms_after": float(after_hinge),
        "onestring_k2d_relative_center_rms_from_M2D": float(after_rel),
        "onestring_k2d_shape_rms_error_to_K3D_before": float(shape_rms_before),
        "onestring_k2d_shape_max_error_to_K3D_before": float(shape_max_before),
        "onestring_k2d_shape_rms_error_to_K3D_after": float(shape_rms_after),
        "onestring_k2d_shape_max_error_to_K3D_after": float(shape_max_after),
        "onestring_k2d_tile_rigidity_preserved": bool(shape_max_after <= max(shape_max_before + 1e-7, 1e-5 * tile_size)),
        "onestring_k2d_rigid_fallback_used": bool(fallback),
        "onestring_k2d_hard_nonoverlap_penetration_tolerance": float(penetration_tol),
        **dict(solver_metrics or {}),
        **hard_metrics,
    }
    print(
        "[OPTCUTS-TEST-K2D-HARD] "
        f"tiles={len(solved)} hinges={len(constraints)} "
        f"penetrations={hard_metrics.get('onestring_k2d_hard_nonoverlap_initial_penetration_count', 0)}"
        f"->{hard_metrics.get('onestring_k2d_hard_nonoverlap_final_penetration_count', 0)} "
        f"max_depth={hard_metrics.get('onestring_k2d_hard_nonoverlap_max_depth_after', 0.0):.6g} "
        f"hinge_rms={before_hinge:.6g}->{after_hinge:.6g} shape_max={shape_max_after:.6g} fallback={fallback}"
    )
    return solved, metrics


def _layout_collision_metrics(pipeline: Any, tiles: np.ndarray) -> tuple[int, float]:
    t = np.asarray(tiles, dtype=float)
    if len(t) < 2:
        return 0, 0.0
    scale = _tile_scale(t)
    tol = max(scale * 1e-9, 1e-12)
    bad = _penetrating_pairs(pipeline, t, tol)
    min_clear = -max((item[3] for item in bad), default=0.0)
    return int(len(bad)), float(min_clear)


def install_optcuts_test_k2d_relative_layout_patch(pipeline: Any) -> None:
    """Replace OptCuts_test K2D flat layout by OneString rigid-tile layout."""
    if getattr(pipeline, "_onestring_optcuts_test_k2d_relative_layout_installed", False):
        return

    base_optimize = pipeline._optimize_k2d
    base_make_layout = pipeline._make_flat_tile_layout

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        result = base_optimize(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        k2d_mesh, report = result
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return result

        rigid_tiles, metrics = _build_rigid_k2d_layout(
            pipeline,
            mesh_2d,
            mesh_3d,
            params,
            progress_callback=progress_callback,
        )
        setattr(k2d_mesh, "_onestring_k2d_rigid_tile_xy", np.asarray(rigid_tiles, dtype=float))
        setattr(k2d_mesh, "_onestring_k2d_m2d_reference_xy", np.asarray(mesh_2d.vertices, dtype=float)[np.asarray(mesh_2d.faces, int), :2])
        try:
            k2d_mesh.metrics.update(metrics)
            k2d_mesh.metrics["objective"] = "OneString K2D rigid tiles; collision-free placement is a hard acceptance constraint"
            k2d_mesh.metrics["actual_backend"] = "OneString SE2 + hard SAT non-overlap projection"
            k2d_mesh.metrics["dominant_backend"] = "cpu rigid-tile local/global + SAT feasibility projection"
        except Exception:
            pass
        try:
            report.objective = "Rigid-tile flat linkage with hard positive-area non-overlap constraint."
            report.constraint_violation = float(metrics.get("onestring_k2d_hard_nonoverlap_final_penetration_count", 0.0))
        except Exception:
            pass
        return k2d_mesh, report

    def make_layout(mesh: Any, params: Any = None):
        layout = base_make_layout(mesh, params)
        rigid = getattr(mesh, "_onestring_k2d_rigid_tile_xy", None)
        if rigid is None:
            return layout
        rigid = np.asarray(rigid, dtype=float)
        if rigid.ndim != 3 or rigid.shape[1:] != (4, 2) or len(rigid) != len(layout.tile_top_vertices_2d):
            return layout
        layout.tile_top_vertices_2d = rigid.copy()
        layout.gap_polygons = []
        collisions, min_clear = _layout_collision_metrics(pipeline, rigid)
        if collisions > 0:
            raise RuntimeError(
                "OPTCUTS_TEST_K2D_LAYOUT_HARD_NONOVERLAP_ASSERTION_FAILED: authoritative rigid K2D "
                f"contains {collisions} positive-area overlaps after optimization."
            )
        try:
            layout.metrics.update(dict(getattr(mesh, "metrics", {}) or {}))
            layout.metrics.update({
                "layout_type": "OneString rigid-tile vertex-hinge flat linkage with hard non-overlap",
                "tile_overlap_count": int(collisions),
                "min_clearance": float(min_clear),
                "k2d_gap_count": int(len(getattr(layout, "hinge_pairs", []) or [])),
                "onestring_k2d_flat_layout_uses_authoritative_rigid_tiles": True,
                "onestring_k2d_legacy_shared_layout_discarded": True,
                "onestring_k2d_hard_nonoverlap_final_assertion": True,
            })
        except Exception:
            pass
        return layout

    pipeline._optimize_k2d = optimize
    pipeline._make_flat_tile_layout = make_layout
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k2d = optimize
        original._make_flat_tile_layout = make_layout

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k2d"] = optimize
            glb["_make_flat_tile_layout"] = make_layout

    pipeline._onestring_optcuts_test_k2d_relative_layout_installed = True


__all__ = ["install_optcuts_test_k2d_relative_layout_patch"]
