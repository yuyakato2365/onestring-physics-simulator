"""Side-face/contact-aware T3D extrusion patch.

This file is intentionally a compatibility wrapper for the user's existing
onestring_pipeline.py.  It loads the backed-up original module and replaces only
_extrude_tiles() with a miter/contact-plane version, so the old Copy-Item based
workflow can be used without shipping a full copy of the large source file.

Expected workflow:
  Copy-Item .\src .\src_backup_before_sideface_contact -Recurse -Force
  Copy-Item .\sideface_contact_tmp\onestring_physics\* .\src\onestring_physics\ -Recurse -Force
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


def _project_root_from_this_file() -> Path:
    # <project>/src/onestring_physics/onestring_pipeline.py
    return Path(__file__).resolve().parents[2]


def _find_original_pipeline() -> Path:
    root = _project_root_from_this_file()
    candidates = [
        root / "src_backup_before_sideface_contact" / "onestring_physics" / "onestring_pipeline.py",
        root / "src_backup_before_mitered_t3d" / "onestring_physics" / "onestring_pipeline.py",
        root / "src_backup_before_sideface_contact" / "src" / "onestring_physics" / "onestring_pipeline.py",
        root / "src_backup_before_mitered_t3d" / "src" / "onestring_physics" / "onestring_pipeline.py",
        root / "src" / "onestring_physics" / "onestring_pipeline.py.bak_mitered_t3d",
    ]
    for path in candidates:
        if not path.exists() or path.resolve() == Path(__file__).resolve():
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:2500]
        except Exception:
            head = ""
        # If the user re-runs the old copy commands after a failed patch, the
        # backup directory may accidentally contain this wrapper instead of the
        # real original file.  Skip wrapper backups to avoid recursive imports and
        # continue to older backups such as src_backup_before_mitered_t3d.
        if "Side-face/contact-aware T3D extrusion patch" in head and "_find_original_pipeline" in head:
            continue
        return path
    tried = "\n  - ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Could not find the original onestring_pipeline.py backup.\n"
        "Run the backup command before copying this patch:\n"
        "  Copy-Item .\\src .\\src_backup_before_sideface_contact -Recurse -Force\n\n"
        f"Tried:\n  - {tried}"
    )


_ORIGINAL_PATH = _find_original_pipeline()
_ORIGINAL_MODULE_NAME = "onestring_physics._onestring_pipeline_original_sideface_contact"

_spec = importlib.util.spec_from_file_location(_ORIGINAL_MODULE_NAME, _ORIGINAL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load original pipeline from {_ORIGINAL_PATH}")
_original = importlib.util.module_from_spec(_spec)
sys.modules[_ORIGINAL_MODULE_NAME] = _original
_spec.loader.exec_module(_original)


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12 or not np.isfinite(n):
        if fallback is None:
            return np.zeros_like(arr, dtype=float)
        fb = np.asarray(fallback, dtype=float)
        fb_n = float(np.linalg.norm(fb))
        return fb / max(fb_n, 1e-12)
    return arr / n


def _edge_inward_normal(top: np.ndarray, face_normal: np.ndarray, edge: tuple[int, int]) -> np.ndarray:
    """Plane normal for the side face through a top edge, pointing into the tile.

    The normal lies in the tile plane and is perpendicular to the edge.  Its sign
    is chosen so that the tile center is on the positive side.
    """
    a, b = edge
    p0 = np.asarray(top[a], dtype=float)
    p1 = np.asarray(top[b], dtype=float)
    center = np.mean(top, axis=0)
    edge_dir = _normalize(p1 - p0, np.array([1.0, 0.0, 0.0]))
    q = np.cross(edge_dir, face_normal)
    q = _normalize(q, np.array([0.0, 1.0, 0.0]))
    mid = 0.5 * (p0 + p1)
    if float(np.dot(q, center - mid)) < 0.0:
        q = -q
    return q


def _build_edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tile_id, face in enumerate(np.asarray(faces, dtype=int)):
        for edge_id, (a, b) in enumerate(local_edges):
            key = tuple(sorted((int(face[a]), int(face[b]))))
            incidence.setdefault(key, []).append((int(tile_id), int(edge_id)))
    return incidence


def _solve_bottom_vertex(
    top: np.ndarray,
    face_normal: np.ndarray,
    thickness: float,
    side_normals: list[np.ndarray],
    vertex_id: int,
) -> tuple[np.ndarray, bool]:
    """Return bottom vertex and whether fallback was used."""
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    center = np.mean(top, axis=0)
    prev_edge = (vertex_id - 1) % 4
    next_edge = vertex_id % 4

    q_prev = side_normals[prev_edge]
    q_next = side_normals[next_edge]
    mid_prev = 0.5 * (top[local_edges[prev_edge][0]] + top[local_edges[prev_edge][1]])
    mid_next = 0.5 * (top[local_edges[next_edge][0]] + top[local_edges[next_edge][1]])

    bottom_plane_c = float(np.dot(face_normal, center) - float(thickness))
    a_mat = np.vstack([face_normal, q_prev, q_next])
    b_vec = np.asarray(
        [
            bottom_plane_c,
            float(np.dot(q_prev, mid_prev)),
            float(np.dot(q_next, mid_next)),
        ],
        dtype=float,
    )

    fallback = np.asarray(top[vertex_id], dtype=float) - float(thickness) * face_normal
    try:
        cond = float(np.linalg.cond(a_mat))
        if not np.isfinite(cond) or cond > 1e10:
            return fallback, True
        out = np.linalg.solve(a_mat, b_vec)
        if not np.all(np.isfinite(out)):
            return fallback, True
        return out, False
    except Exception:
        return fallback, True


def _extrude_tiles(mesh, thickness: float, stage: str):
    """Extrude K3D tiles using shared-edge miter/contact planes.

    Previous behavior:
        bottom = top - thickness * tile_normal

    New behavior:
        - top face remains K3D
        - bottom vertices lie on the offset bottom plane
        - each side face lies on an edge plane
        - shared edges use a single miter/contact plane derived from the two
          adjacent tiles, so neighboring thick panels meet consistently
    """
    import time

    start = time.perf_counter()
    top_tiles = _original._mesh_tiles(mesh)
    tile_count = int(top_tiles.shape[0])
    vertices = np.zeros((tile_count, 8, 3), dtype=float)
    transforms = np.zeros((tile_count, 4, 4), dtype=float)
    local_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    if tile_count == 0:
        top_faces = np.asarray([], dtype=int).reshape(0, 4)
        bottom_faces = np.asarray([], dtype=int).reshape(0, 4)
        side_faces = np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)
        assembly = _original.TileAssembly(
            vertices=vertices,
            top_faces=top_faces,
            bottom_faces=bottom_faces,
            side_faces=side_faces,
            stage=stage,
            metrics={
                "objective": "Contact-aware mitered extrusion.",
                "extrusion_model": "mitered_contact_planes",
                "contact_aware_extrusion": True,
                "tile_thickness": float(thickness),
                "tile_count": 0,
            },
            transform_matrices=transforms,
        )
        report = _original.StageReport(
            name=f"{mesh.stage} -> {stage}",
            objective="Extrude K3D into contact-aware mitered eight-vertex frustum tiles.",
            before_error=0.0,
            after_error=0.0,
            constraint_violation=0.0,
            computation_time=time.perf_counter() - start,
            counts=_original._assembly_counts(assembly),
        )
        return assembly, report

    normals = np.asarray([_original._quad_normal(top) for top in top_tiles], dtype=float)
    raw_side_normals: list[list[np.ndarray]] = []
    for tile_id, top in enumerate(top_tiles):
        raw_side_normals.append([_edge_inward_normal(top, normals[tile_id], edge) for edge in local_edges])

    side_normals: list[list[np.ndarray]] = [[raw_side_normals[i][e].copy() for e in range(4)] for i in range(tile_count)]
    incidence = _build_edge_incidence(mesh.faces)
    internal_miter_edge_count = 0
    boundary_side_plane_count = 0
    nonmanifold_edge_count = 0

    for entries in incidence.values():
        if len(entries) == 1:
            boundary_side_plane_count += 1
            continue
        if len(entries) != 2:
            nonmanifold_edge_count += 1
            continue
        (tile_a, edge_a), (tile_b, edge_b) = entries
        q_a = raw_side_normals[tile_a][edge_a]
        q_b = raw_side_normals[tile_b][edge_b]
        miter = _normalize(q_a - q_b, q_a)
        if float(np.linalg.norm(miter)) <= 1e-12:
            miter = q_a
        side_normals[tile_a][edge_a] = miter
        side_normals[tile_b][edge_b] = -miter
        internal_miter_edge_count += 1

    fallback_count = 0
    for tile_id, top in enumerate(top_tiles):
        normal = normals[tile_id]
        bottom = np.zeros((4, 3), dtype=float)
        for vertex_id in range(4):
            bottom[vertex_id], used_fallback = _solve_bottom_vertex(
                top,
                normal,
                float(thickness),
                side_normals[tile_id],
                vertex_id,
            )
            fallback_count += int(used_fallback)

        vertices[tile_id, :4] = top
        vertices[tile_id, 4:] = bottom

        # IMPORTANT for T2D/animation compatibility:
        # Do not store a shearing/affine top->bottom map here.  The original
        # T2D builder treats transform_matrices as a stable per-tile geometric
        # offset when it lays out thick panels in the flat state.  A least-squares
        # affine map can inject shear/scale into T2D and break the deployment
        # animation.  Keep this transform rigid/translation-only as a safe seed;
        # the patched T2D builder below then rigidly places the full mitered T3D
        # solid so per-tile shape is preserved.
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = np.mean(bottom, axis=0) - np.mean(top, axis=0)
        transforms[tile_id] = transform

    top_faces = np.asarray([[0, 1, 2, 3] for _ in range(tile_count)], dtype=int)
    bottom_faces = np.asarray([[4, 7, 6, 5] for _ in range(tile_count)], dtype=int)
    side_faces = np.asarray([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)

    planarity = _original._tile_face_planarity(vertices)
    face_planarity = _original._tile_face_planarity_by_group(vertices)
    signed_thickness = np.sum((vertices[:, :4] - vertices[:, 4:]) * normals[:, None, :], axis=2)
    thickness_error = signed_thickness - float(thickness)
    center_shift = np.mean(vertices[:, 4:], axis=1) - np.mean(vertices[:, :4], axis=1)
    normal_shift_error = np.linalg.norm(center_shift + float(thickness) * normals, axis=1)

    assembly = _original.TileAssembly(
        vertices=vertices,
        top_faces=top_faces,
        bottom_faces=bottom_faces,
        side_faces=side_faces,
        stage=stage,
        metrics={
            "objective": "Contact-aware mitered extrusion and face planarity report.",
            "extrusion_model": "mitered_contact_planes",
            "contact_aware_extrusion": True,
            "mitered_shared_edge_planes": True,
            "legacy_normal_translation_extrusion": False,
            "t2d_transform_seed_model": "translation_only_center_shift_no_affine_shear",
            "face_planarity_error": planarity,
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            "tile_thickness": float(thickness),
            "thickness_target": float(thickness),
            "thickness_error_rms": float(np.sqrt(np.mean(thickness_error * thickness_error))) if thickness_error.size else 0.0,
            "thickness_error_max": float(np.max(np.abs(thickness_error))) if thickness_error.size else 0.0,
            "normal_translation_center_shift_error_rms": float(np.sqrt(np.mean(normal_shift_error * normal_shift_error))) if normal_shift_error.size else 0.0,
            "internal_miter_edge_count": int(internal_miter_edge_count),
            "boundary_side_plane_count": int(boundary_side_plane_count),
            "nonmanifold_edge_count": int(nonmanifold_edge_count),
            "bottom_vertex_solve_fallback_count": int(fallback_count),
            "surface_fit_error": float(mesh.metrics.get("surface_fit_error_after", 0.0)),
            "tile_count": int(tile_count),
            "k3d_fallback_warning": str(mesh.metrics.get("approximation_warning", "")),
            **_original._tile_orientation_metrics(vertices, f"{stage.lower()}"),
        },
        transform_matrices=transforms,
    )
    report = _original.StageReport(
        name=f"{mesh.stage} -> {stage}",
        objective="Extrude K3D into contact-aware mitered eight-vertex frustum tiles.",
        before_error=0.0,
        after_error=planarity,
        constraint_violation=planarity,
        computation_time=time.perf_counter() - start,
        counts=_original._assembly_counts(assembly),
    )
    return assembly, report



_ORIGINAL_MAKE_T2D_FROM_TRANSFORMS = _original._make_t2d_from_transforms

_ORIGINAL_OPTIMIZE_T2D_FOOTPRINT_LAYOUT = _original._optimize_t2d_footprint_layout
_ORIGINAL_OPTIMIZE_RIGID_ASSEMBLY_HINGE_LAYOUT_2D = _original._optimize_rigid_assembly_hinge_layout_2d


def _grid_with_layout_gap(grid, minimum_gap: float):
    """Return a shallow grid copy whose gap_size is large enough for void layout.

    The original layout solvers use grid.gap_size mainly to set the collision /
    clearance scale.  Increasing it here gives the panel placement stage more
    room to keep voids open without changing the actual K2D/K3D mesh topology.
    """
    import copy

    out = copy.copy(grid)
    try:
        out.gap_size = max(float(getattr(grid, "gap_size", 0.0)), float(minimum_gap))
    except Exception:
        return grid
    return out


def _free_layout_parameters(
    grid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    initial_expansion: float,
    max_center_drift_tiles: float,
) -> dict[str, float | int]:
    tile_size = max(float(getattr(grid, "tile_size", 1.0)), 1e-8)
    requested_gap = float(getattr(grid, "gap_size", 0.08))
    # A larger optimization-only void clearance.  This does not rewrite the mesh;
    # it only tells the placement optimizer to leave visible air between panels.
    layout_gap = max(requested_gap * 1.75, tile_size * 0.10)
    return {
        "iterations": int(max(240, int(iterations) * 3)),
        "connection_weight": float(max(40.0, float(connection_weight) * 12.0)),
        "collision_weight": float(max(3.0, float(collision_weight) * 3.0)),
        # Keep the initial pose as a weak prior, not as a cage.  The old values
        # were too anchor-heavy for mitered solids and could collapse the holes.
        "anchor_weight": float(max(0.003, min(0.025, float(anchor_weight) * 0.25))),
        "initial_expansion": float(max(1.22, float(initial_expansion))),
        "max_center_drift_tiles": float(max(4.0, float(max_center_drift_tiles))),
        "layout_gap": float(layout_gap),
        "clearance": float(max(layout_gap * 0.65, tile_size * 0.035)),
    }


def _layout_quality_for_top_xy(layout: np.ndarray, transforms: np.ndarray, faces: np.ndarray, grid, constraints) -> dict[str, float | int]:
    layout = np.asarray(layout, dtype=float)
    if layout.size == 0:
        return {"hinge_error": 0.0, "collision_count": 0, "min_clearance": 0.0}
    footprints = _original._apply_t2d_transforms_to_top_xy(layout, transforms)[:, :, :2]
    pad = max(float(getattr(grid, "gap_size", 0.08)) * 8.0, float(getattr(grid, "tile_size", 1.0)) * 0.25)
    pairs = _original._spatial_candidate_pairs_for_tiles(footprints, pad=pad)
    specs = _original._vertex_hinge_specs_from_faces(faces)
    return {
        "hinge_error": float(_original._vertex_layout_hinge_error(layout, specs)),
        "collision_count": int(_original._count_2d_footprint_collisions_from_pairs(footprints, pairs)),
        "min_clearance": float(_original._min_footprint_clearance_2d_from_pairs(footprints, pairs)),
    }


def _optimize_t2d_footprint_layout(
    top_xy: np.ndarray,
    transforms: np.ndarray,
    faces: np.ndarray,
    grid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.0,
    max_center_drift_tiles: float = 2.0,
    progress_callback=None,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """More permissive T2D placement for contact-aware thick panels.

    Goal ordering:
      1. vertex hinges should be effectively closed;
      2. projected top+bottom footprints should leave visible voids;
      3. the solution should remain near the expanded initial layout.

    This keeps the original local/global SE(2) solve, but gives it more freedom:
    larger expansion/drift, weaker anchor, stronger connection, and a larger
    collision clearance.  A final hinge-polish pass is accepted only if it does
    not introduce a large collision regression.
    """
    rest = np.asarray(top_xy, dtype=float)
    if len(rest) == 0:
        return rest.copy(), {"t2d_footprint_optimizer": "empty_free_layout"}

    specs = _original._vertex_hinge_specs_from_faces(faces)
    constraints = _original._hinge_constraint_tuples_from_specs(specs)

    def footprint_builder(layout: np.ndarray) -> np.ndarray:
        return _original._apply_t2d_transforms_to_top_xy(layout, transforms)[:, :, :2]

    free = _free_layout_parameters(
        grid,
        iterations,
        connection_weight,
        collision_weight,
        anchor_weight,
        initial_expansion,
        max_center_drift_tiles,
    )
    free_grid = _grid_with_layout_gap(grid, float(free["layout_gap"]))
    before = _layout_quality_for_top_xy(rest, transforms, faces, free_grid, constraints)

    solved, metrics = _original._paper_local_global_se2_layout(
        rest,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=rest,
        iterations=int(free["iterations"]),
        connection_weight=float(free["connection_weight"]),
        collision_weight=float(free["collision_weight"]),
        anchor_weight=float(free["anchor_weight"]),
        clearance=float(free["clearance"]),
        stage_name="T2D Top Hinge free void-preserving placement",
        time_budget_sec=max(float(time_budget_sec), 12.0),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=max(2, int(collision_sweeps_per_iteration)),
        initial_expansion=float(free["initial_expansion"]),
        max_center_drift_tiles=float(free["max_center_drift_tiles"]),
        progress_callback=progress_callback,
    )
    after_free = _layout_quality_for_top_xy(solved, transforms, faces, free_grid, constraints)

    # Hinge polish: make the hinge term even harder.  Because this can close some
    # holes, keep the polished result only if collision/clearance does not regress
    # too far compared with the free-layout solution.
    polished, polish_metrics = _original._paper_local_global_se2_layout(
        rest,
        constraints,
        footprint_builder=footprint_builder,
        initial_xy=solved,
        iterations=max(80, int(iterations)),
        connection_weight=max(120.0, float(free["connection_weight"]) * 2.0),
        collision_weight=max(2.0, float(free["collision_weight"]) * 0.75),
        anchor_weight=max(0.002, float(free["anchor_weight"]) * 0.5),
        clearance=float(free["clearance"]) * 0.75,
        stage_name="T2D Top Hinge hard-hinge polish",
        time_budget_sec=max(4.0, float(time_budget_sec) * 0.5),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=max(1, int(collision_sweeps_per_iteration)),
        initial_expansion=1.0,
        max_center_drift_tiles=float(free["max_center_drift_tiles"]),
        progress_callback=None,
    )
    after_polish = _layout_quality_for_top_xy(polished, transforms, faces, free_grid, constraints)
    accept_polish = (
        after_polish["hinge_error"] <= after_free["hinge_error"] * 0.85 + 1e-8
        and after_polish["collision_count"] <= after_free["collision_count"] + max(1, int(len(rest) * 0.03))
    )
    if accept_polish:
        solved = polished
        final = after_polish
    else:
        final = after_free

    shape_rms = _original._tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
    )
    shape_max = _original._tile_shape_distance_error(
        np.dstack([solved, np.zeros(solved.shape[:2])]),
        np.dstack([rest, np.zeros(rest.shape[:2])]),
        use_max=True,
    )
    out = {
        "t2d_footprint_optimizer": "free local/global SE(2) layout with hard-hinge priority and void clearance",
        "t2d_free_layout_enabled": True,
        "t2d_free_layout_goal_order": "hard hinges > open voids/collision clearance > weak initial-layout anchor",
        "t2d_free_layout_iterations": int(free["iterations"]),
        "t2d_free_layout_connection_weight": float(free["connection_weight"]),
        "t2d_free_layout_collision_weight": float(free["collision_weight"]),
        "t2d_free_layout_anchor_weight": float(free["anchor_weight"]),
        "t2d_free_layout_initial_expansion": float(free["initial_expansion"]),
        "t2d_free_layout_max_center_drift_tiles": float(free["max_center_drift_tiles"]),
        "t2d_free_layout_gap_size_used_for_clearance": float(free["layout_gap"]),
        "t2d_free_layout_clearance": float(free["clearance"]),
        "t2d_hard_hinge_polish_accepted": bool(accept_polish),
        "t2d_footprint_collision_checked_on": "top+bottom projected footprint with SAT, enlarged optimization-only clearance",
        "t2d_footprint_hinge_error_before": float(before["hinge_error"]),
        "t2d_footprint_hinge_error_after": float(final["hinge_error"]),
        "t2d_footprint_collision_count_before": int(before["collision_count"]),
        "t2d_footprint_collision_count_after": int(final["collision_count"]),
        "t2d_footprint_min_clearance_before": float(before["min_clearance"]),
        "t2d_footprint_min_clearance_after": float(final["min_clearance"]),
        "t2d_top_tile_shape_rms_error_after_footprint_layout": float(shape_rms),
        "t2d_top_tile_shape_max_error_after_footprint_layout": float(shape_max),
        "t2d_top_shape_preserved_by_rigid_pose_fit": bool(shape_max < 1e-8),
        **metrics,
    }
    out.update({f"hard_hinge_polish_{k}": v for k, v in polish_metrics.items() if isinstance(v, (int, float, str, bool))})
    return solved, out


def _optimize_rigid_assembly_hinge_layout_2d(
    rest_vertices: np.ndarray,
    hinges,
    grid,
    iterations: int,
    connection_weight: float,
    collision_weight: float,
    anchor_weight: float,
    time_budget_sec: float = 8.0,
    max_candidate_pairs: int = 3000,
    collision_sweeps_per_iteration: int = 2,
    initial_expansion: float = 1.08,
    max_center_drift_tiles: float = 2.0,
    progress_callback=None,
):
    """More permissive dual-hinge/full-panel placement.

    This wraps the original rigid assembly optimizer but deliberately relaxes the
    anchor and expands the trust region, so panels can rearrange to open voids.
    Connection and collision weights are raised to keep hinges closed and panels
    separated.
    """
    free = _free_layout_parameters(
        grid,
        iterations,
        connection_weight,
        collision_weight,
        anchor_weight,
        initial_expansion,
        max_center_drift_tiles,
    )
    free_grid = _grid_with_layout_gap(grid, float(free["layout_gap"]))
    vertices, metrics = _ORIGINAL_OPTIMIZE_RIGID_ASSEMBLY_HINGE_LAYOUT_2D(
        rest_vertices=rest_vertices,
        hinges=hinges,
        grid=free_grid,
        iterations=int(free["iterations"]),
        connection_weight=max(60.0, float(free["connection_weight"])),
        collision_weight=max(3.5, float(free["collision_weight"])),
        anchor_weight=float(free["anchor_weight"]),
        time_budget_sec=max(float(time_budget_sec), 12.0),
        max_candidate_pairs=int(max_candidate_pairs),
        collision_sweeps_per_iteration=max(2, int(collision_sweeps_per_iteration)),
        initial_expansion=float(free["initial_expansion"]),
        max_center_drift_tiles=float(free["max_center_drift_tiles"]),
        progress_callback=progress_callback,
    )

    # Final rigid hinge closure pass.  This translates whole tiles toward their
    # hinge midpoints and reprojects each tile onto its original rigid shape.  It
    # gives the user the intended behavior: hinges are treated as nearly hard
    # constraints, while the preceding solve already made room for voids.
    repaired = vertices.copy()
    before_hinge = float(_original._hinge_connection_error(repaired, hinges)) if hinges else 0.0
    for _ in range(16):
        _original._project_hinge_tile_translations(repaired, hinges, 1.0)
        _original._project_aabb_collisions(repaired, 0.08, grid=free_grid, all_pairs=False)
        _original._project_rigid_tiles(repaired, rest_vertices, 1.0)
    after_hinge = float(_original._hinge_connection_error(repaired, hinges)) if hinges else 0.0
    # Use the hard-closed result unless it catastrophically increases AABB overlaps.
    old_coll = int(_original._count_aabb_collisions(vertices, free_grid))
    new_coll = int(_original._count_aabb_collisions(repaired, free_grid))
    accept_repair = after_hinge <= before_hinge + 1e-8 and new_coll <= old_coll + max(1, int(len(repaired) * 0.04))
    if accept_repair:
        vertices = repaired
    else:
        after_hinge = before_hinge
        new_coll = old_coll

    metrics = dict(metrics)
    metrics.update(
        {
            "dual_hinge_free_layout_enabled": True,
            "dual_hinge_free_layout_goal_order": "hard hinges > open voids/collision clearance > weak initial-layout anchor",
            "dual_hinge_free_layout_iterations": int(free["iterations"]),
            "dual_hinge_free_layout_connection_weight": float(max(60.0, float(free["connection_weight"]))),
            "dual_hinge_free_layout_collision_weight": float(max(3.5, float(free["collision_weight"]))),
            "dual_hinge_free_layout_anchor_weight": float(free["anchor_weight"]),
            "dual_hinge_free_layout_initial_expansion": float(free["initial_expansion"]),
            "dual_hinge_free_layout_max_center_drift_tiles": float(free["max_center_drift_tiles"]),
            "dual_hinge_free_layout_gap_size_used_for_clearance": float(free["layout_gap"]),
            "dual_hinge_hard_hinge_repair_accepted": bool(accept_repair),
            "dual_hinge_hard_hinge_error_before_repair": float(before_hinge),
            "dual_hinge_hard_hinge_error_after_repair": float(after_hinge),
            "dual_hinge_collision_count_after_hard_repair": int(new_coll),
        }
    )
    return vertices, metrics



def _make_t2d_from_transforms(mesh_2d, flat_layout, mesh_3d, tiles_3d, stage: str, params=None):
    """Build T2D while preserving the full mitered T3D tile shape.

    The first side-face patch changed T3D tiles from translation extrusions into
    mitered frusta.  Those tiles are no longer representable by a single affine
    top->bottom transform without shear.  The old T2D path used the transform to
    create bottom vertices from K2D top vertices, so an affine transform could
    distort the flat panels and break the animation.

    Compatibility strategy:
    1. Let the original T2D builder solve the flat top/footprint layout, using
       the safe translation-only transform seed stored by _extrude_tiles().
    2. Replace each resulting tile by a rigid placement of the actual mitered
       T3D solid at that solved flat top pose.

    This keeps the original working T2D layout behavior but restores the most
    important physical invariant for deployment: each T2D tile and its T3D target
    are the same rigid 8-vertex solid up to rotation/translation.
    """
    start = time.perf_counter()
    base_assembly, base_report = _ORIGINAL_MAKE_T2D_FROM_TRANSFORMS(
        mesh_2d,
        flat_layout,
        mesh_3d,
        tiles_3d,
        stage,
        params,
    )
    if len(base_assembly.vertices) == 0:
        return base_assembly, base_report

    placed_vertices = np.zeros_like(base_assembly.vertices)
    rigid_transforms = np.zeros((len(base_assembly.vertices), 4, 4), dtype=float)
    top_errors = []
    for tile_id in range(len(base_assembly.vertices)):
        flat_top = base_assembly.vertices[tile_id, :4]
        placed, transform = _original._rigidly_place_t3d_tile_in_flat_layout(
            tiles_3d.vertices[tile_id],
            flat_top,
        )
        placed_vertices[tile_id] = placed
        rigid_transforms[tile_id] = transform
        top_errors.append(np.linalg.norm(placed[:4, :2] - flat_top[:, :2], axis=1))

    top_errors_arr = np.asarray(top_errors, dtype=float).reshape(-1) if top_errors else np.zeros(0)
    face_planarity = _original._tile_face_planarity_by_group(placed_vertices)
    full_shape_rms = _original._tile_shape_distance_error(placed_vertices, tiles_3d.vertices)
    full_shape_max = _original._tile_shape_distance_error(placed_vertices, tiles_3d.vertices, use_max=True)
    top_shape_rms = _original._tile_shape_distance_error(placed_vertices[:, :4, :], tiles_3d.vertices[:, :4, :])
    top_shape_max = _original._tile_shape_distance_error(placed_vertices[:, :4, :], tiles_3d.vertices[:, :4, :], use_max=True)

    metrics = dict(base_assembly.metrics)
    metrics.update(
        {
            "t2d_geometry_repair_applied": True,
            "t2d_geometry_repair_reason": "mitered T3D cannot be represented by affine/shear top-to-bottom transforms without breaking rigid-panel animation",
            "t2d_geometry_model": "rigidly_placed_mitered_T3D_tiles_after_original_flat_layout",
            "transform_source": "rigid placement of each full mitered T3D tile onto the solved flat top pose",
            "fabrication_geometry_model": "T2D preserves the complete 8-vertex mitered T3D tile shape; top pose comes from the original K2D/T2D layout solve",
            "rigid_copy_of_T3D_forced": True,
            "paper_t2d_extrusion_model": True,
            "t2d_t3d_congruent_tile_geometry": bool(full_shape_max < 1e-6),
            "tile_shape_rms_error_to_T3D": float(full_shape_rms),
            "tile_shape_max_error_to_T3D": float(full_shape_max),
            "top_tile_shape_rms_error_to_K3D": float(top_shape_rms),
            "top_tile_shape_max_error_to_K3D": float(top_shape_max),
            "top_vertices_match_pre_repair_flat_layout_max_error": float(np.max(top_errors_arr)) if top_errors_arr.size else 0.0,
            "top_vertices_match_pre_repair_flat_layout_rms_error": float(np.sqrt(np.mean(top_errors_arr * top_errors_arr))) if top_errors_arr.size else 0.0,
            "face_planarity_error": _original._tile_face_planarity(placed_vertices),
            "top_face_planarity_error": face_planarity["top"],
            "bottom_face_planarity_error": face_planarity["bottom"],
            "side_face_planarity_error": face_planarity["side"],
            **_original._tile_orientation_metrics(placed_vertices, "t2d"),
        }
    )
    repaired = _original.TileAssembly(
        vertices=placed_vertices,
        top_faces=base_assembly.top_faces.copy(),
        bottom_faces=base_assembly.bottom_faces.copy(),
        side_faces=base_assembly.side_faces.copy(),
        stage=base_assembly.stage,
        metrics=metrics,
        transform_matrices=rigid_transforms,
    )
    report = _original.StageReport(
        name=base_report.name,
        objective="Generate T2D by original flat layout solve, then rigidly place contact-aware mitered T3D tiles.",
        before_error=base_report.before_error,
        after_error=float(full_shape_rms),
        constraint_violation=float(metrics.get("top_vertices_match_pre_repair_flat_layout_rms_error", 0.0)),
        computation_time=float(base_report.computation_time) + (time.perf_counter() - start),
        failed_constraints=list(getattr(base_report, "failed_constraints", [])),
        counts=_original._assembly_counts(repaired),
    )
    return repaired, report


# Patch the original module in-place. Functions such as build_onestring_design()
# keep their original global namespace, so this assignment is what makes them call
# the new extrusion implementation.
_original._extrude_tiles = _extrude_tiles
_original._optimize_t2d_footprint_layout = _optimize_t2d_footprint_layout
_original._optimize_rigid_assembly_hinge_layout_2d = _optimize_rigid_assembly_hinge_layout_2d
_original._make_t2d_from_transforms = _make_t2d_from_transforms

# Re-export the original module's API from this wrapper.
for _name, _value in _original.__dict__.items():
    if _name in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "_extrude_tiles",
    }:
        continue
    globals()[_name] = _value

globals()["_extrude_tiles"] = _extrude_tiles
globals()["_optimize_t2d_footprint_layout"] = _optimize_t2d_footprint_layout
globals()["_optimize_rigid_assembly_hinge_layout_2d"] = _optimize_rigid_assembly_hinge_layout_2d
globals()["_make_t2d_from_transforms"] = _make_t2d_from_transforms
globals()["SIDEFACE_CONTACT_PATCH_ACTIVE"] = True
globals()["SIDEFACE_CONTACT_PATCH_ORIGINAL_PATH"] = str(_ORIGINAL_PATH)



# ---------------------------------------------------------------------------
# Streamlit animation/simulation cache
# ---------------------------------------------------------------------------
_ORIGINAL_SIMULATE_ONESTRING_DEPLOYMENT = _original.simulate_onestring_deployment


def _deployment_params_cache_key(params) -> str:
    """Stable-ish key for deployment settings used by the Streamlit UI cache."""
    try:
        import dataclasses
        import json
        import hashlib

        if dataclasses.is_dataclass(params):
            payload = dataclasses.asdict(params)
        else:
            payload = dict(getattr(params, "__dict__", {}))
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    except Exception:
        return repr(params)


def _state_cache_key(state) -> str:
    try:
        import streamlit as st
        pipeline_key = st.session_state.get("pipeline_key")
        if pipeline_key is not None:
            return repr(pipeline_key)
    except Exception:
        pass
    try:
        v = np.asarray(state.tiles_2d_dual_hinge.vertices)
        t = np.asarray(state.tiles_3d.vertices)
        summary = (
            tuple(v.shape),
            tuple(t.shape),
            float(np.nanmean(v)) if v.size else 0.0,
            float(np.nanmean(t)) if t.size else 0.0,
            float(np.nanstd(v)) if v.size else 0.0,
            float(np.nanstd(t)) if t.size else 0.0,
        )
        return repr(summary)
    except Exception:
        return str(id(state))


def simulate_onestring_deployment(state, params=None, progress_callback=None):
    """Cached wrapper around the original deployment simulation.

    The app's Assembly Animation view can be rerun many times while the user only
    changes camera/player UI.  Keep a session-state cache of previously generated
    simulation frames so returning to the same settings reuses the animation
    instead of recomputing it.
    """
    cache_enabled = True
    cache = None
    key = None
    try:
        import streamlit as st
        cache = st.session_state.setdefault("onestring_animation_result_cache", {})
        key = ("deployment", _state_cache_key(state), _deployment_params_cache_key(params))
        if key in cache:
            if progress_callback is not None:
                try:
                    progress_callback("Cached deployment simulation", 1.0, "reusing previously generated animation frames")
                except Exception:
                    pass
            return cache[key]
    except Exception:
        cache_enabled = False

    result = _ORIGINAL_SIMULATE_ONESTRING_DEPLOYMENT(state, params, progress_callback=progress_callback)

    if cache_enabled and cache is not None and key is not None:
        try:
            cache[key] = result
            # Avoid unbounded growth while letting the user switch between a few
            # frame counts / solver settings during tuning.
            if len(cache) > 8:
                oldest_key = next(iter(cache.keys()))
                if oldest_key != key:
                    cache.pop(oldest_key, None)
        except Exception:
            pass
    return result


_original.simulate_onestring_deployment = simulate_onestring_deployment
globals()["simulate_onestring_deployment"] = simulate_onestring_deployment
globals()["ONESTRING_ANIMATION_CACHE_ACTIVE"] = True
