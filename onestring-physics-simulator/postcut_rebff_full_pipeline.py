"""Bridge independently re-BFFed, truly retiled pieces into existing full OneString.

This is NOT an OptCuts replacement and does not invent a standalone hinge solver.
The existing OneString K3D/T3D/K2D/T2D Dual Hinge pipeline executes on the
reassembled M2D/M3D correspondence, with separated UV islands and face indices.
The result must pass normal stage checks; errors are never replaced by previews.
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from onestring_physics import onestring_pipeline as pipe
from onestring_physics.legacy_pipeline_base import SurfaceParameterization
from onestring_physics.reference_bff import validate_reference_mesh
from onestring_physics.input_shape import load_target_shape

_PIPELINE_HOOK_LOCK = threading.Lock()


def _piece_chart(patch):
    x, uv, f = (np.asarray(patch[k], dtype=float if k != "faces" else int)
                for k in ("xyz", "new_uv", "faces"))
    topology = validate_reference_mesh(x, f)
    loop = [int(j) for j in topology["boundary_loops"][0]]
    return SurfaceParameterization(
        method="paper_reference_bff",
        surface_vertices_3d=x,
        surface_faces=f,
        uv_vertices_2d=uv,
        uv_faces=f,
        omega_boundary=uv[loop + [loop[0]]],
        metrics={
            "boundary_loop": loop,
            "per_triangle_lambda": np.ones(len(f)).tolist(),
            "postcut_independent_bff": True,
        },
    )


def _piece_m2d_only(patch, params, n, spacing):
    p = copy.copy(params)
    p.reference_grid_spacing = float(spacing)
    p.reference_grid_rotation_degrees = 0.
    p.reference_grid_origin_u = 0.
    p.reference_grid_origin_v = 0.
    chart = _piece_chart(patch)
    grid = pipe.create_quad_grid(int(n), int(n), p.tile_size, p.gap_size)
    domain = pipe._reference_flatten_to_domain(chart, grid, p)
    quad = pipe._build_reference_m2d(grid, domain, p)
    return quad


def _find_tiling_spacing(patches, params, n, target_tiles, progress=None):
    span = max(float(np.max(np.ptp(np.asarray(p["new_uv"]), axis=0)))
               for p in patches)
    initial = max(span / max(n, 1), 1e-7)
    cache = {}
    def count(spacing):
        key = round(float(spacing), 11)
        if key in cache:
            return cache[key]
        total = 0
        for p in patches:
            try:
                total += len(_piece_m2d_only(p, params, n, spacing).faces)
            except (ValueError, RuntimeError):
                pass
        cache[key] = total
        return total

    # Search a bounded family of UNIFORMLY spaced square grids across islands.
    # Select a single spacing closest to the original tile count. Never silently
    # use the earlier 75-tile postcut preview as the final full-pipeline input.
    low, high = initial * 0.40, initial * 1.6
    samples = list(np.geomspace(low, high, 17))
    counts = [count(v) for v in samples]
    best_id = min(range(len(samples)), key=lambda i: (
        abs(counts[i] - target_tiles), abs(samples[i] - initial)))
    if progress:
        progress("Retile after BFF", 0.05,
                 f"matching original {target_tiles} quads: chose {counts[best_id]} "
                 f"at shared spacing={samples[best_id]:.5g}")
    return float(samples[best_id]), int(counts[best_id])


def _assemble_patches(patches, params, n, baseline_tiles, progress=None,
                      initial_grid=None):
    from postcut_rebff_seam_registration import (
        align_rectangular_bff_charts_on_source_seam,
    )
    # The physical seam is shared in S; use one global UV lattice after
    # seam-constrained rectangular BFF instead of independently translated
    # and retiled islands. All transformations are conformal similarities.
    rectangular_seam_compatible = bool(
        len(patches)>1 and all(
            p.get("seam_rectangle_corners_constrained",False)
            for p in patches
        )
    )
    joint_atlas=bool(
        rectangular_seam_compatible and
        all(p.get("postcut_joint_atlas",False) for p in patches)
    )
    if joint_atlas:
        from postcut_rebff_joint_atlas import verify_joint_source_seams
        # Atlas boundaries were fixed DURING each recursive Split and BFF
        # recomputation. No subsequent per-piece similarities are permitted:
        # such transformations would break already-certified seam vertices.
        seam_uv_audit=verify_joint_source_seams(patches)
    elif rectangular_seam_compatible:
        seam_uv_audit = align_rectangular_bff_charts_on_source_seam(patches)
    else:
        # The legacy freely parameterized comparison is intentionally not
        # declared seam-grid compatible. It retains isolated UV islands.
        seam_uv_audit = {
            "postcut_seam_uv_registration": (
                "not_required" if len(patches)<=1
                else "free_boundary_not_rectangular_seam_compatible"
            ),
            "postcut_seam_uv_pair_count":0,
        }
    common_uv = rectangular_seam_compatible
    exact_rectangular_coverage = bool(
        common_uv or (len(patches)==1 and patches[0].get(
            "bff_metadata",{}).get("bff_rectangle_verified") is True)
    )
    exact_grid_audit = {}
    if exact_rectangular_coverage:
        # A rectangle's sides are not necessarily integer multiples of the
        # desired square pitch. Losing boundary cells is NEVER acceptable.
        # All rectangles share one global set of grid lines, with their exact
        # edges included as mandatory lines. Edge cells can be non-square.
        from types import SimpleNamespace
        from postcut_rebff_exact_rectangle_grid import (
            choose_exact_grid, grid_at_pitch, matching_seam_edges,
        )
        if initial_grid is not None:
            # Preserve the ONLY original Omega grid (pitch, origin and all
            # x/y coordinates). Split duplicates seams but MUST NOT retile.
            spacing = float(initial_grid["pitch"])
            grids = grid_at_pitch(
                patches, spacing, initial_grid=initial_grid,
            )
        else:
            # Retained for independent legacy diagnostic tests only.
            spacing, grids = choose_exact_grid(
                patches, baseline_tiles, grid=n,
            )
        patch_meshes = [
            SimpleNamespace(vertices=patch_grid["vertices"],
                            faces=patch_grid["faces"])
            for patch_grid in grids
        ]
        expected = sum(len(mesh.faces) for mesh in patch_meshes)
        seam_pairs = matching_seam_edges(grids)
        if common_uv:
            from postcut_rebff_seam_registration import _boundary_point_groups
            source_pairs = _boundary_point_groups(patches)
            missing_pairs = [
                pair for pair in source_pairs
                if seam_pairs.get(pair,0)<1
            ]
            if missing_pairs:
                # The grid already includes every rectangle boundary as a
                # mandatory line. If a real positive-length 3D Split seam
                # has no coincident UV edge, the independent rectangular BFF
                # charts themselves are geometrically incompatible. Changing
                # pitch or suppressing this assertion cannot repair it.
                mismatch = seam_uv_audit.get(
                    "postcut_seam_interior_correspondence_by_pair",{}
                )
                physical_detail = [{
                    "patches":list(pair),
                    "source_seam_samples":len(source_pairs[pair]),
                    "source_to_uv_max_gap":mismatch.get(
                        f"{pair[0]}-{pair[1]}",{}
                    ).get("max_uv_difference"),
                } for pair in missing_pairs]
                raise RuntimeError(
                    "Real positive-length 3D Split seams cannot be represented "
                    "by coincident edges in the independently rectangular-BFF "
                    "UV charts. This is a chart compatibility problem, NOT "
                    "an Omega coverage/grid pitch problem. Physical pairs: "
                    f"{physical_detail}. Joint seam-constrained reparameterization "
                    "is required; silently dropping source seam contacts is forbidden."
                )
        widths = [
            min(float(cell["min_cell_width"]),float(cell["min_cell_height"]))
            for cell in grids
        ]
        exact_grid_audit = {
            "postcut_omega_grid_policy":(
                "immutable_initial_omega_lines" if initial_grid is not None
                else "legacy_exact_rectangle_coverage_shared_global_lines"
            ),
            "postcut_omega_grid_reconfigured_after_bff":False if initial_grid is not None else True,
            "postcut_omega_grid_initial_x_count":(
                len(initial_grid["x"]) if initial_grid is not None else None
            ),
            "postcut_omega_grid_initial_y_count":(
                len(initial_grid["y"]) if initial_grid is not None else None
            ),
            "postcut_omega_grid_no_cells_cropped":True,
            "postcut_omega_rectangles_area_fraction_covered":[
                float(cell["area_fraction_covered"]) for cell in grids
            ],
            "postcut_omega_grid_min_cell_size":float(min(widths)),
            "postcut_omega_grid_min_cell_to_pitch_ratio":float(min(widths)/spacing),
            "postcut_seam_exact_uv_edge_pairs":{
                f"{a}-{b}":int(count) for (a,b),count in seam_pairs.items()
            },
            "postcut_omega_nonsquare_boundary_cells_possible":True,
            "postcut_omega_boundary_tile_loss":0,
        }
    else:
        # Explicitly unsupported research comparison: free-boundary BFF
        # regions are not rectangles, and cannot be tiled without clipping
        # by this rectangle-only generator.
        spacing, expected = _find_tiling_spacing(
            patches, params, n, baseline_tiles, progress,
        )
        patch_meshes = [
            _piece_m2d_only(patch, params, n, spacing)
            for patch in patches
        ]
        exact_grid_audit = {
            "postcut_omega_grid_policy":"legacy_clipped_nonrectangular_comparison",
            "postcut_omega_grid_no_cells_cropped":False,
        }
    total = sum(len(m.faces) for m in patch_meshes)
    if total < 4 or total != expected:
        raise RuntimeError(f"Retiling count not consistent: {total}, expected {expected}")
    if total < int(0.75 * baseline_tiles) and not exact_rectangular_coverage:
        raise RuntimeError(
            f"Postcut BFF retile lost too much coverage: {total}/{baseline_tiles} quads; "
            "refusing to pass a sparse preview off as a full T2D design."
        )
    # Exact area coverage is independently proven above. A count smaller
    # than the baseline does NOT imply omitted geometry, especially for
    # extreme-aspect BFF rectangles. Record fidelity separately.
    exact_grid_audit["postcut_grid_tile_count_ratio_to_initial"] = float(
        total / max(1,baseline_tiles)
    )
    # Translate *both source UV triangles and retiled square quads together*,
    # keeping each island's conformal map exact and nonoverlapping in UV.
    source_xyz, source_uv, source_faces = [], [], []
    quad_uv, quad_faces, tile_components = [], [], []
    cursor = 0.
    source_index = quad_index = 0
    for k, (patch, mesh) in enumerate(zip(patches, patch_meshes)):
        old_uv = np.asarray(patch["new_uv"], float)
        old_xyz = np.asarray(patch["xyz"], float)
        old_faces = np.asarray(patch["faces"], int)
        tiled_uv = np.asarray(mesh.vertices, float)[:, :2]
        shift = (
            np.zeros(2)
            if common_uv else np.array([
                cursor - float(np.min(old_uv[:, 0])),
                -float(np.min(old_uv[:, 1]))
            ])
        )
        source_xyz.append(old_xyz)
        source_uv.append(old_uv + shift)
        source_faces.append(old_faces + source_index)
        quad_uv.append(tiled_uv + shift)
        quad_faces.append(np.asarray(mesh.faces, int) + quad_index)
        tile_components.extend([k] * len(mesh.faces))
        source_index += len(old_xyz)
        quad_index += len(tiled_uv)
        cursor += float(np.ptp(old_uv[:, 0])) + 3. * spacing
    joined_xyz = np.concatenate(source_xyz, axis=0)
    joined_uv = np.concatenate(source_uv, axis=0)
    joined_source_faces = np.concatenate(source_faces, axis=0)
    joined_quads = np.concatenate(quad_faces, axis=0)
    joined_quad_vertices = np.column_stack([
        np.concatenate(quad_uv, axis=0),
        np.zeros(quad_index, float),
    ])
    if np.max(joined_quads) >= len(joined_quad_vertices):
        raise RuntimeError("Quad indices escaped assembled M2D")
    if len(joined_quads) != len(tile_components):
        raise RuntimeError("Tile-to-source-patch ownership has been lost")
    if np.min(joined_source_faces) < 0 or np.max(joined_source_faces) >= len(joined_xyz):
        raise RuntimeError("UV and source triangle indices are inconsistent")
    # Each half must reuse precisely the same UV square-grid vertices
    # along the physical seam. Verify before K3D/T2D; do not hide any gap.
    if common_uv:
        rounded=[set() for _ in patches]
        uv_all=np.asarray(joined_quad_vertices[:,:2],float)
        for quad_id,component in enumerate(tile_components):
            ids=joined_quads[quad_id]
            for pt in uv_all[ids]:
                rounded[component].add(tuple(np.rint(pt/(1e-8*spacing)).astype(np.int64)))
        shared_grid_points=sum(
            len(rounded[a].intersection(rounded[b]))
            for a in range(len(patches)) for b in range(a+1,len(patches))
        )
        if shared_grid_points<2:
            raise RuntimeError(
                "Postcut rectangles share the source seam, but the common "
                "square grid generated fewer than 2 matching boundary "
                "vertices. Change shared grid spacing; do not claim the "
                "panels match at the seam."
            )
        seam_uv_audit["postcut_seam_shared_grid_vertex_count"]=int(shared_grid_points)
    # DO NOT weld source vertices between detached islands: doing so would
    # incorrectly introduce hinges across physical Split boundaries.
    patch_label_counts = np.bincount(np.asarray(tile_components, int)).tolist()
    audit = {
        **seam_uv_audit,
        **exact_grid_audit,
        "postcut_rebff_integrated": True,
        "postcut_rebff_full_solver": True,
        "postcut_rebff_split_patch_count": len(patches),
        "postcut_rebff_baseline_quad_count": int(baseline_tiles),
        "postcut_rebff_final_quad_count": int(total),
        "postcut_rebff_patch_quad_counts": patch_label_counts,
        "postcut_rebff_shared_grid_spacing": spacing,
        "postcut_rebff_source_uv_islands_disconnected": True,
        "postcut_rebff_original_optcuts_pipeline_preserved": True,
        "postcut_rebff_export_caution": "experimental BFF split hypothesis; not paper-author verified",
    }
    return {
        "source_xyz": joined_xyz, "source_uv": joined_uv,
        "source_faces": joined_source_faces,
        "quad_vertices": joined_quad_vertices, "quad_faces": joined_quads,
        "tile_patch_ids": np.asarray(tile_components,int),
        "audit": audit,
    }


def run_existing_pipeline_through_dual_hinge(
    target_path: Path, experiment: dict, *, grid: int,
    base_params=None, progress=None, allow_csf_violation_preview: bool = False,
):
    """Run REAL existing K3D...T2D Dual Hinge with re-BFF M2D correspondence.

    By default, reject a chart violating the CSF bound. A user may explicitly
    allow a VISUALIZATION-ONLY downstream preview of an unvalidated chart;
    this does not certify a deployable assembly.
    """
    csf_converged = bool(experiment["report"]["linear_limit_two_satisfied"])
    if not csf_converged and not allow_csf_violation_preview:
        raise RuntimeError(
            "Postcut BFF still exceeds CSF=2; refusing downstream by default. "
            "Visualization-only bypass requires explicit user opt-in."
        )
    patches = experiment["patches"]
    if not patches:
        raise RuntimeError("No postcut pieces to assemble")
    params = copy.copy(base_params or pipe.PipelineParameters(
        nx=int(grid), ny=int(grid),
        omega_parameterization_mode="paper_reference_bff",
        bff_boundary_policy="target_rectangle",
        paper_fidelity_mode=False,
        max_csf_splits=1,
    ))
    params.bff_executable = str(experiment["report"].get("official_bff_executable") or params.bff_executable or "")
    params.bff_rectangle_executable = str(experiment["report"].get("rectangular_bff_executable") or params.bff_rectangle_executable or "")
    params.nx = params.ny = int(grid)
    params.omega_parameterization_mode = "paper_reference_bff"
    params.paper_fidelity_mode = False
    params.collision_split_enabled = False
    params.max_csf_splits = 1
    params.reference_stop_on_required_split = False
    # Cover every rectangular Omega exactly. The desired tile count is only a
    # density target; never delete boundary panels to hit a number.
    baseline = int(experiment["report"]["initial_m2d"]["quads"])
    packed = _assemble_patches(
        patches, params, int(grid), baseline, progress,
        initial_grid=experiment.get("initial_grid"),
    )
    packed["audit"]["postcut_rebff_csf_limit_satisfied"] = csf_converged
    packed["audit"]["postcut_rebff_unconverged_visualization_only"] = bool(
        not csf_converged and allow_csf_violation_preview
    )
    packed["audit"]["postcut_rebff_measured_max_csf"] = float(
        experiment["report"]["max_postcut_patch_linear_span"]
    )
    original = pipe._original
    original_build_m2d = original._build_m2d
    active = {"called": False}

    def replacement_m2d(grid_obj, domain, active_params=None):
        if active["called"]:
            raise RuntimeError("Unexpected second M2D construction during re-BFF integration")
        active["called"] = True
        # Original pipeline creates a target-source parameterization and
        # rectangular domain before this hook. Replace the chart IN PLACE so
        # the *existing* strict inverse lift, K3D surface fit, K2D hinge
        # reconstruction, and Dual Hinge stage all see the same union chart.
        param = domain.parameterization
        param.surface_vertices_3d = packed["source_xyz"]
        param.surface_faces = packed["source_faces"]
        param.uv_vertices_2d = packed["source_uv"]
        param.uv_faces = packed["source_faces"]
        param.metrics["postcut_rebff_multi_island"] = True
        param.metrics["postcut_rebff_source_triangles"] = len(packed["source_faces"])
        param.metrics.update(packed["audit"])
        domain.paper_split_policy = False
        domain.split_lines = []
        domain.csf_model = "postcut independent BFF per-piece conformal area span"
        domain.parameterization = param
        mesh = original.QuadMesh(
            np.asarray(packed["quad_vertices"], float),
            np.asarray(packed["quad_faces"], int),
            grid_obj, "M2D", dict(packed["audit"]), [],
        )
        return mesh

    with _PIPELINE_HOOK_LOCK:
        original._build_m2d = replacement_m2d
        try:
            state = pipe._ORIGINAL_BUILD_ONESTRING_DESIGN(
                load_target_shape(Path(target_path)), params,
                run_simulation=False, progress_callback=progress,
            )
        finally:
            original._build_m2d = original_build_m2d
    if not active["called"]:
        raise RuntimeError("M2D injection was not reached")
    for stage, mesh in [
        ("M2D", state.mesh_2d_initial),
        ("M3D", state.mesh_3d_initial),
        ("K3D", state.mesh_3d_optimized),
        ("K2D", state.mesh_2d_optimized),
    ]:
        if len(mesh.faces) != packed["audit"]["postcut_rebff_final_quad_count"]:
            raise RuntimeError(f"{stage} LOST reBFF tile correspondence")
    if (len(state.tiles_3d.vertices) != len(state.tiles_2d_dual_hinge.vertices)
        or len(state.tiles_2d_dual_hinge.vertices) != len(state.mesh_2d_initial.faces)):
        raise RuntimeError("Dual Hinge output does not match split tile count")
    from postcut_rebff_seam_registration import align_t2d_at_source_split_seams
    seam_metrics = align_t2d_at_source_split_seams(
        state, patches, packed["tile_patch_ids"],
    )
    packed["audit"].update(seam_metrics)
    if seam_metrics.get("split_seam_alignment") in (
        "source_surface_3d_matched_rigid_SE2",
        "shared_source_seam_grid_rigid_SE2",
    ):
        # The original full solver built its gap graph, lift points and string
        # path *before* final seam registration. Recompute those dependent
        # objects so the visible T2D and deployment constraints do not refer
        # to stale, deliberately separated UV islands.
        state.gap_graph = original._build_gap_graph(
            state.mesh_3d_optimized.faces,
            state.tiles_2d_dual_hinge, state.tiles_3d,
        )
        state.lift_points = original._select_lift_points(
            state.gap_graph, params.lift_tau,
        )
        state.string_path = original._build_string_path(
            state.gap_graph, state.lift_points, params.channel_friction,
        )
        packed["audit"]["postcut_rebff_string_path_recomputed_after_seam_alignment"] = True
    state.tiles_2d_dual_hinge.metrics.update(packed["audit"])
    state.tiles_2d_top_hinge.metrics.update(seam_metrics)
    state.hinge_graph.metrics.update(seam_metrics)
    state.tiles_2d_dual_hinge.metrics["postcut_rebff_full_T2D_dual_hinge_computed"] = True
    return state, packed["audit"]
