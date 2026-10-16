#!/usr/bin/env python3
"""Interactive, REAL post-cut official-BFF experiment (not a full T2D pipeline).

Run on Mac: streamlit run app_postcut_rebff.py
Original OneString design/split algorithms are left untouched.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from onestring_physics.input_shape import load_target_shape
from onestring_physics.onestring_pipeline import (
    PipelineParameters, build_paper_reference_initialization,
    create_quad_grid, _reference_flatten_to_domain,
    _build_reference_m2d, _lift_m2d_to_m3d,
)
from onestring_physics.legacy_pipeline_base import SurfaceParameterization
from onestring_physics.paper_fidelity_mode import configure_paper_fidelity
from onestring_physics.reference_bff import (
    run_official_bff, find_official_bff_executable,
    find_rectangular_bff_executable, ReferenceBFFUnavailableError,
)
from scripts.test_lilium_postcut_bff_hypothesis import field
from postcut_rebff_iterative import iterative_postcut_rebff


def _find_bff_in_other_mac_worktrees(*, rectangle: bool) -> Path | None:
    """Reuse separately built official BFF executables from another worktree.

    git worktree includes tracked sources but not ignored CMake build binaries.
    Avoid changing the existing research worktree or recompiling needlessly.
    """
    relative = (
        Path("third_party/bff_rectangle_build/bff-rect-command-line")
        if rectangle else
        Path("third_party/boundary-first-flattening/build/bff-command-line")
    )
    documents = Path.home() / "Documents"
    candidates = [
        ROOT / relative,
        documents / "OneString-large-steps-mesh-conditioning" /
        "onestring-physics-simulator" / relative,
    ]
    if documents.is_dir():
        # Look only one folder below Documents, not every file in home.
        candidates.extend(
            p / relative for p in documents.glob("*/onestring-physics-simulator")
            if p.is_dir()
        )
        candidates.extend(
            p / relative for p in documents.glob("onestring-physics-simulator")
            if p.is_dir()
        )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def _locate_bff_cli(explicit: str, *, rectangle: bool) -> Path:
    finder = (find_rectangular_bff_executable if rectangle
              else find_official_bff_executable)
    if explicit.strip():
        # Respect a manually entered path, including a misspelled one.
        return finder(explicit.strip())
    try:
        return finder()
    except ReferenceBFFUnavailableError as original_error:
        candidate = _find_bff_in_other_mac_worktrees(rectangle=rectangle)
        if candidate is not None:
            return finder(str(candidate))
        raise ReferenceBFFUnavailableError(
            str(original_error)
            + "\nChecked sibling OneString worktrees under ~/Documents for an "
              "existing official BFF executable; none was available. "
              "Build with: bash scripts/repair_postcut_bff_mac.sh"
        ) from original_error


def _jacobian(xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points3 = np.asarray(xyz, float)[faces]
    points2 = np.asarray(uv, float)[faces]
    a3, b3 = points3[:, 1] - points3[:, 0], points3[:, 2] - points3[:, 0]
    a2, b2 = points2[:, 1] - points2[:, 0], points2[:, 2] - points2[:, 0]
    j = np.linalg.norm(np.cross(a3, b3), axis=1) / np.maximum(
        np.abs(a2[:, 0] * b2[:, 1] - a2[:, 1] * b2[:, 0]), 1e-300
    )
    return np.sqrt(j)


def _figure(xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray, *,
            surface: bool, title: str) -> go.Figure:
    coords = np.asarray(xyz if surface else np.column_stack(
        (uv, np.zeros(len(uv)))), float)
    lam = _jacobian(xyz, uv, faces)
    fig = go.Figure(go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=np.log(np.maximum(lam, 1e-12)), intensitymode="cell",
        colorscale="Viridis", opacity=1, flatshading=True, showscale=True,
        colorbar=dict(title="log(CSF)", len=0.7),
        hovertemplate="log(CSF): %{intensity:.3f}<extra></extra>",
    ))
    camera = dict(eye=dict(x=0, y=0, z=2.4)) if not surface else None
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=48, b=0), height=440,
        scene=dict(aspectmode="data", camera=camera,
                   xaxis_title="u" if not surface else "X",
                   yaxis_title="v" if not surface else "Y",
                   zaxis_title="" if not surface else "Z"),
    )
    return fig



def _rebff_piece_m2d_m3d(patch: dict, params: PipelineParameters,
                         grid_resolution: int, spacing: float,
                         initial_grid=None) -> dict:
    """Actually retile one newly BFF-mapped source patch, then lift M2D to M3D.

    The two results are separate UV islands and are not wired into K3D/T2D.
    """
    from onestring_physics.reference_bff import validate_reference_mesh
    from copy import copy
    xyz, uv, faces = patch["xyz"], patch["new_uv"], patch["faces"]
    topo = validate_reference_mesh(xyz, faces)
    loop = [int(v) for v in topo["boundary_loops"][0]]
    chart = SurfaceParameterization(
        method="paper_reference_bff",
        surface_vertices_3d=xyz,
        surface_faces=faces,
        uv_vertices_2d=uv,
        uv_faces=faces,
        omega_boundary=uv[loop + [loop[0]]],
        metrics={"boundary_loop": loop,
                 "per_triangle_lambda": _jacobian(xyz, uv, faces).tolist()},
    )
    piece_params = copy(params)
    piece_params.reference_grid_spacing = float(spacing)
    piece_params.reference_grid_rotation_degrees = 0.0
    if initial_grid is not None:
        # The preview uses the SAME immutable global tile coordinates that
        # will feed K3D, not another per-piece regular-grid construction.
        from postcut_rebff_exact_rectangle_grid import grid_at_pitch
        tile=grid_at_pitch([patch],float(spacing),initial_grid=initial_grid)[0]
        piece_grid=create_quad_grid(
            int(grid_resolution),int(grid_resolution),
            piece_params.tile_size,piece_params.gap_size,
        )
        from onestring_physics.legacy_pipeline_base import QuadMesh
        m2d=QuadMesh(
            np.asarray(tile["vertices"],float),
            np.asarray(tile["faces"],int),
            piece_grid,"M2D",
            {"m2d_grid_origin_policy":"immutable_initial_omega"},[],
        )
    else:
        piece_params.reference_grid_origin_u = 0.0
        piece_params.reference_grid_origin_v = 0.0
        piece_grid = create_quad_grid(
            int(grid_resolution), int(grid_resolution),
            piece_params.tile_size, piece_params.gap_size,
        )
        domain = _reference_flatten_to_domain(chart, piece_grid, piece_params)
        m2d = _build_reference_m2d(piece_grid, domain, piece_params)
    m3d, _ = _lift_m2d_to_m3d(None, m2d, chart, piece_params)
    return {"uv": np.asarray(m2d.vertices, float)[:, :2],
            "xyz": np.asarray(m3d.vertices, float),
            "quads": np.asarray(m2d.faces, int),
            "roundtrip_max": float(m3d.metrics.get("m3d_round_trip_error_max", 0.0))}


def _quad_preview_obj(xyz: np.ndarray, uv: np.ndarray, quads: np.ndarray) -> bytes:
    lines = ["# Post-cut BFF quad preview with 3D vertices and new UV coordinates"]
    lines.extend(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}" for v in xyz)
    lines.extend(f"vt {v[0]:.17g} {v[1]:.17g}" for v in uv)
    for face in quads:
        ids = [int(i)+1 for i in face]
        lines.append("f " + " ".join(f"{i}/{i}" for i in ids))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _obj_bytes(xyz: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> bytes:
    out = ["# Post-cut BFF experiment. v = source 3D positions, vt = new BFF UV."]
    out.extend(f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}" for p in xyz)
    out.extend(f"vt {q[0]:.17g} {q[1]:.17g}" for q in uv)
    for a, b, c in np.asarray(faces, int):
        a, b, c = int(a) + 1, int(b) + 1, int(c) + 1
        out.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    return ("\n".join(out) + "\n").encode("utf-8")


def _download_zip(result: dict) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("comparison.json", json.dumps(result["report"], indent=2,
                                                        ensure_ascii=False))
        for patch in result["patches"]:
            name = patch["side"]
            archive.writestr(
                f"{name}_postcut_rebff.obj",
                _obj_bytes(patch["xyz"], patch["new_uv"], patch["faces"]),
            )
            archive.writestr(
                f"{name}_fixed_original_uv.obj",
                _obj_bytes(patch["xyz"], patch["original_uv"], patch["faces"]),
            )
            if "quad_preview" in patch:
                q = patch["quad_preview"]
                archive.writestr(
                    f"{name}_new_M3D_tiles.obj",
                    _quad_preview_obj(q["xyz"], q["uv"], q["quads"]),
                )
    return data.getvalue()


def _execute(mesh_path: Path, grid: int, standard_exe: str, rect_exe: str,
             phase: float, cut_mode: str, manual_axis: str,
             manual_fraction: float, post_policy: str,
             max_splits: int = 8, progress=None,
             initial_rectangle_mode: str = "source_boundary_shape_fit") -> dict:
    standard_path = str(_locate_bff_cli(standard_exe, rectangle=False))
    rectangle_path = str(_locate_bff_cli(rect_exe, rectangle=True))
    matched_free = post_policy == "automatic_reference_matched_seam"
    if matched_free and int(max_splits) != 1:
        raise ValueError(
            "Free-boundary shared-seam experiment currently supports exactly "
            "one Split. Later cuts would move previously registered seams."
        )
    effective_post_policy = (
        "automatic_reference" if matched_free else post_policy
    )

    params = PipelineParameters(
        nx=grid, ny=grid, omega_parameterization_mode="paper_reference_bff",
        bff_boundary_policy="target_rectangle", paper_fidelity_mode=True,
        max_csf_splits=1,
        bff_executable=standard_path, bff_rectangle_executable=rectangle_path,
    )
    params, _ = configure_paper_fidelity(params)
    params.bff_executable = standard_path
    params.bff_rectangle_executable = rectangle_path
    params.bff_rectangle_corner_phase_fraction = float(phase)
    params.bff_rectangle_selection_mode = initial_rectangle_mode
    params.max_csf_splits = 1
    initial = build_paper_reference_initialization(load_target_shape(mesh_path), params)
    param = initial.surface_parameterization
    xyz = np.asarray(param.surface_vertices_3d, float)
    uv = np.asarray(param.uv_vertices_2d, float)
    faces = np.asarray(param.surface_faces, int)
    initial_report = field(xyz, uv, faces)

    from onestring_physics.reference_bff import (
        validate_reference_mesh, validate_rectangular_boundary,
    )
    rect_loop = validate_reference_mesh(xyz, faces)["boundary_loops"][0]
    initial_rectangle_check = validate_rectangular_boundary(uv, rect_loop)
    initial_rect_metadata = {
        **dict(param.metrics),
        **initial_rectangle_check,
        "bff_boundary_policy_effective": "target_rectangle",
    }
    cut_records = initial.mesh_2d_initial.metrics.get("csf_split_step_analysis") or []
    needs_split = bool(initial_report["linear_span"] > 2.0 + 1e-10)
    first_cut = None
    axis = coordinate = source_vertex = curvature = None
    if needs_split and cut_mode == "曲率優先の最初のSplit":
        if cut_records:
            first = cut_records[0]
            axis = str(first["axis"])
            coordinate = float(first["snapped_value"])
            source_vertex = first.get("curvature_vertex_id")
            curvature = first.get("gaussian_curvature")
            first_cut = dict(
                axis=axis, coordinate=coordinate, origin=cut_mode,
                source_vertex=source_vertex, curvature=curvature,
            )
        # If the original M2D split analysis has no candidate, the iterative
        # routine searches the ACTUAL current grid, never an arbitrary center cut.
    elif needs_split:
        axis = manual_axis
        dim = {"row": 1, "col": 0}[axis]
        requested = float(np.min(uv[:, dim]) +
                          manual_fraction * np.ptp(uv[:, dim]))
        # Snap manual requests to the same reference-grid lattice as M2D.
        reference_domain = initial.conformal_domain
        spacing = float(reference_domain.reference_grid_spacing)
        origin = np.asarray(reference_domain.reference_grid_origin, float)
        coordinate = float(origin[dim] + round((requested-origin[dim])/spacing)*spacing)
        if not (float(np.min(uv[:, dim])) + 1e-9 < coordinate
                < float(np.max(uv[:, dim])) - 1e-9):
            raise ValueError(
                "Manual Split snapped outside the interior of Omega; "
                "adjust its position along the reference grid."
            )
        first_cut = dict(
            axis=axis, coordinate=coordinate,
            origin="manual_grid_snapped", source_vertex=None, curvature=None,
        )

    def _rebff(x, f, *, rectangle_corner_positions=None):
        if post_policy == "target_rectangle" and rectangle_corner_positions is None:
            raise RuntimeError("Split BFF rectangle requires exact physical seam endpoints")
        # Both ends of the physical cut are common 3D vertices of the child
        # patches. Fix them as neighboring rectangle corners in BOTH BFFs.
        bff = run_official_bff(
            x, f, executable=standard_path,
            rectangle_executable=rectangle_path,
            rectangle_corner_positions=(
                np.asarray(rectangle_corner_positions,float)
                if post_policy == "target_rectangle" else None
            ),
            boundary_policy=effective_post_policy,
        )
        if post_policy == "target_rectangle":
            if (bff.metrics.get("bff_boundary_policy_effective") != "target_rectangle"
                    or bff.metrics.get("bff_rectangle_verified") is not True):
                raise RuntimeError(
                    "Split child did not return geometrically verified rectangular BFF; "
                    "refusing fallback to free-boundary parameterization."
                )
        return np.asarray(bff.uv_vertices, float), bff.metrics

    # Enforce a common physical seam edge only for rectangular BFF.
    # Free-boundary comparison intentionally retains its original behavior.
    _rebff.seam_corners_required = (post_policy == "target_rectangle")
    _rebff.match_free_seam = bool(matched_free)
    # A recursive Split must stay on the same GLOBAL Omega lattice. Deriving
    # a new pitch from each smaller child's bbox would manufacture cuts inside
    # cells already proven indivisible by the original grid.
    if post_policy == "target_rectangle":
        lattice = initial.conformal_domain
        spacing = float(lattice.reference_grid_spacing)
        origin = np.asarray(lattice.reference_grid_origin,float).reshape(2)
        if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(origin).all():
            raise RuntimeError("Original M2D lattice is invalid; cannot continue Split")
        _rebff.fixed_grid_spacing = spacing
        _rebff.fixed_grid_origin = origin

    # Establish immutable M2D coordinates BEFORE the first BFF Split.
    # Subsequent BFF calls may change chart interiors, never panel layout.
    initial_grid = None
    if post_policy == "target_rectangle":
        from postcut_rebff_exact_rectangle_grid import build_initial_omega_grid
        initial_grid = build_initial_omega_grid(
            {"xyz":xyz,"new_uv":uv,"faces":faces},
            spacing,origin,
        )

    trial = iterative_postcut_rebff(
        xyz, uv, faces, grid=int(grid), rebff=_rebff, max_splits=int(max_splits),
        csf_limit=2.0, progress=progress,
        first_cut=first_cut,
        initial_linear_span=float(initial_report["linear_span"]),
        initial_bff_metadata=initial_rect_metadata,
    )
    patches = trial["patches"]
    # Preview the ORIGINAL Omega grid partitioned by Split. No new tile
    # density, origins or boundary line insertion is allowed after re-BFF.
    maximum_span = max(float(np.max(np.ptp(p["new_uv"], axis=0))) for p in patches)
    common_spacing = (
        float(initial_grid["pitch"]) if initial_grid is not None
        else max(maximum_span / max(int(grid), 1), 1e-8)
    )
    for patch in patches:
        try:
            patch["quad_preview"] = _rebff_piece_m2d_m3d(
                patch, params, int(grid), common_spacing,
                initial_grid=initial_grid,
            )
        except Exception as exc:
            patch["quad_preview_error"] = f"{type(exc).__name__}: {exc}"

    worst = max(p["post"]["linear_span"] for p in patches)
    report = {
        "experiment": (
            "Initial rectangular BFF -> one full-grid Split -> free-boundary "
            "official BFF on both pieces -> exact shared seam Dirichlet warp "
            "-> recompute actual UV CSF"
            if matched_free else
            "Initial rectangular BFF -> grid Split -> independent child BFF "
            "-> seam-constrained shared atlas -> measure atlas CSF and repeat Split"
        ),
        "not_paper_author_verified": True,
        "not_applied_to_K3D_or_T2D": True,
        "input_mesh": str(mesh_path), "grid": int(grid),
        "postcut_free_shared_seam_mode":bool(matched_free),
        "postcut_free_shared_seam_diagnostics":[
            p.get("postcut_free_seam_audit") for p in patches
        ] if matched_free else None,
        "postcut_free_shared_seam_uv_verified":bool(
            matched_free and len(patches)==2
            and all(p.get("postcut_free_seam_audit",{}).get("seam_uv_error_max",float("inf"))<1e-9
                    for p in patches)
        ),
        "postcut_free_shared_seam_not_original_grid_compatible":bool(matched_free),
        "official_bff_executable": standard_path,
        "rectangular_bff_executable": rectangle_path,
        "initial_bff": "target_rectangle", "postcut_bff": post_policy,
        "initial_rectangle_selection_mode": initial_rectangle_mode,
        "initial_rectangle_selection_method": param.metrics.get("bff_corner_selection_method"),
        "initial_rectangle_selected_score": param.metrics.get("bff_corner_selection_selected_score"),
        "initial_rectangle_selected_corner_vertex_ids": param.metrics.get("bff_corner_selection_selected_source_vertex_ids"),
        "initial_rectangle_selected_corner_xyz": param.metrics.get("bff_corner_selection_selected_source_xyz"),
        "initial_rectangle_selection_attempts": param.metrics.get("bff_corner_selection_attempts"),
        "initial_bff_linear_csf": float(initial_report["linear_span"]),
        "initial_bff_satisfies_csf_limit": not needs_split,
        "initial_bff_rectangle_verified": True,
        "all_postcut_patches_rectangle_verified": all(
            p.get("bff_metadata", {}).get("bff_rectangle_verified") is True
            for p in patches
        ) if post_policy == "target_rectangle" else None,
        "postcut_seam_physical_corner_constraints_enforced": bool(
            post_policy == "target_rectangle" and all(
                p.get("seam_rectangle_corners_constrained",False)
                for p in patches
            )
        ),
        "postcut_joint_atlas_active": bool(
            post_policy == "target_rectangle"
            and all(p.get("postcut_joint_atlas",False) for p in patches)
            and len(patches)>1
        ),
        "postcut_joint_atlas_note": (
            "Physical source seam vertices inherited from parent are fixed. "
            "The independently computed rectangular BFF supplies an interior "
            "target only. The final joined atlas is NOT itself an independent "
            "conformal BFF; the CSF gate is recomputed on the atlas."
            if post_policy == "target_rectangle" else None
        ),
        "rectangle_corner_phase": float(phase),
        "split_grid_policy": "fixed_initial_rectangle_lattice" if post_policy == "target_rectangle" else "per_child_preview_lattice",
        "m2d_grid_policy": ("immutable_initial_omega_lines"
                            if initial_grid is not None else "legacy_retiling"),
        "m2d_initial_grid_x": (initial_grid["x"].tolist()
                              if initial_grid is not None else None),
        "m2d_initial_grid_y": (initial_grid["y"].tolist()
                              if initial_grid is not None else None),
        "split_grid_pitch": getattr(_rebff,"fixed_grid_spacing",None),
        "split_grid_origin": (
            _rebff.fixed_grid_origin.tolist()
            if hasattr(_rebff,"fixed_grid_origin") else None
        ),
        "split_count": trial["split_count"],
        "split_history": trial["split_history"],
        "rejected_candidates": trial["rejected_candidates"],
        "unsplittable_regions": trial["unsplittable_regions"],
        "unsplittable_region_count": trial["unsplittable_region_count"],
        "violating_patch_count": trial["violating_patch_count"],
        "stop_reason": trial["stop_reason"],
        "csf_limit": trial["csf_limit"],
        "max_splits_budget": trial["max_splits"],
        "split": {
            "axis": axis, "uv_coordinate": coordinate,
            "origin": (cut_mode if first_cut is not None else "no_split_required"),
            "source_curvature_vertex": source_vertex, "gaussian_curvature": curvature,
        },
        "initial_source_triangle_csf": initial_report,
        "initial_m2d": {
            "quads": int(len(initial.mesh_2d_initial.faces)),
            "initial_linear_csf": initial.mesh_2d_initial.metrics.get("max_csf_before_split"),
            "after_single_cut_linear_csf": initial.mesh_2d_initial.metrics.get("max_csf_after_split"),
        },
        "patches": [
            {
                "side": p["side"], "triangles": int(len(p["faces"])),
                "fixed_original_chart": p["pre"],
                "postcut_rebff": p["post"],
                "independent_rectangular_bff_csf":p.get("independent_bff_post"),
                "postcut_joint_atlas":p.get("postcut_joint_atlas_metadata"),
            } for p in patches
        ],
        "new_piece_m2d_quad_counts": [
            int(len(p["quad_preview"]["quads"])) if "quad_preview" in p else None
            for p in patches
        ],
        "new_piece_m2d_grid_spacing": float(common_spacing),
        "new_piece_m2d_errors": [p.get("quad_preview_error") for p in patches],
        "max_postcut_patch_linear_span": float(worst),
        "max_independent_rectangular_bff_linear_span": max(
            float(p.get("independent_bff_post",p["post"])["linear_span"])
            for p in patches
        ),
        "postcut_csf_measurement_chart":"actual_shared_atlas" if (
            post_policy=="target_rectangle" and len(patches)>1
        ) else "independent_BFF",
        "linear_limit_two_satisfied": trial["all_pass"],
        "caution": "Intermediate per-island M2D/M3D previews are not the full "
                   "design. Full-solver K3D/T3D/K2D/T2D is available separately; "
                   "its fabrication quality is not certified. Joint-atlas fixed seam "
                   "boundary is a nonconformal experimental coupling, NOT "
                   "an author-verified independent BFF result.",
    }
    return {"report": report, "patches": patches,
            "initial_grid": initial_grid,
            "initial": {"xyz": xyz, "uv": uv, "faces": faces}}


def _mesh_top_stage_figure(vertices, faces, *, title: str):
    """Visualize REAL mesh vertices/faces returned by the original solver."""
    xyz = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    tris = np.concatenate([f[:, [0, 1, 2]], f[:, [0, 2, 3]]])
    fig = go.Figure(go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
        color="#3995aa", opacity=0.9, flatshading=True,
    ))
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      height=480, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _real_tile_stage_figure(tiles: np.ndarray, title: str, flat: bool = False,
                            hinge_graph=None):
    """Draw the ACTUAL eight-vertex T3D/T2D polygons and joint locations."""
    verts = np.asarray(tiles, float).copy()
    if verts.ndim != 3 or verts.shape[1:] != (8, 3):
        raise ValueError(f"Expected REAL (tiles, 8, 3), got {verts.shape}")
    if flat:
        verts[:, :, 2] = 0.
    all_vertices = verts.reshape(-1, 3)
    local_tris = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], int)
    triangles = np.concatenate([
        local_tris + 8*i for i in range(len(verts))
    ])
    fig = go.Figure(go.Mesh3d(
        x=all_vertices[:, 0], y=all_vertices[:, 1], z=all_vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color="#41a0b3", opacity=0.35, flatshading=True,
        hoverinfo="skip", showscale=False,
    ))
    lines = []
    for tile in verts:
        for indices in ([0, 1, 2, 3, 0], [4, 5, 6, 7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]):
            lines.extend(tile[indices].tolist())
            lines.append([None, None, None])
    seq = np.asarray(lines, dtype=object)
    fig.add_trace(go.Scatter3d(
        x=seq[:, 0], y=seq[:, 1], z=seq[:, 2],
        mode="lines", line=dict(color="#224c68", width=2),
        showlegend=False, hoverinfo="skip",
    ))
    if hinge_graph is not None:
        points = []
        for hinge in hinge_graph.hinges:
            xy = np.asarray(hinge.rest_position_2d, float).reshape(-1)
            if len(xy) >= 2 and np.all(np.isfinite(xy[:2])):
                points.append([xy[0], xy[1], 0.])
        if points:
            positions = np.asarray(points, float)
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode="markers", marker=dict(size=2.4, color="#d35f70"),
                name="Dual Hinge joints",
            ))
    fig.update_layout(title=title, height=580,
                      scene=dict(aspectmode="data",
                                 camera=dict(eye=dict(x=0, y=0, z=2.6)) if flat else None),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _tile_stage_obj_bytes(tiles: np.ndarray, label: str) -> bytes:
    geom = np.asarray(tiles, float)
    if geom.ndim != 3 or geom.shape[1:] != (8, 3):
        raise ValueError(f"Unexpected {label} stage geometry: {geom.shape}")
    lines = [f"# Actual OneString stage: {label}"]
    for tile in geom:
        for v in tile:
            lines.append("v " + " ".join(f"{float(x):.17g}" for x in v))
    for i in range(len(geom)):
        offset = 8*i
        for ids in ([1, 2, 3, 4], [8, 7, 6, 5],
                    [1, 5, 6, 2], [2, 6, 7, 3],
                    [3, 7, 8, 4], [4, 8, 5, 1]):
            lines.append("f " + " ".join(str(offset+j) for j in ids))
    return ("\n".join(lines)+"\n").encode("utf-8")


def _export_real_pipeline_zip(state, audit):
    data = io.BytesIO()
    stages = {
        "T3D": state.tiles_3d.vertices,
        "T2D_Top_Hinge": state.tiles_2d_top_hinge.vertices,
        "T2D_Dual_Hinge": state.tiles_2d_dual_hinge.vertices,
    }
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, tiles in stages.items():
            archive.writestr(
                name + ".obj", _tile_stage_obj_bytes(tiles, name)
            )
        archive.writestr("full_pipeline_audit.json",
                         json.dumps(audit, indent=2, ensure_ascii=False))
    return data.getvalue()


def _show_existing_pipeline_state(state, audit):
    if audit.get("postcut_rebff_unconverged_visualization_only", False):
        st.error("これはCSF上限2を満たさない矩形BFFからの試験的なT2D表示です。"
                 "製造可能な結果として使用しないでください。")
    else:
        st.success("既存のOneStringソルバーでT2D Dual Hingeまで計算が完了しました。")
    st.json({
        "M2D/M3D shared tile count": len(state.mesh_2d_initial.faces),
        "K3D tile count": len(state.mesh_3d_optimized.faces),
        "T3D tile count": len(state.tiles_3d.vertices),
        "K2D tile count": len(state.mesh_2d_optimized.faces),
        "T2D Dual Hinge tile count": len(state.tiles_2d_dual_hinge.vertices),
        "Dual Hinge count": len(state.hinge_graph.hinges),
        **audit,
    })
    st.subheader("既存の最適化器が出力した全段階の形状")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_mesh_top_stage_figure(
            state.mesh_3d_optimized.vertices, state.mesh_3d_optimized.faces,
            title="K3D：平面性・四角形性・曲面追従最適化後",
        ), use_container_width=True)
    with b:
        st.plotly_chart(_real_tile_stage_figure(
            state.tiles_3d.vertices, title="T3D：実タイル（上面輪郭）",
        ), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_real_tile_stage_figure(
            state.tiles_2d_top_hinge.vertices,
            title="T2D Top Hinge：実計算結果", flat=True,
        ), use_container_width=True)
    with b:
        st.plotly_chart(_real_tile_stage_figure(
            state.tiles_2d_dual_hinge.vertices,
            title="T2D Dual Hinge：実計算結果", flat=True,
            hinge_graph=state.hinge_graph,
        ), use_container_width=True)
    st.download_button(
        "実計算したT3D・T2D Top Hinge・T2D Dual HingeのOBJをZIP保存",
        data=_export_real_pipeline_zip(state, audit),
        file_name="onestring_rebff_full_T2D_dual_hinge.zip",
        mime="application/zip",
    )
    with st.expander("Dual Hingeソルバーの評価値"):
        st.json(state.tiles_2d_dual_hinge.metrics)


def main() -> None:
    st.set_page_config(page_title="OneString Split → re-BFF", layout="wide")
    st.title("OneString: Split後のBFF再計算（Mac実験モード）")
    st.caption("まず初期矩形BFFのCSFを評価 → 2を超えた領域だけGridに沿ってSplit → 各子領域で矩形BFFを再計算")
    st.warning(
        "研究用の独立した実験画面です。新しいUVとOBJを実際に生成しますが、"
        "領域別プレビューはK3D/T2Dではありません。下の既存ソルバー連携を実行すると実際のDual Hinge結果を計算します。"
        "通常のシミュレーターの計算経路は変更しません。"
    )
    with st.sidebar:
        st.header("実験条件")
        mesh = st.text_input("OBJのパス", value="~/Downloads/lilium.obj")
        grid = st.number_input("グリッド（nx=ny）", min_value=2, max_value=60, value=14)
        rect = st.text_input("矩形BFF実行ファイル（空欄なら自動検出）", value="")
        standard = st.text_input("通常BFF実行ファイル（空欄なら自動検出）", value="")
        initial_rectangle_mode = st.selectbox(
            "初期Ωの矩形BFF四隅の選択",
            ["source_boundary_shape_fit", "fixed_phase"],
            index=0,
            format_func=lambda mode: (
                "入力メッシュSの境界に合わせて自動探索"
                if mode == "source_boundary_shape_fit" else
                "境界周長の4等分（従来方式）"
            ),
            help="自動モードでは元Sの3D境界をPCAで評価し、"
                 "複数の四隅候補で公式矩形BFFを実行、境界適合度とCSFを比較します。",
        )
        phase = st.select_slider(
            "矩形BFF四隅の位相（従来の初期Ωのみ）",
            options=[0.0, 0.0625, 0.125, 0.1875], value=0.0,
        )
        if initial_rectangle_mode == "source_boundary_shape_fit":
            st.caption("初期Ωは入力境界への適合度で自動選択します。"
                       "Split後の矩形BFFでは分割境界の両端を必ず矩形角に固定します。")
        cut_mode = st.radio(
            "切断位置", ["曲率優先の最初のSplit", "手動指定"], index=0,
        )
        axis = "row"
        fraction = 0.5
        if cut_mode == "手動指定":
            axis = st.selectbox("切断方向", ["row", "col"])
            fraction = st.slider("UV境界幅に対する切断位置", 0.05, 0.95, 0.5, 0.01)
        # Keep a free-boundary comparison only as an explicit optional mode;
        # never choose it silently when the user wants rectangles.
        post_policy = st.selectbox(
            "Split後のBFF境界条件",
            ["target_rectangle", "automatic_reference",
             "automatic_reference_matched_seam"],
            index=0,
            format_func=lambda value: {
                "target_rectangle": "矩形境界（初期BFFと同じ・推奨）",
                "automatic_reference": "自由境界（境界一致なし・比較実験）",
                "automatic_reference_matched_seam":
                    "自由境界＋物理Split境界のUVを完全一致（1本・実験）",
            }[value],
            help="自由境界＋境界一致は、公式BFF後に同じ物理切断点を"
                 "同一UV座標へ固定して内部を調和補正します。補正後CSFを再評価し、"
                 "UVの反転を禁止します。元のGridやT2Dの製造適合性は保証しません。",
        )
        st.caption("初回BFFだけでCSF≤2ならSplitは0本。"
                   "再BFF後も違反があればGridに沿って追加Splitします。"
                   "矩形境界モードでは子領域も幾何学的に検証します。")
        max_splits = int(st.number_input(
            "最大Split数（初回を含む）", min_value=1,
            max_value=(1 if post_policy=="automatic_reference_matched_seam" else 64),
            value=(1 if post_policy=="automatic_reference_matched_seam" else 8),
            step=1,
            help="CSF>2の領域だけを切断。自由境界の境界一致実験は"
                 "既存の共有境界を壊さないよう1本に制限しています。",
        ))
        st.caption("CSF判定は全領域の sqrt(max(J)/min(J)) ≤ 2（J=3D/2D面積倍率）")
        start = st.button("Split → BFF再計算を収束まで繰り返す", type="primary")

    if start:
        # Do not show a stale run as success when the new grid/BFF attempt fails.
        st.session_state.pop("postcut_rebff_data", None)
        st.session_state.pop("postcut_rebff_full_pipeline_state", None)
        st.session_state.pop("postcut_rebff_full_pipeline_audit", None)
        path = Path(mesh).expanduser().resolve()
        if not path.is_file():
            st.error(f"OBJが存在しません: {path}")
        else:
            try:
                live = st.empty()
                with st.spinner("各Split後にBFFを再計算し、CSFを再判定中…"):
                    st.session_state["postcut_rebff_data"] = _execute(
                        path, int(grid), standard, rect, float(phase),
                        cut_mode, axis, float(fraction), post_policy,
                        max_splits=max_splits,
                        progress=lambda message: live.info(message),
                        initial_rectangle_mode=initial_rectangle_mode,
                    )
                live.empty()
            except ReferenceBFFUnavailableError as exc:
                st.error("公式BFF CLIが見つからないか、起動に失敗しました。"
                         "OptCutsではなく、実験用の公式BFF実行ファイルが必要です。")
                st.code("bash scripts/repair_postcut_bff_mac.sh", language="bash")
                st.caption("ターミナルで上のコマンドを実行した後、実験ページを再実行してください。"
                           "既存のOneString・OptCutsのコードは変更しません。")
                st.exception(exc)
            except Exception as exc:
                st.exception(exc)
    data = st.session_state.get("postcut_rebff_data")
    if data is None:
        st.info(
            "公式BFFが未セットアップの場合はターミナルで "
            "bash scripts/repair_postcut_bff_mac.sh を実行してください。"
            "その後、実行ボタンを押します。"
        )
        return

    report = data["report"]
    passed = bool(report["linear_limit_two_satisfied"])
    summary = (
        f"Split={report['split_count']}本（上限{report['max_splits_budget']}本）、"
        f"領域={len(data['patches'])}、"
        f"最大CSF比={report['max_postcut_patch_linear_span']:.4f}、"
        f"停止理由={report['stop_reason']}"
    )
    st.caption(f"初期BFF: {report['initial_bff']} ｜ Split後BFF: {report['postcut_bff']} ｜ "
               f"全子領域の矩形検証: {report.get('all_postcut_patches_rectangle_verified')}")
    st.caption(f"初期Ω四隅選択: {report['initial_rectangle_selection_mode']} ｜ "
               f"選択スコア: {report.get('initial_rectangle_selected_score')}")
    if report.get("initial_rectangle_selection_attempts"):
        with st.expander("入力S境界に合わせた初期矩形の候補と比較結果"):
            st.json(report["initial_rectangle_selection_attempts"])
    if report.get("postcut_free_shared_seam_mode"):
        st.info(
            "自由境界BFF後に、同じ3D切断点が完全に一致するようUVを調和補正しました。"
            "以下は補正後のCSFです。Gridの完全保持やT2Dの境界一致は未検証です。"
        )
        st.json(report.get("postcut_free_shared_seam_diagnostics"))
    if report.get("postcut_free_shared_seam_mode") and len(data["patches"])==2:
        from onestring_physics.reference_bff import validate_reference_mesh
        from postcut_free_bff_matched_seam import _physical_seam_samples
        import numpy as np
        split_spec=report.get("split",{})
        if split_spec.get("axis") in ("row","col"):
            ids,_=_physical_seam_samples(
                data["patches"],split_spec["axis"],
                float(split_spec["uv_coordinate"]),
            )
            seam_fig=go.Figure()
            colors=("#287bb7","#d58130")
            for index,(patch,indices) in enumerate(zip(data["patches"],ids)):
                xyz=np.asarray(patch["xyz"],float)
                uv=np.asarray(patch["new_uv"],float)
                boundary=np.asarray(
                    validate_reference_mesh(
                        xyz,np.asarray(patch["faces"],int)
                    )["boundary_loops"][0],int
                )
                outline=uv[np.r_[boundary,boundary[:1]]]
                seam=uv[indices]
                seam_fig.add_trace(go.Scatter(
                    x=outline[:,0],y=outline[:,1],mode="lines",
                    line=dict(color=colors[index],width=2),
                    name=f"領域{index+1} 外周",
                ))
                seam_fig.add_trace(go.Scatter(
                    x=seam[:,0],y=seam[:,1],mode="lines+markers",
                    line=dict(color=colors[index],width=4),
                    marker=dict(size=5),
                    name=f"領域{index+1} 共有境界",
                ))
            seam_fig.update_layout(
                title="共有する物理Split境界の2D一致（2色が重なれば一致）",
                xaxis=dict(title="u",scaleanchor="y",scaleratio=1),
                yaxis=dict(title="v"),
                height=510,
            )
            st.plotly_chart(seam_fig,use_container_width=True)
            st.caption(
                "境界線のUV一致は検証済みですが、これだけではGrid・K3D・"
                "T2Dのパネル境界の一致や製造可能性は保証できません。"
            )
    if passed:
        st.success("全領域のCSFが2以下になりました。" + summary)
    else:
        st.error("全領域のCSFを2以下にできませんでした。" + summary)
    left, middle, right = st.columns(3)
    left.metric("元BFF（元3Dメッシュ）CSF比",
                f"{report['initial_source_triangle_csf']['linear_span']:.4f}")
    middle.metric("最終の最大CSF比",
                  f"{report['max_postcut_patch_linear_span']:.4f}")
    right.metric("Split本数", str(report["split_count"]))
    st.subheader("初期Ωの自動矩形選択：入力Sと矩形UV")
    st.caption("初期BFFの結果です。オレンジ色の点はSの境界から選択した四隅です。"
               "Split後の子領域の矩形BFFとは別の計算です。")
    initial_data = data["initial"]
    source_xyz = np.asarray(initial_data["xyz"], float)
    source_uv = np.asarray(initial_data["uv"], float)
    source_faces = np.asarray(initial_data["faces"], int)
    source_fig = _figure(
        source_xyz, source_uv, source_faces,
        surface=True, title="入力S：初期BFFで使用したメッシュ",
    )
    uv_fig = _figure(
        source_xyz, source_uv, source_faces,
        surface=False, title="選択された初期Ω：矩形BFF",
    )
    chosen_corners = report.get("initial_rectangle_selected_corner_vertex_ids")
    if chosen_corners:
        ids = np.asarray(chosen_corners, int)
        if np.all((ids >= 0) & (ids < len(source_xyz))):
            pos = source_xyz[ids]
            uvpos = source_uv[ids]
            source_fig.add_trace(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode="markers+text", text=["A", "B", "C", "D"],
                textposition="top center",
                marker=dict(size=5, color="#f3982c"),
                name="選択した矩形の四隅",
            ))
            uv_fig.add_trace(go.Scatter3d(
                x=uvpos[:, 0], y=uvpos[:, 1],
                z=np.zeros(len(ids)), mode="markers+text",
                text=["A", "B", "C", "D"], textposition="top center",
                marker=dict(size=6, color="#f3982c"),
                name="矩形の四隅",
            ))
    left_fig, right_fig = st.columns(2)
    with left_fig:
        st.plotly_chart(source_fig, use_container_width=True)
    with right_fig:
        st.plotly_chart(uv_fig, use_container_width=True)

    st.subheader("Split履歴")
    for item in report["split_history"]:
        st.write(
            f"**#{item['step']}**　{item['parent']} → {item['axis']}="
            f"{item['uv_coordinate']:.6f}　|　"
            f"CSF={item['all_patches_max_csf_before']:.4f} → "
            f"{item['all_patches_max_linear_span']:.4f}　|　"
            f"改善={'あり' if item['csf_improved'] else 'なし'}　|　"
            f"CSF>2の領域={item['violating_patch_count']}"
        )
    if report["rejected_candidates"]:
        with st.expander("採用できなかった切断候補"):
            st.json(report["rejected_candidates"])

    st.divider()
    st.subheader("従来OneStringに接続してT2D Dual Hingeまで実計算")
    st.caption("従来のOptCuts対応OneStringパイプラインのK3D・T3D・K2D・"
               "T2D Top Hinge・Dual Hinge最適化器をそのまま使用します。"
               "再BFF後は全領域のタイルを対応付きで統合し、元M2Dに近い"
               "タイル数になる共通格子間隔を探索します。"
               "K3D以降の制約違反や最適化失敗は隠さず表示します。")
    opt_in_preview = False
    if not report["linear_limit_two_satisfied"]:
        st.warning(
            "矩形境界BFFでCSFが2以下になっていません。"
            "この状態のT2Dは製造可能な設計とは判定できません。"
        )
        opt_in_preview = st.checkbox(
            "CSF未達でも、実験用としてT2D Dual Hingeの計算を試す"
            "（品質・製造可能性は保証しない）",
            value=False,
        )
    if (report["linear_limit_two_satisfied"] or opt_in_preview) and st.button(
        "従来の全段階を計算してT2D Dual Hingeを表示", type="primary"
    ):
        try:
            from postcut_rebff_full_pipeline import run_existing_pipeline_through_dual_hinge
            live_progress = st.empty()
            def on_full_progress(stage, value, description=""):
                live_progress.info(
                    f"{stage}（{float(value)*100:.1f}%）: {description}"
                )
            with st.spinner("既存のOneStringでK3D→T3D→K2D→T2D Dual Hingeを計算中…"):
                state, audit = run_existing_pipeline_through_dual_hinge(
                    Path(mesh).expanduser().resolve(), data,
                    grid=int(grid), progress=on_full_progress,
                    allow_csf_violation_preview=bool(opt_in_preview),
                )
                st.session_state["postcut_rebff_full_pipeline_state"] = state
                st.session_state["postcut_rebff_full_pipeline_audit"] = audit
            live_progress.empty()
        except Exception as exc:
            st.error("元のOneStringソルバーが停止しました。"
                     "T2Dを偽の結果で代替していません。")
            st.exception(exc)
    final_state = st.session_state.get("postcut_rebff_full_pipeline_state")
    final_audit = st.session_state.get("postcut_rebff_full_pipeline_audit")
    if final_state is not None and final_audit is not None:
        _show_existing_pipeline_state(final_state, final_audit)

    st.subheader("2D展開の比較")
    st.caption(
        "各パッチの位置・回転・拡大率は独立です。左＝切断前のUVを保持、"
        "右＝実際に再計算されたBFFのUV。色は各三角形のlog(CSF)。"
    )
    for patch in data["patches"]:
        st.markdown(f"**{patch['side']} 領域（{len(patch['faces'])} triangles）**")
        m1, m2 = st.columns(2)
        m1.metric("既存UVのCSF比", f"{patch['pre']['linear_span']:.4f}")
        m2.metric("再BFFのCSF比", f"{patch['post']['linear_span']:.4f}")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_figure(
                patch["xyz"], patch["original_uv"], patch["faces"],
                surface=False, title="切断前のUVを維持",
            ), use_container_width=True)
        with right:
            st.plotly_chart(_figure(
                patch["xyz"], patch["new_uv"], patch["faces"],
                surface=False, title="Split後にBFFを再計算",
            ), use_container_width=True)
    st.subheader("切断後の3D形状")
    for patch in data["patches"]:
        with st.expander(f"{patch['side']}（{len(patch['faces'])} triangles）"):
            st.plotly_chart(_figure(
                patch["xyz"], patch["new_uv"], patch["faces"],
                surface=True, title=f"{patch['side']}：最終3D形状",
            ), use_container_width=True)

    st.subheader("中間検証のみ：再BFF後の各領域のM2D / M3D")
    st.warning("この領域別プレビューは最終設計ではありません。"
               "最終タイル配置とT2D Dual Hingeは上の「従来の全段階を計算」から確認してください。")
    st.caption("各領域に共通間隔の正方形タイルを作り、新しいBFFの逆写像で"
               "3Dへリフトしました。ここでは中間比較のみを表示します。最終段階は既存ソルバーのDual Hinge結果を参照してください。")
    for patch in data["patches"]:
        if "quad_preview" not in patch:
            st.error(f"{patch['side']}: M2D→M3D生成失敗: {patch.get('quad_preview_error')}")
            continue
        q = patch["quad_preview"]
        st.markdown(f"**{patch['side']}：{len(q['quads'])}枚の新M2D/M3Dタイル**")
        triangles = np.concatenate((
            q["quads"][:, [0, 1, 2]], q["quads"][:, [0, 2, 3]],
        ), axis=0)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_figure(
                q["xyz"], q["uv"], triangles,
                surface=False, title="再BFF後に生成したM2D",
            ), use_container_width=True)
        with right:
            st.plotly_chart(_figure(
                q["xyz"], q["uv"], triangles,
                surface=True, title="再BFFからリフトしたM3D",
            ), use_container_width=True)

    st.download_button(
        "最終全領域のBFF UV・M3D四角形・Split履歴をZIP保存",
        data=_download_zip(data), file_name="onestring_postcut_rebff.zip",
        mime="application/zip",
    )
    with st.expander("詳細CSFと実験条件"):
        st.json(report)


if __name__ == "__main__":
    main()
