"""2026-09-14 OptCuts variant with an LSCM-equivalent K2D stage.

The dated OptCuts mode may differ from LSCM before K2D (Omega, seams, M2D
geometry/topology, and K3D), but the K2D numerical solver is the same common
OneString solver used by LSCM.

A second equivalence rule is required after shared-vertex K2D: OptCuts seams may
produce several disconnected fabrication panels.  The ordinary LSCM flat-linkage
builder is therefore applied independently to each edge-connected K2D component,
then the component layouts are merged back in the original face order.  No hinge
is ever created across an OptCuts seam, and disconnected components do not enter
one global SE(2) solve together.

Crucially, component submeshes retain the original K2D vertex array and original
face vertex IDs.  The LSCM flat-layout code infers the regular-grid row/column
parity from those IDs to choose pairwise hinge corners; compactly renumbering a
component would therefore change the LSCM hinge rule and is intentionally not
done here.

OptCuts-specific rigid-K2D replacement, hard-SAT K2D feasibility, all-tile SE(2)
K2D solves, and post-K2D M2D-centroid re-alignment remain disabled for this dated
variant.
"""
from __future__ import annotations

from collections import defaultdict, deque
import os
from typing import Any

import numpy as np


VARIANT = "3"
VERSION_ID = "2026-09-14-paper-k2d-eq5-stage-separated"


def _active(params: Any) -> bool:
    explicit = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == VARIANT
    latest_ui = os.environ.get("ONESTRING_PAPER_K2D_20260914", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return bool(
        (explicit or latest_ui)
        and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"
    )


def _edge_components(faces: Any) -> list[np.ndarray]:
    """Face components under shared mesh edges; seam-duplicated ids stay disconnected."""
    f = np.asarray(faces, dtype=int)
    if len(f) == 0:
        return []
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(f):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edge = tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))
            edge_to_faces[edge].append(int(fi))
    adjacency: list[set[int]] = [set() for _ in range(len(f))]
    for touching in edge_to_faces.values():
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = int(touching[i]), int(touching[j])
                adjacency[a].add(b)
                adjacency[b].add(a)
    unseen = set(range(len(f)))
    out: list[np.ndarray] = []
    while unseen:
        root = unseen.pop()
        queue = deque([root])
        group = [root]
        while queue:
            cur = queue.popleft()
            for nxt in adjacency[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    group.append(nxt)
        out.append(np.asarray(sorted(group), dtype=int))
    out.sort(key=lambda x: int(x[0]) if len(x) else -1)
    return out


def _submesh_for_faces(mesh: Any, face_ids: np.ndarray) -> Any:
    """Select one component without renumbering the original regular-grid vertex IDs."""
    all_faces = np.asarray(mesh.faces, dtype=int)
    selected = all_faces[np.asarray(face_ids, dtype=int)].copy()
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    used = np.unique(selected.reshape(-1)) if len(selected) else np.asarray([], dtype=int)
    metrics = dict(getattr(mesh, "metrics", {}) or {})
    metrics.update(
        {
            "k2d_component_local_face_count": int(len(selected)),
            "k2d_component_used_vertex_count": int(len(used)),
            "k2d_component_preserves_global_vertex_ids": True,
            "k2d_component_preserves_lscm_lattice_parity_rule": True,
        }
    )
    cls = type(mesh)
    try:
        return cls(
            vertices,
            selected,
            mesh.grid,
            mesh.stage,
            metrics,
            list(getattr(mesh, "split_lines", [])),
        )
    except TypeError:
        return cls(
            vertices=vertices,
            faces=selected,
            grid=mesh.grid,
            stage=mesh.stage,
            metrics=metrics,
            split_lines=list(getattr(mesh, "split_lines", [])),
        )


def _tag_lscm_equivalent_result(result: Any, report: Any) -> tuple[Any, Any]:
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics.update(
        {
            "version_id": VERSION_ID,
            "k2d_solver_equivalent_to_lscm": True,
            "k2d_common_solver_authoritative": True,
            "k2d_optcuts_specific_rigid_tile_replacement": False,
            "k2d_optcuts_hard_sat_stage_in_k2d": False,
            "k2d_optcuts_global_se2_stage_in_k2d": False,
            "k2d_split_panel_centroid_realign_disabled": True,
            "k2d_split_panel_post_eq5_translation_applied": False,
            "paper_k2d_shared_vertex_authoritative": True,
            "paper_k2d_independent_rigid_tile_layout_in_k2d": False,
            "paper_k2d_hinge_alignment_in_k2d": False,
            "paper_k2d_hinge_layout_deferred_to_section_4_4": True,
            "paper_k2d_relative_layout_preserved_after_split": True,
            "paper_k2d_absolute_m2d_layout_reinjected_after_optimization": False,
            "k2d_20260914_policy": (
                "Use the exact same common _optimize_k2d solver path as LSCM; "
                "OptCuts-specific rigid/hard/global K2D replacement is disabled."
            ),
        }
    )
    try:
        result.metrics.clear()
        result.metrics.update(metrics)
    except Exception:
        pass
    try:
        report.objective = (
            "Common LSCM/OneString K2D optimization from M2D and K3D; "
            "OptCuts-specific rigid-tile K2D replacement disabled."
        )
    except Exception:
        pass
    return result, report


def _wire_k2d(pipeline: Any, fn: Any) -> None:
    pipeline._optimize_k2d = fn
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k2d = fn
    for build_fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k2d"] = fn


def _wire_flat_layout(pipeline: Any, fn: Any) -> None:
    pipeline._make_flat_tile_layout = fn
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._make_flat_tile_layout = fn
    for build_fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(build_fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_make_flat_tile_layout"] = fn


def _build_componentwise_flat_layout(
    mesh: Any,
    params: Any,
    base_make_layout: Any,
) -> Any:
    """Run the ordinary LSCM flat-layout builder once per disconnected panel."""
    components = _edge_components(mesh.faces)
    if len(components) <= 1:
        layout = base_make_layout(mesh, params)
        try:
            layout.metrics.update(
                {
                    "k2d_flat_layout_componentwise_lscm": True,
                    "k2d_flat_layout_component_count": int(len(components)),
                    "k2d_flat_layout_global_cross_component_solve": False,
                    "k2d_flat_layout_preserves_original_vertex_ids": True,
                }
            )
        except Exception:
            pass
        return layout

    face_count = int(len(np.asarray(mesh.faces)))
    merged_tiles = np.zeros((face_count, 4, 2), dtype=float)
    merged_hinge_pairs: list[tuple[int, int]] = []
    merged_gap_polygons: list[np.ndarray] = []
    component_metrics: list[dict[str, Any]] = []
    first_layout: Any | None = None

    for component_id, face_ids in enumerate(components):
        submesh = _submesh_for_faces(mesh, face_ids)
        local_layout = base_make_layout(submesh, params)
        if first_layout is None:
            first_layout = local_layout

        local_tiles = np.asarray(local_layout.tile_top_vertices_2d, dtype=float)
        if len(local_tiles) != len(face_ids):
            raise RuntimeError(
                "2026-09-14 component K2D layout face-count mismatch: "
                f"component={component_id} faces={len(face_ids)} tiles={len(local_tiles)}"
            )
        merged_tiles[np.asarray(face_ids, dtype=int)] = local_tiles

        for a, b in list(getattr(local_layout, "hinge_pairs", []) or []):
            ia, ib = int(a), int(b)
            if 0 <= ia < len(face_ids) and 0 <= ib < len(face_ids):
                merged_hinge_pairs.append((int(face_ids[ia]), int(face_ids[ib])))
        merged_gap_polygons.extend(
            [np.asarray(poly, dtype=float).copy() for poly in list(getattr(local_layout, "gap_polygons", []) or [])]
        )
        component_metrics.append(dict(getattr(local_layout, "metrics", {}) or {}))

    if first_layout is None:
        return base_make_layout(mesh, params)

    collision_count = int(sum(int(m.get("tile_overlap_count", 0) or 0) for m in component_metrics))
    clearances = [float(m.get("min_clearance", 0.0)) for m in component_metrics if m.get("min_clearance") is not None]
    min_clearance = float(min(clearances)) if clearances else 0.0
    metrics = dict(component_metrics[0]) if component_metrics else {}
    metrics.update(
        {
            "layout_type": "component-wise LSCM independent rigid K2D tile linkage layout",
            "tile_count": int(face_count),
            "hinge_pair_count": int(len(merged_hinge_pairs)),
            "k2d_gap_count": int(len(merged_gap_polygons)),
            "tile_overlap_count": collision_count,
            "min_clearance": min_clearance,
            "k2d_flat_layout_componentwise_lscm": True,
            "k2d_flat_layout_component_count": int(len(components)),
            "k2d_flat_layout_global_cross_component_solve": False,
            "k2d_flat_layout_cross_component_hinges": 0,
            "k2d_flat_layout_preserves_original_vertex_ids": True,
            "k2d_flat_layout_component_face_counts": [int(len(c)) for c in components],
            "k2d_flat_layout_policy": (
                "Apply the ordinary LSCM _make_flat_tile_layout independently to each "
                "edge-connected K2D panel while preserving original grid vertex IDs; "
                "merge results in original face order; no cross-seam hinges."
            ),
        }
    )

    layout_cls = type(first_layout)
    try:
        merged = layout_cls(
            tile_top_vertices_2d=merged_tiles,
            tile_ids=list(range(face_count)),
            hinge_pairs=merged_hinge_pairs,
            gap_polygons=merged_gap_polygons,
            metrics=metrics,
        )
    except TypeError:
        merged = layout_cls(
            merged_tiles,
            list(range(face_count)),
            merged_hinge_pairs,
            merged_gap_polygons,
            metrics,
        )

    print(
        "[2026-09-14-K2D-COMPONENT-LAYOUT] "
        f"components={len(components)} faces={face_count} hinges={len(merged_hinge_pairs)} "
        f"gaps={len(merged_gap_polygons)} global_cross_component_solve=False "
        "original_vertex_ids=True"
    )
    return merged


def _install_simple_split_bypass() -> None:
    """Keep both K2D solver and flat layout on the dated LSCM-equivalent route."""
    try:
        from . import simple_split_panel_patch as simple_split_module
    except Exception:
        return

    if getattr(simple_split_module, "_onestring_20260914_lscm_k2d_bypass_installed", False):
        return

    original_installer = simple_split_module.install_simple_split_panel_patch

    def install_with_20260914_lscm_k2d(pipeline_module: Any, optimization_debug_module: Any) -> None:
        original_installer(pipeline_module, optimization_debug_module)
        legacy_split_k2d = pipeline_module._optimize_k2d
        legacy_flat_layout = pipeline_module._make_flat_tile_layout
        dated_solver = getattr(pipeline_module, "_onestring_20260914_lscm_k2d_solver", None)
        dated_layout = getattr(pipeline_module, "_onestring_20260914_component_flat_layout", None)
        if not callable(dated_solver):
            dated_solver = legacy_split_k2d
        if not callable(dated_layout):
            dated_layout = legacy_flat_layout

        def dispatch(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
            if not _active(params):
                return legacy_split_k2d(
                    mesh_2d,
                    mesh_3d,
                    params,
                    progress_callback=progress_callback,
                )

            result, report = dated_solver(
                mesh_2d,
                mesh_3d,
                params,
                progress_callback=progress_callback,
            )
            result, report = _tag_lscm_equivalent_result(result, report)

            copy_attrs = getattr(simple_split_module, "_copy_attrs", None)
            if callable(copy_attrs):
                try:
                    copy_attrs(mesh_2d, result)
                except Exception:
                    pass

            print(
                "[2026-09-14-K2D-LSCM-EQUIVALENT] "
                "common LSCM K2D solver used; OptCuts rigid/hard/global K2D wrappers bypassed; "
                "post-K2D M2D-centroid realignment disabled"
            )
            return result, report

        def layout_dispatch(mesh: Any, params: Any = None):
            if params is not None and _active(params):
                return dated_layout(mesh, params)
            return legacy_flat_layout(mesh, params)

        _wire_k2d(pipeline_module, dispatch)
        _wire_flat_layout(pipeline_module, layout_dispatch)

    simple_split_module.install_simple_split_panel_patch = install_with_20260914_lscm_k2d
    simple_split_module._onestring_20260914_lscm_k2d_bypass_installed = True


def install_optcuts_paper_k2d_20260914_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_paper_k2d_20260914_installed", False):
        return

    # Capture the same common functions used by ordinary LSCM before OptCuts-only
    # wrappers are installed later by app_optcuts.
    lscm_common_k2d = pipeline._optimize_k2d
    lscm_common_flat_layout = pipeline._make_flat_tile_layout

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        result, report = lscm_common_k2d(
            mesh_2d,
            mesh_3d,
            params,
            progress_callback=progress_callback,
        )
        if not _active(params):
            return result, report

        result, report = _tag_lscm_equivalent_result(result, report)
        print(
            "[2026-09-14-K2D-BASE] "
            "used common LSCM _optimize_k2d; no 2026-09-14 extra K2D refinement"
        )
        return result, report

    def component_flat_layout(mesh: Any, params: Any = None):
        if params is None or not _active(params):
            return lscm_common_flat_layout(mesh, params)
        return _build_componentwise_flat_layout(mesh, params, lscm_common_flat_layout)

    pipeline._onestring_20260914_lscm_k2d_solver = optimize
    pipeline._onestring_20260914_component_flat_layout = component_flat_layout
    _wire_k2d(pipeline, optimize)
    _wire_flat_layout(pipeline, component_flat_layout)
    _install_simple_split_bypass()
    pipeline._onestring_optcuts_paper_k2d_20260914_installed = True


__all__ = ["install_optcuts_paper_k2d_20260914_patch", "VERSION_ID"]
