"""Strict OneString/OptCuts grid fusion.

This module fixes two subtle but important deviations in the first
``optcuts_grid_constrained_*`` implementation:

1. the two UV boundary copies of ONE physical OptCuts seam were allowed to land
   on two different parallel lattice lines.  That creates a geometric gap even
   though there is only one physical cut;
2. M2D used the same lattice, but did not explicitly disconnect topology along
   the now grid-aligned physical seam.

The strict policy implemented here is exactly:

* official OptCuts chooses the physical cut topology;
* the final UV embedding keeps that SAME cut topology;
* each maximal physical seam chain is ONE straight line;
* every line is parallel to one common axis or its perpendicular;
* the fixed existing ``tile_size`` is the lattice unit;
* both UV copies of one physical seam occupy the SAME geometric line (zero width)
  while remaining topologically distinct;
* the free UV vertices are reoptimized by the existing constrained SD solver;
* M2D is generated on the same lattice and topology is cut exactly along those
  grid edges by vertex duplication.  No second seam, seam healing, staircase
  snapping, or chart-strip deletion is introduced.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import optcuts_grid_constrained_parameterization_patch as constrained_param
from . import optcuts_pipeline_patch as optcuts_pipeline
from .optcuts_grid_seam_topology_patch import _duplicate_vertices_along_cut_edges


def _strict_hard_seam_targets(
    parameterization: Any,
    h: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build hard targets for ONE zero-width rectilinear line per physical seam.

    A physical seam has two UV boundary copies after cutting.  They are topology,
    not two different fabrication cuts, so corresponding copy vertices receive
    identical target coordinates here.
    """
    h = max(float(h), 1e-12)
    uv0 = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    records = constrained_param._surface_seam_records(parameterization)
    physical_chains = constrained_param._physical_chains(records)
    lookup = constrained_param._record_lookup(records)

    entries: list[dict[str, Any]] = []
    copy_chains: list[list[int]] = []
    for physical in physical_chains:
        # A non-trivial closed physical loop cannot be represented by one straight
        # finite segment.  Do not silently violate the requested seam model.
        if len(physical) >= 3 and int(physical[0]) == int(physical[-1]):
            raise RuntimeError(
                "OPTCUTS_GRID_CONSTRAINT_CLOSED_SEAM: a closed OptCuts seam cannot "
                "be represented by one straight line under the strict model"
            )
        side0, side1 = constrained_param._uv_copy_chains_for_physical_chain(physical, lookup)
        if len(side0) < 2 or len(side1) < 2:
            continue
        if len(side0) != len(physical) or len(side1) != len(physical):
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_COPY_CHAIN_LENGTH_MISMATCH")
        entries.append({
            "physical": [int(v) for v in physical],
            "copies": [[int(v) for v in side0], [int(v) for v in side1]],
        })
        copy_chains.extend([side0, side1])

    if not entries:
        return uv0.copy(), np.full_like(uv0, np.nan), {
            "seam_chain_count": 0,
            "seam_copy_chain_count": 0,
            "axis_rotation_degrees": 0.0,
            "constrained_vertex_count": 0,
            "zero_width_physical_seam": True,
            "physical_seam_segment_count": 0,
        }

    angle = constrained_param._dominant_axis_angle(uv0, copy_chains)
    rotated, _ = constrained_param._rotate_uv(uv0, -angle)

    # Pick H/V for each PHYSICAL chain, never separately for its two UV copies.
    for entry in entries:
        deltas = []
        for copy in entry["copies"]:
            pts = rotated[np.asarray(copy, dtype=int)]
            deltas.append(pts[-1] - pts[0])
        d = np.mean(np.asarray(deltas, dtype=float), axis=0)
        entry["axis"] = 0 if abs(float(d[0])) >= abs(float(d[1])) else 1

    # Same-axis chains that meet at a physical junction must lie on one common
    # grid line.  Union those chains before choosing the constant coordinate.
    parent = list(range(len(entries)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    incident: dict[int, list[int]] = defaultdict(list)
    for ci, entry in enumerate(entries):
        physical = entry["physical"]
        incident[int(physical[0])].append(ci)
        incident[int(physical[-1])].append(ci)
    for touching in incident.values():
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                a, b = touching[i], touching[j]
                if entries[a]["axis"] == entries[b]["axis"]:
                    union(a, b)

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ci, entry in enumerate(entries):
        groups[(int(entry["axis"]), find(ci))].append(ci)
    for (axis, _root), members in groups.items():
        const_coord = 1 - int(axis)
        sample_ids = sorted({
            int(u)
            for ci in members
            for copy in entries[ci]["copies"]
            for u in copy
        })
        mean_value = float(np.mean(rotated[np.asarray(sample_ids, dtype=int), const_coord]))
        constant = round(mean_value / h) * h
        for ci in members:
            entries[ci]["constant"] = float(constant)

    # One lattice point per PHYSICAL terminal/junction.  All UV copies of that
    # physical vertex will eventually receive this same point.
    physical_uv_samples: dict[int, list[int]] = defaultdict(list)
    for record in records:
        a, b = record["surface_edge"]
        for copy in record["copies"]:
            physical_uv_samples[int(a)].append(int(copy[int(a)]))
            physical_uv_samples[int(b)].append(int(copy[int(b)]))

    terminal_target: dict[int, np.ndarray] = {}
    for surface_vertex, touching in incident.items():
        uv_ids = sorted(set(physical_uv_samples[int(surface_vertex)]))
        if not uv_ids:
            raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_TERMINAL_WITHOUT_UV_COPY")
        p = np.mean(rotated[np.asarray(uv_ids, dtype=int)], axis=0)
        vertical_constants = [
            float(entries[ci]["constant"])
            for ci in touching
            if int(entries[ci]["axis"]) == 1
        ]
        horizontal_constants = [
            float(entries[ci]["constant"])
            for ci in touching
            if int(entries[ci]["axis"]) == 0
        ]
        p[0] = float(np.mean(vertical_constants)) if vertical_constants else round(float(p[0]) / h) * h
        p[1] = float(np.mean(horizontal_constants)) if horizontal_constants else round(float(p[1]) / h) * h
        terminal_target[int(surface_vertex)] = np.round(p / h) * h

    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    targets = np.full_like(rotated, np.nan)
    physical_segments: list[list[list[float]]] = []

    # Assign one common straight segment to both UV copies.  Corresponding
    # physical seam vertices are identical geometrically in the final UV.
    for entry in entries:
        physical = entry["physical"]
        side0, side1 = entry["copies"]
        axis = int(entry["axis"])
        const = float(entry["constant"])
        start_sv, end_sv = int(physical[0]), int(physical[-1])
        p0 = terminal_target[start_sv].copy()
        p1 = terminal_target[end_sv].copy()
        p0[1 - axis] = const
        p1[1 - axis] = const

        varying0 = float(p0[axis])
        varying1 = float(p1[axis])
        if abs(varying1 - varying0) < h:
            # Maintain ordering from the OptCuts proposal but enforce at least one
            # fabrication unit of usable seam length.
            proposal_delta = float(
                np.mean([
                    rotated[side0[-1], axis] - rotated[side0[0], axis],
                    rotated[side1[-1], axis] - rotated[side1[0], axis],
                ])
            )
            sign = 1.0 if proposal_delta >= 0.0 else -1.0
            varying1 = varying0 + sign * h
            p1[axis] = varying1
            terminal_target[end_sv] = p1.copy()

        xyz = surface_xyz[np.asarray(physical, dtype=int)]
        edge_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        total = float(cumulative[-1])
        tvals = cumulative / total if total > 1e-12 else np.linspace(0.0, 1.0, len(physical))

        for k, t in enumerate(tvals):
            target = np.zeros(2, dtype=float)
            target[axis] = (1.0 - float(t)) * varying0 + float(t) * varying1
            target[1 - axis] = const
            # Same physical point, same geometric seam point, BOTH UV copies.
            for u in (int(side0[k]), int(side1[k])):
                if np.all(np.isfinite(targets[u])) and np.linalg.norm(targets[u] - target) > 1e-7 * h:
                    # At a junction multiple chain constraints must intersect.
                    # A conflict means the H/V assignment itself is inconsistent;
                    # do not average into a non-grid point.
                    raise RuntimeError(
                        f"OPTCUTS_GRID_CONSTRAINT_JUNCTION_CONFLICT: uv={u} "
                        f"old={targets[u].tolist()} new={target.tolist()}"
                    )
                targets[u] = target
        physical_segments.append([p0.tolist(), p1.tolist()])

    fixed = np.all(np.isfinite(targets), axis=1)
    info = {
        "seam_chain_count": int(len(entries)),
        "seam_copy_chain_count": int(2 * len(entries)),
        "axis_rotation_degrees": float(np.degrees(-angle)),
        "constrained_vertex_count": int(np.count_nonzero(fixed)),
        "zero_width_physical_seam": True,
        "physical_seam_segment_count": int(len(physical_segments)),
        "physical_seam_segments": physical_segments,
        "seam_copy_lines_are_coincident": True,
    }
    return rotated, targets, info


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _point_on_axis_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    p = np.asarray(point, dtype=float)[:2]
    a = np.asarray(a, dtype=float)[:2]
    b = np.asarray(b, dtype=float)[:2]
    if abs(float(a[0] - b[0])) <= tol:
        return abs(float(p[0] - a[0])) <= tol and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    if abs(float(a[1] - b[1])) <= tol:
        return abs(float(p[1] - a[1])) <= tol and min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
    return False


def install_strict_grid_seam_topology_patch(pipeline: Any) -> None:
    """Disconnect M2D exactly on the already-constrained physical seam lines."""
    if getattr(pipeline, "_onestring_strict_grid_seam_topology_installed", False):
        return
    base_build = pipeline._build_m2d

    def build_with_exact_cut(grid: Any, domain: Any, params: Any = None):
        mesh = base_build(grid, domain, params)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        if not bool(metrics.get("optcuts_grid_constrained_m2d", False)):
            return mesh
        parameterization = getattr(domain, "_optcuts_grid_constrained_parameterization", None)
        if parameterization is None:
            return mesh
        pmetrics = dict(getattr(parameterization, "metrics", {}) or {})
        segments = np.asarray(pmetrics.get("physical_seam_segments", []), dtype=float)
        if len(segments) == 0:
            metrics.update({
                "optcuts_exact_seam_cut_applied": False,
                "optcuts_exact_seam_cut_edge_count": 0,
            })
            mesh.metrics.update(metrics)
            return mesh

        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        h = max(float(pmetrics.get("optcuts_grid_unit", getattr(params, "tile_size", 0.0))), 1e-12)
        tol = max(1e-9, 1e-6 * h)
        all_edges: set[tuple[int, int]] = set()
        for face in faces:
            ids = [int(v) for v in face]
            for i in range(len(ids)):
                all_edges.add(_edge_key(ids[i], ids[(i + 1) % len(ids)]))

        cut_edges: set[tuple[int, int]] = set()
        for edge in all_edges:
            a_id, b_id = edge
            pa, pb = vertices[a_id, :2], vertices[b_id, :2]
            midpoint = 0.5 * (pa + pb)
            for segment in segments:
                s0, s1 = segment[0], segment[1]
                if (
                    _point_on_axis_segment(pa, s0, s1, tol)
                    and _point_on_axis_segment(pb, s0, s1, tol)
                    and _point_on_axis_segment(midpoint, s0, s1, tol)
                ):
                    cut_edges.add(edge)
                    break

        if not cut_edges:
            raise RuntimeError(
                "OPTCUTS_GRID_SEAM_NOT_ON_M2D_EDGES: constrained seam exists but no exact M2D grid edge matches it"
            )

        cut_vertices, cut_faces, duplicated = _duplicate_vertices_along_cut_edges(
            vertices,
            faces,
            cut_edges,
        )
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
        metrics.update({
            "optcuts_exact_seam_cut_applied": True,
            "optcuts_exact_seam_cut_edge_count": int(len(cut_edges)),
            "optcuts_exact_seam_duplicated_vertex_count": int(duplicated),
            "optcuts_exact_seam_gap": 0.0,
            "optcuts_posthoc_extra_seam": False,
            "optcuts_posthoc_seam_snap": False,
            "optcuts_physical_seam_topology_only": True,
        })
        out = cls(
            np.asarray(cut_vertices, dtype=float),
            np.asarray(cut_faces, dtype=int),
            mesh.grid,
            mesh.stage,
            metrics,
            [],
        )
        setattr(out, "_optcuts_exact_cut_edges", sorted(cut_edges))
        print(
            "[OPTCUTS-EXACT-SEAM] "
            f"segments={len(segments)} cut_edges={len(cut_edges)} "
            f"duplicated_vertices={duplicated} gap=0"
        )
        return out

    pipeline._build_m2d = build_with_exact_cut
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_m2d = build_with_exact_cut
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_with_exact_cut
    pipeline._onestring_strict_grid_seam_topology_installed = True


def install_strict_optcuts_grid_fusion() -> None:
    """Patch module-level behavior before the runtime pipeline is executed."""
    # The legacy official bridge used to rotate the OptCuts result for a later
    # post-hoc seam adapter.  The strict constrained solver owns the common frame
    # now, so disable that old extra transform.
    def no_legacy_axis_rotation(parameterization: Any) -> float:
        parameterization.metrics["optcuts_legacy_axis_alignment_disabled"] = True
        return 0.0

    optcuts_pipeline._align_uv_to_optcuts_seam_axis = no_legacy_axis_rotation
    constrained_param._build_hard_seam_targets = _strict_hard_seam_targets


__all__ = [
    "install_strict_optcuts_grid_fusion",
    "install_strict_grid_seam_topology_patch",
    "_strict_hard_seam_targets",
]
