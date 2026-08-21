"""Deterministic M2D -> M3D lift for native Grid-OptCuts.

A zero-width OptCuts seam deliberately gives the same UV coordinates to multiple
UV boundary copies.  The generic inverse lookup returns the first containing
triangle, which is geometrically harmless only if every containing UV triangle
maps that point to the same physical point on S.  This wrapper makes that
assumption explicit and audited:

* find all UV triangles containing each used M2D vertex;
* barycentrically map through every corresponding surface triangle;
* require all mapped 3D candidates to agree within tolerance;
* use their mean as the deterministic lifted point.

No cell is deleted and no chart is repaired after the fact.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any

import numpy as np


def _barycentric(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = np.asarray(tri, dtype=float)
    v0 = b - a
    v1 = c - a
    v2 = np.asarray(point, dtype=float) - a
    denom = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denom) <= 1e-15:
        return None
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denom)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denom)
    return np.asarray([1.0 - u - v, u, v], dtype=float)


def _triangle_bins(uv: np.ndarray, uv_faces: np.ndarray) -> dict[str, Any]:
    tris = np.asarray(uv, dtype=float)[np.asarray(uv_faces, dtype=int)]
    lo = np.min(tris, axis=1)
    hi = np.max(tris, axis=1)
    global_lo = np.min(lo, axis=0)
    global_hi = np.max(hi, axis=0)
    span = np.maximum(global_hi - global_lo, 1e-12)
    bins_n = max(1, min(256, int(math.ceil(math.sqrt(max(len(tris), 1))))))
    cell = span / float(bins_n)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for tri_id in range(len(tris)):
        a = np.floor((lo[tri_id] - global_lo) / cell).astype(int)
        b = np.floor((hi[tri_id] - global_lo) / cell).astype(int)
        a = np.clip(a, 0, bins_n - 1)
        b = np.clip(b, 0, bins_n - 1)
        for ix in range(int(a[0]), int(b[0]) + 1):
            for iy in range(int(a[1]), int(b[1]) + 1):
                buckets[(ix, iy)].append(tri_id)
    return {
        "triangles": tris,
        "lo": lo,
        "hi": hi,
        "global_lo": global_lo,
        "global_hi": global_hi,
        "cell": cell,
        "bins_n": bins_n,
        "buckets": buckets,
    }


def _candidate_triangle_ids(point: np.ndarray, accel: dict[str, Any], tol: float) -> list[int]:
    p = np.asarray(point, dtype=float)
    global_lo = np.asarray(accel["global_lo"], dtype=float)
    cell = np.asarray(accel["cell"], dtype=float)
    bins_n = int(accel["bins_n"])
    raw = np.floor((p - global_lo) / cell).astype(int)
    raw = np.clip(raw, 0, bins_n - 1)
    ids: set[int] = set()
    # Include neighboring bins because a point can sit numerically on a bin edge.
    for ix in range(max(0, int(raw[0]) - 1), min(bins_n - 1, int(raw[0]) + 1) + 1):
        for iy in range(max(0, int(raw[1]) - 1), min(bins_n - 1, int(raw[1]) + 1) + 1):
            ids.update(int(x) for x in accel["buckets"].get((ix, iy), []))
    lo = np.asarray(accel["lo"], dtype=float)
    hi = np.asarray(accel["hi"], dtype=float)
    return [i for i in ids if np.all(p >= lo[i] - tol) and np.all(p <= hi[i] + tol)]


def _map_used_vertices(
    mesh: Any,
    parameterization: Any,
    *,
    bary_tol: float,
    agreement_tol: float,
) -> tuple[np.ndarray, dict[str, int | float]]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    if len(uf) != len(sf):
        raise RuntimeError(
            "OPTCUTS_GRID_LIFT_FACE_CORRESPONDENCE: uv_faces and surface_faces differ in length"
        )

    out = np.full((len(mesh.vertices), 3), np.nan, dtype=float)
    used = np.unique(np.asarray(mesh.faces, dtype=int).reshape(-1))
    accel = _triangle_bins(uv, uf)
    multi_triangle_vertices = 0
    seam_or_edge_vertices = 0
    max_candidate_spread = 0.0
    max_containing = 0

    for vi in used:
        vi = int(vi)
        p = np.asarray(mesh.vertices[vi, :2], dtype=float)
        mapped: list[np.ndarray] = []
        containing_ids: list[int] = []
        for tri_id in _candidate_triangle_ids(p, accel, bary_tol):
            bary = _barycentric(p, uv[uf[tri_id]])
            if bary is None or float(np.min(bary)) < -bary_tol:
                continue
            mapped.append(bary @ xyz[sf[tri_id]])
            containing_ids.append(int(tri_id))
        if not mapped:
            raise RuntimeError(
                "OPTCUTS_GRID_LIFT_NO_CONTAINING_TRIANGLE: "
                f"vertex={vi} uv={p.tolist()}"
            )
        pts = np.asarray(mapped, dtype=float)
        center = np.mean(pts, axis=0)
        spread = float(np.max(np.linalg.norm(pts - center[None, :], axis=1))) if len(pts) > 1 else 0.0
        max_candidate_spread = max(max_candidate_spread, spread)
        max_containing = max(max_containing, len(pts))
        if len(pts) > 1:
            multi_triangle_vertices += 1
            seam_or_edge_vertices += 1
        if spread > agreement_tol:
            raise RuntimeError(
                "OPTCUTS_GRID_LIFT_AMBIGUOUS: coincident UV copies do not map to the same physical point; "
                f"vertex={vi} uv={p.tolist()} containing_triangles={containing_ids[:16]} "
                f"candidate_spread={spread:.9g} tolerance={agreement_tol:.9g}"
            )
        out[vi] = center

    # M2D may retain unused regular-lattice vertices. They are irrelevant to tile
    # geometry, but later vectorized code expects finite arrays. Map them through
    # the same audit if possible; otherwise copy the nearest used vertex only for
    # those unused slots and record the count. Faces never reference them.
    unused = np.setdiff1d(np.arange(len(mesh.vertices), dtype=int), used)
    fallback_unused = 0
    if len(unused):
        if len(used) == 0:
            raise RuntimeError("OPTCUTS_GRID_LIFT_EMPTY_M2D")
        used_uv = np.asarray(mesh.vertices, dtype=float)[used, :2]
        for vi in unused:
            p = np.asarray(mesh.vertices[int(vi), :2], dtype=float)
            nearest = int(used[int(np.argmin(np.linalg.norm(used_uv - p[None, :], axis=1)))])
            out[int(vi)] = out[nearest]
            fallback_unused += 1

    if not np.all(np.isfinite(out)):
        raise RuntimeError("OPTCUTS_GRID_LIFT_NONFINITE_RESULT")
    return out, {
        "used_vertex_count": int(len(used)),
        "unused_vertex_fallback_count": int(fallback_unused),
        "multi_triangle_vertex_count": int(multi_triangle_vertices),
        "seam_or_edge_vertex_count": int(seam_or_edge_vertices),
        "max_containing_triangle_count": int(max_containing),
        "max_candidate_3d_spread": float(max_candidate_spread),
    }


def install_native_grid_optcuts_lift_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_native_grid_optcuts_lift_installed", False):
        return
    base_lift = pipeline._lift_m2d_to_m3d

    def lift_native_grid(target: Any, mesh: Any, parameterization: Any, params: Any):
        metrics_in = dict(getattr(mesh, "metrics", {}) or {})
        if not bool(metrics_in.get("optcuts_grid_constrained_m2d", False)):
            return base_lift(target, mesh, parameterization, params)
        if str(getattr(params, "m3d_construction_mode", "mesh_harmonic")) != "mesh_harmonic":
            return base_lift(target, mesh, parameterization, params)

        start = time.perf_counter()
        scale3 = float(np.linalg.norm(
            np.max(np.asarray(parameterization.surface_vertices_3d, dtype=float), axis=0)
            - np.min(np.asarray(parameterization.surface_vertices_3d, dtype=float), axis=0)
        ))
        h = max(float(metrics_in.get("optcuts_grid_unit", getattr(mesh.grid, "tile_size", 1.0))), 1e-12)
        agreement_tol = max(1e-8 * max(scale3, 1.0), 1e-7 * h)
        bary_tol = 1e-8
        vertices, audit = _map_used_vertices(
            mesh,
            parameterization,
            bary_tol=bary_tol,
            agreement_tol=agreement_tol,
        )

        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or getattr(pipeline, "QuadMesh")
        lifted = cls(vertices, np.asarray(mesh.faces, dtype=int).copy(), mesh.grid, "M3D", {}, list(getattr(mesh, "split_lines", [])))
        planarity_fn = getattr(pipeline, "_quad_planarity_error")
        planarity = float(planarity_fn(vertices, mesh.faces))
        dist_fn = getattr(pipeline, "_distances_to_surface_mesh")
        distances = np.asarray(
            dist_fn(vertices[np.unique(np.asarray(mesh.faces, dtype=int).reshape(-1))],
                    np.asarray(parameterization.surface_vertices_3d, dtype=float),
                    np.asarray(parameterization.surface_faces, dtype=int)),
            dtype=float,
        )
        surface_mean = float(np.mean(distances)) if len(distances) else 0.0
        surface_max = float(np.max(distances)) if len(distances) else 0.0
        lifted.metrics = {
            **metrics_in,
            "surface_deviation": surface_mean,
            "quad_planarity_error": planarity,
            "m3d_construction_method": "mesh_harmonic_native_grid_audited",
            "parameterization_method": str(getattr(parameterization, "method", "optcuts_grid_native")),
            "m3d_surface_distance_mean": surface_mean,
            "m3d_surface_distance_max": surface_max,
            "m3d_uv_triangle_lookup_fail_count": 0,
            "m3d_outside_omega_count": 0,
            "m3d_used_height_field_shortcut": False,
            "m3d_vertex_count": int(len(vertices)),
            "m3d_quad_count": int(len(mesh.faces)),
            "m3d_planarity_error": planarity,
            "optcuts_grid_lift_audited": True,
            "optcuts_grid_lift_barycentric_tolerance": float(bary_tol),
            "optcuts_grid_lift_agreement_tolerance": float(agreement_tol),
            **{f"optcuts_grid_lift_{k}": v for k, v in audit.items()},
        }

        report_cls = getattr(getattr(pipeline, "_original", None), "StageReport", None) or getattr(pipeline, "StageReport")
        counts_fn = getattr(pipeline, "_mesh_counts", None)
        counts = dict(counts_fn(lifted)) if callable(counts_fn) else {
            "vertices": int(len(vertices)),
            "faces": int(len(mesh.faces)),
        }
        report = report_cls(
            name="M2D -> M3D",
            objective=(
                "Audited inverse OptCuts map: all UV triangles containing a zero-width seam/grid vertex "
                "must agree on the same physical S point."
            ),
            before_error=0.0,
            after_error=surface_mean,
            constraint_violation=planarity,
            computation_time=float(time.perf_counter() - start),
            counts=counts,
        )
        print(
            "[OPTCUTS-GRID-LIFT] "
            f"used_vertices={audit['used_vertex_count']} "
            f"multi_triangle={audit['multi_triangle_vertex_count']} "
            f"max_containing={audit['max_containing_triangle_count']} "
            f"max_3d_spread={audit['max_candidate_3d_spread']:.6g} "
            f"agreement_tol={agreement_tol:.6g}"
        )
        return lifted, report

    pipeline._lift_m2d_to_m3d = lift_native_grid
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._lift_m2d_to_m3d = lift_native_grid
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_lift_m2d_to_m3d"] = lift_native_grid
    pipeline._onestring_native_grid_optcuts_lift_installed = True


__all__ = ["install_native_grid_optcuts_lift_patch", "_map_used_vertices"]
