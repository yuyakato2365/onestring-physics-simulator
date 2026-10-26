"""Experimental 2026-10-26 OptCuts -> planar all-quad M2D adapter.

NOT an implementation of Zhang et al. (2021).  This first prototype uses
Gmsh's planar recombination and strictly rejects non-quad, inverted, outside-UV,
or non-bijectively lifted results. It preserves the original OptCuts UV chart,
surface triangle correspondence, and seam topology. Its purpose is to test
whether replacing the 50%-mask staircase is beneficial before paper integration.

Optional runtime dependencies: gmsh>=4.11, shapely>=2.0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np


MODE = "optcuts_planar_quad_20261026"


class PlanarQuadMeshingError(RuntimeError):
    """A candidate must fail, not quietly fall back to cropped squares."""


@dataclass(frozen=True)
class QuadTile:
    id: int
    row: int
    col: int
    vertex_ids: tuple[int, int, int, int]


@dataclass(frozen=True)
class QuadHinge:
    tile_a: int
    corner_a0: int
    corner_a1: int
    tile_b: int
    corner_b0: int
    corner_b1: int
    direction: str


class UnstructuredQuadGrid:
    """Preserve the QuadGrid face/hinge interface without inventing nx*ny cells.

    Consumers which require a rectangular row/column lattice must be updated;
    this adapter never claims the irregular connectivity is a regular grid.
    """
    def __init__(self, uv: np.ndarray, faces: np.ndarray, tile_size: float, gap_size: float):
        self.vertex_positions = np.column_stack((uv, np.zeros(len(uv))))
        self.tiles = [QuadTile(i, -1, -1, tuple(int(v) for v in f))
                      for i, f in enumerate(faces)]
        self.tile_size = float(tile_size)
        self.gap_size = float(gap_size)
        self.nx = 0
        self.ny = 0
        edge_owner: dict[tuple[int, int], tuple[int, int]] = {}
        hinges = []
        for tile in self.tiles:
            vs = tile.vertex_ids
            for k in range(4):
                edge = tuple(sorted((vs[k], vs[(k + 1) % 4])))
                if edge in edge_owner:
                    previous, previous_k = edge_owner.pop(edge)
                    hinges.append(QuadHinge(previous, previous_k, (previous_k + 1) % 4,
                                            tile.id, k, (k + 1) % 4, "unstructured"))
                else:
                    edge_owner[edge] = (tile.id, k)
        self.hinges = hinges

    @property
    def num_tiles(self) -> int:
        return len(self.tiles)

    def tile_corners_from_vertices(self, vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, float)
        ids = np.asarray([t.vertex_ids for t in self.tiles], int)
        return vertices[ids]

    def flat_tile_corners(self, include_gap: bool = True) -> np.ndarray:
        quads = self.tile_corners_from_vertices(self.vertex_positions).copy()
        if not include_gap or self.gap_size <= 0:
            return quads
        center = quads.mean(axis=1, keepdims=True)
        vec = quads - center
        norm = np.linalg.norm(vec[..., :2], axis=2, keepdims=True)
        scale = np.maximum(norm - self.gap_size / 2, 0.) / np.maximum(norm, 1e-12)
        quads[..., :2] = center[..., :2] + vec[..., :2] * scale
        return quads


def _triangle_inverse(uv_vertices: np.ndarray, uv_faces: np.ndarray,
                      xyz_vertices: np.ndarray, xyz_faces: np.ndarray,
                      samples: np.ndarray) -> np.ndarray:
    """Exact piecewise-linear inverse on the *original* paired OptCuts triangles."""
    uv = np.asarray(uv_vertices, float)
    tris = np.asarray(uv_faces, int)
    xyz = np.asarray(xyz_vertices, float)
    xyz_tri = np.asarray(xyz_faces, int)
    if len(tris) != len(xyz_tri):
        raise PlanarQuadMeshingError("QUAD_SOURCE_TRIANGLE_PAIR_COUNT_MISMATCH")
    a, b, c = uv[tris[:, 0]], uv[tris[:, 1]], uv[tris[:, 2]]
    u, v = b - a, c - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    scale = max(float(np.max(np.ptp(uv, axis=0))), 1e-9)
    if np.any(np.abs(det) <= 1e-14 * scale * scale) or np.any(det * det[0] <= 0):
        raise PlanarQuadMeshingError("QUAD_SOURCE_UV_FLIPPED_OR_DEGENERATE")
    low = np.min(uv[tris], axis=1) - 1e-9 * scale
    high = np.max(uv[tris], axis=1) + 1e-9 * scale
    out = []
    for index, p in enumerate(np.asarray(samples, float)):
        candidates = np.flatnonzero(np.all((p >= low) & (p <= high), axis=1))
        if not len(candidates):
            raise PlanarQuadMeshingError(f"QUAD_OUTSIDE_ORIGINAL_OMEGA: vertex={index}")
        d = p - a[candidates]
        s = (d[:, 0] * v[candidates, 1] - d[:, 1] * v[candidates, 0]) / det[candidates]
        t = (u[candidates, 0] * d[:, 1] - u[candidates, 1] * d[:, 0]) / det[candidates]
        good = np.flatnonzero((s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9))
        if not len(good):
            raise PlanarQuadMeshingError(f"QUAD_OUTSIDE_ORIGINAL_OMEGA: vertex={index}")
        j = int(candidates[good[0]])
        weights = np.array([1. - s[good[0]] - t[good[0]], s[good[0]], t[good[0]]])
        out.append(weights @ xyz[xyz_tri[j]])
    return np.asarray(out, float)


def _quad_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    quads = np.asarray(vertices, float)[np.asarray(faces, int)]
    e = np.roll(quads, -1, axis=1) - quads
    lengths = np.linalg.norm(e, axis=-1)
    length_min = float(np.min(lengths))
    length_max = float(np.max(lengths))
    median = max(float(np.median(lengths)), 1e-12)
    previous = np.roll(quads, 1, axis=1) - quads
    nxt = np.roll(quads, -1, axis=1) - quads
    cosines = np.sum(previous * nxt, axis=-1) / np.maximum(
        np.linalg.norm(previous, axis=-1) * np.linalg.norm(nxt, axis=-1), 1e-20)
    angles = np.rad2deg(np.arccos(np.clip(cosines, -1., 1.)))
    areas = .5 * (np.linalg.norm(np.cross(quads[:, 1] - quads[:, 0],
                                         quads[:, 2] - quads[:, 0]), axis=-1)
                  + np.linalg.norm(np.cross(quads[:, 3] - quads[:, 0],
                                            quads[:, 2] - quads[:, 0]), axis=-1))
    return {"edge_min": length_min, "edge_max_to_median": length_max / median,
            "angle_min_degrees": float(np.min(angles)),
            "angle_max_degrees": float(np.max(angles)),
            "area_min": float(np.min(areas)),
            "area_max_to_median": float(np.max(areas)) / max(float(np.median(areas)), 1e-12)}


def build_planar_quad(parameterization: Any, target_faces: int, *, refine_m3d: bool = True):
    """Gmsh conforming 2D quadrangulation; no masked-square fallback."""
    try:
        import gmsh
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise PlanarQuadMeshingError(
            "PLANAR_QUAD_DEPENDENCY_MISSING: install with "
            "'python3 -m pip install gmsh shapely'; the 2021 paper reference "
            "implementation has not yet been integrated"
        ) from exc

    uv_original = np.asarray(parameterization.uv_vertices_2d, float)
    boundary = np.asarray(parameterization.omega_boundary, float)[:, :2]
    if len(boundary) > 1 and np.linalg.norm(boundary[0] - boundary[-1]) < 1e-10:
        boundary = boundary[:-1]
    polygon = Polygon(boundary)
    if not polygon.is_valid or polygon.area <= 0 or len(polygon.interiors):
        raise PlanarQuadMeshingError("PLANAR_QUAD_SOURCE_NOT_SIMPLE_DISK")
    target_faces = max(4, int(target_faces))
    h = math.sqrt(float(polygon.area) / target_faces)
    # Boundary decimation is deliberately bounded; no arbitrary topology repair.
    simplified = polygon.simplify(min(h * .03, math.sqrt(polygon.area) * .005),
                                  preserve_topology=True)
    if simplified.geom_type != "Polygon" or not simplified.is_valid:
        raise PlanarQuadMeshingError("PLANAR_QUAD_BOUNDARY_SIMPLIFY_INVALID")
    boundary = np.asarray(simplified.exterior.coords, float)[:-1, :2]
    if len(boundary) < 4:
        boundary = np.asarray(polygon.exterior.coords, float)[:-1, :2]
    if len(boundary) < 4:
        raise PlanarQuadMeshingError("PLANAR_QUAD_BOUNDARY_TOO_FEW_CORNERS")
    # Disk all-quad topology needs even boundary-edge parity.
    if len(boundary) % 2:
        spans = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
        j = int(np.argmax(spans))
        boundary = np.insert(boundary, j + 1,
                             .5 * (boundary[j] + boundary[(j + 1) % len(boundary)]), axis=0)

    # Gmsh is process-global. Always clear and finalize even after failure.
    owner = not gmsh.isInitialized()
    if owner:
        gmsh.initialize()
    try:
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h * .75)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 1.25)
        gmsh.model.add("onestring_planar_quad_20261026")
        points = [gmsh.model.geo.addPoint(float(x), float(y), 0., h)
                  for x, y in boundary]
        lines = [gmsh.model.geo.addLine(points[i], points[(i + 1) % len(points)])
                 for i in range(len(points))]
        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.setRecombine(2, surface)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(coords, float).reshape(-1, 3)
        tags = np.asarray(tags, int)
        types, _elem_tags, node_tags = gmsh.model.mesh.getElements(2, surface)
        pieces = []
        for kind, conn in zip(types, node_tags):
            _, dim, order, count, *_ = gmsh.model.mesh.getElementProperties(kind)
            if int(dim) != 2 or int(order) != 1 or int(count) != 4:
                raise PlanarQuadMeshingError(
                    f"PLANAR_QUAD_NON_QUAD_ELEMENTS: element_type={kind} "
                    "(Gmsh recombination failed; not accepted as all-quad)")
            pieces.append(np.asarray(conn, int).reshape(-1, 4))
        if not pieces:
            raise PlanarQuadMeshingError("PLANAR_QUAD_NO_FACES")
        faces_tags = np.concatenate(pieces)
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        faces = np.asarray([[lookup[int(t)] for t in face] for face in faces_tags], int)
        uv = xyz[:, :2]
        used = np.unique(faces)
        remap = np.full(len(uv), -1, int)
        remap[used] = np.arange(len(used))
        faces = remap[faces]
        uv = uv[used]
    finally:
        gmsh.clear()
        if owner:
            gmsh.finalize()

    # Strict all-quad connectivity and subsequent geometry must be verified.
    polys = []
    orientations = []
    for i, ids in enumerate(faces):
        pts = uv[ids]
        signed = .5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                            - pts[:, 1] * np.roll(pts[:, 0], -1))
        orientations.append(signed)
        quad = Polygon(pts)
        if not quad.is_valid or quad.area <= 1e-12 * polygon.area:
            raise PlanarQuadMeshingError(f"PLANAR_QUAD_INVERTED_OR_DEGENERATE: face={i}")
        polys.append(quad)
    orientations = np.asarray(orientations, float)
    if (np.any(np.abs(orientations) <= 1e-12 * polygon.area)
            or np.any(orientations * orientations[0] <= 0)):
        raise PlanarQuadMeshingError("PLANAR_QUAD_MIXED_OR_ZERO_ORIENTATION")
    union = unary_union(polys)
    tol_area = 1e-5 * polygon.area
    if abs(sum(q.area for q in polys) - union.area) > tol_area:
        raise PlanarQuadMeshingError("PLANAR_QUAD_OVERLAPPING_FACES")
    if polygon.difference(union).area > .02 * polygon.area:
        raise PlanarQuadMeshingError("PLANAR_QUAD_DOMAIN_UNDERCOVERED_GT_2_PERCENT")
    if union.difference(polygon).area > tol_area:
        raise PlanarQuadMeshingError("PLANAR_QUAD_OUTSIDE_ORIGINAL_OMEGA")
    # Check intersections at quad edges as well as at vertices.
    checks = np.concatenate([uv, uv[faces].mean(axis=1),
        .5 * (uv[faces] + np.roll(uv[faces], -1, axis=1)).reshape(-1, 2)])
    mapped = _triangle_inverse(uv_original, parameterization.uv_faces,
                               parameterization.surface_vertices_3d,
                               parameterization.surface_faces, checks)
    xyz3 = mapped[:len(uv)]
    q2 = _quad_quality(np.column_stack((uv, np.zeros(len(uv)))), faces)
    q3 = _quad_quality(xyz3, faces)
    refine_report = {
        "planar_quad_m3d_quality_optimized": False,
        "planar_quad_refine_3d_moves_accepted": 0,
        "planar_quad_optimization_model": "original OptCuts UV unchanged; no 3D relaxation necessary",
    }
    if refine_m3d and (q3["angle_min_degrees"] < 55.
                       or q3["edge_max_to_median"] > 2.5
                       or q3["area_max_to_median"] > 3.5):
        from .optcuts_planar_quad_refine_20261026 import relax_m3d_quality
        uv, refine_report = relax_m3d_quality(
            uv, faces, parameterization, _triangle_inverse, sweeps=2, max_moves=2000)
        xyz3 = _triangle_inverse(uv_original, parameterization.uv_faces,
                                parameterization.surface_vertices_3d,
                                parameterization.surface_faces, uv)
        q2 = _quad_quality(np.column_stack((uv, np.zeros(len(uv)))), faces)
        q3 = _quad_quality(xyz3, faces)
    if q2["angle_min_degrees"] < 8 or q3["area_min"] <= 1e-14:
        raise PlanarQuadMeshingError("PLANAR_QUAD_BAD_SHAPE_OR_3D_COLLAPSE")
    report = {
        "planar_quad_engine": "Gmsh frontal-delaunay+recombine; NOT Zhang2021 reference code",
        "planar_quad_method": MODE,
        "planar_quad_faces": int(len(faces)),
        "planar_quad_target_faces": int(target_faces),
        "planar_quad_target_edge_length": float(h),
        "planar_quad_omega_undercoverage_fraction": float(
            polygon.difference(union).area / polygon.area),
        "planar_quad_omega_overcoverage_fraction": float(
            union.difference(polygon).area / polygon.area),
        "planar_quad_uv_inverse_exact_piecewise_linear": True,
        **{"planar_quad_m2d_" + k: float(v) for k, v in q2.items()},
        **{"planar_quad_m3d_" + k: float(v) for k, v in q3.items()},
        **refine_report,
        "planar_quad_m3d_quality_policy": "conditionally improved by constrained interior UV moves",
    }
    return uv, faces, report


def install_planar_quad_20261026(pipeline: Any) -> None:
    """Outermost hook installed after legacy OptCuts / seam patches."""
    if getattr(pipeline, "_planar_quad_20261026_installed", False):
        return
    base_parameterization = pipeline._build_surface_parameterization
    base_domain = pipeline._flatten_to_domain
    base_m2d = pipeline._build_m2d

    def with_parameterization(surface, target, grid, params):
        result = base_parameterization(surface, target, grid, params)
        if bool(getattr(params, "optcuts_planar_quad_20261026_enabled", False)):
            result.metrics["requested_omega_parameterization_mode"] = MODE
            result.metrics["planar_quad_source_seam_policy"] = "original OptCuts fixed"
            if bool(getattr(params, "planar_quad_uv_reparam", True)):
                from .optcuts_planar_quad_uv_reparam_20261026 import (
                    harmonically_smoothed_uv, boundary_cycle)
                old_uv = np.asarray(result.uv_vertices_2d, float)
                next_uv, report = harmonically_smoothed_uv(
                    result.surface_vertices_3d, result.surface_faces,
                    old_uv, result.uv_faces,
                    intensity=float(getattr(params, "planar_quad_uv_reparam_intensity", .12)))
                from .optcuts_planar_quad_uv_reparam_20261026 import (
                    coarse_boundary_reparameterize)
                final_uv, coarse_metrics = coarse_boundary_reparameterize(
                    result.surface_vertices_3d, result.surface_faces, next_uv,
                    result.uv_faces, int(getattr(params, "planar_quad_target_faces", 35)))
                report.update(coarse_metrics)
                result.uv_vertices_2d = final_uv
                bd = boundary_cycle(np.asarray(result.uv_faces, int))
                result.omega_boundary = np.vstack((final_uv[bd], final_uv[bd[0]]))
                # No stale acceleration structure may reference old UV.
                if hasattr(result, "triangle_acceleration"):
                    result.triangle_acceleration = None
                result.metrics.update(report)
            else:
                result.metrics.update(
                    planar_quad_uv_boundary_reparameterized=False,
                    planar_quad_uv_reparameterization_disabled=True)
        return result

    def with_domain(parameterization, grid, params=None):
        result = base_domain(parameterization, grid, params)
        if params is not None and bool(getattr(params, "optcuts_planar_quad_20261026_enabled", False)):
            result._planar_quad_source = parameterization
            result._planar_quad_target_faces = int(getattr(params, "planar_quad_target_faces", 35))
            # Retain the audited CSF Split stage. On irregular quad layouts it
            # may reject cuts that do not follow a complete shared edge chain.
            result.paper_split_policy = True
        return result

    def with_m2d(grid, domain, params=None):
        if not bool(getattr(params, "optcuts_planar_quad_20261026_enabled", False)):
            return base_m2d(grid, domain, params)
        if not hasattr(domain, "_planar_quad_source"):
            raise PlanarQuadMeshingError("PLANAR_QUAD_MISSING_OPTCUTS_SOURCE")
        uv, faces, report = build_planar_quad(
            domain._planar_quad_source, domain._planar_quad_target_faces,
            refine_m3d=bool(getattr(params, "planar_quad_refine_m3d", True)))
        # Unlike the regular 50%-mask mode, irregular faces are never
        # silently reconstructed on a rectangular lattice. The existing Split
        # module is allowed to test complete cuts, and reports failure to
        # certify the CSF bound if no cut is geometrically admissible.
        # Match the mesh's actual UV length scale; passing the old regular-grid
        # tile_size unchanged would make T2D gap/shrink geometry meaningless.
        all_edges = np.linalg.norm(np.roll(uv[faces], -1, axis=1) - uv[faces], axis=2)
        effective_size = float(np.median(all_edges))
        original_size = max(float(getattr(grid, "tile_size", 1.)), 1e-12)
        effective_gap = float(getattr(grid, "gap_size", .08)) * effective_size / original_size
        irregular = UnstructuredQuadGrid(uv, faces, effective_size, effective_gap)
        report["planar_quad_effective_uv_tile_size"] = effective_size
        report["planar_quad_effective_uv_gap_size"] = effective_gap
        xyz = np.column_stack((uv, np.zeros(len(uv))))
        mesh_type = pipeline._original.QuadMesh
        mesh = mesh_type(xyz, np.asarray(faces, int), irregular,
                         "M2D", report, [])
        print(f"[PLANAR-QUAD-20261026] validated {len(faces)} all-quad faces "
              f"from {len(uv)} UV vertices; "
              f"min3Dangle={report['planar_quad_m3d_angle_min_degrees']:.2f}",
              flush=True)
        if bool(getattr(domain, "paper_split_policy", False)):
            from .simple_split_panel_patch import apply_paper_split
            mesh = apply_paper_split(pipeline, mesh, domain, params)
            mesh.metrics["planar_quad_csf_split_certified"] = bool(
                mesh.metrics.get("csf_split_bound_certified", False))
            mesh.metrics["planar_quad_csf_split_mechanism"] = (
                "existing OneString complete-edge Split with explicit certification metrics")
            if (bool(getattr(params, "enable_csf_splits", True))
                    and float(getattr(domain, "csf_before", 1.)) >
                        float(getattr(params, "csf_split_threshold", 2.))
                    and not mesh.metrics["planar_quad_csf_split_certified"]):
                print("[PLANAR-QUAD-20261026] WARNING: CSF limit NOT "
                      "certified; no fictitious irregular cut has been invented",
                      flush=True)
        return mesh

    pipeline._build_surface_parameterization = with_parameterization
    pipeline._flatten_to_domain = with_domain
    pipeline._build_m2d = with_m2d
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = with_parameterization
        original._flatten_to_domain = with_domain
        original._build_m2d = with_m2d
    for fn in (getattr(pipeline, "build_onestring_design", None),
               getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
               getattr(original, "build_onestring_design", None) if original else None):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = with_parameterization
            glb["_flatten_to_domain"] = with_domain
            glb["_build_m2d"] = with_m2d
    pipeline._planar_quad_20261026_installed = True
