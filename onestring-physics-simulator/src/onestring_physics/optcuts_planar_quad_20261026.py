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
from threading import RLock
import numpy as np


MODE = "optcuts_planar_quad_20261026"

# Gmsh has a process-global model/option state. Streamlit may dispatch
# independent browser sessions on separate script threads. Serialize our
# initialize/mesh/finalize transaction rather than clearing another run's model.
_GMSH_LOCK = RLock()


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
        uv = np.asarray(uv, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if (uv.ndim != 2 or uv.shape[1] != 2 or faces.ndim != 2
                or faces.shape[1] != 4 or not len(faces)
                or not np.isfinite(uv).all()):
            raise PlanarQuadMeshingError("PLANAR_QUAD_GRID_INVALID_ARRAYS")
        if np.any(faces < 0) or np.any(faces >= len(uv)):
            raise PlanarQuadMeshingError("PLANAR_QUAD_GRID_VERTEX_INDEX_INVALID")
        if any(len(set(map(int, face))) != 4 for face in faces):
            raise PlanarQuadMeshingError("PLANAR_QUAD_GRID_REPEATED_FACE_VERTEX")
        self.vertex_positions = np.column_stack((uv, np.zeros(len(uv))))
        self.tiles = [QuadTile(i, -1, -1, tuple(int(v) for v in f))
                      for i, f in enumerate(faces)]
        self.tile_size = float(tile_size)
        self.gap_size = float(gap_size)
        self.nx = 0
        self.ny = 0
        edge_owner: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for tile in self.tiles:
            vs = tile.vertex_ids
            for k in range(4):
                a, b = int(vs[k]), int(vs[(k + 1) % 4])
                edge_owner.setdefault((min(a, b), max(a, b)), []).append(
                    (tile.id, k, a, b))
        hinges = []
        for edge, owners in edge_owner.items():
            if len(owners) > 2:
                raise PlanarQuadMeshingError(
                    f"PLANAR_QUAD_GRID_NONMANIFOLD_EDGE: edge={edge} owners={len(owners)}")
            if len(owners) == 2:
                (prev, pk, pa, pb), (cur, ck, ca, cb) = owners
                if pa != cb or pb != ca:
                    raise PlanarQuadMeshingError(
                        f"PLANAR_QUAD_GRID_SHARED_EDGE_WINDING: edge={edge}")
                hinges.append(QuadHinge(prev, pk, (pk + 1) % 4,
                                        cur, ck, (ck + 1) % 4, "unstructured"))
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
            "area_max_to_median": float(np.max(areas)) / max(float(np.median(areas)), 1e-12),
            # An average or minimum alone cannot locate a Bunny neck failure.
            "worst_angle_face_index": int(np.argmin(np.min(angles, axis=1))),
            "largest_area_face_index": int(np.argmax(areas)),
            "smallest_area_face_index": int(np.argmin(areas)),
            "longest_edge_face_index": int(np.argmax(np.max(lengths, axis=1)))}


def _choose_contained_omega_boundary(polygon: Any, edge_length: float) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Reduce boundary complexity only if the whole candidate stays in Ω.

    Shapely's preserve_topology=True prevents self-intersections; it does NOT
    promise geometric containment. Removing a shallow concave notch can create
    a shortcut *outside* the original OptCuts disk and invalidate UV inverse.
    """
    original = np.asarray(polygon.exterior.coords, float)[:-1, :2]
    area = float(polygon.area)
    tolerance = min(float(edge_length) * .03, math.sqrt(area) * .005)
    if len(original) < 4:
        raise PlanarQuadMeshingError("PLANAR_QUAD_BOUNDARY_TOO_FEW_CORNERS")
    chosen = original
    chosen_tolerance = 0.
    rejected_outside = 0
    for factor in (1., .5, .25, .125, .0625):
        candidate = polygon.simplify(tolerance * factor, preserve_topology=True)
        if candidate.geom_type != "Polygon" or not candidate.is_valid:
            continue
        coords = np.asarray(candidate.exterior.coords, float)[:-1, :2]
        if len(coords) < 4 or len(coords) >= len(chosen):
            continue
        # A valid polygon need not be a subset of the original Ω.
        if float(candidate.difference(polygon).area) > 1e-12 * area:
            rejected_outside += 1
            continue
        if float(polygon.difference(candidate).area) > .02 * area:
            continue
        chosen = coords
        chosen_tolerance = tolerance * factor
        break
    return chosen.copy(), {
        "planar_quad_original_boundary_vertices": int(len(original)),
        "planar_quad_meshed_boundary_vertices": int(len(chosen)),
        "planar_quad_boundary_simplification_tolerance": float(chosen_tolerance),
        "planar_quad_boundary_outside_rejected_candidates": int(rejected_outside),
        "planar_quad_boundary_source_preserved": bool(
            len(chosen) == len(original) and np.array_equal(chosen, original)),
    }


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
    # Only simplify when the whole resulting contour remains inside the
    # original OptCuts Ω. Topology-preserving does not imply containment.
    boundary, boundary_report = _choose_contained_omega_boundary(polygon, h)
    # Each Gmsh curve gets even 1D segments; even corner parity is unnecessary.
    meshed_polygon = Polygon(boundary)
    meshed_outside_fraction = float(
        meshed_polygon.difference(polygon).area / polygon.area)
    if meshed_outside_fraction > 1e-12:
        raise PlanarQuadMeshingError(
            "PLANAR_QUAD_BOUNDARY_NOT_CONTAINED: "
            f"outside_fraction={meshed_outside_fraction:.9g}")

    # Gmsh is process-global. Always clear and finalize even after failure.
    with _GMSH_LOCK:
        owner = not gmsh.isInitialized()
        if owner:
            # Streamlit executes scripts on worker threads. Gmsh's default
            # interruptible=True installs a SIGINT handler, which Python permits
            # only on the main thread. This flag disables only that handler.
            gmsh.initialize(interruptible=False)
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
            # Full-quad recombination (Mesh.RecombinationAlgorithm=2)
            # internally coarsens *each individual 1D curve* by pairing
            # adjacent line elements. Merely giving the whole polygon an
            # even number of edges does NOT suffice: a short boundary curve
            # independently meshed into one/three elements triggers
            # Gmsh's "1D mesh cannot be divided by 2".
            # Give EVERY boundary curve an even positive element count.
            # The node count for setTransfiniteCurve is elements + 1.
            edge_lengths = np.linalg.norm(
                np.roll(boundary, -1, axis=0) - boundary, axis=1)
            if np.any(edge_lengths <= 1e-12 * max(float(h), 1.)):
                raise PlanarQuadMeshingError(
                    "PLANAR_QUAD_ZERO_LENGTH_BOUNDARY_CURVE")
            curve_elements = np.maximum(
                2, 2 * np.ceil(edge_lengths / (2. * h)).astype(int))
            loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            # gmsh.model.mesh.* operates on the synchronized model; the
            # uncommitted geo.addLine entities do not exist there yet.
            gmsh.model.geo.synchronize()
            for curve, count in zip(lines, curve_elements):
                gmsh.model.mesh.setTransfiniteCurve(int(curve), int(count) + 1)
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

    # This is required even when refine_m3d=False. In the old implementation
    # the disk/nonmanifold audit ran only inside the optional refinement.
    from .optcuts_planar_quad_refine_20261026 import validate_disk_mesh
    try:
        validate_disk_mesh(uv, faces)
    except ValueError as exc:
        raise PlanarQuadMeshingError(
            "PLANAR_QUAD_GRID_TOPOLOGY_INVALID: " + str(exc)
        ) from exc

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
    # Positive polygon area alone accepts concave four-gons. Require every
    # corner to retain the orientation of its containing quadrilateral.
    local_edges = np.roll(uv[faces], -1, axis=1) - uv[faces]
    turns = np.cross(local_edges, np.roll(local_edges, -1, axis=1))
    if np.any(turns * orientations[:, None] <= 1e-12 * polygon.area):
        raise PlanarQuadMeshingError("PLANAR_QUAD_CONCAVE_OR_DEGENERATE_FACE")
    if (np.any(np.abs(orientations) <= 1e-12 * polygon.area)
            or np.any(orientations * orientations[0] <= 0)):
        raise PlanarQuadMeshingError("PLANAR_QUAD_MIXED_OR_ZERO_ORIENTATION")
    union = unary_union(polys)
    tol_area = 1e-5 * polygon.area
    if abs(sum(q.area for q in polys) - union.area) > tol_area:
        raise PlanarQuadMeshingError("PLANAR_QUAD_OVERLAPPING_FACES")
    if polygon.difference(union).area > .02 * polygon.area:
        raise PlanarQuadMeshingError("PLANAR_QUAD_DOMAIN_UNDERCOVERED_GT_2_PERCENT")
    outside_area = float(union.difference(polygon).area)
    if outside_area > tol_area:
        raise PlanarQuadMeshingError(
            "PLANAR_QUAD_OUTSIDE_ORIGINAL_OMEGA: "
            f"mesh_overcoverage_fraction={outside_area / polygon.area:.9g} "
            f"boundary_overcoverage_fraction={meshed_outside_fraction:.9g} "
            f"original_boundary_vertices={boundary_report['planar_quad_original_boundary_vertices']} "
            f"meshed_boundary_vertices={len(boundary)}; "
            "refusing to accept outside UV geometry")
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
    if (q2["angle_min_degrees"] < 20
            or q2["edge_max_to_median"] > 6.
            or q2["area_max_to_median"] > 8.
            or q3["angle_min_degrees"] < 12
            or q3["area_min"] <= 1e-8 * max(
                float(np.median(np.linalg.norm(
                    np.cross(xyz3[faces][:,1]-xyz3[faces][:,0],
                             xyz3[faces][:,2]-xyz3[faces][:,0]), axis=1))),1e-12)):
        raise PlanarQuadMeshingError(
            "PLANAR_QUAD_PANEL_QUALITY_INVALID: "
            f"m2d={q2}; m3d={q3}; no invalid panel accepted")
    report = {
        "planar_quad_engine": "Gmsh frontal-delaunay+recombine; NOT Zhang2021 reference code",
        "planar_quad_method": MODE,
        "planar_quad_faces": int(len(faces)),
        "planar_quad_target_faces": int(target_faces),
        "planar_quad_target_edge_length": float(h),
        **boundary_report,
        "planar_quad_gmsh_boundary_curve_count": int(len(boundary)),
        "planar_quad_gmsh_boundary_curve_elements": int(np.sum(curve_elements)),
        "planar_quad_gmsh_per_curve_even_subdivisions": bool(
            np.all(curve_elements >= 2) and np.all(curve_elements % 2 == 0)),
        "planar_quad_omega_undercoverage_fraction": float(
            polygon.difference(union).area / polygon.area),
        "planar_quad_omega_overcoverage_fraction": float(
            union.difference(polygon).area / polygon.area),
        "planar_quad_uv_inverse_exact_piecewise_linear": True,
        **{"planar_quad_m2d_" + k: int(v) if k.endswith("_face_index") else float(v)
           for k, v in q2.items()},
        **{"planar_quad_m3d_" + k: int(v) if k.endswith("_face_index") else float(v)
           for k, v in q3.items()},
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
            result.metrics["requested_omega_parameterization_mode"] = (
                "optcuts_zhang2021_reference_only"
                if bool(getattr(params, "optcuts_zhang2021_reference_enabled", False))
                else "optcuts_zhang2021_coarse_layout_experiment"
                if bool(getattr(params, "optcuts_zhang2021_experimental_enabled", False))
                else MODE)
            # Preserve the ORIGINAL OptCuts domain for a faithful source run.
            # Arbitrary harmonic smoothing is prohibited in reference-only mode.
            if (bool(getattr(params, "optcuts_zhang2021_reference_enabled", False))
                    and bool(getattr(params, "planar_quad_uv_reparam", False))):
                raise PlanarQuadMeshingError(
                    "ZHANG_REFERENCE_SOURCE_UV_MUST_NOT_BE_PREMODIFIED")
            result.metrics["planar_quad_source_seam_policy"] = "original OptCuts fixed"
            result.metrics["planar_quad_20261026_enabled"] = True
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
        if not bool(getattr(params, "optcuts_planar_quad_20261026_enabled", False)):
            return base_domain(parameterization, grid, params)
        # The previous prototype invoked the regular cropped-grid domain
        # builder even in this mode. That route constructs an entirely
        # unnecessary rectangular overlay and can fail on the SAME Ω that
        # the independent planar quad mesher should handle. Construct only
        # the audited source chart and real area-Jacobian metadata.
        if params is None:
            raise PlanarQuadMeshingError("PLANAR_QUAD_DOMAIN_MISSING_PARAMETERS")
        original = getattr(pipeline, "_original", None)
        cls = getattr(original, "PlanarDomain", None)
        area_metrics_fn = getattr(pipeline, "_paper_lambda_vertices_for_existing_omega", None)
        if cls is None or area_metrics_fn is None:
            raise PlanarQuadMeshingError(
                "PLANAR_QUAD_DOMAIN_DEPENDENCY_UNAVAILABLE: source area field")
        vertex, triangle, differential = area_metrics_fn(parameterization)
        vertex = np.asarray(vertex, float)
        triangle = np.asarray(triangle, float)
        if (len(vertex) != len(parameterization.uv_vertices_2d)
                or not len(triangle) or not np.isfinite(triangle).all()
                or not np.isfinite(vertex).all()):
            raise PlanarQuadMeshingError("PLANAR_QUAD_DOMAIN_CSF_INVALID")
        threshold = float(getattr(params, "csf_split_threshold", 2.))
        if not np.isfinite(threshold) or threshold < 1.:
            raise PlanarQuadMeshingError("PLANAR_QUAD_DOMAIN_CSF_THRESHOLD_INVALID")
        result = cls(
            np.asarray(parameterization.omega_boundary, float).copy(),
            np.asarray(parameterization.uv_vertices_2d, float).copy(),
            f"{parameterization.method} surface parameterization + independent planar quad",
            vertex.copy(), [])
        result.parameterization = parameterization
        result._planar_quad_source = parameterization
        result._planar_quad_target_faces = int(getattr(params, "planar_quad_target_faces", 35))
        result.paper_split_policy = True
        result.csf_before = float(np.max(triangle))
        result.csf_after_split = result.csf_before
        result.csf_split_threshold = threshold
        result.max_csf_splits = max(0, int(getattr(params, "max_csf_splits", 64)))
        result.csf_split_enabled = bool(getattr(params, "enable_csf_splits", True))
        result.csf_model = "original paired UV/3D triangle area Jacobian"
        result.csf_split_exactness_label = "pending unstructured M2D Split audit"
        result.paper_split_diagnostics = {
            "status": "pending_unstructured_m2d",
            "area_expansion_bound": threshold,
            "lambda_max": result.csf_before,
            "split_rule": "complete existing quad-edge-chain only",
        }
        result.localize_csf_splits = False
        result.localized_split_segments = []
        result.reference_mode = False
        parameterization.metrics["split_diagnostics"] = dict(result.paper_split_diagnostics)
        parameterization.metrics["planar_quad_independent_domain"] = True
        parameterization.metrics["planar_quad_legacy_regular_overlay_bypassed"] = True
        return result

    def with_m2d(grid, domain, params=None):
        if not bool(getattr(params, "optcuts_planar_quad_20261026_enabled", False)):
            return base_m2d(grid, domain, params)
        if not hasattr(domain, "_planar_quad_source"):
            raise PlanarQuadMeshingError("PLANAR_QUAD_MISSING_OPTCUTS_SOURCE")
        if bool(getattr(params, "optcuts_zhang2021_reference_enabled", False)):
            if bool(getattr(params, "optcuts_zhang2021_experimental_enabled", False)):
                raise PlanarQuadMeshingError("ZHANG_BACKENDS_MUTUALLY_EXCLUSIVE")
            import shlex
            from .zhang2021_reference_bridge import run_reference_adapter
            cmd = shlex.split(str(getattr(
                params, "zhang2021_reference_adapter_command", "")).strip())
            if not cmd:
                # User may install the ORIGINAL author executables separately.
                # This is an exact native 3-stage adapter, NEVER a generator
                # substitution; source ZIP/license stay outside this repo.
                import os
                import sys
                from pathlib import Path
                native_root = (
                    str(getattr(params, "zhang2021_author_exe_root", "")).strip()
                    or os.environ.get("ONESTRING_ZHANG2021_EXE_ROOT", "").strip())
                if native_root:
                    adapter = (Path(__file__).resolve().parents[2]
                               / "scripts/zhang2021_authors_native_adapter.py")
                    if not adapter.is_file():
                        raise PlanarQuadMeshingError(
                            "ZHANG_AUTHORS_NATIVE_ADAPTER_SCRIPT_MISSING")
                    cmd = [
                        sys.executable, str(adapter), "--input", "{input}",
                        "--output", "{output}", "--target-quads", "{target_quads}",
                        "--exe-root", str(Path(native_root).resolve()),
                    ]
                    wine = (
                        str(getattr(params, "zhang2021_author_wine_command", "")).strip()
                        or os.environ.get(
                            "ONESTRING_ZHANG2021_WINE_COMMAND", "").strip())
                    if wine:
                        cmd.extend(["--wine-command", *shlex.split(wine)])
                    numeric = (
                        str(getattr(params, "zhang2021_independent_solver_dll", "")).strip()
                        or os.environ.get(
                            "ONESTRING_ZHANG2021_INDEPENDENT_EIGEN_DLL", "").strip())
                    if numeric:
                        dll = Path(numeric).expanduser().resolve()
                        if not dll.is_file():
                            raise PlanarQuadMeshingError(
                                "ZHANG_INDEPENDENT_EIGEN_DLL_NOT_FOUND: " + str(dll))
                        cmd.extend(["--independent-eigen-solver-dll", str(dll)])
                    haus_bound = float(getattr(
                        params, "zhang2021_author_hausdorff_bound", .01))
                    sheet_width = float(getattr(
                        params, "zhang2021_author_sheet_width", .05))
                    if (not np.isfinite(haus_bound) or haus_bound <= 0
                            or not np.isfinite(sheet_width) or sheet_width <= 0):
                        raise PlanarQuadMeshingError(
                            "ZHANG_ORIGINAL_NUMERIC_PARAMETERS_INVALID")
                    cmd.extend([
                        "--hausdorff-bound", str(haus_bound),
                        "--sheet-width", str(sheet_width),
                    ])
            audited = run_reference_adapter(
                domain._planar_quad_source, cmd,
                target_quads=int(domain._planar_quad_target_faces),
                require_seam_correspondence=True)
            uv, faces = audited.vertices_uv, audited.faces
            report = dict(audited.diagnostics)
            report["planar_quad_engine"] = (
                "external reference-only adapter; algorithm identity UNVERIFIED")
            report["planar_quad_target_faces"] = int(
                domain._planar_quad_target_faces)
            report["planar_quad_faces"] = int(len(faces))
            report["planar_quad_count_target_met"] = bool(
                len(faces) <= int(domain._planar_quad_target_faces))
            report["planar_quad_count_target_excess"] = int(max(
                0, len(faces) - int(domain._planar_quad_target_faces)))
            report["planar_quad_reference_identity_verified"] = False
            q2 = _quad_quality(
                np.column_stack((uv, np.zeros(len(uv)))), faces)
            q3 = _quad_quality(audited.vertices_3d, faces)
            for name, value in q2.items():
                report["planar_quad_m2d_" + name] = value
            for name, value in q3.items():
                report["planar_quad_m3d_" + name] = value
        elif bool(getattr(params, "optcuts_zhang2021_experimental_enabled", False)):
            # OneString's strict preflight applies to both engines.
            # This is a real, independent quad generator, not the authors'
            # C++ Zhang 2021 reference method. Never fall back to Gmsh.
            from .zhang2021_field_layout_experiment import generate_field_layout
            candidate = generate_field_layout(
                domain._planar_quad_source,
                domain._planar_quad_target_faces,
                require_seam_correspondence=True,
            )
            uv, faces, report = (
                candidate.uv, candidate.quads, dict(candidate.metrics))
        else:
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
        from .optcuts_seam_exact_closure import (
            discover_exact_seam_topology, SeamTopologyMismatch)
        try:
            seam = discover_exact_seam_topology(domain._planar_quad_source, uv, faces)
        except SeamTopologyMismatch as exc:
            raise PlanarQuadMeshingError(
                "PLANAR_QUAD_SOURCE_SEAM_PREFLIGHT_FAILED: " + str(exc)
            ) from exc
        mesh.metrics.update({
            "planar_quad_seam_preflight_status": str(seam["status"]),
            "planar_quad_seam_preflight_stitch_mode": str(
                seam.get("seam_stitch_mode", "none")),
            "planar_quad_seam_preflight_unmatched": int(seam.get("unmatched_edges",0)),
            "planar_quad_seam_preflight_unclassified": int(
                seam.get("unclassified_boundary_edges",0)),
        })
        if str(seam["status"]) == "nonconforming":
            raise PlanarQuadMeshingError(
                "PLANAR_QUAD_SEAM_NOT_CONFORMING: boundary quad edges "
                "cannot be proven to follow the paired original OptCuts cut. "
                f"unmatched={seam.get('unmatched_edges')} "
                f"unclassified={seam.get('unclassified_boundary_edges')}; "
                "accepting this candidate would break inverse correspondence "
                "or seam closure. A seam-aware remeshing stage is still needed."
            )
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
            # The Split module can duplicate vertex IDs and disconnect panels.
            # The original irregular grid still contains PRE-SPLIT face IDs.
            # Reconstruct adjacency from the actual final faces only; no
            # guessed row/col, no fabricated hinge and no face reordering.
            post_faces = np.asarray(mesh.faces, int)
            post_uv = np.asarray(mesh.vertices, float)[:, :2]
            changed = len(post_uv) != len(uv) or not np.array_equal(post_faces, faces)
            if changed:
                mesh.grid = UnstructuredQuadGrid(
                    post_uv, post_faces, effective_size, effective_gap)
            mesh.metrics["planar_quad_post_split_grid_reconstructed"] = bool(changed)
            mesh.metrics["planar_quad_post_split_face_count"] = int(len(post_faces))
            mesh.metrics["planar_quad_post_split_actual_hinge_count"] = int(len(mesh.grid.hinges))
            if (bool(getattr(params, "enable_csf_splits", True))
                    and float(getattr(domain, "csf_before", 1.)) >
                        float(getattr(params, "csf_split_threshold", 2.))
                    and not mesh.metrics["planar_quad_csf_split_certified"]):
                raise PlanarQuadMeshingError(
                    "PLANAR_QUAD_CSF_SPLIT_UNCERTIFIED: "
                    f"source_lambda_max={float(domain.csf_before):.6g} "
                    f"threshold={float(getattr(params, 'csf_split_threshold', 2.)):.6g}; "
                    "unstructured quad edge-chain Split did not certify the bound")
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
