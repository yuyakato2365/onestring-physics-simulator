"""Isolated research bridge for the ORIGINAL Zhang et al. (CAD 2021) backend.

This module DOES NOT implement or claim to reproduce Zhang's algorithm.
It defines an explicit, testable artifact contract for an independently
verified reference-code adapter, then audits its quad output for OneString.
No Gmsh, regular-grid crop, or algorithmic fallback is ever performed.

Stage A is genuinely unavailable until the authors' C++ source/executable,
its license, and its true CLI/data formats have been inspected.  The NPZ
contract below is for our OWN small wrapper around that future binary,
NOT a claim that the authors' binary already accepts NPZ files.

Source:
https://doi.org/10.1016/j.cad.2021.103084
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
from typing import Any, Sequence

import numpy as np


class Zhang2021BridgeError(RuntimeError):
    """Failed closed: the reference result was missing or invalid."""


@dataclass(frozen=True)
class AuditedQuadMesh:
    """Post-validated result; this does not certify the reference algorithm."""
    vertices_uv: np.ndarray
    faces: np.ndarray
    vertices_3d: np.ndarray
    diagnostics: dict[str, Any]


def _source_arrays(source: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        uv = np.asarray(source.uv_vertices_2d, float)
        uv_faces = np.asarray(source.uv_faces, int)
        xyz = np.asarray(source.surface_vertices_3d, float)
        xyz_faces = np.asarray(source.surface_faces, int)
        boundary = np.asarray(source.omega_boundary, float)[:, :2]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise Zhang2021BridgeError("SOURCE_CHART_MISSING_OR_INVALID") from exc
    if (uv.ndim != 2 or uv.shape[1] != 2 or xyz.ndim != 2 or xyz.shape[1] != 3
            or uv_faces.ndim != 2 or uv_faces.shape[1] != 3
            or xyz_faces.shape != uv_faces.shape or len(uv_faces) == 0
            or boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3):
        raise Zhang2021BridgeError("SOURCE_CHART_INVALID_SHAPES")
    if not all(np.isfinite(a).all() for a in (uv, xyz, boundary)):
        raise Zhang2021BridgeError("SOURCE_CHART_NONFINITE")
    if (np.any(uv_faces < 0) or np.any(uv_faces >= len(uv))
            or np.any(xyz_faces < 0) or np.any(xyz_faces >= len(xyz))):
        raise Zhang2021BridgeError("SOURCE_CHART_INVALID_TRIANGLE_INDEX")
    return uv, uv_faces, xyz, xyz_faces, boundary


def export_source_chart(source: Any, file_path: str | Path) -> None:
    """Write immutable OptCuts provenance into our documented NPZ contract."""
    uv, uv_faces, xyz, xyz_faces, boundary = _source_arrays(source)
    np.savez_compressed(file_path, source_uv_vertices=uv,
                        source_uv_faces=uv_faces,
                        source_xyz_vertices=xyz,
                        source_xyz_faces=xyz_faces,
                        omega_boundary=boundary)


def audit_reference_quads(
    source: Any, vertices_uv: np.ndarray, quad_faces: np.ndarray, *,
    max_undercoverage_fraction: float = 1e-7,
    max_overcoverage_fraction: float = 1e-7,
    require_seam_correspondence: bool = True,
) -> AuditedQuadMesh:
    """OneString preflight; never silently crop, project or repair a candidate.

    Geometrically verifies all faces and the source domain.  Confirms each
    output vertex, edge midpoint and quad center has an exact barycentric
    inverse in the original paired OptCuts triangles.  When required, runs
    the existing *edge-level* OptCuts seam correspondence audit.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .optcuts_planar_quad_refine_20261026 import validate_disk_mesh
    from .optcuts_planar_quad_20261026 import _triangle_inverse
    from .optcuts_seam_exact_closure import (
        discover_exact_seam_topology, SeamTopologyMismatch)

    original_uv, original_faces, xyz, xyz_faces, boundary = _source_arrays(source)
    qv = np.asarray(vertices_uv, float)
    qf = np.asarray(quad_faces, int)
    if qv.ndim != 2 or qv.shape[1] != 2 or not np.isfinite(qv).all():
        raise Zhang2021BridgeError("QUAD_UV_INVALID_SHAPE_OR_NONFINITE")
    if not 0 <= max_undercoverage_fraction < 1:
        raise Zhang2021BridgeError("INVALID_UNDERCOVERAGE_POLICY")
    if not 0 <= max_overcoverage_fraction < 1:
        raise Zhang2021BridgeError("INVALID_OVERCOVERAGE_POLICY")
    try:
        validate_disk_mesh(qv, qf)
    except (ValueError, IndexError) as exc:
        raise Zhang2021BridgeError(f"INVALID_QUAD_TOPOLOGY: {exc}") from exc

    source_polygon = Polygon(boundary)
    if (not source_polygon.is_valid or source_polygon.area <= 0
            or len(source_polygon.interiors) != 0):
        raise Zhang2021BridgeError("SOURCE_OMEGA_INVALID_OR_NON_DISK")

    # Check original paired UV triangles cover the declared boundary.
    source_triangles = [Polygon(original_uv[tri]) for tri in original_faces]
    if not all(p.is_valid and p.area > 0 for p in source_triangles):
        raise Zhang2021BridgeError("SOURCE_UV_TRIANGLE_INVALID")
    source_union = unary_union(source_triangles)
    original_discrepancy = float(source_union.symmetric_difference(source_polygon).area)
    if original_discrepancy > 1e-7 * source_polygon.area:
        raise Zhang2021BridgeError(
            "SOURCE_BOUNDARY_TRIANGLE_MISMATCH: "
            f"fraction={original_discrepancy / source_polygon.area:.9g}")

    polygons = [Polygon(qv[face]) for face in qf]
    # Non-convex quads are inappropriate for planar rigid tiles.
    corners = qv[qf]
    edges = np.roll(corners, -1, axis=1) - corners
    signed_turns = np.cross(edges, np.roll(edges, -1, axis=1))
    signed_areas = 0.5 * np.sum(
        corners[:, :, 0] * np.roll(corners[:, :, 1], -1, axis=1)
        - corners[:, :, 1] * np.roll(corners[:, :, 0], -1, axis=1),
        axis=1)
    # Clockwise source meshes are valid. The requirement is a consistent
    # orientation and convex turns, NOT globally positive winding.
    if (np.any(np.abs(signed_areas) <= 1e-12 * source_polygon.area)
            or np.any(signed_turns * np.sign(signed_areas)[:, None]
                      <= 1e-12 * source_polygon.area)):
        raise Zhang2021BridgeError("QUAD_CONCAVE_OR_DEGENERATE_CORNER")

    union = unary_union(polygons)
    area_sum = sum(p.area for p in polygons)
    overlap = float(area_sum - union.area)
    if overlap > 1e-9 * source_polygon.area:
        raise Zhang2021BridgeError("QUAD_OVERLAP")
    outside = float(union.difference(source_polygon).area / source_polygon.area)
    missing = float(source_polygon.difference(union).area / source_polygon.area)
    if outside > max_overcoverage_fraction:
        raise Zhang2021BridgeError(
            f"QUAD_OUTSIDE_SOURCE_OMEGA: fraction={outside:.9g}")
    if missing > max_undercoverage_fraction:
        raise Zhang2021BridgeError(
            f"QUAD_UNDERCOVER_SOURCE_OMEGA: fraction={missing:.9g}")

    midpoints = 0.5 * (corners + np.roll(corners, -1, axis=1))
    samples = np.vstack((qv, midpoints.reshape(-1, 2), corners.mean(axis=1)))
    try:
        mapped = _triangle_inverse(original_uv, original_faces,
                                   xyz, xyz_faces, samples)
    except Exception as exc:
        raise Zhang2021BridgeError(
            "QUAD_SAMPLE_HAS_NO_EXACT_SOURCE_TRIANGLE_INVERSE: " + str(exc)
        ) from exc
    lifted = mapped[:len(qv)]

    seam_status = "unchecked"
    seam_unmatched = -1
    if require_seam_correspondence:
        try:
            seam = discover_exact_seam_topology(source, qv, qf)
        except (SeamTopologyMismatch, ValueError, IndexError) as exc:
            raise Zhang2021BridgeError(
                "QUAD_SEAM_PREFLIGHT_FAILED: " + str(exc)) from exc
        seam_status = str(seam["status"])
        seam_unmatched = int(seam.get("unmatched_edges", 0))
        if seam_status not in ("conforming", "source_open_boundary_only", "no_boundary"):
            raise Zhang2021BridgeError(
                f"QUAD_SEAM_NOT_CONFORMING: status={seam_status} "
                f"unmatched={seam_unmatched}")

    # This report documents validation, never algorithm identity or fidelity.
    return AuditedQuadMesh(
        vertices_uv=qv.copy(), faces=qf.copy(), vertices_3d=lifted.copy(),
        diagnostics={
            "engine": "external_adapter_output",
            "zhang2021_algorithm_verified": False,
            "status": "onestring_geometric_preflight_passed",
            "quad_count": int(len(qf)),
            "omega_undercoverage_fraction": missing,
            "omega_overcoverage_fraction": outside,
            "seam_status": seam_status,
            "seam_unmatched_edges": seam_unmatched,
            "checked_inverse_samples": int(len(samples)),
            "inverse_method": "original_optcuts_triangle_barycentric",
        },
    )


def run_reference_adapter(
    source: Any, adapter_command: Sequence[str], *,
    timeout_seconds: int = 1200,
    target_quads: int = 35,
    require_seam_correspondence: bool = True,
) -> AuditedQuadMesh:
    """Execute ONLY a caller-supplied, separately verified adapter.

    The command must include literal {input} and {output} placeholder tokens.
    Our adapter is responsible for translating NPZ to the REAL authors' native
    format and back; no assumptions about their executable's CLI are made.

    Adapter input NPZ: source_uv_vertices, source_uv_faces,
        source_xyz_vertices, source_xyz_faces, omega_boundary.
    Adapter output NPZ: vertices_uv (Nx2), faces (Mx4).
    No automatic Gmsh fallback; errors are explicit.
    """
    cmd = list(adapter_command)
    if (not cmd or not all(isinstance(x, str) and x for x in cmd)
            or not any("{input}" in x for x in cmd)
            or not any("{output}" in x for x in cmd)):
        raise Zhang2021BridgeError(
            "REFERENCE_ADAPTER_NOT_CONFIGURED: supply a verified external "
            "adapter command with {input} and {output}; the original Zhang "
            "executable's CLI has NOT been established")
    if timeout_seconds < 1:
        raise Zhang2021BridgeError("INVALID_ADAPTER_TIMEOUT")
    if int(target_quads) < 4:
        raise Zhang2021BridgeError("INVALID_NATIVE_TARGET_QUAD_COUNT")
    with TemporaryDirectory(prefix="onestring_zhang2021_") as directory:
        source_file = Path(directory) / "optcuts_source.npz"
        output_file = Path(directory) / "candidate_quads.npz"
        export_source_chart(source, source_file)
        expanded = [arg.replace("{input}", str(source_file))
                        .replace("{output}", str(output_file))
                        .replace("{target_quads}", str(int(target_quads))) for arg in cmd]
        try:
            completed = subprocess.run(
                expanded, check=False, capture_output=True, text=True,
                timeout=timeout_seconds, cwd=directory, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Zhang2021BridgeError(
                "REFERENCE_ADAPTER_EXECUTION_FAILED: " + str(exc)) from exc
        if completed.returncode:
            raise Zhang2021BridgeError(
                f"REFERENCE_ADAPTER_EXIT_{completed.returncode}: "
                + (completed.stderr or completed.stdout)[-1600:])
        if not output_file.is_file():
            raise Zhang2021BridgeError(
                "REFERENCE_ADAPTER_DID_NOT_PRODUCE_QUAD_OUTPUT")
        try:
            with np.load(output_file, allow_pickle=False) as output:
                vertices = output["vertices_uv"].copy()
                faces = output["faces"].copy()
        except (OSError, KeyError, ValueError) as exc:
            raise Zhang2021BridgeError(
                "REFERENCE_ADAPTER_OUTPUT_INVALID") from exc
        # The native adapter writes provenance beside its output; retain it
        # before the temp directory is deleted. This claim does not by itself
        # establish that an arbitrary external command is the authors' binary.
        claimed = {}
        manifest_path = output_file.with_suffix(".provenance.json")
        if manifest_path.is_file():
            try:
                claimed = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise Zhang2021BridgeError("REFERENCE_PROVENANCE_CORRUPT") from exc
    result = audit_reference_quads(
        source, vertices, faces,
        require_seam_correspondence=require_seam_correspondence,
    )
    result.diagnostics["reference_manifest_present"] = bool(claimed)
    result.diagnostics["reference_backend_declared"] = str(
        claimed.get("engine", "unverified_adapter"))
    result.diagnostics["authors_native_stages_reported"] = [
        str(step.get("program", "")) for step in claimed.get("stages", [])
        if isinstance(step, dict)
    ]
    return result


def to_onestring_m2d(audit: AuditedQuadMesh, *, tile_size: float,
                     gap_size: float) -> Any:
    """Produce existing pipeline's true face-indexed mesh after validation."""
    from .optcuts_planar_quad_20261026 import UnstructuredQuadGrid
    from .legacy_pipeline_base import QuadMesh

    uv, faces = audit.vertices_uv, audit.faces
    if tile_size <= 0 or gap_size < 0:
        raise Zhang2021BridgeError("INVALID_TILE_SIZE_OR_GAP")
    grid = UnstructuredQuadGrid(uv, faces, float(tile_size), float(gap_size))
    xyz = np.column_stack((uv, np.zeros(len(uv))))
    return QuadMesh(xyz, faces.copy(), grid, "M2D",
                    dict(audit.diagnostics), [])
