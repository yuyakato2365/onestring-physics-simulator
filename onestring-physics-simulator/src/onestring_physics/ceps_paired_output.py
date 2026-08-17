"""Validate official CEPS ordinary-UV output as one seam-free disk chart.

The official CEPS common-refinement OBJ stores texture coordinates per face
corner.  A surface vertex may therefore have several UV values when the CEPS
layout contains a cut seam.  The current OneString M2D -> M3D stage requires one
single-valued UV coordinate per surface vertex; it cannot faithfully consume a
multi-chart/projective CEPS layout.

Older revisions tried to stitch cut copies with independent 2D similarities.
That reconstruction is not part of CEPS and can change the map, bridge the
physical boundary, and visually cap an open surface.  This module now accepts
only an already seam-free ordinary-UV chart.  Unsupported cut charts fail
explicitly instead of being replaced by a convex hull, nearest mapping, or an
invented planar development.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


_EPS = 1e-12


def _obj_index(value: str, count: int) -> int:
    index = int(value)
    return index - 1 if index > 0 else count + index


def _triangulate_obj(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    vertices: list[list[float]] = []
    textures: list[list[float]] = []
    polygons: list[list[tuple[int, int]]] = []

    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        fields = raw.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(list(map(float, fields[1:4])))
        elif fields[0] == "vt" and len(fields) >= 3:
            textures.append(list(map(float, fields[1:3])))
        elif fields[0] == "f" and len(fields) >= 4:
            refs: list[tuple[int, int]] = []
            for token in fields[1:]:
                parts = token.split("/")
                if len(parts) < 2 or not parts[0] or not parts[1]:
                    raise RuntimeError(
                        f"CEPS OBJ face lacks texture indices at line {line_number}"
                    )
                refs.append(
                    (
                        _obj_index(parts[0], len(vertices)),
                        _obj_index(parts[1], len(textures)),
                    )
                )
            polygons.append(refs)

    if not vertices or not textures or not polygons:
        raise RuntimeError("official CEPS output OBJ is incomplete")

    surface_faces: list[list[int]] = []
    texture_faces: list[list[int]] = []
    for polygon in polygons:
        for index in range(1, len(polygon) - 1):
            triangle = [polygon[0], polygon[index], polygon[index + 1]]
            surface_faces.append([item[0] for item in triangle])
            texture_faces.append([item[1] for item in triangle])

    surface = np.asarray(vertices, dtype=float)
    raw_uv = np.asarray(textures, dtype=float)[:, :2]
    faces = np.asarray(surface_faces, dtype=int)
    texture_ids = np.asarray(texture_faces, dtype=int)
    if (
        np.min(faces) < 0
        or np.max(faces) >= len(surface)
        or np.min(texture_ids) < 0
        or np.max(texture_ids) >= len(raw_uv)
    ):
        raise RuntimeError("official CEPS output OBJ contains an invalid index")
    if not np.all(np.isfinite(surface)) or not np.all(np.isfinite(raw_uv)):
        raise RuntimeError("official CEPS output OBJ contains non-finite coordinates")

    return surface, faces, raw_uv[texture_ids], len(raw_uv)


def _canonical_edge_sample(
    a: int,
    b: int,
    uv_a: np.ndarray,
    uv_b: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray]:
    if a < b:
        return (a, b), np.asarray([uv_a, uv_b], dtype=float)
    return (b, a), np.asarray([uv_b, uv_a], dtype=float)


def _validate_single_valued_chart(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    face_uv: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    faces = np.asarray(surface_faces, dtype=int)[:, :3]
    local_uv = np.asarray(face_uv, dtype=float)[:, :3, :2]
    if len(faces) == 0:
        raise RuntimeError("official CEPS common refinement has no triangles")

    uv_span = float(np.max(np.ptp(local_uv.reshape(-1, 2), axis=0)))
    tolerance = max(1e-10, 1e-8 * max(uv_span, 1.0))

    vertex_samples: dict[int, list[np.ndarray]] = defaultdict(list)
    edge_samples: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for face, triangle_uv in zip(faces, local_uv):
        for local_index, vertex_id in enumerate(face):
            vertex_samples[int(vertex_id)].append(triangle_uv[local_index].copy())
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            key, sample = _canonical_edge_sample(
                int(face[ia]),
                int(face[ib]),
                triangle_uv[ia],
                triangle_uv[ib],
            )
            edge_samples[key].append(sample)

    nonmanifold_edges = [
        edge for edge, samples in edge_samples.items() if len(samples) > 2
    ]
    if nonmanifold_edges:
        raise RuntimeError(
            "official CEPS common refinement is non-manifold at "
            f"{len(nonmanifold_edges)} edges"
        )

    seam_edges: list[tuple[int, int]] = []
    for edge, samples in edge_samples.items():
        if len(samples) != 2:
            continue
        mismatch = float(np.max(np.linalg.norm(samples[0] - samples[1], axis=1)))
        if mismatch > tolerance:
            seam_edges.append(edge)

    uv_vertices = np.full((len(surface_vertices), 2), np.nan, dtype=float)
    multivalued_vertices: list[int] = []
    for vertex_id, samples in vertex_samples.items():
        values = np.asarray(samples, dtype=float)
        center = np.mean(values, axis=0)
        deviation = float(np.max(np.linalg.norm(values - center, axis=1)))
        if deviation > tolerance:
            multivalued_vertices.append(int(vertex_id))
        uv_vertices[int(vertex_id)] = center

    unused_vertices = np.flatnonzero(~np.all(np.isfinite(uv_vertices), axis=1))
    if len(unused_vertices):
        raise RuntimeError(
            "official CEPS common refinement contains "
            f"{len(unused_vertices)} surface vertices without UV samples"
        )

    if seam_edges or multivalued_vertices:
        raise RuntimeError(
            "official CEPS ordinary-UV output contains internal cut seams "
            f"({len(seam_edges)} seam edges, "
            f"{len(multivalued_vertices)} multi-valued vertices). "
            "The current OneString pipeline requires one continuous single-valued "
            "UV disk. It will not invent a stitched chart, use a convex hull, or "
            "apply nearest-triangle fallback. A chart-aware/projective CEPS "
            "integration is required for this output."
        )

    boundary_edge_count = sum(
        1 for samples in edge_samples.values() if len(samples) == 1
    )
    metrics: dict[str, Any] = {
        "ceps_continuous_chart_reconstructed": False,
        "ceps_ordinary_uv_chart_already_single_valued": True,
        "ceps_internal_cut_seam_edge_count": 0,
        "ceps_internal_cut_seams_stitched": False,
        "ceps_internal_cut_seams_supported": False,
        "ceps_common_refinement_surface_vertex_count_before_stitch": int(
            len(surface_vertices)
        ),
        "ceps_common_refinement_surface_vertex_count_after_stitch": int(
            len(surface_vertices)
        ),
        "ceps_physical_boundary_edge_count": int(boundary_edge_count),
        "ceps_uv_single_value_tolerance": float(tolerance),
        "ceps_convex_hull_boundary_used": False,
        "ceps_nearest_inverse_fallback_allowed": False,
        "ceps_artificial_cap_faces_added": 0,
    }
    return uv_vertices, metrics


def _seam_free_parser(module: Any, path: Path) -> Any:
    surface, faces, face_uv, raw_texture_count = _triangulate_obj(path)
    uv_vertices, metrics = _validate_single_valued_chart(surface, faces, face_uv)
    module._CEPS_LAST_CHART_METRICS = dict(metrics)
    vertex_uv = {
        int(index): uv_vertices[index].copy() for index in range(len(uv_vertices))
    }
    return module.CepsObjResult(
        surface,
        faces,
        uv_vertices,
        faces.copy(),
        vertex_uv,
        raw_texture_count,
    )


def install_ceps_paired_output(module: Any) -> None:
    """Install strict seam-free ordinary-UV validation into ``official_ceps``.

    The historical function name is retained for import compatibility. No seam
    vertices are duplicated or stitched by this implementation.
    """
    if getattr(module, "_CEPS_PAIRED_OUTPUT_INSTALLED", False):
        return

    def parse(path: Path) -> Any:
        return _seam_free_parser(module, path)

    module._parse_ceps_obj = parse
    module._CEPS_PAIRED_OUTPUT_INSTALLED = True
