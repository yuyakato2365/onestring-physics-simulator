"""Convert official CEPS per-corner UV output into one continuous disk chart.

CEPS lays out the doubled surface after inserting a cut graph. Its OBJ therefore
stores texture coordinates per face corner, and an internal cut can give one
common-refinement surface vertex several UV copies. OneString cannot treat those
copies as independent charts: its M2D grid assumes one continuous map from the
input disk to Omega. This module stitches the CEPS cut copies back together by
walking the *surface* face adjacency and aligning adjacent target triangles by a
2D similarity across every shared edge.

The result keeps the official CEPS common-refinement surface connectivity, uses
one UV coordinate per common-refinement surface vertex, and preserves the true
physical boundary. No convex hull and no artificial cap faces are introduced.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


_EPS = 1e-12


def _obj_index(value: str, count: int) -> int:
    index = int(value)
    return index - 1 if index > 0 else count + index


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _triangulate_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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
    return surface, faces, raw_uv[texture_ids], texture_ids, len(raw_uv)


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    incidence: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(np.asarray(faces, dtype=int)[:, :3]):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            incidence[tuple(sorted((int(a), int(b))))].append(int(face_id))
    bad = [edge for edge, incident in incidence.items() if len(incident) > 2]
    if bad:
        raise RuntimeError(
            f"official CEPS common refinement is non-manifold at {len(bad)} edges"
        )
    return incidence


def _candidate_transform(
    local_triangle: np.ndarray,
    face: np.ndarray,
    edge: tuple[int, int],
    global_uv: np.ndarray,
    reference_face: np.ndarray,
) -> tuple[np.ndarray, float]:
    a, b = map(int, edge)
    local_a = int(np.flatnonzero(face == a)[0])
    local_b = int(np.flatnonzero(face == b)[0])
    qa, qb = local_triangle[local_a], local_triangle[local_b]
    pa, pb = global_uv[a], global_uv[b]

    local_length = float(np.linalg.norm(qb - qa))
    global_length = float(np.linalg.norm(pb - pa))
    if local_length <= _EPS or global_length <= _EPS:
        raise RuntimeError("official CEPS produced a degenerate cut edge")

    local_axis = (qb - qa) / local_length
    local_perp = np.asarray([-local_axis[1], local_axis[0]], dtype=float)
    global_axis = (pb - pa) / global_length
    global_perp = np.asarray([-global_axis[1], global_axis[0]], dtype=float)
    scale = global_length / local_length

    reference_third = next(int(v) for v in reference_face if int(v) not in edge)
    reference_side = _cross2(pb - pa, global_uv[reference_third] - pa)

    candidates: list[tuple[float, np.ndarray]] = []
    for reflection in (1.0, -1.0):
        transformed: list[np.ndarray] = []
        for point in local_triangle:
            relative = point - qa
            x = float(np.dot(relative, local_axis))
            y = float(np.dot(relative, local_perp))
            transformed.append(
                pa + scale * (x * global_axis + reflection * y * global_perp)
            )
        transformed_array = np.asarray(transformed, dtype=float)
        neighbor_third_local = next(
            index for index, vertex in enumerate(face) if int(vertex) not in edge
        )
        neighbor_side = _cross2(
            pb - pa, transformed_array[neighbor_third_local] - pa
        )
        score = reference_side * neighbor_side
        candidates.append((score, transformed_array))

    opposite = [item for item in candidates if item[0] < -1e-14]
    chosen = min(opposite or candidates, key=lambda item: item[0])
    relative_edge_error = abs(local_length - global_length) / max(
        local_length, global_length, _EPS
    )
    return chosen[1], float(relative_edge_error)


def _stitch_cut_chart(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    face_uv: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    faces = np.asarray(surface_faces, dtype=int)[:, :3]
    local_uv = np.asarray(face_uv, dtype=float)[:, :3, :2]
    if len(faces) == 0:
        raise RuntimeError("official CEPS common refinement has no triangles")

    incidence = _edge_incidence(faces)
    adjacency: list[list[tuple[int, tuple[int, int]]]] = [
        [] for _ in range(len(faces))
    ]
    seam_edge_count = 0
    for edge, incident in incidence.items():
        if len(incident) == 2:
            a, b = incident
            adjacency[a].append((b, edge))
            adjacency[b].append((a, edge))
            samples: list[np.ndarray] = []
            for face_id in incident:
                face = faces[face_id]
                samples.append(
                    np.asarray(
                        [
                            local_uv[face_id, int(np.flatnonzero(face == edge[0])[0])],
                            local_uv[face_id, int(np.flatnonzero(face == edge[1])[0])],
                        ]
                    )
                )
            direct = max(
                float(np.linalg.norm(samples[0][0] - samples[1][0])),
                float(np.linalg.norm(samples[0][1] - samples[1][1])),
            )
            swapped = max(
                float(np.linalg.norm(samples[0][0] - samples[1][1])),
                float(np.linalg.norm(samples[0][1] - samples[1][0])),
            )
            if min(direct, swapped) > 1e-9:
                seam_edge_count += 1

    global_uv = np.full((len(surface_vertices), 2), np.nan, dtype=float)
    placed_faces = np.zeros(len(faces), dtype=bool)
    queue: deque[int] = deque([0])
    root = local_uv[0].copy()
    if _cross2(root[1] - root[0], root[2] - root[0]) < 0.0:
        root[:, 1] *= -1.0
    global_uv[faces[0]] = root
    placed_faces[0] = True

    max_edge_relative_error = 0.0
    max_cycle_closure_error = 0.0
    while queue:
        face_id = queue.popleft()
        for neighbor_id, edge in adjacency[face_id]:
            if placed_faces[neighbor_id]:
                continue
            transformed, edge_error = _candidate_transform(
                local_uv[neighbor_id],
                faces[neighbor_id],
                edge,
                global_uv,
                faces[face_id],
            )
            max_edge_relative_error = max(max_edge_relative_error, edge_error)
            for local_index, vertex_id in enumerate(faces[neighbor_id]):
                vertex_id = int(vertex_id)
                if np.all(np.isfinite(global_uv[vertex_id])):
                    max_cycle_closure_error = max(
                        max_cycle_closure_error,
                        float(np.linalg.norm(global_uv[vertex_id] - transformed[local_index])),
                    )
                else:
                    global_uv[vertex_id] = transformed[local_index]
            placed_faces[neighbor_id] = True
            queue.append(neighbor_id)

    if not bool(np.all(placed_faces)) or not bool(np.all(np.isfinite(global_uv))):
        raise RuntimeError(
            "official CEPS common refinement could not be stitched into one connected disk chart"
        )

    span = float(np.max(np.ptp(global_uv, axis=0)))
    scale_reference = max(span, 1.0)
    normalized_cycle_error = max_cycle_closure_error / scale_reference
    if max_edge_relative_error > 5e-5 or normalized_cycle_error > 5e-5:
        raise RuntimeError(
            "official CEPS cut chart is not consistently stitchable: "
            f"edge_error={max_edge_relative_error:.3e}, "
            f"cycle_error={normalized_cycle_error:.3e}. "
            "The OneString adapter will not replace this inconsistency with a convex hull."
        )

    signed = np.asarray(
        [
            0.5
            * _cross2(
                global_uv[face[1]] - global_uv[face[0]],
                global_uv[face[2]] - global_uv[face[0]],
            )
            for face in faces
        ],
        dtype=float,
    )
    if float(np.median(signed)) < 0.0:
        global_uv[:, 1] *= -1.0
        signed *= -1.0
    flipped = int(np.count_nonzero(signed < -1e-12))
    degenerate = int(np.count_nonzero(np.abs(signed) <= 1e-12))
    if flipped or degenerate:
        raise RuntimeError(
            "stitched official CEPS chart contains invalid triangles: "
            f"flipped={flipped}, degenerate={degenerate}"
        )

    boundary_edges = sum(1 for incident in incidence.values() if len(incident) == 1)
    return global_uv, {
        "ceps_continuous_chart_reconstructed": True,
        "ceps_internal_cut_seam_edge_count": int(seam_edge_count),
        "ceps_internal_cut_seams_stitched": True,
        "ceps_stitch_max_edge_relative_error": float(max_edge_relative_error),
        "ceps_stitch_max_cycle_closure_error": float(max_cycle_closure_error),
        "ceps_stitch_max_cycle_closure_error_normalized": float(normalized_cycle_error),
        "ceps_common_refinement_surface_vertex_count_before_stitch": int(len(surface_vertices)),
        "ceps_common_refinement_surface_vertex_count_after_stitch": int(len(surface_vertices)),
        "ceps_physical_boundary_edge_count": int(boundary_edges),
        "ceps_convex_hull_boundary_used": False,
        "ceps_artificial_cap_faces_added": 0,
    }


def _continuous_parser(module: Any, path: Path) -> Any:
    surface, faces, face_uv, _texture_ids, raw_texture_count = _triangulate_obj(path)
    stitched_uv, metrics = _stitch_cut_chart(surface, faces, face_uv)
    module._CEPS_LAST_CHART_METRICS = dict(metrics)
    vertex_uv = {int(index): stitched_uv[index].copy() for index in range(len(stitched_uv))}
    return module.CepsObjResult(
        surface,
        faces,
        stitched_uv,
        faces.copy(),
        vertex_uv,
        raw_texture_count,
    )


def install_ceps_paired_output(module: Any) -> None:
    """Install continuous-chart reconstruction into ``official_ceps``.

    The historical function name is retained for import compatibility. The
    implementation no longer duplicates seam vertices; it stitches CEPS cut
    copies into one UV coordinate per common-refinement surface vertex.
    """
    if getattr(module, "_CEPS_PAIRED_OUTPUT_INSTALLED", False):
        return

    def parse(path: Path) -> Any:
        return _continuous_parser(module, path)

    module._parse_ceps_obj = parse
    module._CEPS_PAIRED_OUTPUT_INSTALLED = True
