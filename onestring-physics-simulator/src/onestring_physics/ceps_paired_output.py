"""Normalize official CEPS OBJ output to a paired 3D/UV vertex mesh.

CEPS writes ordinary texture coordinates per face corner.  Around parameterization
cuts, one common-refinement surface vertex can therefore have several UV values.
The OneString pipeline and its diagnostics are simplest and safest when each array
index denotes one paired ``(surface position, UV position)`` sample.  This module
replaces the raw OBJ parser with a seam-aware parser that duplicates the 3D vertex
where required and gives ``surface_faces`` and ``uv_faces`` identical connectivity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _obj_index(value: str, count: int) -> int:
    index = int(value)
    return index - 1 if index > 0 else count + index


def _paired_parser(module: Any, path: Path) -> Any:
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

    original_vertices = np.asarray(vertices, dtype=float)
    raw_uv = np.asarray(textures, dtype=float)
    surface_faces: list[list[int]] = []
    texture_faces: list[list[int]] = []
    for polygon in polygons:
        for index in range(1, len(polygon) - 1):
            triangle = [polygon[0], polygon[index], polygon[index + 1]]
            surface_faces.append([item[0] for item in triangle])
            texture_faces.append([item[1] for item in triangle])

    sf = np.asarray(surface_faces, dtype=int)
    tf = np.asarray(texture_faces, dtype=int)
    if (
        np.min(sf) < 0
        or np.max(sf) >= len(original_vertices)
        or np.min(tf) < 0
        or np.max(tf) >= len(raw_uv)
    ):
        raise RuntimeError("official CEPS output OBJ contains an invalid index")

    # Preserve the original CEPS surface-vertex -> UV samples used to locate the
    # four input corner vertices during alignment.  A seam vertex can have more
    # than one UV sample, matching the behavior of the previous parser.
    samples: dict[int, list[np.ndarray]] = {}
    for surface_face, texture_face in zip(sf, tf):
        for surface_id, texture_id in zip(surface_face, texture_face):
            samples.setdefault(int(surface_id), []).append(raw_uv[int(texture_id)])
    vertex_uv = {
        vertex_id: np.mean(np.asarray(values, dtype=float), axis=0)
        for vertex_id, values in samples.items()
    }

    # A UV seam must duplicate the corresponding 3D vertex.  Pairing by original
    # surface ID and rounded UV coordinates keeps ordinary shared vertices welded,
    # while preserving distinct copies on the two sides of a cut.
    paired_surface_vertices: list[np.ndarray] = []
    paired_uv_vertices: list[np.ndarray] = []
    paired_faces = np.empty_like(sf)
    lookup: dict[tuple[int, float, float], int] = {}

    for face_index, (surface_face, texture_face) in enumerate(zip(sf, tf)):
        for corner_index, (surface_id, texture_id) in enumerate(
            zip(surface_face, texture_face)
        ):
            uv = raw_uv[int(texture_id), :2]
            rounded = np.round(uv, 12)
            key = (int(surface_id), float(rounded[0]), float(rounded[1]))
            paired_id = lookup.get(key)
            if paired_id is None:
                paired_id = len(paired_surface_vertices)
                lookup[key] = paired_id
                paired_surface_vertices.append(original_vertices[int(surface_id)].copy())
                paired_uv_vertices.append(uv.copy())
            paired_faces[face_index, corner_index] = paired_id

    paired_surface = np.asarray(paired_surface_vertices, dtype=float)
    paired_uv = np.asarray(paired_uv_vertices, dtype=float)
    if len(paired_surface) != len(paired_uv):
        raise RuntimeError("CEPS paired 3D/UV vertex construction failed")

    return module.CepsObjResult(
        paired_surface,
        paired_faces,
        paired_uv,
        paired_faces.copy(),
        vertex_uv,
        len(raw_uv),
    )


def install_ceps_paired_output(module: Any) -> None:
    """Install the seam-aware parser into ``onestring_physics.official_ceps``."""
    if getattr(module, "_CEPS_PAIRED_OUTPUT_INSTALLED", False):
        return

    def parse(path: Path) -> Any:
        return _paired_parser(module, path)

    module._parse_ceps_obj = parse
    module._CEPS_PAIRED_OUTPUT_INSTALLED = True
