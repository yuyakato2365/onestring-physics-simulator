"""Robust OptCuts seam extraction independent of OBJ vertex-index conventions."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def extract_connected_seam_payload_robust(parameterization: Any) -> dict[str, Any]:
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    empty = {"segments": np.zeros((0, 2, 2), dtype=float), "nodes": {}, "edges": []}
    if len(sf) == 0 or len(sf) != len(uf) or len(xyz) == 0 or len(uv) == 0:
        return empty

    span3 = max(float(np.nanmax(xyz) - np.nanmin(xyz)), 1.0)
    tol3 = max(1e-10 * span3, 1e-12)
    # OptCuts may duplicate a 3D vertex when it creates a cut. Canonicalize exact
    # geometric copies so both sides of the original surface edge meet again for
    # seam detection.
    key_to_canonical: dict[tuple[int, int, int], int] = {}
    canonical_of = np.empty(len(xyz), dtype=int)
    canonical_xyz: list[np.ndarray] = []
    for vi, p in enumerate(xyz[:, :3]):
        key = tuple(np.rint(p / tol3).astype(np.int64).tolist())
        cid = key_to_canonical.get(key)
        if cid is None:
            cid = len(canonical_xyz)
            key_to_canonical[key] = cid
            canonical_xyz.append(p.copy())
        canonical_of[vi] = int(cid)

    # For each canonical 3D edge, store how each incident triangle maps its two
    # endpoints into UV. A real seam is detected by geometric UV separation, not
    # merely by different vt indices; this avoids classifying harmless duplicate
    # indices as cuts.
    incidence: dict[tuple[int, int], list[dict[int, np.ndarray]]] = defaultdict(list)
    for face3, face2 in zip(sf, uf):
        cface = canonical_of[np.asarray(face3, dtype=int)]
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            ca, cb = int(cface[ia]), int(cface[ib])
            if ca == cb:
                continue
            key = tuple(sorted((ca, cb)))
            incidence[key].append(
                {
                    ca: np.asarray(uv[int(face2[ia])], dtype=float),
                    cb: np.asarray(uv[int(face2[ib])], dtype=float),
                }
            )

    uv_span = max(float(np.nanmax(uv) - np.nanmin(uv)), 1.0)
    uv_tol = max(1e-8 * uv_span, 1e-10)
    seam_edges: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for edge, copies in incidence.items():
        if len(copies) != 2:
            continue
        a, b = edge
        c0, c1 = copies
        if a not in c0 or b not in c0 or a not in c1 or b not in c1:
            continue
        da = float(np.linalg.norm(c0[a] - c1[a]))
        db = float(np.linalg.norm(c0[b] - c1[b]))
        if da <= uv_tol and db <= uv_tol:
            continue
        pa = 0.5 * (c0[a] + c1[a])
        pb = 0.5 * (c0[b] + c1[b])
        seam_edges.append((int(a), int(b), pa, pb))

    if not seam_edges:
        return empty
    accum: dict[int, list[np.ndarray]] = defaultdict(list)
    edges: list[tuple[int, int]] = []
    for a, b, pa, pb in seam_edges:
        accum[a].append(np.asarray(pa, dtype=float))
        accum[b].append(np.asarray(pb, dtype=float))
        edges.append((a, b))
    nodes = {int(k): np.mean(np.asarray(vals, dtype=float), axis=0) for k, vals in accum.items()}
    segments = np.asarray([[nodes[a], nodes[b]] for a, b in edges], dtype=float)
    return {
        "segments": segments,
        "nodes": nodes,
        "edges": edges,
        "canonical_surface_vertex_count": int(len(canonical_xyz)),
        "raw_surface_vertex_count": int(len(xyz)),
        "uv_separation_tolerance": float(uv_tol),
    }


def install_robust_optcuts_seam_extraction() -> None:
    from . import optcuts_grid_seam_patch as seam_module

    seam_module._extract_connected_seam_payload = extract_connected_seam_payload_robust


__all__ = ["extract_connected_seam_payload_robust", "install_robust_optcuts_seam_extraction"]
