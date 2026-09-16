"""Robust global UV orientation handling for native Grid-OptCuts.

The native backend historically decided whether to reflect the exported UV map
from the sign of the first reconstructed boundary loop.  ``_boundary_loops``
walks an undirected boundary graph, so that loop direction is arbitrary: the
same valid UV map may be returned clockwise or counter-clockwise.  Using that
arbitrary sign can therefore mirror an otherwise valid map and make every
triangle appear flipped.

This patch preserves the existing backend flow but orients the *reported* main
boundary loop to agree with the majority orientation of the UV triangles.  The
backend's existing reflection step then performs a global reflection iff the
actual map is globally reversed.  If positive and negative triangles are truly
mixed, the later per-triangle audit still rejects the map as a real local flip.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from . import optcuts_grid_native_backend as _backend


def _triangle_orientation_counts(uv: np.ndarray, uv_faces: np.ndarray) -> tuple[int, int, int]:
    pts = np.asarray(uv, dtype=float)
    faces = np.asarray(uv_faces, dtype=int)
    if len(faces) == 0:
        return 0, 0, 0
    tri = pts[faces]
    twice_area = (
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    )
    scale = max(float(np.max(np.abs(pts))) if pts.size else 1.0, 1.0)
    tol = max(1.0e-14 * scale * scale, 1.0e-15)
    pos = int(np.count_nonzero(twice_area > tol))
    neg = int(np.count_nonzero(twice_area < -tol))
    deg = int(len(twice_area) - pos - neg)
    return pos, neg, deg


def _orient_primary_loop_to_triangle_majority(
    uv: np.ndarray,
    uv_faces: np.ndarray,
    loops: Iterable[Iterable[int]],
) -> list[list[int]]:
    """Orient loop[0] so its sign encodes the actual global triangle orientation."""
    out = [list(map(int, loop)) for loop in loops]
    if not out or len(out[0]) < 3:
        return out

    pos, neg, _ = _triangle_orientation_counts(uv, uv_faces)
    if pos == 0 and neg == 0:
        return out

    # The backend reflects iff the main loop area is negative.  Therefore make
    # the loop negative iff negative triangles are the global majority.
    want_negative = neg > pos
    area = float(_backend._signed_area(np.asarray(uv, dtype=float)[np.asarray(out[0], dtype=int)]))
    is_negative = area < 0.0
    if is_negative != want_negative:
        out[0].reverse()
    return out


def install_optcuts_grid_orientation_patch() -> None:
    if getattr(_backend, "_onestring_grid_orientation_patch_installed", False):
        return

    original_to_frame = _backend._to_fabrication_frame
    original_boundary_loops = _backend._boundary_loops

    def to_frame_and_remember(uv: np.ndarray, angle_rad: float) -> np.ndarray:
        framed = original_to_frame(uv, angle_rad)
        _backend._onestring_grid_orientation_uv = np.asarray(framed, dtype=float).copy()
        return framed

    def boundary_loops_with_stable_global_orientation(faces: np.ndarray) -> list[list[int]]:
        loops = original_boundary_loops(faces)
        remembered = getattr(_backend, "_onestring_grid_orientation_uv", None)
        if remembered is None:
            return loops
        pos, neg, deg = _triangle_orientation_counts(remembered, faces)
        _backend._onestring_grid_orientation_counts = {
            "positive": int(pos),
            "negative": int(neg),
            "degenerate": int(deg),
        }
        return _orient_primary_loop_to_triangle_majority(remembered, faces, loops)

    _backend._to_fabrication_frame = to_frame_and_remember
    _backend._boundary_loops = boundary_loops_with_stable_global_orientation
    _backend._onestring_grid_orientation_patch_installed = True


__all__ = [
    "_orient_primary_loop_to_triangle_majority",
    "_triangle_orientation_counts",
    "install_optcuts_grid_orientation_patch",
]
