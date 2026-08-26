"""Reliability fixes for process animations exposed through Streamlit View stage.

This module does not alter Omega/K2D numerics or Split geometry.  It only makes
recorded optimization states survive long enough to be selected later from the
View stage control.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import process_animation_view_patch as process_views


def _boundary_loop_from_triangles(faces: np.ndarray) -> np.ndarray:
    """Recover one ordered boundary loop from a disk-like triangle mesh."""
    tris = np.asarray(faces, dtype=int)[:, :3]
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in tris:
        ids = [int(v) for v in tri]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            counts[tuple(sorted((a, b)))] += 1
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if not boundary_edges:
        return np.zeros(0, dtype=int)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    for _ in range(len(boundary_edges) + 2):
        options = [v for v in adjacency.get(current, []) if v != previous]
        if not options:
            break
        nxt = options[0]
        if nxt == start:
            break
        loop.append(nxt)
        previous, current = current, nxt
    return np.asarray(loop, dtype=int)


def install_direct_omega_process_cache(optimization_debug_module: Any) -> None:
    """Persist Omega accepted states at recorder completion, not renderer time.

    The previous implementation cached the payload only when
    render_omega_flip_debug_animation() happened to be called through the patched
    module attribute.  Some pipeline code keeps an already-imported renderer
    reference, so the View-stage option could exist while its payload remained
    empty.  Hooking the recorder class itself is independent of that call path.
    """
    cls = getattr(optimization_debug_module, "OmegaAcceptedStateRecorder", None)
    if cls is None or getattr(cls, "_onestring_viewstage_direct_cache_installed", False):
        return
    original_capture_final = cls.capture_final

    def capture_final_and_cache(self: Any, uv: np.ndarray, metrics: dict[str, Any]) -> None:
        original_capture_final(self, uv, metrics)
        try:
            faces = np.asarray(self.faces, dtype=int).copy()
            boundary_loop = _boundary_loop_from_triangles(faces)
            payload = {
                "frames": list(self.frames),
                "faces": faces,
                "boundary_loop": boundary_loop,
                "summary": dict(self.summary()),
            }
            process_views._session_set(process_views._SESSION_OMEGA, payload)
            process_views._session_set("_onestring_omega_process_cache_status", {
                "snapshot_count": int(len(self.frames)),
                "boundary_vertex_count": int(len(boundary_loop)),
                "source": "OmegaAcceptedStateRecorder.capture_final",
            })
        except Exception as exc:
            process_views._session_set("_onestring_omega_process_cache_status", {
                "snapshot_count": 0,
                "source": "OmegaAcceptedStateRecorder.capture_final",
                "error": repr(exc),
            })

    cls.capture_final = capture_final_and_cache
    cls._onestring_viewstage_direct_cache_installed = True


def install_process_view_reliability_fixes(optimization_debug_module: Any) -> None:
    install_direct_omega_process_cache(optimization_debug_module)


__all__ = ["install_process_view_reliability_fixes", "install_direct_omega_process_cache"]
