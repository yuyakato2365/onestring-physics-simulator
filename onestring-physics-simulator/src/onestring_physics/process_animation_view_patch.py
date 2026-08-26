"""Expose actual optimization-process animations through Streamlit View stage.

This module is deliberately independent of Split semantics.  It only records
already-computed optimizer states and makes them selectable from ``View stage``.

Omega:
- reuses the existing accepted-state recorder/renderer;
- caches the exact accepted UV snapshots that the existing debug animation gets.

K2D:
- records the actual ``xy``/``xy_t`` state at K2D progress emission points;
- never synthesizes intermediate states between M2D and final K2D;
- always stores initial and final states, and stores bounded solver checkpoints
  whenever the projective/strict solver emits progress.
"""
from __future__ import annotations

import inspect
import os
from typing import Any

import numpy as np


PROCESS_ANIMATION_VIEWS = (
    "Animation: Omega optimization (accepted states)",
    "Animation: K2D optimization (actual iterations)",
)

_SESSION_SELECTED = "_onestring_process_animation_choice"
_SESSION_OMEGA = "_onestring_omega_process_payload"
_SESSION_K2D = "_onestring_k2d_process_payload"


def _session_set(key: str, value: Any) -> None:
    try:
        import streamlit as st
        st.session_state[key] = value
    except Exception:
        pass


def _numpy_xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
        return None
    arr = arr[:, :2].copy()
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _edge_errors(xy: np.ndarray, edges: np.ndarray, targets: np.ndarray | None) -> tuple[float, float]:
    if targets is None or len(edges) == 0 or len(targets) != len(edges):
        return float("nan"), float("nan")
    lengths = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    error = np.abs(lengths - targets)
    return float(np.mean(error)), float(np.max(error))


def _quad_edges(faces: np.ndarray) -> np.ndarray:
    items: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            items.add(tuple(sorted((ids[i], ids[(i + 1) % len(ids)]))))
    return np.asarray(sorted(items), dtype=int) if items else np.zeros((0, 2), dtype=int)


def _edge_lines(xy: np.ndarray, edges: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    for a, b in np.asarray(edges, dtype=int):
        x.extend([float(xy[a, 0]), float(xy[b, 0]), None])
        y.extend([float(xy[a, 1]), float(xy[b, 1]), None])
    return x, y


class K2DProgressRecorder:
    def __init__(self, faces: np.ndarray, initial_xy: np.ndarray, mesh_3d: Any, max_frames: int = 48) -> None:
        self.faces = np.asarray(faces, dtype=int).copy()
        self.edges = _quad_edges(self.faces)
        self.initial_xy = np.asarray(initial_xy, dtype=float)[:, :2].copy()
        self.max_frames = max(8, int(max_frames))
        self.frames: list[dict[str, Any]] = []
        self._seen_signatures: set[tuple[str, int, float]] = set()
        self.targets: np.ndarray | None = None
        try:
            m3 = np.asarray(mesh_3d.vertices, dtype=float)
            self.targets = np.asarray(
                [np.linalg.norm(m3[a] - m3[b]) for a, b in self.edges], dtype=float
            )
        except Exception:
            self.targets = None
        self.capture(self.initial_xy, "initial M2D", 0, "initial")

    def capture(self, xy: Any, stage: str, iteration: int, detail: str = "") -> None:
        value = _numpy_xy(xy)
        if value is None or len(value) != len(self.initial_xy):
            return
        if len(self.frames) >= self.max_frames:
            return
        # Avoid duplicate progress callbacks that expose the same solver state.
        checksum = float(np.sum(value[: min(len(value), 32)]))
        signature = (str(stage), int(iteration), round(checksum, 10))
        if signature in self._seen_signatures:
            return
        self._seen_signatures.add(signature)
        mean_err, max_err = _edge_errors(value, self.edges, self.targets)
        self.frames.append(
            {
                "xy": value.astype(np.float32, copy=True),
                "stage": str(stage),
                "iteration": int(iteration),
                "detail": str(detail),
                "mean_edge_error": float(mean_err),
                "max_edge_error": float(max_err),
            }
        )

    def maybe_capture_caller(self, caller: Any, stage: str, detail: str) -> None:
        if caller is None or "k2d" not in str(stage).lower():
            return
        local = getattr(caller, "f_locals", {}) or {}
        xy = local.get("xy")
        if xy is None:
            xy = local.get("xy_t")
        if xy is None:
            # Some helpers use a generic current/result naming convention.
            for key in ("current_xy", "current", "candidate"):
                candidate = local.get(key)
                arr = _numpy_xy(candidate)
                if arr is not None and len(arr) == len(self.initial_xy):
                    xy = candidate
                    break
        if xy is None:
            return
        iteration = 0
        for key in ("it", "iteration", "step", "projective_iterations_done"):
            try:
                if key in local:
                    iteration = int(local[key]) + (1 if key == "it" else 0)
                    break
            except Exception:
                pass
        self.capture(xy, stage, iteration, detail)

    def capture_final(self, result: Any) -> None:
        try:
            xy = np.asarray(result.vertices, dtype=float)[:, :2]
            iteration = int((getattr(result, "metrics", {}) or {}).get("optimizer_iterations", len(self.frames)))
        except Exception:
            return
        if len(self.frames) >= self.max_frames:
            self.frames[-1] = {
                **self.frames[-1],
                "xy": xy.astype(np.float32, copy=True),
                "stage": "final K2D",
                "iteration": iteration,
                "detail": "final accepted result",
            }
        else:
            self.capture(xy, "final K2D", iteration, "final accepted result")

    def payload(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "faces": self.faces,
            "edges": self.edges,
            "initial_xy": self.initial_xy,
            "actual_trace": len(self.frames) > 2,
        }


def install_k2d_process_recorder(pipeline_module: Any) -> None:
    """Wrap the current outer K2D function once, without modifying Split code."""
    if getattr(pipeline_module, "_K2D_PROCESS_RECORDER_INSTALLED", False):
        return
    original_optimize = pipeline_module._optimize_k2d
    base_module = getattr(pipeline_module, "_original", pipeline_module)

    def optimize_with_process_trace(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
        max_frames = int(os.getenv("ONESTRING_K2D_DEBUG_MAX_FRAMES", "48"))
        recorder = K2DProgressRecorder(
            np.asarray(mesh_2d.faces, dtype=int),
            np.asarray(mesh_2d.vertices, dtype=float)[:, :2],
            mesh_3d,
            max_frames=max_frames,
        )
        original_emit = getattr(base_module, "_emit_progress", None)

        def traced_emit(callback: Any, stage: str, progress: float, detail: str = "") -> None:
            try:
                caller = inspect.currentframe().f_back
                recorder.maybe_capture_caller(caller, str(stage), str(detail))
            except Exception:
                pass
            if original_emit is not None:
                return original_emit(callback, stage, progress, detail)
            if callback is not None:
                try:
                    callback(stage, progress, detail)
                except Exception:
                    pass
            return None

        if original_emit is not None:
            base_module._emit_progress = traced_emit
        try:
            result, report = original_optimize(
                mesh_2d, mesh_3d, params, progress_callback=progress_callback
            )
        finally:
            if original_emit is not None:
                base_module._emit_progress = original_emit
        recorder.capture_final(result)
        _session_set(_SESSION_K2D, recorder.payload())
        try:
            result.metrics["k2d_process_debug_snapshot_count"] = int(len(recorder.frames))
            result.metrics["k2d_process_debug_actual_iteration_trace"] = bool(len(recorder.frames) > 2)
        except Exception:
            pass
        return result, report

    pipeline_module._optimize_k2d = optimize_with_process_trace
    pipeline_module._K2D_PROCESS_RECORDER_INSTALLED = True


def install_omega_process_cache(optimization_debug_module: Any) -> None:
    """Cache the exact payload already sent to the existing Omega renderer."""
    if getattr(optimization_debug_module, "_OMEGA_PROCESS_VIEW_CACHE_INSTALLED", False):
        return
    original_render = optimization_debug_module.render_omega_flip_debug_animation

    def render_and_cache(frames: Any, faces: Any, boundary_loop: Any, summary: Any = None):
        try:
            payload = {
                "frames": list(frames),
                "faces": np.asarray(faces, dtype=int).copy(),
                "boundary_loop": np.asarray(boundary_loop, dtype=int).copy(),
                "summary": dict(summary or {}),
            }
            _session_set(_SESSION_OMEGA, payload)
        except Exception:
            pass
        return original_render(frames, faces, boundary_loop, summary)

    optimization_debug_module.render_omega_flip_debug_animation = render_and_cache
    optimization_debug_module._OMEGA_PROCESS_VIEW_CACHE_INSTALLED = True


def install_process_animation_selector() -> None:
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_process_animation_selector_installed", False):
        return
    previous = st.selectbox

    def selectbox_with_process_views(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label != "View stage":
            return previous(*args, **kwargs)
        if len(args) >= 2:
            options = list(args[1])
            for item in PROCESS_ANIMATION_VIEWS:
                if item not in options:
                    options.append(item)
            args = (args[0], options, *args[2:])
        else:
            options = list(kwargs.get("options", []))
            for item in PROCESS_ANIMATION_VIEWS:
                if item not in options:
                    options.append(item)
            kwargs = {**kwargs, "options": options}
        selected = previous(*args, **kwargs)
        if selected in PROCESS_ANIMATION_VIEWS:
            st.session_state[_SESSION_SELECTED] = selected
            return "T3D"
        st.session_state[_SESSION_SELECTED] = None
        return selected

    st.selectbox = selectbox_with_process_views
    st._onestring_process_animation_selector_installed = True


def _render_k2d(payload: dict[str, Any]) -> None:
    import plotly.graph_objects as go
    import streamlit as st

    frames = list(payload.get("frames", []) or [])
    faces = np.asarray(payload.get("faces", []), dtype=int)
    if not frames or not len(faces):
        st.info("Run the design calculation once to record K2D optimization states.")
        return
    edges = _quad_edges(faces)
    all_xy = np.vstack([np.asarray(frame["xy"], dtype=float) for frame in frames])
    lo = np.min(all_xy, axis=0)
    hi = np.max(all_xy, axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pad = 0.06 * float(np.max(span))
    ghost = np.asarray(frames[0]["xy"], dtype=float)
    gx, gy = _edge_lines(ghost, edges)

    def data_for(record: dict[str, Any]):
        xy = np.asarray(record["xy"], dtype=float)
        ex, ey = _edge_lines(xy, edges)
        return [
            go.Scattergl(x=gx, y=gy, mode="lines", line=dict(width=1, dash="dot"), opacity=0.3, name="initial M2D"),
            go.Scattergl(x=ex, y=ey, mode="lines+markers", line=dict(width=2), marker=dict(size=3), name="actual K2D state"),
        ]

    fig = go.Figure(data=data_for(frames[0]))
    pframes = []
    for i, record in enumerate(frames):
        title = (
            f"K2D actual optimization state | {record.get('stage', '')} | "
            f"iter={int(record.get('iteration', 0))} | "
            f"mean edge error={float(record.get('mean_edge_error', float('nan'))):.4g} | "
            f"max={float(record.get('max_edge_error', float('nan'))):.4g}"
        )
        pframes.append(go.Frame(name=str(i), data=data_for(record), layout=go.Layout(title=title)))
    fig.frames = pframes
    steps = [
        {
            "method": "animate",
            "label": f"{i}: {record.get('stage', '')} {int(record.get('iteration', 0))}",
            "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for i, record in enumerate(frames)
    ]
    fig.update_layout(
        title="K2D actual optimization state | initial M2D",
        height=720,
        xaxis=dict(range=[float(lo[0] - pad), float(hi[0] + pad)], title="x"),
        yaxis=dict(range=[float(lo[1] - pad), float(hi[1] + pad)], title="y", scaleanchor="x", scaleratio=1),
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 220, "redraw": True}, "transition": {"duration": 0}}]},
            {"label": "■ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
        ]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "checkpoint "}, "steps": steps}],
        margin=dict(l=30, r=20, t=80, b=90),
    )
    st.markdown("### Animation: K2D optimization (actual iterations)")
    if bool(payload.get("actual_trace", False)):
        st.caption(
            "These are actual solver states captured at K2D progress checkpoints; no M2D→K2D interpolation is used. "
            "The dotted mesh is the initial M2D reference."
        )
    else:
        st.warning(
            "This backend did not expose intermediate K2D checkpoints in this run, so only the actual initial/final states were recorded."
        )
    st.plotly_chart(fig, config={"responsive": True})


def render_selected_process_animation(optimization_debug_module: Any) -> bool:
    try:
        import streamlit as st
    except Exception:
        return False
    selected = st.session_state.get(_SESSION_SELECTED)
    if selected not in PROCESS_ANIMATION_VIEWS:
        return False

    if selected == PROCESS_ANIMATION_VIEWS[0]:
        payload = st.session_state.get(_SESSION_OMEGA)
        if not payload:
            st.info(
                "Run the design calculation once with an Omega backend that records accepted optimization states (for example bijective_free_boundary)."
            )
            return True
        st.markdown("### Animation: Omega optimization (accepted states)")
        optimization_debug_module.render_omega_flip_debug_animation(
            payload["frames"], payload["faces"], payload["boundary_loop"], payload.get("summary")
        )
        return True

    payload = st.session_state.get(_SESSION_K2D)
    if not payload:
        st.info("Run the design calculation once to record K2D optimization states.")
        return True
    _render_k2d(payload)
    return True


__all__ = [
    "PROCESS_ANIMATION_VIEWS",
    "install_k2d_process_recorder",
    "install_omega_process_cache",
    "install_process_animation_selector",
    "render_selected_process_animation",
]
