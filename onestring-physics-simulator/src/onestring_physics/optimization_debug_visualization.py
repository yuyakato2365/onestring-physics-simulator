"""Interactive debug visualizations for S -> Omega and M2D -> K2D.

The Omega recorder intentionally captures *accepted* optimization states rather
than interpolating between the initial/final maps. It hooks the progress
emission points of the full CUDA/MPS solvers and copies only a bounded set of
UV snapshots back to CPU. This keeps the optimizer GPU-resident while making
triangle-orientation failures inspectable.

The K2D visualization has a different purpose: preserve vertex identity so the
user can see where every final K2D grid vertex came from in M2D. It therefore
uses a correspondence morph between M2D and the final K2D state; it is clearly
labelled as a correspondence view rather than an optimizer-iteration trace.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import math
import os
from typing import Any, Iterator

import numpy as np


def _streamlit_available() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _signed_double_areas_np(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    points = np.asarray(uv, dtype=float)[np.asarray(faces, dtype=int)[:, :3]]
    a = points[:, 1] - points[:, 0]
    b = points[:, 2] - points[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _orientation_stats_np(uv: np.ndarray, faces: np.ndarray, absolute_floor: float) -> dict[str, Any]:
    signed = _signed_double_areas_np(uv, faces)
    finite = signed[np.isfinite(signed)]
    if not len(finite):
        return {
            "minimum_signed_area": float("nan"),
            "median_positive_area": float("nan"),
            "minimum_area_ratio": float("nan"),
            "flip_count": int(len(signed)),
            "near_degenerate_count": int(len(signed)),
            "risk_triangle_ids": [],
            "near_triangle_ids": [],
            "flip_triangle_ids": list(range(len(signed))),
        }
    positive = finite[finite > 0]
    median_positive_double = float(np.median(positive)) if len(positive) else 0.0
    relative_floor = 0.01 * median_positive_double
    near_threshold = max(float(absolute_floor) * 10.0, relative_floor)
    flip_ids = np.flatnonzero(signed <= 0.0)
    near_ids = np.flatnonzero((signed > 0.0) & (signed <= near_threshold))
    order = np.argsort(np.where(np.isfinite(signed), signed, -np.inf))
    risk_ids = [int(v) for v in order[: min(12, len(order))]]
    minimum = float(np.min(finite))
    ratio = minimum / median_positive_double if median_positive_double > 0 else float("nan")
    return {
        "minimum_signed_area": 0.5 * minimum,
        "median_positive_area": 0.5 * median_positive_double,
        "minimum_area_ratio": float(ratio),
        "flip_count": int(len(flip_ids)),
        "near_degenerate_count": int(len(near_ids)),
        "risk_triangle_ids": risk_ids,
        "near_triangle_ids": [int(v) for v in near_ids],
        "flip_triangle_ids": [int(v) for v in flip_ids],
    }


class OmegaAcceptedStateRecorder:
    """Bounded recorder for accepted UV states of the full CUDA/MPS optimizer."""

    def __init__(self, faces: np.ndarray, config: Any, max_frames: int = 48) -> None:
        self.faces = np.asarray(faces, dtype=int)[:, :3]
        self.config = config
        self.max_frames = max(8, int(max_frames))
        self.max_iterations = max(1, int(getattr(config, "max_iterations", 1)))
        self.minimum_double_area = float(getattr(config, "minimum_signed_double_area", 1.0e-12))
        self.interval = max(1, int(math.ceil(self.max_iterations / max(1, self.max_frames - 2))))
        self.next_iteration = 1
        self.frames: list[dict[str, Any]] = []
        self._initial_uv: np.ndarray | None = None
        self._last_iteration = -1
        self._lowest_ratio_seen = float("inf")

    def capture_initial(self, uv: np.ndarray) -> None:
        if self._initial_uv is not None:
            return
        value = np.asarray(uv, dtype=np.float32).copy()
        self._initial_uv = value
        stats = _orientation_stats_np(value, self.faces, self.minimum_double_area)
        self.frames.append({
            "iteration": 0,
            "phase": "initial Floater/Tutte",
            "energy": float("nan"),
            "shrink_energy": float("nan"),
            "safe_step_limit": float("nan"),
            "uv": value,
            **stats,
        })
        ratio = float(stats.get("minimum_area_ratio", float("nan")))
        if np.isfinite(ratio):
            self._lowest_ratio_seen = ratio

    def maybe_capture_from_solver_frame(self, frame: Any, stage: str) -> None:
        stage_name = str(stage)
        if not (stage_name.startswith("CUDA Omega") or stage_name.startswith("MPS Omega")):
            return
        local = getattr(frame, "f_locals", {}) or {}
        uv = local.get("uv")
        accel = local.get("accel")
        if uv is None or accel is None:
            return
        try:
            iteration = int(local.get("iteration", -1)) + 1
        except Exception:
            return
        if iteration <= self._last_iteration:
            return
        risky = False
        try:
            tri = uv[accel.faces]
            a = tri[:, 1] - tri[:, 0]
            b = tri[:, 2] - tri[:, 0]
            signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            min_double = float(signed.min().item())
            positive = signed[signed > 0]
            median_double = float(positive.median().item()) if int(positive.numel()) else 0.0
            ratio = min_double / median_double if median_double > 0 else float("-inf")
            if ratio < 0.75 * self._lowest_ratio_seen or min_double <= 0.0:
                risky = True
                self._lowest_ratio_seen = min(self._lowest_ratio_seen, ratio)
        except Exception:
            ratio = float("nan")

        periodic = iteration >= self.next_iteration
        if not periodic and not risky:
            return
        if len(self.frames) >= self.max_frames - 1:
            return
        try:
            uv_np = uv.detach().cpu().numpy().astype(np.float32, copy=True)
        except Exception:
            return
        stats = _orientation_stats_np(uv_np, self.faces, self.minimum_double_area)
        energy = local.get("energy")
        shrink = local.get("shrink_energy")
        safe = local.get("safe")

        def scalar(value: Any) -> float:
            try:
                if hasattr(value, "item"):
                    return float(value.item())
                return float(value)
            except Exception:
                return float("nan")

        self.frames.append({
            "iteration": int(iteration),
            "phase": str(local.get("phase", "accepted")),
            "energy": scalar(energy),
            "shrink_energy": scalar(shrink),
            "safe_step_limit": scalar(safe),
            "uv": uv_np,
            **stats,
        })
        self._last_iteration = int(iteration)
        while self.next_iteration <= iteration:
            self.next_iteration += self.interval

    def capture_final(self, uv: np.ndarray, metrics: dict[str, Any]) -> None:
        value = np.asarray(uv, dtype=np.float32).copy()
        iteration = int(metrics.get("optimization_iteration_count", self.max_iterations) or self.max_iterations)
        stats = _orientation_stats_np(value, self.faces, self.minimum_double_area)
        frame = {
            "iteration": iteration,
            "phase": "final accepted",
            "energy": float(metrics.get("final_energy", float("nan"))),
            "shrink_energy": float(metrics.get("final_shrink_energy", float("nan"))),
            "safe_step_limit": float("nan"),
            "uv": value,
            **stats,
        }
        if self.frames and self.frames[-1]["iteration"] == iteration:
            self.frames[-1] = frame
        elif len(self.frames) < self.max_frames:
            self.frames.append(frame)
        else:
            self.frames[-1] = frame
        if self.frames:
            self.frames[0]["energy"] = float(metrics.get("initial_energy", float("nan")))
            self.frames[0]["shrink_energy"] = float(metrics.get("initial_shrink_energy", float("nan")))

    def summary(self) -> dict[str, Any]:
        if not self.frames:
            return {"omega_debug_snapshot_count": 0, "omega_debug_any_accepted_flip": False}
        flips = [int(f.get("flip_count", 0)) for f in self.frames]
        ratios = [float(f.get("minimum_area_ratio", float("nan"))) for f in self.frames]
        finite_ratios = [v for v in ratios if np.isfinite(v)]
        return {
            "omega_debug_snapshot_count": int(len(self.frames)),
            "omega_debug_any_accepted_flip": bool(any(v > 0 for v in flips)),
            "omega_debug_max_accepted_flip_count": int(max(flips, default=0)),
            "omega_debug_min_signed_area_ratio": float(min(finite_ratios)) if finite_ratios else float("nan"),
            "omega_debug_snapshot_policy": "accepted states only; periodic + new risk minima; bounded GPU->CPU copies",
        }


@contextmanager
def capture_omega_accepted_states(base_module: Any, faces: np.ndarray, config: Any) -> Iterator[OmegaAcceptedStateRecorder]:
    max_frames = int(os.getenv("ONESTRING_OMEGA_DEBUG_MAX_FRAMES", "48"))
    recorder = OmegaAcceptedStateRecorder(faces, config, max_frames=max_frames)
    original_emit = base_module._emit_progress
    original_tutte = base_module._tutte_embedding

    def wrapped_tutte(*args: Any, **kwargs: Any):
        result = original_tutte(*args, **kwargs)
        try:
            recorder.capture_initial(result)
        except Exception:
            pass
        return result

    def wrapped_emit(callback: Any, stage: str, progress: float, detail: str = "") -> None:
        try:
            caller = inspect.currentframe().f_back
            if caller is not None:
                recorder.maybe_capture_from_solver_frame(caller, str(stage))
        except Exception:
            pass
        return original_emit(callback, stage, progress, detail)

    if _env_bool("ONESTRING_OMEGA_DEBUG_ANIMATION", True):
        base_module._tutte_embedding = wrapped_tutte
        base_module._emit_progress = wrapped_emit
    try:
        yield recorder
    finally:
        base_module._tutte_embedding = original_tutte
        base_module._emit_progress = original_emit


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    tris = np.asarray(faces, dtype=int)[:, :3]
    if not len(tris):
        return np.zeros((0, 2), dtype=int)
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _edge_line_xy(uv: np.ndarray, edges: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    pts = np.asarray(uv, dtype=float)
    x: list[float | None] = []
    y: list[float | None] = []
    for a, b in np.asarray(edges, dtype=int):
        x.extend([float(pts[a, 0]), float(pts[b, 0]), None])
        y.extend([float(pts[a, 1]), float(pts[b, 1]), None])
    return x, y


def _triangle_line_xy(uv: np.ndarray, faces: np.ndarray, triangle_ids: list[int]) -> tuple[list[float | None], list[float | None]]:
    pts = np.asarray(uv, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    x: list[float | None] = []
    y: list[float | None] = []
    for triangle_id in triangle_ids:
        if not (0 <= int(triangle_id) < len(tris)):
            continue
        ids = tris[int(triangle_id)]
        loop = pts[np.asarray([ids[0], ids[1], ids[2], ids[0]], dtype=int)]
        x.extend([*loop[:, 0].tolist(), None])
        y.extend([*loop[:, 1].tolist(), None])
    return x, y


def render_omega_flip_debug_animation(frames: list[dict[str, Any]], faces: np.ndarray, boundary_loop: list[int] | np.ndarray, summary: dict[str, Any] | None = None) -> None:
    if not frames or not _streamlit_available() or not _env_bool("ONESTRING_OMEGA_DEBUG_ANIMATION", True):
        return
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except Exception:
        return

    tris = np.asarray(faces, dtype=int)[:, :3]
    edges = _unique_edges(tris)
    boundary = [int(v) for v in np.asarray(boundary_loop, dtype=int).reshape(-1).tolist()]
    all_uv = np.vstack([np.asarray(frame["uv"], dtype=float) for frame in frames])
    min_xy = np.nanmin(all_uv, axis=0)
    max_xy = np.nanmax(all_uv, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0e-8)
    padding = 0.06 * float(max(span))

    def payload(frame: dict[str, Any]):
        uv = np.asarray(frame["uv"], dtype=float)
        mesh_x, mesh_y = _edge_line_xy(uv, edges)
        flip_set = set(frame.get("flip_triangle_ids", []))
        risk_ids = [int(v) for v in frame.get("risk_triangle_ids", []) if int(v) not in flip_set]
        risk_x, risk_y = _triangle_line_xy(uv, tris, risk_ids)
        flip_x, flip_y = _triangle_line_xy(uv, tris, list(frame.get("flip_triangle_ids", [])))
        if boundary:
            ids = np.asarray(boundary + [boundary[0]], dtype=int)
            b = uv[ids]
            bx, by = b[:, 0].tolist(), b[:, 1].tolist()
        else:
            bx, by = [], []
        return mesh_x, mesh_y, risk_x, risk_y, flip_x, flip_y, bx, by

    p0 = payload(frames[0])
    fig = go.Figure(data=[
        go.Scattergl(x=p0[0], y=p0[1], mode="lines", line=dict(width=1), opacity=0.55, name="Omega mesh"),
        go.Scattergl(x=p0[2], y=p0[3], mode="lines", line=dict(width=4), name="lowest signed-area triangles"),
        go.Scattergl(x=p0[4], y=p0[5], mode="lines", line=dict(width=6), name="FLIPPED triangles"),
        go.Scattergl(x=p0[6], y=p0[7], mode="lines", line=dict(width=3), name="Omega boundary"),
    ])
    plotly_frames = []
    for index, record in enumerate(frames):
        p = payload(record)
        title = (
            f"Accepted Ω state | iter={int(record.get('iteration', 0))} | phase={record.get('phase', '')} | "
            f"flips={int(record.get('flip_count', 0))} | near={int(record.get('near_degenerate_count', 0))} | "
            f"min area={float(record.get('minimum_signed_area', float('nan'))):.3g} | "
            f"min/median={float(record.get('minimum_area_ratio', float('nan'))):.3g} | "
            f"E={float(record.get('energy', float('nan'))):.5g}"
        )
        plotly_frames.append(go.Frame(name=str(index), data=[
            go.Scattergl(x=p[0], y=p[1]),
            go.Scattergl(x=p[2], y=p[3]),
            go.Scattergl(x=p[4], y=p[5]),
            go.Scattergl(x=p[6], y=p[7]),
        ], layout=go.Layout(title=title)))
    fig.frames = plotly_frames
    steps = [{
        "method": "animate",
        "label": str(int(record.get("iteration", 0))),
        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for i, record in enumerate(frames)]
    fig.update_layout(
        title="Accepted Ω state | initial", height=720,
        xaxis=dict(title="u", range=[float(min_xy[0] - padding), float(max_xy[0] + padding)]),
        yaxis=dict(title="v", range=[float(min_xy[1] - padding), float(max_xy[1] + padding)], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=0.0, y=1.08),
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0.0, "y": -0.12, "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 220, "redraw": True}, "transition": {"duration": 0}}]},
            {"label": "■ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
        ]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "iteration "}, "steps": steps, "x": 0.18, "len": 0.80, "y": -0.10}],
        margin=dict(l=30, r=20, t=80, b=120),
    )
    st.subheader("Ω flip / degeneracy debug animation")
    info = dict(summary or {})
    any_flip = bool(info.get("omega_debug_any_accepted_flip", any(int(f.get("flip_count", 0)) > 0 for f in frames)))
    if any_flip:
        st.error("At least one captured ACCEPTED Ω state contains a flipped triangle. This indicates a bijectivity-preservation bug.")
    else:
        st.success("No flipped triangle was found in the captured ACCEPTED Ω states. Watch the highlighted low-area triangles for near-degeneracy.")
    st.caption(
        "This is an actual accepted-state trace, not interpolation. The animation stores a bounded sample of accepted GPU states (CUDA/MPS); "
        "the thick risk overlay tracks triangles with the smallest signed UV areas. Rejected line-search candidates are not shown."
    )
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def _quad_unique_edges(faces: np.ndarray) -> np.ndarray:
    quads = np.asarray(faces, dtype=int)
    edges: list[tuple[int, int]] = []
    for face in quads:
        ids = [int(v) for v in face]
        for i in range(len(ids)):
            edges.append(tuple(sorted((ids[i], ids[(i + 1) % len(ids)]))))
    if not edges:
        return np.zeros((0, 2), dtype=int)
    return np.unique(np.asarray(edges, dtype=int), axis=0)


def _quad_edge_lines(vertices: np.ndarray, edges: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    xy = np.asarray(vertices, dtype=float)[:, :2]
    x: list[float | None] = []
    y: list[float | None] = []
    for a, b in np.asarray(edges, dtype=int):
        x.extend([float(xy[a, 0]), float(xy[b, 0]), None])
        y.extend([float(xy[a, 1]), float(xy[b, 1]), None])
    return x, y


def render_k2d_correspondence_morph(mesh_2d: Any, mesh_k2d: Any, mesh_k3d: Any | None = None) -> None:
    if not _streamlit_available() or not _env_bool("ONESTRING_K2D_DEBUG_ANIMATION", True):
        return
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except Exception:
        return

    start = np.asarray(mesh_2d.vertices, dtype=float)
    final = np.asarray(mesh_k2d.vertices, dtype=float)
    faces = np.asarray(mesh_2d.faces, dtype=int)
    if start.ndim != 2 or final.ndim != 2 or len(start) != len(final) or not len(start) or not len(faces):
        return
    edges = _quad_unique_edges(faces)
    ghost_x, ghost_y = _quad_edge_lines(start, edges)
    n_steps = max(12, min(36, int(os.getenv("ONESTRING_K2D_MORPH_FRAMES", "28"))))
    ts = np.linspace(0.0, 1.0, n_steps)
    grid = getattr(mesh_2d, "grid", None)
    nx = int(getattr(grid, "nx", -1)) if grid is not None else -1
    ny = int(getattr(grid, "ny", -1)) if grid is not None else -1
    row_width = nx + 1 if nx >= 0 else 0
    regular_count = (nx + 1) * (ny + 1) if nx >= 0 and ny >= 0 else 0
    displacement = np.linalg.norm(final[:, :2] - start[:, :2], axis=1)
    hover: list[str] = []
    for vertex_id in range(len(start)):
        if row_width > 0 and vertex_id < regular_count:
            row, col = divmod(vertex_id, row_width)
            grid_label = f"grid(row={row}, col={col})"
        else:
            grid_label = "derived / split vertex"
        hover.append(
            f"vertex {vertex_id}<br>{grid_label}<br>"
            f"M2D=({start[vertex_id,0]:.5g}, {start[vertex_id,1]:.5g})<br>"
            f"K2D=({final[vertex_id,0]:.5g}, {final[vertex_id,1]:.5g})<br>"
            f"|Δ|={displacement[vertex_id]:.5g}"
        )
    all_xy = np.vstack([start[:, :2], final[:, :2]])
    min_xy = np.nanmin(all_xy, axis=0)
    max_xy = np.nanmax(all_xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0e-8)
    padding = 0.06 * float(max(span))
    current_x, current_y = _quad_edge_lines(start, edges)
    fig = go.Figure(data=[
        go.Scattergl(x=ghost_x, y=ghost_y, mode="lines", line=dict(width=1, dash="dot"), opacity=0.35, name="Original M2D grid (fixed ghost)"),
        go.Scattergl(x=current_x, y=current_y, mode="lines", line=dict(width=2), name="Current M2D → K2D mesh"),
        go.Scattergl(x=start[:, 0], y=start[:, 1], mode="markers",
                     marker=dict(size=5, color=displacement, colorscale="Viridis", showscale=True, colorbar=dict(title="final |Δ|")),
                     text=hover, hoverinfo="text", name="vertex identity"),
    ])
    animation_frames = []
    for i, t in enumerate(ts):
        current = (1.0 - float(t)) * start + float(t) * final
        ex, ey = _quad_edge_lines(current, edges)
        animation_frames.append(go.Frame(name=str(i), data=[
            go.Scattergl(x=ghost_x, y=ghost_y),
            go.Scattergl(x=ex, y=ey),
            go.Scattergl(x=current[:, 0], y=current[:, 1], text=hover),
        ], layout=go.Layout(title=f"M2D → K2D correspondence morph | {100.0*float(t):.0f}%")))
    fig.frames = animation_frames
    steps = [{
        "method": "animate", "label": f"{100.0*float(t):.0f}%",
        "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for i, t in enumerate(ts)]
    fig.update_layout(
        title="M2D → K2D correspondence morph | 0%", height=700,
        xaxis=dict(title="x", range=[float(min_xy[0] - padding), float(max_xy[0] + padding)]),
        yaxis=dict(title="y", range=[float(min_xy[1] - padding), float(max_xy[1] + padding)], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=0.0, y=1.08),
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0.0, "y": -0.12, "buttons": [
            {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 180, "redraw": True}, "transition": {"duration": 0}}]},
            {"label": "■ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
        ]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "morph "}, "steps": steps, "x": 0.18, "len": 0.80, "y": -0.10}],
        margin=dict(l=30, r=20, t=80, b=120),
    )
    st.subheader("M2D → K2D grid correspondence animation")
    st.caption(
        "The dotted ghost is the original M2D grid and never moves. Hover a current vertex to see its original vertex id and grid(row,col). "
        "This is a correspondence morph from M2D to the final K2D result, not a replay of every internal K2D optimizer iteration."
    )
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def install_k2d_correspondence_animation(pipeline_module: Any) -> None:
    if getattr(pipeline_module, "_K2D_CORRESPONDENCE_ANIMATION_PATCH_INSTALLED", False):
        return
    original = pipeline_module._optimize_k2d

    def wrapped(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback: Any = None):
        result, report = original(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        try:
            render_k2d_correspondence_morph(mesh_2d, result, mesh_3d)
        except Exception:
            pass
        return result, report

    pipeline_module._optimize_k2d = wrapped
    if hasattr(pipeline_module, "_original"):
        pipeline_module._original._optimize_k2d = wrapped
    pipeline_module._K2D_CORRESPONDENCE_ANIMATION_PATCH_INSTALLED = True


__all__ = [
    "OmegaAcceptedStateRecorder",
    "capture_omega_accepted_states",
    "render_omega_flip_debug_animation",
    "render_k2d_correspondence_morph",
    "install_k2d_correspondence_animation",
]
