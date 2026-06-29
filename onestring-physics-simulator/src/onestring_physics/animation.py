from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .design_optimizer import DesignResult
from .onestring_pipeline import OneStringDesignState, TileAssembly
from .visualization import add_gap_graph, add_rope, add_tile_assembly, add_tiles, _style_scene


def deployment_animation(
    design: DesignResult,
    tile_frames: list[np.ndarray],
    rope_frames: list[np.ndarray] | None = None,
    handle_frames: list[np.ndarray] | None = None,
    step: int = 4,
) -> go.Figure:
    if not tile_frames:
        return go.Figure()
    stride = max(1, step)
    first_tiles = tile_frames[0]
    first_rope = rope_frames[0] if rope_frames else None
    fig = go.Figure()
    add_tiles(fig, first_tiles)
    if first_rope is not None:
        add_rope(fig, first_rope)
    if handle_frames:
        h = handle_frames[0]
        fig.add_trace(go.Scatter3d(x=[h[0]], y=[h[1]], z=[h[2]], mode="markers", marker=dict(size=7, color="#b11226"), name="pull handle"))

    frames = []
    for frame_id in range(0, len(tile_frames), stride):
        data = []
        for tile in tile_frames[frame_id]:
            data.append(
                go.Mesh3d(
                    x=tile[:, 0],
                    y=tile[:, 1],
                    z=tile[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color="#2dd4bf",
                    opacity=0.78,
                    flatshading=True,
                    showscale=False,
                )
            )
            closed = np.vstack([tile, tile[0]])
            data.append(
                go.Scatter3d(
                    x=closed[:, 0],
                    y=closed[:, 1],
                    z=closed[:, 2],
                    mode="lines",
                    line=dict(color="#111827", width=3),
                    showlegend=False,
                )
            )
        if rope_frames:
            rope = rope_frames[frame_id]
            closed = np.vstack([rope, rope[0]])
            data.append(
                go.Scatter3d(
                    x=closed[:, 0],
                    y=closed[:, 1],
                    z=closed[:, 2],
                    mode="lines+markers",
                    line=dict(color="#dc2626", width=5),
                    marker=dict(size=4, color="#ef4444"),
                )
            )
        if handle_frames:
            h = handle_frames[frame_id]
            data.append(go.Scatter3d(x=[h[0]], y=[h[1]], z=[h[2]], mode="markers", marker=dict(size=7, color="#b11226")))
        frames.append(go.Frame(data=data, name=str(frame_id)))
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=frame.name)
                    for frame in frames
                ]
            )
        ],
    )
    _style_scene(fig)
    return fig


def write_animation_html(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")


def tile_assembly_animation(
    template: TileAssembly,
    tile_frames: list[np.ndarray],
    step: int = 3,
    title: str = "OneString actuation simulation",
) -> go.Figure:
    if not tile_frames:
        return go.Figure()
    stride = max(1, step)
    first = TileAssembly(
        vertices=tile_frames[0],
        top_faces=template.top_faces,
        bottom_faces=template.bottom_faces,
        side_faces=template.side_faces,
        stage=template.stage,
    )
    fig = go.Figure()
    add_tile_assembly(fig, first)
    frames = []
    for frame_id in range(0, len(tile_frames), stride):
        frame_assembly = TileAssembly(
            vertices=tile_frames[frame_id],
            top_faces=template.top_faces,
            bottom_faces=template.bottom_faces,
            side_faces=template.side_faces,
            stage=template.stage,
        )
        frame_fig = go.Figure()
        add_tile_assembly(frame_fig, frame_assembly)
        frames.append(go.Frame(data=frame_fig.data, name=str(frame_id)))
    fig.frames = frames
    fig.update_layout(
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}], label=frame.name)
                    for frame in frames
                ]
            )
        ],
    )
    _style_scene(fig)
    return fig


def assembly_progress_animation(
    state: OneStringDesignState,
    frame_count: int = 56,
    title: str = "OneString assembly progression",
    max_tiles: int | None = None,
    show_target: bool = True,
    show_path: bool = False,
    motion_mode: str = "simultaneous_hinge_contraction",
) -> go.Figure:
    """Build a Plotly browser-side animation.

    Plotly/Streamlit has to send every frame to the browser up front.  Large
    grids therefore become huge JSON payloads and the Play button appears to do
    nothing.  max_tiles intentionally creates a preview subset so that the
    animation remains playable in the app.  The full geometry can still be
    inspected in the normal T2D/T3D views.
    """
    frame_count = max(2, int(frame_count))
    tile_ids = _animation_tile_indices(state, max_tiles)
    start_all = state.tiles_2d_dual_hinge.vertices
    target_all = state.tiles_3d.vertices
    rank_all = _tile_activation_rank(state)
    start = start_all[tile_ids]
    target = target_all[tile_ids]
    rank = rank_all[tile_ids]

    fig = go.Figure()
    if show_target:
        target_preview = _subset_assembly(state.tiles_3d, target, stage="T3D target preview")
        add_tile_assembly(fig, target_preview, color="#94a3b8", opacity=0.16, name="T3D target")
    first = _subset_assembly(state.tiles_2d_dual_hinge, start, stage="assembly frame")
    add_tile_assembly(fig, first, color="#2dd4bf", opacity=0.82, name="assembling tiles")
    if show_path:
        add_gap_graph(fig, state.gap_graph, string_path=state.string_path, lift_gap_ids=[lift.gap_id for lift in state.lift_points])

    frames = []
    for frame_id in range(frame_count):
        vertices = _assembly_vertices_at_frame(start, target, rank, frame_id, frame_count, motion_mode=motion_mode)
        frame_assembly = _subset_assembly(state.tiles_2d_dual_hinge, vertices, stage="assembly frame")
        frame_fig = go.Figure()
        if show_target:
            target_preview = _subset_assembly(state.tiles_3d, target, stage="T3D target preview")
            add_tile_assembly(frame_fig, target_preview, color="#94a3b8", opacity=0.16, name="T3D target")
        add_tile_assembly(frame_fig, frame_assembly, color="#2dd4bf", opacity=0.82, name="assembling tiles")
        if show_path:
            add_gap_graph(frame_fig, state.gap_graph, string_path=state.string_path, lift_gap_ids=[lift.gap_id for lift in state.lift_points])
        frames.append(go.Frame(data=frame_fig.data, name=str(frame_id)))
    fig.frames = frames
    fig.update_layout(
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.06,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 70, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}],
                    ),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue={"prefix": "frame "},
                pad={"t": 35},
                steps=[
                    dict(method="animate", args=[[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=frame.name)
                    for frame in frames
                ],
            )
        ],
    )
    _style_scene(fig)
    _fix_scene_bounds(fig, start, target)
    return fig


def assembly_progress_frame_figure(
    state: OneStringDesignState,
    frame_index: int,
    frame_count: int = 56,
    max_tiles: int | None = 900,
    show_target: bool = True,
    show_path: bool = False,
    title: str = "OneString assembly progression frame",
    motion_mode: str = "simultaneous_hinge_contraction",
) -> go.Figure:
    """Build one lightweight assembly frame for Streamlit-side playback.

    Unlike Plotly frames, this sends only one frame at a time to the browser.
    It is slower per frame but much more reliable for large grids because the
    browser does not receive a massive all-frame JSON bundle.
    """
    frame_count = max(2, int(frame_count))
    frame_index = int(np.clip(frame_index, 0, frame_count - 1))
    tile_ids = _animation_tile_indices(state, max_tiles)
    start = state.tiles_2d_dual_hinge.vertices[tile_ids]
    target = state.tiles_3d.vertices[tile_ids]
    rank = _tile_activation_rank(state)[tile_ids]
    vertices = _assembly_vertices_at_frame(start, target, rank, frame_index, frame_count, motion_mode=motion_mode)

    fig = go.Figure()
    if show_target:
        target_preview = _subset_assembly(state.tiles_3d, target, stage="T3D target preview")
        add_tile_assembly(fig, target_preview, color="#94a3b8", opacity=0.15, name="T3D target")
    frame_assembly = _subset_assembly(state.tiles_2d_dual_hinge, vertices, stage="assembly frame")
    add_tile_assembly(fig, frame_assembly, color="#2dd4bf", opacity=0.84, name="assembling tiles")
    if show_path:
        add_gap_graph(fig, state.gap_graph, string_path=state.string_path, lift_gap_ids=[lift.gap_id for lift in state.lift_points])
    fig.update_layout(title=f"{title}: {frame_index + 1}/{frame_count}")
    _style_scene(fig)
    _fix_scene_bounds(fig, start, target)
    return fig


def _assembly_vertices_at_frame(
    start: np.ndarray,
    target: np.ndarray,
    rank: np.ndarray,
    frame_index: int,
    frame_count: int,
    motion_mode: str = "simultaneous_hinge_contraction",
) -> np.ndarray:
    """Return preview vertices for the assembly animation.

    The default intentionally does not follow the string path tile-by-tile.
    The fabricated mechanism is better represented as many hinge/channel gaps
    shortening at the same time: the layout contracts in x/y while the local
    tile geometry rises toward T3D.  The old boundary-order animation is kept as
    a debug mode because it is useful for checking the path, but it is not the
    physical intuition the user expects to see.
    """
    t = float(frame_index) / max(1, frame_count - 1)
    if motion_mode == "boundary_string_order":
        alpha = _smoothstep(np.clip((t - rank) / 0.28, 0.0, 1.0))[:, None, None]
        return (1.0 - alpha) * start + alpha * target
    return _simultaneous_hinge_contraction_vertices(start, target, t)


def _simultaneous_hinge_contraction_vertices(start: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    """Synchronous hinge-shortening style preview.

    Each tile is animated as a centroid plus local shape.  Centroids contract
    from the flat T2D layout toward their assembled T3D positions at the same
    time, and local tile orientation/thickness blends toward T3D.  This avoids
    the previous wave-like activation and makes the whole sheet rise together.
    """
    t = float(np.clip(t, 0.0, 1.0))
    # Hinge/channel shortening is mostly an in-plane contraction first; height
    # follows smoothly so the object appears to rise rather than teleport.
    center_alpha = _smoothstep(t)
    local_alpha = _smoothstep(np.clip((t - 0.04) / 0.96, 0.0, 1.0))
    z_alpha = _smoothstep(np.clip((t - 0.02) / 0.98, 0.0, 1.0))

    start_center = np.nanmean(start, axis=1, keepdims=True)
    target_center = np.nanmean(target, axis=1, keepdims=True)
    center = (1.0 - center_alpha) * start_center + center_alpha * target_center

    start_local = start - start_center
    target_local = target - target_center
    local = (1.0 - local_alpha) * start_local + local_alpha * target_local
    vertices = center + local

    # Bias z separately so the visual reads as simultaneous lifting caused by
    # shrinking hinges/gaps, not as independent tiles crawling along a path.
    vertices[..., 2] = (1.0 - z_alpha) * start[..., 2] + z_alpha * target[..., 2]
    return vertices


def _animation_tile_indices(state: OneStringDesignState, max_tiles: int | None) -> np.ndarray:
    tile_count = state.tiles_2d_dual_hinge.tile_count
    if max_tiles is None or max_tiles <= 0 or tile_count <= max_tiles:
        return np.arange(tile_count, dtype=int)
    stride = int(np.ceil(tile_count / max_tiles))
    ids = np.arange(0, tile_count, stride, dtype=int)
    if ids[-1] != tile_count - 1:
        ids = np.append(ids, tile_count - 1)
    return ids


def _tile_activation_rank(state: OneStringDesignState) -> np.ndarray:
    tile_count = state.tiles_2d_dual_hinge.tile_count
    order = _tile_activation_order(state)
    if len(order) != tile_count:
        order = np.arange(tile_count)
    rank = np.empty(tile_count, dtype=float)
    rank[order] = np.linspace(0.0, 0.72, tile_count) if tile_count > 1 else 0.0
    return rank


def _subset_assembly(template: TileAssembly, vertices: np.ndarray, stage: str) -> TileAssembly:
    return TileAssembly(
        vertices=vertices,
        top_faces=template.top_faces,
        bottom_faces=template.bottom_faces,
        side_faces=template.side_faces,
        stage=stage,
        metrics=dict(getattr(template, "metrics", {}) or {}),
    )


def _fix_scene_bounds(fig: go.Figure, start: np.ndarray, target: np.ndarray) -> None:
    pts = np.concatenate([start.reshape(-1, 3), target.reshape(-1, 3)], axis=0)
    if len(pts) == 0:
        return
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    center = 0.5 * (lo + hi)
    radius = float(np.nanmax(hi - lo) * 0.56)
    if not np.isfinite(radius) or radius <= 1e-9:
        radius = 1.0
    fig.update_layout(
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[center[0] - radius, center[0] + radius], showbackground=False),
            yaxis=dict(range=[center[1] - radius, center[1] + radius], showbackground=False),
            zaxis=dict(range=[center[2] - radius, center[2] + radius], showbackground=False),
        )
    )

def _smoothstep(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def _tile_activation_order(state: OneStringDesignState) -> np.ndarray:
    seen: list[int] = []
    for gap_id in state.string_path.gap_ids:
        if gap_id >= len(state.gap_graph.gaps):
            continue
        for tile_id in state.gap_graph.gaps[gap_id].surrounding_tiles:
            if tile_id not in seen:
                seen.append(tile_id)
    for tile_id in range(state.tiles_2d_dual_hinge.tile_count):
        if tile_id not in seen:
            seen.append(tile_id)
    return np.asarray(seen, dtype=int)
