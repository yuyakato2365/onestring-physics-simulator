from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .design_optimizer import DesignResult
from .visualization import add_rope, add_tiles, _style_scene


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
