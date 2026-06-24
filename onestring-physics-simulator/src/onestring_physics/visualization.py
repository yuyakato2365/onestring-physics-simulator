from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .design_optimizer import DesignResult


def figure_target(design: DesignResult) -> go.Figure:
    grid = design.target.sample_grid(design.grid.nx, design.grid.ny, design.grid.tile_size)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=grid[..., 0],
            y=grid[..., 1],
            z=grid[..., 2],
            colorscale="Viridis",
            opacity=0.72,
            showscale=False,
            name="target",
        )
    )
    _style_scene(fig)
    return fig


def figure_tiles(
    tiles: np.ndarray,
    design: DesignResult | None = None,
    rope: np.ndarray | None = None,
    pull_handle: np.ndarray | None = None,
    title: str = "tiles",
) -> go.Figure:
    fig = go.Figure()
    add_tiles(fig, tiles)
    if design is not None:
        add_hinges(fig, tiles, design)
    if rope is not None:
        add_rope(fig, rope)
    if pull_handle is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[pull_handle[0]],
                y=[pull_handle[1]],
                z=[pull_handle[2]],
                mode="markers",
                marker=dict(size=7, color="#b11226"),
                name="pull handle",
            )
        )
    fig.update_layout(title=title)
    _style_scene(fig)
    return fig


def figure_comparison(design: DesignResult, final_tiles: np.ndarray) -> go.Figure:
    fig = go.Figure()
    add_tiles(fig, design.assembled_tiles, color="#3b82f6", opacity=0.38, name="optimized assembled")
    add_tiles(fig, final_tiles, color="#f97316", opacity=0.72, name="physical final")
    _style_scene(fig)
    return fig


def figure_loss(loss_history: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode="lines+markers", name="mean squared residual"))
    fig.update_layout(xaxis_title="recorded step", yaxis_title="loss", height=280)
    return fig


def add_tiles(
    fig: go.Figure,
    tiles: np.ndarray,
    color: str = "#2dd4bf",
    opacity: float = 0.78,
    name: str = "rigid tiles",
) -> None:
    first = True
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    for tile in np.asarray(tiles, dtype=float):
        fig.add_trace(
            go.Mesh3d(
                x=tile[:, 0],
                y=tile[:, 1],
                z=tile[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=opacity,
                flatshading=False,
                lighting=lighting,
                name=name,
                showscale=False,
                showlegend=first,
            )
        )
        closed = np.vstack([tile, tile[0]])
        fig.add_trace(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                line=dict(color="#111827", width=3),
                name="tile edges",
                showlegend=False,
            )
        )
        first = False


def add_hinges(fig: go.Figure, tiles: np.ndarray, design: DesignResult) -> None:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for hinge in design.hinges:
        a = (tiles[hinge.tile_a, hinge.corner_a0] + tiles[hinge.tile_a, hinge.corner_a1]) * 0.5
        b = (tiles[hinge.tile_b, hinge.corner_b0] + tiles[hinge.tile_b, hinge.corner_b1]) * 0.5
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            line=dict(color="#4b5563", width=4),
            marker=dict(size=3, color="#4b5563"),
            name="hinges",
        )
    )


def add_rope(fig: go.Figure, rope: np.ndarray) -> None:
    rope = np.asarray(rope, dtype=float)
    closed = np.vstack([rope, rope[0]]) if len(rope) > 2 else rope
    fig.add_trace(
        go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines+markers",
            line=dict(color="#dc2626", width=5),
            marker=dict(size=4, color="#ef4444"),
            name="rope",
        )
    )


def _style_scene(fig: go.Figure) -> None:
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, b=0, t=38),
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
    )
