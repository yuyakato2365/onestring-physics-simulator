"""Highlight residual optcuts_test K2D overlap tiles in the flat-layout view."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def install_optcuts_test_k2d_overlap_visualization_patch() -> None:
    from . import visualization

    if getattr(visualization, "_onestring_k2d_overlap_visualization_installed", False):
        return

    base = visualization.figure_flat_tile_layout

    def figure_with_overlap_diagnostics(layout, title="K2D flat tile layout", hinge_graph=None):
        fig = base(layout, title=title, hinge_graph=hinge_graph)
        metrics = dict(getattr(layout, "metrics", {}) or {})
        raw_ids = metrics.get("onestring_k2d_overlap_tile_ids", []) or []
        try:
            overlap_ids = sorted({int(v) for v in raw_ids})
        except Exception:
            overlap_ids = []
        tiles = np.asarray(getattr(layout, "tile_top_vertices_3d", []), dtype=float)
        overlap_ids = [i for i in overlap_ids if 0 <= i < len(tiles)]
        if not overlap_ids:
            return fig

        x: list[float] = []
        y: list[float] = []
        z: list[float] = []
        ii: list[int] = []
        jj: list[int] = []
        kk: list[int] = []
        ex: list[float | None] = []
        ey: list[float | None] = []
        ez: list[float | None] = []
        hover: list[str] = []
        z_lift = 0.002
        pair_count = int(metrics.get("onestring_k2d_residual_overlap_pair_count", 0))

        for tile_id in overlap_ids:
            tile = np.asarray(tiles[tile_id], dtype=float).copy()
            tile[:, 2] += z_lift
            base_idx = len(x)
            x.extend(tile[:, 0].tolist())
            y.extend(tile[:, 1].tolist())
            z.extend(tile[:, 2].tolist())
            ii.extend([base_idx, base_idx])
            jj.extend([base_idx + 1, base_idx + 2])
            kk.extend([base_idx + 2, base_idx + 3])
            hover.extend([f"overlapping K2D tile {tile_id}"] * 4)
            closed = np.vstack([tile, tile[0]])
            ex.extend([*closed[:, 0].tolist(), None])
            ey.extend([*closed[:, 1].tolist(), None])
            ez.extend([*closed[:, 2].tolist(), None])

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=ii,
                j=jj,
                k=kk,
                color="#ef4444",
                opacity=0.92,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                text=hover,
                hoverinfo="text",
                name=f"residual overlap tiles ({len(overlap_ids)})",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color="#991b1b", width=5),
                name="residual overlap tile edges",
            )
        )
        fig.update_layout(
            title=(
                f"{title} | residual overlap: {pair_count} pairs / "
                f"{len(overlap_ids)} tiles (red, nonfatal)"
            )
        )
        return fig

    visualization.figure_flat_tile_layout = figure_with_overlap_diagnostics
    visualization._onestring_k2d_overlap_visualization_installed = True


__all__ = ["install_optcuts_test_k2d_overlap_visualization_patch"]
