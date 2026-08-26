"""Visual diagnostics for residual optcuts_test K2D constraint violations."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _overlay_tiles(fig, tiles: np.ndarray, ids: list[int], *, color: str, edge_color: str, name: str, z_lift: float) -> None:
    if not ids:
        return
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
    for tile_id in ids:
        tile = np.asarray(tiles[tile_id], dtype=float).copy()
        tile[:, 2] += z_lift
        base_idx = len(x)
        x.extend(tile[:, 0].tolist())
        y.extend(tile[:, 1].tolist())
        z.extend(tile[:, 2].tolist())
        ii.extend([base_idx, base_idx])
        jj.extend([base_idx + 1, base_idx + 2])
        kk.extend([base_idx + 2, base_idx + 3])
        hover.extend([f"{name} tile {tile_id}"] * 4)
        closed = np.vstack([tile, tile[0]])
        ex.extend([*closed[:, 0].tolist(), None])
        ey.extend([*closed[:, 1].tolist(), None])
        ez.extend([*closed[:, 2].tolist(), None])
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z, i=ii, j=jj, k=kk,
        color=color, opacity=0.88, flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        text=hover, hoverinfo="text", name=f"{name} ({len(ids)})",
    ))
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color=edge_color, width=5), name=f"{name} edges",
    ))


def install_optcuts_test_k2d_overlap_visualization_patch() -> None:
    from . import visualization

    if getattr(visualization, "_onestring_k2d_overlap_visualization_installed", False):
        return

    base = visualization.figure_flat_tile_layout

    def figure_with_overlap_diagnostics(layout, title="K2D flat tile layout", hinge_graph=None):
        fig = base(layout, title=title, hinge_graph=hinge_graph)
        metrics = dict(getattr(layout, "metrics", {}) or {})
        tiles = np.asarray(getattr(layout, "tile_top_vertices_3d", []), dtype=float)

        def valid_ids(key: str) -> list[int]:
            try:
                ids = sorted({int(v) for v in (metrics.get(key, []) or [])})
            except Exception:
                ids = []
            return [i for i in ids if 0 <= i < len(tiles)]

        overlap_ids = valid_ids("onestring_k2d_overlap_tile_ids")
        hinge_ids = valid_ids("onestring_k2d_kinematic_hinge_violation_tile_ids")
        # Red overlap diagnostics take visual priority over orange loop-hinge diagnostics.
        hinge_only = [i for i in hinge_ids if i not in set(overlap_ids)]
        _overlay_tiles(fig, tiles, hinge_only, color="#f59e0b", edge_color="#92400e", name="loop-hinge violation", z_lift=0.0015)
        _overlay_tiles(fig, tiles, overlap_ids, color="#ef4444", edge_color="#991b1b", name="residual overlap", z_lift=0.0025)

        feasible = metrics.get("onestring_k2d_kinematic_feasible", True)
        if feasible is False:
            pair_count = int(metrics.get("onestring_k2d_residual_overlap_pair_count", 0))
            loop_max = float(metrics.get("onestring_k2d_kinematic_loop_hinge_max_error", 0.0))
            hinge_tol = float(metrics.get("onestring_k2d_hard_hinge_tolerance", 0.0))
            tree_max = float(metrics.get("onestring_k2d_kinematic_tree_hinge_max_error", 0.0))
            fig.update_layout(title=(
                f"{title} | CONSTRAINTS NOT SATISFIED (nonfatal diagnostic): "
                f"overlap pairs={pair_count}, loop hinge max={loop_max:.4g} "
                f"(tol={hinge_tol:.4g}), tree hinge max={tree_max:.3g}"
            ))
            fig.add_annotation(
                text="K2D kinematic constraints are not fully satisfied; downstream results are diagnostic.",
                x=0.5, y=1.02, xref="paper", yref="paper", showarrow=False,
                font=dict(color="#b91c1c", size=13),
            )
        elif overlap_ids:
            pair_count = int(metrics.get("onestring_k2d_residual_overlap_pair_count", 0))
            fig.update_layout(title=f"{title} | residual overlap: {pair_count} pairs / {len(overlap_ids)} tiles")
        return fig

    visualization.figure_flat_tile_layout = figure_with_overlap_diagnostics
    visualization._onestring_k2d_overlap_visualization_installed = True


__all__ = ["install_optcuts_test_k2d_overlap_visualization_patch"]
