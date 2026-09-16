"""Fast AssemblyAnimation path, with a macOS-oriented payload cap.

The legacy animation recomputes a per-tile SVD/rotation for every frame and
re-sends static target/path traces in every Plotly frame.  This patch keeps the
same visual motion while:

1. computing all tile rigid rotations once with batched NumPy SVD,
2. reusing those rotation vectors for every frame,
3. keeping target/path traces static instead of duplicating them in frames,
4. using conservative frame/tile preview caps on macOS to keep the browser
   payload responsive.

It patches only ``onestring_physics.animation.assembly_progress_animation``.
No OneString numerical pipeline, Split, OptCuts, K2D, or T3D data is changed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _prepare_rigid_motion(start: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """Precompute per-tile rigid motion once for all animation frames."""
    start_arr = np.asarray(start, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    start_center = np.nanmean(start_arr, axis=1, keepdims=True)
    target_center = np.nanmean(target_arr, axis=1, keepdims=True)
    start_local = start_arr - start_center
    target_local = target_arr - target_center

    valid = np.all(np.isfinite(start_local), axis=(1, 2)) & np.all(
        np.isfinite(target_local), axis=(1, 2)
    )
    rotvec = np.zeros((len(start_arr), 3), dtype=float)

    if np.any(valid):
        source = start_local[valid]
        dest = target_local[valid]
        covariance = np.swapaxes(source, 1, 2) @ dest
        try:
            u, _s, vt = np.linalg.svd(covariance)
            rotations = np.swapaxes(vt, 1, 2) @ np.swapaxes(u, 1, 2)
            negative = np.linalg.det(rotations) < 0.0
            if np.any(negative):
                vt = vt.copy()
                vt[negative, -1, :] *= -1.0
                rotations = np.swapaxes(vt, 1, 2) @ np.swapaxes(u, 1, 2)
            from scipy.spatial.transform import Rotation

            rotvec[valid] = Rotation.from_matrix(rotations).as_rotvec()
        except Exception:
            valid[:] = False

    return {
        "start_center": start_center,
        "target_center": target_center,
        "start_local": start_local,
        "target_local": target_local,
        "valid": valid,
        "rotvec": rotvec,
    }


def _smoothstep_scalar(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _fast_vertices_at_frame(
    start: np.ndarray,
    target: np.ndarray,
    rank: np.ndarray,
    frame_index: int,
    frame_count: int,
    motion_mode: str,
    rigid: dict[str, np.ndarray] | None,
) -> np.ndarray:
    t = float(frame_index) / max(1, int(frame_count) - 1)
    if motion_mode == "boundary_string_order":
        x = np.clip((t - rank) / 0.28, 0.0, 1.0)
        alpha = (x * x * (3.0 - 2.0 * x))[:, None, None]
        return (1.0 - alpha) * start + alpha * target

    if rigid is None:
        rigid = _prepare_rigid_motion(start, target)

    center_alpha = _smoothstep_scalar(t)
    local_alpha = _smoothstep_scalar((t - 0.04) / 0.96)
    z_alpha = _smoothstep_scalar((t - 0.02) / 0.98)

    start_center = rigid["start_center"]
    target_center = rigid["target_center"]
    center = (1.0 - center_alpha) * start_center + center_alpha * target_center
    center[..., 2] = (
        (1.0 - z_alpha) * start_center[..., 2]
        + z_alpha * target_center[..., 2]
    )

    start_local = rigid["start_local"]
    target_local = rigid["target_local"]
    valid = rigid["valid"]
    local = (1.0 - local_alpha) * start_local + local_alpha * target_local

    if np.any(valid):
        try:
            from scipy.spatial.transform import Rotation

            matrices = Rotation.from_rotvec(rigid["rotvec"][valid] * local_alpha).as_matrix()
            local[valid] = start_local[valid] @ np.swapaxes(matrices, 1, 2)
        except Exception:
            pass

    return center + local


def assembly_progress_animation_fast(
    state: Any,
    frame_count: int = 56,
    title: str = "OneString assembly progression",
    max_tiles: int | None = None,
    show_target: bool = True,
    show_path: bool = False,
    motion_mode: str = "simultaneous_hinge_contraction",
) -> go.Figure:
    from . import animation as legacy
    from .visualization import add_gap_graph, add_tile_assembly, _style_scene

    requested_frames = max(2, int(frame_count))
    requested_max_tiles = max_tiles
    if sys.platform == "darwin":
        frame_cap = max(8, _env_int("ONESTRING_ASSEMBLY_FRAMES_MAC", 36))
        tile_cap = max(80, _env_int("ONESTRING_ASSEMBLY_MAX_TILES_MAC", 500))
        frame_count = min(requested_frames, frame_cap)
        max_tiles = tile_cap if max_tiles is None else min(int(max_tiles), tile_cap)
    else:
        frame_count = requested_frames

    tile_ids = legacy._animation_tile_indices(state, max_tiles)
    start = np.asarray(state.tiles_2d_dual_hinge.vertices, dtype=float)[tile_ids]
    target = np.asarray(state.tiles_3d.vertices, dtype=float)[tile_ids]
    rank = legacy._tile_activation_rank(state)[tile_ids]
    rigid = None
    if motion_mode != "boundary_string_order":
        rigid = _prepare_rigid_motion(start, target)

    fig = go.Figure()
    if show_target:
        target_preview = legacy._subset_assembly(
            state.tiles_3d, target, stage="T3D target preview"
        )
        add_tile_assembly(
            fig,
            target_preview,
            color="#94a3b8",
            opacity=0.16,
            name="T3D target",
        )

    dynamic_start = len(fig.data)
    first = legacy._subset_assembly(
        state.tiles_2d_dual_hinge, start, stage="assembly frame"
    )
    add_tile_assembly(
        fig,
        first,
        color="#2dd4bf",
        opacity=0.82,
        name="assembling tiles",
    )
    dynamic_indices = list(range(dynamic_start, len(fig.data)))

    if show_path:
        add_gap_graph(
            fig,
            state.gap_graph,
            string_path=state.string_path,
            lift_gap_ids=[lift.gap_id for lift in state.lift_points],
        )

    frames: list[go.Frame] = []
    for frame_id in range(frame_count):
        vertices = _fast_vertices_at_frame(
            start,
            target,
            rank,
            frame_id,
            frame_count,
            motion_mode,
            rigid,
        )
        frame_assembly = legacy._subset_assembly(
            state.tiles_2d_dual_hinge, vertices, stage="assembly frame"
        )
        frame_fig = go.Figure()
        add_tile_assembly(
            frame_fig,
            frame_assembly,
            color="#2dd4bf",
            opacity=0.82,
            name="assembling tiles",
        )
        frames.append(
            go.Frame(
                data=frame_fig.data,
                traces=dynamic_indices,
                name=str(frame_id),
            )
        )

    fig.frames = frames
    fig.update_layout(
        title=title,
        meta={
            "assembly_animation_backend": "fast_precomputed_rigid_motion",
            "requested_frame_count": int(requested_frames),
            "effective_frame_count": int(frame_count),
            "requested_max_tiles": requested_max_tiles,
            "effective_tile_count": int(len(tile_ids)),
            "platform": sys.platform,
        },
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
                        args=[
                            None,
                            {
                                "frame": {"duration": 55, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue={"prefix": "frame "},
                pad={"t": 35},
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [frame.name],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=frame.name,
                    )
                    for frame in frames
                ],
            )
        ],
    )
    _style_scene(fig)
    legacy._fix_scene_bounds(fig, start, target)
    return fig


def install_fast_assembly_animation_patch() -> None:
    from . import animation

    if getattr(animation, "_onestring_fast_assembly_animation_installed", False):
        return
    animation._onestring_original_assembly_progress_animation = (
        animation.assembly_progress_animation
    )
    animation.assembly_progress_animation = assembly_progress_animation_fast
    animation._onestring_fast_assembly_animation_installed = True
