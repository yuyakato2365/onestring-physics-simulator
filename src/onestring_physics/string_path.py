from __future__ import annotations

import numpy as np

from .quad_grid import QuadGrid


def boundary_string_path(grid: QuadGrid) -> np.ndarray:
    return grid.boundary_midpoints()


def nearest_tile_corners(points: np.ndarray, tile_positions: np.ndarray) -> list[tuple[int, int]]:
    flat = tile_positions.reshape(-1, 3)
    out: list[tuple[int, int]] = []
    for point in np.asarray(points, dtype=float):
        idx = int(np.argmin(np.linalg.norm(flat - point, axis=1)))
        out.append((idx // 4, idx % 4))
    return out


def lift_anchor_points(tile_positions: np.ndarray) -> np.ndarray:
    return np.mean(tile_positions, axis=1)
