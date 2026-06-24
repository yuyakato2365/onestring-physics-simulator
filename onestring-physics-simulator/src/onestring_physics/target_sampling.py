from __future__ import annotations

import numpy as np

from .heightfields import HeightField


def sample_tile_centers(target: HeightField, nx: int, ny: int, tile_size: float) -> np.ndarray:
    xs = (np.arange(nx) + 0.5 - nx / 2.0) * tile_size
    ys = (np.arange(ny) + 0.5 - ny / 2.0) * tile_size
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = target.height(xx, yy)
    return np.stack([xx, yy, zz], axis=-1)


def sample_vertices(target: HeightField, nx: int, ny: int, tile_size: float) -> np.ndarray:
    return target.sample_grid(nx, ny, tile_size)
