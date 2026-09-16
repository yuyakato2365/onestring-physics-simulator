from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    id: int
    row: int
    col: int
    vertex_ids: tuple[int, int, int, int]


@dataclass(frozen=True)
class HingeSpec:
    tile_a: int
    corner_a0: int
    corner_a1: int
    tile_b: int
    corner_b0: int
    corner_b1: int
    direction: str


@dataclass
class QuadGrid:
    nx: int
    ny: int
    tile_size: float = 1.0
    gap_size: float = 0.08
    tiles: list[TileSpec] | None = None
    hinges: list[HingeSpec] | None = None
    vertex_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.tiles = self._build_tiles()
        self.hinges = self._build_hinges()
        self.vertex_positions = self._build_vertex_positions()

    @property
    def num_tiles(self) -> int:
        return self.nx * self.ny

    def _vertex_id(self, row: int, col: int) -> int:
        return row * (self.nx + 1) + col

    def _build_tiles(self) -> list[TileSpec]:
        tiles: list[TileSpec] = []
        for row in range(self.ny):
            for col in range(self.nx):
                tile_id = row * self.nx + col
                tiles.append(
                    TileSpec(
                        id=tile_id,
                        row=row,
                        col=col,
                        vertex_ids=(
                            self._vertex_id(row, col),
                            self._vertex_id(row, col + 1),
                            self._vertex_id(row + 1, col + 1),
                            self._vertex_id(row + 1, col),
                        ),
                    )
                )
        return tiles

    def _build_hinges(self) -> list[HingeSpec]:
        hinges: list[HingeSpec] = []
        for row in range(self.ny):
            for col in range(self.nx):
                tile = row * self.nx + col
                if col + 1 < self.nx:
                    right = row * self.nx + col + 1
                    hinges.append(HingeSpec(tile, 1, 2, right, 0, 3, "x"))
                if row + 1 < self.ny:
                    up = (row + 1) * self.nx + col
                    hinges.append(HingeSpec(tile, 3, 2, up, 0, 1, "y"))
        return hinges

    def _build_vertex_positions(self) -> np.ndarray:
        xs = (np.arange(self.nx + 1) - self.nx / 2.0) * self.tile_size
        ys = (np.arange(self.ny + 1) - self.ny / 2.0) * self.tile_size
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        return np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)

    def flat_tile_corners(self, include_gap: bool = True) -> np.ndarray:
        corners = np.zeros((self.num_tiles, 4, 3), dtype=float)
        shrink = max(0.0, self.gap_size * 0.5) if include_gap else 0.0
        for tile in self.tiles or []:
            pts = self.vertex_positions[list(tile.vertex_ids)].copy()
            center = np.mean(pts, axis=0)
            if shrink > 0.0:
                vec = pts - center
                length = np.linalg.norm(vec[:, :2], axis=1, keepdims=True)
                scale = np.maximum(length - shrink, 0.0) / np.maximum(length, 1e-8)
                pts[:, :2] = center[:2] + vec[:, :2] * scale
            corners[tile.id] = pts
        return corners

    def tile_corners_from_vertices(self, vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=float).reshape((self.ny + 1) * (self.nx + 1), 3)
        corners = np.zeros((self.num_tiles, 4, 3), dtype=float)
        for tile in self.tiles or []:
            corners[tile.id] = vertices[list(tile.vertex_ids)]
        return corners

    def boundary_midpoints(self) -> np.ndarray:
        flat = self.flat_tile_corners(include_gap=True)
        pts: list[np.ndarray] = []
        for col in range(self.nx):
            pts.append((flat[col, 0] + flat[col, 1]) * 0.5)
        for row in range(self.ny):
            tile = row * self.nx + self.nx - 1
            pts.append((flat[tile, 1] + flat[tile, 2]) * 0.5)
        for col in reversed(range(self.nx)):
            tile = (self.ny - 1) * self.nx + col
            pts.append((flat[tile, 2] + flat[tile, 3]) * 0.5)
        for row in reversed(range(self.ny)):
            tile = row * self.nx
            pts.append((flat[tile, 3] + flat[tile, 0]) * 0.5)
        return np.asarray(pts, dtype=float)


def create_quad_grid(nx: int, ny: int | None = None, tile_size: float = 1.0, gap_size: float = 0.08) -> QuadGrid:
    ny = nx if ny is None else ny
    if nx < 1 or ny < 1:
        raise ValueError("grid dimensions must be positive")
    return QuadGrid(nx=nx, ny=ny, tile_size=tile_size, gap_size=gap_size)
