from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

try:  # scipy is the preferred optimizer; the fallback keeps demos importable.
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - exercised only in minimal environments
    least_squares = None

from .heightfields import HeightField
from .quad_grid import HingeSpec, QuadGrid, create_quad_grid


@dataclass
class DesignParameters:
    max_iterations: int = 30
    w_target: float = 1.0
    w_rigid: float = 0.12
    w_hinge: float = 0.2
    w_smooth: float = 0.08
    w_flat: float = 0.02


@dataclass
class DesignResult:
    grid: QuadGrid
    target: HeightField
    assembled_tiles: np.ndarray
    flat_tiles: np.ndarray
    vertex_positions: np.ndarray
    target_vertices: np.ndarray
    hinges: list[HingeSpec]
    boundary_string_path: np.ndarray
    lift_anchors: np.ndarray
    loss_history: list[float] = field(default_factory=list)

    @property
    def assembled_centers(self) -> np.ndarray:
        return np.mean(self.assembled_tiles, axis=1)

    @property
    def flat_centers(self) -> np.ndarray:
        return np.mean(self.flat_tiles, axis=1)

    def target_vertices_for_tiles(self) -> np.ndarray:
        return self.grid.tile_corners_from_vertices(self.target_vertices)


def optimize_design(
    target: HeightField,
    nx: int = 3,
    ny: int | None = None,
    tile_size: float = 1.0,
    gap_size: float = 0.08,
    params: DesignParameters | None = None,
) -> DesignResult:
    ny = nx if ny is None else ny
    params = params or DesignParameters()
    grid = create_quad_grid(nx, ny, tile_size, gap_size)
    base_vertices = grid.vertex_positions.copy()
    target_vertices = target.sample_grid(nx, ny, tile_size).reshape(-1, 3)
    z0 = np.clip(target_vertices[:, 2], -2.0 * tile_size, 2.0 * tile_size)
    rest_edges = _grid_edge_lengths(base_vertices, grid)
    loss_history: list[float] = []

    def residual(z_values: np.ndarray) -> np.ndarray:
        vertices = base_vertices.copy()
        vertices[:, 2] = z_values
        tile_corners = grid.tile_corners_from_vertices(vertices)
        res: list[np.ndarray] = []

        target_z = target.height(vertices[:, 0], vertices[:, 1])
        res.append(params.w_target * (vertices[:, 2] - target_z))

        rigidity_terms: list[float] = []
        for tile in grid.tiles or []:
            pts = tile_corners[tile.id]
            pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
            flat = grid.flat_tile_corners(include_gap=False)[tile.id]
            for a, b in pairs:
                rigidity_terms.append(np.linalg.norm(pts[a] - pts[b]) - np.linalg.norm(flat[a] - flat[b]))
        res.append(params.w_rigid * np.asarray(rigidity_terms))

        hinge_terms: list[float] = []
        for hinge in grid.hinges or []:
            a0 = tile_corners[hinge.tile_a, hinge.corner_a0]
            a1 = tile_corners[hinge.tile_a, hinge.corner_a1]
            b0 = tile_corners[hinge.tile_b, hinge.corner_b0]
            b1 = tile_corners[hinge.tile_b, hinge.corner_b1]
            hinge_terms.extend((a0 - b0).tolist())
            hinge_terms.extend((a1 - b1).tolist())
            hinge_terms.extend((((a0 + a1) * 0.5) - ((b0 + b1) * 0.5)).tolist())
        if hinge_terms:
            res.append(params.w_hinge * np.asarray(hinge_terms))

        smooth_terms: list[float] = []
        z_grid = vertices[:, 2].reshape(ny + 1, nx + 1)
        smooth_terms.extend(np.diff(z_grid, n=2, axis=0).ravel().tolist())
        smooth_terms.extend(np.diff(z_grid, n=2, axis=1).ravel().tolist())
        if smooth_terms:
            res.append(params.w_smooth * np.asarray(smooth_terms))

        res.append(params.w_flat * z_values)
        return np.concatenate([r.ravel() for r in res if r.size])

    def callback_like(z_values: np.ndarray) -> None:
        r = residual(z_values)
        loss_history.append(float(np.mean(r * r)))

    # scipy's least_squares callback is version-dependent, so use a bounded
    # max_nfev and record the final loss plus a cheap initial estimate.
    callback_like(z0)
    if least_squares is not None:
        opt = least_squares(residual, z0, max_nfev=max(5, params.max_iterations), method="trf")
    else:
        opt = SimpleNamespace(x=_fallback_height_fit(z0, nx, ny, params.max_iterations))
    callback_like(opt.x)

    vertices = base_vertices.copy()
    vertices[:, 2] = opt.x
    assembled_tiles = grid.tile_corners_from_vertices(vertices)
    flat_tiles = grid.flat_tile_corners(include_gap=True)
    lift_anchors = np.mean(flat_tiles, axis=1)

    return DesignResult(
        grid=grid,
        target=target,
        assembled_tiles=assembled_tiles,
        flat_tiles=flat_tiles,
        vertex_positions=vertices,
        target_vertices=target_vertices,
        hinges=list(grid.hinges or []),
        boundary_string_path=grid.boundary_midpoints(),
        lift_anchors=lift_anchors,
        loss_history=loss_history,
    )


def _grid_edge_lengths(vertices: np.ndarray, grid: QuadGrid) -> np.ndarray:
    lengths: list[float] = []
    for tile in grid.tiles or []:
        pts = vertices[list(tile.vertex_ids)]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
            lengths.append(np.linalg.norm(pts[a] - pts[b]))
    return np.asarray(lengths)


def _fallback_height_fit(z0: np.ndarray, nx: int, ny: int, iterations: int) -> np.ndarray:
    z = np.asarray(z0, dtype=float).reshape(ny + 1, nx + 1).copy()
    for _ in range(max(1, iterations // 2)):
        padded = np.pad(z, 1, mode="edge")
        neighbor_avg = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) * 0.25
        z = 0.85 * z + 0.15 * neighbor_avg
    return z.ravel()
