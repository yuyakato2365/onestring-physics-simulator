from __future__ import annotations

import numpy as np


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def tile_deformation(tile_positions: np.ndarray, rest_tile_positions: np.ndarray) -> float:
    current = _tile_lengths(tile_positions)
    rest = _tile_lengths(rest_tile_positions)
    return float(np.max(np.abs(current - rest))) if current.size else 0.0


def kinetic_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    return float(0.5 * mass * np.sum(velocities * velocities))


def potential_energy(positions: np.ndarray, gravity: np.ndarray, mass: float = 1.0) -> float:
    gmag = float(np.linalg.norm(gravity))
    return float(mass * gmag * np.sum(positions[..., 2]))


def _tile_lengths(tile_positions: np.ndarray) -> np.ndarray:
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    values: list[float] = []
    for tile in tile_positions:
        for a, b in pairs:
            values.append(float(np.linalg.norm(tile[a] - tile[b])))
    return np.asarray(values)
