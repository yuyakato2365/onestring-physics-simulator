from __future__ import annotations

import numpy as np


def solve_distance_positions(
    p0: np.ndarray,
    p1: np.ndarray,
    inv_m0: float,
    inv_m1: float,
    rest_length: float,
    stiffness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return p0, p1, 0.0
    total_inv = inv_m0 + inv_m1
    if total_inv <= 1e-12:
        return p0, p1, length - rest_length

    constraint = length - rest_length
    correction = stiffness * constraint * (delta / length) / total_inv
    return p0 + inv_m0 * correction, p1 - inv_m1 * correction, constraint


def integrate_verlet(
    positions: np.ndarray,
    previous_positions: np.ndarray,
    inv_masses: np.ndarray,
    gravity: np.ndarray,
    dt: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    velocities = (positions - previous_positions) * max(0.0, 1.0 - damping)
    previous = positions.copy()
    movable = inv_masses > 0.0
    positions[movable] = positions[movable] + velocities[movable] + gravity * (dt * dt)
    return positions, previous


def update_velocities(positions: np.ndarray, previous_positions: np.ndarray, dt: float) -> np.ndarray:
    return (positions - previous_positions) / max(dt, 1e-8)
