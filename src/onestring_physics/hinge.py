from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PointHingeConstraint:
    tile_a: int
    corner_a: int
    tile_b: int
    corner_b: int
    stiffness: float = 1.0

    def solve(self, tile_positions: np.ndarray, inv_masses: np.ndarray) -> float:
        pa = tile_positions[self.tile_a, self.corner_a]
        pb = tile_positions[self.tile_b, self.corner_b]
        total = inv_masses[self.tile_a] + inv_masses[self.tile_b]
        error = pb - pa
        if total <= 1e-12:
            return float(np.linalg.norm(error))
        correction = self.stiffness * error / total
        tile_positions[self.tile_a, self.corner_a] += inv_masses[self.tile_a] * correction
        tile_positions[self.tile_b, self.corner_b] -= inv_masses[self.tile_b] * correction
        return float(np.linalg.norm(error))


@dataclass
class AngularHingeLimit:
    min_dot: float = -0.95

    def violation(self, normal_a: np.ndarray, normal_b: np.ndarray) -> float:
        na = normal_a / max(float(np.linalg.norm(normal_a)), 1e-12)
        nb = normal_b / max(float(np.linalg.norm(normal_b)), 1e-12)
        return max(0.0, self.min_dot - float(np.dot(na, nb)))


@dataclass
class HingeSpring:
    rest_distance: float
    stiffness: float = 0.1
