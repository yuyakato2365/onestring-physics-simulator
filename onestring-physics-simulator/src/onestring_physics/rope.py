from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .xpbd_solver import solve_distance_positions


@dataclass
class RopeParticle:
    position: np.ndarray
    previous_position: np.ndarray
    velocity: np.ndarray
    mass: float = 0.05
    pinned: bool = False

    @property
    def inv_mass(self) -> float:
        return 0.0 if self.pinned or self.mass <= 0.0 else 1.0 / self.mass


@dataclass
class DistanceConstraint:
    a: int
    b: int
    rest_length: float
    stiffness: float = 1.0

    def solve(self, particles: list[RopeParticle]) -> float:
        pa = particles[self.a]
        pb = particles[self.b]
        new_a, new_b, violation = solve_distance_positions(
            pa.position,
            pb.position,
            pa.inv_mass,
            pb.inv_mass,
            self.rest_length,
            self.stiffness,
        )
        pa.position = new_a
        pb.position = new_b
        return abs(float(violation))


@dataclass
class RopeTileAttachment:
    particle: int
    tile: int
    corner: int | None = None
    stiffness: float = 0.8

    def solve(self, particles: list[RopeParticle], tile_positions: np.ndarray, tile_inv_masses: np.ndarray) -> float:
        rp = particles[self.particle]
        tile_point = (
            np.mean(tile_positions[self.tile], axis=0)
            if self.corner is None
            else tile_positions[self.tile, self.corner]
        )
        inv_tile = tile_inv_masses[self.tile] / 4.0
        new_rope, new_tile, violation = solve_distance_positions(
            rp.position,
            tile_point,
            rp.inv_mass,
            inv_tile,
            0.0,
            self.stiffness,
        )
        rp.position = new_rope
        delta = new_tile - tile_point
        if self.corner is None:
            tile_positions[self.tile] += delta
        else:
            tile_positions[self.tile, self.corner] += delta
        return abs(float(violation))


class Rope:
    def __init__(self, particles: list[RopeParticle], constraints: list[DistanceConstraint]):
        self.particles = particles
        self.constraints = constraints

    @classmethod
    def from_polyline(cls, points: np.ndarray, stiffness: float = 1.0, closed: bool = True) -> "Rope":
        points = np.asarray(points, dtype=float)
        particles = [
            RopeParticle(position=p.copy(), previous_position=p.copy(), velocity=np.zeros(3), pinned=False)
            for p in points
        ]
        constraints: list[DistanceConstraint] = []
        count = len(points)
        last = count if not closed else count + 1
        for i in range(last - 1):
            a = i % count
            b = (i + 1) % count
            rest = float(np.linalg.norm(points[a] - points[b]))
            constraints.append(DistanceConstraint(a, b, rest, stiffness))
        return cls(particles, constraints)

    def positions(self) -> np.ndarray:
        return np.asarray([p.position for p in self.particles], dtype=float)

    def integrate(self, gravity: np.ndarray, dt: float, damping: float) -> None:
        for p in self.particles:
            if p.pinned:
                p.previous_position = p.position.copy()
                p.velocity[:] = 0.0
                continue
            velocity = (p.position - p.previous_position) * max(0.0, 1.0 - damping)
            p.previous_position = p.position.copy()
            p.position = p.position + velocity + gravity * dt * dt

    def solve(self, pull_scale: float = 1.0) -> float:
        max_error = 0.0
        for constraint in self.constraints:
            old = constraint.rest_length
            constraint.rest_length = old * pull_scale
            max_error = max(max_error, constraint.solve(self.particles))
            constraint.rest_length = old
        return max_error
