from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


@dataclass
class RigidTile:
    id: int
    mass: float
    position: np.ndarray
    orientation: np.ndarray
    local_corners: np.ndarray
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    collision_radius: float = 0.75

    @classmethod
    def from_corners(cls, id: int, corners: np.ndarray, mass: float = 1.0) -> "RigidTile":
        corners = np.asarray(corners, dtype=float)
        center = np.mean(corners, axis=0)
        return cls(
            id=id,
            mass=mass,
            position=center,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            local_corners=corners - center,
            collision_radius=float(np.max(np.linalg.norm(corners - center, axis=1))),
        )

    def world_corners(self) -> np.ndarray:
        # v0.1 keeps orientation updates lightweight; local corners are carried
        # by the particle constraints in PhysicsWorld.
        return self.position + self.local_corners

    def accumulate_force(self, force: np.ndarray, point: np.ndarray | None = None) -> None:
        self.force += np.asarray(force, dtype=float)
        if point is not None:
            self.torque += np.cross(np.asarray(point, dtype=float) - self.position, force)

    def integrate(self, dt: float, damping: float = 0.02) -> None:
        acc = self.force / max(self.mass, 1e-8)
        self.linear_velocity = (self.linear_velocity + acc * dt) * max(0.0, 1.0 - damping)
        self.position = self.position + self.linear_velocity * dt

        angular_acc = self.torque / max(self.mass * self.collision_radius**2, 1e-8)
        self.angular_velocity = (self.angular_velocity + angular_acc * dt) * max(0.0, 1.0 - damping)
        speed = float(np.linalg.norm(self.angular_velocity))
        if speed > 1e-12:
            axis = self.angular_velocity / speed
            half = 0.5 * speed * dt
            dq = np.array([np.cos(half), *(np.sin(half) * axis)])
            self.orientation = quat_normalize(quat_multiply(dq, self.orientation))

        self.force[:] = 0.0
        self.torque[:] = 0.0
