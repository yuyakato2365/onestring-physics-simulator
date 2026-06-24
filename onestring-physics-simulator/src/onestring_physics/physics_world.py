from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .design_optimizer import DesignResult
from .hinge import PointHingeConstraint
from .metrics import kinetic_energy, potential_energy, rms_distance, tile_deformation
from .rope import Rope, RopeTileAttachment
from .string_path import nearest_tile_corners
from .xpbd_solver import integrate_verlet, solve_distance_positions, update_velocities


@dataclass
class PhysicsParameters:
    dt: float = 0.01
    substeps: int = 4
    solver_iterations: int = 20
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    damping: float = 0.04
    rope_stiffness: float = 0.95
    hinge_stiffness: float = 0.92
    tile_stiffness: float = 0.98
    rope_pull_speed: float = 1.0
    rope_rest_length_scale: float = 0.55
    debug_goal_attraction: bool = False
    debug_goal_strength: float = 0.02


@dataclass
class PhysicsResult:
    frames: list[np.ndarray]
    rope_frames: list[np.ndarray]
    final_tiles: np.ndarray
    metrics: dict[str, float | bool]
    pull_handle_frames: list[np.ndarray] = field(default_factory=list)


class PhysicsWorld:
    def __init__(self, design: DesignResult, params: PhysicsParameters | None = None):
        self.design = design
        self.params = params or PhysicsParameters()
        self.tile_positions = np.asarray(design.flat_tiles, dtype=float).copy()
        self.previous_tile_positions = self.tile_positions.copy()
        self.tile_inv_masses = np.ones(self.tile_positions.shape[0], dtype=float)
        self.gravity = np.asarray(self.params.gravity, dtype=float)
        self.rest_tile_positions = self.tile_positions.copy()
        self.tile_constraints = self._build_tile_constraints()
        self.hinges = self._build_hinges()
        self.rope = Rope.from_polyline(design.boundary_string_path, self.params.rope_stiffness, closed=True)
        self.rope_attachments = self._build_rope_attachments()
        self.handle_start = np.array([0.0, 0.0, -0.15], dtype=float)
        max_z = float(np.max(design.assembled_tiles[..., 2]))
        self.handle_end = np.array([0.0, 0.0, max(0.75, max_z + 0.8)], dtype=float)
        self.tendon_guides, self.tendon_rest_start, self.tendon_rest_goal = self._build_lift_tendons()
        self.time = 0.0

    def step(self, pull_progress: float) -> dict[str, float]:
        sub_dt = self.params.dt / max(1, self.params.substeps)
        max_hinge = 0.0
        max_rope = 0.0
        for _ in range(max(1, self.params.substeps)):
            flat = self.tile_positions.reshape(-1, 3)
            prev = self.previous_tile_positions.reshape(-1, 3)
            inv = np.repeat(self.tile_inv_masses / 4.0, 4)
            flat, prev = integrate_verlet(flat, prev, inv, self.gravity, sub_dt, self.params.damping)
            self.tile_positions = flat.reshape(self.tile_positions.shape)
            self.previous_tile_positions = prev.reshape(self.previous_tile_positions.shape)
            self.rope.integrate(self.gravity * 0.15, sub_dt, self.params.damping)

            for _ in range(max(1, self.params.solver_iterations)):
                max_rope = max(max_rope, self.rope.solve())
                for attachment in self.rope_attachments:
                    max_rope = max(
                        max_rope,
                        attachment.solve(self.rope.particles, self.tile_positions, self.tile_inv_masses),
                    )
                max_hinge = max(max_hinge, self._solve_tile_rigidity())
                for hinge in self.hinges:
                    max_hinge = max(max_hinge, hinge.solve(self.tile_positions, self.tile_inv_masses))
                self._solve_pull_tendons(pull_progress)
                if self.params.debug_goal_attraction:
                    self.tile_positions += (
                        self.design.assembled_tiles - self.tile_positions
                    ) * self.params.debug_goal_strength

            self.time += sub_dt
        return {"hinge_violation": max_hinge, "rope_stretch_error": max_rope}

    def simulate(self, num_frames: int = 200) -> PhysicsResult:
        frames: list[np.ndarray] = []
        rope_frames: list[np.ndarray] = []
        handle_frames: list[np.ndarray] = []
        last_stats = {"hinge_violation": 0.0, "rope_stretch_error": 0.0}
        for i in range(num_frames):
            raw = min(1.0, self.params.rope_pull_speed * i / max(1, num_frames - 1))
            pull_progress = 1.0 - (1.0 - raw) ** 2
            last_stats = self.step(pull_progress)
            frames.append(self.tile_positions.copy())
            rope_frames.append(self.rope.positions().copy())
            handle_frames.append(self.pull_handle(pull_progress).copy())

        velocities = update_velocities(
            self.tile_positions.reshape(-1, 3),
            self.previous_tile_positions.reshape(-1, 3),
            self.params.dt,
        )
        final_error = rms_distance(self.tile_positions, self.design.assembled_tiles)
        target_error = rms_distance(self.design.assembled_tiles, self.design.target_vertices_for_tiles())
        deform = tile_deformation(self.tile_positions, self.rest_tile_positions)
        metrics = {
            "target_fitting_error": target_error,
            "final_physical_deployment_error": final_error,
            "hinge_constraint_violation": last_stats["hinge_violation"],
            "rope_stretch_error": last_stats["rope_stretch_error"],
            "max_tile_deformation": deform,
            "total_kinetic_energy": kinetic_energy(velocities),
            "potential_energy": potential_energy(self.tile_positions, self.gravity),
            "stable_state": bool(kinetic_energy(velocities) < 0.5 and last_stats["hinge_violation"] < 0.25),
        }
        return PhysicsResult(frames, rope_frames, self.tile_positions.copy(), metrics, handle_frames)

    def pull_handle(self, progress: float) -> np.ndarray:
        return (1.0 - progress) * self.handle_start + progress * self.handle_end

    def _build_tile_constraints(self) -> list[tuple[int, int, int, float]]:
        constraints: list[tuple[int, int, int, float]] = []
        for tile_index, tile in enumerate(self.rest_tile_positions):
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
                constraints.append((tile_index, a, b, float(np.linalg.norm(tile[a] - tile[b]))))
        return constraints

    def _build_hinges(self) -> list[PointHingeConstraint]:
        hinges: list[PointHingeConstraint] = []
        for spec in self.design.hinges:
            hinges.append(PointHingeConstraint(spec.tile_a, spec.corner_a0, spec.tile_b, spec.corner_b0, self.params.hinge_stiffness))
            hinges.append(PointHingeConstraint(spec.tile_a, spec.corner_a1, spec.tile_b, spec.corner_b1, self.params.hinge_stiffness))
        return hinges

    def _build_rope_attachments(self) -> list[RopeTileAttachment]:
        attachments: list[RopeTileAttachment] = []
        matches = nearest_tile_corners(self.design.boundary_string_path, self.tile_positions)
        for particle, (tile, corner) in enumerate(matches):
            attachments.append(RopeTileAttachment(particle, tile, corner, stiffness=0.55))
        return attachments

    def _build_lift_tendons(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat_centers = np.mean(self.rest_tile_positions, axis=1)
        assembled_centers = np.mean(self.design.assembled_tiles, axis=1)
        guide_z = max(float(np.max(assembled_centers[:, 2])) + 1.2, 1.5)
        guides = flat_centers.copy()
        guides[:, 2] = guide_z

        start = np.linalg.norm(flat_centers - guides, axis=1)
        target_lift = np.maximum(assembled_centers[:, 2] - flat_centers[:, 2], 0.0)
        lift_gain = 1.0 + (1.0 - self.params.rope_rest_length_scale) * 0.25
        goal_z = flat_centers[:, 2] + target_lift * lift_gain
        goal = np.maximum(guide_z - goal_z, start * 0.35)
        return guides, start, goal

    def _solve_tile_rigidity(self) -> float:
        max_error = 0.0
        for tile_index, a, b, rest in self.tile_constraints:
            p0 = self.tile_positions[tile_index, a]
            p1 = self.tile_positions[tile_index, b]
            inv = self.tile_inv_masses[tile_index] / 4.0
            n0, n1, error = solve_distance_positions(p0, p1, inv, inv, rest, self.params.tile_stiffness)
            self.tile_positions[tile_index, a] = n0
            self.tile_positions[tile_index, b] = n1
            max_error = max(max_error, abs(float(error)))
        return max_error

    def _solve_pull_tendons(self, pull_progress: float) -> float:
        max_error = 0.0
        for tile_index in range(self.tile_positions.shape[0]):
            center = np.mean(self.tile_positions[tile_index], axis=0)
            rest = (1.0 - pull_progress) * self.tendon_rest_start[tile_index]
            rest += pull_progress * self.tendon_rest_goal[tile_index]
            new_center, _, error = solve_distance_positions(
                center,
                self.tendon_guides[tile_index],
                self.tile_inv_masses[tile_index],
                0.0,
                rest,
                self.params.rope_stiffness,
            )
            self.tile_positions[tile_index] += new_center - center
            max_error = max(max_error, abs(float(error)))
        return max_error


def simulate_deployment(
    design: DesignResult,
    params: PhysicsParameters | None = None,
    num_frames: int = 200,
) -> PhysicsResult:
    return PhysicsWorld(design, params).simulate(num_frames=num_frames)
