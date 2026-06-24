from __future__ import annotations

from dataclasses import dataclass

from .design_optimizer import DesignParameters, DesignResult, optimize_design
from .input_shape import create_builtin_shape
from .physics_world import PhysicsParameters, PhysicsResult, simulate_deployment


@dataclass
class DemoResult:
    design: DesignResult
    physics: PhysicsResult

    @property
    def metrics(self) -> dict:
        return self.physics.metrics


def run_default_demo(num_frames: int = 200) -> DemoResult:
    target = create_builtin_shape("dome", {"amplitude": 0.75, "radius": 2.2})
    design = optimize_design(
        target,
        nx=3,
        ny=3,
        tile_size=1.0,
        gap_size=0.08,
        params=DesignParameters(max_iterations=30),
    )
    physics = simulate_deployment(
        design,
        PhysicsParameters(
            dt=0.01,
            substeps=4,
            solver_iterations=20,
            rope_stiffness=0.95,
            hinge_stiffness=0.92,
            damping=0.04,
        ),
        num_frames=num_frames,
    )
    return DemoResult(design, physics)
