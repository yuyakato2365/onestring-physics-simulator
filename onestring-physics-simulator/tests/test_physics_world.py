from onestring_physics.design_optimizer import DesignParameters, optimize_design
from onestring_physics.input_shape import create_builtin_shape
from onestring_physics.physics_world import PhysicsParameters, simulate_deployment


def test_physics_world_runs_and_returns_metrics():
    target = create_builtin_shape("dome", {"amplitude": 0.4, "radius": 2.0})
    design = optimize_design(target, nx=3, params=DesignParameters(max_iterations=6))
    result = simulate_deployment(
        design,
        PhysicsParameters(dt=0.01, substeps=1, solver_iterations=3, damping=0.05),
        num_frames=5,
    )

    assert len(result.frames) == 5
    assert result.final_tiles.shape == design.flat_tiles.shape
    assert "final_physical_deployment_error" in result.metrics
    assert result.final_tiles[..., 2].max() != design.flat_tiles[..., 2].max()
