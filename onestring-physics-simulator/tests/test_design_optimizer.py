from onestring_physics.design_optimizer import DesignParameters, optimize_design
from onestring_physics.input_shape import create_builtin_shape


def test_design_optimizer_returns_expected_artifacts():
    target = create_builtin_shape("dome", {"amplitude": 0.5, "radius": 2.0})
    result = optimize_design(target, nx=3, params=DesignParameters(max_iterations=8))

    assert result.assembled_tiles.shape == (9, 4, 3)
    assert result.flat_tiles.shape == (9, 4, 3)
    assert len(result.hinges) == 12
    assert result.boundary_string_path.shape[0] == 12
    assert result.assembled_tiles[..., 2].max() > 0.1
