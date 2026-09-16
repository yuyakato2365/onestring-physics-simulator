import numpy as np

from onestring_physics.quad_grid import create_quad_grid


def test_quad_grid_counts_and_corners():
    grid = create_quad_grid(3, tile_size=1.0, gap_size=0.08)

    assert grid.num_tiles == 9
    assert len(grid.hinges) == 12
    corners = grid.flat_tile_corners()
    assert corners.shape == (9, 4, 3)
    assert np.allclose(corners[..., 2], 0.0)


def test_boundary_string_path_has_perimeter_midpoints():
    grid = create_quad_grid(4, tile_size=1.0)
    path = grid.boundary_midpoints()

    assert path.shape == (16, 3)
