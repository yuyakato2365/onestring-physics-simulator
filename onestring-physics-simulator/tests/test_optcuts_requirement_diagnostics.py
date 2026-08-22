import numpy as np

from onestring_physics.optcuts_requirement_diagnostics_patch import _grid_alignment_error


def test_grid_alignment_error_is_zero_for_horizontal_grid_chain():
    nodes = {
        0: np.array([0.1, 0.25]),
        1: np.array([0.6, 0.25]),
        2: np.array([1.1, 0.25]),
    }
    error, details = _grid_alignment_error(
        nodes,
        [(0, 1), (1, 2)],
        h=0.25,
        phase_u=0.0,
        phase_v=0.0,
        angle_degrees=0.0,
    )
    assert error < 1e-12
    assert details[0]["axis"] == "grid_u"


def test_grid_alignment_error_detects_off_grid_chain():
    nodes = {
        0: np.array([0.1, 0.27]),
        1: np.array([0.6, 0.27]),
        2: np.array([1.1, 0.27]),
    }
    error, _ = _grid_alignment_error(
        nodes,
        [(0, 1), (1, 2)],
        h=0.25,
        phase_u=0.0,
        phase_v=0.0,
        angle_degrees=0.0,
    )
    assert abs(error - 0.02) < 1e-12
