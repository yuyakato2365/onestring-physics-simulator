from __future__ import annotations

import numpy as np

from onestring_physics.optcuts_grid_fusion_v6 import (
    _DSU,
    _edge_axes_smoothed,
    _integer_axis_solve,
)


def test_integer_axis_solver_separates_zero_length_candidate() -> None:
    ids = [0, 1]
    dsu = _DSU(ids)
    # Both endpoints are nearest to the same lattice coordinate, but the segment
    # requires one non-zero grid step.  v5 raised ZERO_LENGTH_SEGMENT here.
    coords = np.asarray([0.04, 0.06], dtype=float)
    roots, _groups, info = _integer_axis_solve(
        ids,
        dsu,
        coords,
        phase=0.0,
        h=0.1,
        varying_edges=[(0, 1, 1.0, 2.5)],
        soft_pairs=[],
    )
    assert info["ok"] is True
    assert roots[dsu.find(0)] != roots[dsu.find(1)]


def test_integer_axis_solver_handles_shared_junction_globally() -> None:
    ids = [0, 1, 2]
    dsu = _DSU(ids)
    coords = np.asarray([0.0, 0.02, 0.04], dtype=float)
    roots, _groups, info = _integer_axis_solve(
        ids,
        dsu,
        coords,
        phase=0.0,
        h=0.1,
        varying_edges=[
            (0, 1, 1.0, 2.5),
            (1, 2, 1.0, 2.5),
        ],
        soft_pairs=[],
    )
    assert info["ok"] is True
    x0 = roots[dsu.find(0)]
    x1 = roots[dsu.find(1)]
    x2 = roots[dsu.find(2)]
    assert x0 != x1
    assert x1 != x2


def test_integer_axis_solver_respects_constant_coordinate_union() -> None:
    ids = [0, 1, 2]
    dsu = _DSU(ids)
    dsu.union(0, 1)
    coords = np.asarray([0.01, 0.02, 0.31], dtype=float)
    roots, groups, info = _integer_axis_solve(
        ids,
        dsu,
        coords,
        phase=0.0,
        h=0.1,
        varying_edges=[(1, 2, 3.0, 2.5)],
        soft_pairs=[],
    )
    assert info["ok"] is True
    assert groups[0] == groups[1]
    assert roots[dsu.find(1)] != roots[dsu.find(2)]


def test_closed_loop_axis_assignment_contains_both_axes() -> None:
    # A square loop must not be compiled into one collinear run.
    uv = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    ids = [0, 1, 2, 3, 4]
    axes = _edge_axes_smoothed(uv, ids, ids, closed=True)
    assert 0 in axes
    assert 1 in axes
