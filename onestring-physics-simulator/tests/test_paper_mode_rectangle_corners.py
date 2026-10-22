"""Paper-oriented mode puts the rectangular BFF corners on genuine boundary corners."""
import os
from pathlib import Path

import numpy as np
import pytest

from onestring_physics.paper_split_reflatten import lobe_tip_corner_positions


def _square_sheet(n=24, height=None):
    xs = np.linspace(-1.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    Z = np.zeros_like(X) if height is None else height(X, Y)
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    F = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i*(n+1)+j, (i+1)*(n+1)+j, (i+1)*(n+1)+j+1, i*(n+1)+j+1
            F += [[a, b, c], [a, c, d]]
    return V, np.asarray(F)


@pytest.mark.parametrize("height", [None, lambda x, y: 0.5 * (x * x - y * y)], ids=["flat", "saddle"])
def test_paper_mode_corners_are_the_square_corners(height):
    V, F = _square_sheet(height=height)
    corners = lobe_tip_corner_positions(V, F, snap_to_corners=True)
    assert sorted(map(tuple, np.round(np.abs(corners[:, :2]), 9))) == [(1.0, 1.0)] * 4


def test_default_selection_is_unchanged_for_other_routes():
    V, F = _square_sheet()
    default = lobe_tip_corner_positions(V, F)
    # previous behaviour: first vertex of each flat maximum (not on the corners)
    assert not np.allclose(np.abs(default[:, :2]), 1.0)


def test_lilium_corners_identical_in_paper_mode():
    obj = os.environ.get("ONESTRING_LILIUM_OBJ")
    if not (obj and Path(obj).is_file()):
        pytest.skip("needs ONESTRING_LILIUM_OBJ")
    V, F = [], []
    for line in open(obj):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    V, F = np.asarray(V), np.asarray(F)
    assert np.array_equal(lobe_tip_corner_positions(V, F), lobe_tip_corner_positions(V, F, snap_to_corners=True))


def test_elongated_smooth_outline_uses_shoulder_corners():
    # smooth 2:1 elliptic dome: no genuine corners, corners must sit at the shoulders
    t = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    rings = [np.column_stack([2.0 * r * np.cos(t), r * np.sin(t), np.full(160, 0.3 * (1 - r * r))]) for r in np.linspace(0.05, 1.0, 20)]
    V = np.vstack([[0.0, 0.0, 0.3]] + rings)
    F = [[0, 1 + j, 1 + (j + 1) % 160] for j in range(160)]
    for k in range(19):
        a0, a1 = 1 + k * 160, 1 + (k + 1) * 160
        for j in range(160):
            a, b, c, d = a0 + j, a0 + (j + 1) % 160, a1 + (j + 1) % 160, a1 + j
            F += [[a, d, c], [a, c, b]]
    C = lobe_tip_corner_positions(V, np.asarray(F), snap_to_corners=True)
    assert np.all(np.abs(C[:, 0]) > 0.8) and np.all(np.abs(C[:, 1]) > 0.3)   # two at each end, off the axis
    assert sorted(np.sign(C[:, 0]).tolist()) == [-1, -1, 1, 1]
