"""Strict rectangle boundary BFF: geometry and no hidden harmonic fallback."""
from __future__ import annotations
from math import pi
import numpy as np
import pytest

from onestring_physics.reference_bff import (
    ReferenceBFFUnavailableError,
    find_rectangular_bff_executable,
    validate_rectangular_boundary,
    run_official_bff,
)


def _rectangle(radians=0.):
    points=np.array([
        [-2.,-1.], [0.,-1.], [2.,-1.],
        [2.,0.], [2.,1.], [0.,1.],
        [-2.,1.], [-2.,0.],
    ])
    rot=np.array([[np.cos(radians),-np.sin(radians)],
                  [np.sin(radians),np.cos(radians)]])
    return points@rot.T


def test_rectangle_validates_regardless_of_rotation():
    for angle in [0.,.3,pi/2]:
        pts=_rectangle(angle)
        report=validate_rectangular_boundary(pts,list(range(8)))
        assert report["bff_rectangle_verified"]
        assert report["bff_rectangle_side_max_deviation"]<1e-10


def test_free_boundary_polygon_is_rejected():
    points=_rectangle()
    points[1,1]+=.35
    with pytest.raises(ReferenceBFFUnavailableError,match="geometric verification"):
        validate_rectangular_boundary(points,list(range(8)))


def test_disallow_ordinary_cli_as_fake_rectangular_adapter(tmp_path,monkeypatch):
    monkeypatch.delenv("ONESTRING_BFF_RECTANGLE_EXECUTABLE",raising=False)
    ordinary=tmp_path/"bff-command-line"
    ordinary.write_text("#!/bin/sh\nexit 0\n")
    ordinary.chmod(0o755)
    with pytest.raises(ReferenceBFFUnavailableError,match="separately built"):
        find_rectangular_bff_executable(ordinary)
    with pytest.raises(ReferenceBFFUnavailableError,match="separately built"):
        run_official_bff(
            np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.3,.3,0.]]),
            np.array([[0,1,3],[1,2,3],[2,0,3]],int),
            boundary_policy="target_rectangle",
            rectangle_executable=ordinary,
        )


def test_wrong_adapter_name_rejected_even_with_execute_permissions(tmp_path):
    wrong=tmp_path/"not-rect"
    wrong.write_text("#!/bin/sh\nexit 0\n")
    wrong.chmod(0o755)
    with pytest.raises(ReferenceBFFUnavailableError,match="separately built"):
        find_rectangular_bff_executable(wrong)
