import numpy as np
import pytest
from types import SimpleNamespace
from onestring_physics.optcuts_planar_quad_20261026 import (
    PlanarQuadMeshingError, UnstructuredQuadGrid, _triangle_inverse,
    _quad_quality, build_planar_quad,
)

def test_original_cut_uv_inverse_exact_piecewise_linear():
    # Surface has a genuine fold in 3D but a simple, bijective UV disk.
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mapped = _triangle_inverse(uv, faces, xyz, faces, np.array([[.75, .25], [.25, .75]]))
    assert mapped.shape == (2, 3)
    assert np.allclose(mapped[0], [.75, .25, .25])
    assert np.allclose(mapped[1], [.25, .75, .25])

def test_outside_original_omega_rejected():
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    with pytest.raises(PlanarQuadMeshingError, match="OUTSIDE_ORIGINAL_OMEGA"):
        _triangle_inverse(uv, faces, np.column_stack((uv, np.zeros(4))), faces,
                          np.array([[1.2, .3]]))

def test_valid_clockwise_uv_source_is_not_misclassified_as_folded():
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    faces = np.array([[2, 1, 0], [3, 2, 0]])
    result = _triangle_inverse(uv, faces, np.column_stack((uv, np.zeros(4))),
                               faces, np.array([[.7, .2]]))
    assert np.allclose(result[0, :2], [.7, .2])

def test_unstructured_quad_grid_corners_and_hinge_connectivity():
    uv = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    faces = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], int)
    grid = UnstructuredQuadGrid(uv, faces, 1., 0.)
    assert grid.num_tiles == 2
    assert len(grid.hinges) == 1
    assert grid.flat_tile_corners().shape == (2, 4, 3)
    assert _quad_quality(grid.vertex_positions, faces)["angle_min_degrees"] == pytest.approx(90)

def test_degenerate_quad_detected_by_quality():
    uv = np.array([[0,0,0],[1,0,0],[1,1e-8,0],[0,1e-8,0]], float)
    quality = _quad_quality(uv, np.array([[0,1,2,3]]))
    assert quality["area_min"] < 1e-6

def test_gmsh_missing_dependency_fails_closed(monkeypatch):
    # Does not require gmsh installed on CI.
    import builtins
    orig = builtins.__import__
    def fail_gmsh(name, *a, **k):
        if name == "gmsh":
            raise ImportError("synthetic gmsh missing")
        return orig(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fail_gmsh)
    with pytest.raises(PlanarQuadMeshingError, match="DEPENDENCY_MISSING"):
        build_planar_quad(SimpleNamespace(), 35)
