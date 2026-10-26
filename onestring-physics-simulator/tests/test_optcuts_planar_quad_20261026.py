import numpy as np
import pytest
from types import SimpleNamespace
from onestring_physics.optcuts_planar_quad_20261026 import (
    PlanarQuadMeshingError, UnstructuredQuadGrid, _triangle_inverse,
    _quad_quality, build_planar_quad, _choose_contained_omega_boundary,
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

def test_gmsh_planar_rectangle_all_quads_and_inverse():
    pytest.importorskip("gmsh")
    pytest.importorskip("shapely")
    uv = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    xyz = np.column_stack((uv, np.zeros(4)))
    faces = np.array([[0,1,2],[0,2,3]], dtype=int)
    src = SimpleNamespace(uv_vertices_2d=uv, uv_faces=faces,
                          surface_vertices_3d=xyz, surface_faces=faces,
                          omega_boundary=np.vstack((uv,uv[0])))
    quv, quads, metrics = build_planar_quad(src, 16)
    assert len(quads) > 0
    assert quads.shape[1] == 4
    assert metrics["planar_quad_uv_inverse_exact_piecewise_linear"]
    assert metrics["planar_quad_omega_undercoverage_fraction"] < .02
    assert metrics["planar_quad_m2d_angle_min_degrees"] >= 8

def test_gmsh_concave_L_domain_all_quads_no_outside():
    pytest.importorskip("gmsh")
    pytest.importorskip("shapely")
    uv = np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]],float)
    triangles = np.array([[0,1,3],[1,2,3],[0,3,5],[3,4,5]],int)
    xyz = np.column_stack((uv,np.zeros(len(uv))))
    src = SimpleNamespace(uv_vertices_2d=uv,uv_faces=triangles,
                          surface_vertices_3d=xyz,surface_faces=triangles,
                          omega_boundary=np.vstack((uv,uv[0])))
    qv, qf, report = build_planar_quad(src, 35)
    assert len(qf) > 0 and qf.shape[1] == 4
    assert report["planar_quad_omega_overcoverage_fraction"] <= 1e-5
    assert report["planar_quad_omega_undercoverage_fraction"] < .02
    assert report["planar_quad_m2d_area_min"] > 0.


def test_shallow_concave_notch_is_not_filled_by_boundary_simplification():
    """Topology preserving simplify may go OUTSIDE the original UV polygon.

    Shapely removes a shallow inward notch, then the quad footprint exceeds
    Omega and the old pipeline raises PLANAR_QUAD_OUTSIDE_ORIGINAL_OMEGA.
    The new preflight must reject that particular boundary simplification.
    """
    from shapely.geometry import Polygon
    uv_boundary = np.array([
        [0., 0.], [2., 0.], [2., 2.],
        [1.1, 2.], [1.0, 1.997], [.9, 2.], [0., 2.],
    ])
    source = Polygon(uv_boundary)
    h = float(np.sqrt(source.area / 35))
    old = source.simplify(min(h * .03, np.sqrt(source.area) * .005),
                          preserve_topology=True)
    assert old.is_valid
    assert old.difference(source).area > 1e-5 * source.area
    chosen, report = _choose_contained_omega_boundary(source, h)
    assert Polygon(chosen).difference(source).area <= 1e-12 * source.area
    assert report["planar_quad_boundary_outside_rejected_candidates"] >= 1
    assert report["planar_quad_meshed_boundary_vertices"] <= len(uv_boundary)


def test_convex_corner_simplification_is_allowed_when_it_stays_inside():
    """Don't disable helpful simplification everywhere as a workaround."""
    from shapely.geometry import Polygon
    source = Polygon([(0., 0.), (2., 0.), (2., 2.), (1.1, 2.),
                      (1., 2.003), (.9, 2.), (0., 2.)])
    h = float(np.sqrt(source.area / 35))
    chosen, report = _choose_contained_omega_boundary(source, h)
    assert Polygon(chosen).difference(source).area <= 1e-12 * source.area
    assert report["planar_quad_meshed_boundary_vertices"] <= len(source.exterior.coords)-1
