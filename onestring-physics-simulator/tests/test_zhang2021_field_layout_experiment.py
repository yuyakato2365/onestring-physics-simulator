"""Actual generation tests, not a mock of the generator or claim of fidelity."""
from types import SimpleNamespace
import numpy as np
import pytest
from shapely.geometry import Polygon

from onestring_physics.zhang2021_field_layout_experiment import (
    generate_field_layout, FieldLayoutError)
from onestring_physics.optcuts_planar_quad_refine_20261026 import validate_disk_mesh


def source_from_polygon(coordinates):
    import shapely
    from shapely.ops import triangulate
    outline = np.asarray(coordinates, float)
    polygon = Polygon(outline)
    if hasattr(shapely, "constrained_delaunay_triangles"):
        triangles = list(shapely.constrained_delaunay_triangles(polygon).geoms)
    else:
        triangles = [t for t in triangulate(polygon) if polygon.covers(t)]
    pts = []
    mapping = {}
    faces = []
    for tri in triangles:
        f = []
        for v in np.asarray(tri.exterior.coords)[:-1, :2]:
            key = tuple(v)
            if key not in mapping:
                mapping[key] = len(pts)
                pts.append(v)
            f.append(mapping[key])
        p = np.asarray([pts[k] for k in f])
        if np.cross(p[1] - p[0], p[2] - p[0]) < 0:
            f.reverse()
        faces.append(f)
    uv = np.asarray(pts, float)
    f = np.asarray(faces, int)
    return SimpleNamespace(
        uv_vertices_2d=uv, uv_faces=f,
        surface_vertices_3d=np.column_stack((uv, np.zeros(len(uv)))),
        surface_faces=f,
        omega_boundary=np.vstack((outline, outline[0])),
        metrics={}, method="optcuts_test",
    )


@pytest.mark.parametrize("polygon", [
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]],
    [[0, 0], [3, 0], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [0, 2]],
])
def test_actual_conforming_quad_generation(polygon):
    source = source_from_polygon(polygon)
    out = generate_field_layout(
        source, target_quads=35, require_seam_correspondence=True)
    validate_disk_mesh(out.uv, out.quads)
    assert len(out.quads) >= 4
    assert out.quads.shape[1] == 4
    assert out.metrics["omega_undercoverage_fraction"] < 1e-8
    assert out.metrics["omega_overcoverage_fraction"] < 1e-8
    assert out.metrics["seam_status"] == "source_open_boundary_only"
    assert not out.metrics["zhang2021_algorithm_verified"]
    assert "Gmsh" not in out.metrics["planar_quad_engine"]


def test_boundary_outside_not_tolerated():
    # 2D source chart polygon cannot be different from its paired triangles.
    source = source_from_polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    source.omega_boundary[1, 0] = 1.1
    with pytest.raises(FieldLayoutError, match="SOURCE_BOUNDARY_TRIANGLE_MISMATCH"):
        generate_field_layout(source, target_quads=35)


def test_boundary_refinement_can_be_switched_off():
    source = source_from_polygon(
        [[0, 0], [3, 0], [3, 2], [2, 2], [2, 1], [1, 1], [1, 2], [0, 2]])
    out = generate_field_layout(
        source, 35, allow_outline_simplification=False)
    assert out.metrics["planar_quad_boundary_simplification_tolerance"] == 0


def test_no_fictitious_exact_paper_claim():
    source = source_from_polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    out = generate_field_layout(source)
    assert out.metrics["planar_quad_method"] == "field_guided_patch_templates_experiment"
    assert out.metrics["zhang2021_algorithm_verified"] is False
