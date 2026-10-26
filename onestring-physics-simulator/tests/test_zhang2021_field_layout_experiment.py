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


def test_paper_inspired_generated_quads_reach_all_onestring_stages_on_dome():
    """Run genuine M2D generation through K3D,T3D,K2D,T2D, no Gmsh."""
    from onestring_physics.input_shape import create_builtin_shape
    from onestring_physics import onestring_pipeline as pipeline
    from onestring_physics.optcuts_planar_quad_20261026 import install_planar_quad_20261026
    from onestring_physics.optcuts_planar_quad_downstream_20261026 import (
        install_unstructured_quad_downstream_20261026)
    install_unstructured_quad_downstream_20261026()
    install_planar_quad_20261026(pipeline)
    params = pipeline.PipelineParameters(
        nx=3,
        max_3d_iterations=4,
        max_2d_iterations=4,
        omega_boundary_mode="shape_preserving_experimental",
        omega_parameterization_mode="pca_debug",
        allow_experimental_pipeline=True,
        optcuts_planar_quad_20261026_enabled=True,
        optcuts_zhang2021_experimental_enabled=True,
        planar_quad_target_faces=35,
        planar_quad_uv_reparam=False,
        planar_quad_refine_m3d=False,
        enable_csf_splits=False,
        collision_split_enabled=False,
    )
    target = create_builtin_shape("dome", {"amplitude": .25, "radius": 2.})
    state = pipeline.build_onestring_design(target, params)
    m2d = state.mesh_2d_initial
    assert m2d.metrics["zhang2021_algorithm_verified"] is False
    assert "coarse_layout" in m2d.metrics["planar_quad_engine"]
    assert np.array_equal(m2d.faces, state.mesh_3d_initial.faces)
    assert np.array_equal(m2d.faces, state.mesh_3d_optimized.faces)
    assert np.array_equal(m2d.faces, state.mesh_2d_optimized.faces)
    assert len(state.tiles_2d_dual_hinge.vertices) == len(m2d.faces)


def test_cli_generates_actual_mesh_and_serialized_report(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from onestring_physics.zhang2021_reference_bridge import export_source_chart
    source = source_from_polygon([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    path = tmp_path / "source.npz"
    output = tmp_path / "generated.npz"
    report = tmp_path / "report.json"
    export_source_chart(source, path)
    root = Path(__file__).resolve().parents[1]
    task = subprocess.run(
        [sys.executable, "scripts/run_field_layout_experiment.py",
         "--source", str(path), "--output", str(output),
         "--report", str(report), "--target-quads", "35"],
        cwd=root, capture_output=True, text=True, check=False)
    assert task.returncode == 0, task.stdout + task.stderr
    assert output.is_file() and report.is_file()
    with np.load(output) as mesh:
        assert len(mesh["faces"]) >= 4
        assert mesh["faces"].shape[1] == 4
    data = json.loads(report.read_text())
    assert data["quad_count"] == len(np.load(output)["faces"])
    assert data["zhang2021_algorithm_verified"] is False
