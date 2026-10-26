"""Real OneString stage-by-stage integration on a built-in open test surface.

This is intentionally independent of the official OptCuts binary and therefore
NOT a Bunny / OptCuts certification. It catches mismatched grid/mesh APIs and
silent loss of M2D face topology downstream. Failures must fail CI.
"""
import numpy as np
import pytest

from onestring_physics.input_shape import create_builtin_shape
from onestring_physics import onestring_pipeline as pipeline
from onestring_physics.optcuts_planar_quad_20261026 import install_planar_quad_20261026
from onestring_physics.optcuts_planar_quad_downstream_20261026 import (
    install_unstructured_quad_downstream_20261026,
)


@pytest.mark.parametrize("target_faces", [16, 30, 35])
def test_planar_quad_runs_all_onestring_stages_on_open_dome(target_faces):
    # Distinct from optcuts_test, whose official binary is unavailable on CI.
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
        planar_quad_target_faces=target_faces,
        planar_quad_uv_reparam=(target_faces != 16),
        planar_quad_refine_m3d=(target_faces != 16),
        enable_csf_splits=False,
        collision_split_enabled=False,
    )
    target = create_builtin_shape("dome", {"amplitude": 0.25, "radius": 2.0})
    state = pipeline.build_onestring_design(target, params)
    # Feature mode must never depend on the old rectangular grid overlay.
    assert state.surface_parameterization.metrics["planar_quad_independent_domain"] is True
    assert state.surface_parameterization.metrics["planar_quad_legacy_regular_overlay_bypassed"] is True
    assert not hasattr(state.conformal_domain, "overlay_nx")
    assert len(state.conformal_domain.uv_vertices) == len(state.surface_parameterization.uv_vertices_2d)
    m2d = state.mesh_2d_initial
    m3d = state.mesh_3d_initial
    k3d = state.mesh_3d_optimized
    k2d = state.mesh_2d_optimized
    assert len(m2d.faces) > 0
    assert all(len(face) == 4 for face in m2d.faces)
    assert m2d.metrics["planar_quad_engine"].startswith("Gmsh ")
    # Keep the actual vs requested face count visible, never relabel the target as achieved.
    assert int(m2d.metrics["planar_quad_target_faces"]) == target_faces
    assert int(m2d.metrics["planar_quad_faces"]) == len(m2d.faces)
    assert m2d.metrics["planar_quad_m2d_angle_min_degrees"] >= 20.
    assert m2d.metrics["planar_quad_m3d_angle_min_degrees"] >= 12.
    for stage in ("m2d", "m3d"):
        for criterion in ("worst_angle", "largest_area", "smallest_area", "longest_edge"):
            face_idx = m2d.metrics[f"planar_quad_{stage}_{criterion}_face_index"]
            assert isinstance(face_idx, int)
            assert 0 <= face_idx < len(m2d.faces)
    assert np.array_equal(m2d.faces, m3d.faces)
    assert np.array_equal(m2d.faces, k3d.faces)
    assert np.array_equal(m2d.faces, k2d.faces)
    assert len(m2d.faces) == len(state.tiles_3d.vertices)
    assert len(m2d.faces) == len(state.tiles_2d_dual_hinge.vertices)
    assert int(m3d.metrics["m3d_outside_omega_count"]) == 0
    assert np.isfinite(m3d.vertices).all()
    assert np.isfinite(k3d.vertices).all()
    assert np.isfinite(k2d.vertices).all()
    assert np.isfinite(state.tiles_2d_dual_hinge.vertices).all()
    assert "flat_collision_count" in state.tiles_2d_dual_hinge.metrics
