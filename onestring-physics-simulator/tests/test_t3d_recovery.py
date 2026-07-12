import numpy as np
import pytest

from onestring_physics.onestring_pipeline import QuadMesh, _extrude_tiles, export_t3d_stl
from onestring_physics.t3d_recovery import (
    T3DConstructionError,
    T3D_FAILED_INVALID_TOP,
    T3D_FAILED_NONMANIFOLD_CONTACT,
    T3D_OK_NOMINAL_FRUSTUM,
    T3D_RECOVERED_CAPPED_FRUSTUM,
    T3D_RECOVERED_LOCAL_THICKNESS,
    T3D_RECOVERED_PYRAMID,
    T3D_RECOVERED_WEDGE,
    build_tile_polyhedron,
    clip_convex_polyhedron,
    polyhedron_validation,
)


def test_convex_cube_plane_clip_is_watertight_and_preserves_volume():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    clipped_vertices, clipped_faces, info = clip_convex_polyhedron(
        vertices, faces, np.asarray([1.0, 0.0, 0.0]), 0.6
    )
    quality = polyhedron_validation(clipped_vertices, clipped_faces, 1e-9, 1e-9)
    assert info["cap_added"] is True
    assert quality["valid"] is True
    assert quality["volume"] == pytest.approx(0.6)


def test_flat_shared_pair_uses_nominal_authoritative_solids_and_exports_stl():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=int)
    mesh = QuadMesh(vertices, faces, None, "K3D", {"t3d_variable_topology_enabled": True})
    assembly, _report = _extrude_tiles(mesh, 0.2, "T3D")
    assert assembly.metrics["t3d_recovery_status_counts"] == {T3D_OK_NOMINAL_FRUSTUM: 2}
    assert all(solid.metrics["watertight"] for solid in assembly.authoritative_solids)
    assert all(solid.metrics["manifold"] for solid in assembly.authoritative_solids)
    data, metrics = export_t3d_stl(assembly)
    assert data.startswith(b"solid onestring_t3d")
    assert metrics["t3d_export_tile_count"] == 2
    assert metrics["t3d_export_watertight_tile_count"] == 2


def test_far_miter_intersection_generates_cap_instead_of_normal_prism_fallback():
    top = np.asarray([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=float)
    side_planes = [
        (np.asarray([0.0, -1.0, 10.0]), 0.5),
        (np.asarray([1.0, 0.0, 10.0]), 0.5),
        (np.asarray([0.0, 1.0, 10.0]), 0.5),
        (np.asarray([-1.0, 0.0, 10.0]), 0.5),
    ]
    solid = build_tile_polyhedron(
        tile_id=0,
        top=top,
        normal=np.asarray([0.0, 0.0, 1.0]),
        side_planes=side_planes,
        requested_thickness=0.2,
        minimum_thickness=0.05,
        minimum_volume=1e-9,
        minimum_feature_size=1e-9,
        miter_jump_limit=0.75,
    )
    assert solid.recovery_status == T3D_RECOVERED_CAPPED_FRUSTUM
    assert solid.metrics["cap_face_count"] > 0
    assert solid.metrics["watertight"] is True
    assert "large_miter_jump" in solid.recovery_reasons


def test_invalid_k3d_top_is_a_fundamental_failure():
    invalid_top = np.asarray([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    with pytest.raises(T3DConstructionError) as caught:
        build_tile_polyhedron(
            tile_id=7,
            top=invalid_top,
            normal=np.asarray([0.0, 0.0, 1.0]),
            side_planes=[],
            requested_thickness=0.2,
            minimum_thickness=0.05,
            minimum_volume=1e-9,
            minimum_feature_size=1e-9,
            miter_jump_limit=0.75,
        )
    assert caught.value.status == T3D_FAILED_INVALID_TOP


def test_bottom_quad_can_collapse_to_a_valid_quadrilateral_pyramid():
    top = np.asarray([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=float)
    side_planes = [
        (np.asarray([0.0, -1.0, -2.5]), 0.5),
        (np.asarray([1.0, 0.0, -2.5]), 0.5),
        (np.asarray([0.0, 1.0, -2.5]), 0.5),
        (np.asarray([-1.0, 0.0, -2.5]), 0.5),
    ]
    solid = build_tile_polyhedron(
        tile_id=0, top=top, normal=np.asarray([0.0, 0.0, 1.0]), side_planes=side_planes,
        requested_thickness=0.2, minimum_thickness=0.05, minimum_volume=1e-10,
        minimum_feature_size=1e-10, miter_jump_limit=0.75,
    )
    assert solid.recovery_status == T3D_RECOVERED_PYRAMID
    assert len(solid.vertices) == 5
    assert solid.metrics["volume"] > 0.0
    assert solid.metrics["watertight"] is True


def test_bottom_can_collapse_to_a_watertight_ridge_wedge():
    top = np.asarray([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=float)
    side_planes = [
        (np.asarray([0.0, -1.0, 0.0]), 0.5),
        (np.asarray([1.0, 0.0, -2.5]), 0.5),
        (np.asarray([0.0, 1.0, 0.0]), 0.5),
        (np.asarray([-1.0, 0.0, -2.5]), 0.5),
    ]
    solid = build_tile_polyhedron(
        tile_id=0, top=top, normal=np.asarray([0.0, 0.0, 1.0]), side_planes=side_planes,
        requested_thickness=0.2, minimum_thickness=0.05, minimum_volume=1e-10,
        minimum_feature_size=1e-10, miter_jump_limit=0.75,
    )
    assert solid.recovery_status == T3D_RECOVERED_WEDGE
    assert solid.metrics["volume"] > 0.0
    assert solid.metrics["manifold"] is True


def test_infeasible_nominal_depth_reports_local_thickness_recovery():
    top = np.asarray([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=float)
    side_planes = [
        (np.asarray([0.0, -1.0, -5.0]), 0.5),
        (np.asarray([1.0, 0.0, -5.0]), 0.5),
        (np.asarray([0.0, 1.0, -5.0]), 0.5),
        (np.asarray([-1.0, 0.0, -5.0]), 0.5),
    ]
    solid = build_tile_polyhedron(
        tile_id=0, top=top, normal=np.asarray([0.0, 0.0, 1.0]), side_planes=side_planes,
        requested_thickness=0.2, minimum_thickness=0.05, minimum_volume=1e-10,
        minimum_feature_size=1e-10, miter_jump_limit=0.75,
    )
    assert solid.recovery_status == T3D_RECOVERED_LOCAL_THICKNESS
    assert solid.metrics["actual_max_depth"] == pytest.approx(0.1)
    assert solid.metrics["local_thickness_ratio"] == pytest.approx(0.5)


def test_nonmanifold_contact_fails_without_explicit_junction_design():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0], [1, -1, 0], [0, -1, 0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2, 3], [1, 0, 4, 5], [0, 1, 6, 7]], dtype=int)
    mesh = QuadMesh(vertices, faces, None, "K3D", {"t3d_variable_topology_enabled": True})
    with pytest.raises(T3DConstructionError) as caught:
        _extrude_tiles(mesh, 0.2, "T3D")
    assert caught.value.status == T3D_FAILED_NONMANIFOLD_CONTACT
