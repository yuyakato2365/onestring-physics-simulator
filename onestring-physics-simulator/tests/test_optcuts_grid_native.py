from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_grid_constrained_m2d_patch import (
    _disconnect_faces_along_edges,
    _grid_cut_edges_from_segments,
    _internal_seam_segments,
)
from onestring_physics.optcuts_grid_native_backend import (
    NATIVE_GRID_MARKER,
    NATIVE_GRID_VERSION,
    _to_fabrication_frame,
)
from onestring_physics.optcuts_grid_native_pipeline_patch import (
    install_native_grid_optcuts_pipeline_patch,
)
from onestring_physics.optcuts_uv_overlap_guard import positive_area_uv_overlaps


def test_native_grid_version_is_paired_seam_v4():
    assert NATIVE_GRID_VERSION == 4
    assert "version=4" in NATIVE_GRID_MARKER


def test_fabrication_frame_rotation_preserves_fixed_lattice():
    angle = np.deg2rad(30.0)
    u = np.asarray([np.cos(angle), np.sin(angle)])
    v = np.asarray([-np.sin(angle), np.cos(angle)])
    h = 0.2
    phase_u, phase_v = 0.07, -0.03
    ij = np.asarray([[0, 0], [1, 0], [1, 2], [-2, 3]], dtype=float)
    world = np.asarray(
        [(phase_u + i * h) * u + (phase_v + j * h) * v for i, j in ij],
        dtype=float,
    )
    frame = _to_fabrication_frame(world, angle)
    expected = np.column_stack([phase_u + ij[:, 0] * h, phase_v + ij[:, 1] * h])
    assert np.allclose(frame, expected, atol=1e-12)


def test_internal_optcuts_seam_detected_from_surface_uv_index_mismatch():
    parameterization = SimpleNamespace(
        surface_faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int),
        uv_faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
        uv_vertices_2d=np.asarray(
            [
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    segments = _internal_seam_segments(parameterization)
    assert segments.shape == (1, 2, 2)
    assert {tuple(p) for p in segments[0]} == {(0.0, 0.0), (1.0, 1.0)}


def test_m2d_can_disconnect_a_native_seam_that_coincides_with_shared_grid_edge():
    # V4 normally produces two distinct UV boundary sides. This helper remains
    # necessary for the special case where an exported native seam lies on a
    # shared M2D grid edge: geometry stays coincident while topology disconnects.
    h = 1.0
    segment = np.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=float)
    cut_edges = _grid_cut_edges_from_segments(
        segment, origin=np.asarray([0.0, 0.0]), h=h, nx=2, ny=1
    )
    assert cut_edges == {(1, 4)}

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=int)
    cut_vertices, cut_faces, info = _disconnect_faces_along_edges(vertices, faces, cut_edges)

    assert info["active_cut_edges"] == 1
    assert info["face_components"] == 2
    assert info["duplicated_vertices"] == 2
    assert len(cut_vertices) == len(vertices) + 2
    assert set(cut_faces[0]).isdisjoint(set(cut_faces[1]))
    for p in cut_vertices[len(vertices) :]:
        assert np.min(np.linalg.norm(vertices - p[None, :], axis=1)) <= 1e-12


def test_global_overlap_guard_allows_shared_edge_but_detects_positive_area_overlap():
    adjacent_uv = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
    )
    adjacent_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=int)
    count, examples = positive_area_uv_overlaps(adjacent_uv, adjacent_faces)
    assert count == 0
    assert examples == []

    overlap_uv = np.asarray(
        [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.2, 0.2], [1.2, 0.2], [0.2, 1.2],
        ],
        dtype=float,
    )
    overlap_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int)
    count, examples = positive_area_uv_overlaps(overlap_uv, overlap_faces)
    assert count == 1
    assert examples and examples[0][2] > 0.0


def test_native_pipeline_does_not_intercept_official_optcuts():
    sentinel = object()

    def base_builder(surface, target, grid, params):
        return sentinel

    pipeline = SimpleNamespace(_build_surface_parameterization=base_builder)
    install_native_grid_optcuts_pipeline_patch(pipeline)
    params = SimpleNamespace(omega_parameterization_mode="optcuts")
    assert pipeline._build_surface_parameterization(None, None, None, params) is sentinel


def test_v4_patcher_encodes_grid_search_not_posthoc_seam_snap():
    root = Path(__file__).resolve().parents[1]
    algorithm = (root / "scripts" / "patch_optcuts_native_grid.py").read_text(encoding="utf-8")
    installer = (root / "scripts" / "patch_optcuts_native_grid_v4.py").read_text(encoding="utf-8")

    assert "ONESTRING_GRID_NATIVE_V4" in algorithm
    assert "oneStringGridComputeLocalLDec" in algorithm
    assert "oneStringTryInteriorGridCut" in algorithm
    assert "oneStringTryBoundaryGridCut" in algorithm
    assert "oneStringTotalSD" in algorithm
    assert "A-B-C" in algorithm and "A-D-C" in algorithm
    assert "oneStringGridLockedVert" in algorithm
    assert "oneStringMainInitialGridBoundary" in algorithm
    assert "version=4" in installer
    assert "_replace_in_function" in installer
    assert "No completed free seam" not in installer  # implementation, not a post-hoc adapter


def test_setup_uses_function_scoped_v4_installer():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "scripts" / "setup_optcuts.py").read_text(encoding="utf-8")
    assert "from patch_optcuts_native_grid_v4 import apply_native_grid_patch" in setup
