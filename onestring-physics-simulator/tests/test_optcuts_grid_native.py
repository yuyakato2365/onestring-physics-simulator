from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.optcuts_backend import OptCutsConfig
from onestring_physics.optcuts_grid_constrained_m2d_patch import (
    _disconnect_faces_along_edges,
    _grid_cut_edges_from_segments,
    _internal_seam_segments,
    install_optcuts_grid_constrained_m2d_patch,
)
from onestring_physics.optcuts_grid_native_backend import (
    NATIVE_GRID_MARKER,
    NATIVE_GRID_VERSION,
    _to_fabrication_frame,
    run_native_grid_optcuts,
)
from onestring_physics.optcuts_grid_native_lift_patch import _map_used_vertices
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


def test_v4_algorithm_is_grid_search_not_posthoc_seam_snap():
    root = Path(__file__).resolve().parents[1]
    algorithm = (root / "scripts" / "patch_optcuts_native_grid.py").read_text(encoding="utf-8")
    installer = (root / "scripts" / "patch_optcuts_native_grid_v4_final.py").read_text(encoding="utf-8")

    assert "ONESTRING_GRID_NATIVE_V4" in algorithm
    assert "oneStringGridComputeLocalLDec" in algorithm
    assert "oneStringTryInteriorGridCut" in algorithm
    assert "oneStringTryBoundaryGridCut" in algorithm
    assert "oneStringTotalSD" in algorithm
    assert "A-B-C" in algorithm and "A-D-C" in algorithm
    assert "oneStringGridLockedVert" in algorithm
    assert "oneStringMainInitialGridBoundary" in algorithm
    assert "igl::map_vertices_to_circle" in installer
    assert "version=4" in installer


def test_setup_uses_verified_v4_installer_and_cpp14():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "scripts" / "setup_optcuts.py").read_text(encoding="utf-8")
    assert "from patch_optcuts_native_grid_v4_final import apply_native_grid_patch" in setup
    assert "-DCMAKE_CXX_STANDARD=14" in setup


def test_native_grid_binary_backend_seams_m2d_and_lift_end_to_end():
    """Real binary integration test, enabled in CI after OptCuts has been built."""
    executable = os.environ.get("ONESTRING_OPTCUTS_EXECUTABLE", "").strip()
    if not executable or not Path(executable).is_file():
        pytest.skip("native Grid-OptCuts executable is not configured")

    h = 0.5
    xyz = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=int
    )
    result = run_native_grid_optcuts(
        xyz,
        faces,
        OptCutsConfig(
            executable=executable,
            distortion_bound=4.1,
            lambda_init=0.999,
            method_type=0,
            use_bijectivity=False,
            initial_cut_option=0,
            timeout_seconds=120.0,
        ),
        grid_h=h,
        angle_degrees=0.0,
        phase_u=0.0,
        phase_v=0.0,
        max_snap_steps=4.0,
    )
    assert result.metrics["optcuts_grid_native_version"] == 4
    assert result.metrics["optcuts_grid_postprocess_used"] is False
    assert result.metrics["uv_triangle_flip_count"] == 0
    assert result.metrics["uv_degenerate_triangle_count"] == 0
    assert result.metrics["optcuts_grid_global_overlap_count"] == 0

    parameterization = SimpleNamespace(
        surface_vertices_3d=np.asarray(result.surface_vertices_3d, dtype=float),
        surface_faces=np.asarray(result.surface_faces, dtype=int),
        uv_vertices_2d=np.asarray(result.uv_vertices_2d, dtype=float),
        uv_faces=np.asarray(result.uv_faces, dtype=int),
        boundary_loops=result.boundary_loops,
        omega_boundary=np.asarray(
            result.uv_vertices_2d[result.boundary_loops[0] + [result.boundary_loops[0][0]]],
            dtype=float,
        ),
        metrics={
            **result.metrics,
            "optcuts_grid_constrained": True,
            "grid_phase_u": 0.0,
            "grid_phase_v": 0.0,
        },
    )

    # Audit the *actual cut produced by OptCuts*, not a synthetic seam.
    seam_segments = _internal_seam_segments(parameterization)
    assert len(seam_segments) > 0
    lattice_tol = 2.0e-6
    for segment in seam_segments:
        a, b = np.asarray(segment, dtype=float)
        assert np.max(np.abs(a / h - np.rint(a / h))) <= lattice_tol
        assert np.max(np.abs(b / h - np.rint(b / h))) <= lattice_tol
        delta = np.abs(b - a)
        assert float(np.linalg.norm(delta)) > 1.0e-10
        assert delta[0] <= 1.0e-8 or delta[1] <= 1.0e-8

    # Exercise the same fixed-lattice M2D builder that app_optcuts installs.
    class FakeQuadMesh:
        def __init__(self, vertices, quad_faces, grid, stage, metrics, split_lines):
            self.vertices = np.asarray(vertices, dtype=float)
            self.faces = np.asarray(quad_faces, dtype=int)
            self.grid = grid
            self.stage = stage
            self.metrics = dict(metrics)
            self.split_lines = list(split_lines)

    def base_flatten(param, grid, params=None):
        return SimpleNamespace(
            uv_vertices=np.zeros((0, 2), dtype=float),
            boundary=np.asarray(param.omega_boundary, dtype=float),
            split_lines=[],
        )

    def base_build(grid, domain, params=None):
        raise AssertionError("native Grid M2D builder was not selected")

    fake_pipeline = SimpleNamespace(
        _flatten_to_domain=base_flatten,
        _build_m2d=base_build,
        QuadMesh=FakeQuadMesh,
        _original=None,
    )
    install_optcuts_grid_constrained_m2d_patch(fake_pipeline)
    grid = SimpleNamespace(nx=8, ny=8, tile_size=h, gap_size=0.0)
    params = SimpleNamespace(tile_size=h, omega_overlay_margin=1)
    domain = fake_pipeline._flatten_to_domain(parameterization, grid, params)
    m2d = fake_pipeline._build_m2d(grid, domain, params)
    assert len(m2d.faces) > 0
    assert m2d.metrics["optcuts_grid_constrained_m2d"] is True
    assert m2d.metrics["optcuts_grid_posthoc_seam_snap"] is False

    # Finally audit the real M2D -> physical-surface inverse map.
    lifted, audit = _map_used_vertices(
        m2d,
        parameterization,
        bary_tol=1.0e-8,
        agreement_tol=1.0e-6,
    )
    assert np.all(np.isfinite(lifted))
    assert audit["used_vertex_count"] > 0
    assert audit["max_candidate_3d_spread"] <= 1.0e-6
