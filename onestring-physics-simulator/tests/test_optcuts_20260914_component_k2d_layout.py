from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_paper_k2d_20260914_patch import (
    install_optcuts_paper_k2d_20260914_patch,
)


@dataclass
class DummyMesh:
    vertices: np.ndarray
    faces: np.ndarray
    grid: object = None
    stage: str = "K2D"
    metrics: dict = field(default_factory=dict)
    split_lines: list = field(default_factory=list)


@dataclass
class DummyLayout:
    tile_top_vertices_2d: np.ndarray
    tile_ids: list[int]
    hinge_pairs: list[tuple[int, int]]
    gap_polygons: list[np.ndarray] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _two_component_mesh() -> DummyMesh:
    vertices = np.zeros((20, 3), dtype=float)
    for i in range(len(vertices)):
        vertices[i, :2] = [float(i % 5), float(i // 5)]
    # Two disconnected 1x2 strips.  The dated patch must still pass all four
    # faces to the common LSCM flat-layout function in one call.
    faces = np.asarray(
        [
            [0, 1, 6, 5],
            [1, 2, 7, 6],
            [10, 11, 16, 15],
            [11, 12, 17, 16],
        ],
        dtype=int,
    )
    return DummyMesh(vertices=vertices, faces=faces)


def test_latest_flat_layout_calls_exact_lscm_builder_once_on_whole_mesh(monkeypatch):
    monkeypatch.setenv("ONESTRING_PAPER_K2D_20260914", "1")
    monkeypatch.setenv("ONESTRING_OPTCUTS_TEST_VARIANT", "0")

    mesh = _two_component_mesh()
    params = SimpleNamespace(omega_parameterization_mode="optcuts_test")
    calls: list[object] = []

    def fake_k2d(mesh_2d, mesh_3d, params, progress_callback=None):
        del mesh_3d, params, progress_callback
        report = SimpleNamespace(objective="base")
        return mesh_2d, report

    def fake_lscm_make_layout(input_mesh, input_params):
        assert input_params is params
        calls.append(input_mesh)
        tiles = np.asarray(input_mesh.vertices, dtype=float)[np.asarray(input_mesh.faces, dtype=int), :2]
        return DummyLayout(
            tile_top_vertices_2d=tiles,
            tile_ids=list(range(len(tiles))),
            hinge_pairs=[],
            gap_polygons=[],
            metrics={
                # Simulate the branch selected by the common LSCM function.
                "k2d_independent_fast_large_layout": True,
                "k2d_independent_fast_tile_threshold": 150,
            },
        )

    pipeline = SimpleNamespace(
        _optimize_k2d=fake_k2d,
        _make_flat_tile_layout=fake_lscm_make_layout,
        _original=None,
        build_onestring_design=None,
        _ORIGINAL_BUILD_ONESTRING_DESIGN=None,
    )

    install_optcuts_paper_k2d_20260914_patch(pipeline)
    layout = pipeline._make_flat_tile_layout(mesh, params)

    # This is the regression: disconnected OptCuts topology must NOT cause a
    # component-wise re-entry into the common layout function.
    assert len(calls) == 1
    assert calls[0] is mesh
    assert np.array_equal(calls[0].faces, mesh.faces)

    assert layout.metrics["k2d_flat_layout_solver_equivalent_to_lscm"] is True
    assert layout.metrics["k2d_flat_layout_exact_same_captured_function"] is True
    assert layout.metrics["k2d_flat_layout_whole_mesh_call"] is True
    assert layout.metrics["k2d_flat_layout_componentwise_override"] is False
    assert layout.metrics["k2d_flat_layout_parallel_override"] is False
    assert layout.metrics["k2d_flat_layout_total_input_face_count"] == 4
    assert layout.metrics["k2d_independent_fast_large_layout"] is True


def test_latest_flat_layout_does_not_change_common_branch_input_size(monkeypatch):
    monkeypatch.setenv("ONESTRING_PAPER_K2D_20260914", "1")
    monkeypatch.setenv("ONESTRING_OPTCUTS_TEST_VARIANT", "0")

    # 151 independent faces deliberately exceed the common 150-tile threshold.
    # Even though every face is disconnected, the dated route must present all
    # 151 faces to the common function at once, exactly as LSCM would.
    vertices = np.zeros((151 * 4, 3), dtype=float)
    faces = np.arange(151 * 4, dtype=int).reshape(151, 4)
    mesh = DummyMesh(vertices=vertices, faces=faces)
    params = SimpleNamespace(omega_parameterization_mode="optcuts_test")
    observed_face_counts: list[int] = []

    def fake_k2d(mesh_2d, mesh_3d, params, progress_callback=None):
        del mesh_3d, params, progress_callback
        return mesh_2d, SimpleNamespace(objective="base")

    def fake_lscm_make_layout(input_mesh, input_params):
        del input_params
        face_count = int(len(input_mesh.faces))
        observed_face_counts.append(face_count)
        tiles = np.zeros((face_count, 4, 2), dtype=float)
        return DummyLayout(
            tile_top_vertices_2d=tiles,
            tile_ids=list(range(face_count)),
            hinge_pairs=[],
            metrics={
                "k2d_independent_fast_large_layout": bool(face_count > 150),
                "k2d_independent_fast_tile_threshold": 150,
            },
        )

    pipeline = SimpleNamespace(
        _optimize_k2d=fake_k2d,
        _make_flat_tile_layout=fake_lscm_make_layout,
        _original=None,
        build_onestring_design=None,
        _ORIGINAL_BUILD_ONESTRING_DESIGN=None,
    )

    install_optcuts_paper_k2d_20260914_patch(pipeline)
    layout = pipeline._make_flat_tile_layout(mesh, params)

    assert observed_face_counts == [151]
    assert layout.metrics["k2d_independent_fast_large_layout"] is True
    assert layout.metrics["k2d_flat_layout_whole_mesh_call"] is True
