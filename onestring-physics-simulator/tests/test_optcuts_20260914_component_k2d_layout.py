from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from onestring_physics.optcuts_paper_k2d_20260914_patch import (
    _build_componentwise_flat_layout,
    _edge_components,
    _submesh_for_faces,
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
    # Two 1x2 strips.  Component 2 deliberately uses high/global vertex IDs so
    # the test catches accidental compact renumbering.
    vertices = np.zeros((20, 3), dtype=float)
    for i in range(len(vertices)):
        vertices[i, :2] = [float(i % 5), float(i // 5)]
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


def test_edge_components_respect_seam_disconnection():
    mesh = _two_component_mesh()
    components = _edge_components(mesh.faces)
    assert [c.tolist() for c in components] == [[0, 1], [2, 3]]


def test_submesh_preserves_original_vertex_ids():
    mesh = _two_component_mesh()
    sub = _submesh_for_faces(mesh, np.asarray([2, 3], dtype=int))
    assert np.array_equal(sub.faces, mesh.faces[[2, 3]])
    assert len(sub.vertices) == len(mesh.vertices)
    assert sub.metrics["k2d_component_preserves_global_vertex_ids"] is True
    assert sub.metrics["k2d_component_preserves_lscm_lattice_parity_rule"] is True


def test_componentwise_layout_calls_lscm_builder_per_component_and_never_cross_hinges():
    mesh = _two_component_mesh()
    calls: list[np.ndarray] = []

    def fake_lscm_make_layout(submesh, params):
        del params
        calls.append(np.asarray(submesh.faces, dtype=int).copy())
        tiles = np.asarray(submesh.vertices, dtype=float)[np.asarray(submesh.faces, dtype=int), :2]
        hinges = [(0, 1)] if len(tiles) == 2 else []
        return DummyLayout(
            tile_top_vertices_2d=tiles,
            tile_ids=list(range(len(tiles))),
            hinge_pairs=hinges,
            gap_polygons=[np.zeros((4, 2), dtype=float)] if hinges else [],
            metrics={
                "layout_type": "ordinary LSCM test layout",
                "tile_overlap_count": 0,
                "min_clearance": 1.0,
            },
        )

    merged = _build_componentwise_flat_layout(mesh, object(), fake_lscm_make_layout)

    assert len(calls) == 2
    assert np.array_equal(calls[0], mesh.faces[[0, 1]])
    assert np.array_equal(calls[1], mesh.faces[[2, 3]])
    assert merged.tile_top_vertices_2d.shape == (4, 4, 2)
    assert set(merged.hinge_pairs) == {(0, 1), (2, 3)}
    assert (1, 2) not in merged.hinge_pairs
    assert merged.metrics["k2d_flat_layout_component_count"] == 2
    assert merged.metrics["k2d_flat_layout_global_cross_component_solve"] is False
    assert merged.metrics["k2d_flat_layout_cross_component_hinges"] == 0
    assert merged.metrics["k2d_flat_layout_preserves_original_vertex_ids"] is True
