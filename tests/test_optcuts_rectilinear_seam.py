from __future__ import annotations

import numpy as np

from onestring_physics.optcuts_rectilinear_seam_patch import (
    _build_rectilinear_cut_network,
    _extract_source_chains,
)


def _regular_grid(nx: int, ny: int, h: float = 1.0):
    vertices = np.asarray([[x * h, y * h] for y in range(ny + 1) for x in range(nx + 1)], dtype=float)

    def vid(x: int, y: int) -> int:
        return y * (nx + 1) + x

    faces = []
    for y in range(ny):
        for x in range(nx):
            faces.append([vid(x, y), vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    return vertices, np.asarray(faces, dtype=int)


def test_degree_two_optcuts_points_collapse_to_one_chain():
    nodes = {
        0: np.asarray([0.1, 0.2]),
        1: np.asarray([0.8, 0.4]),
        2: np.asarray([1.4, 0.9]),
        3: np.asarray([2.1, 1.1]),
    }
    chains = _extract_source_chains(nodes, [(0, 1), (1, 2), (2, 3)])
    assert chains == [[0, 1, 2, 3]]


def test_shared_junction_is_preserved_across_chains():
    nodes = {
        0: np.asarray([0.0, 1.0]),
        1: np.asarray([1.0, 1.0]),
        2: np.asarray([2.0, 1.0]),
        3: np.asarray([1.0, 2.0]),
    }
    chains = _extract_source_chains(nodes, [(0, 1), (1, 2), (1, 3)])
    ends = [{chain[0], chain[-1]} for chain in chains]
    assert {0, 1} in ends
    assert {1, 2} in ends
    assert {1, 3} in ends


def test_free_diagonal_chain_becomes_axis_aligned_grid_edges():
    vertices, faces = _regular_grid(4, 4, 1.0)
    nodes = {
        0: np.asarray([0.1, 0.1]),
        1: np.asarray([1.0, 0.8]),
        2: np.asarray([2.0, 1.7]),
        3: np.asarray([3.9, 3.9]),
    }
    edges = [(0, 1), (1, 2), (2, 3)]
    cut_edges, paths, stats = _build_rectilinear_cut_network(vertices, faces, nodes, edges, 1.0)
    assert stats["chain_count"] == 1
    assert len(paths) == 1
    assert cut_edges
    for a, b in cut_edges:
        delta = np.abs(vertices[a] - vertices[b])
        assert np.isclose(delta[0], 0.0) or np.isclose(delta[1], 0.0)
        assert np.isclose(np.sum(delta), 1.0)


def test_rectilinear_path_uses_fixed_grid_unit():
    vertices, faces = _regular_grid(5, 3, 0.25)
    nodes = {0: np.asarray([0.02, 0.02]), 1: np.asarray([1.23, 0.73])}
    cut_edges, paths, _stats = _build_rectilinear_cut_network(vertices, faces, nodes, [(0, 1)], 0.25)
    assert paths
    for a, b in cut_edges:
        length = float(np.linalg.norm(vertices[a] - vertices[b]))
        assert np.isclose(length, 0.25)
