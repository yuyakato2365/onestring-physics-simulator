"""Visual-only K3D mode: geometry is unchanged and planarity view is retained."""
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from onestring_physics.panel_quality_visualization import (
    figure_k3d_plain_blue, figure_k3d_planarity,
)


def _mesh(flat=False):
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0.0 if flat else .25], [0, 1, 0],
        [2, 0, 0], [2, 1, 0],
    ], dtype=float)
    q = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=int)
    return SimpleNamespace(vertices=v, faces=q, stage="K3D")


def test_plain_blue_keeps_exact_optimized_k3d_geometry_and_black_quad_edges():
    mesh = _mesh()
    original = mesh.vertices.copy()
    fig = figure_k3d_plain_blue(mesh)
    solids = [t for t in fig.data if isinstance(t, go.Mesh3d)]
    outlines = [t for t in fig.data if isinstance(t, go.Scatter3d)]
    assert len(solids) == 1 and len(outlines) == 1
    body = solids[0]
    assert body.color == "#5097f5"
    assert body.vertexcolor is None
    assert body.facecolor is None
    assert body.intensity is None
    assert body.showscale is False
    assert len(body.i) == 2 * len(mesh.faces)
    expected = mesh.vertices[mesh.faces].reshape(-1, 3)
    assert np.allclose(np.column_stack([body.x, body.y, body.z]), expected)
    assert outlines[0].mode == "lines"
    assert outlines[0].line.color == "#0f172a"
    assert np.array_equal(mesh.vertices, original)


def test_flat_k3d_is_still_blue_and_heatmap_is_still_available():
    mesh = _mesh(flat=True)
    blue = figure_k3d_plain_blue(mesh)
    heatmap = figure_k3d_planarity(mesh)
    assert blue.data[0].color == "#5097f5"
    assert heatmap.data[0].intensity is not None
    assert heatmap.data[0].colorscale is not None
    assert len(heatmap.data[0].i) == len(blue.data[0].i)
