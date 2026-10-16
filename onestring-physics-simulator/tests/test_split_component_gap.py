"""Paper-oriented Split island gap spacing regression tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import onestring_pipeline as pipe


def _run(monkeypatch, multiplier, connected=False):
    import onestring_physics.paper_t2d as paper_t2d
    import onestring_physics.solid_collision_audit as audit
    monkeypatch.setattr(paper_t2d, "validate_flat_linkage", lambda *a: {
        "flat_collision_count": 0, "fabrication_feasible": True,
    })
    monkeypatch.setattr(audit, "audit_solid_collisions", lambda *a: {
        "solid_collision_pair_count": 0,
    })
    tile = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, -.2], [1, 0, -.2], [1, 1, -.2], [0, 1, -.2],
    ], float)
    source = np.stack([tile, tile + np.array([1., 0., 0.])])
    transforms = np.repeat(np.eye(4)[None], 2, axis=0)
    def assembly():
        return SimpleNamespace(
            vertices=source.copy(), transform_matrices=transforms.copy(),
            top_to_bottom_transforms=transforms.copy(), metrics={},
        )
    final, top = assembly(), assembly()
    final.metrics["hard_collision_feasible"] = True
    hinges = [SimpleNamespace(
        tile_a=0, tile_b=1, local_vertex_a=1, local_vertex_b=0,
        rest_position_2d=np.array([1., 0., 0.]),
    )] if connected else []
    graph = SimpleNamespace(hinges=hinges, metrics={})
    grid = SimpleNamespace(tile_size=1., gap_size=.1)
    params = pipe.PipelineParameters(
        paper_fidelity_mode=True, collision_split_gap_multiplier=multiplier,
        hinge_layout_collision_mode="hard",
    )
    out, graph = pipe._apply_paper_split_component_gap(
        final, graph, top, object(), grid, params
    )
    return source, out, top, graph


def _separation(assembly):
    a = assembly.vertices[0, :4, :2]
    b = assembly.vertices[1, :4, :2]
    return max(np.max(b.min(axis=0)-a.max(axis=0)),
               np.max(a.min(axis=0)-b.max(axis=0)))


def test_gap_three_moves_entire_islands_further_than_gap_one(monkeypatch):
    source, one, _, _ = _run(monkeypatch, 1.)
    _, three, top, graph = _run(monkeypatch, 3.)
    assert _separation(one) >= .2 - 1e-8
    assert _separation(three) >= .6 - 1e-8
    assert _separation(three) > _separation(one) + .35
    assert len(graph.hinges) == 0
    assert three.metrics["split_component_gap_applied"] is True
    assert three.metrics["fabrication_feasible"] is True
    for k in range(2):
        delta = three.vertices[k, 0] - source[k, 0]
        assert np.allclose(three.vertices[k], source[k] + delta)
        assert np.allclose(top.vertices[k], source[k] + delta)
        assert np.allclose(three.transform_matrices[k, :2, 3], delta[:2])


def test_connected_hinges_stay_connected(monkeypatch):
    source, out, top, graph = _run(monkeypatch, 3., connected=True)
    assert np.array_equal(out.vertices, source)
    assert np.array_equal(top.vertices, source)
    assert len(graph.hinges) == 1
    assert out.metrics["split_component_gap_applied"] is False


def test_invalid_multiplier_rejected(monkeypatch):
    with pytest.raises(ValueError, match="between 1 and 10"):
        _run(monkeypatch, 11.)


def test_current_default_params_match_requested_preset():
    p = pipe.PipelineParameters()
    assert p.w_square == 100.
    assert p.hinge_layout_iterations == 1000
    assert p.hinge_layout_connection_weight == 1.
    assert p.hinge_layout_collision_weight == 100.
    assert p.hinge_layout_collision_mode == "hard"
    assert p.collision_split_gap_multiplier == 3.
