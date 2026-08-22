import numpy as np

from onestring_physics.optcuts_seam_requirement_patch import _chain_targets


def _line_error(points: np.ndarray) -> float:
    pts = np.asarray(points, float)
    if len(pts) <= 2:
        return 0.0
    a, b = pts[0], pts[-1]
    d = b - a
    denom = float(np.dot(d, d))
    if denom <= 1e-20:
        return float(np.max(np.linalg.norm(pts - a[None, :], axis=1)))
    t = ((pts - a[None, :]) @ d) / denom
    proj = a[None, :] + t[:, None] * d[None, :]
    return float(np.max(np.linalg.norm(pts - proj, axis=1)))


def test_stage2_straightens_degree2_seam_chain():
    nodes = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.35]),
        2: np.array([2.0, -0.20]),
        3: np.array([3.0, 0.10]),
    }
    edges = [(0, 1), (1, 2), (2, 3)]
    targets, stats = _chain_targets(nodes, edges)
    pts = np.asarray([targets[i] for i in range(4)], float)
    assert stats["chain_count"] == 1
    assert _line_error(pts) < 1e-10


def test_stage3_places_straight_chain_on_real_horizontal_grid_line():
    nodes = {
        0: np.array([0.12, 0.37]),
        1: np.array([0.62, 0.37]),
        2: np.array([1.12, 0.37]),
    }
    edges = [(0, 1), (1, 2)]
    targets, stats = _chain_targets(
        nodes,
        edges,
        grid_size=0.25,
        phase_u=0.0,
        phase_v=0.0,
        angle_degrees=0.0,
    )
    pts = np.asarray([targets[i] for i in range(3)], float)
    assert stats["chain_axes"] == ["grid_u"]
    assert np.allclose(pts[:, 1], 0.25)
    assert np.allclose(pts[:, 0], [0.12, 0.62, 1.12])


def test_stage3_places_straight_chain_on_real_vertical_grid_line():
    nodes = {
        0: np.array([0.61, 0.1]),
        1: np.array([0.61, 0.6]),
        2: np.array([0.61, 1.1]),
    }
    edges = [(0, 1), (1, 2)]
    targets, stats = _chain_targets(
        nodes,
        edges,
        grid_size=0.2,
        phase_u=0.0,
        phase_v=0.0,
        angle_degrees=0.0,
    )
    pts = np.asarray([targets[i] for i in range(3)], float)
    assert stats["chain_axes"] == ["grid_v"]
    assert np.allclose(pts[:, 0], 0.6)
    assert np.allclose(pts[:, 1], [0.1, 0.6, 1.1])


def test_grid_alignment_respects_grid_rotation():
    # A 45-degree world-space seam is horizontal in a grid rotated by 45 degrees.
    t = np.sqrt(0.5)
    nodes = {
        0: np.array([0.0, 0.0]),
        1: np.array([t, t]),
        2: np.array([2.0 * t, 2.0 * t]),
    }
    edges = [(0, 1), (1, 2)]
    targets, stats = _chain_targets(
        nodes,
        edges,
        grid_size=0.25,
        phase_u=0.0,
        phase_v=0.0,
        angle_degrees=45.0,
    )
    pts = np.asarray([targets[i] for i in range(3)], float)
    assert stats["chain_axes"] == ["grid_u"]
    assert _line_error(pts) < 1e-10
