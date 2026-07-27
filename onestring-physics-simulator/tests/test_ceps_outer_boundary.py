from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onestring_physics.official_ceps import _omega_boundary
from onestring_physics.onestring_pipeline import _clip_m2d_faces_to_omega_boundary


def test_ceps_boundary_uses_true_concave_topological_loop_not_convex_hull() -> None:
    # L-shaped disk. The physical boundary contains the concave corner (1, 1),
    # which a convex hull would delete and thereby invent UV coverage.
    uv = np.asarray(
        [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float
    )
    faces = np.asarray(
        [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]], dtype=int
    )

    boundary, metrics = _omega_boundary(uv, faces)

    assert np.allclose(boundary[0], boundary[-1])
    assert any(np.allclose(point, [1.0, 1.0]) for point in boundary[:-1])
    assert metrics["ceps_omega_boundary_source"] == "physical_boundary_loop_of_stitched_common_refinement"
    assert metrics["ceps_omega_boundary_convex_hull_used"] is False
    assert metrics["ceps_input_open_boundary_preserved"] is True
    assert metrics["ceps_surface_boundary_loop_count"] == 1


def test_ceps_physical_boundary_keeps_regular_m2d_quads() -> None:
    uv = np.asarray(
        [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0], [0.0, 0.0]],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    boundary, _metrics = _omega_boundary(uv, faces)

    xs = np.linspace(-2.0, 2.0, 5)
    ys = np.linspace(-1.0, 1.0, 3)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.column_stack([xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size)])
    quads = []
    for row in range(2):
        for column in range(4):
            a = row * 5 + column
            quads.append([a, a + 1, a + 6, a + 5])

    mesh = SimpleNamespace(vertices=vertices, faces=np.asarray(quads, dtype=int))
    domain = SimpleNamespace(boundary=boundary, omega_boundary_forced_rectangle=True)
    params = SimpleNamespace(m2d_crop_policy="center")

    kept, metrics = _clip_m2d_faces_to_omega_boundary(mesh, domain, params)

    assert len(kept) == 8
    assert metrics["m2d_boundary_clip_policy_effective"] == "strict_vertices"
