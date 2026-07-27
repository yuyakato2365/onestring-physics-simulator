from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from onestring_physics.official_ceps import _omega_boundary
from onestring_physics.onestring_pipeline import _clip_m2d_faces_to_omega_boundary


def test_ceps_outer_boundary_ignores_internal_seam_vertices() -> None:
    # Four exterior rectangle corners plus points belonging to an internal CEPS
    # cut seam. The external Omega polygon must be the rectangle hull, not an
    # arbitrary boundary loop formed by the duplicated seam connectivity.
    uv = np.asarray(
        [
            [-2.0, -1.0],
            [2.0, -1.0],
            [2.0, 1.0],
            [-2.0, 1.0],
            [-0.2, -0.1],
            [0.2, -0.1],
            [0.0, 0.2],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 5],
            [2, 3, 6],
            [3, 0, 4],
            [4, 5, 6],
        ],
        dtype=int,
    )

    boundary, metrics = _omega_boundary(uv, faces)

    assert np.allclose(boundary[0], boundary[-1])
    assert np.allclose(np.min(boundary[:-1], axis=0), [-2.0, -1.0])
    assert np.allclose(np.max(boundary[:-1], axis=0), [2.0, 1.0])
    assert metrics["ceps_omega_boundary_source"] == "convex_hull_of_all_paired_ceps_uv_vertices"
    assert metrics["ceps_omega_boundary_convex"] is True
    assert metrics["ceps_internal_uv_seams_excluded_from_omega_boundary"] is True


def test_ceps_outer_boundary_keeps_regular_m2d_quads() -> None:
    uv = np.asarray(
        [
            [-2.0, -1.0],
            [2.0, -1.0],
            [2.0, 1.0],
            [-2.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    boundary, _metrics = _omega_boundary(uv, faces)

    # A 4x2 regular grid exactly covering the CEPS rectangle. Strict rectangular
    # clipping must retain all eight quads rather than raising the user's
    # "removed all quads" failure.
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
