"""M2D/M3D effective quad-area CSF is intentionally not BFF triangle CSF.

Paper-oriented Split computes a temporary M3D on the *uncut* M2D Grid;
later cuts disconnect existing quads but never recompute UV, distort panels,
or use final K3D/T3D in this diagnostic.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.paper_mesh_splitting import (
    _m2d_m3d_panel_area_field, _face_samples,
)


def _surface(uv, xyz, faces):
    return SimpleNamespace(
        uv_vertices_2d=np.asarray(uv, float),
        uv_faces=np.asarray(faces, int),
        surface_vertices_3d=np.asarray(xyz, float),
        surface_faces=np.asarray(faces, int),
        metrics={},
    )


def test_two_actual_m3d_quads_have_one_flat_and_one_slanted_area():
    # Two unit squares, right-hand quad folds up along its outer vertical edge.
    uv = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
    ], float)
    xyz = np.column_stack([uv, [0., 0., 1., 0., 0., 1.]])
    triangles = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
    ])
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    field, info = _m2d_m3d_panel_area_field(uv, quads, _surface(uv, xyz, triangles))
    assert field == pytest.approx([1., np.sqrt(2.)])
    assert info["split_panel_area_m2d_values"] == pytest.approx([1., 1.])
    assert info["split_panel_area_m3d_values"] == pytest.approx([1., np.sqrt(2.)])
    assert np.sqrt(field.max()/field.min()) == pytest.approx(2.**.25)


def test_hidden_source_spike_is_not_a_panel_area_expansion():
    # The source triangulation has a tall midpoint, but the actual M2D
    # grid has ONLY four corners. Strict M3D vertex lifting ignores this
    # interior feature, so the panel area is 1 even while BFF-triangle
    # Jacobians are much larger. This difference is the intended experiment.
    uv = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [.5, .5],
    ], float)
    xyz = np.column_stack([uv, [0., 0., 0., 0., 2.]])
    triangles = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
    ])
    quad = np.array([[0, 1, 2, 3]])
    surface = _surface(uv, xyz, triangles)
    j, info = _m2d_m3d_panel_area_field(uv, quad, surface)
    assert j == pytest.approx([1.])
    lo, hi, samples = _face_samples(
        uv, quad, surface, threshold=2., linear_csf=True,
        sampling_method="m2d_m3d_panel_area",
    )
    assert lo == pytest.approx([1.])
    assert hi == pytest.approx([1.])
    assert np.sqrt(hi.max()/lo.min()) == pytest.approx(1.)
    assert info["split_panel_area_source"] == "pre_split_M2D_M3D_strict_bff_inverse"
    old_lo, old_hi, _ = _face_samples(
        uv, quad, surface, threshold=2., linear_csf=True,
        sampling_method="overlap_extrema",
    )
    assert old_lo[0] > 2.
    assert old_hi[0] > 2.
    assert set(samples[0].tolist()) == set(range(5))


def test_invalid_zero_area_quad_is_rejected():
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    xyz = np.column_stack([uv, np.zeros(4)])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    surface = _surface(uv, xyz, triangles)
    with pytest.raises(ValueError, match="zero/invalid quad area"):
        _m2d_m3d_panel_area_field(
            uv, np.array([[0, 1, 1, 0]]), surface
        )
