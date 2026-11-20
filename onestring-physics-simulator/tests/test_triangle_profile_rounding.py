import numpy as np
import trimesh

from tools.triangle_omega.scripts.fair.round_profile import round_profile


def test_rounded_spike_meets_profile_and_hinge_targets():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    vertices = mesh.vertices.copy()
    vertices[vertices[:, 2] > .99, 2] += .35
    result, stats, before, after = round_profile(vertices, mesh.faces, .65, 45.)
    assert before.max() > 45. and after.max() <= 45.
    assert stats['hinges_above_before'] > 0 and stats['hinges_above_after'] == 0
    assert stats['intersections_after'] == 0
    original = trimesh.Trimesh(vertices, mesh.faces, process=False)
    rounded = trimesh.Trimesh(result, mesh.faces, process=False)
    assert abs(rounded.volume / original.volume - 1) < .001


def test_flat_open_surface_is_unchanged():
    vertices = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [1.,1.,0.]])
    faces = np.array([[0,1,2], [1,3,2]])
    result, stats, _, _ = round_profile(vertices, faces, .1, 45.)
    np.testing.assert_array_equal(result, vertices)
    assert stats['vertices_above_after'] == 0
