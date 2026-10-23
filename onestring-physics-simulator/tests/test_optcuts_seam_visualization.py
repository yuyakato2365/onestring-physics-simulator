from types import SimpleNamespace

import numpy as np

from onestring_physics.optcuts_visualization_compat_patch import (
    _optcuts_seam_segments_2d,
    _seam_adjacent_m2d_face_ids,
)


def _state_with_duplicated_zero_width_seam():
    # Two quads share the same geometric seam x=1, but the right quad uses
    # duplicated seam vertices (6, 7), exactly as a topology cut does.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1 seam original
            [1.0, 1.0, 0.0],  # 2 seam original
            [0.0, 1.0, 0.0],  # 3
            [2.0, 0.0, 0.0],  # 4
            [2.0, 1.0, 0.0],  # 5
            [1.0, 0.0, 0.0],  # 6 seam duplicate
            [1.0, 1.0, 0.0],  # 7 seam duplicate
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2, 3],
            [6, 4, 5, 7],
        ],
        dtype=int,
    )
    mesh = SimpleNamespace(
        vertices=vertices,
        faces=faces,
        _optcuts_grid_seam_paths=[[1, 2]],
    )
    return SimpleNamespace(mesh_2d_initial=mesh)


def test_final_grid_seam_segments_are_read_from_optcuts_paths():
    state = _state_with_duplicated_zero_width_seam()
    segments = _optcuts_seam_segments_2d(state)
    assert segments.shape == (1, 2, 2)
    assert np.allclose(segments[0], [[1.0, 0.0], [1.0, 1.0]])


def test_seam_adjacency_finds_both_sides_after_vertex_duplication():
    state = _state_with_duplicated_zero_width_seam()
    face_ids = _seam_adjacent_m2d_face_ids(state)
    assert face_ids.tolist() == [0, 1]



def _peak_marker_state(records):
    # Source-chart vertices and M2D grid vertices belong to different spaces.
    # The reflatten peak refers to grid vertex #3, which is NOT a source UV id.
    from types import SimpleNamespace
    chart_uv = np.array([[0., 0.], [1., 0.]], dtype=float)
    chart_xyz = np.array([[0., 0., 0.], [1., 0., .1]], dtype=float)
    grid_uv = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    grid_xyz = np.array([[0., 0., 0.], [1., 0., .1],
                         [0., 1., .2], [1., 1., .8]], dtype=float)
    p = SimpleNamespace(uv_vertices_2d=chart_uv)
    mesh = SimpleNamespace(
        vertices=grid_uv,
        _split_panel_source_vertices=grid_uv,
        metrics=dict(paper_split_finalized=True, simple_split_records=records),
    )
    return SimpleNamespace(
        surface_parameterization=p,
        mesh_2d_initial=mesh,
        mesh_3d_initial=SimpleNamespace(vertices=grid_xyz),
    ), chart_xyz, grid_uv, grid_xyz


def test_split_peak_markers_skip_collision_only_records(monkeypatch):
    from onestring_physics import visualization
    from onestring_physics import paper_mesh_splitting

    records = [
        dict(collision_driven=True, axis="row", snapped_value=.5),
        dict(reflatten_csf_cut=True, curvature_vertex_id=3),
        dict(collision_driven=True, axis="col", snapped_value=.75),
    ]
    state, chart_xyz, grid_uv, grid_xyz = _peak_marker_state(records)
    monkeypatch.setattr(paper_mesh_splitting, "chart_geometry",
                        lambda p: (chart_xyz, None, None))
    uv, xyz = visualization._surface_peak_markers(state)
    np.testing.assert_allclose(uv, grid_uv[[3]])
    np.testing.assert_allclose(xyz, grid_xyz[[3]])


def test_split_peak_markers_retain_fixed_uv_source_index(monkeypatch):
    from onestring_physics import visualization
    from onestring_physics import paper_mesh_splitting

    records = [
        dict(curvature_vertex_id=1, collision_driven=False),
        dict(curvature_vertex_id=1, collision_driven=False),
        dict(collision_driven=True, axis="row", snapped_value=.5),
        dict(reflatten_csf_cut=True, curvature_vertex_id=999),
    ]
    state, chart_xyz, _grid_uv, _grid_xyz = _peak_marker_state(records)
    monkeypatch.setattr(paper_mesh_splitting, "chart_geometry",
                        lambda p: (chart_xyz, None, None))
    uv, xyz = visualization._surface_peak_markers(state)
    np.testing.assert_allclose(uv, [[1., 0.]])
    np.testing.assert_allclose(xyz, [[1., 0., .1]])


def test_split_peak_markers_without_curvature_peaks_return_empty(monkeypatch):
    from onestring_physics import visualization
    from onestring_physics import paper_mesh_splitting

    state, chart_xyz, _, _ = _peak_marker_state([
        dict(collision_driven=True, axis="col"),
        dict(collision_driven=True, axis="row"),
    ])
    monkeypatch.setattr(paper_mesh_splitting, "chart_geometry",
                        lambda p: (chart_xyz, None, None))
    uv, xyz = visualization._surface_peak_markers(state)
    assert uv.shape == (0, 2)
    assert xyz.shape == (0, 3)
