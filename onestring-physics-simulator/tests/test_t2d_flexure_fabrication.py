"""Geometry and authoritative-side regression tests for optional print flexures."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.t2d_flexure_fabrication import (
    FlexureParameters,
    export_final_t2d_flexure_stl,
    make_flexure_triangles,
)


def tile(xy, ztop=2., zbottom=0.):
    xy = np.asarray(xy, dtype=float)
    top = np.column_stack((xy, np.full(4, ztop)))
    bottom = np.column_stack((xy, np.full(4, zbottom)))
    return np.concatenate((top, bottom))


def fixture():
    a = tile([[-10, -10], [0, -10], [0, 0], [-10, 0]])
    b = tile([[0, 0], [10, 0], [10, 10], [0, 10]])
    return np.stack((a, b))


def hinge(side="top", ca=None, cb=None, a=0, b=1):
    offset = 0 if side == "top" else 4
    return SimpleNamespace(
        tile_a=a, tile_b=b,
        local_vertex_a=2 + offset if ca is None else ca,
        local_vertex_b=0 + offset if cb is None else cb,
        surface=side,
    )


def assert_closed_shell(triangles):
    """Each ribbon is a closed shell (the later tile union is still separate)."""
    counts = Counter()
    for tri in triangles:
        for i in range(3):
            x = tuple(np.round(tri[i], 7))
            y = tuple(np.round(tri[(i + 1) % 3], 7))
            counts[tuple(sorted((x, y)))] += 1
    assert counts and all(n == 2 for n in counts.values())


@pytest.mark.parametrize("side,expected_low,expected_high", [
    ("top", 1.4, 2.0),
    ("bottom", 0.0, 0.6),
])
def test_uses_authoritative_top_bottom_surface(side, expected_low, expected_high):
    mesh, metrics = make_flexure_triangles(
        fixture(), SimpleNamespace(hinges=[hinge(side)]),
        FlexureParameters(skin_mm=.6, embed_mm=.2),
    )
    arr = np.array(mesh)
    assert arr.shape == (20, 3, 3)
    assert np.min(arr[..., 2]) == pytest.approx(expected_low)
    assert np.max(arr[..., 2]) == pytest.approx(expected_high)
    assert metrics[f"flexure_{side}_count"] == 1
    assert metrics["flexure_hinge_count"] == 1
    assert_closed_shell(arr)


def test_top_and_bottom_hinges_do_not_get_reclassified():
    # The attachment coordinates coincide in XY, yet use separate faces.
    g = SimpleNamespace(hinges=[hinge("top"), hinge("bottom")])
    tris, metrics = make_flexure_triangles(fixture(), g)
    assert len(tris) == 40
    assert metrics["flexure_top_count"] == 1
    assert metrics["flexure_bottom_count"] == 1


def test_explicit_empty_split_topology_is_never_rewelded():
    tris, metrics = make_flexure_triangles(
        fixture(), SimpleNamespace(hinges=[])
    )
    assert tris == []
    assert metrics["flexure_hinge_count"] == 0


def test_wrong_local_vertex_surface_raises_instead_of_guessing():
    bad = hinge("bottom", ca=2, cb=0)
    with pytest.raises(ValueError, match="disagrees"):
        make_flexure_triangles(fixture(), SimpleNamespace(hinges=[bad]))


def test_t2d_mismatched_joint_is_rejected():
    tiles = fixture()
    tiles[1, :, 0] += 2.
    with pytest.raises(ValueError, match="attachment gap"):
        make_flexure_triangles(tiles, SimpleNamespace(hinges=[hinge()]))



@pytest.mark.parametrize("side", ["top", "bottom"])
def test_realistic_residual_hinge_gap_is_bridged_on_correct_side(side):
    tiles = fixture()
    tiles[1, :, 0] += .732
    graph = SimpleNamespace(hinges=[hinge(side)])
    triangles, metrics = make_flexure_triangles(tiles, graph)
    assert len(triangles) == 20
    assert metrics["flexure_max_joint_gap_mm"] == pytest.approx(.732)
    assert metrics["flexure_nonzero_gap_count"] == 1
    assert metrics["flexure_bridge_limit_mm"] == pytest.approx(1.5)
    assert metrics[f"flexure_{side}_count"] == 1
    assert_closed_shell(triangles)
    with pytest.raises(ValueError, match=r"exceeds the printable bridge limit 0.6 mm"):
        make_flexure_triangles(
            tiles, graph, FlexureParameters(max_joint_gap_mm=.6)
        )


def test_large_gap_is_still_rejected_after_relaxation():
    tiles = fixture()
    tiles[1, :, 0] += 6.
    with pytest.raises(ValueError, match="bridge limit"):
        make_flexure_triangles(
            tiles, SimpleNamespace(hinges=[hinge()]),
            FlexureParameters(max_joint_gap_mm=5.),
        )

def test_t2d_duplicate_links_are_not_printed_twice():
    g = SimpleNamespace(hinges=[hinge(), hinge()])
    with pytest.raises(ValueError, match="duplicate hinge"):
        make_flexure_triangles(fixture(), g)


def test_flexure_handles_tilted_print_face_without_flattening():
    tiles = fixture()
    tiles[0, 0, 2] += 1.
    before = tiles.copy()
    triangles, metrics = make_flexure_triangles(
        tiles, SimpleNamespace(hinges=[hinge()]),
    )
    assert len(triangles) == 20
    assert metrics["flexure_max_original_face_z_spread_mm"] >= 1.
    np.testing.assert_array_equal(tiles, before)
    assert_closed_shell(triangles)


def test_final_export_merges_tile_and_flexure_triangles_without_touching_source():
    from onestring_physics import onestring_pipeline as p

    tiles = fixture()
    t2d = SimpleNamespace(
        vertices=tiles.copy(),
        top_faces=np.array([[0, 1, 2, 3]]),
        bottom_faces=np.array([[4, 7, 6, 5]]),
        metrics={},
    )
    state = SimpleNamespace(
        tiles_2d_dual_hinge=t2d,
        hinge_graph=SimpleNamespace(hinges=[hinge()]),
    )
    raw, metrics = export_final_t2d_flexure_stl(
        state, panel_size_mm=40,
    )
    assert raw.startswith(b"solid onestring_t2d_dual_flexure_mm")
    assert raw.count(b"facet normal") == 2 * 12 + 20
    assert metrics["t2d_face_count"] == 44
    assert metrics["flexure_top_count"] == 1
    assert metrics["t2d_fabrication_unit"].startswith("mm")
    np.testing.assert_allclose(t2d.vertices, tiles)
    assert t2d.metrics == {}


def test_invalid_flexure_parameters_rejected():
    with pytest.raises(ValueError, match="finite positive"):
        make_flexure_triangles(
            fixture(), SimpleNamespace(hinges=[hinge()]),
            FlexureParameters(skin_mm=0.),
        )


def test_flexure_thickness_does_not_rewrite_thin_panel_geometry():
    # Flexure parameters never authorize changing or flattening original tiles.
    tiles = fixture()
    tiles[:, 4:, 2] = 1.8  # Deliberately thin 0.2 mm plates.
    before = tiles.copy()
    assembly = SimpleNamespace(
        vertices=tiles, top_faces=np.array([[0, 1, 2, 3]]),
        bottom_faces=np.array([[4, 7, 6, 5]]), metrics={},
    )
    state = SimpleNamespace(
        tiles_2d_dual_hinge=assembly,
        hinge_graph=SimpleNamespace(hinges=[hinge()]),
    )
    raw, metrics = export_final_t2d_flexure_stl(
        state, panel_size_mm=40.,
        parameters=FlexureParameters(skin_mm=.6),
    )
    assert metrics["flexure_original_panel_geometry_preserved"]
    assert raw.count(b"facet normal") == 44
    np.testing.assert_array_equal(assembly.vertices, before)


def extract_stl_triangles(raw):
    points = [
        [float(n) for n in line.strip().split()[1:]]
        for line in raw.decode("ascii").splitlines()
        if line.strip().startswith("vertex ")
    ]
    return np.asarray(points, dtype=float).reshape(-1, 3, 3)


@pytest.mark.parametrize("side", ["top", "bottom"])
def test_print_export_projects_small_nonplanarity_without_reclassifying_side(side):
    # Numerical K3D/T2D planarity is not exact. Neighbor tiles can also
    # retain different Z levels even after 2 mm per-tile extrusion scaling.
    from onestring_physics import onestring_pipeline as p

    source = fixture()
    source[0, :4, 2] += np.array([.05, -.09, .06, -.02])
    source[0, 4:, 2] += np.array([.01, -.04, .02, -.03])
    source[1, :4, 2] += np.array([.18, .25, .22, .19])
    source[1, 4:, 2] += np.array([-.11, -.06, -.04, -.08])
    unchanged = source.copy()
    t2d = SimpleNamespace(vertices=source, top_faces=np.array([[0, 1, 2, 3]]),
                          bottom_faces=np.array([[4, 7, 6, 5]]), metrics={})
    state = SimpleNamespace(
        tiles_2d_dual_hinge=t2d,
        hinge_graph=SimpleNamespace(hinges=[hinge(side)]),
    )
    data, metrics = export_final_t2d_flexure_stl(
        state, panel_size_mm=40.,
        parameters=FlexureParameters(skin_mm=.6, embed_mm=.2),
    )
    assert not metrics["flexure_print_z_normalized"]
    assert metrics["flexure_original_panel_geometry_preserved"]
    assert metrics["flexure_preprint_top_level_spread_mm"] > 0
    assert metrics["flexure_preprint_bottom_level_spread_mm"] > 0
    assert metrics["flexure_preprint_max_face_warp_mm"] > 0
    assert metrics["flexure_preprint_max_best_fit_face_nonplanarity_mm"] > 0
    assert metrics["flexure_flattening_diagnostic_threshold_mm"] > 0
    assert metrics[f"flexure_{side}_count"] == 1
    assert data.count(b"facet normal") == 44
    # The 24 panel triangles are the first 24 STL facets and must exactly
    # equal ALL original T2D eight-corner prisms after ONE uniform transform.
    from onestring_physics import onestring_pipeline as p
    scale = p._t2d_export_scale(t2d, 40.)
    expected = unchanged * scale
    expected[:, :, 2] -= float(expected[..., 2].min())
    panel_tris = np.array([
        tile[list(indices)]
        for tile in expected
        for indices in p._prism_triangle_indices()
    ])
    np.testing.assert_allclose(
        extract_stl_triangles(data)[:len(panel_tris)],
        panel_tris, atol=1e-6, rtol=1e-9,
    )
    assert metrics["flexure_panel_scale_uniform"] == pytest.approx(scale)
    assert metrics["flexure_panel_relative_positions_preserved"]
    # Distances among every point remain in fixed proportions; no tile
    # thickness normalization or face projection is allowed.
    all_orig = unchanged.reshape(-1, 3)
    all_print = expected.reshape(-1, 3)
    np.testing.assert_allclose(
        np.linalg.norm(all_print[:, None, :] - all_print[None, :, :], axis=-1),
        scale * np.linalg.norm(all_orig[:, None, :] - all_orig[None, :, :], axis=-1),
        atol=1e-8,
    )
    # The original optimized (imperfectly planar) source is NEVER modified.
    np.testing.assert_array_equal(t2d.vertices, unchanged)
    assert t2d.metrics == {}


def test_print_export_does_not_flatten_source_for_split_boundary_without_joint():
    source = fixture()
    source[:, :4, 2] += .3
    unchanged = source.copy()
    t2d = SimpleNamespace(
        vertices=source, top_faces=np.array([[0, 1, 2, 3]]),
        bottom_faces=np.array([[4, 7, 6, 5]]), metrics={},
    )
    state = SimpleNamespace(
        tiles_2d_dual_hinge=t2d,
        hinge_graph=SimpleNamespace(hinges=[]),
    )
    raw, metrics = export_final_t2d_flexure_stl(state)
    assert raw.count(b"facet normal") == 24
    assert metrics["flexure_hinge_count"] == 0
    np.testing.assert_array_equal(t2d.vertices, unchanged)

def test_preserve_original_t2d_even_with_large_interpanel_height_difference():
    source = fixture()
    source[1, :, 2] += 1.25
    before = source.copy()
    state = SimpleNamespace(
        tiles_2d_dual_hinge=SimpleNamespace(
            vertices=source, top_faces=np.array([[0, 1, 2, 3]]),
            bottom_faces=np.array([[4, 7, 6, 5]]), metrics={},
        ),
        hinge_graph=SimpleNamespace(hinges=[hinge("top")]),
    )
    raw, metrics = export_final_t2d_flexure_stl(
        state, panel_size_mm=40.,
        parameters=FlexureParameters(max_joint_gap_mm=50.),
    )
    assert metrics["flexure_max_original_joint_height_difference_mm"] == pytest.approx(5.)
    assert metrics["flexure_requires_geometry_review"]
    assert not metrics["flexure_print_z_normalized"]
    assert raw.count(b"facet normal") == 44
    np.testing.assert_array_equal(state.tiles_2d_dual_hinge.vertices, before)
