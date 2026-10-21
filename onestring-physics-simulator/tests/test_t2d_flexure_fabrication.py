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
    ("top", 1.8, 2.4),
    ("bottom", -0.4, 0.2),
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


def test_t2d_duplicate_links_are_not_printed_twice():
    g = SimpleNamespace(hinges=[hinge(), hinge()])
    with pytest.raises(ValueError, match="duplicate hinge"):
        make_flexure_triangles(fixture(), g)


def test_flexure_rejects_tilted_print_face():
    tiles = fixture()
    tiles[0, 0, 2] += 1
    with pytest.raises(ValueError, match="tilted top face"):
        make_flexure_triangles(tiles, SimpleNamespace(hinges=[hinge()]))


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
        state, panel_size_mm=40, panel_thickness_mm=2,
    )
    assert raw.startswith(b"solid onestring_t2d_dual_flexure_mm")
    assert raw.count(b"facet normal") == 2 * 12 + 20
    assert metrics["t2d_face_count"] == 44
    assert metrics["flexure_top_count"] == 1
    assert metrics["t2d_fabrication_unit"].startswith("mm")
    np.testing.assert_allclose(t2d.vertices, tiles)
    assert t2d.metrics == {}


def test_invalid_flexure_parameters_rejected():
    with pytest.raises(ValueError, match="smaller than skin"):
        make_flexure_triangles(
            fixture(), SimpleNamespace(hinges=[hinge()]),
            FlexureParameters(skin_mm=.2, embed_mm=.2),
        )
