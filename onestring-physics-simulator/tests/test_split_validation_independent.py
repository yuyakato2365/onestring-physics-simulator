"""No-native-BFF tests for the independent Split validator."""
import numpy as np
import pytest

from onestring_physics.reference_bff import validate_reference_mesh
from onestring_physics.split_validation.core import (
    area_jacobians, clip_source_to_box, grid_rectangles,
    lobe_tip_corners, scale_span, split_tile_ids, tile_jacobians,
)


def flat_square():
    xyz = np.array([[0,0,0],[2,0,0],[2,2,0],[0,2,0]], float)
    uv = xyz[:,:2].copy()
    faces = np.array([[0,1,2],[0,2,3]],int)
    return xyz, uv, faces


def test_uniform_scale_is_area_span_invariant():
    xyz, uv, tri = flat_square()
    for scale in (0.1, 1., 10.):
        values = area_jacobians(xyz, uv*scale, tri)
        assert scale_span(values) == pytest.approx(1.)
        assert scale_span(values,units="linear_ratio") == pytest.approx(1.)


def test_area_and_linear_threshold_are_different_units():
    assert scale_span([1,4],units="area_ratio") == pytest.approx(4)
    assert scale_span([1,4],units="linear_ratio") == pytest.approx(2)
    with pytest.raises(ValueError,match="units"):
        scale_span([1,4],units="ratio_unknown")


def test_parent_not_bounded_by_product_of_children():
    # A mathematically explicit counterexample to the handoff's disputed claim.
    a, b = np.ones(10), np.ones(10)*100
    assert scale_span(a) == scale_span(b) == 1.
    assert scale_span(np.r_[a,b]) == 100.


def test_negative_flip_and_degenerate_are_rejected():
    xyz, uv, tri = flat_square()
    broken = uv.copy()
    broken[2] = broken[1]
    with pytest.raises(ValueError,match="Degenerate"):
        area_jacobians(xyz, broken, tri)
    folded = tri.copy()
    folded[1] = folded[1,::-1]
    with pytest.raises(ValueError,match="Folded"):
        area_jacobians(xyz, uv, folded)


def test_clip_half_plane_is_disk_with_area_unchanged():
    xyz, uv, tri = flat_square()
    child_xyz, child_uv, child_tri = clip_source_to_box(
        xyz, uv, tri, (0,0,1,2))
    check = validate_reference_mesh(child_xyz, child_tri)
    assert check["disk_topology"]
    assert scale_span(area_jacobians(child_xyz, child_uv, child_tri)) == pytest.approx(1)


def test_grid_is_centered_and_has_no_incomplete_cells():
    u = np.array([[0,0],[1.009,0],[1.009,1],[0,1]],float)
    squares, meta = grid_rectangles(u,14,"fit_center_min_side")
    assert len(squares) == 196
    assert meta["pitch"] == pytest.approx(1/14)
    assert squares[0][2] == pytest.approx(.0045)
    assert squares[0][3] == pytest.approx(0)


def test_grid_cut_maintains_tile_ids():
    uv = np.array([[0,0],[2,0],[2,2],[0,2]],float)
    tiles, _ = grid_rectangles(uv,4)
    left,right = split_tile_ids(tiles,list(range(16)),"col",1.)
    assert len(left)==len(right)==8
    assert sorted(left+right) == list(range(16))
    with pytest.raises(ValueError, match="crosses"):
        split_tile_ids(tiles,list(range(16)),"col",1.1)


def test_tile_weighted_jacobian_preserves_full_coverage():
    xyz, uv, faces = flat_square()
    tiles, _ = grid_rectangles(uv, 2)
    values, coverage = tile_jacobians(uv,np.ones(len(faces)),faces,tiles)
    assert len(values)==4
    assert np.allclose(values,1.)
    assert np.allclose(coverage,1.)
    with pytest.raises(ValueError,match="Unmeasured"):
        tile_jacobians(uv,np.ones(1),faces[:1],tiles)


def test_lobe_tip_detector_is_deterministic_on_rectangle():
    xyz, uv, tri=flat_square()
    one=lobe_tip_corners(xyz,tri)
    two=lobe_tip_corners(xyz,tri)
    assert one == two
    assert len(one)==4
    assert len(set(one))==4
