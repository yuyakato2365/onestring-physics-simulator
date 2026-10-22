"""Regression: neither choosing nor using an official BFF UV may depend on gauge."""
from pathlib import Path

import numpy as np
import pytest

from onestring_physics.bff_rectangle_selection import (
    whole_surface_area_span, _verify_selected_bff_csf_compatible,
)
from onestring_physics.reference_bff import (
    normalize_uv_and_compute_csf, strict_inverse_map_uv_to_surface,
    triangle_jacobian_diagnostics,
)


def square():
    xyz = np.array([[0.,0.,0.],[1.,0.,0.],
                    [1.,1.,0.],[0.,1.,0.]])
    uv = xyz[:,:2].copy()
    faces = np.array([[0,1,2],[0,2,3]],dtype=int)
    return xyz, uv, faces


@pytest.mark.parametrize("scale", [1e-10, 1e-6, 1., 1e6, 1e10])
@pytest.mark.parametrize("normalization", ["min_to_one_hypothesis_a", "none_unspecified"])
def test_official_bff_uniform_scale_does_not_change_degeneracy(
    scale, normalization
):
    xyz, uv, faces = square()
    chosen_uv = uv * scale
    assert whole_surface_area_span(xyz,chosen_uv,faces) == pytest.approx(1.)
    _verify_selected_bff_csf_compatible(xyz,faces,chosen_uv,normalization)
    transformed, diagnostic = normalize_uv_and_compute_csf(
        xyz,chosen_uv,faces,normalization=normalization,
    )
    assert diagnostic["uv_triangle_flip_count"] == 0
    assert diagnostic["uv_degenerate_triangle_count"] == 0
    if normalization == "min_to_one_hypothesis_a":
        np.testing.assert_allclose(transformed, uv, atol=1e-8)
    # The same inverse map must work in either similarity-scale gauge.
    query = np.array([.75,.25])*float(
        1. if normalization == "min_to_one_hypothesis_a" else scale
    )
    point, tri, bary, error = strict_inverse_map_uv_to_surface(
        query,transformed,faces,xyz,faces
    )
    np.testing.assert_allclose(point[:2],[.75,.25],atol=1e-7)
    assert error < 1e-7


@pytest.mark.parametrize("scale", [1e-8, 1., 1e8])
def test_degenerate_uv_stays_invalid_regardless_of_scale(scale):
    xyz,uv,faces=square()
    uv[2] = [2.,0.]
    metric = triangle_jacobian_diagnostics(xyz,uv*scale,faces)
    assert metric["uv_degenerate_triangle_count"] >= 1
    with pytest.raises(ValueError, match="degenerate"):
        whole_surface_area_span(xyz,uv*scale,faces)


@pytest.mark.parametrize("scale", [1e-9, 1., 1e9])
def test_real_foldover_remains_invalid_regardless_of_scale(scale):
    xyz,uv,faces=square()
    uv[2]=[-.2,.2]
    m=triangle_jacobian_diagnostics(xyz,uv*scale,faces)
    assert m["uv_triangle_flip_count"] == 1
    with pytest.raises(ValueError,match="orientation"):
        _verify_selected_bff_csf_compatible(
            xyz,faces,uv*scale,"min_to_one_hypothesis_a"
        )


def test_lobe_tip_uses_general_rectangular_bff_only_if_own_uv_invalid():
    # A successful Lilium lobe tip must keep the exact old result.
    code=(Path(__file__).parents[1]/
          "src/onestring_physics/onestring_pipeline.py").read_text()
    assert '"bff_lobe_tip_original_candidate_rejected": True' in code
    assert 'allow_nonconformal_rescue=False' in code
    assert 'whole_surface_area_span(' in code
