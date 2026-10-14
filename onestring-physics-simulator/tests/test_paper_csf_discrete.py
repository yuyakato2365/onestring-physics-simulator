"""Synthetic tests for optional M2D-grid vertex log-Jacobian interpolation.

These tests do not establish that the paper authors used this discretization.
"""
import numpy as np
import pytest
from onestring_physics.paper_csf_discrete import interpolate_jacobian_to_grid_vertices


def _chart():
    uv = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.5,.5]])
    faces = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    return uv,faces


def test_uniform_conformal_scale_stays_uniform_on_all_grid_vertices():
    uv, faces = _chart()
    lo,hi,d = interpolate_jacobian_to_grid_vertices(
        uv,faces,np.full(len(faces),9.),
        uv[:4],np.array([[0,1,2,3]])
    )
    np.testing.assert_allclose(lo,[9.],atol=1e-12)
    np.testing.assert_allclose(hi,[9.],atol=1e-12)
    assert d["max_quad_linear_csf"] == pytest.approx(1.)
    assert d["m2d_samples"] == 4


def test_variable_field_uses_finite_element_vertex_sampling_not_triangle_extrema():
    uv, faces = _chart()
    lo,hi,d = interpolate_jacobian_to_grid_vertices(
        uv,faces,np.array([1.,1.,16.,16.]),
        uv[:4],np.array([[0,1,2,3]])
    )
    # Source vertices shared between J=1 and J=16 receive geometric
    # means under the UV-area-weighted log(J) interpolation.
    assert lo[0] == pytest.approx(1.)
    assert hi[0] == pytest.approx(16.)
    assert d["max_quad_linear_csf"] == pytest.approx(4.)
    assert d["min_triangle_barycentric"] >= -1e-9


def test_outside_m2d_vertices_are_not_silently_clamped_to_boundary():
    uv,faces = _chart()
    quads=np.array([[0,1,2,3]],int)
    shifted=uv[:4]+np.array([3.,0.])
    with pytest.raises(ValueError,match="not covered"):
        interpolate_jacobian_to_grid_vertices(
            uv,faces,np.ones(len(faces)),shifted,quads
        )
