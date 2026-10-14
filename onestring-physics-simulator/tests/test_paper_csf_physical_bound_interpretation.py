"""Check the units ambiguity explicitly rather than silently calling length limit 2 an area limit.

These are mathematical ideal-mechanism tests, not a reproduction of the
OneString authors' unpublished internal CSF discretization.
"""
import math
import numpy as np
from onestring_physics.paper_mesh_splitting import (
    _component_area_span,
    _component_conformal_scale_span,
)


def test_length_ratio_can_pass_two_while_area_ratio_fails_two():
    lo=np.array([1.0, 2.25])
    hi=lo.copy()
    c=np.array([0,1])
    linear=_component_conformal_scale_span(lo,hi,c)
    area=_component_area_span(lo,hi,c)
    assert abs(linear-1.5)<1e-12
    assert abs(area-2.25)<1e-12
    assert linear<=2.0 and area>2.0


def test_uniform_independent_piece_scaling_cannot_change_jacobian_span():
    lo=np.array([1.0,2.0,3.0])
    hi=np.array([1.1,2.3,3.4])
    c=np.arange(3)
    before=_component_conformal_scale_span(lo,hi,c)
    for scale in (0.5,0.8,1.,1.6,9.):
        after=_component_conformal_scale_span(lo/(scale**2),hi/(scale**2),c)
        np.testing.assert_allclose(after,before,rtol=1e-12,atol=1e-12)


def test_rotating_square_ideal_linkage_area_expansion_limit_two():
    # Toy ideal rotating-squares lattice: planar pitch cos(theta)+sin(theta)
    # relative to initially closed squares for theta in [0,pi/4].
    theta=np.linspace(0,math.pi/4,1001)
    length_pitch=np.cos(theta)+np.sin(theta)
    area_pitch=length_pitch**2
    assert length_pitch[0]==1.
    assert area_pitch.max()<=2.0+1e-12
    assert np.isclose(area_pitch[-1],2.0,atol=1e-12)
    assert np.isclose(length_pitch[-1],math.sqrt(2.),atol=1e-12)
