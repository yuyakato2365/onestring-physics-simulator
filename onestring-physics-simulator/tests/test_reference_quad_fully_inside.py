"""Reference M2D crop must keep ONLY fully-contained quads (Paper §4.1)."""
import numpy as np

from onestring_physics.onestring_pipeline import _reference_quad_fully_inside


UNIT=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])


def test_fully_covered_quad_in_convex_omega_is_kept():
    square=np.array([[-1.,-1.],[2.,-1.],[2.,2.],[-1.,2.]])
    assert _reference_quad_fully_inside(UNIT,square)


def test_quad_crossing_straight_boundary_is_rejected():
    boundary=np.array([[-1.,-1.],[.5,-1.],[.5,2.],[-1.,2.]])
    assert not _reference_quad_fully_inside(UNIT,boundary)


def test_concave_omega_notch_touches_quad_side_without_proper_edge_crossing():
    # Entire top-row polygon boundary makes a narrow inward notch:
    # (0.4,1.0) -> (0.5,0.9) -> (0.6,1.0).
    # ALL four corners of UNIT are inside. No boundary crosses a quad
    # edge properly; two boundary vertices TOUCH that edge instead.
    # But the triangular 0.01 area notch is outside Omega.
    boundary=np.array([
        [-1.,-1.],[2.,-1.],[2.,2.],[.6,2.],[.6,1.],
        [.5,.9],[.4,1.],[.4,2.],[-1.,2.],
    ])
    assert not _reference_quad_fully_inside(UNIT,boundary)


def test_edge_coincident_with_outer_boundary_is_allowed():
    omega=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    assert _reference_quad_fully_inside(UNIT,omega)


def test_boundary_touch_only_does_not_reject_interior_quad():
    # Touching one corner is a legal boundary contact under closed-domain inclusion.
    omega=np.array([[0.,0.],[2.,0.],[2.,2.],[0.,2.]])
    assert _reference_quad_fully_inside(UNIT,omega)
