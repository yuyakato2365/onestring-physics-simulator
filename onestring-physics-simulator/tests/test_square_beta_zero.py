"""beta=0 must be the exact, differentiable mean-deviation LSE limit."""
import numpy as np
import pytest

from onestring_physics.optcuts_test_k3d_augmented_lagrangian_patch import (
    _square_energy_and_gradient,
)


def _two_skewed_quads():
    points=np.array([
        [0.,0.,0.], [1.,0.,0.], [1.15,1.,0.], [0.,1.,0.],
        [3.,0.,0.], [4.,0.,0.], [4.5,1.2,0.], [3.,1.,0.],
    ],float)
    faces=np.array([[0,1,2,3],[4,5,6,7]],int)
    return points,faces,np.array([1.0,1.0])


def test_beta_zero_is_the_exact_mean_deviation_limit(monkeypatch):
    verts,faces,scales=_two_skewed_quads()
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4.0")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","0")
    zero,grad,diag=_square_energy_and_gradient(verts,faces,scales,3.0)
    assert np.isfinite(zero)
    assert np.isfinite(grad).all()
    assert diag["square_tail_beta"]==0.
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","1e-6")
    near,near_grad,near_diag=_square_energy_and_gradient(
        verts,faces,scales,3.0
    )
    assert zero==pytest.approx(near,rel=1e-5,abs=1e-8)
    assert np.allclose(grad,near_grad,rtol=1e-4,atol=1e-6)


def test_beta_zero_analytic_gradient_matches_finite_difference(monkeypatch):
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","0")
    x,f,scales=_two_skewed_quads()
    energy,grad,_=_square_energy_and_gradient(x,f,scales,3.)
    eps=1e-6
    for vid,axis in ((0,0),(1,2),(2,1),(6,0),(7,1)):
        plus=x.copy();minus=x.copy()
        plus[vid,axis]+=eps;minus[vid,axis]-=eps
        eplus=_square_energy_and_gradient(plus,f,scales,3.)[0]
        eminus=_square_energy_and_gradient(minus,f,scales,3.)[0]
        assert grad[vid,axis]==pytest.approx(
            (eplus-eminus)/(2*eps),rel=2e-4,abs=2e-5
        )


def test_alpha_zero_disables_the_extra_tail_term_even_for_beta_zero(monkeypatch):
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_BETA","0")
    x,f,scales=_two_skewed_quads()
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","0")
    without,_,_= _square_energy_and_gradient(x,f,scales,3.)
    monkeypatch.setenv("ONESTRING_K3D_SQUARE_TAIL_ALPHA","4")
    with_tail,_,_= _square_energy_and_gradient(x,f,scales,3.)
    assert with_tail>without>0
