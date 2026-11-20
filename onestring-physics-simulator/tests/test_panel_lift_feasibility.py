"""UV-valid motion must not turn a feasible curved 3D panel non-convex."""
import numpy as np
from onestring_physics import zhang2021_panel_optimizer as optimizer


def test_interior_only_optimization_preserves_lifted_convexity(monkeypatch):
    uv = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    xyz = np.column_stack((uv, [0.,10.,0.,1.]))
    faces = np.array([[0,1,2],[0,2,3]])
    lift = optimizer.ChartLift(uv, faces, xyz, faces)
    start = uv.copy(); start[1] = [.1,0.]
    quads = np.array([[0,1,2,3]])
    X0 = lift.lift(start, lift.locate(start))
    assert optimizer._lift_is_valid(X0, quads)
    # This target stays inside the valid UV square but creates a concave 3D quad.
    assert not optimizer._lift_is_valid(xyz, quads)
    def drive_toward_bad_3d_target(X, Q, L, w):
        delta = X[1, 0] - xyz[1, 0]
        gradient = np.zeros_like(X); gradient[1, 0] = 2*delta
        return float(delta * delta), gradient, {}
    monkeypatch.setattr(optimizer, '_quad_terms', drive_toward_bad_3d_target)
    U, X, report = optimizer.optimize_panel_quality(
        lift, start, quads, np.array([True,False,True,True]),
        weights=optimizer.PanelOptimizerWeights(barrier=0., smooth=0.), max_iter=100)
    assert optimizer._lift_is_valid(X, quads)
    assert np.all(optimizer.quad_validity(U, quads)[0] > 0)
    assert report['panel_optimizer_energy_after'] < report['panel_optimizer_energy_before']
    assert np.array_equal(U[[0,2,3]], start[[0,2,3]])
