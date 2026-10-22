"""Check the NON-paper, OptCuts-test launcher truly installs improved K3D."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from onestring_physics import optcuts_test_k3d_augmented_lagrangian_patch as al


def test_launcher_installs_k3d_al_before_ui_and_guards():
    source = (Path(__file__).parents[1] / "app_optcuts_core_20260916.py").read_text()
    tree = ast.parse(source)
    calls = [
        n.value.func.id
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
    ]
    assert "install_optcuts_test_k3d_augmented_lagrangian_patch" in calls
    assert (
        calls.index("install_optcuts_test_seam_metadata_bridge")
        < calls.index("install_optcuts_test_k3d_augmented_lagrangian_patch")
        < calls.index("_install_optcuts_test_k3d_ui_params")
        < calls.index("install_optcuts_k3d_validity_patch")
        < calls.index("install_optcuts_test2_acceleration_patch")
    )


def test_k3d_al_patch_is_not_merely_an_import(monkeypatch):
    """The wrapper must actually call the AL solver on optcuts_test."""
    invoked = []

    def fake_al(reference, faces, **kwargs):
        invoked.append({"reference": reference.copy(), "faces": faces.copy(), "kwargs": kwargs})
        return reference + np.array([0., 0., .001]), {
            "k3d_augmented_lagrangian_applied": True,
            "k3d_hard_planarity_max_distance_after": 0.,
            "k3d_shape_preserving_mode": True,
            "k3d_global_square_weight_effective": kwargs["square_weight"],
            "k3d_shape_preservation_weight": 100.,
        }

    monkeypatch.setattr(al, "_augmented_lagrangian_planarize", fake_al)
    monkeypatch.setenv("ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY", "1")
    vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                         [1., 1., 0.], [0., 1., 0.]])

    def ordinary(target, mesh, parameterization, params):
        state = SimpleNamespace(vertices=vertices.copy(),
                                faces=np.array([[0, 1, 2, 3]]), metrics={})
        return state, SimpleNamespace(objective="original", constraint_violation=.1)

    pipeline = SimpleNamespace(_optimize_k3d=ordinary, _original=None)
    al.install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline)
    params = SimpleNamespace(
        omega_parameterization_mode="optcuts_test", w_square=42.,
        k3d_al_anchor_weight=1., k3d_al_rho0=10.,
        k3d_al_outer_iterations=1, k3d_al_inner_iterations=1,
        k3d_al_relative_planarity_tolerance=1e-6,
        k3d_al_restoration_sweeps=1, k3d_al_final_polish_sweeps=1
    )
    result, report = pipeline._optimize_k3d(None, None, None, params)
    assert invoked and invoked[0]["kwargs"]["square_weight"] == 42.
    assert result.metrics["k3d_augmented_lagrangian_applied"] is True
    assert result.metrics["k3d_al_authoritative_result"] is True
    assert result.metrics["k3d_global_square_weight_effective"] == 42.
    assert "hard quad planarity" in report.objective

    # The optcuts_test AL never silently changes other Omega backends.
    params.omega_parameterization_mode = "optcuts"
    other, _ = pipeline._optimize_k3d(None, None, None, params)
    assert len(invoked) == 1
    np.testing.assert_allclose(other.vertices, vertices)
