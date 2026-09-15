"""Performance defaults for the expensive ``optcuts_test`` stages.

This patch does not change the mathematical model.  It only injects bounded
iteration/candidate defaults before the existing K3D AL and kinematic K2D
wrappers read ``params``.

K3D:
- Augmented-Lagrangian outer iterations: 8 (was 16).

K2D kinematic solve:
- outer passes: 2 (was 4),
- least-squares max function evaluations per pass: 30 (was 70),
- collision candidates per pass: 800 (was 2500).

The kinematic stage is diagnostic/non-fatal, so an unfinished solve still
returns its best result and publishes feasibility metrics downstream.
"""
from __future__ import annotations

from typing import Any


def _set_param(params: Any, name: str, value: Any) -> None:
    try:
        setattr(params, name, value)
    except Exception:
        pass


def install_optcuts_test_performance_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_performance_patch_installed", False):
        return

    base_k3d = pipeline._optimize_k3d
    base_k2d = pipeline._optimize_k2d

    def optimize_k3d_fast_defaults(target: Any, mesh: Any, parameterization: Any, params: Any):
        if str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            _set_param(params, "k3d_al_outer_iterations", 8)
            print("[OPTCUTS-TEST-PERF] K3D AL outer_iterations=8")
        return base_k3d(target, mesh, parameterization, params)

    def optimize_k2d_fast_defaults(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        if str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            _set_param(params, "k2d_kinematic_outer_passes", 2)
            _set_param(params, "k2d_kinematic_max_nfev", 30)
            _set_param(params, "k2d_kinematic_max_collision_pairs", 800)
            print(
                "[OPTCUTS-TEST-PERF] K2D kinematic outer_passes=2 "
                "max_nfev=30 max_collision_pairs=800"
            )
        return base_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

    pipeline._optimize_k3d = optimize_k3d_fast_defaults
    pipeline._optimize_k2d = optimize_k2d_fast_defaults

    original = getattr(pipeline, "_original", None)
    if original is not None:
        # Do not overwrite the original module's implementation directly; the
        # outer pipeline wrappers will route through the functions above.
        pass

    pipeline._onestring_optcuts_test_performance_patch_installed = True


__all__ = ["install_optcuts_test_performance_patch"]
