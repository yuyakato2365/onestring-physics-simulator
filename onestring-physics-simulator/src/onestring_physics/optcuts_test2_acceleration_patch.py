"""Acceleration controls for the optcuts_test2 experimental variant.

The visible UI mode ``optcuts_test2`` is internally routed through the existing
``optcuts_test`` implementation so test1 remains bit-for-bit unchanged.  A
process environment flag selects this faster variant.

Test2 changes only runtime budgets:
- K3D Augmented Lagrangian outer iterations: 8 instead of 16.
- K2D kinematic outer passes: 2 instead of 4.
- K2D least-squares function evaluations: 40 instead of 70.
- K2D collision candidate cap: 900 instead of 2500.
- When the installed SciPy ``least_squares`` supports ``workers``, finite-
  difference residual evaluations use a ThreadPoolExecutor.

No constraint failure is hidden: the existing kinematic patch still publishes
loop-hinge/collision feasibility metrics and returns the best-effort result.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import os
from typing import Any


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "1").strip() == "2"


def install_optcuts_test2_acceleration_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test2_acceleration_installed", False):
        return

    # Inject mode-specific runtime budgets before K3D/K2D wrappers read params.
    base_k3d = pipeline._optimize_k3d
    base_k2d = pipeline._optimize_k2d

    def k3d_fast(target: Any, mesh: Any, parameterization: Any, params: Any):
        if _is_test2() and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            for key, value in {
                "k3d_al_outer_iterations": 8,
            }.items():
                try:
                    setattr(params, key, value)
                except Exception:
                    try:
                        object.__setattr__(params, key, value)
                    except Exception:
                        pass
            print("[OPTCUTS-TEST2-K3D-FAST] AL outer_iterations=8")
        return base_k3d(target, mesh, parameterization, params)

    def k2d_fast(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        if _is_test2() and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            values = {
                "k2d_kinematic_outer_passes": 2,
                "k2d_kinematic_max_nfev": 40,
                "k2d_kinematic_max_collision_pairs": 900,
            }
            for key, value in values.items():
                try:
                    setattr(params, key, value)
                except Exception:
                    try:
                        object.__setattr__(params, key, value)
                    except Exception:
                        pass
            print(
                "[OPTCUTS-TEST2-K2D-FAST] outer_passes=2 max_nfev=40 "
                "max_collision_pairs=900"
            )
        return base_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

    pipeline._optimize_k3d = k3d_fast
    pipeline._optimize_k2d = k2d_fast
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = k3d_fast
        original._optimize_k2d = k2d_fast
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = k3d_fast
            glb["_optimize_k2d"] = k2d_fast

    # SciPy added a workers hook to least_squares in newer versions.  Wrap it
    # once and only supply workers for test2; older SciPy versions continue with
    # the ordinary serial call without error.
    try:
        import scipy.optimize as spo

        original_ls = spo.least_squares
        if not getattr(original_ls, "_onestring_test2_workers_wrapper", False):
            signature = inspect.signature(original_ls)
            supports_workers = "workers" in signature.parameters
            max_workers = max(1, min(8, int(os.environ.get("ONESTRING_TEST2_WORKERS", os.cpu_count() or 1))))
            executor = ThreadPoolExecutor(max_workers=max_workers) if supports_workers else None

            def least_squares_with_test2_workers(*args: Any, **kwargs: Any):
                if _is_test2() and supports_workers and "workers" not in kwargs and executor is not None:
                    kwargs["workers"] = executor.map
                    if not getattr(least_squares_with_test2_workers, "_logged", False):
                        print(f"[OPTCUTS-TEST2-K2D-PARALLEL] least_squares workers={max_workers}")
                        least_squares_with_test2_workers._logged = True
                return original_ls(*args, **kwargs)

            least_squares_with_test2_workers._onestring_test2_workers_wrapper = True
            least_squares_with_test2_workers._logged = False
            spo.least_squares = least_squares_with_test2_workers
            print(
                "[OPTCUTS-TEST2-PARALLEL-SETUP] "
                f"least_squares_workers_supported={supports_workers} workers={max_workers if supports_workers else 1}"
            )
    except Exception as exc:
        print(f"[OPTCUTS-TEST2-PARALLEL-SETUP] unavailable: {type(exc).__name__}: {exc}")

    pipeline._onestring_optcuts_test2_acceleration_installed = True


__all__ = ["install_optcuts_test2_acceleration_patch"]
