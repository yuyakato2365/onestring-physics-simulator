"""Acceleration controls for the optcuts_test2 experimental variant.

The visible UI mode ``optcuts_test2`` is internally routed through the existing
``optcuts_test`` implementation so test1 remains unchanged. A process environment
flag selects this variant.

Test2 now changes only:
- K3D Augmented Lagrangian outer iterations: 8 instead of 16.
- When the installed SciPy ``least_squares`` supports ``workers``, finite-
  difference residual evaluations use a ThreadPoolExecutor.

K2D numerical budgets are deliberately identical to ``optcuts_test``:
- kinematic outer passes: 4
- least-squares max_nfev: 70
- collision candidate cap: 2500

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

    # Inject only the K3D AL budget change. K2D budgets intentionally remain
    # identical to optcuts_test so test2 does not trade constraint quality for
    # runtime. K2D acceleration is limited to SciPy workers when available.
    base_k3d = pipeline._optimize_k3d
    base_k2d = pipeline._optimize_k2d

    def k3d_fast(target: Any, mesh: Any, parameterization: Any, params: Any):
        if _is_test2() and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            try:
                setattr(params, "k3d_al_outer_iterations", 8)
            except Exception:
                try:
                    object.__setattr__(params, "k3d_al_outer_iterations", 8)
                except Exception:
                    pass
            print("[OPTCUTS-TEST2-K3D-FAST] AL outer_iterations=8")
        return base_k3d(target, mesh, parameterization, params)

    def k2d_full_budget(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        if _is_test2() and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            # Explicitly restore the same numerical budgets as optcuts_test in
            # case a reused params object still carries older test2 fast values.
            values = {
                "k2d_kinematic_outer_passes": 4,
                "k2d_kinematic_max_nfev": 70,
                "k2d_kinematic_max_collision_pairs": 2500,
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
                "[OPTCUTS-TEST2-K2D-FULL] outer_passes=4 max_nfev=70 "
                "max_collision_pairs=2500; no numerical budget reduction"
            )
        return base_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

    pipeline._optimize_k3d = k3d_fast
    pipeline._optimize_k2d = k2d_full_budget
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = k3d_fast
        original._optimize_k2d = k2d_full_budget
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = k3d_fast
            glb["_optimize_k2d"] = k2d_full_budget

    # SciPy added a workers hook to least_squares in newer versions. Wrap it
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
