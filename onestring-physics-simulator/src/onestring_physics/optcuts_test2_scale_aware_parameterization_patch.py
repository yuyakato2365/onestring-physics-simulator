"""Enable the internally modified OptCuts seam objective for ``optcuts_test2``.

Unlike the previous temporary implementation, this module does not run OptCuts
multiple times and choose a completed result in Python.  It patches/rebuilds the
local OptCuts C++ source once, then the *original OptCuts topology search itself*
evaluates OneString scale-factor improvement for every split/merge candidate.

The rebuilt binary is shared with the baselines, but the new term is gated by
``ONESTRING_OPTCUTS_INTERNAL_SCALE_ENABLED``.  Therefore ordinary ``optcuts`` and
``optcuts_test`` retain the upstream objective; only visible ``optcuts_test2``
enables the extra term.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig
from .optcuts_internal_scale_factor_patch import ensure_internal_scale_binary
from . import optcuts_pipeline_patch as optcuts_pipeline


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == "2"


def _hard_scale_audit(result: Any, bound: float) -> tuple[float, bool]:
    """Audit the finished OptCuts map with the same UV->surface scale convention.

    ``per_triangle_sigma2`` is sigma_min(surface->UV), so the local
    sigma_max(UV->surface) is its reciprocal.  The max/min ratio is therefore
    identical to max(sigma2)/min(sigma2), after removing global similarity scale.
    """
    sigma2 = np.asarray(result.metrics.get("per_triangle_sigma2", []), dtype=float)
    sigma2 = sigma2[np.isfinite(sigma2) & (sigma2 > 1.0e-15)]
    if len(sigma2) == 0:
        return float("inf"), False
    scale_range = float(np.max(sigma2) / np.min(sigma2))
    return scale_range, bool(scale_range <= float(bound) + 1.0e-9)


def install_optcuts_test2_scale_aware_parameterization_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_test2_scale_aware_parameterization_installed", False):
        return

    original_run = optcuts_pipeline.run_official_optcuts
    if getattr(original_run, "_onestring_internal_scale_dispatch", False):
        pipeline._onestring_test2_scale_aware_parameterization_installed = True
        return

    def run_dispatch(surface_vertices, surface_faces, config=None):
        cfg = config if config is not None else OptCutsConfig()
        if _is_test2():
            os.environ["ONESTRING_OPTCUTS_INTERNAL_SCALE_ENABLED"] = "1"
            os.environ.setdefault("ONESTRING_OPTCUTS_INTERNAL_SCALE_BOUND", "2.0")
            os.environ.setdefault("ONESTRING_OPTCUTS_INTERNAL_SCALE_WEIGHT", "40.0")
            requested = Path(cfg.executable).expanduser() if cfg.executable else None
            binary = ensure_internal_scale_binary(requested)
            cfg = replace(cfg, executable=str(binary))
            result = original_run(surface_vertices, surface_faces, cfg)
            bound = float(os.environ.get("ONESTRING_OPTCUTS_INTERNAL_SCALE_BOUND", "2.0"))
            scale_range, hard_feasible = _hard_scale_audit(result, bound)
            result.metrics.update(
                {
                    "optcuts_internal_scale_factor_enabled": True,
                    "optcuts_internal_scale_factor_model": (
                        "upstream OptCuts split/merge score + soft decrease of area-weighted "
                        "OneString log-scale-band violation"
                    ),
                    "optcuts_internal_scale_factor_bound": bound,
                    "optcuts_internal_scale_factor_weight": float(
                        os.environ.get("ONESTRING_OPTCUTS_INTERNAL_SCALE_WEIGHT", "40.0")
                    ),
                    "optcuts_internal_scale_factor_source_patched": True,
                    "optcuts_outer_multi_run_selector_used": False,
                    "optcuts_internal_scale_factor_final_range": scale_range,
                    "optcuts_internal_scale_factor_final_hard_feasible": hard_feasible,
                }
            )
            print(
                "[OPTCUTS-INTERNAL-SCALE-FINAL] "
                f"range={scale_range:.9g} bound={bound:.9g} hard_feasible={hard_feasible}"
            )
            return result

        # The same rebuilt binary is safe for the baselines: the C++ term is a
        # no-op unless this environment flag is true.
        os.environ["ONESTRING_OPTCUTS_INTERNAL_SCALE_ENABLED"] = "0"
        return original_run(surface_vertices, surface_faces, cfg)

    run_dispatch._onestring_internal_scale_dispatch = True
    run_dispatch._onestring_original_run_official_optcuts = original_run
    optcuts_pipeline.run_official_optcuts = run_dispatch

    pipeline._onestring_test2_scale_aware_parameterization_installed = True
    print(
        "[OPTCUTS-TEST2-INTERNAL-SCALE-ROUTE] installed; "
        "test2 patches/rebuilds OptCuts C++ and changes its internal candidate score"
    )


__all__ = ["install_optcuts_test2_scale_aware_parameterization_patch"]
