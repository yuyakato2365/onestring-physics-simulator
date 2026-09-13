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

from .optcuts_backend import OptCutsConfig
from .optcuts_internal_scale_factor_patch import ensure_internal_scale_binary
from . import optcuts_pipeline_patch as optcuts_pipeline


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == "2"


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
            result.metrics.update(
                {
                    "optcuts_internal_scale_factor_enabled": True,
                    "optcuts_internal_scale_factor_model": (
                        "upstream OptCuts split/merge score + soft decrease of area-weighted "
                        "OneString log-scale-band violation"
                    ),
                    "optcuts_internal_scale_factor_bound": float(
                        os.environ.get("ONESTRING_OPTCUTS_INTERNAL_SCALE_BOUND", "2.0")
                    ),
                    "optcuts_internal_scale_factor_weight": float(
                        os.environ.get("ONESTRING_OPTCUTS_INTERNAL_SCALE_WEIGHT", "40.0")
                    ),
                    "optcuts_internal_scale_factor_source_patched": True,
                    "optcuts_outer_multi_run_selector_used": False,
                }
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
