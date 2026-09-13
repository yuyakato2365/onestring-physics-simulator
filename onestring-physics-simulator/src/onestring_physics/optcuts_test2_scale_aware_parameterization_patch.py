"""Inject OneString scale-aware selection into the official OptCuts call used by test2.

The important implementation detail is that we do *not* replace the existing
``optcuts_test`` S->Omega wrapper.  That wrapper has accumulated seam metadata,
grid-outline reparameterization, diagnostics, and compatibility layers.  Instead
we replace only the module-global ``run_official_optcuts`` function looked up by
``optcuts_pipeline_patch._build_optcuts_parameterization`` at runtime.

Thus the normal stack remains:
    official OptCuts result -> optcuts_test grid-outline reparameterization -> ...
with the sole test2 change that the first result is selected from several official
OptCuts solutions using seam length + distortion + OneString scale-factor score.
"""
from __future__ import annotations

import os
from typing import Any

from .optcuts_backend import OptCutsConfig
from . import optcuts_pipeline_patch as optcuts_pipeline
from .optcuts_scale_aware_selection_patch import run_scale_aware_optcuts


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == "2"


def install_optcuts_test2_scale_aware_parameterization_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_test2_scale_aware_parameterization_installed", False):
        return

    original_run = optcuts_pipeline.run_official_optcuts
    if getattr(original_run, "_onestring_scale_aware_dispatch", False):
        pipeline._onestring_test2_scale_aware_parameterization_installed = True
        return

    def run_dispatch(surface_vertices, surface_faces, config=None):
        cfg = config if config is not None else OptCutsConfig()
        if _is_test2():
            return run_scale_aware_optcuts(surface_vertices, surface_faces, cfg)
        return original_run(surface_vertices, surface_faces, cfg)

    run_dispatch._onestring_scale_aware_dispatch = True
    run_dispatch._onestring_original_run_official_optcuts = original_run
    optcuts_pipeline.run_official_optcuts = run_dispatch

    pipeline._onestring_test2_scale_aware_parameterization_installed = True
    print(
        "[OPTCUTS-TEST2-SCALE-AWARE-ROUTE] installed at official OptCuts call; "
        "ordinary optcuts/optcuts_test wrapper stack preserved"
    )


__all__ = ["install_optcuts_test2_scale_aware_parameterization_patch"]