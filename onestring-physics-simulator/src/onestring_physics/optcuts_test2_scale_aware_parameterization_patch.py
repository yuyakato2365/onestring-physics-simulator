"""Use the dedicated source-modified OptCuts binary for ``optcuts_test2``.

The OneString scale-factor term lives in OptCuts' own ``TriMesh::computeLocalLDec``
candidate objective.  This module does not rewrite C++ source and does not select
among completed OptCuts runs.  Build the modified binary once with
``python3 scripts/build_optcuts_onestring.py``; test2 then invokes that binary
directly while test1 continues to use the ordinary upstream OptCuts binary.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig, OptCutsUnavailableError
from . import optcuts_pipeline_patch as optcuts_pipeline


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip() == "2"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _onestring_binary() -> Path:
    root = _project_root()
    stamp = root / ".onestring_optcuts_binary"
    candidates: list[Path] = []
    if stamp.is_file():
        try:
            value = stamp.read_text(encoding="utf-8").strip()
            if value:
                candidates.append(Path(value).expanduser())
        except Exception:
            pass
    build = root / "third_party" / "OptCuts" / "build_onestring"
    candidates.extend([
        build / "OptCuts_bin",
        build / "OptCuts_bin.exe",
        build / "Release" / "OptCuts_bin",
        build / "Release" / "OptCuts_bin.exe",
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise OptCutsUnavailableError(
        "The source-modified OneString OptCuts binary is not built. Run:\n"
        "  python3 scripts/build_optcuts_onestring.py\n"
        "This applies the repository-tracked OptCuts source change and builds "
        "third_party/OptCuts/build_onestring/OptCuts_bin."
    )


def _hard_scale_audit(result: Any, bound: float) -> tuple[float, bool]:
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
    if getattr(original_run, "_onestring_source_modified_dispatch", False):
        pipeline._onestring_test2_scale_aware_parameterization_installed = True
        return

    def run_dispatch(surface_vertices, surface_faces, config=None):
        cfg = config if config is not None else OptCutsConfig()
        if _is_test2():
            bound = float(os.environ.get("ONESTRING_OPTCUTS_SCALE_BOUND", "2.0"))
            weight = float(os.environ.get("ONESTRING_OPTCUTS_SCALE_WEIGHT", "40.0"))
            os.environ["ONESTRING_OPTCUTS_SCALE_BOUND"] = str(bound)
            os.environ["ONESTRING_OPTCUTS_SCALE_WEIGHT"] = str(weight)
            binary = _onestring_binary()
            cfg = replace(cfg, executable=str(binary))
            print(
                "[OPTCUTS-TEST2-SOURCE-MODIFIED] "
                f"binary={binary} scale_bound={bound:g} scale_weight={weight:g}"
            )
            result = original_run(surface_vertices, surface_faces, cfg)
            scale_range, hard_feasible = _hard_scale_audit(result, bound)
            result.metrics.update({
                "optcuts_internal_scale_factor_enabled": True,
                "optcuts_internal_scale_factor_model": (
                    "OptCuts TriMesh::computeLocalLDec seam/distortion objective + "
                    "soft OneString scale-factor violation decrease"
                ),
                "optcuts_internal_scale_factor_bound": bound,
                "optcuts_internal_scale_factor_weight": weight,
                "optcuts_source_modified_binary": str(binary),
                "optcuts_runtime_source_patch_used": False,
                "optcuts_outer_multi_run_selector_used": False,
                "optcuts_internal_scale_factor_final_range": scale_range,
                "optcuts_internal_scale_factor_final_hard_feasible": hard_feasible,
            })
            print(
                "[OPTCUTS-INTERNAL-SCALE-FINAL] "
                f"range={scale_range:.9g} bound={bound:.9g} hard_feasible={hard_feasible}"
            )
            return result
        return original_run(surface_vertices, surface_faces, cfg)

    run_dispatch._onestring_source_modified_dispatch = True
    run_dispatch._onestring_original_run_official_optcuts = original_run
    optcuts_pipeline.run_official_optcuts = run_dispatch

    pipeline._onestring_test2_scale_aware_parameterization_installed = True
    print(
        "[OPTCUTS-TEST2-SOURCE-MODIFIED-ROUTE] installed; "
        "test2 uses a dedicated OptCuts binary built from the tracked source modification"
    )


__all__ = ["install_optcuts_test2_scale_aware_parameterization_patch"]
