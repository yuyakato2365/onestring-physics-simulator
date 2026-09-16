"""Authoritative seam transfer for native Grid-OptCuts.

Native Grid-OptCuts already knows the exact cut topology in C++ through ``cohE``.
Do not reconstruct that topology from the exported OBJ.  The canonical C++
patch writes ``finalResult_grid_seams.txt`` directly from the final cohesive
edges after validating every seam side against the fabrication lattice.

This module imports that sidecar, applies exactly the same rigid fabrication
frame transform/reflection as the UV result, and makes M2D consume only those
segments for native Grid-OptCuts runs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsOutputError
from . import optcuts_grid_native_backend as _backend
from . import optcuts_grid_native_pipeline_patch as _pipeline
from . import optcuts_grid_constrained_m2d_patch as _m2d


def _sidecar_path(output_obj: str | Path) -> Path:
    obj = Path(output_obj)
    name = obj.name
    if not name.endswith("_mesh.obj"):
        raise OptCutsOutputError(
            f"OPTCUTS_GRID_NATIVE_UNEXPECTED_OUTPUT_NAME: cannot derive seam sidecar from {obj}"
        )
    return obj.with_name(name[:-len("_mesh.obj")] + "_grid_seams.txt")


def _read_raw_seam_sidecar(path: Path) -> np.ndarray:
    if not path.is_file():
        raise OptCutsOutputError(
            "OPTCUTS_GRID_NATIVE_SEAM_SIDECAR_MISSING: C++ Grid-OptCuts did not export its "
            f"authoritative cohesive seams: {path}. Re-run scripts/setup_optcuts.py."
        )
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise OptCutsOutputError(
                    f"OPTCUTS_GRID_NATIVE_SEAM_SIDECAR_INVALID: {path}:{line_no} expected 4 values"
                )
            try:
                row = [float(x) for x in parts]
            except ValueError as exc:
                raise OptCutsOutputError(
                    f"OPTCUTS_GRID_NATIVE_SEAM_SIDECAR_INVALID: non-numeric row at {path}:{line_no}"
                ) from exc
            if not np.all(np.isfinite(row)):
                raise OptCutsOutputError(
                    f"OPTCUTS_GRID_NATIVE_SEAM_SIDECAR_INVALID: non-finite row at {path}:{line_no}"
                )
            rows.append(row)
    if not rows:
        return np.zeros((0, 2, 2), dtype=float)
    return np.asarray(rows, dtype=float).reshape(-1, 2, 2)


def _to_final_fabrication_segments(
    raw_segments: np.ndarray,
    *,
    angle_degrees: float,
    reflected_v: bool,
) -> np.ndarray:
    raw = np.asarray(raw_segments, dtype=float)
    if raw.size == 0:
        return np.zeros((0, 2, 2), dtype=float)
    flat = raw.reshape(-1, 2)
    framed = _backend._to_fabrication_frame(flat, math.radians(float(angle_degrees)))
    if bool(reflected_v):
        framed = np.asarray(framed, dtype=float).copy()
        framed[:, 1] *= -1.0
    return np.asarray(framed, dtype=float).reshape(-1, 2, 2)


def install_optcuts_grid_seam_sidecar_patch() -> None:
    if getattr(_pipeline, "_onestring_grid_seam_sidecar_patch_installed", False):
        return

    base_run = _pipeline.run_native_grid_optcuts
    fallback_internal_seams = _m2d._internal_seam_segments

    def run_with_authoritative_seams(*args: Any, **kwargs: Any):
        result = base_run(*args, **kwargs)
        sidecar = _sidecar_path(result.output_obj)
        raw_segments = _read_raw_seam_sidecar(sidecar)
        angle_degrees = float(
            result.metrics.get(
                "optcuts_grid_search_angle_degrees",
                kwargs.get("angle_degrees", 0.0),
            )
        )
        segments = _to_final_fabrication_segments(
            raw_segments,
            angle_degrees=angle_degrees,
            reflected_v=bool(result.metrics.get("optcuts_uv_fabrication_v_reflected", False)),
        )
        result.metrics["optcuts_grid_native_seam_sidecar"] = str(sidecar)
        result.metrics["optcuts_grid_native_seam_segment_count"] = int(len(segments))
        result.metrics["optcuts_grid_native_seam_segments"] = segments.tolist()
        result.metrics["optcuts_grid_native_seam_source"] = "cpp_cohE_sidecar"
        return result

    def authoritative_internal_seams(parameterization: Any) -> np.ndarray:
        metrics = dict(getattr(parameterization, "metrics", {}) or {})
        if bool(metrics.get("optcuts_grid_native", False)):
            if "optcuts_grid_native_seam_segments" not in metrics:
                raise RuntimeError(
                    "OPTCUTS_GRID_NATIVE_SEAM_SIDECAR_MISSING: native Grid-OptCuts M2D refuses "
                    "to infer seams from OBJ; the authoritative C++ cohE sidecar is missing"
                )
            arr = np.asarray(metrics["optcuts_grid_native_seam_segments"], dtype=float)
            if arr.size == 0:
                return np.zeros((0, 2, 2), dtype=float)
            if arr.ndim != 3 or arr.shape[1:] != (2, 2) or not np.all(np.isfinite(arr)):
                raise RuntimeError("OPTCUTS_GRID_NATIVE_SEAM_SIDECAR_INVALID")
            return arr
        return fallback_internal_seams(parameterization)

    _pipeline.run_native_grid_optcuts = run_with_authoritative_seams
    _m2d._internal_seam_segments = authoritative_internal_seams
    _pipeline._onestring_grid_seam_sidecar_patch_installed = True


__all__ = [
    "_read_raw_seam_sidecar",
    "_sidecar_path",
    "_to_final_fabrication_segments",
    "install_optcuts_grid_seam_sidecar_patch",
]
