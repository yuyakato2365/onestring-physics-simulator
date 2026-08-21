"""Native Grid-OptCuts execution wrapper.

Unlike the retired Python post-processing prototypes, this module activates the
OneString patch inside the OptCuts C++ candidate search itself.  The subprocess
receives the fixed fabrication lattice parameters through environment variables.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
import os
from typing import Iterator

import numpy as np

from .optcuts_backend import OptCutsConfig, OptCutsError, OptCutsResult, run_official_optcuts


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_native_grid_optcuts(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    config: OptCutsConfig,
    *,
    grid_h: float,
    angle_degrees: float = 0.0,
    phase_u: float = 0.0,
    phase_v: float = 0.0,
    max_snap_steps: float = 2.0,
) -> OptCutsResult:
    h = float(grid_h)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"Grid-OptCuts requires positive finite tile_size; got {grid_h!r}")
    angle = math.radians(float(angle_degrees))
    max_snap = float(max_snap_steps)
    if not math.isfinite(max_snap) or max_snap < 0.5:
        raise ValueError("Grid-OptCuts max_snap_steps must be >= 0.5")

    # Native V1 uses exact coincident UV seam copies on the fabrication grid.
    # The old OptCuts scaffold expects separated boundary copies, so candidate
    # and global optimization use the orientation-preserving non-scaffold path.
    # Per-triangle flip checks remain active in OptCuts and the Python bridge.
    grid_config = replace(config, use_bijectivity=False, initial_cut_option=0)
    env = {
        "ONESTRING_OPTCUTS_GRID_NATIVE": "1",
        "ONESTRING_OPTCUTS_GRID_H": f"{h:.17g}",
        "ONESTRING_OPTCUTS_GRID_ANGLE_RAD": f"{angle:.17g}",
        "ONESTRING_OPTCUTS_GRID_PHASE_U": f"{float(phase_u):.17g}",
        "ONESTRING_OPTCUTS_GRID_PHASE_V": f"{float(phase_v):.17g}",
        "ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS": f"{max_snap:.17g}",
    }
    with _temporary_env(env):
        result = run_official_optcuts(surface_vertices, surface_faces, grid_config)

    stdout_tail = str(result.metrics.get("optcuts_stdout_tail", ""))
    if "[ONESTRING-GRID] native_candidate_search enabled" not in stdout_tail:
        raise OptCutsError(
            "OPTCUTS_GRID_NATIVE_BINARY_NOT_PATCHED: the selected OptCuts executable did not "
            "report the native Grid-OptCuts marker. Re-run `python scripts/setup_optcuts.py` "
            "to patch and rebuild third_party/OptCuts before using optcuts_grid."
        )

    result.metrics.update({
        "parameterization_backend_name": "optcuts_native_grid_constrained",
        "parameterization_method": "optcuts_grid",
        "omega_parameterization_mode": "optcuts_grid",
        "optcuts_grid_native": True,
        "optcuts_grid_postprocess_used": False,
        "optcuts_grid_candidate_search_modified": True,
        "optcuts_grid_spacing": h,
        "optcuts_grid_angle_degrees": float(angle_degrees),
        "optcuts_grid_phase_u": float(phase_u),
        "optcuts_grid_phase_v": float(phase_v),
        "grid_phase_u": float(phase_u),
        "grid_phase_v": float(phase_v),
        "optcuts_grid_max_snap_steps": max_snap,
        "optcuts_grid_merge_enabled": False,
        "optcuts_grid_initial_cut_option_forced": 0,
        "optcuts_grid_bijectivity_scaffold_enabled": False,
        "optcuts_grid_constraint_model": (
            "OptCuts split candidate search restricted to fixed-h orthogonal lattice embeddings "
            "(H, V, H-V, V-H); selected seam vertices remain fixed on the lattice"
        ),
        "optcuts_grid_topology_resolution_limit": (
            "candidate cuts still follow existing source triangle-mesh edges; arbitrary "
            "grid-line/triangle intersection vertex insertion is not implemented in native V1"
        ),
    })
    return result


__all__ = ["run_native_grid_optcuts"]
