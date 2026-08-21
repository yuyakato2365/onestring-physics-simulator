"""Native Grid-OptCuts subprocess bridge.

This path intentionally does NOT call ``run_official_optcuts`` because the
official OneString bridge recenters/rescales UV area after OptCuts exits.  Such a
post-scale would destroy the fixed lattice spacing used by native Grid-OptCuts.
The native bridge therefore invokes the same executable directly and imports the
raw UV coordinates exactly as written by the patched C++ code.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Iterator
import uuid

import numpy as np

from .optcuts_backend import (
    OptCutsConfig,
    OptCutsError,
    OptCutsOutputError,
    OptCutsResult,
    OptCutsUnavailableError,
    _boundary_loops,
    _find_output_obj,
    _infer_optcuts_root,
    _read_obj_with_uv,
    _sha256,
    _signed_area,
    _triangle_differential_metrics,
    _write_triangle_obj,
    resolve_optcuts_executable,
)


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

    if not (4.0 < float(config.distortion_bound) < float("inf")):
        raise ValueError("OptCuts distortion_bound must be > 4")
    if not (0.0 < float(config.lambda_init) < 1.0):
        raise ValueError("OptCuts lambda_init must satisfy 0 < lambda_init < 1")
    if int(config.method_type) not in {0, 1, 2, 3}:
        raise ValueError("OptCuts method_type must be 0, 1, 2, or 3")

    # Native V1 uses exact coincident seam copies on the fabrication lattice.
    # The original scaffold assumes separated UV boundaries, therefore V1 uses
    # OptCuts' orientation-preserving non-scaffold solve.  Triangle inversion is
    # still rejected by OptCuts and audited again after import.
    cfg = replace(config, use_bijectivity=False, initial_cut_option=0)
    executable = resolve_optcuts_executable(cfg.executable)
    root = _infer_optcuts_root(executable)
    tag = f"onestring_grid_{uuid.uuid4().hex[:12]}"
    started = time.time()
    env_values = {
        "ONESTRING_OPTCUTS_GRID_NATIVE": "1",
        "ONESTRING_OPTCUTS_GRID_H": f"{h:.17g}",
        "ONESTRING_OPTCUTS_GRID_ANGLE_RAD": f"{angle:.17g}",
        "ONESTRING_OPTCUTS_GRID_PHASE_U": f"{float(phase_u):.17g}",
        "ONESTRING_OPTCUTS_GRID_PHASE_V": f"{float(phase_v):.17g}",
        "ONESTRING_OPTCUTS_GRID_MAX_SNAP_STEPS": f"{max_snap:.17g}",
    }

    temp_ctx = tempfile.TemporaryDirectory(prefix="onestring_grid_optcuts_")
    try:
        temp_dir = Path(temp_ctx.name)
        input_obj = temp_dir / "surface.obj"
        _write_triangle_obj(input_obj, surface_vertices, surface_faces)
        command = [
            str(executable),
            "100",
            str(input_obj.resolve()),
            f"{float(cfg.lambda_init):.17g}",
            "1",
            str(int(cfg.method_type)),
            f"{float(cfg.distortion_bound):.17g}",
            "0",  # no scaffold in native V1
            "0",  # grid-aware one-point initial cut
            tag,
        ]
        try:
            with _temporary_env(env_values):
                completed = subprocess.run(
                    command,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=float(cfg.timeout_seconds),
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise OptCutsError(f"Native Grid-OptCuts timed out after {cfg.timeout_seconds:g} s") from exc
        except OSError as exc:
            raise OptCutsUnavailableError(f"Failed to execute {executable}: {exc}") from exc

        combined = completed.stdout + "\n" + completed.stderr
        if completed.returncode != 0:
            tail = "\n".join(combined.splitlines()[-60:])
            raise OptCutsError(
                f"Native Grid-OptCuts exited with code {completed.returncode}.\nLast output:\n{tail}"
            )
        if "[ONESTRING-GRID] native_candidate_search enabled" not in combined:
            raise OptCutsError(
                "OPTCUTS_GRID_NATIVE_BINARY_NOT_PATCHED: selected OptCuts executable has no "
                "native Grid-OptCuts marker. Run `python scripts/setup_optcuts.py` to patch and rebuild it."
            )

        result_obj = _find_output_obj(root, tag, started)
        xyz, faces, uv, uv_faces = _read_obj_with_uv(result_obj)
        loops = _boundary_loops(uv_faces)
        if len(loops) != 1:
            raise OptCutsOutputError(
                "Native Grid-OptCuts currently requires exactly one UV boundary loop; "
                f"got {len(loops)}"
            )

        # Reflection preserves the lattice only when applied consistently to phase.
        # Prefer rejecting a globally reversed export over silently changing the
        # fabrication frame after the C++ search.
        if _signed_area(uv[loops[0]]) < 0.0:
            raise OptCutsOutputError(
                "OPTCUTS_GRID_NATIVE_REVERSED_UV: raw Grid-OptCuts UV orientation is reversed; "
                "refusing post-hoc reflection because it would change the configured grid frame."
            )

        differential = _triangle_differential_metrics(xyz, faces, uv, uv_faces)
        if int(differential.get("uv_triangle_flip_count", 0)) != 0:
            raise OptCutsOutputError(
                "OPTCUTS_GRID_NATIVE_FLIPPED_UV: patched OptCuts returned flipped triangles; "
                f"count={differential['uv_triangle_flip_count']}"
            )
        metrics: dict[str, object] = {
            "parameterization_backend_name": "optcuts_native_grid_constrained",
            "parameterization_backend_version": "official_OptCuts_plus_OneString_native_grid_v1",
            "parameterization_method": "optcuts_grid",
            "omega_parameterization_mode": "optcuts_grid",
            "flattening_backend": "optcuts_native_grid_cpp",
            "optcuts_executable": str(executable),
            "optcuts_executable_sha256": _sha256(executable),
            "optcuts_working_root": str(root),
            "optcuts_output_obj": str(result_obj),
            "optcuts_command_mode": 100,
            "optcuts_lambda_init": float(cfg.lambda_init),
            "optcuts_method_type": int(cfg.method_type),
            "optcuts_distortion_bound": float(cfg.distortion_bound),
            "optcuts_use_bijectivity": False,
            "optcuts_initial_cut_option": 0,
            "optcuts_runtime_seconds": float(time.time() - started),
            "optcuts_stdout_tail": "\n".join(completed.stdout.splitlines()[-80:]),
            "optcuts_stderr_tail": "\n".join(completed.stderr.splitlines()[-80:]),
            "optcuts_uv_area_normalization_scale": 1.0,
            "optcuts_uv_post_scale_applied": False,
            "optcuts_uv_boundary_loop_count": int(len(loops)),
            "optcuts_uv_boundary_vertex_count": int(len(loops[0])),
            "surface_vertex_count_after_cut_export": int(len(xyz)),
            "surface_triangle_count_after_cut_export": int(len(faces)),
            "uv_vertex_count_after_cut": int(len(uv)),
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
                "(H, V, H-V, V-H); selected seam vertices remain hard-fixed on the lattice"
            ),
            "optcuts_grid_topology_resolution_limit": (
                "candidate cuts follow existing source triangle-mesh edges; arbitrary "
                "grid-line/triangle intersection insertion is not implemented in native V1"
            ),
            **differential,
        }
        return OptCutsResult(
            surface_vertices_3d=xyz,
            surface_faces=faces,
            uv_vertices_2d=uv,
            uv_faces=uv_faces,
            boundary_loops=loops,
            output_obj=str(result_obj),
            metrics=metrics,
        )
    finally:
        temp_ctx.cleanup()


__all__ = ["run_native_grid_optcuts"]
