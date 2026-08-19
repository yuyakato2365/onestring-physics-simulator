"""Install CUDA-capable Large Steps mesh conditioning before bijective S -> Omega."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .large_steps_mesh_conditioning import (
    LargeStepsMeshConditioningConfig,
    condition_mesh_with_large_steps,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _streamlit_value(key: str, default: Any) -> Any:
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception:
        return default


def _setting(params: Any, attr: str, session_key: str, default: Any) -> Any:
    if hasattr(params, attr):
        return getattr(params, attr)
    return _streamlit_value(session_key, default)


def _make_streamlit_progress_callback():
    """Create an in-place detailed progress renderer when called from Streamlit."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return None
    except Exception:
        return None

    st.caption("Large Steps mesh conditioning progress")
    bar = st.progress(0.0, text="Large Steps: preparing device and mesh…")
    log_slot = st.empty()
    rows: list[str] = []

    def callback(info: dict[str, Any]) -> None:
        event = str(info.get("event", "iteration"))
        fraction = float(np.clip(info.get("fraction", 0.0), 0.0, 1.0))
        iteration = int(info.get("iteration", 0) or 0)
        maximum = int(info.get("max_iterations", 0) or 0)
        elapsed = float(info.get("elapsed_seconds", 0.0) or 0.0)

        if event == "setup":
            device_name = str(info.get("device_name", info.get("device", "unknown")))
            device = str(info.get("device", "unknown"))
            dtype = str(info.get("dtype", ""))
            line = (
                f"[setup] {device} / {device_name} / {dtype} | "
                f"V={int(info.get('vertex_count', 0))}, F={int(info.get('face_count', 0))}, "
                f"interior={int(info.get('interior_vertex_count', 0))} | "
                f"minAngle={float(info.get('minimum_angle_degrees', 0.0)):.3f}° "
                f"q05={float(info.get('triangle_quality_p05', 0.0)):.4f}"
            )
            text = f"Large Steps setup — {device_name}"
        elif event == "iteration":
            line = (
                f"[{iteration:4d}/{maximum}] "
                f"E={float(info.get('energy', 0.0)):.6g} | "
                f"minAngle={float(info.get('minimum_angle_degrees', 0.0)):.3f}° | "
                f"q05={float(info.get('triangle_quality_p05', 0.0)):.4f} | "
                f"edgeCV={float(info.get('edge_length_cv', 0.0)):.4f} | "
                f"accepted={int(info.get('accepted_steps', 0))} "
                f"rejected={int(info.get('rejected_steps', 0))} | "
                f"step={float(info.get('line_search_step_scale', 0.0)):.3g} | "
                f"CG(g/d)={int(info.get('cg_gradient_iterations', 0))}/"
                f"{int(info.get('cg_direction_iterations', 0))} | "
                f"{elapsed:.1f}s"
            )
            text = (
                f"Large Steps {iteration}/{maximum} — "
                f"min angle {float(info.get('minimum_angle_degrees', 0.0)):.2f}°, "
                f"q05 {float(info.get('triangle_quality_p05', 0.0)):.3f}"
            )
        elif event == "stalled":
            line = (
                f"[stalled {iteration}/{maximum}] no valid decreasing step | "
                f"accepted={int(info.get('accepted_steps', 0))} "
                f"rejected={int(info.get('rejected_steps', 0))} | {elapsed:.1f}s"
            )
            text = f"Large Steps stalled at {iteration}/{maximum}"
        elif event == "done":
            line = (
                f"[done] {str(info.get('device_name', info.get('device', '')))} | "
                f"iterations={iteration}/{maximum} | "
                f"minAngle={float(info.get('minimum_angle_degrees', 0.0)):.3f}° | "
                f"q05={float(info.get('triangle_quality_p05', 0.0)):.4f} | "
                f"rejected={int(info.get('rejected_steps', 0))} | "
                f"reason={str(info.get('termination_reason', 'done'))} | {elapsed:.1f}s"
            )
            text = f"Large Steps completed in {elapsed:.1f}s"
            fraction = 1.0
        else:
            line = f"[{event}] iteration={iteration}/{maximum} | {elapsed:.1f}s"
            text = f"Large Steps: {event}"

        rows.append(line)
        if len(rows) > 14:
            del rows[:-14]
        try:
            bar.progress(fraction, text=text)
            log_slot.code("\n".join(rows), language="text")
        except Exception:
            pass

    return callback


def install_large_steps_conditioning(pipeline_module: Any) -> None:
    """Wrap S -> Omega so sampled input meshes are conditioned first."""
    if getattr(pipeline_module, "_LARGE_STEPS_CONDITIONING_PATCH_INSTALLED", False):
        return
    legacy = pipeline_module._build_surface_parameterization

    def build(surface: Any, target: Any, grid: Any, params: Any) -> Any:
        mode = str(getattr(params, "omega_parameterization_mode", "bff"))
        sampled = str(getattr(surface, "kind", getattr(target, "kind", ""))) == "sampled"
        default_enabled = _env_bool("ONESTRING_LARGE_STEPS_CONDITIONING", True)
        enabled = bool(
            _setting(
                params,
                "large_steps_conditioning_enabled",
                "large_steps_conditioning_enabled",
                default_enabled,
            )
        )
        if mode != "bijective_free_boundary" or not sampled or not enabled:
            try:
                setattr(surface, "_large_steps_conditioning_applied", False)
            except Exception:
                pass
            return legacy(surface, target, grid, params)

        config = LargeStepsMeshConditioningConfig(
            enabled=True,
            lambda_=float(
                _setting(params, "large_steps_conditioning_lambda", "large_steps_conditioning_lambda", 10.0)
            ),
            max_iterations=int(
                _setting(params, "large_steps_conditioning_max_iterations", "large_steps_conditioning_max_iterations", 120)
            ),
            learning_rate=float(
                _setting(params, "large_steps_conditioning_learning_rate", "large_steps_conditioning_learning_rate", 0.06)
            ),
            quality_weight=float(
                _setting(params, "large_steps_conditioning_quality_weight", "large_steps_conditioning_quality_weight", 1.0)
            ),
            edge_uniformity_weight=float(
                _setting(params, "large_steps_conditioning_edge_weight", "large_steps_conditioning_edge_weight", 0.12)
            ),
            surface_normal_weight=float(
                _setting(params, "large_steps_conditioning_surface_weight", "large_steps_conditioning_surface_weight", 12.0)
            ),
            position_weight=float(
                _setting(params, "large_steps_conditioning_position_weight", "large_steps_conditioning_position_weight", 0.01)
            ),
            minimum_orientation_ratio=float(
                _setting(params, "large_steps_conditioning_min_orientation_ratio", "large_steps_conditioning_min_orientation_ratio", 0.03)
            ),
            project_to_original_surface=True,
            device=str(
                _setting(params, "large_steps_conditioning_device", "large_steps_conditioning_device", "auto")
            ),
            dtype=str(
                _setting(params, "large_steps_conditioning_dtype", "large_steps_conditioning_dtype", "auto")
            ),
            cg_tolerance=float(
                _setting(params, "large_steps_conditioning_cg_tolerance", "large_steps_conditioning_cg_tolerance", 1.0e-6)
            ),
            cg_max_iterations=int(
                _setting(params, "large_steps_conditioning_cg_max_iterations", "large_steps_conditioning_cg_max_iterations", 160)
            ),
            progress_log_every=int(
                _setting(params, "large_steps_conditioning_progress_log_every", "large_steps_conditioning_progress_log_every", 5)
            ),
        )

        conditioned_vertices, conditioning_metrics = condition_mesh_with_large_steps(
            np.asarray(surface.vertices, dtype=float),
            np.asarray(surface.faces, dtype=int)[:, :3],
            config,
            progress_callback=_make_streamlit_progress_callback(),
        )
        conditioned_surface = pipeline_module._original.SurfaceMesh(
            vertices=np.asarray(conditioned_vertices, dtype=float),
            faces=np.asarray(surface.faces, dtype=int).copy(),
            kind=str(getattr(surface, "kind", "sampled")),
        )
        try:
            setattr(surface, "_large_steps_conditioning_applied", True)
            setattr(surface, "_large_steps_conditioned_vertices", np.asarray(conditioned_vertices, dtype=float))
            setattr(surface, "_large_steps_conditioning_metrics", dict(conditioning_metrics))
        except Exception:
            pass

        result = legacy(conditioned_surface, target, grid, params)
        result.metrics.update(conditioning_metrics)
        result.metrics.update(
            {
                "large_steps_conditioning_stage": "S_input -> S_conditioned -> bijective_free_boundary -> Omega",
                "large_steps_conditioned_surface_used_for_parameterization": True,
                "large_steps_raw_surface_vertex_count": int(len(surface.vertices)),
                "large_steps_conditioned_surface_vertex_count": int(len(conditioned_vertices)),
                "large_steps_conditioned_surface_face_count": int(len(surface.faces)),
            }
        )
        return result

    pipeline_module._build_surface_parameterization = build
    pipeline_module._original._build_surface_parameterization = build
    pipeline_module._LARGE_STEPS_CONDITIONING_PATCH_INSTALLED = True


__all__ = ["install_large_steps_conditioning"]
