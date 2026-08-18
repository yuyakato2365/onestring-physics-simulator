"""Install Large Steps mesh conditioning before bijective S -> Omega."""
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
        )

        conditioned_vertices, conditioning_metrics = condition_mesh_with_large_steps(
            np.asarray(surface.vertices, dtype=float),
            np.asarray(surface.faces, dtype=int)[:, :3],
            config,
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
