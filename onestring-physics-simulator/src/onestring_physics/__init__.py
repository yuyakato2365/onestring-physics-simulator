"""Design and deployment tools for a OneString-inspired simulator."""
from __future__ import annotations

import importlib
from typing import Any

from . import onestring_pipeline as _onestring_pipeline
from .discrete_bff import install_discrete_bff
from .official_ceps import install_official_ceps


def _install_parameterization_backends(module: Any) -> None:
    """Install BFF and optional official CEPS after every pipeline load/reload."""
    # importlib.reload() preserves a module dictionary. Clear old installation
    # markers because the re-executed compatibility wrapper replaced the patch.
    module._DISCRETE_BFF_PATCH_INSTALLED = False
    module._OFFICIAL_CEPS_PATCH_INSTALLED = False
    install_discrete_bff(module)
    install_official_ceps(module)

    backend_build = module._build_surface_parameterization

    def with_rectangle_semantics(surface: Any, target: Any, grid: Any, params: Any) -> Any:
        result = backend_build(surface, target, grid, params)
        metrics = result.metrics
        if metrics.get("bff_backend_used") == "local_discrete_bff":
            metrics["omega_boundary_forced_rectangle"] = True
            metrics["omega_boundary_fixed"] = False
            metrics["omega_boundary_shape"] = "rectangular"
        elif metrics.get("ceps_backend_used") == "official_ceps_cli":
            metrics["omega_boundary_forced_rectangle"] = True
            metrics["omega_boundary_fixed"] = False
            metrics["omega_boundary_shape"] = "rectangular"
        return result

    module._build_surface_parameterization = with_rectangle_semantics
    module._original._build_surface_parameterization = with_rectangle_semantics
    module._ONESTRING_PARAMETERIZATION_BACKENDS_INSTALLED = True


_install_parameterization_backends(_onestring_pipeline)


def _install_reload_guard() -> None:
    """Reinstall the backends after the legacy Streamlit app reloads the wrapper."""
    if getattr(importlib, "_onestring_parameterization_reload_guard", False):
        return
    original_reload = importlib.reload

    def reload_and_reinstall(module: Any) -> Any:
        result = original_reload(module)
        if getattr(result, "__name__", "") == "onestring_physics.onestring_pipeline":
            _install_parameterization_backends(result)
        return result

    importlib.reload = reload_and_reinstall
    importlib._onestring_parameterization_reload_guard = True


def _install_streamlit_ceps_option() -> None:
    """Expose ``ceps`` in the compatibility app without rewriting the large app."""
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_ceps_selectbox_patch", False):
        return
    original_selectbox = st.selectbox

    def selectbox_with_ceps(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label == "Omega parameterization mode":
            if len(args) >= 2:
                options = list(args[1])
                if "ceps" not in options:
                    insertion = options.index("bff") + 1 if "bff" in options else 0
                    options.insert(insertion, "ceps")
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if "ceps" not in options:
                    insertion = options.index("bff") + 1 if "bff" in options else 0
                    options.insert(insertion, "ceps")
                kwargs = {**kwargs, "options": options}
        return original_selectbox(*args, **kwargs)

    st.selectbox = selectbox_with_ceps
    st._onestring_ceps_selectbox_patch = True


_install_reload_guard()
_install_streamlit_ceps_option()

from .design_optimizer import DesignParameters, DesignResult, optimize_design
from .input_shape import create_builtin_shape, load_target_shape, normalize_shape, sample_target_surface
from .onestring_pipeline import (
    ComputeConfig,
    DeploymentParameters,
    DeploymentResult,
    FlatTileLayout,
    OneStringDesignState,
    PipelineParameters,
    SurfaceParameterization,
    build_onestring_design,
    complexity_metrics,
    compute_backend_info,
    export_t2d_stl,
    gpu_self_test,
    inverse_map_uv_to_surface,
    nvidia_smi_probe,
    paper_consistency_report,
    run_simulator_gpu_benchmark,
    safe_capstan_friction,
    simulate_onestring_deployment,
)
from .physics_world import PhysicsParameters, PhysicsResult, PhysicsWorld, simulate_deployment
from .quad_grid import QuadGrid, create_quad_grid

__all__ = [
    "DesignParameters",
    "DesignResult",
    "ComputeConfig",
    "DeploymentParameters",
    "DeploymentResult",
    "FlatTileLayout",
    "OneStringDesignState",
    "PhysicsParameters",
    "PhysicsResult",
    "PhysicsWorld",
    "PipelineParameters",
    "QuadGrid",
    "SurfaceParameterization",
    "build_onestring_design",
    "complexity_metrics",
    "compute_backend_info",
    "export_t2d_stl",
    "create_builtin_shape",
    "create_quad_grid",
    "gpu_self_test",
    "inverse_map_uv_to_surface",
    "load_target_shape",
    "nvidia_smi_probe",
    "paper_consistency_report",
    "run_simulator_gpu_benchmark",
    "normalize_shape",
    "optimize_design",
    "sample_target_surface",
    "safe_capstan_friction",
    "simulate_onestring_deployment",
    "simulate_deployment",
]
