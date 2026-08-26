"""Design and deployment tools for a OneString-inspired simulator."""
from __future__ import annotations

import importlib
from typing import Any

from . import onestring_pipeline as _onestring_pipeline
from . import official_ceps as _official_ceps
from .ceps_outer_boundary import install_ceps_outer_boundary
from .ceps_paired_output import install_ceps_paired_output
from .ceps_strict_adapter import install_ceps_strict_adapter
from .dynamic_free_boundary_v2 import (
    BijectiveFreeBoundaryConfig,
    bijective_free_boundary_parameterization,
    install_bijective_free_boundary,
)
from .discrete_bff import install_discrete_bff
from .fast_t3d_preview import install_fast_t3d_preview
from .large_steps_mesh_conditioning import (
    LargeStepsMeshConditioningConfig,
    condition_mesh_with_large_steps,
)
from .large_steps_pipeline_patch import install_large_steps_conditioning
from .official_ceps import install_official_ceps


install_ceps_paired_output(_official_ceps)
install_ceps_outer_boundary(_official_ceps)


def _install_parameterization_backends(module: Any) -> None:
    """Install parameterization backends and runtime acceleration patches."""
    module._DISCRETE_BFF_PATCH_INSTALLED = False
    module._BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED = False
    module._OFFICIAL_CEPS_PATCH_INSTALLED = False
    module._CEPS_STRICT_ADAPTER_INSTALLED = False
    module._FAST_T3D_PREVIEW_PATCH_INSTALLED = False
    module._LARGE_STEPS_CONDITIONING_PATCH_INSTALLED = False
    module._K2D_CORRESPONDENCE_ANIMATION_PATCH_INSTALLED = False
    install_discrete_bff(module)
    install_bijective_free_boundary(module)
    install_large_steps_conditioning(module)
    install_official_ceps(module)
    install_ceps_strict_adapter(module)
    install_fast_t3d_preview(module)

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
            metrics["ceps_continuous_surface_uv_vertices"] = (
                len(result.surface_vertices_3d) == len(result.uv_vertices_2d)
            )
            metrics["ceps_surface_uv_face_connectivity_equal"] = bool(
                getattr(result.surface_faces, "shape", None)
                == getattr(result.uv_faces, "shape", None)
                and (result.surface_faces == result.uv_faces).all()
            )
            metrics["ceps_paired_surface_uv_vertices"] = metrics[
                "ceps_continuous_surface_uv_vertices"
            ]
        return result

    module._build_surface_parameterization = with_rectangle_semantics
    module._original._build_surface_parameterization = with_rectangle_semantics
    module._ONESTRING_PARAMETERIZATION_BACKENDS_INSTALLED = True


_install_parameterization_backends(_onestring_pipeline)


def _install_reload_guard() -> None:
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


def _install_streamlit_parameterization_options() -> None:
    """Expose installed modes and the conditioned-S inspection stage."""
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_ceps_selectbox_patch", False):
        return
    original_selectbox = st.selectbox

    def selectbox_with_installed_modes(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label == "Omega parameterization mode":
            if len(args) >= 2:
                options = list(args[1])
                if "ceps" not in options:
                    insertion = options.index("bff") + 1 if "bff" in options else 0
                    options.insert(insertion, "ceps")
                if "bijective_free_boundary" not in options:
                    insertion = options.index("bff") + 1 if "bff" in options else 0
                    options.insert(insertion, "bijective_free_boundary")
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if "ceps" not in options:
                    insertion = options.index("bff") + 1 if "bff" in options else 0
                    options.insert(insertion, "ceps")
                if "bijective_free_boundary" not in options:
                    insertion = options.index("bff") + 1 if "bff" in options else 0
                    options.insert(insertion, "bijective_free_boundary")
                kwargs = {**kwargs, "options": options}
        elif label == "View stage":
            if len(args) >= 2:
                options = list(args[1])
                if "Conditioned S" not in options:
                    insertion = options.index("S") + 1 if "S" in options else 1
                    options.insert(insertion, "Conditioned S")
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if "Conditioned S" not in options:
                    insertion = options.index("S") + 1 if "S" in options else 1
                    options.insert(insertion, "Conditioned S")
                kwargs = {**kwargs, "options": options}

        selected = original_selectbox(*args, **kwargs)

        if label == "Omega parameterization mode" and selected == "bijective_free_boundary":
            try:
                with st.expander("Large Steps mesh conditioning", expanded=False):
                    st.checkbox(
                        "condition input mesh before S -> Omega",
                        value=True,
                        key="large_steps_conditioning_enabled",
                        help=(
                            "Automatically redistributes sampled input vertices with the "
                            "Large Steps u=(I+lambda L)v parameterization before the "
                            "bijective free-boundary solve. Connectivity and the 3D boundary stay fixed."
                        ),
                    )
                    st.number_input(
                        "Large Steps lambda",
                        min_value=0.0,
                        max_value=500.0,
                        value=10.0,
                        step=1.0,
                        key="large_steps_conditioning_lambda",
                    )
                    st.number_input(
                        "Large Steps conditioning iterations",
                        min_value=1,
                        max_value=1000,
                        value=120,
                        step=20,
                        key="large_steps_conditioning_max_iterations",
                    )
                    st.number_input(
                        "Large Steps learning rate",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.06,
                        step=0.01,
                        key="large_steps_conditioning_learning_rate",
                    )
                    st.caption(
                        "This is an in-pipeline Large-Steps conditioning stage, not an external Mitsuba preprocessing step. "
                        "Use View stage -> Conditioned S to inspect the result."
                    )
            except Exception:
                pass

        if label == "View stage":
            st.session_state["_onestring_show_conditioned_surface"] = selected == "Conditioned S"
            return "S" if selected == "Conditioned S" else selected
        return selected

    st.selectbox = selectbox_with_installed_modes
    st._onestring_ceps_selectbox_patch = True


_install_reload_guard()
_install_streamlit_parameterization_options()

__version__ = "0.5.0"

from .design_optimizer import DesignParameters, DesignResult, optimize_design
from .abd_backend import (
    ABDBackendConfig,
    ABDBackendError,
    ABDBackendUnavailableError,
    ABDCapabilityError,
    ABDRunResult,
    ShakeTrajectory,
    find_abd_executable,
    prepare_abd_job,
    probe_abd_capabilities,
    run_abd_backend,
)
from .abd_builtin_compat import install_builtin_shape_abd_compatibility
from .abd_layout_compat import install_abd_layout_compatibility

install_builtin_shape_abd_compatibility()
install_abd_layout_compatibility()

from .input_shape import create_builtin_shape, load_target_shape, normalize_shape, sample_target_surface
from .onestring_pipeline import (
    ComputeConfig,
    DeploymentParameters,
    DeploymentResult,
    FlatTileLayout,
    OneStringDesignState,
    PipelineParameters,
    ReferenceInitializationState,
    SurfaceParameterization,
    build_onestring_design,
    build_paper_reference_initialization,
    complexity_metrics,
    compute_backend_info,
    export_t2d_stl,
    export_t3d_stl,
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
from .reference_bff import (
    ReferenceBFFError,
    ReferenceBFFUnavailableError,
    ReferenceInverseMapError,
    ReferenceMeshValidationError,
    run_official_bff,
    triangle_jacobian_diagnostics,
    validate_reference_mesh,
)
from .large_steps_visualization_patch import install_large_steps_visualization_patch
from .visualization_status_patch import install_status_visualization_patch

install_large_steps_visualization_patch()
install_status_visualization_patch()

__all__ = [
    "DesignParameters",
    "DesignResult",
    "BijectiveFreeBoundaryConfig",
    "LargeStepsMeshConditioningConfig",
    "ABDBackendConfig",
    "ABDBackendError",
    "ABDBackendUnavailableError",
    "ABDCapabilityError",
    "ABDRunResult",
    "ShakeTrajectory",
    "ComputeConfig",
    "DeploymentParameters",
    "DeploymentResult",
    "FlatTileLayout",
    "OneStringDesignState",
    "PhysicsParameters",
    "PhysicsResult",
    "PhysicsWorld",
    "PipelineParameters",
    "ReferenceInitializationState",
    "QuadGrid",
    "SurfaceParameterization",
    "ReferenceBFFError",
    "ReferenceBFFUnavailableError",
    "ReferenceInverseMapError",
    "ReferenceMeshValidationError",
    "build_onestring_design",
    "bijective_free_boundary_parameterization",
    "build_paper_reference_initialization",
    "complexity_metrics",
    "compute_backend_info",
    "condition_mesh_with_large_steps",
    "export_t2d_stl",
    "export_t3d_stl",
    "find_abd_executable",
    "prepare_abd_job",
    "probe_abd_capabilities",
    "run_abd_backend",
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
    "run_official_bff",
    "triangle_jacobian_diagnostics",
    "validate_reference_mesh",
    "install_status_visualization_patch",
    "install_large_steps_visualization_patch",
    "install_large_steps_conditioning",
    "install_abd_layout_compatibility",
    "install_bijective_free_boundary",
    "__version__",
]
