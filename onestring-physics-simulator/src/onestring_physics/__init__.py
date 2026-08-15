"""Design and deployment tools for a OneString-inspired simulator."""
from __future__ import annotations

import importlib
from typing import Any

from . import onestring_pipeline as _onestring_pipeline
from . import official_ceps as _official_ceps
from .ceps_outer_boundary import install_ceps_outer_boundary
from .ceps_paired_output import install_ceps_paired_output
from .ceps_strict_adapter import install_ceps_strict_adapter
from .bijective_free_boundary import (
    BijectiveFreeBoundaryConfig,
    bijective_free_boundary_parameterization,
    install_bijective_free_boundary,
)
from .discrete_bff import install_discrete_bff
from .official_ceps import install_official_ceps


# CEPS ordinary UV is stored per face corner. The current OneString inverse map
# can consume it only when those corner values already form one single-valued disk
# chart. Internal CEPS cut seams are rejected explicitly; they are not stitched,
# replaced by a convex hull, or hidden with nearest-triangle fallback. When the
# chart is valid, its true physical boundary loop is used directly.
install_ceps_paired_output(_official_ceps)
install_ceps_outer_boundary(_official_ceps)


def _install_parameterization_backends(module: Any) -> None:
    """Install BFF and optional official CEPS after every pipeline load/reload."""
    # importlib.reload() preserves a module dictionary. Clear old installation
    # markers because the re-executed compatibility wrapper replaced the patch.
    module._DISCRETE_BFF_PATCH_INSTALLED = False
    module._BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED = False
    module._OFFICIAL_CEPS_PATCH_INSTALLED = False
    module._CEPS_STRICT_ADAPTER_INSTALLED = False
    install_discrete_bff(module)
    install_bijective_free_boundary(module)
    install_official_ceps(module)
    install_ceps_strict_adapter(module)

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
            # Backward-compatible metric name used by the existing UI/tests.
            metrics["ceps_paired_surface_uv_vertices"] = metrics[
                "ceps_continuous_surface_uv_vertices"
            ]
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


def _install_streamlit_parameterization_options() -> None:
    """Expose separately installed parameterization modes in the legacy app."""
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
        return original_selectbox(*args, **kwargs)

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

# Install before onestring_pipeline imports run_abd_backend. These patches keep
# the normal routed gap path when valid, provide deterministic guides for
# builtin shapes when needed, and ensure every ABD collision body starts with a
# strictly positive gap, including bodies joined by a hinge.
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
from .visualization_status_patch import install_status_visualization_patch

# Make recovery colors independent of Plotly lighting. In particular, clipped
# cyan/blue solids must never appear gray like the non-authoritative emergency
# normal prism.
install_status_visualization_patch()

__all__ = [
    "DesignParameters",
    "DesignResult",
    "BijectiveFreeBoundaryConfig",
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
    "install_abd_layout_compatibility",
    "install_bijective_free_boundary",
    "__version__",
]
