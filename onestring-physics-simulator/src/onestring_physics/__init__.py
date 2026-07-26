"""Design and deployment tools for a OneString-inspired simulator."""

# Import and patch the compatibility-wrapper pipeline before re-exporting its
# public symbols. Python executes a package's __init__ before resolving
# ``onestring_physics.onestring_pipeline``, so this also covers direct submodule
# imports used by the Streamlit application and tests.
from . import onestring_pipeline as _onestring_pipeline
from .discrete_bff import install_discrete_bff

install_discrete_bff(_onestring_pipeline)

# The discrete BFF boundary is not a fixed Dirichlet rectangle, but the four
# prescribed 90-degree corners plus exact closure still produce a rectangular
# Omega. Preserve the downstream strict-rectangle crop semantics while keeping
# omega_boundary_fixed=False.
_discrete_bff_build = _onestring_pipeline._build_surface_parameterization


def _build_surface_parameterization_with_bff_rectangle_semantics(surface, target, grid, params):
    result = _discrete_bff_build(surface, target, grid, params)
    if result.metrics.get("bff_backend_used") == "local_discrete_bff":
        result.metrics["omega_boundary_forced_rectangle"] = True
        result.metrics["omega_boundary_fixed"] = False
        result.metrics["omega_boundary_shape"] = "rectangular"
    return result


_onestring_pipeline._build_surface_parameterization = _build_surface_parameterization_with_bff_rectangle_semantics
_onestring_pipeline._original._build_surface_parameterization = _build_surface_parameterization_with_bff_rectangle_semantics

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
