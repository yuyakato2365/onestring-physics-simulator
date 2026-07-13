"""Design and deployment tools for a OneString-inspired simulator."""

__version__ = "0.3.0"

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

# Install before onestring_pipeline imports run_abd_backend.  This preserves the
# normal routed gap path and only supplies deterministic guides when a builtin
# shape produces an empty or degenerate path.
install_builtin_shape_abd_compatibility()

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

# Make recovery colors independent of Plotly lighting.  In particular, clipped
# cyan/blue solids must never appear gray like the non-authoritative emergency
# normal prism.
install_status_visualization_patch()

__all__ = [
    "DesignParameters",
    "DesignResult",
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
    "__version__",
]
