"""Lightweight design and deployment tools for a OneString-inspired simulator."""

from .design_optimizer import DesignParameters, DesignResult, optimize_design
from .input_shape import create_builtin_shape, load_target_shape, normalize_shape, sample_target_surface
from .physics_world import PhysicsParameters, PhysicsResult, PhysicsWorld, simulate_deployment
from .quad_grid import QuadGrid, create_quad_grid

__all__ = [
    "DesignParameters",
    "DesignResult",
    "PhysicsParameters",
    "PhysicsResult",
    "PhysicsWorld",
    "QuadGrid",
    "create_builtin_shape",
    "create_quad_grid",
    "load_target_shape",
    "normalize_shape",
    "optimize_design",
    "sample_target_surface",
    "simulate_deployment",
]
