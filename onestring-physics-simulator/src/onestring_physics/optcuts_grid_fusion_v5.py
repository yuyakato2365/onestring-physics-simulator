"""Compatibility shim for the grid-constrained OptCuts fusion.

The former v5 implementation assigned lattice lines first and then tried to
repair short/zero-length segments locally.  That produced repeated
JUNCTION_CONFLICT and ZERO_LENGTH_SEGMENT failures.  Keep the public installer
name used by app_optcuts.py, but route it entirely to the v6 global integer
lattice embedding solver.
"""
from __future__ import annotations

from .optcuts_grid_fusion_v6 import (
    _global_targets,
    install_integer_lattice_grid_fusion,
)


def install_orthogonal_segment_grid_fusion() -> None:
    install_integer_lattice_grid_fusion()


__all__ = ["install_orthogonal_segment_grid_fusion", "_global_targets"]
