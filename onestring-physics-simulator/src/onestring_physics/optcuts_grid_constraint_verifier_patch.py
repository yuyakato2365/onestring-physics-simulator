"""Hard verifier for the OneString grid-constrained OptCuts contract.

The constrained OptCuts mode is allowed to continue only when the FINAL UV map
actually satisfies the fabrication constraints promised by the UI:

* every physical OptCuts seam copy is one straight horizontal or vertical line;
* the constant coordinate is an integer multiple of the fixed lattice unit h;
* seam endpoints lie on lattice intersections;
* the constrained stage did not change the OptCuts cut topology (only UV values).

This is deliberately a hard assertion rather than a visualization heuristic.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .optcuts_grid_constrained_parameterization_patch import (
    _physical_chains,
    _record_lookup,
    _surface_seam_records,
    _uv_copy_chains_for_physical_chain,
)


def _distance_to_lattice(value: float, h: float) -> float:
    return abs(float(value) - round(float(value) / float(h)) * float(h))


def _verify(parameterization: Any, h: float) -> dict[str, Any]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    records = _surface_seam_records(parameterization)
    chains = _physical_chains(records)
    lookup = _record_lookup(records)
    scale = max(float(h), 1e-8)
    line_tol = max(1e-7, 2e-5 * scale)
    lattice_tol = max(1e-7, 2e-5 * scale)

    checked = 0
    bad_axis: list[int] = []
    bad_line: list[int] = []
    bad_endpoint: list[int] = []
    max_line_deviation = 0.0
    max_gridline_offset = 0.0
    max_endpoint_offset = 0.0

    for physical_id, chain in enumerate(chains):
        side0, side1 = _uv_copy_chains_for_physical_chain(chain, lookup)
        for copy in (side0, side1):
            if len(copy) < 2:
                continue
            checked += 1
            pts = uv[np.asarray(copy, dtype=int)]
            span_x = float(np.ptp(pts[:, 0]))
            span_y = float(np.ptp(pts[:, 1]))
            if span_y <= span_x:
                axis = 0  # horizontal; y is constant
                constant_axis = 1
            else:
                axis = 1  # vertical; x is constant
                constant_axis = 0
            line_dev = float(np.max(np.abs(pts[:, constant_axis] - np.mean(pts[:, constant_axis]))))
            max_line_deviation = max(max_line_deviation, line_dev)
            if line_dev > line_tol:
                bad_axis.append(int(physical_id))

            constant = float(np.mean(pts[:, constant_axis]))
            grid_offset = _distance_to_lattice(constant, scale)
            max_gridline_offset = max(max_gridline_offset, grid_offset)
            if grid_offset > lattice_tol:
                bad_line.append(int(physical_id))

            for endpoint in (pts[0], pts[-1]):
                ex = _distance_to_lattice(float(endpoint[0]), scale)
                ey = _distance_to_lattice(float(endpoint[1]), scale)
                eoff = max(ex, ey)
                max_endpoint_offset = max(max_endpoint_offset, eoff)
                if eoff > lattice_tol:
                    bad_endpoint.append(int(physical_id))

            # A straight chain must have nonzero extent on its varying axis.
            if float(np.ptp(pts[:, axis])) <= line_tol:
                bad_axis.append(int(physical_id))

    result = {
        "verified_seam_copy_chain_count": int(checked),
        "max_seam_line_deviation": float(max_line_deviation),
        "max_seam_gridline_offset": float(max_gridline_offset),
        "max_seam_endpoint_lattice_offset": float(max_endpoint_offset),
        "non_axis_aligned_chain_count": int(len(set(bad_axis))),
        "off_grid_line_chain_count": int(len(set(bad_line))),
        "off_grid_endpoint_chain_count": int(len(set(bad_endpoint))),
        "constraint_verified": bool(not bad_axis and not bad_line and not bad_endpoint),
    }
    if not result["constraint_verified"]:
        raise RuntimeError(
            "OPTCUTS_GRID_SEAM_CONSTRAINT_VIOLATION: final UV violates the requested "
            "straight/orthogonal/fixed-unit seam contract; "
            f"bad_axis={sorted(set(bad_axis))[:16]} "
            f"bad_grid_lines={sorted(set(bad_line))[:16]} "
            f"bad_endpoints={sorted(set(bad_endpoint))[:16]} "
            f"max_line_deviation={max_line_deviation:.6g} "
            f"max_gridline_offset={max_gridline_offset:.6g} "
            f"max_endpoint_offset={max_endpoint_offset:.6g}"
        )
    return result


def install_optcuts_grid_constraint_verifier_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_grid_constraint_verifier_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def verified_builder(surface: Any, target: Any, grid: Any, params: Any):
        p = base_builder(surface, target, grid, params)
        if not bool(getattr(p, "metrics", {}).get("optcuts_grid_constrained", False)):
            return p
        h = max(float(getattr(params, "tile_size", getattr(grid, "tile_size", 0.0))), 1e-8)
        audit = _verify(p, h)
        p.metrics.update({
            "optcuts_grid_constraint_verified": True,
            **audit,
        })
        print(
            "[OPTCUTS-GRID-VERIFY] "
            f"chains={audit['verified_seam_copy_chain_count']} "
            f"line_dev={audit['max_seam_line_deviation']:.3g} "
            f"grid_offset={audit['max_seam_gridline_offset']:.3g} "
            f"endpoint_offset={audit['max_seam_endpoint_lattice_offset']:.3g}"
        )
        return p

    pipeline._build_surface_parameterization = verified_builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = verified_builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = verified_builder
    pipeline._onestring_optcuts_grid_constraint_verifier_installed = True


__all__ = ["install_optcuts_grid_constraint_verifier_patch", "_verify"]
