"""Force Split/Panel wrappers into the exact globals used by build_onestring_design.

The project contains wrapper and backed-up pipeline modules. Rebinding module
attributes is not always sufficient because build_onestring_design executes with
its defining module's global dictionary. This helper patches that dictionary
explicitly and adds a fail-fast M2D guard for requested-but-unrealized splits.
"""
from __future__ import annotations

from typing import Any


def install_force_split_panel_wiring(pipeline_module: Any) -> None:
    build_fn = pipeline_module.build_onestring_design
    globals_dict = getattr(build_fn, "__globals__", None)
    if not isinstance(globals_dict, dict):
        raise RuntimeError("Cannot access build_onestring_design globals for Split/Panel wiring")

    patched_build_m2d = pipeline_module._build_m2d
    patched_lift = pipeline_module._lift_m2d_to_m3d
    patched_k2d = pipeline_module._optimize_k2d

    # Patch the exact names resolved by the executing legacy function.
    globals_dict["_build_m2d"] = patched_build_m2d
    globals_dict["_lift_m2d_to_m3d"] = patched_lift
    globals_dict["_optimize_k2d"] = patched_k2d

    # Also mirror into the wrapper/back-up module attributes for callers that do
    # regular attribute lookup instead of function-global lookup.
    try:
        pipeline_module._original._build_m2d = patched_build_m2d
        pipeline_module._original._lift_m2d_to_m3d = patched_lift
        pipeline_module._original._optimize_k2d = patched_k2d
    except Exception:
        pass

    # Fail fast if the active Split wrapper is reached but topology still remains
    # a single panel. This prevents a misleading M2D/K2D result from being shown.
    original_active_build = globals_dict["_build_m2d"]

    def checked_build_m2d(grid: Any, domain: Any, params: Any = None):
        mesh = original_active_build(grid, domain, params)
        metrics = getattr(mesh, "metrics", {}) or {}
        split_applied = bool(metrics.get("csf_split_applied", False))
        panel_count = int(metrics.get("split_panel_count", metrics.get("m2d_connected_component_count_after_split", 1)) or 1)
        if split_applied and panel_count <= 1:
            raise RuntimeError(
                "SPLIT_PANEL_WIRING_REACHED_BUT_NOT_SEPARATED: Split was applied, "
                "but the returned M2D still has one panel. The pipeline will not "
                "continue to M3D/K2D with a fake Split."
            )
        try:
            mesh.metrics["split_panel_force_wiring_active"] = True
            mesh.metrics["split_panel_build_global_patched"] = True
        except Exception:
            pass
        return mesh

    globals_dict["_build_m2d"] = checked_build_m2d
    try:
        pipeline_module._original._build_m2d = checked_build_m2d
    except Exception:
        pass

    pipeline_module._split_panel_force_wiring_installed = True


__all__ = ["install_force_split_panel_wiring"]
