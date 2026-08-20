"""Validation launcher for paper-style Split plus animation View stages.

Key invariants:
- Split wrappers are installed at most once per live pipeline generation.
- Streamlit script reruns do not stack checked/diagnostic/final wrappers.
- A real importlib.reload(onestring_pipeline) gets one fresh Split stack.
- View stage exposes the existing T2D -> T3D assembly animations.
"""
from __future__ import annotations

import importlib
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import onestring_physics as package  # noqa: E402
from onestring_physics import onestring_pipeline as pipeline  # noqa: E402
from onestring_physics import optimization_debug_visualization as opt_debug  # noqa: E402
from onestring_physics import final_split_panel_pass as final_split_module  # noqa: E402
from onestring_physics.split_panel_debug_patch import install_split_panel_debug  # noqa: E402
from onestring_physics.split_panel_force_wiring import install_force_split_panel_wiring  # noqa: E402
from onestring_physics.final_split_panel_pass import install_final_split_panel_pass  # noqa: E402
from onestring_physics.split_diagnostics import install_split_diagnostics  # noqa: E402
from onestring_physics.view_stage_animation_patch import (  # noqa: E402
    install_view_stage_animation_selector,
    render_selected_animation_view,
    wrap_build_to_cache_state,
)

LOG_PATH = ROOT / "logs" / "split_debug.jsonl"


def _fn_info(fn: Any) -> dict[str, Any]:
    code = getattr(fn, "__code__", None)
    return {
        "id": int(id(fn)) if fn is not None else None,
        "module": getattr(fn, "__module__", None),
        "name": getattr(fn, "__name__", None),
        "file": getattr(code, "co_filename", None),
        "line": getattr(code, "co_firstlineno", None),
    }


def _append_route_log(event: str, **payload: Any) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": None,
        "event": event,
        **payload,
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=repr, sort_keys=True) + "\n")


def _normalize_split_axis_parser() -> None:
    """Accept both legacy 'col' and final-pass 'column' names."""
    if not hasattr(final_split_module, "_onestring_original_split_axis_value"):
        final_split_module._onestring_original_split_axis_value = final_split_module._split_axis_value
    base = final_split_module._onestring_original_split_axis_value

    def split_axis_value_compat(line: Any):
        try:
            axis = str(line[0]).strip().lower()
            value = float(line[1])
        except Exception:
            return None
        aliases = {"col": "column", "column": "column", "row": "row"}
        normalized = aliases.get(axis)
        if normalized is None:
            return None
        return base((normalized, value))

    final_split_module._split_axis_value = split_axis_value_compat


def _reset_final_candidate_to_base() -> None:
    """Prevent diagnostic_candidate from being wrapped around itself on reload."""
    if not hasattr(final_split_module, "_onestring_split_candidate_base"):
        final_split_module._onestring_split_candidate_base = final_split_module._complete_cut_candidate
    else:
        final_split_module._complete_cut_candidate = final_split_module._onestring_split_candidate_base


def _wire_build_globals(module: Any) -> None:
    active_build = getattr(module, "_build_m2d", None)
    active_lift = getattr(module, "_lift_m2d_to_m3d", None)
    active_k2d = getattr(module, "_optimize_k2d", None)
    for fn in (
        getattr(module, "build_onestring_design", None),
        getattr(getattr(module, "_original", None), "build_onestring_design", None),
        getattr(module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            if active_build is not None:
                glb["_build_m2d"] = active_build
            if active_lift is not None:
                glb["_lift_m2d_to_m3d"] = active_lift
            if active_k2d is not None:
                glb["_optimize_k2d"] = active_k2d


def _install_split_stack(module: Any, *, reason: str) -> None:
    """Install exactly one Split stack on the current pipeline generation."""
    _normalize_split_axis_parser()
    _reset_final_candidate_to_base()

    # A true module reload recreates these flags.  Never clear flags on an
    # ordinary Streamlit rerun: force wiring has a closure and would stack.
    install_split_panel_debug(module, opt_debug)
    install_force_split_panel_wiring(module)
    install_final_split_panel_pass(module)
    install_split_diagnostics(module, final_split_module, ROOT)
    _wire_build_globals(module)

    base_build = module.build_onestring_design
    cached_build = wrap_build_to_cache_state(base_build)
    module.build_onestring_design = cached_build
    package.build_onestring_design = cached_build
    package.onestring_pipeline = module

    build_globals = getattr(base_build, "__globals__", {})
    global_m2d = build_globals.get("_build_m2d") if isinstance(build_globals, dict) else None
    snapshot = {
        "reason": reason,
        "pipeline_build_m2d": _fn_info(getattr(module, "_build_m2d", None)),
        "build_global_build_m2d": _fn_info(global_m2d),
        "global_and_pipeline_m2d_same_object": bool(global_m2d is getattr(module, "_build_m2d", None)),
    }
    _append_route_log("split_execution_route_wired", **snapshot)
    print(
        "[SPLIT-ROUTE] "
        f"reason={reason} "
        f"global_m2d_same={snapshot['global_and_pipeline_m2d_same_object']} "
        f"m2d={snapshot['pipeline_build_m2d']['name']}"
    )
    if not snapshot["global_and_pipeline_m2d_same_object"]:
        raise RuntimeError(f"SPLIT_EXECUTION_ROUTE_WIRING_FAILED: {snapshot}")


# Install only once for ordinary Streamlit reruns.  A true pipeline reload is
# handled by the one global reload hook below.
if not getattr(package, "_onestring_split_launcher_installed", False):
    _install_split_stack(pipeline, reason="startup")
    package._onestring_split_launcher_installed = True
else:
    # Keep package export synchronized without adding another M2D wrapper layer.
    package.build_onestring_design = pipeline.build_onestring_design


# Wrap the existing parameterization reload guard once per Python process.
if not getattr(importlib, "_onestring_split_reload_hook_installed", False):
    _previous_reload = importlib.reload

    def _reload_with_single_split_reinstall(module: Any) -> Any:
        result = _previous_reload(module)
        if getattr(result, "__name__", "") == "onestring_physics.onestring_pipeline":
            # Reload creates clean pipeline functions and clean installer flags,
            # so one reinstall is correct.  The final candidate lives in another
            # module, hence it is explicitly reset to its saved base above.
            package._onestring_split_launcher_installed = False
            _install_split_stack(result, reason="pipeline_reload")
            package._onestring_split_launcher_installed = True
        return result

    importlib.reload = _reload_with_single_split_reinstall
    importlib._onestring_split_reload_hook_installed = True


install_view_stage_animation_selector()

# Run the normal UI only after Split and View-stage patches are pinned.
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

# Synthetic View-stage animation choices are rendered from the cached design
# after the legacy app body has completed.
render_selected_animation_view()
