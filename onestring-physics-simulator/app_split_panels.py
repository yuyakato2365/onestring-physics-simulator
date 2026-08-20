"""Validation launcher for simple Split panels plus animation View stages.

For this launcher, Split semantics are intentionally simple:
- cut once along each requested existing row/column grid line;
- duplicate interface vertex ids so panels become edge-disconnected;
- preserve the original M2D arrangement and open only a symmetric seam gap;
- never bin-pack or reorder panels.

The legacy Streamlit app reloads ``onestring_pipeline`` at import time.  That is
useful for the normal app but destroys runtime patches here, so this validation
launcher freezes that one reload and keeps exactly one Split wrapper stack.
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
from onestring_physics.simple_split_panel_patch import install_simple_split_panel_patch  # noqa: E402
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
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": None,
                    "event": event,
                    **payload,
                },
                ensure_ascii=False,
                default=repr,
                sort_keys=True,
            )
            + "\n"
        )


def _wire_active_globals(module: Any) -> None:
    active_build = getattr(module, "_build_m2d", None)
    active_lift = getattr(module, "_lift_m2d_to_m3d", None)
    active_k2d = getattr(module, "_optimize_k2d", None)
    for fn in (
        getattr(module, "build_onestring_design", None),
        getattr(module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(module, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = active_build
            glb["_lift_m2d_to_m3d"] = active_lift
            glb["_optimize_k2d"] = active_k2d


def _install_once() -> None:
    install_simple_split_panel_patch(pipeline, opt_debug)
    install_split_diagnostics(pipeline, final_split_module, ROOT)
    _wire_active_globals(pipeline)

    base_build = pipeline.build_onestring_design
    cached_build = wrap_build_to_cache_state(base_build)
    pipeline.build_onestring_design = cached_build
    package.build_onestring_design = cached_build
    package.onestring_pipeline = pipeline

    build_globals = getattr(base_build, "__globals__", {})
    global_m2d = build_globals.get("_build_m2d") if isinstance(build_globals, dict) else None
    snapshot = {
        "pipeline_build_m2d": _fn_info(getattr(pipeline, "_build_m2d", None)),
        "build_global_build_m2d": _fn_info(global_m2d),
        "global_and_pipeline_m2d_same_object": bool(global_m2d is getattr(pipeline, "_build_m2d", None)),
    }
    _append_route_log("simple_split_execution_route_wired", **snapshot)
    print(
        "[SPLIT-ROUTE] simple_split=True "
        f"global_m2d_same={snapshot['global_and_pipeline_m2d_same_object']} "
        f"m2d={snapshot['pipeline_build_m2d']['name']}"
    )
    if not snapshot["global_and_pipeline_m2d_same_object"]:
        raise RuntimeError(f"SIMPLE_SPLIT_EXECUTION_ROUTE_WIRING_FAILED: {snapshot}")


if not getattr(package, "_onestring_simple_split_launcher_installed", False):
    _install_once()
    package._onestring_simple_split_launcher_installed = True


# The backed-up legacy app immediately executes importlib.reload(pipeline).
# In this dedicated validation launcher that would erase the Split wrappers we
# just installed.  Return the already-patched module instead.  Other modules
# retain normal importlib.reload behavior.
if not getattr(importlib, "_onestring_simple_split_freeze_installed", False):
    _real_reload = importlib.reload

    def _reload_except_active_pipeline(module: Any) -> Any:
        if module is pipeline or getattr(module, "__name__", "") == "onestring_physics.onestring_pipeline":
            _append_route_log("pipeline_reload_skipped_for_simple_split")
            print("[SPLIT-ROUTE] skipped legacy pipeline reload; simple Split stack preserved")
            return pipeline
        return _real_reload(module)

    importlib.reload = _reload_except_active_pipeline
    importlib._onestring_simple_split_freeze_installed = True


install_view_stage_animation_selector()

# Run the normal UI after the simple Split route is pinned.
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

# Synthetic View-stage animation choices are rendered from the cached design.
render_selected_animation_view()
