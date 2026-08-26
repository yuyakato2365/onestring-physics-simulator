"""Validation launcher for simple Split panels plus unified process animations.

Numerical Split semantics stay intentionally simple and unchanged:
- cut once along each requested existing row/column grid line;
- duplicate interface vertex ids so panels become edge-disconnected;
- preserve the original M2D arrangement and open only a symmetric seam gap;
- never bin-pack or reorder panels.

UI/debug behavior is deliberately separated from those numerics.  A fresh run
shows exactly one diagnostic sequence (Omega -> Split -> K2D), and the same
Omega/K2D animations can later be selected from View stage without substituting
T3D or another legacy static stage.
"""
from __future__ import annotations

import importlib
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import onestring_physics as package  # noqa: E402
from onestring_physics import onestring_pipeline as pipeline  # noqa: E402
from onestring_physics import optimization_debug_visualization as opt_debug  # noqa: E402
from onestring_physics import final_split_panel_pass as final_split_module  # noqa: E402
from onestring_physics import simple_split_panel_patch as simple_split_module  # noqa: E402
from onestring_physics.simple_split_panel_patch import install_simple_split_panel_patch  # noqa: E402
from onestring_physics.split_diagnostics import install_split_diagnostics  # noqa: E402
from onestring_physics.process_animation_view_patch import (  # noqa: E402
    install_k2d_process_recorder,
)
from onestring_physics.process_animation_view_fix import (  # noqa: E402
    install_process_view_reliability_fixes,
)
from onestring_physics.unified_animation_ui import (  # noqa: E402
    cache_omega_payload,
    install_unified_animation_ui,
    render_postrun_process_sequence,
    render_selected_synthetic_view,
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


def _install_deferred_omega_renderer() -> None:
    """Cache Omega frames during computation; draw them only in the unified UI."""
    if not hasattr(opt_debug, "_onestring_original_omega_renderer_for_unified_view"):
        opt_debug._onestring_original_omega_renderer_for_unified_view = (
            opt_debug.render_omega_flip_debug_animation
        )

    def cache_only(frames: Any, faces: Any, boundary_loop: Any, summary: Any = None) -> None:
        cache_omega_payload(frames, faces, boundary_loop, summary)

    opt_debug.render_omega_flip_debug_animation = cache_only


def _install_once() -> None:
    # Cache accepted Omega states directly at recorder completion as a fallback
    # for backends/callers that bypass the renderer module attribute.
    install_process_view_reliability_fixes(opt_debug)
    _install_deferred_omega_renderer()

    # Preserve the genuine Split diagnostic renderer before installing its
    # numerical wrapper.  The optimizer must not draw UI as a side effect.
    if not hasattr(simple_split_module, "_onestring_original_split_renderer_for_unified_view"):
        simple_split_module._onestring_original_split_renderer_for_unified_view = (
            simple_split_module.render_split_panel_correspondence
        )

    install_simple_split_panel_patch(pipeline, opt_debug)

    # Stop BOTH historical UI side effects:
    # 1) optimize_k2d_keep_panel_layout() used this module-global renderer;
    # 2) the Split installer overwrote the K2D-correspondence renderer slot.
    # The Split/K2D visuals are rendered exactly once after the full build.
    def deferred_noop(*args: Any, **kwargs: Any) -> None:
        return None

    simple_split_module.render_split_panel_correspondence = deferred_noop
    opt_debug.render_k2d_correspondence_morph = deferred_noop

    # Record the outermost active K2D path after Split has wrapped it.  This
    # captures real solver checkpoints and the final accepted K2D result.
    install_k2d_process_recorder(pipeline)

    install_split_diagnostics(pipeline, final_split_module, ROOT)
    _wire_active_globals(pipeline)

    # Keep public imports synchronized with the already-patched pipeline.  The
    # legacy app imports build_onestring_design from the pipeline module itself.
    package.build_onestring_design = pipeline.build_onestring_design
    package.onestring_pipeline = pipeline

    build_fn = pipeline.build_onestring_design
    build_globals = getattr(build_fn, "__globals__", {})
    global_m2d = build_globals.get("_build_m2d") if isinstance(build_globals, dict) else None
    global_k2d = build_globals.get("_optimize_k2d") if isinstance(build_globals, dict) else None
    snapshot = {
        "pipeline_build_m2d": _fn_info(getattr(pipeline, "_build_m2d", None)),
        "build_global_build_m2d": _fn_info(global_m2d),
        "global_and_pipeline_m2d_same_object": bool(global_m2d is getattr(pipeline, "_build_m2d", None)),
        "pipeline_optimize_k2d": _fn_info(getattr(pipeline, "_optimize_k2d", None)),
        "build_global_optimize_k2d": _fn_info(global_k2d),
        "global_and_pipeline_k2d_same_object": bool(global_k2d is getattr(pipeline, "_optimize_k2d", None)),
    }
    _append_route_log("simple_split_execution_route_wired", **snapshot)
    print(
        "[SPLIT-ROUTE] simple_split=True unified_animation_ui=True "
        f"global_m2d_same={snapshot['global_and_pipeline_m2d_same_object']} "
        f"global_k2d_same={snapshot['global_and_pipeline_k2d_same_object']} "
        f"m2d={snapshot['pipeline_build_m2d']['name']} "
        f"k2d={snapshot['pipeline_optimize_k2d']['name']}"
    )
    if not snapshot["global_and_pipeline_m2d_same_object"]:
        raise RuntimeError(f"SIMPLE_SPLIT_EXECUTION_ROUTE_WIRING_FAILED: {snapshot}")
    if not snapshot["global_and_pipeline_k2d_same_object"]:
        raise RuntimeError(f"K2D_PROCESS_EXECUTION_ROUTE_WIRING_FAILED: {snapshot}")


if not getattr(package, "_onestring_simple_split_launcher_installed", False):
    _install_once()
    package._onestring_simple_split_launcher_installed = True
else:
    # Streamlit reruns the launcher in the same Python process.  Reassert UI-only
    # deferrals without stacking any numerical wrappers.
    _install_deferred_omega_renderer()
    simple_split_module.render_split_panel_correspondence = lambda *a, **k: None
    opt_debug.render_k2d_correspondence_morph = lambda *a, **k: None


# The backed-up legacy app immediately executes importlib.reload(pipeline).
# In this dedicated launcher that would erase Split/K2D/Omega instrumentation.
if not getattr(importlib, "_onestring_simple_split_freeze_installed", False):
    _real_reload = importlib.reload

    def _reload_except_active_pipeline(module: Any) -> Any:
        if module is pipeline or getattr(module, "__name__", "") == "onestring_physics.onestring_pipeline":
            _append_route_log("pipeline_reload_skipped_for_simple_split")
            print("[SPLIT-ROUTE] skipped legacy pipeline reload; unified animation stack preserved")
            return pipeline
        return _real_reload(module)

    importlib.reload = _reload_except_active_pipeline
    importlib._onestring_simple_split_freeze_installed = True


# One selector wrapper owns every synthetic animation choice.  It returns a
# private sentinel, never T3D, so the legacy View-stage branch draws no impostor.
install_unified_animation_ui()

# Capture the legacy script globals so we know whether this rerun performed an
# actual fresh pipeline build or was only a display/View-stage rerun.
legacy_globals = runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

state = legacy_globals.get("state")
if state is None:
    state = st.session_state.get("onestring_state")

fresh_run = bool(legacy_globals.get("run_pipeline", False))
split_renderer = getattr(
    simple_split_module,
    "_onestring_original_split_renderer_for_unified_view",
    None,
)

if fresh_run and state is not None and split_renderer is not None:
    # Exactly one fresh-run sequence: Omega -> Split -> K2D.
    render_postrun_process_sequence(state, opt_debug, split_renderer)
else:
    # Display-only reruns (including changing View stage) draw only the selected
    # synthetic animation.  Normal legacy stages were already rendered above.
    render_selected_synthetic_view(opt_debug, state=state)
