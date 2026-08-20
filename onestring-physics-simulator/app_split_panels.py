"""Experimental OneString app with forced paper-style Split execution.

Run with:
  python -m streamlit run app_split_panels.py --server.port 8502

This launcher deliberately owns the Split execution path.  It installs the
Split/Panel wrappers, final complete-grid-line topology pass, structured
logging, and then rebinds both module-level and package-level
``build_onestring_design`` references.  It also survives importlib.reload() of
the pipeline by reinstalling the entire Split stack afterwards.
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

LOG_PATH = ROOT / "logs" / "split_debug.jsonl"


def _fn_info(fn: Any) -> dict[str, Any]:
    code = getattr(fn, "__code__", None)
    return {
        "repr": repr(fn),
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
    """Install Split patches and force every known design entry to use them."""
    # A reload creates a fresh module namespace, so installer guards are normally
    # reset.  For an explicit re-install on the same module, clear only the patch
    # guards owned by this validation launcher.
    for flag in (
        "_onestring_split_panel_debug_installed",
        "_split_panel_force_wiring_installed",
        "_final_split_panel_pass_installed",
        "_split_diagnostics_installed",
    ):
        if reason != "startup" and hasattr(module, flag):
            try:
                setattr(module, flag, False)
            except Exception:
                pass

    install_split_panel_debug(module, opt_debug)
    install_force_split_panel_wiring(module)
    install_final_split_panel_pass(module)
    install_split_diagnostics(module, final_split_module, ROOT)
    _wire_build_globals(module)

    # Critical: the original Streamlit app commonly imports the public package
    # export (``from onestring_physics import build_onestring_design``), not the
    # wrapper module attribute.  Rebind that public export after all patches.
    package.build_onestring_design = module.build_onestring_design

    # Also expose the active module object through the package in case a caller
    # retrieves it after this launcher has patched/reloaded it.
    package.onestring_pipeline = module

    build_fn = module.build_onestring_design
    build_globals = getattr(build_fn, "__globals__", {})
    global_m2d = build_globals.get("_build_m2d") if isinstance(build_globals, dict) else None
    snapshot = {
        "reason": reason,
        "pipeline_build_onestring_design": _fn_info(module.build_onestring_design),
        "package_build_onestring_design": _fn_info(getattr(package, "build_onestring_design", None)),
        "pipeline_build_m2d": _fn_info(getattr(module, "_build_m2d", None)),
        "build_global_build_m2d": _fn_info(global_m2d),
        "pipeline_original_build_m2d": _fn_info(getattr(getattr(module, "_original", None), "_build_m2d", None)),
        "package_and_pipeline_build_same_object": bool(getattr(package, "build_onestring_design", None) is module.build_onestring_design),
        "global_and_pipeline_m2d_same_object": bool(global_m2d is getattr(module, "_build_m2d", None)),
    }
    _append_route_log("split_execution_route_wired", **snapshot)
    print(
        "[SPLIT-ROUTE] "
        f"reason={reason} "
        f"package_build_same={snapshot['package_and_pipeline_build_same_object']} "
        f"global_m2d_same={snapshot['global_and_pipeline_m2d_same_object']} "
        f"m2d={snapshot['pipeline_build_m2d']['name']} "
        f"@ {snapshot['pipeline_build_m2d']['file']}:{snapshot['pipeline_build_m2d']['line']}"
    )
    if not snapshot["package_and_pipeline_build_same_object"] or not snapshot["global_and_pipeline_m2d_same_object"]:
        raise RuntimeError(f"SPLIT_EXECUTION_ROUTE_WIRING_FAILED: {snapshot}")


_install_split_stack(pipeline, reason="startup")

# onestring_physics.__init__ already installs a reload guard for parameterization
# backends.  Wrap the current reload function (rather than replacing its work)
# and reinstall the Split stack whenever the pipeline itself is reloaded.
_previous_reload = importlib.reload


def _reload_with_split_reinstall(module: Any) -> Any:
    result = _previous_reload(module)
    if getattr(result, "__name__", "") == "onestring_physics.onestring_pipeline":
        _install_split_stack(result, reason="pipeline_reload")
    return result


importlib.reload = _reload_with_split_reinstall

# Run the normal UI only after the execution route above is pinned.
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
