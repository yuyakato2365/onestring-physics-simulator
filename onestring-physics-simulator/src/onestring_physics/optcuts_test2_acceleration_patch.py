"""Acceleration controls for the optcuts_test2 experimental variant.

The visible UI mode ``optcuts_test2`` is internally routed through the existing
``optcuts_test`` implementation so test1 remains unchanged. A process environment
flag selects this variant.

Test2 changes only the numerical K3D budget plus safe SciPy worker usage.  This
module also instruments the already-installed OptCuts-test K3D augmented-
Lagrangian layer so Streamlit can show useful progress between the pipeline's
38% M3D checkpoint and 50% K3D checkpoint.

Live K3D preview policy:
- show the ordinary K3D result when the hard-planarity AL starts;
- update the same Plotly placeholder after every AL feasibility-restoration step;
- decimate only the *visualization* to at most 1200 quads; solver data is untouched;
- if one preview render takes more than 0.25 s, disable further live rendering for
  that run so visualization cannot materially slow the numerical solve.

K2D numerical budgets are deliberately identical to ``optcuts_test``:
- kinematic outer passes: 4
- least-squares max_nfev: 70
- collision candidate cap: 2500
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import os
import time
from typing import Any

import numpy as np


def _is_test2() -> bool:
    return os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "1").strip() == "2"


def _install_streamlit_progress_capture() -> None:
    """Remember the pipeline progress bar created by the base Streamlit app."""
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_k3d_progress_capture_installed", False):
        return

    original_progress = st.progress

    def captured_progress(value: Any, *args: Any, **kwargs: Any):
        bar = original_progress(value, *args, **kwargs)
        text = str(kwargs.get("text", ""))
        if text.startswith("Preparing pipeline"):
            st._onestring_pipeline_progress_bar = bar
        return bar

    st.progress = captured_progress
    st._onestring_k3d_progress_capture_installed = True


def _ui_progress(fraction: float, stage: str, detail: str = "") -> None:
    try:
        import streamlit as st

        bar = getattr(st, "_onestring_pipeline_progress_bar", None)
        if bar is None:
            return
        suffix = f" — {detail}" if detail else ""
        bar.progress(float(np.clip(fraction, 0.0, 1.0)), text=f"{fraction * 100:5.1f}%  {stage}{suffix}")
    except Exception:
        return


def _preview_figure(vertices: np.ndarray, faces: np.ndarray, title: str):
    import plotly.graph_objects as go

    verts = np.asarray(vertices, dtype=float)
    f4 = np.asarray(faces, dtype=int)[:, :4]
    if len(f4) > 1200:
        ids = np.linspace(0, len(f4) - 1, 1200, dtype=int)
        shown = f4[ids]
        suffix = f" (preview {len(shown)}/{len(f4)} quads)"
    else:
        shown = f4
        suffix = f" ({len(shown)} quads)"

    tri_a = shown[:, [0, 1, 2]]
    tri_b = shown[:, [0, 2, 3]]
    tri = np.vstack([tri_a, tri_b]) if len(shown) else np.zeros((0, 3), dtype=int)
    fig = go.Figure()
    if len(tri):
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                opacity=0.72,
                flatshading=True,
                name="K3D live",
                showscale=False,
            )
        )
    fig.update_layout(
        title=title + suffix,
        height=430,
        margin=dict(l=0, r=0, t=45, b=0),
        uirevision="onestring-k3d-live-preview",
        scene=dict(aspectmode="data", uirevision="onestring-k3d-live-preview"),
        showlegend=False,
    )
    return fig


def _render_live_preview(context: dict[str, Any], vertices: np.ndarray, faces: np.ndarray, title: str) -> None:
    if bool(context.get("preview_disabled", False)):
        return
    try:
        import streamlit as st

        placeholder = context.get("preview_placeholder")
        if placeholder is None:
            placeholder = st.empty()
            context["preview_placeholder"] = placeholder
        started = time.perf_counter()
        serial = int(context.get("preview_serial", 0))
        placeholder.plotly_chart(
            _preview_figure(vertices, faces, title),
            width="stretch",
            key=f"onestring_k3d_live_{serial}",
        )
        context["preview_serial"] = serial + 1
        render_sec = float(time.perf_counter() - started)
        context["last_preview_render_sec"] = render_sec
        if render_sec > 0.25:
            context["preview_disabled"] = True
            try:
                placeholder.caption(
                    f"K3D live preview paused because one render took {render_sec:.2f}s; numerical progress continues without preview overhead."
                )
            except Exception:
                pass
    except Exception as exc:
        context["preview_disabled"] = True
        print(f"[OPTCUTS-K3D-LIVE-PREVIEW] disabled: {type(exc).__name__}: {exc}")


def _install_k3d_al_live_instrumentation() -> None:
    """Instrument the existing AL module without changing its numerical equations."""
    try:
        from . import optcuts_test_k3d_augmented_lagrangian_patch as al_module
    except Exception as exc:
        print(f"[OPTCUTS-K3D-LIVE] AL module unavailable: {type(exc).__name__}: {exc}")
        return
    if getattr(al_module, "_onestring_live_progress_installed", False):
        return

    original_al = al_module._augmented_lagrangian_planarize
    original_restore = al_module._consensus_planarity_restore
    active: dict[str, Any] = {"context": None}

    def restore_with_live_preview(vertices: Any, faces: Any, **kwargs: Any):
        result = original_restore(vertices, faces, **kwargs)
        context = active.get("context")
        if context is None:
            return result
        solved, used, residual = result
        context["restore_count"] = int(context.get("restore_count", 0)) + 1
        count = int(context["restore_count"])
        total = max(1, int(context.get("outer_iterations", 1)))
        if count <= total:
            local_fraction = count / total
            global_fraction = 0.42 + 0.072 * local_fraction
            label = f"K3D hard-planarity AL {count}/{total}"
        else:
            global_fraction = 0.496
            label = "K3D final planarity polish"
        _ui_progress(global_fraction, label, f"restore sweeps={used}, max plane distance={float(residual):.3g}")
        _render_live_preview(context, solved, faces, label)
        return result

    def al_with_live_progress(reference: Any, faces: Any, **kwargs: Any):
        total = max(1, int(kwargs.get("outer_iterations", 16)))
        context: dict[str, Any] = {
            "outer_iterations": total,
            "restore_count": 0,
            "preview_serial": 0,
            "preview_disabled": False,
        }
        active["context"] = context
        _ui_progress(0.42, "K3D ordinary optimizer complete", "starting hard-planarity Augmented Lagrangian")
        _render_live_preview(context, np.asarray(reference, dtype=float), np.asarray(faces, dtype=int), "Ordinary K3D before hard planarity")
        try:
            solved, metrics = original_al(reference, faces, **kwargs)
            _ui_progress(
                0.498,
                "K3D hard-planarity complete",
                f"max plane distance={float(metrics.get('k3d_hard_planarity_max_distance_after', 0.0)):.3g}",
            )
            _render_live_preview(context, solved, np.asarray(faces, dtype=int), "K3D hard-planarity result")
            return solved, metrics
        finally:
            active["context"] = None

    al_module._consensus_planarity_restore = restore_with_live_preview
    al_module._augmented_lagrangian_planarize = al_with_live_progress
    al_module._onestring_live_progress_installed = True


def install_optcuts_test2_acceleration_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test2_acceleration_installed", False):
        return

    _install_streamlit_progress_capture()
    _install_k3d_al_live_instrumentation()

    base_k3d = pipeline._optimize_k3d
    base_k2d = pipeline._optimize_k2d

    def k3d_fast(target: Any, mesh: Any, parameterization: Any, params: Any):
        is_optcuts_test = str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test"
        if _is_test2() and is_optcuts_test:
            try:
                setattr(params, "k3d_al_outer_iterations", 8)
            except Exception:
                try:
                    object.__setattr__(params, "k3d_al_outer_iterations", 8)
                except Exception:
                    pass
            print("[OPTCUTS-TEST2-K3D-FAST] AL outer_iterations=8")
        if is_optcuts_test:
            _ui_progress(0.385, "M3D -> K3D", "running ordinary K3D optimizer")
        result = base_k3d(target, mesh, parameterization, params)
        if is_optcuts_test:
            _ui_progress(0.50, "M3D -> K3D", "K3D complete")
        return result

    def k2d_full_budget(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        if _is_test2() and str(getattr(params, "omega_parameterization_mode", "")) == "optcuts_test":
            values = {
                "k2d_kinematic_outer_passes": 4,
                "k2d_kinematic_max_nfev": 70,
                "k2d_kinematic_max_collision_pairs": 2500,
            }
            for key, value in values.items():
                try:
                    setattr(params, key, value)
                except Exception:
                    try:
                        object.__setattr__(params, key, value)
                    except Exception:
                        pass
            print(
                "[OPTCUTS-TEST2-K2D-FULL] outer_passes=4 max_nfev=70 "
                "max_collision_pairs=2500; no numerical budget reduction"
            )
        return base_k2d(mesh_2d, mesh_3d, params, progress_callback=progress_callback)

    pipeline._optimize_k3d = k3d_fast
    pipeline._optimize_k2d = k2d_full_budget
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = k3d_fast
        original._optimize_k2d = k2d_full_budget
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = k3d_fast
            glb["_optimize_k2d"] = k2d_full_budget

    try:
        import scipy.optimize as spo

        original_ls = spo.least_squares
        if not getattr(original_ls, "_onestring_test2_workers_wrapper", False):
            signature = inspect.signature(original_ls)
            supports_workers = "workers" in signature.parameters
            max_workers = max(1, min(8, int(os.environ.get("ONESTRING_TEST2_WORKERS", os.cpu_count() or 1))))
            executor = ThreadPoolExecutor(max_workers=max_workers) if supports_workers else None

            def least_squares_with_test2_workers(*args: Any, **kwargs: Any):
                if _is_test2() and supports_workers and "workers" not in kwargs and executor is not None:
                    kwargs["workers"] = executor.map
                    if not getattr(least_squares_with_test2_workers, "_logged", False):
                        print(f"[OPTCUTS-TEST2-K2D-PARALLEL] least_squares workers={max_workers}")
                        least_squares_with_test2_workers._logged = True
                return original_ls(*args, **kwargs)

            least_squares_with_test2_workers._onestring_test2_workers_wrapper = True
            least_squares_with_test2_workers._logged = False
            spo.least_squares = least_squares_with_test2_workers
            print(
                "[OPTCUTS-TEST2-PARALLEL-SETUP] "
                f"least_squares_workers_supported={supports_workers} workers={max_workers if supports_workers else 1}"
            )
    except Exception as exc:
        print(f"[OPTCUTS-TEST2-PARALLEL-SETUP] unavailable: {type(exc).__name__}: {exc}")

    pipeline._onestring_optcuts_test2_acceleration_installed = True


__all__ = ["install_optcuts_test2_acceleration_patch"]
