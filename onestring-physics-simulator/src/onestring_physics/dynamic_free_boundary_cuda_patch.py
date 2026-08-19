"""Route bijective S -> Omega to a full-resident CUDA optimizer when available."""
from __future__ import annotations

import os
from typing import Any

from .dynamic_free_boundary_cuda_backend import _resolve_torch_device
from .dynamic_free_boundary_cuda_full import full_cuda_bijective_free_boundary_parameterization
from .optimization_debug_visualization import (
    capture_omega_accepted_states,
    render_omega_flip_debug_animation,
)
from .robust_floater_patch import install_robust_floater_fallback


def _env_device() -> str:
    value = os.getenv("ONESTRING_BIJECTIVE_DEVICE", "auto").strip().lower()
    return value if value in {"auto", "cuda", "cpu"} else "auto"


def _attach_floater_metrics(base: Any, metrics: dict[str, Any]) -> None:
    info = getattr(base, "_LAST_FLOATER_INITIALIZATION", None)
    if isinstance(info, dict):
        metrics.update(
            {
                "floater_initialization_mode": str(info.get("mode", "unknown")),
                "floater_fallback_used": bool(info.get("fallback_used", False)),
                "floater_primary_error": str(info.get("primary_error", "")),
            }
        )


def _render_streamlit_energy_history(metrics: dict[str, Any]) -> None:
    """Show accepted S -> Omega energy trajectory after optimization."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return
    except Exception:
        return

    records = list(metrics.get("optimization_iteration_log", []) or [])
    if not records:
        return
    iterations, total_energy, shrink_energy = [], [], []
    boundary_x, boundary_y, safe_step = [], [], []
    for row in records:
        try:
            iteration = int(row.get("iteration", 0)) + 1
            energy = float(row.get("energy", float("nan")))
            shrink = float(row.get("shrink_energy", float("nan")))
        except Exception:
            continue
        iterations.append(iteration)
        total_energy.append(energy)
        shrink_energy.append(shrink)
        try:
            safe_step.append(float(row.get("safe_step_limit", float("nan"))))
        except Exception:
            safe_step.append(float("nan"))
        if str(row.get("phase", "")) == "boundary":
            boundary_x.append(iteration)
            boundary_y.append(energy)
    if not iterations:
        return

    st.subheader("S → Ω optimization energy")
    device_name = str(metrics.get("omega_device_name", metrics.get("omega_compute_device", "CPU")))
    termination = str(metrics.get("optimization_termination_reason", "unknown"))
    used = int(metrics.get("optimization_iteration_count", len(records)) or len(records))
    requested = int(metrics.get("optimization_requested_max_iterations", used) or used)
    floater_mode = str(metrics.get("floater_initialization_mode", "mean_value_arc_length"))
    resident = bool(metrics.get("omega_full_cuda_resident", False))
    residency = "FULL GPU-RESIDENT" if resident else "CPU / fallback"
    st.caption(
        f"{residency} | device: {device_name} | iterations: {used}/{requested} | "
        f"termination: {termination} | Floater init: {floater_mode}"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=iterations, y=total_energy, mode="lines", name="Total energy",
                   hovertemplate="iteration=%{x}<br>Total E=%{y:.6g}<extra></extra>"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=shrink_energy, mode="lines", name="Shrink energy",
                   hovertemplate="iteration=%{x}<br>Shrink E=%{y:.6g}<extra></extra>"),
        secondary_y=True,
    )
    if boundary_x:
        fig.add_trace(
            go.Scatter(x=boundary_x, y=boundary_y, mode="markers", name="Accepted boundary update",
                       hovertemplate="boundary update at %{x}<br>Total E=%{y:.6g}<extra></extra>"),
            secondary_y=False,
        )
    fig.update_xaxes(title_text="Optimization iteration")
    fig.update_yaxes(title_text="Total energy", secondary_y=False)
    fig.update_yaxes(title_text="Shrink energy", secondary_y=True)
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=25, b=20), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0))
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

    finite_safe = [v for v in safe_step if v == v and v != float("inf")]
    if finite_safe:
        st.caption(
            f"Safe-step limit: min={min(finite_safe):.3g}, "
            f"median={sorted(finite_safe)[len(finite_safe)//2]:.3g}."
        )


def install_cuda_free_boundary_acceleration(v2_module: Any) -> None:
    """Use full CUDA-resident optimization on CUDA; retain original CPU solver otherwise."""
    if getattr(v2_module, "_CUDA_FREE_BOUNDARY_ACCELERATION_INSTALLED", False):
        return

    base = v2_module.base
    install_robust_floater_fallback(base)
    original_parameterization = v2_module.bijective_free_boundary_parameterization

    def accelerated_parameterization(vertices, faces, config=None, progress_callback=None):
        requested = _env_device()
        try:
            torch, device, dtype = _resolve_torch_device(requested)
        except Exception:
            if requested == "cuda":
                raise
            uv, loop, metrics = original_parameterization(vertices, faces, config, progress_callback)
            _attach_floater_metrics(base, metrics)
            metrics.update({"omega_cuda_used": False, "omega_full_cuda_resident": False,
                            "omega_compute_device": "cpu", "omega_device_name": "CPU"})
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        if device.type == "cuda":
            active_config = config if config is not None else v2_module.BijectiveFreeBoundaryConfig()
            base._emit_progress(
                progress_callback,
                "S -> Omega FULL CUDA",
                0.01,
                f"GPU-resident optimization loop / {torch.cuda.get_device_name(device)} / {str(dtype).replace('torch.', '')}",
            )
            with capture_omega_accepted_states(base, faces, active_config) as recorder:
                uv, loop, metrics = full_cuda_bijective_free_boundary_parameterization(
                    vertices, faces, active_config, progress_callback
                )
            _attach_floater_metrics(base, metrics)
            recorder.capture_final(uv, metrics)
            debug_summary = recorder.summary()
            metrics.update(debug_summary)
            # Render the orientation trace immediately after S -> Omega and
            # before later M2D/K2D pipeline stages can obscure the failure source.
            render_omega_flip_debug_animation(recorder.frames, faces, loop, debug_summary)
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        uv, loop, metrics = original_parameterization(vertices, faces, config, progress_callback)
        _attach_floater_metrics(base, metrics)
        metrics.update({
            "omega_cuda_used": False,
            "omega_full_cuda_resident": False,
            "omega_compute_device": str(device),
            "omega_device_name": "CPU",
            "omega_torch_dtype": str(dtype).replace("torch.", ""),
        })
        _render_streamlit_energy_history(metrics)
        return uv, loop, metrics

    v2_module.bijective_free_boundary_parameterization = accelerated_parameterization
    v2_module._CUDA_FREE_BOUNDARY_ACCELERATION_INSTALLED = True


__all__ = ["install_cuda_free_boundary_acceleration"]
