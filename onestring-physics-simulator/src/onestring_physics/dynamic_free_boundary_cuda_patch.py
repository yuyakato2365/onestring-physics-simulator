"""Install CUDA acceleration into the existing coupled free-boundary optimizer."""
from __future__ import annotations

import contextvars
import os
from typing import Any

from .dynamic_free_boundary_cuda_backend import (
    TorchHarmonicBoundaryResponse,
    TorchOmegaAccelerator,
    _resolve_torch_device,
)

_ACTIVE_ACCELERATOR: contextvars.ContextVar[TorchOmegaAccelerator | None] = contextvars.ContextVar(
    "onestring_omega_accelerator", default=None
)
_REQUESTED_DEVICE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "onestring_omega_requested_device", default="auto"
)


def _env_device() -> str:
    value = os.getenv("ONESTRING_BIJECTIVE_DEVICE", "auto").strip().lower()
    return value if value in {"auto", "cuda", "cpu"} else "auto"


def _render_streamlit_energy_history(metrics: dict[str, Any]) -> None:
    """Show the accepted S -> Omega energy trajectory after optimization."""
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

    iterations: list[int] = []
    total_energy: list[float] = []
    shrink_energy: list[float] = []
    boundary_x: list[int] = []
    boundary_y: list[float] = []
    safe_step: list[float] = []
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
    used_iterations = int(metrics.get("optimization_iteration_count", len(records)) or len(records))
    requested_iterations = int(metrics.get("optimization_requested_max_iterations", used_iterations) or used_iterations)
    st.caption(
        f"device: {device_name} | iterations: {used_iterations}/{requested_iterations} | "
        f"termination: {termination}. Boundary-update acceptances are shown as markers."
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=total_energy,
            mode="lines",
            name="Total energy",
            hovertemplate="iteration=%{x}<br>Total E=%{y:.6g}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=shrink_energy,
            mode="lines",
            name="Shrink energy",
            hovertemplate="iteration=%{x}<br>Shrink E=%{y:.6g}<extra></extra>",
        ),
        secondary_y=True,
    )
    if boundary_x:
        fig.add_trace(
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="markers",
                name="Accepted boundary update",
                hovertemplate="boundary update at %{x}<br>Total E=%{y:.6g}<extra></extra>",
            ),
            secondary_y=False,
        )
    fig.update_xaxes(title_text="Optimization iteration")
    fig.update_yaxes(title_text="Total energy", secondary_y=False)
    fig.update_yaxes(title_text="Shrink energy", secondary_y=True)
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=25, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

    finite_safe = [value for value in safe_step if value == value and value != float("inf")]
    if finite_safe:
        st.caption(
            "Safe-step limit: "
            f"min={min(finite_safe):.3g}, median={sorted(finite_safe)[len(finite_safe)//2]:.3g}. "
            "Extremely small values indicate that near-degenerate UV triangles are still limiting motion."
        )


def install_cuda_free_boundary_acceleration(v2_module: Any) -> None:
    """Patch expensive numeric kernels while retaining the V2 outer algorithm."""
    if getattr(v2_module, "_CUDA_FREE_BOUNDARY_ACCELERATION_INSTALLED", False):
        return

    base = v2_module.base
    original_parameterization = v2_module.bijective_free_boundary_parameterization
    original_energy_gradient = v2_module._energy_gradient
    original_harmonic = v2_module.HarmonicBoundaryResponse
    original_triangle_safe_step = v2_module._triangle_safe_step
    original_area_valid = v2_module._area_valid
    original_boundary_check = base.boundary_self_intersection_count

    class ContextHarmonicBoundaryResponse:
        def __init__(self, faces, vertex_count, boundary_ids):
            requested = _REQUESTED_DEVICE.get()
            try:
                _torch, device, _dtype = _resolve_torch_device(requested)
            except Exception:
                self._backend = original_harmonic(faces, vertex_count, boundary_ids)
                self._cuda = False
                return
            if device.type != "cuda":
                self._backend = original_harmonic(faces, vertex_count, boundary_ids)
                self._cuda = False
                return
            self._backend = TorchHarmonicBoundaryResponse(
                faces,
                vertex_count,
                boundary_ids,
                device="cuda",
                cg_tolerance=1.0e-6,
                cg_max_iterations=160,
            )
            self._cuda = True

        def extend(self, direction):
            return self._backend.extend(direction)

        def __getattr__(self, name):
            return getattr(self._backend, name)

    def energy_gradient(uv, faces, inverse_surface, areas, loop, barrier_epsilon, cfg):
        accelerator = _ACTIVE_ACCELERATOR.get()
        if accelerator is None:
            requested = _REQUESTED_DEVICE.get()
            try:
                _torch, device, _dtype = _resolve_torch_device(requested)
            except Exception:
                return original_energy_gradient(
                    uv, faces, inverse_surface, areas, loop, barrier_epsilon, cfg
                )
            if device.type != "cuda":
                return original_energy_gradient(
                    uv, faces, inverse_surface, areas, loop, barrier_epsilon, cfg
                )
            accelerator = TorchOmegaAccelerator(
                faces=faces,
                inverse_surface=inverse_surface,
                surface_areas=areas,
                boundary_loop=loop,
                barrier_epsilon=barrier_epsilon,
                config=cfg,
                device="cuda",
            )
            _ACTIVE_ACCELERATOR.set(accelerator)
        return accelerator.evaluate(uv)

    def triangle_safe_step(uv, direction, faces, minimum):
        accelerator = _ACTIVE_ACCELERATOR.get()
        if accelerator is None:
            return original_triangle_safe_step(uv, direction, faces, minimum)
        return accelerator.triangle_safe_step(uv, direction, minimum)

    def area_valid(uv, faces, minimum):
        accelerator = _ACTIVE_ACCELERATOR.get()
        if accelerator is None:
            return original_area_valid(uv, faces, minimum)
        return accelerator.area_valid(uv, minimum)

    def boundary_self_intersection_count(uv, loop):
        accelerator = _ACTIVE_ACCELERATOR.get()
        if accelerator is None:
            return original_boundary_check(uv, loop)
        return accelerator.boundary_self_intersection_count(uv)

    def accelerated_parameterization(vertices, faces, config=None, progress_callback=None):
        requested = _env_device()
        try:
            torch, device, dtype = _resolve_torch_device(requested)
        except Exception:
            if requested == "cuda":
                raise
            uv, loop, metrics = original_parameterization(vertices, faces, config, progress_callback)
            metrics.update(
                {
                    "omega_cuda_used": False,
                    "omega_compute_device": "cpu",
                    "omega_device_name": "CPU",
                }
            )
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        if device.type != "cuda":
            uv, loop, metrics = original_parameterization(vertices, faces, config, progress_callback)
            metrics.update(
                {
                    "omega_cuda_used": False,
                    "omega_compute_device": str(device),
                    "omega_device_name": "CPU",
                    "omega_torch_dtype": str(dtype).replace("torch.", ""),
                }
            )
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        token_accel = _ACTIVE_ACCELERATOR.set(None)
        token_device = _REQUESTED_DEVICE.set("cuda")
        base._emit_progress(
            progress_callback,
            "S -> Omega CUDA backend",
            0.01,
            f"cuda / {torch.cuda.get_device_name(device)} / {str(dtype).replace('torch.', '')}",
        )
        try:
            uv, loop, metrics = original_parameterization(
                vertices, faces, config, progress_callback
            )
            accelerator = _ACTIVE_ACCELERATOR.get()
            metrics.update(
                {
                    "omega_cuda_used": True,
                    "omega_compute_device": str(device),
                    "omega_device_name": str(torch.cuda.get_device_name(device)),
                    "omega_torch_dtype": str(dtype).replace("torch.", ""),
                    "omega_cuda_kernel_scope": (
                        "energy_gradient+shrink+conformal+boundary_barrier+"
                        "interior_safe_step+boundary_intersection+harmonic_predictor"
                    ),
                    "omega_outer_optimizer_orchestration": "cpu_numpy",
                    "omega_final_global_overlap_audit": "cpu",
                }
            )
            if accelerator is not None:
                metrics.update(
                    {
                        "omega_cuda_energy_call_count": int(accelerator.energy_call_count),
                        "omega_cuda_energy_seconds": float(accelerator.energy_seconds),
                        "omega_cuda_safe_step_call_count": int(accelerator.safe_call_count),
                        "omega_cuda_safe_step_seconds": float(accelerator.safe_seconds),
                        "omega_cuda_boundary_check_call_count": int(accelerator.boundary_check_count),
                        "omega_cuda_boundary_check_seconds": float(accelerator.boundary_check_seconds),
                    }
                )
            metrics["flattening_backend"] = (
                "cuda_numeric_kernels+" + str(metrics.get("flattening_backend", "coupled_v2"))
            )
            base._emit_progress(
                progress_callback,
                "S -> Omega CUDA complete",
                1.0,
                (
                    f"device={metrics['omega_device_name']}; "
                    f"energy calls={metrics.get('omega_cuda_energy_call_count', 0)}; "
                    f"E={metrics.get('final_energy', 0.0):.5g}"
                ),
            )
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics
        finally:
            _ACTIVE_ACCELERATOR.reset(token_accel)
            _REQUESTED_DEVICE.reset(token_device)

    v2_module._energy_gradient = energy_gradient
    v2_module._triangle_safe_step = triangle_safe_step
    v2_module._area_valid = area_valid
    v2_module.HarmonicBoundaryResponse = ContextHarmonicBoundaryResponse
    base.boundary_self_intersection_count = boundary_self_intersection_count
    v2_module.bijective_free_boundary_parameterization = accelerated_parameterization
    v2_module._CUDA_FREE_BOUNDARY_ACCELERATION_INSTALLED = True


__all__ = ["install_cuda_free_boundary_acceleration"]
