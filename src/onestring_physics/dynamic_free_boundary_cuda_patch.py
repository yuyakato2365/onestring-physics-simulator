"""Route bijective S -> Omega to full-resident CUDA or Apple MPS optimizers.

- NVIDIA: full CUDA-resident coupled Omega solver.
- Apple Silicon: full MPS/Metal-resident coupled Omega solver with matrix-free
  harmonic PCG (no PyTorch sparse-matrix dependency).
- Other platforms: reference CPU coupled solver.

This wrapper also removes an undesirable global scale drift from the free-boundary
result.  The optimizer is still free to change the *shape* of Omega, but if its
final accepted embedding has smaller total area than the initial Floater/Tutte
embedding, the returned UV map is uniformly enlarged about its centroid so that
its total area matches the initial embedding.  Positive uniform scaling preserves
triangle orientation and bijectivity.  The same normalization is applied to the
recorded debug snapshots so the animation visualizes shape change rather than a
meaningless global shrink.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .dynamic_free_boundary_cuda_backend import _resolve_torch_device
from .dynamic_free_boundary_cuda_full import full_cuda_bijective_free_boundary_parameterization
from .dynamic_free_boundary_mps_full import full_mps_bijective_free_boundary_parameterization
from .optimization_debug_visualization import (
    capture_omega_accepted_states,
    render_omega_flip_debug_animation,
)
from .robust_floater_patch import install_robust_floater_fallback


def _env_device() -> str:
    value = os.getenv("ONESTRING_BIJECTIVE_DEVICE", "auto").strip().lower()
    return value if value in {"auto", "cuda", "mps", "cpu"} else "auto"


def _mps_status() -> tuple[bool, str]:
    try:
        import torch
        available = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
        return available, "Apple Metal / MPS" if available else "MPS unavailable"
    except Exception:
        return False, "PyTorch unavailable"


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


def _attach_cpu_fallback_metrics(metrics: dict[str, Any], *, requested: str) -> None:
    available, label = _mps_status()
    metrics.update(
        {
            "omega_cuda_used": False,
            "omega_full_cuda_resident": False,
            "omega_full_gpu_resident": False,
            "omega_compute_device": "cpu",
            "omega_device_name": "CPU",
            "omega_mps_available": bool(available),
            "omega_mps_device_label": label,
            "omega_mps_acceleration_used": False,
            "omega_requested_device": requested,
            "omega_platform_fallback_reason": "No requested GPU backend available; using reference CPU coupled solver.",
        }
    )


def _total_uv_area(uv: Any, faces: Any) -> float:
    """Total unsigned triangle area for a valid 2D embedding."""
    points = np.asarray(uv, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    if points.ndim != 2 or points.shape[1] < 2 or len(tris) == 0:
        return 0.0
    p = points[tris]
    a = p[:, 1, :2] - p[:, 0, :2]
    b = p[:, 2, :2] - p[:, 0, :2]
    signed_double = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    finite = signed_double[np.isfinite(signed_double)]
    if not len(finite):
        return 0.0
    return 0.5 * float(np.sum(np.abs(finite)))


def _normalize_shrunken_omega_area(
    base: Any,
    uv: Any,
    faces: Any,
    loop: Any,
    metrics: dict[str, Any],
    *,
    progress_callback: Any = None,
) -> np.ndarray:
    """Undo only global shrink; never shrink an embedding that already expanded.

    Target area is the area of a fresh Floater/Tutte initialization using the same
    input mesh and configured initial boundary shape.  We reconstruct it from the
    initial-area metric when present, otherwise from the current metrics only if a
    caller has populated ``omega_initial_total_area``.  The accelerated wrappers
    set the target explicitly before calling this helper.
    """
    result = np.asarray(uv, dtype=float).copy()
    initial_area = float(metrics.get("omega_initial_total_area", 0.0) or 0.0)
    raw_area = _total_uv_area(result, faces)
    scale = 1.0
    applied = False
    if initial_area > 0.0 and raw_area > 0.0 and np.isfinite(initial_area) and np.isfinite(raw_area):
        if raw_area < initial_area * (1.0 - 1.0e-10):
            scale = float(np.sqrt(initial_area / raw_area))
            center = np.mean(result[:, :2], axis=0)
            result[:, :2] = center[None, :] + scale * (result[:, :2] - center[None, :])
            applied = True

    normalized_area = _total_uv_area(result, faces)
    metrics.update(
        {
            "omega_global_area_normalization_enabled": True,
            "omega_global_area_normalization_mode": "expand_only_to_initial_floater_area",
            "omega_global_area_normalization_applied": bool(applied),
            "omega_initial_total_area": float(initial_area),
            "omega_raw_final_total_area": float(raw_area),
            "omega_global_area_normalization_scale": float(scale),
            "omega_normalized_final_total_area": float(normalized_area),
            "omega_final_energy_is_pre_area_normalization": bool(applied),
        }
    )
    if applied:
        try:
            base._emit_progress(
                progress_callback,
                "Omega global area normalization",
                0.985,
                f"raw area={raw_area:.6g}; initial area={initial_area:.6g}; expand scale={scale:.6g}; normalized area={normalized_area:.6g}",
            )
        except Exception:
            pass
    return result


def _normalize_debug_frames_to_initial_area(frames: list[dict[str, Any]], faces: Any, initial_area: float) -> None:
    """Normalize recorded accepted states for visualization only."""
    if not frames or initial_area <= 0.0 or not np.isfinite(initial_area):
        return
    for frame in frames:
        try:
            uv = np.asarray(frame.get("uv"), dtype=float)
            raw = _total_uv_area(uv, faces)
            scale = 1.0
            normalized = uv.copy()
            if raw > 0.0 and raw < initial_area * (1.0 - 1.0e-10):
                scale = float(np.sqrt(initial_area / raw))
                center = np.mean(normalized[:, :2], axis=0)
                normalized[:, :2] = center[None, :] + scale * (normalized[:, :2] - center[None, :])
            frame["uv"] = normalized.astype(np.float32, copy=False)
            frame["raw_total_area"] = float(raw)
            frame["area_normalization_scale"] = float(scale)
            frame["normalized_total_area"] = float(_total_uv_area(normalized, faces))
        except Exception:
            continue


def _initial_floater_area(base: Any, vertices: Any, faces: Any, config: Any) -> float:
    """Compute the global area gauge used by the solver before optimization."""
    try:
        xyz = np.asarray(vertices, dtype=float)
        tris = np.asarray(faces, dtype=int)[:, :3]
        loop, _topology = base._extract_single_disk_boundary(tris, len(xyz))
        uv0 = base._tutte_embedding(xyz, tris, loop, config.initial_boundary_shape)
        return float(_total_uv_area(uv0, tris))
    except Exception:
        return 0.0


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
    requested_iterations = int(metrics.get("optimization_requested_max_iterations", used) or used)
    floater_mode = str(metrics.get("floater_initialization_mode", "mean_value_arc_length"))
    resident = bool(metrics.get("omega_full_gpu_resident", metrics.get("omega_full_cuda_resident", False)))
    residency = "FULL GPU-RESIDENT" if resident else "CPU / fallback"
    st.caption(
        f"{residency} | device: {device_name} | iterations: {used}/{requested_iterations} | "
        f"termination: {termination} | Floater init: {floater_mode}"
    )
    platform_note = str(metrics.get("omega_platform_fallback_reason", ""))
    if platform_note:
        st.caption(f"Platform note: {platform_note}")
    if bool(metrics.get("omega_global_area_normalization_enabled", False)):
        raw = float(metrics.get("omega_raw_final_total_area", 0.0) or 0.0)
        target = float(metrics.get("omega_initial_total_area", 0.0) or 0.0)
        scale = float(metrics.get("omega_global_area_normalization_scale", 1.0) or 1.0)
        final_area = float(metrics.get("omega_normalized_final_total_area", raw) or raw)
        st.caption(
            f"Global Ω area normalization: raw={raw:.6g}, initial={target:.6g}, "
            f"scale={scale:.6g}, returned={final_area:.6g} (expand-only)."
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
    """Select full CUDA, full MPS, or reference CPU coupled optimization."""
    if getattr(v2_module, "_CUDA_FREE_BOUNDARY_ACCELERATION_INSTALLED", False):
        return

    base = v2_module.base
    install_robust_floater_fallback(base)
    original_parameterization = v2_module.bijective_free_boundary_parameterization

    def accelerated_parameterization(vertices, faces, config=None, progress_callback=None):
        requested = _env_device()
        mps_available, _mps_label = _mps_status()
        active_config = config if config is not None else v2_module.BijectiveFreeBoundaryConfig()
        initial_area = _initial_floater_area(base, vertices, faces, active_config)

        def normalize_result(uv, loop, metrics):
            metrics["omega_initial_total_area"] = float(initial_area)
            return _normalize_shrunken_omega_area(
                base, uv, faces, loop, metrics, progress_callback=progress_callback
            )

        if requested == "mps" or (requested == "auto" and mps_available):
            if not mps_available:
                raise RuntimeError("ONESTRING_BIJECTIVE_DEVICE=mps was requested, but PyTorch MPS is unavailable")
            base._emit_progress(
                progress_callback,
                "S -> Omega FULL MPS",
                0.01,
                "Metal-resident optimization loop / Apple Silicon / float32",
            )
            with capture_omega_accepted_states(base, faces, active_config) as recorder:
                uv, loop, metrics = full_mps_bijective_free_boundary_parameterization(
                    vertices, faces, active_config, progress_callback
                )
            _attach_floater_metrics(base, metrics)
            uv = normalize_result(uv, loop, metrics)
            recorder.capture_final(uv, metrics)
            _normalize_debug_frames_to_initial_area(recorder.frames, faces, initial_area)
            debug_summary = recorder.summary()
            metrics.update(debug_summary)
            render_omega_flip_debug_animation(recorder.frames, faces, loop, debug_summary)
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        try:
            torch, device, dtype = _resolve_torch_device(requested)
        except Exception:
            if requested == "cuda":
                raise
            uv, loop, metrics = original_parameterization(vertices, faces, config, progress_callback)
            _attach_floater_metrics(base, metrics)
            _attach_cpu_fallback_metrics(metrics, requested=requested)
            uv = normalize_result(uv, loop, metrics)
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        if device.type == "cuda":
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
            uv = normalize_result(uv, loop, metrics)
            recorder.capture_final(uv, metrics)
            _normalize_debug_frames_to_initial_area(recorder.frames, faces, initial_area)
            debug_summary = recorder.summary()
            metrics.update(debug_summary)
            metrics.update({
                "omega_mps_available": bool(mps_available),
                "omega_mps_acceleration_used": False,
                "omega_requested_device": requested,
                "omega_full_gpu_resident": True,
            })
            render_omega_flip_debug_animation(recorder.frames, faces, loop, debug_summary)
            _render_streamlit_energy_history(metrics)
            return uv, loop, metrics

        uv, loop, metrics = original_parameterization(vertices, faces, config, progress_callback)
        _attach_floater_metrics(base, metrics)
        metrics.update({
            "omega_cuda_used": False,
            "omega_full_cuda_resident": False,
            "omega_full_gpu_resident": False,
            "omega_compute_device": str(device),
            "omega_device_name": "CPU",
            "omega_torch_dtype": str(dtype).replace("torch.", ""),
            "omega_mps_available": bool(mps_available),
            "omega_mps_acceleration_used": False,
            "omega_requested_device": requested,
        })
        uv = normalize_result(uv, loop, metrics)
        _render_streamlit_energy_history(metrics)
        return uv, loop, metrics

    v2_module.bijective_free_boundary_parameterization = accelerated_parameterization
    v2_module._CUDA_FREE_BOUNDARY_ACCELERATION_INSTALLED = True


__all__ = ["install_cuda_free_boundary_acceleration"]