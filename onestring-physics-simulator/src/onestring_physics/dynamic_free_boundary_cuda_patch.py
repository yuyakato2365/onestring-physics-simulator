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
            return original_parameterization(vertices, faces, config, progress_callback)

        if device.type != "cuda":
            return original_parameterization(vertices, faces, config, progress_callback)

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
