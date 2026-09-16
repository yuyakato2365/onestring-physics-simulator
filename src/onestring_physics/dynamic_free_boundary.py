"""Dynamic local free-boundary optimizer used by the 0.5.0 experiment.

Boundary and interior vertices are optimized in alternating blocks.  Boundary
updates are independent local normal/tangent motions; the old global polynomial
low-frequency direction is deliberately not used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np

from . import bijective_free_boundary as base
from .reference_bff import triangle_jacobian_diagnostics

EPS = 1.0e-12
ProgressCallback = Callable[[str, float, str], None]


@dataclass(frozen=True)
class BijectiveFreeBoundaryConfig:
    max_iterations: int = 600
    gradient_tolerance: float = 1e-7
    relative_energy_tolerance: float = 1e-9
    line_search_max_steps: int = 16
    line_search_safety: float = 0.9
    initial_step_scale: float = 3.0
    conformal_weight: float = 4.0
    shrink_weight: float = 8.0
    minimum_isotropic_scale: float = 0.90
    boundary_barrier_weight: float = 1.0
    minimum_signed_double_area: float = 1e-12
    initial_boundary_shape: str = "circle"
    interior_steps_per_boundary: int = 2
    lbfgs_history_size: int = 8
    interior_initial_step_scale: float = 1.0
    boundary_step_fraction: float = 0.12
    boundary_tangent_weight: float = 0.20
    boundary_normal_smoothing: float = 0.18
    boundary_gradient_clip_factor: float = 8.0
    low_frequency_metric_weight: float = 0.0
    validate_global_overlap_each_step: bool = False


def _energy_gradient(uv, faces, inverse_surface, areas, loop, barrier_epsilon, cfg):
    total, gradient, distortion, boundary = base._energy_and_gradient(
        uv, faces, inverse_surface, areas, loop, barrier_epsilon,
        cfg.boundary_barrier_weight, cfg.conformal_weight,
    )
    tris = np.asarray(faces, int)[:, :3]
    tri_uv = np.asarray(uv, float)[tris]
    d_uv = np.stack([tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]], axis=2)
    jac = d_uv @ inverse_surface
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    if np.any(det <= EPS) or not np.all(np.isfinite(det)):
        return math.inf, np.zeros_like(uv), math.inf, math.inf, math.inf
    scale = np.sqrt(det)
    deficit = np.maximum(float(cfg.minimum_isotropic_scale) - scale, 0.0)
    shrink = float(cfg.shrink_weight) * float(np.sum(areas * deficit * deficit))
    active = deficit > 0.0
    if cfg.shrink_weight > 0.0 and np.any(active):
        inv_t = np.empty_like(jac)
        inv_t[:, 0, 0] = jac[:, 1, 1] / det
        inv_t[:, 0, 1] = -jac[:, 1, 0] / det
        inv_t[:, 1, 0] = -jac[:, 0, 1] / det
        inv_t[:, 1, 1] = jac[:, 0, 0] / det
        coef = float(cfg.shrink_weight) * areas * (scale - cfg.minimum_isotropic_scale) * scale
        coef = np.where(active, coef, 0.0)
        jgrad = coef[:, None, None] * inv_t
        uvgrad = jgrad @ np.swapaxes(inverse_surface, 1, 2)
        np.add.at(gradient, tris[:, 1], uvgrad[:, :, 0])
        np.add.at(gradient, tris[:, 2], uvgrad[:, :, 1])
        np.add.at(gradient, tris[:, 0], -(uvgrad[:, :, 0] + uvgrad[:, :, 1]))
    return float(total + shrink), gradient, float(distortion), float(boundary), float(shrink)


def _isotropic_stats(uv, faces, inverse_surface):
    tris = np.asarray(faces, int)[:, :3]
    tri_uv = np.asarray(uv, float)[tris]
    d_uv = np.stack([tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]], axis=2)
    jac = d_uv @ inverse_surface
    det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    values = np.sqrt(np.maximum(det, 0.0))
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 0.0
    return float(np.min(values)), float(np.percentile(values, 5.0))


def _boundary_frame(uv, loop):
    ids = np.asarray(loop, int)
    p = np.asarray(uv, float)[ids]
    tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], EPS)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    if area < 0.0:
        normal *= -1.0
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    valid = lengths[np.isfinite(lengths) & (lengths > EPS)]
    return tangent, normal, float(np.median(valid)) if len(valid) else 1.0


def _boundary_direction(uv, gradient, loop, cfg):
    ids = np.asarray(loop, int)
    tangent, normal, edge_scale = _boundary_frame(uv, loop)
    descent = -np.asarray(gradient, float)[ids]
    n = np.sum(descent * normal, axis=1)
    t = np.sum(descent * tangent, axis=1)
    finite = np.abs(n[np.isfinite(n)])
    if len(finite):
        cap = max(float(np.median(finite)) * cfg.boundary_gradient_clip_factor,
                  float(np.percentile(finite, 90.0)), EPS)
        n = np.clip(n, -cap, cap)
    smooth = float(np.clip(cfg.boundary_normal_smoothing, 0.0, 0.95))
    if smooth > 0.0 and len(n) >= 3:
        avg = 0.25 * np.roll(n, 1) + 0.5 * n + 0.25 * np.roll(n, -1)
        n = (1.0 - smooth) * n + smooth * avg
    vector = n[:, None] * normal + float(cfg.boundary_tangent_weight) * t[:, None] * tangent
    direction = np.zeros_like(gradient)
    direction[ids] = vector
    if float(np.sum(gradient * direction)) >= -1e-14:
        direction[:] = 0.0
        direction[ids] = descent
    rms = float(np.linalg.norm(direction[ids]) / math.sqrt(max(1, direction[ids].size)))
    if rms > EPS:
        direction /= rms
    requested = (float(cfg.boundary_step_fraction) * max(float(cfg.initial_step_scale), EPS) * edge_scale)
    return direction, max(requested, np.finfo(float).eps), edge_scale


def _masked_lbfgs(gradient, mask, history):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return np.zeros_like(gradient)
    q = np.asarray(gradient, float)[ids].reshape(-1).copy()
    alphas = []
    for s, y, rho in reversed(history):
        a = rho * float(np.dot(s, q)); alphas.append(a); q -= a * y
    if history:
        s, y, _ = history[-1]
        yy = float(np.dot(y, y))
        q *= max(float(np.dot(s, y)) / yy if yy > EPS else 1.0, 1e-10)
    for (s, y, rho), a in zip(history, reversed(alphas)):
        q += s * (a - rho * float(np.dot(y, q)))
    out = np.zeros_like(gradient)
    out[ids] = -q.reshape((-1, 2))
    return out


def _triangle_safe_step(uv, direction, faces, minimum):
    c = np.asarray(uv, float); d = np.asarray(direction, float); tris = np.asarray(faces, int)[:, :3]
    p = c[tris]; v = d[tris]
    a = p[:, 1] - p[:, 0]; b = p[:, 2] - p[:, 0]
    da = v[:, 1] - v[:, 0]; db = v[:, 2] - v[:, 0]
    roots = base._positive_quadratic_roots_array(
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] - float(minimum),
        da[:, 0] * b[:, 1] - da[:, 1] * b[:, 0] + a[:, 0] * db[:, 1] - a[:, 1] * db[:, 0],
        da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0],
    )
    limit = float(np.min(roots[..., 0])) if len(tris) else math.inf
    return limit, "triangle_degeneracy" if np.isfinite(limit) else "unbounded"


def _area_valid(uv, faces, minimum):
    return bool(np.all(np.isfinite(uv)) and np.all(base._signed_double_areas(uv, faces) > float(minimum)))


def bijective_free_boundary_parameterization(vertices, faces, config=None, progress_callback=None):
    cfg = config or BijectiveFreeBoundaryConfig()
    started = time.perf_counter()
    xyz = np.asarray(vertices, float); tris = np.asarray(faces, int)[:, :3]
    loop, topology = base._extract_single_disk_boundary(tris, len(xyz))
    base._emit_progress(progress_callback, "Dynamic Floater initialization", 0.05, f"V={len(xyz)}, F={len(tris)}, B={len(loop)}")
    t0 = time.perf_counter(); uv = base._tutte_embedding(xyz, tris, loop, cfg.initial_boundary_shape); init_seconds = time.perf_counter() - t0
    initial_uv = uv.copy()
    t0 = time.perf_counter(); inverse_surface, areas = base._surface_differentials(xyz, tris); diff_seconds = time.perf_counter() - t0
    boundary_ids = np.asarray(loop, int)
    boundary_xyz = xyz[boundary_ids]
    barrier_eps = max(0.25 * float(np.mean(np.linalg.norm(np.roll(boundary_xyz, -1, axis=0) - boundary_xyz, axis=1))), 1e-8)
    if not _area_valid(uv, tris, cfg.minimum_signed_double_area) or base.boundary_self_intersection_count(uv, loop) or base._check_global_overlap(uv, tris):
        raise RuntimeError("dynamic free-boundary initialization is not bijective")

    timing = {"energy": 0.0, "safe": 0.0, "boundary": 0.0, "overlap": 0.0}
    counts = {"energy": 0, "safe": 0, "boundary": 0, "overlap": 1, "candidates": 0, "armijo": 0, "validity": 0}
    def evaluate(values):
        t = time.perf_counter(); out = _energy_gradient(values, tris, inverse_surface, areas, loop, barrier_eps, cfg); timing["energy"] += time.perf_counter() - t; counts["energy"] += 1; return out
    def boundary_valid(values):
        if not _area_valid(values, tris, cfg.minimum_signed_double_area): return False
        t = time.perf_counter(); n = base.boundary_self_intersection_count(values, loop); timing["boundary"] += time.perf_counter() - t; counts["boundary"] += 1; return n == 0
    def overlap(values):
        t = time.perf_counter(); n = int(base._check_global_overlap(values, tris)); timing["overlap"] += time.perf_counter() - t; counts["overlap"] += 1; return n

    energy, gradient, distortion, boundary_energy, shrink_energy = evaluate(uv)
    initial_energy, initial_shrink = float(energy), float(shrink_energy)
    initial_conformal = base._conformal_energy(uv, tris, inverse_surface, areas)
    initial_scale_min, initial_scale_p05 = _isotropic_stats(uv, tris, inverse_surface)
    boundary_mask = np.zeros(len(xyz), bool); boundary_mask[boundary_ids] = True; interior_mask = ~boundary_mask
    history = []; accepted_boundary = accepted_interior = rejected = 0
    boundary_moves = []; log = []; converged = False; reason = "maximum_iterations"; small_streak = boundary_failures = max_boundary_failures = 0
    schedule = max(1, int(cfg.interior_steps_per_boundary)) + 1
    max_iterations = max(1, int(cfg.max_iterations))

    for iteration in range(max_iterations):
        gradient = np.asarray(gradient, float); gradient -= np.mean(gradient, axis=0, keepdims=True)
        phase = "boundary" if (iteration + 1) % schedule == 0 or not np.any(interior_mask) else "interior"
        if phase == "boundary":
            direction, requested, edge_scale = _boundary_direction(uv, gradient, loop, cfg)
            phase_mask = boundary_mask
            t = time.perf_counter(); safe, safe_reason = base._safe_step_limit(uv, direction, tris, loop, cfg.minimum_signed_double_area); timing["safe"] += time.perf_counter() - t; counts["safe"] += 1
        else:
            direction = _masked_lbfgs(gradient, interior_mask, history); direction[boundary_mask] = 0.0
            rms = float(np.linalg.norm(direction[interior_mask]) / math.sqrt(max(1, direction[interior_mask].size)))
            if rms > EPS: direction /= rms
            _, _, edge_scale = _boundary_frame(uv, loop)
            requested = max(0.20 * float(cfg.interior_initial_step_scale) * edge_scale, np.finfo(float).eps)
            phase_mask = interior_mask
            t = time.perf_counter(); safe, safe_reason = _triangle_safe_step(uv, direction, tris, cfg.minimum_signed_double_area); timing["safe"] += time.perf_counter() - t; counts["safe"] += 1
        phase_rms = float(np.linalg.norm(gradient[phase_mask]) / math.sqrt(max(1, gradient[phase_mask].size)))
        if phase_rms <= cfg.gradient_tolerance:
            if float(np.linalg.norm(gradient) / math.sqrt(max(1, gradient.size))) <= cfg.gradient_tolerance:
                converged = True; reason = "gradient_tolerance"; break
            continue
        derivative = float(np.sum(gradient * direction))
        if not np.isfinite(derivative) or derivative >= -1e-14:
            if phase == "interior": history.clear(); continue
            boundary_failures += 1; max_boundary_failures = max(max_boundary_failures, boundary_failures)
            if boundary_failures >= 5: reason = "repeated_boundary_not_descent"; break
            continue
        step = requested if not np.isfinite(safe) else min(requested, float(cfg.line_search_safety) * safe)
        result = None
        for _ in range(max(1, int(cfg.line_search_max_steps))):
            if step <= np.finfo(float).eps: break
            counts["candidates"] += 1
            candidate = uv + step * direction; candidate -= np.mean(candidate, axis=0, keepdims=True)
            valid = _area_valid(candidate, tris, cfg.minimum_signed_double_area) if phase == "interior" else boundary_valid(candidate)
            if not valid: counts["validity"] += 1; rejected += 1; step *= 0.5; continue
            data = evaluate(candidate)
            if not np.isfinite(data[0]) or float(data[0]) > float(energy) + 1e-4 * step * derivative:
                counts["armijo"] += 1; rejected += 1; step *= 0.5; continue
            if cfg.validate_global_overlap_each_step and overlap(candidate): rejected += 1; step *= 0.5; continue
            result = (candidate, *data, float(step)); break
        if result is None:
            if phase == "interior": history.clear(); continue
            boundary_failures += 1; max_boundary_failures = max(max_boundary_failures, boundary_failures)
            if boundary_failures >= 5: reason = "repeated_boundary_line_search_exhausted"; break
            continue

        candidate, new_energy, new_gradient, new_distortion, new_boundary, new_shrink, used_step = result
        old_uv, old_gradient = uv, gradient.copy()
        relative = abs(float(energy) - float(new_energy)) / max(abs(float(energy)), 1.0)
        if phase == "interior":
            ids = np.flatnonzero(interior_mask); s = (candidate[ids] - old_uv[ids]).reshape(-1); y = (new_gradient[ids] - old_gradient[ids]).reshape(-1); sy = float(np.dot(s, y))
            if sy > 1e-12:
                history.append((s, y, 1.0 / sy)); history[:] = history[-max(1, int(cfg.lbfgs_history_size)):]
            accepted_interior += 1
        else:
            history.clear(); accepted_boundary += 1; boundary_failures = 0
            move = np.linalg.norm(candidate[boundary_ids] - old_uv[boundary_ids], axis=1); boundary_moves.append(float(np.sqrt(np.mean(move * move))))
        uv, energy, gradient, distortion, boundary_energy, shrink_energy = candidate, float(new_energy), new_gradient, float(new_distortion), float(new_boundary), float(new_shrink)
        small_streak = small_streak + 1 if relative <= cfg.relative_energy_tolerance else 0
        if phase == "boundary" or iteration % max(1, max_iterations // 100) == 0:
            base._emit_progress(progress_callback, f"Dynamic Omega {iteration + 1}/{max_iterations}", 0.15 + 0.75 * (iteration + 1) / max_iterations, f"phase={phase}; E={energy:.5g}; shrink={shrink_energy:.4g}; boundary updates={accepted_boundary}")
        log.append({"iteration": iteration, "phase": phase, "energy": energy, "shrink_energy": shrink_energy, "accepted_step": used_step, "safe_step_limit": float(safe), "safe_step_reason": safe_reason})
        if small_streak >= 2 * schedule: converged = True; reason = "relative_energy_tolerance"; break

    base._emit_progress(progress_callback, "Final validity check", 0.94, "flip / boundary / overlap")
    boundary_crossings = base.boundary_self_intersection_count(uv, loop)
    overlaps = overlap(uv) if _area_valid(uv, tris, cfg.minimum_signed_double_area) else -1
    if not _area_valid(uv, tris, cfg.minimum_signed_double_area) or boundary_crossings or overlaps:
        raise RuntimeError(f"dynamic free-boundary optimization lost validity: boundary={boundary_crossings}, overlaps={overlaps}")
    diagnostics = triangle_jacobian_diagnostics(xyz, uv, tris)
    final_conformal = base._conformal_energy(uv, tris, inverse_surface, areas)
    final_scale_min, final_scale_p05 = _isotropic_stats(uv, tris, inverse_surface)
    signed = base._signed_double_areas(uv, tris)
    displacement = np.linalg.norm(uv[boundary_ids] - initial_uv[boundary_ids], axis=1)
    nonsim, nonsim_rel = base._boundary_nonsimilarity_change(initial_uv[boundary_ids], uv[boundary_ids])
    lambdas = np.asarray(diagnostics["raw_lambda_uv_to_surface_sigma_max"], float); anis = np.asarray(diagnostics["anisotropy"], float)
    valid_l = lambdas[np.isfinite(lambdas) & (lambdas > 0)]; valid_a = anis[np.isfinite(anis)]
    metrics = {
        **topology,
        "parameterization_method": "bijective_free_boundary",
        "parameterization_exactness_label": "dynamic_local_boundary_block_experiment",
        "flattening_backend": "dynamic_boundary_interior_block_symmetric_dirichlet",
        "omega_parameterization_solver": "floater_then_alternating_interior_lbfgs_and_local_boundary_updates",
        "parameterization_runtime_seconds": float(time.perf_counter() - started),
        "parameterization_warning": "" if converged else f"Stopped with {reason}; final UV remains bijective.",
        "initialization_boundary_shape": cfg.initial_boundary_shape,
        "omega_boundary_fixed": False, "omega_boundary_forced_rectangle": False, "omega_boundary_shape": "free",
        "omega_boundary_model": "independent local normal/tangent boundary updates plus interior relaxation",
        "boundary_update_model": "local_normal_dominant_block_step",
        "boundary_global_low_frequency_basis_used": False,
        "lbfgs_low_frequency_metric_weight": 0.0,
        "line_search_initial_step_scale": float(cfg.initial_step_scale),
        "boundary_step_fraction": float(cfg.boundary_step_fraction),
        "boundary_requested_displacement_fraction_of_edge": float(cfg.boundary_step_fraction * cfg.initial_step_scale),
        "boundary_tangent_weight": float(cfg.boundary_tangent_weight), "boundary_normal_smoothing": float(cfg.boundary_normal_smoothing),
        "boundary_loop": list(map(int, loop)), "boundary_vertex_count": len(loop), "surface_vertex_count": len(xyz), "surface_triangle_count": len(tris),
        "uv_triangle_flip_count": int(diagnostics["uv_triangle_flip_count"]), "uv_degenerate_triangle_count": int(diagnostics["uv_degenerate_triangle_count"]),
        "uv_min_triangle_area": 0.5 * float(np.min(signed)), "internal_triangle_overlap_count": int(overlaps), "boundary_self_intersection_count": int(boundary_crossings),
        "initial_energy": initial_energy, "final_energy": float(energy), "initial_conformal_energy": float(initial_conformal), "final_conformal_energy": float(final_conformal),
        "conformal_energy_weight": float(cfg.conformal_weight), "initial_shrink_energy": initial_shrink, "final_shrink_energy": float(shrink_energy),
        "shrink_energy_weight": float(cfg.shrink_weight), "minimum_isotropic_scale_target": float(cfg.minimum_isotropic_scale),
        "initial_isotropic_scale_min": initial_scale_min, "initial_isotropic_scale_p05": initial_scale_p05, "final_isotropic_scale_min": final_scale_min, "final_isotropic_scale_p05": final_scale_p05,
        "boundary_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))), "boundary_displacement_max": float(np.max(displacement)),
        "boundary_nonsimilarity_change_rms": float(nonsim), "boundary_nonsimilarity_change_relative_rms": float(nonsim_rel),
        "initial_boundary_circle_fit_relative_rms": base._circle_fit_relative_rms(initial_uv[boundary_ids]), "final_boundary_circle_fit_relative_rms": base._circle_fit_relative_rms(uv[boundary_ids]),
        "initial_omega_boundary": initial_uv[np.asarray(loop + [loop[0]], int)].tolist(),
        "optimization_requested_max_iterations": max_iterations, "optimization_iteration_count": accepted_boundary + accepted_interior,
        "optimization_boundary_update_count": accepted_boundary, "optimization_interior_update_count": accepted_interior, "optimization_converged": converged, "optimization_termination_reason": reason,
        "optimization_rejected_line_search_step_count": rejected, "maximum_consecutive_boundary_line_search_failures": max_boundary_failures,
        "line_search_candidate_count": counts["candidates"], "armijo_rejected_candidate_count": counts["armijo"], "local_validity_rejected_candidate_count": counts["validity"],
        "boundary_update_accepted_displacement_rms_mean": float(np.mean(boundary_moves)) if boundary_moves else 0.0, "boundary_update_accepted_displacement_rms_max": float(np.max(boundary_moves)) if boundary_moves else 0.0,
        "energy_gradient_call_count": counts["energy"], "energy_gradient_total_seconds": timing["energy"], "safe_step_call_count": counts["safe"], "safe_step_total_seconds": timing["safe"],
        "boundary_self_intersection_check_call_count": counts["boundary"], "boundary_self_intersection_check_total_seconds": timing["boundary"], "overlap_check_call_count": counts["overlap"], "overlap_check_total_seconds": timing["overlap"],
        "floater_initialization_seconds": init_seconds, "tutte_initialization_seconds": init_seconds, "surface_differentials_seconds": diff_seconds, "optimization_iteration_log": log,
        "lambda_min": float(np.min(valid_l)) if len(valid_l) else 0.0, "lambda_median": float(np.median(valid_l)) if len(valid_l) else 0.0, "lambda_max": float(np.max(valid_l)) if len(valid_l) else 0.0,
        "anisotropy_mean": float(np.mean(valid_a)) if len(valid_a) else 0.0, "anisotropy_max": float(np.max(valid_a)) if len(valid_a) else 0.0,
        "per_triangle_lambda": lambdas.tolist(), "per_triangle_log_lambda": np.log(np.maximum(lambdas, EPS)).tolist(), "per_triangle_anisotropy": anis.tolist(),
        "onestring_grid_loss_used": False, "lambda_directly_optimized": False, "shrink_penalty_used": bool(cfg.shrink_weight > 0), "topology_modified": False, "seams_or_cuts_added": False,
    }
    base._emit_progress(progress_callback, "S -> Omega complete", 1.0, f"boundary updates={accepted_boundary}; interior updates={accepted_interior}; E={energy:.5g}")
    return uv, loop, metrics


def install_bijective_free_boundary(pipeline_module: Any) -> None:
    if getattr(pipeline_module, "_BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED", False): return
    legacy = pipeline_module._build_surface_parameterization
    def build(surface, target, grid, params):
        if str(getattr(params, "omega_parameterization_mode", "bff")) != "bijective_free_boundary": return legacy(surface, target, grid, params)
        if str(getattr(params, "omega_boundary_mode", "paper_default")) != "paper_default": raise ValueError("bijective_free_boundary requires omega_boundary_mode='paper_default'")
        cfg = BijectiveFreeBoundaryConfig(
            max_iterations=int(getattr(params, "bijective_free_boundary_max_iterations", 600)),
            gradient_tolerance=float(getattr(params, "bijective_free_boundary_gradient_tolerance", 1e-7)),
            relative_energy_tolerance=float(getattr(params, "bijective_free_boundary_energy_tolerance", 1e-9)),
            line_search_max_steps=int(getattr(params, "bijective_free_boundary_line_search_max_steps", 16)), line_search_safety=float(getattr(params, "bijective_free_boundary_line_search_safety", 0.9)),
            initial_step_scale=float(getattr(params, "bijective_free_boundary_initial_step_scale", 3.0)), conformal_weight=float(getattr(params, "bijective_free_boundary_conformal_weight", 4.0)),
            shrink_weight=float(getattr(params, "bijective_free_boundary_shrink_weight", 8.0)), minimum_isotropic_scale=float(getattr(params, "bijective_free_boundary_minimum_isotropic_scale", 0.90)),
            boundary_barrier_weight=float(getattr(params, "bijective_free_boundary_boundary_barrier_weight", 1.0)), initial_boundary_shape=str(getattr(params, "bijective_free_boundary_initial_boundary_shape", "circle")),
            interior_steps_per_boundary=int(getattr(params, "bijective_free_boundary_interior_steps_per_boundary", 2)), interior_initial_step_scale=float(getattr(params, "bijective_free_boundary_interior_step_scale", 1.0)),
            boundary_step_fraction=float(getattr(params, "bijective_free_boundary_boundary_step_fraction", 0.12)), boundary_tangent_weight=float(getattr(params, "bijective_free_boundary_boundary_tangent_weight", 0.20)), boundary_normal_smoothing=float(getattr(params, "bijective_free_boundary_boundary_normal_smoothing", 0.18)),
        )
        vertices = np.asarray(surface.vertices, float); faces = np.asarray(surface.faces, int)[:, :3]
        uv, loop, metrics = bijective_free_boundary_parameterization(vertices, faces, cfg, getattr(params, "_bijective_free_boundary_progress_callback", None))
        slope = {"mean_slope": 0.0, "max_slope": 0.0} if getattr(target, "kind", "") == "sampled" else pipeline_module._original._heightfield_metric_summary(target, grid)
        metrics.update({"mean_slope": float(slope["mean_slope"]), "max_slope": float(slope["max_slope"]), "height_field_shortcut_used": False, "omega_corresponds_to_S": True, "omega_correspondence_model": "dynamic bijective free-boundary map c:S->Omega", "paper_flow_stage": "S -> Omega by dynamic local free-boundary optimization", "paper_exactness_warning": "Experimental block optimizer; not the Smith-Schaefer reference implementation.", "omega_warning": str(metrics.get("parameterization_warning", ""))})
        output = pipeline_module._original.SurfaceParameterization(method="bijective_free_boundary", surface_vertices_3d=vertices, surface_faces=faces, uv_vertices_2d=uv, uv_faces=faces.copy(), omega_boundary=uv[np.asarray(loop + [loop[0]], int)], triangle_acceleration=None, metrics=metrics)
        marker = getattr(pipeline_module, "_mark_parameterization_mode", None)
        return marker(output, method="bijective_free_boundary", exactness="dynamic_local_free_boundary_experiment", warning=str(metrics.get("parameterization_warning", ""))) if callable(marker) else output
    pipeline_module._build_surface_parameterization = build; pipeline_module._original._build_surface_parameterization = build; pipeline_module._BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED = True


__all__ = ["BijectiveFreeBoundaryConfig", "bijective_free_boundary_parameterization", "install_bijective_free_boundary", "_boundary_direction", "_energy_gradient"]
