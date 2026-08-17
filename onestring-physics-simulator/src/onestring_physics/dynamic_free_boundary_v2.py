"""Free-boundary optimizer with coupled boundary/interior motion.

This corrects the first dynamic_free_boundary experiment.  A boundary proposal is
no longer judged while all interior vertices are frozen.  Instead, every local
boundary displacement is extended harmonically through the interior before the
safe-step and energy tests are performed.  The expensive sparse factorization is
built once and reused for all boundary proposals.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from . import bijective_free_boundary as base
from . import dynamic_free_boundary as previous
from .reference_bff import triangle_jacobian_diagnostics

EPS = 1.0e-12

# Keep the public configuration/API compatible with the preceding experiment.
BijectiveFreeBoundaryConfig = previous.BijectiveFreeBoundaryConfig
_energy_gradient = previous._energy_gradient
_isotropic_stats = previous._isotropic_stats
_boundary_frame = previous._boundary_frame
_boundary_direction = previous._boundary_direction
_masked_lbfgs = previous._masked_lbfgs
_triangle_safe_step = previous._triangle_safe_step
_area_valid = previous._area_valid


class HarmonicBoundaryResponse:
    """Extend boundary displacement to interior vertices with a Dirichlet solve.

    A combinatorial Laplacian is sufficient here because this object transports
    *displacements*, not the parameterization itself.  For a fixed mesh topology,
    L_II is constant, so its sparse factorization is cached once.  Every later
    boundary proposal only needs two sparse triangular solves (x and y).
    """

    def __init__(self, faces: np.ndarray, vertex_count: int, boundary_ids: np.ndarray):
        started = time.perf_counter()
        self.vertex_count = int(vertex_count)
        self.boundary_ids = np.asarray(boundary_ids, dtype=int).reshape(-1)
        boundary_mask = np.zeros(self.vertex_count, dtype=bool)
        boundary_mask[self.boundary_ids] = True
        self.interior_ids = np.flatnonzero(~boundary_mask)
        self.factorization_seconds = 0.0
        self.call_count = 0
        self.solve_seconds = 0.0
        self._solve = None
        self._l_ib = None

        if not len(self.interior_ids):
            self.factorization_seconds = time.perf_counter() - started
            return

        try:
            from scipy import sparse
            from scipy.sparse import linalg as sparse_linalg
        except Exception as exc:  # pragma: no cover - project normally depends on scipy
            raise RuntimeError(
                "scipy is required for coupled harmonic boundary/interior response"
            ) from exc

        tris = np.asarray(faces, dtype=int)[:, :3]
        edges = np.vstack(
            [
                tris[:, [0, 1]],
                tris[:, [1, 2]],
                tris[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        if len(edges):
            edges = np.unique(edges, axis=0)

        degree = np.bincount(edges.reshape(-1), minlength=self.vertex_count).astype(float)
        row = np.concatenate([edges[:, 0], edges[:, 1], np.arange(self.vertex_count)])
        col = np.concatenate([edges[:, 1], edges[:, 0], np.arange(self.vertex_count)])
        data = np.concatenate(
            [
                -np.ones(2 * len(edges), dtype=float),
                degree,
            ]
        )
        laplacian = sparse.coo_matrix(
            (data, (row, col)), shape=(self.vertex_count, self.vertex_count)
        ).tocsr()

        l_ii = laplacian[self.interior_ids[:, None], self.interior_ids].tocsc()
        self._l_ib = laplacian[self.interior_ids[:, None], self.boundary_ids].tocsr()
        # Dirichlet combinatorial Laplacian on a connected disk is SPD.  factorized
        # reuses the numerical factorization for every line-search proposal.
        self._solve = sparse_linalg.factorized(l_ii)
        self.factorization_seconds = time.perf_counter() - started

    def extend(self, boundary_only_direction: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        self.call_count += 1
        supplied = np.asarray(boundary_only_direction, dtype=float)
        out = np.zeros((self.vertex_count, 2), dtype=float)
        if supplied.shape == out.shape:
            boundary_values = supplied[self.boundary_ids]
        elif supplied.shape == (len(self.boundary_ids), 2):
            boundary_values = supplied
        else:
            raise ValueError("boundary displacement has incompatible shape")
        out[self.boundary_ids] = boundary_values

        if len(self.interior_ids):
            rhs = -(self._l_ib @ boundary_values)
            out[self.interior_ids, 0] = np.asarray(self._solve(rhs[:, 0]), dtype=float)
            out[self.interior_ids, 1] = np.asarray(self._solve(rhs[:, 1]), dtype=float)

        self.solve_seconds += time.perf_counter() - started
        return out


def bijective_free_boundary_parameterization(
    vertices,
    faces,
    config: BijectiveFreeBoundaryConfig | None = None,
    progress_callback=None,
):
    cfg = config or BijectiveFreeBoundaryConfig()
    started = time.perf_counter()
    xyz = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=int)[:, :3]
    loop, topology = base._extract_single_disk_boundary(tris, len(xyz))

    base._emit_progress(
        progress_callback,
        "Coupled free-boundary initialization",
        0.05,
        f"V={len(xyz)}, F={len(tris)}, B={len(loop)}",
    )
    t0 = time.perf_counter()
    uv = base._tutte_embedding(xyz, tris, loop, cfg.initial_boundary_shape)
    init_seconds = time.perf_counter() - t0
    initial_uv = uv.copy()

    t0 = time.perf_counter()
    inverse_surface, areas = base._surface_differentials(xyz, tris)
    diff_seconds = time.perf_counter() - t0

    boundary_ids = np.asarray(loop, dtype=int)
    boundary_mask = np.zeros(len(xyz), dtype=bool)
    boundary_mask[boundary_ids] = True
    interior_mask = ~boundary_mask

    base._emit_progress(
        progress_callback,
        "Pre-factorize interior response",
        0.09,
        "Build reusable Dirichlet Laplacian factorization",
    )
    harmonic_response = HarmonicBoundaryResponse(tris, len(xyz), boundary_ids)

    boundary_xyz = xyz[boundary_ids]
    barrier_eps = max(
        0.25
        * float(
            np.mean(
                np.linalg.norm(
                    np.roll(boundary_xyz, -1, axis=0) - boundary_xyz, axis=1
                )
            )
        ),
        1e-8,
    )

    if (
        not _area_valid(uv, tris, cfg.minimum_signed_double_area)
        or base.boundary_self_intersection_count(uv, loop)
        or base._check_global_overlap(uv, tris)
    ):
        raise RuntimeError("coupled free-boundary initialization is not bijective")

    timing = {
        "energy": 0.0,
        "safe": 0.0,
        "boundary": 0.0,
        "overlap": 0.0,
    }
    counts = {
        "energy": 0,
        "safe": 0,
        "boundary": 0,
        "overlap": 1,
        "candidates": 0,
        "armijo": 0,
        "validity": 0,
    }

    def evaluate(values):
        t = time.perf_counter()
        out = _energy_gradient(
            values,
            tris,
            inverse_surface,
            areas,
            loop,
            barrier_eps,
            cfg,
        )
        timing["energy"] += time.perf_counter() - t
        counts["energy"] += 1
        return out

    def boundary_valid(values):
        if not _area_valid(values, tris, cfg.minimum_signed_double_area):
            return False
        t = time.perf_counter()
        n = base.boundary_self_intersection_count(values, loop)
        timing["boundary"] += time.perf_counter() - t
        counts["boundary"] += 1
        return n == 0

    def overlap(values):
        t = time.perf_counter()
        n = int(base._check_global_overlap(values, tris))
        timing["overlap"] += time.perf_counter() - t
        counts["overlap"] += 1
        return n

    energy, gradient, distortion, boundary_energy, shrink_energy = evaluate(uv)
    initial_energy = float(energy)
    initial_shrink = float(shrink_energy)
    initial_conformal = base._conformal_energy(uv, tris, inverse_surface, areas)
    initial_scale_min, initial_scale_p05 = _isotropic_stats(uv, tris, inverse_surface)

    history = []
    accepted_boundary = 0
    accepted_interior = 0
    rejected = 0
    boundary_moves = []
    coupled_interior_moves = []
    log = []
    converged = False
    reason = "maximum_iterations"
    small_streak = 0
    boundary_failures = 0
    max_boundary_failures = 0
    schedule = max(1, int(cfg.interior_steps_per_boundary)) + 1
    max_iterations = max(1, int(cfg.max_iterations))

    for iteration in range(max_iterations):
        gradient = np.asarray(gradient, dtype=float)
        gradient -= np.mean(gradient, axis=0, keepdims=True)
        phase = (
            "boundary"
            if (iteration + 1) % schedule == 0 or not np.any(interior_mask)
            else "interior"
        )

        if phase == "boundary":
            # 1) Build the desired *local boundary* motion from the boundary
            # gradient.  2) Extend that displacement harmonically through all
            # interior vertices.  The candidate is therefore judged only after
            # the interior has already responded to the boundary change.
            boundary_seed, requested, edge_scale = _boundary_direction(
                uv, gradient, loop, cfg
            )
            direction = harmonic_response.extend(boundary_seed)
            phase_mask = boundary_mask
            t = time.perf_counter()
            safe, safe_reason = base._safe_step_limit(
                uv,
                direction,
                tris,
                loop,
                cfg.minimum_signed_double_area,
            )
            timing["safe"] += time.perf_counter() - t
            counts["safe"] += 1
        else:
            direction = _masked_lbfgs(gradient, interior_mask, history)
            direction[boundary_mask] = 0.0
            rms = float(
                np.linalg.norm(direction[interior_mask])
                / math.sqrt(max(1, direction[interior_mask].size))
            )
            if rms > EPS:
                direction /= rms
            _, _, edge_scale = _boundary_frame(uv, loop)
            requested = max(
                0.20 * float(cfg.interior_initial_step_scale) * edge_scale,
                np.finfo(float).eps,
            )
            phase_mask = interior_mask
            t = time.perf_counter()
            safe, safe_reason = _triangle_safe_step(
                uv, direction, tris, cfg.minimum_signed_double_area
            )
            timing["safe"] += time.perf_counter() - t
            counts["safe"] += 1

        phase_rms = float(
            np.linalg.norm(gradient[phase_mask])
            / math.sqrt(max(1, gradient[phase_mask].size))
        )
        if phase_rms <= cfg.gradient_tolerance:
            if (
                float(np.linalg.norm(gradient) / math.sqrt(max(1, gradient.size)))
                <= cfg.gradient_tolerance
            ):
                converged = True
                reason = "gradient_tolerance"
                break
            continue

        derivative = float(np.sum(gradient * direction))
        if not np.isfinite(derivative) or derivative >= -1e-14:
            if phase == "interior":
                history.clear()
                continue
            boundary_failures += 1
            max_boundary_failures = max(max_boundary_failures, boundary_failures)
            if boundary_failures >= 5:
                reason = "repeated_coupled_boundary_not_descent"
                break
            continue

        step = (
            requested
            if not np.isfinite(safe)
            else min(requested, float(cfg.line_search_safety) * safe)
        )
        result = None
        for _ in range(max(1, int(cfg.line_search_max_steps))):
            if step <= np.finfo(float).eps:
                break
            counts["candidates"] += 1
            candidate = uv + step * direction
            candidate -= np.mean(candidate, axis=0, keepdims=True)
            valid = (
                _area_valid(candidate, tris, cfg.minimum_signed_double_area)
                if phase == "interior"
                else boundary_valid(candidate)
            )
            if not valid:
                counts["validity"] += 1
                rejected += 1
                step *= 0.5
                continue

            data = evaluate(candidate)
            if (
                not np.isfinite(data[0])
                or float(data[0]) > float(energy) + 1e-4 * step * derivative
            ):
                counts["armijo"] += 1
                rejected += 1
                step *= 0.5
                continue
            if cfg.validate_global_overlap_each_step and overlap(candidate):
                rejected += 1
                step *= 0.5
                continue
            result = (candidate, *data, float(step))
            break

        if result is None:
            if phase == "interior":
                history.clear()
                continue
            boundary_failures += 1
            max_boundary_failures = max(max_boundary_failures, boundary_failures)
            if boundary_failures >= 5:
                reason = "repeated_coupled_boundary_line_search_exhausted"
                break
            continue

        (
            candidate,
            new_energy,
            new_gradient,
            new_distortion,
            new_boundary,
            new_shrink,
            used_step,
        ) = result
        old_uv = uv
        old_gradient = gradient.copy()
        relative = abs(float(energy) - float(new_energy)) / max(abs(float(energy)), 1.0)

        if phase == "interior":
            ids = np.flatnonzero(interior_mask)
            s = (candidate[ids] - old_uv[ids]).reshape(-1)
            y = (new_gradient[ids] - old_gradient[ids]).reshape(-1)
            sy = float(np.dot(s, y))
            if sy > 1e-12:
                history.append((s, y, 1.0 / sy))
                history[:] = history[-max(1, int(cfg.lbfgs_history_size)) :]
            accepted_interior += 1
        else:
            # Boundary proposal already contains a harmonic interior response.
            # Clear the interior L-BFGS history because the interior has moved by
            # a different model; subsequent interior phases refine this state.
            history.clear()
            accepted_boundary += 1
            boundary_failures = 0
            boundary_delta = np.linalg.norm(
                candidate[boundary_ids] - old_uv[boundary_ids], axis=1
            )
            boundary_moves.append(
                float(np.sqrt(np.mean(boundary_delta * boundary_delta)))
            )
            if np.any(interior_mask):
                interior_delta = np.linalg.norm(
                    candidate[interior_mask] - old_uv[interior_mask], axis=1
                )
                coupled_interior_moves.append(
                    float(np.sqrt(np.mean(interior_delta * interior_delta)))
                )

        uv = candidate
        energy = float(new_energy)
        gradient = new_gradient
        distortion = float(new_distortion)
        boundary_energy = float(new_boundary)
        shrink_energy = float(new_shrink)
        small_streak = (
            small_streak + 1
            if relative <= cfg.relative_energy_tolerance
            else 0
        )

        if phase == "boundary" or iteration % max(1, max_iterations // 100) == 0:
            base._emit_progress(
                progress_callback,
                f"Coupled Omega {iteration + 1}/{max_iterations}",
                0.15 + 0.75 * (iteration + 1) / max_iterations,
                (
                    f"phase={phase}; E={energy:.5g}; shrink={shrink_energy:.4g}; "
                    f"boundary updates={accepted_boundary}"
                ),
            )
        log.append(
            {
                "iteration": iteration,
                "phase": phase,
                "energy": energy,
                "shrink_energy": shrink_energy,
                "accepted_step": used_step,
                "safe_step_limit": float(safe),
                "safe_step_reason": safe_reason,
                "boundary_candidate_interior_response": phase == "boundary",
            }
        )
        if small_streak >= 2 * schedule:
            converged = True
            reason = "relative_energy_tolerance"
            break

    base._emit_progress(
        progress_callback,
        "Final validity check",
        0.94,
        "flip / boundary / overlap",
    )
    boundary_crossings = base.boundary_self_intersection_count(uv, loop)
    overlaps = (
        overlap(uv)
        if _area_valid(uv, tris, cfg.minimum_signed_double_area)
        else -1
    )
    if (
        not _area_valid(uv, tris, cfg.minimum_signed_double_area)
        or boundary_crossings
        or overlaps
    ):
        raise RuntimeError(
            "coupled free-boundary optimization lost validity: "
            f"boundary={boundary_crossings}, overlaps={overlaps}"
        )

    diagnostics = triangle_jacobian_diagnostics(xyz, uv, tris)
    final_conformal = base._conformal_energy(uv, tris, inverse_surface, areas)
    final_scale_min, final_scale_p05 = _isotropic_stats(uv, tris, inverse_surface)
    signed = base._signed_double_areas(uv, tris)
    displacement = np.linalg.norm(
        uv[boundary_ids] - initial_uv[boundary_ids], axis=1
    )
    nonsim, nonsim_rel = base._boundary_nonsimilarity_change(
        initial_uv[boundary_ids], uv[boundary_ids]
    )
    lambdas = np.asarray(
        diagnostics["raw_lambda_uv_to_surface_sigma_max"], dtype=float
    )
    anis = np.asarray(diagnostics["anisotropy"], dtype=float)
    valid_l = lambdas[np.isfinite(lambdas) & (lambdas > 0)]
    valid_a = anis[np.isfinite(anis)]

    metrics = {
        **topology,
        "parameterization_method": "bijective_free_boundary",
        "parameterization_exactness_label": "coupled_dynamic_local_boundary_experiment",
        "flattening_backend": "coupled_boundary_harmonic_response_plus_interior_lbfgs",
        "omega_parameterization_solver": "floater_then_interior_lbfgs_and_coupled_local_boundary_harmonic_response",
        "parameterization_runtime_seconds": float(time.perf_counter() - started),
        "parameterization_warning": "" if converged else f"Stopped with {reason}; final UV remains bijective.",
        "initialization_boundary_shape": cfg.initial_boundary_shape,
        "omega_boundary_fixed": False,
        "omega_boundary_forced_rectangle": False,
        "omega_boundary_shape": "free",
        "omega_boundary_model": "independent local boundary proposal with harmonic interior response before acceptance",
        "boundary_update_model": "local_normal_dominant_boundary_seed_plus_cached_harmonic_interior_extension",
        "boundary_candidate_interior_fixed": False,
        "boundary_candidate_harmonic_interior_response": True,
        "boundary_harmonic_response_laplacian": "combinatorial_dirichlet_displacement_laplacian",
        "boundary_harmonic_factorization_reused": True,
        "boundary_harmonic_factorization_seconds": float(harmonic_response.factorization_seconds),
        "boundary_harmonic_response_call_count": int(harmonic_response.call_count),
        "boundary_harmonic_response_total_seconds": float(harmonic_response.solve_seconds),
        "boundary_global_low_frequency_basis_used": False,
        "lbfgs_low_frequency_metric_weight": 0.0,
        "line_search_initial_step_scale": float(cfg.initial_step_scale),
        "boundary_step_fraction": float(cfg.boundary_step_fraction),
        "boundary_requested_displacement_fraction_of_edge": float(cfg.boundary_step_fraction * cfg.initial_step_scale),
        "boundary_tangent_weight": float(cfg.boundary_tangent_weight),
        "boundary_normal_smoothing": float(cfg.boundary_normal_smoothing),
        "boundary_loop": list(map(int, loop)),
        "boundary_vertex_count": len(loop),
        "surface_vertex_count": len(xyz),
        "surface_triangle_count": len(tris),
        "uv_triangle_flip_count": int(diagnostics["uv_triangle_flip_count"]),
        "uv_degenerate_triangle_count": int(diagnostics["uv_degenerate_triangle_count"]),
        "uv_min_triangle_area": 0.5 * float(np.min(signed)),
        "internal_triangle_overlap_count": int(overlaps),
        "boundary_self_intersection_count": int(boundary_crossings),
        "initial_energy": initial_energy,
        "final_energy": float(energy),
        "initial_conformal_energy": float(initial_conformal),
        "final_conformal_energy": float(final_conformal),
        "conformal_energy_weight": float(cfg.conformal_weight),
        "initial_shrink_energy": initial_shrink,
        "final_shrink_energy": float(shrink_energy),
        "shrink_energy_weight": float(cfg.shrink_weight),
        "minimum_isotropic_scale_target": float(cfg.minimum_isotropic_scale),
        "initial_isotropic_scale_min": initial_scale_min,
        "initial_isotropic_scale_p05": initial_scale_p05,
        "final_isotropic_scale_min": final_scale_min,
        "final_isotropic_scale_p05": final_scale_p05,
        "boundary_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))),
        "boundary_displacement_max": float(np.max(displacement)),
        "boundary_nonsimilarity_change_rms": float(nonsim),
        "boundary_nonsimilarity_change_relative_rms": float(nonsim_rel),
        "initial_boundary_circle_fit_relative_rms": base._circle_fit_relative_rms(initial_uv[boundary_ids]),
        "final_boundary_circle_fit_relative_rms": base._circle_fit_relative_rms(uv[boundary_ids]),
        "initial_omega_boundary": initial_uv[np.asarray(loop + [loop[0]], dtype=int)].tolist(),
        "optimization_requested_max_iterations": max_iterations,
        "optimization_iteration_count": accepted_boundary + accepted_interior,
        "optimization_boundary_update_count": accepted_boundary,
        "optimization_interior_update_count": accepted_interior,
        "optimization_converged": converged,
        "optimization_termination_reason": reason,
        "optimization_rejected_line_search_step_count": rejected,
        "maximum_consecutive_boundary_line_search_failures": max_boundary_failures,
        "line_search_candidate_count": counts["candidates"],
        "armijo_rejected_candidate_count": counts["armijo"],
        "local_validity_rejected_candidate_count": counts["validity"],
        "boundary_update_accepted_displacement_rms_mean": float(np.mean(boundary_moves)) if boundary_moves else 0.0,
        "boundary_update_accepted_displacement_rms_max": float(np.max(boundary_moves)) if boundary_moves else 0.0,
        "boundary_update_coupled_interior_displacement_rms_mean": float(np.mean(coupled_interior_moves)) if coupled_interior_moves else 0.0,
        "boundary_update_coupled_interior_displacement_rms_max": float(np.max(coupled_interior_moves)) if coupled_interior_moves else 0.0,
        "energy_gradient_call_count": counts["energy"],
        "energy_gradient_total_seconds": timing["energy"],
        "safe_step_call_count": counts["safe"],
        "safe_step_total_seconds": timing["safe"],
        "boundary_self_intersection_check_call_count": counts["boundary"],
        "boundary_self_intersection_check_total_seconds": timing["boundary"],
        "overlap_check_call_count": counts["overlap"],
        "overlap_check_total_seconds": timing["overlap"],
        "floater_initialization_seconds": init_seconds,
        "tutte_initialization_seconds": init_seconds,
        "surface_differentials_seconds": diff_seconds,
        "optimization_iteration_log": log,
        "lambda_min": float(np.min(valid_l)) if len(valid_l) else 0.0,
        "lambda_median": float(np.median(valid_l)) if len(valid_l) else 0.0,
        "lambda_max": float(np.max(valid_l)) if len(valid_l) else 0.0,
        "anisotropy_mean": float(np.mean(valid_a)) if len(valid_a) else 0.0,
        "anisotropy_max": float(np.max(valid_a)) if len(valid_a) else 0.0,
        "per_triangle_lambda": lambdas.tolist(),
        "per_triangle_log_lambda": np.log(np.maximum(lambdas, EPS)).tolist(),
        "per_triangle_anisotropy": anis.tolist(),
        "onestring_grid_loss_used": False,
        "lambda_directly_optimized": False,
        "shrink_penalty_used": bool(cfg.shrink_weight > 0),
        "topology_modified": False,
        "seams_or_cuts_added": False,
    }
    base._emit_progress(
        progress_callback,
        "S -> Omega complete",
        1.0,
        (
            f"boundary updates={accepted_boundary}; interior updates={accepted_interior}; "
            f"E={energy:.5g}"
        ),
    )
    return uv, loop, metrics


def install_bijective_free_boundary(pipeline_module: Any) -> None:
    if getattr(pipeline_module, "_BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED", False):
        return
    legacy = pipeline_module._build_surface_parameterization

    def build(surface, target, grid, params):
        if str(getattr(params, "omega_parameterization_mode", "bff")) != "bijective_free_boundary":
            return legacy(surface, target, grid, params)
        if str(getattr(params, "omega_boundary_mode", "paper_default")) != "paper_default":
            raise ValueError("bijective_free_boundary requires omega_boundary_mode='paper_default'")
        cfg = BijectiveFreeBoundaryConfig(
            max_iterations=int(getattr(params, "bijective_free_boundary_max_iterations", 600)),
            gradient_tolerance=float(getattr(params, "bijective_free_boundary_gradient_tolerance", 1e-7)),
            relative_energy_tolerance=float(getattr(params, "bijective_free_boundary_energy_tolerance", 1e-9)),
            line_search_max_steps=int(getattr(params, "bijective_free_boundary_line_search_max_steps", 16)),
            line_search_safety=float(getattr(params, "bijective_free_boundary_line_search_safety", 0.9)),
            initial_step_scale=float(getattr(params, "bijective_free_boundary_initial_step_scale", 3.0)),
            conformal_weight=float(getattr(params, "bijective_free_boundary_conformal_weight", 4.0)),
            shrink_weight=float(getattr(params, "bijective_free_boundary_shrink_weight", 8.0)),
            minimum_isotropic_scale=float(getattr(params, "bijective_free_boundary_minimum_isotropic_scale", 0.70)),
            boundary_barrier_weight=float(getattr(params, "bijective_free_boundary_boundary_barrier_weight", 1.0)),
            initial_boundary_shape=str(getattr(params, "bijective_free_boundary_initial_boundary_shape", "circle")),
            interior_steps_per_boundary=int(getattr(params, "bijective_free_boundary_interior_steps_per_boundary", 2)),
            interior_initial_step_scale=float(getattr(params, "bijective_free_boundary_interior_step_scale", 1.0)),
            boundary_step_fraction=float(getattr(params, "bijective_free_boundary_boundary_step_fraction", 0.12)),
            boundary_tangent_weight=float(getattr(params, "bijective_free_boundary_boundary_tangent_weight", 0.20)),
            boundary_normal_smoothing=float(getattr(params, "bijective_free_boundary_boundary_normal_smoothing", 0.18)),
        )
        vertices = np.asarray(surface.vertices, dtype=float)
        surface_faces = np.asarray(surface.faces, dtype=int)[:, :3]
        uv, loop, metrics = bijective_free_boundary_parameterization(
            vertices,
            surface_faces,
            cfg,
            getattr(params, "_bijective_free_boundary_progress_callback", None),
        )
        slope = (
            {"mean_slope": 0.0, "max_slope": 0.0}
            if getattr(target, "kind", "") == "sampled"
            else pipeline_module._original._heightfield_metric_summary(target, grid)
        )
        metrics.update(
            {
                "mean_slope": float(slope["mean_slope"]),
                "max_slope": float(slope["max_slope"]),
                "height_field_shortcut_used": False,
                "omega_corresponds_to_S": True,
                "omega_correspondence_model": "coupled dynamic bijective free-boundary map c:S->Omega",
                "paper_flow_stage": "S -> Omega by coupled local free-boundary optimization",
                "paper_exactness_warning": "Experimental coupled optimizer; not the Smith-Schaefer reference implementation.",
                "omega_warning": str(metrics.get("parameterization_warning", "")),
            }
        )
        output = pipeline_module._original.SurfaceParameterization(
            method="bijective_free_boundary",
            surface_vertices_3d=vertices,
            surface_faces=surface_faces,
            uv_vertices_2d=uv,
            uv_faces=surface_faces.copy(),
            omega_boundary=uv[np.asarray(loop + [loop[0]], dtype=int)],
            triangle_acceleration=None,
            metrics=metrics,
        )
        marker = getattr(pipeline_module, "_mark_parameterization_mode", None)
        if callable(marker):
            return marker(
                output,
                method="bijective_free_boundary",
                exactness="coupled_dynamic_local_free_boundary_experiment",
                warning=str(metrics.get("parameterization_warning", "")),
            )
        return output

    pipeline_module._build_surface_parameterization = build
    pipeline_module._original._build_surface_parameterization = build
    pipeline_module._BIJECTIVE_FREE_BOUNDARY_PATCH_INSTALLED = True


__all__ = [
    "BijectiveFreeBoundaryConfig",
    "HarmonicBoundaryResponse",
    "bijective_free_boundary_parameterization",
    "install_bijective_free_boundary",
    "_boundary_direction",
    "_energy_gradient",
]
