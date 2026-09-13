"""Scale-aware outer selection for official OptCuts seam solutions.

The upstream OptCuts executable exposes seam-vs-distortion optimization but does
not expose a callback for adding a OneString-specific scale-factor term to each
internal topology operation.  This patch therefore keeps the upstream binary
unchanged and performs the closest reproducible outer optimization available
through its public CLI:

1. run official OptCuts at several progressively tighter Symmetric-Dirichlet
   distortion bounds;
2. measure each returned seam length, actual distortion, and the OneString local
   scale-factor range R=max(lambda)/min(lambda), lambda=sigma_max(J_UV->S);
3. score candidates with a soft objective

       E = E_seam + w_d E_distortion + w_s [max(0, log(R / 2))]^2;

4. prefer hard-feasible candidates (R <= 2) when any exist, otherwise take the
   minimum soft score and leave the final hard-feasibility flag false.

This is intentionally labelled an *outer selection*, not a modification of the
OptCuts C++ topology optimizer.  It is enabled only for the dated/test2 route so
the ordinary OptCuts baseline and optcuts_test remain unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
from typing import Any

import numpy as np

from .optcuts_backend import OptCutsConfig, OptCutsResult, run_official_optcuts


SCALE_BOUND = 2.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _triangle_local_basis(tri: np.ndarray) -> np.ndarray | None:
    p0, p1, p2 = np.asarray(tri, dtype=float)
    e1 = p1 - p0
    l1 = float(np.linalg.norm(e1))
    if not np.isfinite(l1) or l1 <= 1.0e-14:
        return None
    x = e1 / l1
    e2 = p2 - p0
    yv = e2 - float(np.dot(e2, x)) * x
    l2 = float(np.linalg.norm(yv))
    if not np.isfinite(l2) or l2 <= 1.0e-14:
        return None
    y = yv / l2
    return np.asarray(
        [
            [float(np.dot(e1, x)), float(np.dot(e2, x))],
            [float(np.dot(e1, y)), float(np.dot(e2, y))],
        ],
        dtype=float,
    )


def _raw_scale_factors(result: OptCutsResult) -> np.ndarray:
    """Return lambda=sigma_max(J_UV->S) for corresponding triangles."""
    xyz = np.asarray(result.surface_vertices_3d, dtype=float)
    sf = np.asarray(result.surface_faces, dtype=int)[:, :3]
    uv = np.asarray(result.uv_vertices_2d, dtype=float)
    uf = np.asarray(result.uv_faces, dtype=int)[:, :3]
    out = np.full(len(sf), np.nan, dtype=float)
    for i, (f3, f2) in enumerate(zip(sf, uf)):
        ds = _triangle_local_basis(xyz[f3])
        if ds is None:
            continue
        tri2 = uv[f2]
        duv = np.asarray([tri2[1] - tri2[0], tri2[2] - tri2[0]], dtype=float).T
        if not np.isfinite(duv).all() or abs(float(np.linalg.det(duv))) <= 1.0e-14:
            continue
        try:
            jac = ds @ np.linalg.inv(duv)
            s = np.linalg.svd(jac, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        if len(s) and np.all(np.isfinite(s)):
            out[i] = float(np.max(s))
    return out


def _scale_range(result: OptCutsResult) -> tuple[float, float, float]:
    raw = _raw_scale_factors(result)
    valid = raw[np.isfinite(raw) & (raw > 1.0e-14)]
    if len(valid) == 0:
        return float("inf"), float("nan"), float("nan")
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    return float(hi / lo), lo, hi


def _surface_area(result: OptCutsResult) -> float:
    xyz = np.asarray(result.surface_vertices_3d, dtype=float)
    faces = np.asarray(result.surface_faces, dtype=int)[:, :3]
    tri = xyz[faces]
    return 0.5 * float(
        np.sum(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1))
    )


def _internal_seam_length(result: OptCutsResult) -> tuple[float, int]:
    """Measure physical 3D length of interior source edges split in UV."""
    xyz = np.asarray(result.surface_vertices_3d, dtype=float)
    sf = np.asarray(result.surface_faces, dtype=int)[:, :3]
    uf = np.asarray(result.uv_faces, dtype=int)[:, :3]
    uses: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for f3, f2 in zip(sf, uf):
        for (a3, b3), (a2, b2) in zip(
            ((f3[0], f3[1]), (f3[1], f3[2]), (f3[2], f3[0])),
            ((f2[0], f2[1]), (f2[1], f2[2]), (f2[2], f2[0])),
        ):
            source = tuple(sorted((int(a3), int(b3))))
            uv_edge = tuple(sorted((int(a2), int(b2))))
            uses.setdefault(source, []).append(uv_edge)

    length = 0.0
    count = 0
    for (a, b), uv_edges in uses.items():
        # An interior source edge is a seam exactly when its two incident source
        # triangles do not share the same UV edge after cutting.
        if len(uv_edges) == 2 and uv_edges[0] != uv_edges[1]:
            length += float(np.linalg.norm(xyz[b] - xyz[a]))
            count += 1
    return float(length), int(count)


def _candidate_bounds(base: float, count: int) -> list[float]:
    base = max(4.0001, float(base))
    count = max(1, int(count))
    gap = base - 4.0
    values: list[float] = []
    for i in range(count):
        # 4.1 -> 4.1, 4.05, 4.025, ... .  Tightening the ordinary OptCuts
        # distortion bound asks the upstream optimizer to trade more seam for a
        # closer-to-isometric map, producing alternative seam topologies.
        value = 4.0 + gap * (0.5 ** i)
        value = max(4.0001, value)
        if not any(abs(value - old) <= 1.0e-10 for old in values):
            values.append(float(value))
    return values


def _evaluate(result: OptCutsResult, distortion_weight: float, scale_weight: float) -> dict[str, float | int | bool]:
    scale_range, raw_min, raw_max = _scale_range(result)
    seam_length, seam_edges = _internal_seam_length(result)
    characteristic = math.sqrt(max(_surface_area(result), 1.0e-30))
    seam_normalized = float(seam_length / characteristic)
    sd_mean = float(result.metrics.get("symmetric_dirichlet_mean", float("inf")))
    distortion_excess = max(0.0, sd_mean - 4.0) if np.isfinite(sd_mean) else float("inf")
    if np.isfinite(scale_range) and scale_range > 0.0:
        scale_log_violation = max(0.0, math.log(scale_range / SCALE_BOUND))
    else:
        scale_log_violation = float("inf")
    scale_penalty = float(scale_log_violation * scale_log_violation)
    score = float(
        seam_normalized
        + float(distortion_weight) * distortion_excess
        + float(scale_weight) * scale_penalty
    )
    return {
        "scale_range": float(scale_range),
        "scale_raw_min": float(raw_min),
        "scale_raw_max": float(raw_max),
        "scale_bound": float(SCALE_BOUND),
        "scale_hard_feasible": bool(scale_range <= SCALE_BOUND + 1.0e-9),
        "scale_log_violation": float(scale_log_violation),
        "scale_penalty": float(scale_penalty),
        "seam_length_3d": float(seam_length),
        "seam_edge_count": int(seam_edges),
        "seam_length_normalized": float(seam_normalized),
        "symmetric_dirichlet_mean": float(sd_mean),
        "distortion_excess_over_isometry": float(distortion_excess),
        "soft_score": float(score),
    }


def run_scale_aware_optcuts(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    config: OptCutsConfig,
) -> OptCutsResult:
    """Run a small OptCuts candidate family and select by OneString-aware score."""
    count = max(1, min(8, _env_int("ONESTRING_OPTCUTS_SCALE_AWARE_CANDIDATES", 3)))
    distortion_weight = max(0.0, _env_float("ONESTRING_OPTCUTS_SCALE_AWARE_DISTORTION_WEIGHT", 1.0))
    scale_weight = max(0.0, _env_float("ONESTRING_OPTCUTS_SCALE_AWARE_SCALE_WEIGHT", 40.0))
    bounds = _candidate_bounds(float(config.distortion_bound), count)

    evaluated: list[tuple[OptCutsResult, dict[str, float | int | bool], float]] = []
    for index, bound in enumerate(bounds, start=1):
        candidate_cfg = replace(config, distortion_bound=float(bound))
        result = run_official_optcuts(surface_vertices, surface_faces, candidate_cfg)
        metrics = _evaluate(result, distortion_weight, scale_weight)
        evaluated.append((result, metrics, float(bound)))
        print(
            "[OPTCUTS-SCALE-AWARE-CANDIDATE] "
            f"candidate={index}/{len(bounds)} distortion_bound={bound:.9g} "
            f"seam={float(metrics['seam_length_normalized']):.6g} "
            f"sd_mean={float(metrics['symmetric_dirichlet_mean']):.6g} "
            f"scale_range={float(metrics['scale_range']):.6g} "
            f"scale_penalty={float(metrics['scale_penalty']):.6g} "
            f"score={float(metrics['soft_score']):.6g} "
            f"hard_feasible={bool(metrics['scale_hard_feasible'])}"
        )

    feasible = [item for item in evaluated if bool(item[1]["scale_hard_feasible"])]
    pool = feasible if feasible else evaluated
    selected = min(pool, key=lambda item: (float(item[1]["soft_score"]), float(item[2])))
    result, selected_metrics, selected_bound = selected

    candidate_summary = [
        {
            "distortion_bound": float(bound),
            **{k: v for k, v in metrics.items()},
        }
        for _, metrics, bound in evaluated
    ]
    result.metrics.update(
        {
            "optcuts_scale_aware_enabled": True,
            "optcuts_scale_aware_model": (
                "outer official-OptCuts candidate selection: seam + actual Symmetric-Dirichlet + "
                "soft OneString scale-range penalty; hard final scale-range preference"
            ),
            "optcuts_scale_aware_is_internal_cpp_objective": False,
            "optcuts_scale_aware_limitation": (
                "upstream OptCuts CLI has no custom energy callback; scale factor selects among complete "
                "official OptCuts solutions rather than modifying each internal topology operation"
            ),
            "optcuts_scale_aware_candidate_count": int(len(evaluated)),
            "optcuts_scale_aware_distortion_weight": float(distortion_weight),
            "optcuts_scale_aware_scale_weight": float(scale_weight),
            "optcuts_scale_aware_scale_bound": float(SCALE_BOUND),
            "optcuts_scale_aware_selected_distortion_bound": float(selected_bound),
            "optcuts_scale_aware_any_hard_feasible": bool(feasible),
            "optcuts_scale_aware_candidates": candidate_summary,
            **{f"optcuts_scale_aware_selected_{k}": v for k, v in selected_metrics.items()},
        }
    )
    print(
        "[OPTCUTS-SCALE-AWARE-FINAL] "
        f"selected_distortion_bound={selected_bound:.9g} "
        f"scale_range={float(selected_metrics['scale_range']):.6g} "
        f"bound={SCALE_BOUND:.6g} hard_feasible={bool(selected_metrics['scale_hard_feasible'])} "
        f"any_feasible={bool(feasible)} score={float(selected_metrics['soft_score']):.6g}"
    )
    return result


def install_optcuts_scale_aware_selection_patch(pipeline: Any) -> None:
    """Use scale-aware official-OptCuts selection only for visible optcuts_test2."""
    if getattr(pipeline, "_onestring_optcuts_scale_aware_selection_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    # This wrapper must be installed after the ordinary OptCuts backend but
    # before optcuts_test captures its base builder.  optcuts_test2 is routed as
    # optcuts_test with ONESTRING_OPTCUTS_TEST_VARIANT=2; its internal call then
    # asks for ordinary mode="optcuts", which is intercepted here.
    def scale_aware_builder(surface: Any, target: Any, grid: Any, params: Any):
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        variant = os.environ.get("ONESTRING_OPTCUTS_TEST_VARIANT", "0").strip()
        if mode != "optcuts" or variant != "2":
            return base_builder(surface, target, grid, params)

        from .optcuts_pipeline_patch import _config_from_params

        surface_vertices = np.asarray(surface.vertices, dtype=float)
        surface_faces = np.asarray(surface.faces, dtype=int)[:, :3]
        result = run_scale_aware_optcuts(
            surface_vertices,
            surface_faces,
            _config_from_params(params),
        )

        loop = [int(v) for v in result.boundary_loops[0]]
        boundary = result.uv_vertices_2d[loop + [loop[0]]]
        metrics = {
            **result.metrics,
            "parameterization_exactness_label": "official_optcuts_external_scale_aware_outer_selection",
            "parameterization_warning": (
                "Official OptCuts is run unchanged; OneString scale factor is used to select among "
                "several official OptCuts solutions."
            ),
            "paper_compliance_status": "experimental_nonpaper_parameterization",
            "omega_boundary_mode": "paper_default",
            "omega_parameterization_mode": "optcuts",
            "requested_omega_parameterization_mode": "optcuts",
            "boundary_vertex_count": int(len(loop)),
            "boundary_loop": loop,
            "height_field_shortcut_used": False,
            "harmonic_solve_performed": False,
            "omega_corresponds_to_S": True,
            "omega_correspondence_model": "scale-aware selection among official OptCuts cut topology + UV embeddings",
            "paper_flow_stage": "S -> Omega by scale-aware outer selection of official OptCuts solutions",
            "bff_implemented": False,
            "optcuts_implemented": True,
            "optcuts_grid_mode": False,
            "optcuts_internal_method": "optcuts_official",
            "omega_boundary_fixed": False,
            "omega_boundary_forced_rectangle": False,
            "omega_boundary_shape": "free",
            "omega_boundary_constraint_model": "selected official OptCuts optimized seam",
            "fallbacks_used": [],
        }
        parameterization = pipeline.SurfaceParameterization(
            method="optcuts_official",
            surface_vertices_3d=np.asarray(result.surface_vertices_3d, dtype=float),
            surface_faces=np.asarray(result.surface_faces, dtype=int),
            uv_vertices_2d=np.asarray(result.uv_vertices_2d, dtype=float),
            uv_faces=np.asarray(result.uv_faces, dtype=int),
            omega_boundary=np.asarray(boundary, dtype=float),
            triangle_acceleration=None,
            metrics=metrics,
        )
        return parameterization

    pipeline._build_surface_parameterization = scale_aware_builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = scale_aware_builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = scale_aware_builder
    pipeline._onestring_optcuts_scale_aware_selection_installed = True


__all__ = [
    "SCALE_BOUND",
    "run_scale_aware_optcuts",
    "install_optcuts_scale_aware_selection_patch",
]
