"""End-to-end diagnostics for the fixed OptCuts seam requirement.

Writes JSONL to ``logs/optcuts_requirement.jsonl`` and prints concise progress
messages to the terminal.  This module does not change any numerical result.
It only observes the required sequence:

Official OptCuts -> seam extraction -> Omega straightening -> grid alignment -> M2D.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any

import numpy as np


_LOG_LOCK = threading.Lock()
_TLS = threading.local()
_HEARTBEAT_SECONDS = 5.0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def log_path() -> Path:
    path = _project_root() / "logs" / "optcuts_requirement.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _current_run_id() -> str | None:
    return getattr(_TLS, "run_id", None)


@contextmanager
def _run_context(run_id: str):
    previous = _current_run_id()
    _TLS.run_id = run_id
    try:
        yield
    finally:
        _TLS.run_id = previous


def emit(event: str, *, run_id: str | None = None, **fields: Any) -> None:
    rid = run_id or _current_run_id() or "unscoped"
    record = {
        "ts_unix": time.time(),
        "ts_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": rid,
        "event": str(event),
        **{k: _jsonable(v) for k, v in fields.items()},
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with _LOG_LOCK:
        with log_path().open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    summary = fields.get("summary")
    suffix = f" {summary}" if summary else ""
    print(f"[OPTCUTS-REQ][{rid}][{event}]{suffix}", flush=True)


def _grid_alignment_error(
    nodes: dict[int, np.ndarray],
    edges: list[tuple[int, int]],
    *,
    h: float,
    phase_u: float,
    phase_v: float,
    angle_degrees: float,
) -> tuple[float, list[dict[str, Any]]]:
    from .optcuts_rectilinear_seam_patch import _extract_source_chains

    theta = math.radians(float(angle_degrees))
    c, s = math.cos(theta), math.sin(theta)
    world_from_grid = np.asarray([[c, -s], [s, c]], dtype=float)
    grid_from_world = world_from_grid.T
    h = max(float(h), 1e-12)
    worst = 0.0
    details: list[dict[str, Any]] = []
    for chain_index, chain in enumerate(_extract_source_chains(nodes, edges)):
        pts = np.asarray([nodes[int(v)] for v in chain if int(v) in nodes], dtype=float)
        if len(pts) < 2:
            continue
        q = (grid_from_world @ pts[:, :2].T).T
        span_u = float(np.ptp(q[:, 0]))
        span_v = float(np.ptp(q[:, 1]))
        if span_u >= span_v:
            lattice = float(phase_v) + np.rint((q[:, 1] - float(phase_v)) / h) * h
            err = float(np.max(np.abs(q[:, 1] - lattice)))
            axis = "grid_u"
        else:
            lattice = float(phase_u) + np.rint((q[:, 0] - float(phase_u)) / h) * h
            err = float(np.max(np.abs(q[:, 0] - lattice)))
            axis = "grid_v"
        worst = max(worst, err)
        details.append({"chain": chain_index, "axis": axis, "max_grid_error": err})
    return worst, details


def _install_official_optcuts_observer() -> None:
    from . import optcuts_backend as backend
    from . import optcuts_pipeline_patch as pipeline_patch

    if getattr(backend, "_onestring_requirement_diagnostics_installed", False):
        return
    base_run = backend.run_official_optcuts

    def observed_run(surface_vertices: np.ndarray, surface_faces: np.ndarray, config: Any = None):
        run_id = uuid.uuid4().hex[:12]
        started = time.time()
        stop = threading.Event()
        cfg = config or backend.OptCutsConfig()
        emit(
            "official_optcuts_start",
            run_id=run_id,
            summary=f"V={len(surface_vertices)} F={len(surface_faces)} timeout={float(cfg.timeout_seconds):g}s",
            surface_vertex_count=int(len(surface_vertices)),
            surface_triangle_count=int(len(surface_faces)),
            timeout_seconds=float(cfg.timeout_seconds),
            distortion_bound=float(cfg.distortion_bound),
            lambda_init=float(cfg.lambda_init),
            method_type=int(cfg.method_type),
            use_bijectivity=bool(cfg.use_bijectivity),
            initial_cut_option=int(cfg.initial_cut_option),
            log_file=str(log_path()),
        )

        def heartbeat() -> None:
            while not stop.wait(_HEARTBEAT_SECONDS):
                elapsed = time.time() - started
                emit(
                    "official_optcuts_heartbeat",
                    run_id=run_id,
                    summary=f"still running: {elapsed:.1f}s",
                    elapsed_seconds=float(elapsed),
                )

        thread = threading.Thread(target=heartbeat, daemon=True, name="optcuts-requirement-heartbeat")
        thread.start()
        try:
            with _run_context(run_id):
                result = base_run(surface_vertices, surface_faces, cfg)
            elapsed = time.time() - started
            result.metrics["optcuts_requirement_run_id"] = run_id
            result.metrics["optcuts_requirement_log_file"] = str(log_path())
            emit(
                "official_optcuts_finish",
                run_id=run_id,
                summary=f"completed in {elapsed:.3f}s",
                elapsed_seconds=float(elapsed),
                uv_vertex_count=int(len(result.uv_vertices_2d)),
                output_obj=str(result.output_obj),
            )
            return result
        except Exception as exc:
            emit(
                "official_optcuts_error",
                run_id=run_id,
                summary=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=float(time.time() - started),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            raise
        finally:
            stop.set()
            thread.join(timeout=0.2)

    backend.run_official_optcuts = observed_run
    # optcuts_pipeline_patch imported the function by value, so replace its bound symbol too.
    pipeline_patch.run_official_optcuts = observed_run
    backend._onestring_requirement_diagnostics_installed = True


def _install_omega_stage_observers() -> None:
    from . import optcuts_seam_requirement_patch as req

    if getattr(req, "_onestring_requirement_stage_diagnostics_installed", False):
        return
    base_targets = req._chain_targets
    base_apply = req._apply_node_targets

    def observed_targets(nodes: dict[int, np.ndarray], edges: list[tuple[int, int]], **kwargs: Any):
        grid_size = kwargs.get("grid_size", None)
        stage = "grid_align_targets" if grid_size is not None else "omega_straighten_targets"
        started = time.time()
        targets, stats = base_targets(nodes, edges, **kwargs)
        emit(
            stage,
            summary=f"chains={stats.get('chain_count', 0)} targets={len(targets)}",
            elapsed_seconds=float(time.time() - started),
            node_count=int(len(nodes)),
            edge_count=int(len(edges)),
            target_count=int(len(targets)),
            stats=stats,
            grid_size=grid_size,
        )
        return targets, stats

    def observed_apply(parameterization: Any, node_targets: dict[int, np.ndarray]):
        started = time.time()
        result = base_apply(parameterization, node_targets)
        emit(
            "omega_harmonic_deform",
            summary=f"hard_targets={len(node_targets)} elapsed={time.time()-started:.3f}s",
            elapsed_seconds=float(time.time() - started),
            hard_target_count=int(len(node_targets)),
            uv_vertex_count=int(len(np.asarray(parameterization.uv_vertices_2d))),
        )
        return result

    req._chain_targets = observed_targets
    req._apply_node_targets = observed_apply
    req._onestring_requirement_stage_diagnostics_installed = True


def install_requirement_pipeline_diagnostics(pipeline: Any) -> None:
    """Wrap the final Omega requirement stage and arrange final M2D verification logging."""
    _install_official_optcuts_observer()
    _install_omega_stage_observers()

    if getattr(pipeline, "_onestring_requirement_pipeline_diagnostics_installed", False):
        return
    base_flatten = pipeline._flatten_to_domain

    def observed_flatten(parameterization: Any, grid: Any, params: Any = None):
        if str(getattr(parameterization, "method", "")) != "optcuts":
            return base_flatten(parameterization, grid, params)
        run_id = str(getattr(parameterization, "metrics", {}).get("optcuts_requirement_run_id", "unscoped"))
        with _run_context(run_id):
            emit("requirement_postprocess_start", summary="seam -> Omega straightening -> grid alignment")
            started = time.time()
            try:
                domain = base_flatten(parameterization, grid, params)
            except Exception as exc:
                emit(
                    "omega_requirement_verdict",
                    summary=f"FAIL: {type(exc).__name__}: {exc}",
                    passed=False,
                    elapsed_seconds=float(time.time() - started),
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
                raise

            sequence = getattr(domain, "_optcuts_requirement_sequence", None)
            metrics = dict(getattr(domain, "_optcuts_requirement_metrics", {}) or {})
            aligned_param = getattr(domain, "_optcuts_requirement_parameterization", None)
            max_grid_error = float("inf")
            grid_details: list[dict[str, Any]] = []
            if aligned_param is not None:
                from .optcuts_seam_extraction_patch import extract_connected_seam_payload_robust
                payload = extract_connected_seam_payload_robust(aligned_param)
                nodes = {int(k): np.asarray(v, float) for k, v in dict(payload.get("nodes", {}) or {}).items()}
                edges = [(int(a), int(b)) for a, b in list(payload.get("edges", []) or [])]
                h = max(float(getattr(grid, "tile_size", 0.0) or getattr(params, "tile_size", 0.0) or 0.0), 1e-8)
                phase_u = float(os.environ.get("ONESTRING_OPTCUTS_GRID_PHASE_U", "0") or 0.0)
                phase_v = float(os.environ.get("ONESTRING_OPTCUTS_GRID_PHASE_V", "0") or 0.0)
                angle = float(os.environ.get("ONESTRING_OPTCUTS_GRID_ANGLE_DEGREES", "0") or 0.0)
                max_grid_error, grid_details = _grid_alignment_error(
                    nodes, edges, h=h, phase_u=phase_u, phase_v=phase_v, angle_degrees=angle
                )
                grid_tol = max(1e-8, 1e-5 * h)
            else:
                grid_tol = float("nan")

            line_error = float(metrics.get("max_final_chain_line_error", float("inf")))
            line_tol = float(metrics.get("line_tolerance", float("nan")))
            passed = (
                sequence == "seam->omega_straight->grid_align"
                and math.isfinite(line_error)
                and math.isfinite(max_grid_error)
                and line_error <= line_tol
                and max_grid_error <= grid_tol
            )
            diagnostics = {
                **metrics,
                "max_final_grid_alignment_error": float(max_grid_error),
                "grid_alignment_tolerance": float(grid_tol),
                "grid_alignment_details": grid_details,
                "requirement_passed_at_omega": bool(passed),
            }
            setattr(domain, "_optcuts_requirement_diagnostics", diagnostics)
            emit(
                "omega_requirement_verdict",
                summary=(
                    f"{'PASS' if passed else 'FAIL'} line={line_error:.3g}/{line_tol:.3g} "
                    f"grid={max_grid_error:.3g}/{grid_tol:.3g}"
                ),
                passed=bool(passed),
                sequence=sequence,
                elapsed_seconds=float(time.time() - started),
                diagnostics=diagnostics,
            )
            if not passed:
                raise RuntimeError(
                    "OPTCUTS_REQUIREMENT_DIAGNOSTIC_FAILED: final Omega seam did not satisfy both "
                    f"straightness and fabrication-grid alignment; line={line_error:g}/{line_tol:g}, "
                    f"grid={max_grid_error:g}/{grid_tol:g}"
                )
            return domain

    pipeline._flatten_to_domain = observed_flatten
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = observed_flatten

    # Simple Split later installs the final M2D stack. Hook that installer and
    # add diagnostics only after all existing M2D adapters/verifiers are present.
    from . import simple_split_panel_patch as simple_split
    original_installer = simple_split.install_simple_split_panel_patch

    if not getattr(simple_split, "_onestring_requirement_m2d_diagnostics_hook_installed", False):
        def install_then_observe_m2d(pipeline_module: Any, optimization_debug_module: Any) -> None:
            original_installer(pipeline_module, optimization_debug_module)
            if getattr(pipeline_module, "_onestring_requirement_m2d_observer_installed", False):
                return
            base_build = pipeline_module._build_m2d

            def observed_build(grid: Any, domain: Any, params: Any = None):
                if getattr(domain, "_optcuts_requirement_sequence", None) != "seam->omega_straight->grid_align":
                    return base_build(grid, domain, params)
                aligned = getattr(domain, "_optcuts_requirement_parameterization", None)
                run_id = "unscoped"
                if aligned is not None:
                    run_id = str(getattr(aligned, "metrics", {}).get("optcuts_requirement_run_id", "unscoped"))
                started = time.time()
                with _run_context(run_id):
                    emit("m2d_requirement_start", summary="building M2D with required straight grid seam")
                    try:
                        mesh = base_build(grid, domain, params)
                    except Exception as exc:
                        emit(
                            "m2d_requirement_verdict",
                            summary=f"FAIL: {type(exc).__name__}: {exc}",
                            passed=False,
                            elapsed_seconds=float(time.time() - started),
                            exception_type=type(exc).__name__,
                            exception_message=str(exc),
                        )
                        raise
                    metrics = dict(getattr(mesh, "metrics", {}) or {})
                    nonstraight = int(metrics.get("optcuts_requirement_nonstraight_grid_path_count", 0))
                    path_count = int(metrics.get("optcuts_requirement_grid_path_count", 0))
                    passed = nonstraight == 0
                    emit(
                        "m2d_requirement_verdict",
                        summary=f"{'PASS' if passed else 'FAIL'} paths={path_count} nonstraight={nonstraight}",
                        passed=bool(passed),
                        elapsed_seconds=float(time.time() - started),
                        grid_path_count=path_count,
                        nonstraight_grid_path_count=nonstraight,
                        metrics=metrics,
                    )
                    return mesh

            pipeline_module._build_m2d = observed_build
            for fn in (
                getattr(pipeline_module, "build_onestring_design", None),
                getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
                getattr(getattr(pipeline_module, "_original", None), "build_onestring_design", None),
            ):
                glb = getattr(fn, "__globals__", None)
                if isinstance(glb, dict):
                    glb["_build_m2d"] = observed_build
            pipeline_module._onestring_requirement_m2d_observer_installed = True

        simple_split.install_simple_split_panel_patch = install_then_observe_m2d
        simple_split._onestring_requirement_m2d_diagnostics_hook_installed = True

    pipeline._onestring_requirement_pipeline_diagnostics_installed = True


__all__ = ["install_requirement_pipeline_diagnostics", "emit", "log_path"]
