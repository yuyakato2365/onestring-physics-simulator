"""Structured diagnostics for Split failures.

Installed only by app_split_panels.py.  Writes JSONL records to
<project>/logs/split_debug.jsonl and mirrors concise markers to stdout.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any
import uuid

import numpy as np

_CURRENT_RUN: ContextVar[str | None] = ContextVar("onestring_split_debug_run", default=None)


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write(log_path: Path, event: str, **payload: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": _CURRENT_RUN.get(),
        "event": event,
        **{k: _jsonable(v) for k, v in payload.items()},
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _sample(values: np.ndarray | list[Any], limit: int = 40) -> list[Any]:
    seq = list(np.asarray(values).reshape(-1)) if isinstance(values, np.ndarray) else list(values)
    if len(seq) <= limit:
        return [_jsonable(v) for v in seq]
    half = max(1, limit // 2)
    return [_jsonable(v) for v in seq[:half]] + ["..."] + [_jsonable(v) for v in seq[-half:]]


def _candidate_diagnostics(final_module: Any, vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, axis: str, requested: float) -> dict[str, Any]:
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    comp = np.asarray(face_ids, dtype=int)
    coord = 1 if str(axis) == "row" else 0
    other = 1 - coord
    report: dict[str, Any] = {
        "axis": str(axis),
        "requested_value": float(requested),
        "component_face_count": int(len(comp)),
        "component_vertex_count": int(len(np.unique(f[comp].reshape(-1)))) if len(comp) else 0,
    }

    internal = final_module._internal_grid_values(verts, f, comp, coord)
    report["internal_grid_value_count"] = int(len(internal))
    report["internal_grid_values_sample"] = _sample(internal)
    if len(internal) == 0:
        report["failure_stage"] = "no_internal_grid_values"
        return report

    value = float(internal[int(np.argmin(np.abs(internal - float(requested))))])
    report["snapped_value"] = value
    report["snap_distance"] = abs(value - float(requested))
    span = max(float(np.ptp(verts[:, coord])), 1.0)
    tol = max(1e-9 * span, 1e-11)
    report["tolerance"] = tol

    centroids = np.mean(verts[f[comp]][:, :, coord], axis=1)
    neg = comp[centroids < value - tol]
    pos = comp[centroids > value + tol]
    mid = comp[np.abs(centroids - value) <= tol]
    report.update({
        "negative_face_count": int(len(neg)),
        "positive_face_count": int(len(pos)),
        "mid_face_count": int(len(mid)),
    })
    if len(neg) == 0 or len(pos) == 0:
        report["failure_stage"] = "empty_side_after_centroid_partition"
        return report

    edge_to_faces, boundary_edges, boundary_vertices = final_module._component_edge_data(f, comp)
    report["boundary_edge_count"] = int(len(boundary_edges))
    report["boundary_vertex_count"] = int(len(boundary_vertices))
    neg_set, pos_set = set(map(int, neg)), set(map(int, pos))
    seam_edges: list[tuple[int, int]] = []
    line_edges: list[tuple[int, int]] = []
    for edge, inc in edge_to_faces.items():
        if len(inc) != 2:
            continue
        a, b = edge
        if abs(float(verts[a, coord]) - value) <= tol and abs(float(verts[b, coord]) - value) <= tol:
            line_edges.append(edge)
            sides = {(-1 if fi in neg_set else 1 if fi in pos_set else 0) for fi in inc}
            if -1 in sides and 1 in sides:
                seam_edges.append(edge)
    report["all_internal_edges_on_line_count"] = int(len(line_edges))
    report["seam_edge_count"] = int(len(seam_edges))
    if not seam_edges:
        report["failure_stage"] = "no_cross_side_seam_edges"
        return report

    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in seam_edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    unseen = set(graph)
    graph_components: list[list[int]] = []
    while unseen:
        root = unseen.pop()
        q = deque([root])
        group = [root]
        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    q.append(nxt)
                    group.append(nxt)
        graph_components.append(group)
    degrees = {int(v): int(len(nbrs)) for v, nbrs in graph.items()}
    degree_hist = Counter(degrees.values())
    endpoints = [v for v, degree in degrees.items() if degree == 1]
    report.update({
        "seam_graph_component_count": int(len(graph_components)),
        "seam_graph_component_sizes": [int(len(g)) for g in graph_components],
        "seam_degree_histogram": {str(k): int(v) for k, v in sorted(degree_hist.items())},
        "endpoint_count": int(len(endpoints)),
        "endpoint_ids": endpoints,
        "endpoint_on_boundary": [bool(v in boundary_vertices) for v in endpoints],
        "line_other_coord_range": [
            float(np.min(verts[list(graph), other])) if graph else None,
            float(np.max(verts[list(graph), other])) if graph else None,
        ],
    })
    if len(graph_components) != 1:
        report["failure_stage"] = "seam_graph_disconnected"
        return report
    if len(endpoints) != 2 or any(d not in {1, 2} for d in degrees.values()):
        report["failure_stage"] = "seam_not_single_chain"
        return report
    if not all(v in boundary_vertices for v in endpoints):
        report["failure_stage"] = "seam_endpoints_not_on_component_boundary"
        return report

    seam_vertices = set(graph.keys())
    neg_vertices = set(map(int, f[neg].reshape(-1)))
    pos_vertices = set(map(int, f[pos].reshape(-1)))
    shared = neg_vertices & pos_vertices
    report.update({
        "seam_vertex_count": int(len(seam_vertices)),
        "shared_vertex_count": int(len(shared)),
        "shared_not_on_seam_count": int(len(shared - seam_vertices)),
        "seam_not_shared_count": int(len(seam_vertices - shared)),
        "shared_not_on_seam_sample": _sample(sorted(shared - seam_vertices), 20),
        "seam_not_shared_sample": _sample(sorted(seam_vertices - shared), 20),
    })
    if shared != seam_vertices:
        report["failure_stage"] = "shared_vertices_do_not_equal_seam_vertices"
        return report

    report["failure_stage"] = None
    report["candidate_valid"] = True
    return report


def install_split_diagnostics(pipeline_module: Any, final_module: Any, project_root: str | Path) -> Path:
    """Instrument the actual final M2D Split execution path."""
    if getattr(pipeline_module, "_split_diagnostics_installed", False):
        return Path(project_root) / "logs" / "split_debug.jsonl"

    log_path = Path(project_root) / "logs" / "split_debug.jsonl"
    original_candidate = final_module._complete_cut_candidate

    def diagnostic_candidate(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, axis: str, requested: float):
        report = _candidate_diagnostics(final_module, vertices, faces, face_ids, axis, requested)
        result = original_candidate(vertices, faces, face_ids, axis, requested)
        report["original_candidate_returned"] = bool(result is not None)
        if bool(result is not None) != bool(report.get("candidate_valid", False)):
            report["diagnostic_disagreement"] = True
        _write(log_path, "split_candidate_check", **report)
        return result

    final_module._complete_cut_candidate = diagnostic_candidate

    previous_build = pipeline_module._build_m2d

    def diagnostic_build_m2d(grid: Any, domain: Any, params: Any = None):
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + uuid.uuid4().hex[:6]
        token = _CURRENT_RUN.set(run_id)
        split_lines = list(getattr(domain, "split_lines", []) or [])
        localized = list(getattr(domain, "localized_split_segments", []) or [])
        csf_values = np.asarray(getattr(domain, "csf_values", []), dtype=float).reshape(-1)
        print(f"[SPLIT-DEBUG] run={run_id} START lines={split_lines} localized={localized}")
        _write(
            log_path,
            "m2d_split_start",
            split_lines=split_lines,
            localized_split_segments=localized,
            csf_before=getattr(domain, "csf_before", None),
            csf_after_split_reported=getattr(domain, "csf_after_split", None),
            csf_threshold=getattr(domain, "csf_split_threshold", None),
            csf_value_count=int(len(csf_values)),
            csf_min=float(np.min(csf_values)) if len(csf_values) else None,
            csf_max=float(np.max(csf_values)) if len(csf_values) else None,
            csf_median=float(np.median(csf_values)) if len(csf_values) else None,
            peak_uv_target=getattr(domain, "peak_uv_target", None),
            peak_grid_alignment_shift=getattr(domain, "peak_grid_alignment_shift", None),
            localize_csf_splits=getattr(domain, "localize_csf_splits", None),
            params_repr=repr(params),
        )
        try:
            mesh = previous_build(grid, domain, params)
            metrics = dict(getattr(mesh, "metrics", {}) or {})
            faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
            components = final_module._edge_components(faces) if faces.size else []
            summary = {
                "csf_split_applied": metrics.get("csf_split_applied"),
                "split_locations": metrics.get("split_locations"),
                "raw_split_locations": metrics.get("raw_split_locations"),
                "final_split_panel_pass_applied": metrics.get("final_split_panel_pass_applied"),
                "paper_style_complete_split": metrics.get("paper_style_complete_split"),
                "final_split_panel_count": metrics.get("final_split_panel_count"),
                "split_panel_count": metrics.get("split_panel_count"),
                "final_split_panel_per_line": metrics.get("final_split_panel_per_line"),
                "final_split_panel_rejected_lines": metrics.get("final_split_panel_rejected_lines"),
                "rewelded_earlier_split_duplicates": metrics.get("paper_style_rewelded_earlier_split_duplicates"),
                "returned_edge_component_count": int(len(components)),
                "returned_component_face_counts": [int(len(c)) for c in components],
                "returned_vertex_count": int(len(np.asarray(getattr(mesh, "vertices", [])))),
                "returned_face_count": int(len(faces)),
            }
            _write(log_path, "m2d_split_result", **summary)
            print(
                f"[SPLIT-DEBUG] run={run_id} RESULT "
                f"split_applied={summary['csf_split_applied']} "
                f"final_pass={summary['final_split_panel_pass_applied']} "
                f"components={summary['returned_edge_component_count']}"
            )
            return mesh
        except Exception as exc:
            _write(
                log_path,
                "m2d_split_exception",
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                traceback=traceback.format_exc(),
            )
            print(f"[SPLIT-DEBUG] run={run_id} EXCEPTION {type(exc).__name__}: {exc}")
            raise
        finally:
            _CURRENT_RUN.reset(token)

    pipeline_module._build_m2d = diagnostic_build_m2d
    try:
        pipeline_module._original._build_m2d = diagnostic_build_m2d
    except Exception:
        pass

    for fn in (
        getattr(pipeline_module, "build_onestring_design", None),
        getattr(pipeline_module._original, "build_onestring_design", None),
        getattr(pipeline_module, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = diagnostic_build_m2d

    pipeline_module._split_diagnostics_installed = True
    pipeline_module._split_diagnostics_log_path = str(log_path)
    print(f"[SPLIT-DEBUG] logging enabled: {log_path}")
    return log_path


__all__ = ["install_split_diagnostics"]
