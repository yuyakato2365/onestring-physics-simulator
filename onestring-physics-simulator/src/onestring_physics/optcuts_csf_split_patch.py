"""Exact OptCuts conformal-scale diagnostics and paper-style automatic M2D splits.

This patch is intentionally limited to ``omega_parameterization_mode == "optcuts_test"``
(the numerical carrier also used by the dated test2 route).

Why this exists
---------------
The legacy pipeline already had a heuristic CSF proxy and split plumbing. For the
current OptCuts experiment we want a directly measured quantity instead:

    lambda_t = sigma_max(J_{UV -> S})

for every source triangle t. A global similarity scale in UV is arbitrary, so
reported values are normalized by the smallest valid lambda. Hence

    reported_lambda_t = raw_lambda_t / min(raw_lambda)

and the global scale-factor range is max(raw_lambda) / min(raw_lambda).

For a disconnected split part, the same similarity-scale freedom applies to that
part independently. Its required range is therefore

    sigma(part) = max(raw_lambda in part) / min(raw_lambda in part).

The OneString paper allows a quad-linkage scale factor up to 2. We therefore
plan complete row/column M2D cuts until every conceptual part has sigma <= 2,
or until the split budget is exhausted. The existing Simple Split patch performs
the actual topology cut by snapping these requested cuts to fabrication-grid lines.

Important: this patch does NOT fake a post-split CSF by clamping values to 2.
``csf_after_split`` is the maximum measured raw-lambda ratio of the conceptual
parts after the planned hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np


BOUND = 2.0
EPS = 1.0e-9


def _triangle_local_basis(tri: np.ndarray) -> np.ndarray | None:
    """Return a 2x2 local coordinate matrix for one 3D triangle."""
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


def _triangle_raw_lambda(parameterization: Any) -> np.ndarray:
    """Measure sigma_max(J_UV->S) per corresponding source/UV triangle.

    ``surface_faces`` and ``uv_faces`` are intentionally kept separate because an
    OptCuts seam can duplicate UV vertices while retaining source-surface vertices.
    """
    surface_vertices = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    surface_faces = np.asarray(parameterization.surface_faces, dtype=int)[:, :3]
    uv_vertices = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)[:, :3]

    if len(surface_faces) != len(uv_faces):
        raise RuntimeError(
            "OPTCUTS_CSF_FACE_COUNT_MISMATCH: "
            f"surface_faces={len(surface_faces)} uv_faces={len(uv_faces)}"
        )

    out = np.full(len(surface_faces), np.nan, dtype=float)
    for fid, (sf, uf) in enumerate(zip(surface_faces, uv_faces)):
        tri3 = surface_vertices[np.asarray(sf, dtype=int)]
        tri2 = uv_vertices[np.asarray(uf, dtype=int)]

        ds = _triangle_local_basis(tri3)
        if ds is None:
            continue

        duv = np.asarray(
            [
                tri2[1] - tri2[0],
                tri2[2] - tri2[0],
            ],
            dtype=float,
        ).T
        det = float(np.linalg.det(duv))
        if not np.isfinite(det) or abs(det) <= 1.0e-14:
            continue

        # Local 3D coordinates satisfy DS = J_(UV->S) * DUV.
        try:
            jac = ds @ np.linalg.inv(duv)
            singular = np.linalg.svd(jac, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        if len(singular) and np.all(np.isfinite(singular)):
            out[fid] = float(np.max(singular))
    return out


def _normalized_global(raw: np.ndarray) -> tuple[np.ndarray, float]:
    valid = np.isfinite(raw) & (raw > 1.0e-14)
    if not np.any(valid):
        raise RuntimeError("OPTCUTS_CSF_NO_VALID_TRIANGLES")
    minimum = float(np.min(raw[valid]))
    normalized = np.full_like(raw, np.nan, dtype=float)
    normalized[valid] = raw[valid] / minimum
    return normalized, minimum


def _vertex_csf(parameterization: Any, normalized: np.ndarray) -> np.ndarray:
    """Conservative UV-vertex field: maximum incident triangle scale factor."""
    uv_count = len(np.asarray(parameterization.uv_vertices_2d))
    uv_faces = np.asarray(parameterization.uv_faces, dtype=int)[:, :3]
    values = np.full(uv_count, np.nan, dtype=float)
    for fid, face in enumerate(uv_faces):
        lam = float(normalized[fid])
        if not np.isfinite(lam):
            continue
        for vid in face:
            vid = int(vid)
            if not np.isfinite(values[vid]) or lam > values[vid]:
                values[vid] = lam
    return values


def _surface_boundary_vertices(surface_faces: np.ndarray) -> set[int]:
    counts: dict[tuple[int, int], int] = {}
    for tri in np.asarray(surface_faces, dtype=int)[:, :3]:
        ids = [int(v) for v in tri]
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edge = (a, b) if a < b else (b, a)
            counts[edge] = counts.get(edge, 0) + 1
    return {v for e, n in counts.items() if n == 1 for v in e}


def _gaussian_angle_defect(parameterization: Any) -> np.ndarray:
    """Discrete angle defect on the *input open surface*.

    Interior target angle is 2*pi; boundary target angle is pi.
    This is used only to choose a split location, following the paper's
    high-Gaussian-curvature heuristic.
    """
    vertices = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    faces = np.asarray(parameterization.surface_faces, dtype=int)[:, :3]
    angle_sum = np.zeros(len(vertices), dtype=float)

    for tri in faces:
        pts = vertices[np.asarray(tri, dtype=int)]
        for local in range(3):
            a = pts[(local + 1) % 3] - pts[local]
            b = pts[(local + 2) % 3] - pts[local]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1.0e-14 or nb <= 1.0e-14:
                continue
            cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            angle_sum[int(tri[local])] += math.acos(cosine)

    boundary = _surface_boundary_vertices(faces)
    defects = np.empty(len(vertices), dtype=float)
    for vid in range(len(vertices)):
        target = math.pi if vid in boundary else 2.0 * math.pi
        defects[vid] = target - angle_sum[vid]
    return defects


def _uv_for_surface_vertex(parameterization: Any, surface_vid: int, face_ids: np.ndarray) -> np.ndarray | None:
    """Return a representative UV copy of a source vertex within a part."""
    sf = np.asarray(parameterization.surface_faces, dtype=int)[:, :3]
    uf = np.asarray(parameterization.uv_faces, dtype=int)[:, :3]
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    samples: list[np.ndarray] = []
    for fid in np.asarray(face_ids, dtype=int):
        for local, vid in enumerate(sf[int(fid)]):
            if int(vid) == int(surface_vid):
                samples.append(np.asarray(uv[int(uf[int(fid), local])], dtype=float))
    if not samples:
        return None
    # If a source vertex has seam copies, use the median copy in this part.
    return np.median(np.vstack(samples), axis=0)


def _part_sigma(raw: np.ndarray, face_ids: np.ndarray) -> float:
    vals = np.asarray(raw[np.asarray(face_ids, dtype=int)], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 1.0e-14)]
    if len(vals) == 0:
        return float("nan")
    return float(np.max(vals) / np.min(vals))


def _face_uv_centroids(parameterization: Any) -> np.ndarray:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    faces = np.asarray(parameterization.uv_faces, dtype=int)[:, :3]
    return np.mean(uv[faces], axis=1)


def _part_surface_vertices(parameterization: Any, face_ids: np.ndarray) -> np.ndarray:
    faces = np.asarray(parameterization.surface_faces, dtype=int)[:, :3]
    return np.unique(faces[np.asarray(face_ids, dtype=int)].reshape(-1))


def _split_partition(
    centroids: np.ndarray,
    face_ids: np.ndarray,
    axis: str,
    value: float,
) -> tuple[np.ndarray, np.ndarray]:
    coord = 1 if axis == "row" else 0
    ids = np.asarray(face_ids, dtype=int)
    left = ids[centroids[ids, coord] < float(value)]
    right = ids[centroids[ids, coord] >= float(value)]
    return left, right


def _choose_complete_grid_split(
    parameterization: Any,
    face_ids: np.ndarray,
    curvature: np.ndarray,
    centroids: np.ndarray,
) -> tuple[str, float, int] | None:
    """Choose a complete u/v cut through the strongest-curvature vertex.

    The paper describes splitting along a grid direction through the point of
    highest Gaussian curvature. Here the input is an open surface; we rank by
    absolute discrete angle defect so both strongly convex and strongly saddle-like
    regions can trigger the geometric location. The exact CSF threshold remains
    the criterion for whether a split is needed.
    """
    part_vertices = _part_surface_vertices(parameterization, face_ids)
    if len(part_vertices) == 0:
        return None
    finite = part_vertices[np.isfinite(curvature[part_vertices])]
    if len(finite) == 0:
        return None

    # Prefer the strongest magnitude. Deterministic tie break by vertex id.
    ranked = sorted(
        (int(v) for v in finite),
        key=lambda v: (-abs(float(curvature[v])), v),
    )

    for surface_vid in ranked:
        peak_uv = _uv_for_surface_vertex(parameterization, surface_vid, face_ids)
        if peak_uv is None or not np.all(np.isfinite(peak_uv)):
            continue

        candidates: list[tuple[float, str, float]] = []
        # A "row" split is y=constant; a "col" split is x=constant.
        for axis, value in (("row", float(peak_uv[1])), ("col", float(peak_uv[0]))):
            a, b = _split_partition(centroids, face_ids, axis, value)
            if len(a) == 0 or len(b) == 0:
                continue
            balance = abs(len(a) - len(b)) / max(1, len(face_ids))
            candidates.append((float(balance), axis, value))
        if candidates:
            _, axis, value = min(candidates, key=lambda item: (item[0], item[1]))
            return axis, float(value), int(surface_vid)

    # Curvature peak may lie too close to an existing boundary for both directions.
    # Fall back to a median complete cut, still along a fabrication-grid direction.
    ids = np.asarray(face_ids, dtype=int)
    if len(ids) < 2:
        return None
    spread_x = float(np.ptp(centroids[ids, 0]))
    spread_y = float(np.ptp(centroids[ids, 1]))
    if spread_x <= 1.0e-14 and spread_y <= 1.0e-14:
        return None
    if spread_x >= spread_y:
        return "col", float(np.median(centroids[ids, 0])), -1
    return "row", float(np.median(centroids[ids, 1])), -1


@dataclass
class _Part:
    face_ids: np.ndarray
    depth: int


def _hierarchical_split_plan(
    parameterization: Any,
    raw: np.ndarray,
    threshold: float = BOUND,
    max_splits: int = 16,
) -> tuple[list[tuple[str, float]], list[dict[str, Any]], float]:
    valid = np.flatnonzero(np.isfinite(raw) & (raw > 1.0e-14))
    if len(valid) == 0:
        return [], [], float("nan")

    centroids = _face_uv_centroids(parameterization)
    curvature = _gaussian_angle_defect(parameterization)
    queue: list[_Part] = [_Part(valid, 0)]
    accepted: list[np.ndarray] = []
    lines: list[tuple[str, float]] = []
    events: list[dict[str, Any]] = []

    while queue:
        part = queue.pop(0)
        sigma = _part_sigma(raw, part.face_ids)
        if not np.isfinite(sigma) or sigma <= float(threshold) + EPS:
            accepted.append(part.face_ids)
            continue
        if len(lines) >= int(max_splits):
            accepted.append(part.face_ids)
            events.append(
                {
                    "status": "budget_exhausted",
                    "depth": int(part.depth),
                    "faces": int(len(part.face_ids)),
                    "sigma_before": float(sigma),
                }
            )
            continue

        chosen = _choose_complete_grid_split(
            parameterization, part.face_ids, curvature, centroids
        )
        if chosen is None:
            accepted.append(part.face_ids)
            events.append(
                {
                    "status": "no_valid_complete_cut",
                    "depth": int(part.depth),
                    "faces": int(len(part.face_ids)),
                    "sigma_before": float(sigma),
                }
            )
            continue

        axis, value, peak_vid = chosen
        first, second = _split_partition(centroids, part.face_ids, axis, value)
        if len(first) == 0 or len(second) == 0:
            accepted.append(part.face_ids)
            events.append(
                {
                    "status": "degenerate_cut",
                    "depth": int(part.depth),
                    "faces": int(len(part.face_ids)),
                    "sigma_before": float(sigma),
                    "axis": axis,
                    "value": float(value),
                }
            )
            continue

        line = (str(axis), float(value))
        # De-duplicate numerically identical requests.
        if not any(a == line[0] and abs(v - line[1]) <= 1.0e-10 for a, v in lines):
            lines.append(line)
        event = {
            "status": "split",
            "step": int(len(lines)),
            "depth": int(part.depth),
            "faces": int(len(part.face_ids)),
            "sigma_before": float(sigma),
            "axis": str(axis),
            "value": float(value),
            "peak_surface_vertex": int(peak_vid),
            "peak_abs_angle_defect": (
                float(abs(curvature[peak_vid])) if peak_vid >= 0 else float("nan")
            ),
            "child_faces": [int(len(first)), int(len(second))],
            "child_sigma": [
                float(_part_sigma(raw, first)),
                float(_part_sigma(raw, second)),
            ],
        }
        events.append(event)
        queue.append(_Part(first, part.depth + 1))
        queue.append(_Part(second, part.depth + 1))

    residuals = [
        _part_sigma(raw, ids)
        for ids in accepted
        if len(ids) and np.isfinite(_part_sigma(raw, ids))
    ]
    residual_max = float(max(residuals)) if residuals else float("nan")
    return lines, events, residual_max


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(0, int(raw))
    except ValueError:
        return int(default)


def _write_domain_field(domain: Any, name: str, value: Any) -> None:
    try:
        setattr(domain, name, value)
    except Exception:
        metrics = getattr(domain, "metrics", None)
        if isinstance(metrics, dict):
            metrics[name] = value
        else:
            raise


def install_optcuts_csf_split_patch(pipeline: Any) -> None:
    """Wrap S->Omega domain construction with exact CSF measurement/split planning."""
    if getattr(pipeline, "_onestring_optcuts_csf_split_patch_installed", False):
        return

    base = pipeline._flatten_to_domain

    def measured_flatten(parameterization: Any, grid: Any, params: Any):
        domain = base(parameterization, grid, params)
        mode = str(getattr(params, "omega_parameterization_mode", ""))
        if mode != "optcuts_test":
            return domain

        raw = _triangle_raw_lambda(parameterization)
        normalized, raw_min = _normalized_global(raw)
        valid = np.isfinite(normalized)
        lambda_min = float(np.min(normalized[valid]))
        lambda_max = float(np.max(normalized[valid]))
        violating = np.flatnonzero(normalized > BOUND + EPS)
        max_splits = _env_int("ONESTRING_OPTCUTS_CSF_MAX_SPLITS", 16)

        lines, events, residual_max = _hierarchical_split_plan(
            parameterization,
            raw,
            threshold=BOUND,
            max_splits=max_splits,
        )
        vertex_values = _vertex_csf(parameterization, normalized)

        _write_domain_field(domain, "csf_values", vertex_values)
        _write_domain_field(domain, "csf_before", lambda_max)
        _write_domain_field(domain, "csf_after_split", residual_max)
        _write_domain_field(domain, "csf_split_threshold", BOUND)
        _write_domain_field(domain, "split_lines", list(lines))
        _write_domain_field(domain, "localized_split_segments", list(lines))
        _write_domain_field(domain, "localize_csf_splits", False)
        _write_domain_field(domain, "peak_uv_target", None)
        _write_domain_field(domain, "peak_grid_alignment_shift", 0.0)

        metrics = getattr(parameterization, "metrics", None)
        if isinstance(metrics, dict):
            metrics.update(
                {
                    "optcuts_csf_exact_measurement": True,
                    "optcuts_csf_definition": "sigma_max(J_UV_to_surface)",
                    "optcuts_csf_global_normalization": "divide by global minimum raw lambda",
                    "optcuts_csf_raw_min": float(raw_min),
                    "optcuts_csf_lambda_min": float(lambda_min),
                    "optcuts_csf_lambda_max": float(lambda_max),
                    "optcuts_csf_bound": float(BOUND),
                    "optcuts_csf_violating_triangle_count": int(len(violating)),
                    "optcuts_csf_valid_triangle_count": int(np.count_nonzero(valid)),
                    "optcuts_csf_planned_split_count": int(len(lines)),
                    "optcuts_csf_split_lines": [[a, float(v)] for a, v in lines],
                    "optcuts_csf_residual_part_max": float(residual_max),
                    "optcuts_csf_max_splits": int(max_splits),
                    "optcuts_csf_split_events": events,
                    "optcuts_csf_paper_style_note": (
                        "complete row/column cuts are requested through high-curvature locations; "
                        "existing Simple Split snaps requests to fabrication-grid lines"
                    ),
                }
            )

        print(
            "[OPTCUTS-CSF] "
            f"lambda_min={lambda_min:.9g} lambda_max={lambda_max:.9g} "
            f"bound={BOUND:g} violating_triangles={len(violating)}/{np.count_nonzero(valid)} "
            f"planned_splits={len(lines)} residual_part_max={residual_max:.9g}"
        )
        for event in events:
            if event.get("status") == "split":
                print(
                    "[OPTCUTS-CSF-SPLIT] "
                    f"step={event['step']} depth={event['depth']} "
                    f"sigma_before={event['sigma_before']:.9g} "
                    f"line={event['axis']}:{event['value']:.9g} "
                    f"children={event['child_sigma']}"
                )
            else:
                print(f"[OPTCUTS-CSF-SPLIT-WARN] {event}")
        if np.isfinite(residual_max) and residual_max > BOUND + EPS:
            print(
                "[OPTCUTS-CSF-WARN] split plan exhausted before all parts reached "
                f"sigma<=2; residual_part_max={residual_max:.9g}"
            )
        return domain

    pipeline._flatten_to_domain = measured_flatten

    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._flatten_to_domain = measured_flatten

    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = measured_flatten

    pipeline._onestring_optcuts_csf_split_patch_installed = True


__all__ = [
    "BOUND",
    "_triangle_raw_lambda",
    "_normalized_global",
    "_part_sigma",
    "_split_partition",
    "_hierarchical_split_plan",
    "install_optcuts_csf_split_patch",
]
