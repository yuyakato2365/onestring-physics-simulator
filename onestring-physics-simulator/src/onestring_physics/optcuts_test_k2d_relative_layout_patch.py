"""Keep OptCuts_test K2D close to the relative M2D tile layout.

The ordinary K2D solver remains authoritative for K3D edge-length matching.  In
``optcuts_test`` only, this patch adds a conservative post-projection that keeps
adjacent tile-center vectors close to their M2D values (up to one global scale),
while repeatedly re-projecting mesh edges back to their K3D target lengths.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _unique_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in np.asarray(faces, dtype=int):
        ids = [int(v) for v in np.asarray(face).reshape(-1)]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            if a != b:
                edges.add(_edge_key(a, b))
    return sorted(edges)


def _adjacent_face_pairs(faces: np.ndarray) -> list[tuple[int, int]]:
    incidence: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(np.asarray(faces, dtype=int)):
        ids = [int(v) for v in np.asarray(face).reshape(-1)]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            if a != b:
                incidence[_edge_key(a, b)].append(int(fi))
    pairs: set[tuple[int, int]] = set()
    for owners in incidence.values():
        if len(owners) == 2:
            a, b = sorted((int(owners[0]), int(owners[1])))
            pairs.add((a, b))
    return sorted(pairs)


def _edge_errors(xy: np.ndarray, edges: list[tuple[int, int]], target: np.ndarray) -> tuple[float, float]:
    if not edges:
        return 0.0, 0.0
    vals = np.asarray([np.linalg.norm(xy[b] - xy[a]) for a, b in edges], dtype=float)
    err = np.abs(vals - np.asarray(target, dtype=float))
    return float(np.mean(err)), float(np.max(err))


def _project_edge_lengths(xy: np.ndarray, edges: list[tuple[int, int]], target: np.ndarray, strength: float = 0.55) -> np.ndarray:
    out = np.asarray(xy, dtype=float).copy()
    if not edges:
        return out
    acc = np.zeros_like(out)
    weight = np.zeros((len(out), 1), dtype=float)
    for (a, b), length in zip(edges, target):
        d = out[b] - out[a]
        n = float(np.linalg.norm(d))
        if n <= 1e-12:
            continue
        corr = 0.5 * float(strength) * (n - float(length)) * (d / n)
        acc[a] += corr
        acc[b] -= corr
        weight[a, 0] += 1.0
        weight[b, 0] += 1.0
    active = weight[:, 0] > 0
    out[active] += acc[active] / np.maximum(weight[active], 1.0)
    return out


def _relative_center_rms(xy: np.ndarray, faces: np.ndarray, pairs: list[tuple[int, int]], targets: np.ndarray) -> float:
    if not pairs:
        return 0.0
    centers = np.mean(xy[np.asarray(faces, dtype=int)], axis=1)
    vals = np.asarray([(centers[j] - centers[i]) - targets[k] for k, (i, j) in enumerate(pairs)], dtype=float)
    return float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0


def _preserve_relative_layout(mesh_2d: Any, mesh_3d: Any, k2d: Any) -> dict[str, float | int | bool | str]:
    faces = np.asarray(mesh_2d.faces, dtype=int)
    base_xy = np.asarray(mesh_2d.vertices, dtype=float)[:, :2]
    xy0 = np.asarray(k2d.vertices, dtype=float)[:, :2]
    if len(faces) == 0 or len(xy0) == 0:
        return {"optcuts_test_k2d_relative_layout_applied": False}

    pairs = _adjacent_face_pairs(faces)
    edges = _unique_edges(faces)
    if not pairs or not edges:
        return {
            "optcuts_test_k2d_relative_layout_applied": False,
            "optcuts_test_k2d_relative_layout_pair_count": int(len(pairs)),
        }

    target_lengths = np.asarray(
        [np.linalg.norm(np.asarray(mesh_3d.vertices, float)[b] - np.asarray(mesh_3d.vertices, float)[a]) for a, b in edges],
        dtype=float,
    )
    base_lengths = np.asarray([np.linalg.norm(base_xy[b] - base_xy[a]) for a, b in edges], dtype=float)
    mean_base = float(np.mean(base_lengths)) if len(base_lengths) else 1.0
    mean_target = float(np.mean(target_lengths)) if len(target_lengths) else mean_base
    global_scale = mean_target / max(mean_base, 1e-12)

    base_centers = np.mean(base_xy[faces], axis=1)
    rel_targets = np.asarray(
        [global_scale * (base_centers[j] - base_centers[i]) for i, j in pairs],
        dtype=float,
    )

    before_rel = _relative_center_rms(xy0, faces, pairs, rel_targets)
    before_mean, before_max = _edge_errors(xy0, edges, target_lengths)
    xy = xy0.copy()

    # Strong enough to stop component scattering, but edge projection remains
    # authoritative so K2D still tracks the K3D metric.
    relative_step = 0.22
    iterations = 36
    for _ in range(iterations):
        centers = np.mean(xy[faces], axis=1)
        face_shift = np.zeros((len(faces), 2), dtype=float)
        face_weight = np.zeros((len(faces), 1), dtype=float)
        for k, (i, j) in enumerate(pairs):
            err = (centers[j] - centers[i]) - rel_targets[k]
            delta = 0.5 * relative_step * err
            face_shift[i] += delta
            face_shift[j] -= delta
            face_weight[i, 0] += 1.0
            face_weight[j, 0] += 1.0

        vertex_shift = np.zeros_like(xy)
        vertex_weight = np.zeros((len(xy), 1), dtype=float)
        for fi, face in enumerate(faces):
            if face_weight[fi, 0] <= 0:
                continue
            shift = face_shift[fi] / face_weight[fi, 0]
            for vid in face:
                vertex_shift[int(vid)] += shift
                vertex_weight[int(vid), 0] += 1.0
        active = vertex_weight[:, 0] > 0
        xy[active] += vertex_shift[active] / np.maximum(vertex_weight[active], 1.0)

        # Restore K3D edge-length agreement after each relative-layout step.
        xy = _project_edge_lengths(xy, edges, target_lengths, strength=0.65)
        xy = _project_edge_lengths(xy, edges, target_lengths, strength=0.65)

        # Remove global translation drift only; do not rotate the sheet.
        xy -= np.mean(xy, axis=0, keepdims=True) - np.mean(xy0, axis=0, keepdims=True)

    after_rel = _relative_center_rms(xy, faces, pairs, rel_targets)
    after_mean, after_max = _edge_errors(xy, edges, target_lengths)

    # Never let the relative-layout regularizer materially destroy the primary
    # K2D requirement (edge lengths inherited from K3D).  Blend back until the
    # max error is within 10% of the incoming K2D solution (plus numerical slack).
    allowed_max = max(before_max * 1.10, before_max + 1e-6)
    blend = 1.0
    while after_max > allowed_max and blend > 1.0 / 64.0:
        blend *= 0.5
        trial = xy0 + blend * (xy - xy0)
        trial_mean, trial_max = _edge_errors(trial, edges, target_lengths)
        if trial_max <= allowed_max:
            xy = trial
            after_mean, after_max = trial_mean, trial_max
            after_rel = _relative_center_rms(xy, faces, pairs, rel_targets)
            break
    else:
        if after_max > allowed_max:
            xy = xy0.copy()
            after_mean, after_max = before_mean, before_max
            after_rel = before_rel
            blend = 0.0

    k2d.vertices[:, :2] = xy
    try:
        k2d.metrics.update({
            "optcuts_test_k2d_relative_layout_applied": True,
            "optcuts_test_k2d_relative_layout_model": "adjacent tile center vectors anchored to globally-scaled M2D + alternating K3D edge projection",
            "optcuts_test_k2d_relative_layout_pair_count": int(len(pairs)),
            "optcuts_test_k2d_relative_layout_global_scale": float(global_scale),
            "optcuts_test_k2d_relative_layout_rms_before": float(before_rel),
            "optcuts_test_k2d_relative_layout_rms_after": float(after_rel),
            "optcuts_test_k2d_relative_layout_edge_mean_before": float(before_mean),
            "optcuts_test_k2d_relative_layout_edge_mean_after": float(after_mean),
            "optcuts_test_k2d_relative_layout_edge_max_before": float(before_max),
            "optcuts_test_k2d_relative_layout_edge_max_after": float(after_max),
            "optcuts_test_k2d_relative_layout_blend": float(blend),
            "optcuts_test_k2d_relative_layout_iterations": int(iterations),
        })
    except Exception:
        pass
    print(
        "[OPTCUTS-TEST-K2D-RELATIVE] "
        f"pairs={len(pairs)} rel_rms={before_rel:.6g}->{after_rel:.6g} "
        f"edge_max={before_max:.6g}->{after_max:.6g} blend={blend:.4f}"
    )
    return dict(getattr(k2d, "metrics", {}) or {})


def install_optcuts_test_k2d_relative_layout_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_k2d_relative_layout_installed", False):
        return
    base = pipeline._optimize_k2d

    def optimize(mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        result = base(mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        k2d, report = result
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return result
        _preserve_relative_layout(mesh_2d, mesh_3d, k2d)
        return k2d, report

    pipeline._optimize_k2d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k2d = optimize
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k2d"] = optimize
    pipeline._onestring_optcuts_test_k2d_relative_layout_installed = True


__all__ = ["install_optcuts_test_k2d_relative_layout_patch"]
