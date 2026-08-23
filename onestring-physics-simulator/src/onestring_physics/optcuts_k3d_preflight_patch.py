"""Authoritative T3D preflight for the OptCuts seam path.

Ordinary OptCuts keeps this as a final assertion.  OptCuts_test is a diagnostic
mode: invalid tops are logged and tagged on the mesh, but the pipeline is
allowed to continue so the offending tiles can be visualized downstream.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .t3d_recovery import validate_top_quad


def _optcuts_active(pipeline: Any, mesh: Any) -> bool:
    if bool(getattr(pipeline, "_onestring_optcuts_active_run", False)):
        return True
    original_module = getattr(pipeline, "_original", None)
    if bool(getattr(original_module, "_onestring_optcuts_active_run", False)):
        return True
    metrics = dict(getattr(mesh, "metrics", {}) or {})
    return bool(
        metrics.get("optcuts_grid_seam_enabled", False)
        or metrics.get("optcuts_grid_seam_applied", False)
        or metrics.get("flattening_backend") == "official_optcuts_external"
        or metrics.get("omega_parameterization_mode") in ("optcuts", "optcuts_test")
        or metrics.get("parameterization_method") in ("optcuts", "optcuts_test")
    )


def _optcuts_test_mode(pipeline: Any, mesh: Any) -> bool:
    metrics = dict(getattr(mesh, "metrics", {}) or {})
    if metrics.get("omega_parameterization_mode") == "optcuts_test":
        return True
    if metrics.get("parameterization_method") == "optcuts_test":
        return True
    if bool(metrics.get("optcuts_test_boundary_clip", False)):
        return True
    if bool(getattr(mesh, "_optcuts_test_face_sources", None)):
        return True
    if bool(getattr(pipeline, "_onestring_optcuts_test_active_run", False)):
        return True
    original_module = getattr(pipeline, "_original", None)
    return bool(getattr(original_module, "_onestring_optcuts_test_active_run", False))


def _t3d_top_tiles(pipeline: Any, mesh: Any) -> np.ndarray:
    """Use the exact tile-construction path consumed by T3D."""
    original_module = getattr(pipeline, "_original", None)
    fn = getattr(original_module, "_mesh_tiles", None)
    if not callable(fn):
        fn = getattr(pipeline, "_mesh_tiles", None)
    if callable(fn):
        return np.asarray(fn(mesh), dtype=float)
    return np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=int)]


def _invalid_tops(pipeline: Any, mesh: Any) -> list[dict[str, Any]]:
    tops = _t3d_top_tiles(pipeline, mesh)
    faces = np.asarray(mesh.faces, dtype=int)
    invalid: list[dict[str, Any]] = []
    for tile_id, top in enumerate(tops):
        valid, reason = validate_top_quad(np.asarray(top, dtype=float))
        if valid:
            continue
        face = faces[tile_id] if tile_id < len(faces) else np.asarray([], dtype=int)
        invalid.append({
            "tile_id": int(tile_id),
            "reason": str(reason),
            "face_vertex_ids": [int(v) for v in np.asarray(face, dtype=int).reshape(-1)],
            "top": np.asarray(top, dtype=float).tolist(),
        })
    return invalid


def _write_failure_log(mesh: Any, invalid: list[dict[str, Any]]) -> str:
    root = Path(__file__).resolve().parents[2]
    path = root / "logs" / "optcuts_k3d_invalid.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(getattr(mesh, "stage", "")),
        "vertex_count": int(len(np.asarray(mesh.vertices))),
        "face_count": int(len(np.asarray(mesh.faces))),
        "reasons": dict(Counter(str(item["reason"]) for item in invalid)),
        "invalid": invalid[:64],
        "metrics": {
            key: value
            for key, value in dict(getattr(mesh, "metrics", {}) or {}).items()
            if isinstance(value, (str, int, float, bool))
        },
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path)


def _tag_invalid_for_visualization(mesh: Any, invalid: list[dict[str, Any]], stage: str) -> None:
    ids = [int(item["tile_id"]) for item in invalid]
    reasons = {int(item["tile_id"]): str(item["reason"]) for item in invalid}
    try:
        setattr(mesh, "_optcuts_test_invalid_face_ids", ids)
        setattr(mesh, "_optcuts_test_invalid_face_reasons", reasons)
        setattr(mesh, "_optcuts_test_invalid_stage", str(stage))
    except Exception:
        pass
    try:
        mesh.metrics.update({
            "optcuts_test_invalid_face_ids": ids,
            "optcuts_test_invalid_face_count": int(len(ids)),
            "optcuts_test_invalid_stage": str(stage),
        })
    except Exception:
        pass


def install_optcuts_k3d_preflight_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_k3d_preflight_installed", False):
        return
    base_extrude = pipeline._extrude_tiles

    def extrude_with_optcuts_k3d_preflight(mesh: Any, thickness: float, stage: str):
        if str(stage) != "T3D":
            return base_extrude(mesh, thickness, stage)

        active = _optcuts_active(pipeline, mesh)
        if not active:
            print("[OPTCUTS-K3D-PREFLIGHT] skipped active_run=False")
            return base_extrude(mesh, thickness, stage)

        invalid = _invalid_tops(pipeline, mesh)
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        reason_counts = dict(Counter(str(item["reason"]) for item in invalid))
        metrics.update({
            "optcuts_k3d_preflight_applied": True,
            "optcuts_k3d_preflight_top_source": "same _mesh_tiles path as T3D",
            "optcuts_k3d_invalid_top_count": int(len(invalid)),
            "optcuts_k3d_invalid_top_reason_counts": reason_counts,
        })
        try:
            mesh.metrics.update(metrics)
        except Exception:
            pass

        if invalid:
            log_path = _write_failure_log(mesh, invalid)
            _tag_invalid_for_visualization(mesh, invalid, "K3D->T3D preflight")
            examples = [(i["tile_id"], i["reason"]) for i in invalid[:8]]
            if _optcuts_test_mode(pipeline, mesh):
                print(
                    "[OPTCUTS-TEST-NONFATAL][K3D-PREFLIGHT] "
                    f"invalid_tops={len(invalid)} reasons={reason_counts} "
                    f"examples={examples} log={log_path}; continuing to T3D"
                )
                try:
                    result = base_extrude(mesh, thickness, stage)
                    # Preserve diagnostic ids on the produced T3D object when possible.
                    out_mesh = result[0] if isinstance(result, tuple) and result else result
                    if out_mesh is not None:
                        _tag_invalid_for_visualization(out_mesh, invalid, "K3D->T3D preflight")
                    return result
                except Exception as exc:
                    print(
                        "[OPTCUTS-TEST][T3D-EXTRUSION-ERROR] "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise

            print(
                "[OPTCUTS-K3D-PREFLIGHT] FAILED "
                f"invalid_tops={len(invalid)} reasons={reason_counts} "
                f"examples={examples} log={log_path}"
            )
            raise RuntimeError(
                "OPTCUTS_K3D_PREFLIGHT_FAILED: authoritative T3D tops are invalid; "
                f"reasons={reason_counts}; log={log_path}"
            )

        print("[OPTCUTS-K3D-PREFLIGHT] active_run=True invalid_tops=0 source=_mesh_tiles")
        return base_extrude(mesh, thickness, stage)

    pipeline._extrude_tiles = extrude_with_optcuts_k3d_preflight
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_extrude_tiles"] = extrude_with_optcuts_k3d_preflight
    pipeline._onestring_optcuts_k3d_preflight_installed = True


__all__ = ["install_optcuts_k3d_preflight_patch", "_invalid_tops"]
