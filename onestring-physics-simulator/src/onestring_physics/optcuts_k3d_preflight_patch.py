"""Authoritative T3D preflight for the OptCuts seam path.

Ordinary OptCuts keeps this as a final assertion. In optcuts_test, invalid K3D
panels remain visible on K3D but are removed from the copy passed to T3D and
all later tile-topology processing.  The retained original face ids are carried
forward so hinge/gap topology can be rebuilt on the same compact tile indexing.
"""
from __future__ import annotations

from collections import Counter
import copy
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
    if bool(metrics.get("optcuts_k3d_nonfatal_diagnostic_mode", False)):
        return True
    if bool(metrics.get("optcuts_test_invalid_face_ids", [])):
        return True
    if bool(getattr(mesh, "_optcuts_test_face_sources", None)):
        return True
    if bool(getattr(mesh, "_optcuts_invalid_face_ids", None)):
        return True
    if bool(getattr(mesh, "_optcuts_test_invalid_face_ids", None)):
        return True
    if bool(getattr(pipeline, "_onestring_optcuts_test_active_run", False)):
        return True
    original_module = getattr(pipeline, "_original", None)
    return bool(getattr(original_module, "_onestring_optcuts_test_active_run", False))


def _t3d_top_tiles(pipeline: Any, mesh: Any) -> np.ndarray:
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


def _filtered_mesh_without_invalid(mesh: Any, invalid: list[dict[str, Any]]) -> tuple[Any, list[int]]:
    faces = np.asarray(mesh.faces, dtype=int)
    n_faces = int(len(faces))
    drop = {int(item["tile_id"]) for item in invalid if 0 <= int(item["tile_id"]) < n_faces}
    keep_ids = [i for i in range(n_faces) if i not in drop]
    filtered = copy.copy(mesh)
    filtered.faces = faces[np.asarray(keep_ids, dtype=int)].copy()

    try:
        filtered.metrics = dict(getattr(mesh, "metrics", {}) or {})
        filtered.metrics.update({
            "optcuts_test_invalid_panels_excluded": True,
            "optcuts_test_excluded_original_face_ids": sorted(drop),
            "optcuts_test_excluded_face_count": int(len(drop)),
            "optcuts_test_retained_face_count": int(len(keep_ids)),
            "optcuts_test_retained_original_face_ids": keep_ids,
        })
    except Exception:
        pass

    for name in (
        "_optcuts_test_face_sources",
        "_optcuts_test_face_uv",
        "_polygon_faces",
        "_polygon_face_sources",
    ):
        value = getattr(mesh, name, None)
        if value is None:
            continue
        try:
            if len(value) != n_faces:
                continue
            if isinstance(value, np.ndarray):
                new_value = value[np.asarray(keep_ids, dtype=int)].copy()
            else:
                new_value = [value[i] for i in keep_ids]
            setattr(filtered, name, new_value)
        except Exception:
            pass

    try:
        setattr(filtered, "_optcuts_test_retained_original_face_ids", keep_ids)
        setattr(filtered, "_optcuts_test_excluded_original_face_ids", sorted(drop))
        setattr(filtered, "_optcuts_test_invalid_face_ids", [])
        setattr(filtered, "_optcuts_invalid_face_ids", [])
    except Exception:
        pass
    return filtered, keep_ids


def _retained_ids_from_assembly(assembly: Any) -> list[int] | None:
    value = getattr(assembly, "_optcuts_test_retained_original_face_ids", None)
    if value is None:
        value = dict(getattr(assembly, "metrics", {}) or {}).get("optcuts_test_retained_original_face_ids")
    if value is None:
        return None
    try:
        return [int(v) for v in value]
    except Exception:
        return None


def _remap_topology_faces(mesh_faces: np.ndarray, assembly: Any, label: str) -> np.ndarray:
    """Compact original M2D face rows to the retained T3D tile ordering.

    After invalid K3D tiles are dropped, TileAssembly rows are compacted to
    0..N-1.  Topology builders must therefore use the same retained original
    face rows; otherwise HingeSpec tile ids still refer to the pre-drop array.
    """
    faces = np.asarray(mesh_faces, dtype=int)
    tile_count = int(getattr(assembly, "tile_count", len(getattr(assembly, "vertices", []))))
    if len(faces) == tile_count:
        return faces
    retained = _retained_ids_from_assembly(assembly)
    if retained is None or len(retained) != tile_count:
        return faces
    if retained and (min(retained) < 0 or max(retained) >= len(faces)):
        return faces
    remapped = faces[np.asarray(retained, dtype=int)].copy()
    print(
        f"[OPTCUTS-TEST-TOPOLOGY-REMAP][{label}] "
        f"original_faces={len(faces)} retained_faces={len(remapped)}"
    )
    return remapped


def install_optcuts_k3d_preflight_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_k3d_preflight_installed", False):
        return

    base_extrude = pipeline._extrude_tiles
    base_hinge = getattr(pipeline, "_build_hinge_graph", None)
    base_dual = getattr(pipeline, "_optimize_dual_hinges", None)
    base_gap = getattr(pipeline, "_build_gap_graph", None)

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
                filtered_mesh, keep_ids = _filtered_mesh_without_invalid(mesh, invalid)
                print(
                    "[OPTCUTS-TEST-DROP-INVALID][K3D-PREFLIGHT] "
                    f"excluded={len(invalid)} retained={len(keep_ids)} "
                    f"reasons={reason_counts} examples={examples} log={log_path}; "
                    "extruding only valid panels"
                )
                result = base_extrude(filtered_mesh, thickness, stage)
                out_mesh = result[0] if isinstance(result, tuple) and result else result
                if out_mesh is not None:
                    try:
                        out_mesh.metrics.update({
                            "optcuts_test_invalid_panels_excluded": True,
                            "optcuts_test_excluded_original_face_ids": [int(i["tile_id"]) for i in invalid],
                            "optcuts_test_retained_original_face_ids": keep_ids,
                        })
                        setattr(out_mesh, "_optcuts_test_retained_original_face_ids", keep_ids)
                        setattr(out_mesh, "_optcuts_test_excluded_original_face_ids", [int(i["tile_id"]) for i in invalid])
                    except Exception:
                        pass
                return result

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

    def hinge_with_optcuts_remap(grid, mesh_faces, t2d, t3d, dual):
        faces = _remap_topology_faces(mesh_faces, t3d, "HINGE")
        return base_hinge(grid, faces, t2d, t3d, dual)

    def dual_with_optcuts_remap(grid, mesh_faces, t2d, t3d, params=None, progress_callback=None):
        faces = _remap_topology_faces(mesh_faces, t3d, "DUAL")
        return base_dual(grid, faces, t2d, t3d, params, progress_callback=progress_callback)

    def gap_with_optcuts_remap(mesh_faces, t2d, t3d):
        faces = _remap_topology_faces(mesh_faces, t3d, "GAP")
        return base_gap(faces, t2d, t3d)

    pipeline._extrude_tiles = extrude_with_optcuts_k3d_preflight
    if callable(base_hinge):
        pipeline._build_hinge_graph = hinge_with_optcuts_remap
    if callable(base_dual):
        pipeline._optimize_dual_hinges = dual_with_optcuts_remap
    if callable(base_gap):
        pipeline._build_gap_graph = gap_with_optcuts_remap

    original = getattr(pipeline, "_original", None)
    if original is not None:
        if callable(base_hinge):
            original._build_hinge_graph = hinge_with_optcuts_remap
        if callable(base_dual):
            original._optimize_dual_hinges = dual_with_optcuts_remap
        if callable(base_gap):
            original._build_gap_graph = gap_with_optcuts_remap

    # The build function and the dual optimizer resolve these helpers through
    # their module globals, so patch those globals too.
    functions = [
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
        base_dual,
    ]
    for fn in functions:
        glb = getattr(fn, "__globals__", None)
        if not isinstance(glb, dict):
            continue
        glb["_extrude_tiles"] = extrude_with_optcuts_k3d_preflight
        if callable(base_hinge):
            glb["_build_hinge_graph"] = hinge_with_optcuts_remap
        if callable(base_dual):
            glb["_optimize_dual_hinges"] = dual_with_optcuts_remap
        if callable(base_gap):
            glb["_build_gap_graph"] = gap_with_optcuts_remap

    pipeline._onestring_optcuts_k3d_preflight_installed = True


__all__ = [
    "install_optcuts_k3d_preflight_patch",
    "_invalid_tops",
    "_filtered_mesh_without_invalid",
    "_remap_topology_faces",
]
