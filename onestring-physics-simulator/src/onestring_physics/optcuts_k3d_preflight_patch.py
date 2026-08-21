"""Final K3D validity preflight for the OptCuts seam path.

T3D requires every quad top to satisfy exactly the same conditions checked by
``t3d_recovery.validate_top_quad``.  M2D can be valid and still become invalid
after K2D/K3D optimization (for example a bow-tie/self-intersecting quad).

This patch is intentionally OptCuts-only.  Immediately before T3D extrusion it
validates every K3D top with the authoritative validator, removes invalid tile
faces, records the reason/tile ids, and then delegates to the unchanged T3D
constructor.  Non-OptCuts runs are untouched.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .t3d_recovery import validate_top_quad


def _copy_mesh_with_faces(pipeline: Any, mesh: Any, faces: np.ndarray, metrics: dict[str, Any]):
    cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
    out = cls(
        np.asarray(mesh.vertices, dtype=float).copy(),
        np.asarray(faces, dtype=int).copy(),
        mesh.grid,
        mesh.stage,
        metrics,
        list(getattr(mesh, "split_lines", [])),
    )
    for name in (
        "_split_panel_source_vertices",
        "_split_panel_face_components",
        "_split_panel_vertex_components",
        "_split_panel_offsets",
        "_optcuts_grid_seam_cut_edges",
        "_optcuts_grid_seam_paths",
    ):
        if hasattr(mesh, name):
            try:
                setattr(out, name, getattr(mesh, name))
            except Exception:
                pass
    return out


def _invalid_top_faces(mesh: Any) -> list[tuple[int, str]]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    invalid: list[tuple[int, str]] = []
    for fi, face in enumerate(faces):
        try:
            top = vertices[np.asarray(face, dtype=int)]
        except Exception:
            invalid.append((int(fi), "invalid_face_index"))
            continue
        valid, reason = validate_top_quad(np.asarray(top, dtype=float))
        if not valid:
            invalid.append((int(fi), str(reason)))
    return invalid


def install_optcuts_k3d_preflight_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_k3d_preflight_installed", False):
        return
    base_extrude = pipeline._extrude_tiles

    def extrude_with_optcuts_k3d_preflight(mesh: Any, thickness: float, stage: str):
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        optcuts_active = bool(
            metrics.get("optcuts_grid_seam_enabled", False)
            or metrics.get("optcuts_grid_seam_applied", False)
            or metrics.get("flattening_backend") == "official_optcuts_external"
            or metrics.get("omega_parameterization_mode") == "optcuts"
        )
        if not optcuts_active or str(stage) != "T3D":
            return base_extrude(mesh, thickness, stage)

        invalid = _invalid_top_faces(mesh)
        if not invalid:
            metrics.update({
                "optcuts_k3d_preflight_applied": True,
                "optcuts_k3d_invalid_top_count": 0,
                "optcuts_k3d_invalid_top_reason_counts": {},
            })
            try:
                mesh.metrics.update(metrics)
            except Exception:
                pass
            print("[OPTCUTS-K3D-PREFLIGHT] invalid_tops=0")
            return base_extrude(mesh, thickness, stage)

        faces = np.asarray(mesh.faces, dtype=int)
        bad_ids = {int(fi) for fi, _reason in invalid}
        keep = np.asarray([fi for fi in range(len(faces)) if fi not in bad_ids], dtype=int)
        if len(keep) == 0:
            raise RuntimeError(
                "OPTCUTS_K3D_PREFLIGHT_FAILED: every K3D tile is invalid; "
                f"examples={invalid[:12]}"
            )
        reason_counts = Counter(reason for _fi, reason in invalid)
        metrics.update({
            "optcuts_k3d_preflight_applied": True,
            "optcuts_k3d_invalid_top_count": int(len(invalid)),
            "optcuts_k3d_invalid_top_face_ids": [int(fi) for fi, _reason in invalid[:256]],
            "optcuts_k3d_invalid_top_reason_counts": dict(reason_counts),
            "optcuts_k3d_preflight_removed_face_count": int(len(invalid)),
            "optcuts_k3d_preflight_model": "authoritative validate_top_quad before T3D extrusion",
        })
        filtered = _copy_mesh_with_faces(pipeline, mesh, faces[keep], metrics)
        print(
            "[OPTCUTS-K3D-PREFLIGHT] "
            f"invalid_tops={len(invalid)} removed={len(invalid)} "
            f"reasons={dict(reason_counts)} examples={invalid[:8]}"
        )
        return base_extrude(filtered, thickness, stage)

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


__all__ = ["install_optcuts_k3d_preflight_patch", "_invalid_top_faces"]
