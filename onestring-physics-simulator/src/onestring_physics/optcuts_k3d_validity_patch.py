"""Validity-preserving K3D wrapper for the OptCuts fabrication-seam path.

The experimental OptCuts/rectilinear seam route can create K3D optimization
candidates whose quad vertex ordering crosses (a bow-tie) even when the lifted
M3D mesh is valid.  T3D correctly rejects such tops.

Do not delete those tiles: downstream T2D/hinge/string code relies on stable
face/tile ids.  Instead backtrack the whole K3D displacement from the valid M3D
state toward the optimizer candidate and keep the largest safe step for which
*every* quad passes the exact authoritative ``validate_top_quad`` check used by
T3D.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .t3d_recovery import validate_top_quad


def _invalid_faces(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[int, str]]:
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    invalid: list[tuple[int, str]] = []
    for fi, face in enumerate(f):
        try:
            top = verts[np.asarray(face, dtype=int)]
        except Exception:
            invalid.append((int(fi), "invalid_face_index"))
            continue
        valid, reason = validate_top_quad(np.asarray(top, dtype=float))
        if not valid:
            invalid.append((int(fi), str(reason)))
    return invalid


def _optcuts_active(pipeline: Any, mesh: Any | None = None) -> bool:
    if bool(getattr(pipeline, "_onestring_optcuts_active_run", False)):
        return True
    metrics = dict(getattr(mesh, "metrics", {}) or {}) if mesh is not None else {}
    return bool(
        metrics.get("optcuts_grid_seam_enabled", False)
        or metrics.get("optcuts_grid_seam_applied", False)
        or metrics.get("flattening_backend") == "official_optcuts_external"
        or metrics.get("omega_parameterization_mode") == "optcuts"
        or metrics.get("parameterization_method") == "optcuts"
    )


def install_optcuts_k3d_validity_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_k3d_validity_patch_installed", False):
        return

    base_optimize = pipeline._optimize_k3d

    def optimize_k3d_with_validity(target: Any, mesh: Any, parameterization: Any, params: Any):
        out, report = base_optimize(target, mesh, parameterization, params)
        if not _optcuts_active(pipeline, mesh) and str(getattr(parameterization, "method", "")) != "optcuts":
            return out, report

        base_vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(out.faces, dtype=int)
        candidate = np.asarray(out.vertices, dtype=float)
        base_invalid = _invalid_faces(base_vertices, faces)
        if base_invalid:
            reasons = dict(Counter(reason for _fi, reason in base_invalid))
            raise RuntimeError(
                "OPTCUTS_INVALID_M3D_BEFORE_K3D: lifted M3D already contains invalid quads; "
                f"reasons={reasons} examples={base_invalid[:12]}"
            )

        candidate_invalid = _invalid_faces(candidate, faces)
        metrics = dict(getattr(out, "metrics", {}) or {})
        metrics.update({
            "optcuts_k3d_validity_guard_applied": True,
            "optcuts_k3d_invalid_candidate_count": int(len(candidate_invalid)),
            "optcuts_k3d_invalid_candidate_reason_counts": dict(
                Counter(reason for _fi, reason in candidate_invalid)
            ),
        })
        if not candidate_invalid:
            metrics.update({
                "optcuts_k3d_validity_backtracked": False,
                "optcuts_k3d_validity_step_alpha": 1.0,
                "optcuts_k3d_invalid_final_count": 0,
            })
            out.metrics.update(metrics)
            print("[OPTCUTS-K3D-GUARD] candidate_valid=True alpha=1.000000 invalid_final=0")
            return out, report

        # M3D (alpha=0) is valid and the optimizer candidate (alpha=1) is invalid.
        # Binary-search the validity boundary, then retreat a little farther to
        # avoid returning a numerically marginal quad exactly at the crossing.
        displacement = candidate - base_vertices
        low = 0.0   # known valid
        high = 1.0  # known invalid
        for _ in range(20):
            mid = 0.5 * (low + high)
            trial = base_vertices + mid * displacement
            if _invalid_faces(trial, faces):
                high = mid
            else:
                low = mid

        alpha = max(0.0, min(1.0, low * 0.95))
        repaired = base_vertices + alpha * displacement
        final_invalid = _invalid_faces(repaired, faces)
        if final_invalid:
            # Numerical fallback: alpha=0 is guaranteed valid because it was
            # checked above.  Preserve topology rather than dropping tiles.
            alpha = 0.0
            repaired = base_vertices.copy()
            final_invalid = _invalid_faces(repaired, faces)
        if final_invalid:
            raise RuntimeError(
                "OPTCUTS_K3D_VALIDITY_GUARD_FAILED: even the validated M3D fallback became invalid; "
                f"examples={final_invalid[:12]}"
            )

        out.vertices = repaired
        metrics.update({
            "optcuts_k3d_validity_backtracked": True,
            "optcuts_k3d_validity_step_alpha": float(alpha),
            "optcuts_k3d_validity_boundary_alpha": float(low),
            "optcuts_k3d_invalid_final_count": 0,
            "optcuts_k3d_validity_model": (
                "global M3D->K3D displacement backtracking under authoritative validate_top_quad"
            ),
        })
        out.metrics.update(metrics)
        # Keep report useful without assuming its implementation type.
        try:
            report.failed_constraints.append("optcuts_k3d_top_validity_backtrack")
        except Exception:
            pass
        print(
            "[OPTCUTS-K3D-GUARD] "
            f"candidate_valid=False candidate_invalid={len(candidate_invalid)} "
            f"reasons={dict(Counter(reason for _fi, reason in candidate_invalid))} "
            f"alpha={alpha:.6f} boundary_alpha={low:.6f} invalid_final=0"
        )
        return out, report

    pipeline._optimize_k3d = optimize_k3d_with_validity
    original_module = getattr(pipeline, "_original", None)
    if original_module is not None:
        original_module._optimize_k3d = optimize_k3d_with_validity
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original_module, "build_onestring_design", None) if original_module is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize_k3d_with_validity

    pipeline._onestring_optcuts_k3d_validity_patch_installed = True


__all__ = ["install_optcuts_k3d_validity_patch", "_invalid_faces"]
