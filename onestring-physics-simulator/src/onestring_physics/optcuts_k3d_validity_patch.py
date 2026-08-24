"""Validity wrapper for OptCuts paths, with strict final checks in optcuts_test."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any
import numpy as np

from .t3d_recovery import validate_top_quad


def _invalid_faces(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[int, str]]:
    verts = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    invalid: list[tuple[int, str]] = []
    for fi, face in enumerate(f):
        try:
            top = verts[np.asarray(face, int)]
        except Exception:
            invalid.append((int(fi), "invalid_face_index"))
            continue
        valid, reason = validate_top_quad(np.asarray(top, float))
        if not valid:
            invalid.append((int(fi), str(reason)))
    return invalid


def _quad_plane_distances(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Robust quad planarity distance using the maximum-area base triangle.

    The old check always used corners (0,1,2). For clipped triangle->quad
    surrogates those can be A-mid(A,B)-B and therefore collinear, making the
    plane undefined and producing false huge distances. For each quad we select
    the largest-area triangle among its four corners and measure the remaining
    corner to that plane.
    """
    verts = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)[:, :4]
    if len(f) == 0:
        return np.zeros(0, dtype=float)
    out = np.full(len(f), np.inf, dtype=float)
    local_ids = (0, 1, 2, 3)
    for fi, face in enumerate(f):
        pts = verts[face]
        best_combo = None
        best_area = -1.0
        for combo in combinations(local_ids, 3):
            a, b, c = pts[list(combo)]
            area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
            if area2 > best_area:
                best_area = area2
                best_combo = combo
        if best_combo is None or best_area <= 1e-12:
            continue
        remaining = next(idx for idx in local_ids if idx not in best_combo)
        p0, p1, p2 = pts[list(best_combo)]
        p3 = pts[remaining]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            continue
        out[fi] = abs(float(np.dot(p3 - p0, normal))) / normal_norm
    return out


def _diagnose_invalid(mesh: Any, invalid: list[tuple[int, str]], limit: int = 12):
    sources = list(getattr(mesh, "_optcuts_test_face_sources", []) or [])
    uv_faces = list(getattr(mesh, "_optcuts_test_face_uv", []) or [])
    rows = []
    for fi, reason in invalid[:limit]:
        row = {"face": int(fi), "reason": str(reason)}
        if 0 <= fi < len(sources):
            row["source"] = str(sources[fi])
        if 0 <= fi < len(uv_faces):
            try:
                row["m2d_uv"] = np.round(np.asarray(uv_faces[fi], float), 6).tolist()
            except Exception:
                pass
        rows.append(row)
    return rows


def _optcuts_active(pipeline: Any, mesh: Any | None = None) -> bool:
    if bool(getattr(pipeline, "_onestring_optcuts_active_run", False)):
        return True
    metrics = dict(getattr(mesh, "metrics", {}) or {}) if mesh is not None else {}
    return bool(
        metrics.get("optcuts_grid_seam_enabled", False)
        or metrics.get("optcuts_grid_seam_applied", False)
        or metrics.get("flattening_backend") == "official_optcuts_external"
        or metrics.get("omega_parameterization_mode") in ("optcuts", "optcuts_test")
        or metrics.get("parameterization_method") in ("optcuts", "optcuts_test")
    )


def _is_test_mode(mesh: Any, parameterization: Any) -> bool:
    if bool(getattr(mesh, "_optcuts_test_boundary_clipped", False)):
        return True
    metrics = dict(getattr(mesh, "metrics", {}) or {})
    if metrics.get("omega_parameterization_mode") == "optcuts_test" or metrics.get("parameterization_method") == "optcuts_test":
        return True
    return str(getattr(parameterization, "method", "")) == "optcuts_test"


def _store_invalid(mesh: Any, invalid: list[tuple[int, str]], stage: str) -> None:
    ids = [int(fi) for fi, _ in invalid]
    reasons = {int(fi): str(reason) for fi, reason in invalid}
    setattr(mesh, "_optcuts_invalid_face_ids", ids)
    setattr(mesh, "_optcuts_invalid_face_reasons", reasons)
    metrics = dict(getattr(mesh, "metrics", {}) or {})
    metrics.update(
        {
            "optcuts_invalid_face_ids": ids,
            "optcuts_invalid_face_reasons": reasons,
            "optcuts_invalid_face_count": len(ids),
            "optcuts_invalid_face_stage": stage,
        }
    )
    try:
        mesh.metrics.update(metrics)
    except Exception:
        pass


def install_optcuts_k3d_validity_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_k3d_validity_patch_installed", False):
        return
    base_optimize = pipeline._optimize_k3d

    try:
        from .optcuts_invalid_visualization_patch import install_optcuts_invalid_visualization_patch
        install_optcuts_invalid_visualization_patch()
    except Exception as exc:
        print(f"[OPTCUTS-INVALID-VIZ] install skipped: {type(exc).__name__}: {exc}")

    def optimize_k3d_with_validity(target, mesh, parameterization, params):
        out, report = base_optimize(target, mesh, parameterization, params)
        if not _optcuts_active(pipeline, mesh) and str(getattr(parameterization, "method", "")) not in ("optcuts", "optcuts_test"):
            return out, report

        test_mode = _is_test_mode(mesh, parameterization)
        base_vertices = np.asarray(mesh.vertices, float)
        faces = np.asarray(out.faces, int)
        candidate = np.asarray(out.vertices, float)
        base_invalid = _invalid_faces(base_vertices, faces)
        candidate_invalid = _invalid_faces(candidate, faces)
        metrics = dict(getattr(out, "metrics", {}) or {})

        al_applied = bool(metrics.get("k3d_augmented_lagrangian_applied", False))
        if test_mode and al_applied:
            distances = _quad_plane_distances(candidate, faces)
            final_max_plane = float(np.max(distances)) if len(distances) else 0.0
            final_rms_plane = float(np.sqrt(np.mean(distances * distances))) if len(distances) else 0.0
            tolerance = float(metrics.get("k3d_hard_planarity_tolerance", 0.0))
            if tolerance <= 0.0:
                scales = []
                for face in faces[:, :4]:
                    tile = candidate[np.asarray(face, dtype=int)]
                    edges = [float(np.linalg.norm(tile[(i + 1) % 4] - tile[i])) for i in range(4)]
                    positive = [e for e in edges if e > 1e-10]
                    scales.append(float(np.median(positive if positive else edges)))
                tile_scale = float(np.median(scales)) if scales else 1.0
                tolerance = max(1e-10, tile_scale * 1e-6)

            metrics.update(
                {
                    "optcuts_k3d_validity_guard_applied": True,
                    "optcuts_k3d_final_assertion_after_augmented_lagrangian": True,
                    "optcuts_k3d_al_result_authoritative": True,
                    "optcuts_k3d_validity_backtracked": False,
                    "optcuts_k3d_invalid_final_count": int(len(candidate_invalid)),
                    "optcuts_k3d_invalid_final_reason_counts": dict(Counter(r for _, r in candidate_invalid)),
                    "k3d_hard_planarity_final_metric": "max-area base triangle to remaining corner distance",
                    "k3d_hard_planarity_max_distance_final_authoritative": final_max_plane,
                    "k3d_hard_planarity_rms_distance_final_authoritative": final_rms_plane,
                    "k3d_hard_planarity_tolerance_final_authoritative": tolerance,
                    "k3d_hard_planarity_constraint_satisfied_final_authoritative": bool(final_max_plane <= tolerance),
                }
            )
            out.metrics.update(metrics)

            if not np.isfinite(final_max_plane) or final_max_plane > tolerance:
                raise RuntimeError(
                    "OPTCUTS_TEST_FINAL_K3D_PLANARITY_FAILED: final authoritative K3D after "
                    "Augmented Lagrangian is not planar within tolerance under the robust max-area "
                    "base-triangle metric; "
                    f"max_plane_distance={final_max_plane:.9g}, tolerance={tolerance:.9g}. "
                    "No M3D backtracking was applied."
                )
            if candidate_invalid:
                _store_invalid(out, candidate_invalid, "K3D_FINAL_AFTER_AL")
                out.metrics.update(
                    {
                        "optcuts_k3d_nonfatal_diagnostic_mode": True,
                        "optcuts_test_invalid_panels_pending_exclusion": True,
                        "optcuts_test_invalid_face_ids": [int(fi) for fi, _ in candidate_invalid],
                        "optcuts_test_invalid_face_count": int(len(candidate_invalid)),
                    }
                )
                try:
                    report.failed_constraints.append("optcuts_test_invalid_panels_excluded_downstream")
                except Exception:
                    pass
                print(
                    "[OPTCUTS-TEST-K3D-INVALID-NONFATAL] "
                    f"count={len(candidate_invalid)} reasons={dict(Counter(r for _, r in candidate_invalid))}; "
                    "keeping them visible in K3D and continuing so K3D->T3D preflight can exclude them"
                )
                return out, report

            _store_invalid(out, [], "K3D_FINAL_AFTER_AL")
            print(
                "[OPTCUTS-TEST-K3D-FINAL] "
                f"planarity_ok=True max_plane_dist={final_max_plane:.6g} tol={tolerance:.6g} "
                "validity_ok=True authoritative_AL=True no_backtracking=True robust_metric=True"
            )
            try:
                report.constraint_violation = final_max_plane
            except Exception:
                pass
            return out, report

        if base_invalid:
            reasons = dict(Counter(r for _, r in base_invalid))
            details = _diagnose_invalid(mesh, base_invalid)
            _store_invalid(mesh, base_invalid, "M3D")
            print(f"[OPTCUTS-M3D-INVALID] count={len(base_invalid)} reasons={reasons} details={details}")
            if not test_mode:
                raise RuntimeError(
                    "OPTCUTS_INVALID_M3D_BEFORE_K3D: lifted M3D already contains invalid quads; "
                    f"reasons={reasons} details={details}"
                )
            final_invalid = candidate_invalid
            _store_invalid(out, final_invalid, "K3D")
            out.metrics.update(
                {
                    "optcuts_k3d_validity_guard_applied": True,
                    "optcuts_k3d_nonfatal_diagnostic_mode": True,
                    "optcuts_m3d_invalid_initial_count": len(base_invalid),
                    "optcuts_k3d_invalid_final_count": len(final_invalid),
                    "optcuts_k3d_invalid_final_reason_counts": dict(Counter(r for _, r in final_invalid)),
                }
            )
            try:
                report.failed_constraints.append("optcuts_test_invalid_panels_nonfatal")
            except Exception:
                pass
            print(
                f"[OPTCUTS-TEST-NONFATAL] M3D_invalid={len(base_invalid)} "
                f"K3D_invalid={len(final_invalid)}; continuing pipeline"
            )
            return out, report

        metrics.update(
            {
                "optcuts_k3d_validity_guard_applied": True,
                "optcuts_k3d_invalid_candidate_count": int(len(candidate_invalid)),
                "optcuts_k3d_invalid_candidate_reason_counts": dict(Counter(r for _, r in candidate_invalid)),
            }
        )
        if not candidate_invalid:
            _store_invalid(mesh, [], "M3D")
            _store_invalid(out, [], "K3D")
            metrics.update(
                {
                    "optcuts_k3d_validity_backtracked": False,
                    "optcuts_k3d_validity_step_alpha": 1.0,
                    "optcuts_k3d_invalid_final_count": 0,
                }
            )
            out.metrics.update(metrics)
            print("[OPTCUTS-K3D-GUARD] candidate_valid=True alpha=1.000000 invalid_final=0")
            return out, report

        displacement = candidate - base_vertices
        low = 0.0
        high = 1.0
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
            alpha = 0.0
            repaired = base_vertices.copy()
            final_invalid = _invalid_faces(repaired, faces)
        if final_invalid:
            if test_mode:
                _store_invalid(out, final_invalid, "K3D")
                out.metrics.update(
                    {
                        "optcuts_k3d_nonfatal_diagnostic_mode": True,
                        "optcuts_k3d_invalid_final_count": len(final_invalid),
                    }
                )
                print(f"[OPTCUTS-TEST-NONFATAL] K3D guard could not repair {len(final_invalid)} faces; continuing")
                return out, report
            raise RuntimeError(f"OPTCUTS_K3D_VALIDITY_GUARD_FAILED: examples={_diagnose_invalid(mesh, final_invalid)}")

        out.vertices = repaired
        _store_invalid(out, [], "K3D")
        metrics.update(
            {
                "optcuts_k3d_validity_backtracked": True,
                "optcuts_k3d_validity_step_alpha": float(alpha),
                "optcuts_k3d_validity_boundary_alpha": float(low),
                "optcuts_k3d_invalid_final_count": 0,
                "optcuts_k3d_validity_model": "global M3D->K3D displacement backtracking under validate_top_quad",
            }
        )
        out.metrics.update(metrics)
        try:
            report.failed_constraints.append("optcuts_k3d_top_validity_backtrack")
        except Exception:
            pass
        print(
            f"[OPTCUTS-K3D-GUARD] candidate_valid=False candidate_invalid={len(candidate_invalid)} "
            f"alpha={alpha:.6f} boundary_alpha={low:.6f} invalid_final=0"
        )
        return out, report

    pipeline._optimize_k3d = optimize_k3d_with_validity
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = optimize_k3d_with_validity
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize_k3d_with_validity
    pipeline._onestring_optcuts_k3d_validity_patch_installed = True


__all__ = ["install_optcuts_k3d_validity_patch", "_invalid_faces"]
