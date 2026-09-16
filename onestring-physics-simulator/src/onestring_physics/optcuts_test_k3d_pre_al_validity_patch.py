"""Repair ordinary optcuts_test K3D validity before hard-planarity AL.

This wrapper runs *inside* the Augmented-Lagrangian wrapper.  It may backtrack
an ordinary K3D candidate toward M3D to obtain a valid reference, but after AL
runs no later stage is allowed to rewrite the authoritative planar K3D.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .optcuts_k3d_validity_patch import _invalid_faces


def install_optcuts_test_k3d_pre_al_validity_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_k3d_pre_al_validity_installed", False):
        return

    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        out, report = base(target, mesh, parameterization, params)
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return out, report

        faces = np.asarray(out.faces, dtype=int)
        m3d = np.asarray(mesh.vertices, dtype=float)
        candidate = np.asarray(out.vertices, dtype=float)
        invalid = _invalid_faces(candidate, faces)
        if not invalid:
            try:
                out.metrics.update({
                    "k3d_pre_al_validity_repair_applied": True,
                    "k3d_pre_al_validity_backtracked": False,
                    "k3d_pre_al_validity_invalid_before": 0,
                })
            except Exception:
                pass
            return out, report

        # Find the largest step from M3D toward the ordinary K3D candidate that
        # keeps every top quad valid.  This repaired state becomes AL's reference.
        if _invalid_faces(m3d, faces):
            # M3D itself is not a valid endpoint, so a global segment backtrack
            # cannot guarantee a repair. Leave the candidate untouched; AL and
            # the final strict assertion will decide whether a valid planar state
            # exists rather than silently pretending alpha=0 is safe.
            try:
                out.metrics.update({
                    "k3d_pre_al_validity_repair_applied": True,
                    "k3d_pre_al_validity_backtracked": False,
                    "k3d_pre_al_validity_invalid_before": int(len(invalid)),
                    "k3d_pre_al_validity_m3d_endpoint_invalid": True,
                })
            except Exception:
                pass
            print(
                "[OPTCUTS-TEST-K3D-PRE-AL] ordinary candidate invalid but M3D endpoint is also invalid; "
                "skip backtrack and let AL/final assertion handle it"
            )
            return out, report

        displacement = candidate - m3d
        low, high = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (low + high)
            trial = m3d + mid * displacement
            if _invalid_faces(trial, faces):
                high = mid
            else:
                low = mid

        alpha = max(0.0, min(1.0, low * 0.98))
        repaired = m3d + alpha * displacement
        repaired_invalid = _invalid_faces(repaired, faces)
        if repaired_invalid:
            raise RuntimeError(
                "OPTCUTS_TEST_PRE_AL_VALIDITY_REPAIR_FAILED: could not construct a valid K3D reference "
                f"before Augmented Lagrangian; invalid_count={len(repaired_invalid)}"
            )

        out.vertices[:] = repaired
        try:
            out.metrics.update({
                "k3d_pre_al_validity_repair_applied": True,
                "k3d_pre_al_validity_backtracked": True,
                "k3d_pre_al_validity_invalid_before": int(len(invalid)),
                "k3d_pre_al_validity_invalid_after": 0,
                "k3d_pre_al_validity_step_alpha": float(alpha),
                "k3d_pre_al_validity_reference_is_authoritative_for_al": True,
            })
        except Exception:
            pass
        print(
            "[OPTCUTS-TEST-K3D-PRE-AL] "
            f"invalid_before={len(invalid)} alpha={alpha:.6f} invalid_after=0"
        )
        return out, report

    pipeline._optimize_k3d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._optimize_k3d = optimize
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_optimize_k3d"] = optimize

    pipeline._onestring_optcuts_test_k3d_pre_al_validity_installed = True


__all__ = ["install_optcuts_test_k3d_pre_al_validity_patch"]
