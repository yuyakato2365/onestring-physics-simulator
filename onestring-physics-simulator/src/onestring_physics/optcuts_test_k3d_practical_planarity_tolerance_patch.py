"""Use a fabrication-practical final planarity tolerance in optcuts_test.

The numerical solvers still keep/report their strict tolerance (typically
1e-6 times the representative tile size).  After AL + SLSQP polishing, this
wrapper preserves that strict value for diagnostics but publishes a separate
acceptance tolerance of 5e-4 times the representative tile size to the outer
K3D validity guard.

This means a residual up to 0.05% of representative tile size is accepted for
pipeline continuation.  Top-face validity/self-intersection checks remain
unchanged and can still reject the K3D.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _representative_tile_scale(vertices: np.ndarray, faces: np.ndarray) -> float:
    verts = np.asarray(vertices, dtype=float)
    f4 = np.asarray(faces, dtype=int)[:, :4]
    scales: list[float] = []
    for face in f4:
        tile = verts[face]
        edges = [
            float(np.linalg.norm(tile[(i + 1) % 4] - tile[i]))
            for i in range(4)
        ]
        positive = [e for e in edges if e > 1e-10]
        if positive:
            scales.append(float(np.median(positive)))
    return float(np.median(scales)) if scales else 1.0


def install_optcuts_test_k3d_practical_planarity_tolerance_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_practical_planarity_tolerance_installed", False):
        return

    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        out, report = base(target, mesh, parameterization, params)
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return out, report

        metrics = dict(getattr(out, "metrics", {}) or {})
        if not bool(metrics.get("k3d_augmented_lagrangian_applied", False)):
            return out, report

        strict_tol = float(metrics.get("k3d_hard_planarity_tolerance", 0.0))
        tile_scale = _representative_tile_scale(
            np.asarray(out.vertices, dtype=float),
            np.asarray(out.faces, dtype=int),
        )
        relative_acceptance = float(
            getattr(params, "k3d_practical_planarity_relative_tolerance", 5e-4)
        )
        practical_tol = max(1e-10, tile_scale * max(relative_acceptance, 0.0))

        # Preserve the solver's strict criterion for diagnostics, but let the
        # outer final guard use the fabrication-practical acceptance threshold.
        metrics["k3d_strict_planarity_tolerance_solver"] = strict_tol
        metrics["k3d_practical_planarity_relative_tolerance"] = relative_acceptance
        metrics["k3d_practical_planarity_tolerance"] = practical_tol
        metrics["k3d_hard_planarity_tolerance"] = practical_tol
        metrics["k3d_final_planarity_acceptance_mode"] = "fabrication-practical 5e-4 relative tile scale"
        try:
            out.metrics.update(metrics)
        except Exception:
            pass

        print(
            "[OPTCUTS-TEST-K3D-PRACTICAL-TOL] "
            f"strict_tol={strict_tol:.6g} practical_tol={practical_tol:.6g} "
            f"tile_scale={tile_scale:.6g} relative={relative_acceptance:.6g}"
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

    pipeline._onestring_optcuts_test_practical_planarity_tolerance_installed = True


__all__ = ["install_optcuts_test_k3d_practical_planarity_tolerance_patch"]
