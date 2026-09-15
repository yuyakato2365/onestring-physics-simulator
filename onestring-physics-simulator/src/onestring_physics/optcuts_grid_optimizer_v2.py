"""Continuation solver for the experimental post-constrained OptCuts UV embedding.

This solver is intentionally fail-fast.  If an intermediate continuation stage
already flips triangles, the completed official OptCuts topology is not being
embedded injectively under the imposed straight-grid targets.  Continuing more
stages cannot be treated as a numerical fix for that structural mismatch.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from . import optcuts_grid_constrained_parameterization_patch as constrained_param

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _strict_optimize_constrained_uv(
    parameterization: Any,
    uv_initial: np.ndarray,
    hard_targets: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for grid-constrained OptCuts reparameterization")

    uv0_np = np.asarray(uv_initial, dtype=float)
    faces_np = np.asarray(parameterization.uv_faces, dtype=int)
    inv_source_np = constrained_param._source_inverse_matrices(parameterization)
    fixed_np = np.all(np.isfinite(hard_targets), axis=1)
    if not np.any(fixed_np):
        return uv0_np.copy(), {
            "optimizer": "none-no-internal-seam",
            "iterations": 0,
            "continuation_stages": 0,
        }

    requested = os.environ.get("ONESTRING_BIJECTIVE_DEVICE", "").strip().lower()
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float32 if device.type in {"mps", "cuda"} else torch.float64

    uv0 = torch.tensor(uv0_np, device=device, dtype=dtype)
    final_targets = torch.tensor(np.nan_to_num(hard_targets, nan=0.0), device=device, dtype=dtype)
    fixed = torch.tensor(fixed_np, device=device, dtype=torch.bool)
    faces = torch.tensor(faces_np, device=device, dtype=torch.long)
    inv_source = torch.tensor(inv_source_np, device=device, dtype=dtype)

    with torch.no_grad():
        tri0 = uv0[faces]
        B0 = torch.stack([tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0]], dim=2)
        J0 = torch.bmm(B0, inv_source)
        det0 = J0[:, 0, 0] * J0[:, 1, 1] - J0[:, 0, 1] * J0[:, 1, 0]
        ref_sign = torch.where(det0 >= 0.0, torch.ones_like(det0), -torch.ones_like(det0))
        if torch.any(torch.abs(det0) <= 1.0e-10):
            bad = torch.nonzero(torch.abs(det0) <= 1.0e-10).flatten()[:16].cpu().tolist()
            raise RuntimeError(f"OPTCUTS_GRID_INITIAL_UV_DEGENERATE: triangles={bad}")

    total_iterations = max(40, int(iterations))
    stages = int(os.environ.get("ONESTRING_OPTCUTS_GRID_CONTINUATION_STAGES", "8"))
    stages = max(2, min(stages, 24))
    per_stage = max(8, int(math.ceil(total_iterations / stages)))
    variable = torch.nn.Parameter(uv0.clone())

    eps_det = 1.0e-5
    anchor_weight = 1.0e-4
    flip_weight = 3.0e5
    best_final_loss = float("inf")
    best_final = None
    stage_losses: list[float] = []
    stage_invalid_counts: list[int] = []
    stage_min_oriented_det: list[float] = []

    for stage in range(1, stages + 1):
        alpha = float(stage) / float(stages)
        stage_targets = uv0.clone()
        stage_targets[fixed] = (1.0 - alpha) * uv0[fixed] + alpha * final_targets[fixed]
        optimizer = torch.optim.Adam([variable], lr=0.010 if stage < stages else 0.006)
        best_stage_loss = float("inf")
        best_stage_uv = None

        for _ in range(per_stage):
            optimizer.zero_grad(set_to_none=True)
            uv = torch.where(fixed[:, None], stage_targets, variable)
            tri = uv[faces]
            B = torch.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], dim=2)
            J = torch.bmm(B, inv_source)
            a, b = J[:, 0, 0], J[:, 0, 1]
            c, d = J[:, 1, 0], J[:, 1, 1]
            det = a * d - b * c
            oriented_det = ref_sign * det
            abs_det = torch.clamp(torch.abs(det), min=eps_det)
            frob = a * a + b * b + c * c + d * d
            sd = frob + frob / (abs_det * abs_det)
            barrier = torch.relu(eps_det - oriented_det)
            free_delta = variable - uv0
            if torch.any(~fixed):
                anchor = torch.mean(free_delta[~fixed] * free_delta[~fixed])
            else:
                anchor = torch.zeros((), device=device, dtype=dtype)
            loss = torch.mean(sd) + flip_weight * torch.mean(barrier * barrier) + anchor_weight * anchor
            loss.backward()
            if variable.grad is not None:
                variable.grad[fixed] = 0.0
            torch.nn.utils.clip_grad_norm_([variable], max_norm=80.0)
            optimizer.step()
            with torch.no_grad():
                variable[fixed] = stage_targets[fixed]
            value = float(loss.detach().cpu())
            if math.isfinite(value) and value < best_stage_loss:
                best_stage_loss = value
                best_stage_uv = torch.where(fixed[:, None], stage_targets, variable).detach().clone()
                if stage == stages and value < best_final_loss:
                    best_final_loss = value
                    best_final = best_stage_uv.cpu().numpy()

        if best_stage_uv is None:
            raise RuntimeError(f"OPTCUTS_GRID_CONSTRAINED_OPTIMIZER_FAILED_AT_STAGE:{stage}")
        with torch.no_grad():
            variable.copy_(best_stage_uv)
            tri = best_stage_uv[faces]
            B = torch.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], dim=2)
            J = torch.bmm(B, inv_source)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            oriented = ref_sign * det
            invalid_count = int(torch.count_nonzero(oriented <= 1.0e-10).cpu())
            min_det = float(torch.min(oriented).cpu())
        stage_losses.append(float(best_stage_loss))
        stage_invalid_counts.append(invalid_count)
        stage_min_oriented_det.append(min_det)
        print(
            "[OPTCUTS-GRID-CONTINUATION] "
            f"stage={stage}/{stages} alpha={alpha:.4f} invalid={invalid_count} "
            f"min_oriented_det={min_det:.6g} best_loss={best_stage_loss:.6g}"
        )

        if invalid_count > 0:
            raise RuntimeError(
                "OPTCUTS_GRID_POSTCONSTRAINT_STRUCTURAL_MISMATCH: "
                f"the completed official OptCuts cut topology already produces {invalid_count} "
                f"flipped/degenerate triangles at continuation stage {stage}/{stages} "
                f"(alpha={alpha:.4f}, min_oriented_det={min_det:.6g}). "
                "Do not tune Adam/continuation to hide this. The straight-grid constraint must "
                "be enforced while cut candidates are selected, before the final OptCuts topology "
                "is committed. Use mode='optcuts' for the official stable result."
            )

    if best_final is None:
        raise RuntimeError("OPTCUTS_GRID_CONSTRAINED_OPTIMIZER_FAILED")
    best_final[fixed_np] = hard_targets[fixed_np]

    final_areas = constrained_param._triangle_signed_areas(best_final, faces_np)
    initial_areas = constrained_param._triangle_signed_areas(uv0_np, faces_np)
    reference = np.where(initial_areas >= 0.0, 1.0, -1.0)
    invalid = np.where(reference * final_areas <= 1.0e-10)[0]
    if len(invalid):
        raise RuntimeError(
            "OPTCUTS_GRID_CONSTRAINT_INFEASIBLE: final straight-grid seam constraints "
            f"produce {len(invalid)} flipped/degenerate triangles; examples={invalid[:16].tolist()}; "
            f"stage_invalid_counts={stage_invalid_counts}; "
            f"stage_min_oriented_det={stage_min_oriented_det}"
        )

    return np.asarray(best_final, dtype=float), {
        "optimizer": "torch_continuation_symmetric_dirichlet_hard_paired_grid_seams",
        "device": str(device),
        "iterations": int(per_stage * stages),
        "continuation_stages": int(stages),
        "stage_best_losses": stage_losses,
        "stage_invalid_counts": stage_invalid_counts,
        "stage_min_oriented_det": stage_min_oriented_det,
        "final_loss": float(best_final_loss),
        "fixed_vertex_count": int(np.count_nonzero(fixed_np)),
        "flip_count": 0,
    }


def _strict_distortion_metrics(parameterization: Any, uv: np.ndarray) -> dict[str, float]:
    inv_source = constrained_param._source_inverse_matrices(parameterization)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    tri = np.asarray(uv, dtype=float)[uf]
    values: list[float] = []
    det_abs: list[float] = []
    for q, inv_s in zip(tri, inv_source):
        B = np.column_stack([q[1] - q[0], q[2] - q[0]])
        J = B @ inv_s
        det = float(np.linalg.det(J))
        det_abs.append(abs(det))
        if abs(det) <= 1e-12:
            values.append(float("inf"))
            continue
        values.append(float(np.sum(J * J) + np.sum(np.linalg.inv(J) ** 2)))
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return {
        "grid_constrained_sd_mean": float(np.mean(finite)) if len(finite) else float("inf"),
        "grid_constrained_sd_max": float(np.max(finite)) if len(finite) else float("inf"),
        "grid_constrained_abs_det_min": float(np.min(det_abs)) if det_abs else 0.0,
    }


def install_strict_optcuts_grid_optimizer() -> None:
    constrained_param._optimize_constrained_uv = _strict_optimize_constrained_uv
    constrained_param._distortion_metrics = _strict_distortion_metrics


__all__ = ["install_strict_optcuts_grid_optimizer"]
