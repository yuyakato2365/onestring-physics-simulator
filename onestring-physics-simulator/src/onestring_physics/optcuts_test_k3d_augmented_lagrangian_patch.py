"""OptCuts-test K3D hard planarity with a shape-preserving square-aware AL mode.

The historical mode is preserved: it keeps the ordinary K3D result close by a
vertex-position anchor while enforcing one coplanarity equality per quad.

The 2026-09-15 version now combines three requirements:
- hard quad coplanarity;
- preservation of the ordinary K3D reference metric (four edges + two diagonals);
- a global square-quality term applied to every quad, so malformed / triangle-like
  panels are actively regularized instead of merely being preserved as reference.

Selecting this version also enables the existing hard non-penetration stage so
K2D/T2D panel overlap is not silently accepted in the integrated latest version.
"""
from __future__ import annotations

from itertools import combinations
import os
from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


SHAPE_PRESERVING_VERSION_ID = "2026-09-15-shape-preserving-hard-planarity"
SHAPE_PRESERVING_VERSION_LABEL = "2026-09-15 — Shape-preserving hard planarity"
SHAPE_PRESERVING_VERSION_DESCRIPTION = (
    "hard-planarityを維持しつつordinary K3Dからの変形を抑え、さらに全quadへ"
    "global square regularizationを適用する統合版。K2D/T2D hard non-penetrationも有効化する。"
)


def _shape_preserving_enabled() -> bool:
    return os.environ.get("ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY", "0").lower() in {
        "1", "true", "yes", "on"
    }


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _install_shape_preserving_version_ui() -> None:
    """Expose the integrated 09-15 algorithm as a separate Version entry."""
    try:
        import streamlit as st
    except Exception:
        return
    if getattr(st, "_onestring_shape_preserving_planarity_version_installed", False):
        return

    original_selectbox = st.selectbox

    def selectbox_with_shape_preserving_version(*args: Any, **kwargs: Any) -> Any:
        label = args[0] if args else kwargs.get("label")
        if label == "version":
            if len(args) >= 2:
                options = list(args[1])
                if not any(isinstance(v, dict) and v.get("id") == SHAPE_PRESERVING_VERSION_ID for v in options):
                    template = next(
                        (
                            dict(v)
                            for v in reversed(options)
                            if isinstance(v, dict)
                            and v.get("id") == "2026-09-14-paper-eq5-unified-k2d"
                        ),
                        dict(options[-1]) if options and isinstance(options[-1], dict) else {},
                    )
                    template.update(
                        {
                            "id": SHAPE_PRESERVING_VERSION_ID,
                            "label": SHAPE_PRESERVING_VERSION_LABEL,
                            "description": SHAPE_PRESERVING_VERSION_DESCRIPTION,
                        }
                    )
                    options.append(template)
                args = (args[0], options, *args[2:])
            elif "options" in kwargs:
                options = list(kwargs["options"])
                if not any(isinstance(v, dict) and v.get("id") == SHAPE_PRESERVING_VERSION_ID for v in options):
                    template = next(
                        (
                            dict(v)
                            for v in reversed(options)
                            if isinstance(v, dict)
                            and v.get("id") == "2026-09-14-paper-eq5-unified-k2d"
                        ),
                        dict(options[-1]) if options and isinstance(options[-1], dict) else {},
                    )
                    template.update(
                        {
                            "id": SHAPE_PRESERVING_VERSION_ID,
                            "label": SHAPE_PRESERVING_VERSION_LABEL,
                            "description": SHAPE_PRESERVING_VERSION_DESCRIPTION,
                        }
                    )
                    options.append(template)
                kwargs = {**kwargs, "options": options}

        selected = original_selectbox(*args, **kwargs)
        if label == "version":
            enabled = isinstance(selected, dict) and selected.get("id") == SHAPE_PRESERVING_VERSION_ID
            os.environ["ONESTRING_K3D_SHAPE_PRESERVING_PLANARITY"] = "1" if enabled else "0"
            # The 09-15 version is an integrated latest version.  The inner
            # 09-14 selector wrapper may clear this flag because the IDs differ,
            # so assert it here after the actual selected option is known.
            if enabled:
                os.environ["ONESTRING_HARD_NONPENETRATION"] = "1"
                shape_weight = st.number_input(
                    "K3D shape-preservation weight",
                    min_value=0.0,
                    max_value=10000.0,
                    value=max(0.0, min(10000.0, _safe_float_env("ONESTRING_K3D_SHAPE_PRESERVATION_WEIGHT", 100.0))),
                    step=10.0,
                    format="%.1f",
                    key="onestring_k3d_shape_preservation_weight",
                    help=(
                        "Preserves the ordinary K3D local metric. The ordinary UI w_square / ESquare is now "
                        "also injected directly into this hard-planarity solve for every quad."
                    ),
                )
                os.environ["ONESTRING_K3D_SHAPE_PRESERVATION_WEIGHT"] = str(float(shape_weight))
                st.caption(
                    "09-15 integrated: hard coplanarity + reference preservation + global square regularization "
                    "for every quad + hard K2D/T2D non-penetration. w_square / ESquare now acts inside this AL stage too."
                )
        return selected

    st.selectbox = selectbox_with_shape_preserving_version
    st._onestring_shape_preserving_planarity_version_installed = True


_install_shape_preserving_version_ui()


def _robust_constraint_faces(reference: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=float)
    f4 = np.asarray(faces, dtype=int)[:, :4]
    reordered = np.empty_like(f4)
    areas = np.zeros(len(f4), dtype=float)
    local_ids = (0, 1, 2, 3)
    for fi, face in enumerate(f4):
        pts = ref[face]
        best_combo = (0, 1, 2)
        best_area = -1.0
        for combo in combinations(local_ids, 3):
            a, b, c = pts[list(combo)]
            area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
            if area2 > best_area:
                best_area = area2
                best_combo = combo
        remaining = next(idx for idx in local_ids if idx not in best_combo)
        reordered[fi] = np.asarray(
            [face[best_combo[0]], face[best_combo[1]], face[best_combo[2]], face[remaining]],
            dtype=int,
        )
        areas[fi] = max(best_area, 0.0)
    return reordered, areas


def _quad_constraint_values_and_gradient(
    vertices: np.ndarray,
    constraint_faces: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(constraint_faces, dtype=int)[:, :4]
    q = v[f]
    x0, x1, x2, x3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a = x1 - x0
    b = x2 - x0
    c = x3 - x0
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    ab = np.cross(a, b)
    triple = np.einsum("ij,ij->i", a, bc)
    denom = np.maximum(np.asarray(scales, dtype=float) ** 3, 1e-18)
    values = triple / denom
    grad = np.zeros((len(f), 4, 3), dtype=float)
    grad[:, 1] = bc / denom[:, None]
    grad[:, 2] = ca / denom[:, None]
    grad[:, 3] = ab / denom[:, None]
    grad[:, 0] = -(grad[:, 1] + grad[:, 2] + grad[:, 3])
    return values, grad


def _quad_plane_distances(vertices: np.ndarray, constraint_faces: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(constraint_faces, dtype=int)[:, :4]
    q = v[f]
    a = q[:, 1] - q[:, 0]
    b = q[:, 2] - q[:, 0]
    c = q[:, 3] - q[:, 0]
    n = np.cross(a, b)
    n_norm = np.linalg.norm(n, axis=1)
    triple = np.abs(np.einsum("ij,ij->i", c, n))
    out = np.full(len(f), np.inf, dtype=float)
    good = n_norm > 1e-12
    out[good] = triple[good] / n_norm[good]
    return out


def _quad_triangle_area_diagnostics(
    reference: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
    *, min_reference_area_ratio: float = 1e-7,
) -> dict:
    """Track actual source-referenced triangle collapse, not planarity only.

    The classic plane-distance criterion is +inf when its normal has
    zero magnitude: a degenerate *planar* quad is not valid K3D.
    Report both unsigned area-ratio collapse and oriented inversion.
    """
    ref=np.asarray(reference,float)
    v=np.asarray(vertices,float)
    f=np.asarray(faces,int)[:,:4]
    tri=np.concatenate([f[:,[0,1,2]],f[:,[0,2,3]]],axis=0)
    a=ref[tri]
    base=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
    area=np.linalg.norm(base,axis=1)
    scale=_face_scales(ref,f)
    reference_scale=np.concatenate([scale,scale])
    considered=area>np.maximum(
        min_reference_area_ratio*reference_scale**2,1e-16
    )
    q=v[tri]
    cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    unsigned=np.linalg.norm(cross,axis=1)/np.maximum(area,1e-20)
    orientation=np.einsum("ij,ij->i",cross,base)/np.maximum(area**2,1e-40)
    signed=np.where(considered,orientation,np.inf)
    magnitudes=np.where(considered,unsigned,np.inf)
    invalid=(considered & (
        ~np.isfinite(unsigned) | (unsigned<0.02) |
        ~np.isfinite(orientation) | (orientation<0.01)
    ))
    return {
        "valid_source_triangles":int(np.sum(considered)),
        "min_unsigned_area_ratio":float(np.min(magnitudes,initial=np.inf)),
        "min_oriented_area_ratio":float(np.min(signed,initial=np.inf)),
        "collapsed_or_inverted_triangle_count":int(np.sum(invalid)),
        "collapsed_or_inverted_face_ids":sorted(
            set((np.flatnonzero(invalid)%len(f)).tolist())
        )[:24],
    }


def _quad_area_barrier_energy_gradient(
    vertices: np.ndarray, reference: np.ndarray, faces: np.ndarray,
    scales: np.ndarray, *, weight: float, minimum_ratio: float = 0.04,
) -> tuple[float,np.ndarray]:
    """Differentiable penalty against collapse/inversion of quad triangles.

    Signed projected double-areas are evaluated relative to each
    reference M3D triangle normal. This penalty is an ADDITION to
    SmoothMax and not a substitute for a final geometric validity test.
    """
    x=np.asarray(vertices,float)
    base=np.asarray(reference,float)
    f=np.asarray(faces,int)[:,:4]
    triangles=np.concatenate((f[:,[0,1,2]],f[:,[0,2,3]]),axis=0)
    b=base[triangles]
    n=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    nn=np.linalg.norm(n,axis=1)
    unit=n/np.maximum(nn[:,None],1e-20)
    unit_scale=np.concatenate((np.asarray(scales,float),)*2)
    valid=nn>np.maximum(unit_scale**2*1e-7,1e-16)
    q=x[triangles]
    a=q[:,1]-q[:,0]
    c=q[:,2]-q[:,0]
    oriented=np.einsum("ij,ij->i",np.cross(a,c),unit)
    safe_area=np.maximum(nn,1e-20)
    deficit=np.where(valid,np.minimum(oriented/safe_area-minimum_ratio,0.),0.)
    ntri=max(int(np.sum(valid)),1)
    energy=.5*float(weight)*float(np.dot(deficit,deficit))/ntri
    coeff=float(weight)*deficit/(ntri*safe_area)
    db=np.cross(c,unit)
    dc=np.cross(unit,a)
    grad=np.zeros_like(x)
    delta_b=coeff[:,None]*db
    delta_c=coeff[:,None]*dc
    np.add.at(grad,triangles[:,0],-delta_b-delta_c)
    np.add.at(grad,triangles[:,1],delta_b)
    np.add.at(grad,triangles[:,2],delta_c)
    return energy,grad


def _face_scales(reference: np.ndarray, faces: np.ndarray) -> np.ndarray:
    q = np.asarray(reference, dtype=float)[np.asarray(faces, dtype=int)[:, :4]]
    vals: list[float] = []
    for tile in q:
        lengths = [float(np.linalg.norm(tile[(i + 1) % 4] - tile[i])) for i in range(4)]
        positive = [length for length in lengths if length > 1e-10]
        vals.append(max(float(np.median(positive if positive else lengths)), 1e-8))
    return np.asarray(vals, dtype=float)


_METRIC_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))
_EDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))


def _metric_reference(reference: np.ndarray, faces: np.ndarray, scales: np.ndarray) -> np.ndarray:
    q = np.asarray(reference, dtype=float)[np.asarray(faces, dtype=int)[:, :4]]
    values = []
    for a, b in _METRIC_PAIRS:
        values.append(np.linalg.norm(q[:, a] - q[:, b], axis=1))
    return np.stack(values, axis=1) if len(q) else np.zeros((0, len(_METRIC_PAIRS)), dtype=float)


def _shape_metric_energy_and_gradient(
    vertices: np.ndarray,
    faces: np.ndarray,
    scales: np.ndarray,
    reference_lengths: np.ndarray,
    weight: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Normalized 4-edge + 2-diagonal preservation energy and analytic gradient."""
    x = np.asarray(vertices, dtype=float)
    f4 = np.asarray(faces, dtype=int)[:, :4]
    grad = np.zeros_like(x)
    if len(f4) == 0 or weight <= 0.0:
        return 0.0, grad, np.zeros((len(f4), len(_METRIC_PAIRS)), dtype=float)
    q = x[f4]
    scale = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    residuals = np.zeros((len(f4), len(_METRIC_PAIRS)), dtype=float)
    norm_count = float(max(1, len(f4) * len(_METRIC_PAIRS)))
    energy_sum = 0.0
    for pair_id, (a_local, b_local) in enumerate(_METRIC_PAIRS):
        a_ids = f4[:, a_local]
        b_ids = f4[:, b_local]
        delta = q[:, a_local] - q[:, b_local]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1e-12)
        residual = (length - reference_lengths[:, pair_id]) / scale
        residuals[:, pair_id] = residual
        energy_sum += float(np.dot(residual, residual))
        coeff = (float(weight) / norm_count) * residual / (scale * safe)
        contribution = coeff[:, None] * delta
        np.add.at(grad, a_ids, contribution)
        np.add.at(grad, b_ids, -contribution)
    return 0.5 * float(weight) * energy_sum / norm_count, grad, residuals


def _square_energy_and_gradient(
    vertices: np.ndarray,
    faces: np.ndarray,
    scales: np.ndarray,
    weight: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Per-panel square residuals with a smooth worst-panel penalty.

    E = w * (mean(d_i**2) + alpha * LSE_beta(d)**2) / 2,
    d_i = sqrt((sum of four edge-equality and diagonal-equality residuals**2)/5).
    All residuals are dimensionless (normalized by each panel's scale).
    """
    x = np.asarray(vertices, dtype=float)
    f4 = np.asarray(faces, dtype=int)[:, :4]
    grad = np.zeros_like(x)
    if len(f4) == 0 or weight <= 0.0:
        return 0.0, grad, {"rms": 0.0, "max": 0.0, "worst_edge_ratio": 1.0}
    q = x[f4]
    scale = np.maximum(np.asarray(scales, dtype=float), 1e-12)
    edges = [q[:, a] - q[:, b] for a, b in _EDGE_PAIRS]
    lengths = np.stack([np.linalg.norm(d, axis=1) for d in edges], axis=1)
    residuals = [(lengths[:, k] - lengths[:, (k + 1) % 4]) / scale for k in range(4)]
    diagonals = [q[:, 0] - q[:, 2], q[:, 1] - q[:, 3]]
    diag_lengths = [np.linalg.norm(d, axis=1) for d in diagonals]
    residuals.append((diag_lengths[0] - diag_lengths[1]) / scale)
    residual = np.stack(residuals, axis=1)
    d = np.sqrt(np.mean(residual * residual, axis=1) + 1e-24)
    alpha = max(0.0, _safe_float_env("ONESTRING_K3D_SQUARE_TAIL_ALPHA", 4.0))
    beta = max(0.0, _safe_float_env("ONESTRING_K3D_SQUARE_TAIL_BETA", 12.0))
    if beta == 0.0:
        # Exact continuous extension of log(mean(exp(beta*d)))/beta:
        # L_0=mean(d); derivative w.r.t. each d_i is 1/N.
        lse = float(np.mean(d))
        softmax = np.full(len(d), 1.0 / len(d))
    else:
        z = beta * d
        zmax = float(np.max(z))
        expz = np.exp(z - zmax)
        softmax = expz / np.sum(expz)
        lse = (zmax + np.log(np.mean(expz))) / beta
    energy = 0.5 * float(weight) * (float(np.mean(d * d)) + alpha * lse * lse)
    # dE / d(residual_ik), with five residuals per panel.
    coeffs = float(weight) * (1.0 / len(f4) + alpha * lse * softmax / d)[:, None] * residual / 5.0
    for k in range(4):
        for edge_id, sign in ((k, 1.0), ((k + 1) % 4, -1.0)):
            ia, ib = _EDGE_PAIRS[edge_id]
            delta = edges[edge_id]
            factor = sign * coeffs[:, k] / (scale * np.maximum(lengths[:, edge_id], 1e-12))
            g = factor[:, None] * delta
            np.add.at(grad, f4[:, ia], g)
            np.add.at(grad, f4[:, ib], -g)
    for j, (ia, ib) in enumerate(((0, 2), (1, 3))):
        factor = (1.0 if j == 0 else -1.0) * coeffs[:, 4] / (scale * np.maximum(diag_lengths[j], 1e-12))
        g = factor[:, None] * diagonals[j]
        np.add.at(grad, f4[:, ia], g)
        np.add.at(grad, f4[:, ib], -g)
    ratios = np.min(lengths, axis=1) / np.maximum(np.max(lengths, axis=1), 1e-12)
    return energy, grad, {
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "max": float(np.max(np.abs(residual))),
        "worst_edge_ratio": float(np.min(ratios)),
        "worst_panel_deviation": float(np.max(d)),
        "square_tail_lse": float(lse),
        "square_tail_alpha": float(alpha),
        "square_tail_beta": float(beta),
    }

def _consensus_planarity_restore(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sweeps: int,
    relaxation: float = 1.0,
    tolerance: float | None = None,
    constraint_faces: np.ndarray | None = None,
    equality_inverse: np.ndarray | None = None,
    affine_projector=None,
) -> tuple[np.ndarray, int, float]:
    """Nearest-plane consensus feasibility restoration."""
    x = np.asarray(vertices, dtype=float).copy()
    f4 = np.asarray(faces, dtype=int)[:, :4]
    if len(x) == 0 or len(f4) == 0:
        return x, 0, 0.0
    alpha = float(np.clip(relaxation, 0.05, 1.0))
    counts = np.zeros(len(x), dtype=float)
    np.add.at(counts, f4.reshape(-1), 1.0)
    active = counts > 0.0
    used = 0
    last_max = float("inf")
    for sweep in range(max(0, int(sweeps))):
        q = x[f4]
        centers = np.mean(q, axis=1)
        centered = q - centers[:, None, :]
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normals = vh[:, -1, :]
        except np.linalg.LinAlgError:
            normals = np.empty((len(f4), 3), dtype=float)
            for fi, tile in enumerate(centered):
                _, _, vh_i = np.linalg.svd(tile, full_matrices=False)
                normals[fi] = vh_i[-1]
        n_norm = np.linalg.norm(normals, axis=1)
        good = n_norm > 1e-15
        normals[good] /= n_norm[good, None]
        normals[~good] = np.array([0.0, 0.0, 1.0])
        signed = np.einsum("fki,fi->fk", centered, normals)
        projected = q - signed[:, :, None] * normals[:, None, :]
        accum = np.zeros_like(x)
        for corner in range(4):
            np.add.at(accum, f4[:, corner], projected[:, corner])
        consensus = x.copy()
        consensus[active] = accum[active] / counts[active, None]
        x[active] = (1.0 - alpha) * x[active] + alpha * consensus[active]
        if equality_inverse is not None:
            # AL's subsequent planarity-consensus step must not reopen
            # hard seam equality. Project by the SAME variable quotient.
            labels=np.asarray(equality_inverse,int)
            n=int(labels.max())+1
            total=np.zeros((n,3),float)
            np.add.at(total,labels,x)
            count=np.bincount(labels,minlength=n)
            x=total[labels]/count[labels,None]
        if affine_projector is not None:
            x=affine_projector.project(x,strict=True)
        used = sweep + 1
        if tolerance is not None and constraint_faces is not None:
            distances = _quad_plane_distances(x, constraint_faces)
            last_max = float(np.max(distances)) if len(distances) else 0.0
            if last_max <= float(tolerance):
                break
    if constraint_faces is not None:
        distances = _quad_plane_distances(x, constraint_faces)
        last_max = float(np.max(distances)) if len(distances) else 0.0
    return x, used, float(last_max)


def _augmented_lagrangian_planarize(
    reference: np.ndarray,
    faces: np.ndarray,
    *,
    anchor_weight: float = 1.0,
    square_weight: float = 10.0,
    rho0: float = 10.0,
    outer_iterations: int = 16,
    inner_iterations: int = 70,
    relative_plane_tolerance: float = 1e-6,
    restoration_sweeps: int = 8,
    final_polish_sweeps: int = 400,
    equality_groups: tuple | list = (),
    affine_rows: tuple | list = (),
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    ref = np.asarray(reference, dtype=float)
    f = np.asarray(faces, dtype=int)
    if len(ref) == 0 or len(f) == 0:
        return ref.copy(), {"k3d_augmented_lagrangian_applied": False}
    if minimize is None:
        return ref.copy(), {
            "k3d_augmented_lagrangian_applied": False,
            "k3d_augmented_lagrangian_reason": "scipy.optimize.minimize unavailable",
        }

    f4 = f[:, :4]
    constraint_faces, base_areas = _robust_constraint_faces(ref, f4)
    scales = _face_scales(ref, f4)
    tile_scale = float(np.median(scales)) if len(scales) else 1.0
    plane_tol = max(1e-10, tile_scale * float(relative_plane_tolerance))
    degenerate_base_count = int(np.sum(base_areas <= np.maximum(scales * scales * 1e-10, 1e-16)))
    if degenerate_base_count:
        raise RuntimeError(
            "OPTCUTS_TEST_K3D_PLANARITY_CONSTRAINT_DEGENERATE: "
            f"{degenerate_base_count} panels have no non-collinear triple among their four corners."
        )

    shape_mode = _shape_preserving_enabled()
    shape_weight = max(0.0, _safe_float_env("ONESTRING_K3D_SHAPE_PRESERVATION_WEIGHT", 100.0)) if shape_mode else 0.0
    # Seam welding plus a relative-only square penalty has a collapsed
    # quad as a false zero-energy minimum. In seam-affine AL, preserve
    # 3D source edge/diagonal dimensions and oriented triangle areas
    # even when the optional shape-preserving UI version is OFF.
    seam_geometry_guard=bool(affine_rows or equality_groups)
    if seam_geometry_guard:
        shape_weight=max(
            shape_weight,
            max(8.,3.*float(max(square_weight,0.)))
        )
    area_guard_weight=(
        max(50.,10.*float(max(square_weight,0.)))
        if seam_geometry_guard else 0.
    )
    # The global smooth-maximum square penalty is part of EVERY active
    # optcuts_test hard-planarity solve. Shape-preserving reference metrics
    # remain a separate optional term; they must not gate E_Square.
    effective_square_weight = max(0.0, float(square_weight))
    square_tail_alpha=max(0.,_safe_float_env("ONESTRING_K3D_SQUARE_TAIL_ALPHA",4.))
    square_tail_beta=max(0.,_safe_float_env("ONESTRING_K3D_SQUARE_TAIL_BETA",12.))
    print(
        f"[K3D-SMOOTHMAX] stage=AL enabled="
        f"{bool(effective_square_weight>0 and square_tail_alpha>0)} "
        f"w_square={effective_square_weight:.6g} "
        f"alpha={square_tail_alpha:.6g} beta={square_tail_beta:.6g} "
        f"shape_preserving={shape_mode}",flush=True,
    )
    reference_lengths = _metric_reference(ref, f4, scales)

    # HARD source-Seam equalities: quotient optimizer unknowns by proven
    # pairs, rather than optimizing independently then approximating with
    # a post-hoc positional penalty or averaging.
    parent=np.arange(len(ref),dtype=int)
    def root(k):
        k=int(k)
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=int(parent[k])
        return k
    for group in equality_groups or ():
        ids=list(map(int,group))
        if len(ids)<2:
            continue
        if any(i<0 or i>=len(ref) for i in ids):
            raise ValueError("K3D_SEAM_AL_GROUP_VERTEX_OUT_OF_RANGE")
        representative=root(ids[0])
        for v in ids[1:]:
            parent[root(v)]=representative
    roots=np.asarray([root(i) for i in range(len(ref))],int)
    _,equality_inverse=np.unique(roots,return_inverse=True)
    unique_count=int(equality_inverse.max())+1
    has_equalities=unique_count<len(ref)
    # A vertex-to-EDGE or edge-to-edge constraint cannot be expressed by
    # integer-vertex quotienting. A constant orthogonal affine nullspace
    # projector handles arbitrary seam edge subdivisions, including
    # seam-to-open transitions MID Grid edge.
    affine_projector=None
    if affine_rows:
        from .optcuts_seam_affine_constraints import (
            AffineSeamProjector,
        )
        combined=list(affine_rows)+[
            [(int(g[0]),1.),(int(v),-1.)]
            for g in (equality_groups or ()) if len(g)>1
            for v in g[1:]
        ]
        affine_projector=AffineSeamProjector(len(ref),combined)
        has_equalities=False
        print(
            "[K3D-SEAM-AFFINE] enabled=True "
            f"physical_constraints={len(affine_rows)} "
            f"equivalence_groups={len(equality_groups or ())} "
            f"constraint_rows={affine_projector.count}",
            flush=True,
        )
    counts=np.bincount(equality_inverse,minlength=unique_count)
    def flatten_equal(verts):
        total=np.zeros((unique_count,3),float)
        np.add.at(total,equality_inverse,np.asarray(verts,float))
        return total/counts[:,None]
    def expand_equal(flat):
        return np.asarray(flat,float).reshape(unique_count,3)[equality_inverse]
    def reduce_gradient(grad):
        accumulated=np.zeros((unique_count,3),float)
        np.add.at(accumulated,equality_inverse,np.asarray(grad,float))
        return accumulated.ravel()
    # Initial feasible pair positions are identical, bitwise, and stay
    # identical throughout ALL AL and consensus steps.
    x=expand_equal(flatten_equal(ref)) if has_equalities else ref.copy()
    if affine_projector is not None:
        x=affine_projector.project(ref,strict=True)
    if has_equalities:
        print(
            "[K3D-SEAM-AL] hard_shared_unknowns=True "
            f"groups={len(equality_groups)} "
            f"original_vertices={len(ref)} free_vertices={unique_count}",
            flush=True,
        )
    lambdas = np.zeros(len(f4), dtype=float)
    rho = max(float(rho0), 1e-6)
    w_anchor = max(float(anchor_weight), 1e-10)
    before_dist = _quad_plane_distances(x, constraint_faces)
    max_before = float(np.max(before_dist)) if len(before_dist) else 0.0
    rms_before = float(np.sqrt(np.mean(before_dist * before_dist))) if len(before_dist) else 0.0
    previous_violation = float("inf")
    inner_total = 0
    restoration_total = 0
    converged = False
    outer_done = 0
    last_inner_success = True
    last_inner_message = ""

    def objective_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
        verts = (
            affine_projector.project(np.asarray(flat,float).reshape(ref.shape))
            if affine_projector is not None
            else expand_equal(flat) if has_equalities
            else np.asarray(flat,dtype=float).reshape(ref.shape)
        )
        delta = verts - ref
        energy = 0.5 * w_anchor * float(np.sum(delta * delta))
        grad_total = w_anchor * delta
        if shape_weight > 0.0:
            shape_energy, shape_grad, _ = _shape_metric_energy_and_gradient(
                verts, f4, scales, reference_lengths, shape_weight
            )
            energy += shape_energy
            grad_total += shape_grad
        if effective_square_weight > 0.0:
            square_energy, square_grad, _ = _square_energy_and_gradient(
                verts, f4, scales, effective_square_weight
            )
            energy += square_energy
            grad_total += square_grad
        if area_guard_weight>0.:
            area_energy,area_grad=_quad_area_barrier_energy_gradient(
                verts,ref,f4,scales,weight=area_guard_weight
            )
            energy+=area_energy
            grad_total+=area_grad
        constraints, local_grad = _quad_constraint_values_and_gradient(verts, constraint_faces, scales)
        coeff = lambdas + rho * constraints
        energy += float(np.dot(lambdas, constraints) + 0.5 * rho * np.dot(constraints, constraints))
        contribution = coeff[:, None, None] * local_grad
        for corner in range(4):
            np.add.at(grad_total, constraint_faces[:, corner], contribution[:, corner])
        if affine_projector is not None:
            return energy,affine_projector.project(grad_total).ravel()
        return energy,reduce_gradient(grad_total) if has_equalities else grad_total.ravel()

    restore_relaxation = 0.35 if shape_mode else 1.0

    for outer in range(max(1, int(outer_iterations))):
        outer_done = outer + 1
        result = minimize(
            fun=lambda z: objective_and_grad(z)[0],
            x0=flatten_equal(x).ravel() if has_equalities else x.ravel(),
            jac=lambda z: objective_and_grad(z)[1],
            method="L-BFGS-B",
            options={"maxiter": max(10, int(inner_iterations)), "ftol": 1e-13, "gtol": 1e-10, "maxls": 50},
        )
        inner_total += int(getattr(result, "nit", 0))
        last_inner_success = bool(getattr(result, "success", False))
        last_inner_message = str(getattr(result, "message", ""))
        x = (
            affine_projector.project(np.asarray(result.x,float).reshape(ref.shape),strict=True)
            if affine_projector is not None else
            expand_equal(result.x) if has_equalities
            else np.asarray(result.x,dtype=float).reshape(ref.shape)
        )
        before_restore = _quad_plane_distances(x, constraint_faces)
        max_before_restore = float(np.max(before_restore)) if len(before_restore) else 0.0
        x, used, _ = _consensus_planarity_restore(
            x,
            f4,
            sweeps=max(0, int(restoration_sweeps)),
            relaxation=restore_relaxation,
            tolerance=plane_tol,
            constraint_faces=constraint_faces,
            equality_inverse=equality_inverse if has_equalities else None,
            affine_projector=affine_projector,
        )
        restoration_total += int(used)
        constraints, _ = _quad_constraint_values_and_gradient(x, constraint_faces, scales)
        lambdas = lambdas + rho * constraints
        distances = _quad_plane_distances(x, constraint_faces)
        max_dist = float(np.max(distances)) if len(distances) else 0.0
        normalized_violation = float(np.max(np.abs(constraints))) if len(constraints) else 0.0
        _, _, shape_residuals = _shape_metric_energy_and_gradient(x, f4, scales, reference_lengths, 1.0)
        shape_rms = float(np.sqrt(np.mean(shape_residuals * shape_residuals))) if shape_residuals.size else 0.0
        _, _, square_diag = _square_energy_and_gradient(x, f4, scales, 1.0)
        print(
            "[OPTCUTS-TEST-K3D-AL] "
            f"mode={'shape-preserving' if shape_mode else 'legacy'} outer={outer_done} rho={rho:.6g} "
            f"plane_before_restore={max_before_restore:.6g} max_plane_dist={max_dist:.6g} "
            f"shape_metric_rms={shape_rms:.6g} square_rms={square_diag['rms']:.6g} "
            f"worst_edge_ratio={square_diag['worst_edge_ratio']:.4f} restore_sweeps={used} "
            f"max_constraint={normalized_violation:.6g} inner_success={last_inner_success}"
        )
        if max_dist <= plane_tol:
            converged = True
            break
        if normalized_violation > 0.5 * previous_violation:
            rho = min(rho * 5.0, 1e12)
        previous_violation = normalized_violation

    polish_used = 0
    pre_polish_dist = _quad_plane_distances(x, constraint_faces)
    pre_polish_max = float(np.max(pre_polish_dist)) if len(pre_polish_dist) else 0.0
    if pre_polish_max > plane_tol and int(final_polish_sweeps) > 0:
        x, polish_used, _ = _consensus_planarity_restore(
            x,
            f4,
            sweeps=max(0, int(final_polish_sweeps)),
            relaxation=restore_relaxation if shape_mode else 1.0,
            tolerance=plane_tol,
            constraint_faces=constraint_faces,
            equality_inverse=equality_inverse if has_equalities else None,
            affine_projector=affine_projector,
        )
        restoration_total += int(polish_used)

    if has_equalities:
        exact=expand_equal(flatten_equal(x))
        if np.max(np.abs(exact-x))>1e-12:
            raise RuntimeError(
                "K3D_SEAM_AL_HARD_EQUALITY_LOST_IN_RESTORATION"
            )
        x=exact
    if affine_projector is not None:
        x=affine_projector.project(x,strict=True)
    after_dist = _quad_plane_distances(x, constraint_faces)
    max_after = float(np.max(after_dist)) if len(after_dist) else 0.0
    rms_after = float(np.sqrt(np.mean(after_dist * after_dist))) if len(after_dist) else 0.0
    converged = bool(max_after <= plane_tol)
    displacement = np.linalg.norm(x - ref, axis=1)
    _, _, final_shape_residuals = _shape_metric_energy_and_gradient(x, f4, scales, reference_lengths, 1.0)
    shape_rms = float(np.sqrt(np.mean(final_shape_residuals * final_shape_residuals))) if final_shape_residuals.size else 0.0
    shape_max = float(np.max(np.abs(final_shape_residuals))) if final_shape_residuals.size else 0.0
    _, _, square_diag = _square_energy_and_gradient(x, f4, scales, 1.0)

    geometry_validity=_quad_triangle_area_diagnostics(ref,x,f4)
    print(
        "[OPTCUTS-TEST-K3D-GEOMETRY] "
        f"min_signed_area_ratio={geometry_validity['min_oriented_area_ratio']:.7g} "
        f"min_unsigned_area_ratio={geometry_validity['min_unsigned_area_ratio']:.7g} "
        f"invalid_triangles={geometry_validity['collapsed_or_inverted_triangle_count']} "
        f"face_ids={geometry_validity['collapsed_or_inverted_face_ids']}",
        flush=True,
    )
    print(
        "[OPTCUTS-TEST-K3D-AL-FINAL] "
        f"mode={'shape-preserving' if shape_mode else 'legacy'} pre_polish={pre_polish_max:.6g} "
        f"max_plane_dist={max_after:.6g} tol={plane_tol:.6g} "
        f"shape_metric_rms={shape_rms:.6g} shape_metric_max={shape_max:.6g} "
        f"square_rms={square_diag['rms']:.6g} square_max={square_diag['max']:.6g} "
        f"worst_edge_ratio={square_diag['worst_edge_ratio']:.4f} "
        f"polish_sweeps={polish_used} converged={converged}"
    )

    return x, {
        "k3d_augmented_lagrangian_applied": True,
        "k3d_seam_shape_guard_enabled":bool(seam_geometry_guard),
        "k3d_seam_shape_guard_weight":float(shape_weight),
        "k3d_seam_area_guard_weight":float(area_guard_weight),
        "k3d_seam_min_oriented_triangle_area_ratio":float(
            geometry_validity["min_oriented_area_ratio"]
        ),
        "k3d_seam_min_unsigned_triangle_area_ratio":float(
            geometry_validity["min_unsigned_area_ratio"]
        ),
        "k3d_seam_invalid_triangle_count":int(
            geometry_validity["collapsed_or_inverted_triangle_count"]
        ),
        "k3d_seam_invalid_face_ids":list(
            geometry_validity["collapsed_or_inverted_face_ids"]
        ),
        "k3d_al_source_seam_hard_equality":bool(
            has_equalities or affine_projector is not None
        ),
        "k3d_al_source_seam_affine_constraint_count":int(
            affine_projector.count if affine_projector is not None else 0
        ),
        "k3d_al_source_seam_affine_max_residual":float(
            affine_projector.residual(x)
            if affine_projector is not None else 0.
        ),
        "k3d_al_source_seam_group_count":int(len(equality_groups or ())),
        "k3d_al_source_seam_reduced_unknowns":int(unique_count),
        "k3d_al_source_seam_max_gap":float(
            max((np.max(np.linalg.norm(
                x[np.asarray(g,int)]-x[int(g[0])],axis=1))
                for g in (equality_groups or ()) if len(g)>1),
                default=0.)
        ),
        "k3d_planarity_constraint_type": "robust max-area scalar triple product equality per quad",
        "k3d_planarity_constraint": "max-area base triangle; remaining corner coplanar",
        "k3d_planarity_constraint_reordered_faces": True,
        "k3d_planarity_degenerate_base_count": int(degenerate_base_count),
        "k3d_augmented_lagrangian_reference": "validity-repaired ordinary K3D square+surface solution",
        "k3d_augmented_lagrangian_anchor_weight": float(w_anchor),
        "k3d_augmented_lagrangian_rho_initial": float(rho0),
        "k3d_augmented_lagrangian_rho_final": float(rho),
        "k3d_augmented_lagrangian_outer_iterations": int(outer_done),
        "k3d_augmented_lagrangian_inner_iterations_total": int(inner_total),
        "k3d_augmented_lagrangian_last_inner_success": bool(last_inner_success),
        "k3d_augmented_lagrangian_last_inner_message": str(last_inner_message),
        "k3d_augmented_lagrangian_converged": bool(converged),
        "k3d_planarity_restoration_sweeps_total": int(restoration_total),
        "k3d_planarity_final_polish_sweeps": int(polish_used),
        "k3d_planarity_pre_final_polish_max_distance": float(pre_polish_max),
        "k3d_hard_planarity_tolerance": float(plane_tol),
        "k3d_hard_planarity_max_distance_before": float(max_before),
        "k3d_hard_planarity_rms_distance_before": float(rms_before),
        "k3d_hard_planarity_max_distance_after": float(max_after),
        "k3d_hard_planarity_rms_distance_after": float(rms_after),
        "k3d_hard_planarity_constraint_satisfied": bool(max_after <= plane_tol),
        "k3d_hard_planarity_vertex_displacement_rms": float(np.sqrt(np.mean(displacement * displacement))) if len(displacement) else 0.0,
        "k3d_hard_planarity_vertex_displacement_max": float(np.max(displacement)) if len(displacement) else 0.0,
        "k3d_shape_preserving_mode": bool(shape_mode),
        "k3d_shape_preservation_model": "normalized four edges + two diagonals per quad" if shape_mode else "disabled",
        "k3d_shape_preservation_weight": float(shape_weight),
        "k3d_shape_metric_relative_rms": float(shape_rms),
        "k3d_shape_metric_relative_max": float(shape_max),
        "k3d_shape_preserving_restore_relaxation": float(restore_relaxation),
        "k3d_global_square_regularization_enabled": bool(effective_square_weight > 0.0),
        "k3d_smoothmax_enabled": bool(effective_square_weight>0 and square_tail_alpha>0),
        "k3d_smoothmax_alpha":float(square_tail_alpha),
        "k3d_smoothmax_beta":float(square_tail_beta),
        "k3d_smoothmax_weight":float(effective_square_weight),
        "k3d_smoothmax_stage":"optcuts_test_al",
        "k3d_global_square_regularization_model": "all quads: four cyclic edge equalities + equal diagonals",
        "k3d_global_square_weight_requested": float(square_weight),
        "k3d_global_square_weight_effective": float(effective_square_weight),
        "k3d_global_square_residual_rms": float(square_diag["rms"]),
        "k3d_global_square_residual_max": float(square_diag["max"]),
        "k3d_global_square_worst_edge_ratio": float(square_diag["worst_edge_ratio"]),
    }


def install_optcuts_test_k3d_augmented_lagrangian_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_k3d_al_installed", False):
        return
    base = pipeline._optimize_k3d

    def optimize(target: Any, mesh: Any, parameterization: Any, params: Any):
        result = base(target, mesh, parameterization, params)
        k3d, report = result
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return result
        reference = np.asarray(k3d.vertices, dtype=float).copy()
        requested_square_weight = float(getattr(params, "w_square", 10.0))
        solved, al_metrics = _augmented_lagrangian_planarize(
            reference,
            np.asarray(k3d.faces, dtype=int),
            anchor_weight=float(getattr(params, "k3d_al_anchor_weight", 1.0)),
            square_weight=requested_square_weight,
            rho0=float(getattr(params, "k3d_al_rho0", 10.0)),
            outer_iterations=int(getattr(params, "k3d_al_outer_iterations", 16)),
            inner_iterations=int(getattr(params, "k3d_al_inner_iterations", 70)),
            relative_plane_tolerance=float(getattr(params, "k3d_al_relative_planarity_tolerance", 1e-6)),
            restoration_sweeps=int(getattr(params, "k3d_al_restoration_sweeps", 8)),
            final_polish_sweeps=int(getattr(params, "k3d_al_final_polish_sweeps", 400)),
            equality_groups=(
                getattr(mesh,"metrics",{}).get("seam_exact_vertex_groups",[])
            ),
            affine_rows=(
                getattr(mesh,"metrics",{}).get("seam_exact_affine_rows",[])
            ),
        )
        k3d.vertices[:] = solved
        try:
            k3d.metrics.update(al_metrics)
            if bool(al_metrics.get("k3d_shape_preserving_mode", False)):
                k3d.metrics["k3d_planarity_mode"] = "hard equality + reference metric + global square regularization"
            else:
                k3d.metrics["k3d_planarity_mode"] = "hard equality via robust augmented Lagrangian + feasibility restoration"
            k3d.metrics["k3d_soft_w_planar_is_authoritative"] = False
            k3d.metrics["k3d_al_authoritative_result"] = True
        except Exception:
            pass
        try:
            suffix = (
                " + hard quad planarity (shape-preserving + global square AL)"
                if bool(al_metrics.get("k3d_shape_preserving_mode", False))
                else " + hard quad planarity (Augmented Lagrangian + feasibility restoration)"
            )
            report.objective = str(getattr(report, "objective", "")) + suffix
            report.constraint_violation = float(al_metrics.get("k3d_hard_planarity_max_distance_after", 0.0))
        except Exception:
            pass
        if bool(al_metrics.get("k3d_global_square_regularization_enabled",False)):
            print(
                "[OPTCUTS-TEST-K3D-WEIGHTS] "
                f"w_square requested/effective={requested_square_weight:.6g}/"
                f"{float(al_metrics.get('k3d_global_square_weight_effective', 0.0)):.6g} "
                f"shape_preservation={float(al_metrics.get('k3d_shape_preservation_weight', 0.0)):.6g}"
            )
        return k3d, report

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
    pipeline._onestring_optcuts_test_k3d_al_installed = True


__all__ = [
    "SHAPE_PRESERVING_VERSION_ID",
    "SHAPE_PRESERVING_VERSION_LABEL",
    "install_optcuts_test_k3d_augmented_lagrangian_patch",
    "_robust_constraint_faces",
    "_quad_plane_distances",
    "_consensus_planarity_restore",
    "_augmented_lagrangian_planarize",
]
