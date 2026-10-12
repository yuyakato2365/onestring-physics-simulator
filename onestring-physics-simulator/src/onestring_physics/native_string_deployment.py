"""Experimental rigid-panel deployment driven by ONE unilateral routed string.

This is a CPU position-based rigid-body approximation, NOT the authors' solver,
ABD/IPC or a verified predictor of physical fabrication success. Crucially,
unlike the old snap/lift preview, the T3D target does not actuate the model.

Each T2D solid is one rigid 6-DoF body. Constraints are point hinges, a
unilateral (tension-only) total string length, approximate 3D OBB contacts,
and a fixed support table. String guides are fixed MATERIAL points on tiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class NativeDeploymentError(RuntimeError):
    """Invalid topology or numerically unreliable physical deployment."""


@dataclass
class _Mechanism:
    rest: np.ndarray
    local: np.ndarray
    centers: np.ndarray
    rotations: np.ndarray
    inv_inertia_local: np.ndarray
    hinges: list[tuple[int, int, int, int]]
    guide_bodies: np.ndarray
    guide_locals: np.ndarray
    anchor: np.ndarray
    initial_length: float
    floor_z: float
    tile_scale: float
    axes_local: np.ndarray
    circumradius: np.ndarray


def _safe_unit(a: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(a))
    if length <= 1e-12:
        raise NativeDeploymentError("Degenerate tile geometry: cannot construct a 3D collision frame")
    return a / length


def _physical_frame(tile: np.ndarray) -> np.ndarray:
    # Axis 1 is orthogonalized, so even non-square quads define a proper frame.
    x = _safe_unit(tile[1] - tile[0])
    y = tile[3] - tile[0]
    y = _safe_unit(y - np.dot(y, x) * x)
    z = _safe_unit(np.cross(x, y))
    return np.stack((x, y, z), axis=0)


def _vertex_idx(hinge, which: str) -> int:
    vertex_id = int(getattr(hinge, which))
    if str(getattr(hinge, "surface", "top")).lower() == "bottom" and 0 <= vertex_id < 4:
        vertex_id += 4
    if not 0 <= vertex_id < 8:
        raise NativeDeploymentError(f"Invalid {which}={vertex_id}: eight-vertex T2D panels required")
    return vertex_id


def _prepare(state, *, anchor_height_tiles: float) -> _Mechanism:
    rest = np.asarray(state.tiles_2d_dual_hinge.vertices, dtype=float)
    if rest.ndim != 3 or rest.shape[1:] != (8, 3) or not len(rest):
        raise NativeDeploymentError("Native simulation requires non-empty T2D Dual with 8 vertices per rigid panel")
    if not np.all(np.isfinite(rest)):
        raise NativeDeploymentError("Nonfinite T2D panel coordinates")
    centers = rest.mean(axis=1)
    local = rest - centers[:, None, :]
    n = len(rest)
    scale = float(np.median(np.linalg.norm(rest[:, 1, :] - rest[:, 0, :], axis=1)))
    if scale <= 1e-9:
        raise NativeDeploymentError("Zero-size T2D tile geometry")
    axes = np.asarray([_physical_frame(t[:4]) for t in local])
    inertia_inv = np.zeros((n, 3, 3))
    for idx in range(n):
        r = local[idx]
        inertia = (np.sum(r*r) * np.eye(3) - r.T @ r) / 8.0
        inertia_inv[idx] = np.linalg.pinv(inertia, rcond=1e-8)
    hinges = []
    for h in getattr(state.hinge_graph, "hinges", []):
        a, b = int(h.tile_a), int(h.tile_b)
        if 0 <= a < n and 0 <= b < n and a != b:
            hinges.append((a, b, _vertex_idx(h, "local_vertex_a"),
                           _vertex_idx(h, "local_vertex_b")))
    gaps = {int(g.id): g for g in getattr(state.gap_graph, "gaps", [])}
    guide_bodies = []
    guide_vertices = []
    ids = list(getattr(state.string_path, "gap_ids", []))
    for gap_id in ids:
        gap = gaps.get(int(gap_id))
        if gap is None or not getattr(gap, "surrounding_tiles", None):
            raise NativeDeploymentError(
                f"STRING_GUIDE_MISSING: routed gap {gap_id} has no attached panel"
            )
        tile_id = int(gap.surrounding_tiles[0])
        if tile_id < 0 or tile_id >= n:
            raise NativeDeploymentError(f"STRING_GUIDE_OWNER_INVALID: gap {gap_id}")
        xy = np.asarray(gap.centroid_2d, dtype=float).reshape(-1)[:2]
        if len(xy) != 2 or not np.all(np.isfinite(xy)):
            raise NativeDeploymentError(f"STRING_GUIDE_POSITION_INVALID: gap {gap_id}")
        nearest = int(np.argmin(np.linalg.norm(rest[tile_id, :4, :2] - xy, axis=1)))
        guide_bodies.append(tile_id)
        guide_vertices.append(local[tile_id, nearest].copy())
    if len(guide_bodies) < 2 or len(set(guide_bodies)) < 2:
        raise NativeDeploymentError(
            "STRING_ROUTE_UNDERCONSTRAINED: at least two guides on different panels required; "
            "no hidden T3D lift or global contraction will be substituted"
        )
    bodies = np.asarray(guide_bodies, dtype=int)
    guides = np.asarray(guide_vertices, dtype=float)
    point0 = centers[bodies[0]] + guides[0]
    anchor = point0 + np.array([0., 0., scale * float(anchor_height_tiles)])
    world = centers[bodies] + guides
    points = np.vstack((anchor[None, :], world))
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if length <= 1e-9:
        raise NativeDeploymentError("Routed string has zero initial length")
    return _Mechanism(
        rest=rest.copy(), local=local, centers=centers.copy(),
        rotations=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
        inv_inertia_local=inertia_inv, hinges=hinges,
        guide_bodies=bodies, guide_locals=guides,
        anchor=anchor, initial_length=length,
        floor_z=float(np.min(rest[:, 4:, 2])),
        tile_scale=scale, axes_local=axes,
        circumradius=np.max(np.linalg.norm(local, axis=2), axis=1),
    )


def _rotated_vectors(m: _Mechanism, bodies: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", m.rotations[bodies], vectors)


def _vertices(m: _Mechanism) -> np.ndarray:
    return m.centers[:, None, :] + np.einsum("nij,nkj->nki", m.rotations, m.local)


def _rotate_one(m: _Mechanism, body: int, theta: np.ndarray) -> None:
    norm = float(np.linalg.norm(theta))
    if norm > 0.22:
        theta = theta * (0.22 / norm)
    if np.any(theta):
        m.rotations[body] = Rotation.from_rotvec(theta).as_matrix() @ m.rotations[body]


def _inverse_inertia_world(m: _Mechanism, body: int) -> np.ndarray:
    rot = m.rotations[body]
    return rot @ m.inv_inertia_local[body] @ rot.T


def _apply_point_impulse(m: _Mechanism, body: int, r: np.ndarray, impulse: np.ndarray) -> None:
    m.centers[body] += impulse  # unit body mass
    theta = _inverse_inertia_world(m, body) @ np.cross(r, impulse)
    _rotate_one(m, body, theta)


def _pin_constraints(m: _Mechanism, stiffness: float) -> None:
    for a, b, va, vb in m.hinges:
        ra = m.rotations[a] @ m.local[a, va]
        rb = m.rotations[b] @ m.local[b, vb]
        error = (m.centers[b] + rb) - (m.centers[a] + ra)
        length = float(np.linalg.norm(error))
        if length < 1e-10:
            continue
        direction = error / length
        xa = np.cross(ra, direction)
        xb = np.cross(rb, direction)
        effective = (2.0 + xa @ _inverse_inertia_world(m, a) @ xa
                     + xb @ _inverse_inertia_world(m, b) @ xb)
        impulse = direction * (float(stiffness) * length / max(effective, 1e-12))
        _apply_point_impulse(m, a, ra, impulse)
        _apply_point_impulse(m, b, rb, -impulse)


def _string_geometry(m: _Mechanism) -> tuple[float, np.ndarray, np.ndarray]:
    radii = _rotated_vectors(m, m.guide_bodies, m.guide_locals)
    positions = m.centers[m.guide_bodies] + radii
    points = np.vstack((m.anchor[None, :], positions))
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    u = segments / np.maximum(lengths[:, None], 1e-12)
    grad = u[:-1] - u[1:] if len(positions) > 1 else np.zeros((0, 3))
    # Last guide is the free end, so its gradient has only the last incoming segment.
    guide_gradient = np.vstack((grad, u[-1][None, :]))
    return float(lengths.sum()), guide_gradient, radii


def _string_constraint(m: _Mechanism, command: float, strength: float) -> float:
    current, gradient, radii = _string_geometry(m)
    excess = current - command
    if excess <= 0.0 or strength <= 0.0:
        return max(0.0, excess)
    body_gradient = np.zeros((len(m.centers), 3))
    angular_gradient = np.zeros_like(body_gradient)
    for k, body in enumerate(m.guide_bodies):
        body_gradient[body] += gradient[k]
        angular_gradient[body] += np.cross(radii[k], gradient[k])
    denominator = float(np.sum(body_gradient**2))
    inv_world = np.einsum("nij,njk,nlk->nil", m.rotations, m.inv_inertia_local, m.rotations)
    denominator += float(np.einsum(
        "ni,nij,nj->", angular_gradient, inv_world, angular_gradient
    ))
    if denominator < 1e-12:
        return excess
    lamb = float(strength) * excess / denominator
    # Cap per-iteration correction to prevent explosions for poor initial T2D.
    impulse = -lamb * body_gradient
    magnitudes = np.linalg.norm(impulse, axis=1)
    impulse *= (np.minimum(1.0, 0.2 * m.tile_scale /
                            np.maximum(magnitudes, 1e-12)))[:, None]
    m.centers += impulse
    angular = -lamb * np.einsum("nij,nj->ni", inv_world, angular_gradient)
    norms = np.linalg.norm(angular, axis=1)
    angular *= (np.minimum(1.0, 0.22 / np.maximum(norms, 1e-12)))[:, None]
    dr = Rotation.from_rotvec(angular).as_matrix()
    m.rotations = np.einsum("nij,njk->nik", dr, m.rotations)
    return excess


def _floor_constraint(m: _Mechanism) -> int:
    world = _vertices(m)
    below = np.min(world[:, :, 2], axis=1) - m.floor_z
    indices = np.flatnonzero(below < 0)
    m.centers[indices, 2] -= below[indices]
    return int(len(indices))


def _overlap_obb(a: np.ndarray, b: np.ndarray, axes_a: np.ndarray,
                 axes_b: np.ndarray, tolerance: float) -> tuple[float, np.ndarray]:
    """Approximate convex frustum contact using 15 OBB separating axes."""
    axes = [*axes_a, *axes_b]
    axes.extend(np.cross(x, y) for x in axes_a for y in axes_b)
    smallest = float("inf")
    normal = np.zeros(3)
    centroid_diff = b.mean(axis=0) - a.mean(axis=0)
    for raw in axes:
        norm = float(np.linalg.norm(raw))
        if norm < 1e-9:
            continue
        u = raw / norm
        av = a @ u
        bv = b @ u
        depth = min(float(av.max()), float(bv.max())) - max(float(av.min()), float(bv.min()))
        if depth <= tolerance:
            return 0.0, normal
        if depth < smallest:
            smallest = depth
            normal = u if np.dot(centroid_diff, u) >= 0 else -u
    return (smallest, normal) if np.isfinite(smallest) else (0.0, normal)


def _contacts(m: _Mechanism, *, apply: bool, strength: float) -> int:
    # Spatial broad phase. This is a 3D approximation, NOT CCD/IPC.
    n = len(m.centers)
    if n < 2:
        return 0
    maximum = float(np.max(m.circumradius))
    candidate = cKDTree(m.centers).query_pairs(2.0 * maximum + 1e-5, output_type="ndarray")
    if not len(candidate):
        return 0
    world = _vertices(m)
    axes = np.einsum("nij,nkj->nki", m.rotations, m.axes_local)
    count = 0
    tol = 1e-7 * m.tile_scale
    for i, j in candidate:
        if np.linalg.norm(m.centers[i] - m.centers[j]) > m.circumradius[i] + m.circumradius[j]:
            continue
        depth, normal = _overlap_obb(world[i], world[j], axes[i], axes[j], tol)
        if depth <= 0:
            continue
        count += 1
        if apply and strength > 0:
            correction = 0.5 * min(depth, 0.2 * m.tile_scale) * strength * normal
            m.centers[i] -= correction
            m.centers[j] += correction
    return count


def _hinge_rms(m: _Mechanism) -> float:
    if not m.hinges:
        return 0.0
    errors = []
    for a, b, va, vb in m.hinges:
        pa = m.centers[a] + m.rotations[a] @ m.local[a, va]
        pb = m.centers[b] + m.rotations[b] @ m.local[b, vb]
        errors.append(float(np.dot(pa - pb, pa - pb)))
    return float(np.sqrt(np.mean(errors)))


def simulate_native_string_deployment(state, params, progress_callback=None):
    """Simulate T2D -> 3D using only gravity, physical constraints and string pull.

    Returns the existing DeploymentResult frame contract. Physical success is
    never inferred merely from resemblance to T3D.
    """
    from .legacy_pipeline_base import DeploymentResult

    t0 = time.perf_counter()
    ratio = float(getattr(params, "native_pull_end_ratio", 0.45))
    anchor_height = float(getattr(params, "native_anchor_height_tiles", 2.0))
    timestep = float(getattr(params, "native_timestep", 0.02))
    if not (0.05 <= ratio <= 1 and 0.1 <= anchor_height <= 20
            and 1e-4 <= timestep <= 0.1):
        raise NativeDeploymentError("Invalid native pull ratio, anchor height or timestep")
    m = _prepare(state, anchor_height_tiles=anchor_height)
    total = max(2, int(params.steps))
    iterations = max(1, int(params.solver_iterations))
    substeps = max(1, int(params.solver_substeps))
    damping = float(np.clip(params.damping_ratio, 0, .99))
    gravity = float(max(0., getattr(params, "gravity", 9.81)))
    hinge_strength = float(np.clip(params.hinge_weight, 0., 1.))
    collision_strength = float(np.clip(params.collision_weight, 0., 1.))
    nframes = max(2, int(getattr(params, "max_animation_frames", 40)))
    frames = [m.rest.copy()]
    collision_counts = [_contacts(m, apply=False, strength=0.)]
    v = np.zeros_like(m.centers)
    w = np.zeros_like(m.centers)
    gravity_vec = np.array([0., 0., -gravity])
    floor_touches = 0
    for step in range(total):
        phase = (step + 1) / total
        pull_smooth = phase * phase * (3. - 2. * phase)
        command = m.initial_length * (1. - (1. - ratio) * pull_smooth)
        for substep in range(substeps):
            dt = timestep / substeps
            old_center = m.centers.copy()
            old_rot = m.rotations.copy()
            v += gravity_vec * dt
            m.centers += v * dt
            m.rotations = np.einsum(
                "nij,njk->nik", Rotation.from_rotvec(w * dt).as_matrix(), m.rotations
            )
            for sweep in range(iterations):
                _pin_constraints(m, hinge_strength)
                _string_constraint(m, command, 1.0)
                floor_touches += _floor_constraint(m)
                if collision_strength > 0 and sweep % 2 == 0:
                    _contacts(m, apply=True, strength=collision_strength)
            _pin_constraints(m, hinge_strength)
            _floor_constraint(m)
            v = (m.centers - old_center) / dt
            w = (Rotation.from_matrix(
                np.einsum("nij,nkj->nik", m.rotations, old_rot)
            ).as_rotvec() / dt)
            v *= (1. - damping)
            w *= (1. - damping)
            v = np.clip(v, -20. * m.tile_scale / dt, 20. * m.tile_scale / dt)
            w = np.clip(w, -20., 20.)
            if floor_touches:
                # Approximate tabletop Coulomb resistance while resting.
                mverts = _vertices(m)
                resting = np.min(mverts[:, :, 2], axis=1) <= m.floor_z + 1e-6
                v[resting, :2] *= max(0., 1. - float(params.contact_friction) * dt)
                v[resting, 2] = np.maximum(v[resting, 2], 0.)
        if progress_callback and (step % max(1, total // 30) == 0 or step == total - 1):
            progress_callback(
                "Native rigid OneString pull", min(0.98, (step + 1) / total),
                f"step {step + 1}/{total}; string command {command:.5g}",
            )
        if (len(frames) < nframes - 1 and
                (step + 1) % max(1, total // max(1, nframes - 2)) == 0):
            frames.append(_vertices(m).copy())
            collision_counts.append(_contacts(m, apply=False, strength=0.))
    final = _vertices(m)
    if len(frames) == 0 or not np.array_equal(frames[-1], final):
        frames.append(final.copy())
        collision_counts.append(_contacts(m, apply=False, strength=0.))
    string_length, _, _ = _string_geometry(m)
    final_command = m.initial_length * ratio
    hinge_error = _hinge_rms(m)
    contacts = _contacts(m, apply=False, strength=0.)
    target = np.asarray(state.tiles_3d.vertices, dtype=float)
    target_error = (float(np.sqrt(np.mean((final-target)**2)))
                    if target.shape == final.shape else float("nan"))
    rise = float(np.max(final[:, :, 2]) - np.max(m.rest[:, :, 2]))
    metrics = {
        "physics_backend": "native",
        "actual_backend": "native_cpu_pbd",
        "simulation_model": "approximate_6dof_rigid_body_unilateral_routed_string",
        "physical_solver_fidelity": "experimental; point hinges + OBB contact, no CCD/IPC or channel capstan friction",
        "string_model": "unilateral_total_length",
        "initial_layout": "exact_t2d_dual_hinge",
        "t3d_target_used_as_actuator": False,
        "rigid_error": float(np.max(np.abs(
            np.linalg.norm(final[:, 1] - final[:, 0], axis=1)
            - np.linalg.norm(m.rest[:, 1] - m.rest[:, 0], axis=1)
        ))),
        "hinge_error": hinge_error,
        "collision_count": int(contacts),
        "initial_collision_count": int(collision_counts[0]),
        "initial_string_length": float(m.initial_length),
        "commanded_string_length": float(final_command),
        "actual_string_length": float(string_length),
        "constraint_violation": float(max(0., string_length-final_command)),
        "string_guide_count": int(len(m.guide_bodies)),
        "panel_count": int(len(m.rest)),
        "anchor_height_tiles": anchor_height,
        "rise_from_t2d": rise,
        "final_deployment_error_to_T3D": target_error,
        "snap_error": float("nan"),
        "lift_error": float("nan"),
        "stable_state": bool(
            hinge_error < .05 * m.tile_scale
            and contacts == 0 and
            string_length <= final_command + .02 * m.tile_scale
        ),
        "collision_model": "3d_obb_sat_discrete",
        "simulation_runtime_seconds": float(time.perf_counter()-t0),
        "elapsed_time": float(time.perf_counter()-t0),
    }
    if progress_callback:
        progress_callback("Native rigid OneString completed", 1., "frames and diagnostics available")
    return DeploymentResult(
        frames=frames, final_tiles=final.copy(),
        metrics=metrics, collision_counts=collision_counts,
    )
