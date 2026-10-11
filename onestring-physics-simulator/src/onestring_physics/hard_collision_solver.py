"""Numerically certified XY footprint nonpenetration for rigid SE(2) tiles.

This restores feasibility by projecting collision minimum-translation vectors;
it is a heuristic for a nonconvex constraint, not a proof of feasibility.
Each accepted hard candidate gets an independent full-footprint SAT audit.
"""
from __future__ import annotations

import time
import numpy as np


def footprint_contacts(xy, faces):
    """All penetrating pairs as (tile_i, tile_j, normal, sign, SAT depth).

    Each tile's convex hull includes all eight 3D panel corners projected into
    XY. Contacts between hinge-connected panels are NOT excluded.
    """
    polygons = np.asarray(xy, dtype=float).reshape(-1, 2)[np.asarray(faces, int)]
    if len(polygons) < 2:
        return (np.zeros(0, int), np.zeros(0, int), np.empty((0, 2)),
                np.empty(0), np.empty(0))
    lo, hi = polygons.min(axis=1), polygons.max(axis=1)
    order = np.argsort(lo[:, 0], kind="stable")
    left, right = [], []
    for rank, i in enumerate(order):
        stop = np.searchsorted(lo[order, 0], hi[i, 0], side="left")
        eligible = order[rank + 1:stop]
        eligible = eligible[(lo[eligible, 1] < hi[i, 1]) &
                            (hi[eligible, 1] > lo[i, 1])]
        left.extend([int(i)] * len(eligible))
        right.extend(eligible.tolist())
    if not left:
        return (np.zeros(0, int), np.zeros(0, int), np.empty((0, 2)),
                np.empty(0), np.empty(0))
    ii, jj = np.asarray(left, int), np.asarray(right, int)
    a, b = polygons[ii], polygons[jj]
    edges = np.concatenate((np.roll(a, -1, axis=1) - a,
                            np.roll(b, -1, axis=1) - b), axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    normals = np.stack((-edges[..., 1], edges[..., 0]), axis=-1) / np.maximum(lengths[..., None], 1e-30)
    ap = np.einsum("pvd,pad->pav", a, normals)
    bp = np.einsum("pvd,pad->pav", b, normals)
    forward = ap.max(axis=2) - bp.min(axis=2)
    reverse = bp.max(axis=2) - ap.min(axis=2)
    depths = np.minimum(forward, reverse)
    depths[lengths <= 1e-30] = np.inf
    axes = np.argmin(depths, axis=1)
    depth = depths[np.arange(len(ii)), axes]
    mask = np.isfinite(depth) & (depth > 0.)
    active_axes = axes[mask]
    sign = np.where(forward[mask, active_axes] <= reverse[mask, active_axes], 1., -1.)
    return ii[mask], jj[mask], normals[mask, active_axes], sign, depth[mask]


def repair_nonpenetration(initial_pose, positions, faces, reference_pose,
                          max_center_drift_tiles, scale, deadline, rng,
                          *, max_iterations=480, log_stride=8, callback=None):
    """Try to enforce SAT nonpenetration by rigid SE(2) panel displacements.

    Translation bounds match the original L-BFGS-B bounds. Feasibility means
    every SAT depth is within numerical tolerance 1e-10 * tile scale; failure
    is explicitly reported and never silently converted into success.
    """
    pose = np.asarray(initial_pose, float).copy()
    ref = np.asarray(reference_pose, float)
    drift = float(max_center_drift_tiles)
    pose[:, 1:] = np.clip(pose[:, 1:], ref[:, 1:]-drift, ref[:, 1:]+drift)
    best = pose.copy()
    history = [(0, pose.copy())]
    eps = max(1e-12, float(scale)*1e-10)
    clearance = max(2e-11, float(scale)*1e-7)
    ntiles = len(pose)
    best_quality = None
    stagnation = 0
    diagnostics = []
    for iteration in range(max_iterations+1):
        xy, _ = positions(pose)
        ii, jj, normals, signs, depths = footprint_contacts(xy, faces)
        collisions = int(np.count_nonzero(depths > eps))
        max_depth = float(np.max(depths, initial=0.))
        quality = (collisions, max_depth, float(depths@depths))
        if best_quality is None or quality < best_quality:
            best_quality = quality
            best = pose.copy()
            stagnation = 0
        else:
            stagnation += 1
        diagnostics.append(dict(iteration=int(iteration), collisions=collisions,
                                max_depth=max_depth, penetration_energy=float(depths@depths)))
        if collisions == 0 and max_depth <= eps:
            if history[-1][0] != iteration:
                history.append((iteration, pose.copy()))
            return pose, True, history, diagnostics
        if iteration == max_iterations or time.perf_counter() >= deadline:
            break
        displacement = np.zeros((ntiles, 2), float)
        degree = np.zeros(ntiles, float)
        vector = signs[:, None]*normals*(depths+clearance)[:, None]
        np.add.at(displacement, ii, -vector)
        np.add.at(displacement, jj, vector)
        np.add.at(degree, ii, 1.)
        np.add.at(degree, jj, 1.)
        active = degree > 0
        displacement[active] /= degree[active, None]
        proposal = pose.copy()
        proposal[:, 1:] += .7*displacement/max(scale, 1e-30)
        proposal[:, 1:] = np.clip(proposal[:, 1:], ref[:, 1:]-drift, ref[:, 1:]+drift)
        if stagnation >= 12 and np.any(active):
            magnitude = min(.3, .015*(1+stagnation//12))
            proposal[active, 1:] += rng.normal(0., magnitude, (np.count_nonzero(active), 2))
            proposal[active, 0] += rng.normal(0., .075, np.count_nonzero(active))
            proposal[:, 1:] = np.clip(proposal[:, 1:], ref[:, 1:]-drift, ref[:, 1:]+drift)
            stagnation = 0
        pose = proposal
        if iteration%max(1, log_stride) == 0:
            history.append((iteration+1, pose.copy()))
        if callback and iteration%10 == 0:
            callback(iteration, collisions, max_depth)
    if history[-1][0] != iteration:
        history.append((iteration, best.copy()))
    return best, False, history, diagnostics


def refine_feasible_pose(pose, positions, faces, reference_pose, max_center_drift_tiles,
                         scale, deadline, rng, objective, *, iterations=24):
    """Projected descent on hinge/anchor energy, rejecting nonfeasible steps."""
    current = np.asarray(pose, float).copy()
    current_energy, _ = objective(current)
    history = [(0, current.copy())]
    step = .10
    scale_sq = max(float(scale)**2, 1e-30)
    for iteration in range(iterations):
        if time.perf_counter() >= deadline:
            break
        _, gradient = objective(current)
        direction = -gradient/scale_sq
        direction[:, 0] = np.clip(direction[:, 0], -1., 1.)
        direction[:, 1:] = np.clip(direction[:, 1:], -1., 1.)
        proposed = current+step*direction
        projected, feasible, _, _ = repair_nonpenetration(
            proposed, positions, faces, reference_pose, max_center_drift_tiles,
            scale, deadline, rng, max_iterations=24, log_stride=24,
        )
        if feasible:
            after, _ = objective(projected)
            if after < current_energy-1e-12*scale_sq:
                current = projected
                current_energy = after
                history.append((iteration+1, current.copy()))
                step = min(.5, step*1.2)
                continue
        step *= .5
        if step < 1e-5:
            break
    return current, history
