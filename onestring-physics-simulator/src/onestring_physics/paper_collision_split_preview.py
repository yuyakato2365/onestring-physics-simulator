"""Lightweight geometry snapshots for ACCEPTED and REJECTED Split candidates.

A trial has been fully rebuilt and optimized, but a rejected candidate must not
replace the authoritative OneStringDesignState. Keeping entire rejected states
would retain huge K3D histories, surface caches and optimizer internals.
Snapshots contain only float32 vertices, source faces, and small hinge index
arrays; no objective history, SciPy state or STL source is altered.
"""
from __future__ import annotations

import numpy as np


def capture_split_trial_geometry(state) -> dict | None:
    """Capture the minimum geometry needed to inspect a proposed Split.

    A missing stage (common in mocked unit tests or failed partial builds)
    is reported as unavailable, rather than disrupting feasibility evaluation.
    The snapshot's panel correspondence follows source-face order throughout.
    """
    if state is None:
        return None
    specs = (
        ("M2D", "mesh_2d_initial", 4),
        ("K2D", "mesh_2d_optimized", 4),
        ("T2D Top", "tiles_2d_top_hinge", 8),
        ("T2D Dual", "tiles_2d_dual_hinge", 8),
    )
    captured = {}
    panel_count = None
    for name, attribute, width in specs:
        obj = getattr(state, attribute, None)
        if obj is None or not hasattr(obj, "vertices"):
            continue
        pts = np.asarray(obj.vertices, dtype=float)
        if not np.isfinite(pts).all():
            continue
        if width == 4:
            if not hasattr(obj, "faces"):
                continue
            faces = np.asarray(obj.faces, dtype=np.int32)
            if faces.ndim != 2 or faces.shape[1] != 4 or (
                len(faces) and (faces.min() < 0 or faces.max() >= len(pts))
            ):
                continue
            n = len(faces)
        else:
            faces = None
            if pts.ndim != 3 or pts.shape[1:] != (8, 3):
                continue
            n = len(pts)
        if panel_count is None:
            panel_count = n
        if n != panel_count:
            continue
        if width == 4 and (pts.ndim != 2 or pts.shape[1] != 3):
            continue
        entry = {"vertices": np.asarray(pts, dtype=np.float32).copy()}
        if faces is not None:
            entry["faces"] = faces.copy()
        captured[name] = entry
    if not captured or "T2D Dual" not in captured:
        return None

    joint_records = []
    graph = getattr(state, "hinge_graph", None)
    for hinge in getattr(graph, "hinges", ()) or ():
        try:
            a, b = int(hinge.tile_a), int(hinge.tile_b)
            ca, cb = int(hinge.local_vertex_a), int(hinge.local_vertex_b)
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= a < panel_count and 0 <= b < panel_count and 0 <= ca < 8 and 0 <= cb < 8:
            joint_records.append((a, ca, b, cb))
    joints = np.asarray(joint_records, dtype=np.int32).reshape(-1, 4)

    return {
        "stages": captured,
        "hinges": joints,
        "tile_count": int(panel_count),
        "geometry_only": True,
        "valid_for_fabrication_export": False,
    }


def collision_tile_severity(vertices) -> dict:
    """Per-tile squared SAT penetration for an exact 8-corner XY footprint.

    This is the SAME penetration measure as the Split acceptance diagnostic;
    contacts at a shared corner or edge are not counted as penetrations.
    """
    from .paper_collision_split import _collision_depths

    tiles = np.asarray(vertices, dtype=float)
    if tiles.ndim != 3 or tiles.shape[1:] != (8, 3):
        raise ValueError("Expected N x 8 x 3 rigid panel coordinates")
    n = len(tiles)
    strength = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=np.int32)
    if not n:
        return dict(strength=strength, count=count, pairs=np.empty((0, 2), dtype=np.int32),
                    depth=np.zeros(0), pair_count=0, max_depth=0., energy=0.)
    scale = float(np.median(np.linalg.norm(
        np.roll(tiles[:, :4, :2], -1, axis=1)-tiles[:, :4, :2],axis=2
    )))
    a, b, depths = _collision_depths(tiles)
    active = depths > max(scale, 1e-10)*1e-8
    a, b, depths = a[active], b[active], depths[active]
    np.add.at(strength, a, depths**2)
    np.add.at(strength, b, depths**2)
    np.add.at(count, a, 1)
    np.add.at(count, b, 1)
    return dict(
        strength=strength,
        count=count,
        pairs=np.column_stack((a, b)).astype(np.int32),
        depth=depths,
        pair_count=int(len(depths)),
        max_depth=float(np.max(depths, initial=0.)),
        energy=float(np.sum(depths**2)),
    )


def hinge_gap_segments(vertices, hinges, *, max_segments: int = 100) -> dict:
    """Return the largest joint gaps, with real endpoints (not fake contact loci)."""
    tiles = np.asarray(vertices, dtype=float)
    pairs = np.asarray(hinges, dtype=np.int64).reshape(-1, 4)
    if tiles.ndim != 3 or tiles.shape[1:] != (8, 3) or not len(pairs):
        return {"start": np.empty((0, 3)), "end": np.empty((0, 3)),
                "gap": np.zeros(0), "index": np.zeros(0, dtype=np.int32)}
    a, ca, b, cb = pairs.T
    valid = ((a >= 0) & (a < len(tiles)) & (b >= 0) & (b < len(tiles))
             & (ca >= 0) & (ca < 8) & (cb >= 0) & (cb < 8))
    if not np.any(valid):
        return {"start": np.empty((0, 3)), "end": np.empty((0, 3)),
                "gap": np.zeros(0), "index": np.zeros(0, dtype=np.int32)}
    indices = np.flatnonzero(valid)
    start = tiles[a[valid], ca[valid]]
    end = tiles[b[valid], cb[valid]]
    errors = np.linalg.norm(start-end, axis=1)
    chosen = np.argsort(errors)[::-1][:max(0,int(max_segments))]
    return dict(start=start[chosen], end=end[chosen], gap=errors[chosen],
                index=indices[chosen].astype(np.int32))


def rejection_comparison(before: dict, after: dict) -> dict:
    """Reproduce threshold-relevant facts without modifying accept/reject."""
    e0, e1 = before.get("collision_energy"), after.get("collision_energy")
    p0, p1 = before.get("collision_pairs"), after.get("collision_pairs")
    d0, d1 = before.get("penetration_max"), after.get("penetration_max")
    h0, h1 = before.get("hinge_rms"), after.get("hinge_rms")
    improvement = (
        100.*(float(e0)-float(e1))/float(e0)
        if isinstance(e0,(int,float)) and isinstance(e1,(int,float)) and e0>0
        else None
    )
    allowed = max(2,int(np.ceil(.02*int(p0)))) if isinstance(p0,int) and p0>=0 else None
    tile_scale=float(before.get("tile_scale",1.))
    hinge_ceiling=(
        max(3.*float(h0), tile_scale*.1)
        if isinstance(h0,(int,float)) and h0>=0 else None
    )
    return dict(
        collision_delta=(int(p1)-int(p0) if isinstance(p0,int) and isinstance(p1,int) else None),
        penetration_energy_improvement_percent=improvement,
        max_penetration_delta=(float(d1)-float(d0) if isinstance(d0,(int,float))
                               and isinstance(d1,(int,float)) else None),
        hinge_rms_delta=(float(h1)-float(h0) if isinstance(h0,(int,float))
                         and isinstance(h1,(int,float)) else None),
        extra_pairs_allowance=allowed,
        hinge_rms_rejection_threshold=hinge_ceiling,
    )
