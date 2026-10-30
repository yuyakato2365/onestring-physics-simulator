"""Hinge-gap-driven Split (project extension, 2026-10-29).

The same policy as the collision-driven Split, but triggered by T2D hinges that
stay open instead of by tile collisions:

1. build the design and measure, for every hinge, the distance between its two
   joint points in the final Dual-Hinge T2D (in panel edge lengths);
2. if hinges stay open beyond ``hinge_gap_split_threshold``, propose cuts: a
   cut is a straight chain of M2D edges that runs through the worst open hinge
   and continues to the nearest Omega boundary (a slit / dart);
3. rebuild the complete design with that cut for the best few proposals and
   keep a cut only when it improves the open-hinge measure by at least
   ``hinge_gap_split_min_relative_improvement``; repeat up to
   ``hinge_gap_split_max_additional`` cuts.

A cut removes the hinges across the slit in K2D and T2D only.  In 3D the two
sides of the slit stay closed: K3D is solved on the uncut M3D and the copies of
every slit vertex share that position; T3D receives them as equality groups.
Acceptance: the open-hinge energy sum(max(gap - 0.1, 0)^2) drops by the minimum
relative improvement and the count over the threshold does not grow.  A cut is stored as
the pairs of M2D faces it separates; faces are never
renumbered, so cuts combine and survive rebuilds.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- topology

def cut_quad_mesh(faces: np.ndarray, n_vertices: int, cut_pairs) -> tuple[np.ndarray, np.ndarray]:
    """Split vertices so that the given face pairs are no longer edge-adjacent.

    Returns (new_faces, origin) where origin[k] is the original vertex of new
    vertex k (identity for the first ``n_vertices``).  A vertex whose incident
    faces stay connected around it (a slit tip) is not duplicated.
    """
    faces = np.asarray(faces, int)
    cut = {frozenset(map(int, p)) for p in cut_pairs}
    vf: dict[int, list[int]] = {}
    for fi, f in enumerate(faces):
        for v in f:
            vf.setdefault(int(v), []).append(fi)
    new_faces = faces.copy()
    origin = list(range(int(n_vertices)))
    for v, fs in vf.items():
        if len(fs) < 2:
            continue
        parent = {f: f for f in fs}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        nbr: dict[int, set[int]] = {}
        for f in fs:
            q = list(map(int, faces[f]))
            k = q.index(v)
            nbr[f] = {q[(k + 1) % 4], q[(k - 1) % 4]}
        for i, f in enumerate(fs):
            for g in fs[i + 1:]:
                if nbr[f] & nbr[g] and frozenset((f, g)) not in cut:
                    parent[find(f)] = find(g)
        groups: dict[int, list[int]] = {}
        for f in fs:
            groups.setdefault(find(f), []).append(f)
        if len(groups) < 2:
            continue
        ordered = sorted(groups.values(), key=lambda g: min(g))
        for g in ordered[1:]:
            new_id = len(origin)
            origin.append(v)
            for f in g:
                row = new_faces[f]
                row[row == v] = new_id
    return new_faces, np.asarray(origin, int)


def _edge_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(faces):
        for k in range(4):
            a, b = int(f[k]), int(f[(k + 1) % 4])
            out.setdefault((min(a, b), max(a, b)), []).append(fi)
    return out


def _apply_to_mesh(mesh: Any, faces: np.ndarray, origin: np.ndarray, n0: int) -> None:
    """In-place: duplicate vertex rows by ``origin`` and install the cut faces."""
    mesh.vertices = np.asarray(mesh.vertices, float)[origin].copy()
    mesh.faces = np.asarray(faces, int).copy()
    for name in ("_split_panel_source_vertices", "_seam_original_m2d_uv"):
        arr = getattr(mesh, name, None)
        if isinstance(arr, np.ndarray) and len(arr) == n0:
            setattr(mesh, name, arr[origin].copy())
    comps = getattr(mesh, "_split_panel_vertex_components", None)
    if isinstance(comps, list) and comps:
        lookup = {}
        for ci, comp in enumerate(comps):
            for v in np.asarray(comp, int).ravel():
                lookup[int(v)] = ci
        extra = [[] for _ in comps]
        for k in range(n0, len(origin)):
            ci = lookup.get(int(origin[k]))
            if ci is not None:
                extra[ci].append(k)
        mesh._split_panel_vertex_components = [
            np.concatenate([np.asarray(c, int).ravel(), np.asarray(e, int)]) if e else c
            for c, e in zip(comps, extra)]


def _equality_groups(origin: np.ndarray) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for k, v in enumerate(origin):
        if k != v:
            groups.setdefault(int(v), [int(v)]).append(int(k))
    return list(groups.values())


# --------------------------------------------------------------------------- pipeline hooks

def install_hinge_gap_split(pipeline: Any) -> None:
    """Outermost M3D-lift / K3D wrappers; inactive unless params carry cuts."""
    if getattr(pipeline, "_hinge_gap_split_installed", False):
        return
    old_lift = pipeline._lift_m2d_to_m3d
    old_k3d = pipeline._optimize_k3d

    def lift(target, mesh, parameterization, params):
        out, report = old_lift(target, mesh, parameterization, params)
        out._hinge_cut_m2d = mesh          # K2D/T2D read this object later
        return out, report

    def optimize(target, mesh, parameterization, params):
        cuts = [tuple(map(int, p)) for cut in (getattr(params, "hinge_gap_split_cuts", ()) or ()) for p in cut]
        out, report = old_k3d(target, mesh, parameterization, params)
        if not cuts:
            return out, report
        m2d = getattr(mesh, "_hinge_cut_m2d", None)
        if m2d is None:
            raise RuntimeError("HINGE_GAP_SPLIT: the M2D object of this M3D is unknown")
        n0 = len(mesh.vertices)
        faces, origin = cut_quad_mesh(np.asarray(mesh.faces, int), n0, cuts)
        groups = _equality_groups(origin)
        # K3D was solved on the uncut M3D: every copy takes its vertex's position.
        _apply_to_mesh(out, faces, origin, n0)
        _apply_to_mesh(mesh, faces, origin, n0)
        uv = np.asarray(m2d.vertices, float)[origin]
        m2d.vertices = uv.copy()
        m2d.faces = faces.copy()
        grid = getattr(m2d, "grid", None)
        if grid is not None and hasattr(grid, "vertex_positions"):
            grid.__init__(uv[:, :2], faces, grid.tile_size, grid.gap_size)   # same object: active_grid
            for obj in (out, mesh):
                if getattr(obj, "grid", None) is not None and obj.grid is not grid:
                    obj.grid = grid
        prev = [list(map(int, g)) for g in out.metrics.get("seam_exact_vertex_groups", []) or []]
        out.metrics["seam_exact_vertex_groups"] = prev + groups
        out.metrics["hinge_gap_split_cuts"] = [[list(map(int, p)) for p in cut]
                                               for cut in (getattr(params, "hinge_gap_split_cuts", ()) or ())]
        out.metrics.update({
            "hinge_gap_split_cut_count": len(getattr(params, "hinge_gap_split_cuts", ()) or ()),
            "hinge_gap_split_cut_edges": len(cuts),
            "hinge_gap_split_duplicated_vertices": int(len(origin) - n0),
            "hinge_gap_split_model": "K3D on the uncut M3D; slit copies share positions; hinges removed in K2D/T2D",
        })
        print(f"[HINGE-GAP-SPLIT] applied {len(cuts)} cut edges in "
              f"{out.metrics['hinge_gap_split_cut_count']} cut(s): {len(origin) - n0} vertices duplicated", flush=True)
        return out, report

    pipeline._lift_m2d_to_m3d = lift
    pipeline._optimize_k3d = optimize
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._lift_m2d_to_m3d = lift
        original._optimize_k3d = optimize
    for fn in (getattr(pipeline, "build_onestring_design", None),
               getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
               getattr(original, "build_onestring_design", None) if original is not None else None):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_lift_m2d_to_m3d"] = lift
            glb["_optimize_k3d"] = optimize
    pipeline._hinge_gap_split_installed = True


# --------------------------------------------------------------------------- measure / propose / accept

def hinge_gaps(state: Any, h_panel: float) -> dict[tuple[int, int], float]:
    """Open distance of every T2D hinge, in panel edge lengths, keyed by its face pair."""
    hv = np.asarray(state.tiles_2d_dual_hinge.vertices, float)
    out = {}
    for hg in state.hinge_graph.hinges:
        g = float(np.linalg.norm(hv[hg.tile_a, hg.local_vertex_a] - hv[hg.tile_b, hg.local_vertex_b])) / h_panel
        key = (min(hg.tile_a, hg.tile_b), max(hg.tile_a, hg.tile_b))
        out[key] = max(out.get(key, 0.0), g)
    return out


def hinge_quality(state: Any, h_panel: float, threshold: float) -> dict:
    g = np.array(list(hinge_gaps(state, h_panel).values()))
    if not len(g):
        return {"status": "no_hinges"}
    top = np.sort(g)[-max(1, int(round(0.05 * len(g)))):]
    m = state.tiles_2d_dual_hinge.metrics
    return {"status": "ok", "hinges": int(len(g)), "max": float(g.max()), "tail_mean_5pct": float(top.mean()),
            "open_energy": float(np.sum(np.maximum(g - 0.1, 0.0) ** 2)),
            "over_threshold": int(np.sum(g > threshold)), "over_quarter": int(np.sum(g > 0.25)),
            "mean": float(g.mean()), "EHinge": float(m.get("EHinge", np.nan))}


def _straight_chain(faces: np.ndarray, start_edge: tuple[int, int], boundary: set[int],
                    valence: np.ndarray, max_len: int = 80):
    """Edges of the straight M2D edge line from ``start_edge`` through its second vertex to the boundary."""
    a, b = start_edge
    chain = [(a, b)]
    vf: dict[int, list[int]] = {}
    for fi, f in enumerate(faces):
        for v in f:
            vf.setdefault(int(v), []).append(fi)
    while b not in boundary:
        if valence[b] != 4 or len(chain) > max_len:
            return None                    # cone or runaway: not a straight cut
        # the edge at b that shares no face with (a, b) is the straight continuation
        faces_ab = {fi for fi in vf[b] if a in faces[fi]}
        nxt = None
        for fi in vf[b]:
            if fi in faces_ab:
                continue
            q = list(map(int, faces[fi]))
            k = q.index(b)
            for c in (q[(k + 1) % 4], q[(k - 1) % 4]):
                if not any(c in faces[fj] for fj in faces_ab):
                    nxt = c
        if nxt is None:
            return None
        chain.append((b, nxt))
        a, b = b, nxt
    return chain


def rank_hinge_gap_cuts(state: Any, h_panel: float, *, threshold: float, existing=(), max_candidates: int = 6,
                        length_weight: float = 0.04, seeds: int = 4):
    """Slit proposals through the worst open hinges, ranked by removed open length minus a length cost."""
    m2d = state.mesh_2d_initial
    faces = np.asarray(m2d.faces, int)
    ef = _edge_faces(faces)
    boundary = {v for e, fs in ef.items() if len(fs) == 1 for v in e}
    valence = np.zeros(int(faces.max()) + 1, int)
    for e in ef:
        valence[e[0]] += 1
        valence[e[1]] += 1
    gaps = hinge_gaps(state, h_panel)
    cut_already = {frozenset(p) for cut in existing for p in cut}
    worst = sorted(gaps.items(), key=lambda kv: -kv[1])
    props, seen = [], set()
    used_seeds = 0
    for (fa, fb), g in worst:
        if g <= threshold or used_seeds >= seeds:
            break
        shared = [v for v in faces[fa] if v in faces[fb]]
        if len(shared) != 2:
            continue
        used_seeds += 1
        u, w = map(int, shared)
        for start in ((u, w), (w, u)):
            chain = _straight_chain(faces, start, boundary, valence)
            if chain is None:
                continue
            pairs = []
            for e in chain:
                fs = ef.get((min(e), max(e)), [])
                if len(fs) == 2:
                    pairs.append((min(fs), max(fs)))
            if not pairs or any(frozenset(p) in cut_already for p in pairs):
                continue
            key = frozenset(pairs)
            if key in seen:
                continue
            seen.add(key)
            removed = sum(max(gaps.get(p, 0.0) - 0.1, 0.0) for p in pairs)
            props.append({"pairs": [tuple(map(int, p)) for p in pairs], "length_edges": len(pairs),
                          "seed_gap": float(g), "removed_open": float(removed),
                          "score": float(removed - length_weight * len(pairs))})
    props.sort(key=lambda p: -p["score"])
    return props[:max_candidates]


def build_with_hinge_gap_split(build: Any, target: Any, params: Any, *, h_panel: float, state: Any = None,
                               log: Any = print, on_state: Any = None):
    """Collision-Split-style feedback loop driven by open T2D hinges."""
    threshold = float(getattr(params, "hinge_gap_split_threshold", 0.5))
    limit = int(getattr(params, "hinge_gap_split_max_additional", 3))
    attempts = int(getattr(params, "hinge_gap_split_candidate_attempts", 2))
    minimum = float(getattr(params, "hinge_gap_split_min_relative_improvement", 0.03))
    accepted = list(getattr(params, "hinge_gap_split_cuts", ()) or ())
    if state is None:
        state = build(target, params)
    steps = []
    status = "searching"
    for k in range(limit):
        before = hinge_quality(state, h_panel, threshold)
        if before["status"] != "ok" or before["over_threshold"] == 0:
            status = "no_hinge_over_threshold"
            break
        cands = rank_hinge_gap_cuts(state, h_panel, threshold=threshold, existing=accepted,
                                    max_candidates=max(attempts, 1))
        if not cands:
            status = "no_straight_cut_to_boundary"
            break
        best = None
        for cand in cands[:attempts]:
            trial_params = copy.copy(params)
            trial_params.hinge_gap_split_cuts = tuple(accepted + [tuple(cand["pairs"])])
            try:
                trial = build(target, trial_params)
                after = hinge_quality(trial, h_panel, threshold)
                # accept like the collision Split: the open-hinge energy sum(max(gap - 0.1, 0)^2) must drop by
                # at least ``minimum`` and no more hinges may stay open beyond the threshold
                ok = (after["status"] == "ok" and after["open_energy"] <= (1 - minimum) * before["open_energy"]
                      and after["over_threshold"] <= before["over_threshold"])
                reason = (f"E_open {before['open_energy']:.3f} -> {after['open_energy']:.3f}, "
                          f">thr {before['over_threshold']} -> {after['over_threshold']}, "
                          f"EHinge {before['EHinge']:.3f} -> {after['EHinge']:.3f}")
            except Exception as exc:  # noqa: BLE001 - a failed rebuild only rejects this cut
                trial, after, ok, reason = None, {"status": "build_failed"}, False, f"{type(exc).__name__}: {exc}"[:300]
            steps.append({"cut_number": k + 1, "proposal": cand, "before": before, "after": after,
                          "accepted": bool(ok), "reason": reason})
            log(f"[HINGE-GAP-SPLIT] cut {k + 1} candidate len={cand['length_edges']} seed={cand['seed_gap']:.2f}: "
                f"{'ACCEPT' if ok else 'reject'} ({reason})")
            if on_state is not None and trial is not None:
                on_state(trial, k + 1, len(steps), bool(ok))
            if ok and (best is None or after["open_energy"] < best[1]["open_energy"]):
                best = (cand, after, trial)
        if best is None:
            status = "no_improving_cut"
            break
        accepted.append(tuple(best[0]["pairs"]))
        state = best[2]
        status = "limit_reached" if k == limit - 1 else "searching"
    state.hinge_gap_split_report = {"status": status, "accepted_cuts": accepted, "steps": steps,
                                    "threshold": threshold, "min_relative_improvement": minimum}
    return state


__all__ = ["cut_quad_mesh", "install_hinge_gap_split", "hinge_gaps", "hinge_quality", "rank_hinge_gap_cuts",
           "build_with_hinge_gap_split"]
