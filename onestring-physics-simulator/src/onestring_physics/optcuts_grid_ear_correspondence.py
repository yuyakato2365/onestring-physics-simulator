"""Necessary endpoint feasibility checks for strict Grid-boundary UV charts.

If ALL three vertices of a UV triangle are constrained boundary vertices,
its final area is determined solely by the boundary correspondence. Interior
optimization can never untangle it. This is not a sufficient global
bijectivity criterion: final overlap, harmonic and distortion checks remain.

When an arbitrary arclength/DP corner assignment collapses a boundary ear,
move an existing target grid corner's *source* anchor to the middle boundary
vertex of that ear, preserving order, exact 50% Grid footprint and the full
target outline. Never change seam topology, triangulation, or final guard.
"""
from __future__ import annotations

import numpy as np


def constrained_boundary_faces(uv_faces, boundary_loop, vertex_count=None):
    """Return (face IDs, ordered-local source boundary corner triples)."""
    f = np.asarray(uv_faces, int)
    boundary = np.asarray(boundary_loop, int)
    if vertex_count is None:
        vertex_count = max(int(np.max(f))+1, int(np.max(boundary))+1)
    lookup = np.full(vertex_count, -1, dtype=int)
    lookup[boundary] = np.arange(len(boundary), dtype=int)
    face_local = lookup[f]
    picked = np.flatnonzero(np.all(face_local >= 0, axis=1))
    return picked, face_local[picked]


def signed_area2(triangle_points):
    t = np.asarray(triangle_points, float)
    a = t[..., 1, :] - t[..., 0, :]
    b = t[..., 2, :] - t[..., 0, :]
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def fixed_boundary_ear_report(
    source_uv, uv_faces, boundary_loop, target_boundary, *, ratio_min=.005
):
    """Proof of failure for a fixed boundary mapping when any constrained
    triangle has signed area below the hard acceptance threshold.
    Other triangles remain free, so zero violations is NOT proof of success.
    """
    source = np.asarray(source_uv, float)
    loop = np.asarray(boundary_loop, int)
    targets = np.asarray(target_boundary, float)
    if targets.shape != source[loop].shape:
        raise ValueError("GRID_EAR_BAD_BOUNDARY_SHAPE")
    face_ids, local = constrained_boundary_faces(
        uv_faces, loop, len(source))
    if not len(face_ids):
        return {"face_ids": face_ids, "ratios": np.empty(0),
                "bad_ids": np.empty(0, int),
                "bad_local_triples": np.empty((0,3), int),
                "minimum_fixed_ratio": None}
    base = signed_area2(source[np.asarray(uv_faces, int)[face_ids]])
    if np.any(~np.isfinite(base)) or np.any(np.abs(base) < 1e-18):
        raise ValueError("GRID_EAR_SOURCE_DEGENERATE")
    goal = signed_area2(targets[local])
    ratios = goal/base
    bad = (~np.isfinite(ratios)) | (ratios < float(ratio_min))
    return {
        "face_ids": face_ids,
        "ratios": ratios,
        "bad_ids": face_ids[bad],
        "bad_local_triples": local[bad],
        "minimum_fixed_ratio": float(np.min(ratios)),
    }


def map_boundary_with_corner_indices(old, old_edge_lengths, corners, indices):
    old = np.asarray(old, float)
    n = len(old)
    corners = np.asarray(corners, float)
    inds = [int(i) for i in indices]
    if len(inds) != len(corners) or inds[0] != 0:
        raise ValueError("GRID_EAR_BAD_CORNER_ASSIGNMENT")
    if any(b <= a for a,b in zip(inds,inds[1:])) or inds[-1]>=n:
        raise ValueError("GRID_EAR_NONMONOTONE_ASSIGNMENT")
    mapped = np.empty_like(old)
    for j in range(len(inds)):
        start = inds[j]
        end = inds[j+1] if j+1<len(inds) else n
        segment = np.asarray(old_edge_lengths[start:end], float)
        fraction = np.r_[0., np.cumsum(segment[:-1])] / max(
            float(segment.sum()),1e-12)
        mapped[start:end] = (corners[j] + fraction[:,None] *
                             (corners[(j+1)%len(corners)]-corners[j]))
    return mapped


def repair_boundary_ears(
    source_uv, uv_faces, boundary_loop, corners, indices,
    old_edge_lengths, *, max_edits=6, ratio_min=.005
):
    """Greedy local search over distinct, order-preserving corner anchors.

    This is a *candidate generator*, not a guarantee of feasible UV. Prefer
    the fewest unrepaired boundary ears and then smallest UV displacement.
    Each edit must strictly decrease the number of impossible triangles.
    """
    boundary = np.asarray(boundary_loop,int)
    old = np.asarray(source_uv,float)[boundary]
    inds = [int(i) for i in indices]
    target = map_boundary_with_corner_indices(
        old, old_edge_lengths, corners, inds)
    current = fixed_boundary_ear_report(
        source_uv, uv_faces, boundary, target, ratio_min=ratio_min)
    edits = []
    original_count = len(current["bad_ids"])
    if original_count == 0:
        return target, inds, edits, current
    for _ in range(min(max_edits,original_count)):
        best = None
        for failed_face, local in zip(current["bad_ids"],
                                     current["bad_local_triples"]):
            source_order = np.sort(local)
            # To map a 3-boundary-vertex ear to positive area, at least
            # one interior boundary vertex must land on a turning corner.
            # Wrap-around ears crossing the fixed phase anchor 0 need a
            # separate cyclic phase; never shift the fixed first vertex.
            if source_order[-1]-source_order[0] >= len(boundary)-2:
                continue
            mid = int(source_order[1])
            nearby_corners = np.argsort(np.abs(
                np.asarray(inds[1:]) - mid
            ),kind="stable")[:5] + 1
            for corner_id in nearby_corners:
                corner_id = int(corner_id)
                if not (inds[corner_id-1] < mid and
                        (corner_id+1==len(inds)
                         or mid < inds[corner_id+1])):
                    continue
                if mid == inds[corner_id]:
                    continue
                revised = inds.copy()
                revised[corner_id] = mid
                candidate = map_boundary_with_corner_indices(
                    old, old_edge_lengths, corners, revised)
                audit = fixed_boundary_ear_report(
                    source_uv, uv_faces, boundary, candidate,
                    ratio_min=ratio_min)
                count = len(audit["bad_ids"])
                if count >= len(current["bad_ids"]):
                    continue
                difference = candidate-old
                disp = np.sum(difference**2,axis=1)
                score = (
                    count, float(np.mean(disp)+.15*np.max(disp)),
                    abs(mid-inds[corner_id]), corner_id
                )
                if best is None or score < best[0]:
                    best = (score, candidate, revised, audit,
                            {"face_id":int(failed_face),
                             "corner_id":corner_id,
                             "old_boundary_vertex_index":int(inds[corner_id]),
                             "new_boundary_vertex_index":mid})
        if best is None:
            break
        _, target, inds, current, edit = best
        edits.append(edit)
        if not len(current["bad_ids"]):
            break
    return target, inds, edits, current


class FixedBoundaryEarCollapseError(ValueError):
    """A fixed boundary correspondence *forces* an invalid UV triangle.

    The exception carries constrained original face IDs and area ratios;
    it must be intercepted by a recovery search before expensive fitting.
    """


def require_noncollapsed_boundary_ears(
    source_uv, uv_faces, boundary_loop, target_boundary, *, ratio_min=.005
):
    """Validate a necessary geometric constraint, or request recovery."""
    audit = fixed_boundary_ear_report(
        source_uv, uv_faces, boundary_loop, target_boundary,
        ratio_min=ratio_min,
    )
    if len(audit["bad_ids"]):
        exc = FixedBoundaryEarCollapseError(
            "GRID_EAR_COLLAPSE_RECOVERY_NEEDED: "
            f"fully_constrained_bad_faces={audit['bad_ids'].tolist()}; "
            f"minimum_fixed_area_ratio={audit['minimum_fixed_ratio']:.8g}; "
            "move Grid-corner anchors; never relax the 0.005 area guard."
        )
        exc.face_ids = audit["bad_ids"].tolist()
        exc.area_ratios = audit["ratios"].tolist()
        raise exc
    return audit


def recover_boundary_ear_correspondences(
    source_uv, uv_faces, boundary_loop, corners, indices,
    old_edge_lengths, *, max_depth=3, beam_width=8,
    max_evaluations=64, max_solutions=3, ratio_min=.005
):
    """Automatic *exception recovery*: reposition pinned Grid-corner anchors.

    First try the known monotone local repair, then a bounded beam search
    which can cross intermediate states with unchanged numbers of collapsed
    boundary ears. Return several distinct endpoint-feasible mappings so
    the downstream bijective solver can retry if the first numerical solve
    fails. All candidates retain the exact target stair outline, original
    M2D cell mask, UV triangle connectivity and source/3D surface.

    A feasible endpoint boundary ear condition is NECESSARY, not sufficient
    for global bijectivity. No candidate bypasses final signed-area,
    distortion, overlap or inverse-map audits.
    """
    old = np.asarray(source_uv, float)[np.asarray(boundary_loop, int)]
    seed = tuple(int(v) for v in indices)
    n, m = len(old), len(corners)
    if len(seed) != m:
        raise ValueError("GRID_EAR_RECOVERY_BAD_ASSIGNMENT")
    solutions = []
    seen_solutions = set()
    # Prefer the cheap original local greedy correction. This is already
    # sufficient for many actual OptCuts Bunny collapsed-ear examples.
    first_map, first_ind, first_edits, first_report = repair_boundary_ears(
        source_uv, uv_faces, boundary_loop, corners, seed,
        old_edge_lengths, ratio_min=ratio_min,
    )

    def displacement(target):
        sq = np.sum((target-old)**2, axis=1)
        return float(np.mean(sq) + .15*np.max(sq))

    def add_solution(assignment, target, report, edits):
        identity = tuple(assignment)
        if identity in seen_solutions or len(report["bad_ids"]):
            return False
        seen_solutions.add(identity)
        solutions.append({
            "mapped": target,
            "indices": list(identity),
            "edits": edits,
            "audit": report,
            "score": displacement(target),
            "model": "ear_anchor_exception_recovery",
        })
        return True

    if len(first_report["bad_ids"]) == 0:
        add_solution(first_ind, first_map, first_report, first_edits)
        # Recover more than one eligible geometry: a first downstream
        # harmonic/untangling failure does not mean all reassignments fail.
        # Keep these *same* boundary corners and strict assignment order.
        pivot_corners = sorted(
            {int(edit["corner_id"]) for edit in first_edits}
        )
        for corner_id in pivot_corners:
            for offset in (-2, -1, 1, 2):
                revised = list(first_ind)
                shifted = revised[corner_id] + offset
                if not (revised[corner_id-1] < shifted and
                        (corner_id+1 == m
                         or shifted < revised[corner_id+1])
                        and 0 < shifted < n):
                    continue
                revised[corner_id] = shifted
                target = map_boundary_with_corner_indices(
                    old, old_edge_lengths, corners, revised)
                audit = fixed_boundary_ear_report(
                    source_uv, uv_faces, boundary_loop, target,
                    ratio_min=ratio_min)
                add_solution(revised, target, audit, first_edits+[{
                    "corner_id": corner_id,
                    "old_boundary_vertex_index": int(first_ind[corner_id]),
                    "new_boundary_vertex_index": int(shifted),
                    "reason": "numerical_retry_near_repaired_anchor",
                }])

    # Greedy repair may get stuck because one anchor shift leaves the same
    # number of invalid ears while enabling the next shift to fix both.
    # Search a small collection of such combinations before giving up.
    if len(first_report["bad_ids"]) and len(solutions) < max_solutions:
        queue = [(seed, [])]
        visited = {seed}
        evaluated = 0
        for _depth in range(max_depth):
            next_states = []
            for assignment, edits in queue:
                target = map_boundary_with_corner_indices(
                    old, old_edge_lengths, corners, assignment)
                current = fixed_boundary_ear_report(
                    source_uv, uv_faces, boundary_loop, target,
                    ratio_min=ratio_min)
                if not len(current["bad_ids"]):
                    add_solution(assignment, target, current, edits)
                    if len(solutions) >= max_solutions:
                        break
                    # Search alternatives near a solved assignment without
                    # turning a successful correction into a new constraint.
                    focuses = sorted({e["corner_id"] for e in edits
                                      if "corner_id" in e})
                    focuses = focuses[-2:]
                    focus_source = [
                        int(assignment[j]) for j in focuses
                    ]
                else:
                    focuses = []
                    focus_source = []
                    for loc in current["bad_local_triples"][:3]:
                        order = np.sort(loc)
                        # For wraps, the anchored phase is intrinsically
                        # unsuitable; other cyclic phases remain available.
                        if order[-1]-order[0] >= len(old)-2:
                            continue
                        focus_source.extend(
                            [int(order[1]), int(order[1])-1,
                             int(order[1])+1])
                for mid in sorted(set(focus_source)):
                    if not (0 < mid < n):
                        continue
                    nearest = np.argsort(
                        np.abs(np.asarray(assignment[1:])-mid),
                        kind="stable"
                    )[:6] + 1
                    for pos in nearest:
                        pos = int(pos)
                        for new_anchor in (mid-1, mid, mid+1):
                            if not (
                                0 < new_anchor < n
                                and assignment[pos-1] < new_anchor
                                and (pos+1 == m
                                     or new_anchor < assignment[pos+1])
                                and new_anchor != assignment[pos]
                            ):
                                continue
                            updated = list(assignment)
                            updated[pos] = new_anchor
                            updated = tuple(updated)
                            if updated in visited:
                                continue
                            visited.add(updated)
                            new_target = map_boundary_with_corner_indices(
                                old, old_edge_lengths, corners, updated)
                            report = fixed_boundary_ear_report(
                                source_uv, uv_faces, boundary_loop,
                                new_target, ratio_min=ratio_min)
                            evaluated += 1
                            edit = {
                                "corner_id": pos,
                                "old_boundary_vertex_index":
                                    int(assignment[pos]),
                                "new_boundary_vertex_index":
                                    int(new_anchor),
                                "reason": "bounded_exception_beam",
                            }
                            new_edits = edits+[edit]
                            if not len(report["bad_ids"]):
                                add_solution(updated, new_target,
                                             report, new_edits)
                                if len(solutions) >= max_solutions:
                                    break
                            next_states.append((
                                len(report["bad_ids"]),
                                displacement(new_target),
                                updated, new_edits
                            ))
                            if evaluated >= max_evaluations:
                                break
                        if (evaluated >= max_evaluations
                                or len(solutions) >= max_solutions):
                            break
                    if (evaluated >= max_evaluations
                            or len(solutions) >= max_solutions):
                        break
                if (evaluated >= max_evaluations
                        or len(solutions) >= max_solutions):
                    break
            if evaluated >= max_evaluations or len(solutions) >= max_solutions:
                break
            if not next_states:
                break
            next_states.sort(key=lambda item:(item[0],item[1],item[2]))
            queue=[(entry[2],entry[3])
                   for entry in next_states[:beam_width]]
    solutions.sort(key=lambda x: (x["score"], tuple(x["indices"])))
    return solutions[:max_solutions]
