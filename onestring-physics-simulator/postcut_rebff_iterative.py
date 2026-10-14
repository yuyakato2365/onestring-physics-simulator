"""Iterative post-cut independent official BFF research experiment.

Each step cuts exactly one violating disk patch along a grid-direction line,
recomputes its two new BFF charts, and rechecks ALL remaining pieces.
No smoothing, clipping of CSF extrema, or unreported fallbacks are allowed.
Not part of the normal OneString production split implementation.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import numpy as np

from onestring_physics.paper_mesh_splitting import gaussian_curvature
from scripts.test_lilium_postcut_bff_hypothesis import clip_piece, field


def candidate_lines(xyz, uv, faces, grid: int, *, max_candidates: int = 24):
    """Curvature-prioritized, interior full-grid cut lines on the CURRENT UV chart.

    Re-BFF changes the patch chart. Recompute line locations and curvature for
    each violating child rather than reusing the previous parent chart.
    """
    chart = SimpleNamespace(
        surface_vertices_3d=np.asarray(xyz, float),
        surface_faces=np.asarray(faces, int),
        uv_vertices_2d=np.asarray(uv, float),
        uv_faces=np.asarray(faces, int),
    )
    curvature = gaussian_curvature(chart)
    viable = np.flatnonzero(np.isfinite(curvature))
    if not len(viable):
        return []
    order = viable[np.argsort(-curvature[viable], kind="stable")]
    bounds_min, bounds_max = np.min(uv, axis=0), np.max(uv, axis=0)
    spacing = max(float(np.max(bounds_max-bounds_min))/max(grid, 2), 1e-10)
    by_axis = {}
    for axis, dim in (("row", 1), ("col", 0)):
        # Local candidate grid anchored to the current chart bounding box.
        low = float(bounds_min[dim])
        high = float(bounds_max[dim])
        coords = low + spacing*np.arange(1, int(np.floor((high-low)/spacing))+1)
        coords = [float(v) for v in coords if low + 1e-8 < v < high - 1e-8]
        by_axis[axis] = coords

    proposals = {}
    for v in order[:min(len(order), 24)]:
        for axis, dim in (("row", 1), ("col", 0)):
            coords = by_axis[axis]
            if not coords:
                continue
            position = float(uv[v, dim])
            nearest = sorted(coords, key=lambda value: (abs(value-position), value))[:2]
            for coordinate in nearest:
                key = (axis, round(coordinate, 12))
                candidate = dict(
                    axis=axis, coordinate=coordinate,
                    curvature_vertex=int(v), curvature=float(curvature[v]),
                    peak_distance=float(abs(coordinate-position)),
                    proposal_method="reflattened_patch_curvature_guided_grid_cut",
                )
                priority = (-candidate["curvature"], candidate["peak_distance"],
                            0 if axis == "row" else 1, coordinate)
                if key not in proposals or priority < proposals[key][0]:
                    proposals[key] = priority, candidate
        if len(proposals) >= max_candidates:
            break
    return [entry[1] for entry in
            sorted(proposals.values(), key=lambda pair: pair[0])[:max_candidates]]


def _split_and_rebff(patch, axis, coordinate, rebff: Callable):
    dim = {"row": 1, "col": 0}[axis]
    children = []
    for positive in (False, True):
        xyz, uv_old, faces, topo = clip_piece(
            patch["xyz"], patch["new_uv"], patch["faces"],
            dim, float(coordinate), positive,
        )
        if not topo["disk_topology"] or len(faces) < 4:
            raise ValueError("Split would produce a nondisk or fewer than 4 triangles")
        new_uv, metadata = rebff(xyz, faces)
        new_uv = np.asarray(new_uv, float)
        if new_uv.shape != (len(xyz), 2) or not np.isfinite(new_uv).all():
            raise ValueError("Official BFF returned invalid UV dimensions or nonfinite values")
        prior = field(xyz, uv_old, faces)
        post = field(xyz, new_uv, faces)
        if not np.isfinite(post["linear_span"]):
            raise ValueError("Postcut CSF is nonfinite")
        children.append(dict(
            side=patch["side"] + ("_positive" if positive else "_negative"),
            xyz=xyz, faces=faces, original_uv=uv_old, new_uv=new_uv,
            pre=prior, post=post, bff_metadata=metadata,
        ))
    return children


def iterative_postcut_rebff(
    xyz, uv, faces, *,
    first_cut: dict, rebff: Callable, grid: int,
    csf_limit: float = 2.0, max_splits: int = 8,
    progress: Callable | None = None,
):
    """Return final patches, split log and honest convergence status.

    First cut is user-selected or from the existing OneString curvature split.
    Extra cuts occur ONLY if a final piece still violates the post-BFF CSF
    bound, and only within violating pieces. Accepted cuts are atomic: both
    children must have valid disk topology, BFF, and finite scale statistics.
    """
    if not (1.0 < float(csf_limit) < 100.0):
        raise ValueError("CSF limit must be between 1 and 100")
    if not (1 <= int(max_splits) <= 64):
        raise ValueError("max_splits must be between 1 and 64")
    source = dict(side="initial", xyz=np.asarray(xyz, float),
                  new_uv=np.asarray(uv, float), faces=np.asarray(faces, int))
    pending = [source]
    history = []
    rejected = []
    stop_reason = None

    def notify(message):
        if progress is not None:
            progress(message)

    while len(history) < max_splits:
        if not history:
            patch = source
            options = [dict(
                axis=first_cut["axis"],
                coordinate=float(first_cut["coordinate"]),
                curvature_vertex=first_cut.get("source_vertex"),
                curvature=first_cut.get("curvature"),
                proposal_method=first_cut.get("origin", "initial"),
            )]
        else:
            violating = [p for p in pending
                         if p["post"]["linear_span"] > csf_limit + 1e-10]
            if not violating:
                stop_reason = "all_patches_pass"
                break
            patch = max(violating, key=lambda p: p["post"]["linear_span"])
            options = candidate_lines(
                patch["xyz"], patch["new_uv"], patch["faces"], grid,
            )

        if not options:
            stop_reason = "no_curvature_guided_interior_grid_candidate"
            break

        accepted = None
        for cut in options:
            try:
                children = _split_and_rebff(
                    patch, cut["axis"], cut["coordinate"], rebff,
                )
                accepted = (cut, children)
                break
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(
                    parent=patch["side"], axis=cut["axis"],
                    coordinate=float(cut["coordinate"]),
                    reason=f"{type(exc).__name__}: {str(exc)[:280]}",
                ))
        if accepted is None:
            stop_reason = "all_candidate_cuts_failed"
            break

        cut, children = accepted
        pending = [p for p in pending if p is not patch] + children
        worst = max(p["post"]["linear_span"] for p in pending)
        history.append(dict(
            step=len(history)+1, parent=patch["side"],
            axis=cut["axis"], uv_coordinate=float(cut["coordinate"]),
            selection=cut.get("proposal_method"),
            curvature_vertex=cut.get("curvature_vertex"),
            gaussian_curvature=cut.get("curvature"),
            children=[dict(name=c["side"], triangles=int(len(c["faces"])),
                           source_chart_linear_span=c["pre"]["linear_span"],
                           rebff_linear_span=c["post"]["linear_span"])
                      for c in children],
            all_patches_max_linear_span=float(worst),
            violating_patch_count=sum(p["post"]["linear_span"] >
                                      csf_limit + 1e-10 for p in pending),
        ))
        notify("Split %d/%d → %s; max CSF=%.4f, violating=%d"
               % (len(history), max_splits, patch["side"], worst,
                  history[-1]["violating_patch_count"]))
        if worst <= csf_limit + 1e-10:
            stop_reason = "all_patches_pass"
            break

    final_worst = max(p["post"]["linear_span"] for p in pending) if history else None
    passes = bool(history and final_worst <= csf_limit + 1e-10)
    if stop_reason is None:
        stop_reason = "all_patches_pass" if passes else "max_split_budget_exhausted"
    return dict(
        patches=pending, split_history=history,
        rejected_candidates=rejected, stop_reason=stop_reason,
        split_count=len(history),
        max_linear_span=final_worst, all_pass=passes,
        csf_limit=float(csf_limit),
        max_splits=int(max_splits),
    )
