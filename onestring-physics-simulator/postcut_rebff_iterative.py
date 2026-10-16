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


def candidate_lines(xyz, uv, faces, grid: int, *, max_candidates: int = 24,
                    fixed_spacing=None, fixed_origin=None):
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
    spacing = (float(fixed_spacing) if fixed_spacing is not None else
               max(float(np.max(bounds_max-bounds_min))/max(grid, 2), 1e-10))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Split grid spacing must be positive and finite")
    origin = (np.asarray(fixed_origin,float).reshape(2)
              if fixed_origin is not None else np.zeros(2,float))
    if not np.all(np.isfinite(origin)):
        raise ValueError("Split grid origin must be finite")
    by_axis = {}
    for axis, dim in (("row", 1), ("col", 0)):
        # Match the per-chart M2D lattice origin at 0, instead of shifting
        # grid lines to an arbitrary UV bounding-box minimum after every BFF.
        low = float(bounds_min[dim])
        high = float(bounds_max[dim])
        start = int(np.floor((low-origin[dim]) / spacing)) + 1
        stop = int(np.ceil((high-origin[dim]) / spacing)) - 1
        by_axis[axis] = [
            float(origin[dim] + k * spacing) for k in range(start, stop + 1)
            if low + 1e-8 < origin[dim]+k*spacing < high - 1e-8
        ]

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


def _cut_child_rectangular_corners(parent, child_xyz, child_uv, child_faces,
                                   axis, coordinate, positive):
    """Exactly the 4 boundary corners of a rectangle cut by an interior line.

    The *two seam endpoints* MUST be rectangle corners on BOTH newly created
    charts; the remaining corners are inherited from the parent's rectangle.
    Any fallback to arbitrary arclength quartiles invalidates common seam
    registration and is intentionally rejected.
    """
    from onestring_physics.reference_bff import validate_reference_mesh
    parent_uv=np.asarray(parent["new_uv"],float)
    child_uv=np.asarray(child_uv,float)
    child_xyz=np.asarray(child_xyz,float)
    parent_faces=np.asarray(parent["faces"],int)
    parent_loop=np.asarray(
        validate_reference_mesh(np.asarray(parent["xyz"],float),parent_faces)
        ["boundary_loops"][0],int)
    child_loop=np.asarray(
        validate_reference_mesh(child_xyz,np.asarray(child_faces,int))
        ["boundary_loops"][0],int)
    dim={"row":1,"col":0}[axis]
    other=1-dim
    span=float(np.max(np.ptp(parent_uv,axis=0)))
    tol=max(1e-8*span,1e-10)
    parent_min,parent_max=np.min(parent_uv,axis=0),np.max(parent_uv,axis=0)
    seam=child_loop[
        np.abs(child_uv[child_loop,dim]-float(coordinate))<=tol
    ]
    if len(seam)<2:
        raise ValueError("Grid Split does not intersect source boundary at two seam endpoints")
    endpoints=[int(seam[np.argmin(child_uv[seam,other])]),
               int(seam[np.argmax(child_uv[seam,other])])]
    if endpoints[0]==endpoints[1]:
        raise ValueError("Grid Split seam endpoints are coincident")
    original=[]
    for x in (parent_min[0],parent_max[0]):
        for y in (parent_min[1],parent_max[1]):
            target=np.array([x,y],float)
            dist=np.linalg.norm(parent_uv[parent_loop]-target,axis=1)
            if float(np.min(dist))>tol:
                raise ValueError("Parent BFF boundary is not a four-corner rectangle")
            pvid=int(parent_loop[np.argmin(dist)])
            pos=parent_uv[pvid]
            if (pos[dim]>=float(coordinate)-tol if positive
                    else pos[dim]<=float(coordinate)+tol):
                hits=np.linalg.norm(child_uv[child_loop]-pos,axis=1)
                if float(np.min(hits))>tol:
                    raise ValueError("Grid-cut child lost a parent rectangle corner")
                original.append(int(child_loop[np.argmin(hits)]))
    corners=endpoints+original
    if len(original)!=2 or len(set(corners))!=4:
        raise ValueError(
            f"Grid Split requires two old corners + two seam corners; "
            f"found {len(original)} old and {len(set(corners))} unique."
        )
    # Verify the seam is a single uninterrupted boundary arc. Both BFF charts
    # must use the two endpoints as adjacent rectangle corners.
    positions={int(v):i for i,v in enumerate(child_loop)}
    i,j=positions[endpoints[0]],positions[endpoints[1]]
    arc1=[int(child_loop[k%len(child_loop)])
          for k in range(i,j+1)] if i<=j else [
          int(child_loop[k%len(child_loop)])
          for k in range(i,j+len(child_loop)+1)]
    arc2=[int(child_loop[k%len(child_loop)])
          for k in range(j,i+1)] if j<=i else [
          int(child_loop[k%len(child_loop)])
          for k in range(j,i+len(child_loop)+1)]
    if not (
        all(abs(child_uv[v,dim]-coordinate)<=tol for v in arc1)
        or all(abs(child_uv[v,dim]-coordinate)<=tol for v in arc2)
    ):
        raise ValueError("Seam is not a full rectangular boundary side")
    return child_xyz[corners].copy()


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
        corners_xyz=None
        if getattr(rebff,"seam_corners_required",False):
            corners_xyz=_cut_child_rectangular_corners(
                patch,xyz,uv_old,faces,axis,float(coordinate),positive,
            )
            import inspect
            signature=inspect.signature(rebff)
            if "rectangle_corner_positions" not in signature.parameters:
                raise ValueError(
                    "Rectangular post-cut BFF must accept explicit seam corner positions; "
                    "silent arclength-quartile fallback is forbidden"
                )
            new_uv, metadata = rebff(
                xyz, faces, rectangle_corner_positions=corners_xyz
            )
        else:
            new_uv, metadata = rebff(xyz, faces)
        raw_bff_uv = np.asarray(new_uv, float)
        if raw_bff_uv.shape != (len(xyz), 2) or not np.isfinite(raw_bff_uv).all():
            raise ValueError("Official BFF returned invalid UV dimensions or nonfinite values")
        prior = field(xyz, uv_old, faces)
        raw_post = field(xyz, raw_bff_uv, faces)
        if corners_xyz is not None:
            # Independently computed conformal rectangles do NOT, in
            # general, share the same nonlinear seam parameterization.
            # Every later Split must instead cut the already-registered
            # global rectangular atlas. The raw official BFF remains a
            # separate recorded diagnostic, not the solver's UV geometry.
            from postcut_rebff_joint_atlas import seam_constrained_rectangular_atlas
            new_uv, joint_audit=seam_constrained_rectangular_atlas(
                xyz,faces,uv_old,raw_bff_uv
            )
        else:
            new_uv=raw_bff_uv
            joint_audit={}
        post = field(xyz, new_uv, faces)
        if not np.isfinite(post["linear_span"]):
            raise ValueError("Postcut joint-atlas CSF is nonfinite")
        children.append(dict(
            side=patch["side"] + ("_positive" if positive else "_negative"),
            xyz=xyz, faces=faces, original_uv=uv_old, new_uv=new_uv,
            independent_bff_uv=raw_bff_uv,
            independent_bff_post=raw_post,
            postcut_joint_atlas=bool(corners_xyz is not None),
            postcut_joint_atlas_metadata=joint_audit,
            pre=prior, post=post, bff_metadata=metadata,
            seam_rectangle_corners_xyz=corners_xyz,
            seam_rectangle_corners_constrained=bool(corners_xyz is not None),
        ))
    return children


def iterative_postcut_rebff(
    xyz, uv, faces, *,
    first_cut: dict | None, rebff: Callable, grid: int,
    csf_limit: float = 2.0, max_splits: int = 8,
    progress: Callable | None = None,
    initial_linear_span: float | None = None,
    initial_bff_metadata: dict | None = None,
):
    """Return final patches, split log and honest convergence status.

    Initially measure the complete BFF chart. If CSF <= the bound, return
    without any Split. Otherwise cut on the configured grid, then re-BFF
    BOTH children; repeat only on CSF-violating pieces. Every cut remains
    atomic and rejects malformed topologies and invalid BFF solutions.
    """
    if not (1.0 < float(csf_limit) < 100.0):
        raise ValueError("CSF limit must be between 1 and 100")
    if not (1 <= int(max_splits) <= 64):
        raise ValueError("max_splits must be between 1 and 64")
    xyz = np.asarray(xyz, float)
    uv = np.asarray(uv, float)
    faces = np.asarray(faces, int)
    initial = float(field(xyz, uv, faces)["linear_span"]) if initial_linear_span is None else float(initial_linear_span)
    if not np.isfinite(initial) or initial < 1.0 - 1e-9:
        raise ValueError(f"Invalid initial BFF CSF ratio: {initial}")
    source = dict(
        side="initial", xyz=xyz, new_uv=uv, original_uv=uv.copy(), faces=faces,
        pre={"linear_span": initial}, post={"linear_span": initial},
        bff_metadata=dict(initial_bff_metadata or {}),
    )
    pending = [source]
    history = []
    rejected = []
    # Memoize only FINAL failed regions. A later successful split of a
    # DIFFERENT piece does not make an unchanged region newly splittable.
    # Each branch path is unique in the binary cut tree.
    unsplittable_by_side = {}
    stop_reason = None
    if initial <= csf_limit + 1e-10:
        return dict(
            patches=pending, split_history=history,
            rejected_candidates=rejected, stop_reason="initial_bff_satisfies_limit",
            split_count=0, max_linear_span=float(initial), all_pass=True,
            initial_linear_span=float(initial), csf_limit=float(csf_limit),
            max_splits=int(max_splits),
            unsplittable_regions=[], unsplittable_region_count=0,
            violating_patch_count=0,
        )

    def notify(message):
        if progress is not None:
            progress(message)

    while len(history) < max_splits:
        violating = sorted(
            (p for p in pending
             if p["post"]["linear_span"] > csf_limit + 1e-10),
            key=lambda p: (-float(p["post"]["linear_span"]), p["side"]),
        )
        if not violating:
            stop_reason = "all_patches_pass"
            break

        # Crucial: do NOT stop all Split work merely because the currently
        # highest-CSF region occupies a single cell, has no interior full-grid
        # cut, or has only invalid BFF children. Try the next violating region.
        candidates = [
            p for p in violating if p["side"] not in unsplittable_by_side
        ]
        if not candidates:
            stop_reason = "no_splittable_violating_regions"
            break

        accepted = None
        for patch in candidates:
            options = candidate_lines(
                patch["xyz"], patch["new_uv"], patch["faces"], grid,
                fixed_spacing=getattr(rebff,"fixed_grid_spacing",None),
                fixed_origin=getattr(rebff,"fixed_grid_origin",None),
            )
            if not history and patch is source and first_cut is not None:
                preferred = dict(
                    axis=first_cut["axis"],
                    coordinate=float(first_cut["coordinate"]),
                    curvature_vertex=first_cut.get("source_vertex"),
                    curvature=first_cut.get("curvature"),
                    proposal_method=first_cut.get("origin", "initial_grid_cut"),
                )
                # Give the original paper-grid cut highest priority, but
                # do not label a region physically UNSPLITTABLE simply
                # because this one preferred attempt failed.
                options = [preferred] + [
                    cut for cut in options
                    if cut["axis"] != preferred["axis"]
                    or abs(float(cut["coordinate"])-preferred["coordinate"])>1e-10
                ]
            if not options:
                unsplittable_by_side[patch["side"]] = {
                    "side": patch["side"],
                    "linear_span": float(patch["post"]["linear_span"]),
                    "source_triangles": int(len(patch["faces"])),
                    "reason": "no_interior_full_grid_cut_candidate",
                    "attempted_candidates": 0,
                }
                notify(
                    "Unresolved: %s has CSF=%.4f but no interior grid cut. "
                    "Continuing other regions."
                    % (patch["side"], patch["post"]["linear_span"])
                )
                continue

            for cut in options:
                try:
                    children = _split_and_rebff(
                        patch, cut["axis"], cut["coordinate"], rebff,
                    )
                    # A successful BFF subprocess does not mean that the
                    # resulting *shared* atlas is a useful parameterization.
                    # In particular, fixed-seam harmonic registration can
                    # create almost-collapsed UV triangles, making Jmax/Jmin
                    # hundreds of times larger than on the parent. Never
                    # commit a geometrically regressive cut: keep searching
                    # other legal lines, including plateau (neutral) cuts.
                    # This is an explicit stability safeguard, not a claim
                    # about the unspecified author implementation.
                    parent_span = float(patch["post"]["linear_span"])
                    child_spans = [float(c["post"]["linear_span"]) for c in children]
                    if (
                        not child_spans
                        or any(not np.isfinite(v) or v < 1.0 - 1e-9
                               for v in child_spans)
                    ):
                        raise ValueError(
                            "Invalid postcut shared-atlas CSF: "
                            f"parent={parent_span:.6g}, children={child_spans}"
                        )
                    worst_child = max(child_spans)
                    # The guard targets the fixed-seam rectangular atlas.
                    # Keep the independent free-boundary research comparison
                    # unchanged; its BFF charts are not stitched together.
                    joined_rectangle = all(
                        bool(c.get("postcut_joint_atlas", False))
                        for c in children
                    )
                    if (joined_rectangle and
                        worst_child > parent_span + 1e-9 * max(1.0, parent_span)):
                        raw_spans = [
                            float(c.get("independent_bff_post", {}).get(
                                "linear_span", float("nan")
                            ))
                            for c in children
                        ]
                        raise ValueError(
                            "Postcut shared-atlas CSF regression: "
                            f"parent={parent_span:.6g}, "
                            f"children={[round(v, 6) for v in child_spans]}, "
                            f"raw_BFF={[round(v, 6) for v in raw_spans]}. "
                            "Cut rejected; trying next initial-grid line."
                        )
                    accepted = (patch, cut, children)
                    break
                except (ValueError, RuntimeError) as exc:
                    rejected.append(dict(
                        parent=patch["side"], axis=cut["axis"],
                        coordinate=float(cut["coordinate"]),
                        reason=f"{type(exc).__name__}: {str(exc)[:280]}",
                    ))
            if accepted is not None:
                break
            # All cut candidates for THIS region failed. Preserve the
            # region and its failing CSF, then try other violating regions.
            unsplittable_by_side[patch["side"]] = {
                "side": patch["side"],
                "linear_span": float(patch["post"]["linear_span"]),
                "source_triangles": int(len(patch["faces"])),
                "reason": "all_candidate_cuts_failed",
                "attempted_candidates": len(options),
            }
            notify(
                "Unresolved: %s has CSF=%.4f, all %d cuts rejected. "
                "Continuing other regions."
                % (patch["side"], patch["post"]["linear_span"], len(options))
            )

        if accepted is None:
            stop_reason = "no_splittable_violating_regions"
            break

        patch, cut, children = accepted
        before_worst = max(p["post"]["linear_span"] for p in pending)
        parent_csf = float(patch["post"]["linear_span"])
        pending = [p for p in pending if p is not patch] + children
        worst = max(p["post"]["linear_span"] for p in pending)
        history.append(dict(
            step=len(history)+1, parent=patch["side"],
            parent_linear_span=parent_csf,
            all_patches_max_csf_before=float(before_worst),
            all_patches_max_csf_improvement=float(before_worst - worst),
            csf_improved=bool(worst < before_worst - 1e-9),
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
            unsplittable_region_count=len(unsplittable_by_side),
        ))
        notify(
            "Split %d/%d -> %s; max CSF=%.4f, violating=%d, "
            "unsplittable=%d"
            % (len(history), max_splits, patch["side"], worst,
               history[-1]["violating_patch_count"],
               len(unsplittable_by_side))
        )
        if worst <= csf_limit + 1e-10:
            stop_reason = "all_patches_pass"
            break

    final_worst = max(p["post"]["linear_span"] for p in pending)
    passes = bool(final_worst <= csf_limit + 1e-10)
    unresolved = [
        unsplittable_by_side[p["side"]]
        for p in pending if p["side"] in unsplittable_by_side
    ]
    violating_count = sum(
        p["post"]["linear_span"] > csf_limit + 1e-10 for p in pending
    )
    if stop_reason is None:
        stop_reason = "all_patches_pass" if passes else "max_split_budget_exhausted"
    return dict(
        patches=pending, split_history=history,
        rejected_candidates=rejected, stop_reason=stop_reason,
        split_count=len(history),
        max_linear_span=final_worst, all_pass=passes,
        initial_linear_span=float(initial),
        csf_limit=float(csf_limit),
        max_splits=int(max_splits),
        unsplittable_regions=unresolved,
        unsplittable_region_count=len(unresolved),
        violating_patch_count=int(violating_count),
    )
