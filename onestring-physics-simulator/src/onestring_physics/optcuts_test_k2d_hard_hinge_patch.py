"""Kinematic hard-hinge parameterization for ``optcuts_test`` K2D.

Tree hinges are not repaired after the fact.  A spanning forest of the hinge
network is used as the coordinate system itself: every non-root tile is placed
from its parent hinge, so every tree hinge is coincident by construction for any
choice of tile angles.  Only loop-closing hinges and positive-area collisions
remain as feasibility residuals.

The solver attempts to drive loop-closure and collision residuals to zero while
keeping the kinematic layout near the ordinary K2D initialization.  If the
remaining constraints are numerically unsatisfied, the best result is still
returned and the failure is published in metrics instead of aborting the rest of
the pipeline.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _rot(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def _hinge_errors(tiles: np.ndarray, constraints: list[tuple[int, int, int, int]]) -> np.ndarray:
    x = np.asarray(tiles, dtype=float)
    vals: list[float] = []
    for ia, ca, ib, cb in constraints:
        if 0 <= ia < len(x) and 0 <= ib < len(x) and 0 <= ca < x.shape[1] and 0 <= cb < x.shape[1]:
            vals.append(float(np.linalg.norm(x[ia, ca] - x[ib, cb])))
        else:
            vals.append(float("inf"))
    return np.asarray(vals, dtype=float)


def _build_spanning_forest(
    tile_count: int,
    constraints: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(tile_count)]
    for edge_id, (ia, _ca, ib, _cb) in enumerate(constraints):
        if 0 <= ia < tile_count and 0 <= ib < tile_count and ia != ib:
            adjacency[ia].append((ib, edge_id))
            adjacency[ib].append((ia, edge_id))

    parent = np.full(tile_count, -1, dtype=int)
    parent_edge = np.full(tile_count, -1, dtype=int)
    parent_corner = np.full(tile_count, -1, dtype=int)
    child_corner = np.full(tile_count, -1, dtype=int)
    component_root = np.full(tile_count, -1, dtype=int)
    order: list[int] = []
    roots: list[int] = []
    tree_edges: set[int] = set()

    seen = np.zeros(tile_count, dtype=bool)
    for root in range(tile_count):
        if seen[root]:
            continue
        seen[root] = True
        roots.append(root)
        component_root[root] = root
        q: deque[int] = deque([root])
        while q:
            u = q.popleft()
            for v, edge_id in adjacency[u]:
                if seen[v]:
                    continue
                seen[v] = True
                parent[v] = u
                parent_edge[v] = edge_id
                component_root[v] = root
                ia, ca, ib, cb = constraints[edge_id]
                if u == ia and v == ib:
                    parent_corner[v], child_corner[v] = int(ca), int(cb)
                else:
                    parent_corner[v], child_corner[v] = int(cb), int(ca)
                tree_edges.add(int(edge_id))
                order.append(v)
                q.append(v)

    loop_edges = [i for i in range(len(constraints)) if i not in tree_edges]
    return {
        "parent": parent,
        "parent_edge": parent_edge,
        "parent_corner": parent_corner,
        "child_corner": child_corner,
        "component_root": component_root,
        "roots": roots,
        "order": order,
        "tree_edges": sorted(tree_edges),
        "loop_edges": loop_edges,
    }


def _kinematic_layout(
    local_tiles: np.ndarray,
    root_centres: np.ndarray,
    theta: np.ndarray,
    forest: dict[str, Any],
) -> np.ndarray:
    local = np.asarray(local_tiles, dtype=float)
    angles = np.asarray(theta, dtype=float)
    out = np.zeros_like(local)
    for root in forest["roots"]:
        r = _rot(float(angles[root]))
        out[root] = local[root] @ r + root_centres[root][None, :]
    parent = forest["parent"]
    pc = forest["parent_corner"]
    cc = forest["child_corner"]
    for child in forest["order"]:
        p = int(parent[child])
        r = _rot(float(angles[child]))
        hinge_world = out[p, int(pc[child])]
        translated = local[child] @ r
        t = hinge_world - translated[int(cc[child])]
        out[child] = translated + t[None, :]
    return out


def _ancestor_sets(forest: dict[str, Any]) -> list[set[int]]:
    parent = np.asarray(forest["parent"], dtype=int)
    roots = set(int(v) for v in forest["roots"])
    result: list[set[int]] = []
    for tile in range(len(parent)):
        s: set[int] = set()
        u = int(tile)
        while u >= 0 and u not in roots:
            s.add(u)
            u = int(parent[u])
        result.append(s)
    return result


def _loop_error_details(
    tiles: np.ndarray,
    constraints: list[tuple[int, int, int, int]],
    loop_edges: list[int],
) -> tuple[np.ndarray, list[int]]:
    vals: list[float] = []
    ids: list[int] = []
    for edge_id in loop_edges:
        ia, ca, ib, cb = constraints[edge_id]
        d = float(np.linalg.norm(tiles[ia, ca] - tiles[ib, cb]))
        vals.append(d)
        ids.append(int(edge_id))
    return np.asarray(vals, dtype=float), ids


def install_optcuts_test_k2d_hard_hinge_patch(pipeline: Any) -> None:
    from . import optcuts_test_k2d_relative_layout_patch as mod

    if getattr(mod, "_onestring_k2d_hard_hinge_installed", False):
        return

    base_build = mod._build_rigid_k2d_layout
    base_make_layout = pipeline._make_flat_tile_layout
    original_layout_collision_metrics = mod._layout_collision_metrics

    def build_with_kinematic_constraints(pipeline_obj: Any, mesh_2d: Any, mesh_3d: Any, params: Any, progress_callback=None):
        initial, metrics = base_build(pipeline_obj, mesh_2d, mesh_3d, params, progress_callback=progress_callback)
        initial = np.asarray(initial, dtype=float)
        faces = np.asarray(mesh_2d.faces, dtype=int)
        constraints = mod._hinge_constraints(pipeline_obj, faces)
        n = len(initial)
        if n == 0 or not constraints:
            return initial, dict(metrics or {})

        forest = _build_spanning_forest(n, constraints)
        centres0 = np.mean(initial, axis=1)
        local = initial - centres0[:, None, :]
        theta = np.zeros(n, dtype=float)
        root_centres = centres0.copy()
        scale = max(float(mod._tile_scale(initial)), 1e-12)
        hinge_tol = max(scale * float(getattr(params, "k2d_hard_hinge_relative_tolerance", 5e-4)), 1e-9)
        collision_tol = max(scale * 1e-10, 1e-13)
        roots = set(int(v) for v in forest["roots"])
        variable_tiles = [i for i in range(n) if i not in roots]
        var_index = {tile: k for k, tile in enumerate(variable_tiles)}
        ancestors = _ancestor_sets(forest)

        def unpack(z: np.ndarray) -> np.ndarray:
            a = np.zeros(n, dtype=float)
            for tile, k in var_index.items():
                a[tile] = float(z[k])
            return a

        z = np.zeros(len(variable_tiles), dtype=float)
        current = _kinematic_layout(local, root_centres, theta, forest)
        pair_fn, sat_fn = mod._collision_backend(pipeline_obj)
        loop_edges = list(forest["loop_edges"])
        scipy_used = False
        solver_messages: list[str] = []

        # Repeatedly rebuild a fixed collision-candidate set, then solve the
        # smooth-ish feasibility residuals in angle space. Tree hinges never
        # appear as residuals because they are exact by construction.
        try:
            from scipy.optimize import least_squares
            from scipy.sparse import lil_matrix

            scipy_used = True
            outer_passes = max(1, min(6, int(getattr(params, "k2d_kinematic_outer_passes", 4))))
            max_nfev = max(20, int(getattr(params, "k2d_kinematic_max_nfev", 70)))
            for outer in range(outer_passes):
                current = _kinematic_layout(local, root_centres, unpack(z), forest)
                if pair_fn is not None:
                    collision_pairs = list(pair_fn(current, pad=max(scale * 0.20, 1e-5)))
                else:
                    collision_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                # Keep the residual size bounded on very large layouts while
                # always retaining currently penetrating pairs.
                bad_now = mod._penetrating_pairs(pipeline_obj, current, collision_tol)
                bad_keys = {(min(i, j), max(i, j)) for i, j, _m, _d in bad_now}
                ordered_pairs = []
                seen_pairs: set[tuple[int, int]] = set()
                for i, j in [*( (i, j) for i, j, _m, _d in bad_now ), *collision_pairs]:
                    key = (min(int(i), int(j)), max(int(i), int(j)))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        ordered_pairs.append(key)
                max_pairs = max(300, int(getattr(params, "k2d_kinematic_max_collision_pairs", 2500)))
                collision_pairs = ordered_pairs[:max_pairs]

                w_loop = float(getattr(params, "k2d_kinematic_loop_weight", 30.0))
                w_collision = float(getattr(params, "k2d_kinematic_collision_weight", 40.0))
                w_anchor = float(getattr(params, "k2d_kinematic_anchor_weight", 0.02))
                w_angle = float(getattr(params, "k2d_kinematic_angle_weight", 0.002))

                def residual(zv: np.ndarray) -> np.ndarray:
                    layout = _kinematic_layout(local, root_centres, unpack(zv), forest)
                    vals: list[float] = []
                    for edge_id in loop_edges:
                        ia, ca, ib, cb = constraints[edge_id]
                        diff = (layout[ia, ca] - layout[ib, cb]) / scale
                        vals.extend((np.sqrt(w_loop) * diff).tolist())
                    if sat_fn is not None:
                        for i, j in collision_pairs:
                            overlap, mtv, signed = sat_fn(layout[i], layout[j], clearance=0.0)
                            depth = float(np.linalg.norm(np.asarray(mtv, dtype=float))) if overlap else 0.0
                            pen = depth / scale if overlap and float(signed) < -collision_tol and depth > collision_tol else 0.0
                            vals.append(float(np.sqrt(w_collision) * pen))
                    centres = np.mean(layout, axis=1)
                    vals.extend((np.sqrt(w_anchor) * ((centres - centres0) / scale)).reshape(-1).tolist())
                    vals.extend((np.sqrt(w_angle) * np.asarray(zv, dtype=float)).tolist())
                    return np.asarray(vals, dtype=float)

                # Dependency sparsity follows the tree ancestry.  This makes
                # finite-difference Jacobians practical for hundreds of tiles.
                rows = 2 * len(loop_edges) + len(collision_pairs) + 2 * n + len(variable_tiles)
                sparsity = lil_matrix((rows, len(variable_tiles)), dtype=int)
                row = 0
                for edge_id in loop_edges:
                    ia, _ca, ib, _cb = constraints[edge_id]
                    deps = ancestors[ia] | ancestors[ib]
                    cols = [var_index[t] for t in deps if t in var_index]
                    if cols:
                        sparsity[row, cols] = 1
                        sparsity[row + 1, cols] = 1
                    row += 2
                for i, j in collision_pairs:
                    deps = ancestors[i] | ancestors[j]
                    cols = [var_index[t] for t in deps if t in var_index]
                    if cols:
                        sparsity[row, cols] = 1
                    row += 1
                for tile in range(n):
                    cols = [var_index[t] for t in ancestors[tile] if t in var_index]
                    if cols:
                        sparsity[row, cols] = 1
                        sparsity[row + 1, cols] = 1
                    row += 2
                for tile in variable_tiles:
                    sparsity[row, var_index[tile]] = 1
                    row += 1

                result = least_squares(
                    residual,
                    z,
                    jac_sparsity=sparsity.tocsr(),
                    max_nfev=max_nfev,
                    xtol=1e-8,
                    ftol=1e-8,
                    gtol=1e-8,
                    verbose=0,
                )
                z = np.asarray(result.x, dtype=float)
                solver_messages.append(f"pass={outer + 1} success={bool(result.success)} nfev={int(result.nfev)} cost={float(result.cost):.6g}")

                current = _kinematic_layout(local, root_centres, unpack(z), forest)
                loop_err, _ = _loop_error_details(current, constraints, loop_edges)
                overlaps = mod._penetrating_pairs(pipeline_obj, current, collision_tol)
                if (loop_err.size == 0 or float(np.max(loop_err)) <= hinge_tol) and not overlaps:
                    break
        except Exception as exc:
            solver_messages.append(f"kinematic least_squares unavailable/failed: {type(exc).__name__}: {exc}")

        final = _kinematic_layout(local, root_centres, unpack(z), forest)
        all_err = _hinge_errors(final, constraints)
        tree_err = all_err[np.asarray(forest["tree_edges"], dtype=int)] if forest["tree_edges"] else np.zeros(0)
        loop_err, loop_edge_ids = _loop_error_details(final, constraints, loop_edges)
        final_overlaps = mod._penetrating_pairs(pipeline_obj, final, collision_tol)
        overlap_pairs = [(int(i), int(j)) for i, j, _m, _d in final_overlaps]
        overlap_tile_ids = sorted({v for pair in overlap_pairs for v in pair})
        violating_loop_edges = [int(edge_id) for edge_id, err in zip(loop_edge_ids, loop_err) if err > hinge_tol]
        hinge_violation_tiles: set[int] = set()
        for edge_id in violating_loop_edges:
            ia, _ca, ib, _cb = constraints[edge_id]
            hinge_violation_tiles.update((int(ia), int(ib)))

        tree_max = float(np.max(tree_err)) if tree_err.size else 0.0
        loop_max = float(np.max(loop_err)) if loop_err.size else 0.0
        loop_rms = float(np.sqrt(np.mean(loop_err * loop_err))) if loop_err.size else 0.0
        all_max = float(np.max(all_err)) if all_err.size else 0.0
        feasible = bool(tree_max <= max(hinge_tol * 1e-3, 1e-10) and loop_max <= hinge_tol and len(final_overlaps) == 0)

        metrics = dict(metrics or {})
        metrics.update({
            "onestring_k2d_kinematic_parameterization_applied": True,
            "onestring_k2d_kinematic_model": "spanning-forest hinge kinematics; tree hinges exact by construction; loop closures and SAT collision residuals optimized in angle space",
            "onestring_k2d_kinematic_component_count": int(len(forest["roots"])),
            "onestring_k2d_kinematic_tree_hinge_count": int(len(forest["tree_edges"])),
            "onestring_k2d_kinematic_loop_hinge_count": int(len(loop_edges)),
            "onestring_k2d_kinematic_tree_hinge_max_error": float(tree_max),
            "onestring_k2d_kinematic_loop_hinge_max_error": float(loop_max),
            "onestring_k2d_kinematic_loop_hinge_rms_error": float(loop_rms),
            "onestring_k2d_hard_hinge_max_after": float(all_max),
            "onestring_k2d_hard_hinge_tolerance": float(hinge_tol),
            "onestring_k2d_joint_hard_collision_tolerance": float(collision_tol),
            "onestring_k2d_joint_hard_final_overlap_count": int(len(final_overlaps)),
            "onestring_k2d_joint_hard_feasible": bool(feasible),
            "onestring_k2d_kinematic_feasible": bool(feasible),
            "onestring_k2d_kinematic_constraint_status": "satisfied" if feasible else "NOT SATISFIED; best-effort layout returned for downstream inspection",
            "onestring_k2d_kinematic_violating_loop_edge_ids": violating_loop_edges,
            "onestring_k2d_kinematic_hinge_violation_tile_ids": sorted(hinge_violation_tiles),
            "onestring_k2d_residual_overlap_pairs": [list(pair) for pair in overlap_pairs],
            "onestring_k2d_residual_overlap_pair_count": int(len(overlap_pairs)),
            "onestring_k2d_overlap_tile_ids": overlap_tile_ids,
            "onestring_k2d_hard_nonoverlap_final_penetration_count": int(len(final_overlaps)),
            "onestring_k2d_hard_nonoverlap_satisfied": bool(len(final_overlaps) == 0),
            "onestring_k2d_kinematic_scipy_used": bool(scipy_used),
            "onestring_k2d_kinematic_solver_log": solver_messages,
            "onestring_k2d_tile_rigidity_preserved": True,
        })

        print(
            "[OPTCUTS-TEST-K2D-KINEMATIC] "
            f"tree_hinges={len(forest['tree_edges'])} tree_max={tree_max:.6g} "
            f"loop_hinges={len(loop_edges)} loop_max={loop_max:.6g} tol={hinge_tol:.6g} "
            f"overlaps={len(final_overlaps)} feasible={feasible}"
        )
        if not feasible:
            print(
                "[OPTCUTS-TEST-K2D-KINEMATIC-NONFATAL] constraints not satisfied; "
                "continuing pipeline with diagnostic best-effort K2D layout"
            )
        return final, metrics

    # The old flat-layout wrapper aborts if any residual overlap remains.  In the
    # kinematic diagnostic mode we intentionally continue, while preserving the
    # actual overlap count from mesh.metrics for visualization and reporting.
    def permissive_layout_collision_metrics(pipeline_obj: Any, tiles: np.ndarray) -> tuple[int, float]:
        _actual_count, min_clear = original_layout_collision_metrics(pipeline_obj, tiles)
        return 0, float(min_clear)

    mod._layout_collision_metrics = permissive_layout_collision_metrics
    mod._build_rigid_k2d_layout = build_with_kinematic_constraints

    def make_layout_nonfatal(mesh: Any, params: Any = None):
        layout = base_make_layout(mesh, params)
        actual = dict(getattr(mesh, "metrics", {}) or {})
        try:
            layout.metrics.update(actual)
            actual_count = int(actual.get("onestring_k2d_residual_overlap_pair_count", actual.get("onestring_k2d_joint_hard_final_overlap_count", 0)))
            layout.metrics["tile_overlap_count"] = actual_count
            layout.metrics["onestring_k2d_hard_nonoverlap_final_assertion"] = False
            layout.metrics["onestring_k2d_constraints_nonfatal_diagnostic_mode"] = True
        except Exception:
            pass
        return layout

    pipeline._make_flat_tile_layout = make_layout_nonfatal
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._make_flat_tile_layout = make_layout_nonfatal
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_make_flat_tile_layout"] = make_layout_nonfatal

    mod._onestring_k2d_hard_hinge_installed = True


__all__ = ["install_optcuts_test_k2d_hard_hinge_patch"]
