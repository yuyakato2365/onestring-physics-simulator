"""Grid-constrained reparameterization of an official OptCuts cut topology.

This is the OneString/OptCuts fusion used by ``app_optcuts.py``.

Important design rule
---------------------
The official OptCuts result is used to decide *which surface edges are cut*.
We do NOT keep its arbitrary curved UV seam and then add a second fabrication
seam afterwards.  Instead, the OptCuts cut topology is kept and the UV map is
re-solved under OneString fabrication constraints:

* the existing ``tile_size`` is the fixed lattice unit h;
* one global orthogonal frame is estimated from the OptCuts seam network;
* every physical seam chain is represented by a straight line in that frame;
* every seam line is horizontal or vertical;
* seam endpoints/junctions lie on h-spaced lattice intersections;
* all UV vertices on a seam copy are hard-constrained to that straight line;
* all remaining UV vertices are re-optimized for Symmetric Dirichlet distortion.

Thus the final UV discontinuity is the *same cut topology* that OptCuts chose,
but its geometry is genuinely grid-feasible before M2D is constructed.  This is
not a post-hoc seam overlay.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
import os
from typing import Any

import numpy as np

try:  # optional, but expected in the OneString environment
    import torch
except Exception:  # pragma: no cover
    torch = None


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _surface_seam_records(parameterization: Any) -> list[dict[str, Any]]:
    """Return internal surface edges whose two incident faces use different UV ids.

    Each record stores the canonical surface edge and its two UV boundary copies.
    A UV copy is represented as ``{surface_vertex_id: uv_vertex_id}`` so the two
    sides of one physical seam can be followed without confusing duplicated ids.
    """
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    if len(sf) != len(uf):
        raise RuntimeError("OPTCUTS_GRID_CONSTRAINT_FACE_COUNT_MISMATCH")
    incidences: dict[tuple[int, int], list[dict[int, int]]] = defaultdict(list)
    for face3, face2 in zip(sf, uf):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            sa, sb = int(face3[i]), int(face3[j])
            incidences[_edge_key(sa, sb)].append({sa: int(face2[i]), sb: int(face2[j])})
    records: list[dict[str, Any]] = []
    for edge, copies in incidences.items():
        if len(copies) != 2:
            continue
        a, b = edge
        c0, c1 = copies
        if a not in c0 or b not in c0 or a not in c1 or b not in c1:
            continue
        if c0[a] == c1[a] and c0[b] == c1[b]:
            continue
        records.append({"surface_edge": (a, b), "copies": (c0, c1)})
    return records


def _physical_chains(records: list[dict[str, Any]]) -> list[list[int]]:
    """Collapse degree-2 physical seam vertices into chains.

    OptCuts may branch.  Endpoints and junctions are therefore terminals and a
    physical chain is the maximal path between two terminals.  Pure cycles are
    broken once so that every chain can be assigned one straight fabrication line.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    for record in records:
        a, b = record["surface_edge"]
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    if not adjacency:
        return []
    terminals = {v for v, nbrs in adjacency.items() if len(nbrs) != 2}
    visited: set[tuple[int, int]] = set()
    chains: list[list[int]] = []

    def walk(start: int, nxt: int) -> list[int]:
        chain = [int(start), int(nxt)]
        visited.add(_edge_key(start, nxt))
        prev, cur = int(start), int(nxt)
        while cur not in terminals:
            candidates = [x for x in adjacency[cur] if x != prev and _edge_key(cur, x) not in visited]
            if not candidates:
                break
            following = int(candidates[0])
            visited.add(_edge_key(cur, following))
            chain.append(following)
            prev, cur = cur, following
        return chain

    for terminal in sorted(terminals):
        for nbr in sorted(adjacency[terminal]):
            if _edge_key(terminal, nbr) not in visited:
                chains.append(walk(terminal, int(nbr)))

    # Residual all-degree-2 cycles.
    for a, nbrs in sorted(adjacency.items()):
        for b in sorted(nbrs):
            if _edge_key(a, b) in visited:
                continue
            chain = [int(a), int(b)]
            visited.add(_edge_key(a, b))
            prev, cur = int(a), int(b)
            while True:
                candidates = [x for x in adjacency[cur] if x != prev and _edge_key(cur, x) not in visited]
                if not candidates:
                    break
                following = int(candidates[0])
                visited.add(_edge_key(cur, following))
                chain.append(following)
                prev, cur = cur, following
                if cur == chain[0]:
                    break
            if len(chain) >= 2:
                chains.append(chain)
    return chains


def _record_lookup(records: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {_edge_key(*record["surface_edge"]): record for record in records}


def _uv_copy_chains_for_physical_chain(
    chain: list[int],
    lookup: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Follow the two UV boundary copies of one physical seam chain."""
    if len(chain) < 2:
        return [], []
    first = lookup[_edge_key(chain[0], chain[1])]
    a0, a1 = first["copies"]
    side0 = [int(a0[chain[0]]), int(a0[chain[1]])]
    side1 = [int(a1[chain[0]]), int(a1[chain[1]])]
    for i in range(1, len(chain) - 1):
        prev_surface = int(chain[i])
        next_surface = int(chain[i + 1])
        record = lookup[_edge_key(prev_surface, next_surface)]
        copies = list(record["copies"])

        def extend(side: list[int], candidates: list[dict[int, int]]) -> int:
            current_uv = int(side[-1])
            matching = [c for c in candidates if int(c[prev_surface]) == current_uv]
            if matching:
                return int(matching[0][next_surface])
            # At a branch/junction OptCuts can duplicate the terminal again.  Pick
            # the closest id continuity only as a deterministic fallback; chains
            # stop at junctions so this is uncommon.
            return int(candidates[0][next_surface])

        next0 = extend(side0, copies)
        # Prefer the other copy for side1 when possible.
        next1_candidates = [c for c in copies if int(c[next_surface]) != next0] or copies
        next1 = extend(side1, next1_candidates)
        side0.append(next0)
        side1.append(next1)
    return side0, side1


def _uv_vertex_to_surface_vertex(parameterization: Any) -> np.ndarray:
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    count = len(np.asarray(parameterization.uv_vertices_2d))
    mapping = np.full(count, -1, dtype=int)
    for f3, f2 in zip(sf, uf):
        for s, u in zip(f3, f2):
            u = int(u)
            s = int(s)
            if mapping[u] < 0:
                mapping[u] = s
            elif mapping[u] != s:
                raise RuntimeError("OPTCUTS_UV_VERTEX_HAS_MULTIPLE_SURFACE_VERTICES")
    return mapping


def _dominant_axis_angle(uv: np.ndarray, copy_chains: list[list[int]]) -> float:
    moment = np.zeros((2, 2), dtype=float)
    total = 0.0
    for chain in copy_chains:
        if len(chain) < 2:
            continue
        pts = uv[np.asarray(chain, dtype=int)]
        for d in np.diff(pts, axis=0):
            length = float(np.linalg.norm(d))
            if length <= 1e-12:
                continue
            unit = d / length
            moment += length * np.outer(unit, unit)
            total += length
    if total <= 1e-12:
        return 0.0
    values, vectors = np.linalg.eigh(moment)
    axis = vectors[:, int(np.argmax(values))]
    angle = float(math.atan2(axis[1], axis[0]))
    while angle > math.pi / 2.0:
        angle -= math.pi
    while angle <= -math.pi / 2.0:
        angle += math.pi
    return angle


def _rotate_uv(uv: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(uv, axis=0) if len(uv) else np.zeros(2, dtype=float)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    return (uv - center[None, :]) @ rotation.T + center[None, :], rotation


def _union_find_groups(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(int(a), int(b))
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(count):
        groups[find(i)].append(i)
    return list(groups.values())


def _build_hard_seam_targets(
    parameterization: Any,
    h: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return rotated UV, hard-target array, and constraint diagnostics.

    ``targets[i]`` is NaN for free UV vertices.  A constrained seam vertex has an
    exact lattice-compatible coordinate.
    """
    uv0 = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    records = _surface_seam_records(parameterization)
    physical = _physical_chains(records)
    lookup = _record_lookup(records)
    chain_entries: list[dict[str, Any]] = []
    all_copy_chains: list[list[int]] = []
    for chain in physical:
        side0, side1 = _uv_copy_chains_for_physical_chain(chain, lookup)
        if len(side0) < 2 or len(side1) < 2:
            continue
        all_copy_chains.extend([side0, side1])
        chain_entries.append({"physical": chain, "copies": [side0, side1]})

    if not chain_entries:
        # Open surfaces can legitimately have no OptCuts-created internal seam.
        return uv0.copy(), np.full_like(uv0, np.nan), {
            "seam_chain_count": 0,
            "seam_copy_chain_count": 0,
            "axis_rotation_degrees": 0.0,
            "constrained_vertex_count": 0,
        }

    angle = _dominant_axis_angle(uv0, all_copy_chains)
    rotated, _rotation = _rotate_uv(uv0, -angle)
    uv_to_surface = _uv_vertex_to_surface_vertex(parameterization)
    surface_xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)

    # Physical chain orientation is shared by both UV copies.
    for entry in chain_entries:
        endpoints = []
        for copy in entry["copies"]:
            pts = rotated[np.asarray(copy, dtype=int)]
            endpoints.append(pts[-1] - pts[0])
        direction = np.mean(np.asarray(endpoints, dtype=float), axis=0)
        entry["axis"] = 0 if abs(float(direction[0])) >= abs(float(direction[1])) else 1

    # Every UV-copy chain is an independent straight line, but chains sharing the
    # same UV junction and the same orientation must share the same lattice line.
    copy_entries: list[dict[str, Any]] = []
    for physical_id, entry in enumerate(chain_entries):
        for side_id, copy in enumerate(entry["copies"]):
            copy_entries.append({
                "physical_id": physical_id,
                "side_id": side_id,
                "uv_ids": [int(x) for x in copy],
                "axis": int(entry["axis"]),
            })

    targets = np.full_like(rotated, np.nan)
    for axis in (0, 1):
        indices = [i for i, e in enumerate(copy_entries) if e["axis"] == axis]
        local_index = {global_i: k for k, global_i in enumerate(indices)}
        endpoint_to_chains: dict[int, list[int]] = defaultdict(list)
        for global_i in indices:
            copy = copy_entries[global_i]["uv_ids"]
            endpoint_to_chains[copy[0]].append(global_i)
            endpoint_to_chains[copy[-1]].append(global_i)
        pairs: list[tuple[int, int]] = []
        for touching in endpoint_to_chains.values():
            for j in range(1, len(touching)):
                pairs.append((local_index[touching[0]], local_index[touching[j]]))
        groups = _union_find_groups(len(indices), pairs) if indices else []
        constant_coord = 1 - axis
        for group in groups:
            global_group = [indices[g] for g in group]
            sample_ids = sorted({u for gi in global_group for u in copy_entries[gi]["uv_ids"]})
            mean_c = float(np.mean(rotated[np.asarray(sample_ids, dtype=int), constant_coord]))
            snapped_c = round(mean_c / h) * h
            for gi in global_group:
                copy_entries[gi]["constant"] = float(snapped_c)

    # Determine lattice terminal coordinates.  A terminal may be constrained by a
    # horizontal and a vertical chain simultaneously; that naturally yields an
    # exact grid intersection.
    terminal_incident: dict[int, list[int]] = defaultdict(list)
    for ci, entry in enumerate(copy_entries):
        ids = entry["uv_ids"]
        terminal_incident[ids[0]].append(ci)
        terminal_incident[ids[-1]].append(ci)
    terminal_xy: dict[int, np.ndarray] = {}
    for terminal, incident in terminal_incident.items():
        p = rotated[int(terminal)].copy()
        x_candidates = [copy_entries[i]["constant"] for i in incident if copy_entries[i]["axis"] == 1]
        y_candidates = [copy_entries[i]["constant"] for i in incident if copy_entries[i]["axis"] == 0]
        p[0] = float(np.mean(x_candidates)) if x_candidates else round(float(p[0]) / h) * h
        p[1] = float(np.mean(y_candidates)) if y_candidates else round(float(p[1]) / h) * h
        terminal_xy[int(terminal)] = p

    # Put every seam-copy vertex on its hard straight line.  Internal vertices are
    # parameterized by cumulative 3D seam arclength, not by the noisy original UV.
    for entry in copy_entries:
        ids = entry["uv_ids"]
        axis = int(entry["axis"])
        const = float(entry["constant"])
        p0 = terminal_xy[ids[0]].copy()
        p1 = terminal_xy[ids[-1]].copy()
        p0[1 - axis] = const
        p1[1 - axis] = const
        varying0 = float(p0[axis])
        varying1 = float(p1[axis])
        if abs(varying1 - varying0) < 0.5 * h:
            sign = 1.0 if float(rotated[ids[-1], axis] - rotated[ids[0], axis]) >= 0.0 else -1.0
            varying1 = varying0 + sign * h
            p1[axis] = varying1
            terminal_xy[ids[-1]] = p1
        surface_ids = [int(uv_to_surface[u]) for u in ids]
        if any(s < 0 for s in surface_ids):
            raise RuntimeError("OPTCUTS_SEAM_COPY_WITHOUT_SURFACE_VERTEX")
        xyz = surface_xyz[np.asarray(surface_ids, dtype=int)]
        lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = float(cumulative[-1])
        if total <= 1e-12:
            tvals = np.linspace(0.0, 1.0, len(ids))
        else:
            tvals = cumulative / total
        for u, t in zip(ids, tvals):
            target = np.zeros(2, dtype=float)
            target[axis] = (1.0 - float(t)) * varying0 + float(t) * varying1
            target[1 - axis] = const
            if np.all(np.isfinite(targets[int(u)])):
                # A UV junction belongs to multiple chains.  Their exact constraints
                # should intersect; average only numerical differences.
                targets[int(u)] = 0.5 * (targets[int(u)] + target)
            else:
                targets[int(u)] = target

    # Snap all shared terminal targets exactly to lattice intersections once more.
    for terminal, p in terminal_xy.items():
        if np.all(np.isfinite(targets[int(terminal)])):
            targets[int(terminal)] = np.round(targets[int(terminal)] / h) * h

    return rotated, targets, {
        "seam_chain_count": int(len(chain_entries)),
        "seam_copy_chain_count": int(len(copy_entries)),
        "axis_rotation_degrees": float(math.degrees(-angle)),
        "constrained_vertex_count": int(np.count_nonzero(np.all(np.isfinite(targets), axis=1))),
    }


def _source_inverse_matrices(parameterization: Any) -> np.ndarray:
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    invs = np.zeros((len(sf), 2, 2), dtype=float)
    for i, face in enumerate(sf):
        p = xyz[np.asarray(face, dtype=int)]
        e1 = p[1] - p[0]
        e2 = p[2] - p[0]
        l1 = float(np.linalg.norm(e1))
        n = np.cross(e1, e2)
        nl = float(np.linalg.norm(n))
        if l1 <= 1e-14 or nl <= 1e-14:
            raise RuntimeError(f"OPTCUTS_DEGENERATE_SOURCE_TRIANGLE:{i}")
        x = e1 / l1
        z = n / nl
        y = np.cross(z, x)
        source = np.asarray(
            [[l1, float(np.dot(e2, x))], [0.0, float(np.dot(e2, y))]],
            dtype=float,
        )
        invs[i] = np.linalg.inv(source)
    return invs


def _triangle_signed_areas(uv: np.ndarray, uv_faces: np.ndarray) -> np.ndarray:
    q = np.asarray(uv, dtype=float)[np.asarray(uv_faces, dtype=int)]
    return 0.5 * (
        (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
        - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    )


def _optimize_constrained_uv(
    parameterization: Any,
    uv_initial: np.ndarray,
    hard_targets: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for grid-constrained OptCuts reparameterization")
    uv_faces_np = np.asarray(parameterization.uv_faces, dtype=int)
    inv_source_np = _source_inverse_matrices(parameterization)
    fixed_np = np.all(np.isfinite(hard_targets), axis=1)
    if not np.any(fixed_np):
        return uv_initial.copy(), {"optimizer": "none-no-internal-seam", "iterations": 0}

    requested = os.environ.get("ONESTRING_BIJECTIVE_DEVICE", "").strip().lower()
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float32 if device.type in {"mps", "cuda"} else torch.float64

    uv0 = torch.tensor(np.asarray(uv_initial, dtype=float), device=device, dtype=dtype)
    targets = torch.tensor(np.nan_to_num(hard_targets, nan=0.0), device=device, dtype=dtype)
    fixed = torch.tensor(fixed_np, device=device, dtype=torch.bool)
    faces = torch.tensor(uv_faces_np, device=device, dtype=torch.long)
    inv_source = torch.tensor(inv_source_np, device=device, dtype=dtype)
    variable = torch.nn.Parameter(uv0.clone())
    optimizer = torch.optim.Adam([variable], lr=0.015)
    eps_det = 1.0e-5
    anchor_weight = 2.0e-4
    flip_weight = 5.0e4
    best_loss = float("inf")
    best = None

    for iteration in range(max(20, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        uv = torch.where(fixed[:, None], targets, variable)
        tri = uv[faces]
        b0 = tri[:, 1] - tri[:, 0]
        b1 = tri[:, 2] - tri[:, 0]
        B = torch.stack([b0, b1], dim=2)  # N x 2 x 2
        J = torch.bmm(B, inv_source)
        a, b = J[:, 0, 0], J[:, 0, 1]
        c, d = J[:, 1, 0], J[:, 1, 1]
        det = a * d - b * c
        safe = torch.clamp(det, min=eps_det)
        frob = a * a + b * b + c * c + d * d
        inv_frob = (a * a + b * b + c * c + d * d) / (safe * safe)
        sd = frob + inv_frob
        flip_barrier = torch.relu(eps_det - det)
        free_delta = variable - uv0
        loss = torch.mean(sd) + flip_weight * torch.mean(flip_barrier * flip_barrier)
        loss = loss + anchor_weight * torch.mean(free_delta[~fixed] * free_delta[~fixed])
        loss.backward()
        if variable.grad is not None:
            variable.grad[fixed] = 0.0
        optimizer.step()
        with torch.no_grad():
            variable[fixed] = targets[fixed]
        value = float(loss.detach().cpu())
        if math.isfinite(value) and value < best_loss:
            best_loss = value
            best = torch.where(fixed[:, None], targets, variable).detach().cpu().numpy()

    if best is None:
        raise RuntimeError("OPTCUTS_GRID_CONSTRAINED_OPTIMIZER_FAILED")
    best[fixed_np] = hard_targets[fixed_np]
    areas = _triangle_signed_areas(best, uv_faces_np)
    initial_areas = _triangle_signed_areas(uv_initial, uv_faces_np)
    sign = 1.0 if float(np.median(initial_areas)) >= 0.0 else -1.0
    invalid = np.where(sign * areas <= 1e-10)[0]
    if len(invalid):
        raise RuntimeError(
            "OPTCUTS_GRID_CONSTRAINT_INFEASIBLE: constrained seam layout produced "
            f"{len(invalid)} flipped/degenerate triangles; examples={invalid[:16].tolist()}"
        )
    return np.asarray(best, dtype=float), {
        "optimizer": "torch_adam_symmetric_dirichlet_with_hard_grid_seam_constraints",
        "device": str(device),
        "iterations": int(max(20, int(iterations))),
        "final_loss": float(best_loss),
        "fixed_vertex_count": int(np.count_nonzero(fixed_np)),
        "flip_count": 0,
    }


def _distortion_metrics(parameterization: Any, uv: np.ndarray) -> dict[str, float]:
    inv_source = _source_inverse_matrices(parameterization)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    tri = np.asarray(uv, dtype=float)[uf]
    values: list[float] = []
    for q, inv_s in zip(tri, inv_source):
        B = np.column_stack([q[1] - q[0], q[2] - q[0]])
        J = B @ inv_s
        det = float(np.linalg.det(J))
        if det <= 1e-12:
            values.append(float("inf"))
            continue
        values.append(float(np.sum(J * J) + np.sum(np.linalg.inv(J) ** 2)))
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    return {
        "grid_constrained_sd_mean": float(np.mean(finite)) if len(finite) else float("inf"),
        "grid_constrained_sd_max": float(np.max(finite)) if len(finite) else float("inf"),
    }


def install_optcuts_grid_constrained_parameterization_patch(pipeline: Any) -> None:
    """Wrap the already-installed official OptCuts builder with the hard seam solve."""
    if getattr(pipeline, "_onestring_optcuts_grid_constrained_parameterization_installed", False):
        return
    base_builder = pipeline._build_surface_parameterization

    def constrained_builder(surface: Any, target: Any, grid: Any, params: Any):
        parameterization = base_builder(surface, target, grid, params)
        if str(getattr(parameterization, "method", "")) != "optcuts":
            return parameterization
        h = max(float(getattr(params, "tile_size", getattr(grid, "tile_size", 0.0))), 1e-8)
        uv_rotated, hard_targets, seam_info = _build_hard_seam_targets(parameterization, h)
        iterations = int(os.environ.get("ONESTRING_OPTCUTS_GRID_OPT_ITERS", "180"))
        uv_final, opt_info = _optimize_constrained_uv(
            parameterization,
            uv_rotated,
            hard_targets,
            iterations,
        )
        parameterization.uv_vertices_2d = np.asarray(uv_final, dtype=float)
        loop = [int(x) for x in parameterization.metrics.get("boundary_loop", [])]
        if loop:
            parameterization.omega_boundary = parameterization.uv_vertices_2d[loop + [loop[0]]]
        parameterization.metrics.update({
            "omega_parameterization_mode": "optcuts",
            "optcuts_grid_constrained": True,
            "optcuts_fusion_model": (
                "official OptCuts cut topology + hard fixed-unit orthogonal seam geometry + "
                "Symmetric Dirichlet reparameterization"
            ),
            "optcuts_posthoc_extra_seam": False,
            "optcuts_original_uv_used_as_final": False,
            "optcuts_grid_unit": float(h),
            "optcuts_grid_allowed_seam_directions": "two global orthogonal axes only",
            "optcuts_grid_seam_geometry": "straight line per physical seam chain; no staircase/L-path postprocess",
            **seam_info,
            **opt_info,
            **_distortion_metrics(parameterization, uv_final),
        })
        setattr(parameterization, "_onestring_grid_unit", float(h))
        print(
            "[OPTCUTS-GRID-CONSTRAINED] "
            f"chains={seam_info['seam_chain_count']} copies={seam_info['seam_copy_chain_count']} "
            f"fixed={seam_info['constrained_vertex_count']} h={h:g} "
            f"rotation={seam_info['axis_rotation_degrees']:.3f}deg "
            f"device={opt_info.get('device', 'none')} sd_mean={parameterization.metrics['grid_constrained_sd_mean']:.6g}"
        )
        return parameterization

    pipeline._build_surface_parameterization = constrained_builder
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._build_surface_parameterization = constrained_builder
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_surface_parameterization"] = constrained_builder
    pipeline._onestring_optcuts_grid_constrained_parameterization_installed = True


__all__ = [
    "install_optcuts_grid_constrained_parameterization_patch",
    "_surface_seam_records",
    "_physical_chains",
    "_build_hard_seam_targets",
]
