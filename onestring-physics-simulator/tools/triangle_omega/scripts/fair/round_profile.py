"""Round finite-width surface bends with a position tether and fixed-rim volume."""

import numpy as np
import json
from igl.copyleft.cgal import remesh_self_intersections
from scipy.sparse.linalg import splu
from scipy.sparse import coo_matrix, csr_matrix, diags, eye, kron, vstack
from scipy.sparse.csgraph import connected_components, dijkstra


def round_profile(vertices, faces, span, angle_deg):
    # Work in units of the measurement span so weights are scale independent.
    origin = vertices.mean(axis=0)
    base = (vertices - origin) / span
    count = len(base)
    corners = faces.T
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse, multiplicity = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    rim_edges = directed[multiplicity[inverse] == 1]
    rim = np.unique(rim_edges)
    i, j = edges.T
    lengths = np.linalg.norm(base[i] - base[j], axis=1)

    def neighborhoods(x, radius=1.0):
        lengths = np.linalg.norm(x[i] - x[j], axis=1)
        graph = coo_matrix(
            (np.r_[lengths, lengths], (np.r_[i, j], np.r_[j, i])), shape=(count, count)
        ).tocsr()
        # Batches keep the bounded geodesic query from allocating an N by N matrix.
        starts, ends = [], []
        for start in range(0, count, 128):
            ids = np.arange(start, min(start + 128, count))
            dist = dijkstra(graph, indices=ids, limit=radius)
            row, col = np.nonzero(np.isfinite(dist) & (ids[:, None] < np.arange(count)))
            starts.append(ids[row])
            ends.append(col)
        return np.concatenate(starts), np.concatenate(ends), graph

    a, b, graph = neighborhoods(base)

    incident = coo_matrix(
        (np.ones(faces.size), (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
        shape=(count, len(faces)),
    ).tocsr()

    def geometry(x):
        p, q, r = x[corners]
        cross = np.cross(q - p, r - p)
        summed = incident @ cross
        norm = np.maximum(np.linalg.norm(summed, axis=1), 1e-14)
        return cross, summed / norm[:, None], norm

    def angles(normals):
        return np.arccos(np.clip(np.sum(normals[a] * normals[b], axis=1), -1.0, 1.0))

    cross0, normals0, _ = geometry(base)
    double_area = np.maximum(np.linalg.norm(cross0, axis=1), 1e-14)
    mass = (incident @ double_area) / 6

    def vertex_turn(first, second, turn):
        worst = np.zeros(count)
        np.maximum.at(worst, first, np.degrees(turn))
        np.maximum.at(worst, second, np.degrees(turn))
        return worst

    initial = angles(normals0)
    before = vertex_turn(a, b, initial)
    limit = np.radians(angle_deg)
    sharp = initial > limit
    edge_weight = 0.10 * (mass[i] + mass[j]) / (6 * np.maximum(lengths, 0.08) ** 2)
    normal0 = cross0 / double_area[:, None]
    # Resolve bends at distances shorter than the span as well. Averaged
    # vertex normals alone can hide a corner between two coarse triangles.
    shared = np.flatnonzero(multiplicity == 2)
    order = np.argsort(inverse, kind="stable")
    first = np.cumsum(multiplicity) - multiplicity
    left = order[first[shared]] // 3
    right = order[first[shared] + 1] // 3
    centers = base[faces].mean(axis=1)
    width = np.clip(np.linalg.norm(centers[left] - centers[right], axis=1), 0.12, 1.0)
    face_limit = limit * width
    bend_weight = 0.30 * (double_area[left] + double_area[right]) / (6 * width**2)

    initial_hinges = np.arccos(np.clip(np.sum(normal0[left] * normal0[right], axis=1), -1.0, 1.0))
    sharp_faces = initial_hinges > limit
    hot = np.unique(
        np.r_[a[sharp], b[sharp], faces[left[sharp_faces]].ravel(), faces[right[sharp_faces]].ravel()]
    )
    free = np.zeros(count, bool)
    if len(hot):
        free = np.isfinite(dijkstra(graph, indices=hot, min_only=True, limit=1.0))
    free[rim] = False
    active = np.flatnonzero(free)

    # Close each fixed open rim with a fan solely for volume measurement.
    # Cap contributions are constant because those vertices cannot move.
    cap_volume = 0.0
    if len(rim):
        s, t = rim_edges.T
        rim_graph = coo_matrix((np.ones(len(s)), (s, t)), shape=(count, count))
        _, groups = connected_components(rim_graph, directed=False)
        for label in np.unique(groups[rim]):
            ids = groups[s] == label
            center = base[rim[groups[rim] == label]].mean(axis=0)
            cap_volume += np.sum(np.cross(base[t[ids]], base[s[ids]]) @ center) / 6

    def volume(x, cross):
        return cap_volume + np.sum(x[corners[0]] * cross) / 6

    volume0 = volume(base, cross0)
    if len(active) and abs(volume0) < 1e-14:
        raise ValueError("体積の基準を決められない面です。メッシュの面の向きを確認してください。")

    tri0 = base[faces]
    edge2 = ((tri0 - np.roll(tri0, 1, axis=1)) ** 2).sum(axis=2)
    quality = 2 * np.sqrt(3.0) * double_area / edge2.sum(axis=1)
    # Anchor a few ill-conditioned triangles so they cannot dictate the step elsewhere.
    free[np.unique(faces[quality < 0.012])] = False
    active = np.flatnonzero(free)
    dofs = (3 * active[:, None] + np.arange(3)).ravel()

    def angle_jacobian(first, second, vector, norm):
        # One row for the derivative of the angle between two unit vectors,
        # with respect to the unnormalized vectors at its endpoints.
        cosine = np.clip(np.sum(vector[first] * vector[second], axis=1), -1.0, 1.0)
        sine = np.maximum(np.sqrt(1.0 - cosine * cosine), 1e-7)
        p = -(vector[second] - cosine[:, None] * vector[first]) / (sine * norm[first])[:, None]
        q = -(vector[first] - cosine[:, None] * vector[second]) / (sine * norm[second])[:, None]
        col = np.c_[3 * first[:, None] + np.arange(3), 3 * second[:, None] + np.arange(3)].ravel()
        return csr_matrix(
            (np.c_[p, q].ravel(), col, np.arange(0, 6 * len(first) + 1, 6)),
            shape=(len(first), 3 * len(vector)),
        )

    def cross_jacobian(x):
        # d[(q-p) x (r-p)] for displacement of each triangle corner.
        sides = x[faces[:, [2, 0, 1]]] - x[faces[:, [1, 2, 0]]]
        # d cross / d p = dp x (q-r); equivalently (r-q) x dp.

        ax, ay, az = sides.transpose(2, 0, 1)
        z = np.zeros_like(ax)
        blocks = np.stack([z, -az, ay, az, z, -ax, -ay, ax, z], axis=-1).reshape(len(faces), 3, 3, 3)
        data = blocks.transpose(0, 2, 1, 3).ravel()
        ids = (3 * faces[:, :, None] + np.arange(3)).reshape(len(faces), 9)
        col = np.repeat(ids, 3, axis=0).ravel()
        return csr_matrix((data, col, np.arange(0, len(data) + 1, 9)), shape=(3 * len(faces), 3 * count))[
            :, dofs
        ]

    sum_faces = kron(incident, eye(3), format="csr")
    edges_d = coo_matrix(
        (np.tile([1.0, -1.0], len(i)), (np.repeat(np.arange(len(i)), 2), np.stack([i, j], axis=1).ravel())),
        shape=(len(i), count),
    ).tocsr()
    reg = kron(0.9 * diags(mass) + edges_d.T @ diags(5 * edge_weight) @ edges_d, eye(3), format="csc")[
        dofs
    ][:, dofs]
    change_limit = max(0.0, limit - np.radians(4.0))
    solution = base.copy()
    iterations = 0

    def residuals(x):
        cross, normals, norm = geometry(x)
        turn = angles(normals)
        face_area = np.maximum(np.linalg.norm(cross, axis=1), 1e-14)
        fn = cross / face_area[:, None]
        hinge = np.arccos(np.clip((fn[left] * fn[right]).sum(1), -1.0, 1.0))
        ratios = (cross * normal0).sum(1) / double_area
        return cross, normals, norm, turn, face_area, fn, hinge, ratios

    def intersecting_pairs(x):
        pairs = remesh_self_intersections(x, faces, detect_only=True)[2]
        return {tuple(pair) for pair in np.sort(pairs, axis=1)}

    def targets(x):
        chord = np.linalg.norm(x[a] - x[b], axis=1)
        # The angle swept by a circular arc between two samples.
        half = np.clip(chord * np.sin(change_limit / 2), 0.0, 0.99)
        return 2 * np.arcsin(half)

    def merit(x, turns, hinges, strength):
        pos = (x - base)[active].ravel()
        return (
            0.5 * np.sum(weight * np.maximum(turns - targets(x), 0.0) ** 2) * strength
            + 0.5 * np.sum(bend_weight * np.maximum(hinges - face_limit, 0.0) ** 2) * np.sqrt(strength)
            + 0.5 * pos.dot(reg @ pos)
        )

    # Evaluate accepted passes on geodesic neighborhoods of the actual deformed
    # mesh, not those of the fixed input. Keep the best geometrically valid pass.
    initial_intersections = intersecting_pairs(base)
    best_solution = solution.copy()
    after = before.copy()
    best_rank = (
        len(initial_intersections),
        np.count_nonzero(before > angle_deg),
        np.count_nonzero(initial_hinges > limit),
        np.linalg.norm(np.maximum(before - angle_deg, 0.0))
        + np.linalg.norm(np.degrees(np.maximum(initial_hinges - limit, 0.0))),
    )
    chosen_strength = 0.0
    chosen_intersections = len(initial_intersections)
    for strength in (1.0, 6.0, 32.0, 128.0, 512.0, 1600.0, 4000.0, 8000.0):
        if not len(active):
            break
        a, b, _ = neighborhoods(solution, 1.35)
        degree = np.maximum(np.bincount(np.r_[a, b], minlength=count), 1)
        node_weight = np.maximum(mass, np.median(mass) * 0.25)
        weight = 0.5 * (node_weight[a] / degree[a] + node_weight[b] / degree[b])
        for iteration in range(20):
            cross, normals, norm, turn, face_area, fn, hinge, ratios = residuals(solution)
            # Near-active constraints also resist entering a new violation.
            target = targets(solution)
            pair = turn > target - 0.015
            bend = hinge > face_limit - 0.015
            A = cross_jacobian(solution)
            J = angle_jacobian(a[pair], b[pair], normals, norm) @ (sum_faces @ A)
            diff = solution[a[pair]] - solution[b[pair]]
            length = np.linalg.norm(diff, axis=1)
            slope = (
                2
                * np.sin(change_limit / 2)
                / np.sqrt(np.maximum(1 - np.square(length * np.sin(change_limit / 2)), 0.02))
            )
            slope[length * np.sin(change_limit / 2) >= 0.99] = 0.0
            side = -slope[:, None] * diff / np.maximum(length[:, None], 1e-12)
            ids = np.c_[3 * a[pair, None] + np.arange(3), 3 * b[pair, None] + np.arange(3)].ravel()
            J += csr_matrix(
                (np.c_[side, -side].ravel(), ids, np.arange(0, len(ids) + 1, 6)),
                shape=(len(length), 3 * count),
            )[:, dofs]
            H = angle_jacobian(left[bend], right[bend], fn, face_area) @ A
            weights = np.sqrt(weight[pair] * strength)
            bw = np.sqrt(bend_weight[bend] * np.sqrt(strength))
            mats = [J.multiply(weights[:, None]), H.multiply(bw[:, None])]
            rhs = [
                weights * np.maximum(turn[pair] - target[pair], 0.0),
                bw * np.maximum(hinge[bend] - face_limit[bend], 0.0),
            ]
            bad = ratios < 0.62
            if bad.any():
                ids = np.flatnonzero(bad)
                col = (3 * ids[:, None] + np.arange(3)).ravel()
                factor = np.sqrt(100.0 * double_area[ids])
                nr = csr_matrix(
                    (
                        (factor[:, None] * normal0[ids] / double_area[ids, None]).ravel(),
                        col,
                        np.arange(0, len(col) + 1, 3),
                    ),
                    shape=(len(ids), 3 * len(faces)),
                )
                mats.append(nr @ A)
                rhs.append(factor * np.minimum(ratios[ids] - 0.58, 0.0))
            M = vstack(mats, format="csr")
            off = (solution - base)[active].ravel()
            grad = M.T @ np.concatenate(rhs) + reg @ off
            mat = M.T @ M + reg + diags(np.repeat(mass[active], 3))
            factor = splu(mat.tocsc())
            # Exact first-order volume constraint, restricted to the moving vertices.
            g = (incident @ cross)[active].ravel() / 6
            dz, vg = factor.solve(np.c_[-grad, g]).T
            dv = volume(solution, cross) - volume0
            dz -= vg * (g.dot(dz) + dv) / max(g.dot(vg), 1e-20)
            delta = np.zeros_like(solution)
            delta[active] = dz.reshape(-1, 3)
            current = merit(solution, turn, hinge, strength)
            for k in range(12):
                trial = solution + delta * (0.5**k)
                for repair in range(12):
                    _, _, _, trial_turn, _, _, trial_hinge, trial_ratio = residuals(trial)
                    invalid = trial_ratio < 0.15
                    if not invalid.any():
                        break
                    ids = np.unique(faces[invalid])
                    trial[ids] = 0.5 * (solution[ids] + trial[ids]) if repair < 6 else solution[ids]
                else:
                    continue
                if merit(trial, trial_turn, trial_hinge, strength) < current:
                    break
            else:
                break
            solution = trial
            iterations += 1
            if k > 10:
                break
        a, b, _ = neighborhoods(solution)
        cross, normals, _ = geometry(solution)
        measured = vertex_turn(a, b, angles(normals))
        intersections = intersecting_pairs(solution)
        dv = abs((volume(solution, cross) - volume0) / volume0)
        face_normals = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-14)
        hinges = np.arccos(np.clip(np.sum(face_normals[left] * face_normals[right], axis=1), -1.0, 1.0))
        rank = (
            len(intersections),
            np.count_nonzero(measured > angle_deg),
            np.count_nonzero(hinges > limit),
            np.linalg.norm(np.maximum(measured - angle_deg, 0.0))
            + np.linalg.norm(np.degrees(np.maximum(hinges - limit, 0.0))),
        )
        if dv <= 0.001 and intersections.issubset(initial_intersections) and rank < best_rank:
            best_solution = solution.copy()
            best_rank = rank
            after = measured
            chosen_strength = strength
            chosen_intersections = len(intersections)
        print(
            json.dumps(
                dict(
                    penalty=strength,
                    remaining_vertices=int(rank[1]),
                    intersections=len(intersections),
                    relative_volume_error=dv,
                )
            ),
            flush=True,
        )
        if best_rank[:3] == (0, 0, 0):
            break
    solution = best_solution
    cross, normals, _ = geometry(solution)
    positions = vertices + (solution - base) * span
    shift = np.linalg.norm(positions - vertices, axis=1)
    volume_change = (volume(solution, cross) - volume0) / volume0 if volume0 else 0.0
    stats = dict(
        span=span,
        threshold_degrees=angle_deg,
        iterations=int(iterations),
        solver="constrained_gauss_newton",
        selected_penalty=chosen_strength,
        hinges_above_before=int(np.count_nonzero(initial_hinges > limit)),
        hinges_above_after=int(best_rank[2]),
        intersections_before=len(initial_intersections),
        intersections_after=chosen_intersections,
        volume_relative_change=float(volume_change),
        rms_move=float(np.sqrt(np.mean(shift**2))),
        max_move=float(shift.max(initial=0)),
        vertices_above_before=int(np.sum(before > angle_deg)),
        vertices_above_after=int(np.sum(after > angle_deg)),
        p95_before=float(np.percentile(before, 95)),
        p95_after=float(np.percentile(after, 95)),
    )
    return positions, stats, before, after


if __name__ == "__main__":
    import os
    import pickle
    from pathlib import Path
    import sys
    import trimesh

    src, directory = map(Path, sys.argv[1:])
    directory.mkdir(exist_ok=True, parents=True)
    mesh = trimesh.load(src, process=False)
    result, stats, before, after = round_profile(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        float(os.environ["TRI_PROFILE_SPAN"]),
        float(os.environ["TRI_CREASE_ANGLE"]),
    )
    trimesh.Trimesh(result, mesh.faces, process=False).export(directory / "inflated.obj", digits=15)
    (directory / "inflate_summary.json").write_text(json.dumps(dict(profile_rounding=stats), indent=2))
    with (directory / "chart_inflated.pkl").open("wb") as stream:
        pickle.dump(dict(xyz=result, f3=mesh.faces), stream)
    with (directory / "inflate_fields.pkl").open("wb") as stream:
        pickle.dump(
            dict(V0=mesh.vertices, V=result, F=mesh.faces, turn_before=before, turn_after=after), stream
        )
    print(json.dumps(stats, indent=2), flush=True)
