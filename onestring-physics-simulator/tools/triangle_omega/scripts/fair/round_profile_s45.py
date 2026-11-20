"""Round finite-width surface bends with a position tether and fixed-rim volume."""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def round_profile(vertices, faces, span, angle_deg):
    # Work in units of the measurement span so weights are scale independent.
    origin = vertices.mean(axis=0)
    base = (vertices - origin) / span
    count = len(base)
    corners = faces.T
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse, multiplicity = np.unique(np.sort(directed, axis=1), axis=0,
                                             return_inverse=True, return_counts=True)
    rim_edges = directed[multiplicity[inverse] == 1]
    rim = np.unique(rim_edges)
    i, j = edges.T
    lengths = np.linalg.norm(base[i] - base[j], axis=1)
    graph = coo_matrix((np.r_[lengths, lengths], (np.r_[i, j], np.r_[j, i])),
                       shape=(count, count)).tocsr()
    # Batches keep the bounded geodesic query from allocating an N by N matrix.
    starts, ends = [], []
    for start in range(0, count, 128):
        ids = np.arange(start, min(start + 128, count))
        dist = dijkstra(graph, indices=ids, limit=1.)
        row, col = np.nonzero(np.isfinite(dist) & (ids[:, None] < np.arange(count)))
        starts.extend(ids[row])
        ends.extend(col)
    a, b = np.asarray(starts, int), np.asarray(ends, int)
    incident = coo_matrix((np.ones(faces.size), (faces.ravel(), np.repeat(np.arange(len(faces)), 3))),
                          shape=(count, len(faces))).tocsr()

    def geometry(x):
        p, q, r = x[corners]
        cross = np.cross(q - p, r - p)
        summed = incident @ cross
        norm = np.maximum(np.linalg.norm(summed, axis=1), 1e-14)
        return cross, summed / norm[:, None], norm

    def angles(normals):
        return np.arccos(np.clip(np.sum(normals[a] * normals[b], axis=1), -1., 1.))

    cross0, normals0, _ = geometry(base)
    double_area = np.maximum(np.linalg.norm(cross0, axis=1), 1e-14)
    mass = (incident @ double_area) / 6
    degree = np.maximum(np.bincount(np.r_[a, b], minlength=count), 1)
    weight = .5 * (mass[a] / degree[a] + mass[b] / degree[b])
    initial = angles(normals0)
    limit = np.radians(angle_deg)
    sharp = initial > limit
    edge_weight = .10 * (mass[i] + mass[j]) / (6 * np.maximum(lengths, .08)**2)
    normal0 = cross0 / double_area[:, None]
    # Resolve bends at distances shorter than the span as well. Averaged
    # vertex normals alone can hide a corner between two coarse triangles.
    shared = np.flatnonzero(multiplicity == 2)
    order = np.argsort(inverse, kind='stable')
    first = np.cumsum(multiplicity) - multiplicity
    left = order[first[shared]] // 3
    right = order[first[shared]+1] // 3
    centers = base[faces].mean(axis=1)
    width = np.clip(np.linalg.norm(centers[left]-centers[right], axis=1), .12, 1.)
    face_limit = limit * width
    bend_weight = .30 * (double_area[left] + double_area[right]) / (6 * width**2)

    initial_hinges = np.arccos(np.clip(np.sum(normal0[left]*normal0[right], axis=1), -1., 1.))
    sharp_faces = initial_hinges > limit
    hot = np.unique(np.r_[a[sharp], b[sharp], faces[left[sharp_faces]].ravel(), faces[right[sharp_faces]].ravel()])
    free = np.zeros(count, bool)
    if len(hot):
        free = np.isfinite(dijkstra(graph, indices=hot, min_only=True, limit=1.))
    free[rim] = False
    active = np.flatnonzero(free)

    # Close each fixed open rim with a fan solely for volume measurement.
    # Cap contributions are constant because those vertices cannot move.
    cap_volume = 0.
    if len(rim):
        s, t = rim_edges.T
        rim_graph = coo_matrix((np.ones(len(s)), (s,t)), shape=(count,count))
        _, groups = connected_components(rim_graph, directed=False)
        for label in np.unique(groups[rim]):
            ids = groups[s] == label
            center = base[rim[groups[rim] == label]].mean(axis=0)
            cap_volume += np.sum(np.cross(base[t[ids]], base[s[ids]]) @ center) / 6

    def volume(x, cross):
        return cap_volume + np.sum(x[corners[0]] * cross) / 6

    volume0 = volume(base, cross0)
    if len(active) and abs(volume0) < 1e-14:
        raise ValueError('体積の基準を決められない面です。メッシュの面の向きを確認してください。')
    # For a closed mesh or a fixed boundary, the exact derivative at a free
    # vertex equals the sum of incident oriented face area vectors / 6.
    vol_scale = max(mass[free].sum(), 1e-12)
    multiplier = 0.
    penalty = 100.
    # In the ordinary solve the unknowns are the vertex displacements. If a
    # trial collapses a face, retry in scaled coordinates with an area barrier.
    unit = None
    barrier_weight = None

    def objective(params):
        offset = np.zeros_like(base)
        offset[active] = params.reshape(-1, 3)
        if unit is not None:
            offset[active] *= unit[:, None]
        x = base + offset
        cross, normals, norm = geometry(x)
        turn = angles(normals)
        excess = np.maximum(turn - limit, 0.)
        score = .5 * np.sum(weight * excess**2)
        coeff = -weight * excess / np.maximum(np.sin(turn), 1e-7)
        dn = np.empty_like(x)
        for axis in range(3):
            dn[:, axis] = (np.bincount(a, coeff * normals[b, axis], minlength=count) +
                           np.bincount(b, coeff * normals[a, axis], minlength=count))
        dn -= normals * np.sum(normals * dn, axis=1)[:, None]
        dc = incident.T @ (dn / norm[:, None])
        face_area = np.maximum(np.linalg.norm(cross, axis=1), 1e-14)
        fn = cross / face_area[:, None]
        dots = np.clip(np.sum(fn[left] * fn[right], axis=1), -1., 1.)
        hinge = np.arccos(dots)
        excess_hinge = np.maximum(hinge - face_limit, 0.)
        score += .5 * np.sum(bend_weight * excess_hinge**2)
        coeff = -bend_weight * excess_hinge / np.maximum(np.sin(hinge), 1e-7)
        for axis in range(3):
            change = np.bincount(left, coeff * (fn[right,axis] - dots*fn[left,axis]), minlength=len(faces))
            change += np.bincount(right, coeff * (fn[left,axis] - dots*fn[right,axis]), minlength=len(faces))
            dc[:,axis] += change / face_area
        # Penalize loss of projected face area before a triangle can collapse.
        ratio = np.sum(cross * normal0, axis=1) / double_area
        deficit = np.minimum(ratio - .55, 0.)
        score += 25 * np.dot(double_area, deficit**2)
        dc += 50 * deficit[:, None] * normal0
        if barrier_weight is not None:
            # C1 continuation of -log(t) + t - 1 below the cutoff keeps line
            # searches finite even when they propose crossing zero area.
            target = .4
            cutoff = .04
            t = np.minimum(ratio / target, 1.)
            safe = np.maximum(t, cutoff)
            below = np.minimum(t - cutoff, 0.)
            value = -np.log(safe) + safe - 1 + (1 - 1/cutoff)*below + .5*(below/cutoff)**2
            derivative = 1 - 1/safe + below/cutoff**2
            score += np.dot(barrier_weight, value)
            dc += (barrier_weight * derivative / (target * double_area))[:, None] * normal0
        p, q, r = x[corners]
        dq = np.cross(r-p, dc)
        dr = np.cross(dc, q-p)
        grad = np.zeros_like(x)
        for ids, values in zip(corners, (-dq-dr, dq, dr)):
            for axis in range(3):
                grad[:, axis] += np.bincount(ids, values[:, axis], minlength=count)
        # Position fidelity and smooth variation of displacement preserve the
        # ear axis and damp corrugations between the sampled points.
        score += .5 * .28 * np.sum(mass[:, None] * offset**2)
        grad += .28 * mass[:, None] * offset
        difference = offset[i] - offset[j]
        score += .5 * np.sum(edge_weight[:, None] * difference**2)
        for axis in range(3):
            flux = edge_weight * difference[:, axis]
            grad[:, axis] += np.bincount(i, flux, minlength=count) - np.bincount(j, flux, minlength=count)
        dv = volume(x, cross) - volume0
        score += multiplier * dv + .5 * penalty * dv**2 / vol_scale
        grad += (multiplier + penalty * dv / vol_scale) * (incident @ cross) / 6
        derivative = grad[active]
        if unit is not None:
            derivative *= unit[:, None]
        return score, derivative.ravel()

    solution = base.copy()
    iterations = 0

    def solve():
        nonlocal multiplier, iterations
        multiplier = 0.
        params = np.zeros(3 * len(active))
        for _ in range(4):
            result = minimize(objective, params, jac=True, method='L-BFGS-B',
                              options=dict(maxiter=100, ftol=1e-10, gtol=2e-6))
            if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
                raise ValueError("曲面の丸めで非有限な座標またはエネルギーが発生しました。")
            params = result.x
            displacement = params.reshape(-1, 3)
            if unit is not None:
                displacement = displacement * unit[:, None]
            solution[active] = base[active] + displacement
            iterations += result.nit
            delta = volume(solution, geometry(solution)[0]) - volume0
            if abs(delta) <= 1e-5 * abs(volume0) and (unit is None or result.success):
                break
            multiplier += penalty * delta / vol_scale

    def projected_area_ratio(x):
        return np.sum(geometry(x)[0] * normal0, axis=1) / double_area

    initial_bad_faces = 0
    min_trial_ratio = 1.
    if len(active):
        solve()
        ratios = projected_area_ratio(solution)
        min_trial_ratio = float(ratios.min())
        initial_bad_faces = int(np.sum(ratios < .12))
        if initial_bad_faces:
            # A single near-zero face destabilizes angular derivatives. Estimate
            # the diagonal stiffness from d(face normal)/d(vertex position):
            # its magnitude is bounded by opposite-edge length / double area.
            bend_mass = (np.bincount(left, bend_weight, minlength=len(faces))
                         + np.bincount(right, bend_weight, minlength=len(faces)))
            tri = base[faces]
            opposite2 = np.sum((np.roll(tri, 1, axis=1) - np.roll(tri, -1, axis=1))**2, axis=2)
            stiffness = opposite2 * (bend_mass / double_area**2)[:, None]
            diagonal = np.bincount(faces.ravel(), weights=stiffness.ravel(), minlength=count)
            unit = 1 / np.sqrt(np.maximum(diagonal[active], 1e-9))
            unit /= np.median(unit)
            barrier_weight = 2 * bend_mass + double_area
            solve()
            # Repair only a remaining unsafe patch and recheck its neighbors.
            for _ in range(64):
                invalid = projected_area_ratio(solution) < .12
                if not invalid.any():
                    break
                ids = np.unique(faces[invalid])
                solution[ids] = .5 * (base[ids] + solution[ids])
            else:
                raise ValueError('丸め後の一部の面で向きを維持できませんでした。')
    cross, normals, _ = geometry(solution)
    final = angles(normals)
    positions = vertices + (solution - base) * span
    shift = np.linalg.norm(positions - vertices, axis=1)
    # Give every vertex one vote regardless of its number of neighbors.
    def vertex_turn(turn):
        worst = np.zeros(count)
        np.maximum.at(worst, a, np.degrees(turn))
        np.maximum.at(worst, b, np.degrees(turn))
        return worst
    before, after = vertex_turn(initial), vertex_turn(final)
    volume_change = (volume(solution, cross)-volume0) / volume0 if volume0 else 0.
    stats = dict(span=span, threshold_degrees=angle_deg, iterations=int(iterations),
                 stabilized_retry=bool(initial_bad_faces), trial_faces_below_area_limit=initial_bad_faces,
                 min_trial_area_ratio=min_trial_ratio, min_final_area_ratio=float(projected_area_ratio(solution).min()),
                 volume_relative_change=float(volume_change), rms_move=float(np.sqrt(np.mean(shift**2))),
                 max_move=float(shift.max(initial=0)), vertices_above_before=int(np.sum(before > angle_deg)),
                 vertices_above_after=int(np.sum(after > angle_deg)), p95_before=float(np.percentile(before,95)),
                 p95_after=float(np.percentile(after,95)))
    return positions, stats, before, after


if __name__ == '__main__':
    import json
    import os
    import pickle
    from pathlib import Path
    import sys
    import trimesh

    src, directory = map(Path, sys.argv[1:])
    directory.mkdir(exist_ok=True, parents=True)
    mesh = trimesh.load(src, process=False)
    result, stats, before, after = round_profile(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), float(os.environ['TRI_PROFILE_SPAN']),
        float(os.environ['TRI_CREASE_ANGLE']))
    trimesh.Trimesh(result, mesh.faces, process=False).export(directory/'inflated.obj', digits=15)
    (directory/'inflate_summary.json').write_text(json.dumps(dict(profile_rounding=stats), indent=2))
    with (directory/'chart_inflated.pkl').open('wb') as stream:
        pickle.dump(dict(xyz=result, f3=mesh.faces), stream)
    with (directory/'inflate_fields.pkl').open('wb') as stream:
        pickle.dump(dict(V0=mesh.vertices, V=result, F=mesh.faces, turn_before=before, turn_after=after), stream)
    print(json.dumps(stats, indent=2), flush=True)
