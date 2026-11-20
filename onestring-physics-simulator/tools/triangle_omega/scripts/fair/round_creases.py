"""Round local convex ridges and concave valleys on a triangle surface."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import trimesh


def round_creases(vertices, faces, threshold_deg, radius, radius_violations, inflate_residual=False):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    left, right = mesh.face_adjacency.T
    hinges = mesh.face_adjacency_edges
    edges = mesh.edges_unique
    fixed = np.zeros(len(vertices), bool)
    fixed[edges[np.bincount(mesh.edges_unique_inverse) == 1].ravel()] = True
    i, j = np.concatenate([edges, edges[:, ::-1]]).T
    lengths = np.linalg.norm(vertices[i] - vertices[j], axis=1)
    weights = 1 / np.maximum(lengths, 1e-14)
    count = len(vertices)
    row_weights = np.bincount(i, weights=weights, minlength=count)
    average = csr_matrix((weights / row_weights[i], (i, j)), shape=(count, count))
    local_size = np.bincount(i, weights=lengths, minlength=count) / np.maximum(np.bincount(i, minlength=count), 1)
    original_cross = mesh.triangles_cross
    original_area2 = np.einsum('ij,ij->i', original_cross, original_cross)

    def geometry(x):
        tri = x[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-20)
        dots = np.sum(normals[left] * normals[right], axis=1)
        return cross, np.degrees(np.arccos(np.clip(dots, -1., 1.)))

    def energy(angles):
        return np.sum(np.maximum(angles - threshold_deg, 0.) ** 2)

    x = vertices.copy()
    _, initial = geometry(x)
    angle_limit = np.maximum(threshold_deg, initial) + .01
    angles = initial
    iterations = 0
    for _ in range(48):
        active = angles > threshold_deg + .05
        if not active.any():
            break
        influence = np.zeros(count)
        strength = np.clip((angles[active] - threshold_deg) / 15., .15, 1.)
        np.maximum.at(influence, hinges[active].ravel(), np.repeat(strength, 2))
        # Taper into both neighboring rings to spread a ridge across several faces.
        for _ring in range(2):
            np.maximum.at(influence, i, .6 * influence[j])
        influence[fixed] = 0.
        delta = .4 * influence[:, None] * (average @ x - x)
        score = energy(angles)
        for power in range(10):
            move = x + (.5 ** power) * delta - vertices
            scale = np.minimum(1., local_size / np.maximum(np.linalg.norm(move, axis=1), 1e-20))
            trial = vertices + move * scale[:, None]
            # Restore neighborhoods where rounding would collapse a face,
            # sharpen an existing crease, or violate the tip radius.
            for _repair in range(8):
                cross, new_angles = geometry(trial)
                invalid = np.sum(original_cross * cross, axis=1) < .12 * original_area2
                worsened = new_angles > angle_limit
                if invalid.any() or worsened.any():
                    touched = np.r_[faces[invalid].ravel(), faces[left[worsened]].ravel(), faces[right[worsened]].ravel()]
                    trial[touched] = x[touched]
                    continue
                bad = radius_violations(trial)
                if not len(bad):
                    break
                tree = cKDTree(trial)
                for group in tree.query_ball_point(trial[bad], radius + lengths.max()):
                    trial[group] = x[group]
            else:
                continue
            new_score = energy(new_angles)
            if new_score < score - 1e-8:
                break
        else:
            break
        x, angles = trial, new_angles
        iterations += 1
        if score - new_score < 1e-6 * max(1., score):
            break
    pre_inflate = angles.copy()
    bulge = np.zeros(count)
    inflate_steps = 0
    if inflate_residual and np.any(angles > threshold_deg):
        base = x.copy()
        cross, _ = geometry(base)
        normals = np.zeros_like(x)
        for corner in faces.T:
            np.add.at(normals, corner, cross)
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
        # The input face winding supplies the outward direction, as in the tip offset.
        # Holding those directions fixed makes each accepted displacement outward.
        for _ in range(64):
            cross, _ = geometry(x)
            area = np.maximum(np.linalg.norm(cross, axis=1), 1e-20)
            unit = cross / area[:, None]
            active = angles > threshold_deg
            if not active.any():
                break
            a, b = left[active], right[active]
            theta = np.radians(angles[active])
            # Differentiate the squared angular excess with respect to each
            # face normal, then to its three corner positions.
            weight = np.radians(angles[active] - threshold_deg + .25) / np.maximum(np.sin(theta), 1e-8)
            descent = np.zeros_like(cross)
            cosine = np.cos(theta)[:, None]
            np.add.at(descent, a, weight[:, None] * (unit[b] - cosine * unit[a]) / area[a, None])
            np.add.at(descent, b, weight[:, None] * (unit[a] - cosine * unit[b]) / area[b, None])
            tri = x[faces]
            d1 = np.cross(tri[:, 2] - tri[:, 0], descent)
            d2 = np.cross(descent, tri[:, 1] - tri[:, 0])
            disp = np.zeros_like(x)
            for corner, value in zip(faces.T, (-d1-d2, d1, d2)):
                np.add.at(disp, corner, value)
            drive = np.maximum(np.sum(disp * normals, axis=1) * local_size, 0.)
            drive = drive / (.1 + drive)
            drive[fixed] = 0.
            # Taper the outward offset into the neighboring ring; fix the open rim.
            step = .15 * local_size * (.8 * drive + .2 * (average @ drive))
            step[fixed] = 0.
            score = energy(angles)
            for power in range(10):
                trial_bulge = np.minimum(bulge + step * (.5 ** power), .65 * local_size)
                for _repair in range(10):
                    trial = base + trial_bulge[:, None] * normals
                    new_cross, new_angles = geometry(trial)
                    invalid = np.sum(original_cross * new_cross, axis=1) < .12 * original_area2
                    # Leave already compliant edges within tolerance, and preserve the
                    # best maximum fold angle reached by ordinary crease rounding.
                    worse = new_angles > np.maximum(pre_inflate, threshold_deg) + 1e-7
                    if invalid.any() or worse.any():
                        touched = np.r_[faces[invalid].ravel(), faces[left[worse]].ravel(), faces[right[worse]].ravel()]
                        trial_bulge[touched] = bulge[touched]
                        continue
                    if energy(new_angles) >= score - 1e-8:
                        break
                    bad = radius_violations(trial)
                    if not len(bad):
                        break
                    tree = cKDTree(trial)
                    for group in tree.query_ball_point(trial[bad], radius + lengths.max()):
                        trial_bulge[group] = bulge[group]
                else:
                    continue
                if energy(new_angles) < score - 1e-8:
                    break
            else:
                break
            x, angles, bulge = trial, new_angles, trial_bulge
            inflate_steps += 1
    return x, dict(
        threshold_degrees=threshold_deg, iterations=iterations,
        residual_inflation=inflate_residual, inflation_steps=inflate_steps,
        edges_above_after_rounding=int(np.sum(pre_inflate > threshold_deg)),
        max_degrees_after_rounding=float(pre_inflate.max(initial=0)),
        max_bulge=float(bulge.max(initial=0)),
        edges_above_before=int(np.sum(initial > threshold_deg)), edges_above_after=int(np.sum(angles > threshold_deg)),
        max_degrees_before=float(initial.max(initial=0)), max_degrees_after=float(angles.max(initial=0)),
        rms_excess_before=float(np.sqrt(energy(initial) / len(initial))) if len(initial) else 0.,
        rms_excess_after=float(np.sqrt(energy(angles) / len(angles))) if len(angles) else 0.,
        max_shift=float(np.linalg.norm(x - vertices, axis=1).max(initial=0)),
    )
