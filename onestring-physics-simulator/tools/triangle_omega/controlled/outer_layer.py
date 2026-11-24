"""Solve a second, outward triangle mesh on the same vertex graph (millimetres)."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def check_shared_vertex_clearance(faces, normals, depth, lateral_limit, tolerance_mm):
    if tolerance_mm is None:
        return
    by_vertex = [[] for _ in range(int(faces.max()) + 1)]
    for face, vertices in enumerate(faces):
        for vertex in vertices:
            by_vertex[vertex].append(face)
    largest = 0.
    for vertex, owners in enumerate(by_vertex):
        if len(owners) < 2:
            continue
        incident = normals[owners]
        cosine = np.clip((incident @ incident.T).min(), -1., 1.)
        angle = np.arccos(cosine)
        # Even the bisector requires this much tangential displacement.
        required = max(0., depth[owners].min() - tolerance_mm) * np.tan(angle / 2)
        if required > largest:
            largest, worst, opening = required, vertex, np.degrees(angle)
    if largest > lateral_limit + 1e-7:
        lower_bound = np.ceil(largest * 100) / 100
        raise ValueError(
            f"共有頂点 {worst}（法線の開き {opening:.1f}°）で、指定厚みには"
            f"横ずれ上限を少なくとも {lower_bound:.2f} mm にする必要があります"
            f"（現在 {lateral_limit:g} mm）。この下限は平面配置や干渉を考慮する前の値です。"
        )


def outward_offsets(top, faces, frame, rotation, planar, depth, lateral_limit=1.0, tolerance_mm=None):
    # One unknown 3-vector per mesh vertex. Both owners of a seam therefore
    # receive the same outer edge. Each outer face is a single affine triangle.
    count = int(faces.max()) + 1
    objective = np.zeros((count, 3))
    rows, columns, values, limits = [], [], [], []
    lower_rows = []
    lateral_rows = []

    def constraint(vertex_ids, vectors, limit):
        columns.extend((3 * np.asarray(vertex_ids)[:, None] + np.arange(3)).ravel())
        rows.extend([len(limits)] * (3 * len(vertex_ids)))
        values.extend(np.asarray(vectors).ravel())
        limits.append(limit)

    angles = 2 * np.pi * np.arange(12) / 12
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # Inscribe the polygonal constraint in the selected lateral tolerance disk.
    cap = np.full(len(faces), lateral_limit)
    for i, verts in enumerate(faces):
        normal = frame[i, 2]
        uv = (top[i, 1:] - top[i, 0]) @ frame[i, :2].T
        # Bound slope of the outer plane relative to the inner triangle.
        gradient = np.linalg.solve(uv, np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]))
        for vertex in verts:
            objective[vertex] -= normal
            constraint([vertex], [normal], depth[i])
            minimum = min(0.001, depth[i] * 0.001)
            if tolerance_mm is not None:
                minimum = max(minimum, depth[i] - tolerance_mm)
            lower_rows.append(len(limits))
            constraint([vertex], [-normal], -minimum)
            for tangent in directions @ frame[i, :2]:
                lateral_rows.append(len(limits))
                constraint([vertex], [tangent], cap[i] * np.cos(np.pi / 12))
        for direction in directions:
            constraint(
                verts,
                (direction @ gradient)[:, None] * normal,
                np.tan(np.deg2rad(12)) * np.cos(np.pi / 12),
            )

    if planar is not None:
        polygons = [Polygon(t[:, :2]) for t in planar]
        tree = STRtree(polygons)
        ii, jj = tree.query([p.buffer(2 * cap.max() + 0.01) for p in polygons])
        for i, j in zip(ii[ii < jj], jj[ii < jj]):
            a, b = planar[i, :, :2], planar[j, :, :2]
            edges = np.vstack([np.roll(t, 1, axis=0) - t for t in (a, b)])
            axes = edges[:, ::-1] * [-1, 1]
            axes /= np.linalg.norm(axes, axis=1)[:, None]
            axes = np.vstack([axes, -axes])
            left = (a @ axes.T).max(0)
            right = (b @ axes.T).min(0)
            best = np.argmax(right - left)
            if right[best] < left[best] - 1e-7:
                raise ValueError(f"平面配置の外面が重なっています（{i}, {j}）。")
            cut = (right[best] + left[best]) / 2
            axis = axes[best]
            for face, sign in [(i, 1), (j, -1)]:
                vector = sign * (axis @ rotation[face, :2])
                for v, coord in zip(faces[face], planar[face, :, :2]):
                    constraint([v], [vector], sign * (cut - coord @ axis) + 1e-9)

    mat = coo_matrix((values, (rows, columns)), shape=(len(limits), 3 * count)).tocsr()
    if tolerance_mm is not None and tolerance_mm > 0:
        # First lift the thinnest corner. Then maximize total depth without
        # sacrificing that corner, up to the numerical tolerance of the LP.
        column = coo_matrix((np.ones(len(lower_rows)), (lower_rows, np.zeros(len(lower_rows), int))),
                            shape=(len(limits), 1))
        fair = linprog(np.r_[np.zeros(3 * count), -1.], A_ub=hstack((mat, column)),
                       b_ub=limits, bounds=[(None, None)] * (3 * count) + [(0., None)],
                       method="highs")
        if fair.success:
            limits = np.array(limits)
            limits[lower_rows] -= max(0., fair.x[-1] - 1e-7)
    result = linprog(
        objective.ravel(), A_ub=mat, b_ub=limits, bounds=(None, None), method="highs"
    )
    if result.status == 2:
        # A single nonnegative slack gives the least necessary increase of the
        # lateral warning threshold while preserving the physical constraints.
        slack = coo_matrix((-np.ones(len(lateral_rows)), (lateral_rows, np.zeros(len(lateral_rows),int))),
                           shape=(len(limits),1))
        relaxed = linprog(np.r_[np.zeros(3*count),1.], A_ub=hstack((mat,slack)),
                          b_ub=limits, bounds=[(None,None)]*(3*count)+[(0,None)], method="highs")
        if relaxed.success:
            return relaxed.x[:-1].reshape(-1,3)[faces]
    if not result.success:
        if tolerance_mm is not None and result.status == 2 and planar is None:
            from drift_preview import ThicknessConflict
            raise ThicknessConflict(
                f"組立姿勢で指定厚み（許容する薄化 {tolerance_mm:g} mm）、"
                "共有側面および外面の傾斜上限を、横ずれを自由にしても同時に満たせません。"
            )
        if tolerance_mm is not None and result.status == 2:
            raise ValueError(
                f"指定した厚み（許容する薄化 {tolerance_mm:g} mm）を全パネルの全頂点で"
                "満たせません。現在の平面配置・横ずれ上限・共有側面の条件で解がありません。"
            )
        raise ValueError(f"共有頂点で外側の層を作れません。{result.message}")
    return result.x.reshape(-1, 3)[faces]
