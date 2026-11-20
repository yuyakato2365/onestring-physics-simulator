"""Topology guards for the geometric preprocessing before Botsch remeshing."""
import numpy as np
import trimesh


def require_manifold(vertices, faces, stage):
    """Check edges and vertex links, including open surfaces (not watertightness)."""
    V, F = np.asarray(vertices), np.asarray(faces, dtype=int)
    if not len(F) or not np.isfinite(V).all():
        raise ValueError(f"{stage}: empty mesh or non-finite coordinates")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError(f"{stage}: invalid vertex index")
    if np.any(np.diff(np.sort(F, axis=1), axis=1) == 0):
        raise ValueError(f"{stage}: collapsed triangle")
    if len(np.unique(np.sort(F, axis=1), axis=0)) != len(F):
        raise ValueError(f"{stage}: duplicate triangles")
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError(f"{stage}: {int(np.sum(counts > 2))} non-manifold edges")
    links = {}
    for face in F:
        for k, v in enumerate(face):
            links.setdefault(int(v), []).append((int(face[(k+1) % 3]), int(face[(k+2) % 3])))
    for vertex, pairs in links.items():
        adjacency = {}
        for a, b in pairs:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        degrees = [len(ns) for ns in adjacency.values()]
        seen, todo = set(), [next(iter(adjacency))]
        while todo:
            v = todo.pop()
            if v not in seen:
                seen.add(v)
                todo.extend(adjacency[v] - seen)
        if len(seen) != len(adjacency) or max(degrees) > 2 or degrees.count(1) not in (0, 2):
            raise ValueError(f"{stage}: non-manifold vertex {vertex}")


def clean_mesh(vertices, faces, h):
    """Weld triangle-soup copies and remove collapsed faces, without filling holes."""
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    span = max(float(np.max(np.ptp(mesh.vertices, axis=0))), 1e-30)
    digits = int(np.clip(np.ceil(-np.log10(span * 1e-10)), 0, 15))
    mesh.merge_vertices(digits_vertex=digits)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    P = V[F]
    area2 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    good = np.all(np.diff(np.sort(F, axis=1), axis=1) != 0, axis=1) & (area2 > 2e-12 * h*h)
    mesh.update_faces(good)
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)


def prepare_half(vertices, faces, normal, offset, h):
    """Keep broad snapping only when it preserves a manifold mesh and half-mesh.

    A dense pole can contain multiple vertex rows in the 0.3*h snapping band.
    Projecting all of them onto the mirror plane folds their fans onto each other.
    In that case retry a direct plane cut, projecting numerical plane coincidences
    only. Never pass the broken snapped mesh to the native remesher.
    """
    V, F = clean_mesh(vertices, faces, h)
    require_manifold(V, F, "input after STL vertex welding")
    normal = np.asarray(normal, dtype=float)
    distance = V @ normal - offset
    epsilon = 1e-10 * max(float(np.max(np.ptp(V, axis=0))), h)
    reasons = []
    for band in (0.3 * h, epsilon):
        near = np.abs(distance) <= band
        moved = V.copy()
        moved[near] -= np.outer(distance[near], normal)
        try:
            A, B = clean_mesh(moved, F, h)
            require_manifold(A, B, "after mirror-plane snapping")
            half = trimesh.intersections.slice_mesh_plane(
                trimesh.Trimesh(A, B, process=False), plane_normal=normal,
                plane_origin=normal * offset, cap=False)
            H, T = clean_mesh(half.vertices, half.faces, h)
            dist = H @ normal - offset
            on = np.abs(dist) <= epsilon
            H[on] -= np.outer(dist[on], normal)
            require_manifold(H, T, "after mirror-plane cut")
            return H, T, {"broad_snap": band > epsilon, "snapped_vertices": int(near.sum()),
                          "rejected_attempts": reasons}
        except ValueError as exc:
            reasons.append(str(exc))
    raise ValueError("Cannot prepare a manifold symmetric half: " + "; ".join(reasons))
