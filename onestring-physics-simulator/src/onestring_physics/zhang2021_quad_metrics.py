"""Quality metrics used by Zhang, Chai, Liu, Fu (CAD 2021), Sec. 4.

``Quad meshing with coarse layouts for planar domains`` reports, for every
result quad mesh Q of an input planar triangle mesh M:

* ``N_p``   number of quad patches of the quad layout (base complex),
* ``N_q``   number of quads,
* ``J_min`` / ``J_avg``  minimum / average scaled Jacobian over all quads,
* ``d_H``   two-sided Hausdorff distance between the boundaries of Q and M,
            reported as a percentage of the bounding-box diagonal of M.

This module recomputes those quantities independently of the authors' C++
code so that the reference binaries can be audited.  It is pure numpy and
contains no OneString specific logic.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------- I/O
def read_obj_polygons(path: str | Path) -> tuple[np.ndarray, list[list[int]], np.ndarray | None, list[list[int]] | None]:
    """Read v / vt / f records of an OBJ file (0-based indices).

    Returns ``(V, F, VT, FT)``; ``VT``/``FT`` are ``None`` when the file has no
    texture coordinates.  Polygons of any arity are kept as lists.
    """
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    tfaces: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                xyz = [float(x) for x in parts[1:4]]
                while len(xyz) < 3:
                    xyz.append(0.0)
                vertices.append(xyz)
            elif line.startswith("vt "):
                parts = line.split()
                texcoords.append([float(x) for x in parts[1:3]])
            elif line.startswith("f "):
                vi, ti = [], []
                for token in line.split()[1:]:
                    pieces = token.split("/")
                    vi.append(int(pieces[0]) - 1)
                    if len(pieces) > 1 and pieces[1]:
                        ti.append(int(pieces[1]) - 1)
                faces.append(vi)
                if len(ti) == len(vi):
                    tfaces.append(ti)
    V = np.asarray(vertices, float).reshape(-1, 3)
    VT = np.asarray(texcoords, float).reshape(-1, 2) if texcoords else None
    FT = tfaces if (texcoords and len(tfaces) == len(faces)) else None
    return V, faces, VT, FT


def write_obj(path: str | Path, vertices: np.ndarray, faces: Iterable[Sequence[int]],
              texcoords: np.ndarray | None = None,
              tex_faces: Iterable[Sequence[int]] | None = None) -> None:
    vertices = np.asarray(vertices, float)
    if vertices.shape[1] == 2:
        vertices = np.column_stack((vertices, np.zeros(len(vertices))))
    lines = [f"v {x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in vertices]
    faces = [list(map(int, f)) for f in faces]
    if texcoords is not None:
        lines += [f"vt {u:.17g} {v:.17g}\n" for u, v in np.asarray(texcoords, float)]
        tex_faces = [list(map(int, f)) for f in (tex_faces if tex_faces is not None else faces)]
        for f, t in zip(faces, tex_faces):
            lines.append("f " + " ".join(f"{a + 1}/{b + 1}" for a, b in zip(f, t)) + "\n")
    else:
        for f in faces:
            lines.append("f " + " ".join(str(a + 1) for a in f) + "\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- topology
def edge_face_incidence(faces: Sequence[Sequence[int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map undirected edge -> [(face, local corner index of the edge start)]."""
    owners: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for fi, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            a, b = int(face[k]), int(face[(k + 1) % n])
            owners[(min(a, b), max(a, b))].append((fi, k))
    return owners


def boundary_loops(faces: Sequence[Sequence[int]]) -> list[list[int]]:
    """Oriented boundary loops (vertex index cycles) of a polygon mesh."""
    nxt: dict[int, list[int]] = defaultdict(list)
    owners = edge_face_incidence(faces)
    for (a, b), inc in owners.items():
        if len(inc) != 1:
            continue
        fi, k = inc[0]
        face = faces[fi]
        u, v = int(face[k]), int(face[(k + 1) % len(face)])
        nxt[u].append(v)
    loops: list[list[int]] = []
    used: set[tuple[int, int]] = set()
    for start in sorted(nxt):
        for first in nxt[start]:
            if (start, first) in used:
                continue
            loop = [start]
            used.add((start, first))
            current = first
            guard = 0
            while current != start:
                loop.append(current)
                options = [w for w in nxt[current] if (current, w) not in used]
                if not options:
                    raise ValueError("boundary is not a closed manifold loop")
                w = options[0]
                used.add((current, w))
                current = w
                guard += 1
                if guard > 10_000_000:
                    raise ValueError("boundary loop tracing did not terminate")
            loops.append(loop)
    return loops


# ----------------------------------------------------------------- quality
def scaled_jacobians(vertices: np.ndarray, quads: np.ndarray, orientation: float | None = None) -> np.ndarray:
    """Per-quad minimum corner scaled Jacobian (planar, signed).

    For a corner with outgoing edges e1 (to next) and e2 (to previous), the
    scaled Jacobian is ``cross(e1, e2) / (|e1| |e2|)`` which equals ``sin``
    of the corner angle; it is negative for a flipped/concave corner.
    ``orientation`` (+1/-1) fixes the sign convention; default = sign of the
    total signed area.
    """
    P = np.asarray(vertices, float)[:, :2][np.asarray(quads, int)]
    e_next = np.roll(P, -1, axis=1) - P
    e_prev = np.roll(P, 1, axis=1) - P
    cross = e_next[..., 0] * e_prev[..., 1] - e_next[..., 1] * e_prev[..., 0]
    norm = np.linalg.norm(e_next, axis=-1) * np.linalg.norm(e_prev, axis=-1)
    if orientation is None:
        area = 0.5 * np.sum(P[..., 0] * np.roll(P[..., 1], -1, axis=1) - P[..., 1] * np.roll(P[..., 0], -1, axis=1))
        orientation = 1.0 if area >= 0 else -1.0
    sj = orientation * cross / np.maximum(norm, 1e-300)
    return np.min(sj, axis=1)


def _segments_of_loops(points: np.ndarray, loops: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for loop in loops:
        idx = np.asarray(loop, int)
        a.append(points[idx])
        b.append(points[np.roll(idx, -1)])
    return np.concatenate(a), np.concatenate(b)


def _point_to_segments(points: np.ndarray, sa: np.ndarray, sb: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.empty(len(points))
    d = sb - sa
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        w = p[:, None, :] - sa[None, :, :]
        t = np.clip(np.einsum("pij,ij->pi", w, d) / dd[None, :], 0.0, 1.0)
        proj = sa[None, :, :] + t[..., None] * d[None, :, :]
        out[start:start + chunk] = np.min(np.linalg.norm(p[:, None, :] - proj, axis=-1), axis=1)
    return out


def _densify(sa: np.ndarray, sb: np.ndarray, spacing: float) -> np.ndarray:
    pts = [sa]
    lengths = np.linalg.norm(sb - sa, axis=1)
    for a, b, length in zip(sa, sb, lengths):
        k = int(np.ceil(length / spacing))
        if k > 1:
            t = np.arange(1, k)[:, None] / k
            pts.append(a[None, :] + t * (b - a)[None, :])
    return np.concatenate(pts)


def two_sided_boundary_hausdorff(points_a: np.ndarray, loops_a: Sequence[Sequence[int]],
                                 points_b: np.ndarray, loops_b: Sequence[Sequence[int]],
                                 samples_per_min_edge: int = 4) -> tuple[float, float, float]:
    """Two-sided Hausdorff distance between two sets of boundary polylines.

    Each polyline is densified with a spacing of (shortest edge of either
    polyline) / ``samples_per_min_edge``; distances are exact point-to-segment
    distances.  Returns ``(d_H, d(A->B), d(B->A))``.
    """
    pa = np.asarray(points_a, float)[:, :2]
    pb = np.asarray(points_b, float)[:, :2]
    a0, a1 = _segments_of_loops(pa, loops_a)
    b0, b1 = _segments_of_loops(pb, loops_b)
    lengths = np.concatenate([np.linalg.norm(a1 - a0, axis=1), np.linalg.norm(b1 - b0, axis=1)])
    positive = lengths[lengths > 0]
    spacing = float(np.min(positive)) / max(1, samples_per_min_edge) if len(positive) else 1.0
    # Keep sample counts bounded for huge inputs.
    total = float(np.sum(lengths))
    spacing = max(spacing, total / 400_000.0)
    sample_a = _densify(a0, a1, spacing)
    sample_b = _densify(b0, b1, spacing)
    dab = float(np.max(_point_to_segments(sample_a, b0, b1)))
    dba = float(np.max(_point_to_segments(sample_b, a0, a1)))
    return max(dab, dba), dab, dba


# ------------------------------------------------------------ base complex
def quad_layout_patches(quads: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int]]]:
    """Number of patches of the base complex (quad layout) of a quad mesh.

    Irregular vertices are interior vertices with valence != 4 and boundary
    vertices whose face valence != 2 (i.e. corners and T-like boundary
    singularities).  Separatrices are traced from every irregular vertex
    along every incident edge, continuing straight through regular vertices
    (the edge "opposite" in the cyclic edge order) until another irregular
    vertex or the boundary is reached.  Patches are the connected components
    of faces whose shared edges are not on a separatrix.

    Returns ``(N_p, patch_id_per_quad, separatrix_edges)``.
    """
    quads = np.asarray(quads, int)
    nv = int(quads.max()) + 1 if len(quads) else 0
    owners = edge_face_incidence(quads)
    # Vertex -> ordered list of incident edges (via face corners).
    vertex_faces: list[list[tuple[int, int]]] = [[] for _ in range(nv)]
    for fi, q in enumerate(quads):
        for k in range(4):
            vertex_faces[int(q[k])].append((fi, k))
    boundary_edge = {e for e, inc in owners.items() if len(inc) == 1}
    boundary_vertex = np.zeros(nv, bool)
    for a, b in boundary_edge:
        boundary_vertex[a] = boundary_vertex[b] = True
    face_valence = np.asarray([len(v) for v in vertex_faces])
    irregular = np.zeros(nv, bool)
    for v in range(nv):
        if face_valence[v] == 0:
            continue
        if boundary_vertex[v]:
            irregular[v] = face_valence[v] != 2
        else:
            irregular[v] = face_valence[v] != 4

    def neighbors_in_cyclic_order(v: int) -> list[int]:
        # Order neighbours around v by walking faces (works for interior and
        # boundary fans of a manifold quad mesh).
        incident = vertex_faces[v]
        nxt_of: dict[int, int] = {}
        for fi, k in incident:
            q = quads[fi]
            nxt_of[int(q[(k + 1) % 4])] = int(q[(k + 3) % 4])  # out-edge -> next out-edge (ccw)
        start = None
        targets = set(nxt_of.values())
        for u in nxt_of:
            if u not in targets:
                start = u  # boundary fan start
                break
        if start is None:
            start = next(iter(nxt_of)) if nxt_of else None
        order: list[int] = []
        u = start
        while u is not None and u not in order:
            order.append(u)
            u = nxt_of.get(u)
        return order

    cyclic = {}

    def straight_continuation(prev: int, v: int) -> int | None:
        if v not in cyclic:
            cyclic[v] = neighbors_in_cyclic_order(v)
        order = cyclic[v]
        if prev not in order or len(order) != 4 or boundary_vertex[v]:
            return None
        i = order.index(prev)
        return order[(i + 2) % 4]

    separatrix: set[tuple[int, int]] = set(boundary_edge)
    for v in np.flatnonzero(irregular):
        if v not in cyclic:
            cyclic[v] = neighbors_in_cyclic_order(int(v))
        for u in cyclic[v]:
            prev, cur = int(v), int(u)
            guard = 0
            while True:
                e = (min(prev, cur), max(prev, cur))
                if e in separatrix and e not in boundary_edge and guard > 0:
                    break
                separatrix.add(e)
                if irregular[cur] or boundary_vertex[cur]:
                    break
                nxt = straight_continuation(prev, cur)
                if nxt is None:
                    break
                prev, cur = cur, nxt
                guard += 1
                if guard > 10 * len(owners) + 10:
                    break
    # Union faces across non-separatrix edges.
    parent = list(range(len(quads)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e, inc in owners.items():
        if len(inc) == 2 and e not in separatrix:
            a, b = find(inc[0][0]), find(inc[1][0])
            if a != b:
                parent[a] = b
    roots = np.asarray([find(i) for i in range(len(quads))])
    _, patch_ids = np.unique(roots, return_inverse=True)
    interior_separatrix = sorted(e for e in separatrix if e not in boundary_edge)
    return int(patch_ids.max() + 1 if len(patch_ids) else 0), patch_ids, interior_separatrix


@dataclass
class PaperMetrics:
    n_patches: int
    n_quads: int
    j_min: float
    j_avg: float
    d_h: float
    d_h_percent_bbox: float
    d_q_to_m: float
    d_m_to_q: float
    bbox_diagonal: float
    flipped_quads: int

    def as_dict(self, prefix: str = "") -> dict[str, float | int]:
        return {prefix + k: v for k, v in self.__dict__.items()}


def paper_metrics(quad_vertices: np.ndarray, quads: np.ndarray,
                  tri_vertices: np.ndarray, tris: Sequence[Sequence[int]]) -> PaperMetrics:
    """Compute (N_p, N_q, J_min, J_avg, d_H) exactly as defined in Sec. 4."""
    quads = np.asarray(quads, int)
    qv = np.asarray(quad_vertices, float)[:, :2]
    tv = np.asarray(tri_vertices, float)[:, :2]
    n_p, _, _ = quad_layout_patches(quads)
    sj = scaled_jacobians(qv, quads)
    d_h, d_qm, d_mq = two_sided_boundary_hausdorff(qv, boundary_loops(quads), tv, boundary_loops(tris))
    diag = float(np.linalg.norm(np.ptp(tv, axis=0)))
    return PaperMetrics(n_patches=n_p, n_quads=int(len(quads)), j_min=float(np.min(sj)),
                        j_avg=float(np.mean(sj)), d_h=d_h, d_h_percent_bbox=100.0 * d_h / max(diag, 1e-300),
                        d_q_to_m=d_qm, d_m_to_q=d_mq, bbox_diagonal=diag,
                        flipped_quads=int(np.sum(sj <= 0.0)))
