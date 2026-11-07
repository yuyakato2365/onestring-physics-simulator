"""2026-11-06: a drawstring through the outer boundary of the flat layout (shared by the IPC simulation and the
interactive assembly page).

The string runs once around the outer boundary loop of the flat mesh (M2D; one closed loop, slits included).  It
passes through every boundary tile; between two consecutive boundary tiles it crosses the gap between them, from the
side face of the first tile to the side face of the second that face each other there (the side of each tile that
holds the boundary vertex they share and is not itself on the boundary).  Pulling the string shortens these
crossings, i.e. closes the gaps along the boundary, like gathering a drawstring bag.  The string slides freely
through the tiles, so its tension is the same everywhere; the part of its length inside the tiles does not change.

Tile corner k of a T2D tile (top k, bottom k + 4) is vertex k of the matching M2D face, so tile side k is the face
made of tile vertices (k, k+1, k+5, k+4).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def boundary_loop(faces: np.ndarray) -> list[tuple[int, int, int, int]]:
    """The longest boundary loop of a quad mesh as ordered (face, side, v_from, v_to)."""
    F = np.asarray(faces, int)
    count = Counter()
    owner = {}
    for fi, f in enumerate(F):
        for k in range(4):
            a, b = int(f[k]), int(f[(k + 1) % 4])
            count[(min(a, b), max(a, b))] += 1
            owner[(a, b)] = (fi, k)
    nxt = {}
    for (a, b), (fi, k) in owner.items():
        if count[(min(a, b), max(a, b))] == 1:
            nxt.setdefault(a, []).append((b, fi, k))
    loops, used = [], set()
    for start in list(nxt):
        for first in nxt[start]:
            if (start, first[0]) in used:
                continue
            loop, a, step = [], start, first
            while True:
                b, fi, k = step
                if (a, b) in used:
                    break
                used.add((a, b))
                loop.append((fi, k, a, b))
                a = b
                cand = [s for s in nxt.get(a, []) if (a, s[0]) not in used]
                if not cand:
                    break
                step = cand[0]
            if loop:
                loops.append(loop)
    return max(loops, key=len) if loops else []


def side_face(k: int) -> tuple[int, int, int, int]:
    return (k, (k + 1) % 4, (k + 1) % 4 + 4, k + 4)


def seam_edges(state, loop) -> dict:
    """Boundary edges that lie on a seam (a cut of the flat layout), each mapped to the loop index of its partner edge
    on the other side of the cut.  The two sides of a cut share vertex positions in the assembled mesh (K3D)."""
    from scipy.spatial import cKDTree
    V = np.asarray(state.mesh_3d_optimized.vertices, float)
    twin = {}
    for a, b in cKDTree(V).query_pairs(1e-7 * float(np.ptp(V))):
        twin[a], twin[b] = b, a
    where = {(a, b): i for i, (_, _, a, b) in enumerate(loop)}
    out = {}
    for i, (_, _, a, b) in enumerate(loop):
        if a in twin and b in twin and (twin[b], twin[a]) in where:
            out[i] = where[(twin[b], twin[a])]
    return out


def drawstring_segments(state, route: str = "loop") -> list[dict]:
    """The gap crossings of the drawstring: dict(a, face_a, b, face_b) with face_* = 4 tile vertex ids.

    route "loop": once around the whole outer boundary in loop order (along both sides of a seam, round its end).
    route "seam": around the outer boundary, but across each seam the string laces the two sides together: it crosses
    from every seam tile to the tile facing it on the other side of the cut (their cut faces meet in the assembled
    shape), instead of running down one side of the cut and back up the other.  Without a seam, both are the same."""
    F = np.asarray(state.mesh_2d_initial.faces, int)
    loop = boundary_loop(F)
    if not loop:
        return []
    seam = seam_edges(state, loop) if route == "seam" else {}
    inside = set()                       # loop positions from a seam's mouth to the matching edge on its other side
    for i, j in seam.items():
        if i < j:
            inside.update(range(i, j + 1))
    boundary_sides = defaultdict(set)
    for fi, k, _, _ in loop:
        boundary_sides[fi].add(k)
    segs = []
    m = len(loop)
    for i in range(m):
        fa, ka, _, v = loop[i]
        fb, kb, v2, _ = loop[(i + 1) % m]
        if fa == fb or (i in inside and (i + 1) % m in inside):
            continue

        def facing(fi, vertex):
            f = list(F[fi])
            j = f.index(vertex)
            for k in (j, (j - 1) % 4):        # the two sides of face fi that contain this vertex
                if k not in boundary_sides[fi]:
                    return side_face(k)
            return side_face(j)
        segs.append(dict(a=int(fa), face_a=facing(fa, v), b=int(fb), face_b=facing(fb, v)))
    for i, j in seam.items():
        if i < j:
            (fa, ka, _, _), (fb, kb, _, _) = loop[i], loop[j]
            if fa != fb:
                segs.append(dict(a=int(fa), face_a=side_face(ka), b=int(fb), face_b=side_face(kb), seam=True))
    return segs
