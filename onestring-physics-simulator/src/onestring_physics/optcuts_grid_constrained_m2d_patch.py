"""Fixed-lattice M2D for native Grid-OptCuts.

The C++ Grid-OptCuts stage has already selected the seam and placed it on the
fixed h-lattice.  This module does not invent or snap a second seam.  It:
1. builds M2D on exactly the same h/phase lattice;
2. keeps cells inside the native OptCuts UV domain;
3. extracts the *actual internal OptCuts cut edges* from paired surface/UV faces;
4. converts those already-grid-aligned seam segments to lattice edges; and
5. duplicates vertex ids by face component across those edges (zero-width cut).

Thus geometry remains coincident at a seam while M2D topology is disconnected.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from .quad_grid import create_quad_grid


def _barycentric_2d(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = np.asarray(tri, dtype=float)
    v0, v1 = b - a, c - a
    v2 = np.asarray(point, dtype=float) - a
    denom = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denom) <= 1e-14:
        return None
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denom)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denom)
    return np.asarray([1.0 - u - v, u, v], dtype=float)


def _uv_domain_data(parameterization: Any) -> dict[str, Any]:
    cached = getattr(parameterization, "_onestring_grid_domain_data", None)
    if isinstance(cached, dict):
        return cached
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    triangles = uv[uf]
    tri_lo, tri_hi = np.min(triangles, axis=1), np.max(triangles, axis=1)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for face in uf:
        ids = [int(x) for x in face]
        for i in range(3):
            edge_count[tuple(sorted((ids[i], ids[(i + 1) % 3])))] += 1
    boundary = np.asarray(
        [[uv[a], uv[b]] for (a, b), count in edge_count.items() if count == 1], dtype=float
    ) if edge_count else np.zeros((0, 2, 2), dtype=float)
    data = {
        "uv": uv,
        "uv_faces": uf,
        "triangles": triangles,
        "tri_lo": tri_lo,
        "tri_hi": tri_hi,
        "boundary_segments": boundary,
    }
    setattr(parameterization, "_onestring_grid_domain_data", data)
    return data


def _point_triangle_id(point: np.ndarray, data: dict[str, Any], tol: float = 1e-9) -> int:
    p = np.asarray(point, dtype=float)
    mask = np.all(p[None, :] >= data["tri_lo"] - tol, axis=1) & np.all(
        p[None, :] <= data["tri_hi"] + tol, axis=1
    )
    for tri_id in np.flatnonzero(mask):
        bary = _barycentric_2d(p, data["triangles"][int(tri_id)])
        if bary is not None and float(np.min(bary)) >= -tol:
            return int(tri_id)
    return -1


def _segment_hits_open_rect(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray, eps: float) -> bool:
    lower, upper = np.asarray(lo, float) + eps, np.asarray(hi, float) - eps
    if np.any(lower >= upper):
        return False
    p0, p1 = np.asarray(a, float), np.asarray(b, float)
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(2):
        if abs(float(d[axis])) <= 1e-15:
            if float(p0[axis]) <= float(lower[axis]) or float(p0[axis]) >= float(upper[axis]):
                return False
            continue
        enter = (float(lower[axis]) - float(p0[axis])) / float(d[axis])
        leave = (float(upper[axis]) - float(p0[axis])) / float(d[axis])
        if enter > leave:
            enter, leave = leave, enter
        t0, t1 = max(t0, enter), min(t1, leave)
        if t0 > t1:
            return False
    return t1 >= 0.0 and t0 <= 1.0 and t0 <= t1


def _cell_crossed_by_uv_boundary(points: np.ndarray, data: dict[str, Any], h: float) -> bool:
    pts = np.asarray(points, float)
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    eps = max(1e-10, 1e-7 * float(h))
    for a, b in np.asarray(data["boundary_segments"], float):
        seg_lo, seg_hi = np.minimum(a, b), np.maximum(a, b)
        if np.any(seg_hi < lo + eps) or np.any(seg_lo > hi - eps):
            continue
        if _segment_hits_open_rect(a, b, lo, hi, eps):
            return True
    return False


def _internal_seam_segments(parameterization: Any) -> np.ndarray:
    """Return geometric UV copies of physical surface edges actually cut by OptCuts.

    OBJ texture-coordinate ids are *not* a seam signal: two adjacent faces may
    legally use different ``vt`` ids for exactly the same UV point.  An edge is
    therefore considered cut only when the two incident faces map at least one
    of the shared physical endpoints to genuinely different UV coordinates.
    """
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    if len(sf) != len(uf):
        raise RuntimeError("OPTCUTS_GRID_M2D_FACE_CORRESPONDENCE_MISMATCH")

    incidences: dict[tuple[int, int], list[dict[int, int]]] = defaultdict(list)
    for f3, f2 in zip(sf, uf):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            sa, sb = int(f3[i]), int(f3[j])
            key = tuple(sorted((sa, sb)))
            incidences[key].append({sa: int(f2[i]), sb: int(f2[j])})

    scale = max(float(np.max(np.abs(uv))) if uv.size else 1.0, 1.0)
    coord_tol = max(1e-9 * scale, 1e-10)
    segments: list[np.ndarray] = []
    for (sa, sb), copies in incidences.items():
        if len(copies) != 2:
            continue
        c0, c1 = copies
        if sa not in c0 or sb not in c0 or sa not in c1 or sb not in c1:
            continue

        a0, b0 = uv[c0[sa]], uv[c0[sb]]
        a1, b1 = uv[c1[sa]], uv[c1[sb]]
        same_a = float(np.linalg.norm(a0 - a1)) <= coord_tol
        same_b = float(np.linalg.norm(b0 - b1)) <= coord_tol
        if same_a and same_b:
            # Different OBJ vt ids but geometrically the same UV edge: not a seam.
            continue

        segments.append(np.asarray([a0, b0], dtype=float))
        segments.append(np.asarray([a1, b1], dtype=float))

    if not segments:
        return np.zeros((0, 2, 2), dtype=float)

    # Deduplicate coincident seam copies geometrically.
    unique: dict[tuple[int, ...], np.ndarray] = {}
    dedup_tol = max(1e-10 * scale, 1e-12)
    for seg in segments:
        pts = sorted([tuple(np.rint(p / dedup_tol).astype(np.int64)) for p in seg])
        unique[tuple(pts[0] + pts[1])] = seg
    return np.asarray(list(unique.values()), dtype=float)


def _grid_cut_edges_from_segments(
    segments: np.ndarray,
    *,
    origin: np.ndarray,
    h: float,
    nx: int,
    ny: int,
) -> set[tuple[int, int]]:
    """Convert exact native seam segments to base QuadGrid edge ids."""
    origin = np.asarray(origin, dtype=float)
    tol = max(1e-7 * h, 1e-10)

    def vid(row: int, col: int) -> int:
        return int(row * (nx + 1) + col)

    cut: set[tuple[int, int]] = set()
    for seg in np.asarray(segments, dtype=float):
        a, b = seg
        ga, gb = (a - origin) / h, (b - origin) / h
        ra, rb = np.rint(ga).astype(int), np.rint(gb).astype(int)
        if np.linalg.norm(ga - ra) > tol / h or np.linalg.norm(gb - rb) > tol / h:
            raise RuntimeError(
                "OPTCUTS_GRID_NATIVE_SEAM_OFF_LATTICE: native C++ seam endpoint is not on M2D lattice; "
                f"a={a.tolist()} b={b.tolist()} grid_a={ga.tolist()} grid_b={gb.tolist()}"
            )
        x0, y0 = int(ra[0]), int(ra[1])
        x1, y1 = int(rb[0]), int(rb[1])
        if y0 == y1 and x0 != x1:
            row = y0
            if 0 <= row <= ny:
                for col in range(min(x0, x1), max(x0, x1)):
                    if 0 <= col < nx:
                        cut.add(tuple(sorted((vid(row, col), vid(row, col + 1)))))
        elif x0 == x1 and y0 != y1:
            col = x0
            if 0 <= col <= nx:
                for row in range(min(y0, y1), max(y0, y1)):
                    if 0 <= row < ny:
                        cut.add(tuple(sorted((vid(row, col), vid(row + 1, col)))))
        else:
            raise RuntimeError(
                "OPTCUTS_GRID_NATIVE_SEAM_NOT_ORTHOGONAL: native seam segment is neither horizontal nor vertical "
                f"in fabrication frame; a={a.tolist()} b={b.tolist()}"
            )
    return cut


def _disconnect_faces_along_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    cut_edges: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Zero-width topological cut: duplicate shared vertex ids by face component."""
    verts = np.asarray(vertices, dtype=float).copy()
    out_faces = np.asarray(faces, dtype=int).copy()
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(out_faces):
        ids = [int(x) for x in face]
        for i in range(len(ids)):
            edge_to_faces[tuple(sorted((ids[i], ids[(i + 1) % len(ids)])))].append(fi)
    adjacency: list[set[int]] = [set() for _ in range(len(out_faces))]
    active_cut_edges = 0
    for edge, touching in edge_to_faces.items():
        if edge in cut_edges:
            if len(touching) >= 2:
                active_cut_edges += 1
            continue
        for i in range(len(touching)):
            for j in range(i + 1, len(touching)):
                adjacency[touching[i]].add(touching[j])
                adjacency[touching[j]].add(touching[i])
    component = np.full(len(out_faces), -1, dtype=int)
    comp_count = 0
    for root in range(len(out_faces)):
        if component[root] >= 0:
            continue
        component[root] = comp_count
        q = deque([root])
        while q:
            cur = q.popleft()
            for nxt in adjacency[cur]:
                if component[nxt] < 0:
                    component[nxt] = comp_count
                    q.append(nxt)
        comp_count += 1

    vertex_components: dict[int, set[int]] = defaultdict(set)
    for fi, face in enumerate(out_faces):
        for v in face:
            vertex_components[int(v)].add(int(component[fi]))
    replacement: dict[tuple[int, int], int] = {}
    duplicate_count = 0
    for old, comps in vertex_components.items():
        ordered = sorted(comps)
        for comp in ordered[1:]:
            replacement[(old, comp)] = len(verts)
            verts = np.vstack([verts, verts[old]])
            duplicate_count += 1
    for fi, face in enumerate(out_faces):
        comp = int(component[fi])
        for li, old in enumerate(face):
            new = replacement.get((int(old), comp))
            if new is not None:
                out_faces[fi, li] = new
    return verts, out_faces, {
        "active_cut_edges": int(active_cut_edges),
        "duplicated_vertices": int(duplicate_count),
        "face_components": int(comp_count),
    }


def install_optcuts_grid_constrained_m2d_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_grid_constrained_m2d_installed", False):
        return
    base_flatten = pipeline._flatten_to_domain
    base_build = pipeline._build_m2d

    def flatten_on_fixed_lattice(parameterization: Any, grid: Any, params: Any = None):
        domain = base_flatten(parameterization, grid, params)
        if not bool(getattr(parameterization, "metrics", {}).get("optcuts_grid_constrained", False)):
            return domain
        h = max(float(getattr(params, "tile_size", getattr(grid, "tile_size", 0.0))), 1e-8)
        uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
        margin = int(max(0, getattr(params, "omega_overlay_margin", 1))) if params is not None else 1
        metrics = dict(getattr(parameterization, "metrics", {}) or {})
        phase = np.asarray([
            float(metrics.get("grid_phase_u", 0.0)), float(metrics.get("grid_phase_v", 0.0))
        ], dtype=float)
        lo = phase + np.floor((np.min(uv, axis=0) - phase) / h) * h - margin * h
        hi = phase + np.ceil((np.max(uv, axis=0) - phase) / h) * h + margin * h
        nx = max(1, int(round(float((hi[0] - lo[0]) / h))))
        ny = max(1, int(round(float((hi[1] - lo[1]) / h))))
        xs, ys = lo[0] + np.arange(nx + 1) * h, lo[1] + np.arange(ny + 1) * h
        uu, vv = np.meshgrid(xs, ys, indexing="xy")
        domain.uv_vertices = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        domain.boundary = np.asarray(parameterization.omega_boundary, dtype=float)
        domain.split_lines = []
        domain.overlay_nx, domain.overlay_ny = int(nx), int(ny)
        domain.overlay_margin_tiles = int(margin)
        domain.overlay_step_u = domain.overlay_step_v = float(h)
        domain.original_requested_nx = int(getattr(grid, "nx", nx))
        domain.original_requested_ny = int(getattr(grid, "ny", ny))
        setattr(domain, "_optcuts_grid_constrained_parameterization", parameterization)
        setattr(domain, "_optcuts_grid_origin", np.asarray(lo, dtype=float))
        setattr(domain, "_optcuts_grid_phase", phase.copy())
        setattr(domain, "_optcuts_grid_unit", float(h))
        print(
            "[OPTCUTS-GRID-DOMAIN] "
            f"h={h:g} phase=({phase[0]:.6g},{phase[1]:.6g}) "
            f"origin=({lo[0]:.6g},{lo[1]:.6g}) grid={nx}x{ny} legacy_split=disabled"
        )
        return domain

    def build_from_constrained_uv(grid: Any, domain: Any, params: Any = None):
        parameterization = getattr(domain, "_optcuts_grid_constrained_parameterization", None)
        if parameterization is None:
            return base_build(grid, domain, params)
        h = float(getattr(domain, "_optcuts_grid_unit"))
        nx, ny = int(domain.overlay_nx), int(domain.overlay_ny)
        overlay_grid = create_quad_grid(nx, ny, h, float(getattr(grid, "gap_size", 0.0)))
        vertices = np.column_stack([np.asarray(domain.uv_vertices, float), np.zeros(len(domain.uv_vertices))])
        data = _uv_domain_data(parameterization)
        kept: list[tuple[int, int, int, int]] = []
        kept_triangle_ids: list[int] = []
        outside = crossing = 0
        for tile in overlay_grid.tiles or []:
            face = tuple(int(x) for x in tile.vertex_ids)
            pts = vertices[np.asarray(face), :2]
            tri_id = _point_triangle_id(np.mean(pts, axis=0), data)
            if tri_id < 0:
                outside += 1
                continue
            if _cell_crossed_by_uv_boundary(pts, data, h):
                crossing += 1
                continue
            kept.append(face)
            kept_triangle_ids.append(int(tri_id))
        if not kept:
            raise RuntimeError("OPTCUTS_GRID_M2D_EMPTY: fixed lattice contains no valid UV cells")
        faces = np.asarray(kept, dtype=int)

        seam_segments = _internal_seam_segments(parameterization)
        cut_edges = _grid_cut_edges_from_segments(
            seam_segments,
            origin=np.asarray(getattr(domain, "_optcuts_grid_origin"), dtype=float),
            h=h, nx=nx, ny=ny,
        )
        base_vertex_count = len(vertices)
        if cut_edges:
            vertices, faces, cut_info = _disconnect_faces_along_edges(vertices, faces, cut_edges)
        else:
            cut_info = {"active_cut_edges": 0, "duplicated_vertices": 0, "face_components": 1}

        metrics = {
            "optcuts_grid_constrained_m2d": True,
            "optcuts_grid_unit": float(h),
            "optcuts_grid_origin": np.asarray(getattr(domain, "_optcuts_grid_origin"), float).tolist(),
            "optcuts_grid_phase": np.asarray(getattr(domain, "_optcuts_grid_phase"), float).tolist(),
            "optcuts_grid_nx": nx, "optcuts_grid_ny": ny,
            "optcuts_grid_total_cell_count": int(nx * ny),
            "optcuts_grid_kept_cell_count": int(len(faces)),
            "optcuts_grid_outside_cell_count": int(outside),
            "optcuts_grid_boundary_crossing_cell_count": int(crossing),
            "optcuts_grid_base_vertex_count": int(base_vertex_count),
            "optcuts_grid_native_seam_segment_count": int(len(seam_segments)),
            "optcuts_grid_native_requested_cut_edge_count": int(len(cut_edges)),
            "optcuts_grid_native_active_cut_edge_count": int(cut_info["active_cut_edges"]),
            "optcuts_grid_native_duplicated_vertex_count": int(cut_info["duplicated_vertices"]),
            "optcuts_grid_native_face_component_count": int(cut_info["face_components"]),
            "optcuts_grid_zero_width_topology_cut": True,
            "optcuts_grid_vertex_ids_preserved": False,
            "optcuts_grid_posthoc_seam_snap": False,
            "optcuts_grid_posthoc_seam_cell_deletion": False,
            "optcuts_grid_seam_aligned_before_m2d": True,
            "number_of_splits": 0,
            "split_locations": [],
            "m2d_grid_overlay": "same fixed h-lattice/phase as native Grid-OptCuts; actual native seam transferred topologically",
        }
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or getattr(pipeline, "QuadMesh")
        out = cls(vertices, faces, overlay_grid, "M2D", metrics, [])
        setattr(out, "_optcuts_grid_cell_triangle_ids", np.asarray(kept_triangle_ids, dtype=int))
        print(
            "[OPTCUTS-GRID-M2D] "
            f"kept={len(faces)} outside={outside} boundary_crossing={crossing} h={h:g} "
            f"seam_segments={len(seam_segments)} cut_edges={cut_info['active_cut_edges']} "
            f"duplicated_vertices={cut_info['duplicated_vertices']} components={cut_info['face_components']}"
        )
        return out

    pipeline._flatten_to_domain = flatten_on_fixed_lattice
    pipeline._build_m2d = build_from_constrained_uv
    original = getattr(pipeline, "_original", None)
    if original is not None:
        original._flatten_to_domain = flatten_on_fixed_lattice
        original._build_m2d = build_from_constrained_uv
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(original, "build_onestring_design", None) if original is not None else None,
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_flatten_to_domain"] = flatten_on_fixed_lattice
            glb["_build_m2d"] = build_from_constrained_uv
    pipeline._onestring_optcuts_grid_constrained_m2d_installed = True


__all__ = ["install_optcuts_grid_constrained_m2d_patch"]
