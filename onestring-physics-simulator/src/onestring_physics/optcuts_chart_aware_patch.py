"""Chart-aware OptCuts M2D and M3D mapping.

OptCuts' UV embedding is already discontinuous along the cuts chosen by OptCuts.
A later fabrication seam cannot move that discontinuity by merely changing M2D
connectivity.  This patch therefore treats the OptCuts UV charts as authoritative
for inverse mapping:

* UV triangles are split into connected chart components by shared UV edges.
* Every M2D quad must lie completely inside one chart; chart-crossing quads are
  removed before M3D.
* M2D vertices shared by faces from different charts are duplicated so each
  vertex has one unambiguous chart id.
* M2D -> M3D inverse mapping is restricted to that vertex's chart.

The rectilinear OneString seam remains an additional zero-width fabrication cut;
it no longer pretends to relocate the discontinuity of the original OptCuts map.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from .optcuts_grid_seam_patch import _make_quadmesh


def _barycentric_2d(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    a, b, c = np.asarray(tri, dtype=float)
    v0 = b - a
    v1 = c - a
    v2 = np.asarray(point, dtype=float) - a
    denom = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(denom) <= 1e-14:
        return None
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / denom)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / denom)
    return np.asarray([1.0 - u - v, u, v], dtype=float)


def _chart_data(parameterization: Any) -> dict[str, Any]:
    cached = getattr(parameterization, "_onestring_optcuts_chart_data", None)
    if isinstance(cached, dict):
        return cached
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(uf):
        ids = [int(x) for x in face]
        for i in range(3):
            edge_faces[tuple(sorted((ids[i], ids[(i + 1) % 3])))].append(int(fi))
    adjacency = [set() for _ in range(len(uf))]
    for touching in edge_faces.values():
        if len(touching) != 2:
            continue
        a, b = touching
        adjacency[a].add(b)
        adjacency[b].add(a)
    face_chart = np.full(len(uf), -1, dtype=int)
    charts: list[np.ndarray] = []
    for root in range(len(uf)):
        if face_chart[root] >= 0:
            continue
        cid = len(charts)
        q = deque([root])
        face_chart[root] = cid
        ids: list[int] = []
        while q:
            cur = q.popleft()
            ids.append(int(cur))
            for nxt in adjacency[cur]:
                if face_chart[nxt] < 0:
                    face_chart[nxt] = cid
                    q.append(nxt)
        charts.append(np.asarray(ids, dtype=int))
    chart_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for ids in charts:
        tris = uv[uf[ids]]
        chart_boxes.append((np.min(tris.reshape(-1, 2), axis=0), np.max(tris.reshape(-1, 2), axis=0)))
    data = {"face_chart": face_chart, "charts": charts, "chart_boxes": chart_boxes}
    setattr(parameterization, "_onestring_optcuts_chart_data", data)
    return data


def _point_in_chart(point: np.ndarray, chart_id: int, parameterization: Any, tol: float = 1e-9) -> tuple[bool, int, np.ndarray | None]:
    data = _chart_data(parameterization)
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    uf = np.asarray(parameterization.uv_faces, dtype=int)
    p = np.asarray(point, dtype=float)[:2]
    ids = np.asarray(data["charts"][int(chart_id)], dtype=int)
    lo, hi = data["chart_boxes"][int(chart_id)]
    if np.any(p < lo - tol) or np.any(p > hi + tol):
        return False, -1, None
    best: tuple[float, int, np.ndarray] | None = None
    for fi in ids:
        tri = uv[uf[int(fi)]]
        if np.any(p < np.min(tri, axis=0) - tol) or np.any(p > np.max(tri, axis=0) + tol):
            continue
        bary = _barycentric_2d(p, tri)
        if bary is None:
            continue
        minimum = float(np.min(bary))
        if minimum >= -tol:
            return True, int(fi), bary
        score = abs(minimum)
        if best is None or score < best[0]:
            best = (score, int(fi), bary)
    return False, -1, None


def _quad_chart(points: np.ndarray, parameterization: Any) -> int | None:
    data = _chart_data(parameterization)
    pts = np.asarray(points, dtype=float)[:, :2]
    center = np.mean(pts, axis=0)
    # Center first makes this cheap for the common case.
    candidate: list[int] = []
    for cid in range(len(data["charts"])):
        inside, _fi, _bary = _point_in_chart(center, cid, parameterization)
        if inside:
            candidate.append(cid)
    for cid in candidate:
        if all(_point_in_chart(p, cid, parameterization)[0] for p in pts):
            return int(cid)
    return None


def _duplicate_vertices_by_face_chart(vertices: np.ndarray, faces: np.ndarray, face_charts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    verts = np.asarray(vertices, dtype=float).copy()
    out = np.asarray(faces, dtype=int).copy()
    incident: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for fi, face in enumerate(out):
        cid = int(face_charts[fi])
        for vid in face:
            incident[int(vid)][cid].append(int(fi))
    vertex_chart = np.full(len(verts), -1, dtype=int)
    duplicate_count = 0
    for vid, by_chart in incident.items():
        chart_ids = sorted(by_chart)
        if not chart_ids:
            continue
        vertex_chart[int(vid)] = int(chart_ids[0])
        for cid in chart_ids[1:]:
            new_id = len(verts)
            verts = np.vstack([verts, verts[int(vid)]])
            vertex_chart = np.append(vertex_chart, int(cid))
            duplicate_count += 1
            for fi in by_chart[int(cid)]:
                mask = out[int(fi)] == int(vid)
                out[int(fi), mask] = int(new_id)
    # Fill any unused ids conservatively from the first incident face.
    for fi, face in enumerate(out):
        cid = int(face_charts[fi])
        for vid in face:
            if vertex_chart[int(vid)] < 0:
                vertex_chart[int(vid)] = cid
    return verts, out, vertex_chart, int(duplicate_count)


def _inverse_in_chart(point: np.ndarray, chart_id: int, parameterization: Any) -> tuple[np.ndarray, int]:
    inside, fi, bary = _point_in_chart(np.asarray(point, dtype=float)[:2], int(chart_id), parameterization)
    if not inside or fi < 0 or bary is None:
        raise RuntimeError(f"OPTCUTS_CHART_INVERSE_FAILED: chart={chart_id} uv={np.asarray(point)[:2].tolist()}")
    sf = np.asarray(parameterization.surface_faces, dtype=int)
    xyz = np.asarray(parameterization.surface_vertices_3d, dtype=float)
    return np.asarray(bary @ xyz[sf[int(fi)]], dtype=float), int(fi)


def install_optcuts_chart_aware_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_chart_aware_patch_installed", False):
        return

    base_build_m2d = pipeline._build_m2d
    base_lift = pipeline._lift_m2d_to_m3d

    def build_chart_aware_m2d(grid: Any, domain: Any, params: Any = None):
        mesh = base_build_m2d(grid, domain, params)
        parameterization = getattr(domain, "_optcuts_parameterization", None)
        if parameterization is None or str(getattr(parameterization, "method", "")) != "optcuts":
            return mesh
        data = _chart_data(parameterization)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        keep_faces: list[np.ndarray] = []
        face_charts: list[int] = []
        rejected: list[int] = []
        for fi, face in enumerate(faces):
            cid = _quad_chart(vertices[np.asarray(face, dtype=int)], parameterization)
            if cid is None:
                rejected.append(int(fi))
                continue
            keep_faces.append(np.asarray(face, dtype=int))
            face_charts.append(int(cid))
        if not keep_faces:
            raise RuntimeError("OPTCUTS_CHART_AWARE_M2D_EMPTY: no quad lies wholly inside one OptCuts UV chart")
        kept = np.asarray(keep_faces, dtype=int)
        chart_arr = np.asarray(face_charts, dtype=int)
        new_vertices, new_faces, vertex_charts, duplicated = _duplicate_vertices_by_face_chart(
            vertices, kept, chart_arr
        )
        metrics = dict(getattr(mesh, "metrics", {}) or {})
        metrics.update({
            "optcuts_chart_aware_m2d": True,
            "optcuts_uv_chart_count": int(len(data["charts"])),
            "optcuts_chart_crossing_quad_removed_count": int(len(rejected)),
            "optcuts_chart_crossing_quad_removed_ids_sample": rejected[:128],
            "optcuts_chart_vertex_duplicate_count": int(duplicated),
            "optcuts_chart_aware_face_count": int(len(new_faces)),
        })
        out = _make_quadmesh(pipeline, mesh, new_vertices, new_faces, metrics)
        setattr(out, "_optcuts_face_chart_ids", chart_arr)
        setattr(out, "_optcuts_vertex_chart_ids", vertex_charts)
        print(
            "[OPTCUTS-CHART-M2D] "
            f"charts={len(data['charts'])} kept={len(new_faces)} "
            f"crossing_removed={len(rejected)} chart_vertex_duplicates={duplicated}"
        )
        return out

    def lift_chart_aware(target: Any, mesh: Any, parameterization: Any, params: Any):
        face_charts = getattr(mesh, "_optcuts_face_chart_ids", None)
        vertex_charts = getattr(mesh, "_optcuts_vertex_chart_ids", None)
        if str(getattr(parameterization, "method", "")) != "optcuts" or face_charts is None or vertex_charts is None:
            return base_lift(target, mesh, parameterization, params)
        vertices_2d = np.asarray(mesh.vertices, dtype=float)
        vcharts = np.asarray(vertex_charts, dtype=int)
        mapped = np.zeros((len(vertices_2d), 3), dtype=float)
        tri_ids = np.full(len(vertices_2d), -1, dtype=int)
        for vi, point in enumerate(vertices_2d):
            cid = int(vcharts[vi])
            if cid < 0:
                raise RuntimeError(f"OPTCUTS_VERTEX_WITHOUT_CHART: vertex={vi}")
            mapped[vi], tri_ids[vi] = _inverse_in_chart(point[:2], cid, parameterization)

        # Reuse the original report construction by feeding a temporary parameterization
        # is unsafe because it would globally re-query UV. Construct the same QuadMesh and
        # a lightweight StageReport directly.
        cls = getattr(getattr(pipeline, "_original", None), "QuadMesh", None) or type(mesh)
        lifted = cls(mapped, np.asarray(mesh.faces, dtype=int).copy(), mesh.grid, "M3D", {}, list(getattr(mesh, "split_lines", [])))
        # Preserve OptCuts metadata so the K3D validity guard sees the active path.
        lifted.metrics = dict(getattr(mesh, "metrics", {}) or {})
        lifted.metrics.update({
            "parameterization_method": "optcuts",
            "m3d_construction_method": "chart-aware OptCuts barycentric inverse",
            "m3d_uv_triangle_lookup_fail_count": 0,
            "m3d_outside_omega_count": 0,
            "m3d_vertex_count": int(len(mapped)),
            "m3d_quad_count": int(len(mesh.faces)),
            "optcuts_chart_aware_lift": True,
        })
        # Use original helpers when available for diagnostics/report only.
        planarity_fn = getattr(pipeline, "_quad_planarity_error", None)
        planarity = float(planarity_fn(mapped, mesh.faces)) if callable(planarity_fn) else 0.0
        lifted.metrics["quad_planarity_error"] = planarity
        lifted.metrics["m3d_planarity_error"] = planarity
        report_cls = getattr(getattr(pipeline, "_original", None), "StageReport", None) or getattr(pipeline, "StageReport")
        counts_fn = getattr(pipeline, "_mesh_counts", None)
        counts = counts_fn(lifted) if callable(counts_fn) else {"vertices": int(len(mapped)), "tiles": int(len(mesh.faces))}
        report = report_cls(
            name="M2D -> M3D",
            objective="Chart-aware OptCuts inverse map: every M2D vertex is mapped only inside its assigned UV chart.",
            before_error=0.0,
            after_error=0.0,
            constraint_violation=planarity,
            computation_time=0.0,
            counts=counts,
        )
        print(f"[OPTCUTS-CHART-LIFT] vertices={len(mapped)} quads={len(mesh.faces)} lookup_fail=0")
        return lifted, report

    pipeline._build_m2d = build_chart_aware_m2d
    pipeline._lift_m2d_to_m3d = lift_chart_aware
    for fn in (
        getattr(pipeline, "build_onestring_design", None),
        getattr(pipeline, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None),
        getattr(getattr(pipeline, "_original", None), "build_onestring_design", None),
    ):
        glb = getattr(fn, "__globals__", None)
        if isinstance(glb, dict):
            glb["_build_m2d"] = build_chart_aware_m2d
            glb["_lift_m2d_to_m3d"] = lift_chart_aware
    pipeline._onestring_optcuts_chart_aware_patch_installed = True


__all__ = ["install_optcuts_chart_aware_patch", "_chart_data", "_quad_chart", "_inverse_in_chart"]
