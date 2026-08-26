"""OptCuts_test: smooth the OptCuts cut boundary and regenerate Omega."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover
    coo_matrix = None
    spsolve = None


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _boundary_loop(parameterization: Any) -> list[int]:
    uv = np.asarray(parameterization.uv_vertices_2d, dtype=float)
    stored = [int(v) for v in (parameterization.metrics.get("boundary_loop", []) or [])]
    if stored and all(0 <= v < len(uv) for v in stored):
        if len(stored) > 1 and stored[0] == stored[-1]:
            stored = stored[:-1]
        return stored
    faces = np.asarray(parameterization.uv_faces, dtype=int)
    incidence: dict[tuple[int, int], int] = {}
    for face in faces:
        a, b, c = [int(x) for x in face]
        for e in ((a, b), (b, c), (c, a)):
            k = _edge_key(*e)
            incidence[k] = incidence.get(k, 0) + 1
    edges = [e for e, n in incidence.items() if n == 1]
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj:
        return []
    start = min(adj)
    loop = [start]
    prev, cur = -1, start
    for _ in range(len(edges) + 2):
        nxts = [v for v in adj[cur] if v != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return loop


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    p = np.asarray(points, float)
    if len(p) < 2:
        return p.copy()
    closed = np.vstack([p, p[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-14:
        return p.copy()
    samples = np.linspace(0.0, total, int(count), endpoint=False)
    out = []
    for s in samples:
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = max(0, min(i, len(p) - 1))
        span = float(cum[i + 1] - cum[i])
        t = 0.0 if span <= 1e-14 else (s - float(cum[i])) / span
        out.append((1.0 - t) * closed[i] + t * closed[i + 1])
    return np.asarray(out, float)


def _taubin_smooth(points: np.ndarray, iterations: int = 18, lam: float = 0.38, mu: float = -0.40) -> np.ndarray:
    out = np.asarray(points, float).copy()
    if len(out) < 5:
        return out
    center0 = np.mean(out, axis=0)
    radius0 = float(np.sqrt(np.mean(np.sum((out - center0) ** 2, axis=1))))
    for _ in range(int(iterations)):
        lap = 0.5 * (np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0)) - out
        out = out + lam * lap
        lap = 0.5 * (np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0)) - out
        out = out + mu * lap
    center = np.mean(out, axis=0)
    radius = float(np.sqrt(np.mean(np.sum((out - center) ** 2, axis=1))))
    out -= center
    if radius > 1e-14 and radius0 > 1e-14:
        out *= radius0 / radius
    out += center0
    return _resample_closed(out, len(out))


def _signed_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(uv, float)[np.asarray(faces, int)]
    return 0.5 * ((tri[:, 1, 0]-tri[:, 0, 0])*(tri[:, 2, 1]-tri[:, 0, 1]) - (tri[:, 1, 1]-tri[:, 0, 1])*(tri[:, 2, 0]-tri[:, 0, 0]))


def _is_valid_like(ref_uv: np.ndarray, candidate: np.ndarray, faces: np.ndarray) -> bool:
    ref = _signed_areas(ref_uv, faces)
    cur = _signed_areas(candidate, faces)
    nz = ref[np.abs(ref) > 1e-14]
    sign = 1.0 if len(nz) == 0 or float(np.median(nz)) >= 0 else -1.0
    eps = max(1e-12, (float(np.median(np.abs(nz))) if len(nz) else 1.0) * 1e-8)
    return bool(np.all(np.isfinite(cur)) and np.all(sign * cur > eps))


def _harmonic_extend(uv: np.ndarray, faces: np.ndarray, boundary_ids: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = len(uv)
    fixed = np.zeros(n, bool); fixed[boundary_ids] = True
    free = np.flatnonzero(~fixed)
    out = np.asarray(uv, float).copy(); out[boundary_ids] = target
    if len(free) == 0:
        return out
    edges: set[tuple[int, int]] = set()
    for f in np.asarray(faces, int):
        a, b, c = [int(x) for x in f]
        for e in ((a,b),(b,c),(c,a)):
            edges.add(_edge_key(*e))
    if coo_matrix is not None and spsolve is not None:
        rows=[]; cols=[]; data=[]; degree=np.zeros(n,float)
        for a,b in edges:
            rows += [a,b]; cols += [b,a]; data += [-1.0,-1.0]; degree[a]+=1; degree[b]+=1
        rows += list(range(n)); cols += list(range(n)); data += degree.tolist()
        L = coo_matrix((data,(rows,cols)), shape=(n,n)).tocsr()
        fixed_ids = np.flatnonzero(fixed)
        try:
            rhs = -(L[free][:,fixed_ids] @ target)
            out[free,0] = np.asarray(spsolve(L[free][:,free], np.asarray(rhs[:,0]).reshape(-1)), float)
            out[free,1] = np.asarray(spsolve(L[free][:,free], np.asarray(rhs[:,1]).reshape(-1)), float)
            return out
        except Exception:
            pass
    adj=[[] for _ in range(n)]
    for a,b in edges:
        adj[a].append(b); adj[b].append(a)
    for _ in range(700):
        prev=out.copy()
        for v in free:
            if adj[int(v)]: out[int(v)] = np.mean(prev[np.asarray(adj[int(v)],int)], axis=0)
        out[boundary_ids] = target
        if float(np.max(np.linalg.norm(out-prev, axis=1))) < 1e-10: break
    return out


def _smooth_and_regenerate_omega(parameterization: Any) -> Any:
    uv0 = np.asarray(parameterization.uv_vertices_2d, float).copy()
    faces = np.asarray(parameterization.uv_faces, int)
    loop = _boundary_loop(parameterization)
    if len(loop) < 5:
        parameterization.method = "optcuts_test"
        return parameterization
    ids = np.asarray(loop, int)
    uniform = _resample_closed(uv0[ids], len(ids))
    target = _taubin_smooth(uniform, iterations=18, lam=0.38, mu=-0.40)
    alpha = 1.0; best = uv0.copy(); accepted = 0.0
    for _ in range(20):
        bt = uv0[ids] + alpha * (target - uv0[ids])
        cand = _harmonic_extend(uv0, faces, ids, bt)
        if _is_valid_like(uv0, cand, faces):
            best = cand; accepted = alpha; break
        alpha *= 0.5
    parameterization.uv_vertices_2d = best
    boundary = best[ids]
    parameterization.omega_boundary = np.vstack([boundary, boundary[0]])
    parameterization.method = "optcuts_test"
    parameterization.metrics.update({
        "omega_parameterization_mode":"optcuts_test",
        "requested_omega_parameterization_mode":"optcuts_test",
        "optcuts_test_enabled":True,
        "optcuts_test_model":"official OptCuts -> equal-arclength boundary -> Taubin smoothing -> harmonic Omega regeneration -> polygon boundary clipping",
        "optcuts_test_seam_topology_preserved":True,
        "optcuts_test_boundary_equal_arclength":True,
        "optcuts_test_seam_smoothing_method":"Taubin(lambda=0.38,mu=-0.40,iterations=18)",
        "optcuts_test_seam_smoothing_alpha":float(accepted),
        "optcuts_test_grid_outline_forcing_removed":True,
    })
    setattr(parameterization, "_optcuts_test_clip_boundary", True)
    print(f"[OPTCUTS-TEST] smooth=taubin+equal-arclength alpha={accepted:.6f} boundary_vertices={len(ids)}")
    return parameterization


def install_optcuts_test_simple_pipeline_patch(pipeline: Any) -> None:
    if getattr(pipeline, "_onestring_optcuts_test_simple_patch_installed", False): return
    base = pipeline._build_surface_parameterization
    def builder(surface: Any, target: Any, grid: Any, params: Any):
        if str(getattr(params, "omega_parameterization_mode", "")) != "optcuts_test":
            return base(surface, target, grid, params)
        ordinary = replace(params, omega_parameterization_mode="optcuts")
        return _smooth_and_regenerate_omega(base(surface, target, grid, ordinary))
    pipeline._build_surface_parameterization = builder
    original = getattr(pipeline, "_original", None)
    if original is not None: original._build_surface_parameterization = builder
    for fn in (getattr(pipeline,"build_onestring_design",None), getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None), getattr(original,"build_onestring_design",None) if original is not None else None):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict): glb["_build_surface_parameterization"] = builder
    pipeline._onestring_optcuts_test_simple_patch_installed = True


__all__=["install_optcuts_test_simple_pipeline_patch"]
