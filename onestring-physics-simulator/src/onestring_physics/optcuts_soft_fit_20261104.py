"""2026-11-04: "soft constraint + boundary fit" Omega mode.

* OptCuts runs with the soft sigma1 = sigma2 term (ONESTRING_ANGLE_W=16) and the seam bend penalty
  (ONESTRING_SEAM_BEND_W=2), the same settings as the Igloo soft run.  These terms exist only in the OneString fork of
  OptCuts (tools/optcuts_soft_fit_20261104/optcuts_onestring_fork_20261104.patch); the mode runs
  third_party/OptCuts_csf_onestring/build2/OptCuts_bin (or ONESTRING_SOFT_FIT_OPTCUTS_EXECUTABLE) instead of the
  binary the app would otherwise use.
* When OptCuts leaves a seam inside the surface that does not reach the boundary (the UV chart then has 2+ boundary
  loops, which the one-chart pipeline refuses), the slit is extended from one of its tips to the surface boundary along
  the shortest edge path that touches no other seam, and the cut surface is given back to OptCuts, whose bijective
  solver re-solves the UV map (it may add cuts only if the distortion bound is still violated).
* The structured grid then uses the "seam zipper" (every seam closes with 4 panels) and grid corners from the 3D
  surface's turning (feature preset ``soft_fit_20261104`` in feature_lines_20261029.py).

``configure(active)`` is called by the app on every rerun: it sets or removes the environment and the bridge patch.
"""
from __future__ import annotations

import heapq
import os
from pathlib import Path

import numpy as np

_ENV = {"ONESTRING_ANGLE_W": "16", "ONESTRING_SEAM_BEND_W": "2"}
# 2026-11-04: OptCuts settings of the cloud comparison runs (the artifacts): constraint -> environment of the fork
CONSTRAINTS = {
    "soft": {"ONESTRING_ANGLE_W": "16", "ONESTRING_SEAM_BEND_W": "2"},
    "hard": {"ONESTRING_ANGLE_W": "4096", "ONESTRING_ANGLE_BOUND_SD": "1", "ONESTRING_SEAM_BEND_W": "2"},
    "none": {},
}
# seam start search (multi-start): the symmetry term and the no-branch rule used with each constraint
MULTISTART = {"soft": {"ONESTRING_SYM_W": "1"}, "hard": {"ONESTRING_SYM_W": "0.5", "ONESTRING_NO_BRANCH": "1"}, "none": {}}
_ALL_KEYS = ("ONESTRING_ANGLE_W", "ONESTRING_ANGLE_BOUND_SD", "ONESTRING_SEAM_BEND_W", "ONESTRING_SYM_W", "ONESTRING_NO_BRANCH")
_MARK = "ONESTRING_SOFT_FIT_20261104_ACTIVE"
_STATE = {"installed": False, "orig_read": None, "orig_once": None, "cfg": None, "depth": 0, "exe": "", "constraint": "soft", "multistart": False, "root": None}


def _boundary_loops(f2: np.ndarray, n: int) -> list[set]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    bE = u[c == 1]
    g = coo_matrix((np.ones(len(bE)), (bE[:, 0], bE[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    loops: dict[int, set] = {}
    for e in bE:
        loops.setdefault(int(lab[e[0]]), set()).update(e.tolist())
    return sorted(loops.values(), key=len, reverse=True)


def connect_interior_slits(xyz, f3, uv, f2):
    """Extend every interior slit to the surface boundary (topology only; the UV of new copies = the old one).

    Returns (vmap, f2_new, uv_new, report): vmap[j] = surface vertex of UV vertex j."""
    xyz = np.asarray(xyz, float); f3 = np.asarray(f3, int); uv = np.asarray(uv, float); f2 = np.asarray(f2, int)
    nV = len(uv)
    vid = np.full(nV, -1, int)
    vid[f2.ravel()] = f3.ravel()
    loops = _boundary_loops(f2, nV)
    if len(loops) < 2:
        return vid, f2, uv, []
    copies = np.bincount(vid, minlength=len(xyz))
    E0 = np.sort(np.concatenate([f3[:, [0, 1]], f3[:, [1, 2]], f3[:, [2, 0]]]), 1)
    u0, c0 = np.unique(E0, axis=0, return_counts=True)
    bnd0 = set(u0[c0 == 1].ravel().tolist())
    nbr: list[list] = [[] for _ in range(len(xyz))]
    for a, c in u0:
        L = float(np.linalg.norm(xyz[a] - xyz[c])); nbr[a].append((c, L)); nbr[c].append((a, L))
    seam_touch = set(np.flatnonzero(copies > 1).tolist())
    targets = {v for v in bnd0 if v not in seam_touch}
    cut_edges, report = set(), []
    for li, loop in enumerate(loops[1:], 1):
        cnt: dict[int, int] = {}
        for v in loop:
            cnt[int(vid[v])] = cnt.get(int(vid[v]), 0) + 1
        tips = [o for o, k in cnt.items() if k == 1]
        if len(tips) != 2:
            raise RuntimeError(f"interior loop {li} is not a simple slit ({len(tips)} tips)")
        dist = {t: 0.0 for t in tips}; prev: dict[int, int] = {}; pq = [(0.0, t) for t in tips]; heapq.heapify(pq); end = None
        while pq:
            dv, v = heapq.heappop(pq)
            if dv > dist.get(v, np.inf):
                continue
            if v in targets:
                end = v; break
            for w, L in nbr[v]:
                if (w in seam_touch and w not in tips) or (w in bnd0 and w not in targets):
                    continue
                if dv + L < dist.get(w, np.inf):
                    dist[w] = dv + L; prev[w] = v; heapq.heappush(pq, (dv + L, w))
        if end is None:
            raise RuntimeError(f"interior slit {li}: no path to the boundary")
        path = [end]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path = path[::-1]
        for a, c in zip(path[:-1], path[1:]):
            cut_edges.add((min(a, c), max(a, c)))
        report.append(dict(loop=li, slit_vertices=len(loop), path_edges=len(path) - 1, path_length=float(dist[end])))
    inc: list[list[int]] = [[] for _ in range(nV)]
    for t, f in enumerate(f2):
        for k in range(3):
            inc[f[k]].append(t)
    f2n = f2.copy(); uvn = [uv[i] for i in range(nV)]; vmap = list(vid)
    for v in range(nV):
        ts = inc[v]; o = vid[v]
        if len(ts) < 2 or not any((min(o, vid[x]), max(o, vid[x])) in cut_edges for t in ts for x in f2[t] if vid[x] != o):
            continue
        par = {t: t for t in ts}

        def fd(t):
            while par[t] != t:
                par[t] = par[par[t]]; t = par[t]
            return t
        for i, t in enumerate(ts):
            for s in ts[i + 1:]:
                sh = set(f2[t].tolist()) & set(f2[s].tolist()); sh.discard(v)
                for w in sh:
                    if (min(o, vid[w]), max(o, vid[w])) not in cut_edges:
                        par[fd(t)] = fd(s)
        groups: dict[int, list] = {}
        for t in ts:
            groups.setdefault(fd(t), []).append(t)
        for gi, g in enumerate(sorted(groups.values(), key=len, reverse=True)):
            if gi == 0:
                continue
            nvid = len(uvn); uvn.append(uv[v].copy()); vmap.append(o)
            for t in g:
                f2n[t][f2n[t] == v] = nvid
    return np.asarray(vmap), f2n, np.asarray(uvn), report


def _multistart(ob, V, F, cfg, trace):
    """Seam start search of the cloud runs (tools/optcuts_soft_fit_20261104/optcuts_multistart.py) in a subprocess on
    this exact surface; the winner's chart is returned as the bridge's result.  When the bridge refused the winner's
    chart (a seam that does not reach the boundary), the slit is connected and the UV map re-solved as below."""
    import pickle, subprocess, sys, tempfile, time
    import trimesh
    from scipy.spatial import cKDTree
    root = Path(_STATE["root"] or Path(__file__).resolve().parents[2])
    script = root / "tools" / "optcuts_soft_fit_20261104" / "optcuts_multistart.py"
    out = root / "output" / "optcuts_multistart" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, float); F = np.asarray(F, int)
    mesh = out / "surface.obj"
    trimesh.Trimesh(V, F, process=False).export(mesh)
    env = dict(os.environ, MS_NO_NORMALIZE="1", MS_FINAL_TIMEOUT="0", MS_STALL_SECONDS="180")
    if _STATE.get("exe"):
        env["ONESTRING_OPTCUTS_EXE"] = _STATE["exe"]
    env.setdefault("ONESTRING_TRACE_DIR", str(out / "trace"))
    print(f"[FIT] seam start search ({_STATE['constraint']} constraint) -> {out}", flush=True)
    trace("optcuts.fit", "multistart_start", out=str(out), constraint=_STATE["constraint"])
    proc = subprocess.Popen([sys.executable, str(script), str(mesh), str(out)], cwd=str(root), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if line.startswith("[MS]"):
            print("[FIT] " + line.rstrip(), flush=True)
    proc.wait()
    chart_p = out / "optcuts_chart.pkl"
    if chart_p.exists():
        ch = pickle.load(open(chart_p, "rb"))
        d, idx = cKDTree(V).query(np.asarray(ch["xyz"], float))
        if d.max() > 1e-6:
            raise RuntimeError("multi-start chart does not match the surface")
        f3 = idx[np.asarray(ch["f3"], int)]; uv = np.asarray(ch["uv"], float); f2 = np.asarray(ch["f2"], int)
    else:
        frames = sorted(out.glob("f*.bin"))
        if not frames:
            raise RuntimeError(f"seam start search produced no result (see {out})")
        b = open(frames[-1], "rb").read(); nV, nF = np.frombuffer(b, np.int32, 2, 12)
        A = np.frombuffer(b, np.float64, nV * 5, 20).reshape(nV, 5); F2 = np.frombuffer(b, np.int32, nF * 3, 20 + nV * 40).reshape(nF, 3)
        vid = cKDTree(V).query(A[:, 2:])[1]
        vmap, f2n, uvn, report = connect_interior_slits(V, vid[F2], A[:, :2], F2)
        print(f"[FIT] the winner left a seam inside the surface; connected to the boundary: {report}", flush=True)
        trace("optcuts.fit", "connect", slits=report)
        Vc = V[vmap]
        _STATE["depth"] += 1
        try:
            r = _STATE["orig_once"](Vc, f2n, cfg) if cfg is not None else _STATE["orig_once"](Vc, f2n)
        finally:
            _STATE["depth"] -= 1
        d, idx = cKDTree(Vc).query(np.asarray(r.surface_vertices_3d))
        f3 = vmap[idx[np.asarray(r.surface_faces)]]; uv = np.asarray(r.uv_vertices_2d); f2 = np.asarray(r.uv_faces)
    loops = ob._boundary_loops(f2)
    trace("optcuts.fit", "multistart_done", uv_vertices=int(len(uv)), boundary_loops=len(loops))
    return ob.OptCutsResult(surface_vertices_3d=V.copy(), surface_faces=f3, uv_vertices_2d=uv, uv_faces=f2,
                            boundary_loops=loops, output_obj=str(out),
                            metrics={"parameterization_backend_name": "optcuts_multistart_20261104",
                                     "optcuts_uv_boundary_loop_count": len(loops),
                                     "optcuts_fit_constraint": _STATE["constraint"]})


def _install():
    if _STATE["installed"]:
        return
    from . import optcuts_backend as ob
    from .run_trace import trace
    orig_read, orig_once = ob._read_obj_with_uv, ob._run_official_optcuts_once
    _STATE.update(orig_read=orig_read, orig_once=orig_once)

    def once(V, F, cfg=None, *a, **k):
        if os.environ.get(_MARK) and _STATE.get("exe"):
            # the soft sigma1 = sigma2 term exists only in the OneString fork: use it even if the app named a binary
            import dataclasses
            cfg = dataclasses.replace(cfg if cfg is not None else ob.OptCutsConfig(), executable=_STATE["exe"])
        _STATE["cfg"] = cfg
        if os.environ.get(_MARK) and _STATE["multistart"] and _STATE["depth"] == 0:
            return _multistart(ob, V, F, cfg, trace)
        return orig_once(V, F, cfg, *a, **k) if cfg is not None else orig_once(V, F, *a, **k)

    def read(path):
        xyz, f3, uv, f2 = orig_read(path)
        if not os.environ.get(_MARK) or _STATE["depth"] > 0:
            return xyz, f3, uv, f2
        loops = _boundary_loops(np.asarray(f2, int), len(uv))
        if len(loops) < 2:
            return xyz, f3, uv, f2
        vmap, f2n, uvn, report = connect_interior_slits(xyz, f3, uv, f2)
        print(f"[SOFT-FIT] OptCuts left {len(loops) - 1} seam(s) inside the surface; connected to the boundary: {report}", flush=True)
        trace("optcuts.soft_fit", "connect", loops=len(loops), slits=report)
        Vc = np.asarray(xyz, float)[vmap]
        _STATE["depth"] += 1
        try:
            r = orig_once(Vc, f2n, _STATE["cfg"]) if _STATE["cfg"] is not None else orig_once(Vc, f2n)
        finally:
            _STATE["depth"] -= 1
        from scipy.spatial import cKDTree
        d, idx = cKDTree(Vc).query(np.asarray(r.surface_vertices_3d))
        if d.max() > 1e-6:
            raise RuntimeError("re-solved chart does not match the cut surface")
        f3n = vmap[idx[np.asarray(r.surface_faces)]]
        info = dict(uv_vertices=int(len(r.uv_vertices_2d)), boundary_loops=len(r.boundary_loops))
        print(f"[SOFT-FIT] UV map re-solved by OptCuts on the connected cut: {info}", flush=True)
        trace("optcuts.soft_fit", "resolve", **info)
        return np.asarray(xyz, float), f3n, np.asarray(r.uv_vertices_2d), np.asarray(r.uv_faces)

    ob._run_official_optcuts_once = once
    ob._read_obj_with_uv = read
    _STATE["installed"] = True


def configure(active: bool, project_root: str | Path | None = None, constraint: str = "soft",
              multistart: bool = False) -> None:
    """Switch the mode's OptCuts environment (constraint: soft / hard / none), the seam start search and the
    interior-slit repair on or off (idempotent)."""
    for k in _ALL_KEYS:
        os.environ.pop(k, None)
    if active:
        constraint = constraint if constraint in CONSTRAINTS else "soft"
        os.environ.update(CONSTRAINTS[constraint])
        if multistart:
            os.environ.update(MULTISTART[constraint])
        os.environ[_MARK] = "1"
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        _STATE.update(constraint=constraint, multistart=bool(multistart), root=str(root))
        fork = root / "third_party" / "OptCuts_csf_onestring" / "build2" / "OptCuts_bin"
        exe = os.environ.get("ONESTRING_SOFT_FIT_OPTCUTS_EXECUTABLE") or (str(fork) if fork.is_file() else "")
        _STATE["exe"] = exe
        if not exe and constraint != "none":
            print("[SOFT-FIT] WARNING: the OneString OptCuts fork was not found "
                  f"({fork}); OptCuts runs without the sigma1 = sigma2 term", flush=True)
        _install()
    else:
        os.environ.pop(_MARK, None)
        _STATE.update(exe="", multistart=False)
