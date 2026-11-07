"""2026-11-04: "soft constraint + boundary fit" Omega mode.

* OptCuts runs with the soft sigma1 = sigma2 term (ONESTRING_ANGLE_W=16) and the seam bend penalty
  (ONESTRING_SEAM_BEND_W=2), the same settings as the Igloo soft run.  These terms exist only in the OneString fork of
  OptCuts (tools/optcuts_soft_fit_20261104/optcuts_onestring_fork_20261104.patch); the mode runs
  third_party/OptCuts_csf_onestring/build2/OptCuts_bin (or ONESTRING_SOFT_FIT_OPTCUTS_EXECUTABLE) instead of the
  binary the app would otherwise use.
* When OptCuts leaves a seam inside the surface that does not reach the boundary (the UV chart then has 2+ boundary
  loops, which the one-chart pipeline refuses), every such cut (a slit or a branched tree) is connected to the boundary
  along the least visible edge path that touches no other seam (connect_interior_slits), and the cut surface is given back to OptCuts, whose bijective
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


def _hidden_edge_cost(xyz, f3, u0):
    """Cost factor per surface edge for a connecting cut: where a cut is least visible it is cheapest.

    A cut is hardest to see in a concave crease (a valley between two faces) and easiest to see on a smooth or convex
    part, so the factor is 1 on flat and convex edges and falls to 0.2 in a valley of 45 degrees or more."""
    n = np.cross(xyz[f3[:, 1]] - xyz[f3[:, 0]], xyz[f3[:, 2]] - xyz[f3[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    faces_of = {}
    for t, f in enumerate(f3):
        for k in range(3):
            a, c = int(f[k]), int(f[(k + 1) % 3])
            faces_of.setdefault((min(a, c), max(a, c)), []).append((t, int(f[(k + 2) % 3])))
    out = {}
    for a, c in u0:
        fs = faces_of.get((int(a), int(c)), [])
        if len(fs) != 2:
            out[(int(a), int(c))] = 1.0
            continue
        (t1, _), (t2, o2) = fs
        ang = float(np.degrees(np.arccos(np.clip(n[t1] @ n[t2], -1.0, 1.0))))
        concave = (xyz[o2] - xyz[a]) @ n[t1] > 0          # the second face bends up towards the first face's normal
        out[(int(a), int(c))] = float(np.clip(1.0 - ang / 45.0, 0.2, 1.0)) if concave else 1.0
    return out


def _flipped(uv, f2) -> int:
    """triangles whose UV orientation is opposite to the majority"""
    P = np.asarray(uv, float)[np.asarray(f2, int)]
    s = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    return int(min((s > 0).sum(), (s < 0).sum()))


def _resolve_bijective(ob, Vc, f2n):
    """UV map of the cut surface by OptCuts' fixed-topology distortion minimisation (method 3, DistMin) with its
    bijectivity guarantee: no triangle flips and no overlap, and no new cuts (2026-11-06; before, the re-solve ran full
    OptCuts, which may cut again and leave new inner cuts or pieces)."""
    import dataclasses
    cfg = _STATE["cfg"] if _STATE["cfg"] is not None else ob.OptCutsConfig()
    cfg = dataclasses.replace(cfg, method_type=3, use_bijectivity=True, executable=_STATE.get("exe") or cfg.executable)
    _STATE["depth"] += 1
    try:
        return _STATE["orig_once"](Vc, np.asarray(f2n), cfg)
    finally:
        _STATE["depth"] -= 1


def _chart_pieces(f2: np.ndarray, n: int) -> int:
    """number of connected pieces of a UV mesh"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]]])
    used = np.unique(f2)
    k, lab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)), directed=False)
    return len(np.unique(lab[used]))


def connect_interior_slits(xyz, f3, uv, f2):
    """Connect every cut that lies inside the surface (not reaching the boundary) to the boundary, so the chart has one
    boundary loop again (topology only; the UV of new copies = the old one; the caller re-solves the UV map with
    OptCuts' bijective solver).

    A cut may be a single slit or a branched tree of seams (2026-11-06: before, only simple slits with two tips were
    accepted, which stopped the Stanford bunny with 68 inner cuts, one with 44 tips).  Each inner cut is joined by one
    connecting path, from whichever of its vertices gives the cheapest path, to the boundary, to the surface boundary
    or to a cut that is already connected (a connected cut is part of the boundary from then on).  The path touches no
    other cut, and its cost is edge length x a visibility factor (_hidden_edge_cost) so that it runs through
    concave creases where it is hard to see, and stays short.

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
    hide = _hidden_edge_cost(xyz, f3, u0)
    nbr: list[list] = [[] for _ in range(len(xyz))]
    for a, c in u0:
        L = float(np.linalg.norm(xyz[a] - xyz[c])) * hide[(int(a), int(c))]
        nbr[a].append((c, L)); nbr[c].append((a, L))
    seam_touch = set(np.flatnonzero(copies > 1).tolist())
    # the outer boundary of the chart (largest loop) already reaches the surface boundary
    outer = {int(vid[v]) for v in loops[0]}
    targets = {v for v in bnd0 if v not in seam_touch} | outer
    cut_edges, report = set(), []
    for li, loop in enumerate(loops[1:], 1):
        own = {int(vid[v]) for v in loop}
        # a surface vertex shared with the boundary through a different UV copy does not join the loops: the path
        # must reach a boundary vertex that is not part of this cut
        goal = targets - own
        dist = {t: 0.0 for t in own}; prev: dict[int, int] = {}; pq = [(0.0, t) for t in own]; heapq.heapify(pq); end = None
        while pq:
            dv, v = heapq.heappop(pq)
            if dv > dist.get(v, np.inf):
                continue
            if v in goal:
                end = v; break
            for w, L in nbr[v]:
                if w in seam_touch and w not in own and w not in goal:
                    continue
                if w in bnd0 and w not in goal and w not in own:
                    continue
                if dv + L < dist.get(w, np.inf):
                    dist[w] = dv + L; prev[w] = v; heapq.heappush(pq, (dv + L, w))
        if end is None:
            raise RuntimeError(f"inner cut {li}: no path to the boundary")
        path = [end]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        path = path[::-1]
        for a, c in zip(path[:-1], path[1:]):
            cut_edges.add((min(a, c), max(a, c)))
        targets |= own | set(path)
        seam_touch |= set(path)
        report.append(dict(loop=li, cut_vertices=len(own), path_edges=len(path) - 1,
                           path_length=float(sum(np.linalg.norm(xyz[a] - xyz[c]) for a, c in zip(path[:-1], path[1:])))))
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
        f3, uv, f2 = vid[F2], A[:, :2], F2
        pieces = _chart_pieces(np.asarray(f2, int), len(uv))
        if pieces > 1:
            # connecting cuts can only open a chart further; pieces that OptCuts cut off completely cannot be joined
            raise RuntimeError(f"OptCuts cut the surface into {pieces} separate pieces (not one chart with inner cuts); "
                               "connecting cuts to the boundary cannot join them.  Use a coarser surface or the soft "
                               f"constraint, or split the design into pieces (see {out})")
        # connect the inner cuts to the boundary and let OptCuts' bijective solver re-solve the UV map; OptCuts may
        # add cuts again, so repeat until the chart has one boundary loop (stop if a round does not reduce them)
        prev_loops = None
        while True:
            n_loops = len(_boundary_loops(np.asarray(f2, int), len(uv)))
            if n_loops <= 1:
                break
            if prev_loops is not None and n_loops >= prev_loops:
                raise RuntimeError(f"connecting the inner cuts does not converge: {n_loops} boundary loops after "
                                   f"re-solving (before: {prev_loops})")
            prev_loops = n_loops
            vmap, f2n, uvn, report = connect_interior_slits(V, f3, uv, f2)
            print(f"[FIT] {n_loops - 1} cut(s) inside the surface; connected to the boundary along hidden paths: "
                  f"{len(report)} cuts, {sum(r_['path_edges'] for r_ in report)} edges added", flush=True)
            trace("optcuts.fit", "connect", slits=report)
            Vc = V[vmap]
            r = _resolve_bijective(ob, Vc, f2n)
            d, idx = cKDTree(Vc).query(np.asarray(r.surface_vertices_3d))
            f3 = vmap[idx[np.asarray(r.surface_faces)]]; uv = np.asarray(r.uv_vertices_2d); f2 = np.asarray(r.uv_faces)
        # bijectivity check of the final map: no flipped triangle (OptCuts' bijective solver also prevents overlaps)
        flipped = _flipped(uv, f2)
        print(f"[FIT] final chart: {len(f2)} triangles, flipped {flipped}", flush=True)
        if flipped:
            raise RuntimeError(f"the re-solved UV map is not bijective: {flipped} flipped triangles")
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
        print(f"[FIT] OptCuts call: mode active {bool(os.environ.get(_MARK))}, constraint {_STATE['constraint']}, "
              f"seam start search {_STATE['multistart']}, nested {_STATE['depth']}, executable {getattr(cfg, 'executable', None)}", flush=True)
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
        pieces = _chart_pieces(np.asarray(f2, int), len(uv))
        if pieces > 1:
            raise RuntimeError(f"OptCuts cut the surface into {pieces} separate pieces (not one chart with inner cuts); "
                               "connecting cuts to the boundary cannot join them.  Use a coarser surface or the soft constraint.")
        vmap, f2n, uvn, report = connect_interior_slits(xyz, f3, uv, f2)
        print(f"[SOFT-FIT] OptCuts left {len(loops) - 1} cut(s) inside the surface; connected to the boundary along hidden "
              f"paths ({sum(r_['path_edges'] for r_ in report)} edges)", flush=True)
        trace("optcuts.soft_fit", "connect", loops=len(loops), slits=report)
        Vc = np.asarray(xyz, float)[vmap]
        r = _resolve_bijective(ob, Vc, f2n)
        from scipy.spatial import cKDTree
        d, idx = cKDTree(Vc).query(np.asarray(r.surface_vertices_3d))
        if d.max() > 1e-6:
            raise RuntimeError("re-solved chart does not match the cut surface")
        f3n = vmap[idx[np.asarray(r.surface_faces)]]
        info = dict(uv_vertices=int(len(r.uv_vertices_2d)), boundary_loops=len(r.boundary_loops),
                    flipped=_flipped(np.asarray(r.uv_vertices_2d), np.asarray(r.uv_faces)))
        if info["flipped"]:
            raise RuntimeError(f"the re-solved UV map is not bijective: {info['flipped']} flipped triangles")
        print(f"[SOFT-FIT] UV map re-solved by OptCuts (bijective, cuts kept as they are): {info}", flush=True)
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
        print(f"[FIT] configure: constraint {constraint}, seam start search {bool(multistart)}, root {root}", flush=True)
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
