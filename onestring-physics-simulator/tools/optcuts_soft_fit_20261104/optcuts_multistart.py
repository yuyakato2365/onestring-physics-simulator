"""2026-11-03: OptCuts with several seam starting points solved side by side, pruned while they progress.

  candidates : the boundary vertices OptCuts itself would examine at its first split (top N^0.8 by its own ranking)
  branch     : OptCuts (same settings) forced to start its first seam at one candidate (ONESTRING_FORCE_START_VERTEX)
  rounds     : every live branch is advanced to the same number of topology changes (1, 2, 4, 8, ...; OptCuts is
               deterministic, so a branch is simply re-run with the larger budget), ranked, and the better half kept;
               a branch that finishes on its own keeps its final result.  Repeated until one branch is left, which is
               then run to the end.
  ranking    : a state is feasible when its distortion measure is <= b_d (4.1).  Feasible states are ranked by the
               seam cost with the bend penalty (each seam edge: length x (1 + w_bend (1 - cos(turn)))), smaller first;
               branches with no feasible state yet come after them, ranked by how far their distortion is above b_d.
  symmetry   : (2026-11-03, ONESTRING_SYM_W > 0) the mirror plane of the shape is found automatically (PCA axes refined
               by minimising the distance of the mirrored surface to itself) and passed to OptCuts, which adds
               w x (length of the seam edges off the plane whose mirror image is not a seam edge) to the seam
               length; the same term is added to the seam cost used for ranking, and the boundary vertices on the
               plane are added to the starting points.
env: ONESTRING_ANGLE_W, ONESTRING_ANGLE_BOUND_SD, ONESTRING_SEAM_BEND_W (as for optcuts_dump.py), ONESTRING_SYM_W
usage: python3 optcuts_multistart.py MESH OUT_DIR
"""
import os, sys, json, glob, shutil, pickle, math, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, trimesh
from pathlib import Path
TOOLS = Path(__file__).resolve().parent; ROOT = Path(os.environ.get("ONESTRING_ROOT", TOOLS.parents[1]))
sys.path.insert(0, str(TOOLS)); sys.path.insert(0, str(ROOT / "src"))
EXE = os.environ.get("ONESTRING_OPTCUTS_EXE", str(ROOT / "third_party" / "OptCuts_csf_onestring" / "build2" / "OptCuts_bin"))   # OptCuts with optcuts_onestring_fork_20261104.patch
BD = 4.1
STALL = float(os.environ.get("MS_STALL_SECONDS", "30"))    # no new OptCuts frame for this long = cycling without any move

def load_mesh(mesh):
    from onestring_physics.input_shape import normalize_shape
    m = trimesh.load(mesh, force="mesh", process=False)
    if os.environ.get("MS_NO_NORMALIZE"):   # 2026-11-04: the app passes its own (already normalised) surface
        return np.asarray(m.vertices, float), np.asarray(m.faces, int)
    return normalize_shape(np.asarray(m.vertices, float)), np.asarray(m.faces, int)

def run_branch(args):
    """Run OptCuts once (in a fresh process).  Returns dict(finished, chart or None, dump)."""
    mesh, dump, start, budget, extra_env = args[:5]
    timeout = args[5] if len(args) > 5 else 1800.0     # a branch whose only allowed first move never pays off can loop without progress
    os.chdir(str(ROOT))
    from onestring_physics import optcuts_backend as ob
    os.makedirs(dump, exist_ok=True)
    for f in glob.glob(os.path.join(dump, "f*.bin")): os.remove(f)
    env = {"ONESTRING_CSF_MODE": "0", "ONESTRING_DUMP_DIR": dump}
    env.update(extra_env)
    if start is not None: env["ONESTRING_FORCE_START_VERTEX"] = str(start)
    if budget is not None: env["ONESTRING_TOPO_BUDGET"] = str(budget)
    for k in ("ONESTRING_FORCE_START_VERTEX", "ONESTRING_TOPO_BUDGET", "ONESTRING_PRINT_START_CANDS"): os.environ.pop(k, None)
    os.environ.update(env)
    V, F = load_mesh(mesh)
    import subprocess, tempfile
    def watched_run(cmd, cwd=None, capture_output=True, text=True, timeout=None, check=False, **kw):
        """subprocess.run replacement: stop OptCuts when it has dumped no new frame for STALL seconds (it is then
        cycling without any move, e.g. a forced first seam that never pays off)"""
        fo, fe = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
        p = subprocess.Popen(cmd, cwd=cwd, stdout=fo, stderr=fe, text=True, env=kw.get("env"))
        last, nlast, t0 = time.time(), -1, time.time()
        while p.poll() is None:
            time.sleep(0.5)
            n = len(glob.glob(os.path.join(dump, "f*.bin")))
            if n != nlast: nlast, last = n, time.time()
            elif time.time() - last > STALL or (timeout and time.time() - t0 > timeout): p.kill(); break
        p.wait(); fo.seek(0); fe.seek(0)
        return subprocess.CompletedProcess(cmd, p.returncode, fo.read(), fe.read())
    ob.subprocess.run = watched_run
    def tagged_output_only(root, tag, started_at):
        """2026-11-04: accept only this run's own result.  The bridge otherwise falls back to any finalResult written
        after the run started, so a branch stopped by its topology budget (which writes no result) picked up the
        result of another OptCuts run working at the same time and was scored as finished with a foreign chart."""
        c = [p for p in (root / "output").glob("**/finalResult_mesh.obj") if tag in str(p.parent)]
        if not c: raise ob.OptCutsOutputError("no result of this run (stopped by the topology budget)")
        return max(c, key=lambda p: p.stat().st_mtime)
    ob._find_output_obj = tagged_output_only
    cfg = ob.OptCutsConfig(executable=EXE, csf_tail_refine=False, timeout_seconds=float(timeout))
    try:
        r = ob._run_official_optcuts_once(V, F, cfg)
        chart = {k: np.asarray(v) for k, v in (("xyz", r.surface_vertices_3d), ("f3", r.surface_faces), ("uv", r.uv_vertices_2d), ("f2", r.uv_faces))}
        pickle.dump(chart, open(os.path.join(dump, "chart.pkl"), "wb"))
        return dict(finished=True, dump=dump)
    except Exception as e:
        return dict(finished=False, dump=dump, err=str(e)[:200])

def boundary_loop_count(F, n):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1); u, c = np.unique(E, axis=0, return_counts=True); bE = u[c == 1]
    if not len(bE): return 0
    g = coo_matrix((np.ones(len(bE)), (bE[:, 0], bE[:, 1])), shape=(n, n)); _, lab = connected_components(g, directed=False)
    return int(len(np.unique(lab[np.unique(bE.ravel())])))

def frames_of(dump):
    out = []
    for fn in sorted(glob.glob(os.path.join(dump, "f*.bin"))):
        b = open(fn, "rb").read()
        it, = np.frombuffer(b, np.int32, 1, 0); lam, = np.frombuffer(b, np.float64, 1, 4); nV, nF = np.frombuffer(b, np.int32, 2, 12)
        A = np.frombuffer(b, np.float64, nV * 5, 20).reshape(nV, 5); Fm = np.frombuffer(b, np.int32, nF * 3, 20 + nV * 40).reshape(nF, 3)
        out.append(dict(it=int(it), lam=float(lam), uv=A[:, :2].copy(), x=A[:, 2:].copy(), F=Fm.copy()))
    return out

def distortion(X, F, uv, w_angle, bound_sd):
    P = X[F]; e1 = P[:, 1] - P[:, 0]; e2 = P[:, 2] - P[:, 0]; a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    ux = e1 / np.linalg.norm(e1, axis=1, keepdims=True); n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True); uy = np.cross(n, ux)
    L = np.stack([np.stack([(e1 * ux).sum(1), (e1 * uy).sum(1)], 1), np.stack([(e2 * ux).sum(1), (e2 * uy).sum(1)], 1)], 2)
    Q = uv[F]; U = np.stack([Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]], 2)
    s = np.linalg.svd(U @ np.linalg.inv(L), compute_uv=False); s1, s2 = s[:, 0], np.maximum(s[:, 1], 1e-12)
    A = a3.sum(); sd = float(((s1**2 + s1**-2 + s2**2 + s2**-2) * a3).sum() / A)
    mips = float(((s1 / s2 + s2 / s1 - 2) * a3).sum() / A)
    return sd if bound_sd else sd + w_angle * mips

def seam_polylines(X, F):
    """seam edges (geometric edges on the boundary twice) chained into polylines of 3D points"""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, c = np.unique(E, axis=0, return_counts=True); bE = u[c == 1]
    keys = {}; pid = np.array([keys.setdefault(tuple(p), len(keys)) for p in np.round(X, 9)])
    pos = np.zeros((len(keys), 3)); pos[pid] = X
    ge = np.sort(pid[bE], 1); gk = ge[:, 0] * len(keys) + ge[:, 1]; gu, gc = np.unique(gk, return_counts=True)
    seam = [(int(k // len(keys)), int(k % len(keys))) for k in gu[gc == 2]]
    nb = {}
    for a, b in seam: nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    used = set(); lines = []
    ends = [v for v in nb if len(nb[v]) != 2] or list(nb)[:1]
    for s0 in ends:
        for n0 in nb[s0]:
            if (min(s0, n0), max(s0, n0)) in used: continue
            line = [s0, n0]; used.add((min(s0, n0), max(s0, n0)))
            while len(nb[line[-1]]) == 2:
                nxt = [v for v in nb[line[-1]] if (min(v, line[-1]), max(v, line[-1])) not in used]
                if not nxt: break
                used.add((min(nxt[0], line[-1]), max(nxt[0], line[-1]))); line.append(nxt[0])
            lines.append(pos[line])
    return lines

SYM = None        # dict(n, d, tol, w) when the symmetry term is on

def detect_mirror_plane(V, F):
    """Mirror plane n.x = d of the surface: PCA axes through the area centroid, each refined (Nelder-Mead on the
    normal's two angles and the offset) by minimising the mean distance of the mirrored vertices to the surface.
    Returns (n, d, error / bbox diagonal) of the best plane."""
    from scipy.optimize import minimize
    tm = trimesh.Trimesh(V, F, process=False)
    P = V[F]; A = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    c = (P.mean(1) * A[:, None]).sum(0) / A.sum()
    X = V - c; _, ev = np.linalg.eigh(X.T @ X)
    diag = float(np.linalg.norm(np.ptp(V, 0)))
    sub = V[np.linspace(0, len(V) - 1, min(len(V), 800)).astype(int)]
    def err(prm):
        th, ph, off = prm
        n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        d = float(n @ c) + off
        Rm = sub - 2 * ((sub @ n) - d)[:, None] * n
        _, dist, _ = trimesh.proximity.closest_point(tm, Rm)
        return float(dist.mean()) / diag
    best = None
    for n0 in ev.T:
        th0 = float(np.arccos(np.clip(n0[2], -1, 1))); ph0 = float(np.arctan2(n0[1], n0[0]))
        r = minimize(err, [th0, ph0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-7, maxiter=300))
        if best is None or r.fun < best[0]:
            th, ph, off = r.x
            n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            best = (float(r.fun), n, float(n @ c) + off)
    return best[1], best[2], best[0]

def asym_length(lines):
    """length of the seam edges that are off the mirror plane and whose mirror image is not a seam edge (OptCuts' E_sym)"""
    if SYM is None or not lines: return 0.0
    n, d, tol = SYM["n"], SYM["d"], SYM["tol"]
    segs = [(np.asarray(P[i]), np.asarray(P[i + 1])) for P in lines for i in range(len(P) - 1)]
    if not segs: return 0.0
    A = np.array([a for a, b in segs]); B = np.array([b for a, b in segs])
    out = 0.0
    for a, b in segs:
        if abs(a @ n - d) < tol and abs(b @ n - d) < tol: continue
        ra = a - 2 * (a @ n - d) * n; rb = b - 2 * (b @ n - d) * n
        hit = ((np.linalg.norm(A - ra, axis=1) < tol) & (np.linalg.norm(B - rb, axis=1) < tol)) | \
              ((np.linalg.norm(A - rb, axis=1) < tol) & (np.linalg.norm(B - ra, axis=1) < tol))
        if not hit.any(): out += float(np.linalg.norm(b - a))
    return out

def seam_cost(lines, w_bend, R):
    c = 0.0
    for P in lines:
        d = np.diff(P, axis=0); L = np.linalg.norm(d, axis=1); u = d / np.maximum(L[:, None], 1e-12)
        f = np.ones(len(L)); f[1:] += w_bend * (1 - np.clip((u[1:] * u[:-1]).sum(1), -1, 1))
        c += float((L * f).sum())
    if SYM is not None:
        c += SYM["w"] * asym_length(lines)
    return c / R

def score_states(states, w_angle, bound_sd, w_bend, R):
    """states: list of (X, F, uv).  -> (rank tuple, details of the state that defines it)"""
    best_f = None; best_i = None
    for X, F, uv in states:
        e = distortion(X, F, uv, w_angle, bound_sd)
        if e <= BD + 1e-3:
            lines = seam_polylines(X, F); c = seam_cost(lines, w_bend, R)
            if best_f is None or c < best_f[0]: best_f = (c, e, lines)
        elif best_f is None:
            if best_i is None or e < best_i[0]: best_i = (e, seam_polylines(X, F))
    if best_f: return (0, best_f[0]), dict(feasible=True, seam_cost=best_f[0], distortion=best_f[1], lines=best_f[2])
    return (1, best_i[0] - BD), dict(feasible=False, distortion=best_i[0], seam_cost=seam_cost(best_i[1], w_bend, R), lines=best_i[1])

def main():
    mesh, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    # 2026-11-04: diagnostic trace (JSON lines) next to the result unless a trace directory is given
    os.environ.setdefault("ONESTRING_TRACE_DIR", os.path.join(out, "trace"))
    sys.path.insert(0, str(ROOT / "src"))
    from onestring_physics.run_trace import trace
    trace("optcuts.multistart", "start", mesh=mesh, env={k: v for k, v in os.environ.items() if k.startswith(("ONESTRING_", "MS_"))})
    w_angle = float(os.environ.get("ONESTRING_ANGLE_W", "0")); bound_sd = os.environ.get("ONESTRING_ANGLE_BOUND_SD", "0") == "1"
    w_bend = float(os.environ.get("ONESTRING_SEAM_BEND_W", "0"))
    extra = {k: os.environ[k] for k in ("ONESTRING_ANGLE_W", "ONESTRING_ANGLE_BOUND_SD", "ONESTRING_SEAM_BEND_W", "ONESTRING_NO_BRANCH") if k in os.environ}
    V, F = load_mesh(mesh)
    R = math.sqrt(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / math.pi)
    global SYM
    w_sym = float(os.environ.get("ONESTRING_SYM_W", "0"))
    sym_info = None
    if w_sym > 0:
        n_, d_, e_ = detect_mirror_plane(V, F)
        el = np.linalg.norm(V[F] - np.roll(V[F], 1, axis=1), axis=2)
        tol_ = 0.6 * float(el.mean())
        sym_info = dict(normal=n_.round(6).tolist(), offset=round(float(d_), 6), error=float(e_), tol=tol_, w=w_sym, used=True)
        print(f"[MS] mirror plane: n {np.round(n_, 4)} d {d_:.4f}, mean mirror distance {e_:.2e} of the diagonal", flush=True)
        if e_ < 5e-3:
            SYM = dict(n=n_, d=d_, tol=tol_, w=w_sym)
            extra.update(ONESTRING_SYM_W=str(w_sym), ONESTRING_SYM_PLANE=f"{n_[0]:.9f},{n_[1]:.9f},{n_[2]:.9f},{d_:.9f}",
                         ONESTRING_SYM_TOL=f"{tol_:.9f}")
        else:
            print("[MS] the shape is not mirror symmetric enough; symmetry term off", flush=True)
            sym_info["used"] = False
    t0 = time.time()
    # 1. OptCuts' own candidate list at its first split
    cf = os.path.join(out, "start_candidates.txt")
    run_branch((mesh, os.path.join(out, "probe"), None, None, dict(extra, ONESTRING_PRINT_START_CANDS=cf)))
    ranked = [int(l.split()[0]) for l in open(cf) if l.strip()]
    n = max(2, int(len(ranked) ** 0.8)); cands = ranked[:n]
    print(f"[MS] {len(ranked)} boundary vertices ranked by OptCuts; taking its top {n} as seam starting points", flush=True)
    if SYM is not None:
        # boundary vertices on the mirror plane (a seam along the plane is symmetric by itself)
        on = [v for v in ranked if abs(V[v] @ SYM["n"] - SYM["d"]) < SYM["tol"]]
        add = [v for v in on if v not in cands]
        cands += add
        print(f"[MS] {len(on)} ranked boundary vertices lie on the mirror plane; {len(add)} added as starting points", flush=True)
    alive = list(cands); budget = 1; rounds = []; final = {}
    pool = ProcessPoolExecutor(max_workers=2)
    while True:
        todo = [v for v in alive if v not in final]
        res = list(pool.map(run_branch, [(mesh, os.path.join(out, f"b{v}"), v, budget, extra) for v in todo]))
        scored = {}
        for v, r in zip(todo, res):
            if r["finished"]:
                ch = pickle.load(open(os.path.join(r["dump"], "chart.pkl"), "rb"))
                Xc = np.zeros((len(ch["uv"]), 3)); Xc[ch["f2"].ravel()] = ch["xyz"][ch["f3"].ravel()]
                key, det = score_states([(Xc, ch["f2"], ch["uv"])], w_angle, bound_sd, w_bend, R)
                final[v] = (key, det, len(frames_of(r["dump"])))
            else:
                fr = frames_of(r["dump"])
                if not fr: scored[v] = ((2, 0.0), dict(feasible=False, distortion=float("nan"), seam_cost=0.0, lines=[])); continue
                scored[v] = score_states([(f["x"], f["F"], f["uv"]) for f in fr], w_angle, bound_sd, w_bend, R)
        for v in alive:
            if v in final: scored[v] = final[v][:2]
        order = sorted(alive, key=lambda v: scored[v][0])
        for v in alive:
            r_ = next((r for vv, r in zip(todo, res) if vv == v), None); fr_ = frames_of(os.path.join(out, f"b{v}"))
            last_ = fr_[-1] if fr_ else None
            trace("optcuts.multistart", "branch", budget=budget, start=int(v), finished=v in final, error=(r_ or {}).get("err"),
                  frames=len(fr_), last_uv_vertices=(len(last_["uv"]) if last_ else None),
                  last_boundary_loops=(boundary_loop_count(last_["F"], len(last_["uv"])) if last_ else None),
                  feasible=scored[v][1]["feasible"], distortion=scored[v][1]["distortion"], seam_cost=scored[v][1]["seam_cost"])
        keep = order[:max(1, math.ceil(len(order) / 2))] if len(order) > 1 else order
        rounds.append(dict(budget=budget, branches=[dict(start=int(v), finished=v in final, rank=i, kept=v in keep,
                                                         feasible=scored[v][1]["feasible"], seam_cost=scored[v][1]["seam_cost"],
                                                         distortion=scored[v][1]["distortion"],
                                                         lines=[np.round(P, 4).tolist() for P in scored[v][1]["lines"]])
                                                    for i, v in enumerate(order)]))
        best = order[0]
        print(f"[MS] round budget {budget}: {len(order)} branches -> keep {len(keep)}; best start {best} "
              f"({'feasible, seam cost %.3f' % scored[best][1]['seam_cost'] if scored[best][1]['feasible'] else 'distortion %.4f' % scored[best][1]['distortion']}), "
              f"finished {sum(v in final for v in order)}, {time.time() - t0:.0f}s", flush=True)
        if len(order) == 1 or len(keep) == 1:
            alive = keep; break
        alive = keep; budget *= 2
    win = alive[0]
    if win not in final:
        # 2026-11-03: the winner runs until OptCuts stops on its own (MS_FINAL_TIMEOUT=0: no wall-clock cap)
        r = run_branch((mesh, os.path.join(out, f"b{win}"), win, None, extra, float(os.environ.get("MS_FINAL_TIMEOUT", "1800"))))
        if not r["finished"]:
            # 2026-11-04: keep the record even when the bridge refuses the final chart (e.g. a seam that does not reach
            # the boundary): the winner's frames and the rounds are saved, the chart is left to a repair step.
            wd = os.path.join(out, f"b{win}")
            for f in glob.glob(os.path.join(out, "f*.bin")): os.remove(f)
            for f in sorted(glob.glob(os.path.join(wd, "f*.bin"))): shutil.copy(f, out)
            json.dump(dict(candidates=cands, ranked_all=len(ranked), winner=int(win), w_angle=w_angle, bound_sd=bound_sd, w_bend=w_bend,
                           symmetry=sym_info, final_error=r.get("err", ""), rounds=rounds),
                      open(os.path.join(out, "multistart.json"), "w"))
            print(f"[MS] winner: start vertex {win}; the bridge refused the final chart: {r.get('err', '')[:160]}", flush=True)
            trace("optcuts.multistart", "final_refused", winner=int(win), error=r.get("err", ""), frames=len(frames_of(wd)))
            sys.exit(3)
    wd = os.path.join(out, f"b{win}")
    ch = pickle.load(open(os.path.join(wd, "chart.pkl"), "rb"))
    Vc = np.zeros((len(ch["uv"]), 3)); Vc[ch["f2"].ravel()] = ch["xyz"][ch["f3"].ravel()]
    trimesh.Trimesh(Vc, ch["f2"], process=False).export(os.path.join(out, "cut.obj")); np.save(os.path.join(out, "uv.npy"), ch["uv"])
    pickle.dump(ch, open(os.path.join(out, "optcuts_chart.pkl"), "wb"))
    for f in glob.glob(os.path.join(out, "f*.bin")): os.remove(f)
    for f in sorted(glob.glob(os.path.join(wd, "f*.bin"))): shutil.copy(f, out)
    key, det = score_states([(Vc, ch["f2"], ch["uv"])], w_angle, bound_sd, w_bend, R)
    json.dump(dict(candidates=cands, ranked_all=len(ranked), winner=int(win), w_angle=w_angle, bound_sd=bound_sd, w_bend=w_bend,
                   symmetry=sym_info, final_asym_length=asym_length(det["lines"]),
                   final=dict(feasible=det["feasible"], seam_cost=det["seam_cost"], distortion=det["distortion"]), rounds=rounds),
              open(os.path.join(out, "multistart.json"), "w"))
    trace("optcuts.multistart", "final", winner=int(win), feasible=det["feasible"], seam_cost=det["seam_cost"], distortion=det["distortion"])
    print(f"[MS] winner: start vertex {win}; feasible {det['feasible']}, seam cost {det['seam_cost']:.3f}, distortion {det['distortion']:.4f}; "
          f"{len(rounds)} rounds, {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
