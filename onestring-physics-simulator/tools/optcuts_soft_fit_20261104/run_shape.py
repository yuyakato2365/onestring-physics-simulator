"""Full pipeline on another target shape with the best bunny settings (handoff 4: 10-30 align preset,
w_square 400, Square tail alpha 16 / beta 100, no polish, 600 panels).  Real OptCuts is run.

usage: python3 run_shape.py MESH_FILE OUT_DIR      (env THICK_RATIO: tile thickness / panel side, default 0.74)
"""
import os, pickle, runpy, sys, time
from pathlib import Path
import numpy as np
TOOLS = Path(__file__).resolve().parent; ROOT = Path(os.environ.get("ONESTRING_ROOT", TOOLS.parents[1]))   # repo root (tools/optcuts_soft_fit_20261104/..)
MESH, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ONESTRING_TRACE_DIR", str(OUT / "trace"))   # 2026-11-04: diagnostic trace (docs/TRACE_LOG_FORMAT.md)
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
os.environ["ONESTRING_RUNTIME_BASE_BOOTSTRAPPED"] = "1"
os.environ.setdefault("ONESTRING_K3D_PLANARITY_POLISH", "0")
os.environ.setdefault("ONESTRING_K3D_SQUARE_TAIL_ALPHA", "16")
os.environ.setdefault("ONESTRING_K3D_SQUARE_TAIL_BETA", "100")
_snap_p = ROOT / "logs" / "structured_grid_last_input.pkl.orig"     # only for the saved-bunny-chart mode
SNAP = pickle.load(open(_snap_p, "rb")) if _snap_p.exists() else None
if os.environ.get("TARGET_XYZ"):
    SNAP["xyz"] = np.load(os.environ["TARGET_XYZ"])        # same mesh and chart, vertices moved (bunny only)
    print(f"[RUN] bunny target replaced by {os.environ['TARGET_XYZ']}", flush=True)
_orig = runpy.run_path
runpy.run_path = lambda path, *a, **k: {} if Path(path).name == "app.py" else _orig(path, *a, **k)
_orig(str(ROOT / "app_optcuts_20260923_extrusion_aware.py"), run_name="__main__")
from onestring_physics import onestring_pipeline as pipeline
import onestring_physics as package
from onestring_physics.input_shape import HeightField, normalize_shape
from onestring_physics import paper_local_global_solvers as plg
from onestring_physics import optcuts_backend as ob
import trimesh

BUNNY = MESH.name == "bunny_saved_chart"
if BUNNY:
    # the bunny of the handoff: saved Mac OptCuts chart, target = its (cut-open) surface, as run_feat.py
    P = np.asarray(SNAP["xyz"], float).copy(); Fc = np.asarray(SNAP["f3"], int).copy()
    from onestring_physics import structured_grid_20261028 as sg
    def fake_optcuts_once(V, F, cfg=None):
        loops = ob._boundary_loops(SNAP["f2"])
        return ob.OptCutsResult(surface_vertices_3d=SNAP["xyz"].copy(), surface_faces=SNAP["f3"].copy(),
                                uv_vertices_2d=SNAP["uv"].copy(), uv_faces=SNAP["f2"].copy(), boundary_loops=loops,
                                output_obj="(saved Mac chart)", metrics={"parameterization_backend_name": "saved_mac_optcuts_chart",
                                                                          "optcuts_uv_boundary_loop_count": len(loops)})
    ob._run_official_optcuts_once = fake_optcuts_once
    _orig_plan = sg.plan_structured_grid
    sg.plan_structured_grid = lambda xyz, f3, uv, f2, params: _orig_plan(SNAP["xyz"], SNAP["f3"], SNAP["uv"], SNAP["f2"], params)
elif os.environ.get("CHART_PKL"):
    # 2026-11-02: reuse a saved OptCuts chart (same target mesh), e.g. to rerun the grid stage with other settings
    CH = pickle.load(open(os.environ["CHART_PKL"], "rb"))
    m = trimesh.load(str(MESH), force="mesh", process=False)
    P = np.asarray(m.vertices, float) if os.environ.get("NO_NORMALIZE") else normalize_shape(np.asarray(m.vertices, float))
    Fc = np.asarray(m.faces, int)
    def fake_optcuts_chart(V, F, cfg=None):
        loops = ob._boundary_loops(np.asarray(CH["f2"]))
        return ob.OptCutsResult(surface_vertices_3d=np.asarray(CH["xyz"]).copy(), surface_faces=np.asarray(CH["f3"]).copy(),
                                uv_vertices_2d=np.asarray(CH["uv"]).copy(), uv_faces=np.asarray(CH["f2"]).copy(), boundary_loops=loops,
                                output_obj="(saved chart)", metrics={"parameterization_backend_name": "saved_optcuts_chart",
                                                                     "optcuts_uv_boundary_loop_count": len(loops)})
    ob._run_official_optcuts_once = fake_optcuts_chart
    print(f"[RUN] OptCuts chart taken from {os.environ['CHART_PKL']}", flush=True)
else:
    m = trimesh.load(str(MESH), force="mesh", process=False)
    P = np.asarray(m.vertices, float) if os.environ.get("NO_NORMALIZE") else normalize_shape(np.asarray(m.vertices, float))
    Fc = np.asarray(m.faces, int)
area = 0.5 * np.linalg.norm(np.cross(P[Fc[:, 1]] - P[Fc[:, 0]], P[Fc[:, 2]] - P[Fc[:, 0]]), axis=1).sum()
target = HeightField("sampled", points=P, faces=Fc)

params = pipeline.PipelineParameters()
for k, v in SNAP["params"].items():
    if hasattr(params, k):
        try: setattr(params, k, v)
        except Exception: pass
# same settings as run_feat.py (handoff 4 run) -------------------------------------------------
params.structured_grid_feature_preset = os.environ.get("PRESET", "align_shape_even_keep_20261030")
params.structured_grid_auto_panels = int(os.environ.get("PANELS", "600"))
params.structured_grid_feature_mode = "off"; params.structured_grid_feature_design = False
params.structured_grid_align_weight = 10.0; params.structured_grid_feature_kinds = "valley"
params.w_square = float(os.environ.get("W_SQUARE", "400"))
if os.environ.get("SEAM_ZIPPER"):
    params.structured_grid_seam_zipper = True      # 2026-11-03: seams close with 4 panels everywhere (no 3/5-panel vertices)
    print("[RUN] seam zipper: every seam closes with 4 panels around each of its points", flush=True)
if os.environ.get("TURN_SURFACE"):
    params.structured_grid_turn_reference = "surface"   # 2026-11-03: logical corners from the 3D geodesic turning
    print("[RUN] grid corners placed by the 3D surface turning", flush=True)
if os.environ.get("W_SURFACE"):
    params.w_surface = float(os.environ["W_SURFACE"])   # 2026-11-03: K3D surface-fidelity weight (paper 0.1)
if os.environ.get("W_PLANAR"):
    params.w_planar = float(os.environ["W_PLANAR"])
params.structured_grid_panel_planar_weight = 0.0; params.structured_grid_panel_principal_weight = 0.0
params.structured_grid_feature_crease_hold = "slide"; params.structured_grid_feature_min_mean_bend = 0.0
params.structured_grid_feature_min_length = 2.0; params.structured_grid_feature_schedule = "1,3,10,30,100"
params.structured_grid_feature_slide_weight = 100.0
if os.environ.get("MODE_PAPER"):
    # 2026-11-02: the app's paper-faithful preset (official BFF, rectangle boundary, per-part CSF split) with the
    # paper's E_Assembled weights and the square-auxetic AREA limit 2 (= linear sqrt 2)
    from onestring_physics.paper_fidelity_mode import configure_paper_fidelity
    params.optcuts_structured_grid_20261028_enabled = False
    params, _ov = configure_paper_fidelity(params)
    params.paper_split_area_limit_experiment = os.environ.get("PAPER_LIMIT", "area2") == "area2"
    params.w_planar, params.w_square, params.w_surface = 10000.0, 10.0, 0.1
    params.bff_executable = os.environ.setdefault("ONESTRING_BFF_EXECUTABLE", str(ROOT / "third_party" / "bff" / "build" / "bff-command-line"))
    if os.environ.get("SPLIT_CHOICES"):
        # 2026-11-02: fix the split sequence, e.g. "stop" (no split) or "auto,stop" (one cut at the curvature peak)
        params.paper_split_direction_choices = tuple(os.environ["SPLIT_CHOICES"].split(","))
    if os.environ.get("FIT_RECT"):
        params.reference_grid_fit_rectangle = True      # 2026-11-02: fit the grid to the rectangle Omega (no partial cells)
    if os.environ.get("GRID"):
        params.nx = params.ny = int(os.environ["GRID"])      # 2026-11-02: M2D grid resolution (Lilium: 14 -> about 193 tiles)
    print(f"[RUN] paper preset: omega {params.omega_parameterization_mode}, grid {params.nx}x{params.ny}, "
          f"CSF sampling {getattr(params, 'paper_csf_sampling_method', '?')}, area-limit 2", flush=True)
if os.environ.get("MODE_0918"):
    # 2026-11-02: the 09-18 mainline (OptCuts Omega + paper-aligned local/global K3D/K2D) with the app's settings
    params.omega_parameterization_mode = "optcuts_paper_local_global_k3d_k2d"
    params.optcuts_structured_grid_20261028_enabled = False
    params.w_square = float(SNAP["params"].get("w_square", params.w_square))
    print(f"[RUN] 09-18 mode: grid {params.nx}x{params.ny}, w_square {params.w_square}", flush=True)
if os.environ.get("PAPER_SPLIT"):
    # 2026-11-02: user's idea - cut along the grid line in Omega, carry it back to S, split, re-flatten every
    # part with official BFF (app's experimental "reflatten_tile_area" Split, area-span limit 2 per part)
    params.paper_split_criterion = os.environ["PAPER_SPLIT"]
    params.paper_split_apply_rebff_m2d = bool(os.environ.get("REBFF_M2D"))
    print(f"[RUN] split criterion {params.paper_split_criterion}, reBFF applied to M2D {params.paper_split_apply_rebff_m2d}", flush=True)
if os.environ.get("OMEGA_MINSPREAD"):
    # 2026-11-02: Omega = conformal BFF map whose boundary scale makes the tile area scale as even as possible
    # (bff_minspread.py; BFF construction, same as official BFF when the boundary scale is 0).  No CSF split.
    sys.path.insert(0, str(TOOLS))
    import bff_minspread as _ms
    from onestring_physics.reference_bff import BFFRunResult as _BR
    params.enable_csf_splits = False
    params.bff_rectangle_selection_mode = "fixed_phase"
    params.bff_boundary_policy = "automatic_reference"
    def _minspread_bff(V, F, **kw):
        V = np.asarray(V, float); F = np.asarray(F, int)
        Pf = V[F]; a = 0.5 * np.linalg.norm(np.cross(Pf[:, 1] - Pf[:, 0], Pf[:, 2] - Pf[:, 0]), axis=1).sum()
        uv_, info_ = _ms.minspread_bff(V, F, float(np.sqrt(a / float(os.environ.get("MINSPREAD_PANELS", "252")))))
        print(f"[RUN] min-spread BFF Omega: {len(V)} v, {info_}", flush=True)
        base_ = _orig_bff(V, F, **kw)          # official BFF (boundary scale 0): keeps every metric key the app reads
        return _BR(uv_vertices=uv_, metrics={**base_.metrics, "minspread_boundary_scale": True, "minspread": info_})
    _orig_bff = pipeline.run_official_bff
    pipeline.run_official_bff = _minspread_bff
if os.environ.get("OMEGA_UV_NPY"):
    # 2026-11-02: Omega = a given UV of the target mesh (same vertex order), e.g. OptCuts with the shape penalty.
    # Same downstream as the BFF runs (paper-mode grid on Omega, no CSF split).
    from onestring_physics.reference_bff import BFFRunResult as _BR2
    params.enable_csf_splits = False
    params.bff_rectangle_selection_mode = "fixed_phase"
    params.bff_boundary_policy = "automatic_reference"
    _given_uv = np.load(os.environ["OMEGA_UV_NPY"])
    def _given_bff(V, F, **kw):
        if len(V) != len(_given_uv):
            raise ValueError(f"given UV has {len(_given_uv)} vertices, surface has {len(V)}")
        base_ = _orig_bff2(V, F, **kw)
        print(f"[RUN] Omega taken from {os.environ['OMEGA_UV_NPY']} ({len(V)} v)", flush=True)
        return _BR2(uv_vertices=_given_uv.copy(), metrics={**base_.metrics, "given_uv": os.environ["OMEGA_UV_NPY"]})
    _orig_bff2 = pipeline.run_official_bff
    pipeline.run_official_bff = _given_bff
if os.environ.get("NO_SPLIT"):
    # 2026-11-02: OptCuts seams take the place of Split: no CSF split and no collision-driven split
    params.enable_csf_splits = False
    params.collision_split_enabled = False
    print("[RUN] CSF and collision splits off (OptCuts seams only)", flush=True)
if os.environ.get("T2D_UNLIMITED"):
    # 2026-11-02: let every T2D (dual hinge) trial run until its solver converges (0 = no wall-clock budget)
    params.hinge_layout_time_budget_sec = 0.0
    params.hinge_layout_iterations = 10 ** 7          # L-BFGS-B then stops on its own tolerances (ftol / gtol)
    print("[RUN] T2D time budget and iteration cap off (each trial runs to convergence)", flush=True)
if os.environ.get("PRUNE") is not None:
    params.structured_grid_prune_branch_panels = float(os.environ["PRUNE"])   # 0 keeps every OptCuts cut branch
# thickness: same ratio to the panel side as the bunny run (0.08 / 0.108 = 0.74)
side = np.sqrt(area / params.structured_grid_auto_panels)
params.thickness = float(os.environ.get("THICK_RATIO", "0.74")) * side
if BUNNY:
    params.thickness = 0.08          # as the handoff runs
if os.environ.get("THICK_ABS"):
    params.thickness = float(os.environ["THICK_ABS"])     # keep the thickness of the unsmoothed run
print(f"[RUN] {MESH.name}: {len(P)} v {len(Fc)} f, area {area:.3f}, panel side {side:.4f}, thickness {params.thickness:.4f}, "
      f"preset {params.structured_grid_feature_preset} w_square {params.w_square} alpha {os.environ['ONESTRING_K3D_SQUARE_TAIL_ALPHA']}", flush=True)

# keep the OptCuts chart and the K3D solver output
if os.environ.get("OPTCUTS_DISK_FIX"):
    # 2026-11-01: OptCuts sometimes returns two separate seam trees (UV chart = annulus, two boundary loops).
    # Join them with the shortest interior path, cut along it and relax the UV (SLIM) -> one disk chart.
    sys.path.insert(0, str(TOOLS))
    from annulus_fix import fix as _disk_fix
    _read = ob._read_obj_with_uv
    def _read_fixed(path):
        xyz_, f3_, uv_, f2_ = _read(path)
        n_ = len(ob._boundary_loops(np.asarray(f2_)))
        # 2026-11-02: join any number of loops, two at a time
        while n_ >= 2:
            r_ = _disk_fix(xyz_, f3_, uv_, f2_)
            if r_ is None or r_[2]["loops_after"] >= n_:
                break
            print(f"[RUN] OptCuts chart had {n_} boundary loops; joined two: {r_[2]}", flush=True)
            uv_, f2_ = r_[0], r_[1]; n_ = r_[2]["loops_after"]
        return xyz_, f3_, uv_, f2_
    ob._read_obj_with_uv = _read_fixed
_once = ob._run_official_optcuts_once
def _once_save(V, F, cfg=None):
    t = time.time()
    if os.environ.get("OPTCUTS_EXE"):
        # 2026-11-02: OptCuts copy with the CSF condition (ONESTRING_CSF_* env is passed to the binary)
        from dataclasses import replace as _rep0
        cfg = _rep0(cfg or ob.OptCutsConfig(), executable=os.environ["OPTCUTS_EXE"],
                    timeout_seconds=float(os.environ.get("OPTCUTS_TIMEOUT", "600")))
        print(f"[RUN] OptCuts binary {cfg.executable} (CSF mode {os.environ.get('ONESTRING_CSF_MODE', '0')}), timeout {cfg.timeout_seconds:.0f}s", flush=True)
    try:
        r = _once(V, F, cfg)
    except ob.OptCutsOutputError as e:
        # 2026-11-01: the official OptCuts cut sometimes leaves two boundary loops; retry once from the other initial cut
        from dataclasses import replace as _rep
        c0 = cfg or ob.OptCutsConfig()
        print(f"[RUN] OptCuts output rejected ({str(e)[:80]}); retrying with initial cut option {1 - c0.initial_cut_option}", flush=True)
        r = _once(V, F, _rep(c0, initial_cut_option=1 - c0.initial_cut_option))
    pickle.dump({"xyz": r.surface_vertices_3d, "f3": r.surface_faces, "uv": r.uv_vertices_2d, "f2": r.uv_faces},
                open(OUT / "optcuts_chart.pkl", "wb"))
    print(f"[RUN] OptCuts {time.time() - t:.0f}s: {len(r.uv_vertices_2d)} uv verts, {len(r.boundary_loops)} loops", flush=True)
    return r
ob._run_official_optcuts_once = _once_save
if os.environ.get("ANIM_TRACE"):
    # 2026-11-03: K3D and K2D run until they have converged (the iteration numbers below are only guards) and every
    # K3D step is kept for the animation; T2D keeps a pose every ONESTRING_T2D_HISTORY_STRIDE solver iterations.
    #   K3D converged: the best E_Assembled improved by less than 1e-4 (relative) over the last 50 iterations; the best iterate is returned
    #   K2D converged: the best E_Flat improved by less than 1% over the last 1000 iterations (or the app's own test)
    import inspect as _insp
    from onestring_physics import paper_hinge_solver as _phs
    from onestring_physics import lscm_latest_omega_hybrid_20260914_patch as _lh0
    os.environ["ONESTRING_PAPER_K3D_ITERATIONS"] = os.environ.get("K3D_MAX_ITER", "100000")
    os.environ["ONESTRING_EQ5_ITERATIONS"] = os.environ.get("K2D_MAX_ITER", "100000")
    os.environ.setdefault("ONESTRING_T2D_HISTORY_STRIDE", "50")
    plg._OS_K3D_FRAMES = []
    def _os_k3d_step(x_, records_):
        # converged when the best E_Assembled so far improved by less than 0.01 % over the last 50 iterations
        # (2026-11-03: comparing two single iterates 10 apart stopped one run early when E went down and back up)
        plg._OS_K3D_FRAMES.append((int(records_[-1]["iteration"]), float(records_[-1]["EAssembled"]), np.asarray(x_, float).copy()))
        if len(records_) > 51:
            e_ = np.array([r_["EAssembled"] for r_ in records_])
            return e_.min() >= e_[:-50].min() * (1 - 1e-4)
        return False
    def _os_k2d_converged(records_):
        # K2D's collision set changes from step to step, so E_Flat jitters instead of settling and keeps creeping down
        # very slowly; it is taken as converged when its best value improved by less than 1% over the last 1000
        # iterations (about 1e-5 per iteration, the usual relative tolerance)
        n_ = len(records_)
        if n_ % 500 == 0:
            e_ = [r_["EFlat"] for r_ in records_]
            print(f"[K2D-TRACE] iter {n_} E_Flat {e_[-1]:.6g} best {min(e_):.6g} collisions {records_[-1]['collisions']}", flush=True)
        if n_ > 2000:
            e_ = np.array([r_["EFlat"] for r_ in records_[-2000:]])
            best_before_, best_recent_ = e_[:1000].min(), e_[1000:].min()
            return best_recent_ >= best_before_ * (1 - 1e-2)
        return False
    def _os_k3d_best(x_):
        fr_ = plg._OS_K3D_FRAMES
        if not fr_:
            return x_
        i_ = int(np.argmin([f_[1] for f_ in fr_]))
        plg._OS_K3D_BEST = int(fr_[i_][0])
        print(f"[RUN] K3D: smallest E_Assembled {fr_[i_][1]:.6g} at iteration {fr_[i_][0]} of {fr_[-1][0]} (last {fr_[-1][1]:.6g}); that iterate is used", flush=True)
        return np.asarray(fr_[i_][2], float).copy()
    plg._os_k3d_best = _os_k3d_best
    plg._os_k3d_step = _os_k3d_step; plg._os_k2d_converged = _os_k2d_converged
    _src = _insp.getsource(plg.optimize_paper_local_global_k3d)
    _a = "        records.append(_checkpoint(x,it+1,step))\n        if step<1e-8: break\n"
    assert _src.count(_a) == 1
    _src = _src.replace(_a, _a + "        if _os_k3d_step(x, records): break\n")
    # 2026-11-03: the worst-panel (tail) reweighting changes the objective every iteration; once E_Assembled (the
    # paper's objective) starts to rise again it trades surface fidelity for the squareness of the worst panels.
    # The iterate with the smallest E_Assembled is returned (the whole trajectory stays in the animation).
    _b = "    extrusion_metrics = {}\n"
    assert _src.count(_b) == 1
    _src = _src.replace(_b, "    x = _os_k3d_best(x)\n" + _b)
    exec(compile(_src, plg.__file__, "exec"), plg.__dict__)
    _src = _insp.getsource(plg.optimize_paper_local_global_k2d)
    _a = "        if step<1e-9 and not collision_pairs and fab==0: break\n"
    assert _src.count(_a) == 1
    _src = _src.replace(_a, "        if (step<1e-9 and not collision_pairs and fab==0) or _os_k2d_converged(records): break\n")
    exec(compile(_src, plg.__file__, "exec"), plg.__dict__)
    _lh0.optimize_paper_local_global_k3d = plg.optimize_paper_local_global_k3d
    # 2026-11-03: grid placement of the structured-grid route is recorded too (every accepted SLIM step of the final
    # placement map, every accepted L-BFGS step of the panel-quality step); both run until their own tolerances.
    from onestring_physics import aligned_grid_20261028 as _ag
    from onestring_physics import structured_grid_20261028 as _sg
    from onestring_physics import zhang2021_panel_optimizer as _po
    _src = _insp.getsource(_ag.aligned_slim)
    _a = "        uv, E, J, Esd = trial, Et, Jt, Esd_t\n"
    assert _src.count(_a) == 1
    # convergence of the recorded placement: total energy improved by < 1e-5 (relative) over the last 20 accepted
    # steps (the align stage has no stopping test of its own and the plain stage stops at a fixed count)
    _src = _src.replace(_a, _a + "        if _OS_REC is not None:\n"
                        "            _OS_REC.append((uv.copy(), float(Esd))); _OS_EH.append(float(E))\n"
                        "            if len(_OS_EH) > 20 and (_OS_EH[-21] - _OS_EH[-1]) <= 1e-5 * max(abs(_OS_EH[-1]), 1e-30): break\n")
    _b = "    uv = np.asarray(uv0, float).copy()\n"
    assert _src.count(_b) == 1
    _src = _src.replace(_b, _b + "    _OS_EH.clear()\n")
    exec(compile(_src, _ag.__file__, "exec"), _ag.__dict__)
    _ag._OS_REC = None
    _ag._OS_EH = []
    _src = _insp.getsource(_ag.place_grid_aligned)
    for _a, _b in (("iters=150, area_weight=area_weight)", "iters=100000, area_weight=area_weight)"),
                   ("n_it = 4 * int(features.get(\"iters\", 40)) if soft is None else 0",
                    "n_it = 100000 if soft is None else 0")):
        assert _src.count(_a) == 1, _a
        _src = _src.replace(_a, _b)
    exec(compile(_src, _ag.__file__, "exec"), _ag.__dict__)
    _pga0 = _ag.place_grid_aligned
    M2D_ANIM = {"place": [], "panel": []}
    def _pull_grid(cur, src_, dom_, logical_, walk_, chart_, h_):
        uvm_, f2_ = chart_[0], chart_[1]
        lg_ = np.asarray(logical_, float) * h_
        uv_ = np.full((len(logical_), 2), np.nan)
        chain_of = {k: (verts, ax) for k, verts, ax in _ag._side_chains(src_, dom_)}
        for vtx, k, _j in walk_:
            verts, _ax = chain_of[k]
            d_ = _sg.DIRS[dom_["dirs"][k]].astype(float)
            s_ = cur[verts] @ d_
            target_ = float(lg_[vtx] @ d_)
            i_ = int(np.clip(np.searchsorted(s_, target_) - 1, 0, len(s_) - 2)) if np.all(np.diff(s_) > 0) else int(np.argmin(np.abs(s_[:-1] - target_)))
            lam_ = 0.0 if s_[i_ + 1] == s_[i_] else float(np.clip((target_ - s_[i_]) / (s_[i_ + 1] - s_[i_]), 0.0, 1.0))
            uv_[vtx] = uvm_[verts[i_]] + lam_ * (uvm_[verts[i_ + 1]] - uvm_[verts[i_]])
        bnd_ = np.array(sorted({w[0] for w in walk_}))
        gfree_ = np.setdiff1d(np.arange(len(logical_)), bnd_)
        out_ = _sg._locate_bary(cur, f2_, lg_[gfree_], uvm_)
        if out_ is None:
            return None
        uv_[gfree_] = out_
        return uv_
    def _pga(src_, dom_, logical_, quads_, walk_, chart_, h_, area_weight=0.0, features=None):
        _ag._OS_REC = []
        try:
            res_ = _pga0(src_, dom_, logical_, quads_, walk_, chart_, h_, area_weight=area_weight, features=features)
        finally:
            rec_ = _ag._OS_REC; _ag._OS_REC = None
        frames_ = []
        for cur_, E_ in rec_:
            g_ = _pull_grid(cur_, src_, dom_, logical_, walk_, chart_, h_)
            if g_ is not None:
                frames_.append((g_.astype(np.float32), E_))
        M2D_ANIM["place"] = frames_
        M2D_ANIM["place_quads"] = np.asarray(quads_, int)
        M2D_ANIM["chart_uv"] = np.asarray(chart_[0], np.float32); M2D_ANIM["chart_f"] = np.asarray(chart_[1], int)
        print(f"[RUN] grid placement recorded: {len(frames_)} SLIM steps", flush=True)
        return res_
    _ag.place_grid_aligned = _pga
    _src = _insp.getsource(_po._lbfgs)
    _a = "        x, f, g = xn, fn, gn\n"
    assert _src.count(_a) == 1
    _src = _src.replace(_a, _a + "        if _OS_PREC is not None:\n"
                        "            _OS_PREC.append((x.copy(), float(f)))\n"
                        "            if len(_OS_PREC) > 20 and (_OS_PREC[-21][1] - f) <= 1e-6 * max(abs(f), 1e-30): break\n")
    exec(compile(_src, _po.__file__, "exec"), _po.__dict__)
    _po._OS_PREC = None
    _opq0 = _po.optimize_panel_quality
    def _opq(lift_, quad_uv_, quads_, fixed_, **kw_):
        _po._OS_PREC = []
        try:
            res_ = _opq0(lift_, quad_uv_, quads_, fixed_, **kw_)
        finally:
            rec_ = _po._OS_PREC; _po._OS_PREC = None
        P0_ = np.asarray(quad_uv_, float); free_ = np.flatnonzero(~np.asarray(fixed_, bool)); nfz_ = 2 * len(free_)
        frames_ = []
        for z_, f_ in rec_:
            P_ = P0_.copy(); P_[free_] = np.asarray(z_[:nfz_]).reshape(-1, 2); frames_.append((P_.astype(np.float32), f_))
        M2D_ANIM["panel"] = frames_
        M2D_ANIM["panel_quads"] = np.asarray(quads_, int)
        M2D_ANIM["panel_final"] = np.asarray(res_[0], np.float32)
        print(f"[RUN] panel-quality step recorded: {len(frames_)} L-BFGS steps", flush=True)
        return res_
    _po.optimize_panel_quality = _opq
    params.zhang2021_panel_optimization_iterations = 10 ** 6      # L-BFGS stops on its own tolerances
    _lh0.optimize_paper_local_global_k2d = plg.optimize_paper_local_global_k2d
    _src = _insp.getsource(_phs.optimize_hinge_poses)
    _a = "    history_stride=max(1, int(np.ceil(max(1, params.hinge_layout_iterations)/180)))\n"
    assert _src.count(_a) == 1
    _src = _src.replace(_a, "    history_stride=int(os.environ.get('ONESTRING_T2D_HISTORY_STRIDE','50'))\n")
    # 2026-11-03: the hard collision repair keeps its own 480-step budget.  Running it to feasibility was tried: every
    # attempt then becomes overlap-free only by pushing tiles far apart, and the 24-step feasible refinement cannot
    # pull the hinges back (soft: 98.6 % -> 4.8 % connected), so the original budget is kept for now.
    _phs.os = os
    exec(compile(_src, _phs.__file__, "exec"), _phs.__dict__)
    print("[RUN] K3D / K2D run to convergence (traced); T2D pose history every "
          f"{os.environ['ONESTRING_T2D_HISTORY_STRIDE']} iterations", flush=True)
_k3d = plg.optimize_paper_local_global_k3d
def _k3d_save(target_, mesh, parameterization, params_, **kw):
    out, report = _k3d(target_, mesh, parameterization, params_, **kw)
    pickle.dump({"m3d_v": np.asarray(mesh.vertices, float), "m3d_f": np.asarray(mesh.faces, int),
                 "k3d_v": np.asarray(out.vertices, float), "k3d_f": np.asarray(out.faces, int),
                 "k3d_metrics": dict(out.metrics), "k3d_frames": list(getattr(plg, "_OS_K3D_FRAMES", [])),
                 "k3d_best_iteration": getattr(plg, "_OS_K3D_BEST", None)}, open(OUT / "k3d.pkl", "wb"))
    print("[RUN] K3D solver output saved", flush=True)
    if os.environ.get("ANIM_TRACE") and globals().get("M2D_ANIM"):
        pickle.dump(globals()["M2D_ANIM"], open(OUT / "m2d_anim.pkl", "wb"))
    if os.environ.get("STOP_AFTER_K3D"):
        print("[RUN] stop after K3D (diagnostic)", flush=True); sys.stdout.flush(); os._exit(0)
    return out, report
plg.optimize_paper_local_global_k3d = _k3d_save
from onestring_physics import lscm_latest_omega_hybrid_20260914_patch as _lh
if hasattr(_lh, "optimize_paper_local_global_k3d"):
    _lh.optimize_paper_local_global_k3d = _k3d_save

if os.environ.get("K2D_OPEN"):
    # 2026-11-01 diagnostic: the paper normalises the flat domain so that every tile's opening (flat pitch /
    # 3D tile size) is >= 1.  Here M2D cells are about the K3D tile size, so K2D starts closed with half the
    # tiles too large.  Scale a copy of M2D so that the smallest opening becomes K2D_OPEN (e.g. 1.05).
    import copy as _copy
    from onestring_physics import paper_local_global_solvers as _plg2
    _k2d = _plg2.optimize_paper_local_global_k2d
    def _quad_size(V, F):
        P = np.asarray(V, float)[np.asarray(F, int)][:, :, :3]
        if P.shape[2] == 2:
            P = np.concatenate([P, np.zeros(P.shape[:2] + (1,))], 2)
        return np.sqrt(0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]), axis=1))
    def _k2d_open(mesh_2d, mesh_3d, params_, **kw):
        F_ = np.asarray(mesh_2d.faces, int)
        r_ = _quad_size(mesh_2d.vertices, F_) / _quad_size(mesh_3d.vertices, F_)
        lo_ = float(np.percentile(r_, float(os.environ.get("K2D_OPEN_PCT", "0"))))
        s_ = float(os.environ["K2D_OPEN"]) / lo_
        m2 = _copy.deepcopy(mesh_2d)
        V_ = np.asarray(m2.vertices, float).copy(); c_ = V_.mean(0); c_[2:] = 0
        V_ = c_ + (V_ - c_) * s_
        try:
            m2.vertices = V_
        except Exception:
            object.__setattr__(m2, "vertices", V_)
        print(f"[RUN] K2D start scaled x{s_:.3f} (opening before: min {r_.min():.3f} median {np.median(r_):.3f} max {r_.max():.3f}; "
              f"after: min {r_.min() * s_:.3f} max {r_.max() * s_:.3f})", flush=True)
        return _k2d(m2, mesh_3d, params_, **kw)
    _plg2.optimize_paper_local_global_k2d = _k2d_open
    if hasattr(_lh, "optimize_paper_local_global_k2d"):
        _lh.optimize_paper_local_global_k2d = _k2d_open

t0 = time.time()
from onestring_physics.run_trace import trace as _trace
import re as _re
# only this driver's own switches (and ONESTRING_*) go to the trace -- never the rest of the environment
_RUN_SWITCHES = set(_re.findall(r'os\.environ(?:\.get\(|\[)"([A-Z0-9_]+)"', open(__file__).read()))
_trace("run", "start", mesh=str(MESH), out=str(OUT), argv=sys.argv,
       env={k: v for k, v in os.environ.items() if k.startswith(("ONESTRING_", "MS_")) or k in _RUN_SWITCHES},
       params={k: v for k, v in vars(params).items() if isinstance(v, (int, float, str, bool, type(None)))})
def progress(stage, fraction, detail=""):
    print(f"[PROGRESS {time.time() - t0:7.1f}s] {fraction:.2f} {stage} {detail}", flush=True)
    _trace("progress", str(stage), fraction=float(fraction), detail=str(detail))
try:
    state = package.build_onestring_design(target, params, run_simulation=False, progress_callback=progress)
except BaseException as _exc:
    import traceback as _tb
    _trace("run", "failed", error_type=type(_exc).__name__, error=str(_exc), traceback=_tb.format_exc())
    raise
_trace("run", "done", seconds=time.time() - t0)
print(f"[RUN] done in {time.time() - t0:.1f}s", flush=True)
pickle.dump(state, open(OUT / "state.pkl", "wb"))
if os.environ.get("ANIM_TRACE") and globals().get("M2D_ANIM"):
    pickle.dump(M2D_ANIM, open(OUT / "m2d_anim.pkl", "wb"))
pickle.dump({"P": P, "F": Fc, "area": area, "side": side, "thickness": params.thickness}, open(OUT / "target.pkl", "wb"))
print("[RUN] state saved", flush=True)
