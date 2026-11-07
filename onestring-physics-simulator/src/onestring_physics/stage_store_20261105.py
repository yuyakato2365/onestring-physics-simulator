"""2026-11-05: every stage of a run is saved to a folder named after the run's date, and a later run can continue from
any saved stage (after the app has been changed, too).

Folder: output/stage_results/<YYYY-MM-DD_HH-MM-SS>/
    1_omega.pkl ... 8_t2d_dual.pkl   everything the pipeline has computed up to and including that stage
    final_state.pkl                  the finished result, as shown on the page
    run.json                         when / which app version / which stages are there / where a continued run started

How it works: while a run is going, the body of legacy_pipeline_base.build_onestring_design (the function that runs
S → Ω → M2D → M3D → K3D → T3D → K2D → T2D Top → T2D Dual one after another) is replaced by the same body with a save
after each stage.  It runs with the legacy module's own names, so every patched stage function still runs exactly as
before.  To continue from stage k, the saved state after stage k - 1 is loaded and the body starts at stage k; the
stages before it are not computed again (their files are linked into the new run's folder, so it is complete by
itself).  Each stage file holds everything computed so far in one piece, so objects shared between stages stay
shared after loading.  Only the first pass through the pipeline in one run is saved / continued (the collision-split
trials, when enabled, run further passes that are computed as usual).
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "output" / "stage_results"

# (key, label, progress fraction at the end of the stage, figure kind, the saved name of its main result)
STAGES = [
    ("omega", "S → Ω（OptCuts の平面図）", 0.22, "omega", "parameterization"),
    ("m2d", "Ω → M2D", 0.30, "mesh2d", "mesh_2d_initial"),
    ("m3d", "M2D → M3D", 0.38, "mesh3d", "mesh_3d_initial"),
    ("k3d", "M3D → K3D", 0.50, "mesh3d", "mesh_3d_optimized"),
    ("t3d", "K3D → T3D", 0.56, "tiles3d", "tiles_3d"),
    ("k2d", "M2D → K2D", 0.73, "mesh2d", "mesh_2d_optimized"),
    ("t2d_top", "K2D → T2D Top Hinge", 0.81, "tiles2d", "tiles_2d_top"),
    ("t2d_dual", "T2D Dual Hinge", 0.94, "tiles2d", "tiles_2d_dual"),
]


def _stage_file(k: int) -> str:
    return f"{k + 1}_{STAGES[k][0]}.pkl"


def _write_pickle(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _app_version():
    try:
        from .app_version_20261105 import current_version
        return current_version(ROOT)["n"]
    except Exception:   # noqa: BLE001
        return None


def _new_run_dir() -> Path:
    STORE.mkdir(parents=True, exist_ok=True)
    base = time.strftime("%Y-%m-%d_%H-%M-%S")
    d, n = STORE / base, 1
    while d.exists():
        n += 1
        d = STORE / f"{base}_{n}"
    d.mkdir()
    return d


def _read_info(d: Path) -> dict:
    try:
        return json.loads((d / "run.json").read_text())
    except Exception:   # noqa: BLE001
        return {}


_SAVED_STATE_IDS: set[int] = set()     # results already written as final_state.pkl in this server process


# ------------------------------------------------------------------------------------------------ recording a run
class RunRecorder:
    """Saves the stages of one run; with resume=(folder, k) it continues that saved run from stage k."""

    def __init__(self, resume: dict | None = None, note: str = ""):
        self.dir = _new_run_dir()
        self.sink = None                    # the job's list of finished stages (figures on the page)
        self.start_stage = 0
        self.source = None
        self._claimed = False
        self.info = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), app_version=_app_version(), stages={},
                         note=note, final=False)
        if resume:
            self.source = STORE / resume["run"]
            self.start_stage = int(resume["stage"])
            self.info["continued_from"] = dict(run=resume["run"], stage=self.start_stage,
                                               stage_label=STAGES[self.start_stage][1])
            src_info = _read_info(self.source)
            for k in range(self.start_stage):           # the reused stages: link the files into this run's folder
                name = _stage_file(k)
                try:
                    os.link(self.source / name, self.dir / name)
                except OSError:
                    shutil.copy2(self.source / name, self.dir / name)
                entry = dict(src_info.get("stages", {}).get(STAGES[k][0], {}))
                entry["reused_from"] = resume["run"]
                self.info["stages"][STAGES[k][0]] = entry
        self._write_info()
        print(f"[STAGE-STORE] saving this run's stages to {self.dir}"
              + (f" (continuing {resume['run']} from {STAGES[self.start_stage][1]})" if resume else ""), flush=True)

    def _write_info(self):
        (self.dir / "run.json").write_text(json.dumps(self.info, ensure_ascii=False, indent=1))

    # called by the replaced pipeline body -------------------------------------------------------------------
    def begin_call(self):
        """the first pass through the pipeline is recorded (and continued); later passes are not"""
        if self._claimed:
            return None
        self._claimed = True
        return self

    def load(self) -> dict:
        if self.start_stage == 0:
            return {}
        with open(self.dir / _stage_file(self.start_stage - 1), "rb") as f:
            S = pickle.load(f)
        if self.sink is not None:                       # show the reused stages on the page like computed ones
            for k in range(self.start_stage):
                key, label, _, kind, main = STAGES[k]
                obj = S.get(main)
                if obj is not None:
                    self.sink.append(dict(title=f"{label}（保存した結果 {self.source.name} を使用）", kind=kind, obj=obj,
                                          seconds=0.0))
        return S

    def save(self, k: int, S: dict) -> None:
        key, label = STAGES[k][0], STAGES[k][1]
        t0 = time.perf_counter()
        try:
            _write_pickle(self.dir / _stage_file(k), S)
            self.info["stages"][key] = dict(saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                                            mb=round((self.dir / _stage_file(k)).stat().st_size / 1e6, 1))
            print(f"[STAGE-STORE] {label} saved ({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as e:   # noqa: BLE001  (saving must never stop the calculation)
            self.info["stages"][key] = dict(error=f"{type(e).__name__}: {e}")
            print(f"[STAGE-STORE] could not save {label}: {type(e).__name__}: {e}", flush=True)
        self._write_info()

    # called by the job after the pipeline returned --------------------------------------------------------------
    def save_final(self, state, pipeline_key=None, params=None) -> None:
        try:
            _write_pickle(self.dir / "final_state.pkl",
                          dict(onestring_state=state, pipeline_key=pipeline_key, onestring_last_completed_params=params))
            self.info["final"] = True
            self.info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _SAVED_STATE_IDS.add(id(state))
        except Exception as e:   # noqa: BLE001
            self.info["final_error"] = f"{type(e).__name__}: {e}"
            print(f"[STAGE-STORE] could not save the final result: {type(e).__name__}: {e}", flush=True)
        try:
            self.info["tiles"] = int(len(state.tiles_2d_dual_hinge.vertices))
            self.info["hinges"] = int(len(state.hinge_graph.hinges))
        except Exception:   # noqa: BLE001
            pass
        self._write_info()


def build_onestring_design(target, params=None, run_simulation=False, deployment_params=None, progress_callback=None):
    """The body of legacy_pipeline_base.build_onestring_design with a save after each stage (see the module text).
    Its code is put into the legacy function while a run is recorded, so every name below is the legacy module's."""
    rec = globals().get("__onestring_stage_recorder__")
    rec = rec.begin_call() if rec is not None else None
    params = params or PipelineParameters()
    from onestring_physics import extrusion_aware as experiment
    experimental = experiment.enabled(params)
    if experimental:
        experiment.validate(params)
    _validate_compute_config(params.compute)
    if getattr(params, "strict_paper_flow", False) and params.m3d_construction_mode != "mesh_harmonic":
        raise RuntimeError(
            "Strict paper flow disables analytic_scaled_heightfield_debug. "
            "Use mesh_harmonic so M2D is lifted through the stored S↔Omega parameterization."
        )
    ny = params.nx if params.ny is None else params.ny
    params.ny = ny
    start = rec.start_stage if rec is not None else 0
    S = rec.load() if rec is not None else {}
    rec_stages = __import__("onestring_physics.stage_store_20261105", fromlist=["STAGES"]).STAGES
    for k in range(start):
        _emit_progress(progress_callback, rec_stages[k][1], rec_stages[k][2], "保存した結果を使います")

    def done(k):
        if rec is not None:
            rec.save(k, dict(S))

    if start <= 0:
        S["runtime"] = {}
        _emit_progress(progress_callback, "Initialize grid", 0.02, "Create regular overlay grid")
        S["grid"] = create_quad_grid(params.nx, ny, params.tile_size, params.gap_size)
        S["reports"] = {}
        S["target_surface"] = _build_surface_mesh(target, S["grid"], params.surface_mesh_subdivisions)
        _emit_progress(progress_callback, "S: target surface mesh", 0.08, f"{len(S['target_surface'].vertices)} vertices")
        tick = time.perf_counter()
        S["parameterization"] = _build_surface_parameterization(S["target_surface"], target, S["grid"], params)
        S["runtime"]["parameterization"] = time.perf_counter() - tick
        if experimental:
            experiment.prepare_parameterization(S["parameterization"], params)
        _emit_progress(progress_callback, "S -> Ω", 0.16,
                       str(S["parameterization"].metrics.get("parameterization_method", S["parameterization"].method)))
        S["domain"] = _flatten_to_domain(S["parameterization"], S["grid"], params)
        if experimental and params.use_extrusion_aware_split and "extrusion_split_status" not in S["parameterization"].metrics:
            raise NotImplementedError("Extrusion-aware Split currently requires the common CSF candidate-selection route")
        _emit_progress(progress_callback, "Ω domain", 0.22, "Flattened domain ready")
        done(0)
    reports, runtime = S["reports"], S["runtime"]
    if start <= 1:
        tick = time.perf_counter()
        S["mesh_2d_initial"] = _build_m2d(S["grid"], S["domain"], params)
        _emit_progress(progress_callback, "Ω -> M2D", 0.30, f"kept {len(S['mesh_2d_initial'].faces)} quads")
        runtime["grid"] = time.perf_counter() - tick
        S["active_grid"] = S["mesh_2d_initial"].grid
        done(1)
    if start <= 2:
        S["mesh_3d_initial"], reports["M2D -> M3D"] = _lift_m2d_to_m3d(target, S["mesh_2d_initial"], S["parameterization"], params)
        from .split_geometry_constraints import propagate
        propagate(S["mesh_2d_initial"], S["mesh_3d_initial"])
        _emit_progress(progress_callback, "M2D -> M3D", 0.38, "Inverse map / surface lift done")
        done(2)
    if start <= 3:
        from .split_geometry_constraints import optimize_shared
        tick = time.perf_counter()
        if experimental:
            import sys
            S["mesh_3d_optimized"], reports["M3D -> K3D"] = experiment.optimize_assembled(
                _optimize_k3d, target, S["mesh_2d_initial"], S["mesh_3d_initial"], S["parameterization"], params,
                sys.modules[__name__])
        else:
            S["mesh_3d_optimized"], reports["M3D -> K3D"] = optimize_shared(
                _optimize_k3d, target, S["mesh_3d_initial"], S["parameterization"], params)
        runtime["K3D"] = time.perf_counter() - tick
        if experimental:
            m3 = S["mesh_3d_optimized"]
            mapping = np.asarray(m3.metrics.get("assembled_geometry_map", np.arange(len(m3.vertices))), int)
            view = experiment.geometry_view(m3, mapping)
            m3.metrics["extrusion_pre_t3d"] = experiment.quality_summary(
                experiment.evaluate_extrusion(view.vertices, view.faces, params.thickness))
        _emit_progress(progress_callback, "M3D -> K3D", 0.50, str(S["mesh_3d_optimized"].metrics.get("actual_backend", "cpu")))
        done(3)
    if start <= 4:
        tick = time.perf_counter()
        S["tiles_3d"], reports["K3D -> T3D"] = _extrude_tiles(S["mesh_3d_optimized"], params.thickness, "T3D")
        runtime["T3D"] = time.perf_counter() - tick
        _emit_progress(progress_callback, "K3D -> T3D", 0.56, "Extruded assembled tiles")
        done(4)
    if start <= 5:
        tick = time.perf_counter()
        S["mesh_2d_optimized"], reports["M2D -> K2D"] = _optimize_k2d(
            S["mesh_2d_initial"],
            S["mesh_3d_optimized"],
            params,
            progress_callback=_subprogress(progress_callback, 0.56, 0.70, "M2D -> K2D: "),
        )
        S["k2d_flat_layout"] = _make_flat_tile_layout(S["mesh_2d_optimized"], params)
        runtime["K2D"] = time.perf_counter() - tick
        _emit_progress(progress_callback, "K2D independent tile layout", 0.73, "K2D tile geometry ready for extrusion")
        lay = S["k2d_flat_layout"]
        S["mesh_2d_optimized"].metrics.update(
            {
                "k2d_tile_overlap_count": int(lay.metrics["tile_overlap_count"]),
                "k2d_min_clearance": float(lay.metrics["min_clearance"]),
                "k2d_gap_count": int(lay.metrics["k2d_gap_count"]),
                "flat_layout_type": str(lay.metrics["layout_type"]),
            }
        )
        done(5)
    if start <= 6:
        _emit_progress(progress_callback, "K2D -> T2D Top Hinge", 0.731, "Starting rigid flat-tile construction")
        S["tick_t2d"] = time.perf_counter()
        S["tiles_2d_top"], reports["K2D -> T2D top hinge"] = _make_t2d_from_transforms(
            S["mesh_2d_optimized"],
            S["k2d_flat_layout"],
            S["mesh_3d_optimized"],
            S["tiles_3d"],
            "T2D top hinge",
        )
        _emit_progress(progress_callback, "K2D -> T2D Top Hinge", 0.78, "Top-hinge T2D built from K2D and T3D transforms")
        S["source_faces"] = S["mesh_3d_optimized"].faces  # metric/side adjacency; K2D may use joint vertex IDs
        S["hinge_graph"] = _build_hinge_graph(S["active_grid"], S["source_faces"], S["tiles_2d_top"], S["tiles_3d"], dual=False)
        _emit_progress(progress_callback, "Build hinge graph", 0.81, f"{len(S['hinge_graph'].hinges)} pairwise hinges")
        done(6)
    if start <= 7:
        tick = S.get("tick_t2d", time.perf_counter()) if start < 7 else time.perf_counter()
        S["tiles_2d_dual"], S["hinge_graph"], reports["T2D top hinge -> T2D dual hinge"] = _optimize_dual_hinges(
            S["active_grid"],
            S["source_faces"],
            S["tiles_2d_top"],
            S["tiles_3d"],
            params,
            progress_callback=_subprogress(progress_callback, 0.81, 0.94, "Dual Hinge: "),
        )
        runtime["T2D"] = time.perf_counter() - tick
        done(7)
    tiles_3d, tiles_2d_dual, source_faces = S["tiles_3d"], S["tiles_2d_dual"], S["source_faces"]
    mesh_3d_optimized, mesh_2d_optimized = S["mesh_3d_optimized"], S["mesh_2d_optimized"]
    gap_graph = _build_gap_graph(source_faces, tiles_2d_dual, tiles_3d)
    _emit_progress(progress_callback, "Build gap graph", 0.96, f"{len(gap_graph.gaps)} gaps")
    lift_points = _select_lift_points(gap_graph, params.lift_tau)
    string_path = _build_string_path(gap_graph, lift_points, params.channel_friction)
    _emit_progress(progress_callback, "Lift points / string path", 0.98, f"{len(lift_points)} lift points")

    state = OneStringDesignState(
        target_surface=S["target_surface"],
        surface_parameterization=S["parameterization"],
        conformal_domain=S["domain"],
        mesh_2d_initial=S["mesh_2d_initial"],
        mesh_3d_initial=S["mesh_3d_initial"],
        mesh_3d_optimized=mesh_3d_optimized,
        tiles_3d=tiles_3d,
        mesh_2d_optimized=mesh_2d_optimized,
        k2d_flat_layout=S["k2d_flat_layout"],
        tiles_2d_top_hinge=S["tiles_2d_top"],
        tiles_2d_dual_hinge=tiles_2d_dual,
        hinge_graph=S["hinge_graph"],
        gap_graph=gap_graph,
        lift_points=lift_points,
        string_path=string_path,
        stage_reports=reports,
        backend_reports={
            "K3D": _stage_backend_report("K3D", params.compute, reports["M3D -> K3D"].computation_time, mesh_3d_optimized.metrics),
            "K2D": _stage_backend_report("K2D", params.compute, reports["M2D -> K2D"].computation_time, mesh_2d_optimized.metrics),
            "T2D/T3D": _stage_backend_report(
                "T2D/T3D",
                params.compute,
                reports["K2D -> T2D top hinge"].computation_time + reports["K3D -> T3D"].computation_time,
                {"actual_backend": "cpu", "cpu_stage": True, "fallback_reason": "geometry extrusion and hinge projection are CPU-side"},
            ),
        },
        approximations=[
            "Figure-5 order is now locked: K2D is not hinge-optimized before T2D Top Hinge; Dual Hinge is the only hinge-layout stage.",
            "Remaining mismatch from the paper: Boundary First Flattening is approximated by harmonic UV parameterization.",
            "Remaining mismatch from the paper: ShapeOp/libigl projection stack is approximated by NumPy/PyTorch local projection/least-squares steps.",
            "Remaining mismatch from the paper: CSF split placement is simplified.",
            "Remaining mismatch from the paper: Morse-Smale lift point clustering is approximated with GPE thresholding.",
            "Remaining mismatch from the paper: E_Collision uses AABB/local repulsion rather than full rigid tile collision constraints.",
            "Remaining mismatch from the paper: string routing friction uses a simplified Capstan-style turn cost.",
        ],
    )

    if experimental:
        experiment.record_run(state, params, runtime)

    if run_simulation:
        state.simulation_result = simulate_onestring_deployment(state, deployment_params, progress_callback=_subprogress(progress_callback, 0.98, 1.0, "deployment: "))
    _emit_progress(progress_callback, "Done", 1.0, "Pipeline complete")
    return state


_RECORDING_CODE = build_onestring_design.__code__
_ORIGINAL_CODES: dict[int, object] = {}
_LOCK = threading.Lock()


def _legacy_build_functions() -> list:
    """the legacy build function objects (the private copy the app runs, and the importable one)"""
    import sys
    cands = []
    try:
        from . import legacy_pipeline_base as lb
        cands.append(lb.build_onestring_design)
    except Exception:   # noqa: BLE001
        pass
    for m in list(sys.modules.values()):
        if not str(getattr(m, "__name__", "")).startswith("onestring_physics"):
            continue
        cands.append(getattr(m, "_ORIGINAL_BUILD_ONESTRING_DESIGN", None))
        o = getattr(m, "_original", None)
        if o is not None:
            cands.append(getattr(o, "build_onestring_design", None))
    out = {}
    for f in cands:
        seen = set()
        while f is not None and id(f) not in seen:
            seen.add(id(f))
            code = getattr(f, "__code__", None)
            if code is not None and code.co_name == "build_onestring_design" and (
                    str(code.co_filename).endswith("legacy_pipeline_base.py") or id(f) in _ORIGINAL_CODES):
                out[id(f)] = f
                break
            f = getattr(f, "__wrapped__", None)
    return list(out.values())


class recording:
    """with recording(recorder): the pipeline saves its stages through `recorder` (and continues where it says)"""

    def __init__(self, recorder: RunRecorder):
        self.recorder = recorder
        self.funcs = []

    def __enter__(self):
        _LOCK.acquire()
        self.funcs = _legacy_build_functions()
        for f in self.funcs:
            _ORIGINAL_CODES.setdefault(id(f), f.__code__)
            f.__code__ = _RECORDING_CODE
            f.__globals__["__onestring_stage_recorder__"] = self.recorder
        print(f"[STAGE-STORE] recording {len(self.funcs)} pipeline function(s)", flush=True)
        if not self.funcs:
            print("[STAGE-STORE] the pipeline function was not found: stages are not saved in this run", flush=True)
        return self.recorder

    def __exit__(self, *exc):
        for f in self.funcs:
            f.__code__ = _ORIGINAL_CODES.get(id(f), f.__code__)
            f.__globals__.pop("__onestring_stage_recorder__", None)
        _LOCK.release()
        return False


# --------------------------------------------------------------------------------------------- saved runs on disk
def list_runs() -> list[dict]:
    """newest first"""
    out = []
    for d in sorted(STORE.glob("*"), reverse=True) if STORE.is_dir() else []:
        if not d.is_dir():
            continue
        info = _read_info(d)
        info["run"] = d.name
        info["have"] = [k for k in range(len(STAGES)) if (d / _stage_file(k)).exists()]
        info["has_final"] = (d / "final_state.pkl").exists()
        if info["have"] or info["has_final"]:
            out.append(info)
    return out


def run_label(info: dict) -> str:
    parts = [info["run"]]
    parts.append(f"{len(info['have'])}/{len(STAGES)} 段階")
    if info.get("has_final"):
        parts.append("最終結果あり")
    if info.get("app_version") is not None:
        parts.append(f"バージョン {info['app_version']}")
    c = info.get("continued_from")
    if c:
        parts.append(f"{c['run']} の {c['stage_label']} から続き")
    if info.get("note"):
        parts.append(info["note"])
    return "・".join(parts)


def load_final(run: str, session_state) -> None:
    with open(STORE / run / "final_state.pkl", "rb") as f:
        payload = pickle.load(f)
    for k, v in payload.items():
        if v is not None:
            session_state[k] = v
    state = session_state["onestring_state"]
    try:
        session_state["eq5_history"] = getattr(state.mesh_2d_optimized, "eq5_history", None)
    except Exception:   # noqa: BLE001
        pass
    try:
        from .ui_result_recovery import prepare_completed_pipeline_display
        prepare_completed_pipeline_display(session_state, old_grid_size=None,
                                           new_grid_size=session_state.get("onestring_completed_grid_size") or 0)
    except Exception:   # noqa: BLE001
        pass
    session_state.pop("actuation_key", None)
    _SAVED_STATE_IDS.add(id(state))


def save_shown_result(session_state) -> None:
    """A result on the page that was not made by a recorded run (e.g. finished before this feature) is saved once as
    a run folder with only its final result."""
    state = session_state.get("onestring_state")
    if state is None or id(state) in _SAVED_STATE_IDS:
        return
    _SAVED_STATE_IDS.add(id(state))
    try:
        rec = RunRecorder(note="表示していた結果を保存（段階ごとの結果はなし）")
        rec.save_final(state, session_state.get("pipeline_key"), session_state.get("onestring_last_completed_params"))
    except Exception as e:   # noqa: BLE001
        print(f"[STAGE-STORE] could not save the shown result: {type(e).__name__}: {e}", flush=True)


def render_panel(st) -> None:
    """the saved runs: load a finished result, or continue a run from a chosen stage"""
    runs = list_runs()
    with st.expander(f"保存した各段階の結果（{len(runs)} 回分）　・　途中から続きを計算する",
                     expanded=("onestring_state" not in st.session_state and bool(runs))):
        if not runs:
            st.caption(f"まだありません。計算を実行すると、各段階の結果が {STORE.relative_to(ROOT)}/（実行した日時の名前のフォルダ）に保存されます。")
            return
        info = st.selectbox("保存した実行（フォルダ名 = 実行した日時）", runs, format_func=run_label, key="stage_store_run")
        rows = []
        for k, (key, label, *_rest) in enumerate(STAGES):
            e = info.get("stages", {}).get(key, {})
            have = k in info["have"]
            state_txt = ("✓" if have else "—") + (f"　{e['saved']}" if e.get("saved") else "") + (
                f"　{e['mb']} MB" if e.get("mb") is not None else "") + (
                f"　（{e['reused_from']} から引き継ぎ）" if e.get("reused_from") else "") + (
                f"　保存できず: {e['error']}" if e.get("error") else "")
            rows.append(f"{k + 1}. {label}：{state_txt}")
        st.markdown("  \n".join(rows))
        c1, c2 = st.columns(2)
        if c1.button("この実行の最終結果を読み込む", disabled=not info["has_final"], key="stage_store_load_final"):
            load_final(info["run"], st.session_state)
            st.rerun()
        starts = [k for k in range(len(STAGES)) if k == 0 or (k - 1) in info["have"]]
        k = c2.selectbox("ここから計算し直す", starts, index=len(starts) - 1, key="stage_store_start",
                         format_func=lambda k: f"{k + 1}. {STAGES[k][1]}")
        if c2.button("この段階から続きを計算する", key="stage_store_continue"):
            st.session_state["_stage_store_resume"] = dict(run=info["run"], stage=int(k))
            st.rerun()
        st.caption("選んだ段階より前は保存した結果をそのまま使い、選んだ段階から後を今の設定と今のコードで計算します。"
                   "前の段階に効く設定（入力メッシュ・格子の数・Ω の設定など）は、保存したときのものが使われます。"
                   "続きの結果も、新しいフォルダ（実行した日時の名前）に保存されます。")
