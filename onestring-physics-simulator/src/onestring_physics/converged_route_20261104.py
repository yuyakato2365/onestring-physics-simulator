"""2026-11-04: the convergence rules of the cloud comparison runs (tools/optcuts_soft_fit_20261104/run_shape.py with
ANIM_TRACE=1), available to the app.  ``install()`` patches the solvers once; the rules act only while
``set_active(True)`` (the app switches them per run), so every other mode keeps its previous behaviour.

* K3D (paper local/global): runs until the best E_Assembled improved by less than 1e-4 (relative) over the last 50
  iterations; the iterate with the smallest E_Assembled is returned (the worst-panel reweighting makes E rise again
  late in the run and trades surface fidelity for squareness).
* K2D (paper Eq.5): converged when its best E_Flat improved by less than 1 % over the last 500 iterations (or the
  solver's own test; 1000 until 2026-11-05).
* Grid placement (aligned SLIM of the structured grid): up to 100000 steps, stops when the total energy improved by less
  than 1e-5 (relative) over the last 20 accepted steps.
* Panel-quality step: L-BFGS stops when f improved by less than 1e-6 (relative) over the last 20 steps (the iteration
  number is set by the app to 10^6 while active).
* Iteration guards: ONESTRING_PAPER_K3D_ITERATIONS / ONESTRING_EQ5_ITERATIONS = 100000 while active.
"""
from __future__ import annotations

import inspect
import os

import numpy as np

_STATE = {"installed": False, "active": False}


def active() -> bool:
    return bool(_STATE["active"])


def set_active(flag: bool) -> None:
    _STATE["active"] = bool(flag)
    if flag:
        install()
        os.environ["ONESTRING_PAPER_K3D_ITERATIONS"] = "100000"
        os.environ["ONESTRING_EQ5_ITERATIONS"] = "100000"


def install() -> None:
    if _STATE["installed"]:
        return
    from . import paper_local_global_solvers as plg
    from . import lscm_latest_omega_hybrid_20260914_patch as lh
    from . import aligned_grid_20261028 as ag
    from . import zhang2021_panel_optimizer as po

    plg._OS_K3D_FRAMES = []

    def k3d_step(x_, records_):
        if not _STATE["active"]:
            return False
        plg._OS_K3D_FRAMES.append((int(records_[-1]["iteration"]), float(records_[-1]["EAssembled"]), np.asarray(x_, float).copy()))
        if len(records_) > 51:
            e_ = np.array([r_["EAssembled"] for r_ in records_])
            return bool(e_.min() >= e_[:-50].min() * (1 - 1e-4))
        return False

    def k3d_best(x_):
        fr_ = plg._OS_K3D_FRAMES
        plg._OS_K3D_FRAMES = []
        if not _STATE["active"] or not fr_:
            return x_
        i_ = int(np.argmin([f_[1] for f_ in fr_]))
        plg._OS_K3D_BEST = int(fr_[i_][0])
        print(f"[CONVERGED] K3D: smallest E_Assembled {fr_[i_][1]:.6g} at iteration {fr_[i_][0]} of {fr_[-1][0]} "
              f"(last {fr_[-1][1]:.6g}); that iterate is used", flush=True)
        return np.asarray(fr_[i_][2], float).copy()

    def k2d_converged(records_):
        if not _STATE["active"]:
            return False
        # 2026-11-05: window 500 (was 1000): stop when the best E_Flat of the last 500 iterations is less than 1 %
        # below the best of the 500 before (env ONESTRING_K2D_PLATEAU_WINDOW / ONESTRING_K2D_PLATEAU_REL)
        w_ = int(os.environ.get("ONESTRING_K2D_PLATEAU_WINDOW", "500"))
        rel_ = float(os.environ.get("ONESTRING_K2D_PLATEAU_REL", "0.01"))
        if len(records_) > 2 * w_:
            e_ = np.array([r_["EFlat"] for r_ in records_[-2 * w_:]])
            return bool(e_[w_:].min() >= e_[:w_].min() * (1 - rel_))
        return False

    plg._os_k3d_step, plg._os_k3d_best, plg._os_k2d_converged = k3d_step, k3d_best, k2d_converged
    src = inspect.getsource(plg.optimize_paper_local_global_k3d)
    a = "        records.append(_checkpoint(x,it+1,step))\n        if step<1e-8: break\n"
    b = "    extrusion_metrics = {}\n"
    if src.count(a) == 1 and src.count(b) == 1:
        src = src.replace(a, a + "        if _os_k3d_step(x, records): break\n").replace(b, "    x = _os_k3d_best(x)\n" + b)
        exec(compile(src, plg.__file__, "exec"), plg.__dict__)
    src = inspect.getsource(plg.optimize_paper_local_global_k2d)
    a = "        if step<1e-9 and not collision_pairs and fab==0: break\n"
    if src.count(a) == 1:
        src = src.replace(a, "        if (step<1e-9 and not collision_pairs and fab==0) or _os_k2d_converged(records): break\n")
        exec(compile(src, plg.__file__, "exec"), plg.__dict__)
    for mod in (lh,):
        if hasattr(mod, "optimize_paper_local_global_k3d"):
            mod.optimize_paper_local_global_k3d = plg.optimize_paper_local_global_k3d
        if hasattr(mod, "optimize_paper_local_global_k2d"):
            mod.optimize_paper_local_global_k2d = plg.optimize_paper_local_global_k2d

    # grid placement: plateau stop while active (the align stage has no stopping test of its own)
    src = inspect.getsource(ag.aligned_slim)
    a = "        uv, E, J, Esd = trial, Et, Jt, Esd_t\n"
    b = "    uv = np.asarray(uv0, float).copy()\n"
    if src.count(a) == 1 and src.count(b) == 1:
        src = src.replace(a, a + "        if _OS_CONV_ACTIVE():\n"
                          "            _OS_EH.append(float(E))\n"
                          "            if len(_OS_EH) > 20 and (_OS_EH[-21] - _OS_EH[-1]) <= 1e-5 * max(abs(_OS_EH[-1]), 1e-30): break\n")
        src = src.replace(b, b + "    _OS_EH.clear()\n")
        ag._OS_EH = []
        ag._OS_CONV_ACTIVE = active
        exec(compile(src, ag.__file__, "exec"), ag.__dict__)
    src = inspect.getsource(ag.place_grid_aligned)
    reps = (("iters=150, area_weight=area_weight)", "iters=(100000 if _OS_CONV_ACTIVE() else 150), area_weight=area_weight)"),
            ("n_it = 4 * int(features.get(\"iters\", 40)) if soft is None else 0",
             "n_it = (100000 if _OS_CONV_ACTIVE() else 4 * int(features.get(\"iters\", 40))) if soft is None else 0"))
    if all(src.count(x) == 1 for x, _ in reps):
        for x, y in reps:
            src = src.replace(x, y)
        exec(compile(src, ag.__file__, "exec"), ag.__dict__)
    src = inspect.getsource(po._lbfgs)
    a = "        x, f, g = xn, fn, gn\n"
    r0 = "    S, Y = [], []\n    it = 0\n"
    if src.count(a) == 1 and src.count(r0) == 1:
        src = src.replace(r0, r0 + "    _OS_PH = []\n")
        src = src.replace(a, a + "        if _OS_CONV_ACTIVE():\n"
                          "            _OS_PH.append(float(f))\n"
                          "            if len(_OS_PH) > 20 and (_OS_PH[-21] - f) <= 1e-6 * max(abs(f), 1e-30): break\n")
        po._OS_CONV_ACTIVE = active
        exec(compile(src, po.__file__, "exec"), po.__dict__)
    _STATE["installed"] = True
