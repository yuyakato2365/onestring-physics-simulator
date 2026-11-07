"""2026-11-05: each stage's result shown under the running-process figure as soon as that stage finishes.

``live_stage_results(container)`` wraps the stage functions of ``legacy_pipeline_base`` for one run (outermost, so
every patched route still runs unchanged) and appends one figure per finished stage to ``container``; nothing is
removed while the run goes on, so earlier (provisional) results stay visible above the newer ones.
Stages: S → Ω (OptCuts chart), M2D, M3D, K3D, T3D, K2D, T2D top hinge, T2D dual hinge.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np

_BOX = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                 [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])


def _wire2d(go, V, F, color):
    """polygon outlines (triangles or quads) as one line trace"""
    V = np.asarray(V, float)[:, :2]
    F = np.asarray(F, int)
    xs, ys = [], []
    for f in F:
        p = V[list(f) + [f[0]]]
        xs += p[:, 0].tolist() + [None]
        ys += p[:, 1].tolist() + [None]
    return go.Scattergl(x=xs, y=ys, mode="lines", line=dict(width=1, color=color), hoverinfo="skip", showlegend=False)


def _mesh3d(go, V, F, color):
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    if F.shape[1] == 4:
        F = np.vstack([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])
    return go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], color=color,
                     flatshading=True, lighting=dict(ambient=0.55, diffuse=0.7), hoverinfo="skip")


def _tiles3d(go, T, color):
    T = np.asarray(T, float)
    n = len(T)
    V = T[:, :8].reshape(-1, 3)
    F = (_BOX[None] + 8 * np.arange(n)[:, None, None]).reshape(-1, 3)
    return go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2], color=color,
                     flatshading=True, lighting=dict(ambient=0.55, diffuse=0.7), hoverinfo="skip")


def _figure(kind, obj):
    import plotly.graph_objects as go
    fig = go.Figure()
    if kind == "omega":
        fig.add_trace(_wire2d(go, obj.uv_vertices_2d, obj.uv_faces, "#3b7f8c"))
        b = np.asarray(getattr(obj, "omega_boundary", np.zeros((0, 2))), float)
        if len(b):
            b = np.vstack([b, b[:1]])
            fig.add_trace(go.Scattergl(x=b[:, 0], y=b[:, 1], mode="lines", line=dict(width=2.5, color="#c0392b"),
                                       hoverinfo="skip", showlegend=False))
    elif kind == "mesh2d":
        fig.add_trace(_wire2d(go, obj.vertices, obj.faces, "#2f5d8a"))
    elif kind == "mesh3d":
        fig.add_trace(_mesh3d(go, obj.vertices, obj.faces, "#8fb3d9"))
    elif kind == "tiles3d":
        fig.add_trace(_tiles3d(go, obj.vertices, "#8fb3d9"))
    elif kind == "tiles2d":
        T = np.asarray(obj.vertices, float)
        F = np.arange(len(T) * 4).reshape(-1, 4)
        fig.add_trace(_wire2d(go, T[:, :4, :2].reshape(-1, 2), F, "#2f5d8a"))
    if kind in ("omega", "mesh2d", "tiles2d"):
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    else:
        fig.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                                     zaxis=dict(visible=False)))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _summary(kind, obj):
    if kind == "omega":
        m = getattr(obj, "metrics", {}) or {}
        return f"{len(obj.uv_vertices_2d)} UV 頂点・{len(obj.uv_faces)} 三角形" + (
            f"・方法 {m.get('parameterization_method')}" if m.get("parameterization_method") else "")
    if kind in ("mesh2d", "mesh3d"):
        return f"{len(obj.faces)} パネル"
    return f"{len(obj.vertices)} タイル"


@contextlib.contextmanager
def live_stage_results(container, started=None):
    """Within the block, every finished stage of the OneString pipeline is drawn into ``container``."""
    import streamlit as st
    from . import legacy_pipeline_base as lb
    t0 = started if started is not None else time.perf_counter()
    count = {"n": 0, "extrude": 0}
    plan = [  # (function name, title, figure kind, which item of the return value)
        ("_build_surface_parameterization", "S → Ω：OptCuts の平面図", "omega", None),
        ("_build_m2d", "Ω → M2D：平面図の上の四角形メッシュ", "mesh2d", None),
        ("_lift_m2d_to_m3d", "M2D → M3D：立体に持ち上げたメッシュ", "mesh3d", 0),
        ("_optimize_k3d", "M3D → K3D：3D 最適化", "mesh3d", 0),
        ("_extrude_tiles", "K3D → T3D：押し出したタイル", "tiles3d", 0),
        ("_optimize_k2d", "M2D → K2D：2D 最適化", "mesh2d", 0),
        ("_make_t2d_from_transforms", "K2D → T2D：Top Hinge", "tiles2d", 0),
        ("_optimize_dual_hinges", "T2D：Dual Hinge（最終）", "tiles2d", 0),
    ]
    originals = {}

    def show(title, kind, obj):
        count["n"] += 1
        try:
            with container:
                st.markdown(f"**{count['n']}. {title}**　<span style='color:gray'>{time.perf_counter() - t0:6.0f} 秒 ・ "
                            f"{_summary(kind, obj)}</span>", unsafe_allow_html=True)
                st.plotly_chart(_figure(kind, obj), use_container_width=True, key=f"live_stage_{count['n']}_{time.time()}")
        except Exception as e:   # noqa: BLE001  (a figure must never stop the pipeline)
            print(f"[LIVE-STAGE] could not draw {title}: {type(e).__name__}: {e}", flush=True)

    # the pipeline runs a private copy of legacy_pipeline_base (onestring_pipeline._original, loaded from the same
    # file): wrap the stage functions in every module object made from that file
    import sys
    targets, seen = [], set()
    cands = [lb] + [getattr(m, "_original", None) for m in list(sys.modules.values())
                    if getattr(m, "__name__", "").startswith("onestring_physics")]
    for m in cands:
        if m is not None and str(getattr(m, "__file__", "")).endswith("legacy_pipeline_base.py") and id(m) not in seen:
            seen.add(id(m)); targets.append(m)
    depth = {}

    def wrap(mod, name, title, kind, item):
        f = getattr(mod, name, None)
        if f is None:
            return
        originals[(id(mod), name)] = (mod, name, f)

        def wrapped(*a, **k):
            depth[name] = depth.get(name, 0) + 1
            try:
                out = f(*a, **k)
            finally:
                depth[name] -= 1
            if depth[name] > 0:               # an inner copy of the same stage: shown by the outer call
                return out
            if name == "_extrude_tiles":
                count["extrude"] += 1
                if count["extrude"] > 1:      # only the first extrusion is T3D
                    return out
            obj = out[item] if (item is not None and isinstance(out, tuple)) else out
            show(title, kind, obj)
            return out
        wrapped.__wrapped__ = f
        setattr(mod, name, wrapped)

    for mod in targets:
        for name, title, kind, item in plan:
            wrap(mod, name, title, kind, item)
    print(f"[LIVE-STAGE] stage results shown for {len(targets)} pipeline module(s)", flush=True)
    try:
        yield
    finally:
        for mod, name, f in originals.values():
            setattr(mod, name, f)



# ---- 2026-11-05 (later): background-thread version -----------------------------------------------------------
@contextlib.contextmanager
def capture_stage_results(sink: list, started=None):
    """Like live_stage_results, but only records each finished stage in ``sink`` (no Streamlit call, so it can run in
    the background thread); the page draws them with render_stage_result."""
    import sys
    from . import legacy_pipeline_base as lb
    t0 = started if started is not None else time.perf_counter()
    plan = [
        ("_build_surface_parameterization", "S → Ω：OptCuts の平面図", "omega", None),
        ("_build_m2d", "Ω → M2D：平面図の上の四角形メッシュ", "mesh2d", None),
        ("_lift_m2d_to_m3d", "M2D → M3D：立体に持ち上げたメッシュ", "mesh3d", 0),
        ("_optimize_k3d", "M3D → K3D：3D 最適化", "mesh3d", 0),
        ("_extrude_tiles", "K3D → T3D：押し出したタイル", "tiles3d", 0),
        ("_optimize_k2d", "M2D → K2D：2D 最適化", "mesh2d", 0),
        ("_make_t2d_from_transforms", "K2D → T2D：Top Hinge", "tiles2d", 0),
        ("_optimize_dual_hinges", "T2D：Dual Hinge（最終）", "tiles2d", 0),
    ]
    targets, seen = [], set()
    cands = [lb] + [getattr(m, "_original", None) for m in list(sys.modules.values())
                    if getattr(m, "__name__", "").startswith("onestring_physics")]
    for m in cands:
        if m is not None and str(getattr(m, "__file__", "")).endswith("legacy_pipeline_base.py") and id(m) not in seen:
            seen.add(id(m)); targets.append(m)
    originals, depth, count = {}, {}, {"extrude": 0}

    def wrap(mod, name, title, kind, item):
        f = getattr(mod, name, None)
        if f is None:
            return
        originals[(id(mod), name)] = (mod, name, f)

        def wrapped(*a, **k):
            depth[name] = depth.get(name, 0) + 1
            try:
                out = f(*a, **k)
            finally:
                depth[name] -= 1
            if depth[name] > 0:
                return out
            if name == "_extrude_tiles":
                count["extrude"] += 1
                if count["extrude"] > 1:
                    return out
            obj = out[item] if (item is not None and isinstance(out, tuple)) else out
            sink.append(dict(title=title, kind=kind, obj=obj, seconds=time.perf_counter() - t0))
            return out
        wrapped.__wrapped__ = f
        setattr(mod, name, wrapped)

    for mod in targets:
        for name, title, kind, item in plan:
            wrap(mod, name, title, kind, item)
    try:
        yield
    finally:
        for mod, name, f in originals.values():
            setattr(mod, name, f)


def render_stage_result(container, index: int, item: dict) -> None:
    import streamlit as st
    try:
        with container:
            st.markdown(f"**{index + 1}. {item['title']}**　<span style='color:gray'>{item['seconds']:6.0f} 秒 ・ "
                        f"{_summary(item['kind'], item['obj'])}</span>", unsafe_allow_html=True)
            st.plotly_chart(_figure(item["kind"], item["obj"]), use_container_width=True,
                            key=f"live_stage_{index}_{id(item)}")
    except Exception as e:   # noqa: BLE001
        if type(e).__name__ in ("StopException", "RerunException"):
            raise
        print(f"[LIVE-STAGE] could not draw {item.get('title')}: {type(e).__name__}: {e}", flush=True)
