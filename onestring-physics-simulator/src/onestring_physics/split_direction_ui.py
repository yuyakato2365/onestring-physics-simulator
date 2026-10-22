"""Pick the direction of each re-flatten Split on the M2D candidate lines.

Every recorded cut has two candidate grid lines through the same curvature
peak: vertical ("col", constant u in M2D) and horizontal ("row", constant v).
The figure draws both for every cut; clicking a line handle (or the radio
below) chooses it. Choices are stored per cut in planning order as
``paper_split_direction_choices`` ("auto" | "col" | "row") and the pipeline is
re-run, because a different cut changes every later cut and stage.
"""
from __future__ import annotations

import numpy as np

SESSION_KEY = "paper_split_direction_choices"
RERUN_FLAG = "onestring_split_direction_rerun"
LABEL = {"auto": "自動", "col": "縦", "row": "横"}
COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf")


def _canonical(mesh):
    xy = np.asarray(getattr(mesh, "_split_panel_source_vertices", mesh.vertices), float)[:, :2]
    return xy, np.asarray(mesh.faces, int)


def figure_split_candidates(mesh, steps):
    import plotly.graph_objects as go

    xy, faces = _canonical(mesh)
    fig = go.Figure()
    xs, ys = [], []
    for f in faces:
        q = xy[list(f) + [f[0]]]
        xs += list(q[:, 0]) + [None]; ys += list(q[:, 1]) + [None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(90,100,115,0.35)", width=1),
                             hoverinfo="skip", name="M2D"))
    for k, step in enumerate(steps):
        ids = np.asarray(step.get("parent_face_ids", []), int)
        if not len(ids):
            continue
        lo, hi = xy[faces[ids]].reshape(-1, 2).min(0), xy[faces[ids]].reshape(-1, 2).max(0)
        color = COLORS[k % len(COLORS)]
        for cand in step.get("candidates", []):
            axis, v = cand["axis"], float(cand["value"])
            chosen = axis == step.get("axis") and abs(v - float(step.get("value", np.nan))) < 1e-9
            if axis == "col":
                lx, ly, mid = [v, v], [lo[1], hi[1]], (v, 0.5 * (lo[1] + hi[1]))
            else:
                lx, ly, mid = [lo[0], hi[0]], [v, v], (0.5 * (lo[0] + hi[0]), v)
            sig = " / ".join(f"{s:.3f}" for s in cand.get("child_sigma", []))
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color=color, width=5 if chosen else 2, dash="solid" if chosen else "dash")))
            fig.add_trace(go.Scatter(
                x=[mid[0]], y=[mid[1]], mode="markers+text", customdata=[[k, axis]],
                text=[f"S{k + 1} {LABEL[axis]}"], textposition="top center", textfont=dict(color=color, size=13),
                marker=dict(size=16 if chosen else 13, color=color if chosen else "white",
                            line=dict(color=color, width=2), symbol="circle"),
                hovertemplate=f"Split {k + 1}・{LABEL[axis]}：子のσ {sig}" + ("（採用中）" if chosen else "（クリックで選択）") + "<extra></extra>",
                name=f"Split {k + 1} {LABEL[axis]}", showlegend=False))
    fig.update_layout(title=dict(text="M2D上の候補線（実線＝採用中、点線＝もう一方）。丸をクリックすると向きを選べます", x=0.01),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
                      height=620, margin=dict(l=0, r=0, t=50, b=0), clickmode="event+select",
                      paper_bgcolor="white", plot_bgcolor="white")
    return fig


def render_split_direction_ui(st, state):
    report = (getattr(state.mesh_2d_initial, "metrics", {}) or {}).get("reflatten_split_report") or {}
    steps = [s for s in report.get("steps", []) if s.get("candidates")]
    st.subheader("Splitの向きを選ぶ（M2Dの候補線）")
    if not steps:
        st.caption("この結果には Split の候補線の記録がありません（Split なし、または再展開判定以外）。")
        return
    current = list(st.session_state.get(SESSION_KEY, ()))
    current += ["auto"] * (len(steps) - len(current))
    event = st.plotly_chart(figure_split_candidates(state.mesh_2d_initial, steps), use_container_width=True,
                            key="split_direction_figure", on_select="rerun", selection_mode="points")
    try:
        points = event.selection.points if event is not None else []
    except AttributeError:
        points = (event or {}).get("selection", {}).get("points", []) if isinstance(event, dict) else []
    for point in points or []:
        data = point.get("customdata") if isinstance(point, dict) else None
        if data and len(data) == 2:
            st.session_state[f"split_direction_choice_{int(data[0])}"] = LABEL[str(data[1])]
    inverse = {v: k for k, v in LABEL.items()}
    pending = []
    for k, step in enumerate(steps):
        rows = {c["axis"]: c for c in step["candidates"]}
        detail = "、".join(f"{LABEL[a]}：子のσ {' / '.join(f'{s:.3f}' for s in rows[a]['child_sigma'])}" for a in ("col", "row") if a in rows)
        key = f"split_direction_choice_{k}"
        if key not in st.session_state:
            st.session_state[key] = LABEL[current[k] if current[k] in LABEL else "auto"]
        value = st.radio(f"Split {k + 1}（今回：{LABEL[step['axis']]}、{'指定' if step.get('chosen_by') == 'user' else '自動'}）　{detail}",
                         [LABEL["auto"], LABEL["col"], LABEL["row"]], key=key, horizontal=True)
        pending.append(inverse[value])
    st.caption("向きは Split の順番ごとに効きます。前の Split を変えると、後の Split の位置や候補も変わるので、"
               "上から順に1本ずつ決めて再計算するのが確実です。「縦」「横」は M2D（Ω）の座標での向きです。")
    c1, c2 = st.columns(2)
    if c1.button("この向きで Split をやり直す（パイプライン再計算）", type="primary", key="split_direction_apply"):
        while pending and pending[-1] == "auto":
            pending.pop()
        st.session_state[SESSION_KEY] = tuple(pending)
        st.session_state[RERUN_FLAG] = True
        st.rerun()
    if c2.button("すべて自動に戻す", key="split_direction_reset"):
        st.session_state[SESSION_KEY] = ()
        for k in range(len(steps)):
            st.session_state.pop(f"split_direction_choice_{k}", None)
        st.session_state[RERUN_FLAG] = True
        st.rerun()
