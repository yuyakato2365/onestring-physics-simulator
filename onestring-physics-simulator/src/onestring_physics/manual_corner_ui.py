"""Pick the four rectangle corners by hand on the original S (Paper mode).

Click four boundary points of S (top view), check them in 3D, undo or reset
mistakes, then confirm: the pipeline is re-run with those boundary vertices
as the official rectangular-BFF corners (``bff_rectangle_manual_corners``).
"""
from __future__ import annotations

import numpy as np

PICKS = "manual_corner_picks"
CONFIRMED = "bff_manual_corners"
RERUN_FLAG = "onestring_split_direction_rerun"
COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")


def render_manual_corner_ui(st, state, surface=None, request=False):
    import plotly.graph_objects as go
    from .paper_split_reflatten import _boundary_loop

    if surface is not None:
        V, F = np.asarray(surface[0], float), np.asarray(surface[1], int)[:, :3]
    else:
        V = np.asarray(state.target_surface.vertices, float)
        F = np.asarray(state.target_surface.faces, int)[:, :3]
    loop = _boundary_loop(F)
    B = V[loop]
    picks = [int(b) for b in st.session_state.get(PICKS, []) if 0 <= int(b) < len(loop)]
    confirmed = st.session_state.get(CONFIRMED)
    st.subheader("Ω を作る前に、矩形の隅を4点選んでください（元の S の境界）" if request else "矩形Ωの隅を手動で選ぶ（元の S の境界）")
    st.caption(("現在の計算：手動で確定した4点を隅に使用。" if confirmed else "現在の計算：自動で選んだ隅。")
               + " 真上から見た図で境界の点を4つクリックし、3D で確認して「確定」を押すと、その4点を BFF の隅にして計算し直します。"
               "Paperモードの計算でだけ使われます。選ぶ順番は自由です（境界に沿って自動で並べ替えます）。"
               "選んだ点には、各軸に平行な補助線（点線）を引きます。")
    top = go.Figure()
    top.add_trace(go.Scatter(x=V[:, 0], y=V[:, 1], mode="markers", marker=dict(size=2, color="rgba(120,140,170,0.35)"),
                             hoverinfo="skip", name="S"))
    top.add_trace(go.Scatter(x=B[:, 0], y=B[:, 1], mode="markers+lines", customdata=np.arange(len(loop)),
                             marker=dict(size=8, color="#1b2a31"), line=dict(color="#1b2a31", width=1),
                             hovertemplate="境界点 %{customdata}（クリックで選択）<extra></extra>", name="境界（クリックで選択）"))
    lo, hi = V.min(axis=0), V.max(axis=0)
    pad = 0.06 * float(np.max(hi - lo))
    for k, b in enumerate(picks):
        # guide lines through the picked point, parallel to the x and y axes
        # (layout shapes: not dimmed by the chart's point selection)
        top.add_shape(type="line", x0=lo[0] - pad, x1=hi[0] + pad, y0=B[b, 1], y1=B[b, 1], layer="above",
                      line=dict(color=COLORS[k], width=2, dash="dash"))
        top.add_shape(type="line", x0=B[b, 0], x1=B[b, 0], y0=lo[1] - pad, y1=hi[1] + pad, layer="above",
                      line=dict(color=COLORS[k], width=2, dash="dash"))
        top.add_trace(go.Scatter(x=[B[b, 0]], y=[B[b, 1]], mode="markers+text", text=[f"C{k + 1}"], textposition="top center",
                                 textfont=dict(size=15, color=COLORS[k]), marker=dict(size=17, color=COLORS[k]),
                                 hoverinfo="skip", name=f"C{k + 1}"))
    top.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1), height=520,
                      clickmode="event+select", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02),
                      paper_bgcolor="white", plot_bgcolor="white")
    event = st.plotly_chart(top, use_container_width=True, key=f"manual_corner_top_{len(picks)}",
                            on_select="rerun", selection_mode="points")
    try:
        points = event.selection.points if event is not None else []
    except AttributeError:
        points = []
    for point in points or []:
        if not isinstance(point, dict) or point.get("curve_number", 1) != 1:
            continue
        b = point.get("customdata", point.get("point_index"))
        b = int(b[0] if isinstance(b, (list, tuple)) else b)
        if b not in picks and len(picks) < 4:
            st.session_state[PICKS] = picks + [b]
            st.rerun()
    with st.expander("クリックで選べないときは番号で追加", expanded=False):
        number = st.number_input("境界点の番号（図にマウスを重ねると表示）", 0, len(loop) - 1, 0, 1, key="manual_corner_number")
        if st.button("この番号の点を追加", key="manual_corner_add", disabled=len(picks) >= 4 or int(number) in picks):
            st.session_state[PICKS] = picks + [int(number)]
            st.rerun()
    mesh = go.Figure(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                               color="#9db7d9", opacity=0.9, flatshading=True, hoverinfo="skip", name="S"))
    L = B[list(range(len(loop))) + [0]]
    mesh.add_trace(go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="lines", line=dict(color="#1b2a31", width=4), name="境界"))
    for k, b in enumerate(picks):
        # guide lines through the picked point, parallel to the x, y and z axes
        for axis in range(3):
            ends = np.repeat(B[b][None, :], 2, axis=0)
            ends[:, axis] = [lo[axis] - pad, hi[axis] + pad]
            mesh.add_trace(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="lines",
                                        line=dict(color=COLORS[k], width=3, dash="dash"), hoverinfo="skip", showlegend=False))
        mesh.add_trace(go.Scatter3d(x=[B[b, 0]], y=[B[b, 1]], z=[B[b, 2]], mode="markers+text", text=[f"C{k + 1}"],
                                    textfont=dict(size=16, color=COLORS[k]), marker=dict(size=9, color=COLORS[k]), name=f"C{k + 1}"))
    mesh.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"))
    st.plotly_chart(mesh, use_container_width=True, key="manual_corner_3d")
    st.write(f"選択済み：{len(picks)} / 4 点" + ("（" + "、".join(f"C{k + 1}=境界点{b}" for k, b in enumerate(picks)) + "）" if picks else ""))
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("最後の1点を取り消す", key="manual_corner_undo", disabled=not picks):
        st.session_state[PICKS] = picks[:-1]
        st.rerun()
    if c2.button("すべてやり直す", key="manual_corner_reset", disabled=not picks):
        st.session_state[PICKS] = []
        st.rerun()
    if c3.button("この4点で確定（再計算）", key="manual_corner_confirm", type="primary", disabled=len(picks) != 4):
        st.session_state[CONFIRMED] = [B[b].tolist() for b in picks]
        st.session_state.pop("corner_selection_request", None)
        st.session_state[RERUN_FLAG] = True
        st.rerun()
    if c4.button("手動の隅を解除（自動に戻す）", key="manual_corner_clear", disabled=not confirmed):
        st.session_state.pop(CONFIRMED, None)
        st.session_state[PICKS] = []
        st.session_state[RERUN_FLAG] = True
        st.rerun()
