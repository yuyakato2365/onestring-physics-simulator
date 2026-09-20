"""Streamlit renderer for the authoritative M3D -> K3D optimization trajectory.

The history rendered here is recorded inside the K3D solver that actually
produces the returned mesh (``optimize_paper_local_global_k3d``).  Nothing in
this module runs an optimizer: if a run has no recorded history, it says so
instead of synthesising one.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


ENERGY_SERIES = [
    ("EAssembled", "EAssembled", "#1f4e79"),
    ("w1EPlanar", "ω₁·EPlanar", "#2f8f6f"),
    ("w2ESquare", "ω₂·ESquare", "#b06a2c"),
    ("w3ESurface", "ω₃·ESurface", "#7a4a9e"),
]

DEVIATION_SERIES = [
    ("planarity_max", "max", "#b8422a"),
    ("planarity_rms", "RMS", "#1f4e79"),
    ("planarity_mean", "mean", "#2f8f6f"),
]


def get_k3d_history(state):
    """Return the recorded history dict for this state, or None."""
    mesh = getattr(state, "mesh_3d_optimized", None)
    if mesh is None:
        return None
    history = (getattr(mesh, "metrics", {}) or {}).get("k3d_iteration_history")
    if not history or not history.get("records"):
        return None
    return history


def _series_figure(records, series, *, title, y_title, log_y):
    fig = go.Figure()
    iterations = [int(r["iteration"]) for r in records]
    for key, label, color in series:
        if key not in records[0]:
            continue
        values = [float(r[key]) for r in records]
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=f"{label}<br>iteration %{{x}}<br>%{{y:.6g}}<extra></extra>",
            )
        )
    positive = [
        float(r[k]) for r in records for k, _, _ in series if k in r and float(r[k]) > 0.0
    ]
    use_log = bool(log_y and positive and max(positive) / max(min(positive), 1e-300) > 50)
    y_axis = dict(title=y_title, type="log" if use_log else "linear")
    if use_log:
        # A term that starts at (numerically) zero would otherwise stretch the
        # axis over 30+ decades and flatten every other curve.  Keep at most
        # eight decades below the largest value; smaller points sit on the floor.
        high = max(positive)
        low = max(min(positive), high / 1e8)
        y_axis["range"] = [float(np.log10(low / 3.0)), float(np.log10(high * 3.0))]
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=0, r=0, t=52, b=0),
        xaxis=dict(title="iteration", dtick=max(1, len(records) // 12)),
        yaxis=y_axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def render_k3d_iteration_history(st, state) -> bool:
    """Render the K3D objective and planarity trajectories.  Returns True if drawn."""
    history = get_k3d_history(state)
    if history is None:
        mesh = getattr(state, "mesh_3d_optimized", None)
        solver = str((getattr(mesh, "metrics", {}) or {}).get("k3d_solver_model", "unknown"))
        st.info(
            "K3D optimization trajectory is not available for this run.\n\n"
            f"The K3D solver used here was `{solver}`. Per-iteration history is recorded "
            "inside the paper-aligned local/global K3D solver — select the Omega "
            "parameterization mode "
            "`2026-09-18 | OptCuts Ω + paper-aligned local/global K3D/K2D` and re-run. "
            "No history is synthesised from a second optimization."
        )
        return False

    records = history["records"]
    weights = history.get("weights", {}) or {}
    mesh = state.mesh_3d_optimized
    span = np.ptp(np.asarray(mesh.vertices, dtype=float), axis=0)

    st.markdown("### K3D optimization trajectory — EAssembled")
    st.caption(
        "M3D → K3D の EAssembled(v) = ω₁E_Planar + ω₂E_Square + ω₃E_Surface。"
        f"重みは ω₁={weights.get('w1_planar', float('nan')):.4g} / "
        f"ω₂={weights.get('w2_square', float('nan')):.4g} / "
        f"ω₃={weights.get('w3_surface', float('nan')):.4g}。"
        f"iteration 0 は最適化前（M3D）の値です。checkpoint 数 = {len(records)}。"
    )
    st.plotly_chart(
        _series_figure(
            records,
            ENERGY_SERIES,
            title="EAssembled and its weighted terms",
            y_title="energy [mesh units²]",
            log_y=True,
        ),
        use_container_width=True,
        key="k3d_history_energy",
    )

    st.markdown("### K3D planarity deviation")
    st.caption(
        "エネルギーではなく幾何学的な距離です。各 quad の4頂点について、その quad の "
        "best-fit 平面までの距離を測り、全 quad 頂点にわたる max / RMS / mean を取っています"
        "（論文 Sec. 6 の planarity error と同じ定義）。"
        f"単位は mesh 座標単位（K3D の bbox = {span[0]:.4g} × {span[1]:.4g} × {span[2]:.4g}）。"
        "target を mm 単位の OBJ/STL から読み込んだ場合はそのまま mm です。"
    )
    st.plotly_chart(
        _series_figure(
            records,
            DEVIATION_SERIES,
            title="Vertex distance to each quad's best-fit plane",
            y_title="planar deviation [mesh units]",
            log_y=True,
        ),
        use_container_width=True,
        key="k3d_history_planarity",
    )

    final = records[-1]
    columns = st.columns(4)
    columns[0].metric("EAssembled (final)", f"{final['EAssembled']:.4g}")
    columns[1].metric("max deviation", f"{final['planarity_max']:.4g}")
    columns[2].metric("RMS deviation", f"{final['planarity_rms']:.4g}")
    columns[3].metric("mean deviation", f"{final['planarity_mean']:.4g}")

    with st.expander("K3D trajectory table / CSV", expanded=False):
        st.caption(f"solver: `{history.get('solver')}` — {history.get('backend')}")
        try:
            import pandas as pd

            frame = pd.DataFrame(records)
            st.dataframe(frame, hide_index=True)
            st.download_button(
                "Download K3D trajectory CSV",
                frame.to_csv(index=False),
                "k3d_optimization_trajectory.csv",
                "text/csv",
                key="k3d_history_csv",
            )
        except Exception as exc:  # pandas is optional for the graphs
            st.write(records)
            st.caption(f"table view unavailable: {exc}")
    return True


__all__ = ["render_k3d_iteration_history", "get_k3d_history"]
