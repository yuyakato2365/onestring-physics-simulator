"""Streamlit renderer for M3D -> K3D objective history."""
from __future__ import annotations


def render_k3d_iteration_history(st,state):
    mesh=getattr(state,"mesh_3d_optimized",None)
    if mesh is None:
        return False
    history=getattr(mesh,"metrics",{}).get("k3d_iteration_history")
    if not history or not history.get("records"):
        note=getattr(mesh,"metrics",{}).get("k3d_iteration_history_note")
        if note:
            st.caption(note)
        return False

    import pandas as pd
    records=pd.DataFrame(history["records"])
    if records.empty:
        return False

    st.markdown("### K3D optimization trajectory — EAssembled")
    st.caption(
        "M3D → K3D の ω₁EPlanar + ω₂ESquare + ω₃ESurface の変遷です。"
        "各項は重み適用後の値で、EAssembled は3項の和です。"
    )
    energy_cols=[c for c in ["EAssembled","w1EPlanar","w2ESquare","w3ESurface"] if c in records]
    st.line_chart(records.set_index("evaluation")[energy_cols])

    st.markdown("#### Quad planarity deviation")
    st.caption(
        "各quadについて、最初の3頂点が張る平面から4頂点目までの距離を測定。"
        "max / RMS / mean の変遷を、エネルギーとは別の幾何学的な距離として表示します。"
    )
    planar_cols=[c for c in ["planarity_max","planarity_rms","planarity_mean"] if c in records]
    st.line_chart(records.set_index("evaluation")[planar_cols])
    last=records.iloc[-1]
    st.write({
        "planarity_max_final":float(last["planarity_max"]),
        "planarity_rms_final":float(last["planarity_rms"]),
        "planarity_mean_final":float(last["planarity_mean"]),
        "history_backend":history.get("backend"),
        "solver_message":history.get("message"),
    })
    st.dataframe(records,hide_index=True)
    st.download_button(
        "Download K3D diagnostics CSV",
        records.to_csv(index=False),
        "k3d_diagnostics.csv",
        "text/csv",
        key="k3d_diagnostics_csv",
    )
    return True


__all__=["render_k3d_iteration_history"]
