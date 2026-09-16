"""Fast K2D experiment UI: replay saved checkpoints without rerunning OptCuts/K3D."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

ROOT=Path(__file__).resolve().parent; SRC=ROOT/"src"
if str(SRC) not in sys.path: sys.path.insert(0,str(SRC))
from onestring_physics.optcuts_k2d_global_history import list_checkpoints,load_checkpoint,global_hinge_solve,save_k2d_checkpoint

st.set_page_config(page_title="OneString K2D History Lab",layout="wide")
st.title("K2D History Lab — global all-hinge solver")
st.caption("Saved collision-free rigid K2D checkpoints can be replayed here. OptCuts, M2D and K3D are not rerun.")
files=list_checkpoints()
if not files:
    st.warning("No K2D history yet. Run app_optcuts.py once in optcuts_test/test2; a checkpoint will be saved automatically.")
    st.stop()
labels=[f.name for f in files]
idx=st.selectbox("K2D history",range(len(files)),format_func=lambda i: labels[i])
tiles,constraints,meta=load_checkpoint(files[idx])
st.json(meta)
max_nfev=st.number_input("Global solver max_nfev",min_value=20,max_value=5000,value=300,step=50)


def draw(x,title):
    fig,ax=plt.subplots(figsize=(10,8))
    for poly in np.asarray(x):
        q=np.vstack([poly,poly[0]])
        ax.plot(q[:,0],q[:,1],linewidth=.55)
    ax.set_aspect("equal",adjustable="box"); ax.set_title(title); ax.grid(True,alpha=.2)
    st.pyplot(fig); plt.close(fig)

c1,c2=st.columns(2)
with c1: draw(tiles,"Saved K2D checkpoint")
if st.button("Run NEW global all-hinge feasibility",type="primary"):
    with st.spinner("Solving every hinge simultaneously over all tile SE(2) poses..."):
        solved,metrics=global_hinge_solve(tiles,constraints,max_nfev=int(max_nfev))
    st.session_state["global_k2d_result"]=(solved,metrics)
if "global_k2d_result" in st.session_state:
    solved,metrics=st.session_state["global_k2d_result"]
    with c2: draw(solved,"NEW: global all-hinge result")
    st.subheader("Feasibility diagnostics")
    st.json(metrics)
    if metrics.get("all_hinges_satisfied"):
        st.success("All hinge coincidences satisfy the practical tolerance. Next step: add hard non-overlap while retaining these equalities.")
    else:
        st.error("All hinges could not be closed within tolerance. This isolates a hinge/linkage feasibility problem before collision handling.")
    if st.button("Save this global result to history"):
        p=save_k2d_checkpoint(solved,constraints,metrics,label="global_hinge")
        st.success(f"Saved: {p}")

st.divider()
st.markdown("**Interpretation:** this new mode intentionally solves hinge feasibility first with collision OFF. That is the diagnostic Phase 1. It avoids hiding whether failure comes from loop closure or collision. Once hinge feasibility is confirmed, Phase 2 can enforce non-overlap as an inequality while keeping every hinge equality active.")
