"""Render Eq.(5) K2D snapshots every 10 solver iterations."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def _figure(xy,faces,snapshot):
    xs=[];ys=[]
    for face in faces:
        ids=np.r_[face,face[0]];p=xy[ids];xs.extend(p[:,0].tolist()+[None]);ys.extend(p[:,1].tolist()+[None])
    fig=go.Figure(go.Scattergl(x=xs,y=ys,mode='lines',line=dict(width=1),hoverinfo='skip'))
    it=snapshot['iteration'];c=snapshot.get('collisions',0);f=snapshot.get('fab_violations',0);step=snapshot.get('step',0.)
    fig.update_layout(title=f'iter {it} · collision {c} · fab {f} · step {step:.2e}',height=330,margin=dict(l=5,r=5,t=42,b=5),showlegend=False,xaxis=dict(visible=False,scaleanchor='y',scaleratio=1),yaxis=dict(visible=False,constrain='domain'))
    return fig


def render_eq5_iteration_history(st):
    try:
        from .paper_eq5_k2d_solver import get_last_eq5_history
        history=get_last_eq5_history()
    except Exception:
        return
    if not history or not history.get('snapshots'):return
    faces=np.asarray(history['faces'],int);snaps=history['snapshots']
    st.markdown('### K2D optimization history — every 10 iterations')
    st.caption('同じ表示範囲・同じtopologyで、10 iterationごとのK2Dを左上から時系列に3列表示します。崩れ始めるiterationを特定するためのdiagnosticです。')
    for start in range(0,len(snaps),3):
        cols=st.columns(3,gap='small')
        for j,snapshot in enumerate(snaps[start:start+3]):
            with cols[j]:
                st.plotly_chart(_figure(np.asarray(snapshot['xy'],float),faces,snapshot),width='stretch',key=f"eq5_hist_{snapshot['iteration']}_{start+j}")
    st.caption('各タイトル: iteration / collision pair数 / EFab violation数 / 直前iterationからの最大vertex移動量(step)。')

__all__=['render_eq5_iteration_history']
