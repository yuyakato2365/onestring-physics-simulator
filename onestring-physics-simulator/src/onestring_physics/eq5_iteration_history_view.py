"""Render the raw Eq.(5) K2D solver trajectory on one fixed coordinate scale."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def _bounds(snaps):
    finite=[]
    for s in snaps:
        xy=np.asarray(s['xy'],float);ok=np.all(np.isfinite(xy),axis=1)
        if np.any(ok):finite.append(xy[ok])
    if not finite:return (-1.,1.),(-1.,1.)
    allxy=np.vstack(finite);lo=np.min(allxy,axis=0);hi=np.max(allxy,axis=0);cx,cy=.5*(lo+hi);span=max(float(hi[0]-lo[0]),float(hi[1]-lo[1]),1e-9)*1.08
    return (cx-span/2,cx+span/2),(cy-span/2,cy+span/2)

def _figure(xy,faces,snapshot,xrange,yrange):
    xs=[];ys=[]
    for face in faces:
        ids=np.r_[face,face[0]];p=xy[ids];xs.extend(p[:,0].tolist()+[None]);ys.extend(p[:,1].tolist()+[None])
    fig=go.Figure(go.Scattergl(x=xs,y=ys,mode='lines',line=dict(width=1),hoverinfo='skip'))
    it=snapshot['iteration'];c=snapshot.get('collisions',0);f=snapshot.get('fab_violations',0);step=snapshot.get('step',0.)
    label='Initial raw xy' if snapshot.get('initial') else ('Final raw xy' if snapshot.get('final') else f'iter {it}')
    finite=np.all(np.isfinite(xy),axis=1);extent=np.ptp(xy[finite],axis=0) if np.any(finite) else np.array([np.nan,np.nan])
    fig.update_layout(title=f'{label} · col {c} · fab {f} · step {step:.2e}<br><sup>extent {extent[0]:.3g} × {extent[1]:.3g}</sup>',height=330,margin=dict(l=5,r=5,t=58,b=5),showlegend=False,xaxis=dict(visible=False,range=list(xrange),scaleanchor='y',scaleratio=1),yaxis=dict(visible=False,range=list(yrange),constrain='domain'))
    return fig

def render_completed_k2d(history):
    """Publish before T2D/Eq.6, scoped to this Streamlit session."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx(suppress_warning=True) is None:
            return
        st.session_state["eq5_history"] = history
        st.subheader("K2D result — before T2D / Dual Hinge")
        render_eq5_iteration_history(st, history)
        st.session_state["eq5_rendered_this_run"] = True
    except ImportError:
        return


def render_eq5_iteration_history(st, history=None):
    if history is None:
        history=st.session_state.get("eq5_history")
    if not history or not history.get('snapshots'):return
    faces=np.asarray(history['faces'],int);snaps=history['snapshots'];xrange,yrange=_bounds(snaps)
    st.markdown('### Raw K2D solver trajectory — Initial / every 10 iterations / Final')
    st.caption('FlatTileLayoutへ変換する前のsolver内部 xy を直接描画しています。全パネルで座標範囲を完全に固定しているため、縮小・膨張・飛び出しが起きたiterationをそのまま比較できます。')
    import pandas as pd
    records=pd.DataFrame(history.get('records', []))
    if not records.empty:
        last=records.iloc[-1]
        if last['collisions'] or last['fab_violations']:
            st.warning(f"K2D 制約未充足: collisions={int(last['collisions'])}, fabrication violations={int(last['fab_violations'])}。反復停止を製造可能な収束とは扱いません。")
        st.caption(str(history.get('solver_message','')))
        st.line_chart(records.set_index('iteration')[['EFlat','EEdge','ECollision','EFab']])
        st.dataframe(records, hide_index=True)
        st.download_button('Download K2D diagnostics CSV', records.to_csv(index=False),
                           'k2d_diagnostics.csv', 'text/csv', key='eq5_diagnostics_csv')
    for start in range(0,len(snaps),3):
        cols=st.columns(3,gap='small')
        for j,snapshot in enumerate(snaps[start:start+3]):
            xy=np.asarray(snapshot['xy'],float)
            with cols[j]:st.plotly_chart(_figure(xy,faces,snapshot,xrange,yrange),width='stretch',key=f"eq5_raw_hist_{snapshot['iteration']}_{start+j}")
    st.caption(f'共通表示範囲: x=[{xrange[0]:.4g}, {xrange[1]:.4g}], y=[{yrange[0]:.4g}, {yrange[1]:.4g}]。各タイトルの extent はそのiteration自身の幅×高さです。')

__all__=['render_eq5_iteration_history','render_completed_k2d']
