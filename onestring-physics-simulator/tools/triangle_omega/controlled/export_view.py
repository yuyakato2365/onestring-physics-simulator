"""Standalone, offline view of a completed triangular surface and its cut banks."""
from base64 import b64encode
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def flat_svg(triangles, outline):
    points = np.concatenate((triangles.reshape(-1, 2), outline))
    low, high = points.min(0), points.max(0)
    pad = np.max(high-low) * .025
    def path(row):
        return 'M'+' '.join(f'{x:.8g},{-y:.8g}' for x,y in row)+'Z'
    box = f'{low[0]-pad} {-high[1]-pad} {high[0]-low[0]+2*pad} {high[1]-low[1]+2*pad}'
    lines = ''.join(path(face) for face in triangles)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}">'
           f'<path d="{lines}" fill="#d2e9e4" stroke="#34595d" stroke-width="{pad*.035}"/>'
           f'<path d="{path(outline)}" fill="none" stroke="#ae3865" stroke-width="{pad*.10}"/></svg>')
    return b64encode(svg.encode()).decode()


def portable_view(folder, settings, grid, panels, figure, flat_faces):
    fig = go.Figure()
    seams = []
    for i, name in enumerate(('M3D','M3D（最小角）','入力曲面（seam）')):
        stage = figure(folder, name)
        for trace in stage.data:
            if trace.meta == 'seam':
                if i != 1:
                    trace.update(meta='input' if i==2 else 'panel', visible=i==0)
                    seams.append(trace)
            else:
                trace.update(visible=i==0)
                fig.add_trace(trace)
    fig.add_traces(seams)
    fig.update_layout(showlegend=False,margin=dict(l=0,r=0,t=0,b=0),scene=dict(
        aspectmode='data',dragmode='orbit',uirevision='camera',xaxis_visible=False,
        yaxis_visible=False,zaxis_visible=False,
        camera=dict(eye=dict(x=-.92,y=.29,z=1.44),up=dict(x=0,y=1,z=0))))
    plot=fig.to_html(full_html=False,include_plotlyjs=True,div_id='surface3d',
                     config=dict(responsive=True,scrollZoom=True,displaylogo=False,displayModeBar=False),
                     default_height='100%',default_width='100%')
    title=Path(settings.get('input_label') or 'M3D').stem
    template=Path(__file__).with_name('large_view.html').read_text()
    return (template.replace('@@GRID@@',flat_svg(grid['initial_triangles'],grid['O']))
            .replace('@@M2D@@',flat_svg(grid['Xg'][flat_faces],grid['O']))
            .replace('@@COUNT@@',str(len(panels['T']))).replace('@@MODEL@@',escape(title))
            .replace('@@PLOT@@',plot)).encode()
