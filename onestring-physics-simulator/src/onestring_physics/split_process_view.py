"""Replay recorded Split decisions without rerunning or changing the solver."""
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative


def figure_split_process(mesh, step):
    initial = mesh.metrics.get('csf_split_initial_components')
    records = mesh.metrics.get('csf_split_step_analysis', [])
    if initial is None or any('split_boundary_pairs' not in r for r in records):
        return None, []
    step = max(0, min(int(step), len(records)))
    groups = [(list(c['face_ids']), float(c['sigma'])) for c in initial]
    for record in records[:step]:
        parent = set(record['face_ids'])
        groups = [(ids, sigma) for ids, sigma in groups if set(ids) != parent]
        groups.extend((list(ids), float(sigma)) for ids, sigma in zip(record['child_face_ids'], record['sigma_after']))
    groups.sort(key=lambda c: min(c[0]))
    canonical = np.asarray(getattr(mesh, '_split_panel_source_vertices', mesh.vertices))
    faces = np.asarray(mesh.faces, int)
    fig = go.Figure()
    rows = []
    for index, (ids, sigma) in enumerate(groups):
        xs, ys = [], []
        for face in faces[ids]:
            polygon = canonical[np.r_[face, face[0]], :2]
            xs.extend([*polygon[:, 0], None]); ys.extend([*polygon[:, 1], None])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', fill='toself',
                                fillcolor=qualitative.Pastel[index % len(qualitative.Pastel)],
                                line=dict(color='#64748b', width=.7),
                                name=f'領域 {index + 1} · CSF {sigma:.4g}',
                                hovertemplate=f'領域 {index + 1}<br>{len(ids)} tiles<br>CSF {sigma:.4g}<extra></extra>'))
        rows.append({'領域': index + 1, 'タイル数': len(ids), 'CSF': sigma})
    for newest in (False, True):
        xs, ys = [], []
        selected = records[max(0, step-1):step] if newest else records[:max(0, step-1)]
        for record in selected:
            for a, edge, _, _ in record['split_boundary_pairs']:
                segment = canonical[faces[a, [edge, (edge+1) % 4]], :2]
                xs.extend([*segment[:, 0], None]); ys.extend([*segment[:, 1], None])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                line=dict(color='#ef4444' if newest else '#111827', width=5 if newest else 3),
                                name='今回追加した切断' if newest else 'それまでの切断', hoverinfo='name'))
    fig.update_layout(title=f'Split {step} / {len(records)} · {len(groups)} 領域',
                      xaxis_title='Ω u', yaxis=dict(title='Ω v', scaleanchor='x', scaleratio=1),
                      height=620, uirevision='split-process-canonical', legend=dict(orientation='h'))
    return fig, rows
