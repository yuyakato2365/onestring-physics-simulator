"""Triangle stages drawn with the OneString scene and Omega correspondence colors."""
import pickle

import numpy as np
import plotly.graph_objects as go
from onestring_physics.visualization import _add_quad_mesh_surface, _style_scene, correspondence_colors


def stage_figure(folder, stage):
    with (folder/'omega2_chart.pkl').open('rb') as stream:
        chart = pickle.load(stream)
    fig = go.Figure()
    colors = None
    if stage == 'S':
        points, faces = chart['xyz'], chart['f3']
    elif stage == 'Omega':
        points, faces = chart['uv'], chart['f2']
    else:
        with (folder/'grid/result.pkl').open('rb') as stream:
            grid = pickle.load(stream)
        with (folder/'remesh3d.pkl').open('rb') as stream:
            panels = pickle.load(stream)
        flat_faces = np.delete(grid['Tg'], panels['removed_panels'], axis=0)
        uv = grid['Xg'][flat_faces].reshape(-1, 2)
        colors = correspondence_colors(uv, grid['O'])
        faces = np.arange(len(uv)).reshape(-1, 3)
        points = uv if stage == 'M2D' else panels['X'][panels['T']].reshape(-1, 3)
        if stage == 'M3D':
            target = chart['xyz']
            i, j, k = chart['f3'].T
            fig.add_trace(go.Mesh3d(x=target[:, 0], y=target[:, 1], z=target[:, 2],
                                    i=i, j=j, k=k, name='target surface S', visible='legendonly',
                                    opacity=0.28, color='#94a3b8'))
    planar = points.shape[1] == 2
    if planar:
        points = np.column_stack([points, np.zeros(len(points))])
    _add_quad_mesh_surface(fig, points, faces, color='#7c3aed' if stage == 'S' else '#14b8a6',
                           opacity=1, name=stage, vertex_colors=colors)
    _style_scene(fig)
    eye = dict(x=0, y=0, z=2) if planar else dict(x=1.3, y=0.55, z=1.6)
    fig.update_layout(title=stage, scene_camera=dict(eye=eye, up=dict(x=0, y=1, z=0)))
    return fig
