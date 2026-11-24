"""Corresponding cut banks and welded edges of the final triangle panels."""
import numpy as np
import plotly.graph_objects as go
from ..scripts.os_simple.common import CutMesh

COLORS = ('#a1194e', '#007b77', '#ad5600', '#6750b3', '#126bbe')


def chart_paths(chart, spatial=False):
    cm = CutMesh(chart['xyz'], chart['uv'], chart['f3'], chart['f2'])
    _, runs, pairs = cm.branches()
    for branch, pair in enumerate(pairs):
        for bank in pair[:1] if spatial else pair:
            edges = runs[bank]
            uv_ids = cm.loop[np.r_[edges, (edges[-1]+1) % len(cm.loop)]]
            yield branch, (cm.X if spatial else cm.U)[uv_ids]


def final_flat_faces(grid, panels):
    mapped = grid['wid'][grid['Tg']]
    survived = np.all(np.diff(np.sort(mapped, axis=1), axis=1) != 0, axis=1)
    return np.delete(grid['Tg'][survived], panels['removed_panels'], axis=0)


def panel_paths(grid, panels, flat=False):
    # A fin can disappear during surface relaxation. Emit only edges shared by
    # two surviving panels and split the line wherever a segment disappeared.
    if not grid.get('pairs'):
        return
    remap = np.full(len(grid['VW']), -1, dtype=int)
    remap[panels['kept_vertices']] = np.arange(len(panels['X']))
    faces = panels['T']
    edges = faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2)
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    interior = set(map(tuple, unique[counts == 2]))
    for branch, pair in enumerate(grid.get('pairs', [])):
        for bank in pair if flat else pair[:1]:
            k = grid['sec_of_run'][bank]
            section = grid['sections'][k]
            start = section['start']
            end = (start + section['n']) % len(grid['L'])
            ids = np.array([grid['pin_g'][start], *grid['sec_g'][k], grid['pin_g'][end]])
            final = remap[grid['wid'][ids]]
            xyz = grid['Xg'][ids] if flat else panels['X'][np.maximum(final, 0)]
            pieces = []
            for i, pair_ids in enumerate(zip(final, final[1:])):
                if tuple(sorted(pair_ids)) in interior:
                    pieces.extend((xyz[i], xyz[i+1], np.full(xyz.shape[1], np.nan)))
            if pieces:
                yield branch, np.array(pieces)


def add_paths(fig, paths):
    for branch, line in paths:
        z = line[:,2] if line.shape[1] == 3 else np.zeros(len(line))
        fig.add_trace(go.Scatter3d(x=line[:,0], y=line[:,1], z=z, mode='lines',
            name=f'seam {branch+1}', line=dict(color=COLORS[branch % len(COLORS)], width=8),
            showlegend=False, hoverinfo='name', meta='seam'))
