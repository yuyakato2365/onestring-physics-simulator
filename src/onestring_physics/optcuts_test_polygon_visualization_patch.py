"""Render true 3--5-gon OptCuts_test panels instead of legacy quad surrogates."""
from __future__ import annotations

from typing import Any
import numpy as np
import plotly.graph_objects as go


def install_optcuts_test_polygon_visualization_patch() -> None:
    from . import visualization as vis
    if getattr(vis, "_onestring_optcuts_test_polygon_visualization_installed", False):
        return
    base = vis.figure_quad_mesh

    def figure_quad_mesh(mesh: Any, *args: Any, **kwargs: Any):
        polygon_faces = getattr(mesh, "_polygon_faces", None)
        if not polygon_faces:
            return base(mesh, *args, **kwargs)
        vertices = np.asarray(mesh.vertices, dtype=float)
        title = kwargs.get("title", args[0] if args else str(getattr(mesh, "stage", "mesh")))
        tri_i=[]; tri_j=[]; tri_k=[]
        for face in polygon_faces:
            ids=[int(v) for v in face]
            if len(ids)<3: continue
            for j in range(1,len(ids)-1):
                tri_i.append(ids[0]); tri_j.append(ids[j]); tri_k.append(ids[j+1])
        fig=go.Figure()
        if tri_i:
            fig.add_trace(go.Mesh3d(
                x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                i=tri_i, j=tri_j, k=tri_k,
                color="#63c7bd", opacity=0.88, flatshading=True,
                name=str(getattr(mesh,"stage","mesh")), showscale=False,
            ))
        xs=[]; ys=[]; zs=[]
        for face in polygon_faces:
            ids=[int(v) for v in face]
            if len(ids)<2: continue
            pts=vertices[np.asarray(ids+[ids[0]],dtype=int)]
            xs.extend(pts[:,0].tolist()+[None]); ys.extend(pts[:,1].tolist()+[None]); zs.extend(pts[:,2].tolist()+[None])
        fig.add_trace(go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",line=dict(color="#111827",width=3),name="panel edges",showlegend=False))
        fig.update_layout(title=title)
        try:
            vis._style_scene(fig)
        except Exception:
            fig.update_layout(scene=dict(aspectmode="data"))
        return fig

    vis.figure_quad_mesh = figure_quad_mesh
    vis._onestring_optcuts_test_polygon_visualization_installed = True


__all__=["install_optcuts_test_polygon_visualization_patch"]
