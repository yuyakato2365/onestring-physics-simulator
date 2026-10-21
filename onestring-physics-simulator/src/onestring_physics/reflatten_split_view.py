"""Read-only 3D replay of per-piece reflatten Split decisions.

Plots the actual pipeline M3D quad coordinates (not postcut free-BFF UV charts).
The original T2D/K3D mesh and all optimization states are not modified.
"""
from __future__ import annotations

import numpy as np


def figure_reflatten_split_3d(state, step):
    """Return a Plotly M3D component view and a short per-piece CSF table."""
    # Delay optional graph dependencies until a user opens Split Map.
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    m2d=state.mesh_2d_initial
    m3d=state.mesh_3d_initial
    metrics=getattr(m2d,"metrics",{})
    records=list(metrics.get("csf_split_step_analysis",()))
    initial=metrics.get("csf_split_initial_components")
    if initial is None or metrics.get("csf_split_criterion")!="reflatten_tile_area":
        return None,[]
    faces=np.asarray(m3d.faces,int)
    vertices=np.asarray(m3d.vertices,float)
    if (faces.ndim!=2 or faces.shape[1]!=4 or vertices.ndim!=2 or
            vertices.shape[1]!=3 or not np.isfinite(vertices).all()):
        return None,[]
    if len(m2d.faces)!=len(faces):
        return None,[]

    limit=min(max(int(step),0),len(records))
    groups=[(list(row["face_ids"]),float(row["sigma"])) for row in initial]
    for rec in records[:limit]:
        parent=set(rec["face_ids"])
        groups=[row for row in groups if set(row[0])!=parent]
        groups.extend((list(ids),float(sigma))
                      for ids,sigma in zip(rec["child_face_ids"],rec["sigma_after"]))
    groups.sort(key=lambda x:min(x[0]) if x[0] else -1)
    fig=go.Figure();rows=[]
    for i,(ids,sigma) in enumerate(groups):
        tiles=faces[np.asarray(ids,int)]
        if not len(tiles):continue
        # Flatten each quad into two triangles without changing any vertex.
        tri=np.concatenate([tiles[:,[0,1,2]],tiles[:,[0,2,3]]])
        used,ind=np.unique(tri,return_inverse=True)
        P=vertices[used];T=ind.reshape(-1,3)
        color=qualitative.Pastel[i%len(qualitative.Pastel)]
        fig.add_trace(go.Mesh3d(
            x=P[:,0],y=P[:,1],z=P[:,2],
            i=T[:,0],j=T[:,1],k=T[:,2],
            color=color,opacity=.87,flatshading=True,
            name=f"Piece {i+1} · sigma {sigma:.4g}",
            showlegend=True,
            hovertemplate=f"Piece {i+1}<br>{len(ids)} M2D panels<br>area span {sigma:.4g}<extra></extra>",
        ))
        rows.append({"Piece":i+1,"M2D panel count":len(ids),"reflatten area span":sigma})
    # Draw only TRUE saved broken edges, not inferred new connections.
    for newest in (False,True):
        source=records[max(0,limit-1):limit] if newest else records[:max(0,limit-1)]
        xs=[];ys=[];zs=[]
        for record in source:
            for a,edge,_,_ in record.get("split_boundary_pairs",[]):
                edge_vertices=faces[int(a),[int(edge),(int(edge)+1)%4]]
                pts=vertices[edge_vertices]
                xs.extend([float(pts[0,0]),float(pts[1,0]),None])
                ys.extend([float(pts[0,1]),float(pts[1,1]),None])
                zs.extend([float(pts[0,2]),float(pts[1,2]),None])
        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs,y=ys,z=zs,mode="lines",
                line=dict(color="#ef4444" if newest else "#101828",
                          width=8 if newest else 5),
                name="Latest real source cut" if newest else "Earlier source cuts",
            ))
    fig.update_layout(
        title=f"Reflatten-based source Split · M3D · {limit}/{len(records)} cuts",
        scene=dict(aspectmode="data",xaxis_title="x",yaxis_title="y",zaxis_title="z"),
        margin=dict(l=5,r=5,t=45,b=5),height=630,
        uirevision="reflatten-original-M3D",
    )
    return fig,rows
