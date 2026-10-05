"""Panel-quality diagnostics for K3D/T3D/T2D views.

The diagnostics deliberately reuse the same geometric definitions as the active
paper solvers:
* K3D planarity: max point-to-best-fit-plane distance of each quad.
* T2D collision: the same convex-hull SAT penetration test used by
  paper_hinge_solver / paper_eq5_k2d_solver.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def _best_fit_plane(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[-1]
    return p - np.outer((p - c) @ n, n)


def quad_planarity_residuals(vertices, faces) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    out = np.zeros(len(f), dtype=float)
    for i, face in enumerate(f):
        p = v[np.asarray(face, dtype=int)]
        q = _best_fit_plane(p)
        out[i] = float(np.max(np.linalg.norm(p - q, axis=1)))
    return out


def _quad_mesh_heatmap(vertices, faces, values, title: str, colorbar_title: str) -> go.Figure:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    values = np.asarray(values, dtype=float)
    x=[]; y=[]; z=[]; ii=[]; jj=[]; kk=[]; cell=[]
    ex=[]; ey=[]; ez=[]
    for fi, face in enumerate(f):
        ids = np.asarray(face, dtype=int)
        p = v[ids, :3]
        base = len(x)
        x.extend(p[:,0]); y.extend(p[:,1]); z.extend(p[:,2])
        ii.extend([base, base]); jj.extend([base+1, base+2]); kk.extend([base+2, base+3])
        cell.extend([float(values[fi]), float(values[fi])])
        closed=np.vstack([p,p[0]])
        ex.extend([*closed[:,0].tolist(),None]); ey.extend([*closed[:,1].tolist(),None]); ez.extend([*closed[:,2].tolist(),None])
    fig=go.Figure()
    if ii:
        fig.add_trace(go.Mesh3d(
            x=x,y=y,z=z,i=ii,j=jj,k=kk,
            intensity=cell,intensitymode="cell",colorscale="Turbo",
            cmin=0.0,cmax=max(float(np.max(values)),1e-15),
            colorbar=dict(title=colorbar_title),
            flatshading=True,opacity=1.0,name=title,
            hovertemplate="planarity residual=%{intensity:.6g}<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode="lines",line=dict(color="#303030",width=1),hoverinfo="skip",showlegend=False))
    fig.update_layout(title=title,scene=dict(aspectmode="data"),margin=dict(l=0,r=0,t=40,b=0))
    return fig


def figure_k3d_planarity(mesh) -> go.Figure:
    residuals=quad_planarity_residuals(mesh.vertices,mesh.faces)
    return _quad_mesh_heatmap(mesh.vertices,mesh.faces,residuals,"K3D — panel planarity residual","max plane distance")


def planarized_k3d_from_t3d(mesh, tiles_3d):
    """Reconstruct shared K3D vertices from the solved T3D top faces."""
    faces=np.asarray(mesh.faces,dtype=int)
    original=np.asarray(mesh.vertices,dtype=float)
    tiles=np.asarray(tiles_3d.vertices,dtype=float)
    accum=np.zeros_like(original)
    count=np.zeros(len(original),dtype=int)
    n=min(len(faces),len(tiles))
    for ti in range(n):
        for li,vid in enumerate(faces[ti]):
            accum[int(vid)] += tiles[ti,li,:3]
            count[int(vid)] += 1
    out=original.copy()
    mask=count>0
    out[mask]=accum[mask]/count[mask,None]
    return out


def figure_t3d_planarized_k3d(mesh, tiles_3d) -> go.Figure:
    v=planarized_k3d_from_t3d(mesh,tiles_3d)
    residuals=quad_planarity_residuals(v,mesh.faces)
    return _quad_mesh_heatmap(v,mesh.faces,residuals,"T3D-planarized K3D — solved top surface","max plane distance")


def _collision_pairs_sat(xy: np.ndarray, faces: np.ndarray, tolerance: float, exclude_face_pairs=None) -> np.ndarray:
    """Same SAT overlap/depth criterion as collision_energy_gradient, IDs only."""
    polygons=xy[faces]
    lo,hi=polygons.min(axis=1),polygons.max(axis=1)
    order=np.argsort(lo[:,0],kind="stable")
    pair_a=[];pair_b=[]
    for pos,i in enumerate(order):
        end=np.searchsorted(lo[order,0],hi[i,0],side="left")
        candidates=order[pos+1:end]
        keep=(hi[candidates,1]>lo[i,1]) & (lo[candidates,1]<hi[i,1])
        candidates=candidates[keep]
        pair_a.extend([i]*len(candidates));pair_b.extend(candidates)
    ii,jj=np.asarray(pair_a,int),np.asarray(pair_b,int)
    if exclude_face_pairs and len(ii):
        excluded={tuple(sorted((int(a),int(b)))) for a,b in exclude_face_pairs}
        keep=np.fromiter((tuple(sorted((int(a),int(b)))) not in excluded for a,b in zip(ii,jj)),dtype=bool,count=len(ii))
        ii,jj=ii[keep],jj[keep]
    if len(ii)==0:
        return np.zeros((0,2),dtype=int)
    a,b=polygons[ii],polygons[jj]
    edges=np.concatenate((np.roll(a,-1,axis=1)-a,np.roll(b,-1,axis=1)-b),axis=1)
    lengths=np.linalg.norm(edges,axis=2)
    normals=np.stack((-edges[...,1],edges[...,0]),axis=-1)/np.maximum(lengths[...,None],1e-30)
    ap=np.einsum("pvd,pad->pav",a,normals)
    bp=np.einsum("pvd,pad->pav",b,normals)
    d1,d2=ap.max(axis=2)-bp.min(axis=2),bp.max(axis=2)-ap.min(axis=2)
    depths=np.minimum(d1,d2)
    depths[lengths<=1e-30]=np.inf
    depth=np.min(depths,axis=1)
    live=depth>float(tolerance)
    return np.stack((ii[live],jj[live]),axis=1) if np.any(live) else np.zeros((0,2),dtype=int)


def t2d_collision_tile_ids(assembly, hinge_graph=None) -> tuple[np.ndarray,np.ndarray]:
    rest=np.asarray(assembly.vertices,dtype=float)
    if len(rest)==0:
        return np.zeros(0,dtype=int),np.zeros((0,2),dtype=int)
    hulls=[ConvexHull(tile[:,:2]).vertices.tolist() for tile in rest]
    width=max(map(len,hulls))
    # Match paper_hinge_solver exactly: each tile has eight flattened vertices.
    faces=np.array([[8*i+j for j in h+[h[-1]]*(width-len(h))] for i,h in enumerate(hulls)],dtype=int)
    flat=rest[:,:,:2].reshape(-1,2)
    scale=float(np.median(np.linalg.norm(np.roll(rest[:,:4,:2],-1,axis=1)-rest[:,:4,:2],axis=2)))
    hinge_pairs=set()
    if hinge_graph is not None:
        hinge_pairs={(min(int(h.tile_a),int(h.tile_b)),max(int(h.tile_a),int(h.tile_b))) for h in hinge_graph.hinges}
    pairs=_collision_pairs_sat(flat,faces,scale*1e-8,exclude_face_pairs=hinge_pairs)
    ids=np.unique(pairs.reshape(-1)) if len(pairs) else np.zeros(0,dtype=int)
    return ids.astype(int),pairs.astype(int)


def add_t2d_collision_overlay(fig: go.Figure, assembly, hinge_graph=None) -> tuple[go.Figure,np.ndarray,np.ndarray]:
    ids,pairs=t2d_collision_tile_ids(assembly,hinge_graph=hinge_graph)
    if len(ids)==0:
        return fig,ids,pairs
    vertices=np.asarray(assembly.vertices,dtype=float)
    x=[];y=[];z=[];ii=[];jj=[];kk=[]
    for ti in ids:
        p=vertices[int(ti),:4,:3]
        base=len(x)
        x.extend(p[:,0]);y.extend(p[:,1]);z.extend(p[:,2])
        ii.extend([base,base]);jj.extend([base+1,base+2]);kk.extend([base+2,base+3])
    fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=ii,j=jj,k=kk,color="#ef4444",opacity=.92,flatshading=True,
                            name=f"Colliding panels ({len(ids)})",hoverinfo="name"))
    return fig,ids,pairs


__all__=[
    "quad_planarity_residuals","figure_k3d_planarity","planarized_k3d_from_t3d",
    "figure_t3d_planarized_k3d","t2d_collision_tile_ids","add_t2d_collision_overlay",
]
