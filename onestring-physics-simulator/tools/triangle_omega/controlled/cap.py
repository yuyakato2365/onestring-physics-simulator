"""Cap planar input rims after outer-surface filtering and before OptCuts."""
import os
from pathlib import Path
import sys
import numpy as np
import trimesh
import triangle

def close_rims(vertices, faces):
    edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    rim=edges[counts[inverse]==1]
    if not len(rim): return vertices,faces
    if counts.max()>2 or len(np.unique(rim[:,0]))!=len(rim): raise ValueError('入力の境界が分岐しています。')
    nxt=dict(rim)
    verts=[vertices]; patches=[faces]; base=len(vertices)
    area=np.median(trimesh.triangles.area(vertices[faces]))
    while nxt:
        first=next(iter(nxt));loop=[first];v=nxt.pop(first)
        while v!=first:
            loop.append(v);v=nxt.pop(v)
        loop=np.asarray(loop)
        x=vertices[loop];centre=x.mean(0);_,_,basis=np.linalg.svd(x-centre,full_matrices=False)
        xy=(x-centre)@basis[:2].T
        if np.ptp((x-centre)@basis[2])>1e-5*np.linalg.norm(np.ptp(x,axis=0)):
            raise ValueError('入力に平面でない開口部があります。閉じた入力を指定するか「平面の開口部を閉じる」を外してください。')
        idx=np.arange(len(loop));seg=np.column_stack([idx,np.roll(idx,-1)])
        cap=triangle.triangulate(dict(vertices=xy,segments=seg),'pYq20a'+format(area,'.15f'))
        local=cap['triangles'].copy()
        if np.any((local==0)&(np.roll(local,-1,axis=1)==1)):local=local[:,::-1]
        extra=cap['vertices'][len(loop):]@basis[:2]+centre
        verts.append(extra);patches.append(np.r_[loop,base+np.arange(len(extra))][local]);base+=len(extra)
        print('Closed planar rim:',len(loop),'edges,',len(local),'cap triangles',flush=True)
    v=np.vstack(verts);f=np.vstack(patches)
    mesh=trimesh.Trimesh(v,f,process=False)
    if not (mesh.is_watertight and mesh.is_winding_consistent): raise ValueError('入力の開口部を閉じられませんでした。')
    return v,f


if __name__=='__main__':
    m=trimesh.load(sys.argv[1],process=False)
    v,f=close_rims(m.vertices,m.faces)
    trimesh.Trimesh(v,f,process=False).export(sys.argv[2],digits=17)
