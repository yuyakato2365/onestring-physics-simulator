"""Paper-aligned K3D -> T3D extrusion for OneString Sec. 4.2.

The paper says to offset K3D vertices along mesh normals, then optimize the
extruded vertices with Eq. (2) to enforce planarity of top, bottom, and contact
faces.  This module implements that route as an explicit local/global projection
solver.  It intentionally avoids the repository's miter/contact-plane clipping
route.

The paper does not fully specify all numerical gauge/thickness details.  We use
the normal-offset extrusion as the rest shape and a weak rest-position term only
to remove null/drift modes while planarity projections dominate.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def _normalize(v, fallback=None):
    v=np.asarray(v,float); n=float(np.linalg.norm(v))
    if n>1e-12: return v/n
    return np.asarray([0.,0.,1.] if fallback is None else fallback,float)


def _vertex_normals(vertices, faces):
    v=np.asarray(vertices,float); f=np.asarray(faces,int)
    n=np.zeros_like(v)
    for face in f:
        q=v[face]
        # Area-weighted two-triangle quad normal.
        fn=np.cross(q[1]-q[0],q[2]-q[0])+np.cross(q[2]-q[0],q[3]-q[0])
        for idx in face: n[int(idx)]+=fn
    out=np.empty_like(n)
    for i,x in enumerate(n): out[i]=_normalize(x)
    return out


def _best_fit_plane(points):
    p=np.asarray(points,float); c=p.mean(axis=0)
    _,_,vt=np.linalg.svd(p-c,full_matrices=False)
    normal=vt[-1]
    return p-np.outer((p-c)@normal,normal)


def _solve(n_vertices,constraints,anchor,anchor_weight):
    rows=[];cols=[];data=[];rhs=[];r=0
    for ids,targets,weight in constraints:
        sw=math.sqrt(max(float(weight),0.0))
        if sw==0: continue
        for vid,target in zip(ids,np.asarray(targets,float)):
            for d in range(3):
                rows.append(r);cols.append(int(vid)*3+d);data.append(sw);rhs.append(sw*float(target[d]));r+=1
    if anchor_weight>0:
        sw=math.sqrt(anchor_weight)
        for i,p in enumerate(anchor):
            for d in range(3):
                rows.append(r);cols.append(i*3+d);data.append(sw);rhs.append(sw*float(p[d]));r+=1
    A=sparse.coo_matrix((data,(rows,cols)),shape=(r,n_vertices*3)).tocsr()
    return lsqr(A,np.asarray(rhs),atol=1e-11,btol=1e-11,iter_lim=max(500,4*n_vertices))[0].reshape(n_vertices,3)


def _planarity_error(x,face_sets):
    vals=[]
    for ids in face_sets:
        p=x[np.asarray(ids,int)]
        proj=_best_fit_plane(p)
        vals.append(float(np.max(np.linalg.norm(p-proj,axis=1))))
    return max(vals,default=0.0)


def extrude_paper_face_planarity(mesh,thickness,stage,pipeline):
    """K3D -> T3D: mesh-normal offset followed by Eq.(2) face-planarity solve."""
    started=time.perf_counter()
    top_global=np.asarray(mesh.vertices,float)
    mesh_faces=np.asarray(mesh.faces,int)
    tile_count=len(mesh_faces)
    if tile_count==0:
        return pipeline._extrude_tiles(mesh,thickness,stage)

    normals=_vertex_normals(top_global,mesh_faces)
    bottom_global=top_global-float(thickness)*normals

    # Tiles remain independent rigid bodies downstream, but the initial extrusion
    # is generated from shared mesh vertices exactly as described in the paper.
    x0=np.zeros((tile_count*8,3),float)
    top_face_ids=[];bottom_face_ids=[];side_face_ids=[]
    for ti,face in enumerate(mesh_faces):
        base=8*ti
        x0[base:base+4]=top_global[face]
        x0[base+4:base+8]=bottom_global[face]
        top_face_ids.append([base+i for i in (0,1,2,3)])
        bottom_face_ids.append([base+i for i in (4,7,6,5)])
        side_face_ids.extend([
            [base+i for i in (0,1,5,4)],
            [base+i for i in (1,2,6,5)],
            [base+i for i in (2,3,7,6)],
            [base+i for i in (3,0,4,7)],
        ])

    all_faces=top_face_ids+bottom_face_ids+side_face_ids
    x=x0.copy()
    iterations=max(1,int(getattr(mesh,"metrics",{}).get("paper_t3d_planarity_iterations",40)))
    weight=float(getattr(mesh,"metrics",{}).get("paper_t3d_planarity_weight",1.0))
    anchor_weight=float(getattr(mesh,"metrics",{}).get("paper_t3d_rest_weight",1e-3))
    before=_planarity_error(x,all_faces)
    done=0
    for it in range(iterations):
        constraints=[]
        for ids in all_faces:
            ids_arr=np.asarray(ids,int)
            projected=_best_fit_plane(x[ids_arr])
            constraints.append((ids,projected,weight))
        new=_solve(len(x),constraints,x0,anchor_weight)
        step=float(np.linalg.norm(new-x)/max(math.sqrt(len(x)),1.0))
        x=new;done=it+1
        if step<1e-10: break

    after=_planarity_error(x,all_faces)
    tiles=x.reshape(tile_count,8,3)

    # Recover a proper rigid top->bottom transform per tile, as required by the
    # paper's subsequent K2D -> T2D step.  This is the best rigid transform after
    # face-planarity optimization, not an affine/shear map.
    transforms=np.zeros((tile_count,4,4),float)
    rigid_rms=[]
    for ti,tile in enumerate(tiles):
        a=tile[:4];b=tile[4:]
        ca=a.mean(axis=0);cb=b.mean(axis=0)
        u,_,vt=np.linalg.svd((a-ca).T@(b-cb))
        corr=np.diag([1.,1.,np.linalg.det(vt.T@u.T)])
        R=vt.T@corr@u.T
        trans=cb-R@ca
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=trans;transforms[ti]=T
        pred=a@R.T+trans
        rigid_rms.append(float(np.sqrt(np.mean(np.sum((pred-b)**2,axis=1)))))

    grouped=pipeline._tile_face_planarity_by_group(tiles)
    metrics=dict(getattr(mesh,"metrics",{}))
    metrics.update({
        "objective":"Paper Sec. 4.2: mesh-normal offset + Eq.(2) top/bottom/contact face planarity",
        "extrusion_model":"paper_mesh_normal_offset_face_planarity_local_global",
        "paper_t3d_extrusion":True,
        "paper_t3d_normal_model":"shared K3D mesh vertex normals",
        "paper_t3d_planarity_iterations":done,
        "paper_t3d_planarity_error_before":before,
        "paper_t3d_planarity_error_after":after,
        "face_planarity_error":after,
        "top_face_planarity_error":grouped["top"],
        "bottom_face_planarity_error":grouped["bottom"],
        "side_face_planarity_error":grouped["side"],
        "tile_thickness":float(thickness),
        "paper_t3d_rest_weight":anchor_weight,
        "paper_t3d_rigid_transform_rms":float(np.mean(rigid_rms)) if rigid_rms else 0.0,
        "paper_t3d_note":"Eq.(2) planarity route; weak normal-offset rest term is an implementation gauge because the paper does not specify the numerical gauge."
    })
    local_top=np.tile(np.asarray([0,1,2,3],int),(tile_count,1))
    local_bottom=np.tile(np.asarray([4,7,6,5],int),(tile_count,1))
    local_sides=np.asarray([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]],int)
    assembly=pipeline.TileAssembly(
        vertices=tiles,top_faces=local_top,bottom_faces=local_bottom,
        side_faces=local_sides,stage=stage,metrics=metrics,transform_matrices=transforms)
    report=pipeline.StageReport(
        name=f"{mesh.stage} -> {stage}",
        objective=metrics["objective"],before_error=before,after_error=after,
        constraint_violation=after,computation_time=time.perf_counter()-started,
        counts=pipeline._assembly_counts(assembly))
    print(f"[PAPER-T3D] iterations={done} planarity={before:.6g}->{after:.6g} rigid_rms={metrics['paper_t3d_rigid_transform_rms']:.6g}",flush=True)
    return assembly,report


__all__=["extrude_paper_face_planarity"]
