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
    """K3D -> T3D using shared mesh vertices during Eq.(2) optimization.

    The optimization unknowns are two copies of the *mesh* vertices (top and
    bottom), not eight independent vertices per tile.  Consequently adjacent
    tiles cannot drift apart at a shared K3D edge.  Only after the shared solve
    converges do we expand the result into the downstream tile_count x 8 layout.
    """
    mapping = getattr(mesh, "metrics", {}).get("assembled_geometry_map")
    if mapping is not None:
        from .extrusion_aware import geometry_view
        mapping = np.asarray(mapping, int)
        view = geometry_view(mesh, mapping)
        assembly, report = extrude_paper_face_planarity(view, thickness, stage, pipeline)
        mesh.vertices = view.vertices[mapping].copy()
        mesh.metrics.update(view.metrics)
        assembly.metrics["assembled_geometry_map"] = mapping.tolist()
        # Faces and vertex IDs on the caller are untouched; only geometry moved.
        return assembly, report
    started=time.perf_counter()
    top_global=np.asarray(mesh.vertices,float)
    mesh_faces=np.asarray(mesh.faces,int)
    tile_count=len(mesh_faces)
    if tile_count==0:
        return pipeline._extrude_tiles(mesh,thickness,stage)
    if top_global.ndim != 2 or top_global.shape[1] != 3 or not np.isfinite(top_global).all():
        raise RuntimeError(
            f"K3D -> T3D preflight: invalid K3D vertices shape={top_global.shape}, "
            f"nonfinite={int(np.size(top_global)-np.count_nonzero(np.isfinite(top_global)))}"
        )
    if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 4:
        raise RuntimeError(f"K3D -> T3D preflight: expected quad faces, got {mesh_faces.shape}")
    if int(mesh_faces.min()) < 0 or int(mesh_faces.max()) >= len(top_global):
        raise RuntimeError("K3D -> T3D preflight: face index outside K3D vertex range")

    n_mesh_vertices=len(top_global)
    normals=_vertex_normals(top_global,mesh_faces)
    bottom_global=top_global-float(thickness)*normals

    # Eq.(2) is solved on shared top/bottom mesh vertices.  A mesh edge has one
    # geometric contact quad, even when two tiles are incident to that edge.
    x0=np.vstack([top_global,bottom_global])
    top_face_ids=[]
    bottom_face_ids=[]
    contact_face_ids=[]
    for face in mesh_faces:
        f=[int(v) for v in face]
        top_face_ids.append(f)
        bottom_face_ids.append([n_mesh_vertices+v for v in (f[3],f[2],f[1],f[0])])

    unique_edges=set()
    for face in mesh_faces:
        f=[int(v) for v in face]
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[3]),(f[3],f[0])):
            key=(min(a,b),max(a,b))
            if key in unique_edges:
                continue
            unique_edges.add(key)
            contact_face_ids.append([a,b,n_mesh_vertices+b,n_mesh_vertices+a])

    all_faces=top_face_ids+bottom_face_ids+contact_face_ids
    x=x0.copy()
    metrics_in=getattr(mesh,"metrics",{})
    iterations=max(1,int(metrics_in.get("paper_t3d_planarity_iterations",40)))
    weight=float(metrics_in.get("paper_t3d_planarity_weight",1.0))
    anchor_weight=float(metrics_in.get("paper_t3d_rest_weight",1e-3))
    before=_planarity_error(x,all_faces)
    done=0
    for it in range(iterations):
        constraints=[]
        for ids in all_faces:
            ids_arr=np.asarray(ids,int)
            projected=_best_fit_plane(x[ids_arr])
            constraints.append((ids,projected,weight))
        new=_solve(len(x),constraints,x0,anchor_weight)
        if not np.isfinite(new).all():
            # Never let a failed T3D local/global step poison K2D/T2D with NaNs.
            # The previous iterate is finite and already represents the paper
            # normal-offset extrusion; stop refinement there instead.
            print(f"[PAPER-T3D] non-finite solve at iter={it+1}; keeping previous finite iterate", flush=True)
            break
        step=float(np.linalg.norm(new-x)/max(math.sqrt(len(x)),1.0))
        x=new
        done=it+1
        if step<1e-10:
            break

    if not np.isfinite(x).all():
        raise RuntimeError("K3D -> T3D produced non-finite shared extrusion vertices")
    after=_planarity_error(x,all_faces)
    top_solved=x[:n_mesh_vertices]
    bottom_solved=x[n_mesh_vertices:]

    # The T3D face-planarity solve moves the shared top vertices.  Keep K3D and
    # T3D geometrically consistent by promoting that solved top surface to the
    # final K3D result.  This is an in-place update of the same QuadMesh object
    # retained by the pipeline/state, so downstream K2D construction and the K3D
    # viewer both consume exactly the top surface used by T3D.
    k3d_before=np.asarray(mesh.vertices,float).copy()
    k3d_update_rms=float(np.sqrt(np.mean(np.sum((top_solved-k3d_before)**2,axis=1))))
    k3d_update_max=float(np.max(np.linalg.norm(top_solved-k3d_before,axis=1),initial=0.0))
    mesh.vertices=np.asarray(top_solved,float).copy()
    if hasattr(mesh,"metrics") and isinstance(mesh.metrics,dict):
        mesh.metrics.update({
            "paper_t3d_planarity_promoted_to_k3d":True,
            "paper_t3d_k3d_update_rms":k3d_update_rms,
            "paper_t3d_k3d_update_max":k3d_update_max,
            "paper_t3d_k3d_source":"T3D shared top vertices after face-planarity solve",
        })

    # Expand only after optimization.  Shared vertices are copied verbatim into
    # each tile, so every adjacent tile receives exactly the same joint geometry.
    tiles=np.zeros((tile_count,8,3),float)
    for ti,face in enumerate(mesh_faces):
        f=np.asarray(face,int)
        tiles[ti,:4]=top_solved[f]
        tiles[ti,4:]=bottom_solved[f]

    # Diagnostics for the two paper invariants that matter downstream.
    edge_copy_separation=0.0
    incidence={}
    for ti,face in enumerate(mesh_faces):
        for li,v in enumerate(face):
            incidence.setdefault(int(v),[]).append((ti,li))
    for copies in incidence.values():
        if len(copies)<2:
            continue
        ref_top=tiles[copies[0][0],copies[0][1]]
        ref_bottom=tiles[copies[0][0],4+copies[0][1]]
        for ti,li in copies[1:]:
            edge_copy_separation=max(
                edge_copy_separation,
                float(np.linalg.norm(tiles[ti,li]-ref_top)),
                float(np.linalg.norm(tiles[ti,4+li]-ref_bottom)),
            )

    parallel_angles=[]
    for tile in tiles:
        top_n=_normalize(np.cross(tile[1]-tile[0],tile[2]-tile[0]))
        bot_n=_normalize(np.cross(tile[5]-tile[4],tile[6]-tile[4]))
        dot=float(np.clip(abs(np.dot(top_n,bot_n)),-1.0,1.0))
        parallel_angles.append(math.degrees(math.acos(dot)))

    # The paper's T2D stage consumes a per-tile top->bottom rigid transform.
    transforms=np.zeros((tile_count,4,4),float)
    rigid_rms=[]
    for ti,tile in enumerate(tiles):
        a=tile[:4]
        b=tile[4:]
        ca=a.mean(axis=0)
        cb=b.mean(axis=0)
        u,_,vt=np.linalg.svd((a-ca).T@(b-cb))
        corr=np.diag([1.,1.,np.linalg.det(vt.T@u.T)])
        R=vt.T@corr@u.T
        trans=cb-R@ca
        T=np.eye(4)
        T[:3,:3]=R
        T[:3,3]=trans
        transforms[ti]=T
        pred=a@R.T+trans
        rigid_rms.append(float(np.sqrt(np.mean(np.sum((pred-b)**2,axis=1)))))

    grouped=pipeline._tile_face_planarity_by_group(tiles)
    metrics=dict(metrics_in)
    metrics.update({
        "objective":"Paper Sec. 4.2: shared-mesh normal offset + Eq.(2) top/bottom/contact face planarity",
        "extrusion_model":"paper_shared_mesh_normal_offset_face_planarity_local_global",
        "paper_t3d_extrusion":True,
        "paper_t3d_normal_model":"shared K3D mesh vertex normals",
        "paper_t3d_optimization_topology":"shared top/bottom mesh vertices; expand to 8-vertex tiles after solve",
        "paper_t3d_planarity_iterations":done,
        "paper_t3d_planarity_error_before":before,
        "paper_t3d_planarity_error_after":after,
        "paper_t3d_planarity_promoted_to_k3d":True,
        "paper_t3d_k3d_update_rms":k3d_update_rms,
        "paper_t3d_k3d_update_max":k3d_update_max,
        "face_planarity_error":after,
        "top_face_planarity_error":grouped["top"],
        "bottom_face_planarity_error":grouped["bottom"],
        "side_face_planarity_error":grouped["side"],
        "tile_thickness":float(thickness),
        "paper_t3d_rest_weight":anchor_weight,
        "paper_t3d_shared_vertex_separation_max":edge_copy_separation,
        "paper_t3d_top_bottom_parallel_angle_max_deg":max(parallel_angles,default=0.0),
        "paper_t3d_top_bottom_parallel_angle_mean_deg":float(np.mean(parallel_angles)) if parallel_angles else 0.0,
        "paper_t3d_rigid_transform_rms":float(np.mean(rigid_rms)) if rigid_rms else 0.0,
        "paper_t3d_note":"Eq.(2) solved on shared mesh vertices. Weak normal-offset rest term fixes numerical drift; principal-curvature grid alignment is an upstream optional paper path and is not changed here."
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
    print(
        f"[PAPER-T3D] iterations={done} planarity={before:.6g}->{after:.6g} "
        f"k3d_promoted_rms={k3d_update_rms:.6g} k3d_promoted_max={k3d_update_max:.6g} "
        f"shared_sep={edge_copy_separation:.3g} "
        f"parallel_max_deg={metrics['paper_t3d_top_bottom_parallel_angle_max_deg']:.6g} "
        f"rigid_rms={metrics['paper_t3d_rigid_transform_rms']:.6g}",
        flush=True,
    )
    return assembly,report


__all__=["extrude_paper_face_planarity"]
