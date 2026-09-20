"""Bounded Section 4.4 rigid pose solve for the reconstructed Eq.5 route.

Rigidity is exact (SE(2) poses of the full 8-vertex solid); connections use both
paired vertices' squared residuals. Collision is the conservative 2D convex
footprint SAT surrogate, not the paper's 3D nonpenetration implementation.
No artificial positive clearance is imposed on a coincident hinge.
"""
from __future__ import annotations

import copy
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from .paper_eq5_k2d_solver import collision_energy_gradient


class _Deadline(Exception):
    pass


def optimize_hinge_poses(grid, mesh_faces, t2d, t3d, params, pipeline, progress_callback=None):
    started=time.perf_counter()
    from .paper_t2d import build_hinge_graph, validate_flat_linkage
    graph=build_hinge_graph(t2d,t3d,pipeline,dual=True)
    rest=np.asarray(t2d.vertices,float)
    centers=rest[:,:4,:2].mean(axis=1)
    local=rest[:,:,:2]-centers[:,None,:]
    scale=float(np.median(np.linalg.norm(np.roll(rest[:,:4,:2],-1,axis=1)-rest[:,:4,:2],axis=2)))
    # Hull vertex IDs stay fixed under rigid motion. Pad repeated final vertices;
    # zero-length padding edges are excluded from SAT's axes.
    hulls=[ConvexHull(tile[:,:2]).vertices.tolist() for tile in rest]
    width=max(map(len,hulls))
    faces=np.array([[8*i+j for j in h+[h[-1]]*(width-len(h))] for i,h in enumerate(hulls)],int)
    ha=np.array([8*h.tile_a+h.local_vertex_a for h in graph.hinges],int)
    hb=np.array([8*h.tile_b+h.local_vertex_b for h in graph.hinges],int)
    wconn=float(params.hinge_layout_connection_weight)
    wcoll=float(params.hinge_layout_collision_weight)
    wanchor=float(params.hinge_layout_anchor_weight)
    if min(wconn,wcoll,wanchor)<0 or not np.isfinite([wconn,wcoll,wanchor]).all():
        raise ValueError('Hinge weights must be finite and nonnegative')
    budget=float(params.hinge_layout_time_budget_sec)
    deadline=started+budget if budget>0 else float('inf')
    records=[]
    last=np.zeros((len(rest),3),float)
    def positions(pose):
        angle=pose[:,0];c=np.cos(angle)[:,None];s=np.sin(angle)[:,None]
        rotated=np.stack((c*local[:,:,0]-s*local[:,:,1],s*local[:,:,0]+c*local[:,:,1]),axis=2)
        xy=rotated+centers[:,None,:]+scale*pose[:,None,1:]
        return xy,rotated
    def evaluate(pose):
        xy,rotated=positions(pose)
        flat=xy.reshape(-1,2)
        ec,g,ncoll=collision_energy_gradient(flat,faces,scale*1e-8)
        g*=wcoll
        delta=flat[ha]-flat[hb]
        dz=rest.reshape(-1,3)[ha,2]-rest.reshape(-1,3)[hb,2]
        conn=2*float(np.sum(delta*delta)+dz@dz)
        np.add.at(g,ha,4*wconn*delta);np.add.at(g,hb,-4*wconn*delta)
        g=g.reshape(-1,8,2)
        gradient=np.zeros_like(pose)
        gradient[:,0]=np.sum(g[:,:,0]*(-rotated[:,:,1])+g[:,:,1]*rotated[:,:,0],axis=1)
        gradient[:,1:]=scale*np.sum(g,axis=1)
        anchor=scale**2*float(np.sum(pose[:,1:]**2))
        gradient[:,1:]+=2*wanchor*scale**2*pose[:,1:]
        energy=wconn*conn+wcoll*ec+wanchor*anchor
        stats=dict(EHinge=energy,EConn=conn,ECollision=ec,collisions=ncoll,
                   hinge_rms=float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)+dz*dz))) if len(delta) else 0.)
        return energy,gradient,stats
    def fun(z):
        if time.perf_counter()>=deadline:
            raise _Deadline()
        e,g,_=evaluate(z.reshape(-1,3))
        return e/(scale**2),(g/(scale**2)).ravel()
    def record(pose):
        _,_,stats=evaluate(pose)
        records.append(dict(iteration=len(records),**stats))
        if len(records)>1 and records[-1]['EHinge']>records[-2]['EHinge']+1e-10*scale**2:
            raise RuntimeError('Hinge solver accepted an increasing objective')
        return stats
    record(last)
    def callback(z):
        nonlocal last
        last=z.reshape(-1,3).copy();stats=record(last)
        if progress_callback and (len(records)%10==0 or len(records)==2):
            progress_callback('Eq.6 rigid tile poses',min(.99,(len(records)-1)/max(1,params.hinge_layout_iterations)),
                              f"iter {len(records)-1}; EHinge={stats['EHinge']:.5g}; collisions={stats['collisions']}; hinge_rms={stats['hinge_rms']:.5g}")
    timed_out=False
    try:
        result=minimize(fun,last.ravel(),jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=max(1,params.hinge_layout_iterations),maxls=30,maxcor=20,ftol=1e-13,gtol=1e-8))
        last=result.x.reshape(-1,3);message=str(result.message)
    except _Deadline:
        timed_out=True;message='time budget reached; returning last accepted pose'
    xy,_=positions(last)
    out=copy.deepcopy(t2d);out.vertices[:,:,:2]=xy;out.stage='T2D dual hinge'
    # Compose the SE(2) placement with each original T3D->T2D rigid transform.
    for i,(angle,tx,ty) in enumerate(last):
        c,s=np.cos(angle),np.sin(angle)
        r=np.array([[c,-s],[s,c]])
        transform=np.eye(4);transform[:2,:2]=r
        transform[:2,3]=centers[i]+scale*np.array([tx,ty])-r@centers[i]
        out.transform_matrices[i]=transform@t2d.transform_matrices[i]
        out.top_to_bottom_transforms[i]=transform@t2d.top_to_bottom_transforms[i]@np.linalg.inv(transform)
    _,_,final=evaluate(last)
    out.metrics.update(graph.metrics)
    out.metrics.update(final)
    out.metrics.update(hinge_connection_error=final['hinge_rms'],flat_collision_count=final['collisions'],
                       dual_hinge_final_collision_count=final['collisions'],
                       dual_hinge_layout_optimizer='analytic L-BFGS on rigid SE(2) poses',
                       dual_hinge_timed_out=timed_out,dual_hinge_solver_message=message,
                       dual_hinge_iterations=len(records)-1,dual_hinge_history=records,
                       dual_hinge_collision_model='convex 2D footprint SAT; conservative surrogate for 3D solids',
                       dual_hinge_connection_model='2 * sum squared paired 3D corner distances',
                       tile_shape_max_error_to_T3D=pipeline._tile_shape_distance_error(out.vertices,t3d.vertices,use_max=True),
                       fabrication_feasible=final['collisions']==0 and final['hinge_rms']<scale*1e-4)
    out.metrics.update(validate_flat_linkage(out,t3d,pipeline,graph))
    out.metrics["paper_classification"]["optional_anchor_energy"] = "Project-specific extension"
    graph.metrics=dict(out.metrics)
    for hinge in graph.hinges:
        hinge.rest_position_2d=.5*(out.vertices[hinge.tile_a,hinge.local_vertex_a]+out.vertices[hinge.tile_b,hinge.local_vertex_b])
    report=pipeline.StageReport(name='T2D top hinge -> T2D dual hinge',
                               objective='Rigid SE(2) Eq.6 with convex footprint SAT surrogate',
                               before_error=records[0]['EHinge'],after_error=final['EHinge'],
                               constraint_violation=final['hinge_rms'],computation_time=time.perf_counter()-started,
                               counts={'tiles':len(rest),'hinges':len(graph.hinges)})
    if progress_callback:
        progress_callback('Eq.6 complete',1.,f"{message}; collisions={final['collisions']}; hinge_rms={final['hinge_rms']:.5g}")
    return out,graph,report
