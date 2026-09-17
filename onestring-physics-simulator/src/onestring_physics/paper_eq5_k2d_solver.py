"""Experimental paper Eq.(5) K2D on an explicit auxetic tile topology."""
from __future__ import annotations
import math,os,time
from typing import Any
import numpy as np

_LAST_EQ5_HISTORY=None

def get_last_eq5_history():
    return _LAST_EQ5_HISTORY

def _angle(a,b):
    na,nb=float(np.linalg.norm(a)),float(np.linalg.norm(b))
    if na<=1e-12 or nb<=1e-12:return 0.
    return float(math.acos(np.clip(float(np.dot(a,b))/(na*nb),-1.,1.)))
def _rot(v,angle):
    c,s=math.cos(angle),math.sin(angle);return np.asarray([c*v[0]-s*v[1],s*v[0]+c*v[1]],float)
def _env_float(name,fallback):
    try:return float(os.environ.get(name,str(fallback)))
    except Exception:return float(fallback)
def _env_int(name,fallback):
    try:return int(float(os.environ.get(name,str(fallback))))
    except Exception:return int(fallback)

def _build_auxetic_topology(mesh_2d):
    src_xy=np.asarray(mesh_2d.vertices[:,:2],float);src_faces=np.asarray(mesh_2d.faces,int);nf=len(src_faces)
    xy=np.zeros((4*nf,2),float);faces=np.arange(4*nf,dtype=int).reshape(nf,4);source_ids=np.zeros(4*nf,int);groups={};corner_info={}
    for f,face in enumerate(src_faces):
        for local,source in enumerate(face):
            q=4*f+local;source=int(source);xy[q]=src_xy[source];source_ids[q]=source;groups.setdefault(source,[]).append(q);corner_info[q]=(f,local,4*f+(local-1)%4,4*f+(local+1)%4)
    gap_pairs=[]
    for source,copies in groups.items():
        if len(copies)<2:continue
        center=src_xy[source];entries=[]
        for q in copies:
            f,local,prev,nxt=corner_info[q];vp=xy[prev]-center;vn=xy[nxt]-center
            if np.linalg.norm(vp)<=1e-12 or np.linalg.norm(vn)<=1e-12:continue
            up=vp/np.linalg.norm(vp);un=vn/np.linalg.norm(vn);bis=up+un
            if np.linalg.norm(bis)<=1e-12:bis=up
            phi=math.atan2(float(bis[1]),float(bis[0]));rays=[(math.atan2(float(vp[1]),float(vp[0])),prev),(math.atan2(float(vn[1]),float(vn[0])),nxt)];entries.append((phi,q,rays))
        entries.sort(key=lambda e:e[0]);m=len(entries)
        for i in range(m):
            phi_a,qa,rays_a=entries[i];phi_b,qb,rays_b=entries[(i+1)%m];sector=(phi_b-phi_a)%(2*math.pi)
            if sector<=1e-9 or sector>=math.pi+1e-6:continue
            def rel(angle):return (angle-phi_a)%(2*math.pi)
            cand_a=[(rel(a),n) for a,n in rays_a if rel(a)<=sector+1e-7];cand_b=[(rel(a),n) for a,n in rays_b if rel(a)<=sector+1e-7]
            if not cand_a or not cand_b:continue
            na=max(cand_a,key=lambda x:x[0])[1];nb=min(cand_b,key=lambda x:x[0])[1]
            if na!=nb:gap_pairs.append((qa,na,qb,nb))
    return xy,faces,source_ids,[np.asarray(v,int) for v in groups.values() if len(v)>1],gap_pairs

def _project_hinges(xy,groups):
    out=xy.copy()
    for ids in groups:out[ids]=np.mean(out[ids],axis=0)
    return out

def _project_edges(xy,faces,source_ids,mesh_3d):
    out=xy.copy();acc=np.zeros_like(out);counts=np.zeros((len(out),1));err=0.
    for face in faces:
        for k in range(4):
            a,b=int(face[k]),int(face[(k+1)%4]);sa,sb=int(source_ids[a]),int(source_ids[b]);target=float(np.linalg.norm(mesh_3d.vertices[sa]-mesh_3d.vertices[sb]));d=out[b]-out[a];length=float(np.linalg.norm(d))
            if length<=1e-12:continue
            delta=.5*(length-target)*d/length;acc[a]+=delta;acc[b]-=delta;counts[a]+=1;counts[b]+=1;err+=(length-target)**2
    active=counts[:,0]>0;out[active]+=acc[active]/counts[active];return out,err

def _project_fab(xy,gap_pairs,theta_min):
    out=xy.copy();accum=np.zeros_like(out);counts=np.zeros((len(out),1));energy=0.;violations=0;angles=[]
    for ca,na,cb,nb in gap_pairs:
        va=out[na]-out[ca];vb=out[nb]-out[cb];la2=float(va@va);lb2=float(vb@vb)
        if la2<=1e-16 or lb2<=1e-16:continue
        raw=_angle(va,vb);theta=min(raw,2*math.pi-raw);angles.append(theta);target=min(max(theta,theta_min),.5*math.pi);delta=target-theta
        if abs(delta)<=1e-10:continue
        violations+=1;cross=float(va[0]*vb[1]-va[1]*vb[0]);orient=1. if cross>=0 else -1.;total=la2+lb2;qa=out[ca]+_rot(va,-orient*delta*(lb2/total));qb=out[cb]+_rot(vb,orient*delta*(la2/total));accum[na]+=qa-out[na];accum[nb]+=qb-out[nb];counts[na]+=1;counts[nb]+=1;energy+=float(np.sum((qa-out[na])**2)+np.sum((qb-out[nb])**2))
    active=counts[:,0]>0;out[active]+=accum[active]/counts[active];return out,energy,violations,(min(angles) if angles else 0.),(max(angles) if angles else 0.)

def _sat_mtv(a,b):
    best=None;depth_best=float('inf');ca,cb=np.mean(a,0),np.mean(b,0)
    for poly in (a,b):
        for i in range(len(poly)):
            e=poly[(i+1)%len(poly)]-poly[i];axis=np.asarray([-e[1],e[0]],float);n=float(np.linalg.norm(axis))
            if n<=1e-12:continue
            axis/=n;aa,bb=a@axis,b@axis;depth=min(float(max(aa)),float(max(bb)))-max(float(min(aa)),float(min(bb)))
            if depth<=0:return None
            if depth<depth_best:
                if float((cb-ca)@axis)<0:axis=-axis
                best,depth_best=axis,depth
    return None if best is None else best*depth_best

def _project_collisions(xy,faces):
    out=xy.copy();accum=np.zeros_like(out);counts=np.zeros((len(out),1));collisions=0;energy=0.;polys=[out[f] for f in faces]
    for i in range(len(faces)):
        amin,amax=np.min(polys[i],0),np.max(polys[i],0)
        for j in range(i+1,len(faces)):
            bmin,bmax=np.min(polys[j],0),np.max(polys[j],0)
            if np.any(np.minimum(amax,bmax)-np.maximum(amin,bmin)<=0):continue
            mtv=_sat_mtv(polys[i],polys[j])
            if mtv is None:continue
            collisions+=1;energy+=float(mtv@mtv)
            for v in faces[i]:accum[int(v)]-=.5*mtv;counts[int(v)]+=1
            for v in faces[j]:accum[int(v)]+=.5*mtv;counts[int(v)]+=1
    active=counts[:,0]>0;out[active]+=accum[active]/counts[active];return out,energy,collisions

def optimize_paper_eq5(mesh_2d,mesh_3d,params,*,progress_callback=None,pipeline=None):
    global _LAST_EQ5_HISTORY
    start=time.perf_counter();xy0,faces,source_ids,hinge_groups,gap_pairs=_build_auxetic_topology(mesh_2d);xy=xy0.copy();centroid0=np.mean(xy0,0,keepdims=True)
    w_edge=_env_float('ONESTRING_EQ5_W_EDGE',getattr(params,'w_edge',1.));w_collision=_env_float('ONESTRING_EQ5_W_COLLISION',getattr(params,'w_collision',1.));w_fab=_env_float('ONESTRING_EQ5_W_FAB',getattr(params,'w_fab',.001));theta_deg=_env_float('ONESTRING_EQ5_THETA_MIN_DEG',5.);theta_min=float(np.clip(math.radians(theta_deg),0,.5*math.pi));iterations=max(1,_env_int('ONESTRING_EQ5_ITERATIONS',max(80,int(getattr(params,'max_2d_iterations',40))*6)))
    history=[];_LAST_EQ5_HISTORY={'faces':faces.copy(),'snapshots':history}
    print(f'[PAPER-EQ5-SETTINGS] w_edge={w_edge:g} w_collision={w_collision:g} w_fab={w_fab:g} theta_min_deg={theta_deg:g} iterations={iterations}')
    print(f'[PAPER-EQ5-AUXETIC] source_vertices={len(mesh_2d.vertices)} independent_vertices={len(xy)} tiles={len(faces)} hinge_groups={len(hinge_groups)} physical_gaps={len(gap_pairs)}')
    final_collisions=final_fab_violations=0;amin=amax=0.
    for it in range(iterations):
        old=xy.copy();xy=_project_hinges(xy,hinge_groups);edge_candidate,_=_project_edges(xy,faces,source_ids,mesh_3d);collision_candidate,_,final_collisions=_project_collisions(xy,faces);fab_candidate,_,final_fab_violations,amin,amax=_project_fab(xy,gap_pairs,theta_min);total=max(w_edge+w_collision+w_fab,1e-12);xy=(w_edge*edge_candidate+w_collision*collision_candidate+w_fab*fab_candidate)/total;xy=_project_hinges(xy,hinge_groups);xy+=centroid0-np.mean(xy,0,keepdims=True);step=float(np.max(np.linalg.norm(xy-old,axis=1))) if len(xy) else 0.
        if (it+1)%10==0:history.append({'iteration':it+1,'xy':xy.copy(),'collisions':int(final_collisions),'fab_violations':int(final_fab_violations),'gap_min_deg':float(math.degrees(amin)),'gap_max_deg':float(math.degrees(amax)),'step':step})
        if it==0 or (it+1)%max(1,iterations//8)==0 or it+1==iterations:print(f'[PAPER-EQ5-K2D-ITER] iter={it+1} collisions={final_collisions} fab_violations={final_fab_violations} gap_min_deg={math.degrees(amin):.4g} gap_max_deg={math.degrees(amax):.4g} step={step:.6g}')
        if progress_callback is not None and (it%max(1,iterations//30)==0 or it+1==iterations):
            try:progress_callback('Paper Eq.5 auxetic K2D',(it+1)/iterations,f'iter {it+1}/{iterations}; physical_gaps={len(gap_pairs)}; fab={final_fab_violations}')
            except Exception:pass
        if step<1e-9:break
    if not history or history[-1]['iteration']!=it+1:history.append({'iteration':it+1,'xy':xy.copy(),'collisions':int(final_collisions),'fab_violations':int(final_fab_violations),'gap_min_deg':float(math.degrees(amin)),'gap_max_deg':float(math.degrees(amax)),'step':step,'final':True})
    _,edge_energy=_project_edges(xy,faces,source_ids,mesh_3d);_,collision_energy,final_collisions=_project_collisions(xy,faces);_,fab_energy,final_fab_violations,amin,amax=_project_fab(xy,gap_pairs,theta_min);edge_mean=math.sqrt(max(0.,edge_energy)/max(1,4*len(faces)));vertices=np.column_stack([xy,np.zeros(len(xy))]);metrics=dict(getattr(mesh_2d,'metrics',{}) or {});metrics.update({'objective':'E_Flat = w1*EEdge + w2*ECollision + w3*EFab on adjacency-derived auxetic gaps','paper_eq5_unified':True,'paper_eq5_auxetic_topology':True,'paper_eq5_fab_topology_verified':False,'paper_weight_w1_edge':w_edge,'paper_weight_w2_collision':w_collision,'paper_weight_w3_fab':w_fab,'paper_fab_theta_min_rad':theta_min,'paper_fab_theta_max_rad':.5*math.pi,'paper_fab_gap_constraint_count':len(gap_pairs),'paper_fab_violation_count':final_fab_violations,'paper_fab_gap_min_deg':math.degrees(amin),'paper_fab_gap_max_deg':math.degrees(amax),'paper_fab_projection_energy':fab_energy,'paper_collision_projection_energy':collision_energy,'collision_count_after':final_collisions,'2d_collision_count':final_collisions,'edge_matching_error':edge_mean,'optimizer_iterations':it+1,'actual_backend':'numpy_auxetic_adjacency_projective_eq5','auxetic_source_vertex_ids':source_ids.tolist(),'auxetic_source_face_count':len(mesh_2d.faces),'eq5_history_snapshot_count':len(history)})
    out=type(mesh_2d)(vertices,faces.copy(),mesh_2d.grid,'K2D',metrics,list(getattr(mesh_2d,'split_lines',[])));report_type=getattr(pipeline,'StageReport',None) if pipeline is not None else None
    if report_type is None and pipeline is not None:report_type=getattr(getattr(pipeline,'_original',None),'StageReport',None)
    if report_type is None:raise RuntimeError('Paper Eq.5 auxetic solver could not resolve StageReport type')
    report=report_type(name='M2D -> auxetic K2D',objective=metrics['objective'],before_error=0.,after_error=edge_mean,constraint_violation=float(final_collisions+final_fab_violations),computation_time=time.perf_counter()-start,counts={'vertices':len(vertices),'quads':len(faces),'physical_gaps':len(gap_pairs)})
    print(f'[PAPER-EQ5-K2D] auxetic=True physical_gaps={len(gap_pairs)} collisions={final_collisions} fab_violations={final_fab_violations} gap_min_deg={math.degrees(amin):.4g} gap_max_deg={math.degrees(amax):.4g} edge_mean={edge_mean:.6g} history_snapshots={len(history)}')
    return out,report

__all__=['optimize_paper_eq5','get_last_eq5_history']
