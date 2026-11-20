"""Reduce excessive normal spread on the reference surface with fixed connectivity."""
from itertools import combinations

import igl
import numpy as np

from lift_quality import surface_quality


def relax_fans(points, faces, source, source_faces, min_depth, lateral):
    x = points.copy()
    limit = np.cos(2 * np.arctan2(lateral, min_depth))
    fans = [[] for _ in x]
    for face_id, tri in enumerate(faces):
        for vertex in tri:
            fans[vertex].append(face_id)
    pairs = sorted({pair for fan in fans for pair in combinations(fan, 2)})
    if not pairs:
        return x, None
    a, b = np.asarray(pairs).T

    def state(v):
        t = v[faces]
        u, w = t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]
        cross = np.cross(u, w)
        area = np.linalg.norm(cross, axis=1)
        n = cross / np.maximum(area[:, None], 1e-20)
        dots = (n[a]*n[b]).sum(1)
        excess = np.maximum(limit-dots, 0.)
        quality = area / np.maximum((u*u+w*w+(u-w)**2).sum(1), 1e-20)
        return u, w, area, n, excess, quality, np.degrees(np.arccos(np.clip(dots.min(), -1, 1)))

    initial = state(x)
    if not initial[4].any():
        return x, dict(steps=0, initial_angle=initial[-1], final_angle=initial[-1])
    ed, counts = np.unique(np.sort(faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1),axis=0,return_counts=True)
    boundary = np.unique(ed[counts==1])
    pitch = np.median(np.linalg.norm(x[ed[:,0]]-x[ed[:,1]],axis=1))
    surface = surface_quality(x,faces,source)
    if not surface['valid']:
        return x, dict(steps=0, reason='invalid_input')
    ref_normals = igl.per_face_normals(source,source_faces,np.array([0.,0.,1.]))
    _, start_face, _ = igl.point_mesh_squared_distance(x,source,source_faces)
    steps = 0
    for _ in range(60):
        u,w,area,n,excess,quality,angle = state(x)
        if not excess.any():
            break
        dn = np.zeros_like(n)
        np.add.at(dn,a,-excess[:,None]**3*n[b])
        np.add.at(dn,b,-excess[:,None]**3*n[a])
        dc = (dn-(dn*n).sum(1)[:,None]*n)/area[:,None]
        du,dw = np.cross(w,dc),np.cross(dc,u)
        grad = np.zeros_like(x)
        for corners,g in zip(faces.T,(-du-dw,du,dw)):
            np.add.at(grad,corners,g)
        grad[boundary]=0.
        active = np.linalg.norm(grad,axis=1)>1e-9
        if not active.any():
            break
        direction = grad[active]/np.linalg.norm(grad[active],axis=1)[:,None]
        for step in (.08,.04,.02,.01,.005):
            trial=x.copy()
            _,idx,nearest = igl.point_mesh_squared_distance(x[active]-step*pitch*direction,source,source_faces)
            allowed = ((np.linalg.norm(nearest-points[active],axis=1)<.4*pitch)
                       & ((ref_normals[idx]*ref_normals[start_face[active]]).sum(1)>.5))
            trial[np.flatnonzero(active)[allowed]]=nearest[allowed]
            _,_,_,ns,loss,q,opening=state(trial)
            if (opening>=angle or (loss**4).sum()>=(excess**4).sum()
                or np.any(q<np.minimum(initial[5],.14)-1e-8) or np.any((ns*n).sum(1)<.8)):
                continue
            check=surface_quality(trial,faces,source)
            if (check['valid'] and check['max_distance_over_pitch']<=min(1.,surface['max_distance_over_pitch']+.05)
                and check['p95_distance_over_pitch']<=surface['p95_distance_over_pitch']+.01):
                x=trial
                steps+=1
                break
        else:
            break
    # A sweep that moves all vertices can stall at an unrelated sharp fan.
    # Try tangent moves one vertex at a time and judge only affected normals.
    local_steps = 0
    incident = [np.asarray(fan, int) for fan in fans]
    influenced = [np.flatnonzero(np.isin(a,f) | np.isin(b,f)) for f in incident]
    circles = np.arange(12)*np.pi/6
    for sweep in range(12):
        before=x.copy()
        _,_,_,normals,old_loss,_,old_opening = state(x)
        per_face=np.zeros(len(faces))
        np.maximum.at(per_face,a,old_loss)
        np.maximum.at(per_face,b,old_loss)
        score=np.zeros(len(x))
        np.maximum.at(score,faces.ravel(),np.repeat(per_face,3))
        score[boundary]=0
        for v in np.argsort(-score):
            if score[v]<=0:
                break
            f=incident[v]
            pair=influenced[v]
            n0=ref_normals[start_face[v]]
            u=np.cross(n0,np.eye(3)[np.argmin(abs(n0))])
            u/=np.linalg.norm(u)
            dirs=np.cos(circles)[:,None]*u+np.sin(circles)[:,None]*np.cross(n0,u)
            queries=(x[v]+pitch*np.array([.04,.12,.24])[:,None,None]*dirs).reshape(-1,3)
            _,idx,pts=igl.point_mesh_squared_distance(queries,source,source_faces)
            allowed=((ref_normals[idx]@n0>.5)&(np.linalg.norm(pts-points[v],axis=1)<.4*pitch))
            trials=pts[allowed]
            if not len(trials):
                continue
            t=np.broadcast_to(x[faces[f]],(len(trials),len(f),3,3)).copy()
            t[:,np.arange(len(f)),np.argmax(faces[f]==v,axis=1)]=trials[:,None]
            d1,d2=t[:,:,1]-t[:,:,0],t[:,:,2]-t[:,:,0]
            cross=np.cross(d1,d2)
            area=np.linalg.norm(cross,axis=2)
            ns=cross/np.maximum(area[:,:,None],1e-20)
            q=area/np.maximum((d1*d1+d2*d2+(d1-d2)**2).sum(2),1e-20)
            feasible=((ns*normals[f]).sum(2)>.8).all(1)&(q>=np.minimum(initial[5][f],.14)-1e-8).all(1)
            left=np.broadcast_to(normals[a[pair]],(len(trials),len(pair),3)).copy()
            right=np.broadcast_to(normals[b[pair]],left.shape).copy()
            for ids,pair_n in ((a[pair],left),(b[pair],right)):
                mask=np.isin(ids,f)
                pair_n[:,mask]=ns[:,np.searchsorted(f,ids[mask])]
            loss=np.maximum(0.,limit-(left*right).sum(2))
            cost=(loss**6).sum(1)
            local=np.maximum(0.,limit-(normals[a[pair]]*normals[b[pair]]).sum(1))
            eligible=feasible&(loss.max(1)<=local.max()+1e-10)&(cost<(local**6).sum()-1e-10)
            if eligible.any():
                j=np.argmin(np.where(eligible,cost,np.inf))
                x[v]=trials[j]
                normals[f]=ns[j]
        check=surface_quality(x,faces,source)
        if (state(x)[-1]>=old_opening-1e-7 or not check['valid']
                or check['max_distance_over_pitch']>min(1.,surface['max_distance_over_pitch']+.05)
                or check['p95_distance_over_pitch']>surface['p95_distance_over_pitch']+.01):
            x=before
            break
        local_steps+=1
    steps+=local_steps
    return x,dict(steps=steps,initial_angle=initial[-1],final_angle=state(x)[-1],
                  max_vertex_displacement=float(np.linalg.norm(x-points,axis=1).max()))
