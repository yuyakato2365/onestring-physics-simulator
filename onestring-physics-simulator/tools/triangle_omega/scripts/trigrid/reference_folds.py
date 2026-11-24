"""Repair unsupported concave panel folds by fitting local patches to the reference surface."""
import numpy as np
import trimesh
import igl
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from igl.copyleft.cgal import remesh_self_intersections


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-14)


def normals(points, faces):
    a, b, c = points[faces].transpose(1, 0, 2)
    return unit(np.cross(b-a, c-a))


def shared_edges(faces):
    edges = {}
    for fi, tri in enumerate(faces):
        for i, j, k in zip(tri, np.roll(tri, -1), np.roll(tri, -2)):
            edges.setdefault(tuple(sorted((i,j))), []).append((i,j,k,fi))
    records = [(i,j,k,l,a,b) for pair in edges.values() if len(pair)==2
               for (i,j,k,a),(jj,ii,l,b) in [pair] if (i,j)==(ii,jj)]
    return np.asarray(records, dtype=int).reshape(-1,6)


def repair_reference_folds(vertices, faces, reference, reference_faces):
    edges = shared_edges(faces)
    report = dict(revision=1, detected_edges=0, moved_vertices=0)
    if not len(edges):
        return vertices, report
    i,j,k,l,fa,fb = edges.T
    pitch = np.median(np.linalg.norm(vertices[i]-vertices[j], axis=1))
    if pitch <= 1e-12:
        raise ValueError("M3D has zero representative panel pitch")
    origin = vertices.mean(axis=0)
    x = (vertices-origin)/pitch
    s = (reference-origin)/pitch
    fn = normals(x,faces)
    rn = normals(s,reference_faces)
    tri = s[reference_faces]
    kd = cKDTree(tri.mean(1))
    def match(q, direction):
        _, ids = kd.query(q,k=min(len(tri),32))
        ids = ids.reshape(len(q),-1)
        pts = trimesh.triangles.closest_point(tri[ids].reshape(-1,3,3),
                np.repeat(q, ids.shape[1], axis=0)).reshape(len(q),-1,3)
        cost = ((pts-q[:,None])**2).sum(2)+.35*np.maximum(.25-(rn[ids]*direction[:,None]).sum(2),0)**2
        pick = cost.argmin(1)
        return pts[np.arange(len(q)),pick], rn[ids[np.arange(len(q)),pick]]
    # Use the same orientation for both surfaces even when the imported face order differs.
    orient = np.median((fn*rn[kd.query(x[faces].mean(1))[1]]).sum(1))
    if orient < 0:
        rn = -rn
    orient_sign = 1 if np.sum(x[faces[:,0]]*np.cross(x[faces[:,1]],x[faces[:,2]])) >= 0 else -1
    def angle(na,nb,e):
        return orient_sign*np.arctan2(-(np.cross(na,nb)*unit(e)).sum(1),(na*nb).sum(1))
    def folds(y):
        e = y[j]-y[i]
        a = np.cross(e,y[k]-y[i])
        b = np.cross(y[l]-y[i],e)
        le = np.linalg.norm(e,axis=1)
        gk = unit(a)*(le/np.maximum(np.linalg.norm(a,axis=1),1e-12))[:,None]
        gl = unit(b)*(le/np.maximum(np.linalg.norm(b,axis=1),1e-12))[:,None]
        u = ((y[k]-y[i])*e).sum(1)/np.maximum(le**2,1e-14)
        v = ((y[l]-y[i])*e).sum(1)/np.maximum(le**2,1e-14)
        jac = np.stack([-(1-u[:,None])*gk-(1-v[:,None])*gl,-u[:,None]*gk-v[:,None]*gl,gk,gl],1)
        return angle(unit(a),unit(b),e),orient_sign*jac
    before,_ = folds(x)
    def compare_to_reference(y, angles, panel_normals):
        suspect = angles > np.deg2rad(43)
        ref = np.zeros(len(i))
        if suspect.any():
            edge_mid = (y[i[suspect]] + y[j[suspect]]) / 2
            expected = []
            for t in (.28, .5, .72):
                _, a = match((1-t)*edge_mid + t*y[k[suspect]], panel_normals[fa[suspect]])
                _, b = match((1-t)*edge_mid + t*y[l[suspect]], panel_normals[fb[suspect]])
                expected.append(angle(a, b, y[j[suspect]] - y[i[suspect]]))
            ref[suspect] = np.median(expected, axis=0)
        return ref, suspect & (angles-ref > np.deg2rad(42))

    ref, bad = compare_to_reference(x, before, fn)
    report['detected_edges']=int(bad.sum())
    if not bad.any():
        return vertices,report
    quad=edges[:,:4]; ed=igl.edges(faces)
    graph=coo_matrix((np.ones(len(ed)),ed.T),shape=(len(x),len(x)))
    dist=dijkstra(graph,directed=False,indices=np.unique(quad[bad]),min_only=True,unweighted=True)
    free=dist<=2
    rim=np.unique(igl.boundary_facets(faces)[0])
    free[rim]=False
    cap=np.where(bad,np.maximum(ref+np.deg2rad(3),np.deg2rad(2)),np.maximum(before,0)+np.deg2rad(4))
    floor=np.minimum(igl.internal_angles(x,faces).min(1),np.deg2rad(30))
    area=igl.doublearea(x,faces)
    start_error=float(np.sqrt(igl.point_mesh_squared_distance(s,x,faces)[0].max()))
    y=x.copy()
    # Fit apparent contours sampled on a hemisphere, independent of the display camera.
    # Restrict samples and updates to the detected neighborhoods. This term prevents
    # flattening a notch simply by pulling both of its shoulders below the source.
    source_edges = shared_edges(reference_faces)
    panel_tree = cKDTree(x[free]) if free.any() else None
    contours = []
    direction_count = 48
    smooth_normals=igl.per_vertex_normals(s,reference_faces)
    for index in range(direction_count):
        h = (index+.5)/direction_count
        phi = index*np.pi*(3-np.sqrt(5))
        view = np.array([np.sqrt(1-h*h)*np.cos(phi),h,np.sqrt(1-h*h)*np.sin(phi)])
        front = rn@view
        se = source_edges[front[source_edges[:,4]]*front[source_edges[:,5]]<0,:2]
        pts = s[se].mean(1)
        near = panel_tree.query(pts)[0]<2 if panel_tree is not None else np.zeros(len(pts),bool)
        se,pts = se[near],pts[near]
        sn = unit(smooth_normals[se].mean(1)) if len(se) else np.empty((0,3))
        sn = unit(sn-(sn@view)[:,None]*view)
        contours.append((view,pts,sn,np.linalg.norm(s[se[:,1]]-s[se[:,0]],axis=1)))
    def contour_system(y):
        rows=[];cols=[];vals=[];rhs=[];offset=0
        normals_y=normals(y,faces)
        for view,pts,sn,weights in contours:
            front=normals_y@view
            se=edges[(front[fa]*front[fb]<0)&(free[i]|free[j]),:2]
            if not len(pts) or not len(se):
                continue
            tr=np.eye(3)-.84*np.outer(view,view)
            ends=y[se]@tr; points=pts@tr
            ids=cKDTree(ends.mean(1)).query(points,k=min(8,len(se)))[1].reshape(len(pts),-1)
            a=ends[ids,0];delta=ends[ids,1]-a
            t=np.clip(((points[:,None]-a)*delta).sum(2)/np.maximum((delta*delta).sum(2),1e-12),0,1)
            err=((a+t[:,:,None]*delta-points[:,None])**2).sum(2)
            b=err.argmin(1); idx=np.arange(len(b));use=err[idx,b]<.65
            sel=se[ids[idx,b][use]];t=t[idx,b][use];w=np.sqrt(weights[use]);n=sn[use]*w[:,None]
            jac=np.stack([(1-t[:,None])*n,t[:,None]*n],1);count=len(t)
            rows.extend(np.repeat(np.arange(offset,offset+count),6));cols.extend((3*sel[:,:,None]+np.arange(3)).ravel());vals.extend(jac.ravel())
            rhs.extend((n*pts[use]).sum(1));offset+=count
        B=coo_matrix((vals,(rows,cols)),shape=(offset,3*len(x))).tocsr()
        scale=free.sum()/max(float(B.multiply(B).sum()),1e-8)
        return (B.T@B)*scale,(B.T@np.asarray(rhs))*scale
    if free.any():
        for _ in range(10):
            C,target=contour_system(y)
            q,normal=match(y,igl.per_vertex_normals(y,faces))
            def energy(flat):
                z=x.copy();z[free]=flat.reshape(-1,3);d=z-x
                change=d[ed[:,0]]-d[ed[:,1]]
                h=((z-q)*normal).sum(1)
                pos=z.ravel();cz=C@pos
                val=75*np.dot(pos,cz-2*target)+(d*d).sum()+.4*(change**2).sum()+7*(h*h).sum()
                g=150*(cz-target).reshape(-1,3)+2*d+14*h[:,None]*normal
                np.add.at(g,ed[:,0],.8*change);np.add.at(g,ed[:,1],-.8*change)
                theta,dtheta=folds(z)
                ex=np.maximum(theta-cap,0)
                val+=28*(ex@ex);np.add.at(g,quad,(56*ex)[:,None,None]*dtheta)
                p=z[faces]
                u0,w0=p[:,1]-p[:,0],p[:,2]-p[:,0]
                cross=np.cross(u0,w0); face_n=unit(cross)
                ratio=np.linalg.norm(cross,axis=1)/area
                ex_area=np.minimum(ratio-.74,0)+np.maximum(ratio-1.30,0)
                val+=1500*(ex_area@ex_area)
                gu=np.cross(w0,face_n);gv=np.cross(face_n,u0)
                gf=(3000*ex_area/area)[:,None,None]*np.stack([-gu-gv,gu,gv],1)
                for a,b,c in [(0,1,2),(1,2,0),(2,0,1)]:
                    v=p[:,b]-p[:,a];w=p[:,c]-p[:,a];u=unit(v);n=unit(w);cs=(u*n).sum(1)
                    low=np.maximum(cs-np.cos(floor),0)
                    val+=12000*(low@low)
                    gu=(24000*low/np.maximum(np.linalg.norm(v,axis=1),1e-12))[:,None]*(n-cs[:,None]*u)
                    gv=(24000*low/np.maximum(np.linalg.norm(w,axis=1),1e-12))[:,None]*(u-cs[:,None]*n)
                    gf[:,a]-=gu+gv;gf[:,b]+=gu;gf[:,c]+=gv
                np.add.at(g,faces,gf)
                return val,g[free].ravel()
            y[free]=minimize(energy,y[free].ravel(),jac=True,method='L-BFGS-B',
                            options=dict(maxiter=180,ftol=1e-11)).x.reshape(-1,3)
    for fraction in (1.,.75,.5,.25):
        z=x+fraction*(y-x); ang,_=folds(z)
        if (ang[bad].max()>=before[bad].max()-1e-4 or
            np.min((normals(z,faces)*fn).sum(1))<=0 or
            np.min(igl.doublearea(z,faces)/area)<.65 or
            np.any(igl.internal_angles(z,faces).min(1)<floor-np.deg2rad(1)) or
            len(remesh_self_intersections(z,faces,detect_only=True)[2]) or
            np.sqrt(igl.point_mesh_squared_distance(s,z,faces)[0].max())>start_error+.035):
            continue
        _, remaining = compare_to_reference(z, ang, normals(z, faces))
        if (remaining & ~bad).any():
            continue
        out=vertices.copy();out[free]=origin+pitch*z[free]
        report.update(moved_vertices=int(np.sum(np.linalg.norm(z-x,axis=1)>1e-8)),
                      max_before_deg=float(np.degrees(before[bad].max())),max_after_deg=float(np.degrees(ang[bad].max())),
                      remaining_edges=int(remaining.sum()))
        return out,report
    report.update(remaining_edges=int(bad.sum()),status='no_safe_improvement')
    return vertices,report
