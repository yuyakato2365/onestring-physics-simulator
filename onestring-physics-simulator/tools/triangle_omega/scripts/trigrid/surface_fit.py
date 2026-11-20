"""Redistribute projected panel vertices where coarse chords miss the reference surface."""
import igl
import numpy as np
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from lift_quality import surface_quality


def fit_surface(x, faces, source, source_faces):
    x = x.copy()
    original = x.copy()
    edges, count = np.unique(np.sort(faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    pitch = np.median(np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1))
    locked = np.unique(edges[count==1])
    fixed = np.isin(faces,locked).any(1)
    locked = np.unique(faces[fixed])
    n = len(x)
    adjacency = igl.adjacency_matrix(faces).tocsr()
    lap = diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    normals = igl.per_face_normals(source,source_faces,np.array([0.,0.,1.]))
    _,initial_face,_ = igl.point_mesh_squared_distance(x,source,source_faces)
    def sample(points):
        ds,fi,hit = igl.point_mesh_squared_distance(source,points,faces)
        return np.sqrt(ds),fi,hit
    def errors(dist):
        return np.percentile(dist,95),dist.max(), np.mean((dist/pitch)**4)
    def angular_error(points):
        side=np.roll(points[faces],-1,axis=1)-points[faces]
        side/=np.maximum(np.linalg.norm(side,axis=2,keepdims=True),1e-15)
        angle=np.degrees(np.arccos(np.clip((-side*np.roll(side,1,axis=1)).sum(2),-1,1)))
        return np.max(np.abs(angle-60),axis=1)
    limit=np.maximum(28.,angular_error(original)+1.)
    distance, fi, hit = sample(x)
    before = errors(distance)
    count_steps = 0
    for _ in range(20):
        tri = x[faces[fi]]
        bary = igl.barycentric_coordinates(hit,*tri.transpose(1,0,2))
        matrix = coo_matrix((bary.ravel(),(np.repeat(np.arange(len(hit)),3),faces[fi].ravel())),shape=(len(hit),n)).tocsr()
        weights = diags(np.clip((distance/(.12*pitch))**2,.05,16.))
        rhs = matrix.T @ (weights @ (source-hit))
        shift = spsolve((matrix.T@weights@matrix+4*eye(n)+.75*lap).tocsc(),rhs)
        shift[locked]=0
        accepted = False
        for factor in (1.,.5,.25,.125):
            proposal=x+factor*shift
            _,ids,nearest=igl.point_mesh_squared_distance(proposal,source,source_faces)
            changed=(np.sum(normals[ids]*normals[initial_face],axis=1)>.5)&(np.linalg.norm(nearest-original,axis=1)<.45*pitch)
            changed[locked]=False
            trial=x.copy();trial[changed]=nearest[changed]
            while True:
                bad=angular_error(trial)>limit+1e-8
                if not bad.any():break
                trial[np.unique(faces[bad])]=x[np.unique(faces[bad])]
            dist,which,closest=sample(trial)
            if errors(dist)[2]>=errors(distance)[2]*(1.-1e-5) or dist.max()>before[1]:continue
            if not surface_quality(trial,faces,source)['valid']:continue
            x,distance,fi,hit=trial,dist,which,closest
            count_steps+=1;accepted=True;break
        if not accepted:break
    return x,dict(steps=count_steps,p95_before=before[0],p95_after=errors(distance)[0],
                  max_before=before[1],max_after=distance.max())
