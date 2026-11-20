"""Improve physical rim panels after surface relaxation, preserving face connectivity."""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize


def adjust_rim(vertices, faces, max_fraction, surface=None, log=print):
    edges, counts = np.unique(
        np.sort(np.stack((faces, np.roll(faces, 1, axis=1)), axis=2).reshape(-1, 2), axis=1),
        axis=0, return_counts=True)
    boundary = edges[counts == 1]
    rim = np.unique(boundary)
    if not len(rim):
        return vertices, dict(status="closed", remaining=0)
    pitch = np.median(np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1))
    x0 = vertices / pitch
    triangles = x0[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    rim_mask = np.isin(faces, rim).any(1)
    minimum_area = .5*np.median(areas[~rim_mask] if (~rim_mask).any() else areas)
    normal_sum = np.zeros_like(x0)
    for column in faces.T:
        np.add.at(normal_sum, column, normals)
    def unit(v):
        return v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)
    directed = np.concatenate((boundary, boundary[:, ::-1]))
    directed = directed[np.argsort(directed[:, 0])]
    if np.any(np.unique(directed[:, 0], return_counts=True)[1] != 2):
        raise ValueError("開口縁が単純な閉曲線になっていません。")
    adjacent = directed[:, 1].reshape(-1, 2)
    tangent = unit(x0[adjacent[:, 1]] - x0[adjacent[:, 0]])
    side = unit(np.cross(normal_sum[rim], tangent))
    basis = np.stack((tangent, side), axis=1)
    local_faces = faces[rim_mask]
    local_edges = np.searchsorted(rim, boundary)
    inner = np.setdiff1d(np.unique(local_faces),rim) if surface is not None else np.array([],int)
    movable = np.r_[rim,inner]
    affected = np.isin(faces, movable).any(1)
    if len(inner):
        import igl
        fine, fine_faces = surface
        fine = np.asarray(fine)/pitch
        _, initial_face, _ = igl.point_mesh_squared_distance(x0[inner], fine, fine_faces)
        adjacency = igl.triangle_triangle_adjacency(fine_faces)[0]
        fine_normals = igl.per_face_normals(fine, fine_faces, np.array([0.,0.,1.]))
        patches = []
        for face in initial_face:
            patch = np.array([face])
            for _ in range(3):
                neighbors = adjacency[patch].ravel()
                patch = np.union1d(patch, neighbors[neighbors>=0])
            patches.append(patch[(fine_normals[patch]@fine_normals[face])>.5])
        size = max(map(len,patches))
        patch_ids = np.stack([np.pad(row,(0,size-len(row)), mode='edge') for row in patches])
        patch = jnp.asarray(fine[fine_faces[patch_ids]])
        fn = fine_normals[initial_face]
        axis = unit(np.cross(fn,np.eye(3)[np.argmin(abs(fn),axis=1)]))
        basis = np.concatenate((basis, np.stack((axis,np.cross(fn,axis)),axis=1)))
    def pose(z):
        offset = jnp.einsum('ij,ijk->ik', z.reshape(-1, 2), basis)
        x = jnp.asarray(x0).at[rim].add(offset[:len(rim)])
        if len(inner):
            # Select the nearest point on a connected, same-side fine-surface patch.
            q = (x0[inner]+offset[len(rim):])[:,None,:]
            a,b,c = patch[:,:,0],patch[:,:,1],patch[:,:,2]
            n = jnp.cross(b-a,c-a)
            projected = q-n*(jnp.sum((q-a)*n,axis=-1)/jnp.maximum(jnp.sum(n*n,axis=-1),1e-20))[:,:,None]
            candidates = [projected]
            inside = jnp.ones(projected.shape[:2],dtype=bool)
            for left,right in ((a,b),(b,c),(c,a)):
                edge = right-left
                inside &= jnp.sum(jnp.cross(edge,projected-left)*n,axis=-1)>=0
                t = jnp.clip(jnp.sum((q-left)*edge,axis=-1)/jnp.maximum(jnp.sum(edge*edge,axis=-1),1e-20),0.,1.)
                candidates.append(left+t[:,:,None]*edge)
            candidates = jnp.stack(candidates,axis=2)
            distances = jnp.sum((candidates-q[:,:,None,:])**2,axis=-1)
            distances = distances.at[:,:,0].set(jnp.where(inside,distances[:,:,0],jnp.inf))
            idx = distances.reshape(len(inner),-1).argmin(1)
            x = x.at[inner].set(candidates.reshape(len(inner),-1,3)[jnp.arange(len(inner)),idx])
        return x, offset
    def loss(z, weight):
        x, offset = pose(z)
        t = x[faces[affected]]
        a, b = t-jnp.roll(t, 1, axis=1), t-jnp.roll(t, -1, axis=1)
        lengths = jnp.linalg.norm(a, axis=2)
        cosine = jnp.sum(a*b, axis=2)/(lengths*jnp.roll(lengths, -1, axis=1)+1e-15)
        # Aim slightly inside the requested interval to absorb numerical tolerance.
        violation = jnp.maximum(cosine-np.cos(np.deg2rad(40.2)), 0.) + jnp.maximum(np.cos(np.deg2rad(79.8))-cosine, 0.)
        n = jnp.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        area = jnp.linalg.norm(n, axis=1)
        turn = jnp.sum(n*unit(normals[affected]), axis=1)
        area_floor = jnp.minimum(areas[affected], minimum_area)
        smooth = offset[local_edges[:, 0]]-offset[local_edges[:, 1]]
        return (weight*jnp.sum(violation**2*jnp.where(rim_mask[affected],1.,.03)[:,None]) + 20*jnp.sum(jnp.maximum(0.,area_floor-area)**2)
                + 10000*jnp.sum(jnp.maximum(0., .2*areas[affected]-turn)**2)
                + .03*jnp.sum((cosine-.5)**2) + .08*jnp.sum(offset**2) + .08*jnp.sum(smooth**2))
    gradient = jax.jit(jax.value_and_grad(loss))
    def objective(z, weight):
        value, grad = gradient(z, weight)
        return float(value), np.asarray(grad, dtype=float)
    def measure(x):
        t = x[local_faces]
        cosine = np.sum(unit(t-np.roll(t,1,axis=1))*unit(t-np.roll(t,-1,axis=1)), axis=2)
        angles = np.degrees(np.arccos(np.clip(cosine,-1,1)))
        return int(((angles.min(1)<40)|(angles.max(1)>80)).sum()), float(angles.min()), float(angles.max())
    z = np.zeros(2*len(movable))
    best = x0
    initial = measure(x0)
    remaining, low, high = initial
    key = (remaining,max(40-low,high-80,0.),0.)
    from igl.copyleft.cgal import remesh_self_intersections
    def intersections(x):
        pairs = remesh_self_intersections(x, faces, detect_only=True)[2]
        return set(map(tuple, np.sort(pairs, axis=1)))
    original_crossings = intersections(x0)
    steps = []
    for limit in sorted(set(min(max_fraction,v) for v in [.115,.2,.35,.5,.75,1.,max_fraction])):
        if remaining == 0: break
        # Component bounds guarantee the Euclidean displacement is within the selected radius.
        for weight in (30., 3000.):
            result = minimize(objective,z,args=(weight,),jac=True,method='L-BFGS-B',
                              bounds=[(-limit/np.sqrt(2),limit/np.sqrt(2))]*(2*len(rim)) + [(-min(.4,limit)/2,min(.4,limit)/2)]*(2*len(inner)),
                              options=dict(maxiter=350,ftol=1e-12))
            z = result.x
            trial = np.asarray(pose(z)[0])
            t = trial[faces[affected]]
            n = np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
            if not np.isfinite(trial).all() or np.any(np.sum(n*normals[affected],axis=1)<=0):
                continue
            count, amin, amax = measure(trial)
            move = float(np.linalg.norm(trial[rim]-x0[rim],axis=1).max())
            score = (count,max(40-amin,amax-80,0.),move)
            if score < key and not (intersections(trial) - original_crossings):
                best,key,remaining,low,high=trial,score,count,amin,amax
            if remaining == 0: break
        steps.append(dict(limit=limit,remaining=remaining))
        log(f'[RIM] limit {limit:.1%}, outside 40–80°: {remaining}/{len(local_faces)}')
    return best*pitch, dict(status="finished",panels=len(local_faces),before=initial[0],remaining=remaining,
                            min_angle=low,max_angle=high,max_move_over_pitch=key[2],steps=steps)
