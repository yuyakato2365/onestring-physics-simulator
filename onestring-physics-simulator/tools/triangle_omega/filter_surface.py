"""Rolling guidance normal filtering and gradient-domain reconstruction.

Uses the local Bunny trial's five-pass normal filter and FEM reconstruction.
Only face centers within four spatial standard deviations contribute.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
import trimesh


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)


def rotations(a, b):
    c = np.sum(a*b, axis=1)
    cross = np.cross(a,b)
    k = np.cross(np.eye(3)[None, :, :], cross[:, None, :])
    r = np.eye(3) + k + k@k / np.maximum(1+c[:,None,None], 1e-12)
    for i in np.flatnonzero(c < -1+1e-9):
        axis = normalize(np.cross(a[i], np.eye(3)[np.argmin(abs(a[i]))]))
        r[i] = 2*np.outer(axis,axis) - np.eye(3)
    return r


def filter_mesh(mesh):
    f, x = mesh.faces, mesh.vertices.copy()
    size = np.ptp(x,axis=0).max()
    origin = x.mean(0)
    x = (x-origin)/size
    mesh = trimesh.Trimesh(x,f,process=False)
    normals, areas = mesh.face_normals, mesh.area_faces
    centers = mesh.triangles_center
    if np.any(areas <= 1e-15):
        raise ValueError("Degenerate input triangle")
    sigma = 3*mesh.edges_unique_length.mean()
    distances = cKDTree(centers).sparse_distance_matrix(cKDTree(centers),4*sigma,output_type='coo_matrix')
    i,j = distances.row, distances.col
    spatial = areas[j]*np.exp(-0.5*(distances.data/sigma)**2)
    guide = np.zeros_like(normals)
    for _ in range(5):
        weights = spatial*np.exp(-np.sum((guide[i]-guide[j])**2,axis=1)/(2*.3**2))
        guide = normalize(sp.csr_matrix((weights,(i,j)),shape=distances.shape)@normals)
    tri = x[f]
    grad = np.stack([np.cross(normals,tri[:,(k+2)%3]-tri[:,(k+1)%3]) for k in range(3)],axis=2)/(2*areas[:,None,None])
    nf,nv = len(f),len(x)
    g = sp.coo_matrix((grad.ravel(),(np.repeat(np.arange(3*nf),3),np.tile(f[:,None,:],(1,3,1)).ravel())),shape=(3*nf,nv)).tocsr()
    a,b = mesh.face_adjacency.T
    rotated = rotations(normals[b],normals[a])@grad[b]
    cols = np.tile(np.column_stack([f[a],f[b]])[:,None,:],(1,3,1)).ravel()
    h = sp.coo_matrix((np.concatenate([grad[a],-rotated],axis=2).ravel(),(np.repeat(np.arange(3*len(a)),6),cols)),shape=(3*len(a),nv)).tocsr()
    wa = np.repeat(areas/areas.mean(),3)
    edge = mesh.face_adjacency_edges
    we = np.repeat(np.linalg.norm(x[edge[:,0]]-x[edge[:,1]],axis=1)/mesh.edges_unique_length.mean(),3)
    matrix = (g.T@sp.diags(wa)@g + .5*h.T@sp.diags(we)@h).tocsc()
    target_grad = (g@x).reshape(nf,3,3)@rotations(normals,guide).transpose(0,2,1)
    rhs = g.T@(wa[:,None]*target_grad.reshape(-1,3))
    boundary = np.unique(mesh.edges_unique[np.bincount(mesh.edges_unique_inverse)==1])
    fixed = boundary if len(boundary) else np.array([0])
    free = np.setdiff1d(np.arange(nv),fixed)
    y = x.copy()
    y[free] = splu(matrix[free][:,free].tocsc()).solve(rhs[free]-matrix[free][:,fixed]@x[fixed])
    if not np.isfinite(y).all():
        raise ValueError("Non-finite reconstruction")
    print(json.dumps(dict(passes=5,spatial_sigma_in_mean_edges=3,normal_sigma=.3,
                          fixed_vertices=len(fixed),vertices=nv,faces=nf,max_displacement=float(np.linalg.norm(y-x,axis=1).max()*size))))
    return trimesh.Trimesh(y*size+origin,f,process=False)


if __name__ == '__main__':
    result=filter_mesh(trimesh.load(sys.argv[1],force='mesh',process=False))
    result.export(sys.argv[2],digits=17)
