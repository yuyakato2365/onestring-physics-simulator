"""Rolling guidance on the face-adjacency graph; screened normal integration."""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
import trimesh


def converge_guide(spatial, normals, tolerance=1e-5, *, s45=False):
    # Every row needs the original face normals and the guides in its neighborhood.
    # Reuse the CSR structure; only the bilateral weights change between updates.
    spatial = spatial.tocsr()
    all_faces = np.arange(len(normals))
    guide = np.zeros_like(normals)

    start = 1
    if s45:
        # Retain the arithmetic order and the final accepted update of the S45 run.
        neighborhood = spatial.tocoo()
        for iteration in range(1, 201):
            delta = guide[neighborhood.row] - guide[neighborhood.col]
            weights = neighborhood.data * np.exp(-np.sum(delta**2, axis=1) / .18)
            smooth = sp.csr_matrix((weights, (neighborhood.row, neighborhood.col)), shape=spatial.shape) @ normals
            smooth /= np.maximum(np.linalg.norm(smooth, axis=1)[:, None], 1e-15)
            change = np.max(np.linalg.norm(smooth-guide, axis=1))
            guide = smooth
            if change < tolerance:
                return guide, iteration, float(change)
        # A slower input can still use the extended convergence loop.
        start = 201

    def prepare(faces):
        rows = spatial if faces is all_faces else spatial[faces]
        source = np.repeat(faces, np.diff(rows.indptr))
        return rows, rows.copy(), source

    faces = all_faces
    rows, weighted, source = prepare(faces)
    for iteration in range(start, 2001):
        dist2 = np.zeros_like(rows.data)
        for axis in guide.T:
            diff = axis[source] - axis[rows.indices]
            dist2 += diff * diff
        weighted.data[:] = rows.data * np.exp(-dist2 / .18)
        proposal = weighted @ normals
        length = np.linalg.norm(proposal, axis=1)
        if not np.all(np.isfinite(length)) or np.any(length == 0):
            raise ValueError('Rolling Guidanceで有効な面法線を計算できません。')
        proposal /= length[:, None]
        change = np.linalg.norm(proposal - guide[faces], axis=1)
        pending = change >= tolerance
        full_check = faces is all_faces
        if full_check and (iteration == 1 or iteration % 25 == 0 or iteration >= 200 or not pending.any()):
            print(json.dumps(dict(guide_update=iteration, remaining_faces=int(pending.sum()),
                                  max_guide_change=float(change.max()))), flush=True)
        if full_check and not pending.any():
            # Return the guide whose fixed-point residual was measured on every face.
            return guide, iteration, float(change.max())
        guide[faces] = proposal
        if full_check and iteration >= 200 and 0 < pending.sum() < len(normals) // 2:
            faces = np.flatnonzero(pending)
            rows, weighted, source = prepare(faces)
        elif not full_check and (not pending.any() or iteration % 16 == 0):
            # A moving face can disturb an apparently settled neighbor.
            faces = all_faces
            rows, weighted, source = prepare(faces)
    raise ValueError('Rolling Guidanceが2000回で収束しませんでした。全体の残差は工程ログを確認してください。')


def filter_surface(mesh):
    f, x = mesh.faces, mesh.vertices
    normals, areas = mesh.face_normals, mesh.area_faces
    centers = mesh.triangles_center
    a, b = mesh.face_adjacency.T
    graph = sp.csr_matrix((np.linalg.norm(centers[a]-centers[b], axis=1), (a,b)), shape=(len(f),len(f)))
    graph = graph + graph.T
    sigma = 3*mesh.edges_unique_length.mean()
    blocks = []
    for start in range(0,len(f),128):
        distances = dijkstra(graph, indices=np.arange(start,min(start+128,len(f))), limit=4*sigma)
        row,col = np.where(np.isfinite(distances))
        weight = areas[col]*np.exp(-.5*(distances[row,col]/sigma)**2)
        blocks.append(sp.csr_matrix((weight,(row,col)),shape=distances.shape))
    guide, iterations, change = converge_guide(sp.vstack(blocks, format='csr'), normals, s45=os.environ.get('TRI_S45_ROLLING')=='1')
    # Solve for vertex displacements. Each row is one face-edge orthogonality constraint.
    i = f.ravel(); j = np.roll(f,-1,axis=1).ravel()
    nn = np.repeat(guide,3,axis=0)
    coord = np.arange(3)
    cols = np.c_[3*i[:,None]+coord,3*j[:,None]+coord]
    A = sp.coo_matrix((np.c_[-nn,nn].ravel(),(np.repeat(np.arange(len(i)),6),cols.ravel())),
                      shape=(len(i),3*len(x))).tocsr()
    boundary = mesh.edges_unique[np.bincount(mesh.edges_unique_inverse)==1]
    free = np.repeat(~np.isin(np.arange(len(x)),boundary),3)
    B = A[:,free]
    shift = np.zeros(x.size)
    shift[free] = spsolve((B.T@B + .05*sp.eye(B.shape[1])).tocsc(), -B.T@(A@x.ravel()))
    # The unconstrained normal fit can push slender triangles across their
    # opposite edge. Backtrack only their incident vertex displacements.
    base=np.asarray(x)
    orig=mesh.triangles_cross
    area2=np.sum(orig*orig,axis=1)
    if np.any(area2<=0):
        raise ValueError('Rolling Guidanceの入力に面積ゼロの面があります。')
    weights=np.ones(len(base))
    for backtracks in range(128):
        result=base+weights[:,None]*shift.reshape(-1,3)
        triangles=result[f]
        cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        ratio=np.sum(orig*cross,axis=1)/area2
        bad=ratio<.25
        if not bad.any():break
        weights[np.unique(f[bad])]*=.5
    else:
        raise ValueError('Rolling Guidanceの面の向きを維持できません。')
    print(json.dumps(dict(iterations=iterations, max_guide_change=change, sigma=sigma,
                          position_weight=.05, limited_vertices=int((weights<1).sum()), backtracks=backtracks,
                          min_oriented_area_ratio=float(ratio.min()), max_shift=np.linalg.norm(result-base,axis=1).max())))
    return trimesh.Trimesh(result, f, process=False)


if __name__=='__main__':
    filter_surface(trimesh.load(sys.argv[1],process=False)).export(sys.argv[2],digits=17)
