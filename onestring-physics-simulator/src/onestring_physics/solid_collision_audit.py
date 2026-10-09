"""Independent 3D convex-hull SAT audit, distinct from the XY objective."""
import numpy as np
from scipy.spatial import ConvexHull


def audit_solid_collisions(vertices, tolerance=1e-8):
    tiles = np.asarray(vertices, float)
    hulls = [ConvexHull(v) for v in tiles]
    edge_vectors=[]
    for tile,hull in zip(tiles,hulls):
        edges={tuple(sorted((int(f[i]),int(f[(i+1)%3])))) for f in hull.simplices for i in range(3)}
        edge_vectors.append(np.array([tile[b]-tile[a] for a,b in edges]))
    lo,hi=tiles.min(axis=1),tiles.max(axis=1)
    pairs=[];depths=[]
    for a in range(len(tiles)):
        for b in range(a+1,len(tiles)):
            if np.any(np.minimum(hi[a],hi[b])-np.maximum(lo[a],lo[b])<=tolerance):
                continue
            cross=np.cross(edge_vectors[a][:,None,:],edge_vectors[b][None,:,:]).reshape(-1,3)
            axes=np.vstack((hulls[a].equations[:,:3],hulls[b].equations[:,:3],cross))
            lengths=np.linalg.norm(axes,axis=1);axes=axes[lengths>1e-12]/lengths[lengths>1e-12,None]
            pa,pb=tiles[a]@axes.T,tiles[b]@axes.T
            depth=float(np.minimum(pa.max(axis=0)-pb.min(axis=0),pb.max(axis=0)-pa.min(axis=0)).min())
            if depth>tolerance:
                pairs.append([a,b]);depths.append(depth)
    return dict(solid_collision_pairs=pairs,solid_collision_pair_count=len(pairs),
                solid_penetration_depths=depths,solid_penetration_max=max(depths,default=0.),
                solid_penetration_rms=float(np.sqrt(np.mean(np.square(depths)))) if depths else 0.,
                solid_collision_model='3D convex hull SAT; exact for convex tiles, conservative for nonconvex tiles')
