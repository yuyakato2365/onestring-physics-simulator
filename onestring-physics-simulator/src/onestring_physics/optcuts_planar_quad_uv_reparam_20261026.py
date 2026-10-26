"""Safe UV disk reparameterization before planar all-quad meshing.

A bounded boundary-smoothing displacement is harmonically extended through the
ORIGINAL, paired OptCuts UV triangles. The paired 3D triangle mesh is untouched.
Every line-search trial is rejected unless the new UV chart stays locally
unflipped, its boundary is simple and its true 3D->UV symmetric Dirichlet
energy remains within a small relative budget.

This is an original small-deformation add-on, not Zhang et al. (2021) or SLIM.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import Polygon


class UVReparameterizationError(RuntimeError):
    pass


def boundary_cycle(faces: np.ndarray) -> np.ndarray:
    """Find the single oriented boundary walk from triangle incidence."""
    f = np.asarray(faces, int)
    edge_count = {}
    directed = {}
    for a,b,c in f:
        for u,v in ((a,b),(b,c),(c,a)):
            key = tuple(sorted((int(u),int(v))))
            edge_count[key] = edge_count.get(key,0)+1
            directed[key] = (int(u),int(v))
    if any(count not in (1,2) for count in edge_count.values()):
        raise UVReparameterizationError("UV_REPARAM_NONMANIFOLD_EDGE")
    edge = [directed[e] for e,count in edge_count.items() if count == 1]
    if len(edge) < 3:
        raise UVReparameterizationError("UV_REPARAM_NOT_DISK")
    next_vertex = {}
    for a,b in edge:
        if a in next_vertex:
            raise UVReparameterizationError("UV_REPARAM_BRANCHING_BOUNDARY")
        next_vertex[a] = b
    first = edge[0][0]
    out = [first]
    p = first
    for _ in range(len(edge)-1):
        p = next_vertex.get(p, -1)
        if p < 0 or p in out:
            raise UVReparameterizationError("UV_REPARAM_BOUNDARY_NOT_SINGLE_LOOP")
        out.append(p)
    if next_vertex.get(out[-1]) != first:
        raise UVReparameterizationError("UV_REPARAM_BOUNDARY_NOT_CLOSED")
    if len(next_vertex) != len(out):
        raise UVReparameterizationError("UV_REPARAM_MULTIPLE_BOUNDARIES")
    return np.asarray(out,int)


def surface_uv_sd(xyz: np.ndarray, faces3d: np.ndarray,
                  uv: np.ndarray, faces2d: np.ndarray) -> float:
    """Area-weighted 3D->UV symmetric Dirichlet energy; true metric units."""
    xyz_tri = np.asarray(xyz,float)[np.asarray(faces3d,int)]
    uv_tri = np.asarray(uv,float)[np.asarray(faces2d,int)]
    if len(xyz_tri) != len(uv_tri):
        raise UVReparameterizationError("UV_REPARAM_TRIANGLE_PAIRING_MISSING")
    a,b = xyz_tri[:,1]-xyz_tri[:,0],xyz_tri[:,2]-xyz_tri[:,0]
    la = np.linalg.norm(a,axis=1)
    cross = np.linalg.norm(np.cross(a,b),axis=1)
    if np.any(cross <= 1e-15) or np.any(la <= 1e-15):
        raise UVReparameterizationError("UV_REPARAM_SOURCE_DEGENERATE")
    lx = np.einsum("ij,ij->i",a,b)/la
    ly = cross/la
    bu,bv = uv_tri[:,1]-uv_tri[:,0],uv_tri[:,2]-uv_tri[:,0]
    # UV = J * local 3D planar triangle
    x = bu/la[:,None]
    y = (bv - x*lx[:,None])/ly[:,None]
    det = x[:,0]*y[:,1]-x[:,1]*y[:,0]
    if np.any(np.abs(det) < 1e-14):
        return float("inf")
    norm2 = np.sum(x*x+y*y,axis=1)
    energies = norm2*(1.+1./det**2)
    weights = .5*cross
    return float(np.dot(weights,energies)/np.sum(weights))


def harmonically_smoothed_uv(
    xyz: np.ndarray, faces3d: np.ndarray, uv: np.ndarray, faces2d: np.ndarray,
    *, intensity: float = .12, max_sd_relative: float = 1.035,
) -> tuple[np.ndarray, dict[str,float|bool]]:
    """Return modified UV vertices only after topology and SD validity checks."""
    f = np.asarray(faces2d,int)
    uv = np.asarray(uv,float)
    bd = boundary_cycle(f)
    poly = Polygon(uv[bd])
    if not poly.is_valid or poly.area <= 0:
        raise UVReparameterizationError("UV_REPARAM_SOURCE_BOUNDARY_INVALID")
    signed = np.cross(uv[f[:,1]]-uv[f[:,0]],uv[f[:,2]]-uv[f[:,0]])
    if np.any(signed*signed[0] <= 1e-15):
        raise UVReparameterizationError("UV_REPARAM_SOURCE_TRIANGLES_FLIPPED")
    sd0 = surface_uv_sd(xyz,faces3d,uv,f)
    left = np.roll(uv[bd],1,axis=0)
    right = np.roll(uv[bd],-1,axis=0)
    middle = .5*(left+right)
    # Corner-like turns remain pinned to preserve prominent features.
    di = uv[bd]-left
    do = right-uv[bd]
    dnorm = np.maximum(np.linalg.norm(di,axis=1)*np.linalg.norm(do,axis=1),1e-15)
    turn_angle = np.arccos(np.clip(np.einsum("ij,ij->i",di,do)/dnorm,-1.,1.))
    corner_weight = np.clip(1. - turn_angle/np.deg2rad(52.),0.,1.)**2
    delta_bd = np.clip(intensity,0.,.4)*corner_weight[:,None]*(middle-uv[bd])
    # Construct uniform-Laplacian displacement with immutable boundary IDs.
    nn = len(uv)
    bdset = set(bd.tolist())
    free = np.asarray([i for i in range(nn) if i not in bdset],int)
    delta = np.zeros_like(uv)
    delta[bd] = delta_bd
    if len(free):
        adjacent = [set() for _ in range(nn)]
        for a,b,c in f:
            adjacent[a].update((int(b),int(c)))
            adjacent[b].update((int(a),int(c)))
            adjacent[c].update((int(a),int(b)))
        ids = {int(v):i for i,v in enumerate(free)}
        rows,cols,data=[],[],[]
        rhs = np.zeros((len(free),2),float)
        for i,v in enumerate(free):
            adj = adjacent[int(v)]
            rows.append(i);cols.append(i);data.append(float(len(adj)))
            for j in adj:
                if j in ids:
                    rows.append(i);cols.append(ids[j]);data.append(-1.)
                else:
                    rhs[i]+=delta[j]
        mat=csr_matrix((data,(rows,cols)),shape=(len(free),len(free)))
        for axis in (0,1):
            delta[free,axis]=spsolve(mat,rhs[:,axis])
    scale=max(float(np.max(np.ptp(uv,axis=0))),1e-12)
    alpha=1.
    for _ in range(14):
        trial=uv+alpha*delta
        current=np.cross(trial[f[:,1]]-trial[f[:,0]],
                         trial[f[:,2]]-trial[f[:,0]])
        if np.any(current*signed[0] <= 1e-12*scale*scale):
            alpha*=.5
            continue
        boundary_polygon=Polygon(trial[bd])
        if not boundary_polygon.is_valid or boundary_polygon.area<=0:
            alpha*=.5
            continue
        sdt=surface_uv_sd(xyz,faces3d,trial,f)
        if np.isfinite(sdt) and sdt<=sd0*max_sd_relative:
            return trial,{
                "planar_quad_uv_boundary_reparameterized":bool(np.linalg.norm(alpha*delta)>1e-12),
                "planar_quad_uv_reparam_sd_before":float(sd0),
                "planar_quad_uv_reparam_sd_after":float(sdt),
                "planar_quad_uv_reparam_step":float(alpha),
                "planar_quad_uv_reparam_max_displacement":float(np.max(np.linalg.norm(alpha*delta,axis=1))),
            }
        alpha*=.5
    # Explicitly accept unchanged, geometrically valid source rather than
    # accepting a distorted candidate under a fake success label.
    return uv.copy(),{
        "planar_quad_uv_boundary_reparameterized":False,
        "planar_quad_uv_reparam_sd_before":float(sd0),
        "planar_quad_uv_reparam_sd_after":float(sd0),
        "planar_quad_uv_reparam_step":0.,
        "planar_quad_uv_reparam_max_displacement":0.,
    }
