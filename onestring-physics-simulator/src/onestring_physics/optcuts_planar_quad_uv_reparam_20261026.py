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


def coarse_boundary_reparameterize(
    xyz: np.ndarray, faces3d: np.ndarray, uv: np.ndarray, faces2d: np.ndarray,
    target_faces: int, *, max_sd_relative: float = 1.08,
) -> tuple[np.ndarray, dict[str,float|int|bool]]:
    """Simplify *Omega* by moving paired UV boundary points, never deleting any.

    Unlike simplifying the Gmsh polyline alone, all source UV triangles are
    updated by harmonic interior extension. Their original 3D vertex/face
    identity remains unchanged. A successful full relocation makes adjacent
    source boundary UV points collinear; Gmsh can then use the coarse boundary
    polygon without extending outside the registered UV region.

    Only entirely bijective, SD-bounded full relocation is accepted. On
    failure returns the original UV intact and reports why.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    from shapely.geometry import Polygon
    uv = np.asarray(uv,float)
    faces2d = np.asarray(faces2d,int)
    cycle = boundary_cycle(faces2d)
    boundary = uv[cycle]
    poly = Polygon(boundary)
    if not poly.is_valid or poly.area <= 0:
        raise UVReparameterizationError("PLANAR_QUAD_COARSE_INPUT_NOT_DISK")
    n = len(cycle)
    desired = max(4, int(target_faces))
    h = float(np.sqrt(poly.area / desired))
    source_sd = surface_uv_sd(xyz,faces3d,uv,faces2d)
    scale = max(float(np.max(np.ptp(uv,axis=0))),1e-12)
    orient = np.cross(uv[faces2d[:,1]]-uv[faces2d[:,0]],
                      uv[faces2d[:,2]]-uv[faces2d[:,0]])
    if np.any(orient * orient[0] <= 1e-14 * scale * scale):
        raise UVReparameterizationError("PLANAR_QUAD_COARSE_SOURCE_FOLDOVER")
    # Try finer boundary approximations if coarser segmentation cannot produce
    # a bijective low-distortion UV map.
    for tol_fraction in (.18,.12,.07,.04,.02):
        coarse = poly.simplify(tol_fraction * h, preserve_topology=True)
        if coarse.geom_type != "Polygon" or not coarse.is_valid:
            continue
        coords = np.asarray(coarse.exterior.coords,float)[:-1,:2]
        if len(coords) < 4 or len(coords) >= n - 2:
            continue
        anchors=[]
        for p in coords:
            d=np.linalg.norm(boundary-p,axis=1)
            idx=int(np.argmin(d))
            if d[idx] > 1e-8 * scale:
                anchors=[]
                break
            anchors.append(idx)
        anchors=sorted(set(anchors))
        if len(anchors)<4:
            continue
        updated_bd=boundary.copy()
        for ai in range(len(anchors)):
            left=anchors[ai]
            right=anchors[(ai+1)%len(anchors)]
            ids=np.arange(left,right+1,dtype=int) if right>left else np.r_[np.arange(left,n),np.arange(right+1)]
            if len(ids)<3:
                continue
            segment=np.linalg.norm(boundary[np.mod(ids[1:],n)]-
                                   boundary[np.mod(ids[:-1],n)],axis=1)
            cumul=np.r_[0.,np.cumsum(segment)]
            total=float(cumul[-1])
            if total<1e-14:
                continue
            # Every source boundary vertex remains, with distinct arclength
            # parameters, but now lies on a straight target segment.
            updated_bd[ids%n]=(1.-cumul[:,None]/total)*boundary[left]+(
                cumul[:,None]/total)*boundary[right]
        displacement=np.zeros_like(uv)
        displacement[cycle]=updated_bd-boundary
        if np.max(np.linalg.norm(displacement[cycle],axis=1))<=1e-13*scale:
            continue
        # Harmonic interior continuation; original cut source faces remain paired.
        bset=set(cycle.tolist())
        free=np.asarray([j for j in range(len(uv)) if j not in bset],int)
        if len(free):
            adjacent=[set() for _ in uv]
            for a,b,c in faces2d:
                adjacent[a].update((int(b),int(c)))
                adjacent[b].update((int(a),int(c)))
                adjacent[c].update((int(a),int(b)))
            idx={int(j):k for k,j in enumerate(free)}
            rows=[];cols=[];vals=[]
            rhs=np.zeros((len(free),2),float)
            for k,j in enumerate(free):
                ring=adjacent[int(j)]
                rows.append(k);cols.append(k);vals.append(float(len(ring)))
                for t in ring:
                    if t in idx:
                        rows.append(k);cols.append(idx[t]);vals.append(-1.)
                    else:
                        rhs[k]+=displacement[t]
            mat=csr_matrix((vals,(rows,cols)),shape=(len(free),len(free)))
            for dim in (0,1):
                displacement[free,dim]=spsolve(mat,rhs[:,dim])
        trial=uv+displacement
        new_orient=np.cross(trial[faces2d[:,1]]-trial[faces2d[:,0]],
                            trial[faces2d[:,2]]-trial[faces2d[:,0]])
        if np.any(new_orient*orient[0]<=1e-12*scale*scale):
            continue
        trial_poly=Polygon(trial[cycle])
        if not trial_poly.is_valid or trial_poly.area<=0:
            continue
        sd=surface_uv_sd(xyz,faces3d,trial,faces2d)
        if not np.isfinite(sd) or sd>max_sd_relative*source_sd:
            continue
        return trial,{
            "planar_quad_coarse_uv_accepted":True,
            "planar_quad_coarse_boundary_points_before":int(n),
            "planar_quad_coarse_boundary_points_after":int(len(anchors)),
            "planar_quad_coarse_uv_sd_before":float(source_sd),
            "planar_quad_coarse_uv_sd_after":float(sd),
            "planar_quad_coarse_boundary_tolerance":float(tol_fraction*h),
            "planar_quad_coarse_max_uv_displacement":float(np.max(np.linalg.norm(displacement,axis=1))),
        }
    return uv.copy(),{
        "planar_quad_coarse_uv_accepted":False,
        "planar_quad_coarse_boundary_points_before":int(n),
        "planar_quad_coarse_boundary_points_after":int(n),
        "planar_quad_coarse_uv_sd_before":float(source_sd),
        "planar_quad_coarse_uv_sd_after":float(source_sd),
        "planar_quad_coarse_boundary_tolerance":0.,
        "planar_quad_coarse_max_uv_displacement":0.,
    }
