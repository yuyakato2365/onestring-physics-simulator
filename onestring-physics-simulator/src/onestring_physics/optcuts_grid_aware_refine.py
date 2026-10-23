"""Optional grid-aware UV refinement after the *official* OptCuts solve.

This is an experimental post-OptCuts ablation, NOT a modification of the
authors' C++ topology optimizer. It preserves exact OptCuts cut connectivity
and all 3D/UV face correspondences, softly straightens existing seam sides
onto the lattice used by OneString's FREE-boundary Omega overlay, and
rejects candidates that fold, overlap, exceed distortion guardrails, or
substantially lose complete-grid-cell coverage.

The lattice follows _rebuild_domain_overlay_for_general_omega:
    h_u = (max(u)-min(u)) / (2*nx),
    h_v = (max(v)-min(v)) / (2*ny).
This can be anisotropic; do NOT silently claim fabrication-square cells.
"""
from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import numpy as np


def _seam_sides(result):
    """Both UV copies of each *internal* cut edge, keyed by welded 3D geometry."""
    xyz=np.asarray(result.surface_vertices_3d,float)
    sf=np.asarray(result.surface_faces,int)
    uv=np.asarray(result.uv_vertices_2d,float)
    uf=np.asarray(result.uv_faces,int)
    if len(sf)!=len(uf) or len(sf)==0 or len(xyz)==0:
        return []
    scale=max(float(np.ptp(xyz,axis=0).max()),1.)
    tolerance=max(scale*1e-10,1e-12)
    canonical={}
    vertex_ids=[]
    for xyz_point in xyz:
        key=tuple(np.rint(xyz_point/tolerance).astype(np.int64))
        if key not in canonical:
            canonical[key]=len(canonical)
        vertex_ids.append(canonical[key])
    vertex_ids=np.asarray(vertex_ids,dtype=int)
    incidence=defaultdict(list)
    for f3,fuv in zip(sf,uf):
        ids=vertex_ids[f3]
        for ai,bi in ((0,1),(1,2),(2,0)):
            a,b=int(ids[ai]),int(ids[bi])
            if a==b: continue
            first,last=(ai,bi) if a<b else (bi,ai)
            incidence[(min(a,b),max(a,b))].append((
                int(fuv[first]),int(fuv[last]),int(f3[first]),int(f3[last])
            ))
    out=[]
    uv_tolerance=max(float(np.ptp(uv,axis=0).max())*1e-8,1e-10)
    for copies in incidence.values():
        if len(copies)!=2: continue
        a,b=copies
        if (np.linalg.norm(uv[a[0]]-uv[b[0]]) <= uv_tolerance and
                np.linalg.norm(uv[a[1]]-uv[b[1]]) <= uv_tolerance):
            continue
        length=float(np.linalg.norm(xyz[a[2]]-xyz[a[3]]))
        if length<=1e-12: continue
        out.append(((a[0],a[1]),(b[0],b[1]),length))
    return out


def _lattice(uv, nx, ny):
    xy=np.asarray(uv,float)
    lower=xy.min(axis=0)
    span=np.maximum(np.ptp(xy,axis=0),1e-9)
    return lower,np.maximum(span/np.asarray([2*max(nx,1),2*max(ny,1)]),1e-10)


def _seam_quality(uv, sides, nx, ny):
    """Dimensionless grid alignment error; lower is better."""
    if not sides: return 0.
    lower,h=_lattice(uv,nx,ny)
    error,total=0.,0.
    for a,b,length in sides:
        for v0,v1 in (a,b):
            p,q=uv[[v0,v1]]
            d=p-q
            center=(p+q)*.5
            # Both endpoints must live on ONE constant-u or constant-v
            # lattice line. Penalize diagonal seams and phase mismatch.
            vertical=((d[0]/h[0])**2+
                      ((center[0]-lower[0])/h[0]-np.round((center[0]-lower[0])/h[0]))**2)
            horizontal=((d[1]/h[1])**2+
                        ((center[1]-lower[1])/h[1]-np.round((center[1]-lower[1])/h[1]))**2)
            error+=length*min(float(vertical),float(horizontal))
            total+=length
    return float(error/max(total,1e-12))


def _points_inside_polygon(points, polygon):
    """Vectorized even-odd polygon containment for approximate grid-cell yield."""
    pts=np.asarray(points,float)
    poly=np.asarray(polygon,float)
    x,y=pts[:,0],pts[:,1]
    inside=np.zeros(len(pts),dtype=bool)
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        x0,y0=float(a[0]),float(a[1]);x1,y1=float(b[0]),float(b[1])
        if y0==y1: continue
        crosses=((y0>y)!=(y1>y))
        xcut=x0+(y-y0)*(x1-x0)/(y1-y0)
        inside ^= crosses & (x < xcut)
    return inside


def _cell_yield_proxy(uv, loop, nx, ny):
    """Fraction of bounding-box cells with four corners + center inside Omega.

    Not an exact seam-aware M2D count: boundary-segment crossings and
    multi-chart inverse validity still need to be checked downstream.
    This proxy is only a candidate ranking/diagnostic term.
    """
    if not loop or nx<=0 or ny<=0: return 0.
    xy=np.asarray(uv,float)
    polygon=xy[np.asarray(loop,dtype=int)]
    lower,h=_lattice(xy,nx,ny)
    # Match the actual 2*nx by 2*ny free-boundary overlay density.
    nu,nv=2*nx,2*ny
    x=lower[0]+np.arange(nu+1)*h[0]
    y=lower[1]+np.arange(nv+1)*h[1]
    xx,yy=np.meshgrid(x,y,indexing="xy")
    inside=_points_inside_polygon(np.column_stack([xx.ravel(),yy.ravel()]),polygon).reshape((nv+1,nu+1))
    cx=.5*(x[:-1]+x[1:]);cy=.5*(y[:-1]+y[1:])
    mx,my=np.meshgrid(cx,cy,indexing="xy")
    mid=_points_inside_polygon(np.column_stack([mx.ravel(),my.ravel()]),polygon).reshape((nv,nu))
    complete=(inside[:-1,:-1]&inside[1:,:-1]&inside[:-1,1:]&inside[1:,1:]&mid)
    return float(complete.sum()/max(nu*nv,1))


def _seam_line_targets(uv,sides,nx,ny):
    """Fixed nearest-lattice targets for both copies of each seam edge."""
    lower,h=_lattice(uv,nx,ny)
    values=defaultdict(list)
    for a,b,length in sides:
        mean=np.mean([uv[list(a)],uv[list(b)]],axis=(0,1))
        edge_vector=np.mean([uv[a[1]]-uv[a[0]],uv[b[1]]-uv[b[0]]],axis=0)
        fixed_axis=1 if abs(edge_vector[0]/h[0])>=abs(edge_vector[1]/h[1]) else 0
        for endpoint_pair in (a,b):
            p,q=uv[list(endpoint_pair)]
            # Preserve the seam gap: snap each UV side to its own adjacent
            # grid line, NEVER weld separate seam copies together.
            coord=float(.5*(p[fixed_axis]+q[fixed_axis]))
            target=float(lower[fixed_axis]+np.round((coord-lower[fixed_axis])/h[fixed_axis])*h[fixed_axis])
            for vertex_id in endpoint_pair:
                values[(int(vertex_id),fixed_axis)].append((target,length))
    target=np.zeros_like(uv)
    weights=np.zeros_like(uv)
    for (idx,axis),records in values.items():
        sum_weight=sum(weight for _,weight in records)
        target[idx,axis]=sum(val*w for val,w in records)/max(sum_weight,1e-15)
        weights[idx,axis]=1.
    return target,weights


def _solve_harmonic_displacement(uv,uf,target,weights,strength):
    """Sparse smooth displacement anchored to original UV throughout."""
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import spsolve
    n=len(uv)
    e=np.concatenate([uf[:,[0,1]],uf[:,[1,2]],uf[:,[2,0]]],axis=0)
    e=np.unique(np.sort(e,axis=1),axis=0)
    degree=np.bincount(np.r_[e[:,0],e[:,1]],minlength=n).astype(float)
    row=np.r_[e[:,0],e[:,1]]
    col=np.r_[e[:,1],e[:,0]]
    L=diags(degree,format="csr")+coo_matrix(
        (-np.ones(len(row)),(row,col)),shape=(n,n)).tocsr()
    out=uv.copy()
    smoothness=.3
    anchor=.15
    for axis in range(2):
        w=strength*weights[:,axis]
        A=(smoothness*L+diags(anchor+w,format="csr")).tocsr()
        rhs=w*(target[:,axis]-uv[:,axis])
        out[:,axis]+=spsolve(A,rhs)
    return out


def _oriented_and_nondegenerate(original,new,faces):
    tri0=original[faces];tri=new[faces]
    def cross(q):
        a=q[:,1]-q[:,0];b=q[:,2]-q[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    before=cross(tri0);after=cross(tri)
    return (np.isfinite(new).all() and
            np.all(before*after>0) and
            np.all(np.abs(after)>=np.maximum(1e-12,0.05*np.abs(before))))


def refine_optcuts_grid_aware(result, *, nx, ny, strength=3.0,
                              distortion_bound=4.1, max_coverage_loss=.01):
    """Improve existing OptCuts seam-grid agreement without changing seam topology.

    May return original result unchanged with clear diagnostic rejection status.
    No local candidate can exceed the baseline distortion, flip, overlap,
    or lose more than max_coverage_loss of the bounding grid.
    """
    from .optcuts_quality_audit import quality
    from .optcuts_uv_overlap_guard import positive_area_uv_overlaps
    nx=int(nx);ny=int(ny)
    if nx<2 or ny<2 or nx>100 or ny>100:
        raise ValueError("grid-aware OptCuts requires 2<=nx,ny<=100")
    if not np.isfinite(strength) or strength<=0:
        raise ValueError("grid-aware OptCuts strength must be positive")
    base=np.asarray(result.uv_vertices_2d,float)
    faces=np.asarray(result.uv_faces,int)
    sides=_seam_sides(result)
    original_metrics=dict(result.metrics)
    diagnostics=dict(
        optcuts_grid_aware_enabled=True,
        optcuts_grid_aware_model="post-OptCuts seam-preserving harmonic UV adjustment; NOT joint C++ topology optimization",
        optcuts_grid_aware_grid_nx=nx,
        optcuts_grid_aware_grid_ny=ny,
        optcuts_grid_aware_seam_3d_edge_count=len(sides),
        optcuts_grid_aware_strength=float(strength),
        optcuts_grid_aware_distortion_bound=float(distortion_bound),
    )
    if not sides:
        result.metrics.update(diagnostics,optcuts_grid_aware_status="no_internal_seam_unchanged")
        return result
    if not len(result.boundary_loops):
        result.metrics.update(diagnostics,optcuts_grid_aware_status="no_boundary_unchanged")
        return result
    loop=result.boundary_loops[0]
    alignment_before=_seam_quality(base,sides,nx,ny)
    yield_before=_cell_yield_proxy(base,loop,nx,ny)
    diagnostics.update(
        optcuts_grid_aware_alignment_before=alignment_before,
        optcuts_grid_aware_cell_yield_proxy_before=yield_before,
    )
    target,weights=_seam_line_targets(base,sides,nx,ny)
    if not np.any(weights) or not np.isfinite(alignment_before):
        result.metrics.update(diagnostics,optcuts_grid_aware_status="no_valid_targets_unchanged")
        return result
    try:
        moved=_solve_harmonic_displacement(base,faces,target,weights,float(strength))
    except (ValueError,RuntimeError,ArithmeticError,ImportError) as exc:
        result.metrics.update(diagnostics,optcuts_grid_aware_status="solver_failed_unchanged",
                              optcuts_grid_aware_error=type(exc).__name__+": "+str(exc))
        return result
    from .optcuts_quality_audit import quality as uv_quality
    baseline_quality=uv_quality(result)
    baseline_dist=float(baseline_quality["distortion_area_mean"])
    best=None
    attempts=[]
    for fraction in (.125,.25,.5,.75,1.):
        candidate_uv=base+fraction*(moved-base)
        if not _oriented_and_nondegenerate(base,candidate_uv,faces):
            attempts.append((fraction,"flipped_or_degenerate"))
            continue
        candidate=replace(result,uv_vertices_2d=candidate_uv,metrics=dict(original_metrics))
        candidate_quality=uv_quality(candidate)
        mean_dist=float(candidate_quality["distortion_area_mean"])
        if (not np.isfinite(mean_dist)
                or int(candidate_quality["flipped_triangles"])!=0
                or int(candidate_quality["degenerate_triangles"])!=0
                or int(candidate_quality["injectivity_overlap_pairs"])!=0
                or mean_dist>baseline_dist*1.005+1e-10
                or (baseline_dist<=distortion_bound+1e-8 and
                    mean_dist>distortion_bound+1e-8)):
            attempts.append((fraction,"distortion_or_overlap_guard"))
            continue
        alignment=_seam_quality(candidate_uv,sides,nx,ny)
        coverage=_cell_yield_proxy(candidate_uv,loop,nx,ny)
        if coverage<yield_before-max_coverage_loss:
            attempts.append((fraction,"cell_yield_guard"))
            continue
        if alignment>=alignment_before-1e-8:
            attempts.append((fraction,"no_seam_alignment_gain"))
            continue
        score=(alignment/max(alignment_before,1e-10)
               +2.*max(0.,yield_before-coverage))
        if best is None or score<best[0]:
            best=(score,candidate,alignment,coverage,fraction,mean_dist)
        attempts.append((fraction,"accepted_candidate"))
    diagnostics["optcuts_grid_aware_attempts"]=attempts
    if best is None:
        result.metrics.update(diagnostics,optcuts_grid_aware_status="rejected_retained_original",
                              optcuts_grid_aware_alignment_after=alignment_before,
                              optcuts_grid_aware_cell_yield_proxy_after=yield_before)
        return result
    _,best_result,align,coverage,fraction,dist=best
    diagnostics.update(
        optcuts_grid_aware_status="accepted",
        optcuts_grid_aware_alignment_after=float(align),
        optcuts_grid_aware_cell_yield_proxy_after=float(coverage),
        optcuts_grid_aware_fraction=float(fraction),
        optcuts_grid_aware_mean_symmetric_dirichlet_before=baseline_dist,
        optcuts_grid_aware_mean_symmetric_dirichlet_after=float(dist),
        optcuts_grid_aware_seam_topology_changed=False,
        optcuts_grid_aware_author_code_modified=False,
    )
    best_result.metrics.update(diagnostics)
    return best_result
