"""Experimental area-threshold lattice boundary -> fixed-boundary UV relaxation.

The *original* official OptCuts topology and CSF-tail result are input.
No posthoc changes to the seam graph, K3D/K2D solvers, or original OptCuts.
A fixed lattice is created BEFORE changing Omega. Each UV triangle is clipped
exactly against the intersected lattice rectangles (no Shapely dependency).
Cells containing >= the user-selected fraction of their area are retained.

Only disk-topology, edge-connected masks with a simple boundary are accepted.
The ordered original OptCuts boundary is mapped onto the rectilinear polygon.
Every grid corner needs a distinct UV boundary vertex, or we fail explicitly;
no barycentric extrapolation, edge collapse or silent fallback is allowed.
Harmonic Dirichlet relaxation moves the interior. A full-boundary solution
must be finite, flipped-free, overlap-free and inside the M2D inverse map.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


class GridBoundaryReparameterizationError(RuntimeError):
    pass


def _clip_axis(poly, axis, value, keep_greater):
    if len(poly)==0:
        return poly
    result=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        da=float(a[axis]-value)
        db=float(b[axis]-value)
        inside_a=da>=-1e-12 if keep_greater else da<=1e-12
        inside_b=db>=-1e-12 if keep_greater else db<=1e-12
        if inside_a and inside_b:
            result.append(b)
        elif inside_a != inside_b:
            t=da/(da-db) if abs(da-db)>1e-18 else 0.
            intersect=a+t*(b-a)
            if inside_a:
                result.append(intersect)
            else:
                result.extend([intersect,b])
    return np.asarray(result,float).reshape(-1,2)


def _triangle_rectangle_area(tri,lo,hi):
    clipped=np.asarray(tri,float)
    for axis,value,greater in ((0,lo[0],True),(0,hi[0],False),
                               (1,lo[1],True),(1,hi[1],False)):
        clipped=_clip_axis(clipped,axis,float(value),greater)
        if len(clipped)<3:
            return 0.
    x,y=clipped[:,0],clipped[:,1]
    return abs(.5*float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))


def _intersection_ratios(uv,faces,origin,step,nx,ny):
    areas=np.zeros((ny,nx),float)
    tri=np.asarray(uv,float)[np.asarray(faces,int)]
    for xyz in tri:
        low=np.min(xyz,axis=0)
        high=np.max(xyz,axis=0)
        left=max(0,int(np.floor((low[0]-origin[0])/step[0]-1e-10)))
        right=min(nx-1,int(np.ceil((high[0]-origin[0])/step[0]+1e-10))-1)
        bottom=max(0,int(np.floor((low[1]-origin[1])/step[1]-1e-10)))
        top=min(ny-1,int(np.ceil((high[1]-origin[1])/step[1]+1e-10))-1)
        if left>right or bottom>top:
            continue
        for j in range(bottom,top+1):
            for i in range(left,right+1):
                lo=origin+step*np.asarray([i,j])
                a=_triangle_rectangle_area(xyz,lo,lo+step)
                if a>0: areas[j,i]+=a
    fraction=areas/float(np.prod(step))
    if np.any(fraction>1.00001):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_UV_OVERLAP: original OptCuts UV triangles "
            "overlap and yield more than 100% area inside a cell"
        )
    return np.clip(fraction,0.,1.)


def _grid_boundary(mask):
    selected=np.argwhere(mask)
    if not len(selected):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_NO_CELLS: no candidate tiles at this coverage threshold"
        )
    ny,nx=mask.shape
    edges=[]
    for j,i in selected:
        j,i=int(j),int(i)
        if j==0 or not mask[j-1,i]: edges.append(((i,j),(i+1,j)))
        if i==nx-1 or not mask[j,i+1]: edges.append(((i+1,j),(i+1,j+1)))
        if j==ny-1 or not mask[j+1,i]: edges.append(((i+1,j+1),(i,j+1)))
        if i==0 or not mask[j,i-1]: edges.append(((i,j+1),(i,j)))
    next_of={}
    inbound=defaultdict(int)
    for a,b in edges:
        if a in next_of:
            raise GridBoundaryReparameterizationError(
                "GRID_BOUNDARY_PINCH_VERTEX: selected cells touch at a point "
                "without a single nonbranching boundary"
            )
        next_of[a]=b;inbound[b]+=1
    if any(inbound[a]!=1 for a in next_of) or len(next_of)!=len(inbound):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_NONMANIFOLD: selected tiles have a branching boundary"
        )
    ordered=[]
    first=min(next_of)
    p=first
    for _ in range(len(edges)+1):
        if p==first and ordered:break
        if p in ordered:break
        ordered.append(p)
        p=next_of.get(p)
    if p!=first or len(ordered)!=len(edges):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_MULTI_COMPONENT_OR_HOLE: selected cells are "
            "disconnected or introduce one or more holes"
        )
    # Remove collinear grid lattice vertices; retain all corners.
    points=np.asarray(ordered,int)
    previous=np.roll(points,1,axis=0)
    after=np.roll(points,-1,axis=0)
    incoming=points-previous
    outgoing=after-points
    turn=incoming[:,0]*outgoing[:,1]-incoming[:,1]*outgoing[:,0]
    corners=points[np.abs(turn)>0]
    if len(corners)<4:
        raise GridBoundaryReparameterizationError("GRID_BOUNDARY_INVALID_OUTLINE")
    return corners


def _sarea(points):
    pts=np.asarray(points,float)
    return .5*float(np.dot(pts[:,0],np.roll(pts[:,1],-1))-
                      np.dot(pts[:,1],np.roll(pts[:,0],-1)))


def _monotone_corner_indices_dp(old, old_t, corners, times):
    """Order-preserving landmark matching rather than an arc-length-only guess.

    The first source boundary vertex stays anchored to the first target
    corner; the caller searches cyclic phases separately. Each later corner
    gets a distinct boundary vertex, minimizing geometric corner mismatch
    with a weak arc-length regularizer. This changes neither the target cell
    mask nor the source UV/3D topology.
    """
    n, m = len(old), len(corners)
    if n == m:
        return list(range(n))
    scale = max(float(np.ptp(old, axis=0).max()),
                float(np.ptp(corners, axis=0).max()), 1e-8)
    weight = .05 * scale * scale
    previous = np.full(n, np.inf)
    previous[0] = 0.
    predecessor = np.full((m, n), -1, int)
    for j in range(1, m):
        current = np.full(n, np.inf)
        local = np.sum((old - corners[j])**2, axis=1)
        local += weight * (old_t - times[j])**2
        # Vectorized prefix-min DP: O(n*m) arithmetic in NumPy rather than
        # Python loops over all source boundary vertices (important on Bunny).
        # For each valid corner position i, choose the cheapest predecessor
        # strictly before i; retain enough vertices for remaining corners.
        prefix = np.minimum.accumulate(previous)
        previous_min = np.r_[np.inf, prefix[:-1]]
        improvement = previous < previous_min
        argmin_prefix = np.maximum.accumulate(
            np.where(improvement, np.arange(n), -1)
        )
        allowed = np.arange(j, n - (m - j) + 1)
        predecessor_slots = allowed - 1
        feasible = (np.isfinite(prefix[predecessor_slots])
                    & (argmin_prefix[predecessor_slots] >= 0))
        positions = allowed[feasible]
        current[positions] = (
            prefix[predecessor_slots[feasible]] + local[positions]
        )
        predecessor[j, positions] = argmin_prefix[predecessor_slots[feasible]]
        previous = current
    k = int(np.argmin(previous))
    if not np.isfinite(previous[k]):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_DP_CORRESPONDENCE_FAILED"
        )
    indices = [0] * m
    for j in range(m - 1, 0, -1):
        indices[j] = k
        k = int(predecessor[j, k])
    if k != 0:
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_DP_BACKTRACK_FAILED"
        )
    return indices


def _boundary_target_candidates(uv,loop,outline,max_phases=48):
    """Search cyclic phases and two strictly order-preserving corner fits.

    The arc-length fit can assign an awkward source vertex to a concave grid
    corner. A dynamic-programming candidate also minimizes landmark
    displacement without changing the prescribed target outline.
    """
    old=np.asarray(uv,float)[np.asarray(loop,int)]
    new=np.asarray(outline,float)
    if _sarea(old)*_sarea(new)<0:
        new=new[::-1].copy()
    n,m=len(old),len(new)
    if n<m:
        raise GridBoundaryReparameterizationError(
            f"GRID_BOUNDARY_NOT_ENOUGH_VERTICES: need {m}, got {n}"
        )
    old_length=np.linalg.norm(np.roll(old,-1,axis=0)-old,axis=1)
    new_length=np.linalg.norm(np.roll(new,-1,axis=0)-new,axis=1)
    if min(float(old_length.sum()),float(new_length.sum()))<1e-12:
        raise GridBoundaryReparameterizationError("GRID_BOUNDARY_DEGENERATE_CONTOUR")
    old_t=np.r_[0.,np.cumsum(old_length[:-1])]/old_length.sum()
    nearest=int(np.argmin(np.linalg.norm(new-old[0],axis=1)))
    phases=(list(range(m)) if m<=max_phases else
            list(dict.fromkeys([nearest]+list(np.linspace(
                0,m-1,max_phases,dtype=int)))))
    ranked=[]
    for phase in phases:
        corners=np.roll(new,-phase,axis=0)
        lengths=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)
        times=np.r_[0.,np.cumsum(lengths[:-1])]/lengths.sum()
        indices=[0]
        for j in range(1,m):
            idx=max(indices[-1]+1,int(np.searchsorted(old_t,times[j])))
            idx=min(idx,n-(m-j))
            if idx<=indices[-1]: break
            indices.append(idx)
        if len(indices)!=m: continue
        options=[("arclength",indices)]
        if n>m:
            dp_indices=_monotone_corner_indices_dp(old,old_t,corners,times)
            if dp_indices!=indices:
                options.append(("corner_dp",dp_indices))
        for method,selected_indices in options:
            mapped=np.empty_like(old)
            for j in range(m):
                start=selected_indices[j]
                end=selected_indices[j+1] if j<m-1 else n
                segment=old_length[start:end]
                fraction=np.r_[0.,np.cumsum(segment[:-1])]/max(float(segment.sum()),1e-12)
                mapped[start:end]=corners[j]+fraction[:,None]*(corners[(j+1)%m]-corners[j])
            if not np.isfinite(mapped).all():continue
            sq=np.sum((mapped-old)**2,axis=1)
            score=float(sq.mean()+.15*sq.max())
            ranked.append((score,int(phase),mapped,method))
    if not ranked:
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_ORDERED_CORRESPONDENCE_FAILED"
        )
    return sorted(ranked,key=lambda x:(x[0],x[1]))


def _boundary_target_map(uv,loop,outline):
    return _boundary_target_candidates(uv,loop,outline)[0][2]


def _ensure_boundary_vertices(parameterization,corner_count,loop):
    """Subdivide REAL boundary triangles if the grid stair has more corners.

    Each new UV vertex is paired with an interpolated 3D source vertex,
    and BOTH uv_faces and surface_faces are subdivided in lockstep.
    No fake UV-only boundary vertices and no broken inverse M2D->M3D map.
    """
    loop=[int(i) for i in loop]
    if len(loop)>=corner_count:
        return loop,0
    uv=np.asarray(parameterization.uv_vertices_2d,float).copy()
    xyz=np.asarray(parameterization.surface_vertices_3d,float).copy()
    uf=np.asarray(parameterization.uv_faces,int).copy().tolist()
    sf=np.asarray(parameterization.surface_faces,int).copy().tolist()
    if len(uf)!=len(sf):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_SOURCE_FACE_MISMATCH: UV/3D triangle correspondence lost"
        )
    # Each original boundary edge is evenly subdivided. Worst-case vertex
    # count grows to <2*n_corners, independent of dense 3D interior geometry.
    factor=int(np.ceil(float(corner_count)/len(loop)))
    n_new=0
    for ai in range(len(loop)):
        start=int(loop[ai]);end=int(loop[(ai+1)%len(loop)])
        current=start
        for k in range(1,factor):
            found=False
            for fi,face in enumerate(uf):
                for edge_i in range(3):
                    aa=int(face[edge_i]);bb=int(face[(edge_i+1)%3])
                    if {aa,bb}!={current,end}:
                        continue
                    # Use a fraction of the REMAINING edge so all points
                    # lie evenly along the original boundary edge.
                    remain=factor-k+1
                    t=(1./float(remain) if aa==current
                       else 1.-1./float(remain))
                    z=int(face[(edge_i+2)%3])
                    xyzface=sf[fi]
                    xa=int(xyzface[edge_i]);xb=int(xyzface[(edge_i+1)%3])
                    xz=int(xyzface[(edge_i+2)%3])
                    vi=len(uv);si=len(xyz)
                    uv=np.vstack([uv,(1.-t)*uv[aa]+t*uv[bb]])
                    xyz=np.vstack([xyz,(1.-t)*xyz[xa]+t*xyz[xb]])
                    uf[fi]=[aa,vi,z]
                    sf[fi]=[xa,si,xz]
                    uf.append([vi,bb,z])
                    sf.append([si,xb,xz])
                    current=vi
                    n_new+=1
                    found=True
                    break
                if found:break
            if not found:
                raise GridBoundaryReparameterizationError(
                    "GRID_BOUNDARY_SUBDIVISION_FAILED: original boundary edge "
                    "is missing from paired UV/3D triangles"
                )
    parameterization.uv_vertices_2d=uv
    parameterization.surface_vertices_3d=xyz
    parameterization.uv_faces=np.asarray(uf,int)
    parameterization.surface_faces=np.asarray(sf,int)
    from .optcuts_backend import _boundary_loops
    loops=_boundary_loops(parameterization.uv_faces)
    if len(loops)!=1 or len(loops[0])<corner_count:
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_SUBDIVISION_TOPOLOGY: subdivided UV is not a single disk"
        )
    return loops[0],n_new


def _harmonic_dirichlet(uv,faces,loop,targets):
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import spsolve
    uv=np.asarray(uv,float)
    faces=np.asarray(faces,int)
    n=len(uv)
    bound=np.asarray(loop,int)
    if len(np.unique(bound))!=len(bound):
        raise GridBoundaryReparameterizationError("GRID_BOUNDARY_DUPLICATE_LOOP_VERTEX")
    is_b=np.zeros(n,bool);is_b[bound]=True
    interior=np.flatnonzero(~is_b)
    result=uv.copy()
    result[bound]=targets
    if not len(interior):return result
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    weights=np.ones(len(edges))
    degree=np.bincount(np.r_[edges[:,0],edges[:,1]],minlength=n).astype(float)
    rows=np.r_[edges[:,0],edges[:,1]]
    cols=np.r_[edges[:,1],edges[:,0]]
    L=diags(degree,format="csr")+coo_matrix(
        (-np.r_[weights,weights],(rows,cols)),shape=(n,n)).tocsr()
    # Low-strength anchoring preserves the original low-distortion OptCuts UV,
    # while applying the exact requested boundary to the interior.
    anchor=.025
    A=L[interior][:,interior]+diags(np.full(len(interior),anchor),format="csr")
    B=-L[interior][:,bound]@targets+anchor*uv[interior]
    for axis in (0,1):
        result[interior,axis]=spsolve(A,B[:,axis])
    if not np.isfinite(result).all():
        raise GridBoundaryReparameterizationError("GRID_BOUNDARY_HARMONIC_NONFINITE")
    return result


def _actual_overlay_lattice(domain):
    """Extract *the actual production M2D lattice* before deforming Omega."""
    count_x=int(getattr(domain,"overlay_nx",0))
    count_y=int(getattr(domain,"overlay_ny",0))
    points=np.asarray(domain.uv_vertices,float)[:,:2]
    if count_x<2 or count_y<2 or len(points)!=(count_x+1)*(count_y+1):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_REAL_M2D_LATTICE_MISSING: original Ω must first "
            "generate the actual rectangular M2D overlay"
        )
    origin=points[0].copy()
    dx=points[1]-points[0]
    dy=points[count_x+1]-points[0]
    tol=max(float(np.ptp(points,axis=0).max())*1e-7,1e-10)
    if (abs(dx[1])>tol or abs(dy[0])>tol or
            dx[0]<=1e-12 or dy[1]<=1e-12):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_ROTATED_OR_IRREGULAR_LATTICE_UNSUPPORTED: "
            "cannot change grid phase or pretend the M2D grid is axis-aligned"
        )
    step=np.array([dx[0],dy[1]],float)
    expected=origin+np.stack(np.meshgrid(
        np.arange(count_x+1)*step[0],
        np.arange(count_y+1)*step[1],indexing="xy"),axis=-1).reshape(-1,2)
    if not np.allclose(points,expected,atol=tol,rtol=0):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_REAL_M2D_LATTICE_MISMATCH: overlay not regular"
        )
    return origin,step,count_x,count_y


def construct_grid_target(uv,faces,boundary_loop,*,nx,ny,threshold=.5,
                          margin=1,domain=None):
    """Choose cells from the REAL production M2D lattice, not an inferred one."""
    if not np.isfinite(threshold) or not (0.<threshold<=1.):
        raise ValueError("GRID_BOUNDARY_THRESHOLD_RANGE: coverage must be in (0,1]")
    if domain is not None:
        origin,step,n_cells_x,n_cells_y=_actual_overlay_lattice(domain)
        margin=int(getattr(domain,"overlay_margin_tiles",0))
    else:
        # Pure-geometry tests only. Live runs MUST supply the actual domain.
        nx,ny=int(nx),int(ny)
        if nx<2 or ny<2 or nx>100 or ny>100:
            raise ValueError("GRID_BOUNDARY_GRID_RANGE: grid nx/ny must be in [2,100]")
        boundary_uv=np.asarray(uv,float)[np.asarray(boundary_loop,int)]
        origin=boundary_uv.min(axis=0)
        span=np.maximum(np.ptp(boundary_uv,axis=0),1e-8)
        n_cells_x=max(2*nx,nx+2,2)
        n_cells_y=max(2*ny,ny+2,2)
        step=span/np.array([n_cells_x,n_cells_y],float)
    coverage=_intersection_ratios(uv,faces,origin,step,n_cells_x,n_cells_y)
    mask=coverage>=float(threshold)-1e-12
    corners=_grid_boundary(mask)
    target=origin+np.asarray(corners,float)*step
    return dict(mask=mask,coverage=coverage,grid_outline=target,
                grid_outline_integer=np.asarray(corners,int),
                origin=origin,step=step,nx=n_cells_x,ny=n_cells_y,
                margin=int(max(0,margin)),
                grid_source="real_production_m2d_overlay" if domain is not None else "synthetic_fixture")


def apply_area_grid_boundary(parameterization,grid,params,*,domain=None):
    from .optcuts_backend import _boundary_loops
    uv=np.asarray(parameterization.uv_vertices_2d,float)
    faces=np.asarray(parameterization.uv_faces,int)
    loops=_boundary_loops(faces)
    if len(loops)!=1:
        raise GridBoundaryReparameterizationError(
            f"GRID_BOUNDARY_REQUIRES_ONE_DISK_CHART: found {len(loops)} UV boundary loops"
        )
    loop=loops[0]
    thresh=float(getattr(params,"optcuts_area_grid_coverage_threshold",.5))
    margin=int(getattr(params,"omega_overlay_margin",1))
    target=construct_grid_target(uv,faces,loop,nx=grid.nx,ny=grid.ny,
                                 threshold=thresh,margin=margin,domain=domain)
    loop,inserted=_ensure_boundary_vertices(
        parameterization,len(target["grid_outline"]),loop
    )
    # Work with paired and subdivided triangles throughout; the original
    # 3D surface remains geometrically identical, only triangulation changes.
    uv=np.asarray(parameterization.uv_vertices_2d,float)
    faces=np.asarray(parameterization.uv_faces,int)
    boundary_candidates=_boundary_target_candidates(uv,loop,target["grid_outline"])
    # quality() requires OptCutsResult (not SurfaceParameterization).
    # Evaluate by wrapping the current chart with all corresponding faces.
    # Homotopy is diagnostic and prevents accepting a folded intermediate.
    # The final full-boundary target (not a fractional approximation) must pass.
    base=uv.copy()
    result=None
    from .optcuts_quality_audit import quality as opt_quality
    from .optcuts_backend import OptCutsResult
    def assess(candidate):
        ref=OptCutsResult(
            np.asarray(parameterization.surface_vertices_3d,float),
            np.asarray(parameterization.surface_faces,int),
            np.asarray(candidate,float),
            faces,
            [list(loop)],"area-grid-boundary",{})
        return opt_quality(ref)
    baseline=assess(base)
    if int(baseline["flipped_triangles"]) or int(baseline["injectivity_overlap_pairs"]):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_SOURCE_INVALID: original OptCuts has inverted/overlapping UV")
    from .optcuts_grid_boundary_untangle import fit_by_continuation,oriented_area_ratio
    # Keep the SAME selected cell union, change only cyclic vertex matching.
    phase_errors=[]
    chosen_phase=None
    chosen_method=None
    history=[]
    # The former 6-candidate shortlist could fail without trying the
    # geometrically better monotone corner assignments. Keep a bounded,
    # explicitly audited search instead of silently changing the cell mask.
    for _cost,phase,mapped,method in boundary_candidates[:min(12,len(boundary_candidates))]:
        try:
            result,history=fit_by_continuation(
                base,faces,loop,mapped,_harmonic_dirichlet,assess)
            chosen_phase=int(phase)
            chosen_method=str(method)
            break
        except ValueError as exc:
            phase_errors.append(f"phase={phase}, fit={method}: {exc}")
    if result is None:
        raise GridBoundaryReparameterizationError(
            f"GRID_BOUNDARY_ALL_MONOTONE_CORRESPONDENCES_FAILED: "
            f"attempted={len(phase_errors)} / {len(boundary_candidates)}, "
            f"remaining_untried={len(boundary_candidates)-len(phase_errors)}; "
            + " | ".join(phase_errors)
        )
    final=assess(result)
    if float(final["distortion_area_mean"])>max(
        1.30*float(baseline["distortion_area_mean"]),
        float(baseline["distortion_area_mean"])+.1):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_DISTORTION_EXCESS: mean SD grew too much "
            f"({baseline['distortion_area_mean']:.4g} -> {final['distortion_area_mean']:.4g})"
        )
    # Check the grid boundary vertices exactly lie on the new UV boundary
    # within numerical tolerance, not just near it.
    outline=np.asarray(target["grid_outline"],float)
    uv_bound=np.asarray(result,float)[np.asarray(loop,int)]
    tolerance=max(float(np.max(target["step"]))*1e-6,1e-9)
    for corner in outline:
        if float(np.min(np.linalg.norm(uv_bound-corner,axis=1)))>tolerance:
            raise GridBoundaryReparameterizationError(
                "GRID_BOUNDARY_CORNER_MATCH_FAILED: corner has no matching UV boundary vertex"
            )
    # Keep the actual original contour and the target polyomino for a
    # literal before/mask/after visual audit in the Omega UI.
    source_loops=np.asarray(parameterization.omega_boundary,float).copy()
    parameterization._optcuts_area_grid_source_boundary=source_loops
    parameterization._optcuts_area_grid_target_outline=np.vstack([
        outline,outline[0],
    ])
    parameterization.uv_vertices_2d=result
    parameterization.omega_boundary=result[np.r_[loop,loop[0]]]
    # Clear stale inverse accelerator; it was built for the old coordinates.
    parameterization.triangle_acceleration=None
    # Invalidate obsolete per-triangle diagnostics from BEFORE boundary
    # subdivision / UV reparameterization. They are not the new Jacobians.
    for name in ("per_triangle_lambda","per_triangle_csf",
                 "per_triangle_sigma","per_triangle_distortion"):
        parameterization.metrics.pop(name,None)
    parameterization.metrics["boundary_loop"]=[int(i) for i in loop]
    count=int(target["mask"].sum())
    parameterization.metrics.update(
        optcuts_area_grid_boundary_enabled=True,
        optcuts_area_grid_boundary_status="accepted_exact_grid_outline",
        optcuts_area_grid_coverage_threshold=thresh,
        optcuts_area_grid_cell_count=count,
        optcuts_area_grid_candidate_cell_count=int(target["mask"].size),
        optcuts_area_grid_boundary_corner_count=int(len(outline)),
        optcuts_area_grid_subdivided_boundary_vertex_count=int(inserted),
        optcuts_area_grid_boundary_cyclic_phase=int(chosen_phase),
        optcuts_area_grid_boundary_correspondence_method=chosen_method,
        optcuts_area_grid_boundary_untangle_history=history,
        optcuts_area_grid_min_signed_area_ratio=float(np.min(
            oriented_area_ratio(base,result,faces))),
        optcuts_area_grid_original_mean_sd=float(baseline["distortion_area_mean"]),
        optcuts_area_grid_final_mean_sd=float(final["distortion_area_mean"]),
        optcuts_area_grid_grid_origin=target["origin"].tolist(),
        optcuts_area_grid_grid_step=target["step"].tolist(),
        optcuts_area_grid_overlay_nx=int(target["nx"]),
        optcuts_area_grid_overlay_ny=int(target["ny"]),
        optcuts_area_grid_margin=int(target["margin"]),
        optcuts_area_grid_selection="exact UV-triangle/cell intersection >= threshold",
        optcuts_area_grid_lattice_source=target["grid_source"],
        optcuts_area_grid_reparameterization="arclength/DP monotone corner correspondence + fixed-boundary harmonic Dirichlet and area untangle",
        optcuts_area_grid_seam_topology_preserved=True,
    )
    # Exact mask stored only transiently in the parameterization object
    # (not JSON metrics). The original fixed lattice is authoritative.
    parameterization._optcuts_area_grid_mask=target["mask"]
    parameterization._optcuts_area_grid_coverage=target["coverage"]
    return parameterization


def fix_domain_lattice(domain,parameterization):
    """Restore the PRE-reparameterization lattice rather than recompute it."""
    m=parameterization.metrics
    if not m.get("optcuts_area_grid_boundary_enabled",False):
        return domain
    origin=np.asarray(m["optcuts_area_grid_grid_origin"],float)
    step=np.asarray(m["optcuts_area_grid_grid_step"],float)
    nx=int(m["optcuts_area_grid_overlay_nx"])
    ny=int(m["optcuts_area_grid_overlay_ny"])
    # origin already includes all production margins. Never add them again.
    margin=int(m["optcuts_area_grid_margin"])
    x=origin[0]+np.arange(nx+1)*step[0]
    y=origin[1]+np.arange(ny+1)*step[1]
    xx,yy=np.meshgrid(x,y,indexing="xy")
    domain.uv_vertices=np.stack((xx,yy),axis=2).reshape(-1,2)
    domain.overlay_nx=nx
    domain.overlay_ny=ny
    domain.overlay_margin_tiles=margin
    domain.overlay_step_u=float(step[0])
    domain.overlay_step_v=float(step[1])
    domain.boundary=np.asarray(parameterization.omega_boundary,float)
    domain._optcuts_area_grid_mask=np.asarray(parameterization._optcuts_area_grid_mask,bool).copy()
    domain._optcuts_area_grid_coverage=np.asarray(parameterization._optcuts_area_grid_coverage,float).copy()
    return domain


def keep_cells_on_fixed_lattice(mesh,domain):
    """Select the SAME exact-overlap tiles that defined the snapped boundary."""
    mask=np.asarray(domain._optcuts_area_grid_mask,bool)
    faces=np.asarray(mesh.faces,int)
    src_nx=int(domain.overlay_nx)
    margin=int(domain.overlay_margin_tiles)
    # Overlay cells are row-major by y, x; verify actual geometry, not IDs.
    uv=np.asarray(mesh.vertices,float)
    base=np.asarray(domain.uv_vertices,float)
    if len(uv)!=len(base) or not np.allclose(uv[:,:2],base):
        raise GridBoundaryReparameterizationError(
            "GRID_BOUNDARY_M2D_LATTICE_MISMATCH: M2D and target grid differ"
        )
    keep=np.zeros(len(faces),bool)
    for fi,f in enumerate(faces):
        p=np.asarray(uv[f,:2],float)
        # Cell center maps to integer coordinates relative to fixed origin.
        c=p.mean(axis=0)
        origin=np.asarray(domain.parameterization.metrics["optcuts_area_grid_grid_origin"],float)
        step=np.asarray(domain.parameterization.metrics["optcuts_area_grid_grid_step"],float)
        ij=np.floor((c-origin)/step+1e-9).astype(int)
        i,j=int(ij[0]),int(ij[1])
        if 0<=i<mask.shape[1] and 0<=j<mask.shape[0]:
            keep[fi]=bool(mask[j,i])
    if int(keep.sum())!=int(mask.sum()):
        raise GridBoundaryReparameterizationError(
            f"GRID_BOUNDARY_M2D_FACE_COUNT_MISMATCH: expected {int(mask.sum())}, "
            f"actual {int(keep.sum())}"
        )
    return keep
