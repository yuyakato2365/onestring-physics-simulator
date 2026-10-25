"""Opt-in source-seam lattice-transition Omega parameterization (no staircase).

On each physical cut chain, require u_B=R^k u_A+h*n, k in 0..3,
n in Z².  UV copies are tracked by matching ORIGINAL 3D vertex IDs.
The constrained solve does not straighten seams, modify the OptCuts cut
topology, change the user Grid, or fall back to an older Omega mode.

A seamless UV transition alone is NOT yet a seam-aware quad-grid overlay.
This pilot reports M2D/3D coverage and exact K3D closure as UNVERIFIED.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


class SeamlessLatticeError(RuntimeError):
    pass


ROT = (
    np.eye(2,dtype=int),
    np.array([[0,-1],[1,0]],int),
    -np.eye(2,dtype=int),
    np.array([[0,1],[-1,0]],int),
)


def source_chains(chart):
    from .optcuts_grid_constrained_parameterization_patch import (
        _surface_seam_records, _physical_chains,
        _record_lookup, _uv_copy_chains_for_physical_chain,
    )
    records=_surface_seam_records(chart)
    lookup=_record_lookup(records)
    result=[]
    for chain in _physical_chains(records):
        a,b=_uv_copy_chains_for_physical_chain(chain,lookup)
        if len(a)!=len(chain) or len(b)!=len(chain):
            raise SeamlessLatticeError("SEAMLESS_SOURCE_CHAIN_COPY_MISMATCH")
        result.append((tuple(chain),tuple(a),tuple(b)))
    return result


def best_integer_transitions(uv,chains,h):
    """One common rigid-grid transformation for an ENTIRE source chain."""
    if not np.isfinite(h) or h<=0:
        raise SeamlessLatticeError("SEAMLESS_INVALID_LATTICE_SPACING")
    result=[]
    for chain,a,b in chains:
        lhs,rhs=uv[list(a)],uv[list(b)]
        candidates=[]
        tip_pairs=[idx for idx,(ai,bi) in enumerate(zip(a,b))
                   if int(ai)==int(bi)]
        for k,R in enumerate(ROT):
            if tip_pairs:
                # An uncut terminal is one *shared* UV vertex. It must
                # remain a fixed point of its own transition (a cone).
                # For a quarter turn, the fixed point must be on the
                # integer lattice; search nearby original tip positions.
                if k==0:
                    candidate_steps=[np.array([0,0],int)]
                else:
                    pivots=[]
                    for idx in tip_pairs:
                        origin=np.rint(lhs[idx]/h).astype(int)
                        for dx in (-1,0,1):
                            for dy in (-1,0,1):
                                pivot=origin+np.array([dx,dy],int)
                                pivots.append((np.eye(2,dtype=int)-R)@pivot)
                    candidate_steps=pivots
            else:
                candidate_steps=[np.rint(np.mean(
                    rhs-lhs@R.T,axis=0
                )/h).astype(int)]
            for steps in candidate_steps:
                if tip_pairs and max(
                    float(np.linalg.norm(lhs[idx]@R.T+h*steps-rhs[idx]))
                    for idx in tip_pairs
                )>max(h*1.5,1e-9):
                    # Far from the original tips is allowed only through
                    # a new physical cone/cut planning step, not by
                    # implicitly dragging a source junction across Omega.
                    continue
                difference=rhs-(lhs@R.T+h*steps)
                rms=float(np.sqrt(np.mean(np.sum(
                    difference*difference,axis=1
                ))))
                candidates.append((rms,k,tuple(map(int,steps))))
        if not candidates:
            raise SeamlessLatticeError(
                "SEAMLESS_NO_FEASIBLE_INTEGER_TIP_TRANSITION: "
                f"source_chain={chain[:12]}"
            )
        cost,k,steps=min(candidates,key=lambda x:(x[0],x[1]))
        result.append({"physical":chain,"a":a,"b":b,
                       "k":k,"steps":steps,"rms":cost})
    return result


def _global_lattice_constraints(count,transitions,h):
    """Quotient by exact lattice transforms; diagnose junction holonomy."""
    graph=[[] for _ in range(count)]
    for t in transitions:
        R=ROT[t["k"]]
        shift=h*np.asarray(t["steps"],float)
        for a,b in zip(t["a"],t["b"]):
            graph[a].append((b,R,shift))
            graph[b].append((a,R.T,-R.T@shift))
    component=np.full(count,-1,int)
    matrices=np.zeros((count,2,2),int)
    offsets=np.zeros((count,2),float)
    fixed={}
    groups=0
    tol=max(1e-9,h*1e-7)
    for start in range(count):
        if component[start]>=0:continue
        ci=groups;groups+=1
        component[start]=ci;matrices[start]=np.eye(2,dtype=int)
        pending=deque([start])
        while pending:
            a=pending.popleft()
            for b,R,t in graph[a]:
                M=R@matrices[a]
                d=R@offsets[a]+t
                if component[b]<0:
                    component[b]=ci;matrices[b]=M;offsets[b]=d
                    pending.append(b);continue
                if component[b]!=ci:
                    raise SeamlessLatticeError("SEAMLESS_INTERNAL_COMPONENT_MISMATCH")
                A=matrices[b]-M
                rhs=d-offsets[b]
                if not np.any(A):
                    if np.linalg.norm(rhs)>tol:
                        raise SeamlessLatticeError(
                            "SEAMLESS_INCONSISTENT_TRANSLATIONAL_HOLONOMY: "
                            f"vertex={b} mismatch={np.linalg.norm(rhs):.7g}"
                        )
                    continue
                if abs(float(np.linalg.det(A)))<1e-9:
                    raise SeamlessLatticeError(
                        "SEAMLESS_UNSUPPORTED_CONE_HOLONOMY: "
                        "singular transform difference at source junction"
                    )
                pivot=np.linalg.solve(A.astype(float),rhs)
                if np.linalg.norm(pivot/h-np.rint(pivot/h))>1e-6:
                    raise SeamlessLatticeError(
                        "SEAMLESS_CONE_NOT_ON_INTEGER_GRID: "
                        f"root={start} pivot={pivot.tolist()}"
                    )
                if ci in fixed and np.linalg.norm(fixed[ci]-pivot)>tol:
                    raise SeamlessLatticeError(
                        "SEAMLESS_INCONSISTENT_CONE_ANCHORS"
                    )
                fixed[ci]=pivot
    return component,matrices,offsets,fixed


def solve_seamless_lattice_uv(chart,h,*,edge_weight=.3):
    """Solve chart UVs with EXACT paired-grid transition identities.

    Optimizes original UV vertex positions and edge vectors over the
    independent 2D root positions by sparse least-squares. Guards signed
    area; no accepted foldovers or arbitrary seam straightening.
    """
    old=np.asarray(chart.uv_vertices_2d,float)
    faces=np.asarray(chart.uv_faces,int)
    if old.ndim!=2 or old.shape[1]!=2 or len(old)==0:
        raise SeamlessLatticeError("SEAMLESS_INVALID_UV_CHART")
    chains=source_chains(chart)
    if not chains:
        raise SeamlessLatticeError("SEAMLESS_NO_PHYSICAL_OPT CUT_CHAIN".replace(" ","_"))
    # Source-chain transforms must be selected jointly. The former
    # independent rounding of every chain had no holonomy preflight:
    # two individually good integer shifts can sum to a NONZERO loop.
    from .optcuts_seamless_joint_transitions import select_joint_transitions
    transitions,joint_report=select_joint_transitions(old,chains,h)
    # Re-check the complete (uncompressed) source UV graph before solving.
    # This independent audit also catches accidental solver regressions.
    comp,mat,offset,fixed=_global_lattice_constraints(
        len(old),transitions,h
    )
    free=sorted(set(comp.tolist())-set(fixed))
    columns={g:2*i for i,g in enumerate(free)}
    if not free:raise SeamlessLatticeError("SEAMLESS_NO_FREE_UV_VARIABLES")
    rows=[];cols=[];values=[];rhs=[]
    def formula(i,d):
        g=int(comp[i])
        bias=float(offset[i,d])
        if g in fixed:
            return {},bias+float(mat[i,d]@fixed[g])
        base=columns[g]
        return {base:float(mat[i,d,0]),base+1:float(mat[i,d,1])},bias
    def insert(coef,value,weight):
        index=len(rhs)
        for c,x in coef.items():
            if abs(x)>1e-13:
                rows.append(index);cols.append(c);values.append(weight*x)
        rhs.append(float(weight*value))
    for i in range(len(old)):
        for d in (0,1):
            terms,bias=formula(i,d)
            insert(terms,old[i,d]-bias,1.)
    edges=set()
    for face in faces:
        for a,b in ((0,1),(1,2),(2,0)):
            edges.add(tuple(sorted((int(face[a]),int(face[b])))))
    for i,j in sorted(edges):
        for d in (0,1):
            a,oa=formula(i,d);b,ob=formula(j,d)
            coeff=defaultdict(float)
            for k,v in a.items():coeff[k]+=v
            for k,v in b.items():coeff[k]-=v
            insert(coeff,(old[i,d]-old[j,d])-(oa-ob),float(edge_weight)**.5)
    A=sparse.coo_matrix((values,(rows,cols)),
         shape=(len(rhs),2*len(free))).tocsr()
    roots=lsmr(A,np.asarray(rhs),atol=1e-10,btol=1e-10,
               maxiter=max(200,4*len(free)))[0]
    new=np.zeros_like(old)
    for i in range(len(old)):
        g=int(comp[i])
        base=fixed[g] if g in fixed else roots[
            columns[g]:columns[g]+2]
        new[i]=mat[i]@base+offset[i]
    residual=max(float(np.linalg.norm(
        new[b]-(new[a]@ROT[t["k"]].T+h*np.asarray(t["steps"]))
    )) for t in transitions for a,b in zip(t["a"],t["b"]))
    if residual>max(1e-8,h*1e-7):
        raise SeamlessLatticeError(
            f"SEAMLESS_NONZERO_TRANSITION_RESIDUAL: {residual:.9g}"
        )
    def signed_areas(points):
        tri=points[faces]
        u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0]
        return u[:,0]*v[:,1]-u[:,1]*v[:,0]
    before,after=signed_areas(old),signed_areas(new)
    if np.any(abs(before)<1e-14):
        raise SeamlessLatticeError("SEAMLESS_DEGENERATE_INPUT_UV")
    ratios=after/before
    if np.min(ratios)<=1e-3:
        bad=np.flatnonzero(ratios<=1e-3)
        raise SeamlessLatticeError(
            "SEAMLESS_UV_FLIP_OR_COLLAPSE: "
            f"count={len(bad)} sample={bad[:12].tolist()} "
            f"min_signed_ratio={ratios.min():.7g}; no fallback"
        )
    return new,{
        "seamless_lattice_mode_enabled":True,
        **joint_report,
        "seamless_lattice_source_chain_count":len(transitions),
        "seamless_lattice_transition_residual_max":float(residual),
        "seamless_lattice_transitions":[{
            "k":int(t["k"]),"shift_grid_steps":list(t["steps"]),
            "source_chain":list(t["physical"]),
            "uv_pairs":len(t["a"]),"initial_rms":float(t["rms"])
        } for t in transitions],
        "seamless_lattice_cone_count":len(fixed),
        "seamless_lattice_source_relative_area_min":float(ratios.min()),
        "seamless_lattice_uv_movement_rms":float(np.sqrt(
            np.mean(np.sum((new-old)**2,axis=1)))),
        "seamless_lattice_staircase_snap_applied":False,
        "seamless_lattice_m2d_coverage_proven":False,
        "seamless_lattice_final_seam_gap_proven":False,
    }


def _audit_actual_seamless_m2d(mesh,domain,params):
    """Measure 2D Omega area captured by actual (unchanged) M2D cells.

    A lattice UV transition DOES NOT automatically create 3D coverage:
    the existing strict full-cell crop may miss seam-crossing cells.
    Report this, never label it an exact watertight Seam closure.
    """
    chart=getattr(domain,"parameterization",None)
    if chart is None:
        raise SeamlessLatticeError("SEAMLESS_M2D_MISSING_OMEGA_CHART")
    from .optcuts_area_grid_boundary import _intersection_ratios
    v=np.asarray(domain.uv_vertices,float)[:,:2]
    nx=int(domain.overlay_nx);ny=int(domain.overlay_ny)
    if len(v)!=(nx+1)*(ny+1):
        raise SeamlessLatticeError("SEAMLESS_M2D_OVERLAY_VERTEX_COUNT_MISMATCH")
    origin=v[0].copy()
    step=np.asarray([v[1,0]-v[0,0],
         v[nx+1,1]-v[0,1]],float)
    if np.any(step<=0):
        raise SeamlessLatticeError("SEAMLESS_M2D_INVALID_GRID_STEP")
    coverage=_intersection_ratios(
        np.asarray(chart.uv_vertices_2d,float),
        np.asarray(chart.uv_faces,int),
        origin,step,nx,ny,
    )
    occupied=np.zeros((ny,nx),bool)
    all_v=np.asarray(mesh.vertices,float)
    faces=np.asarray(mesh.faces,int)
    for f in faces:
        center=np.mean(all_v[f,:2],axis=0)
        ij=np.floor((center-origin)/step+1e-8).astype(int)
        i,j=int(ij[0]),int(ij[1])
        if i<0 or i>=nx or j<0 or j>=ny:
            raise SeamlessLatticeError(
                "SEAMLESS_M2D_CELL_OUTSIDE_ORIGINAL_GRID"
            )
        occupied[j,i]=True
    denominator=max(float(np.sum(coverage)),1e-20)
    ratio=float(np.sum(coverage[occupied])/denominator)
    mesh.metrics["seamless_lattice_m2d_source_uv_area_coverage"]=ratio
    mesh.metrics["seamless_lattice_m2d_source_uv_uncovered_area_fraction"]=float(
        1.-ratio
    )
    mesh.metrics["seamless_lattice_m2d_source_uv_coverage_pass_98pct"]=bool(
        ratio>=.98
    )
    mesh.metrics["seamless_lattice_m2d_3d_area_coverage_verified"]=False
    mesh.metrics["seamless_lattice_m2d_selected_cell_count"]=int(
        np.sum(occupied)
    )
    print(
        "[OMEGA-SEAMLESS-M2D-AUDIT] "
        f"strict_full_cells={int(np.sum(occupied))} "
        f"source_uv_area_coverage={ratio:.3%} "
        "3D_coverage=UNMEASURED exact_seam=UNPROVEN",
        flush=True,
    )
    return mesh


def install_seamless_lattice_mode_patch(pipeline:Any):
    if getattr(pipeline,"_onestring_seamless_lattice_mode_installed",False):
        return
    base=pipeline._build_surface_parameterization
    base_m2d=pipeline._build_m2d
    def seamless_builder(surface,target,grid,params):
        chart=base(surface,target,grid,params)
        if not bool(getattr(params,"optcuts_seamless_lattice_enabled",False)):
            return chart
        if str(getattr(chart,"method",""))!="optcuts_test":
            raise SeamlessLatticeError("SEAMLESS_MUST_USE_OPTCUTS_TEST_CARRIER")
        h=float(getattr(params,"tile_size",getattr(grid,"tile_size",0.)))
        uv,report=solve_seamless_lattice_uv(chart,h)
        chart.uv_vertices_2d=uv
        from .optcuts_backend import _boundary_loops
        loops=_boundary_loops(np.asarray(chart.uv_faces,int))
        if len(loops)!=1:
            raise SeamlessLatticeError(
                f"SEAMLESS_REQUIRES_ONE_DISK_CHART: loops={len(loops)}"
            )
        loop=[int(i) for i in loops[0]]
        chart.omega_boundary=uv[loop+[loop[0]]]
        chart.triangle_acceleration=None
        chart.metrics.update(report)
        chart.metrics["requested_omega_parameterization_mode"]="optcuts_seamless_lattice"
        chart.metrics["boundary_loop"]=loop
        print(
            "[OMEGA-SEAMLESS-LATTICE] "
            f"chains={report['seamless_lattice_source_chain_count']} "
            f"UV_cut_residual={report['seamless_lattice_transition_residual_max']:.3g} "
            "staircase_snap=OFF M2D_coverage=UNVERIFIED",
            flush=True,
        )
        return chart
    def seamless_m2d(grid,domain,params=None):
        mesh=base_m2d(grid,domain,params)
        if bool(getattr(params,"optcuts_seamless_lattice_enabled",False)):
            return _audit_actual_seamless_m2d(mesh,domain,params)
        return mesh
    pipeline._build_surface_parameterization=seamless_builder
    pipeline._build_m2d=seamless_m2d
    original=getattr(pipeline,"_original",None)
    if original is not None:
        original._build_surface_parameterization=seamless_builder
        original._build_m2d=seamless_m2d
    for fn in (
        getattr(pipeline,"build_onestring_design",None),
        getattr(pipeline,"_ORIGINAL_BUILD_ONESTRING_DESIGN",None),
        getattr(original,"build_onestring_design",None)
        if original is not None else None,
    ):
        glb=getattr(fn,"__globals__",None)
        if isinstance(glb,dict):
            glb["_build_surface_parameterization"]=seamless_builder
            glb["_build_m2d"]=seamless_m2d
    pipeline._onestring_seamless_lattice_mode_installed=True
