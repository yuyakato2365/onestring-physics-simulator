"""OneString Section 5.4 / Supplement Appendix E actuation reference mode.

Paper-matched ACTUATION constraints, not authors' numerical solver:
  - start at actual T2D Dual Hinge eight-vertex panels;
  - position/velocity prediction with gravity (Appendix E);
  - Snap: paired *side face midpoints* on the optimized closed string route;
  - Lift: the designated GPE lift points toward their assembled 3D positions;
  - rigid panel Kabsch projection, hinge links and 3D collision projections.
No extra T3D tile pose fitting, no artificial per-gap target contraction,
and no global shortened virtual rope. String friction affects route selection,
NOT the simulation as an invented capstan force.

The authors' exact PD weights, global solver and continuous contact treatment
are not fully specified/reproduced here. The local/global sweeps and discrete
15-axis OBB SAT are explicitly tagged approximations; contact is not IPC.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from .native_string_deployment import _contacts, _floor_constraint, _vertices, _physical_frame


class PaperPDInputError(ValueError):
    pass


@dataclass(frozen=True)
class _Snap:
    gap_id: int
    a: int
    b: int
    face_a: tuple[int, int, int, int]
    face_b: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Lift:
    gap_id: int
    tiles: tuple[int, ...]
    faces: tuple[tuple[int, ...], ...]
    reference: np.ndarray
    target: np.ndarray


def _closest_side_face(rest_tile: np.ndarray, gap_xy: np.ndarray) -> tuple[int, int, int, int]:
    """Identify the actual T2D side face closest to the void/gap centroid.

    A gap in the routing graph is *not itself* a physical string channel.
    This is a geometry-based side-face proxy and is reported as such.
    """
    corners = np.asarray(rest_tile[:4, :2], float)
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    def dist(pair):
        p, q = corners[list(pair)]
        v = q - p
        t = float(np.clip(np.dot(gap_xy - p, v) / max(np.dot(v, v), 1e-16), 0., 1.))
        return float(np.linalg.norm(p + t*v - gap_xy))
    ia, ib = min(pairs, key=dist)
    return (ia, ib, ib + 4, ia + 4)


def _validate_route(state) -> tuple[list[_Snap], list[_Lift], dict]:
    graph = state.gap_graph
    route = state.string_path
    route_meta = getattr(route, "metrics", {})
    graph_meta = getattr(graph, "metrics", {})
    if not graph_meta.get("paper_section_5_3_graph", False):
        raise PaperPDInputError(
            "PAPER_PD_ROUTE_NOT_SECTION53: regenerate the explicit-corner "
            "gap graph and optimized paper Section 5.3 string route"
        )
    if route_meta.get("string_path_model") != "Sec. 5.3 minimum-turn closed-cycle MILP":
        raise PaperPDInputError("PAPER_PD_ROUTE_NOT_MILP: Section 5.3 routing is required")
    if route_meta.get("string_path_valid") is not True:
        raise PaperPDInputError(
            f"PAPER_PD_INVALID_ROUTE: {route_meta.get('string_path_status','unknown')}"
        )
    ids = list(map(int, route.gap_ids))
    if len(ids) < 4 or ids[0] != ids[-1] or len(set(ids[:-1])) != len(ids)-1:
        raise PaperPDInputError("PAPER_PD_ROUTE_OPEN_OR_REPEATED: expected a simple closed cycle")
    if graph_meta.get("split_boundary_missing_pairs"):
        raise PaperPDInputError("PAPER_PD_BROKEN_SPLIT_BRIDGE")
    gaps = {int(x.id): x for x in graph.gaps}
    if any(idx not in gaps for idx in ids):
        raise PaperPDInputError("PAPER_PD_ROUTE_UNKNOWN_GAP")
    boundaries = {int(g.id) for g in graph.gaps if g.boundary}
    if not boundaries.intersection(ids):
        raise PaperPDInputError("PAPER_PD_NO_BOUNDARY_ANCHOR")
    n = len(state.tiles_2d_dual_hinge.vertices)
    rest = np.asarray(state.tiles_2d_dual_hinge.vertices, float)
    snaps = []
    for gid in ids[:-1]:
        gap = gaps[gid]
        if gap.boundary:
            continue
        tiles = sorted(set(map(int, gap.surrounding_tiles)))
        if len(tiles) != 2:
            # A non-two-panel internal gap is not a paired side-face snap.
            # Do not invent a contraction between arbitrary bodies.
            continue
        a, b = tiles
        if min(a,b) < 0 or max(a,b) >= n:
            raise PaperPDInputError(f"PAPER_PD_BAD_GAP_OWNER: {gid}")
        xy = np.asarray(gap.centroid_2d, float)[:2]
        snaps.append(_Snap(gid,a,b,_closest_side_face(rest[a],xy),
                           _closest_side_face(rest[b],xy)))
    if not snaps:
        raise PaperPDInputError("PAPER_PD_NO_SNAP_GAPS: routed cycle lacks paired internal side faces")
    lifts = []
    route_set = set(ids[:-1])
    for point in state.lift_points:
        gid = int(point.gap_id)
        if gid not in route_set:
            raise PaperPDInputError(f"PAPER_PD_LIFT_NOT_ON_ROUTE: gap {gid}")
        gap = gaps[gid]
        owned = tuple(sorted(set(map(int,gap.surrounding_tiles))))
        if not owned or min(owned) < 0 or max(owned) >= n:
            raise PaperPDInputError(f"PAPER_PD_INVALID_LIFT_OWNER: gap {gid}")
        xy = np.asarray(gap.centroid_2d,float)[:2]
        faces = tuple(_closest_side_face(rest[t],xy)[:2] for t in owned)
        reference = np.mean(
            [rest[t,list(face)].mean(axis=0) for t,face in zip(owned,faces)],axis=0
        )
        target = np.asarray(point.position_3d,float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise PaperPDInputError(f"PAPER_PD_INVALID_LIFT_TARGET: gap {gid}")
        lifts.append(_Lift(gid,owned,faces,reference,target))
    if not lifts:
        raise PaperPDInputError("PAPER_PD_NO_LIFT_POINTS: Section 5.2 has no selected lift gaps")
    return snaps,lifts,dict(
        routing_model=route_meta["string_path_model"],
        snap_geometry="nearest actual T2D side-face midpoints; geometry proxy",
        lift_geometry="nearest side top-edge midpoint; geometry proxy",
        route_nodes=len(ids)-1,route_closed=True,
        routed_snap_gaps=len(snaps),routed_lift_gaps=len(lifts),
        route_optimization_status=route_meta.get("string_path_status","unknown"),
    )


def _rigid_fit(rest: np.ndarray, proposed: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Batch SO(3) Kabsch projection onto the exact T2D rest geometry."""
    centers0=rest.mean(axis=1)
    centers=proposed.mean(axis=1)
    local=rest-centers0[:,None,:]
    shifted=proposed-centers[:,None,:]
    cov=np.einsum("nki,nkj->nij",local,shifted)
    u,_,vt=np.linalg.svd(cov)
    reflection=np.ones((len(rest),3))
    reflection[:,-1]=np.where(np.linalg.det(u@vt)>=0.,1.,-1.)
    q=np.einsum("nik,nk,nkj->nij",u,reflection,vt)  # row-vector transform
    placed=np.einsum("nki,nij->nkj",local,q)+centers[:,None,:]
    return placed,centers,np.transpose(q,(0,2,1))


def _project_hinges(p: np.ndarray, links: list, weight: float) -> None:
    for hinge in links:
        a,b=int(hinge.tile_a),int(hinge.tile_b)
        if not (0<=a<len(p) and 0<=b<len(p)) or a==b:
            continue
        va,vb=int(hinge.local_vertex_a),int(hinge.local_vertex_b)
        if str(getattr(hinge,"surface","top"))=="bottom":
            if va<4: va+=4
            if vb<4: vb+=4
        if not (0<=va<8 and 0<=vb<8):
            raise PaperPDInputError("PAPER_PD_HINGE_VERTEX_OUT_OF_RANGE")
        delta = (p[b,vb]-p[a,va])*float(weight)*.5
        p[a,va]+=delta
        p[b,vb]-=delta


def _project_snaps(p: np.ndarray, snaps: list[_Snap], weight: float) -> None:
    for snap in snaps:
        pa=p[snap.a,list(snap.face_a)].mean(axis=0)
        pb=p[snap.b,list(snap.face_b)].mean(axis=0)
        delta=(pb-pa)*(.5*weight)
        # Projection gradient of the side-face midpoint: distribute equally
        # over the FOUR vertices defining the physical side face.
        p[snap.a,list(snap.face_a)] += delta
        p[snap.b,list(snap.face_b)] -= delta


def _project_lifts(p: np.ndarray, lifts: list[_Lift], phase: float, weight: float) -> None:
    for lift in lifts:
        desired=(1.-phase)*lift.reference+phase*lift.target
        current=np.mean(
            [p[t,list(face)].mean(axis=0) for t,face in zip(lift.tiles,lift.faces)],axis=0
        )
        delta=(desired-current)*weight/len(lift.tiles)
        for tile,face in zip(lift.tiles,lift.faces):
            p[tile,list(face)] += delta


def _snap_rms(p:np.ndarray, snaps:list[_Snap])->float:
    values=[float(np.linalg.norm(p[x.a,list(x.face_a)].mean(axis=0)
                                -p[x.b,list(x.face_b)].mean(axis=0)))**2 for x in snaps]
    return float(np.sqrt(np.mean(values))) if values else 0.


def _lift_rms(p:np.ndarray,lifts:list[_Lift])->float:
    values=[]
    for x in lifts:
        pos=np.mean([p[t,list(f)].mean(axis=0) for t,f in zip(x.tiles,x.faces)],axis=0)
        values.append(float(np.sum((pos-x.target)**2)))
    return float(np.sqrt(np.mean(values))) if values else 0.


def _hinge_rms(p:np.ndarray,links:list)->float:
    e=[]
    for h in links:
        va,vb=int(h.local_vertex_a),int(h.local_vertex_b)
        if str(getattr(h,"surface","top"))=="bottom":
            if va<4: va+=4
            if vb<4: vb+=4
        if 0<=va<8 and 0<=vb<8 and h.tile_a!=h.tile_b:
            e.append(float(np.sum((p[h.tile_a,va]-p[h.tile_b,vb])**2)))
    return float(np.sqrt(np.mean(e))) if e else 0.


def _contact_proxy(rest:np.ndarray,p:np.ndarray):
    """Rigid 3D OBB proxy with rotational collision response (discrete)."""
    _,centers,rot=_rigid_fit(rest,p)
    local=rest-rest.mean(axis=1)[:,None,:]
    moments=np.zeros((len(rest),3,3))
    axes=np.asarray([_physical_frame(tile[:4]) for tile in local])
    for k in range(len(rest)):
        inertia=(np.sum(local[k]*local[k])*np.eye(3)-local[k].T@local[k])/8
        moments[k]=np.linalg.pinv(inertia,rcond=1e-8)
    m=SimpleNamespace(
        rest=rest,centers=centers.copy(),local=local,rotations=rot.copy(),
        inv_inertia_local=moments,axes_local=axes,
        circumradius=np.max(np.linalg.norm(local,axis=2),axis=1),
        tile_scale=float(np.median(np.linalg.norm(rest[:,1]-rest[:,0],axis=1))),
        floor_z=float(np.min(rest[:,:,2])),
    )
    return m


def simulate_paper_pd(state,params,progress_callback=None):
    """Appendix-E Snap/Lift actuation with a disclosed approximate PD local solver."""
    from .legacy_pipeline_base import DeploymentResult

    start=time.perf_counter()
    initial=np.asarray(state.tiles_2d_dual_hinge.vertices,float)
    target=np.asarray(state.tiles_3d.vertices,float)
    if initial.ndim!=3 or initial.shape[1:]!=(8,3) or initial.shape!=target.shape:
        raise PaperPDInputError("PAPER_PD_INVALID_T2D_T3D: expected corresponding (N,8,3) panels")
    if not np.isfinite(initial).all():
        raise PaperPDInputError("PAPER_PD_NONFINITE_PANEL")
    snaps,lifts,route_info=_validate_route(state)
    iterations=max(1,int(params.solver_iterations))
    steps=max(2,int(params.steps))
    fps=max(2,int(params.max_animation_frames))
    dt=float(getattr(params,"paper_pd_timestep",0.015))
    if not (0<dt<=.05):
        raise PaperPDInputError("PAPER_PD_BAD_TIMESTEP")
    gamma=float(np.clip(params.damping_ratio,0,.99))
    gravity=np.array([0.,0.,-float(max(0.,params.gravity))])
    rest=initial.copy()
    current=initial.copy()
    velocity=np.zeros_like(current)
    frames=[current.copy()]
    save_at=set(np.unique(np.rint(np.linspace(0,steps,fps)).astype(int))[1:])
    links=list(state.hinge_graph.hinges)
    c0=_contact_proxy(rest,current)
    collisions=[_contacts(c0,apply=False,strength=0.)]
    hinge_curve=[_hinge_rms(current,links)]
    snap_curve=[_snap_rms(current,snaps)]
    lift_curve=[_lift_rms(current,lifts)]
    rigid_err_curve=[0.]
    contact_strength=float(np.clip(params.collision_weight,0.,1.))
    for i in range(steps):
        phase=(i+1)/steps
        # Appendix E: predicted location = x + dt*v + dt²*a.
        predicted=current+dt*velocity+(dt**2)*gravity
        p=predicted.copy()
        for j in range(iterations):
            # PD-like local projections of explicit Appendix E energies.
            _project_hinges(p,links,float(np.clip(params.hinge_weight,0.,1.)))
            _project_snaps(p,snaps,float(np.clip(params.snap_weight,0.,1.))*phase)
            _project_lifts(p,lifts,phase,float(np.clip(params.lift_weight,0.,1.)))
            p,_,_=_rigid_fit(rest,p)
            if contact_strength>0.:
                m=_contact_proxy(rest,p)
                _contacts(m,apply=True,strength=contact_strength)
                _floor_constraint(m)
                p=_vertices(m)
        p,_,_=_rigid_fit(rest,p)
        # Appendix E reports a damped velocity update with external acceleration.
        # No arbitrary target-pose attraction or explicit string-length contraction.
        velocity=(1.-gamma)*velocity+dt*gravity
        current=p
        if i+1 in save_at:
            frames.append(current.copy())
            m=_contact_proxy(rest,current)
            collisions.append(_contacts(m,apply=False,strength=0.))
            hinge_curve.append(_hinge_rms(current,links))
            snap_curve.append(_snap_rms(current,snaps))
            lift_curve.append(_lift_rms(current,lifts))
            rigid_err_curve.append(float(np.max(np.abs(
                np.linalg.norm(current[:,1]-current[:,0],axis=1)
                -np.linalg.norm(rest[:,1]-rest[:,0],axis=1)))))
        if progress_callback and (i%max(1,steps//25)==0 or i+1==steps):
            progress_callback("Paper Appendix E Snap/Lift PD", (i+1)/steps,
                              f"step {i+1}/{steps}: snapped gaps {len(snaps)}, lift gaps {len(lifts)}")
    if not np.array_equal(frames[-1],current):
        frames.append(current.copy())
        mm=_contact_proxy(rest,current)
        collisions.append(_contacts(mm,apply=False,strength=0.))
        hinge_curve.append(_hinge_rms(current,links))
        snap_curve.append(_snap_rms(current,snaps))
        lift_curve.append(_lift_rms(current,lifts))
    scale=float(np.median(np.linalg.norm(initial[:,1]-initial[:,0],axis=1)))
    hinge_error=hinge_curve[-1]
    snap_error=snap_curve[-1]
    lift_error=lift_curve[-1]
    final_contacts=collisions[-1]
    valid=(final_contacts==0 and hinge_error<.05*scale
           and snap_error<.05*scale and lift_error<.10*scale)
    elapsed=time.perf_counter()-start
    result=DeploymentResult(
        frames=frames,final_tiles=current.copy(),collision_counts=collisions,
        metrics={
            "physics_backend":"paper_pd",
            "actual_backend":"cpu_paper_appendix_e_projective_approx",
            "simulation_model":"Sec5.4_SuppE_snap_lift_PD",
            "paper_alignment":"actuation objectives and topology paper-aligned; numerical solver/contact approximated",
            "initial_layout":"exact_T2D_Dual",
            "route_model":route_info["routing_model"],
            "route_optimization_status":route_info["route_optimization_status"],
            "route_closed":True,
            "snap_midpoint_model":route_info["snap_geometry"],
            "lift_point_model":route_info["lift_geometry"],
            "actuated_snap_gap_count":len(snaps),
            "actuated_lift_gap_count":len(lifts),
            "string_model":"geometric_Snap_Lift_constraints_no_rope_dynamics",
            "target_fit_weight":0.,
            "target_contact_guard_weight":0.,
            "t3d_target_tile_pose_actuator":False,
            "time_integration":"Appendix_E_prediction_and_velocity_update",
            "collision_model":"discrete_3D_15_axis_OBB_SAT_rotation_correction_not_IPC",
            "collision_count":int(final_contacts),
            "initial_collision_count":int(collisions[0]),
            "hinge_error":hinge_error,"snap_error":snap_error,"lift_error":lift_error,
            "rigid_error":float(max(rigid_err_curve) if rigid_err_curve else 0.),
            "final_deployment_error_to_T3D":float(np.sqrt(np.mean((current-target)**2))),
            "stable_state":bool(valid),
            "status":"constraints_satisfied" if valid else "ended_with_constraint_residuals",
            "solver_steps":steps,"solver_iterations":iterations,"timestep":dt,
            "elapsed_time":elapsed,"simulation_runtime_seconds":elapsed,
            "paper_target_pose_forced":False,
        }
    )
    result.hinge_error_history=hinge_curve
    result.snap_error_history=snap_curve
    result.lift_error_history=lift_curve
    return result
