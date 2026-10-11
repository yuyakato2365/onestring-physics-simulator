"""Bounded Section 4.4 rigid pose solve for the reconstructed Eq.5 route.

Rigidity is exact (SE(2) poses of the full 8-vertex solid); connections use both
paired vertices' squared residuals. Collision is the conservative 2D convex
footprint SAT surrogate, not the paper's 3D nonpenetration implementation.
No artificial positive clearance is imposed on a coincident hinge.
"""
from __future__ import annotations

import copy
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from .paper_eq5_k2d_solver import collision_energy_gradient


class _Deadline(Exception):
    pass


def optimize_hinge_poses(grid, mesh_faces, t2d, t3d, params, pipeline, progress_callback=None):
    started=time.perf_counter()
    from .paper_t2d import build_hinge_graph, validate_flat_linkage
    from .solid_collision_audit import audit_solid_collisions
    graph=build_hinge_graph(t2d,t3d,pipeline,dual=True)
    rest=np.asarray(t2d.vertices,float)
    centers=rest[:,:4,:2].mean(axis=1)
    local=rest[:,:,:2]-centers[:,None,:]
    scale=float(np.median(np.linalg.norm(np.roll(rest[:,:4,:2],-1,axis=1)-rest[:,:4,:2],axis=2)))
    # Hull vertex IDs stay fixed under rigid motion. Pad repeated final vertices;
    # zero-length padding edges are excluded from SAT's axes.
    hulls=[ConvexHull(tile[:,:2]).vertices.tolist() for tile in rest]
    width=max(map(len,hulls))
    faces=np.array([[8*i+j for j in h+[h[-1]]*(width-len(h))] for i,h in enumerate(hulls)],int)
    ha=np.array([8*h.tile_a+h.local_vertex_a for h in graph.hinges],int)
    hb=np.array([8*h.tile_b+h.local_vertex_b for h in graph.hinges],int)
    # Contact has zero SAT depth already. A hinge does not authorize positive
    # penetration of the two tiles; use the same all-pair policy as validation.
    wconn=float(params.hinge_layout_connection_weight)
    wcoll=float(params.hinge_layout_collision_weight)
    wanchor=float(params.hinge_layout_anchor_weight)
    if min(wconn,wcoll,wanchor)<0 or not np.isfinite([wconn,wcoll,wanchor]).all():
        raise ValueError('Hinge weights must be finite and nonnegative')
    budget=float(params.hinge_layout_time_budget_sec)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError('hinge_layout_time_budget_sec must be finite and nonnegative (0 = unlimited)')
    # The time budget applies to the numerical solve, not to graph construction,
    # initial collision audits or the first solver-history snapshot. In large
    # meshes those preparations can otherwise consume the entire short budget.
    deadline=float('inf')
    # Stabilization controls are part of the actual dual-hinge solve.  Expansion
    # is applied to tile centers before optimization; max drift is enforced with
    # L-BFGS-B translation bounds around that expanded reference.  This makes
    # the UI controls authoritative instead of display-only.
    initial_expansion=float(getattr(params,'hinge_layout_initial_expansion',1.0))
    max_center_drift_tiles=float(getattr(params,'hinge_layout_max_center_drift_tiles',2.0))
    max_candidate_pairs=int(getattr(params,'hinge_layout_max_candidate_pairs',3000))
    if not np.isfinite(initial_expansion) or initial_expansion<=0:
        raise ValueError('hinge_layout_initial_expansion must be finite and positive')
    if not np.isfinite(max_center_drift_tiles) or max_center_drift_tiles<0:
        raise ValueError('hinge_layout_max_center_drift_tiles must be finite and nonnegative')
    if max_candidate_pairs<=0:
        raise ValueError('hinge_layout_max_candidate_pairs must be positive')
    layout_center=centers.mean(axis=0)
    expanded_centers=layout_center+initial_expansion*(centers-layout_center)
    center_offsets=expanded_centers-centers
    # Keep the objective, reference placement and translation bounds identical
    # across ALL restarts. Only the initial SE(2) pose is perturbed.
    reference_pose=np.zeros((len(rest),3),float)
    reference_pose[:,1:]=center_offsets/max(scale,1e-30)
    drift_bound=max_center_drift_tiles
    collision_mode=str(getattr(params, 'hinge_layout_collision_mode', 'soft'))
    if collision_mode not in ('soft', 'hard'):
        raise ValueError(f'Unsupported hinge_layout_collision_mode: {collision_mode}')
    search_mode=str(getattr(params, 'hinge_layout_search_mode', 'single'))
    if search_mode not in ('single', 'multi_start'):
        raise ValueError(f'Unsupported hinge_layout_search_mode: {search_mode}')
    restarts=int(getattr(params, 'hinge_layout_restart_count', 6)) if search_mode == 'multi_start' else 0
    if restarts < 0 or restarts > 50:
        raise ValueError('hinge_layout_restart_count must be between 0 and 50')
    translation_sigma=float(getattr(params, 'hinge_layout_perturb_translation', 0.35))
    rotation_sigma=float(getattr(params, 'hinge_layout_perturb_rotation_deg', 20.0))
    seed=int(getattr(params, 'hinge_layout_random_seed', 0))
    if not np.isfinite([translation_sigma, rotation_sigma]).all() or translation_sigma < 0 or rotation_sigma < 0:
        raise ValueError('Hinge perturbation scales must be finite and nonnegative')
    rotation_sigma=np.deg2rad(rotation_sigma)
    rng=np.random.default_rng(seed)
    history_stride=max(1, int(np.ceil(max(1, params.hinge_layout_iterations)/180)))
    trial_summaries=[]
    trial_trajectories=[]
    def positions(pose):
        angle=pose[:,0];c=np.cos(angle)[:,None];s=np.sin(angle)[:,None]
        rotated=np.stack((c*local[:,:,0]-s*local[:,:,1],s*local[:,:,0]+c*local[:,:,1]),axis=2)
        xy=rotated+centers[:,None,:]+scale*pose[:,None,1:]
        return xy,rotated
    def evaluate(pose):
        xy,rotated=positions(pose)
        flat=xy.reshape(-1,2)
        ec,g,ncoll,collision_stats=collision_energy_gradient(flat,faces,scale*1e-8,return_diagnostics=True)
        g*=wcoll
        delta=flat[ha]-flat[hb]
        dz=rest.reshape(-1,3)[ha,2]-rest.reshape(-1,3)[hb,2]
        conn=2*float(np.sum(delta*delta)+dz@dz)
        np.add.at(g,ha,4*wconn*delta);np.add.at(g,hb,-4*wconn*delta)
        g=g.reshape(-1,8,2)
        gradient=np.zeros_like(pose)
        gradient[:,0]=np.sum(g[:,:,0]*(-rotated[:,:,1])+g[:,:,1]*rotated[:,:,0],axis=1)
        gradient[:,1:]=scale*np.sum(g,axis=1)
        translation_delta=pose[:,1:]-reference_pose[:,1:]
        anchor=scale**2*float(np.sum(translation_delta**2))
        gradient[:,1:]+=2*wanchor*scale**2*translation_delta
        energy=wconn*conn+wcoll*ec+wanchor*anchor
        errors=np.sqrt(np.sum(delta*delta,axis=1)+dz*dz)
        stats=dict(EHinge=energy,EConn=conn,ECollision=ec,EAnchor=anchor,ERigid=0.,
                   wConnEConn=wconn*conn,wCollisionECollision=wcoll*ec,wAnchorEAnchor=wanchor*anchor,
                   rigid_constraint='exact SE(2) poses; no soft weight',
                   collisions=ncoll,**collision_stats,
                   hinge_mean=float(errors.mean()) if len(errors) else 0.,
                   hinge_rms=float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.,
                   hinge_p95=float(np.percentile(errors,95)) if len(errors) else 0.,
                   hinge_max=float(errors.max()) if len(errors) else 0.,
                   hinge_fixed_z_error_lower_bound=float(np.max(np.abs(dz))) if len(dz) else 0.)
        return energy,gradient,stats
    def fun(z):
        if time.perf_counter()>=deadline:
            raise _Deadline()
        e,g,_=evaluate(z.reshape(-1,3))
        return e/(scale**2),(g/(scale**2)).ravel()
    # The exact same objective and bounds are reused for every local solve.
    bounds=[]
    for i in range(len(rest)):
        bounds.append((None,None))
        bounds.append((reference_pose[i,1]-drift_bound,reference_pose[i,1]+drift_bound))
        bounds.append((reference_pose[i,2]-drift_bound,reference_pose[i,2]+drift_bound))
    setup_elapsed=time.perf_counter()-started
    optimization_started=time.perf_counter()
    global_deadline=optimization_started+budget if budget>0 else float('inf')
    # Hard mode reserves a large part of the wall-clock budget to restore
    # nonpenetration. Soft multi-start alone must not consume the entire budget.
    soft_deadline=(optimization_started+0.55*budget
                   if collision_mode=='hard' and budget>0 else global_deadline)

    def run_trial(initial_pose, trial_id, source, trial_deadline):
        nonlocal deadline
        deadline=trial_deadline
        current=initial_pose.copy()
        records=[]
        frames=[]
        trial_started=time.perf_counter()

        def record(pose):
            _,_,stats=evaluate(pose)
            if records and stats['EHinge'] > records[-1]['EHinge'] + 1e-10*scale**2:
                raise RuntimeError('Hinge solver accepted an increasing objective within a trial')
            records.append(dict(iteration=len(records),**stats))
            if (len(records)-1)%history_stride == 0:
                frames.append((len(records)-1,pose.copy()))
            return stats

        initial_stats=record(current)

        def callback(z):
            nonlocal current
            current=z.reshape(-1,3).copy()
            stats=record(current)
            if progress_callback and (len(records)%10==0 or len(records)==2):
                progress_callback('Eq.6 rigid tile poses',min(.99,(len(records)-1)/max(1,params.hinge_layout_iterations)),
                                  f"trial {trial_id}/{restarts}; {source}; iter {len(records)-1}; EHinge={stats['EHinge']:.5g}; collisions={stats['collisions']}")

        success=False
        status=None
        nfev=0
        nit=0
        timed_out=False
        message='not started'
        try:
            result=minimize(fun,current.ravel(),jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
                            options=dict(maxiter=max(1,params.hinge_layout_iterations),maxls=30,maxcor=20,ftol=1e-13,gtol=1e-8))
            proposed=result.x.reshape(-1,3)
            success=bool(result.success)
            status=int(result.status)
            nfev=int(result.nfev)
            nit=int(result.nit)
            message=str(result.message)
            # SciPy can return a final accepted iterate without invoking callback.
            # Only append if it does not worsen the last accepted pose.
            if not np.array_equal(proposed,current):
                _,_,proposed_stats=evaluate(proposed)
                if proposed_stats['EHinge'] <= records[-1]['EHinge'] + 1e-10*scale**2:
                    current=proposed.copy()
                    record(current)
        except _Deadline:
            timed_out=True
            message='trial time budget reached before convergence; returning last accepted pose'
        if not frames or frames[-1][0] != len(records)-1:
            frames.append((len(records)-1,current.copy()))
        _,_,final_stats=evaluate(current)
        summary=dict(trial_id=int(trial_id),initial_source=source,
                     initial_EHinge=float(initial_stats['EHinge']),
                     final_EHinge=float(final_stats['EHinge']),
                     collision_pair_count=int(final_stats['collision_pair_count']),
                     penetration_max=float(final_stats['penetration_max']),
                     hinge_rms=float(final_stats['hinge_rms']),
                     iterations=int(len(records)-1),solver_nit=nit,solver_nfev=nfev,
                     elapsed_sec=float(time.perf_counter()-trial_started),
                     solver_success=success,solver_status=status,
                     timed_out=timed_out,solver_message=message)
        return dict(pose=current,history=records,frames=frames,summary=summary)

    def perturb_pose(base):
        # Independent local tile perturbations (not a rigid deformation).
        # Mix sparse and dense tile subsets; coordinated collision patches are
        # deliberately deferred to a separate future search mode.
        candidate=base.copy()
        if len(candidate):
            coverage=float(rng.choice([0.25,0.5,1.0]))
            count=max(1,int(np.ceil(coverage*len(candidate))))
            indices=rng.choice(len(candidate),size=count,replace=False)
            candidate[indices,0]+=rng.normal(0.,rotation_sigma,size=count)
            candidate[indices,1:]+=rng.normal(0.,translation_sigma,size=(count,2))
        candidate[:,1:]=np.clip(candidate[:,1:],
                                reference_pose[:,1:]-drift_bound,
                                reference_pose[:,1:]+drift_bound)
        return candidate

    trials_requested=restarts+1
    best=None
    baseline_energy=None
    trial_candidates=[]
    # Distribute approximately 70% of restarts around the incumbent best
    # and 30% around the original Top Hinge reference (not around a new rest).
    best_restart_count=int(round(.7*restarts))
    for trial_id in range(trials_requested):
        now=time.perf_counter()
        if now>=soft_deadline:
            break
        if trial_id==0:
            source='baseline Top Hinge'
            initial=reference_pose.copy()
            parent_trial_id=None
        else:
            restart_index=trial_id-1
            # Accumulator alternates restart sources while preserving 70/30.
            best_source=((restart_index+1)*best_restart_count+restarts-1)//restarts > (restart_index*best_restart_count+restarts-1)//restarts
            source='perturbed best' if best_source else 'perturbed Top Hinge'
            parent_trial_id=int(best['summary']['trial_id']) if best_source else None
            initial=perturb_pose(best['pose'] if best_source else reference_pose)
        remaining_trials=trials_requested-trial_id
        local_deadline=(min(soft_deadline, now+(soft_deadline-now)/remaining_trials)
                        if np.isfinite(soft_deadline) else float('inf'))
        trial=run_trial(initial,trial_id,source,local_deadline)
        summary=trial['summary']
        summary['parent_trial_id']=parent_trial_id
        improved=(best is None or summary['final_EHinge'] < best['summary']['final_EHinge']-1e-12*max(scale**2,1e-30))
        summary['became_best']=bool(improved)
        if improved:
            best=trial
        trial_summaries.append(summary)
        trial_candidates.append(dict(trial_id=int(trial_id), pose=trial['pose'].copy(),
                                     energy=float(summary['final_EHinge'])))
        trial_trajectories.append(dict(trial_id=int(trial_id),initial_source=source,parent_trial_id=parent_trial_id,
                                       pose_history=[{'iteration': int(i),'poses':p.tolist()} for i,p in trial['frames']]))
        if trial_id==0:
            baseline_energy=float(summary['final_EHinge'])
        if progress_callback:
            progress_callback('Eq.6 multi-start',min(.99,(trial_id+1)/trials_requested),
                              f"trial {trial_id}/{restarts}: E={summary['final_EHinge']:.5g}, best trial={best['summary']['trial_id']}")
    if best is None:
        raise RuntimeError('No T2D hinge optimization trial was started')
    optimization_elapsed=time.perf_counter()-optimization_started
    selected=best['summary']
    last=best['pose'].copy()
    records=best['history']
    pose_history=best['frames']
    timed_out=bool(selected['timed_out'])
    solver_success=bool(selected['solver_success'])
    solver_status=selected['solver_status']
    message=selected['solver_message']
    chosen_trial=int(selected['trial_id'])
    chosen_source=selected['initial_source']
    global_budget_exhausted=(len(trial_summaries)<trials_requested)
    hard_feasible=None
    hard_attempts=[]
    hard_selected_source_trial=None
    hard_history=[]
    hard_repair_elapsed=0.0
    hard_refinement_elapsed=0.0
    hard_remaining_collision_count=None
    hard_remaining_max_depth=None
    hard_tolerance=max(1e-12,scale*1e-10)
    if collision_mode=='hard':
        from .hard_collision_solver import (
            footprint_contacts, repair_nonpenetration, refine_feasible_pose,
        )
        hard_started=time.perf_counter()
        hard_feasible=False
        # Prioritize the best soft energy but also try geometrically different
        # soft endpoints. Each candidate retains the ORIGINAL pose bounds.
        ordered=sorted(trial_candidates,key=lambda item:item['energy'])
        repair_sources=ordered[:min(len(ordered),6)]
        winning_hard=None
        near_miss=None
        repair_duration_fraction=.75
        for idx,item in enumerate(repair_sources):
            now=time.perf_counter()
            if now>=global_deadline:
                break
            remaining=len(repair_sources)-idx
            if np.isfinite(global_deadline):
                allotted=max(0.,(global_deadline-now)*repair_duration_fraction/remaining)
                attempt_deadline=min(global_deadline, now+allotted)
            else:
                attempt_deadline=float('inf')
            before_pose=item['pose']
            before_xy,_=positions(before_pose)
            _,_,before_count,_=collision_energy_gradient(
                before_xy.reshape(-1,2),faces,scale*1e-8,return_diagnostics=True)
            if before_count==0:
                candidate=before_pose.copy()
                repaired=True
                repair_frames=[(0,candidate.copy())]
                steps=[]
            else:
                candidate,repaired,repair_frames,steps=repair_nonpenetration(
                    before_pose,positions,faces,reference_pose,drift_bound,
                    scale,attempt_deadline,rng,max_iterations=480,
                )
            candidate_xy,_=positions(candidate)
            ci,cj,cn,cs,depths=footprint_contacts(candidate_xy,faces)
            _,_,ncount,_=collision_energy_gradient(
                candidate_xy.reshape(-1,2),faces,scale*1e-8,return_diagnostics=True)
            max_depth=float(np.max(depths,initial=0.))
            certified=bool(repaired and ncount==0 and max_depth<=hard_tolerance)
            candidate_energy,_,_ = evaluate(candidate)
            diag=dict(source_trial_id=int(item['trial_id']),
                      initial_collision_count=int(before_count),
                      final_collision_count=int(ncount),
                      final_max_depth=max_depth,
                      resulting_EHinge=float(candidate_energy),
                      feasible=certified,
                      repair_iterations=int(steps[-1]['iteration']) if steps else 0)
            hard_attempts.append(diag)
            miss_score=(int(ncount),max_depth,float(candidate_energy))
            if near_miss is None or miss_score<near_miss[0]:
                near_miss=(miss_score,diag)
            if certified and (winning_hard is None or candidate_energy<winning_hard['energy']):
                winning_hard=dict(pose=candidate.copy(),energy=float(candidate_energy),
                                  source_trial_id=int(item['trial_id']),
                                  frames=repair_frames)
            if progress_callback:
                progress_callback('Hard collision repair',min(.99,(idx+1)/len(repair_sources)),
                                  f"repair {idx+1}/{len(repair_sources)}; XY overlaps={ncount}; feasible={certified}")
        hard_repair_elapsed=time.perf_counter()-hard_started
        if winning_hard is not None:
            hard_refinement_started=time.perf_counter()
            refined, refine_frames=refine_feasible_pose(
                winning_hard['pose'],positions,faces,reference_pose,
                drift_bound,scale,global_deadline,rng,
                lambda pose: (lambda energy, gradient, stats: (energy,gradient))(*evaluate(pose)),
                iterations=24,
            )
            hard_refinement_elapsed=time.perf_counter()-hard_refinement_started
            refined_xy,_=positions(refined)
            _,_,refined_count,_=collision_energy_gradient(
                refined_xy.reshape(-1,2),faces,scale*1e-8,return_diagnostics=True)
            _,_,_,_,refined_depths=footprint_contacts(refined_xy,faces)
            hard_feasible=bool(refined_count==0 and np.max(refined_depths,initial=0.)<=hard_tolerance)
            if hard_feasible:
                last=refined.copy()
                hard_selected_source_trial=winning_hard['source_trial_id']
                chosen_trial=hard_selected_source_trial
                chosen_source=f"hard-feasible repair from soft trial #{chosen_trial}"
                records=[]  # Hard phase has separate trajectory and energy scale.
                raw_frames=winning_hard['frames']
                hard_history=[
                    dict(iteration=int(k),poses=p.tolist(),phase='collision repair')
                    for k,p in raw_frames
                ]
                offset=int(hard_history[-1]['iteration'])+1
                hard_history.extend(
                    dict(iteration=offset+int(k),poses=p.tolist(),phase='feasible refinement')
                    for k,p in refine_frames
                )
                # Final accepted state must be the final frame.
                if not np.array_equal(np.asarray(hard_history[-1]['poses']),last):
                    hard_history.append(dict(iteration=hard_history[-1]['iteration']+1,
                                             poses=last.tolist(),phase='feasible refinement'))
                _,_,hard_stats=evaluate(last)
                records=[dict(iteration=i,**hard_stats) for i in range(1)]
                pose_history=[(int(row['iteration']),np.asarray(row['poses'])) for row in hard_history]
            else:
                hard_attempts.append(dict(source_trial_id=winning_hard['source_trial_id'],
                                          feasible=False,reason='refinement audit failed'))
        if not hard_feasible:
            # A hard request that cannot be satisfied is NOT silently certified
            # or replaced by an arbitrary high-connection-error configuration.
            # Keep the best soft solution for diagnosis, mark it INVALID.
            if near_miss is not None:
                hard_remaining_collision_count=near_miss[1]['final_collision_count']
                hard_remaining_max_depth=near_miss[1]['final_max_depth']
        else:
            hard_remaining_collision_count=0
            hard_remaining_max_depth=0.
        optimization_elapsed=time.perf_counter()-optimization_started
    xy,_=positions(last)
    out=copy.deepcopy(t2d);out.vertices[:,:,:2]=xy;out.stage='T2D dual hinge'
    # Compose the SE(2) placement with each original T3D->T2D rigid transform.
    for i,(angle,tx,ty) in enumerate(last):
        c,s=np.cos(angle),np.sin(angle)
        r=np.array([[c,-s],[s,c]])
        transform=np.eye(4);transform[:2,:2]=r
        transform[:2,3]=centers[i]+scale*np.array([tx,ty])-r@centers[i]
        out.transform_matrices[i]=transform@t2d.transform_matrices[i]
        out.top_to_bottom_transforms[i]=transform@t2d.top_to_bottom_transforms[i]@np.linalg.inv(transform)
    _,_,final=evaluate(last)
    # The chosen trial has its own completed history. Never call a callback
    # local to an already finished trial from this outer scope.
    if not np.isclose(final['EHinge'], records[-1]['EHinge'], rtol=1e-10, atol=1e-12*scale**2):
        records.append(dict(iteration=len(records),**final))
        pose_history.append((len(records)-1,last.copy()))
    out.metrics.update(graph.metrics)
    out.metrics.update(final)
    out.metrics.update(hinge_connection_error=final['hinge_rms'],flat_collision_count=final['collisions'],
                       dual_hinge_final_collision_count=final['collisions'],
                       dual_hinge_layout_optimizer=('analytic L-BFGS-B with SE(2) multi-start' if search_mode=='multi_start' else 'analytic L-BFGS on rigid SE(2) poses'),
                       dual_hinge_timed_out=timed_out,dual_hinge_solver_message=message,
                       dual_hinge_time_budget_sec=budget,
                       dual_hinge_search_mode=search_mode,
                       dual_hinge_collision_constraint_mode=collision_mode,
                       hard_collision_feasible=hard_feasible,
                       hard_collision_repair_attempts=hard_attempts,
                       hard_collision_repair_source_trial=hard_selected_source_trial,
                       hard_collision_history=hard_history,
                       hard_collision_tolerance=hard_tolerance,
                       hard_collision_repair_elapsed_sec=hard_repair_elapsed,
                       hard_collision_refinement_elapsed_sec=hard_refinement_elapsed,
                       hard_collision_best_unresolved_count=hard_remaining_collision_count,
                       hard_collision_best_unresolved_max_depth=hard_remaining_max_depth,
                       dual_hinge_selected_trial=chosen_trial,
                       dual_hinge_selected_source=chosen_source,
                       dual_hinge_trials_requested=trials_requested,
                       dual_hinge_trials_completed=len(trial_summaries),
                       dual_hinge_search_global_budget_exhausted=global_budget_exhausted,
                       dual_hinge_baseline_EHinge=baseline_energy,
                       dual_hinge_best_improvement=float(baseline_energy-final['EHinge']),
                       dual_hinge_trial_summary=trial_summaries,
                       dual_hinge_trial_pose_histories=trial_trajectories,
                       dual_hinge_search_seed=seed,
                       dual_hinge_setup_elapsed_sec=setup_elapsed,
                       dual_hinge_optimization_elapsed_sec=optimization_elapsed,
                       dual_hinge_stop_reason=('time_budget' if timed_out else ('converged' if solver_success else 'solver_stopped')),
                       dual_hinge_solver_success=solver_success,dual_hinge_solver_status=solver_status,
                       dual_hinge_weights={'connection':wconn,'collision':wcoll,'anchor':wanchor},
                       dual_hinge_rigidity='exact in-plane rigid poses; fixed Z, not unrestricted paper 3D motion',
                       dual_hinge_fixed_z_infeasible=final['hinge_fixed_z_error_lower_bound']>=scale*1e-4,
                       dual_hinge_collision_pair_exclusions=0,
                       dual_hinge_collision_candidate_limit_applied=False,
                       dual_hinge_iterations=len(records)-1,dual_hinge_history=records,
                       dual_hinge_pose_history=[{'iteration': i, 'poses': p.tolist()} for i,p in pose_history],
                       dual_hinge_initial_expansion=initial_expansion,
                       dual_hinge_max_center_drift_tiles=max_center_drift_tiles,
                       dual_hinge_max_candidate_pairs=max_candidate_pairs,
                       dual_hinge_collision_model='convex 2D footprint SAT; conservative surrogate for 3D solids',
                       dual_hinge_connection_model='2 * sum squared paired 3D corner distances',
                       tile_shape_max_error_to_T3D=pipeline._tile_shape_distance_error(out.vertices,t3d.vertices,use_max=True),
                       fabrication_feasible=final['collisions']==0 and final['hinge_rms']<scale*1e-4)
    out.metrics.update(validate_flat_linkage(out,t3d,pipeline,graph))
    out.metrics['dual_hinge_initial_solid_collisions'] = audit_solid_collisions(rest,scale*1e-8)
    out.metrics.update(audit_solid_collisions(out.vertices,scale*1e-8))
    out.metrics['fabrication_feasible'] = bool(out.metrics['fabrication_feasible'] and out.metrics['solid_collision_pair_count']==0)
    if collision_mode=='hard':
        out.metrics['fabrication_feasible']=bool(out.metrics['fabrication_feasible'] and hard_feasible)
        out.metrics['hard_collision_status']=('verified_feasible' if hard_feasible else 'NONPENETRATION_NOT_FOUND')
    out.metrics["paper_classification"]["optional_anchor_energy"] = "Project-specific extension"
    out.metrics["paper_classification"]["multi_start_perturbation_search"] = ("Project-specific extension" if search_mode=="multi_start" else "Disabled (paper-style single local solve)")
    out.metrics["paper_classification"]["hard_collision_projection"] = ("Project-specific extension" if collision_mode=="hard" else "Disabled")
    graph.metrics=dict(out.metrics)
    for hinge in graph.hinges:
        hinge.rest_position_2d=.5*(out.vertices[hinge.tile_a,hinge.local_vertex_a]+out.vertices[hinge.tile_b,hinge.local_vertex_b])
    report=pipeline.StageReport(name='T2D top hinge -> T2D dual hinge',
                               objective='Rigid SE(2) Eq.6 with convex footprint SAT surrogate',
                               before_error=trial_summaries[0]['initial_EHinge'],after_error=final['EHinge'],
                               constraint_violation=final['hinge_rms'],computation_time=time.perf_counter()-started,
                               counts={'tiles':len(rest),'hinges':len(graph.hinges)})
    if progress_callback:
        progress_callback('Eq.6 complete',1.,f"selected trial {chosen_trial} ({chosen_source}); {message}; collisions={final['collisions']}; hinge_rms={final['hinge_rms']:.5g}")
    return out,graph,report
