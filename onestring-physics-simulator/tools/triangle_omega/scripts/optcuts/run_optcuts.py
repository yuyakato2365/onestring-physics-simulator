"""OptCuts as in the 2026-10-23 area-threshold mode of onestring-physics-simulator (official OptCuts objective: symmetric
Dirichlet + seam length, b_d = 4.1, lambda_init = 0.999, bijective scaffold, alternative initial cut tried; then the
CSF-tail post-refinement, which is kept only if it improves the result).

usage:   run_optcuts.py MESH.obj OUT_DIR
env:     ONESTRING_REPO = path of a checkout of github.com/yuyakato2365/onestring-physics-simulator (commit dfe09c7)
         OPTCUTS_BIN    = path of the OptCuts binary built from upstream OptCuts cd23026 + patches/optcuts_fork_onestring.patch
outputs: OUT_DIR/result.pkl  dict(xyz, f3 = 3D surface, uv, f2 = Omega, metrics), OUT_DIR/frames/f*.bin (every iteration)
Fork options are set explicitly from the triangle workflow after clearing inherited settings.
"""
import os, sys, pickle, json
mesh, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]); os.makedirs(out, exist_ok=True)
R = os.environ['ONESTRING_REPO']; BIN = os.environ['OPTCUTS_BIN']
sys.path.insert(0, R + '/src'); sys.path.insert(0, R + '/tools/optcuts_soft_fit_20261104')
from optcuts_multistart import load_mesh
from onestring_physics import optcuts_backend as ob
for k in list(os.environ):
    if k.startswith('ONESTRING_') and k not in ('ONESTRING_DUMP_DIR', 'ONESTRING_REPO'): os.environ.pop(k)
if os.environ.get('TRI_NORMALIZED') != '1':
    os.environ['ONESTRING_DUMP_DIR'] = os.path.join(out, 'frames'); os.makedirs(os.environ['ONESTRING_DUMP_DIR'], exist_ok=True)
else:
    os.environ.pop('ONESTRING_DUMP_DIR', None)
if os.environ.get('TRI_NECK') == '1':
    os.environ['ONESTRING_KEEP_CONNECTED']='1'
if os.environ.get('TRI_NORMALIZED') == '1':
    from trimesh import load
    surface = load(mesh, process=False)
    V, F = surface.vertices.copy(), surface.faces.copy()
else:
    V, F = load_mesh(mesh)
os.chdir(R)
cfg = ob.OptCutsConfig(executable=BIN, timeout_seconds=86400.0, try_alternative_initial_cut=True)
from dataclasses import replace
from pathlib import Path
import numpy as np

from conformal_bound import ChartConstraintError, constrain_chart

variation_weight = float(os.environ.get('TRI_SEAM_CURVATURE_WEIGHT', 0))
if not np.isfinite(variation_weight) or variation_weight < 0:
    raise ValueError('曲面の曲率変化の重みは0以上の有限値で指定してください。')
if variation_weight > 0:
    curvature_binary = Path(BIN).with_name('OptCuts_neck_bin')
    if not curvature_binary.is_file() or b'ONESTRING_SEAM_CURVATURE_W' not in curvature_binary.read_bytes():
        raise RuntimeError('曲率変化に対応したOptCutsを controlled/build_optcuts.py でビルドしてください。')
    os.environ['ONESTRING_SEAM_CURVATURE_W'] = str(variation_weight)
    cfg = replace(cfg, executable=str(curvature_binary))
print(f'seamに沿った曲面の曲率変化の重み {variation_weight:g}', flush=True)

epsilon = float(os.environ.get('TRI_OPTCUTS_CONFORMAL_LIMIT', .02))
if not np.isfinite(epsilon) or epsilon < 0:
    raise ValueError('等角歪み上限は0以上の有限値で指定してください。')
bend_weight = float(os.environ.get('TRI_SEAM_BEND_WEIGHT', 0))
if not np.isfinite(bend_weight) or bend_weight < 0:
    raise ValueError('seamの曲がりの重みは0以上の有限値で指定してください。')
os.environ['ONESTRING_SEAM_BEND_W'] = str(bend_weight)
print(f'OptCuts seamの曲がりの重み {bend_weight:g}', flush=True)
neck_enabled=os.environ.get('TRI_NECK')=='1'
if os.environ.get('TRI_DIRECT_OMEGA') != '1': print(f'OptCuts 各面の等角歪み上限 {epsilon:g} · SD上限 {cfg.distortion_bound:g}', flush=True)

def solve_candidate(cfg):
    # The official seed must not inherit neck settings from a previous candidate.
    for key in ('ONESTRING_NECK_PROGRESS', 'ONESTRING_NECK_W', 'ONESTRING_NECK_D0', 'ONESTRING_NECK_TOPO_W'):
        os.environ.pop(key, None)
    seed_cfg=replace(cfg,executable=str(Path(BIN).with_name('OptCuts_bin'))) if neck_enabled and not variation_weight else cfg
    try:
        initial=ob.run_official_optcuts(V,F,seed_cfg)
    except ob.OptCutsBoundaryCountError:
        if os.environ.get('TRI_NORMALIZED')!='1':raise
        print('複数のUV境界を検出。単一境界の候補を記録して初期展開を再計算します。',flush=True)
        os.environ['ONESTRING_KEEP_CONNECTED']='1'
        disk_cfg=replace(cfg,executable=str(Path(BIN).with_name('OptCuts_neck_bin')))
        initial=ob._run_official_optcuts_once(V,F,disk_cfg,recover_single_disk=True)
        initial=ob._refine_uv_for_csf_tail(initial,disk_cfg)
    if neck_enabled:
        os.environ.update(ONESTRING_NECK_PROGRESS=os.path.join(out,'iterations.log'), ONESTRING_NECK_W='100',
                          ONESTRING_NECK_D0='0.12', ONESTRING_NECK_TOPO_W='100')
        seed=initial.uv_vertices_2d.copy()
        tri=seed[initial.uv_faces]
        if np.linalg.det(tri[:,1:]-tri[:,:1]).mean()<0:
            seed[:,1]*=-1
        print('Starting the neck-penalized optimization from the official single-chart result.',flush=True)
        r=ob._run_official_optcuts_once(initial.surface_vertices_3d,initial.surface_faces,cfg,
                                        initial_uv_chart=(seed,initial.uv_faces),recover_single_disk=True)
    else:
        r = initial
    return r, initial, seed if neck_enabled else None

attempts = []
# Tightening the native search budget requests alternative cuts if the fixed
# seam UV solve fails. The acceptance bound stays at 4.1 for every candidate.
if os.environ.get('TRI_DIRECT_OMEGA') == '1':
    search_bound = cfg.distortion_bound
    native_cfg = replace(cfg, executable=str(Path(BIN).with_name('OptCuts_neck_bin')))
    r = initial = ob._run_official_optcuts_once(V, F, native_cfg, recover_single_disk=True)
    uv = r.uv_vertices_2d.copy()
    triangles = uv[r.uv_faces]
    if np.all(np.linalg.det(triangles[:,1:]-triangles[:,:1]) < 0):
        uv[:,1] *= -1
    from shapely.geometry import LinearRing
    triangles = uv[r.uv_faces]
    if np.any(np.linalg.det(triangles[:,1:]-triangles[:,:1]) <= 0) or not LinearRing(uv[r.boundary_loops[0]]).is_simple:
        raise ChartConstraintError('OptCutsの直接展開に面の反転または外周の自己交差があります。')
    hard = dict(direct_omega=True)
    print('OptCutsの単一チャートをそのまま格子に使用します。', flush=True)
else:
    for search_bound in (cfg.distortion_bound, 4.05, 4.025):
        print(f'Ω候補を探索 · OptCuts探索上限 {search_bound:g}', flush=True)
        try:
            r, initial, seed = solve_candidate(replace(cfg, distortion_bound=search_bound))
            uv = r.uv_vertices_2d.copy()
            triangles = uv[r.uv_faces]
            signs = np.linalg.det(triangles[:,1:] - triangles[:,:1])
            if np.all(signs < 0):
                uv[:,1] *= -1
            uv, hard = constrain_chart(r.surface_vertices_3d, r.surface_faces, uv, r.uv_faces,
                                       r.boundary_loops[0], epsilon, cfg.distortion_bound)
            break
        except (ChartConstraintError, ob.OptCutsOutputError) as exc:
            attempts.append(dict(search_bound=search_bound, error=str(exc)))
            Path(out, 'conformal_attempts.json').write_text(json.dumps(attempts, ensure_ascii=False, indent=2))
            print(str(exc), flush=True)
    else:
        raise ValueError(f'Ωの候補が等角歪み上限 {epsilon:g} とSD上限 {cfg.distortion_bound:g} を満たしません。'
                         f' 最終候補: {attempts[-1]["error"]}')

hard.update(seam_curvature_weight=variation_weight, seam_bend_weight=bend_weight, optcuts_distortion_bound=cfg.distortion_bound, optcuts_search_bound=search_bound)
r.metrics.update(hard)
if 'conformal_max' in hard: print(f'Ω確定 · 最大等角歪み {hard["conformal_max"]:.8f} · SD {hard["conformal_sd"]:.6f}', flush=True)
differential = ob._triangle_differential_metrics(r.surface_vertices_3d, r.surface_faces, uv, r.uv_faces)
r.metrics.update(differential)
ratio = np.asarray(differential['per_triangle_sigma1']) / differential['per_triangle_sigma2']
tri3 = r.surface_vertices_3d[r.surface_faces]
areas = np.linalg.norm(np.cross(tri3[:, 1]-tri3[:, 0], tri3[:, 2]-tri3[:, 0]), axis=1)
summary=dict(added_penalties=os.environ.get('TRI_NECK')=='1',
             **hard, rejected_candidates=attempts, sigma_ratio_p95=float(np.percentile(ratio, 95)),
             sigma_ratio_max=float(ratio.max()), mean_angle_penalty=float(np.average(ratio+1/ratio-2, weights=areas)),
             boundary_loops=r.metrics['optcuts_uv_boundary_loop_count'],
             seed_disk_checkpoint=initial.metrics.get('optcuts_used_disk_checkpoint',False),
             final_disk_checkpoint=r.metrics.get('optcuts_used_disk_checkpoint',False),
             executable_sha256=r.metrics['optcuts_executable_sha256'])
if summary['added_penalties']:
    same=bool(np.array_equal(initial.uv_faces,r.uv_faces))
    summary.update(neck_weight=100,neck_distance_over_radius=.12,neck_cut_weight=100,
                   keep_connected=True,same_cut_as_initial=same)
    if same and seed.shape==uv.shape:
        X=seed-seed.mean(0); Y=uv-uv.mean(0)
        u,_,v=np.linalg.svd(X.T@Y)
        summary['max_uv_change_after_rigid_alignment']=float(np.linalg.norm(X@u@v-Y,axis=1).max())
with open(os.path.join(out,'summary.json'),'w') as stream:json.dump(summary,stream,indent=2)
with open(os.path.join(out, 'result.tmp'), 'wb') as stream:
    pickle.dump(dict(xyz=r.surface_vertices_3d, f3=r.surface_faces, uv=uv, f2=r.uv_faces,
                 metrics={k: v for k, v in r.metrics.items() if isinstance(v, (int, float, str))}), stream)
os.replace(os.path.join(out, 'result.tmp'), os.path.join(out, 'result.pkl'))
print('OK', {k: v for k, v in r.metrics.items() if 'csf_tail_refine_status' in k})
