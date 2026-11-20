"""OptCuts as in the 2026-10-23 area-threshold mode of onestring-physics-simulator (official OptCuts objective: symmetric
Dirichlet + seam length, b_d = 4.1, lambda_init = 0.999, bijective scaffold, alternative initial cut tried; then the
CSF-tail post-refinement, which is kept only if it improves the result).

usage:   run_optcuts.py MESH.obj OUT_DIR
env:     ONESTRING_REPO = path of a checkout of github.com/yuyakato2365/onestring-physics-simulator (commit dfe09c7)
         OPTCUTS_BIN    = path of the OptCuts binary built from upstream OptCuts cd23026 + patches/optcuts_fork_onestring.patch
outputs: OUT_DIR/result.pkl  dict(xyz, f3 = 3D surface, uv, f2 = Omega, metrics), OUT_DIR/frames/f*.bin (every iteration)
Every other ONESTRING_* variable is removed first, so none of the experimental fork options (neck energy etc.) is active.
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

neck_enabled=os.environ.get('TRI_NECK')=='1'
seed_cfg=replace(cfg,executable=str(Path(BIN).with_name('OptCuts_bin'))) if neck_enabled else cfg
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
uv=r.uv_vertices_2d.copy()
if os.environ.get('TRI_NORMALIZED') == '1':
    import numpy as np
    triangles=uv[r.uv_faces]
    signs=np.linalg.det(triangles[:,1:]-triangles[:,:1])
    if np.all(signs<0):
        uv[:,1]*=-1
        print('Reflected clockwise UV chart to the CCW convention.')
    elif np.any(signs<=0):
        raise ValueError('OptCuts chart has mixed orientations or a collapsed face.')
if os.environ.get('TRI_NORMALIZED')=='1':
    summary=dict(added_penalties=os.environ.get('TRI_NECK')=='1',
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
pickle.dump(dict(xyz=r.surface_vertices_3d, f3=r.surface_faces, uv=uv, f2=r.uv_faces,
                 metrics={k: v for k, v in r.metrics.items() if isinstance(v, (int, float, str))}), open(os.path.join(out, 'result.pkl'), 'wb'))
print('OK', {k: v for k, v in r.metrics.items() if 'csf_tail_refine_status' in k})
