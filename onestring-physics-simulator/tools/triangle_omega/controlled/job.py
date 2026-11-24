"""Run the controlled triangle workflow as individually viewable stages."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from time import monotonic, time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[1]
DEFAULT_CONFORMAL_LIMIT = 0.02
DEFAULT_SEAM_BEND_WEIGHT = 2.0
DEFAULT_SEAM_CURVATURE_WEIGHT = 0.5
STAGES = ('normalize','rolling_guidance','post_rolling','inflate','post_rounding','cap','optcuts','outline','omega','grid','remesh','tiles','flat','closure','print')
LABELS = dict(normalize='S（正規化）',rolling_guidance='S（平滑化）',post_rolling='S（平滑化後のLargeSteps）',inflate='S（丸め後）',
              post_rounding='S（丸め後のLargeSteps）',cap='S（底面を閉じた形状）',optcuts='Omega',
              outline='Omega（外形平滑化）',omega='Omega2',grid='M2D',remesh='M3D',tiles='T3D',flat='K2D',closure='T2D',print='印刷用STL')


# Reproduce the dense Bunny seam study on the selected input surface.
DIRECT_GRID = dict(direct_omega=True, cap=False, rolling=False, radius=0., round_method='none',
                   post_rolling_large_steps=False, post_rounding_large_steps=False, neck=False,
                   outline=False, size_mode='uniform', stretch=0., seam_spacing=True,
                   paired_protrusions=True, initial_rule='area_fraction', boundary_lift=False,
                   rim_move_percent=15., conformal_limit=None)


def stages(config):
    enabled = {'inflate': not config.get('direct_omega',False),'cap':config.get('cap',False),'rolling_guidance':config['rolling'], 'post_rolling':config['rolling'] and config.get('post_rolling_large_steps',True), 'post_rounding':config.get('post_rounding_large_steps', True), 'outline':config['outline']}
    return [s for s in STAGES[:STAGES.index(config['stop'])+1] if enabled.get(s,True)]


def completed_stages(events):
    completed=set()
    guarded=False
    for event in events:
        stage=event['stage']
        if stage=='run':continue
        if event['status']=='invalidated':
            completed.difference_update(STAGES[STAGES.index(stage):])
        if event['status']=='started':
            completed.discard(stage)
            if stage=='rolling_guidance':completed.discard('post_rolling')
        if event['status']=='finished':
            completed.add(stage)
            if stage=='rolling_guidance':guarded=event.get('rolling_revision')==2
    # Old runs that already passed inflation can keep their original geometry.
    # Retry the unguarded normal fit before continuing an older failed run.
    if not guarded and 'inflate' not in completed:
        completed.difference_update(('rolling_guidance','post_rolling'))
    return completed


def saved_stage_order(settings, completed):
    # A legacy run that stopped at rounding can use the new step immediately.
    # If later geometry is already complete, continue from that same surface.
    return {
        'post_rolling_large_steps': settings.get('post_rolling_large_steps',
                                                  'inflate' not in completed or 'post_rolling' in completed),
        'post_rounding_large_steps': settings.get('post_rounding_large_steps',
                                                  not completed.intersection(STAGES[STAGES.index('cap'):])),
    }


def panel_depth_spec(settings):
    """Return the depth rule in mesh units. Historical runs use 10% of each face's mean edge."""
    import math
    mode = settings.get('panel_depth_mode', 'edge_ratio')
    value = settings.get('panel_depth_value', .1 if mode == 'edge_ratio' else 10.)
    if mode not in ('edge_ratio', 'mm') or not math.isfinite(value) or value <= 0:
        raise ValueError('T3D の目標厚みには正の有限値を指定してください。')
    if mode == 'mm':
        scale = settings['mm_per_unit']
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('印刷倍率には正の有限値を指定してください。')
        value /= scale
    return mode, value


def grid_selection_rule(folder, settings):
    if 'initial_rule' in settings:
        return settings['initial_rule']
    report = folder/'grid/summary.json'
    if report.is_file():
        return json.loads(report.read_text())['lattice'].get('initial_rule', 'centroid')
    return 'area_fraction'


def copy_through_stage(source, destination, stage):
    """Reuse the completed upstream geometry for a separate grid or thickness trial."""
    if stage not in ('optcuts_input','omega','grid','remesh','flat'):
        raise ValueError('Unsupported trial starting stage')
    from tempfile import TemporaryDirectory
    with (source/'.job.lock').open('a') as lock, TemporaryDirectory(dir=destination.parent) as temp:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        events = [json.loads(line) for line in (source/'events.jsonl').read_text().splitlines()]
        completed = completed_stages(events)
        if stage == 'optcuts_input':
            # Rounding and capping are optional. Reuse exactly the surface used by the saved run.
            stage = next(s for s in reversed(STAGES[:STAGES.index('optcuts')]) if s in completed)
        if stage not in completed:
            raise ValueError(LABELS[stage]+' の計算が完了していません。')
        saved = json.loads((source/'settings.json').read_text())
        saved.update(saved_stage_order(saved, completed_stages(events)))
        saved['stop'] = stage
        prefix = STAGES[:STAGES.index(stage)+1]
        dest = Path(temp)/'run'
        dest.mkdir()
        for item in source.iterdir():
            if item.name in ('input.obj','normalized.obj','filtered.obj','capped.obj') or item.name.split('.')[0] in prefix or ('omega' in prefix and (item.name == 'smooth' or item.name.startswith('omega2_'))) or ('remesh' in prefix and item.name.startswith('remesh3d')) or (stage=='flat' and item.name in ('hinges.json','tiles_summary.json')):
                if item.is_dir():
                    shutil.copytree(item, dest/item.name)
                else:
                    shutil.copy2(item, dest/item.name)
        if stage=='flat' and (source/'prototype/T3D.stl').exists():
            (dest/'prototype').mkdir(exist_ok=True)
            shutil.copy2(source/'prototype/T3D.stl',dest/'prototype/T3D.stl')
        (dest/'settings.json').write_text(json.dumps(saved, indent=2))
        (dest/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events if e['stage'] in prefix))
        dest.rename(destination)


def run(out,source,c,resume=False):
    if not resume:out.mkdir(parents=True,exist_ok=False)
    with (out/'.job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_locked(out,source,c,resume)


def run_locked(out, source, c, resume):
    import math
    if c.get('size_mode', 'uniform') not in ('uniform','adaptive'):
        raise ValueError('Ω2 の目標サイズ設定が不正です。')
    if not 0 < c.get('crease_angle',45.) <= 180:raise ValueError('折れ目の角度は0度より大きく180度以下で指定してください。')
    if not all(math.isfinite(c[k]) for k in ('radius','stretch','mm_per_unit','hinge_mm','hole_mm')):
        raise ValueError('Non-finite dimension')
    if not (c['count'] >= 1 and c['radius'] >= 0 and 0 <= c['stretch'] <= 1
            and min(c['mm_per_unit'],c['hinge_mm']) > 0 and c['hole_mm'] >= 2):
        raise ValueError('Invalid size or target count')
    if c.get('round_method', 'profile') not in ('profile','profile_strong','inflate','none'):
        raise ValueError('Unknown surface rounding method')
    if not math.isfinite(c.get('profile_span', .10)) or c.get('profile_span', .10) <= 0:
        raise ValueError('尖りを測る幅は正の有限な長さで指定してください。')
    src=out/'input.obj'
    completed=set()
    if resume:
        previous=json.loads((out/'settings.json').read_text())
        c.setdefault('direct_omega', previous.get('direct_omega', False))
        if c['direct_omega'] != previous.get('direct_omega', False):
            raise ValueError('Ωの生成方式を変更する場合は新しい計算を開始してください。')
        for key in ('count','rolling','radius','neck','outline','cap'):
            if previous.get(key)!=c.get(key):raise ValueError('計算条件が保存した結果と一致しません。')
        events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
        completed=completed_stages(events)
        saved_variation = previous.get('seam_curvature_weight', 0.)
        c.setdefault('seam_curvature_weight', saved_variation)
        if 'optcuts' in completed and c['seam_curvature_weight'] != saved_variation:
            raise ValueError('曲面の曲率変化の重みを変える場合はΩを再計算してください。')
        saved_bend = previous.get('seam_bend_weight', 0.)
        c.setdefault('seam_bend_weight', saved_bend)
        if 'optcuts' in completed:
            if c['seam_bend_weight'] != saved_bend:
                raise ValueError('seamの曲がりの重みを変える場合はΩを再計算してください。')
            if c.get('conformal_limit', previous.get('conformal_limit')) != previous.get('conformal_limit'):
                raise ValueError('等角歪み上限を変える場合はΩを再計算してください。')
            c['conformal_limit'] = previous.get('conformal_limit')
        else:
            c.setdefault('conformal_limit', previous.get('conformal_limit', DEFAULT_CONFORMAL_LIMIT))
        c.setdefault('rim_move_percent', previous.get('rim_move_percent', 0.))
        if 'remesh' in completed and c['rim_move_percent'] != previous.get('rim_move_percent', 0.):
            raise ValueError('開口縁の移動幅を変更するときはM2DからM3Dを再計算してください。')
        saved_rule = grid_selection_rule(out, previous)
        c.setdefault('boundary_lift',previous.get('boundary_lift',False))
        if 'grid' in completed and c['boundary_lift'] != previous.get('boundary_lift',False):
            raise ValueError('M3D逆写像の方式を変更するときはΩ2から格子を再計算してください。')
        c.setdefault('seam_spacing',previous.get('seam_spacing',False))
        if 'grid' in completed and c['seam_spacing'] != previous.get('seam_spacing',False):
            raise ValueError('切れ目の辺数配分を変更するときはΩ2から格子を再計算してください。')
        c.setdefault('paired_protrusions', previous.get('paired_protrusions', 'grid' not in completed))
        if 'grid' in completed and c['paired_protrusions'] != previous.get('paired_protrusions', False):
            raise ValueError('突出の採用条件を変更する場合はΩ2から格子を再計算してください。')
        c.setdefault('initial_rule', saved_rule)
        if 'grid' in completed and c['initial_rule'] != saved_rule:
            raise ValueError('仮採用の方式を変更する場合は、保存したΩ2から格子を再計算してください。')
        c.setdefault('panel_depth_mode', previous.get('panel_depth_mode','edge_ratio'))
        c.setdefault('panel_depth_value', previous.get('panel_depth_value', .1 if c['panel_depth_mode']=='edge_ratio' else 10.))
        c.setdefault('depth_tolerance_mm',previous.get('depth_tolerance_mm'))
        c.setdefault('outer_spacing_percent',previous.get('outer_spacing_percent',0.))
        c.setdefault('outer_lateral_mm',previous.get('outer_lateral_mm',1.))
        if 'tiles' in completed and previous.get('extrusion_mode')=='outer_layer' and (c.get('depth_tolerance_mm')!=previous.get('depth_tolerance_mm') or c['outer_lateral_mm']!=previous.get('outer_lateral_mm',1.) or c['outer_spacing_percent']!=previous.get('outer_spacing_percent',0.)):
            raise ValueError('押し出し条件を変更するときはM3Dから再計算してください。')
        c.setdefault('extrusion_mode',previous.get('extrusion_mode','miter'))
        if 'tiles' in completed and (c['extrusion_mode']!=previous.get('extrusion_mode','miter') or (c['extrusion_mode'] in ('fixed_skin','outer_layer') and c['mm_per_unit']!=previous['mm_per_unit'])):
            raise ValueError('押し出し方式や実寸倍率を変更する場合はM3DからT3Dを再計算してください。')
        if 'closure' in completed and c['extrusion_mode'] in ('fixed_skin','outer_layer') and (c['hinge_mm']!=previous['hinge_mm'] or c['hole_mm']!=previous['hole_mm']):
            raise ValueError('ヒンジ厚や紐穴径の変更にはK2DからT2Dの再計算が必要です。')
        if 'tiles' in completed and panel_depth_spec(c) != panel_depth_spec(previous):
            raise ValueError('厚みを変更するときは保存したM3Dから別のT3Dを計算してください。')
        c.setdefault('size_mode', previous.get('size_mode','uniform'))
        if 'omega' in completed and c['size_mode'] != previous.get('size_mode','uniform'):
            raise ValueError('Ω2 の目標サイズを変更する場合は新しく計算してください。')
        if 'omega' in completed and previous['stretch'] != c['stretch']:
            raise ValueError('Ω2 の伸縮を変えて再計算する場合は新しい実行を開始してください。')
        for key, fallback in [('round_method', 'inflate'), ('profile_span', .10)]:
            saved = previous.get(key, fallback)
            if key in c and c[key] != saved:
                raise ValueError('丸め方を変更する場合は新しい計算を開始してください。')
            c[key] = saved
        # Both earlier implementations used the same method ID. Identify completed
        # strong-rounding runs by their own summary when continuing later stages.
        summary = out/'inflate/inflate_summary.json'
        if 'inflate' in completed and c['round_method']=='profile' and summary.is_file():
            if json.loads(summary.read_text()).get('profile_rounding',{}).get('solver')=='constrained_gauss_newton':
                c['round_method']='profile_strong'
        old_angle=previous.get('crease_angle',180. if 'inflate' in completed else 45.)
        if 'crease_angle' in c and c['crease_angle']!=old_angle:
            raise ValueError('折れ目の角度を変更する場合は新しい計算を開始してください。')
        old_inflate=previous.get('crease_inflate','inflate' not in completed)
        if 'crease_inflate' in c and c['crease_inflate']!=old_inflate:
            raise ValueError('折れ目の膨らませ設定を変更する場合は新しい計算を開始してください。')
        c['crease_inflate']=old_inflate
        c['crease_angle']=old_angle
        # Keep an older completed inflation and its downstream artifacts on one pipeline version.
        c.update(saved_stage_order(previous, completed))
        if 'print' in completed:raise ValueError('STLは生成済みです。')
    else:
        shutil.copyfile(source,src)
    variation_weight = c.setdefault('seam_curvature_weight', DEFAULT_SEAM_CURVATURE_WEIGHT)
    if not math.isfinite(variation_weight) or variation_weight < 0:
        raise ValueError('曲面の曲率変化の重みは0以上の有限値で指定してください。')
    bend_weight = c.setdefault('seam_bend_weight', DEFAULT_SEAM_BEND_WEIGHT)
    if not math.isfinite(bend_weight) or bend_weight < 0:
        raise ValueError('seamの曲がりの重みは0以上の有限値で指定してください。')
    epsilon = c.setdefault('conformal_limit', DEFAULT_CONFORMAL_LIMIT)
    if epsilon is None and not c.get('direct_omega') and 'optcuts' not in completed:
        epsilon = c['conformal_limit'] = DEFAULT_CONFORMAL_LIMIT
    if epsilon is not None and (not math.isfinite(epsilon) or epsilon < 0):
        raise ValueError('等角歪み上限は0以上の有限値で指定してください。')
    if 'optcuts' not in completed:
        c.pop('optcuts_angle_weight', None)
    c.setdefault('boundary_lift', False)
    c.setdefault('seam_spacing', False)
    c.setdefault('paired_protrusions', True)
    c.setdefault('initial_rule', 'area_fraction')
    if c['initial_rule'] not in ('area_fraction', 'centroid'):
        raise ValueError('パネルの仮採用方式が不正です。')
    c.setdefault('size_mode','uniform')
    c.setdefault('round_method','profile')
    c.setdefault('profile_span',.10)
    c.setdefault('crease_angle',45.)
    c.setdefault('crease_inflate',True)
    c.setdefault('post_rolling_large_steps',True)
    if not resume:
        c['post_rounding_large_steps'] = not c.get('direct_omega', False)
    rim_percent = c.setdefault('rim_move_percent', 0.)
    if not math.isfinite(rim_percent) or rim_percent < 0:
        raise ValueError('開口縁の移動上限は有限の非負値で指定してください。')
    tolerance = c.get('depth_tolerance_mm')
    if tolerance is not None and c.get('extrusion_mode')=='outer_layer':
        if c.get('panel_depth_mode')!='mm' or not math.isfinite(tolerance) or not 0 <= tolerance <= c['panel_depth_value']:
            raise ValueError('厚みの許容値は0以上、指定厚み以下のmm値にしてください。')
    if c.get('extrusion_mode')=='outer_layer' and not (math.isfinite(c.get('outer_lateral_mm',1.)) and c.get('outer_lateral_mm',1.)>0):
        raise ValueError('横ずれには正の有限値を指定してください。')
    if c.get('extrusion_mode')=='outer_layer' and not (math.isfinite(c.get('outer_spacing_percent',0.)) and c.get('outer_spacing_percent',0.)>=0):
        raise ValueError('中心間隔の拡大率は有限の非負値にしてください。')
    if c.setdefault('extrusion_mode','miter') not in ('fixed_skin','outer_layer','miter'):
        raise ValueError('不明な押し出し方式です。')
    if c['extrusion_mode'] in ('fixed_skin','outer_layer') and c['stop']=='print':
        raise ValueError('外面固定の試作STLはT2D工程で生成します。紐穴と材料分割には未対応です。')
    panel_depth_spec(c)
    settings=dict(c,input_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    path=out/'settings.tmp'
    path.write_text(json.dumps(settings,indent=2))
    path.replace(out/'settings.json')
    binary=REPO/'third_party/OptCuts_csf_onestring/build2'/('OptCuts_neck_bin' if c['neck'] else 'OptCuts_bin')
    env=dict(os.environ,ONESTRING_REPO=str(REPO),OPTCUTS_BIN=str(binary),TRI_SKIP_VIDEOS='1',
             TRI_DIRECT_OMEGA=str(int(c.get('direct_omega',False))),
             TRI_SEAM_CURVATURE_WEIGHT=str(variation_weight),
             TRI_SEAM_BEND_WEIGHT=str(bend_weight),
             TRI_OPTCUTS_CONFORMAL_LIMIT=str(epsilon if epsilon is not None else DEFAULT_CONFORMAL_LIMIT),
             TRI_PAIRED_PROTRUSIONS=str(int(c['paired_protrusions'])),
             TRI_RIM_MAX_FRACTION=str(rim_percent/100.),
             TRI_BOUNDARY_LIFT=str(int(c['boundary_lift'])),
             TRI_SEAM_SPACING=str(int(c['seam_spacing'])),TRI_ORDERED_BOUNDARY='1',TRI_INITIAL_RULE=c['initial_rule'],TRI_SIZE_MODE=c['size_mode'],TRI_NORMALIZED='1',TRI_SMALL_WEIGHT='10000',
             TRI_S45_ROLLING=str(int(c['round_method']=='profile')),TRI_PROFILE_SPAN=str(c['profile_span']),TRI_CREASE_ANGLE=str(c['crease_angle']),TRI_CREASE_INFLATE=str(int(c['crease_inflate'])),TRI_NECK=str(int(c['neck'])),FAIR_R=str(c['radius']),NP=str(c['count']),TRI_STRETCH=str(c['stretch']),
             OPENBLAS_NUM_THREADS='1',W_SHAPE='100')
    for k in ('MS_NO_NORMALIZE','SKIP_STEP2','STOP_AFTER_STEP2'):
        env.pop(k,None)
    def event(stage,status,**kw):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(stage=stage,status=status,unix_time=time(),**kw))+'\n')
    if resume and 'omega' in completed and 'grid' not in completed and 'grid' in stages(c):
        previous_omega = json.loads((out/'omega2_summary.json').read_text()) if (out/'omega2_summary.json').exists() else {}
        if 'injectivity_limited' not in previous_omega:
            import numpy as np
            from shapely.geometry import MultiLineString
            with (out/'omega2_chart.pkl').open('rb') as stream:
                old = pickle.load(stream)
            edges, counts = np.unique(np.sort(old['f2'][:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),
                                      axis=0, return_counts=True)
            if not MultiLineString(old['uv'][edges[counts==1]].tolist()).is_simple:
                event('omega','invalidated',reason='保存済みΩ2の外周が自己交差')
                completed.difference_update(STAGES[STAGES.index('omega'):])
    if resume and 'tiles' not in completed and 'remesh' in completed and any(
            e['stage']=='tiles' and e['status']=='failed' for e in events):
        old_report = json.loads((out/'remesh3d_summary.json').read_text())
        if 'surface_quality' not in old_report:
            event('omega','invalidated',reason='旧版の押し出し失敗後にΩ2の外周と溶接を再評価')
            completed.difference_update(STAGES[STAGES.index('omega'):])
        elif (c.get('extrusion_mode') == 'outer_layer' and c.get('panel_depth_mode') == 'mm'
              and c.get('depth_tolerance_mm') is not None and c['panel_depth_value'] > c['depth_tolerance_mm']
              and old_report.get('extrusion_fan_relaxation',{}).get('revision',0)<2 and (out/'tiles.log').is_file()
              and '法線の開き' in (out/'tiles.log').read_text(errors='replace')):
            event('remesh','invalidated',reason='厚みを考慮して保存済み格子からM3Dを再計算')
            completed.difference_update(STAGES[STAGES.index('remesh'):])
    if (out/'curvature_feedback.pkl').is_file():
        ops_feedback = dict(TRI_CURVATURE_FEEDBACK=str(out/'curvature_feedback.pkl'),
                            TRI_DEPTH_FLOOR_MM=str(c['panel_depth_value']-c['depth_tolerance_mm']),
                            TRI_LATERAL_MM=str(c['outer_lateral_mm']))
    else:
        ops_feedback = {}
    rounded = out/'inflate/inflated.obj'
    conditioned = out/'post_rounding.obj'
    final_source = out/'normalized.obj' if c.get('direct_omega') else (conditioned if c['post_rounding_large_steps'] else rounded)
    chart=out/'optcuts/result.pkl'
    smooth=out/'smooth/U2.pkl'
    normalized=out/'normalized.obj'
    filtered=out/'filtered.obj'
    rounding_script = {
        'profile': 'scripts/fair/round_profile_s45.py',
        'profile_strong': 'scripts/fair/round_profile.py',
        'inflate': 'scripts/fair/inflate.py',
        'none': None,
    }[c['round_method']]
    ops={
        'normalize':('scripts/optcuts/normalize_mesh.py',src,normalized),
        'rolling_guidance':('controlled/rolling.py',normalized,filtered),
        'post_rolling':('controlled/post_rolling.py',filtered,out/'post_rolling.obj'),
        'inflate':(rounding_script,(out/'post_rolling.obj' if c['post_rolling_large_steps'] else filtered) if c['rolling'] else normalized,out/'inflate'),
        'post_rounding':('controlled/post_rolling.py',rounded,conditioned),
        'cap':('controlled/cap.py',final_source,out/'capped.obj'),
        'optcuts':('scripts/optcuts/run_optcuts.py',out/'capped.obj' if c.get('cap') else final_source,out/'optcuts'),
        'outline':('scripts/trigrid/pipeline2.py',chart,out/'smooth',c['count']),
        'omega':('scripts/trigrid/omega2.py',smooth,chart,out),
        'grid':('scripts/trigrid/pipeline2.py',out/'omega2_chart.pkl',out/'grid',c['count']),
        'remesh':('scripts/trigrid/remesh3d.py',out,out/'omega2_chart.pkl'),
    }
    for stage in ('tiles','flat','closure','print'):
        ops[stage]=('controlled/panels.py',stage,out)
    for stage in stages(c):
        if stage in completed:continue
        script,*args=ops[stage]
        event(stage,'started')
        start=monotonic()
        try:
            if stage=='inflate' and c['round_method']=='none':
                rounded.parent.mkdir(exist_ok=True)
                shutil.copyfile(ops['inflate'][1],rounded)
                event(stage,'finished',seconds=monotonic()-start,rounding='off')
                continue
            if stage=='omega' and c.get('direct_omega'):
                import numpy as np
                with chart.open('rb') as stream:
                    native = pickle.load(stream)
                shutil.copyfile(chart, out/'omega2_chart.pkl')
                with (out/'omega2_fields.pkl').open('wb') as stream:
                    pickle.dump(dict(U_omega=native['uv'], s_star=np.ones(len(native['xyz']))), stream)
                (out/'omega2_summary.json').write_text(json.dumps(dict(direct_omega=True, injectivity_limited=False)))
                event(stage,'finished',seconds=monotonic()-start)
                continue
            if stage=='omega' and not c['outline']:
                smooth.parent.mkdir(exist_ok=True)
                with chart.open('rb') as f:
                    smooth.write_bytes(pickle.dumps(pickle.load(f)['uv']))
            py=REPO/'.venv311/bin/python' if stage in ('normalize','optcuts','cap') else sys.executable
            opt={'STOP_AFTER_STEP2':'1'} if stage=='outline' else {'SKIP_STEP2':'1', **ops_feedback} if stage=='grid' else {}
            with (out/(stage+'.log')).open('w') as f:
                subprocess.run([str(py),str(BASE/script),*map(str,args)],env={**env,**opt},stdout=f,stderr=subprocess.STDOUT,check=True)
            version={'rolling_revision':2} if stage=='rolling_guidance' else {}
            event(stage,'finished',seconds=monotonic()-start,**version)
        except subprocess.CalledProcessError as exc:
            if stage=='tiles' and exc.returncode==3 and (out/'drift_preview.npz').is_file():
                event('tiles','preview',reason='共有側面と厚みが両立しないため診断形状を保存')
                event('run','preview')
                return
            event(stage,'failed',error=str(exc),seconds=monotonic()-start)
            event('run','failed')
            raise
        except Exception as exc:
            event(stage,'failed',error=str(exc),seconds=monotonic()-start)
            event('run','failed')
            raise
    event('run','finished')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    parser.add_argument('config',type=Path)
    parser.add_argument('--input',type=Path,default=REPO/'assets/Bunny_open_isotropic_10251.obj')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.out.resolve(),args.input,json.loads(args.config.read_text()),args.resume)
