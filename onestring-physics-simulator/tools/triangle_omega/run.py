"""Run one isolated triangle Omega job. Execute with .venv-triangle/bin/python."""
import argparse
import hashlib
import shutil
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from time import monotonic, time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEFAULT_TARGET_PANELS = 500


def run(out, mode, rolling, input_path, target_panels=DEFAULT_TARGET_PANELS):
    if type(target_panels) is not int or target_panels < 1:
        raise ValueError('Target panel count must be a positive integer')
    out.mkdir(parents=True, exist_ok=False)
    source = input_path or ROOT/'bunny.obj'
    if input_path is not None:
        source = Path(shutil.copyfile(source, out/'input.obj'))
    cached = input_path is None and not rolling
    env = dict(os.environ, TRI_SIZE_MODE=mode, OPENBLAS_NUM_THREADS='1',
               TRI_SKIP_VIDEOS='1', ONESTRING_REPO=str(REPO),
               OPTCUTS_BIN=str(REPO/'third_party/OptCuts_csf_onestring/build2/OptCuts_bin'))
    env.pop('MS_NO_NORMALIZE',None)
    config = dict(mode=mode, rolling_guidance=rolling, input=str(source), target_panels=target_panels,
                  sha256=hashlib.sha256(source.read_bytes()).hexdigest(), reference_chart=cached,
                  python=sys.version, executable=sys.executable,
                  versions={name:version(name) for name in ['numpy','scipy','jax','libigl','trimesh','shapely']})
    (out/'settings.json').write_text(json.dumps(config,indent=2))
    def event(**fields):
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(unix_time=time(),**fields))+'\n')
    def step(name,script,*args,app_python=False,extra=None):
        command=[str(REPO/'.venv311/bin/python') if app_python else sys.executable,str(ROOT/script),*map(str,args)]
        event(stage=name,status='started',command=command)
        start=monotonic()
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(command,env={**env,**(extra or {})},stdout=log,stderr=subprocess.STDOUT)
        event(stage=name,status='failed' if result.returncode else 'finished',seconds=monotonic()-start,exit_code=result.returncode)
        result.check_returncode()
    try:
        if cached:
            chart=ROOT/'optcuts_result.pkl'
            smoothed=out/'U2.pkl'
            with (ROOT/'omega2_fields.pkl').open('rb') as stream:
                smoothed.write_bytes(pickle.dumps(pickle.load(stream)['U_omega']))
            event(stage='optcuts',status='reused_reference')
        else:
            if input_path is not None:
                step('normalize','scripts/optcuts/normalize_mesh.py',source,out/'normalized.obj',app_python=True)
                step('inflate','scripts/fair/inflate.py',out/'normalized.obj',out/'inflate')
                source=out/'inflate/inflated.obj'
            if rolling:
                step('rolling_guidance','filter_surface.py',source,out/'filtered.obj')
                source=out/'filtered.obj'
            step('optcuts','scripts/optcuts/run_optcuts.py',source,out/'optcuts',app_python=True)
            chart=out/'optcuts/result.pkl'
            step('outline','scripts/trigrid/pipeline2.py',chart,out/'smooth',target_panels,extra={'STOP_AFTER_STEP2':'1'})
            smoothed=out/'smooth/U2.pkl'
        step('omega','scripts/trigrid/omega2.py',smoothed,chart,out,extra={'NP': str(target_panels)})
        step(
            'grid', 'scripts/trigrid/pipeline2.py', out/'omega2_chart.pkl', out/'grid', target_panels,
            extra={'SKIP_STEP2': '1', 'TRI_ORDERED_BOUNDARY': '1' if rolling else '0'},
        )
        step('remesh','scripts/trigrid/remesh3d.py',out,out/'omega2_chart.pkl')
        step('viewer','scripts/trigrid/build_viewer.py',out,chart,out/'M3D.html')
        try:
            step('hinges','hinges.py',out)
        except subprocess.CalledProcessError:
            event(stage='hinges',status='unavailable',details='See hinges.log for the topology diagnostic')
        event(stage='run',status='finished')
    except Exception:
        event(stage='run',status='failed',error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('out',type=Path)
    parser.add_argument('--mode',choices=['adaptive','uniform'],required=True)
    parser.add_argument('--rolling',action='store_true')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--input',type=Path,default=REPO/'assets/Bunny_open_LargeSteps_3500.obj')
    source.add_argument('--reference',action='store_true',help='Use the historical reference Bunny and its matching cached chart.')
    parser.add_argument('--target-panels', type=int, default=DEFAULT_TARGET_PANELS)
    args=parser.parse_args()
    if args.target_panels < 1:
        parser.error('--target-panels must be a positive integer')
    run(args.out.resolve(),args.mode,args.rolling,None if args.reference else args.input,args.target_panels)
