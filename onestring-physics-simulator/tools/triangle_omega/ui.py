from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import tempfile
from time import monotonic, sleep
from uuid import uuid4
import zipfile

import streamlit as st
import trimesh

from .run import DEFAULT_TARGET_PANELS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def debug_zip(folder):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.jsonl', '.log'}:
                archive.write(path, path.relative_to(folder))
    return buffer.getvalue()


STAGE_LABELS = {
    'normalize': '入力メッシュの正規化',
    'inflate': '入力メッシュの膨張処理',
    'rolling_guidance': 'Rolling Guidance',
    'optcuts': 'OptCutsで切断・展開',
    'outline': '平面輪郭の平滑化',
    'omega': 'Ω2の計算',
    'grid': '三角形格子の生成・調整',
    'remesh': 'M3Dの生成・調整',
    'viewer': '3Dビューの生成',
    'hinges': '角ヒンジの対応計算',
}


def run_with_progress(command, folder, progress, stages, start, labels=None, on_stage=None):
    labels = STAGE_LABELS if labels is None else labels
    events_path = folder/'events.jsonl'
    completed = set()
    active = '計算の準備中'
    failure = None

    def update():
        nonlocal active, failure
        previous = len(completed)
        if events_path.is_file():
            content = events_path.read_bytes()
            # The worker can be appending an event while Streamlit reads the file.
            for line in content.split(b'\n')[:-1]:
                event = json.loads(line)
                stage = event['stage']
                if stage not in stages:
                    continue
                status = event['status']
                if status in ('finished', 'reused_reference', 'unavailable'):
                    completed.add(stage)
                if status == 'started':
                    active = labels[stage]
                    failure = None
                elif status == 'failed':
                    failure = stage
        if on_stage and len(completed) > previous:
            on_stage(next(s for s in reversed(stages) if s in completed))
        elapsed = int(monotonic() - start)
        count = f'完了 {len(completed)}/{len(stages)} 工程 · 経過 {elapsed // 60}:{elapsed % 60:02d}'
        detail=''
        iteration_log=folder/'optcuts/iterations.log'
        if active==labels.get('optcuts') and 'optcuts' not in completed and iteration_log.is_file():
            with iteration_log.open('rb') as stream:
                stream.seek(max(0,iteration_log.stat().st_size-512))
                lines=[l for l in stream.read().splitlines() if b'vertices=' in l]
            if lines:detail=' · '+lines[-1].split()[0].decode(errors='replace')+' 反復'
        progress.progress(len(completed)/len(stages), text=f'{active}{detail} · {count}')
        return count

    with tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr) as worker:
            while worker.poll() is None:
                update()
                sleep(0.5)
            count = update()
            if worker.returncode:
                label = labels[failure] if failure else active
                progress.progress(len(completed)/len(stages), text=f'{label}で失敗 · {count}')
                stderr.seek(max(0, stderr.tell()-2000))
                traceback = stderr.read().decode(errors='replace').strip()
                stage_log = folder/f'{failure}.log' if failure else None
                details = ''
                if stage_log is not None and stage_log.is_file():
                    with stage_log.open('rb') as log:
                        log.seek(max(0, stage_log.stat().st_size-8000))
                        details = log.read().decode(errors='replace').strip()
                if 'No lattice candidate met trimming and seam constraints' in details:
                    st.error('継ぎ目の頂点対応と輪郭への収まりを両方満たす三角形格子が見つかりませんでした。')
                elif details:
                    st.error(details.splitlines()[-1])
                else:
                    st.error('生成に失敗しました。')
                if details or traceback:
                    with st.expander('エラーの詳細'):
                        st.code(details or traceback, language=None)
            elif (folder/'drift_preview.npz').is_file() and not (folder/'tiles.npz').is_file():
                progress.progress(len(completed)/len(stages), text=f'T3Dの診断形状を表示 · {count}')
            else:
                progress.progress(1.0, text=f'生成完了 · {count}')


def triangle_settings(target_kind, uploaded):
    bundled = st.checkbox('参照Bunnyを使う',value=uploaded is None and target_kind=='Remeshed_Bunny.stl',
                          key='tri_reference_'+target_kind+str(uploaded is not None))
    adaptive = st.checkbox('曲率の高い部分・薄い部分のパネルを小さくする', value=True)
    rolling = st.checkbox('Rolling Guidance', value=False,
                         help='OptCutsの前に面法線を5回フィルタリングして頂点位置を再構成します。')
    target_panels = st.number_input(
        'パネル枚数の目安', min_value=1, value=DEFAULT_TARGET_PANELS, step=1,
        help='面全体の目標枚数です。輪郭と継ぎ目に合わせるため、完成時の枚数は前後します。',
    )
    mode = 'adaptive' if adaptive else 'uniform'
    sample = REPO/'runs/triangle_omega/bunny_uniform_rg_off'
    if bundled and not rolling and not adaptive and target_panels == DEFAULT_TARGET_PANELS and (sample/'M3D.html').is_file():
        if st.button('計算済みの均一Bunnyを表示'):
            st.session_state['tri_last_run'] = str(sample)
    return bundled, mode, rolling, target_panels


def render(build_target, options, run_pipeline):
    bundled, mode, rolling, target_panels = options
    if run_pipeline:
        folder = REPO/'runs/triangle_omega'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
        st.session_state['tri_last_run'] = str(folder)
        worker = REPO/'.venv-triangle/bin/python'
        command=[str(worker),str(HERE/'run.py'),str(folder),'--mode',mode,'--target-panels',str(target_panels)]
        if bundled: command.append('--reference')
        if rolling: command.append('--rolling')
        folder.parent.mkdir(parents=True,exist_ok=True)
        stages = []
        if not bundled:
            stages.extend(['normalize', 'inflate'])
        if rolling:
            stages.append('rolling_guidance')
        stages.append('optcuts')
        if rolling or not bundled:
            stages.append('outline')
        stages.extend(['omega', 'grid', 'remesh', 'viewer', 'hinges'])
        start = monotonic()
        progress = st.progress(0.0, text=f'入力を準備中 · 完了 0/{len(stages)} 工程')
        with tempfile.TemporaryDirectory() as tmp:
            if not bundled:
                target=build_target()
                source=Path(tmp)/'target.obj'
                trimesh.Trimesh(target.points,target.faces,process=False).export(source,digits=17)
                command.extend(['--input',str(source)])
            run_with_progress(command, folder, progress, stages, start)
    last=st.session_state.get('tri_last_run')
    if not last or not Path(last).is_dir():
        st.info('設定を選び、Run OneString pipeline を実行してください。')
        return
    folder=Path(last)
    if not (folder/'settings.json').is_file():
        st.error('設定ログを作成する前に処理が終了しました。')
        return
    settings=json.loads((folder/'settings.json').read_text())
    count_label = f"目標 {settings.get('target_panels', DEFAULT_TARGET_PANELS)} 枚"
    summary_path = folder/'remesh3d_summary.json'
    if summary_path.is_file():
        actual = json.loads(summary_path.read_text()).get('panels')
        if actual is None and (folder/'hinges.json').is_file():
            actual = len(json.loads((folder/'hinges.json').read_text())['panel_ids'])
        if actual is not None:
            count_label += f" · M3D {actual} 枚"
    if (folder/'remesh3d.pkl').is_file():
        from .views import stage_figure
        stage = st.container().selectbox('View stage', ['S', 'Omega', 'M2D', 'M3D'], index=3, key='triangle_view_stage')
        if getattr(st, '_onestring_paper_ui_selectbox_wrapped', False):
            from onestring_physics.paper_ui_20260916_patch import _render
            _render(st, 'Original Figure 5 · blue frame = displayed stage', stage, 1)
        st.plotly_chart(stage_figure(folder, stage), use_container_width=True, key='triangle_view_'+stage)
    st.caption(f"{count_label} · {settings['mode']} · Rolling Guidance {'ON' if settings['rolling_guidance'] else 'OFF'}")
    with st.expander('出力・解析ログ'):
        st.code(str(folder), language=None)
        st.download_button('解析ログを保存', debug_zip(folder), file_name=folder.name+'-logs.zip')
        if (folder/'M3D.html').is_file():
            st.download_button('角度誤差・サイズの診断ビューを保存', (folder/'M3D.html').read_bytes(), file_name='M3D.html')
        if (folder/'hinges.json').is_file():
            info = json.loads((folder/'hinges.json').read_text())
            st.download_button(f"角ヒンジの対応 {info['count']} 組を保存", (folder/'hinges.json').read_bytes(), file_name='hinges.json')
