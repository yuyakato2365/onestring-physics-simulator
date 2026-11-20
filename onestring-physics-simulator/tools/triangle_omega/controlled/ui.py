"""Kagome stages inside the existing OneString Streamlit interface."""
from datetime import datetime
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from time import monotonic
from uuid import uuid4

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import trimesh

from ..ui import REPO, debug_zip, run_with_progress
from ..scripts.trigrid.omega2_metric import relative_stretch
from .job import HERE, STAGES, LABELS, completed_stages, copy_through_stage, grid_selection_rule, panel_depth_spec, saved_stage_order, stages
from onestring_physics.visualization import _add_quad_mesh_surface, _style_scene


INITIAL_RULE_LABELS = {'area_fraction':'面積の50%以上がΩ2の内側', 'centroid':'重心がΩ2の内側'}


def settings(target_kind, uploaded):
    reference = uploaded is None and target_kind=='Bunny_S45_volume（当時の入力）'
    target_id = (target_kind, getattr(uploaded, 'file_id', None))
    if st.session_state.get('kg_display_target') != target_id:
        st.session_state.pop('kg_last_run', None)
        st.session_state.pop('kg_continue', None)
        st.session_state['kg_display_target'] = target_id
    source = None
    if reference:
        source = HERE/'bunny_open_bottom_repaired.obj'
        st.caption('Bunny_S45_volume の入力（3,680頂点）。当時の設定は Rolling Guidance オン、体積を保つ丸め、45°、測定幅 0.10 です。')
    elif uploaded is None and target_kind in ('Bunny_open_LargeSteps_3500.obj', 'Bunny_open_LargeSteps.obj'):
        source = REPO/'assets'/target_kind
    preset_key = '_s45' if reference else ''
    c = {}
    c['cap'] = st.checkbox('平面の開口部を閉じる', value=source is None, help='同梱Bunnyの底を三角形で塞ぎ、底も含めて展開します。')
    c['count'] = st.number_input('パネル枚数の目安', 1, value=500, step=1, key='kg_count')
    c['rolling'] = st.checkbox('Rolling Guidance', value=True, key='kg_rolling'+preset_key,help='面法線を平滑化し、その曲面上でLargeStepsをかけてから耳先を丸めます。')
    methods = {'profile':'体積と元の形状を保つ（Bunny_S45_volume版）',
               'profile_strong':'45°超過を強く抑える', 'inflate':'局所的に膨らませる（従来）'}
    c['round_method'] = st.selectbox('曲面の丸め方', list(methods), format_func=methods.get, key='kg_round_method'+preset_key)
    profile = c['round_method'] in ('profile','profile_strong')
    c['crease_angle'] = st.slider('曲がりの許容角（度）', 15, 180, 45, 1, key='kg_angle'+preset_key,
                                 help='曲面上の指定幅で測る向きの変化の目標です。' if profile else '隣り合う面の法線の角度差で判定します。')
    if profile:
        c['profile_span'] = st.number_input('尖りを測る幅（正規化座標）', min_value=.001, value=.10, step=.01, key='kg_span'+preset_key,
                                           format='%.3f', help='曲面に沿ったこの距離まで、複数の面をまたぐ曲がりを測ります。')
        c.update(radius=0., crease_inflate=False)
    else:
        c['radius'] = st.number_input('耳先の丸め半径（正規化座標）', min_value=0., value=.08,
                                     step=.005, format='%.3f', help='0 で耳先の膨らませを無効にします。')
        c['crease_inflate'] = st.checkbox('残った折れ目を局所的に膨らませる', value=True)
    c['input_label'] = uploaded.name if uploaded is not None else target_kind
    c['neck'] = st.checkbox('OptCutsにくびれ・細い領域への罰則と切断制限を追加',value=True)
    c['outline'] = st.checkbox('Ω の外形を平滑化', value=True)
    adaptive = st.checkbox('曲率の高い部分・薄い部分のパネルを小さくする', value=False, key='kg_adaptive',
                           help='オフでは3Dの目標辺長を一定にします。オンでは曲率と厚みから場所ごとの目標辺長を求めます。')
    c['size_mode'] = 'adaptive' if adaptive else 'uniform'
    c['stretch'] = st.slider('Ω2 の伸縮の強さ', 0., 1., 1., .05, help='0 で Ω の座標を保持し、1 で調和近似の伸縮をすべて適用します。')
    c['initial_rule'] = st.radio('パネルの仮採用', list(INITIAL_RULE_LABELS), format_func=INITIAL_RULE_LABELS.get,
                                 key='kg_initial_rule', help='どちらの方式でも、緩和後にΩ2との重なりが50%未満のパネルを除きます。')
    c['boundary_lift'] = st.checkbox('M3Dの逆写像で境界パネルの歪みを抑える',value=False,
                                    help='切れ目の両岸を固定し、境界近くの頂点が曲面上で対応する位置を調整します。')
    flexible_rim = st.checkbox('M3Dの開口縁を柔軟にして境界パネルを整える', value=False,
                               help='開口縁を段階的に動かし、接する全パネルの内角が40〜80°に入るか、移動上限に達すると終了します。')
    c['rim_move_percent'] = st.number_input('開口縁の移動上限（代表辺長の%）', min_value=0., value=50., step=5.) if flexible_rim else 0.
    c['seam_spacing'] = st.checkbox('3Dの長さに応じて境界の辺数を配分（試験）', value=False,
                                   help='切れ目の両岸に同じ辺数を割り当て、3Dの目標辺長に合わせて対応する頂点の位置を選びます。')
    c['paired_protrusions'] = st.checkbox('Ω2からの突出は2枚以上が連なる場合に採用', value=True,
                                          help='格子の採用時に、Ω2の外側で辺を共有するパネルがある突出だけを残します。')
    c['extrusion_mode']=st.radio('T3D の押し出し方式', ['miter','fixed_skin','outer_layer'],
        format_func=lambda v: '従来の留め継ぎ' if v=='miter' else ('外面固定・局所押し出し' if v=='fixed_skin' else '外側の対応メッシュをつなぐ'),
        help='外側の対応メッシュ方式では元の三角形と外側の三角形を結び、横ずれに上限を設けます。外面固定ではK3Dを保持し、平面配置で干渉する辺や角の押し出しを浅くします。T2Dでヒンジ付き試作STLを出力します。')
    if c['extrusion_mode']=='outer_layer':
        c['restore_panel_area']=st.checkbox('押し出し後に縮んだパネル面を近傍と調整',value=True)
        c['outer_lateral_mm']=st.number_input('外側の層の最大横ずれ (mm)',min_value=.05,value=4.,step=.25,
            help='小さいほど横への張り出しを抑えます。曲率の高い場所では達成できる厚みが小さくなります。')
        c['outer_spacing_percent']=st.number_input('T2D のパネル中心間隔の拡大 (%)',min_value=0.,value=20.,step=5.,
            help='パネルを剛体のまま平面上で離し、厚みを確保します。ヒンジはその分長くなります。')
    choices=[s for s in STAGES if (s!='print' or c['extrusion_mode'] not in ('fixed_skin','outer_layer')) and (s not in ('rolling_guidance','post_rolling') or c['rolling']) and (s!='outline' or c['outline']) and (s!='cap' or c['cap'])]
    c['stop'] = st.selectbox('計算を終える工程', choices, index=choices.index('post_rounding' if reference else 'closure'), format_func=LABELS.get, key='kg_stop_s45' if reference else 'kg_stop')
    mode = st.radio('T3D のパネル厚', ['edge_ratio','mm'], horizontal=True,
                    index=int(c['extrusion_mode'] in ('fixed_skin','outer_layer')),key='kg_depth_rule_'+c['extrusion_mode'],
                    format_func=lambda v: '平均辺長の倍率' if v=='edge_ratio' else 'mm 指定')
    c['panel_depth_mode'] = mode
    if mode=='mm':
        c['panel_depth_value'] = st.number_input('T3D の目標厚み (mm)', min_value=.01, value=10., step=.5)
    else:
        c['panel_depth_value'] = st.number_input('T3D の目標厚み / パネルの平均辺長', min_value=.001,
                                                value=.1, step=.01, format='%.3f')
    c['depth_tolerance_mm'] = None
    if mode=='mm' and c['extrusion_mode']=='outer_layer' and st.checkbox('全パネルの厚みをそろえる', value=True):
        target = c['panel_depth_value']
        c['depth_tolerance_mm'] = st.number_input('目標厚みから許容する薄化 (mm)', min_value=0., max_value=target, value=min(1.,target), step=.05, help='三角形の3頂点すべてに適用します。0で指定厚みに固定します。条件が成立しないときはT3Dで停止します。')
    c['avoid_t2d_overlap'] = c['extrusion_mode'] in ('fixed_skin','outer_layer') or st.checkbox('T2Dのパネル交差を禁止', value=True, help='厚みを含む平面投影を分離し、必要なヒンジ間隔を表示します。')
    with st.expander('印刷設定'):
        c['mm_per_unit'] = st.number_input('正規化座標 1 単位あたりの mm',min_value=.01,value=113.,step=1.)
        c['hinge_mm'] = st.number_input('ヒンジ厚 (mm)', min_value=.1,value=2.,step=.1,key='kg_hinge')
        c['hole_mm'] = st.number_input('紐穴の直径 (mm)', min_value=2.,value=2.,step=.1,key='kg_hole')
        if c['extrusion_mode'] in ('fixed_skin','outer_layer'):
            c['string_route']=st.selectbox('紐経路', ['radial','perimeter'], format_func=lambda x: '従来の中央分岐' if x=='radial' else '外周一周・直線穴と丸い入口')
            c['material']='separate'
            st.caption('T2D試作の材料構成 · 符号付き曲率に応じた上下面の角ヒンジ。膜はパネル厚の内側に割り当て、紐穴の外壁を残します。')
        else:
            c['material']=st.radio('材料構成', ['integrated','separate'], format_func=lambda k:
                               '一体型（穴の周りだけPLA）' if k=='integrated' else '材料別（PLAパネル・TPUヒンジ）')
    saved=sorted((f for f in (REPO/'runs/triangle_kagome').glob('*') if (f/'events.jsonl').is_file()),
                 key=lambda f:(f/'events.jsonl').stat().st_mtime,reverse=True)
    if saved:
        with st.expander('保存した計算結果'):
            selected=st.selectbox('実行フォルダ',saved,format_func=lambda f:f.name)
            if st.button('この結果を表示'):
                st.session_state['kg_last_run']=str(selected)
    return source,c


def load(path):
    with path.open('rb') as f:
        return pickle.load(f)


VIEWS = {'normalize':['S（正規化）'],'rolling_guidance':['S（平滑化）'],'post_rolling':[LABELS['post_rolling']],'inflate':[LABELS['inflate']],
         'post_rounding':[LABELS['post_rounding']],
         'cap':['S（底面を閉じた形状）'],'optcuts':['Omega'],'outline':['Omega（外形平滑化）'],'omega':['Omega2'],'grid':['Omega2 + Grid','M2D','M3D（逆写像）'],
         'remesh':['M3D'],'tiles':['K3D','T3D'],'flat':['K2D'],'closure':['T2D'],'print':[]}


def figure(folder, view):
    if view=='T3D（紐穴付き同一パネル）':
        with np.load(folder/'assembled_drilled.npz') as arc:
            xyz,faces=arc['vertices'],arc['faces']
        fig=go.Figure(go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],
            color='#249c92',name=view,flatshading=True))
        _style_scene(fig)
        fig.update_layout(title=view,scene_camera=dict(eye=dict(x=1.3,y=.55,z=1.6),up=dict(x=0,y=1,z=0)))
        return fig
    if view == 'Omega2 + Grid':
        from shapely.geometry import LineString, Polygon
        fig = figure(folder, 'Omega2')
        grid = load(folder/'grid/result.pkl')
        log = json.loads((folder/'grid/summary.json').read_text())['lattice']
        count = json.loads((folder/'settings.json').read_text())['count']
        if 'sampling_uv' in grid:
            uv = grid['sampling_uv']
            source = load(folder/'omega2_fields.pkl')['U_omega']
            faces = load(folder/'omega2_chart.pkl')['f2']
            scale = np.log2(relative_stretch(source, uv, faces))
            extent = max(float(np.percentile(np.abs(scale),95)),.03)
            fig.data[0].update(x=uv[:,0], y=uv[:,1], intensity=scale, cmin=-extent, cmax=extent,
                               colorbar=dict(title='局所伸縮',tickvals=[-extent,0,extent],
                                             ticktext=[f'{2**t:.2f}×' for t in (-extent,0,extent)]))
        poly = Polygon(grid['O'])
        pitch = np.sqrt(poly.area / (count*np.sqrt(3)/4))
        theta = np.radians(log['rotation_deg'])
        angles = theta+np.arange(3)*np.pi/3
        basis = pitch*np.column_stack([np.cos(angles),np.sin(angles)])
        origin = np.asarray(log['offset']) @ basis[:2]
        bounds = grid['O']-origin
        extent = np.max(np.linalg.norm(bounds,axis=1))+pitch
        segments = []
        for direction in basis/pitch:
            normal = np.array([-direction[1],direction[0]])
            offsets = bounds@normal/(pitch*np.sqrt(3)/2)
            for k in range(int(np.ceil(offsets.min())),int(np.floor(offsets.max()))+1):
                center = origin+k*(pitch*np.sqrt(3)/2)*normal
                hit = poly.intersection(LineString([center-extent*direction,center+extent*direction]))
                for segment in getattr(hit,'geoms',[hit]):
                    if segment.geom_type=='LineString' and not segment.is_empty:
                        segments.extend([*segment.coords,(np.nan,np.nan)])
        def draw_lines(points, name, color, visible, width):
            points=np.asarray(points)
            fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=np.full(len(points),pitch*1e-4),
                mode='lines',name=name,showlegend=True,visible=visible,line=dict(color=color,width=width),hoverinfo='skip'))
        draw_lines(segments,'一様格子（緩和前）','#23303b',True,2)
        edges=np.unique(np.sort(grid['Tg'][:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
        fitted=np.concatenate([grid['Xg'][edges],np.full((len(edges),1,2),np.nan)],axis=1).reshape(-1,2)
        draw_lines(fitted,'境界に合わせた M2D','#822e12','legendonly',3)
        fitted_outline = grid['U4'][grid['Lw']]
        draw_lines(np.vstack([fitted_outline, fitted_outline[0]]),'採用パネルに合わせた Ω2 外周','#137769','legendonly',4)
        fig.update_layout(title='Ω2 ＋ Grid' + ('（格子の局所再配分）' if 'sampling_uv' in grid else ''),showlegend=True,legend=dict(orientation='h',y=-.04))
        return fig
    if view in (LABELS['inflate'], LABELS['post_rounding']):
        # The comparison HTML uses the solver coordinates before OBJ serialization.
        fields = folder/'inflate/inflate_fields.pkl'
        if view == LABELS['inflate'] and fields.is_file():
            data = load(fields)
            xyz, faces = data['V'], data['F']
        else:
            path = 'post_rounding.obj' if view == LABELS['post_rounding'] else 'inflate/inflated.obj'
            mesh = trimesh.load(folder/path, process=False)
            xyz, faces = mesh.vertices, mesh.faces
        fig = go.Figure(go.Mesh3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2], color='#9571ce', flatshading=True,
            lighting=dict(ambient=.65, diffuse=.6, specular=.07), hoverinfo='skip', showlegend=False))
        mesh = trimesh.Trimesh(xyz, faces, process=False)
        edges = xyz[mesh.edges_unique]
        lines = np.concatenate([edges, np.full((len(edges),1,3), np.nan)], axis=1).reshape(-1,3)
        fig.add_trace(go.Scatter3d(x=lines[:,0], y=lines[:,1], z=lines[:,2], mode='lines',
            line=dict(color='#4b3f5e', width=1), hoverinfo='skip', showlegend=False))
        fig.update_layout(title=view, height=620, margin=dict(l=10,r=10,b=10,t=38),
            scene=dict(aspectmode='data', camera=dict(eye=dict(x=-.85,y=.23,z=.5), up=dict(x=0,y=1,z=0)),
                       xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
        return fig
    if view.startswith('S（'):
        path={'S（底面を閉じた形状）':'capped.obj','S（正規化）':'normalized.obj','S（平滑化）':'filtered.obj',LABELS['post_rolling']:'post_rolling.obj'}[view]
        m=trimesh.load(folder/path,process=False)
        xyz,faces=np.asarray(m.vertices),np.asarray(m.faces)
    elif view.startswith('Omega'):
        m=load(folder/('omega2_chart.pkl' if view=='Omega2' else 'optcuts/result.pkl'))
        xyz=load(folder/'smooth/U2.pkl') if view=='Omega（外形平滑化）' else m['uv']
        faces=m['f2']
        if view == 'Omega2' and (folder/'omega2_fields.pkl').is_file():
            source = load(folder/'omega2_fields.pkl')['U_omega']
            scale = relative_stretch(source, xyz, faces)
            extent = max(float(np.percentile(np.abs(np.log2(scale)), 95)), .03)
            ticks = np.array([-extent, 0., extent])
            fig = go.Figure(go.Mesh3d(x=xyz[:,0], y=xyz[:,1], z=np.zeros(len(xyz)),
                i=faces[:,0], j=faces[:,1], k=faces[:,2], intensitymode='cell', intensity=np.log2(scale),
                colorscale='RdBu_r', cmin=-extent, cmax=extent,
                colorbar=dict(title='局所伸縮', tickvals=ticks, ticktext=[f'{2**x:.2f}×' for x in ticks]),
                lighting=dict(ambient=1., diffuse=0., specular=0.), hoverinfo='skip'))
            _style_scene(fig)
            fig.update_layout(title='Ω2 · 一様な拡大を除いた局所伸縮',scene_camera=dict(eye=dict(x=0,y=0,z=2),
                              up=dict(x=0,y=1,z=0)), meta=dict(scale_percentiles=np.percentile(scale,[5,95]).tolist()))
            return fig
    elif view in ('M2D','M3D（逆写像）','M3D','K3D'):
        if view.startswith('M3D（') or view=='M2D':
            m=load(folder/'grid/result.pkl')
            xyz,faces=(m['Xg'],m['Tg']) if view=='M2D' else (m['VW'],m['TW'])
            if view=='M2D' and (folder/'remesh3d.pkl').is_file():
                faces=np.delete(faces,load(folder/'remesh3d.pkl')['removed_panels'],axis=0)
        else:
            m=load(folder/'remesh3d.pkl');xyz,faces=m['X'],m['T']
    else:
        f={'K2D':'flat','T2D':'closure','T2D（干渉回避案）':'closure','T3D':'tiles'}[view]
        with np.load(folder/(f+'.npz')) as arr:
            solids=arr['separated' if view=='T2D（干渉回避案）' else view]
        n=len(solids); xyz=solids.reshape(-1,3); k=solids.shape[1]
        tri=np.array([[0,1,2]]) if k==3 else np.array([[0,1,2],[5,4,3],[1,0,3],[1,3,4],[2,1,4],[2,4,5],[0,2,5],[0,5,3]])
        if k in (6,14):
            with np.load(folder/'tiles.npz') as tiles:
                tri=tiles.get('surface_tris',tri)
        faces=(tri+k*np.arange(n)[:,None,None]).reshape(-1,3)
    horizontal=xyz.shape[1]==2 or view in ('K2D','T2D','T2D（干渉回避案）')
    if xyz.shape[1]==2: xyz=np.column_stack([xyz,np.zeros(len(xyz))])
    fig=go.Figure()
    printed_mesh = False
    fixed_skin_view = view in ('T2D','T3D') and k==14
    if fixed_skin_view and view=='T2D':
        with np.load(folder/'closure.npz') as archive:
            if 'pla_vertices' in archive:
                xyz,faces=archive['pla_vertices'],archive['pla_faces']
                printed_mesh=True
    if fixed_skin_view:
        fig.add_trace(go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],
            i=faces[:,0],j=faces[:,1],k=faces[:,2],color='#20a89d',flatshading=True,
            lighting=dict(ambient=.75,diffuse=.48,specular=.04),name=view))
        # Only the three outer edges belong to the panel outline. The other
        # triangles tessellate its variable-depth shell.
        outlines=np.concatenate((solids[:,[0,1,2,0]],np.full((n,1,3),np.nan)),axis=1).reshape(-1,3)
        if not printed_mesh:fig.add_trace(go.Scatter3d(x=outlines[:,0],y=outlines[:,1],z=outlines[:,2],
            mode='lines',line=dict(color='#25645f',width=1),hoverinfo='skip',showlegend=False))
    else:
        _add_quad_mesh_surface(fig,xyz,faces,color='#7c3aed' if view.startswith('S（') else '#14b8a6',opacity=1.,name=view)
    if view in ('K2D','T2D','T2D（干渉回避案）'):
        with np.load(folder/'flat.npz') as layout:
            joints=layout['hinges'].reshape(-1,4)
        if view=='T2D' and k==14:
            with np.load(folder/'closure.npz') as archive:
                v,f=archive['hinge_vertices'],archive['hinge_faces']
            if len(f):
                fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],
                    color='#e98c37',name='TPUヒンジ',flatshading=True))
        elif len(joints):
            left,li,right,ri=joints.T
            ends=np.stack([solids[left,li],solids[right,ri]],axis=1)
            lines=np.concatenate([ends,np.full((len(ends),1,3),np.nan)],axis=1).reshape(-1,3)
            fig.add_trace(go.Scatter3d(x=lines[:,0],y=lines[:,1],z=lines[:,2],mode='lines+markers',
                line=dict(width=2,color='#334155'),marker=dict(size=2,color='#334155'),name='角ヒンジ',visible='legendonly'))
        if view=='T2D' and (folder/'closure.json').exists():
            diagnostic=json.loads((folder/'closure.json').read_text())
            blocked=diagnostic.get('unavoidable_hinge_pairs',[])
            hits=np.unique(diagnostic.get('boundary_contact_panel_ids',[])).astype(int)
            if len(hits):
                with np.load(folder/'flat.npz') as layout:
                    hit_mask=np.isin(layout['panel_ids'],hits)
                    panel_ids=layout['panel_ids'][hit_mask]
                points=solids[hit_mask,:3].mean(axis=1)
                fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',
                    text=panel_ids,hovertemplate='パネル %{text}<extra></extra>',
                    marker=dict(size=4,color='#bf462a'),name='境界の干渉に関わるパネル',visible='legendonly'))
            if blocked:
                left,li,_,_=np.array(blocked).T
                points=solids[left,li]
                fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',
                    marker=dict(size=4,color='#c94136'),name='回転で干渉を解消できないヒンジ',visible='legendonly'))
    _style_scene(fig)
    eye=dict(x=0,y=0,z=2) if horizontal else dict(x=1.3,y=.55,z=1.6)
    fig.update_layout(title=view,scene_camera=dict(eye=eye,up=dict(x=0,y=1,z=0)))
    if fixed_skin_view and view=='T2D':
        span=np.ptp(xyz,axis=0)
        ratios=1.65*span/max(span.max(),1e-12)
        fig.update_scenes(aspectmode='manual',aspectratio=dict(zip(('x','y','z'),ratios)),
            camera_projection_type='orthographic',uirevision='fixed-skin-t2d-top-v2',
            xaxis_visible=False,yaxis_visible=False,zaxis_visible=False)
    return fig


def render(build_target, options, run_pipeline):
    source,c=options
    continuation=st.session_state.pop('kg_continue',None)
    if run_pipeline or continuation:
        folder=REPO/'runs/triangle_kagome'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
        if continuation:
            original = Path(continuation[0])
            if len(continuation)>2:
                stage = {'new_thickness':'remesh', 'new_grid':'omega', 'curvature_grid':'omega', 'new_rim':'grid', 'new_layout':'flat'}[continuation[2]]
                copy_through_stage(original, folder, stage)
                if continuation[2]=='curvature_grid':
                    from shutil import copyfile
                    copyfile(original/'remesh3d.pkl',folder/'curvature_feedback.pkl')
                    copyfile(original/'grid/summary.json',folder/'curvature_reference_grid.json')
            else:
                folder = original
            previous=json.loads((folder/'settings.json').read_text())
            completed = completed_stages(json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines())
            previous.update(saved_stage_order(previous, completed))
            if 'remesh' not in completed:
                previous['rim_move_percent'] = c['rim_move_percent']
            if 'grid' not in completed:
                previous['boundary_lift'] = c['boundary_lift']
                previous['seam_spacing'] = c['seam_spacing']
                previous['initial_rule'] = c['initial_rule']
                previous['paired_protrusions'] = c['paired_protrusions']
            if 'omega' not in completed:
                previous.update(stretch=c['stretch'], size_mode=c['size_mode'])
            previous.update({key:c[key] for key in ('mm_per_unit','hole_mm','hinge_mm','material')})
            previous['string_route']=c.get('string_route','radial')
            if 'closure' not in completed:
                previous['avoid_t2d_overlap']=c['avoid_t2d_overlap']
            if 'tiles' not in completed:
                previous.update({key:c[key] for key in ('panel_depth_mode','panel_depth_value','extrusion_mode')})
                if c['extrusion_mode']=='outer_layer':
                    previous.update({k:c[k] for k in ('outer_lateral_mm','outer_spacing_percent','depth_tolerance_mm','restore_panel_area')})
            if len(continuation)>2 and continuation[2]=='curvature_grid':
                previous=json.loads((original/'settings.json').read_text())
            previous['stop']=continuation[1]
            c=previous
        folder.parent.mkdir(parents=True,exist_ok=True)
        st.session_state['kg_last_run']=str(folder)
        # The general Figure-5 wrapper infers a stage from the overall fraction.
        # Kagome has its own finer stages, recorded directly by the worker.
        progress=st.container().progress(0.,text='入力を準備中')
        preview=st.empty()
        def draw(stage):
            if VIEWS[stage]:
                v=VIEWS[stage][-1]
                preview.plotly_chart(figure(folder,v),use_container_width=True,key='kg_live_'+folder.name+stage)
        with TemporaryDirectory() as tmp:
            config=Path(tmp)/'settings.json';config.write_text(json.dumps(c))
            cmd=[str(REPO/'.venv-triangle/bin/python'),str(HERE/'job.py'),str(folder),str(config)]
            if continuation:
                cmd.append('--resume')
            elif source is not None:
                cmd.extend(['--input',str(source)])
            else:
                target=build_target()
                src=Path(tmp)/'input.obj'
                trimesh.Trimesh(target.points,target.faces,process=False).export(src,digits=17)
                cmd.extend(['--input',str(src)])
            run_with_progress(cmd,folder,progress,stages(c),monotonic(),labels=LABELS,on_stage=draw)
        preview.empty()
    last=st.session_state.get('kg_last_run')
    if not last or not (Path(last)/'events.jsonl').is_file():
        st.info('設定を選び、Run OneString pipeline を実行してください。')
        return
    folder=Path(last)
    cfg=json.loads((folder/'settings.json').read_text())
    events=[json.loads(l) for l in (folder/'events.jsonl').read_bytes().split(b'\n')[:-1]]
    choices=list(dict.fromkeys(v for e in events if e['status']=='finished' for v in VIEWS.get(e['stage'],[])))
    if 'T2D' in choices and (folder/'assembled_drilled.npz').is_file():
        choices.insert(choices.index('T2D'),'T3D（紐穴付き同一パネル）')
    if 'T2D' in choices and (folder/'closure.json').is_file() and 'separated_layout' in json.loads((folder/'closure.json').read_text()):
        choices.append('T2D（干渉回避案）')

    failures=[e for e in events if e['status']=='failed' and e['stage'] in LABELS]
    if failures and events[-1]['status']=='failed' and not run_pipeline:
        failed=failures[-1]['stage']; log=folder/(failed+'.log')
        tail=log.read_text(errors='replace').strip().splitlines() if log.exists() else []
        st.error(LABELS[failed]+'で停止しました。'+(tail[-1] if tail else ''))
    if choices:
        view=st.container().selectbox('View stage',choices,index=len(choices)-1,key='kg_stage_'+folder.name+choices[-1])
        fig = figure(folder,view)
        if view in (LABELS['inflate'], LABELS['post_rounding']):
            input_name = cfg.get('input_label', '保存された入力')
            st.caption(f"表示中の計算 {folder.name} · {input_name} · {len(fig.data[0].x):,} 頂点 · {len(fig.data[0].i):,} 面")
        if view=='M3D' and (folder/'remesh3d_summary.json').exists():
            stats=json.loads((folder/'remesh3d_summary.json').read_text())
            st.caption(f"逆写像後、重複したヒレを {stats['fins_removed']} 組除去し、曲面上で辺長と角度を緩和。"
                       f"最小内角 {stats['before']['angle_min']:.1f}° → {stats['after']['angle_min']:.1f}° · "
                       f"最大内角 {stats['before']['angle_max']:.1f}° → {stats['after']['angle_max']:.1f}°")
        if view == 'Omega2' and fig.layout.meta:
            lo, hi = fig.layout.meta['scale_percentiles']
            sizing = '曲率・厚みに応じたサイズ' if cfg.get('size_mode')=='adaptive' else '3Dで均一なサイズ'
            st.caption(f"{sizing} · この計算の伸縮の強さ {cfg['stretch']:.2f} · 面ごとの長さ倍率（面積比の平方根）の5–95%範囲 {lo:.3f}–{hi:.3f} 倍。赤は拡大、青は縮小です。")
        st.plotly_chart(fig,use_container_width=True,key='kg_view_'+folder.name+view)
        if view == 'S（丸め後）' and (folder/'inflate/inflate_summary.json').is_file():
            summary = json.loads((folder/'inflate/inflate_summary.json').read_text())
            profile_stats = summary.get('profile_rounding')
            if profile_stats:
                st.caption(f"許容角を超える頂点　{profile_stats['vertices_above_before']:,} → {profile_stats['vertices_above_after']:,} 点　"
                           f"曲がりの95パーセンタイル　{profile_stats['p95_before']:.1f}° → {profile_stats['p95_after']:.1f}°")
                st.caption(f"測定幅 {profile_stats['span']:g} · 許容角 {profile_stats['threshold_degrees']:g}° · "
                           f"丸め前からの体積変化 {profile_stats['volume_relative_change']:+.3%} · "
                           f"頂点移動 RMS {profile_stats['rms_move'] * cfg['mm_per_unit']:.2f} mm")
            rounding = summary.get('crease_rounding')
            if rounding:
                caption = f"折れ角 {rounding['threshold_degrees']:g}° を超える辺　処理前 {rounding['edges_above_before']} 本"
                if rounding.get('residual_inflation'):
                    mm = rounding['max_bulge'] * cfg['mm_per_unit']
                    caption += (f" → 丸め後 {rounding['edges_above_after_rounding']} 本"
                                f" → 膨らませ後 {rounding['edges_above_after']} 本（最大膨らみ {mm:.2f} mm）")
                else:
                    caption += f" → 丸め後 {rounding['edges_above_after']} 本"
                st.caption(caption)
        if view in ('K2D','T2D','T2D（干渉回避案）'):
            with np.load(folder/'flat.npz') as layout:
                opening=layout['opening']
                outside=np.count_nonzero((opening<1.-1e-10)|(opening>2.+1e-10))
            if outside:st.caption(f'開き具合 λ が 1〜2 の範囲を外れる {outside} 枚では、端の角度を使いました。')
    label=f"目標 {cfg['count']} 枚"
    if (folder/'remesh3d_summary.json').is_file():
        label+=f" · M3D {json.loads((folder/'remesh3d_summary.json').read_text())['panels']} 枚"
    st.caption(label)
    if (folder/'closure.json').is_file():
        d=json.loads((folder/'closure.json').read_text())
        if d.get('extrusion_mode') in ('fixed_skin','outer_layer'):
            st.caption(f"T2D · ヒンジ {d['hinges']} 個 · 未接合の端 {d['unattached_hinge_ends']} 個 · 連結体 {d['components']} 個 · 意図しない接合 {d['unintended_contacts']} 件 · ヒンジ同士の干渉 {d['joint_intersections']} 件")
            if d.get('hinge_design')=='dual_triangle':
                st.caption(f"基準面ヒンジ {d['inner_hinges']} 個・押し出し先 {d['offset_hinges']} 個 · 最大角間隔 {d['max_gap_mm']:.3f} mm · 組立時の側面ずれ {d['assembled_side_gap_mm']:.6f} mm。組立経路は別検証です。")
        else:
            msg=f"最大ヒンジずれ {d['max_gap']*cfg['mm_per_unit']:.3f} mm · 最大食い込み（面内分離距離） {d['max_penetration']*cfg['mm_per_unit']:.3f} mm"
            if 'separated_layout' in d:
                alt=d['separated_layout']
                st.caption(f"干渉回避案 · 干渉 {alt['overlap_pairs']} 組 · 必要なヒンジ間隔は最大 {alt['max_gap']*cfg['mm_per_unit']:.2f} mm")
            if d.get('layout_seed')=='separated':
                st.caption(f"T2D · パネル交差 {d['footprint_overlap_pairs']} 組 · ヒンジ間隔の中央値 {d['median_gap']*cfg['mm_per_unit']:.2f} mm · 最大 {d['max_gap']*cfg['mm_per_unit']:.2f} mm")
            elif d['converged']: st.caption(msg)
            elif d['max_gap']<=d['tolerance']:
                st.warning('角ヒンジは閉じていますが、厚みのあるパネルが干渉しています。'+msg)
            else: st.warning('T2Dは未収束です。'+msg)
        if d.get('hinge_design')=='dual_triangle' and (folder/'motion_check.json').exists():
            motion=json.loads((folder/'motion_check.json').read_text())
            failed=sum(x['panel_intersections']>0 for x in motion['between_poses'])
            st.warning(f"試した中間姿勢を剛体補間すると {failed} / {len(motion['between_poses'])} 区間の中点に干渉があります。連続組立経路は未成立です。")
        if 'boundary_overlap_pairs' in d:
            st.caption(f"境界パネルが関わる干渉 {d['boundary_overlap_pairs']} 組 · 最大食い込み {d['boundary_max_penetration']*cfg['mm_per_unit']:.3f} mm · 最大ヒンジずれ {d['boundary_max_gap']*cfg['mm_per_unit']:.3f} mm")
        if d.get('unavoidable_hinge_pairs'):
            message=f"ヒンジ点を一致させたまま回転しても干渉が残る接続が {len(d['unavoidable_hinge_pairs'])} 組あります。"
            if 'boundary_unavoidable_hinges' in d:
                message+=f"うち境界に関わるものは {d['boundary_unavoidable_hinges']} 組です。"
            st.caption(message)
    if (folder/'tiles_summary.json').is_file():
        info=json.loads((folder/'tiles_summary.json').read_text())
        if info.get('extrusion_mode') in ('fixed_skin','outer_layer'):
            st.caption(f"{'外側の対応メッシュ' if info['extrusion_mode']=='outer_layer' else '外面固定'} · 中央厚の中央値 {info['center_depth_mm'][1]:.2f} mm · T2D / T3Dのパネル交差 {info['flat_body_intersections']} / {info['assembled_body_intersections']} 組")
            if 'maximum_lateral_mm' in info:
                st.caption(f"元の面に平行な方向への頂点の横ずれ · 最大 {info['maximum_lateral_mm']:.2f} mm")
            if info['near_zero_columns']:
                st.caption(f"目標の押し出し量の1%未満になった箇所 {info['near_zero_columns']} / {info['columns']}。")
        if 'corner_depth_range' in info:
            lo,hi = np.array(info['corner_depth_range'])*cfg['mm_per_unit']
            st.caption(f"T3D の実測厚み（各角をパネル法線方向に測定） {lo:.2f}–{hi:.2f} mm")
        if 'boundary_minimum_depth_fraction' in info:
            st.caption(f"境界で最も薄い角は指定厚の {info['boundary_minimum_depth_fraction']:.1%}（パネル {info['boundary_thinnest_panel_id']}）")
        cut=len(info['miter_apex_panels'])
        if cut: st.caption(f'留め継ぎ面が指定深さより手前で交わるパネルが {cut} 枚あります。')
    completed=completed_stages(events)
    cfg.update(saved_stage_order(cfg, completed))
    if 'omega' in completed and c['size_mode'] != cfg.get('size_mode','uniform'):
        st.caption('変更した目標サイズは Run OneString pipeline で新しく計算できます。')
    if 'omega' in completed and c['stretch'] != cfg['stretch']:
        st.caption(f"表示中のΩ2は強さ {cfg['stretch']:.2f} で計算済みです。変更した強さ {c['stretch']:.2f} は Run OneString pipeline で新しく計算できます。")
    if 'grid' in completed:
        actual_rule = grid_selection_rule(folder, cfg)
        st.caption('表示中の格子の仮採用方式 · '+INITIAL_RULE_LABELS[actual_rule])
        paired = cfg.get('paired_protrusions', False)
        st.caption('突出の採用条件 · '+('Ω2の外側で2枚以上が辺を共有' if paired else '枚数の条件なし'))
        report_file = folder/'grid/summary.json'
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        design = report.get('boundary_design')
        lift = report.get('boundary_lift',{})
        if 'after' in lift:
            st.caption(f"M3D逆写像 · 境界の内角 {lift['after']['min_angle']:.1f}〜{lift['after']['max_angle']:.1f}° · 対応位置を調整した頂点 {lift['moved']} 個")
        if design and 'seam_spacing' in design:
            allocations = '、'.join(f"{pair['runs'][0]}–{pair['runs'][1]} は {pair['edges']} 辺（目標 {pair['target']:.1f}）" for pair in design['seam_spacing'])
            st.caption('3D長で配分した切れ目の両岸 · '+allocations if allocations else '3D長で外周の辺数を配分')
        if design:
            st.caption(f"元のΩ2とM2Dの外周差 · 最大距離は目標辺長の {design['max_gap_over_pitch']:.2f} 倍 · 面積差（対称差）はΩ2の {100*design['area_difference_fraction']:.2f}%")
        if (folder/'remesh3d_summary.json').exists():
            summary=json.loads((folder/'remesh3d_summary.json').read_text())
            comparison=summary.get('curvature_sampling_comparison')
            if comparison:
                st.caption(f"格子の再配分前後の最大法線角 {comparison['before_max']:.1f}° → {comparison['after_max']:.1f}°")
            rim=summary.get('flexible_rim',{})
            if rim.get('status')=='finished':
                st.caption(f"開口縁の40〜80°範囲外 · {rim['before']} → {rim['remaining']} / {rim['panels']} 枚 · 最大移動 {100*rim['max_move_over_pitch']:.1f}%")
            elif rim.get('status')=='closed':
                st.caption('このメッシュには開口縁がありません。')
            quality=summary.get('boundary_k3d')
            if quality:
                st.caption(f"K3Dの境界パネルの内角 {quality['min_angle']:.1f}〜{quality['max_angle']:.1f}° · 最大辺長比 {quality['max_edge_ratio']:.2f}")
        if (not design or design.get('method') != 'panel_shape' or c['initial_rule'] != actual_rule or c['paired_protrusions'] != paired or c['seam_spacing'] != cfg.get('seam_spacing',False) or c['boundary_lift'] != cfg.get('boundary_lift',False)) and st.button('このΩ2から選んだ方式で格子を再計算', key='kg_regrid_'+folder.name):
            st.session_state['kg_continue']=(str(folder),'grid','new_grid')
            st.rerun()
    if ('remesh' in completed and 'tiles' not in completed and cfg.get('extrusion_mode')=='outer_layer'
            and cfg.get('panel_depth_mode')=='mm' and cfg.get('depth_tolerance_mm') is not None
            and cfg['panel_depth_value']>cfg['depth_tolerance_mm'] and (folder/'tiles.log').is_file()
            and '法線の開き' in (folder/'tiles.log').read_text(errors='replace')):
        if st.button('折れの大きい領域に格子を増やし、同じ印刷条件で再計算',key='kg_curvature_grid_'+folder.name):
            st.session_state['kg_continue']=(str(folder),'closure','curvature_grid')
            st.rerun()
    if 'remesh' in completed and c['rim_move_percent'] != cfg.get('rim_move_percent',0.):
        if st.button('このM2Dから開口縁の設定を変えてM3Dを再計算', key='kg_rerim_'+folder.name):
            st.session_state['kg_continue']=(str(folder),'remesh','new_rim')
            st.rerun()
    if 'closure' in completed and cfg.get('extrusion_mode') in ('fixed_skin','outer_layer') and (c['hinge_mm']!=cfg['hinge_mm'] or c['hole_mm']!=cfg['hole_mm'] or c.get('string_route','radial')!=cfg.get('string_route','radial') or json.loads((folder/'closure.json').read_text()).get('hinge_design')!='dual_triangle'):
        if st.button('曲率に応じた角ヒンジでT2Dを再計算',key='kg_fixed_rehinge_'+folder.name):
            st.session_state['kg_continue']=(str(folder),'closure','new_layout')
            st.rerun()
    if 'closure' in completed and cfg.get('extrusion_mode') not in ('fixed_skin','outer_layer') and c['avoid_t2d_overlap'] != cfg.get('avoid_t2d_overlap',False):
        if st.button('このK2Dから交差の設定を変えてT2Dを再計算',key='kg_relayout_'+folder.name):
            st.session_state['kg_continue']=(str(folder),'closure','new_layout')
            st.rerun()
    next_stage=next((s for s in stages(dict(cfg,stop='closure' if cfg.get('extrusion_mode') in ('fixed_skin','outer_layer') else 'print')) if s not in completed),None)
    if 'remesh' in completed and (panel_depth_spec(c) != panel_depth_spec(cfg) or (c['extrusion_mode']=='outer_layer' and (c.get('depth_tolerance_mm')!=cfg.get('depth_tolerance_mm') or c.get('outer_lateral_mm',1.)!=cfg.get('outer_lateral_mm',1.) or c.get('outer_spacing_percent',0.)!=cfg.get('outer_spacing_percent',0.) or c.get('restore_panel_area',False)!=cfg.get('restore_panel_area',False))) or c['extrusion_mode']!=cfg.get('extrusion_mode','miter') or (cfg.get('extrusion_mode') in ('fixed_skin','outer_layer') and c['mm_per_unit']!=cfg['mm_per_unit'])):
        if st.button('このM3Dから選んだ押し出し方式・厚みでT3Dを再計算', key='kg_retile_'+folder.name):
            st.session_state['kg_continue']=(str(folder),'tiles','new_thickness')
            st.rerun()
    elif next_stage and st.button('この結果から '+LABELS[next_stage]+' を計算',key='kg_continue_'+folder.name):
        st.session_state['kg_continue']=(str(folder),next_stage)
        st.rerun()
    if cfg.get('extrusion_mode') in ('fixed_skin','outer_layer'):
        if (folder/'prototype/TPU_membrane.stl').is_file():
            current=json.loads((folder/'closure.json').read_text()).get('hinge_design')=='dual_triangle'
            st.caption('PLA・TPUを共通原点で読み込みます。3側面の紐穴はパネル中心で交わります。' if current else '以前の底面膜の試作です。紐穴は含まれません。')
            for name in ('PLA_tiles','TPU_membrane'):
                path=folder/'prototype'/(name+'.stl')
                st.download_button(name+'.stl',path.read_bytes(),file_name=path.name)
        else:
            st.caption('試作STL · 組立途中の干渉と最小肉厚は未検証です。')
        for name,stage in [('T3D','tiles'),('T2D','closure')]:
            path=folder/'prototype'/(name+'.stl')
            if stage in completed and path.is_file():
                st.download_button(name+' 試作STL',path.read_bytes(),file_name=path.name)
    for path in sorted(folder.glob('print/*.stl')):
        st.download_button(path.name,path.read_bytes(),file_name=path.name)
    with st.expander('出力・解析ログ'):
        st.code(str(folder),language=None)
        st.download_button('解析ログを保存',debug_zip(folder),file_name=folder.name+'-logs.zip')
