"""Run the actual nested launcher in an isolated Streamlit AppTest process."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

root=Path(__file__).resolve().parents[1]
at=AppTest.from_file(str(root/'app_optcuts_20260920_paper_t3d.py'),default_timeout=90).run()
assert not at.exception, at.exception
next(w for w in at.number_input if w.label=='grid size').set_value(4)
at.run()
next(w for w in at.button if w.label=='Run OneString pipeline').click().run()
assert not at.exception, at.exception
state=at.session_state['onestring_state']
assert state is not None
original_mesh=state.mesh_3d_optimized
original_history=original_mesh.metrics['k3d_iteration_history']
report={}
for stage in ['K3D','T3D','M2D','M3D','K2D','T2D Top Hinge','T2D Dual Hinge','K3D']:
    at.selectbox(key='onestring_view_stage').select(stage).run()
    assert not at.exception, at.exception
    assert at.session_state['onestring_state'] is state
    assert state.mesh_3d_optimized is original_mesh
    assert original_mesh.metrics['k3d_iteration_history'] is original_history
    charts=[json.loads(x.proto.spec) for x in at.get('plotly_chart')]
    three_d=[c for c in charts if any(t['type'] in ('mesh3d','scatter3d') for t in c['data'])]
    assert len(three_d)==(0 if stage=='K2D' else 1), (stage,len(three_d))
    if stage in ('K3D','T3D'):
        assert len(charts)==(3 if stage=='K3D' else 1), (stage,len(charts))
    assert not any(t['type']=='scattergl' for c in charts for t in c['data'])
    if stage=='K3D':
        assert len(original_history['records'])>1
        assert charts[-2]['data'][0]['name']=='EAssembled'
        assert charts[-1]['data'][0]['name']=='max'
    at.run()  # explicit rerun, no widget change / pipeline recomputation
    assert not at.exception, at.exception
    assert at.selectbox(key='onestring_view_stage').value==stage
    assert at.session_state['onestring_state'] is state
    report[stage]=dict(charts=len(charts),three_d=len(three_d),state_retained=True)
print('VIEW_RERUN_RESULT='+json.dumps(report))
