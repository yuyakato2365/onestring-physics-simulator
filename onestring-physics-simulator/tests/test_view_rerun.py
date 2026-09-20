"""Exercise the real launcher and guard against WebGL history exhaustion."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

@pytest.fixture(scope='module')
def view_results():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'tests/view_rerun_probe.py')],cwd=root,
                          env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True,timeout=180)
    assert result.returncode==0, (result.stdout+result.stderr)[-10000:]
    line=next(line for line in result.stdout.splitlines() if line.startswith('VIEW_RERUN_RESULT='))
    return json.loads(line.split('=',1)[1])


def test_k3d_state_survives_view_rerun(view_results):
    assert view_results['K3D']['state_retained']


def test_k3d_view_does_not_get_replaced_by_synthetic_view(view_results):
    assert view_results['K3D']['three_d']==1
    assert view_results['K3D']['charts']==3


def test_t3d_view_does_not_get_replaced_by_synthetic_view(view_results):
    assert view_results['T3D']['three_d']==1
    assert view_results['T3D']['charts']==1


def test_eq5_history_does_not_allocate_webgl_contexts():
    from onestring_physics.eq5_iteration_history_view import _figure
    for iteration in range(27):
        fig=_figure(np.array([[0,0],[1,0],[1,1],[0,1]]),np.array([[0,1,2,3]]),
                    {'iteration':iteration},(-1,2),(-1,2))
        assert all(t.type=='scatter' for t in fig.data)
