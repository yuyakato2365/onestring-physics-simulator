"""Geometry regressions for opt-in reBFF M2D -> K2D continuation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics.rebff_k2d_guard import (
    _face_turns, count_overlaps, guard_rebff_k2d,
)


def _two_quads(target_middle=.30, solved_middle=-.50):
    xy=np.array([[0.,0.],[1.,0.],[2.,0.],
                 [0.,1.],[1.,1.],[2.,1.]])
    faces=np.array([[0,1,4,3],[1,2,5,4]],int)
    target=xy.copy()
    target[[1,4],0]=target_middle
    solver=xy.copy()
    solver[[1,4],0]=solved_middle
    source=SimpleNamespace(vertices=np.c_[xy,np.zeros(len(xy))],
                           faces=faces,metrics={"m2d_postcut_rebff_applied":True})
    m3d=SimpleNamespace(vertices=np.c_[target,np.zeros(len(target))],
                        faces=faces)
    result=SimpleNamespace(vertices=np.c_[solver,np.zeros(len(solver))],
                           faces=faces)
    return source,m3d,result


def test_folded_k2d_solver_output_is_backtracked_to_nonoverlapping_quads():
    original,m3d,solved=_two_quads()
    assert np.any(_face_turns(solved.vertices[:,:2],solved.faces)<=0.)
    corrected,report=guard_rebff_k2d(original,m3d,solved)
    assert report["k2d_rebff_guard_applied"]
    assert report["k2d_rebff_guard_clipped"]
    assert 0.<report["k2d_rebff_guard_step_fraction"]<1.
    assert report["k2d_rebff_guard_final_mean_edge_error"] < report["k2d_rebff_guard_start_mean_edge_error"]
    assert np.all(_face_turns(corrected,original.faces)>0.)
    assert count_overlaps(corrected,original.faces)==0
    # Never rewrite source M2D or K3D input during the postprocessing.
    np.testing.assert_allclose(original.vertices[1,:2],[1.,0.])


def test_valid_solver_output_is_unchanged():
    original,m3d,solved=_two_quads(target_middle=.7,solved_middle=.7)
    corrected,report=guard_rebff_k2d(original,m3d,solved)
    np.testing.assert_allclose(corrected,solved.vertices[:,:2])
    assert report["k2d_rebff_guard_step_fraction"]==1.
    assert report["k2d_rebff_guard_clipped"] is False


def test_invalid_rebff_m2d_input_is_not_misreported_as_k2d_failure():
    original,m3d,solved=_two_quads()
    original.vertices[[1,4],:2]=[[-.5,0.],[-.5,1.]]
    with pytest.raises(RuntimeError,match="K2D_REBFF_INPUT_INVALID"):
        guard_rebff_k2d(original,m3d,solved)


def test_overlapping_independent_m2d_quads_fail_before_k2d():
    xy=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],
                 [.5,.25],[1.5,.25],[1.5,1.25],[.5,1.25]])
    faces=np.array([[0,1,2,3],[4,5,6,7]],int)
    mesh=SimpleNamespace(vertices=np.c_[xy,np.zeros(len(xy))],faces=faces)
    with pytest.raises(RuntimeError,match="K2D_REBFF_INPUT_OVERLAP"):
        guard_rebff_k2d(mesh,mesh,mesh)


def test_k2d_guard_is_gated_to_rebff_manufacturing_m2d():
    root=Path(__file__).parents[1]
    app=(root/"src/onestring_physics/simple_split_panel_patch.py").read_text()
    assert 'm2d_postcut_rebff_applied",False' in app
    assert 'guard_rebff_k2d(' in app
    assert 'k2d_rebff_guard_final_mean_edge_error' in app


def test_rebff_m2d_geometry_is_paper_only_and_normal_optcuts_is_preserved():
    """Outside Paper, an old mesh's reBFF flag alone must not activate K2D guard."""
    root=Path(__file__).parents[1]
    ui=(root/"app_runtime_base.py").read_text()
    panel=(root/"src/onestring_physics/simple_split_panel_patch.py").read_text()
    engine=(root/"src/onestring_physics/paper_split_reflatten.py").read_text()
    assert 'or not paper_fidelity_enabled' in ui
    assert 'paper_split_apply_rebff_m2d = False' in ui
    assert 'getattr(params,"paper_fidelity_mode",False)' in panel
    assert 'getattr(params,"paper_split_apply_rebff_m2d",False)' in panel
    assert "POSTCUT_REBFF_M2D_PAPER_ONLY" in engine


def test_programmatic_nonpaper_rebff_m2d_request_is_explicitly_rejected():
    """A direct pipeline caller cannot silently turn on this Paper-only patch."""
    from types import SimpleNamespace
    from onestring_physics.paper_split_reflatten import split_mesh_reflatten

    uv=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    tri=np.array([[0,1,2],[0,2,3]],dtype=int)
    chart=SimpleNamespace(
        uv_vertices_2d=uv,surface_vertices_3d=np.c_[uv,np.zeros(4)],
        uv_faces=tri,surface_faces=tri,method="harmonic",metrics={},
    )
    domain=SimpleNamespace(
        parameterization=chart,reference_grid_rotation_degrees=0.,max_csf_splits=4,
    )
    params=SimpleNamespace(
        paper_fidelity_mode=False,paper_split_apply_rebff_m2d=True,
        bff_executable=None,bff_timeout_seconds=1.,
    )
    with pytest.raises(ValueError,match="POSTCUT_REBFF_M2D_PAPER_ONLY"):
        split_mesh_reflatten(uv,np.array([[0,1,2,3]],int),domain,params)


def test_version_selector_no_longer_assigns_streamlit_widget_state():
    """The reported freeze log shows a selectbox session-state/default conflict."""
    root=Path(__file__).parents[1]
    ui_patch=(root/"src/onestring_physics/paper_ui_20260916_patch.py").read_text()
    app=(root/"app_runtime_base.py").read_text()
    assert 'k["key"]=key' in ui_patch
    assert 'st.session_state[key]=opts[wanted]' not in ui_patch
    assert "PAPER_UI_RERUN_LOOP_DETECTED" in app
    assert "paper_fidelity_computation_mode" in app
