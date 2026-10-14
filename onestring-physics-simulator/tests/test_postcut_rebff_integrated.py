"""Original OneString UI adapter: preserve diagnostics and all-stage solver output."""
from types import SimpleNamespace

import numpy as np
import pytest

import postcut_rebff_integrated as integrated


def target():
    return SimpleNamespace(
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],
                           [1.,1.,0.],[0.,1.,0.]]),
        faces=np.array([[0,1,2],[0,2,3]],dtype=int),
    )


def report(csf, converged):
    return {
        "linear_limit_two_satisfied":converged,
        "split_count":1,
        "max_postcut_patch_linear_span":csf,
        "stop_reason":"all_patches_pass" if converged else "all_candidate_cuts_failed",
        "rejected_candidates":[{
            "axis":"row","coordinate":0.5,
            "reason":"ReferenceBFFUnavailableError: child BFF rectangle failed"
        }] if not converged else [],
    }


def test_unconverged_rectangle_raises_with_rejected_cut_details(monkeypatch):
    observed=[]
    monkeypatch.setattr(
        integrated,"_execute",
        lambda *a,**kw:{"report":report(7109.1819,False)},
    )
    def fail_if_optimizing(*a,**kw):
        raise AssertionError("Never produce a valid T2D without explicit user opt-in")
    monkeypatch.setattr(integrated,
                        "run_existing_pipeline_through_dual_hinge",fail_if_optimizing)
    with pytest.raises(integrated.PostcutReBFFDidNotConverge,
                       match="all_candidate_cuts_failed"):
        integrated.run_original_onestring_with_postcut_rebff(
            target(),SimpleNamespace(nx=14),
            diagnostic_callback=lambda r:observed.append(r),
        )
    assert observed[0]["max_postcut_patch_linear_span"]==7109.1819


def test_unconverged_preview_reuses_real_pipeline_and_remains_marked(monkeypatch):
    seen={}
    monkeypatch.setattr(
        integrated,"_execute",
        lambda *a,**kw:(seen.update({"options":kw,"args":a}) or
                        {"report":report(7109.1819,False)}),
    )
    tiles=SimpleNamespace(metrics={})
    state=SimpleNamespace(mesh_2d_initial=SimpleNamespace(metrics={}),
                          tiles_2d_dual_hinge=tiles)
    def full_pipeline(path,result,**options):
        seen["path"]=str(path)
        seen["full_options"]=options
        return state,{"postcut_rebff_unconverged_visualization_only":True}
    monkeypatch.setattr(integrated,
                        "run_existing_pipeline_through_dual_hinge",full_pipeline)
    result,audit,diagnostic=integrated.run_original_onestring_with_postcut_rebff(
        target(),SimpleNamespace(nx=14),
        max_splits=8,
        initial_rectangle_mode="source_boundary_shape_fit",
        allow_unconverged_t2d_preview=True,
    )
    assert result is state
    assert audit["postcut_rebff_csf_condition_met"] is False
    assert audit["postcut_rebff_unconverged_visualization_only"] is True
    assert diagnostic["stop_reason"]=="all_candidate_cuts_failed"
    assert seen["options"]["initial_rectangle_mode"]=="source_boundary_shape_fit"
    assert seen["args"][8]=="target_rectangle"
    assert seen["full_options"]["allow_csf_violation_preview"] is True
    assert state.tiles_2d_dual_hinge.metrics["postcut_rebff_full_T2D_dual_hinge_computed"]


def test_fully_converged_rebff_runs_original_full_solver(monkeypatch):
    mock_report=report(1.7,True)
    monkeypatch.setattr(integrated,"_execute",lambda *a,**kw:{"report":mock_report})
    state=SimpleNamespace(
        mesh_2d_initial=SimpleNamespace(metrics={}),
        tiles_2d_dual_hinge=SimpleNamespace(metrics={}),
    )
    monkeypatch.setattr(integrated,"run_existing_pipeline_through_dual_hinge",
                        lambda *a,**kw:(state,{}))
    _,audit,_=integrated.run_original_onestring_with_postcut_rebff(
        target(),SimpleNamespace(nx=14))
    assert audit["postcut_rebff_csf_condition_met"] is True
    assert not audit["postcut_rebff_all_candidate_rejections"]


def test_save_target_rejects_non_disk_mesh(tmp_path):
    closed=SimpleNamespace(
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float),
        faces=np.array([[0,1,2],[0,3,1],[1,3,2],[0,2,3]],int),
    )
    with pytest.raises(Exception):
        integrated._save_target_obj(closed,tmp_path/"bad.obj")
