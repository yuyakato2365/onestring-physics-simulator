"""Original OneString UI adapter: preserve diagnostics and all-stage solver output."""
from types import SimpleNamespace

import numpy as np
import pytest

import postcut_rebff_integrated as integrated
from onestring_physics.heightfields import HeightField
from onestring_physics.onestring_pipeline import PipelineParameters


def target():
    # app_runtime_base.build_target() returns HeightField, not SurfaceMesh.
    return HeightField(
        kind="sampled",
        points=np.array([[0.,0.,0.],[1.,0.,0.],
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
            target(),PipelineParameters(nx=14,ny=14),
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
    closed=HeightField(
        "sampled",
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float),
        faces=np.array([[0,1,2],[0,3,1],[1,3,2],[0,2,3]],int),
    )
    with pytest.raises(Exception):
        integrated._save_target_obj(closed,tmp_path/"bad.obj",
                                    PipelineParameters(nx=3,ny=3))


def test_sampled_heightfield_writes_original_vertices_and_triangle_indices(tmp_path):
    path=tmp_path/"S_sampled.obj"
    mesh=target()
    integrated._save_target_obj(mesh,path,PipelineParameters(nx=4,ny=4))
    rows=path.read_text().splitlines()
    assert len([line for line in rows if line.startswith("v ")])==len(mesh.points)
    assert len([line for line in rows if line.startswith("f ")])==len(mesh.faces)
    assert "v 1 1 0" in rows
    assert "f 1 2 3" in rows


def test_sampled_quad_faces_get_triangulated_exactly_like_original(tmp_path):
    square=HeightField(
        "sampled",points=target().points,
        faces=np.array([[0,1,2,3]],int),
    )
    path=tmp_path/"S_quad.obj"
    integrated._save_target_obj(square,path,PipelineParameters(nx=4,ny=4))
    rows=path.read_text().splitlines()
    assert len([line for line in rows if line.startswith("f ")])==2


@pytest.mark.parametrize("kind",["flat","dome","saddle"])
def test_builtin_analytic_heightfield_uses_original_surface_mesh(tmp_path,kind):
    # The original OneString UI also supplies analytic HeightField inputs.
    path=tmp_path/f"S_{kind}.obj"
    analytic=HeightField(kind,parameters={"amplitude":0.3,"radius":1.7})
    p=PipelineParameters(nx=4,ny=3,surface_mesh_subdivisions=2)
    integrated._save_target_obj(analytic,path,p)
    rows=path.read_text().splitlines()
    assert len([line for line in rows if line.startswith("v ")])>10
    assert len([line for line in rows if line.startswith("f ")])>10

