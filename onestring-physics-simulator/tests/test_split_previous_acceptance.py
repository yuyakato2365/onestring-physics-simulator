"""Paper-only Split exceptions, preserving ordinary OFF behavior.

Both states use the OFF sqrt(area Jacobian) CSF and the same grid-seam
machinery. Paper ON additionally uses highest curvature in CSF>threshold
tiles and does not reject a cut merely because its CSF fails to improve.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from onestring_physics import paper_mesh_splitting as split


def _grid(n=4):
    vertices=np.array([[float(i),float(j),0.]
                       for j in range(n+1) for i in range(n+1)])
    faces=np.array([[j*(n+1)+i,j*(n+1)+i+1,
                     (j+1)*(n+1)+i+1,(j+1)*(n+1)+i]
                    for j in range(n) for i in range(n)],dtype=int)
    return vertices,faces


def _run(monkeypatch, paper_mode, lo, hi, *, threshold=2., budget=64,
         forced=(), curvature=None, samples=None):
    vertices,faces=_grid()
    if curvature is None:
        curvature=np.full(len(vertices),-1.)
        curvature[12]=10.  # x=2,y=2 internal source vertex
    if samples is None:
        samples=[np.array([12],dtype=int) for _ in faces]
    monkeypatch.setattr(split,"_face_samples",lambda *a:(lo,hi,samples))
    monkeypatch.setattr(split,"gaussian_curvature",lambda *a:curvature)
    domain=SimpleNamespace(
        parameterization=SimpleNamespace(metrics={},uv_vertices_2d=vertices[:,:2]),
        csf_split_enabled=True,
        csf_split_threshold=threshold,
        max_csf_splits=budget,
        reference_grid_rotation_degrees=0.,
    )
    params=SimpleNamespace(paper_fidelity_mode=paper_mode,
                           collision_split_forced_lines=forced)
    return split._split_mesh_single(vertices,faces,domain,params)


@pytest.mark.parametrize("jacobian", [1.9,4.2,12.5])
def test_paper_uses_normal_csf_but_preserves_off_unchanged(monkeypatch,jacobian):
    lo=np.ones(16)
    hi=np.full(16,jacobian)
    off=_run(monkeypatch,False,lo,hi)
    on=_run(monkeypatch,True,lo,hi)
    expected_csf=np.sqrt(jacobian)
    for result in (off,on):
        assert result[3]["max_csf_before_split"]==pytest.approx(expected_csf)
        assert result[3]["max_csf_after_split"]==pytest.approx(expected_csf)
        assert result[3]["csf_split_threshold"]==2.
    assert off[3]["number_of_splits"]==0
    assert off[3]["csf_split_algorithm_same_as_normal"] is True
    assert on[3]["csf_split_paper_exceptions_active"] is True
    if jacobian<=4:
        assert on[3]["number_of_splits"]==0
        assert on[3]["csf_split_stopping_reason"]=="threshold_met"
    else:
        # No within-quad-floor veto or immediate improvement veto in Paper ON.
        assert on[3]["number_of_splits"]>0
        assert on[2][0]["worst_sigma_reduction"]==pytest.approx(0.)
        assert on[3]["csf_split_unresolved_component_count"]>0
        assert on[3]["csf_split_stopping_reason"] in (
            "no_admissible_cut_remaining","budget_exhausted"
        )
        assert off[3]["csf_split_rejected_components"][0]["reason"]==(
            "within_quad_distortion_floor"
        )


def test_paper_prioritizes_curvature_only_inside_violating_tiles(monkeypatch):
    lo=np.ones(16)
    hi=np.ones(16)
    hi[5]=6.25   # CSF 2.5 at source vertex 6
    hi[7]=20.25  # CSF 4.5 at source vertex 8
    sources=[np.array([0],dtype=int) for _ in range(16)]
    sources[5]=np.array([6],dtype=int)
    sources[7]=np.array([8],dtype=int)
    sources[11]=np.array([13],dtype=int)  # low CSF, high curvature
    curvature=np.full(25,-1.)
    curvature[6]=100.
    curvature[8]=10.
    curvature[13]=1000.
    result=_run(monkeypatch,True,lo,hi,budget=1,
                curvature=curvature,samples=sources)
    assert len(result[2])==1
    assert result[2][0]["curvature_vertex_id"]==6
    assert result[3]["csf_split_stopping_reason"]=="budget_exhausted"
    assert result[3]["csf_split_unresolved_component_count"]>0


def test_paper_cut_is_accepted_without_improvement_while_off_rejects(monkeypatch):
    lo=np.ones(16)
    hi=np.full(16,9.)
    off=_run(monkeypatch,False,lo,hi,budget=1)
    on=_run(monkeypatch,True,lo,hi,budget=1)
    assert off[3]["number_of_splits"]==0
    assert on[3]["number_of_splits"]==1
    assert on[2][0]["worst_sigma_reduction"]==pytest.approx(0.)
    assert on[3]["max_csf_after_split"]==pytest.approx(3.)
    assert on[3]["csf_split_stopping_reason"]=="budget_exhausted"
    assert on[3]["csf_split_unresolved_component_count"]==2
    assert not on[3]["csf_split_bound_certified"]


def test_user_threshold_and_budget_are_preserved(monkeypatch):
    lo=np.ones(16)
    hi=np.full(16,9.)
    for enabled in (False,True):
        out=_run(monkeypatch,enabled,lo,hi,threshold=1.7,budget=1)
        assert out[3]["csf_split_threshold"]==1.7
        assert out[3]["max_csf_splits"]==1
        assert out[3]["csf_split_face_sigma"]==pytest.approx([3.]*16)


def test_separate_collision_feedback_cut_works_in_both_modes(monkeypatch):
    lo=np.ones(16)
    hi=np.ones(16)
    forced=[dict(axis="col",value=2.,source_face_id=5)]
    off=_run(monkeypatch,False,lo,hi,forced=forced,budget=0)
    on=_run(monkeypatch,True,lo,hi,forced=forced,budget=0)
    np.testing.assert_array_equal(off[0],on[0])
    np.testing.assert_array_equal(off[1],on[1])
    assert off[3]["collision_split_count"]==1
    assert on[3]["collision_split_count"]==1
    assert off[3]["csf_split_step_count"]==0
    assert on[3]["csf_split_step_count"]==0
